from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
from torch import nn

from .stateful_flow import entmax15


def _center_stocks(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("values must have a non-empty stock axis")
    return values - values.mean(dim=0, keepdim=True)


def _bounded_logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    return log(probability / (1.0 - probability))


@dataclass(frozen=True)
class HierarchicalRoleOutput:
    sender: torch.Tensor
    receiver: torch.Tensor
    coarse_sender: torch.Tensor
    coarse_receiver: torch.Tensor
    sender_candidate: torch.Tensor
    receiver_candidate: torch.Tensor
    sender_mix: torch.Tensor
    receiver_mix: torch.Tensor


class HierarchicalMarketRoleResidual(nn.Module):
    """Market-conditioned coarse roles that perturb, but never replace, A9 roles."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        fine_role_count: int,
        *,
        coarse_role_count: int = 4,
        mix_max: float = 0.10,
        mix_init: float = 0.02,
    ) -> None:
        super().__init__()
        if min(d_model, context_dim, fine_role_count, coarse_role_count) < 1:
            raise ValueError("hierarchical role dimensions must be positive")
        if fine_role_count < 2 or coarse_role_count < 2:
            raise ValueError("hierarchical role counts must be at least two")
        if not 0.0 < mix_max < 1.0:
            raise ValueError("mix_max must lie strictly between zero and one")
        if not 0.0 < mix_init < mix_max:
            raise ValueError("mix_init must lie strictly between zero and mix_max")

        self.d_model = int(d_model)
        self.context_dim = int(context_dim)
        self.fine_role_count = int(fine_role_count)
        self.coarse_role_count = int(coarse_role_count)
        self.mix_max = float(mix_max)

        self.stock_norm = nn.LayerNorm(self.d_model)
        self.sender_coarse = nn.Linear(
            self.d_model + self.context_dim,
            self.coarse_role_count,
        )
        self.receiver_coarse = nn.Linear(
            self.d_model + self.context_dim,
            self.coarse_role_count,
        )
        expansion_dim = self.coarse_role_count * self.fine_role_count
        self.sender_expansion = nn.Linear(self.context_dim, expansion_dim)
        self.receiver_expansion = nn.Linear(self.context_dim, expansion_dim)
        self.mix_head = nn.Linear(self.context_dim, 2)

        nn.init.zeros_(self.mix_head.weight)
        initial_ratio = float(mix_init) / self.mix_max
        nn.init.constant_(self.mix_head.bias, _bounded_logit(initial_ratio))

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        base_sender: torch.Tensor,
        base_receiver: torch.Tensor,
    ) -> HierarchicalRoleOutput:
        stocks = z.shape[0]
        if z.ndim != 2 or z.shape[1] != self.d_model:
            raise ValueError("z must be [stock, d_model]")
        if context.shape != (self.context_dim,):
            raise ValueError("context must be [context_dim]")
        expected_membership = (stocks, self.fine_role_count)
        if base_sender.shape != expected_membership or base_receiver.shape != expected_membership:
            raise ValueError("base role memberships have invalid shape")

        # This side branch must not distort the local GRU anchor in joint training.
        normalized = self.stock_norm(z.detach())
        expanded_context = context.unsqueeze(0).expand(stocks, -1)
        coarse_features = torch.cat([normalized, expanded_context], dim=-1)
        coarse_sender = entmax15(self.sender_coarse(coarse_features), dim=-1)
        coarse_receiver = entmax15(self.receiver_coarse(coarse_features), dim=-1)

        sender_expansion = self.sender_expansion(context).reshape(
            self.coarse_role_count,
            self.fine_role_count,
        ).softmax(dim=-1)
        receiver_expansion = self.receiver_expansion(context).reshape(
            self.coarse_role_count,
            self.fine_role_count,
        ).softmax(dim=-1)
        sender_candidate = coarse_sender @ sender_expansion
        receiver_candidate = coarse_receiver @ receiver_expansion

        mix = self.mix_max * torch.sigmoid(self.mix_head(context))
        sender_mix, receiver_mix = mix.unbind(dim=0)
        sender = (1.0 - sender_mix) * base_sender + sender_mix * sender_candidate
        receiver = (
            (1.0 - receiver_mix) * base_receiver
            + receiver_mix * receiver_candidate
        )
        return HierarchicalRoleOutput(
            sender=sender,
            receiver=receiver,
            coarse_sender=coarse_sender,
            coarse_receiver=coarse_receiver,
            sender_candidate=sender_candidate,
            receiver_candidate=receiver_candidate,
            sender_mix=sender_mix,
            receiver_mix=receiver_mix,
        )


@dataclass(frozen=True)
class InnovationSourceOutput:
    information: torch.Tensor
    innovation: torch.Tensor
    group_state: torch.Tensor
    residual: torch.Tensor
    gate: torch.Tensor
    relative_norm: torch.Tensor


class InnovationSourceResidual(nn.Module):
    """A bounded group-relative innovation added to the original A9 source."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        flow_dim: int,
        *,
        max_fraction: float = 0.20,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if min(d_model, context_dim, flow_dim) < 1:
            raise ValueError("innovation source dimensions must be positive")
        if not 0.0 < max_fraction < 1.0:
            raise ValueError("max_fraction must lie strictly between zero and one")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.d_model = int(d_model)
        self.context_dim = int(context_dim)
        self.flow_dim = int(flow_dim)
        self.max_fraction = float(max_fraction)
        self.eps = float(eps)

        self.innovation_norm = nn.LayerNorm(self.d_model)
        self.encoder = nn.Sequential(
            nn.Linear(self.d_model + self.context_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.flow_dim),
        )
        self.gate_head = nn.Linear(self.context_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        coarse_sender: torch.Tensor,
        base_information: torch.Tensor,
    ) -> InnovationSourceOutput:
        stocks = z.shape[0]
        if z.ndim != 2 or z.shape[1] != self.d_model:
            raise ValueError("z must be [stock, d_model]")
        if context.shape != (self.context_dim,):
            raise ValueError("context must be [context_dim]")
        if coarse_sender.ndim != 2 or coarse_sender.shape[0] != stocks:
            raise ValueError("coarse_sender must be [stock, coarse_role]")
        if base_information.shape != (stocks, self.flow_dim):
            raise ValueError("base_information must be [stock, flow_dim]")

        weights = coarse_sender.detach()
        detached_z = z.detach()
        denominator = weights.sum(dim=0, keepdim=True).transpose(0, 1) + self.eps
        coarse_state = (weights.transpose(0, 1) @ detached_z) / denominator
        group_state = (weights @ coarse_state).detach()
        innovation = self.innovation_norm(detached_z - group_state)
        expanded_context = context.unsqueeze(0).expand(stocks, -1)
        raw_residual = _center_stocks(
            self.encoder(torch.cat([innovation, expanded_context], dim=-1))
        )

        base_norm = base_information.detach().norm(dim=-1, keepdim=True)
        raw_norm = raw_residual.norm(dim=-1, keepdim=True)
        cap = self.max_fraction * base_norm
        bound_scale = torch.clamp(cap / (raw_norm + self.eps), max=1.0)
        gate = torch.sigmoid(self.gate_head(context)).reshape(())
        residual = gate * bound_scale * raw_residual
        information = _center_stocks(base_information + residual)
        relative_norm = residual.norm(dim=-1) / (base_norm.squeeze(-1) + self.eps)
        return InnovationSourceOutput(
            information=information,
            innovation=innovation,
            group_state=group_state,
            residual=residual,
            gate=gate,
            relative_norm=relative_norm,
        )


@dataclass(frozen=True)
class StrictNormFusionOutput:
    fused: torch.Tensor
    projected_flow: torch.Tensor
    correction: torch.Tensor
    delta: torch.Tensor
    fraction: torch.Tensor
    relative_norm: torch.Tensor


class StrictNormBoundedFusion(nn.Module):
    """Shared-head Flow correction with a true per-stock L2 perturbation cap."""

    def __init__(
        self,
        flow_dim: int,
        d_model: int,
        context_dim: int,
        hidden_dim: int,
        *,
        max_fraction: float = 0.08,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if min(flow_dim, d_model, context_dim, hidden_dim) < 1:
            raise ValueError("strict fusion dimensions must be positive")
        if not 0.0 < max_fraction < 1.0:
            raise ValueError("max_fraction must lie strictly between zero and one")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.flow_dim = int(flow_dim)
        self.d_model = int(d_model)
        self.context_dim = int(context_dim)
        self.max_fraction = float(max_fraction)
        self.eps = float(eps)

        self.flow_projection = nn.Linear(self.flow_dim, self.d_model, bias=False)
        self.interaction_projection = nn.Linear(
            self.d_model,
            self.d_model,
            bias=False,
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * self.d_model + self.context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.interaction_projection.weight)
        final_gate = self.gate[-1]
        if not isinstance(final_gate, nn.Linear):
            raise TypeError("strict fusion gate output must be linear")
        nn.init.zeros_(final_gate.weight)
        nn.init.zeros_(final_gate.bias)

    def forward(
        self,
        z: torch.Tensor,
        flow: torch.Tensor,
        context: torch.Tensor,
    ) -> StrictNormFusionOutput:
        stocks = z.shape[0]
        if z.ndim != 2 or z.shape[1] != self.d_model:
            raise ValueError("z must be [stock, d_model]")
        if flow.shape != (stocks, self.flow_dim):
            raise ValueError("flow must be [stock, flow_dim]")
        if context.shape != (self.context_dim,):
            raise ValueError("context must be [context_dim]")

        projected_flow = self.flow_projection(flow)
        detached_z = z.detach()
        interaction = torch.tanh(
            self.interaction_projection(detached_z * projected_flow)
        )
        correction = projected_flow + interaction
        expanded_context = context.unsqueeze(0).expand(stocks, -1)
        gate_features = torch.cat(
            [detached_z, correction, expanded_context],
            dim=-1,
        )
        fraction = self.max_fraction * torch.sigmoid(self.gate(gate_features))

        anchor_norm = detached_z.norm(dim=-1, keepdim=True)
        correction_norm = correction.norm(dim=-1, keepdim=True)
        cap = fraction * anchor_norm
        scale = torch.clamp(cap / (correction_norm + self.eps), max=1.0)
        delta = scale * correction
        fused = z + delta
        relative_norm = delta.norm(dim=-1) / (anchor_norm.squeeze(-1) + self.eps)
        return StrictNormFusionOutput(
            fused=fused,
            projected_flow=projected_flow,
            correction=correction,
            delta=delta,
            fraction=fraction,
            relative_norm=relative_norm,
        )


__all__ = [
    "HierarchicalMarketRoleResidual",
    "HierarchicalRoleOutput",
    "InnovationSourceOutput",
    "InnovationSourceResidual",
    "StrictNormBoundedFusion",
    "StrictNormFusionOutput",
]
