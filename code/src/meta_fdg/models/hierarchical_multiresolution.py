from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from .b_flow import right_gauge_projection
from .directed_graph import role_scores, sender_softmax
from .regressor import StaticFDGRegressor


HierarchicalControl = Literal[
    "candidate",
    "coarse_off",
    "sender_role_shift",
    "receiver_role_shift",
    "receiver_collapsed",
    "coarse_core_transposed",
    "self_graph",
]


def normalise_coarse_core(core: torch.Tensor) -> torch.Tensor:
    """Right-gauge a coarse core and lock its Frobenius scale to sqrt(K)."""

    if core.ndim != 2 or core.shape[0] != core.shape[1]:
        raise ValueError("coarse core must be a square matrix")
    gauged = right_gauge_projection(core)
    norm = torch.linalg.vector_norm(gauged)
    if not torch.isfinite(norm):
        raise FloatingPointError("coarse core norm is non-finite")
    return math.sqrt(core.shape[0]) * gauged / norm.clamp_min(1e-12)


def center_over_valid_senders(
    logits: torch.Tensor,
    stock_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Subtract each receiver row's valid-sender mean.

    Centering is a softmax gauge fix.  It must never use padded sender slots.
    """

    if logits.ndim < 2 or logits.shape[-2] != logits.shape[-1]:
        raise ValueError("adjacency logits must end in [receiver, sender]")
    if stock_mask is None:
        return logits - logits.mean(dim=-1, keepdim=True)
    if stock_mask.shape != logits.shape[:-1]:
        raise ValueError("stock_mask must match the receiver dimensions")
    sender_weight = stock_mask.unsqueeze(-2).to(dtype=logits.dtype)
    count = sender_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (logits * sender_weight).sum(dim=-1, keepdim=True) / count
    return logits - mean


def mean_receiver_adjacency_kl(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    stock_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return mean KL(candidate || reference) over valid receiver rows per day."""

    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference adjacency shapes must match")
    if candidate.ndim < 2 or candidate.shape[-2] != candidate.shape[-1]:
        raise ValueError("adjacency must end in [receiver, sender]")
    term = candidate * (
        candidate.clamp_min(1e-12).log()
        - reference.clamp_min(1e-12).log()
    )
    per_receiver = term.sum(dim=-1)
    if stock_mask is None:
        return per_receiver.mean(dim=-1)
    if stock_mask.shape != per_receiver.shape:
        raise ValueError("stock_mask must match adjacency receiver dimensions")
    weight = stock_mask.to(dtype=per_receiver.dtype)
    return (per_receiver * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def _normalise_daily_budget(
    budget: float | torch.Tensor,
    *,
    fine_logits: torch.Tensor,
) -> torch.Tensor:
    daily_shape = fine_logits.shape[:-2]
    value = torch.as_tensor(
        budget,
        dtype=fine_logits.dtype,
        device=fine_logits.device,
    )
    if torch.any(value < 0):
        raise ValueError("adjacency KL budget must be non-negative")
    if not daily_shape:
        if value.numel() != 1:
            raise ValueError("an unbatched day requires a scalar KL budget")
        return value.reshape(())
    try:
        return torch.broadcast_to(value, daily_shape)
    except RuntimeError as error:
        raise ValueError(
            f"adjacency KL budget must broadcast to daily shape {daily_shape}"
        ) from error


def shrink_logit_residual_to_adjacency_kl(
    fine_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    *,
    budget: float | torch.Tensor,
    stock_mask: torch.Tensor | None = None,
    bisection_steps: int = 20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Shrink a per-day logit residual to a sender-adjacency KL budget.

    The bisection scale is detached.  Gradients still flow through the final
    scaled residual.  Every leading element is solved independently, so one
    day can never borrow a trust-region scale from another day.
    """

    if fine_logits.shape != residual_logits.shape:
        raise ValueError("fine and residual logits must have matching shapes")
    if fine_logits.ndim < 2 or fine_logits.shape[-2] != fine_logits.shape[-1]:
        raise ValueError("logits must end in [receiver, sender]")
    if stock_mask is not None and stock_mask.shape != fine_logits.shape[:-1]:
        raise ValueError("stock_mask must match logit receiver dimensions")
    if int(bisection_steps) <= 0:
        raise ValueError("bisection_steps must be positive")

    daily_budget = _normalise_daily_budget(budget, fine_logits=fine_logits)
    reference = sender_softmax(fine_logits, 1.0, stock_mask)
    raw = sender_softmax(fine_logits + residual_logits, 1.0, stock_mask)
    raw_kl = mean_receiver_adjacency_kl(raw, reference, stock_mask)

    with torch.no_grad():
        low = torch.zeros_like(raw_kl)
        high = torch.ones_like(raw_kl)
        feasible_at_one = raw_kl.detach() <= daily_budget.detach()
        low = torch.where(feasible_at_one, torch.ones_like(low), low)
        for _ in range(int(bisection_steps)):
            midpoint = 0.5 * (low + high)
            scale_shape = (*midpoint.shape, 1, 1)
            proposal = sender_softmax(
                fine_logits.detach()
                + midpoint.reshape(scale_shape) * residual_logits.detach(),
                1.0,
                stock_mask,
            )
            proposal_kl = mean_receiver_adjacency_kl(
                proposal,
                reference.detach(),
                stock_mask,
            )
            feasible = proposal_kl <= daily_budget.detach()
            low = torch.where(feasible, midpoint, low)
            high = torch.where(feasible, high, midpoint)
        scale = torch.where(feasible_at_one, torch.ones_like(low), low)

    scale_shape = (*scale.shape, 1, 1)
    scaled_residual = residual_logits * scale.reshape(scale_shape)
    candidate = sender_softmax(fine_logits + scaled_residual, 1.0, stock_mask)
    final_kl = mean_receiver_adjacency_kl(candidate, reference, stock_mask)
    return scaled_residual, {
        "reference_adjacency": reference,
        "candidate_adjacency": candidate,
        "raw_adjacency_kl": raw_kl,
        "final_adjacency_kl": final_kl,
        "adjacency_kl_scale": scale,
        "adjacency_kl_budget": daily_budget,
    }


def permute_valid_stock_assignments(
    roles: torch.Tensor,
    stock_mask: torch.Tensor | None,
    *,
    shift: int = 1,
) -> torch.Tensor:
    """Cyclically permute role assignments within each day's valid stocks."""

    if roles.ndim < 2:
        raise ValueError("roles must end in [stock, role]")
    if stock_mask is None:
        return torch.roll(roles, shifts=int(shift), dims=-2)
    if stock_mask.shape != roles.shape[:-1]:
        raise ValueError("stock_mask must match role stock dimensions")

    stock_count, role_count = roles.shape[-2:]
    flat_roles = roles.reshape(-1, stock_count, role_count)
    flat_mask = stock_mask.reshape(-1, stock_count)
    rows: list[torch.Tensor] = []
    for day_roles, day_mask in zip(flat_roles, flat_mask, strict=True):
        indices = torch.nonzero(day_mask, as_tuple=False).squeeze(-1)
        day_output = day_roles.clone()
        if indices.numel() > 1:
            day_output[indices] = torch.roll(
                day_roles[indices],
                shifts=int(shift),
                dims=0,
            )
        rows.append(day_output)
    return torch.stack(rows, dim=0).reshape_as(roles)


class HierarchicalMultiResolutionSEFDG(nn.Module):
    """A parent-anchored K32+K8 residual inside one SE-FDG adjacency.

    The full Stage-I parent is frozen.  Coarse latent market roles can change
    only the logits consumed by the *single* sender softmax.  Values, skip
    path, hidden layers, and the sole prediction head are the parent's frozen
    modules.
    """

    controls = frozenset(
        {
            "candidate",
            "coarse_off",
            "sender_role_shift",
            "receiver_role_shift",
            "receiver_collapsed",
            "coarse_core_transposed",
            "self_graph",
        }
    )

    def __init__(
        self,
        base: StaticFDGRegressor,
        *,
        n_coarse_roles: int = 8,
        coarse_role_temperature: float = 2.0,
        coarse_init_seed: int = 17,
        gamma_max: float = 1.0,
        adjacency_kl_budget: float = 0.002,
        adjacency_kl_bisection_steps: int = 20,
    ) -> None:
        super().__init__()
        if not isinstance(base, StaticFDGRegressor):
            raise TypeError("base must be a StaticFDGRegressor")
        if int(base.temporal_lookback) != 1:
            raise ValueError("Phase20 requires the current-day Stage-I parent")
        n_fine_roles = int(base.B0.shape[0])
        if n_coarse_roles <= 1 or n_coarse_roles >= n_fine_roles:
            raise ValueError("coarse role count must lie between 1 and the fine count")
        if n_fine_roles % int(n_coarse_roles) != 0:
            raise ValueError("fine roles must divide evenly into coarse roles")
        if gamma_max <= 0:
            raise ValueError("gamma_max must be positive")
        if coarse_role_temperature <= 0:
            raise ValueError("coarse_role_temperature must be positive")
        if adjacency_kl_budget <= 0:
            raise ValueError("adjacency_kl_budget must be positive")
        if int(adjacency_kl_bisection_steps) <= 0:
            raise ValueError("adjacency_kl_bisection_steps must be positive")

        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        self.n_fine_roles = n_fine_roles
        self.n_coarse_roles = int(n_coarse_roles)
        self.coarse_role_temperature = float(coarse_role_temperature)
        self.gamma_max = float(gamma_max)
        self.adjacency_kl_budget = float(adjacency_kl_budget)
        self.adjacency_kl_bisection_steps = int(adjacency_kl_bisection_steps)

        d_model = int(base.graph.sender_projection.in_features)
        self.coarse_sender_projection = nn.Linear(
            d_model,
            self.n_coarse_roles,
            bias=False,
        )
        self.coarse_receiver_projection = nn.Linear(
            d_model,
            self.n_coarse_roles,
            bias=False,
        )
        # Make the K8 partition independent of ambient runner RNG consumption.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(int(coarse_init_seed))
            self.coarse_sender_projection.reset_parameters()
            self.coarse_receiver_projection.reset_parameters()
        self.coarse_core_raw = nn.Parameter(torch.eye(self.n_coarse_roles))
        self.gamma_raw = nn.Parameter(torch.zeros(()))

    def train(self, mode: bool = True) -> HierarchicalMultiResolutionSEFDG:
        super().train(mode)
        # The parent contains dropout but is a bitwise-frozen inference anchor.
        self.base.eval()
        return self

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.tanh(self.gamma_raw)

    def active_coarse_core(self, *, transpose: bool = False) -> torch.Tensor:
        core = normalise_coarse_core(self.coarse_core_raw)
        return core.transpose(-1, -2) if transpose else core

    def _predict_from_message(
        self,
        representation: torch.Tensor,
        message: torch.Tensor,
        stock_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.base.skip_projection(representation) + message
        embedding = self.base.hidden(fused)
        prediction = self.base.output(embedding).squeeze(-1)
        if stock_mask is not None:
            prediction = prediction.masked_fill(~stock_mask, 0.0)
        return prediction, embedding

    def forward(
        self,
        features: torch.Tensor,
        *,
        stock_mask: torch.Tensor | None = None,
        control: HierarchicalControl = "candidate",
        return_graph: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if control not in self.controls:
            raise ValueError(f"unsupported Phase20 control: {control}")
        representation = self.base.encoder(features)
        parent_graph = self.base.graph(representation, stock_mask=stock_mask)
        fine_sender = parent_graph["S"]
        fine_receiver = parent_graph["R"]
        coarse_sender = torch.softmax(
            self.coarse_sender_projection(representation)
            / self.coarse_role_temperature,
            dim=-1,
        )
        coarse_receiver = torch.softmax(
            self.coarse_receiver_projection(representation)
            / self.coarse_role_temperature,
            dim=-1,
        )
        if stock_mask is not None:
            valid = stock_mask.unsqueeze(-1).to(dtype=representation.dtype)
            coarse_sender = coarse_sender * valid
            coarse_receiver = coarse_receiver * valid

        if control == "sender_role_shift":
            coarse_sender = torch.roll(coarse_sender, shifts=1, dims=-1)
        elif control == "receiver_role_shift":
            coarse_receiver = torch.roll(coarse_receiver, shifts=1, dims=-1)
        elif control == "receiver_collapsed":
            if stock_mask is None:
                collapsed = coarse_receiver.mean(dim=-2, keepdim=True)
            else:
                weight = stock_mask.unsqueeze(-1).to(dtype=coarse_receiver.dtype)
                collapsed = (coarse_receiver * weight).sum(dim=-2, keepdim=True) / (
                    weight.sum(dim=-2, keepdim=True).clamp_min(1.0)
                )
            coarse_receiver = collapsed.expand_as(coarse_receiver)
            if stock_mask is not None:
                coarse_receiver = coarse_receiver * stock_mask.unsqueeze(-1).to(
                    dtype=coarse_receiver.dtype
                )

        coarse_core = self.active_coarse_core(
            transpose=control == "coarse_core_transposed"
        )
        coarse_scores = role_scores(coarse_receiver, coarse_core, coarse_sender)
        centered_coarse_logits = center_over_valid_senders(coarse_scores, stock_mask)
        gamma = self.gamma
        residual_logits = gamma * centered_coarse_logits
        if control == "coarse_off":
            residual_logits = torch.zeros_like(residual_logits)

        fine_logits = parent_graph["scores"] / float(self.base.graph.tau_a)
        scaled_residual, trust = shrink_logit_residual_to_adjacency_kl(
            fine_logits,
            residual_logits,
            budget=self.adjacency_kl_budget,
            stock_mask=stock_mask,
            bisection_steps=self.adjacency_kl_bisection_steps,
        )
        adjacency = trust["candidate_adjacency"]
        if control == "self_graph":
            stock_count = adjacency.shape[-1]
            identity = torch.eye(
                stock_count,
                dtype=adjacency.dtype,
                device=adjacency.device,
            )
            while identity.ndim < adjacency.ndim:
                identity = identity.unsqueeze(0)
            adjacency = identity.expand_as(adjacency)
            if stock_mask is not None:
                valid_pair = stock_mask.unsqueeze(-1) & stock_mask.unsqueeze(-2)
                adjacency = adjacency * valid_pair.to(dtype=adjacency.dtype)
        value = self.base.graph.value_projection(parent_graph["value_input"])
        message = adjacency @ value
        prediction, embedding = self._predict_from_message(
            representation,
            message,
            stock_mask,
        )
        if not return_graph:
            return prediction

        parent_prediction, _ = self._predict_from_message(
            representation,
            parent_graph["message"],
            stock_mask,
        )
        return {
            "prediction": prediction,
            "parent_prediction": parent_prediction,
            "embedding": embedding,
            "representation": representation,
            "A": adjacency,
            "A0": trust["reference_adjacency"],
            "S32": fine_sender,
            "R32": fine_receiver,
            "S8": coarse_sender,
            "R8": coarse_receiver,
            "coarse_sender_projection_weight": self.coarse_sender_projection.weight,
            "coarse_receiver_projection_weight": self.coarse_receiver_projection.weight,
            "coarse_core": coarse_core,
            "fine_logits": fine_logits,
            "centered_coarse_logits": centered_coarse_logits,
            "raw_residual_logits": residual_logits,
            "scaled_residual_logits": scaled_residual,
            "gamma": gamma,
            "raw_adjacency_kl": trust["raw_adjacency_kl"],
            "final_adjacency_kl": trust["final_adjacency_kl"],
            "adjacency_kl_scale": trust["adjacency_kl_scale"],
            "adjacency_kl_budget": trust["adjacency_kl_budget"],
            "message": message,
        }
