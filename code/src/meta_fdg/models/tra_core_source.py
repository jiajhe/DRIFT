from __future__ import annotations

from math import sqrt
from typing import Mapping

import torch
from torch import nn

from .pattern_routed_source import PatternRoutedTemporalOutput


class TRACoreRoutedTemporalSourceEncoder(nn.Module):
    """Reuse TRA's LSTM-attention and latent-representation routing core.

    Official TRA combines the recurrent latent representation with a causal
    history of per-state prediction errors.  Flow-FDG already owns the causal
    recurrent market state, so this adapter deliberately reuses the stable LR
    half of the TRA router and omits the separate error-memory/TPE branch.  The
    three TRA predictors seed three state-specific representations which are
    routed into one local state and one shared Flow reservoir.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        *,
        state_count: int = 3,
        router_temperature: float = 1.0,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError("TRA core dimensions must be positive")
        if int(state_count) < 2:
            raise ValueError("TRA core needs at least two latent states")
        if float(router_temperature) <= 0.0:
            raise ValueError("TRA router temperature must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.state_count = int(state_count)
        self.router_temperature = float(router_temperature)
        self.eps = float(eps)
        self._force_uniform_router = False

        # These names and shapes match the official TRA LSTM attention W/u.
        self.attention_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_score = nn.Linear(self.hidden_dim, 1, bias=False)
        tra_hidden_dim = 2 * self.hidden_dim
        self.state_predictors = nn.Linear(tra_hidden_dim, self.state_count)
        self.latent_router = nn.Linear(tra_hidden_dim, self.state_count)
        self.state_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(tra_hidden_dim + 1, self.output_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.LayerNorm(self.output_dim),
                )
                for _ in range(self.state_count)
            ]
        )

    def set_uniform_router(self, enabled: bool) -> None:
        self._force_uniform_router = bool(enabled)

    @property
    def force_uniform_router(self) -> bool:
        return self._force_uniform_router

    def load_official_tra_state(
        self,
        model_state: Mapping[str, torch.Tensor],
        tra_state: Mapping[str, torch.Tensor],
    ) -> dict[str, object]:
        """Load attention, predictors, and the LR half of an official TRA router."""

        required_model = {"W.weight", "W.bias", "u.weight"}
        required_tra = {
            "predictors.weight",
            "predictors.bias",
            "fc.weight",
            "fc.bias",
        }
        if not required_model <= set(model_state):
            raise ValueError("official TRA checkpoint is missing attention tensors")
        if not required_tra <= set(tra_state):
            raise ValueError("official TRA checkpoint is missing routing tensors")

        with torch.no_grad():
            self.attention_projection.weight.copy_(model_state["W.weight"])
            self.attention_projection.bias.copy_(model_state["W.bias"])
            self.attention_score.weight.copy_(model_state["u.weight"])
            self.state_predictors.weight.copy_(tra_state["predictors.weight"])
            self.state_predictors.bias.copy_(tra_state["predictors.bias"])

            fc_weight = tra_state["fc.weight"]
            expected_latent = 2 * self.hidden_dim
            if fc_weight.shape[0] != self.state_count:
                raise ValueError("TRA router state count does not match the model")
            if fc_weight.shape[1] <= expected_latent:
                raise ValueError("TRA router does not contain a TPE plus LR input")
            latent_weight = fc_weight[:, -expected_latent:]
            self.latent_router.weight.copy_(latent_weight)
            self.latent_router.bias.copy_(tra_state["fc.bias"])

        return {
            "state_count": self.state_count,
            "hidden_dim": self.hidden_dim,
            "attention_loaded": True,
            "predictors_loaded": True,
            "router_component": "latent_representation_only",
            "state_score_anchored": True,
            "source_score_axis": 0,
            "source_router_tpe_dim": int(fc_weight.shape[1] - expected_latent),
        }

    def forward(
        self,
        sequence: torch.Tensor,
        last_state: torch.Tensor,
    ) -> PatternRoutedTemporalOutput:
        if sequence.ndim != 3 or last_state.ndim != 2:
            raise ValueError("TRA core inputs have invalid rank")
        stocks, steps, width = sequence.shape
        if width != self.hidden_dim:
            raise ValueError("TRA recurrent sequence width is invalid")
        if last_state.shape != (stocks, self.hidden_dim):
            raise ValueError("TRA last state must be [stock, hidden_dim]")
        if stocks < 2 or steps < 1:
            raise ValueError("TRA core needs at least two stocks and one step")
        if not torch.isfinite(sequence).all() or not torch.isfinite(last_state).all():
            raise FloatingPointError("TRA core inputs must be finite")

        attention_logits = self.attention_score(
            torch.tanh(self.attention_projection(sequence))
        ).squeeze(-1)
        base_attention = torch.softmax(attention_logits, dim=1)
        attention_context = torch.einsum(
            "nt,nth->nh", base_attention, sequence
        )
        latent = torch.cat([last_state, attention_context], dim=-1)
        state_scores = self.state_predictors(latent)
        route_logits = self.latent_router(latent) / self.router_temperature
        if self._force_uniform_router:
            route = route_logits.new_full(
                route_logits.shape, 1.0 / float(self.state_count)
            )
        else:
            # Soft routing preserves TRA's mixture semantics without injecting
            # Gumbel noise into the already stateful Flow recurrence.
            route = torch.softmax(route_logits, dim=-1)
        source_score = torch.einsum("nm,nm->n", route, state_scores)

        patterns = torch.stack(
            [
                projection(
                    torch.cat([latent, state_scores[:, index : index + 1]], dim=-1)
                )
                for index, projection in enumerate(self.state_projections)
            ],
            dim=1,
        )
        # Preserve the strongest train/Valid-only transferable TRA signal in a
        # named representation coordinate.  This is not a prediction residual:
        # the unified Flow-FDG head still owns the final score, while roles and
        # Flow can consume and update the routed state-specific information.
        patterns = torch.cat(
            [state_scores.unsqueeze(-1), patterns[..., 1:]],
            dim=-1,
        )
        fused = torch.einsum("nm,nmd->nd", route, patterns)

        attention = base_attention[:, None, :].expand(
            stocks, self.state_count, steps
        )
        base_entropy = -(
            base_attention * base_attention.clamp_min(1e-12).log()
        ).sum(dim=-1)
        attention_entropy = base_entropy[:, None].expand(stocks, self.state_count)
        lags = torch.arange(
            steps - 1, -1, -1, device=sequence.device, dtype=sequence.dtype
        )
        expected = torch.einsum("nt,t->n", base_attention, lags)
        expected_lag = expected[:, None].expand(stocks, self.state_count)
        mean_load = route.mean(dim=0)
        load_balance = (
            mean_load - 1.0 / float(self.state_count)
        ).square().sum()
        attention_diversity = base_attention.new_ones(())

        return PatternRoutedTemporalOutput(
            fused=fused,
            patterns=patterns,
            route=route,
            attention=attention,
            attention_entropy=attention_entropy,
            expected_lag=expected_lag,
            load_balance=load_balance,
            attention_diversity=attention_diversity,
            base_attention_entropy=base_entropy,
            source_score=source_score,
        )


__all__ = ["TRACoreRoutedTemporalSourceEncoder"]
