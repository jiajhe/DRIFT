from __future__ import annotations

from math import sqrt

import torch
from torch import nn


class StateAttentiveTemporalPooling(nn.Module):
    """Route causal multi-horizon attention with the current market state.

    Each attention head is restricted to a fixed trailing horizon. A label-free
    router combines the stock-local terminal state with the same-day market
    mean and scale, then chooses how much each stock uses each horizon.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        horizons: tuple[int, ...] = (5, 20, 60),
        attention_dim: int = 64,
        router_hidden_dim: int = 64,
        router_temperature: float = 1.0,
        mix_init: float = -1.0,
        identity_residual: bool = False,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        resolved_horizons = tuple(int(value) for value in horizons)
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if len(resolved_horizons) < 2 or any(
            value <= 0 for value in resolved_horizons
        ):
            raise ValueError(
                "state-attentive pooling needs at least two positive horizons"
            )
        if tuple(sorted(set(resolved_horizons))) != resolved_horizons:
            raise ValueError("attention horizons must be strictly increasing")
        if int(attention_dim) <= 0 or int(router_hidden_dim) <= 0:
            raise ValueError("attention and router dimensions must be positive")
        if float(router_temperature) <= 0.0:
            raise ValueError("router_temperature must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.hidden_dim = int(hidden_dim)
        self.horizons = resolved_horizons
        self.state_count = len(self.horizons)
        self.attention_dim = int(attention_dim)
        self.router_temperature = float(router_temperature)
        self.identity_residual = bool(identity_residual)
        self.eps = float(eps)

        self.attention_key = nn.Linear(
            self.hidden_dim, self.attention_dim, bias=False
        )
        self.attention_query = nn.Linear(
            self.hidden_dim,
            self.state_count * self.attention_dim,
            bias=False,
        )
        self.local_router = nn.Linear(
            self.hidden_dim, self.state_count, bias=False
        )
        self.market_router = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, int(router_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(router_hidden_dim), self.state_count),
        )
        self.context_projection = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.residual_logit = nn.Parameter(
            torch.tensor(
                0.0 if self.identity_residual else float(mix_init)
            )
        )
        self.dropout = nn.Dropout(float(dropout))
        self.output_norm: nn.Module = (
            nn.Identity()
            if self.identity_residual
            else nn.LayerNorm(self.hidden_dim)
        )

    def forward(
        self,
        sequence: torch.Tensor,
        last_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or last_state.ndim != 2:
            raise ValueError(
                "recurrent sequence and last state have invalid rank"
            )
        if sequence.shape[0] != last_state.shape[0]:
            raise ValueError(
                "recurrent sequence and last state disagree on stocks"
            )
        if sequence.shape[2] != self.hidden_dim:
            raise ValueError(
                "recurrent sequence width does not match hidden_dim"
            )
        if last_state.shape[1] != self.hidden_dim:
            raise ValueError("last state width does not match hidden_dim")

        stocks, steps, _ = sequence.shape
        keys = torch.tanh(self.attention_key(sequence))
        queries = self.attention_query(last_state).reshape(
            stocks,
            self.state_count,
            self.attention_dim,
        )
        attention_logits = torch.einsum(
            "nta,nka->ntk",
            keys,
            queries,
        ) / sqrt(float(self.attention_dim))
        positions = torch.arange(steps, device=sequence.device)
        allowed = torch.stack(
            [
                positions >= steps - min(int(horizon), steps)
                for horizon in self.horizons
            ],
            dim=-1,
        )
        attention_logits = attention_logits.masked_fill(
            ~allowed.unsqueeze(0),
            -torch.inf,
        )
        attention = torch.softmax(attention_logits, dim=1)
        contexts = torch.einsum("ntk,nth->nkh", attention, sequence)

        market_mean = last_state.mean(dim=0)
        market_scale = torch.sqrt(
            last_state.var(dim=0, unbiased=False) + self.eps
        )
        market_features = torch.cat([market_mean, market_scale], dim=-1)
        route_logits = self.local_router(last_state) + self.market_router(
            market_features
        ).unsqueeze(0)
        route = torch.softmax(
            route_logits / self.router_temperature,
            dim=-1,
        )
        routed_context = torch.einsum("nk,nkh->nh", route, contexts)
        mix = (
            torch.tanh(self.residual_logit)
            if self.identity_residual
            else torch.sigmoid(self.residual_logit)
        )
        pooled = self.output_norm(
            last_state
            + mix * self.dropout(self.context_projection(routed_context))
        )
        attention_entropy = -(
            attention * attention.clamp_min(1e-12).log()
        ).sum(dim=1)
        return pooled, route, attention_entropy


class ConcatenatedTemporalAttentionPooling(nn.Module):
    """Concatenate the recurrent state with a causal attention summary."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        attention_dim: int | None = None,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        resolved_attention_dim = (
            int(hidden_dim)
            if attention_dim is None
            else int(attention_dim)
        )
        if resolved_attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = resolved_attention_dim
        self.key_projection = nn.Linear(
            self.hidden_dim,
            self.attention_dim,
        )
        self.score_projection = nn.Linear(
            self.attention_dim,
            1,
            bias=False,
        )

    def forward(
        self,
        sequence: torch.Tensor,
        layer_mean_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or layer_mean_state.ndim != 2:
            raise ValueError("attention inputs have invalid rank")
        if sequence.shape[0] != layer_mean_state.shape[0]:
            raise ValueError("attention inputs disagree on stocks")
        if sequence.shape[2] != self.hidden_dim:
            raise ValueError("sequence width does not match hidden_dim")
        if layer_mean_state.shape[1] != self.hidden_dim:
            raise ValueError("state width does not match hidden_dim")
        logits = self.score_projection(
            torch.tanh(self.key_projection(sequence))
        ).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        context = torch.einsum("nt,nth->nh", weights, sequence)
        entropy = -(
            weights * weights.clamp_min(1e-12).log()
        ).sum(dim=1)
        return torch.cat([layer_mean_state, context], dim=-1), entropy
