from __future__ import annotations

from dataclasses import dataclass
from math import log10, sqrt

import torch
from torch import nn

from .state_attentive import ConcatenatedTemporalAttentionPooling
from .stateful_flow import entmax15


@dataclass(frozen=True)
class PatternRoutedTemporalOutput:
    """Causal temporal patterns and their stock-level sparse routes."""

    fused: torch.Tensor
    patterns: torch.Tensor
    route: torch.Tensor
    attention: torch.Tensor
    attention_entropy: torch.Tensor
    expected_lag: torch.Tensor
    load_balance: torch.Tensor
    attention_diversity: torch.Tensor
    base_attention_entropy: torch.Tensor | None
    source_score: torch.Tensor | None = None


class PatternRoutedTemporalSourceEncoder(nn.Module):
    """Generate routed temporal information sources from one shared LSTM.

    The module learns a small set of attention slots over the same recurrent
    sequence.  A label-free stock router uses the terminal recurrent state,
    the average slot state, and same-day market moments.  Economic labels are
    intentionally not assigned to slots during training.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        *,
        pattern_count: int = 4,
        attention_dim: int = 64,
        router_hidden_dim: int = 64,
        max_len: int = 60,
        router_temperature: float = 1.0,
        state_fusion: str = "attention_only",
        lag_bias_init: str = "zero",
        local_state_mode: str = "pattern_routed",
        dropout: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        positive = {
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "pattern_count": pattern_count,
            "attention_dim": attention_dim,
            "router_hidden_dim": router_hidden_dim,
            "max_len": max_len,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(pattern_count) < 2:
            raise ValueError("pattern_count must be at least two")
        if float(router_temperature) <= 0.0:
            raise ValueError("router_temperature must be positive")
        if str(state_fusion) not in {
            "attention_only",
            "concat_terminal",
            "residual_terminal",
        }:
            raise ValueError(
                "state_fusion must be attention_only, concat_terminal, "
                "or residual_terminal"
            )
        if str(lag_bias_init) not in {
            "zero",
            "diverse_geometric",
            "terminal_geometric",
        }:
            raise ValueError(
                "lag_bias_init must be zero, diverse_geometric, or terminal_geometric"
            )
        if str(local_state_mode) not in {
            "pattern_routed",
            "concat_terminal_attention",
        }:
            raise ValueError(
                "local_state_mode must be pattern_routed or "
                "concat_terminal_attention"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.pattern_count = int(pattern_count)
        self.attention_dim = int(attention_dim)
        self.max_len = int(max_len)
        self.router_temperature = float(router_temperature)
        self.state_fusion = str(state_fusion)
        self.lag_bias_init = str(lag_bias_init)
        self.local_state_mode = str(local_state_mode)
        self.eps = float(eps)
        self._force_uniform_router = False

        self.history_projection = nn.Linear(
            self.hidden_dim, self.attention_dim, bias=False
        )
        self.current_projection = nn.Linear(
            self.hidden_dim, self.attention_dim, bias=False
        )
        self.attention_queries = nn.Parameter(
            torch.empty(self.pattern_count, self.attention_dim)
        )
        # Column zero is lag zero. This remains causal for every lookback.
        self.lag_bias = nn.Parameter(
            torch.zeros(self.pattern_count, self.max_len)
        )
        self.terminal_pattern_projection: nn.Module | None = None
        self.pattern_output_norm: nn.Module = nn.Identity()
        if self.state_fusion == "residual_terminal":
            self.pattern_projection: nn.Module = nn.Linear(
                self.hidden_dim,
                self.output_dim,
                bias=False,
            )
            self.terminal_pattern_projection = (
                nn.Identity()
                if self.hidden_dim == self.output_dim
                else nn.Linear(self.hidden_dim, self.output_dim, bias=False)
            )
            self.pattern_output_norm = nn.LayerNorm(self.output_dim)
        else:
            pattern_input_dim = (
                2 * self.hidden_dim
                if self.state_fusion == "concat_terminal"
                else self.hidden_dim
            )
            self.pattern_projection = nn.Sequential(
                nn.Linear(pattern_input_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
        self.current_state_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        self.base_temporal_pooling: nn.Module | None = None
        self.base_state_projection: nn.Module | None = None
        if self.local_state_mode == "concat_terminal_attention":
            self.base_temporal_pooling = ConcatenatedTemporalAttentionPooling(
                self.hidden_dim,
                attention_dim=self.hidden_dim,
            )
            self.base_state_projection = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
        router_input_dim = self.hidden_dim + 3 * self.output_dim
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, int(router_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(router_hidden_dim), self.pattern_count),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.attention_queries,
            mean=0.0,
            std=1.0 / sqrt(float(self.attention_dim)),
        )
        if self.lag_bias_init == "zero":
            nn.init.zeros_(self.lag_bias)
        else:
            lag = torch.arange(
                self.max_len,
                dtype=self.lag_bias.dtype,
                device=self.lag_bias.device,
            )
            with torch.no_grad():
                if self.lag_bias_init == "diverse_geometric":
                    scales = torch.logspace(
                        log10(2.0),
                        log10(float(self.max_len)),
                        self.pattern_count,
                        device=self.lag_bias.device,
                    )
                    self.lag_bias.copy_(
                        -lag.unsqueeze(0) / scales.unsqueeze(1)
                    )
                else:
                    self.lag_bias.zero_()
                    self.lag_bias[0].copy_(-8.0 * lag)
                    if self.pattern_count > 1:
                        scales = torch.logspace(
                            log10(2.0),
                            log10(float(self.max_len)),
                            self.pattern_count - 1,
                            device=self.lag_bias.device,
                        )
                        self.lag_bias[1:].copy_(
                            -lag.unsqueeze(0) / scales.unsqueeze(1)
                        )
        if self.state_fusion == "residual_terminal":
            if not isinstance(self.pattern_projection, nn.Linear):
                raise TypeError("residual pattern projection must be linear")
            nn.init.normal_(
                self.pattern_projection.weight,
                mean=0.0,
                std=0.01,
            )
            if isinstance(self.terminal_pattern_projection, nn.Linear):
                nn.init.orthogonal_(self.terminal_pattern_projection.weight)
        final = self.router[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("router output must be linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def set_uniform_router(self, enabled: bool) -> None:
        self._force_uniform_router = bool(enabled)

    @property
    def force_uniform_router(self) -> bool:
        return self._force_uniform_router

    def forward(
        self,
        sequence: torch.Tensor,
        last_state: torch.Tensor,
    ) -> PatternRoutedTemporalOutput:
        if sequence.ndim != 3 or last_state.ndim != 2:
            raise ValueError("temporal source inputs have invalid rank")
        stocks, steps, width = sequence.shape
        if last_state.shape != (stocks, self.hidden_dim):
            raise ValueError("last_state must be [stock, hidden_dim]")
        if width != self.hidden_dim:
            raise ValueError("sequence width does not match hidden_dim")
        if steps < 1 or steps > self.max_len:
            raise ValueError("sequence length must lie in [1, max_len]")
        if stocks < 2:
            raise ValueError("temporal source needs at least two stocks")
        if not torch.isfinite(sequence).all() or not torch.isfinite(last_state).all():
            raise FloatingPointError("temporal source inputs must be finite")

        energy = torch.tanh(
            self.history_projection(sequence)
            + self.current_projection(last_state).unsqueeze(1)
        )
        logits = torch.einsum(
            "nta,ma->nmt", energy, self.attention_queries
        ) / sqrt(float(self.attention_dim))
        lags = torch.arange(
            steps - 1, -1, -1, device=sequence.device, dtype=torch.long
        )
        logits = logits + self.lag_bias[:, lags].unsqueeze(0)
        attention = torch.softmax(logits, dim=-1)
        contexts = torch.einsum("nmt,nth->nmh", attention, sequence)
        if self.state_fusion == "concat_terminal":
            terminal = last_state[:, None, :].expand(
                stocks, self.pattern_count, self.hidden_dim
            )
            pattern_inputs = torch.cat([terminal, contexts], dim=-1)
            patterns = self.pattern_projection(pattern_inputs)
        elif self.state_fusion == "residual_terminal":
            if self.terminal_pattern_projection is None:
                raise TypeError("terminal pattern projection is missing")
            terminal = self.terminal_pattern_projection(last_state)
            patterns = self.pattern_output_norm(
                terminal[:, None, :]
                + self.pattern_projection(contexts)
            )
        else:
            patterns = self.pattern_projection(contexts)

        current = self.current_state_projection(last_state)
        market_mean = current.mean(dim=0)
        market_scale = torch.sqrt(
            current.var(dim=0, unbiased=False) + self.eps
        )
        market = torch.cat([market_mean, market_scale], dim=-1)
        router_input = torch.cat(
            [
                last_state,
                patterns.mean(dim=1),
                market.unsqueeze(0).expand(stocks, -1),
            ],
            dim=-1,
        )
        route_logits = self.router(router_input) / self.router_temperature
        if self._force_uniform_router:
            route = route_logits.new_full(
                route_logits.shape, 1.0 / float(self.pattern_count)
            )
        else:
            route = entmax15(route_logits, dim=-1)
        routed_patterns = torch.einsum("nm,nmd->nd", route, patterns)
        base_attention_entropy: torch.Tensor | None = None
        if self.local_state_mode == "concat_terminal_attention":
            if (
                self.base_temporal_pooling is None
                or self.base_state_projection is None
            ):
                raise TypeError("base concat-attention modules are missing")
            base_concat, base_attention_entropy = self.base_temporal_pooling(
                sequence,
                last_state,
            )
            fused = self.base_state_projection(base_concat)
        else:
            fused = routed_patterns

        attention_entropy = -(
            attention * attention.clamp_min(1e-12).log()
        ).sum(dim=-1)
        expected_lag = torch.einsum(
            "nmt,t->nm", attention, lags.to(dtype=attention.dtype)
        )
        mean_load = route.mean(dim=0)
        load_balance = (
            mean_load - 1.0 / float(self.pattern_count)
        ).square().sum()
        overlaps = attention @ attention.transpose(1, 2)
        off_diagonal = ~torch.eye(
            self.pattern_count,
            dtype=torch.bool,
            device=attention.device,
        )
        attention_diversity = overlaps[:, off_diagonal].square().mean()
        return PatternRoutedTemporalOutput(
            fused=fused,
            patterns=patterns,
            route=route,
            attention=attention,
            attention_entropy=attention_entropy,
            expected_lag=expected_lag,
            load_balance=load_balance,
            attention_diversity=attention_diversity,
            base_attention_entropy=base_attention_entropy,
        )


__all__ = [
    "PatternRoutedTemporalOutput",
    "PatternRoutedTemporalSourceEncoder",
]
