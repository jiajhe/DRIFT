from __future__ import annotations

import torch
from torch import nn

from .directed_graph import DirectedRoleGraph


class ResidualFeatureEncoder(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(features + self.residual(features))


class CausalTemporalFeatureEncoder(nn.Module):
    """Fuse a causal per-stock feature history into the FDG role representation."""

    def __init__(
        self,
        d_current: int,
        d_temporal: int,
        lookback: int,
        bottleneck_dim: int,
        temporal_hidden_dim: int,
        dropout: float,
        temporal_arch: str = "gru",
        temporal_num_layers: int = 1,
        temporal_mix_init: float = -2.0,
        temporal_pool: str = "last",
    ) -> None:
        super().__init__()
        if lookback <= 1:
            raise ValueError("temporal encoder requires a lookback greater than one")
        if temporal_arch not in {"gru", "lstm"}:
            raise ValueError("temporal_arch must be gru or lstm")
        if temporal_num_layers <= 0:
            raise ValueError("temporal_num_layers must be positive")
        if temporal_pool not in {"last", "last_attn"}:
            raise ValueError("temporal_pool must be last or last_attn")
        self.lookback = int(lookback)
        self.temporal_dim = int(d_temporal)
        self.temporal_arch = str(temporal_arch)
        self.temporal_pool = str(temporal_pool)
        self.current = ResidualFeatureEncoder(d_current, bottleneck_dim, dropout)
        recurrent = nn.GRU if self.temporal_arch == "gru" else nn.LSTM
        self.temporal = recurrent(
            input_size=d_temporal,
            hidden_size=temporal_hidden_dim,
            num_layers=int(temporal_num_layers),
            batch_first=True,
            dropout=float(dropout) if temporal_num_layers > 1 else 0.0,
        )
        if self.temporal_pool == "last_attn":
            self.temporal_attention = nn.Sequential(
                nn.Linear(temporal_hidden_dim, temporal_hidden_dim),
                nn.Tanh(),
                nn.Linear(temporal_hidden_dim, 1, bias=False),
            )
            projection_dim = 2 * temporal_hidden_dim
        else:
            self.temporal_attention = None
            projection_dim = temporal_hidden_dim
        self.temporal_projection = nn.Linear(projection_dim, d_current)
        self.mix_logit = nn.Parameter(torch.tensor(float(temporal_mix_init)))
        self.norm = nn.LayerNorm(d_current)

    def forward(
        self, current_features: torch.Tensor, temporal_features: torch.Tensor | None
    ) -> torch.Tensor:
        if temporal_features is None:
            raise ValueError("temporal FDG requires causal temporal features")
        if temporal_features.ndim != 4:
            raise ValueError("temporal encoder expects [batch, stock, lookback, feature]")
        if (
            temporal_features.shape[-2] != self.lookback
            or temporal_features.shape[-1] != self.temporal_dim
        ):
            raise ValueError("temporal feature shape differs from model configuration")
        if current_features.shape[:2] != temporal_features.shape[:2]:
            raise ValueError("current and temporal feature batches must share batch and stock axes")
        batch, stocks = temporal_features.shape[:2]
        sequence_input = temporal_features.reshape(
            batch * stocks, *temporal_features.shape[-2:]
        )
        sequence, hidden = self.temporal(sequence_input)
        if self.temporal_arch == "lstm":
            hidden = hidden[0]
        last = hidden[-1]
        if self.temporal_pool == "last_attn":
            if self.temporal_attention is None:
                raise RuntimeError("attention pooling was not initialized")
            attention_logits = self.temporal_attention(sequence).squeeze(-1)
            observed = sequence_input.abs().sum(dim=-1) > 0.0
            # A freshly listed stock may have an all-zero causal history after
            # preprocessing.  Its final (current-day) slot is nevertheless a
            # valid fallback so softmax never receives an all-masked row.
            observed[:, -1] = True
            attention_logits = attention_logits.masked_fill(~observed, -torch.inf)
            weights = torch.softmax(attention_logits, dim=-1)
            pooled = (weights.unsqueeze(-1) * sequence).sum(dim=1)
            temporal_input = torch.cat([last, pooled], dim=-1)
        else:
            temporal_input = last
        temporal = self.temporal_projection(temporal_input).reshape(batch, stocks, -1)
        current = self.current(current_features)
        return self.norm(current + torch.sigmoid(self.mix_logit) * temporal)


class StaticFDGRegressor(nn.Module):
    """Direction-correct static FDG predictor used by Stage-I pretraining."""

    def __init__(
        self,
        d_input: int,
        n_roles: int,
        *,
        bottleneck_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        tau_s: float = 1.0,
        tau_r: float = 1.0,
        tau_a: float = 1.0,
        role_mode: str = "separate",
        core_mode: str = "directed",
        temporal_lookback: int = 1,
        temporal_feature_dim: int | None = None,
        temporal_hidden_dim: int = 64,
        temporal_arch: str = "gru",
        temporal_num_layers: int = 1,
        temporal_mix_init: float = -2.0,
        temporal_pool: str = "last",
    ) -> None:
        super().__init__()
        self.temporal_lookback = int(temporal_lookback)
        if self.temporal_lookback > 1:
            if temporal_feature_dim is None or int(temporal_feature_dim) <= 0:
                raise ValueError("temporal FDG requires a positive temporal feature dimension")
            self.encoder = CausalTemporalFeatureEncoder(
                d_input,
                int(temporal_feature_dim),
                self.temporal_lookback,
                bottleneck_dim,
                int(temporal_hidden_dim),
                dropout,
                temporal_arch=str(temporal_arch),
                temporal_num_layers=int(temporal_num_layers),
                temporal_mix_init=float(temporal_mix_init),
                temporal_pool=str(temporal_pool),
            )
        else:
            self.encoder = ResidualFeatureEncoder(d_input, bottleneck_dim, dropout)
        self.graph = DirectedRoleGraph(
            d_model=d_input,
            n_roles=n_roles,
            tau_s=tau_s,
            tau_r=tau_r,
            tau_a=tau_a,
            role_mode=role_mode,
            core_mode=core_mode,
        )
        self.skip_projection = nn.Linear(d_input, d_input)
        self.hidden = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, 1)

    @property
    def B0(self) -> torch.Tensor:
        return self.graph.B0

    def forward(
        self,
        features: torch.Tensor,
        *,
        stock_mask: torch.Tensor | None = None,
        core: torch.Tensor | None = None,
        temporal_features: torch.Tensor | None = None,
        sender_role_bias: torch.Tensor | None = None,
        receiver_role_bias: torch.Tensor | None = None,
        sender_role_kl_budget: float | torch.Tensor | None = None,
        receiver_role_kl_budget: float | torch.Tensor | None = None,
        role_kl_bisection_steps: int = 16,
        sender_value_scale: torch.Tensor | None = None,
        sender_value_shift: torch.Tensor | None = None,
        return_graph: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.temporal_lookback > 1:
            representation = self.encoder(features, temporal_features)
        else:
            representation = self.encoder(features)
        graph_output = self.graph(
            representation,
            core=core,
            stock_mask=stock_mask,
            sender_role_bias=sender_role_bias,
            receiver_role_bias=receiver_role_bias,
            sender_role_kl_budget=sender_role_kl_budget,
            receiver_role_kl_budget=receiver_role_kl_budget,
            role_kl_bisection_steps=role_kl_bisection_steps,
            sender_value_scale=sender_value_scale,
            sender_value_shift=sender_value_shift,
        )
        fused = self.skip_projection(representation) + graph_output["message"]
        embedding = self.hidden(fused)
        prediction = self.output(embedding).squeeze(-1)
        if stock_mask is not None:
            prediction = prediction.masked_fill(~stock_mask, 0.0)
        if not return_graph:
            return prediction
        return {
            "prediction": prediction,
            "embedding": embedding,
            "representation": representation,
            **graph_output,
        }
