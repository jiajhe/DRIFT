from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn

from .directed_graph import DirectedRoleGraph
from .dynamic_route import center_and_bound_role_bias


class PackedCausalTemporalEncoder(nn.Module):
    """Encode one causal, right-padded history per stock.

    The caller is responsible for applying the training-window RobustZ
    transform.  Packing makes every encoded stock depend only on its declared
    prefix, while the attention pool is explicitly masked to the same prefix.
    """

    def __init__(
        self,
        d_current: int,
        d_temporal: int,
        *,
        d_model: int = 128,
        lookback: int = 60,
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(
            int(d_current),
            int(d_temporal),
            int(d_model),
            int(lookback),
            int(temporal_hidden_dim),
            int(temporal_num_layers),
        ) <= 0:
            raise ValueError("all concept-temporal encoder dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.d_current = int(d_current)
        self.d_temporal = int(d_temporal)
        self.d_model = int(d_model)
        self.lookback = int(lookback)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.temporal_num_layers = int(temporal_num_layers)

        self.current_projection = nn.Sequential(
            nn.Linear(self.d_current, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.temporal = nn.LSTM(
            input_size=self.d_temporal,
            hidden_size=self.temporal_hidden_dim,
            num_layers=self.temporal_num_layers,
            batch_first=True,
            dropout=float(dropout) if self.temporal_num_layers > 1 else 0.0,
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.temporal_hidden_dim, self.temporal_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.temporal_hidden_dim, 1, bias=False),
        )
        self.temporal_projection = nn.Linear(
            2 * self.temporal_hidden_dim,
            self.d_model,
        )
        self.temporal_dropout = nn.Dropout(float(dropout))
        self.fusion_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        current_features: torch.Tensor,
        temporal_features: torch.Tensor,
        temporal_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if current_features.ndim < 2:
            raise ValueError("current features must end in [stock, feature]")
        if current_features.shape[-1] != self.d_current:
            raise ValueError("current feature dimension differs from model configuration")
        if temporal_features.ndim != current_features.ndim + 1:
            raise ValueError(
                "temporal features must append [lookback, feature] to current stock axes"
            )
        if temporal_features.shape[:-2] != current_features.shape[:-1]:
            raise ValueError("current and temporal features must share all stock axes")
        if temporal_features.shape[-2:] != (self.lookback, self.d_temporal):
            raise ValueError("temporal feature shape differs from model configuration")
        if temporal_lengths.shape != current_features.shape[:-1]:
            raise ValueError("temporal lengths must match current stock axes")

        flat_history = temporal_features.reshape(
            -1,
            self.lookback,
            self.d_temporal,
        )
        flat_lengths = temporal_lengths.reshape(-1).to(
            device=flat_history.device,
            dtype=torch.long,
        )
        if torch.any(flat_lengths < 0) or torch.any(flat_lengths > self.lookback):
            raise ValueError("temporal lengths must lie in [0, lookback]")

        # pack_padded_sequence cannot accept a zero length.  A zero-history
        # stock gets a synthetic all-zero step which is removed again below.
        has_history = flat_lengths > 0
        packed_lengths = flat_lengths.clamp_min(1)
        safe_history = torch.where(
            has_history[:, None, None],
            flat_history,
            torch.zeros_like(flat_history),
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            safe_history,
            packed_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_sequence, (hidden, _) = self.temporal(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(
            packed_sequence,
            batch_first=True,
            total_length=self.lookback,
        )
        last = hidden[-1]

        positions = torch.arange(self.lookback, device=flat_history.device)
        observed = positions.unsqueeze(0) < packed_lengths.unsqueeze(1)
        attention_logits = self.temporal_attention(sequence).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~observed, -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        pooled = (attention.unsqueeze(-1) * sequence).sum(dim=1)
        temporal = self.temporal_projection(torch.cat([last, pooled], dim=-1))
        temporal = self.temporal_dropout(temporal)
        temporal = temporal * has_history.unsqueeze(-1).to(dtype=temporal.dtype)

        current = self.current_projection(current_features)
        temporal = temporal.reshape(*current_features.shape[:-1], self.d_model)
        return self.fusion_norm(current + temporal)


class FixedConceptRoleBias(nn.Module):
    """Project fixed stock concepts into sender/receiver latent-role biases.

    The fixed matrix is a buffer, never a predictor input.  Row zero must
    already be the asset's reserved all-zero unknown-stock sentinel.  A shared
    trainable concept embedding is averaged with row-normalised fixed weights;
    only the final sender/receiver role projections are separate.
    """

    def __init__(
        self,
        concept_matrix: torch.Tensor,
        n_roles: int,
        *,
        concept_embedding_dim: int = 32,
        max_logit_bias: float = 0.15,
    ) -> None:
        super().__init__()
        matrix = torch.as_tensor(concept_matrix, dtype=torch.float32)
        if matrix.ndim != 2 or min(matrix.shape) <= 0:
            raise ValueError("concept matrix must be a non-empty [stock, concept] tensor")
        if not torch.isfinite(matrix).all():
            raise ValueError("concept matrix must be finite")
        if torch.any(matrix < 0):
            raise ValueError("concept matrix weights must be non-negative")
        if torch.count_nonzero(matrix[0]).item() != 0:
            raise ValueError("concept row zero must be the all-zero unknown sentinel")
        if min(int(n_roles), int(concept_embedding_dim)) <= 0:
            raise ValueError("n_roles and concept_embedding_dim must be positive")
        if not math.isfinite(float(max_logit_bias)) or float(max_logit_bias) <= 0:
            raise ValueError("max_logit_bias must be finite and positive")
        self.concept_dim = int(matrix.shape[1])
        self.n_roles = int(n_roles)
        self.concept_embedding_dim = int(concept_embedding_dim)
        self.max_logit_bias = float(max_logit_bias)
        self.register_buffer("concept_matrix", matrix.clone())
        self.concept_embedding = nn.Parameter(
            torch.empty(self.concept_dim, self.concept_embedding_dim)
        )
        nn.init.xavier_uniform_(self.concept_embedding)
        self.sender_projection = nn.Linear(
            self.concept_embedding_dim,
            self.n_roles,
            bias=False,
        )
        self.receiver_projection = nn.Linear(
            self.concept_embedding_dim,
            self.n_roles,
            bias=False,
        )

    @property
    def unknown_row(self) -> int:
        return 0

    def _resolve_rows(self, concept_rows: torch.Tensor) -> torch.Tensor:
        if concept_rows.dtype == torch.bool or concept_rows.dtype.is_floating_point:
            raise TypeError("concept rows must contain integer indices")
        rows = concept_rows.to(device=self.concept_matrix.device, dtype=torch.long)
        if torch.any(rows < -1) or torch.any(rows >= int(self.concept_matrix.shape[0])):
            raise ValueError("concept rows must index a known row, -1, or the unknown row")
        return torch.where(rows == -1, self.unknown_row, rows)

    def embed_rows(
        self,
        concept_rows: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        rows = self._resolve_rows(concept_rows)
        weights = self.concept_matrix.index_select(0, rows.reshape(-1))
        weights = weights.reshape(*rows.shape, self.concept_dim)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        target_device = self.concept_embedding.device if device is None else device
        target_dtype = self.concept_embedding.dtype if dtype is None else dtype
        weights = weights.to(device=target_device, dtype=target_dtype)
        embedding = self.concept_embedding.to(
            device=target_device,
            dtype=target_dtype,
        )
        return weights @ embedding

    def forward(
        self,
        concept_rows: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concepts = self.embed_rows(concept_rows, dtype=dtype, device=device)
        sender = center_and_bound_role_bias(
            self.sender_projection(concepts),
            self.max_logit_bias,
        )
        receiver = center_and_bound_role_bias(
            self.receiver_projection(concepts),
            self.max_logit_bias,
        )
        return sender, receiver


class ConceptTemporalSEFDGRegressor(nn.Module):
    """A single-head temporal predictor whose only stock mixing is SE-FDG.

    Concepts are structurally isolated from the temporal representation,
    value path, skip path, and prediction head.  In directed mode they can
    change only sender/receiver latent-role logits.  An optional per-day KL
    trust region constrains the resulting role movement relative to the
    concept-off roles, rather than relying on the logit cap alone.
    ``graph_mode='self'`` skips both concept lookup and role construction and
    uses the graph value projection stock-by-stock.
    """

    def __init__(
        self,
        d_current: int,
        d_temporal: int,
        n_roles: int,
        concept_matrix: torch.Tensor,
        *,
        d_model: int = 128,
        lookback: int = 60,
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 2,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
        tau_s: float = 1.0,
        tau_r: float = 1.0,
        tau_a: float = 1.0,
        graph_mode: str = "directed",
        concept_scale: float = 1.0,
        concept_embedding_dim: int = 32,
        max_concept_logit_bias: float = 0.15,
        concept_role_kl_budget: float | None = None,
    ) -> None:
        super().__init__()
        if graph_mode not in {"directed", "self"}:
            raise ValueError("graph_mode must be directed or self")
        if int(head_hidden_dim) <= 0:
            raise ValueError("head_hidden_dim must be positive")
        if (
            not isinstance(concept_scale, Real)
            or not math.isfinite(float(concept_scale))
            or not 0.0 <= float(concept_scale) <= 1.0
        ):
            raise ValueError("concept_scale must be a finite scalar in [0, 1]")
        self.n_roles = int(n_roles)
        self.d_model = int(d_model)
        self.graph_mode = str(graph_mode)
        self.concept_scale = float(concept_scale)
        if concept_role_kl_budget is not None and (
            not isinstance(concept_role_kl_budget, Real)
            or not math.isfinite(float(concept_role_kl_budget))
            or float(concept_role_kl_budget) < 0.0
        ):
            raise ValueError("concept_role_kl_budget must be finite and non-negative")
        self.concept_role_kl_budget = (
            None
            if concept_role_kl_budget is None
            else float(concept_role_kl_budget)
        )

        self.encoder = PackedCausalTemporalEncoder(
            d_current=d_current,
            d_temporal=d_temporal,
            d_model=self.d_model,
            lookback=lookback,
            temporal_hidden_dim=temporal_hidden_dim,
            temporal_num_layers=temporal_num_layers,
            dropout=dropout,
        )
        self.concept_role_bias = FixedConceptRoleBias(
            concept_matrix,
            self.n_roles,
            concept_embedding_dim=concept_embedding_dim,
            max_logit_bias=max_concept_logit_bias,
        )
        self.graph = DirectedRoleGraph(
            d_model=self.d_model,
            n_roles=self.n_roles,
            tau_s=tau_s,
            tau_r=tau_r,
            tau_a=tau_a,
            role_mode="separate",
            core_mode="directed",
        )
        self.skip_projection = nn.Linear(self.d_model, self.d_model)
        self.graph_fusion_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, int(head_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_dim), 1),
        )

    @property
    def B0(self) -> torch.Tensor:
        return self.graph.B0

    @property
    def unknown_concept_row(self) -> int:
        return self.concept_role_bias.unknown_row

    @staticmethod
    def _scale_tensor(
        concept_scale: float | torch.Tensor,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.as_tensor(
            concept_scale,
            dtype=reference.dtype,
            device=reference.device,
        )
        if (
            scale.numel() != 1
            or not torch.isfinite(scale).all()
            or bool((scale < 0.0).any())
            or bool((scale > 1.0).any())
        ):
            raise ValueError("concept_scale must be one finite scalar in [0, 1]")
        return scale.reshape(())

    def _self_graph_output(
        self,
        representation: torch.Tensor,
        stock_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        stock_count = int(representation.shape[-2])
        leading = representation.shape[:-2]
        dtype = representation.dtype
        device = representation.device
        if stock_mask is not None and stock_mask.shape != representation.shape[:-1]:
            raise ValueError("stock_mask must match representation stock axes")

        value = self.graph.value_projection(representation)
        if stock_mask is None:
            valid = torch.ones(
                representation.shape[:-1],
                dtype=dtype,
                device=device,
            )
        else:
            valid = stock_mask.to(dtype=dtype, device=device)
        message = value * valid.unsqueeze(-1)
        eye = torch.eye(stock_count, dtype=dtype, device=device)
        adjacency = eye.expand(*leading, stock_count, stock_count)
        adjacency = adjacency * valid.unsqueeze(-1) * valid.unsqueeze(-2)

        roles = representation.new_zeros(*leading, stock_count, self.n_roles)
        scores = representation.new_zeros(*leading, stock_count, stock_count)
        daily = representation.new_zeros(*leading)
        core = representation.new_zeros(self.n_roles, self.n_roles)
        return {
            "A": adjacency,
            "S": roles,
            "R": roles,
            "S0": roles,
            "R0": roles,
            "sender_route_raw_kl": daily,
            "sender_route_kl": daily,
            "sender_route_kl_scale": daily,
            "receiver_route_raw_kl": daily,
            "receiver_route_kl": daily,
            "receiver_route_kl_scale": daily,
            "core": core,
            "scores": scores,
            "value_input": representation,
            "message": message,
            "concept_sender_bias": roles,
            "concept_receiver_bias": roles,
        }

    def forward(
        self,
        current_features: torch.Tensor,
        temporal_features: torch.Tensor,
        temporal_lengths: torch.Tensor,
        *,
        concept_rows: torch.Tensor | None = None,
        stock_mask: torch.Tensor | None = None,
        core: torch.Tensor | None = None,
        graph_mode: str | None = None,
        concept_scale: float | torch.Tensor | None = None,
        return_graph: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        mode = self.graph_mode if graph_mode is None else str(graph_mode)
        if mode not in {"directed", "self"}:
            raise ValueError("graph_mode must be directed or self")
        representation = self.encoder(
            current_features,
            temporal_features,
            temporal_lengths,
        )

        if mode == "self":
            # Deliberately branch before inspecting concept_rows, concept_scale,
            # role projections, or B.  The self control is therefore an exact
            # stock-wise path even when those ignored inputs are permuted.
            graph_output = self._self_graph_output(representation, stock_mask)
        else:
            selected_scale = self.concept_scale if concept_scale is None else concept_scale
            scale = self._scale_tensor(selected_scale, reference=representation)
            if bool(scale.detach().cpu().item() == 0.0):
                sender_bias = None
                receiver_bias = None
                zero_bias = representation.new_zeros(
                    *representation.shape[:-1],
                    self.n_roles,
                )
            else:
                if concept_rows is None:
                    raise ValueError("directed concept routing requires concept_rows")
                if concept_rows.shape != representation.shape[:-1]:
                    raise ValueError("concept_rows must match representation stock axes")
                sender_bias, receiver_bias = self.concept_role_bias(
                    concept_rows,
                    dtype=representation.dtype,
                    device=representation.device,
                )
                sender_bias = scale * sender_bias
                receiver_bias = scale * receiver_bias
                zero_bias = None
            graph_output = self.graph(
                representation,
                core=core,
                stock_mask=stock_mask,
                sender_role_bias=sender_bias,
                receiver_role_bias=receiver_bias,
                sender_role_kl_budget=self.concept_role_kl_budget,
                receiver_role_kl_budget=self.concept_role_kl_budget,
            )
            graph_output["concept_sender_bias"] = (
                zero_bias if sender_bias is None else sender_bias
            )
            graph_output["concept_receiver_bias"] = (
                zero_bias if receiver_bias is None else receiver_bias
            )

        skip = self.skip_projection(representation)
        fused = self.graph_fusion_norm(skip + graph_output["message"])
        prediction = self.head(fused).squeeze(-1)
        if stock_mask is not None:
            if stock_mask.shape != prediction.shape:
                raise ValueError("stock_mask must match prediction stock axes")
            prediction = prediction.masked_fill(~stock_mask, 0.0)
        if not return_graph:
            return prediction
        return {
            "prediction": prediction,
            "representation": representation,
            "skip": skip,
            "fused": fused,
            **graph_output,
        }
