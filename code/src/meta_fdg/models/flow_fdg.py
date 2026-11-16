from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import torch
from torch import nn

from .factor_axis_lstm import (
    ALPHA360_LAST_SNAPSHOT,
    orient_factor_axis_features,
)
from .hmc_flow import (
    HierarchicalMarketRoleResidual,
    InnovationSourceResidual,
    StrictNormBoundedFusion,
)
from .pattern_routed_source import PatternRoutedTemporalSourceEncoder
from .state_attentive import (
    ConcatenatedTemporalAttentionPooling,
    StateAttentiveTemporalPooling,
)
from .stateful_flow import entmax15
from .tra_core_source import TRACoreRoutedTemporalSourceEncoder


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} must contain only finite values")


def center_stocks(values: torch.Tensor) -> torch.Tensor:
    """Remove the cross-sectional market mode along the first (stock) axis."""

    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("values must have a non-empty leading stock axis")
    _require_finite("cross-sectional values", values)
    return values - values.mean(dim=0, keepdim=True)


def _initialize_gelu_axis_identity_head(
    head: nn.Sequential,
    *,
    axis: int,
) -> dict[str, object]:
    """Initialize a two-layer GELU head to return one input coordinate.

    For exact GELU, ``GELU(x) - GELU(-x) == x``.  An antipodal pair therefore
    preserves the pre-trained TRA score through the unified model head without
    adding a prediction residual or a second output head.
    """

    if len(head) != 4:
        raise ValueError("source-score head must be Linear/GELU/Dropout/Linear")
    first, activation, dropout, final = head
    if (
        not isinstance(first, nn.Linear)
        or not isinstance(activation, nn.GELU)
        or not isinstance(dropout, nn.Dropout)
        or not isinstance(final, nn.Linear)
        or first.out_features < 2
        or final.in_features != first.out_features
        or final.out_features != 1
        or not 0 <= int(axis) < first.in_features
    ):
        raise ValueError("source-score head has an incompatible structure")
    if float(dropout.p) != 0.0:
        raise ValueError("exact source-score head initialization requires dropout=0")
    with torch.no_grad():
        first.weight.zero_()
        first.bias.zero_()
        final.weight.zero_()
        final.bias.zero_()
        first.weight[0, int(axis)] = 1.0
        first.weight[1, int(axis)] = -1.0
        final.weight[0, 0] = 1.0
        final.weight[0, 1] = -1.0
    return {
        "mode": "gelu_antipodal_identity",
        "source_score_axis": int(axis),
        "prediction_residual": False,
        "exact_in_eval_mode": True,
    }


def _initialize_strict_source_anchor(
    fusion: StrictNormBoundedFusion,
) -> dict[str, object]:
    """Make the first unified prediction exactly equal to the TRA source.

    The strict fusion already constrains every learned Flow correction in
    representation norm.  Zeroing its Flow projection makes the initial
    correction exactly zero while retaining a nonzero gradient for subsequent
    Train/Valid-only optimization.
    """

    if not isinstance(fusion, StrictNormBoundedFusion):
        raise TypeError("source anchoring requires strict norm-bounded fusion")
    with torch.no_grad():
        fusion.flow_projection.weight.zero_()
        fusion.interaction_projection.weight.zero_()
    return {
        "mode": "zero_initialized_strict_norm_flow_delta",
        "initial_flow_delta_exactly_zero": True,
        "maximum_relative_flow_delta": float(fusion.max_fraction),
        "prediction_residual": False,
        "trainable_after_initialization": True,
    }


def helmert_contrast(role_count: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return a deterministic orthonormal basis of the role-contrast subspace."""

    role_count = int(role_count)
    if role_count < 2:
        raise ValueError("role_count must be at least two")
    basis = torch.zeros(role_count, role_count - 1, dtype=dtype)
    for column in range(role_count - 1):
        width = column + 1
        denominator = sqrt(float(width * (width + 1)))
        basis[:width, column] = 1.0 / denominator
        basis[width, column] = -float(width) / denominator
    return basis


def implicit_offdiag_apply(
    receiver_coordinates: torch.Tensor,
    role_operator: torch.Tensor,
    sender_coordinates: torch.Tensor,
    values: torch.Tensor,
    gamma: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the normalized zero-diagonal operator without forming ``N x N``.

    ``receiver_coordinates @ role_operator @ sender_coordinates.T`` obeys the
    repository convention: row ``i`` is receiver stock ``i`` and column ``j``
    is sender stock ``j``.  The returned tuple contains the transformed values,
    the removed raw diagonal, the off-diagonal Frobenius norm, and the scalar
    normalization factor.
    """

    if receiver_coordinates.ndim != 2 or sender_coordinates.ndim != 2:
        raise ValueError("role coordinates must be rank-two [stock, contrast]")
    if receiver_coordinates.shape != sender_coordinates.shape:
        raise ValueError("sender and receiver coordinates must share shape")
    stocks, contrasts = receiver_coordinates.shape
    if role_operator.shape != (contrasts, contrasts):
        raise ValueError("role_operator must be square on the contrast space")
    if values.shape[0] != stocks or values.ndim not in (1, 2):
        raise ValueError("values must be [stock] or [stock, channel]")
    if gamma.numel() != 1:
        raise ValueError("gamma must be scalar")
    for name, tensor in (
        ("receiver_coordinates", receiver_coordinates),
        ("role_operator", role_operator),
        ("sender_coordinates", sender_coordinates),
        ("values", values),
        ("gamma", gamma),
    ):
        _require_finite(name, tensor)

    squeeze = values.ndim == 1
    payload = values.unsqueeze(-1) if squeeze else values
    receiver_operator = receiver_coordinates @ role_operator
    removed_diagonal = (receiver_operator * sender_coordinates).sum(dim=-1)
    mixed = receiver_operator @ (sender_coordinates.transpose(0, 1) @ payload)
    offdiag_mixed = mixed - removed_diagonal.unsqueeze(-1) * payload

    receiver_gram = receiver_coordinates.transpose(0, 1) @ receiver_coordinates
    sender_gram = sender_coordinates.transpose(0, 1) @ sender_coordinates
    full_frobenius_sq = torch.trace(
        role_operator.transpose(0, 1)
        @ receiver_gram
        @ role_operator
        @ sender_gram
    )
    offdiag_frobenius_sq = torch.clamp_min(
        full_frobenius_sq - removed_diagonal.square().sum(), 0.0
    )
    offdiag_frobenius = torch.sqrt(offdiag_frobenius_sq + float(eps))
    normalization = gamma / torch.sqrt(1.0 + offdiag_frobenius_sq)
    transformed = normalization * offdiag_mixed
    if squeeze:
        transformed = transformed.squeeze(-1)
    return transformed, removed_diagonal, offdiag_frobenius, normalization


def explicit_offdiag_operator(
    receiver_coordinates: torch.Tensor,
    role_operator: torch.Tensor,
    sender_coordinates: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Materialize the operator for tests and structural counterfactuals only."""

    raw = receiver_coordinates @ role_operator @ sender_coordinates.transpose(0, 1)
    operator = raw - torch.diag_embed(torch.diagonal(raw))
    return gamma * operator / torch.sqrt(1.0 + operator.square().sum())


def explicit_offdiag_operator_counterfactual(
    receiver_coordinates: torch.Tensor,
    role_operator: torch.Tensor,
    sender_coordinates: torch.Tensor,
    gamma: torch.Tensor,
    *,
    transform: str,
    sender_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize one frozen-checkpoint structural operator control.

    The transformation is applied to the raw zero-diagonal stock operator and
    the result is normalized with the same bounded Frobenius rule as the main
    model.  ``sender_permuted`` reassigns sender-role coordinates across stock
    identities before rebuilding the diagonal-free operator.
    """

    transform = str(transform)
    allowed = {"forward", "reverse", "symmetric", "sender_permuted"}
    if transform not in allowed:
        raise ValueError(f"unknown operator counterfactual: {transform}")
    if receiver_coordinates.ndim != 2 or sender_coordinates.ndim != 2:
        raise ValueError("role coordinates must be rank-two [stock, contrast]")
    if receiver_coordinates.shape != sender_coordinates.shape:
        raise ValueError("sender and receiver coordinates must share shape")
    stocks, contrasts = receiver_coordinates.shape
    if role_operator.shape != (contrasts, contrasts):
        raise ValueError("role_operator must be square on the contrast space")
    if gamma.numel() != 1:
        raise ValueError("gamma must be scalar")
    for name, tensor in (
        ("receiver_coordinates", receiver_coordinates),
        ("role_operator", role_operator),
        ("sender_coordinates", sender_coordinates),
        ("gamma", gamma),
    ):
        _require_finite(name, tensor)

    resolved_sender = sender_coordinates
    if transform == "sender_permuted":
        if sender_permutation is None:
            raise ValueError("sender_permuted requires sender_permutation")
        if sender_permutation.shape != (stocks,) or sender_permutation.dtype != torch.long:
            raise ValueError("sender_permutation must be int64 [stock]")
        if sender_permutation.device != sender_coordinates.device:
            raise ValueError("sender_permutation must share the coordinate device")
        expected = torch.arange(stocks, device=sender_permutation.device)
        if not torch.equal(torch.sort(sender_permutation).values, expected):
            raise ValueError("sender_permutation must contain every stock index once")
        resolved_sender = sender_coordinates[sender_permutation]
    elif sender_permutation is not None:
        raise ValueError("sender_permutation is only valid for sender_permuted")

    raw = receiver_coordinates @ role_operator @ resolved_sender.transpose(0, 1)
    if transform == "reverse":
        raw = raw.transpose(0, 1)
    elif transform == "symmetric":
        raw = 0.5 * (raw + raw.transpose(0, 1))
    operator = raw - torch.diag_embed(torch.diagonal(raw))
    return gamma * operator / torch.sqrt(1.0 + operator.square().sum())


class ResidualMultiScaleTemporalPooling(nn.Module):
    """Add learnable short- and full-window means to the recurrent last state.

    Both residual coefficients start at zero, so a newly constructed model is
    exactly equivalent to last-state pooling before optimization.
    """

    def __init__(self, short_window: int = 5) -> None:
        super().__init__()
        if int(short_window) <= 0:
            raise ValueError("short_window must be positive")
        self.short_window = int(short_window)
        self.short_logit = nn.Parameter(torch.zeros(()))
        self.long_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self, sequence: torch.Tensor, last_state: torch.Tensor
    ) -> torch.Tensor:
        if sequence.ndim != 3 or last_state.ndim != 2:
            raise ValueError("recurrent sequence and last state have invalid rank")
        if sequence.shape[0] != last_state.shape[0]:
            raise ValueError("recurrent sequence and last state disagree on stocks")
        if sequence.shape[2] != last_state.shape[1]:
            raise ValueError("recurrent sequence and last state disagree on width")
        width = min(self.short_window, int(sequence.shape[1]))
        short_mean = sequence[:, -width:].mean(dim=1)
        long_mean = sequence.mean(dim=1)
        return (
            last_state
            + torch.tanh(self.short_logit) * short_mean
            + torch.tanh(self.long_logit) * long_mean
        )


@dataclass(frozen=True)
class FlowFDGState:
    """Instrument-aligned information state and elapsed trading-day gaps."""

    flow_prev: torch.Tensor
    gap_days: torch.Tensor
    pattern_prev: torch.Tensor | None = None
    pattern_valid: torch.Tensor | None = None

    def detach(self) -> FlowFDGState:
        return FlowFDGState(
            flow_prev=self.flow_prev.detach(),
            gap_days=self.gap_days.detach(),
            pattern_prev=(
                None if self.pattern_prev is None else self.pattern_prev.detach()
            ),
            pattern_valid=(
                None if self.pattern_valid is None else self.pattern_valid.detach()
            ),
        )


@dataclass(frozen=True)
class FlowFDGOutput:
    prediction: torch.Tensor
    p_local: torch.Tensor | None
    flow_readout: torch.Tensor | None
    flow: torch.Tensor
    state: FlowFDGState
    diagnostics: Mapping[str, torch.Tensor]


class FlowFDG(nn.Module):
    """Single, end-to-end Flow-FDG with coupled graph/flow dynamics.

    In fully-coupled mode there are no local or flow prediction heads.  The
    graph reads both ``Z_t`` and ``F_{t-1}``, and the sole prediction is made
    from ``[Z_t, F_t, Z_t * W_F F_t]`` after the Flow update.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        temporal_arch: str = "gru",
        d_model: int = 64,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        transformer_max_len: int = 64,
        transformer_nhead: int = 4,
        transformer_ff_dim: int = 128,
        temporal_mlp_token_dim: int = 32,
        temporal_mlp_hidden_dim: int = 96,
        temporal_input_layout: str = "outer_time",
        temporal_outer_lookback: int = 20,
        alpha360_field_count: int = 6,
        temporal_pooling_mode: str = "last",
        temporal_pooling_short_window: int = 5,
        temporal_pooling_horizons: tuple[int, ...] = (5, 20, 60),
        temporal_attention_dim: int = 64,
        temporal_router_hidden_dim: int = 64,
        temporal_router_temperature: float = 1.0,
        temporal_attention_mix_init: float = -1.0,
        temporal_attention_identity_residual: bool = False,
        temporal_pattern_count: int = 4,
        pattern_flow_dim: int = 8,
        pattern_router_scope: str = "both",
        pattern_information_route_strength: float = 1.0,
        pattern_recurrence_mode: str = "full",
        pattern_state_fusion: str = "attention_only",
        pattern_lag_bias_init: str = "zero",
        pattern_local_state_mode: str = "pattern_routed",
        shared_pattern_reservoir: bool = False,
        market_feature_dim: int = 0,
        market_context_stop_gradient: bool = False,
        hierarchical_market_roles: bool = False,
        coarse_role_count: int = 4,
        hierarchy_mix_max: float = 0.10,
        hierarchy_mix_init: float = 0.02,
        innovation_source_residual: bool = False,
        source_residual_max: float = 0.20,
        local_hidden_dim: int = 64,
        role_count: int = 16,
        role_projection_dim: int = 64,
        flow_dim: int = 16,
        context_dim: int = 32,
        fast_rank: int = 4,
        active_fast_rank: int | None = None,
        fast_coefficient_mode: str = "signed_tanh",
        slow_rank: int = 4,
        learned_slow_state: bool = False,
        learned_slow_state_init: float = 0.0,
        fusion_hidden_dim: int = 96,
        readout_mode: str = "independent_concat",
        architecture_mode: str = "legacy_auxiliary_diagnostics",
        coupled_flow_projection_mode: str = "normalized",
        coupled_source_anchored_init: bool = False,
        couple_lagged_flow_to_roles: bool = True,
        coupled_fusion_mode: str = "mlp",
        max_flow_fraction: float = 0.08,
        normalized_energy_gate: bool = False,
        energy_gate_target: float = 0.04,
        evidence_aware_flow_reliability: bool = False,
        flow_reliability_floor: float = 0.25,
        expose_forwarding_readout: bool = False,
        forwarding_readout_scale: float = 0.25,
        dropout: float = 0.1,
        sender_temperature: float = 1.0,
        receiver_temperature: float = 1.0,
        gamma_max: float = 0.95,
        dissipation_floor: float = 0.05,
        retention_scale: float = 1.0,
        forwarding_scale: float = 1.0,
        suspension_max_days: int = 20,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        resolved_pooling_horizons = tuple(
            int(value) for value in temporal_pooling_horizons
        )
        positive = {
            "input_dim": input_dim,
            "d_model": d_model,
            "gru_hidden_dim": gru_hidden_dim,
            "gru_num_layers": gru_num_layers,
            "transformer_max_len": transformer_max_len,
            "transformer_nhead": transformer_nhead,
            "transformer_ff_dim": transformer_ff_dim,
            "temporal_mlp_token_dim": temporal_mlp_token_dim,
            "temporal_mlp_hidden_dim": temporal_mlp_hidden_dim,
            "temporal_outer_lookback": temporal_outer_lookback,
            "alpha360_field_count": alpha360_field_count,
            "temporal_pooling_short_window": temporal_pooling_short_window,
            "temporal_attention_dim": temporal_attention_dim,
            "temporal_router_hidden_dim": temporal_router_hidden_dim,
            "temporal_pattern_count": temporal_pattern_count,
            "pattern_flow_dim": pattern_flow_dim,
            "local_hidden_dim": local_hidden_dim,
            "role_count": role_count,
            "role_projection_dim": role_projection_dim,
            "flow_dim": flow_dim,
            "context_dim": context_dim,
            "fast_rank": fast_rank,
            "slow_rank": slow_rank,
            "fusion_hidden_dim": fusion_hidden_dim,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= int(market_feature_dim) < int(input_dim):
            raise ValueError("market_feature_dim must lie in [0, input_dim)")
        if int(coarse_role_count) < 2 or int(coarse_role_count) > int(role_count):
            raise ValueError("coarse_role_count must lie in [2, role_count]")
        if not 0.0 < float(hierarchy_mix_max) < 1.0:
            raise ValueError("hierarchy_mix_max must lie strictly between zero and one")
        if not 0.0 < float(hierarchy_mix_init) < float(hierarchy_mix_max):
            raise ValueError(
                "hierarchy_mix_init must lie strictly between zero and hierarchy_mix_max"
            )
        if not 0.0 < float(source_residual_max) < 1.0:
            raise ValueError("source_residual_max must lie strictly between zero and one")
        if bool(innovation_source_residual) and not bool(hierarchical_market_roles):
            raise ValueError(
                "innovation source residual requires hierarchical market roles"
            )
        if int(role_count) < 2:
            raise ValueError("role_count must be at least two")
        if str(temporal_arch).lower() not in {
            "gru",
            "lstm",
            "transformer",
            "mlp",
        }:
            raise ValueError(
                "temporal_arch must be 'gru', 'lstm', 'transformer', or 'mlp'"
            )
        if str(temporal_input_layout) not in {
            "outer_time",
            ALPHA360_LAST_SNAPSHOT,
        }:
            raise ValueError(
                "temporal_input_layout must be 'outer_time' or "
                f"'{ALPHA360_LAST_SNAPSHOT}'"
            )
        if (
            str(temporal_input_layout) == ALPHA360_LAST_SNAPSHOT
            and str(temporal_arch).lower() not in {"gru", "lstm"}
        ):
            raise ValueError(
                "Alpha360 factor-axis input requires a recurrent encoder"
            )
        if (
            str(temporal_input_layout) == ALPHA360_LAST_SNAPSHOT
            and int(input_dim) - int(market_feature_dim) != 360
        ):
            raise ValueError(
                "Alpha360 factor-axis input requires 360 local features"
            )
        if (
            str(temporal_input_layout) == ALPHA360_LAST_SNAPSHOT
            and 360 % int(alpha360_field_count) != 0
        ):
            raise ValueError(
                "Alpha360 local features must be divisible by the field count"
            )
        if (
            str(temporal_arch).lower() == "transformer"
            and int(d_model) % int(transformer_nhead) != 0
        ):
            raise ValueError("d_model must be divisible by transformer_nhead")
        if str(temporal_pooling_mode) not in {
            "last",
            "residual_multiscale",
            "state_attentive",
            "concat_attentive",
            "pattern_routed_source",
            "tra_core_routed_source",
        }:
            raise ValueError(
                "temporal_pooling_mode must be 'last', "
                "'residual_multiscale', 'state_attentive', "
                "'concat_attentive', 'pattern_routed_source', or "
                "'tra_core_routed_source'"
            )
        if (
            str(temporal_pooling_mode) == "residual_multiscale"
            and str(temporal_arch).lower() not in {"gru", "lstm"}
        ):
            raise ValueError(
                "residual_multiscale pooling requires a recurrent encoder"
            )
        if (
            str(temporal_pooling_mode) in {"state_attentive", "concat_attentive"}
            and str(temporal_arch).lower() != "lstm"
        ):
            raise ValueError(
                f"{temporal_pooling_mode} pooling requires an LSTM encoder"
            )
        if (
            str(temporal_pooling_mode)
            in {"pattern_routed_source", "tra_core_routed_source"}
            and str(temporal_arch).lower() not in {"gru", "lstm"}
        ):
            raise ValueError(
                "routed-source pooling requires a recurrent encoder"
            )
        if (
            str(temporal_pooling_mode) == "tra_core_routed_source"
            and str(temporal_arch).lower() != "lstm"
        ):
            raise ValueError("TRA-core routing requires an LSTM encoder")
        if len(resolved_pooling_horizons) < 2 or any(
            value <= 0 for value in resolved_pooling_horizons
        ):
            raise ValueError(
                "state-attentive pooling needs positive horizons"
            )
        if (
            tuple(sorted(set(resolved_pooling_horizons)))
            != resolved_pooling_horizons
        ):
            raise ValueError(
                "attention horizons must be strictly increasing"
            )
        if str(temporal_pooling_mode) in {
            "pattern_routed_source",
            "tra_core_routed_source",
        }:
            if (
                not bool(shared_pattern_reservoir)
                and int(flow_dim)
                != int(temporal_pattern_count) * int(pattern_flow_dim)
            ):
                raise ValueError(
                    "pattern-routed flow_dim must equal "
                    "temporal_pattern_count * pattern_flow_dim"
                )
            if str(pattern_router_scope) not in {
                "both",
                "local_only",
                "information_only",
                "none",
            }:
                raise ValueError(
                    "pattern_router_scope must be both, local_only, "
                    "information_only, or none"
                )
            if not 0.0 <= float(pattern_information_route_strength) <= 1.0:
                raise ValueError(
                    "pattern_information_route_strength must lie in [0, 1]"
                )
            if str(pattern_recurrence_mode) not in {"full", "one_step"}:
                raise ValueError(
                    "pattern_recurrence_mode must be full or one_step"
                )
            if (
                str(pattern_local_state_mode) == "concat_terminal_attention"
                and str(pattern_router_scope)
                not in {"information_only", "none"}
            ):
                raise ValueError(
                    "concat base local state requires an information-only "
                    "or disabled pattern router"
                )
        if float(temporal_router_temperature) <= 0.0:
            raise ValueError(
                "temporal_router_temperature must be positive"
            )
        resolved_active_fast_rank = (
            int(fast_rank) if active_fast_rank is None else int(active_fast_rank)
        )
        if not 1 <= resolved_active_fast_rank <= int(fast_rank):
            raise ValueError("active_fast_rank must lie in [1, fast_rank]")
        if fast_coefficient_mode not in {
            "signed_tanh",
            "nonnegative_sigmoid",
        }:
            raise ValueError(
                "fast_coefficient_mode must be 'signed_tanh' or "
                "'nonnegative_sigmoid'"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if min(float(sender_temperature), float(receiver_temperature)) <= 0.0:
            raise ValueError("role temperatures must be positive")
        if not 0.0 < float(gamma_max) < 1.0:
            raise ValueError("gamma_max must lie strictly between zero and one")
        if not 0.0 < float(dissipation_floor) < 1.0:
            raise ValueError("dissipation_floor must lie strictly between zero and one")
        if not 0.0 <= float(retention_scale) <= 1.0:
            raise ValueError("retention_scale must lie in [0, 1]")
        if not 0.0 <= float(forwarding_scale) <= 1.0:
            raise ValueError("forwarding_scale must lie in [0, 1]")
        if int(suspension_max_days) < 0:
            raise ValueError("suspension_max_days must be non-negative")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        if readout_mode not in {
            "independent_concat",
            "shared_gated",
            "bounded_shared_gated",
            "strict_norm_bounded_shared",
        }:
            raise ValueError(
                "readout_mode must be 'independent_concat', 'shared_gated', "
                "'bounded_shared_gated', or 'strict_norm_bounded_shared'"
            )
        if architecture_mode not in {
            "legacy_auxiliary_diagnostics",
            "fully_coupled_single_head",
        }:
            raise ValueError(
                "architecture_mode must be legacy_auxiliary_diagnostics or "
                "fully_coupled_single_head"
            )
        if architecture_mode == "fully_coupled_single_head":
            if str(temporal_pooling_mode) not in {
                "pattern_routed_source",
                "tra_core_routed_source",
            }:
                raise ValueError(
                    "fully coupled mode requires routed temporal sources"
                )
            if readout_mode != "independent_concat":
                raise ValueError(
                    "fully coupled mode requires one independent-concat joint head"
                )
            if bool(expose_forwarding_readout):
                raise ValueError(
                    "fully coupled mode cannot expose a separate forwarding readout"
                )
        if (
            readout_mode == "strict_norm_bounded_shared"
            and architecture_mode == "fully_coupled_single_head"
        ):
            raise ValueError(
                "strict norm-bounded shared readout requires the A9 legacy anchor"
            )
        if coupled_flow_projection_mode not in {
            "normalized",
            "scale_preserving",
        }:
            raise ValueError(
                "coupled_flow_projection_mode must be normalized or scale_preserving"
            )
        if architecture_mode != "fully_coupled_single_head" and (
            coupled_flow_projection_mode != "normalized"
            or bool(coupled_source_anchored_init)
        ):
            raise ValueError(
                "coupled initialization options require fully coupled mode"
            )
        if coupled_fusion_mode not in {"mlp", "linear"}:
            raise ValueError("coupled_fusion_mode must be mlp or linear")
        if (
            architecture_mode != "fully_coupled_single_head"
            and coupled_fusion_mode != "mlp"
        ):
            raise ValueError(
                "coupled_fusion_mode is defined only for fully coupled mode"
            )
        if not 0.0 < float(max_flow_fraction) < 1.0:
            raise ValueError("max_flow_fraction must lie strictly between zero and one")
        if (
            bool(normalized_energy_gate)
            and not 0.0 < float(energy_gate_target) < float(max_flow_fraction)
        ):
            raise ValueError(
                "energy_gate_target must lie strictly between zero and max_flow_fraction"
            )
        if bool(normalized_energy_gate) and architecture_mode != "fully_coupled_single_head":
            raise ValueError(
                "normalized energy gating requires fully coupled single-head mode"
            )
        if bool(normalized_energy_gate) and readout_mode == "strict_norm_bounded_shared":
            raise ValueError(
                "strict norm-bounded readout already supplies the energy gate"
            )
        if not 0.0 <= float(flow_reliability_floor) < 1.0:
            raise ValueError("flow_reliability_floor must lie in [0, 1)")
        if not 0.0 <= float(forwarding_readout_scale) <= 1.0:
            raise ValueError("forwarding_readout_scale must lie in [0, 1]")
        if (
            bool(evidence_aware_flow_reliability)
            and readout_mode != "bounded_shared_gated"
        ):
            raise ValueError(
                "evidence-aware reliability requires the bounded shared gate"
            )
        if readout_mode == "strict_norm_bounded_shared" and bool(
            expose_forwarding_readout
        ):
            raise ValueError(
                "strict norm-bounded readout cannot expose forwarding separately"
            )

        self.input_dim = int(input_dim)
        self.market_feature_dim = int(market_feature_dim)
        self.local_input_dim = self.input_dim - self.market_feature_dim
        self.market_context_stop_gradient = bool(market_context_stop_gradient)
        self.hierarchical_market_roles = bool(hierarchical_market_roles)
        self.coarse_role_count = int(coarse_role_count)
        self.hierarchy_mix_max = float(hierarchy_mix_max)
        self.hierarchy_mix_init = float(hierarchy_mix_init)
        self.innovation_source_residual = bool(innovation_source_residual)
        self.source_residual_max = float(source_residual_max)
        self.temporal_arch = str(temporal_arch).lower()
        self.d_model = int(d_model)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru_num_layers = int(gru_num_layers)
        self.transformer_max_len = int(transformer_max_len)
        self.transformer_nhead = int(transformer_nhead)
        self.transformer_ff_dim = int(transformer_ff_dim)
        self.temporal_mlp_token_dim = int(temporal_mlp_token_dim)
        self.temporal_mlp_hidden_dim = int(temporal_mlp_hidden_dim)
        self.temporal_input_layout = str(temporal_input_layout)
        self.temporal_outer_lookback = int(temporal_outer_lookback)
        self.alpha360_field_count = int(alpha360_field_count)
        self.temporal_pooling_mode = str(temporal_pooling_mode)
        self.temporal_pooling_short_window = int(
            temporal_pooling_short_window
        )
        self.temporal_pooling_horizons = resolved_pooling_horizons
        self.temporal_attention_dim = int(temporal_attention_dim)
        self.temporal_router_hidden_dim = int(
            temporal_router_hidden_dim
        )
        self.temporal_router_temperature = float(
            temporal_router_temperature
        )
        self.temporal_attention_mix_init = float(
            temporal_attention_mix_init
        )
        self.temporal_attention_identity_residual = bool(
            temporal_attention_identity_residual
        )
        self.pattern_routed = self.temporal_pooling_mode in {
            "pattern_routed_source",
            "tra_core_routed_source",
        }
        self.pattern_count = (
            int(temporal_pattern_count) if self.pattern_routed else 0
        )
        self.pattern_state_dim = self.d_model if self.pattern_routed else 0
        self.pattern_flow_dim = (
            int(pattern_flow_dim) if self.pattern_routed else 0
        )
        self.pattern_router_scope = (
            str(pattern_router_scope) if self.pattern_routed else "none"
        )
        self.pattern_information_route_strength = (
            float(pattern_information_route_strength)
            if self.pattern_routed else 0.0
        )
        self.pattern_recurrence_mode = (
            str(pattern_recurrence_mode) if self.pattern_routed else "full"
        )
        self.pattern_state_fusion = (
            str(pattern_state_fusion)
            if self.pattern_routed
            else "attention_only"
        )
        self.pattern_lag_bias_init = (
            str(pattern_lag_bias_init) if self.pattern_routed else "zero"
        )
        self.pattern_local_state_mode = (
            str(pattern_local_state_mode)
            if self.pattern_routed
            else "pattern_routed"
        )
        self.shared_pattern_reservoir = bool(
            shared_pattern_reservoir and self.pattern_routed
        )
        self.local_hidden_dim = int(local_hidden_dim)
        self.role_count = int(role_count)
        self.role_contrast_dim = self.role_count - 1
        self.role_projection_dim = int(role_projection_dim)
        self.flow_dim = int(flow_dim)
        self.context_dim = int(context_dim)
        self.fast_rank = int(fast_rank)
        self.active_fast_rank = resolved_active_fast_rank
        self.fast_coefficient_mode = str(fast_coefficient_mode)
        self.slow_rank = int(slow_rank)
        self.learned_slow_state = bool(learned_slow_state)
        self.learned_slow_state_init = float(learned_slow_state_init)
        if not 0.0 <= self.learned_slow_state_init <= 1.0:
            raise ValueError("learned_slow_state_init must lie in [0, 1]")
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.readout_mode = str(readout_mode)
        self.architecture_mode = str(architecture_mode)
        self.fully_coupled = (
            self.architecture_mode == "fully_coupled_single_head"
        )
        self.coupled_flow_projection_mode = str(
            coupled_flow_projection_mode
        )
        self.coupled_source_anchored_init = bool(
            coupled_source_anchored_init
        )
        self.couple_lagged_flow_to_roles = bool(
            couple_lagged_flow_to_roles
        )
        self._detach_lagged_flow_roles = False
        self._force_coupled_temporal_only = False
        self.coupled_fusion_mode = str(coupled_fusion_mode)
        self.max_flow_fraction = float(max_flow_fraction)
        self.normalized_energy_gate = bool(normalized_energy_gate)
        self.energy_gate_target = float(energy_gate_target)
        self.evidence_aware_flow_reliability = bool(
            evidence_aware_flow_reliability
        )
        self.flow_reliability_floor = float(flow_reliability_floor)
        self.expose_forwarding_readout = bool(expose_forwarding_readout)
        self.forwarding_readout_scale = float(forwarding_readout_scale)
        self.dropout = float(dropout)
        self.sender_temperature = float(sender_temperature)
        self.receiver_temperature = float(receiver_temperature)
        self.gamma_max = float(gamma_max)
        self.dissipation_floor = float(dissipation_floor)
        self.retention_scale = float(retention_scale)
        self.forwarding_scale = float(forwarding_scale)
        self.suspension_max_days = int(suspension_max_days)
        self.eps = float(eps)

        self.temporal_input_projection: nn.Module | None = None
        self.temporal_position_embedding: nn.Module | None = None
        self.temporal_residual_pooling: nn.Module | None = None
        self.temporal_state_pooling: nn.Module | None = None
        self.temporal_concat_pooling: nn.Module | None = None
        self.temporal_pattern_pooling: nn.Module | None = None
        if self.temporal_arch == "transformer":
            self.temporal_input_projection = nn.Linear(
                self.local_input_dim, self.d_model
            )
            self.temporal_position_embedding = nn.Embedding(
                self.transformer_max_len, self.d_model
            )
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.transformer_nhead,
                dim_feedforward=self.transformer_ff_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder: nn.Module = nn.TransformerEncoder(
                layer, num_layers=self.gru_num_layers
            )
            self.temporal_projection: nn.Module = nn.LayerNorm(self.d_model)
        elif self.temporal_arch == "mlp":
            self.temporal_input_projection = nn.Sequential(
                nn.Linear(self.local_input_dim, self.temporal_mlp_token_dim),
                nn.GELU(),
                nn.LayerNorm(self.temporal_mlp_token_dim),
            )
            self.temporal_encoder = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(
                    self.transformer_max_len * self.temporal_mlp_token_dim,
                    self.temporal_mlp_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.temporal_mlp_hidden_dim, self.d_model),
            )
            self.temporal_projection = nn.LayerNorm(self.d_model)
        else:
            recurrent_type: type[nn.GRU] | type[nn.LSTM]
            recurrent_type = nn.GRU if self.temporal_arch == "gru" else nn.LSTM
            recurrent_input_size = (
                self.alpha360_field_count
                if self.temporal_input_layout == ALPHA360_LAST_SNAPSHOT
                else self.local_input_dim
            )
            self.temporal_encoder = recurrent_type(
                input_size=recurrent_input_size,
                hidden_size=self.gru_hidden_dim,
                num_layers=self.gru_num_layers,
                batch_first=True,
                dropout=self.dropout if self.gru_num_layers > 1 else 0.0,
            )
            if self.pattern_routed:
                self.temporal_projection = nn.Identity()
            else:
                recurrent_projection_dim = (
                    2 * self.gru_hidden_dim
                    if self.temporal_pooling_mode == "concat_attentive"
                    else self.gru_hidden_dim
                )
                self.temporal_projection = nn.Sequential(
                    nn.Linear(recurrent_projection_dim, self.d_model),
                    nn.LayerNorm(self.d_model),
                )
            if self.temporal_pooling_mode == "residual_multiscale":
                self.temporal_residual_pooling = ResidualMultiScaleTemporalPooling(
                    self.temporal_pooling_short_window
                )
            elif self.temporal_pooling_mode == "state_attentive":
                self.temporal_state_pooling = StateAttentiveTemporalPooling(
                    self.gru_hidden_dim,
                    horizons=self.temporal_pooling_horizons,
                    attention_dim=self.temporal_attention_dim,
                    router_hidden_dim=self.temporal_router_hidden_dim,
                    router_temperature=self.temporal_router_temperature,
                    mix_init=self.temporal_attention_mix_init,
                    identity_residual=(
                        self.temporal_attention_identity_residual
                    ),
                    dropout=self.dropout,
                    eps=self.eps,
                )
            elif self.temporal_pooling_mode == "concat_attentive":
                self.temporal_concat_pooling = (
                    ConcatenatedTemporalAttentionPooling(
                        self.gru_hidden_dim,
                        attention_dim=self.temporal_attention_dim,
                    )
                )
            elif self.temporal_pooling_mode == "pattern_routed_source":
                self.temporal_pattern_pooling = (
                    PatternRoutedTemporalSourceEncoder(
                        self.gru_hidden_dim,
                        self.d_model,
                        pattern_count=self.pattern_count,
                        attention_dim=self.temporal_attention_dim,
                        router_hidden_dim=self.temporal_router_hidden_dim,
                        max_len=self.transformer_max_len,
                        router_temperature=self.temporal_router_temperature,
                        state_fusion=self.pattern_state_fusion,
                        lag_bias_init=self.pattern_lag_bias_init,
                        local_state_mode=self.pattern_local_state_mode,
                        dropout=self.dropout,
                        eps=self.eps,
                    )
                )
            elif self.temporal_pooling_mode == "tra_core_routed_source":
                self.temporal_pattern_pooling = (
                    TRACoreRoutedTemporalSourceEncoder(
                        self.gru_hidden_dim,
                        self.d_model,
                        state_count=self.pattern_count,
                        router_temperature=self.temporal_router_temperature,
                        dropout=self.dropout,
                        eps=self.eps,
                    )
                )
        self.local_head = nn.Sequential(
            nn.Linear(self.d_model, self.local_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.local_hidden_dim, 1),
        )
        self.tra_source_score_head_initialization: dict[str, object] | None = None

        if self.fully_coupled:
            self.local_head = None
            if self.coupled_flow_projection_mode == "scale_preserving":
                self.role_flow_projection: nn.Module | None = nn.Sequential(
                    nn.Linear(self.flow_dim, self.d_model, bias=False),
                    nn.Tanh(),
                )
            else:
                self.role_flow_projection = nn.Sequential(
                    nn.Linear(self.flow_dim, self.d_model, bias=False),
                    nn.LayerNorm(self.d_model),
                )
            self.role_state_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(3 * self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.LayerNorm(self.d_model),
            )
        else:
            self.role_flow_projection = None
            self.role_state_encoder = None

        self.context_encoder = nn.Sequential(
            nn.Linear(2 * self.d_model, self.context_dim),
            nn.GELU(),
            nn.LayerNorm(self.context_dim),
        )
        self.market_feature_encoder: nn.Module | None = None
        self.market_context_norm: nn.Module | None = None
        if self.market_feature_dim > 0:
            self.market_feature_encoder = nn.Sequential(
                nn.LayerNorm(self.market_feature_dim),
                nn.Linear(self.market_feature_dim, self.context_dim),
                nn.GELU(),
                nn.Linear(self.context_dim, self.context_dim),
            )
            self.market_context_norm = nn.LayerNorm(self.context_dim)
        self.role_norm = nn.LayerNorm(self.d_model)
        self.sender_query = nn.Linear(
            self.d_model, self.role_projection_dim, bias=False
        )
        self.receiver_query = nn.Linear(
            self.d_model, self.role_projection_dim, bias=False
        )
        self.sender_prototypes = nn.Parameter(
            torch.empty(self.role_count, self.role_projection_dim)
        )
        self.receiver_prototypes = nn.Parameter(
            torch.empty(self.role_count, self.role_projection_dim)
        )
        activity_input = self.d_model + self.context_dim
        self.sender_activity = nn.Sequential(
            nn.Linear(activity_input, self.context_dim),
            nn.GELU(),
            nn.Linear(self.context_dim, 1),
        )
        self.receiver_activity = nn.Sequential(
            nn.Linear(activity_input, self.context_dim),
            nn.GELU(),
            nn.Linear(self.context_dim, 1),
        )
        self.register_buffer(
            "role_contrast_basis", helmert_contrast(self.role_count), persistent=True
        )
        self.hierarchical_role_residual: nn.Module | None = None
        if self.hierarchical_market_roles:
            self.hierarchical_role_residual = HierarchicalMarketRoleResidual(
                self.d_model,
                self.context_dim,
                self.role_count,
                coarse_role_count=self.coarse_role_count,
                mix_max=self.hierarchy_mix_max,
                mix_init=self.hierarchy_mix_init,
            )

        contrast = self.role_contrast_dim
        self.base_role_operator = nn.Parameter(torch.empty(contrast, contrast))
        self.fast_left = nn.Parameter(torch.empty(contrast, self.fast_rank))
        self.fast_right = nn.Parameter(torch.empty(contrast, self.fast_rank))
        self.slow_left = nn.Parameter(torch.empty(contrast, self.slow_rank))
        self.slow_right = nn.Parameter(torch.empty(contrast, self.slow_rank))
        if self.learned_slow_state:
            initial_slow = torch.full(
                (self.slow_rank,), self.learned_slow_state_init
            )
            initial_slow[1::2].neg_()
            self.slow_state_parameter = nn.Parameter(initial_slow)
        else:
            self.register_parameter("slow_state_parameter", None)
        self.fast_state_head = nn.Linear(self.context_dim, self.fast_rank)
        self.register_buffer(
            "fast_rank_mask",
            torch.cat(
                [
                    torch.ones(self.active_fast_rank),
                    torch.zeros(self.fast_rank - self.active_fast_rank),
                ]
            ),
            persistent=False,
        )

        self.info_encoder: nn.Module | None = None
        self.pattern_info_encoders: nn.ModuleList | None = None
        self.mix_base_logits: nn.Parameter | None = None
        self.mix_head: nn.Module | None = None
        self.pattern_mix_logits: nn.Parameter | None = None
        if self.pattern_routed:
            pattern_information_dim = (
                self.flow_dim
                if self.shared_pattern_reservoir
                else self.pattern_flow_dim
            )
            self.pattern_info_encoders = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.d_model, self.d_model),
                        nn.GELU(),
                        nn.LayerNorm(self.d_model),
                        nn.Linear(self.d_model, pattern_information_dim),
                    )
                    for _ in range(self.pattern_count)
                ]
            )
            self.pattern_mix_logits = nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0]] * self.pattern_count)
            )
        else:
            self.info_encoder = nn.Sequential(
                nn.Linear(self.d_model + 1, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.flow_dim),
            )
            self.mix_base_logits = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
            self.mix_head = nn.Linear(self.context_dim, 3)
        self.innovation_source_module: nn.Module | None = None
        if self.innovation_source_residual:
            self.innovation_source_module = InnovationSourceResidual(
                self.d_model,
                self.context_dim,
                self.flow_dim,
                max_fraction=self.source_residual_max,
                eps=self.eps,
            )
        self.gamma_base_logit = nn.Parameter(torch.tensor(0.0))
        self.gamma_head = nn.Linear(self.context_dim, 1)

        self.strict_norm_fusion: nn.Module | None = None
        if self.readout_mode == "strict_norm_bounded_shared":
            self.flow_projection = nn.Identity()
        elif self.fully_coupled and self.coupled_flow_projection_mode == "scale_preserving":
            self.flow_projection = nn.Sequential(
                nn.Linear(self.flow_dim, self.d_model, bias=False),
                nn.Tanh(),
            )
        else:
            self.flow_projection = nn.Sequential(
                nn.Linear(self.flow_dim, self.d_model),
                nn.LayerNorm(self.d_model),
            )
        if self.readout_mode == "independent_concat":
            if self.fully_coupled and self.coupled_fusion_mode == "linear":
                self.fusion = nn.Sequential(
                    nn.Linear(3 * self.d_model, self.d_model),
                    nn.LayerNorm(self.d_model),
                )
            else:
                self.fusion = nn.Sequential(
                    nn.Linear(3 * self.d_model, self.fusion_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.fusion_hidden_dim, self.d_model),
                    nn.LayerNorm(self.d_model),
                )
            self.prediction_head: nn.Module | None = nn.Sequential(
                nn.Linear(self.d_model, self.local_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.local_hidden_dim, 1),
            )
            self.fusion_gate: nn.Module | None = None
            self.interaction_projection: nn.Module | None = None
            self.shared_fusion_norm: nn.Module | None = None
        elif self.readout_mode == "strict_norm_bounded_shared":
            self.fusion = None
            self.prediction_head = None
            self.fusion_gate = None
            self.interaction_projection = None
            self.shared_fusion_norm = None
            self.strict_norm_fusion = StrictNormBoundedFusion(
                self.flow_dim,
                self.d_model,
                self.context_dim,
                self.fusion_hidden_dim,
                max_fraction=self.max_flow_fraction,
                eps=self.eps,
            )
        else:
            self.fusion = None
            self.prediction_head = None
            self.fusion_gate = nn.Linear(2 * self.d_model, self.d_model)
            self.interaction_projection = nn.Linear(
                self.d_model, self.d_model, bias=False
            )
            self.shared_fusion_norm = nn.LayerNorm(self.d_model)
        self.flow_diagnostic_head: nn.Module | None = None
        if not self.fully_coupled:
            self.flow_diagnostic_head = nn.Linear(
                self.flow_dim, 1, bias=False
            )
        self.flow_reliability_gate: nn.Module | None = None
        self.energy_gate: nn.Module | None = None
        if self.normalized_energy_gate:
            energy_input_dim = 2 * self.d_model + self.pattern_count
            self.energy_gate = nn.Sequential(
                nn.Linear(energy_input_dim, self.context_dim),
                nn.GELU(),
                nn.Linear(self.context_dim, 1),
            )
        self.forwarding_projection: nn.Module | None = None
        if self.expose_forwarding_readout:
            self.forwarding_projection = nn.Linear(
                self.flow_dim, self.d_model, bias=False
            )

        self.reset_parameters()
        if self.evidence_aware_flow_reliability:
            self.flow_reliability_gate = nn.Linear(5, 1)
            nn.init.zeros_(self.flow_reliability_gate.weight)
            nn.init.constant_(self.flow_reliability_gate.bias, 3.0)

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.sender_prototypes)
        nn.init.orthogonal_(self.receiver_prototypes)
        nn.init.normal_(self.base_role_operator, mean=0.0, std=0.04)
        for value in (
            self.fast_left,
            self.fast_right,
            self.slow_left,
            self.slow_right,
        ):
            nn.init.normal_(value, mean=0.0, std=0.04)
        if isinstance(self.mix_head, nn.Linear):
            nn.init.zeros_(self.mix_head.weight)
            nn.init.zeros_(self.mix_head.bias)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        if self.readout_mode in {"shared_gated", "bounded_shared_gated"}:
            assert isinstance(self.fusion_gate, nn.Linear)
            assert isinstance(self.interaction_projection, nn.Linear)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(
                self.fusion_gate.bias,
                0.0 if self.readout_mode == "bounded_shared_gated" else 3.0,
            )
            nn.init.zeros_(self.interaction_projection.weight)
        if self.flow_reliability_gate is not None:
            assert isinstance(self.flow_reliability_gate, nn.Linear)
            nn.init.zeros_(self.flow_reliability_gate.weight)
            nn.init.constant_(self.flow_reliability_gate.bias, 3.0)
        if self.forwarding_projection is not None:
            assert isinstance(self.forwarding_projection, nn.Linear)
            nn.init.zeros_(self.forwarding_projection.weight)
        if self.energy_gate is not None:
            final_energy = self.energy_gate[-1]
            if not isinstance(final_energy, nn.Linear):
                raise TypeError("energy gate output must be linear")
            nn.init.zeros_(final_energy.weight)
            target_fraction = self.energy_gate_target / self.max_flow_fraction
            nn.init.constant_(
                final_energy.bias,
                torch.logit(torch.tensor(target_fraction)).item(),
            )
        if self.fully_coupled and self.coupled_source_anchored_init:
            if not isinstance(self.role_state_encoder, nn.Sequential):
                raise TypeError("coupled role-state encoder is missing")
            if not isinstance(self.fusion, nn.Sequential):
                raise TypeError("coupled joint fusion is missing")
            role_input = self.role_state_encoder[0]
            fusion_input = self.fusion[0]
            if not isinstance(role_input, nn.Linear) or not isinstance(
                fusion_input, nn.Linear
            ):
                raise TypeError("coupled source-anchor inputs must be linear")
            with torch.no_grad():
                role_input.weight[:, self.d_model :].mul_(0.01)
                fusion_input.weight[:, self.d_model :].mul_(0.01)

    def set_pattern_router_uniform(self, enabled: bool) -> None:
        if self.temporal_pattern_pooling is None:
            return
        if not isinstance(
            self.temporal_pattern_pooling,
            (
                PatternRoutedTemporalSourceEncoder,
                TRACoreRoutedTemporalSourceEncoder,
            ),
        ):
            raise TypeError("invalid pattern-routed temporal source module")
        self.temporal_pattern_pooling.set_uniform_router(enabled)

    def initialize_tra_core(
        self,
        model_state: Mapping[str, torch.Tensor],
        tra_state: Mapping[str, torch.Tensor],
    ) -> dict[str, object]:
        """Warm-start the recurrent and routed source modules from official TRA."""

        if self.temporal_pooling_mode != "tra_core_routed_source":
            raise ValueError("TRA initialization requires tra_core_routed_source")
        if not isinstance(self.temporal_encoder, nn.LSTM):
            raise TypeError("TRA initialization requires an LSTM encoder")
        if not isinstance(
            self.temporal_pattern_pooling,
            TRACoreRoutedTemporalSourceEncoder,
        ):
            raise TypeError("TRA-core routed source module is missing")
        recurrent_state = {
            key.removeprefix("rnn."): value
            for key, value in model_state.items()
            if key.startswith("rnn.")
        }
        if not recurrent_state:
            raise ValueError("official TRA checkpoint is missing recurrent tensors")
        self.temporal_encoder.load_state_dict(recurrent_state, strict=True)
        details = self.temporal_pattern_pooling.load_official_tra_state(
            model_state,
            tra_state,
        )
        if self.local_head is None:
            raise TypeError("TRA initialization requires the unified local head")
        self.tra_source_score_head_initialization = (
            _initialize_gelu_axis_identity_head(
                self.local_head,
                axis=0,
            )
        )
        if not isinstance(self.strict_norm_fusion, StrictNormBoundedFusion):
            raise TypeError(
                "TRA source anchoring requires strict norm-bounded shared readout"
            )
        strict_source_anchor = _initialize_strict_source_anchor(
            self.strict_norm_fusion
        )
        return {
            **details,
            "recurrent_tensors_loaded": len(recurrent_state),
            "source": "official_qlib_tra",
            "unified_head_initialization": dict(
                self.tra_source_score_head_initialization
            ),
            "strict_source_anchor": strict_source_anchor,
        }

    def set_coupled_temporal_only(self, enabled: bool) -> None:
        if bool(enabled) and not self.fully_coupled:
            raise ValueError(
                "temporal-only curriculum is defined only for fully coupled mode"
            )
        self._force_coupled_temporal_only = bool(enabled)

    def set_detach_lagged_flow_roles(self, enabled: bool) -> None:
        if bool(enabled) and not self.fully_coupled:
            raise ValueError(
                "lagged-Flow role detachment is defined only for fully coupled mode"
            )
        self._detach_lagged_flow_roles = bool(enabled)

    @property
    def force_coupled_temporal_only(self) -> bool:
        return self._force_coupled_temporal_only

    @property
    def detach_lagged_flow_roles(self) -> bool:
        return self._detach_lagged_flow_roles

    def structure_dict(self) -> dict[str, object]:
        return {
            "model": type(self).__name__,
            "single_model": True,
            "prediction_residual": False,
            "tra_source_score_head_initialization": (
                dict(self.tra_source_score_head_initialization)
                if self.tra_source_score_head_initialization is not None
                else None
            ),
            "architecture_mode": self.architecture_mode,
            "fully_coupled_graph_and_flow": self.fully_coupled,
            "auxiliary_prediction_heads": not self.fully_coupled,
            "coupled_flow_projection_mode": (
                self.coupled_flow_projection_mode if self.fully_coupled else None
            ),
            "coupled_source_anchored_init": self.coupled_source_anchored_init,
            "couple_lagged_flow_to_roles": (
                self.couple_lagged_flow_to_roles if self.fully_coupled else None
            ),
            "supports_detached_lagged_flow_role_warmup": self.fully_coupled,
            "coupled_fusion_mode": (
                self.coupled_fusion_mode if self.fully_coupled else None
            ),
            "supports_single_head_temporal_curriculum": self.fully_coupled,
            "prediction_readout": (
                "shared_head(Z+strict_l2_bounded_flow_delta)"
                if self.readout_mode == "strict_norm_bounded_shared"
                else "unified(Z,F,Z*W_FF)"
            ),
            "readout_mode": self.readout_mode,
            "shared_local_and_final_head": (not self.fully_coupled) and self.readout_mode
            in {
                "shared_gated",
                "bounded_shared_gated",
                "strict_norm_bounded_shared",
            },
            "max_flow_fraction": (
                self.max_flow_fraction
                if self.readout_mode
                in {"bounded_shared_gated", "strict_norm_bounded_shared"}
                or self.normalized_energy_gate
                else None
            ),
            "strict_per_stock_l2_flow_bound": (
                self.readout_mode == "strict_norm_bounded_shared"
            ),
            "normalized_energy_gate": self.normalized_energy_gate,
            "energy_gate_target": (
                self.energy_gate_target if self.normalized_energy_gate else None
            ),
            "evidence_aware_flow_reliability": self.evidence_aware_flow_reliability,
            "flow_reliability_floor": (
                self.flow_reliability_floor
                if self.evidence_aware_flow_reliability
                else None
            ),
            "expose_forwarding_readout": self.expose_forwarding_readout,
            "forwarding_readout_scale": (
                self.forwarding_readout_scale
                if self.expose_forwarding_readout
                else None
            ),
            "input_dim": self.input_dim,
            "local_input_dim": self.local_input_dim,
            "temporal_input_layout": self.temporal_input_layout,
            "temporal_outer_lookback": self.temporal_outer_lookback,
            "alpha360_field_count": (
                self.alpha360_field_count
                if self.temporal_input_layout == ALPHA360_LAST_SNAPSHOT
                else None
            ),
            "recurrent_steps": (
                self.local_input_dim // self.alpha360_field_count
                if self.temporal_input_layout == ALPHA360_LAST_SNAPSHOT
                else self.temporal_outer_lookback
            ),
            "market_feature_dim": self.market_feature_dim,
            "market_context_stop_gradient": self.market_context_stop_gradient,
            "hierarchical_market_roles": self.hierarchical_market_roles,
            "coarse_role_count": (
                self.coarse_role_count if self.hierarchical_market_roles else None
            ),
            "hierarchy_mix_max": (
                self.hierarchy_mix_max if self.hierarchical_market_roles else None
            ),
            "innovation_source_residual": self.innovation_source_residual,
            "source_residual_max": (
                self.source_residual_max if self.innovation_source_residual else None
            ),
            "temporal_encoder": self.temporal_arch,
            "transformer_max_len": (
                self.transformer_max_len
                if self.temporal_arch == "transformer"
                else None
            ),
            "transformer_nhead": (
                self.transformer_nhead
                if self.temporal_arch == "transformer"
                else None
            ),
            "transformer_ff_dim": (
                self.transformer_ff_dim
                if self.temporal_arch == "transformer"
                else None
            ),
            "temporal_mlp_max_len": (
                self.transformer_max_len if self.temporal_arch == "mlp" else None
            ),
            "temporal_mlp_token_dim": (
                self.temporal_mlp_token_dim if self.temporal_arch == "mlp" else None
            ),
            "temporal_mlp_hidden_dim": (
                self.temporal_mlp_hidden_dim if self.temporal_arch == "mlp" else None
            ),
            "temporal_pooling_mode": self.temporal_pooling_mode,
            "temporal_pooling_horizons": (
                list(self.temporal_pooling_horizons)
                if self.temporal_pooling_mode == "state_attentive"
                else None
            ),
            "temporal_attention_dim": (
                self.temporal_attention_dim
                if self.temporal_pooling_mode
                in {
                    "state_attentive",
                    "concat_attentive",
                    "pattern_routed_source",
                    "tra_core_routed_source",
                }
                else None
            ),
            "temporal_router_hidden_dim": (
                self.temporal_router_hidden_dim
                if self.temporal_pooling_mode
                in {
                    "state_attentive",
                    "pattern_routed_source",
                    "tra_core_routed_source",
                }
                else None
            ),
            "temporal_router_temperature": (
                self.temporal_router_temperature
                if self.temporal_pooling_mode
                in {
                    "state_attentive",
                    "pattern_routed_source",
                    "tra_core_routed_source",
                }
                else None
            ),
            "temporal_pattern_count": (
                self.pattern_count if self.pattern_routed else None
            ),
            "pattern_flow_dim": (
                self.pattern_flow_dim if self.pattern_routed else None
            ),
            "shared_pattern_reservoir": self.shared_pattern_reservoir,
            "temporal_source": (
                "tra_core_lr_routed_temporal_source_encoder"
                if self.temporal_pooling_mode == "tra_core_routed_source"
                else "pattern_routed_temporal_source_encoder"
                if self.temporal_pooling_mode == "pattern_routed_source"
                else None
            ),
            "pattern_router_scope": (
                self.pattern_router_scope if self.pattern_routed else None
            ),
            "pattern_information_route_strength": (
                self.pattern_information_route_strength
                if self.pattern_routed else None
            ),
            "router_target": (
                {
                    "both": "local_state_and_graph_information_innovation",
                    "local_only": "local_state_only",
                    "information_only": "graph_information_innovation_only",
                    "none": "disabled_equal_average",
                }[self.pattern_router_scope]
                if self.pattern_routed
                else None
            ),
            "pattern_recurrence_mode": (
                self.pattern_recurrence_mode
                if self.pattern_routed
                else None
            ),
            "pattern_state_fusion": (
                self.pattern_state_fusion if self.pattern_routed else None
            ),
            "pattern_lag_bias_init": (
                self.pattern_lag_bias_init if self.pattern_routed else None
            ),
            "pattern_local_state_mode": (
                self.pattern_local_state_mode
                if self.pattern_routed
                else None
            ),
            "temporal_attention_identity_residual": (
                self.temporal_attention_identity_residual
                if self.temporal_pooling_mode == "state_attentive"
                else None
            ),
            "temporal_pooling_short_window": (
                self.temporal_pooling_short_window
                if self.temporal_pooling_mode == "residual_multiscale"
                else None
            ),
            "d_model": self.d_model,
            "gru_hidden_dim": self.gru_hidden_dim,
            "gru_num_layers": self.gru_num_layers,
            "flow_dim": self.flow_dim,
            "latent_market_roles": self.role_count,
            "role_contrast_dim": self.role_contrast_dim,
            "role_basis": "fixed_orthonormal_helmert_contrast",
            "role_assignment": "entmax15_membership_times_dynamic_activity",
            "role_state": (
                "phi_role([Z_t,F_t_minus,Z_t*W_fF_t_minus])"
                if self.fully_coupled
                else "Z_t"
            ),
            "role_operator": "B0+Vf_diag(g_t)_WfT+Vs_diag(s_e)_WsT",
            "fast_rank": self.fast_rank,
            "active_fast_rank": self.active_fast_rank,
            "fast_coefficient_mode": self.fast_coefficient_mode,
            "slow_rank": self.slow_rank,
            "operator": "implicit_zero_diagonal_frobenius_normalized",
            "direction_convention": "K[i,j]=sender_j_to_receiver_i",
            "gamma_max": self.gamma_max,
            "dissipation_floor": self.dissipation_floor,
            "retention_scale": self.retention_scale,
            "forwarding_scale": self.forwarding_scale,
            "recurrence": (
                "F=C(c_keep*alpha_route*Fprev+K*sum_m(Um)+c_fwd*rho_route*K*Fprev)"
                if self.shared_pattern_reservoir
                else "Fm=C(c_keep*alpha_m*Fm_prev+K*Um+c_fwd*rho_m*K*Fm_prev)"
                if self.pattern_routed
                else "F=C(c_keep*alpha*Fprev+K*U+c_fwd*rho*K*Fprev)"
            ),
            "shared_stock_operator_across_patterns": self.pattern_routed,
            "single_shared_flow_state": self.shared_pattern_reservoir,
            "suspension_max_days": self.suspension_max_days,
            "learned_slow_state": self.learned_slow_state,
            "learned_slow_state_init": (
                self.learned_slow_state_init if self.learned_slow_state else None
            ),
            "se_status": (
                "outer_learned_slow_state; frozen_during_test"
                if self.learned_slow_state
                else "slow_state_interface_present; core_anchor_uses_s_e=0"
            ),
        }

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        encoder_modules = tuple(
            module
            for module in (
                self.temporal_input_projection,
                self.temporal_position_embedding,
                self.temporal_encoder,
                self.temporal_projection,
                self.temporal_residual_pooling,
                self.temporal_state_pooling,
                self.temporal_concat_pooling,
                self.temporal_pattern_pooling,
                self.local_head,
            )
            if module is not None
        )
        encoder = tuple(
            parameter
            for module in encoder_modules
            for parameter in module.parameters()
        )
        encoder_ids = {id(parameter) for parameter in encoder}
        flow = tuple(
            parameter for parameter in self.parameters() if id(parameter) not in encoder_ids
        )
        return {"encoder": encoder, "flow": flow}

    def flow_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Split state/operator generation from its bounded unified readout."""
        readout_modules = tuple(
            module
            for module in (
                self.flow_projection,
                self.fusion,
                self.prediction_head,
                self.fusion_gate,
                self.interaction_projection,
                self.shared_fusion_norm,
                self.flow_diagnostic_head,
                self.flow_reliability_gate,
                self.energy_gate,
                self.forwarding_projection,
                self.strict_norm_fusion,
            )
            if module is not None
        )
        readout = tuple(
            parameter
            for module in readout_modules
            for parameter in module.parameters()
        )
        readout_ids = {id(parameter) for parameter in readout}
        slow = (
            (self.slow_state_parameter,)
            if self.slow_state_parameter is not None
            else ()
        )
        slow_ids = {id(parameter) for parameter in slow}
        flow = self.parameter_groups()["flow"]
        graph = tuple(
            parameter
            for parameter in flow
            if id(parameter) not in readout_ids | slow_ids
        )
        if len(graph) + len(readout) + len(slow) != len(flow):
            raise AssertionError("Flow-FDG flow parameter split is not exhaustive")
        return {"graph": graph, "readout": readout, "slow": slow}

    def _encode(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 3:
            raise ValueError("features must be [stock, lookback, feature]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature dimension {features.shape[-1]} != configured {self.input_dim}"
            )
        if features.shape[0] < 2 or features.shape[1] < 1:
            raise ValueError("each day needs at least two stocks and one lookback step")
        _require_finite("features", features)
        local_features = features[..., : self.local_input_dim]
        if features.shape[1] != self.temporal_outer_lookback:
            raise ValueError(
                "feature lookback does not match temporal_outer_lookback"
            )
        if self.temporal_input_layout == ALPHA360_LAST_SNAPSHOT:
            local_features = orient_factor_axis_features(
                local_features,
                layout=ALPHA360_LAST_SNAPSHOT,
                input_dim=self.local_input_dim,
                outer_lookback=self.temporal_outer_lookback,
                alpha360_field_count=self.alpha360_field_count,
            )
        if self.temporal_arch == "transformer":
            if features.shape[1] > self.transformer_max_len:
                raise ValueError("feature lookback exceeds transformer_max_len")
            assert self.temporal_input_projection is not None
            assert self.temporal_position_embedding is not None
            positions = torch.arange(features.shape[1], device=features.device)
            tokens = self.temporal_input_projection(local_features) + (
                self.temporal_position_embedding(positions).unsqueeze(0)
            )
            causal_mask = torch.triu(
                torch.ones(
                    features.shape[1],
                    features.shape[1],
                    dtype=torch.bool,
                    device=features.device,
                ),
                diagonal=1,
            )
            encoded = self.temporal_encoder(tokens, mask=causal_mask)
            return self.temporal_projection(encoded[:, -1]), {}
        if self.temporal_arch == "mlp":
            if features.shape[1] != self.transformer_max_len:
                raise ValueError(
                    "MLP feature lookback must equal the configured temporal max length"
                )
            assert self.temporal_input_projection is not None
            tokens = self.temporal_input_projection(local_features)
            return self.temporal_projection(self.temporal_encoder(tokens)), {}
        encoded_sequence, recurrent_state = self.temporal_encoder(local_features)
        hidden = (
            recurrent_state[0]
            if isinstance(recurrent_state, tuple)
            else recurrent_state
        )
        last_state = (
            hidden.mean(dim=0)
            if self.temporal_concat_pooling is not None
            else hidden[-1]
        )
        if self.temporal_residual_pooling is not None:
            last_state = self.temporal_residual_pooling(
                encoded_sequence, last_state
            )
        temporal_diagnostics: dict[str, torch.Tensor] = {}
        if self.temporal_state_pooling is not None:
            if not isinstance(
                self.temporal_state_pooling,
                StateAttentiveTemporalPooling,
            ):
                raise TypeError("invalid temporal state pooling module")
            last_state, route, attention_entropy = (
                self.temporal_state_pooling(
                    encoded_sequence,
                    last_state,
                )
            )
            temporal_diagnostics["temporal_state_route"] = route
            temporal_diagnostics[
                "temporal_attention_entropy"
            ] = attention_entropy
        if self.temporal_concat_pooling is not None:
            if not isinstance(
                self.temporal_concat_pooling,
                ConcatenatedTemporalAttentionPooling,
            ):
                raise TypeError("invalid concatenated temporal pooling module")
            last_state, concat_entropy = self.temporal_concat_pooling(
                encoded_sequence,
                last_state,
            )
            temporal_diagnostics[
                "temporal_concat_attention_entropy"
            ] = concat_entropy
        if self.temporal_pattern_pooling is not None:
            if not isinstance(
                self.temporal_pattern_pooling,
                (
                    PatternRoutedTemporalSourceEncoder,
                    TRACoreRoutedTemporalSourceEncoder,
                ),
            ):
                raise TypeError("invalid pattern-routed temporal source module")
            pattern_terminal = (
                hidden.mean(dim=0)
                if self.pattern_local_state_mode
                == "concat_terminal_attention"
                else hidden[-1]
            )
            pattern_output = self.temporal_pattern_pooling(
                encoded_sequence,
                pattern_terminal,
            )
            temporal_diagnostics.update(
                {
                    "pattern_states": pattern_output.patterns,
                    "pattern_route": pattern_output.route,
                    "pattern_attention": pattern_output.attention,
                    "pattern_attention_entropy": pattern_output.attention_entropy,
                    "pattern_expected_lag": pattern_output.expected_lag,
                    "pattern_load_balance": pattern_output.load_balance,
                    "pattern_attention_diversity": (
                        pattern_output.attention_diversity
                    ),
                }
            )
            if pattern_output.base_attention_entropy is not None:
                temporal_diagnostics[
                    "pattern_base_attention_entropy"
                ] = pattern_output.base_attention_entropy
            if pattern_output.source_score is not None:
                temporal_diagnostics[
                    "tra_latent_route_source_score"
                ] = pattern_output.source_score
            return pattern_output.fused, temporal_diagnostics
        return self.temporal_projection(last_state), temporal_diagnostics

    def _market_context(
        self,
        z: torch.Tensor,
        market_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_source = z.detach() if self.market_context_stop_gradient else z
        mean = context_source.mean(dim=0)
        scale = torch.sqrt(
            context_source.var(dim=0, unbiased=False) + self.eps
        )
        context = self.context_encoder(torch.cat([mean, scale], dim=-1))
        if self.market_feature_dim == 0:
            if market_features is not None:
                raise ValueError("market features were supplied to a stock-only model")
            return context
        if market_features is None or market_features.shape != (
            z.shape[0],
            self.market_feature_dim,
        ):
            raise ValueError("market_features must be [stock, market_feature_dim]")
        if self.market_feature_encoder is None or self.market_context_norm is None:
            raise TypeError("market context modules are missing")
        pooled_market = market_features.detach().mean(dim=0)
        market_context = self.market_feature_encoder(pooled_market)
        return self.market_context_norm(context + market_context)

    def _role_coordinates(
        self, z: torch.Tensor, context: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        normalized = self.role_norm(z)
        role_scale = sqrt(float(self.role_projection_dim))
        sender_logits = (
            self.sender_query(normalized) @ self.sender_prototypes.transpose(0, 1)
        ) / (role_scale * self.sender_temperature)
        receiver_logits = (
            self.receiver_query(normalized) @ self.receiver_prototypes.transpose(0, 1)
        ) / (role_scale * self.receiver_temperature)
        base_sender = entmax15(sender_logits, dim=-1)
        base_receiver = entmax15(receiver_logits, dim=-1)
        sender_membership = base_sender
        receiver_membership = base_receiver
        role_diagnostics: dict[str, torch.Tensor] = {
            "S_base": base_sender,
            "R_base": base_receiver,
        }
        if self.hierarchical_role_residual is not None:
            if not isinstance(
                self.hierarchical_role_residual,
                HierarchicalMarketRoleResidual,
            ):
                raise TypeError("invalid hierarchical role residual module")
            hierarchical = self.hierarchical_role_residual(
                z,
                context,
                base_sender,
                base_receiver,
            )
            sender_membership = hierarchical.sender
            receiver_membership = hierarchical.receiver
            role_diagnostics.update(
                {
                    "C_sender": hierarchical.coarse_sender,
                    "C_receiver": hierarchical.coarse_receiver,
                    "S_hierarchical_candidate": hierarchical.sender_candidate,
                    "R_hierarchical_candidate": hierarchical.receiver_candidate,
                    "sender_hierarchy_mix": hierarchical.sender_mix,
                    "receiver_hierarchy_mix": hierarchical.receiver_mix,
                }
            )
        expanded_context = context.unsqueeze(0).expand(z.shape[0], -1)
        activity_features = torch.cat([z, expanded_context], dim=-1)
        sender_activity = torch.sigmoid(self.sender_activity(activity_features)).squeeze(-1)
        receiver_activity = torch.sigmoid(self.receiver_activity(activity_features)).squeeze(-1)
        sender_weighted = sender_membership * sender_activity.unsqueeze(-1)
        receiver_weighted = receiver_membership * receiver_activity.unsqueeze(-1)
        sender_coordinates = center_stocks(sender_weighted) @ self.role_contrast_basis
        receiver_coordinates = center_stocks(receiver_weighted) @ self.role_contrast_basis
        return (
            sender_membership,
            receiver_membership,
            sender_activity,
            receiver_activity,
            sender_coordinates,
            receiver_coordinates,
            role_diagnostics,
        )

    def _role_operator(
        self, context: torch.Tensor, slow_state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast_logits = self.fast_state_head(context)
        if self.fast_coefficient_mode == "signed_tanh":
            fast_state = torch.tanh(fast_logits)
        else:
            fast_state = torch.sigmoid(fast_logits)
        fast_state = fast_state * self.fast_rank_mask
        if slow_state is None:
            resolved_slow = (
                context.new_zeros(self.slow_rank)
                if self.slow_state_parameter is None
                else torch.tanh(self.slow_state_parameter)
            )
        else:
            if slow_state.shape != (self.slow_rank,):
                raise ValueError("slow_state must match configured slow_rank")
            if slow_state.device != context.device or slow_state.dtype != context.dtype:
                raise ValueError("slow_state must share context dtype and device")
            _require_finite("slow_state", slow_state)
            resolved_slow = torch.tanh(slow_state)
        fast_delta = (self.fast_left * fast_state.unsqueeze(0)) @ self.fast_right.transpose(0, 1)
        slow_delta = (self.slow_left * resolved_slow.unsqueeze(0)) @ self.slow_right.transpose(0, 1)
        return self.base_role_operator + fast_delta + slow_delta, fast_state, resolved_slow

    def _mixing_coefficients(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pattern_routed:
            if self.pattern_mix_logits is None:
                raise TypeError("pattern mixing logits are missing")
            base = torch.softmax(self.pattern_mix_logits, dim=-1)
            alpha = base[:, 0]
            rho = base[:, 1]
            delta = base[:, 2]
        else:
            if self.mix_base_logits is None or self.mix_head is None:
                raise TypeError("single-channel mixing parameters are missing")
            base = torch.softmax(
                self.mix_base_logits + self.mix_head(context),
                dim=-1,
            )
            available = 1.0 - self.dissipation_floor
            alpha = available * base[0]
            rho = available * base[1]
            delta = self.dissipation_floor + available * base[2]
        gamma = self.gamma_max * torch.sigmoid(
            self.gamma_base_logit + self.gamma_head(context).squeeze(-1)
        )
        return alpha, rho, delta, gamma

    def _resolve_state(
        self, z: torch.Tensor, state: FlowFDGState | None, alpha: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stocks = z.shape[0]
        if state is None:
            return z.new_zeros(stocks, self.flow_dim), torch.ones(
                stocks, dtype=torch.long, device=z.device
            )
        if state.flow_prev.shape != (stocks, self.flow_dim):
            raise ValueError("state.flow_prev must be instrument-aligned [stock, flow_dim]")
        if state.gap_days.shape != (stocks,) or state.gap_days.dtype != torch.long:
            raise ValueError("state.gap_days must be int64 [stock]")
        if state.flow_prev.device != z.device or state.flow_prev.dtype != z.dtype:
            raise ValueError("state.flow_prev must share feature dtype and device")
        if state.gap_days.device != z.device:
            raise ValueError("state.gap_days must share feature device")
        _require_finite("state.flow_prev", state.flow_prev)
        if torch.any(state.gap_days < 1):
            raise ValueError("state gaps must be at least one trading day")
        missing_days = state.gap_days - 1
        retained = missing_days <= self.suspension_max_days
        if self.shared_pattern_reservoir:
            if alpha.shape not in {(stocks,), (stocks, 1)}:
                raise ValueError(
                    "shared-reservoir alpha must be stock aligned"
                )
            stock_alpha = alpha.reshape(stocks)
            decay = stock_alpha.pow(
                missing_days.to(dtype=z.dtype)
            ).unsqueeze(-1)
            flow_minus = (
                state.flow_prev * decay * retained.unsqueeze(-1)
            )
        elif self.pattern_routed:
            if alpha.shape != (self.pattern_count,):
                raise ValueError("pattern alpha must be [pattern]")
            previous = state.flow_prev.reshape(
                stocks, self.pattern_count, self.pattern_flow_dim
            )
            decay = alpha.unsqueeze(0).pow(
                missing_days.to(dtype=z.dtype).unsqueeze(-1)
            )
            resolved = (
                previous
                * decay.unsqueeze(-1)
                * retained[:, None, None]
            )
            flow_minus = resolved.reshape(stocks, self.flow_dim)
        else:
            if alpha.numel() != 1:
                raise ValueError("single-channel alpha must be scalar")
            decay = alpha.pow(missing_days.to(dtype=z.dtype)).unsqueeze(-1)
            flow_minus = state.flow_prev * decay * retained.unsqueeze(-1)
        return flow_minus, state.gap_days

    def _resolve_pattern_previous(
        self,
        z: torch.Tensor,
        state: FlowFDGState | None,
        gaps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stocks = z.shape[0]
        expected = (stocks, self.pattern_count, self.pattern_state_dim)
        if not self.pattern_routed:
            raise RuntimeError("pattern state requested by a single-channel model")
        if state is None or state.pattern_prev is None:
            return z.new_zeros(expected), torch.zeros(
                stocks, dtype=torch.bool, device=z.device
            )
        if state.pattern_prev.shape != expected:
            raise ValueError(
                "state.pattern_prev must be [stock, pattern, pattern_state_dim]"
            )
        if (
            state.pattern_prev.device != z.device
            or state.pattern_prev.dtype != z.dtype
        ):
            raise ValueError("state.pattern_prev must share feature dtype and device")
        _require_finite("state.pattern_prev", state.pattern_prev)
        if state.pattern_valid is None:
            valid = gaps == 1
        else:
            if (
                state.pattern_valid.shape != (stocks,)
                or state.pattern_valid.dtype != torch.bool
            ):
                raise ValueError("state.pattern_valid must be bool [stock]")
            if state.pattern_valid.device != z.device:
                raise ValueError("state.pattern_valid must share feature device")
            valid = state.pattern_valid
        valid = valid & (gaps == 1)
        return state.pattern_prev, valid

    def explicit_operator(
        self,
        receiver_coordinates: torch.Tensor,
        role_operator: torch.Tensor,
        sender_coordinates: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        return explicit_offdiag_operator(
            receiver_coordinates, role_operator, sender_coordinates, gamma
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        state: FlowFDGState | None = None,
        slow_state: torch.Tensor | None = None,
        compute_graph_diagnostics: bool = False,
    ) -> FlowFDGOutput:
        z, temporal_diagnostics = self._encode(features)
        market_features = (
            features[:, -1, self.local_input_dim :]
            if self.market_feature_dim > 0
            else None
        )
        pattern_states = temporal_diagnostics.pop("pattern_states", None)
        if self.pattern_routed:
            if pattern_states is None:
                raise TypeError("pattern-routed encoder omitted pattern states")
            if self.pattern_router_scope == "none":
                temporal_diagnostics["pattern_route"] = z.new_full(
                    (z.shape[0], self.pattern_count),
                    1.0 / float(self.pattern_count),
                )
                temporal_diagnostics["pattern_load_balance"] = z.new_zeros(())
            if (
                self.pattern_local_state_mode == "pattern_routed"
                and self.pattern_router_scope in {"information_only", "none"}
            ):
                z = pattern_states.mean(dim=1)
        seed_context = self._market_context(z, market_features)
        seed_alpha, _, _, _ = self._mixing_coefficients(seed_context)
        if self.shared_pattern_reservoir:
            seed_route = temporal_diagnostics["pattern_route"]
            seed_alpha = (seed_route * seed_alpha.unsqueeze(0)).sum(dim=-1)
        flow_minus, gaps = self._resolve_state(z, state, seed_alpha)
        if self.fully_coupled:
            if self.role_flow_projection is None or self.role_state_encoder is None:
                raise TypeError("fully coupled role-state modules are missing")
            if self.couple_lagged_flow_to_roles:
                role_flow_input = (
                    flow_minus.detach()
                    if self._detach_lagged_flow_roles
                    else flow_minus
                )
                role_flow = self.role_flow_projection(role_flow_input)
            else:
                role_flow = z.new_zeros(z.shape)
            role_state = self.role_state_encoder(
                torch.cat(
                    [z, role_flow, z * role_flow],
                    dim=-1,
                )
            )
        else:
            role_flow = z.new_zeros(z.shape)
            role_state = z
        context = self._market_context(role_state, market_features)
        p_local: torch.Tensor | None = None
        if not self.fully_coupled:
            if self.local_head is None:
                raise TypeError("legacy local diagnostic head is missing")
            tra_source_score = temporal_diagnostics.get(
                "tra_latent_route_source_score"
            )
            p_local = center_stocks(
                tra_source_score
                if tra_source_score is not None
                else self.local_head(z).squeeze(-1)
            )
        (
            sender_membership,
            receiver_membership,
            sender_activity,
            receiver_activity,
            sender_coordinates,
            receiver_coordinates,
            role_diagnostics,
        ) = self._role_coordinates(role_state, context)
        role_operator, fast_state, resolved_slow = self._role_operator(
            context, slow_state
        )
        alpha, rho, delta, gamma = self._mixing_coefficients(context)
        pattern_alpha = alpha
        pattern_rho = rho
        pattern_delta = delta
        if self.shared_pattern_reservoir:
            route = temporal_diagnostics["pattern_route"]
            alpha = (route * pattern_alpha.unsqueeze(0)).sum(dim=-1, keepdim=True)
            rho = (route * pattern_rho.unsqueeze(0)).sum(dim=-1, keepdim=True)
            delta = (route * pattern_delta.unsqueeze(0)).sum(dim=-1, keepdim=True)

        pattern_information: torch.Tensor | None = None
        pattern_change_rms: torch.Tensor | None = None
        source_diagnostics: dict[str, torch.Tensor] = {}
        if self.pattern_routed:
            if pattern_states is None:
                raise TypeError("pattern-routed encoder omitted pattern states")
            route = temporal_diagnostics["pattern_route"]
            if route.shape != (z.shape[0], self.pattern_count):
                raise ValueError("pattern route has invalid shape")
            equal_route = route.new_full(
                route.shape,
                1.0 / float(self.pattern_count),
            )
            if self.pattern_router_scope in {"both", "information_only"}:
                information_route = equal_route + (
                    self.pattern_information_route_strength
                    * (route - equal_route)
                )
            else:
                information_route = equal_route
            previous_patterns, pattern_valid = self._resolve_pattern_previous(
                z, state, gaps
            )
            pattern_change = (
                pattern_states - previous_patterns
            ) * pattern_valid[:, None, None]
            pattern_change_rms = torch.sqrt(
                pattern_change.square().mean(dim=(0, 2)) + self.eps
            )
            if self.pattern_info_encoders is None:
                raise TypeError("pattern information encoders are missing")
            pattern_information = torch.stack(
                [
                    information_route[:, mode : mode + 1]
                    * encoder(
                        torch.cat(
                            [
                                pattern_states[:, mode],
                                pattern_change[:, mode],
                            ],
                            dim=-1,
                        )
                    )
                    for mode, encoder in enumerate(self.pattern_info_encoders)
                ],
                dim=1,
            )
            pattern_information = center_stocks(pattern_information)
            if self.shared_pattern_reservoir:
                information = center_stocks(pattern_information.sum(dim=1))
            else:
                information = pattern_information.reshape(z.shape[0], self.flow_dim)
        else:
            if self.info_encoder is None:
                raise TypeError("single-channel information encoder is missing")
            if p_local is None:
                raise TypeError("A9 information source requires the local diagnostic")
            base_information = center_stocks(
                self.info_encoder(
                    torch.cat([z, p_local.detach().unsqueeze(-1)], dim=-1)
                )
            )
            information = base_information
            if self.innovation_source_module is not None:
                if not isinstance(
                    self.innovation_source_module,
                    InnovationSourceResidual,
                ):
                    raise TypeError("invalid innovation source residual module")
                coarse_sender = role_diagnostics.get("C_sender")
                if coarse_sender is None:
                    raise TypeError(
                        "innovation source requires coarse sender memberships"
                    )
                innovation_output = self.innovation_source_module(
                    z,
                    context,
                    coarse_sender,
                    base_information,
                )
                information = innovation_output.information
                source_diagnostics.update(
                    {
                        "U_base": base_information,
                        "U_group_innovation": innovation_output.innovation,
                        "U_group_state": innovation_output.group_state,
                        "U_residual": innovation_output.residual,
                        "U_residual_gate": innovation_output.gate,
                        "U_residual_relative_norm": (
                            innovation_output.relative_norm
                        ),
                    }
                )

        joined_payload = torch.cat([information, flow_minus], dim=-1)
        propagated, removed_diagonal, operator_frobenius, operator_scale = (
            implicit_offdiag_apply(
                receiver_coordinates,
                role_operator,
                sender_coordinates,
                joined_payload,
                gamma,
                eps=self.eps,
            )
        )
        propagated_new, propagated_history = propagated.split(self.flow_dim, dim=-1)
        effective_alpha = self.retention_scale * alpha
        effective_rho = self.forwarding_scale * rho
        if self.shared_pattern_reservoir:
            keep = effective_alpha * flow_minus
            new = propagated_new
            forwarding = effective_rho * propagated_history
            flow = center_stocks(keep + new + forwarding)
        elif self.pattern_routed:
            flow_minus_modes = flow_minus.reshape(
                z.shape[0], self.pattern_count, self.pattern_flow_dim
            )
            new_modes = propagated_new.reshape(
                z.shape[0], self.pattern_count, self.pattern_flow_dim
            )
            history_modes = propagated_history.reshape(
                z.shape[0], self.pattern_count, self.pattern_flow_dim
            )
            if self.pattern_recurrence_mode == "one_step":
                keep_modes = torch.zeros_like(flow_minus_modes)
                forwarding_modes = torch.zeros_like(history_modes)
                flow_modes = center_stocks(new_modes)
            else:
                keep_modes = effective_alpha[None, :, None] * flow_minus_modes
                forwarding_modes = (
                    effective_rho[None, :, None] * history_modes
                )
                flow_modes = center_stocks(
                    keep_modes + new_modes + forwarding_modes
                )
            keep = keep_modes.reshape(z.shape[0], self.flow_dim)
            new = new_modes.reshape(z.shape[0], self.flow_dim)
            forwarding = forwarding_modes.reshape(z.shape[0], self.flow_dim)
            flow = flow_modes.reshape(z.shape[0], self.flow_dim)
        else:
            keep = effective_alpha * flow_minus
            new = propagated_new
            forwarding = effective_rho * propagated_history
            flow = center_stocks(keep + new + forwarding)

        strict_fusion_output = None
        if self.readout_mode == "strict_norm_bounded_shared":
            flow_projection = z.new_zeros(z.shape)
        else:
            flow_projection = self.flow_projection(flow)
        normalized_flow_projection: torch.Tensor | None = None
        energy_gate_value: torch.Tensor | None = None
        if self.normalized_energy_gate:
            if self.energy_gate is None:
                raise TypeError("normalized energy gate is missing")
            flow_rms = torch.sqrt(
                flow_projection.square().mean(dim=0, keepdim=True) + self.eps
            )
            normalized_flow_projection = flow_projection / flow_rms
            route = temporal_diagnostics["pattern_route"]
            energy_gate_value = self.max_flow_fraction * torch.sigmoid(
                self.energy_gate(
                    torch.cat([z, normalized_flow_projection, route], dim=-1)
                )
            )
            flow_projection = energy_gate_value * normalized_flow_projection
        if self.forwarding_projection is None:
            forwarding_readout_projection = flow_projection.new_zeros(
                flow_projection.shape
            )
        else:
            forwarding_readout_projection = torch.tanh(
                self.forwarding_projection(forwarding)
            )
            flow_projection = flow_projection + (
                self.forwarding_readout_scale * forwarding_readout_projection
            )
        if self._force_coupled_temporal_only:
            flow_projection = torch.zeros_like(flow_projection)
        if self.readout_mode == "independent_concat":
            assert self.fusion is not None and self.prediction_head is not None
            fused = self.fusion(
                torch.cat([z, flow_projection, z * flow_projection], dim=-1)
            )
            prediction = center_stocks(self.prediction_head(fused).squeeze(-1))
            fusion_gate = z.new_full(z.shape, float("nan"))
        elif self.readout_mode == "strict_norm_bounded_shared":
            if not isinstance(self.strict_norm_fusion, StrictNormBoundedFusion):
                raise TypeError("strict norm-bounded fusion module is missing")
            if self.local_head is None:
                raise TypeError("strict shared readout requires the A9 local head")
            strict_fusion_output = self.strict_norm_fusion(z, flow, context)
            fused = strict_fusion_output.fused
            flow_projection = strict_fusion_output.projected_flow
            flow_fraction = strict_fusion_output.fraction
            fusion_gate = 1.0 - flow_fraction
            prediction = center_stocks(self.local_head(fused).squeeze(-1))
        else:
            assert self.fusion_gate is not None
            assert self.interaction_projection is not None
            assert self.shared_fusion_norm is not None
            gate_logits = self.fusion_gate(
                torch.cat([z, flow_projection], dim=-1)
            )
            if self.readout_mode == "bounded_shared_gated":
                if self.flow_reliability_gate is None:
                    flow_reliability = gate_logits.new_ones(gate_logits.shape[0])
                    reliability_evidence = gate_logits.new_zeros(
                        gate_logits.shape[0], 5
                    )
                else:
                    state_rms = torch.sqrt(flow.square().mean(dim=-1) + self.eps)
                    new_rms = torch.sqrt(new.square().mean(dim=-1) + self.eps)
                    forwarding_rms = torch.sqrt(
                        forwarding.square().mean(dim=-1) + self.eps
                    )
                    state_level = state_rms / (1.0 + state_rms)
                    state_share = state_rms / (state_rms + new_rms + self.eps)
                    forwarding_share = forwarding_rms / (
                        forwarding_rms + new_rms + self.eps
                    )
                    operator_strength = operator_frobenius / (
                        1.0 + operator_frobenius
                    )
                    reliability_evidence = torch.stack(
                        [
                            state_level,
                            state_share,
                            forwarding_share,
                            gamma.expand_as(state_level),
                            operator_strength.expand_as(state_level),
                        ],
                        dim=-1,
                    )
                    raw_reliability = torch.sigmoid(
                        self.flow_reliability_gate(reliability_evidence)
                    ).squeeze(-1)
                    flow_reliability = self.flow_reliability_floor + (
                        1.0 - self.flow_reliability_floor
                    ) * raw_reliability
                flow_fraction = (
                    self.max_flow_fraction
                    * torch.sigmoid(gate_logits)
                    * flow_reliability.unsqueeze(-1)
                )
                bounded_interaction = torch.tanh(
                    self.interaction_projection(z * flow_projection)
                )
                flow_candidate = flow_projection + bounded_interaction
                fusion_gate = 1.0 - flow_fraction
                fused = self.shared_fusion_norm(
                    fusion_gate * z + flow_fraction * flow_candidate
                )
            else:
                fusion_gate = torch.sigmoid(gate_logits)
                flow_fraction = 1.0 - fusion_gate
                fused = self.shared_fusion_norm(
                    fusion_gate * z
                    + flow_fraction * flow_projection
                    + self.interaction_projection(z * flow_projection)
                )
            prediction = center_stocks(self.local_head(fused).squeeze(-1))
        flow_readout: torch.Tensor | None = None
        if self.flow_diagnostic_head is not None:
            flow_readout = center_stocks(
                self.flow_diagnostic_head(flow).squeeze(-1)
            )
        next_state = FlowFDGState(
            flow_prev=flow,
            gap_days=torch.ones_like(gaps),
            pattern_prev=pattern_states,
            pattern_valid=(
                torch.ones(
                    z.shape[0],
                    dtype=torch.bool,
                    device=z.device,
                )
                if pattern_states is not None
                else None
            ),
        )
        diagnostic_alpha = alpha.mean() if alpha.ndim > 0 else alpha
        diagnostic_rho = rho.mean() if rho.ndim > 0 else rho
        diagnostic_delta = delta.mean() if delta.ndim > 0 else delta
        diagnostic_effective_rho = (
            effective_rho.mean()
            if effective_rho.ndim > 0
            else effective_rho
        )
        diagnostic_effective_alpha = (
            effective_alpha.mean()
            if effective_alpha.ndim > 0
            else effective_alpha
        )
        diagnostics: dict[str, torch.Tensor] = {
            "Z": z,
            "Q": role_state,
            "F_role": role_flow,
            "market_context": context,
            "S": sender_membership,
            "R": receiver_membership,
            "sender_activity": sender_activity,
            "receiver_activity": receiver_activity,
            "sender_coordinates": sender_coordinates,
            "receiver_coordinates": receiver_coordinates,
            "B": role_operator,
            "g_fast": fast_state,
            "s_slow": resolved_slow,
            "alpha": diagnostic_alpha,
            "effective_alpha": diagnostic_effective_alpha,
            "rho": diagnostic_rho,
            "effective_rho": diagnostic_effective_rho,
            "delta": diagnostic_delta,
            "gamma": gamma,
            "U": information,
            "F_minus": flow_minus,
            "P_keep": keep,
            "P_new": new,
            "P_fwd": forwarding,
            "P_fwd_readout": forwarding_readout_projection,
            "F": flow,
            "coupled_temporal_only": z.new_tensor(
                float(self._force_coupled_temporal_only)
            ),
            "lagged_flow_roles_detached": z.new_tensor(
                float(self._detach_lagged_flow_roles)
            ),
            "operator_raw_diagonal": removed_diagonal,
            "operator_frobenius": operator_frobenius,
            "operator_scale": operator_scale,
            "contraction_upper_bound": (
                (effective_alpha + effective_rho * gamma).max()
                if self.pattern_recurrence_mode == "full"
                else alpha.new_zeros(())
            ),
        }
        diagnostics.update(role_diagnostics)
        diagnostics.update(source_diagnostics)
        if self.pattern_routed:
            if (
                pattern_information is None
                or pattern_change_rms is None
            ):
                raise TypeError("pattern diagnostics are incomplete")
            diagnostics.update(
                {
                    "pattern_alpha": pattern_alpha,
                    "pattern_rho": pattern_rho,
                    "pattern_delta": pattern_delta,
                    "pattern_lifetime_simplex_error": (
                        pattern_alpha + pattern_rho + pattern_delta - 1.0
                    ).abs().max(),
                    "pattern_information": pattern_information,
                    "pattern_change_rms": pattern_change_rms,
                }
            )
        diagnostics.update(temporal_diagnostics)
        if self.pattern_count == 2 and "pattern_expected_lag" in diagnostics:
            expected_lag = diagnostics["pattern_expected_lag"]
            diagnostics["pattern_lag_order_violation"] = torch.relu(
                expected_lag[:, 0] - expected_lag[:, 1] + 1.0
            ).mean()
        if energy_gate_value is not None:
            diagnostics["flow_fraction"] = energy_gate_value
            diagnostics["flow_energy_mean"] = energy_gate_value.mean()
            assert normalized_flow_projection is not None
            diagnostics["normalized_flow_projection"] = normalized_flow_projection
        if self.readout_mode in {
            "shared_gated",
            "bounded_shared_gated",
            "strict_norm_bounded_shared",
        }:
            diagnostics["fusion_gate"] = fusion_gate
            diagnostics["flow_fraction"] = flow_fraction
        if self.readout_mode == "bounded_shared_gated":
            diagnostics["flow_reliability"] = flow_reliability
            diagnostics["flow_reliability_evidence"] = reliability_evidence
        if self.readout_mode == "strict_norm_bounded_shared":
            if strict_fusion_output is None:
                raise TypeError("strict fusion diagnostics are missing")
            diagnostics.update(
                {
                    "flow_energy_mean": flow_fraction.mean(),
                    "flow_projection": strict_fusion_output.projected_flow,
                    "fusion_correction": strict_fusion_output.correction,
                    "fusion_delta": strict_fusion_output.delta,
                    "fusion_relative_norm": strict_fusion_output.relative_norm,
                }
            )
        if compute_graph_diagnostics:
            diagnostics["K"] = explicit_offdiag_operator(
                receiver_coordinates,
                role_operator,
                sender_coordinates,
                gamma,
            )
        finite_values: list[tuple[str, torch.Tensor]] = [
            ("prediction", prediction),
            ("flow", flow),
            *diagnostics.items(),
        ]
        if p_local is not None:
            finite_values.append(("p_local", p_local))
        if flow_readout is not None:
            finite_values.append(("flow_readout", flow_readout))
        for name, value in finite_values:
            _require_finite(name, value)
        return FlowFDGOutput(
            prediction=prediction,
            p_local=p_local,
            flow_readout=flow_readout,
            flow=flow,
            state=next_state,
            diagnostics=diagnostics,
        )


__all__ = [
    "FlowFDG",
    "FlowFDGOutput",
    "FlowFDGState",
    "center_stocks",
    "explicit_offdiag_operator",
    "explicit_offdiag_operator_counterfactual",
    "helmert_contrast",
    "implicit_offdiag_apply",
]
