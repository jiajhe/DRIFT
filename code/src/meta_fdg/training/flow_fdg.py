from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from meta_fdg.data.temporal_cross_section import TemporalDay, TemporalSplit
from meta_fdg.models import FlowFDG, FlowFDGOutput, FlowFDGState
from meta_fdg.models.flow_fdg import center_stocks

from .metrics import daily_signal_metrics, summarize_daily_metrics


FIXED_SPLITS = {
    "train": ["2016-01-01", "2020-12-31"],
    "valid": ["2021-01-01", "2021-12-31"],
    "test": ["2022-01-01", "2025-12-31"],
}
SHIFT2Y_SHORTTEST_LOCK_ID = "shift2y-shorttest-v1-20260721"
SHIFT2Y_SHORTTEST_SPLITS = {
    "train": ["2018-01-01", "2022-12-31"],
    "valid": ["2023-01-01", "2023-12-31"],
    "test": ["2024-01-01", "2025-12-31"],
}


@dataclass(frozen=True)
class FlowAlignmentAudit:
    carried: int
    entered: int
    reentered: int
    exited: int
    reset_after_long_gap: int


@dataclass(frozen=True)
class FlowEpochResult:
    metrics: dict[str, Any]
    component_means: dict[str, float]


@dataclass(frozen=True)
class FlowEvaluationResult:
    daily_full: pd.DataFrame
    daily_local: pd.DataFrame
    daily_flow: pd.DataFrame
    predictions_full: pd.DataFrame
    predictions_local: pd.DataFrame
    predictions_flow: pd.DataFrame
    diagnostics_daily: pd.DataFrame
    summary_full: dict[str, Any]
    summary_local: dict[str, Any]
    summary_flow: dict[str, Any]
    counterfactual_daily: Mapping[str, pd.DataFrame]
    counterfactual_predictions: Mapping[str, pd.DataFrame]
    counterfactual_summaries: Mapping[str, dict[str, Any]]


@dataclass(frozen=True)
class FlowHopDecomposition:
    """Exact direct-receipt versus hop-at-least-one state decomposition."""

    direct: torch.Tensor
    multi_hop: torch.Tensor
    direct_minus: torch.Tensor
    multi_hop_minus: torch.Tensor
    reconstruction_max_abs: torch.Tensor


class FlowStateBank:
    """Chronological per-instrument Flow and temporal-pattern state."""

    def __init__(
        self,
        *,
        flow_dim: int,
        suspension_max_days: int,
        pattern_count: int = 0,
        pattern_state_dim: int = 0,
    ) -> None:
        self.flow_dim = int(flow_dim)
        self.suspension_max_days = int(suspension_max_days)
        self.pattern_count = int(pattern_count)
        self.pattern_state_dim = int(pattern_state_dim)
        if (self.pattern_count == 0) != (self.pattern_state_dim == 0):
            raise ValueError(
                "pattern_count and pattern_state_dim must both be zero or positive"
            )
        self.values: dict[str, torch.Tensor] = {}
        self.pattern_values: dict[str, torch.Tensor] = {}
        self.last_seen: dict[str, int] = {}
        self.previous: tuple[str, ...] = ()

    @staticmethod
    def _unique(instruments: Sequence[str]) -> tuple[str, ...]:
        values = tuple(str(value) for value in instruments)
        if len(values) != len(set(values)):
            raise ValueError("daily instruments must be unique")
        return values

    def prepare(
        self,
        instruments: Sequence[str],
        *,
        day_index: int,
        reference: torch.Tensor,
    ) -> tuple[FlowFDGState, FlowAlignmentAudit]:
        current = self._unique(instruments)
        previous_set = set(self.previous)
        values: list[torch.Tensor] = []
        pattern_values: list[torch.Tensor] = []
        pattern_valid: list[bool] = []
        gaps: list[int] = []
        carried = reentered = reset = 0
        for instrument in current:
            if instrument not in self.values:
                values.append(reference.new_zeros(self.flow_dim))
                if self.pattern_count:
                    pattern_values.append(
                        reference.new_zeros(
                            self.pattern_count,
                            self.pattern_state_dim,
                        )
                    )
                    pattern_valid.append(False)
                gaps.append(1)
                continue
            gap = int(day_index) - int(self.last_seen[instrument])
            if gap < 1:
                raise ValueError("day_index must advance strictly chronologically")
            if gap == 1:
                carried += 1
            else:
                reentered += 1
            long_gap = gap - 1 > self.suspension_max_days
            if long_gap:
                values.append(reference.new_zeros(self.flow_dim))
                reset += 1
            else:
                values.append(self.values[instrument])
            if self.pattern_count:
                has_previous_pattern = (
                    gap == 1
                    and not long_gap
                    and instrument in self.pattern_values
                )
                pattern_values.append(
                    self.pattern_values[instrument]
                    if has_previous_pattern
                    else reference.new_zeros(
                        self.pattern_count,
                        self.pattern_state_dim,
                    )
                )
                pattern_valid.append(has_previous_pattern)
            gaps.append(gap)
        state = FlowFDGState(
            flow_prev=torch.stack(values, dim=0),
            gap_days=torch.tensor(gaps, dtype=torch.long, device=reference.device),
            pattern_prev=(
                torch.stack(pattern_values, dim=0)
                if self.pattern_count
                else None
            ),
            pattern_valid=(
                torch.tensor(
                    pattern_valid,
                    dtype=torch.bool,
                    device=reference.device,
                )
                if self.pattern_count
                else None
            ),
        )
        current_set = set(current)
        entered = sum(instrument not in previous_set for instrument in current)
        audit = FlowAlignmentAudit(
            carried=carried,
            entered=entered,
            reentered=reentered,
            exited=len(previous_set - current_set),
            reset_after_long_gap=reset,
        )
        return state, audit

    def update(
        self,
        instruments: Sequence[str],
        flow: torch.Tensor,
        *,
        day_index: int,
        pattern_state: torch.Tensor | None = None,
    ) -> None:
        current = self._unique(instruments)
        if flow.shape != (len(current), self.flow_dim):
            raise ValueError("flow state shape does not match current instruments")
        if self.pattern_count:
            expected = (
                len(current),
                self.pattern_count,
                self.pattern_state_dim,
            )
            if pattern_state is None or pattern_state.shape != expected:
                raise ValueError(
                    "pattern state shape does not match current instruments"
                )
        elif pattern_state is not None:
            raise ValueError("single-channel state bank cannot store pattern state")
        for position, instrument in enumerate(current):
            self.values[instrument] = flow[position]
            if pattern_state is not None:
                self.pattern_values[instrument] = pattern_state[position]
            self.last_seen[instrument] = int(day_index)
        self.previous = current

    def detach(self) -> None:
        self.values = {name: value.detach() for name, value in self.values.items()}
        self.pattern_values = {
            name: value.detach() for name, value in self.pattern_values.items()
        }

def align_explicit_operator_by_instrument(
    source_operator: torch.Tensor,
    source_instruments: Sequence[str],
    target_instruments: Sequence[str],
) -> tuple[torch.Tensor, int]:
    """Align a saved stock operator to another day's instrument order.

    Stocks absent from the source day receive zero rows and columns. Existing
    sender/receiver entries retain their source-day values, so a path-order
    placebo changes only operator time order rather than stock identity.
    """

    source_names = tuple(str(value) for value in source_instruments)
    target_names = tuple(str(value) for value in target_instruments)
    if len(source_names) != len(set(source_names)):
        raise ValueError("source instruments must be unique")
    if len(target_names) != len(set(target_names)):
        raise ValueError("target instruments must be unique")
    if source_operator.shape != (len(source_names), len(source_names)):
        raise ValueError("source operator shape must match source instruments")
    if not torch.isfinite(source_operator).all():
        raise FloatingPointError("source operator must be finite")
    if not torch.allclose(
        torch.diagonal(source_operator),
        source_operator.new_zeros(len(source_names)),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("source operator must have zero diagonal")

    source_positions = {name: index for index, name in enumerate(source_names)}
    matched_target: list[int] = []
    matched_source: list[int] = []
    for target_index_value, name in enumerate(target_names):
        source_index_value = source_positions.get(name)
        if source_index_value is not None:
            matched_target.append(target_index_value)
            matched_source.append(source_index_value)
    aligned = source_operator.new_zeros(len(target_names), len(target_names))
    if matched_target:
        target_index = torch.tensor(
            matched_target, dtype=torch.long, device=source_operator.device
        )
        source_index = torch.tensor(
            matched_source, dtype=torch.long, device=source_operator.device
        )
        aligned[target_index[:, None], target_index[None, :]] = source_operator[
            source_index[:, None], source_index[None, :]
        ]
    return aligned, len(matched_target)


def decompose_flow_hops(
    model: FlowFDG,
    output: FlowFDGOutput,
    direct_state: FlowFDGState,
) -> FlowHopDecomposition:
    """Split the current state into direct receipt and hop-at-least-one parts.

    The decomposition is algebraically exact for the existing recurrence.  It
    keeps direct information while it is merely retained and moves it into the
    multi-hop component only after another application of the stock operator.
    """

    flow_minus = output.diagnostics["F_minus"]
    alpha = output.diagnostics["alpha"]
    if direct_state.flow_prev.shape != flow_minus.shape:
        raise ValueError("direct_state must align with the resolved Flow state")
    if direct_state.gap_days.shape != flow_minus.shape[:1]:
        raise ValueError("direct_state gaps must align with the stock axis")
    if direct_state.gap_days.dtype != torch.long:
        raise ValueError("direct_state gaps must be int64")
    if direct_state.flow_prev.device != flow_minus.device:
        raise ValueError("direct_state must share the Flow device")
    if direct_state.flow_prev.dtype != flow_minus.dtype:
        raise ValueError("direct_state must share the Flow dtype")
    if direct_state.gap_days.device != flow_minus.device:
        raise ValueError("direct_state gaps must share the Flow device")
    if torch.any(direct_state.gap_days < 1):
        raise ValueError("direct_state gaps must be at least one day")

    missing_days = direct_state.gap_days - 1
    retained = missing_days <= model.suspension_max_days
    decay = alpha.pow(missing_days.to(dtype=flow_minus.dtype)).unsqueeze(-1)
    direct_minus = direct_state.flow_prev * decay * retained.unsqueeze(-1)
    multi_hop_minus = flow_minus - direct_minus
    direct = center_stocks(
        alpha * direct_minus + output.diagnostics["P_new"]
    )
    multi_hop = center_stocks(
        alpha * multi_hop_minus + output.diagnostics["P_fwd"]
    )
    reconstruction_max_abs = (direct + multi_hop - output.flow).abs().max()
    return FlowHopDecomposition(
        direct=direct,
        multi_hop=multi_hop,
        direct_minus=direct_minus,
        multi_hop_minus=multi_hop_minus,
        reconstruction_max_abs=reconstruction_max_abs,
    )


def propagate_flow_counterfactual(
    model: FlowFDG,
    output: FlowFDGOutput,
    state: FlowFDGState,
    operator: torch.Tensor,
    *,
    include_forwarding: bool,
    forwarding_operator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance an aligned frozen-checkpoint state under an explicit operator.

    All neural quantities come from the unmodified daily forward pass.  Only
    the stock transport operator and, optionally, the forwarding term are
    changed.  Suspension decay and cross-sectional centering exactly match the
    main recurrence.
    """

    stocks = output.diagnostics["Z"].shape[0]
    if state.flow_prev.shape != (stocks, model.flow_dim):
        raise ValueError("counterfactual state must be [stock, flow_dim]")
    if state.gap_days.shape != (stocks,) or state.gap_days.dtype != torch.long:
        raise ValueError("counterfactual gaps must be int64 [stock]")
    if operator.shape != (stocks, stocks):
        raise ValueError("counterfactual operator must be [stock, stock]")
    if forwarding_operator is not None and forwarding_operator.shape != (
        stocks,
        stocks,
    ):
        raise ValueError("forwarding operator must be [stock, stock]")
    if forwarding_operator is not None and not include_forwarding:
        raise ValueError("forwarding_operator requires include_forwarding=True")
    reference = output.diagnostics["Z"]
    if state.flow_prev.device != reference.device or state.flow_prev.dtype != reference.dtype:
        raise ValueError("counterfactual state must share model dtype and device")
    if state.gap_days.device != reference.device or operator.device != reference.device:
        raise ValueError("counterfactual state and operator must share the model device")
    if operator.dtype != reference.dtype:
        raise ValueError("counterfactual operator must share the model dtype")
    if forwarding_operator is not None:
        if forwarding_operator.device != reference.device:
            raise ValueError("forwarding operator must share the model device")
        if forwarding_operator.dtype != reference.dtype:
            raise ValueError("forwarding operator must share the model dtype")
    if torch.any(state.gap_days < 1):
        raise ValueError("counterfactual gaps must be at least one day")
    if not torch.isfinite(state.flow_prev).all() or not torch.isfinite(operator).all():
        raise FloatingPointError("counterfactual state and operator must be finite")
    if not torch.allclose(
        torch.diagonal(operator),
        operator.new_zeros(stocks),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("counterfactual operator must have zero diagonal")
    resolved_forwarding_operator = (
        operator if forwarding_operator is None else forwarding_operator
    )
    if not torch.isfinite(resolved_forwarding_operator).all():
        raise FloatingPointError("forwarding operator must be finite")
    if not torch.allclose(
        torch.diagonal(resolved_forwarding_operator),
        operator.new_zeros(stocks),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("forwarding operator must have zero diagonal")

    alpha = output.diagnostics["alpha"]
    rho = output.diagnostics["rho"]
    missing_days = state.gap_days - 1
    retained = missing_days <= model.suspension_max_days
    decay = alpha.pow(missing_days.to(dtype=reference.dtype)).unsqueeze(-1)
    flow_minus = state.flow_prev * decay * retained.unsqueeze(-1)
    new = operator @ output.diagnostics["U"]
    forwarding = (
        rho * (resolved_forwarding_operator @ flow_minus)
        if include_forwarding
        else 0.0
    )
    flow = center_stocks(alpha * flow_minus + new + forwarding)
    if not torch.isfinite(flow).all():
        raise FloatingPointError("counterfactual Flow state must be finite")
    return flow


def bounded_flow_counterfactual_prediction(
    model: FlowFDG,
    z: torch.Tensor,
    flow: torch.Tensor,
    *,
    forwarding: torch.Tensor | None = None,
    include_interaction: bool = True,
) -> torch.Tensor:
    """Replay A9's bounded shared readout for a counterfactual Flow state.

    This diagnostic intentionally supports only the evidence-free bounded gate
    used by A9.  A learned reliability gate would require recomputing its state
    evidence and is rejected rather than silently reusing full-state evidence.
    """

    if model.readout_mode != "bounded_shared_gated":
        raise ValueError("counterfactual replay requires bounded_shared_gated")
    if model.flow_reliability_gate is not None:
        raise ValueError("counterfactual replay does not reuse reliability evidence")
    if z.shape != (flow.shape[0], model.d_model):
        raise ValueError("z must be [stock, d_model] and align with flow")
    if flow.shape[1:] != (model.flow_dim,):
        raise ValueError("flow must be [stock, flow_dim]")
    if forwarding is not None and forwarding.shape != flow.shape:
        raise ValueError("forwarding must align with flow")
    assert model.fusion_gate is not None
    assert model.interaction_projection is not None
    assert model.shared_fusion_norm is not None
    flow_projection = model.flow_projection(flow)
    if model.forwarding_projection is not None:
        if forwarding is None:
            raise ValueError(
                "forwarding-exposed readout requires its P_fwd counterfactual"
            )
        flow_projection = flow_projection + model.forwarding_readout_scale * torch.tanh(
            model.forwarding_projection(forwarding)
        )
    gate_logits = model.fusion_gate(torch.cat([z, flow_projection], dim=-1))
    flow_fraction = model.max_flow_fraction * torch.sigmoid(gate_logits)
    interaction = torch.tanh(model.interaction_projection(z * flow_projection))
    if not include_interaction:
        interaction = torch.zeros_like(interaction)
    flow_candidate = flow_projection + interaction
    fused = model.shared_fusion_norm(
        (1.0 - flow_fraction) * z + flow_fraction * flow_candidate
    )
    return center_stocks(model.local_head(fused).squeeze(-1))


def validate_flow_fdg_config(config: Mapping[str, Any]) -> None:
    required = {
        "tier",
        "job_id",
        "method",
        "lock_id",
        "seed",
        "data",
        "dataset",
        "model_config",
        "train",
        "loss",
        "test_report",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Flow-FDG config misses required keys: {missing}")
    if config["method"] != "flow_fdg_v2":
        raise ValueError("method must be flow_fdg_v2")
    split_contract = str(config.get("split_contract", "paper_fixed"))
    if split_contract == "paper_fixed":
        expected_lock_id = "paper-fixed-split-v1-20260716"
        expected_splits = FIXED_SPLITS
    elif split_contract == "shift2y_shorttest_v1":
        expected_lock_id = SHIFT2Y_SHORTTEST_LOCK_ID
        expected_splits = SHIFT2Y_SHORTTEST_SPLITS
    else:
        raise ValueError(f"unsupported Flow-FDG split contract: {split_contract}")
    if config["lock_id"] != expected_lock_id:
        raise ValueError(
            f"Flow-FDG split contract {split_contract} requires lock {expected_lock_id}"
        )
    if tuple(config.get("evaluation_splits", ())) != ("valid", "test"):
        raise ValueError("Flow-FDG must select on Valid and directly report Test")
    data = config["data"]
    if (
        {name: list(value) for name, value in data.get("splits", {}).items()}
        != expected_splits
    ):
        raise ValueError(f"data.splits must equal split contract {split_contract}")
    handler = str(data.get("handler"))
    market = str(data.get("market"))
    fair_alpha_markets = (
        config.get("protocol_id") == "fair-alpha-markets-shift2y-v1"
        and config.get("tier") == "rolling_development"
        and split_contract == "shift2y_shorttest_v1"
    )
    us_sp500_fixed_extension = (
        config.get("protocol_id") == "us-sp500-pit-alpha158-ohlcv-v1"
        and config.get("tier") == "us_headline"
        and split_contract == "paper_fixed"
    )
    us_sp500_csi_matched_extension = (
        config.get("protocol_id")
        == "us-sp500-pit-alpha158-ohlcv-csi-matched-v1"
        and config.get("tier") == "us_csi_matched"
        and split_contract == "shift2y_shorttest_v1"
    )
    us_sp500_extension = (
        us_sp500_fixed_extension or us_sp500_csi_matched_extension
    )
    if fair_alpha_markets:
        if handler not in {"Alpha158", "Alpha360"} or market not in {
            "csi300",
            "csi500",
            "csi800",
        }:
            raise ValueError(
                "the fair Alpha-market extension requires Alpha158 or Alpha360 "
                "on CSI300, CSI500, or CSI800"
            )
        if (
            data.get("handler_class") != handler
            or data.get("handler_module_path") != "qlib.contrib.data.handler"
        ):
            raise ValueError(
                "the fair Alpha-market extension must use the declared official "
                "Qlib Alpha handler"
            )
    elif us_sp500_extension:
        expected_us_fields = {
            "handler": "Alpha158USOHLCV",
            "handler_class": "Alpha158USOHLCV",
            "handler_module_path": "meta_fdg.data.us_ohlcv_handler",
            "market": "sp500_pit",
            "region": "us",
        }
        observed_us_fields = {
            name: data.get(name) for name in expected_us_fields
        }
        if observed_us_fields != expected_us_fields:
            raise ValueError(
                "the US extension must use the audited PIT S&P 500 "
                "Alpha158-minus-VWAP handler"
            )
        if data.get("preprocessing_version") != (
            "us-sp500-pit-alpha158-ohlcv-v1__stateful_flow__v1"
        ):
            raise ValueError(
                "the US extension must use the locked stateful-flow "
                "preprocessing profile"
            )
    elif handler not in {"Alpha158", "MasterAlpha158"} or market != "csi300":
        raise ValueError(
            "the anchor must be Alpha158 or its MASTER-market extension on CSI300"
        )
    if handler == "MasterAlpha158" and (
        data.get("handler_class") != "MasterAlpha158"
        or data.get("handler_module_path")
        != "meta_fdg.baselines.master_qlib"
    ):
        raise ValueError(
            "MasterAlpha158 must use the audited in-repository handler adapter"
        )
    if tuple(data.get("load_splits", ())) != ("train", "valid"):
        raise ValueError("training materialization must be exactly train+valid")
    if str(data.get("materialization_end_time")) != expected_splits["valid"][1]:
        raise ValueError("development materialization must stop at Valid end")
    train = config["train"]
    if train.get("full_epoch_required") is not True:
        raise ValueError("full chronological epochs are required")
    if train.get("sampling") is not False or train.get("shuffle_dates") is not False:
        raise ValueError("Flow-FDG may neither sample nor shuffle dates")
    if int(train.get("tbptt_days", 0)) < 1:
        raise ValueError("train.tbptt_days must be positive")
    if int(train.get("pattern_router_uniform_epochs", 0)) < 0:
        raise ValueError("train.pattern_router_uniform_epochs must be non-negative")
    if int(train.get("coupled_temporal_only_epochs", 0)) < 0:
        raise ValueError(
            "train.coupled_temporal_only_epochs must be non-negative"
        )
    if train.get("stages") is not None and int(
        train.get("coupled_temporal_only_epochs", 0)
    ) > 0:
        raise ValueError("coupled temporal curriculum requires joint-only training")
    stages = train.get("stages")
    if stages is None:
        if int(train.get("epochs", 0)) < 1:
            raise ValueError("train.epochs must be positive")
    else:
        names = tuple(str(stage.get("name")) for stage in stages)
        if names != ("local", "flow_warm", "joint"):
            raise ValueError(
                "staged Flow-FDG must use local -> flow_warm -> joint"
            )
        if any(int(stage.get("epochs", 0)) < 1 for stage in stages):
            raise ValueError("each Flow-FDG stage needs positive epochs")
    if str(train.get("selection_metric")) != "valid_IC":
        raise ValueError("checkpoint selection is locked to Valid IC")
    report = config["test_report"]
    if report.get("enabled") is not True:
        raise ValueError("every formal Flow-FDG candidate requires direct Test reporting")
    if report.get("checkpoint") != "best_valid":
        raise ValueError("Test must use the best-Valid checkpoint")
    if report.get("gradient_updates") is not False:
        raise ValueError("Test labels must never drive gradients")
    if report.get("slow_state_updates") is not False:
        raise ValueError("primary Test must freeze the slow state")
    if config.get("se_mode", "disabled") not in {
        "disabled",
        "outer_learned_slow_state",
    }:
        raise ValueError("unsupported SE mode")
    nonnegative = (
        "main_ic_weight",
        "main_point_weight",
        "local_aux_weight",
        "local_ic_weight",
        "local_point_weight",
        "flow_aux_weight",
        "flow_aux_ic_weight",
        "flow_aux_point_weight",
        "shared_effect_aux_weight",
        "role_balance_weight",
        "activity_balance_weight",
        "state_l2_weight",
        "operator_l2_weight",
        "operator_direction_smooth_weight",
        "fast_state_l2_weight",
        "forwarding_floor_weight",
        "pattern_balance_weight",
        "pattern_diversity_weight",
        "pattern_lag_order_weight",
        "energy_gate_weight",
    )
    for name in nonnegative:
        if float(config["loss"].get(name, 0.0)) < 0.0:
            raise ValueError(f"loss.{name} must be non-negative")
    if config["loss"].get("rankic_in_selection_gate") is not False:
        raise ValueError("RankIC is reporting-only and cannot veto selection")


def model_kwargs_from_config(
    model_config: Mapping[str, Any], *, input_dim: int
) -> dict[str, Any]:
    encoder = model_config["encoder"]
    roles = model_config["latent_market_roles"]
    flow = model_config["flow"]
    dynamics = model_config["dynamics"]
    output = model_config["output"]
    market_conditioning = model_config.get("market_conditioning", {})
    hierarchical_roles = model_config.get("hierarchical_roles", {})
    innovation_source = model_config.get("innovation_source", {})
    temporal_arch = str(encoder.get("arch", "gru")).lower()
    if temporal_arch not in {"gru", "lstm", "transformer", "mlp"}:
        raise ValueError(
            "Flow-FDG temporal encoder must be an in-model GRU, LSTM, "
            "causal Transformer, or fixed-window MLP"
        )
    if roles.get("assignment") != "entmax15":
        raise ValueError("latent market roles must use Entmax1.5")
    if output.get("residual_readout") is not False:
        raise ValueError("Flow-FDG final readout must not contain a residual score")
    return {
        "input_dim": int(input_dim),
        "temporal_arch": temporal_arch,
        "d_model": int(encoder["output_dim"]),
        "gru_hidden_dim": int(encoder["hidden_dim"]),
        "gru_num_layers": int(encoder["num_layers"]),
        "transformer_max_len": int(encoder.get("max_len", encoder["lookback"])),
        "transformer_nhead": int(encoder.get("nhead", 4)),
        "transformer_ff_dim": int(encoder.get("ff_dim", 2 * encoder["output_dim"])),
        "temporal_mlp_token_dim": int(encoder.get("token_dim", 32)),
        "temporal_mlp_hidden_dim": int(encoder.get("mlp_hidden_dim", 96)),
        "temporal_input_layout": str(
            encoder.get("input_layout", "outer_time")
        ),
        "temporal_outer_lookback": int(encoder["lookback"]),
        "alpha360_field_count": int(encoder.get("field_count", 6)),
        "temporal_pooling_mode": str(encoder.get("pooling", "last")),
        "temporal_pooling_short_window": int(
            encoder.get("pooling_short_window", 5)
        ),
        "temporal_pooling_horizons": tuple(
            int(value)
            for value in encoder.get(
                "attention_horizons",
                [5, 20, 60],
            )
        ),
        "temporal_attention_dim": int(
            encoder.get("attention_dim", 64)
        ),
        "temporal_router_hidden_dim": int(
            encoder.get("router_hidden_dim", 64)
        ),
        "temporal_router_temperature": float(
            encoder.get("router_temperature", 1.0)
        ),
        "temporal_attention_mix_init": float(
            encoder.get("attention_mix_init", -1.0)
        ),
        "temporal_attention_identity_residual": bool(
            encoder.get("attention_identity_residual", False)
        ),
        "temporal_pattern_count": int(encoder.get("pattern_count", 4)),
        "pattern_flow_dim": int(flow.get("per_pattern_dimension", 8)),
        "pattern_router_scope": str(encoder.get("router_scope", "both")),
        "pattern_information_route_strength": float(
            encoder.get("information_route_strength", 1.0)
        ),
        "pattern_recurrence_mode": str(flow.get("recurrence_mode", "full")),
        "pattern_state_fusion": str(
            encoder.get("pattern_state_fusion", "attention_only")
        ),
        "pattern_lag_bias_init": str(
            encoder.get("lag_bias_init", "zero")
        ),
        "pattern_local_state_mode": str(
            encoder.get("local_state_mode", "pattern_routed")
        ),
        "shared_pattern_reservoir": bool(
            flow.get("shared_reservoir", False)
        ),
        "market_feature_dim": (
            int(market_conditioning.get("feature_dim", 0))
            if bool(market_conditioning.get("enabled", False))
            else 0
        ),
        "market_context_stop_gradient": bool(
            market_conditioning.get("stop_gradient_from_context", False)
        ),
        "hierarchical_market_roles": bool(
            hierarchical_roles.get("enabled", False)
        ),
        "coarse_role_count": int(hierarchical_roles.get("coarse_count", 4)),
        "hierarchy_mix_max": float(hierarchical_roles.get("mix_max", 0.10)),
        "hierarchy_mix_init": float(hierarchical_roles.get("mix_init", 0.02)),
        "innovation_source_residual": bool(
            innovation_source.get("enabled", False)
        ),
        "source_residual_max": float(
            innovation_source.get("max_residual_fraction", 0.20)
        ),
        "local_hidden_dim": int(
            output.get(
                "prediction_hidden_dim", output.get("local_hidden_dim", 64)
            )
        ),
        "role_count": int(roles["count"]),
        "role_projection_dim": int(roles.get("projection_dim", encoder["output_dim"])),
        "flow_dim": int(flow["dimension"]),
        "context_dim": int(dynamics["context_dim"]),
        "fast_rank": int(dynamics["fast_rank"]),
        "active_fast_rank": int(
            dynamics.get("active_fast_rank", dynamics["fast_rank"])
        ),
        "fast_coefficient_mode": str(
            dynamics.get("fast_coefficient_mode", "signed_tanh")
        ),
        "slow_rank": int(dynamics["slow_rank"]),
        "learned_slow_state": bool(dynamics.get("learned_slow_state", False)),
        "learned_slow_state_init": float(
            dynamics.get("learned_slow_state_init", 0.0)
        ),
        "fusion_hidden_dim": int(output["fusion_hidden_dim"]),
        "readout_mode": str(output.get("readout_mode", "independent_concat")),
        "architecture_mode": str(
            output.get("architecture_mode", "legacy_auxiliary_diagnostics")
        ),
        "coupled_flow_projection_mode": str(
            output.get("coupled_flow_projection_mode", "normalized")
        ),
        "coupled_source_anchored_init": bool(
            output.get("coupled_source_anchored_init", False)
        ),
        "couple_lagged_flow_to_roles": bool(
            output.get("couple_lagged_flow_to_roles", True)
        ),
        "coupled_fusion_mode": str(
            output.get("coupled_fusion_mode", "mlp")
        ),
        "max_flow_fraction": float(output.get("max_flow_fraction", 0.08)),
        "normalized_energy_gate": bool(
            output.get("normalized_energy_gate", False)
        ),
        "energy_gate_target": float(
            output.get("energy_gate_target", 0.04)
        ),
        "evidence_aware_flow_reliability": bool(
            output.get("evidence_aware_flow_reliability", False)
        ),
        "flow_reliability_floor": float(
            output.get("flow_reliability_floor", 0.25)
        ),
        "expose_forwarding_readout": bool(
            output.get("expose_forwarding_readout", False)
        ),
        "forwarding_readout_scale": float(
            output.get("forwarding_readout_scale", 0.25)
        ),
        "dropout": float(encoder.get("dropout", 0.1)),
        "sender_temperature": float(roles.get("sender_temperature", 1.0)),
        "receiver_temperature": float(roles.get("receiver_temperature", 1.0)),
        "gamma_max": float(dynamics["gamma_max"]),
        "dissipation_floor": float(dynamics["dissipation_floor"]),
        "retention_scale": float(flow.get("retention_scale", 1.0)),
        "forwarding_scale": float(flow.get("forwarding_scale", 1.0)),
        "suspension_max_days": int(flow["suspension_max_days"]),
        "eps": float(dynamics.get("epsilon", 1e-8)),
    }


def build_flow_fdg_model(
    input_dim: int,
    model_config: Mapping[str, Any],
    device: torch.device,
) -> FlowFDG:
    return FlowFDG(**model_kwargs_from_config(model_config, input_dim=input_dim)).to(device)


def negative_pearson_ic(
    prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("prediction and target must share one-dimensional shape")
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    left = prediction - prediction.mean()
    right = target - target.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum() + float(eps))
    return -(left * right).sum() / denominator


def _point_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: str,
    huber_delta: float,
) -> torch.Tensor:
    if kind == "mse":
        return (prediction - target).square().mean()
    if kind == "huber":
        return nn.functional.huber_loss(prediction, target, delta=float(huber_delta))
    raise ValueError(f"unsupported point loss {kind!r}")


def flow_fdg_loss(
    output: FlowFDGOutput,
    target: torch.Tensor,
    loss_config: Mapping[str, Any],
    *,
    stage: str = "joint",
    previous_role_operator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if stage not in {"local", "flow_warm", "joint"}:
        raise ValueError(f"unknown Flow-FDG stage {stage!r}")
    if target.shape != output.prediction.shape:
        raise ValueError("target must match Flow-FDG prediction")
    kind = str(loss_config.get("point_loss", "huber"))
    huber_delta = float(loss_config.get("huber_delta", 1.0))
    main_ic = negative_pearson_ic(output.prediction, target)
    main_point = _point_loss(
        output.prediction, target, kind=kind, huber_delta=huber_delta
    )
    auxiliary_available = (
        output.p_local is not None and output.flow_readout is not None
    )
    if (output.p_local is None) != (output.flow_readout is None):
        raise ValueError("local and flow auxiliary heads must be jointly available")
    if auxiliary_available:
        assert output.p_local is not None
        assert output.flow_readout is not None
        local_ic = negative_pearson_ic(output.p_local, target)
        local_point = _point_loss(
            output.p_local, target, kind=kind, huber_delta=huber_delta
        )
        flow_residual_target = target - output.p_local.detach()
        flow_aux_ic = negative_pearson_ic(
            output.flow_readout, flow_residual_target
        )
        flow_aux_point = _point_loss(
            output.flow_readout,
            flow_residual_target,
            kind=kind,
            huber_delta=huber_delta,
        )
        shared_effect = output.prediction - output.p_local
        shared_effect_ic = negative_pearson_ic(
            shared_effect, flow_residual_target
        )
    else:
        forbidden = {
            name: float(loss_config.get(name, 0.0))
            for name in (
                "local_aux_weight",
                "flow_aux_weight",
                "shared_effect_aux_weight",
            )
        }
        if any(value != 0.0 for value in forbidden.values()):
            raise ValueError(
                f"single-head loss cannot enable auxiliary prediction terms: {forbidden}"
            )
        local_ic = main_ic.new_zeros(())
        local_point = main_ic.new_zeros(())
        flow_aux_ic = main_ic.new_zeros(())
        flow_aux_point = main_ic.new_zeros(())
        shared_effect_ic = main_ic.new_zeros(())
    sender = output.diagnostics["S"]
    receiver = output.diagnostics["R"]
    uniform = 1.0 / float(sender.shape[-1])
    role_balance = (sender.mean(dim=0) - uniform).square().mean() + (
        receiver.mean(dim=0) - uniform
    ).square().mean()
    sender_activity = output.diagnostics["sender_activity"]
    receiver_activity = output.diagnostics["receiver_activity"]
    activity_target = float(loss_config.get("activity_target", 0.5))
    activity_balance = (sender_activity.mean() - activity_target).square() + (
        receiver_activity.mean() - activity_target
    ).square()
    state_l2 = output.flow.square().mean()
    operator_l2 = output.diagnostics["B"].square().mean()
    if previous_role_operator is None:
        operator_direction_smooth = operator_l2.new_zeros(())
    else:
        role_operator = output.diagnostics["B"]
        if previous_role_operator.shape != role_operator.shape:
            raise ValueError("previous_role_operator must match the current role operator")
        if (
            previous_role_operator.device != role_operator.device
            or previous_role_operator.dtype != role_operator.dtype
        ):
            raise ValueError(
                "previous_role_operator must share the current operator dtype and device"
            )
        if not torch.isfinite(previous_role_operator).all():
            raise FloatingPointError("previous_role_operator must be finite")
        current_direction = role_operator / (role_operator.norm() + 1e-8)
        previous_direction = previous_role_operator / (
            previous_role_operator.norm() + 1e-8
        )
        operator_direction_smooth = 0.5 * (
            current_direction - previous_direction
        ).square().sum()
    fast_state_l2 = output.diagnostics["g_fast"].square().mean()
    forwarding_floor = torch.relu(
        float(loss_config.get("rho_floor", 0.1)) - output.diagnostics["rho"]
    ).square()
    pattern_load_balance = output.diagnostics.get(
        "pattern_load_balance",
        main_ic.new_zeros(()),
    )
    pattern_attention_diversity = output.diagnostics.get(
        "pattern_attention_diversity",
        main_ic.new_zeros(()),
    )
    pattern_regularization = (
        float(loss_config.get("pattern_balance_weight", 0.0))
        * pattern_load_balance
        + float(loss_config.get("pattern_diversity_weight", 0.0))
        * pattern_attention_diversity
    )
    pattern_lag_order = output.diagnostics.get(
        "pattern_lag_order_violation",
        main_ic.new_zeros(()),
    )
    energy_gate_regularization = main_ic.new_zeros(())
    if "flow_energy_mean" in output.diagnostics and stage != "local":
        energy_target = float(loss_config.get("energy_gate_target", 0.04))
        energy_gate_regularization = (
            output.diagnostics["flow_energy_mean"] - energy_target
        ).square()
    joint_total = (
        float(loss_config.get("main_ic_weight", 1.0)) * main_ic
        + float(loss_config.get("main_point_weight", 0.0)) * main_point
        + float(loss_config.get("local_aux_weight", 0.0))
        * (
            float(loss_config.get("local_ic_weight", 1.0)) * local_ic
            + float(loss_config.get("local_point_weight", 0.0)) * local_point
        )
        + float(loss_config.get("flow_aux_weight", 0.0))
        * (
            float(loss_config.get("flow_aux_ic_weight", 1.0)) * flow_aux_ic
            + float(loss_config.get("flow_aux_point_weight", 0.0))
            * flow_aux_point
        )
        + float(loss_config.get("shared_effect_aux_weight", 0.0))
        * shared_effect_ic
        + float(loss_config.get("role_balance_weight", 0.0)) * role_balance
        + float(loss_config.get("activity_balance_weight", 0.0)) * activity_balance
        + float(loss_config.get("state_l2_weight", 0.0)) * state_l2
        + float(loss_config.get("operator_l2_weight", 0.0)) * operator_l2
        + float(loss_config.get("operator_direction_smooth_weight", 0.0))
        * operator_direction_smooth
        + float(loss_config.get("fast_state_l2_weight", 0.0)) * fast_state_l2
        + float(loss_config.get("forwarding_floor_weight", 0.0)) * forwarding_floor
        + pattern_regularization
        + float(loss_config.get("pattern_lag_order_weight", 0.0))
        * pattern_lag_order
        + float(loss_config.get("energy_gate_weight", 0.0))
        * energy_gate_regularization
    )
    if stage == "local":
        if auxiliary_available:
            total = (
                float(loss_config.get("local_stage_ic_weight", 1.0)) * local_ic
                + float(loss_config.get("local_stage_point_weight", 0.05))
                * local_point
                + pattern_regularization
            )
        else:
            total = (
                float(loss_config.get("main_ic_weight", 1.0)) * main_ic
                + float(loss_config.get("main_point_weight", 0.0)) * main_point
                + pattern_regularization
                + float(loss_config.get("pattern_lag_order_weight", 0.0))
                * pattern_lag_order
            )
    else:
        total = joint_total
    components = {
        "main_ic_loss": main_ic,
        "main_point_loss": main_point,
        "local_ic_loss": local_ic,
        "local_point_loss": local_point,
        "flow_aux_ic_loss": flow_aux_ic,
        "flow_aux_point_loss": flow_aux_point,
        "shared_effect_ic_loss": shared_effect_ic,
        "role_balance": role_balance,
        "activity_balance": activity_balance,
        "state_l2": state_l2,
        "operator_l2": operator_l2,
        "operator_direction_smooth": operator_direction_smooth,
        "fast_state_l2": fast_state_l2,
        "forwarding_floor": forwarding_floor,
        "pattern_load_balance": pattern_load_balance,
        "pattern_attention_diversity": pattern_attention_diversity,
        "pattern_lag_order": pattern_lag_order,
        "energy_gate_regularization": energy_gate_regularization,
        "total_loss": total,
    }
    if not torch.isfinite(total):
        raise FloatingPointError("Flow-FDG produced a non-finite training loss")
    return total, components


def set_stage_trainability(model: FlowFDG, stage: str) -> None:
    if stage not in {"local", "flow_warm", "joint"}:
        raise ValueError(f"unknown Flow-FDG stage {stage!r}")
    groups = model.parameter_groups()
    encoder_trainable = stage in {"local", "joint"}
    for parameter in groups["encoder"]:
        parameter.requires_grad_(encoder_trainable)
    if model.fully_coupled:
        flow_groups = model.flow_parameter_groups()
        for parameter in flow_groups["graph"]:
            parameter.requires_grad_(stage in {"flow_warm", "joint"})
        for parameter in flow_groups["readout"]:
            parameter.requires_grad_(True)
        for parameter in flow_groups["slow"]:
            parameter.requires_grad_(stage in {"flow_warm", "joint"})
    else:
        flow_trainable = stage in {"flow_warm", "joint"}
        for parameter in groups["flow"]:
            parameter.requires_grad_(flow_trainable)


def build_optimizer(
    model: FlowFDG,
    train_config: Mapping[str, Any],
    stage: str = "joint",
) -> torch.optim.AdamW:
    set_stage_trainability(model, stage)
    groups = model.parameter_groups()
    parameter_groups: list[dict[str, Any]] = []
    if stage in {"local", "joint"}:
        encoder_lr = (
            float(train_config["encoder_learning_rate"])
            if stage == "local"
            else float(
                train_config.get(
                    "joint_encoder_learning_rate",
                    train_config["encoder_learning_rate"],
                )
            )
        )
        parameter_groups.append(
            {"name": "encoder", "params": groups["encoder"], "lr": encoder_lr}
        )
        if stage == "local" and model.fully_coupled:
            readout = model.flow_parameter_groups()["readout"]
            parameter_groups.append(
                {"name": "readout", "params": readout, "lr": encoder_lr}
            )
    if stage in {"flow_warm", "joint"}:
        flow_learning_rate = (
            float(train_config["flow_learning_rate"])
            if stage == "flow_warm"
            else float(
                train_config.get(
                    "joint_flow_learning_rate",
                    train_config["flow_learning_rate"],
                )
            )
        )
        split_joint_flow = stage == "joint" and (
            "joint_graph_learning_rate" in train_config
            or "joint_readout_learning_rate" in train_config
        )
        if split_joint_flow:
            flow_groups = model.flow_parameter_groups()
            parameter_groups.extend(
                [
                    {
                        "name": "graph",
                        "params": flow_groups["graph"],
                        "lr": float(
                            train_config.get(
                                "joint_graph_learning_rate", flow_learning_rate
                            )
                        ),
                    },
                    {
                        "name": "readout",
                        "params": flow_groups["readout"],
                        "lr": float(
                            train_config.get(
                                "joint_readout_learning_rate", flow_learning_rate
                            )
                        ),
                    },
                ]
            )
            if flow_groups["slow"]:
                graph_learning_rate = float(
                    train_config.get(
                        "joint_graph_learning_rate", flow_learning_rate
                    )
                )
                parameter_groups.append(
                    {
                        "name": "slow",
                        "params": flow_groups["slow"],
                        "lr": float(
                            train_config.get(
                                "joint_slow_state_learning_rate",
                                0.1 * graph_learning_rate,
                            )
                        ),
                    }
                )
        else:
            parameter_groups.append(
                {
                    "name": "flow",
                    "params": groups["flow"],
                    "lr": flow_learning_rate,
                }
            )
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )


def _assert_finite_parameters_and_state(
    model: FlowFDG, optimizer: torch.optim.Optimizer
) -> None:
    bad_parameters = [
        name
        for name, parameter in model.named_parameters()
        if not torch.isfinite(parameter).all()
    ]
    if bad_parameters:
        raise FloatingPointError(f"non-finite model parameters: {bad_parameters}")
    bad_state: list[str] = []
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                bad_state.append(f"{tuple(parameter.shape)}:{name}")
    if bad_state:
        raise FloatingPointError(f"non-finite optimizer state: {bad_state}")


def train_flow_epoch(
    model: FlowFDG,
    split: TemporalSplit,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_config: Mapping[str, Any],
    tbptt_days: int,
    grad_clip: float,
    stage: str = "joint",
) -> FlowEpochResult:
    if stage not in {"local", "flow_warm", "joint"}:
        raise ValueError(f"unknown Flow-FDG stage {stage!r}")
    model.train()
    if stage == "flow_warm":
        model.temporal_encoder.eval()
        model.temporal_projection.eval()
        if model.temporal_residual_pooling is not None:
            model.temporal_residual_pooling.eval()
        if model.temporal_state_pooling is not None:
            model.temporal_state_pooling.eval()
        if model.temporal_pattern_pooling is not None:
            model.temporal_pattern_pooling.eval()
        if model.local_head is not None:
            model.local_head.eval()
    bank = FlowStateBank(
        flow_dim=model.flow_dim,
        suspension_max_days=model.suspension_max_days,
        pattern_count=model.pattern_count,
        pattern_state_dim=model.pattern_state_dim,
    )
    component_totals: dict[str, float] = {}
    emitted_days = emitted_rows = optimizer_steps = 0
    carried = entered = reentered = exited = reset = 0
    gradient_norm_max = 0.0
    alpha_total = rho_total = delta_total = gamma_total = 0.0
    forwarding_ratio_total = 0.0
    previous_role_operator: torch.Tensor | None = None

    days = split.days
    for chunk_start in range(0, len(days), int(tbptt_days)):
        chunk = days[chunk_start : chunk_start + int(tbptt_days)]
        optimizer.zero_grad(set_to_none=True)
        chunk_loss: torch.Tensor | None = None
        for offset, day in enumerate(chunk):
            day_index = chunk_start + offset
            features = day.sequences.to(device, non_blocking=True)
            target = day.target.to(device, non_blocking=True)
            state, audit = bank.prepare(
                day.instruments,
                day_index=day_index,
                reference=features,
            )
            output = model(features, state=state, compute_graph_diagnostics=False)
            loss, components = flow_fdg_loss(
                output,
                target,
                loss_config,
                stage=stage,
                previous_role_operator=previous_role_operator,
            )
            previous_role_operator = output.diagnostics["B"]
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(
                    value.detach().cpu()
                )
            bank.update(
                day.instruments,
                output.flow,
                day_index=day_index,
                pattern_state=output.state.pattern_prev,
            )
            emitted_days += 1
            emitted_rows += day.row_count
            carried += audit.carried
            entered += audit.entered
            reentered += audit.reentered
            exited += audit.exited
            reset += audit.reset_after_long_gap
            alpha_total += float(output.diagnostics["alpha"].detach().cpu())
            rho_total += float(output.diagnostics["rho"].detach().cpu())
            delta_total += float(output.diagnostics["delta"].detach().cpu())
            gamma_total += float(output.diagnostics["gamma"].detach().cpu())
            denominator = output.diagnostics["P_new"].square().mean().sqrt() + 1e-8
            forwarding_ratio_total += float(
                (
                    output.diagnostics["P_fwd"].square().mean().sqrt()
                    / denominator
                )
                .detach()
                .cpu()
            )
        if chunk_loss is None:
            raise AssertionError("empty TBPTT chunk")
        (chunk_loss / float(len(chunk))).backward()
        nonfinite_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if nonfinite_gradients:
            raise FloatingPointError(
                f"non-finite Flow-FDG gradients: {nonfinite_gradients}"
            )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(grad_clip), error_if_nonfinite=True
        )
        gradient_norm_max = max(
            gradient_norm_max, float(gradient_norm.detach().cpu())
        )
        optimizer.step()
        _assert_finite_parameters_and_state(model, optimizer)
        optimizer_steps += 1
        bank.detach()
        if previous_role_operator is not None:
            previous_role_operator = previous_role_operator.detach()

    if emitted_days != split.coverage.emitted_days:
        raise AssertionError("Flow-FDG train epoch omitted chronological days")
    if emitted_rows != split.coverage.emitted_rows:
        raise AssertionError("Flow-FDG train epoch omitted rows")
    divisor = float(emitted_days)
    return FlowEpochResult(
        metrics={
            "train_days_seen": emitted_days,
            "train_days_expected": split.coverage.emitted_days,
            "train_rows_seen": emitted_rows,
            "train_rows_expected": split.coverage.emitted_rows,
            "coverage_complete": True,
            "tbptt_days": int(tbptt_days),
            "optimizer_steps": optimizer_steps,
            "state_detach_count": optimizer_steps,
            "alignment_carried": carried,
            "alignment_entered": entered,
            "alignment_reentered": reentered,
            "alignment_exited": exited,
            "alignment_reset_after_long_gap": reset,
            "gradient_norm_max": gradient_norm_max,
            "alpha_mean": alpha_total / divisor,
            "rho_mean": rho_total / divisor,
            "delta_mean": delta_total / divisor,
            "gamma_mean": gamma_total / divisor,
            "forwarding_to_new_rms_ratio_mean": forwarding_ratio_total / divisor,
            "stage": stage,
        },
        component_means={
            name: value / divisor for name, value in component_totals.items()
        },
    )


def _scalar(value: torch.Tensor, reduction: str = "mean") -> float:
    detached = value.detach().float()
    if reduction == "mean":
        result = detached.mean()
    elif reduction == "std":
        result = detached.std(unbiased=False)
    elif reduction == "max":
        result = detached.max()
    else:
        raise ValueError(f"unknown reduction {reduction}")
    return float(result.cpu())


@torch.no_grad()
def evaluate_flow_fdg(
    model: FlowFDG,
    evaluation_split: TemporalSplit,
    *,
    warmup_days: Sequence[TemporalDay],
    device: torch.device,
    loss_config: Mapping[str, Any],
    split_name: str,
    transport_counterfactuals: bool = False,
) -> FlowEvaluationResult:
    if split_name not in {"valid", "test"} or evaluation_split.split != split_name:
        raise ValueError("evaluation split identity mismatch")
    if transport_counterfactuals and model.pattern_routed:
        raise ValueError(
            "single-channel hop counterfactuals are not defined for pattern Flow"
        )
    model.eval()
    bank = FlowStateBank(
        flow_dim=model.flow_dim,
        suspension_max_days=model.suspension_max_days,
        pattern_count=model.pattern_count,
        pattern_state_dim=model.pattern_state_dim,
    )
    direct_bank = (
        FlowStateBank(
            flow_dim=model.flow_dim,
            suspension_max_days=model.suspension_max_days,
            pattern_count=model.pattern_count,
            pattern_state_dim=model.pattern_state_dim,
        )
        if transport_counterfactuals
        else None
    )
    day_index = 0
    for day in warmup_days:
        features = day.sequences.to(device)
        state, _ = bank.prepare(
            day.instruments, day_index=day_index, reference=features
        )
        direct_state = None
        if direct_bank is not None:
            direct_state, _ = direct_bank.prepare(
                day.instruments, day_index=day_index, reference=features
            )
        output = model(features, state=state, compute_graph_diagnostics=False)
        if direct_bank is not None:
            assert direct_state is not None
            hop = decompose_flow_hops(model, output, direct_state)
            if float(hop.reconstruction_max_abs.cpu()) > 1e-5:
                raise AssertionError("hop decomposition does not reconstruct Flow")
            direct_bank.update(
                day.instruments,
                hop.direct,
                day_index=day_index,
                pattern_state=output.state.pattern_prev,
            )
        bank.update(
            day.instruments,
            output.flow,
            day_index=day_index,
            pattern_state=output.state.pattern_prev,
        )
        day_index += 1

    full_daily: list[dict[str, Any]] = []
    local_daily: list[dict[str, Any]] = []
    flow_daily: list[dict[str, Any]] = []
    full_predictions: list[dict[str, Any]] = []
    local_predictions: list[dict[str, Any]] = []
    flow_predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    counterfactual_daily_rows: dict[str, list[dict[str, Any]]] = {
        "direct_only": [],
        "multi_hop_only": [],
        "full_no_interaction": [],
    }
    counterfactual_prediction_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in counterfactual_daily_rows
    }
    full_losses: list[float] = []
    local_losses: list[float] = []
    auxiliary_available = not model.fully_coupled
    rows_seen = 0

    for day in evaluation_split.days:
        features = day.sequences.to(device)
        target = day.target.to(device)
        state, audit = bank.prepare(
            day.instruments, day_index=day_index, reference=features
        )
        direct_state = None
        if direct_bank is not None:
            direct_state, direct_audit = direct_bank.prepare(
                day.instruments, day_index=day_index, reference=features
            )
            if direct_audit != audit:
                raise AssertionError("direct and full state alignment diverged")
        output = model(features, state=state, compute_graph_diagnostics=False)
        hop = None
        counterfactual_scores: dict[str, np.ndarray] = {}
        if direct_bank is not None:
            assert direct_state is not None
            hop = decompose_flow_hops(model, output, direct_state)
            if float(hop.reconstruction_max_abs.cpu()) > 1e-5:
                raise AssertionError("hop decomposition does not reconstruct Flow")
            replay = bounded_flow_counterfactual_prediction(
                model,
                output.diagnostics["Z"],
                output.flow,
                forwarding=output.diagnostics["P_fwd"],
            )
            if not torch.allclose(replay, output.prediction, atol=1e-6, rtol=1e-6):
                raise AssertionError("bounded counterfactual readout does not replay full")
            counterfactual_scores = {
                "direct_only": bounded_flow_counterfactual_prediction(
                    model,
                    output.diagnostics["Z"],
                    hop.direct,
                    forwarding=torch.zeros_like(output.diagnostics["P_fwd"]),
                ).cpu().numpy(),
                "multi_hop_only": bounded_flow_counterfactual_prediction(
                    model,
                    output.diagnostics["Z"],
                    hop.multi_hop,
                    forwarding=output.diagnostics["P_fwd"],
                ).cpu().numpy(),
                "full_no_interaction": bounded_flow_counterfactual_prediction(
                    model,
                    output.diagnostics["Z"],
                    output.flow,
                    forwarding=output.diagnostics["P_fwd"],
                    include_interaction=False,
                ).cpu().numpy(),
            }
        _, components = flow_fdg_loss(output, target, loss_config)
        full_losses.append(float(components["main_point_loss"].cpu()))
        full_score = output.prediction.cpu().numpy()
        label = day.raw_target.numpy()
        full_metrics = daily_signal_metrics(full_score, label)
        if int(full_metrics["n"]) != day.row_count:
            raise AssertionError("Flow-FDG evaluation dropped rows")
        full_daily.append(
            {"split": split_name, "variant": "full", "date": day.date, **full_metrics}
        )
        local_score: np.ndarray | None = None
        flow_score: np.ndarray | None = None
        if auxiliary_available:
            if output.p_local is None or output.flow_readout is None:
                raise TypeError("legacy diagnostics disappeared during evaluation")
            local_losses.append(float(components["local_point_loss"].cpu()))
            local_score = output.p_local.cpu().numpy()
            flow_score = output.flow_readout.cpu().numpy()
            local_metrics = daily_signal_metrics(local_score, label)
            flow_metrics = daily_signal_metrics(flow_score, label)
            local_daily.append(
                {"split": split_name, "variant": "local_diagnostic", "date": day.date, **local_metrics}
            )
            flow_daily.append(
                {"split": split_name, "variant": "flow_diagnostic", "date": day.date, **flow_metrics}
            )
        for name, score in counterfactual_scores.items():
            metrics = daily_signal_metrics(score, label)
            if int(metrics["n"]) != day.row_count:
                raise AssertionError(f"{name} counterfactual dropped rows")
            counterfactual_daily_rows[name].append(
                {"split": split_name, "variant": name, "date": day.date, **metrics}
            )
        for position, instrument in enumerate(day.instruments):
            base = {
                "split": split_name,
                "date": day.date,
                "instrument": instrument,
                "label": float(label[position]),
            }
            full_predictions.append({**base, "score": float(full_score[position])})
            if local_score is not None and flow_score is not None:
                local_predictions.append({**base, "score": float(local_score[position])})
                flow_predictions.append({**base, "score": float(flow_score[position])})
            for name, score in counterfactual_scores.items():
                counterfactual_prediction_rows[name].append(
                    {**base, "variant": name, "score": float(score[position])}
                )

        sender = output.diagnostics["S"]
        receiver = output.diagnostics["R"]
        sender_entropy = -(sender * sender.clamp_min(1e-12).log()).sum(dim=-1)
        receiver_entropy = -(receiver * receiver.clamp_min(1e-12).log()).sum(dim=-1)
        p_new_rms = output.diagnostics["P_new"].square().mean().sqrt()
        p_fwd_rms = output.diagnostics["P_fwd"].square().mean().sqrt()
        pattern_daily: dict[str, float] = {}
        if model.pattern_routed:
            route = output.diagnostics["pattern_route"]
            expected_lag = output.diagnostics["pattern_expected_lag"]
            pattern_daily.update(
                {
                    "pattern_route_zero_fraction": _scalar(
                        (route == 0).float()
                    ),
                    "pattern_load_balance": _scalar(
                        output.diagnostics["pattern_load_balance"]
                    ),
                    "pattern_attention_diversity": _scalar(
                        output.diagnostics["pattern_attention_diversity"]
                    ),
                    "pattern_lifetime_simplex_error": _scalar(
                        output.diagnostics["pattern_lifetime_simplex_error"]
                    ),
                }
            )
            for mode in range(model.pattern_count):
                pattern_daily.update(
                    {
                        f"pattern_{mode}_route_mean": _scalar(route[:, mode]),
                        f"pattern_{mode}_expected_lag": _scalar(
                            expected_lag[:, mode]
                        ),
                        f"pattern_{mode}_alpha": _scalar(
                            output.diagnostics["pattern_alpha"][mode]
                        ),
                        f"pattern_{mode}_rho": _scalar(
                            output.diagnostics["pattern_rho"][mode]
                        ),
                        f"pattern_{mode}_delta": _scalar(
                            output.diagnostics["pattern_delta"][mode]
                        ),
                        f"pattern_{mode}_change_rms": _scalar(
                            output.diagnostics["pattern_change_rms"][mode]
                        ),
                    }
                )
        diagnostics.append(
            {
                "split": split_name,
                "date": day.date,
                "n": day.row_count,
                "alpha": _scalar(output.diagnostics["alpha"]),
                "rho": _scalar(output.diagnostics["rho"]),
                "delta": _scalar(output.diagnostics["delta"]),
                "gamma": _scalar(output.diagnostics["gamma"]),
                "contraction_upper_bound": _scalar(
                    output.diagnostics["contraction_upper_bound"]
                ),
                "prediction_std": _scalar(output.prediction, "std"),
                **(
                    {
                        "local_diagnostic_std": _scalar(output.p_local, "std"),
                        "flow_diagnostic_std": _scalar(output.flow_readout, "std"),
                    }
                    if output.p_local is not None
                    and output.flow_readout is not None
                    else {}
                ),
                "state_rms": _scalar(output.flow.square().mean().sqrt()),
                "new_rms": _scalar(p_new_rms),
                "forwarding_rms": _scalar(p_fwd_rms),
                "forwarding_to_new_rms_ratio": _scalar(p_fwd_rms / (p_new_rms + 1e-8)),
                "keep_rms": _scalar(
                    output.diagnostics["P_keep"].square().mean().sqrt()
                ),
                "operator_frobenius": _scalar(
                    output.diagnostics["operator_frobenius"]
                ),
                "role_operator_frobenius": float(
                    output.diagnostics["B"].float().norm().cpu()
                ),
                "fast_state_rms": _scalar(
                    output.diagnostics["g_fast"].square().mean().sqrt()
                ),
                "sender_activity_mean": _scalar(
                    output.diagnostics["sender_activity"]
                ),
                "receiver_activity_mean": _scalar(
                    output.diagnostics["receiver_activity"]
                ),
                "sender_role_entropy": _scalar(sender_entropy),
                "receiver_role_entropy": _scalar(receiver_entropy),
                "sender_role_zero_fraction": _scalar((sender == 0).float()),
                "receiver_role_zero_fraction": _scalar((receiver == 0).float()),
                **(
                    {
                        "flow_processed_residual_ic": _scalar(
                            -negative_pearson_ic(
                                output.flow_readout, target - output.p_local
                            )
                        )
                    }
                    if output.p_local is not None
                    and output.flow_readout is not None
                    else {}
                ),
                **(
                    {"fusion_gate_mean": _scalar(output.diagnostics["fusion_gate"])}
                    if "fusion_gate" in output.diagnostics
                    else {}
                ),
                **(
                    {"flow_fraction_mean": _scalar(output.diagnostics["flow_fraction"])}
                    if "flow_fraction" in output.diagnostics
                    else {}
                ),
                **(
                    {
                        "flow_reliability_mean": _scalar(
                            output.diagnostics["flow_reliability"]
                        ),
                        "flow_state_level_evidence_mean": _scalar(
                            output.diagnostics["flow_reliability_evidence"][:, 0]
                        ),
                        "flow_state_share_evidence_mean": _scalar(
                            output.diagnostics["flow_reliability_evidence"][:, 1]
                        ),
                        "flow_forwarding_share_evidence_mean": _scalar(
                            output.diagnostics["flow_reliability_evidence"][:, 2]
                        ),
                    }
                    if "flow_reliability" in output.diagnostics
                    else {}
                ),
                "alignment_carried": audit.carried,
                "alignment_entered": audit.entered,
                "alignment_reentered": audit.reentered,
                "alignment_exited": audit.exited,
                "alignment_reset_after_long_gap": audit.reset_after_long_gap,
                **pattern_daily,
                **(
                    {
                        "hop_direct_rms": _scalar(
                            hop.direct.square().mean().sqrt()
                        ),
                        "hop_multi_hop_rms": _scalar(
                            hop.multi_hop.square().mean().sqrt()
                        ),
                        "hop_multi_to_direct_rms_ratio": _scalar(
                            hop.multi_hop.square().mean().sqrt()
                            / (hop.direct.square().mean().sqrt() + 1e-8)
                        ),
                        "hop_reconstruction_max_abs": _scalar(
                            hop.reconstruction_max_abs
                        ),
                    }
                    if hop is not None
                    else {}
                ),
            }
        )
        if direct_bank is not None:
            assert hop is not None
            direct_bank.update(
                day.instruments,
                hop.direct,
                day_index=day_index,
                pattern_state=output.state.pattern_prev,
            )
        bank.update(
            day.instruments,
            output.flow,
            day_index=day_index,
            pattern_state=output.state.pattern_prev,
        )
        day_index += 1
        rows_seen += day.row_count

    if len(full_daily) != evaluation_split.coverage.emitted_days:
        raise AssertionError("Flow-FDG evaluation omitted days")
    if rows_seen != evaluation_split.coverage.emitted_rows:
        raise AssertionError("Flow-FDG evaluation omitted rows")
    full_frame = pd.DataFrame(full_daily)
    local_frame = pd.DataFrame(local_daily)
    flow_frame = pd.DataFrame(flow_daily)
    summary_full = summarize_daily_metrics(full_frame)
    common = {
        "rows": rows_seen,
        "days_expected": evaluation_split.coverage.emitted_days,
        "rows_expected": evaluation_split.coverage.emitted_rows,
        "coverage_complete": True,
        "warmup_days": len(warmup_days),
    }
    summary_full.update({**common, "point_loss": float(np.mean(full_losses))})
    if auxiliary_available:
        summary_local = summarize_daily_metrics(local_frame)
        summary_flow = summarize_daily_metrics(flow_frame)
        summary_local.update(
            {**common, "point_loss": float(np.mean(local_losses))}
        )
        summary_flow.update(common)
    else:
        unavailable = {
            **common,
            "available": False,
            "reason": "fully_coupled_single_head_has_no_auxiliary_predictions",
        }
        summary_local = dict(unavailable)
        summary_flow = dict(unavailable)
    counterfactual_daily = {
        name: pd.DataFrame(rows)
        for name, rows in counterfactual_daily_rows.items()
        if rows
    }
    counterfactual_predictions = {
        name: pd.DataFrame(rows)
        for name, rows in counterfactual_prediction_rows.items()
        if rows
    }
    counterfactual_summaries: dict[str, dict[str, Any]] = {}
    for name, frame in counterfactual_daily.items():
        summary = summarize_daily_metrics(frame)
        summary.update(common)
        counterfactual_summaries[name] = summary
    return FlowEvaluationResult(
        daily_full=full_frame,
        daily_local=local_frame,
        daily_flow=flow_frame,
        predictions_full=pd.DataFrame(full_predictions),
        predictions_local=pd.DataFrame(local_predictions),
        predictions_flow=pd.DataFrame(flow_predictions),
        diagnostics_daily=pd.DataFrame(diagnostics),
        summary_full=summary_full,
        summary_local=summary_local,
        summary_flow=summary_flow,
        counterfactual_daily=counterfactual_daily,
        counterfactual_predictions=counterfactual_predictions,
        counterfactual_summaries=counterfactual_summaries,
    )


def checkpoint_payload(
    model: FlowFDG,
    *,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    feature_dim: int,
    lookback: int,
    epoch: int,
    valid_summary: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "schema_version": "flow_fdg_checkpoint_v1",
        "method": "flow_fdg_v2",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "resolved_config": dict(config),
        "model_config": dict(model_config),
        "feature_dim": int(feature_dim),
        "lookback": int(lookback),
        "epoch": int(epoch),
        "valid_summary": dict(valid_summary),
        "selection_split": "valid",
        "test_used_for_checkpoint_selection": False,
    }


__all__ = [
    "FlowAlignmentAudit",
    "FlowEpochResult",
    "FlowEvaluationResult",
    "FlowHopDecomposition",
    "FlowStateBank",
    "align_explicit_operator_by_instrument",
    "bounded_flow_counterfactual_prediction",
    "build_flow_fdg_model",
    "build_optimizer",
    "checkpoint_payload",
    "decompose_flow_hops",
    "propagate_flow_counterfactual",
    "evaluate_flow_fdg",
    "flow_fdg_loss",
    "model_kwargs_from_config",
    "negative_pearson_ic",
    "set_stage_trainability",
    "train_flow_epoch",
    "validate_flow_fdg_config",
]
