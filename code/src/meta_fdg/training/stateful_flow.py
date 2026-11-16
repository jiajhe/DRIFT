from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import traceback
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from meta_fdg.data.temporal_cross_section import (
    FullEpochIterator,
    TemporalDay,
    TemporalSplit,
)
from meta_fdg.models import StatefulFlowFDG, StatefulFlowOutput, StatefulFlowState

from .metrics import daily_signal_metrics, summarize_daily_metrics
from .run_io import atomic_json


FIXED_SPLITS = {
    "train": ["2016-01-01", "2020-12-31"],
    "valid": ["2021-01-01", "2021-12-31"],
    "test": ["2022-01-01", "2025-12-31"],
}
SUPPORTED_STAGES = ("local", "flow_warm", "joint")
FAILURE_TAXONOMY = {
    "CONTRACT_VIOLATION",
    "DATA_CONTRACT_FAILURE",
    "FULL_EPOCH_INCOMPLETE",
    "NUMERICAL_INSTABILITY",
    "RESOURCE_EXHAUSTION",
    "OPTIMIZATION_NO_SIGNAL",
    "FLOW_COLLAPSE",
    "METRIC_SHORTFALL",
    "EXECUTION_CRASH",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Append one complete JSON record under an inter-process file lock."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(dict(payload))
    with target.open("ab", buffering=0) as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            fcntl = None  # type: ignore[assignment]
        try:
            handle.write(encoded)
            os.fsync(handle.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class AlignmentAudit:
    carried: int
    entered: int
    reentered: int
    exited: int


@dataclass(frozen=True)
class EpochResult:
    metrics: dict[str, Any]
    component_means: dict[str, float]


@dataclass(frozen=True)
class EvaluationResult:
    daily_full: pd.DataFrame
    daily_local: pd.DataFrame
    predictions_full: pd.DataFrame
    predictions_local: pd.DataFrame
    diagnostics_daily: pd.DataFrame
    summary_full: dict[str, Any]
    summary_local: dict[str, Any]


def _unique_instruments(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    instruments = tuple(str(value) for value in values)
    if len(instruments) != len(set(instruments)):
        raise ValueError(f"{label} contains duplicate instruments")
    return instruments


def align_state_by_instrument(
    state: StatefulFlowState | None,
    previous_instruments: Sequence[str] | None,
    current_instruments: Sequence[str],
    *,
    seen_instruments: set[str] | None = None,
) -> tuple[StatefulFlowState | None, AlignmentAudit]:
    """Map stock state from yesterday to today and zero every non-contiguous entry.

    A stock is carried only when it was active on the immediately preceding day.
    Therefore a stock returning after any absence receives zero ``v_prev`` and
    ``p_local_prev`` even when it appeared earlier in the epoch.  Shared latent
    channel history remains chronological and is never cleared by stock churn.
    The gather/where implementation preserves gradients for continuously active
    stocks inside one TBPTT chunk.
    """

    current = _unique_instruments(current_instruments, label="current_instruments")
    seen = set() if seen_instruments is None else set(seen_instruments)
    if state is None:
        entered = len(current)
        reentered = sum(value in seen for value in current)
        return None, AlignmentAudit(
            carried=0,
            entered=entered,
            reentered=reentered,
            exited=0,
        )
    if previous_instruments is None:
        raise ValueError("previous_instruments are required for a non-empty state")
    previous = _unique_instruments(previous_instruments, label="previous_instruments")
    if state.v_prev.ndim != 1 or state.p_local_prev.ndim != 1:
        raise ValueError("trainer stock state must be one-dimensional")
    if len(previous) != state.v_prev.numel() or state.p_local_prev.shape != state.v_prev.shape:
        raise ValueError("previous instrument metadata does not match stock state")

    lookup = {instrument: index for index, instrument in enumerate(previous)}
    positions = [lookup.get(instrument, -1) for instrument in current]
    valid = torch.tensor(
        [position >= 0 for position in positions],
        dtype=torch.bool,
        device=state.v_prev.device,
    )
    safe = torch.tensor(
        [max(position, 0) for position in positions],
        dtype=torch.long,
        device=state.v_prev.device,
    )
    gathered_v = state.v_prev.index_select(0, safe)
    gathered_local = state.p_local_prev.index_select(0, safe)
    zero_v = torch.zeros_like(gathered_v)
    zero_local = torch.zeros_like(gathered_local)
    aligned = StatefulFlowState(
        v_prev=torch.where(valid, gathered_v, zero_v),
        p_local_prev=torch.where(valid, gathered_local, zero_local),
        carrier_history=state.carrier_history,
    )
    entered_names = [value for value, is_valid in zip(current, valid.tolist(), strict=True) if not is_valid]
    return aligned, AlignmentAudit(
        carried=int(valid.sum().item()),
        entered=len(entered_names),
        reentered=sum(value in seen for value in entered_names),
        exited=len(set(previous) - set(current)),
    )


def validate_stateful_flow_config(config: Mapping[str, Any]) -> None:
    """Fail closed on every round-one training and data permission contract."""

    required = {"tier", "job_id", "seed", "data", "dataset", "train", "loss", "attempt"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Stateful Flow-FDG config misses required keys: {missing}")
    if config.get("method") != "stateful_flow_fdg_v1":
        raise ValueError("method must be stateful_flow_fdg_v1")
    if config.get("lock_id") != "paper-fixed-split-v1-20260716":
        raise ValueError("Stateful Flow-FDG must use the public fixed-split lock")
    evaluation_splits = tuple(config.get("evaluation_splits", ()))
    if evaluation_splits != ("valid",):
        raise ValueError("round-one development must evaluate valid only")

    data = config["data"]
    normalized_splits = {name: list(period) for name, period in data.get("splits", {}).items()}
    if normalized_splits != FIXED_SPLITS:
        raise ValueError("data.splits must equal the single public fixed split")
    if data.get("handler") != "Alpha158" or data.get("market") != "csi300":
        raise ValueError("round one is locked to Alpha158/CSI300")
    if tuple(data.get("load_splits", ("train", "valid"))) != ("train", "valid"):
        raise ValueError("development data permission must be exactly train+valid")
    if pd.Timestamp(data.get("materialization_end_time", "9999-12-31")) > pd.Timestamp(
        FIXED_SPLITS["valid"][1]
    ):
        raise ValueError("development materialization must end no later than validation")
    if str(data.get("context_start_time")) != "2015-07-01":
        raise ValueError("causal temporal context must begin at 2015-07-01")
    dataset = config["dataset"]
    if dataset.get("class") != "TSDatasetH" or int(dataset.get("kwargs", {}).get("step_len", -1)) < 1:
        raise ValueError("dataset must be TSDatasetH with a positive step_len")
    if dataset.get("sampler_fillna_type") not in (None, "ffill+bfill"):
        raise ValueError(
            "dataset.sampler_fillna_type must be 'ffill+bfill' or null"
        )

    train = config["train"]
    if train.get("full_epoch_required") is not True:
        raise ValueError("every epoch must be declared full_epoch_required")
    if train.get("sampling") is not False or train.get("shuffle_dates") is not False:
        raise ValueError("date/stock sampling and date shuffling are forbidden")
    if int(train.get("tbptt_days", -1)) != 20:
        raise ValueError("round one uses the locked TBPTT length of 20 trading days")
    if train.get("optimizer") != "AdamW":
        raise ValueError("round one optimizer must be AdamW")
    if float(train.get("grad_clip", 0.0)) <= 0:
        raise ValueError("gradient clipping must be positive")
    stages = list(train.get("stages", ()))
    if not stages:
        raise ValueError("at least one training stage is required")
    names = tuple(str(stage.get("name")) for stage in stages)
    if len(names) != len(set(names)) or any(name not in SUPPORTED_STAGES for name in names):
        raise ValueError(f"stages must be unique members of {SUPPORTED_STAGES}")
    if tuple(SUPPORTED_STAGES.index(name) for name in names) != tuple(
        sorted(SUPPORTED_STAGES.index(name) for name in names)
    ):
        raise ValueError("stages must follow local -> flow_warm -> joint order")
    expected_flags = {
        "local": (True, False),
        "flow_warm": (False, True),
        "joint": (True, True),
    }
    for stage in stages:
        name = str(stage["name"])
        if int(stage.get("epochs", 0)) <= 0:
            raise ValueError(f"stage {name} must request at least one full epoch")
        flags = (bool(stage.get("encoder_trainable")), bool(stage.get("flow_trainable")))
        if flags != expected_flags[name]:
            raise ValueError(f"stage {name} trainability flags differ from its locked contract")

    loss = config["loss"]
    if loss.get("point_loss", "mse") not in {"mse", "huber"}:
        raise ValueError("loss.point_loss must be mse or huber")
    if loss.get("family", "ic_mse") not in {
        "ic_mse",
        "ic_huber",
        "ic_point",
        "ic",
        "mse",
        "huber",
    }:
        raise ValueError("unsupported loss family")
    if loss.get("rankic_in_selection_gate") is not False:
        raise ValueError("RankIC is report-only and cannot enter the selection gate")
    nonnegative_weights = (
        "main_ic_weight",
        "main_point_weight",
        "local_aux_weight",
        "local_ic_weight",
        "local_point_weight",
        "role_balance_weight",
        "state_l2_weight",
        "contraction_weight",
        "lag_entropy_weight",
    )
    for key in nonnegative_weights:
        if float(loss.get(key, 0.0)) < 0:
            raise ValueError(f"loss.{key} must be non-negative")


def model_kwargs_from_config(model_config: Mapping[str, Any], *, input_dim: int) -> dict[str, Any]:
    encoder = model_config["encoder"]
    local_head = model_config["local_head"]
    roles = model_config["latent_market_roles"]
    transport = model_config["transport"]
    if str(encoder.get("arch")) != "gru":
        raise ValueError("Stateful Flow-FDG v1 currently implements only the locked GRU encoder")
    if str(roles.get("assignment")) != "entmax15":
        raise ValueError("latent market roles must use entmax15")
    output = model_config.get("output", {})
    if float(output.get("lambda_initial", 0.0)) != 0.0:
        raise ValueError("round-one lambda must initialize at exactly zero")
    encoder_dropout = float(encoder.get("dropout", 0.1))
    local_dropout = float(local_head.get("dropout", encoder_dropout))
    return {
        "input_dim": int(input_dim),
        "d_model": int(encoder["output_dim"]),
        "gru_hidden_dim": int(encoder["hidden_dim"]),
        "gru_num_layers": int(encoder["num_layers"]),
        "local_hidden_dim": int(local_head["hidden_dim"]),
        "n_roles": int(roles["count"]),
        "role_dim": int(roles.get("projection_dim", encoder["output_dim"])),
        "gate_feature_dim": int(transport.get("gate_feature_dim", 16)),
        "gate_hidden_dim": int(transport["gate_hidden_dim"]),
        "context_dim": int(transport.get("context_dim", 0)),
        "lags": tuple(int(value) for value in transport["lags"]),
        "source_mode": str(transport["source"]),
        "dropout": encoder_dropout,
        "encoder_dropout": encoder_dropout,
        "local_dropout": local_dropout,
        "tau_sender": float(roles.get("sender_temperature", 1.0)),
        "tau_receiver": float(roles.get("receiver_temperature", 1.0)),
        "rho_init": float(transport["rho_initial"]),
        "rho_max": float(transport.get("rho_max", 0.9)),
        "signed_strength_init": float(transport.get("direction_init_scale", 0.02)),
        "reliability_init": float(transport.get("reliability_initial", 0.5)),
        "eps": float(transport.get("epsilon", 1e-6)),
    }


def build_stateful_flow_model(
    input_dim: int,
    model_config: Mapping[str, Any],
    device: torch.device,
) -> StatefulFlowFDG:
    return StatefulFlowFDG(**model_kwargs_from_config(model_config, input_dim=input_dim)).to(device)


def set_stage_trainability(model: StatefulFlowFDG, stage: str) -> None:
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    groups = model.parameter_groups()
    local_trainable = stage in {"local", "joint"}
    flow_trainable = stage in {"flow_warm", "joint"}
    for parameter in groups["local"]:
        parameter.requires_grad_(local_trainable)
    for parameter in groups["flow"]:
        parameter.requires_grad_(flow_trainable)


def build_stage_optimizer(
    model: StatefulFlowFDG,
    train_config: Mapping[str, Any],
    stage: str,
) -> torch.optim.AdamW:
    set_stage_trainability(model, stage)
    groups = model.parameter_groups()
    parameters: list[dict[str, Any]] = []
    if stage in {"local", "joint"}:
        learning_rate = (
            float(train_config["encoder_learning_rate"])
            if stage == "local"
            else float(train_config["joint_encoder_learning_rate"])
        )
        parameters.append(
            {"params": [value for value in groups["local"] if value.requires_grad], "lr": learning_rate, "name": "local"}
        )
    if stage in {"flow_warm", "joint"}:
        parameters.append(
            {
                "params": [value for value in groups["flow"] if value.requires_grad],
                "lr": float(train_config["flow_learning_rate"]),
                "name": "flow",
            }
        )
    if not parameters or any(not group["params"] for group in parameters):
        raise ValueError(f"stage {stage} resolved an empty AdamW parameter group")
    return torch.optim.AdamW(parameters, weight_decay=float(train_config.get("weight_decay", 0.0)))


def negative_pearson_ic(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mask is not None:
        prediction = prediction[mask]
        target = target[mask]
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    left = prediction - prediction.mean()
    right = target - target.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum() + float(eps))
    return -(left * right).sum() / denominator


def _point_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    kind: str,
    *,
    huber_delta: float,
) -> torch.Tensor:
    if mask is not None:
        prediction = prediction[mask]
        target = target[mask]
    if kind == "mse":
        return (prediction - target).square().mean()
    if kind == "huber":
        return nn.functional.huber_loss(prediction, target, delta=float(huber_delta))
    raise ValueError(f"unsupported point loss {kind!r}")


def stateful_flow_loss(
    output: StatefulFlowOutput,
    target: torch.Tensor,
    loss_config: Mapping[str, Any],
    *,
    mask: torch.Tensor | None = None,
    flow_regularizers_enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    point_kind = str(loss_config.get("point_loss", "mse"))
    huber_delta = float(loss_config.get("huber_delta", 1.0))
    main_ic = negative_pearson_ic(output.prediction, target, mask)
    main_point = _point_loss(output.prediction, target, mask, point_kind, huber_delta=huber_delta)
    local_ic = negative_pearson_ic(output.p_local, target, mask)
    local_point = _point_loss(output.p_local, target, mask, point_kind, huber_delta=huber_delta)

    sender = output.diagnostics["S"]
    receiver = output.diagnostics["R"]
    sender_occupancy = sender.mean(dim=-2)
    receiver_occupancy = receiver.mean(dim=-2)
    uniform = 1.0 / float(sender.shape[-1])
    role_balance = (sender_occupancy - uniform).square().mean() + (
        receiver_occupancy - uniform
    ).square().mean()
    state_l2 = output.v.square().mean()
    contraction_weight = float(loss_config.get("contraction_weight", 0.0))
    if "T" in output.diagnostics:
        transport = output.diagnostics["T"]
        row_l1 = transport.abs().sum(dim=-1)
        contraction = torch.relu(
            output.diagnostics["rho"] * row_l1.max() - 1.0
        ).square()
    else:
        if flow_regularizers_enabled and contraction_weight > 0.0:
            raise ValueError(
                "a positive contraction_weight requires the exact signed "
                "transport diagnostic T"
            )
        contraction = output.v.sum() * 0.0
    lag_weights = output.diagnostics["pi"]
    # Positive weight encourages a non-degenerate, high-entropy lag kernel.
    lag_negative_entropy = (lag_weights * lag_weights.clamp_min(1e-12).log()).sum(dim=-1).mean()

    components = {
        "main_ic_loss": main_ic,
        "main_point_loss": main_point,
        "local_ic_loss": local_ic,
        "local_point_loss": local_point,
        "role_balance": role_balance,
        "state_l2": state_l2,
        "contraction": contraction,
        "lag_negative_entropy": lag_negative_entropy,
    }
    flow_regularizer_scale = 1.0 if flow_regularizers_enabled else 0.0
    total = (
        float(loss_config.get("main_ic_weight", 1.0)) * main_ic
        + float(loss_config.get("main_point_weight", 0.0)) * main_point
        + float(loss_config.get("local_aux_weight", 0.0))
        * (
            float(loss_config.get("local_ic_weight", 1.0)) * local_ic
            + float(loss_config.get("local_point_weight", 0.0)) * local_point
        )
        + flow_regularizer_scale
        * (
            float(loss_config.get("role_balance_weight", 0.0)) * role_balance
            + float(loss_config.get("state_l2_weight", 0.0)) * state_l2
            + contraction_weight * contraction
            + float(loss_config.get("lag_entropy_weight", 0.0))
            * lag_negative_entropy
        )
    )
    components["total_loss"] = total
    if not torch.isfinite(total):
        raise FloatingPointError("Stateful Flow-FDG produced a non-finite training loss")
    return total, components


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().square().sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu())


def _numeric_context(
    *,
    stage: str,
    stage_epoch: int | None,
    chunk_index: int,
    dates: Sequence[pd.Timestamp],
) -> str:
    return (
        f"stage={stage} stage_epoch={stage_epoch if stage_epoch is not None else 'unknown'} "
        f"chunk={chunk_index} "
        f"dates={dates[0].date()}..{dates[-1].date()}"
    )


def _assert_finite_trainable_gradients(
    model: nn.Module,
    *,
    stage: str,
    stage_epoch: int | None,
    chunk_index: int,
    dates: Sequence[pd.Timestamp],
) -> None:
    bad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    if bad:
        raise FloatingPointError(
            "non-finite trainable gradients before clipping: "
            f"{_numeric_context(stage=stage, stage_epoch=stage_epoch, chunk_index=chunk_index, dates=dates)} "
            f"parameters={bad}"
        )


def _assert_finite_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    stage: str,
    stage_epoch: int | None,
    chunk_index: int,
    dates: Sequence[pd.Timestamp],
) -> None:
    bad_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not torch.isfinite(parameter).all()
    ]
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    bad_states: list[str] = []
    for parameter, state in optimizer.state.items():
        parameter_name = names_by_id.get(id(parameter), "<unnamed>")
        for state_name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                bad_states.append(f"{parameter_name}.{state_name}")
    if bad_parameters or bad_states:
        raise FloatingPointError(
            "non-finite values after optimizer step: "
            f"{_numeric_context(stage=stage, stage_epoch=stage_epoch, chunk_index=chunk_index, dates=dates)} "
            f"parameters={bad_parameters} optimizer_states={bad_states}"
        )


@contextmanager
def _numeric_phase_audit(
    numeric_audit_log: str | Path | None,
    *,
    stage: str,
    stage_epoch: int | None,
    chunk_index: int,
    dates: Sequence[pd.Timestamp],
    phase: str,
    active_date: pd.Timestamp | None = None,
) -> Iterator[None]:
    """Persist the first failing numeric phase with exact temporal context."""

    try:
        yield
    except BaseException as error:
        if numeric_audit_log is not None:
            append_jsonl(
                numeric_audit_log,
                {
                    "event": "numeric_chunk_failure",
                    "at_utc": _utc_now(),
                    "stage": stage,
                    "stage_epoch": stage_epoch,
                    "chunk_index": chunk_index,
                    "date_min": str(dates[0].date()),
                    "date_max": str(dates[-1].date()),
                    "active_date": None
                    if active_date is None
                    else str(active_date.date()),
                    "phase": phase,
                    "exception_type": type(error).__name__,
                    "error": str(error),
                    "test_touched": False,
                },
            )
        raise


def _day_count_from_chunk(mask: torch.Tensor, offset: int) -> int:
    count = int(mask[offset].sum().item())
    if count < 1:
        raise AssertionError("full epoch emitted an empty daily cross-section")
    return count


def train_full_epoch(
    model: StatefulFlowFDG,
    split: TemporalSplit,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_config: Mapping[str, Any],
    tbptt_days: int = 20,
    grad_clip: float = 3.0,
    stage: str = "joint",
    stage_epoch: int | None = None,
    numeric_audit_log: str | Path | None = None,
) -> EpochResult:
    """Train exactly once over every chronological day and row in ``split``."""

    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"unknown training stage {stage!r}")
    model.train()
    flow_regularizers_enabled = stage != "local"
    compute_transport_diagnostics = (
        flow_regularizers_enabled
        and float(loss_config.get("contraction_weight", 0.0)) > 0.0
    )
    if not any(parameter.requires_grad for parameter in model.parameter_groups()["local"]):
        # A frozen encoder/head must also have frozen stochastic behavior.
        model.temporal_encoder.eval()
        model.temporal_projection.eval()
        model.local_head.eval()
    iterator = FullEpochIterator(
        split,
        chunk_size=int(tbptt_days),
        expected_days=split.coverage.emitted_days,
        expected_rows=split.coverage.emitted_rows,
    )
    state: StatefulFlowState | None = None
    previous_instruments: tuple[str, ...] | None = None
    seen: set[str] = set()
    emitted_days = 0
    emitted_rows = 0
    carried = entered = reentered = exited = 0
    component_totals: dict[str, float] = {}
    gradient_norm_max = 0.0
    eta_gradient_abs_max = 0.0
    flow_v_std_total = 0.0
    optimizer_steps = 0

    for chunk_index, chunk in enumerate(iterator, start=1):
        optimizer.zero_grad(set_to_none=True)
        chunk_loss: torch.Tensor | None = None
        for offset, instruments in enumerate(chunk.instruments):
            count = _day_count_from_chunk(chunk.mask, offset)
            if count != len(instruments):
                raise AssertionError("padded mask and instrument coverage differ")
            aligned_state, audit = align_state_by_instrument(
                state,
                previous_instruments,
                instruments,
                seen_instruments=seen,
            )
            features = chunk.sequences[offset, :count].to(device, non_blocking=True)
            target = chunk.target[offset, :count].to(device, non_blocking=True)
            with _numeric_phase_audit(
                numeric_audit_log,
                stage=stage,
                stage_epoch=stage_epoch,
                chunk_index=chunk_index,
                dates=chunk.dates,
                phase="forward",
                active_date=chunk.dates[offset],
            ):
                output = model(
                    features,
                    state=aligned_state,
                    compute_graph_diagnostics=False,
                    compute_transport_diagnostics=compute_transport_diagnostics,
                )
            with _numeric_phase_audit(
                numeric_audit_log,
                stage=stage,
                stage_epoch=stage_epoch,
                chunk_index=chunk_index,
                dates=chunk.dates,
                phase="loss",
                active_date=chunk.dates[offset],
            ):
                loss, components = stateful_flow_loss(
                    output,
                    target,
                    loss_config,
                    flow_regularizers_enabled=flow_regularizers_enabled,
                )
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value.detach().cpu())
            flow_v_std_total += float(output.v.detach().std(unbiased=False).cpu())
            state = output.state
            previous_instruments = tuple(instruments)
            seen.update(instruments)
            emitted_days += 1
            emitted_rows += count
            carried += audit.carried
            entered += audit.entered
            reentered += audit.reentered
            exited += audit.exited

        if chunk_loss is None:
            raise AssertionError("full epoch yielded an empty TBPTT chunk")
        with _numeric_phase_audit(
            numeric_audit_log,
            stage=stage,
            stage_epoch=stage_epoch,
            chunk_index=chunk_index,
            dates=chunk.dates,
            phase="backward",
        ):
            (chunk_loss / float(chunk.day_count)).backward()
        with _numeric_phase_audit(
            numeric_audit_log,
            stage=stage,
            stage_epoch=stage_epoch,
            chunk_index=chunk_index,
            dates=chunk.dates,
            phase="preclip_guard",
        ):
            _assert_finite_trainable_gradients(
                model,
                stage=stage,
                stage_epoch=stage_epoch,
                chunk_index=chunk_index,
                dates=chunk.dates,
            )
        if model.eta.grad is not None:
            eta_gradient_abs_max = max(
                eta_gradient_abs_max, float(model.eta.grad.detach().abs().cpu())
            )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradient_norm_max = max(gradient_norm_max, _gradient_norm(trainable))
        with _numeric_phase_audit(
            numeric_audit_log,
            stage=stage,
            stage_epoch=stage_epoch,
            chunk_index=chunk_index,
            dates=chunk.dates,
            phase="clip",
        ):
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                float(grad_clip),
                error_if_nonfinite=True,
            )
            if not torch.isfinite(clipped_norm):
                raise FloatingPointError(
                    "non-finite gradient norm during clipping: "
                    f"{_numeric_context(stage=stage, stage_epoch=stage_epoch, chunk_index=chunk_index, dates=chunk.dates)}"
                )
        with _numeric_phase_audit(
            numeric_audit_log,
            stage=stage,
            stage_epoch=stage_epoch,
            chunk_index=chunk_index,
            dates=chunk.dates,
            phase="optimizer_step",
        ):
            optimizer.step()
            _assert_finite_optimizer_step(
                model,
                optimizer,
                stage=stage,
                stage_epoch=stage_epoch,
                chunk_index=chunk_index,
                dates=chunk.dates,
            )
        if numeric_audit_log is not None:
            append_jsonl(
                numeric_audit_log,
                {
                    "event": "numeric_chunk_complete",
                    "at_utc": _utc_now(),
                    "stage": stage,
                    "stage_epoch": stage_epoch,
                    "chunk_index": chunk_index,
                    "date_min": str(chunk.dates[0].date()),
                    "date_max": str(chunk.dates[-1].date()),
                    "days": int(chunk.day_count),
                    "rows": int(chunk.row_count),
                    "gradient_norm_preclip": float(clipped_norm.detach().cpu()),
                    "grad_clip": float(grad_clip),
                    "preclip_gradients_finite": True,
                    "poststep_parameters_finite": True,
                    "poststep_optimizer_state_finite": True,
                },
            )
        optimizer_steps += 1
        # Preserve numeric payload/carrier values while cutting only autograd history.
        if state is not None:
            state = state.detach()

    if emitted_days != split.coverage.emitted_days or emitted_rows != split.coverage.emitted_rows:
        raise AssertionError(
            "full epoch coverage failed: "
            f"days={emitted_days}/{split.coverage.emitted_days}, "
            f"rows={emitted_rows}/{split.coverage.emitted_rows}"
        )
    means = {name: value / float(emitted_days) for name, value in component_totals.items()}
    metrics: dict[str, Any] = {
        "train_days_seen": emitted_days,
        "train_days_expected": split.coverage.emitted_days,
        "train_rows_seen": emitted_rows,
        "train_rows_expected": split.coverage.emitted_rows,
        "coverage_complete": True,
        "tbptt_days": int(tbptt_days),
        "optimizer_steps": optimizer_steps,
        "state_detach_count": optimizer_steps,
        "state_clear_count": 1,
        "alignment_carried": carried,
        "alignment_entered": entered,
        "alignment_reentered": reentered,
        "alignment_exited": exited,
        "gradient_norm_max": gradient_norm_max,
        "eta_gradient_abs_max": eta_gradient_abs_max,
        "flow_v_std_mean": flow_v_std_total / float(emitted_days),
        "rho": float(model.rho.detach().cpu()),
        "lambda": float(model.flow_lambda.detach().cpu()),
        "stage": stage,
        "flow_regularizers_enabled": flow_regularizers_enabled,
        "exact_transport_regularizer": compute_transport_diagnostics,
    }
    return EpochResult(metrics=metrics, component_means=means)


def _diagnostic_scalar(value: torch.Tensor, reduction: str = "mean") -> float:
    detached = value.detach().float()
    if detached.numel() == 0:
        return math.nan
    if reduction == "mean":
        result = detached.mean()
    elif reduction == "std":
        result = detached.std(unbiased=False)
    elif reduction == "max":
        result = detached.max()
    elif reduction == "min":
        result = detached.min()
    else:
        raise ValueError(f"unknown diagnostic reduction {reduction!r}")
    return float(result.cpu())


def _assert_finite_evaluation_output(
    output: StatefulFlowOutput,
    *,
    split_name: str,
    date: pd.Timestamp,
) -> None:
    tensors: dict[str, torch.Tensor] = {
        "prediction": output.prediction,
        "p_local": output.p_local,
        "v": output.v,
        "state.v_prev": output.state.v_prev,
        "state.p_local_prev": output.state.p_local_prev,
        "state.carrier_history": output.state.carrier_history,
        **{f"diagnostics.{name}": value for name, value in output.diagnostics.items()},
    }
    nonfinite = [
        name for name, value in tensors.items() if not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise FloatingPointError(
            f"non-finite {split_name} evaluation output on {date.date()}: {nonfinite}"
        )


@torch.no_grad()
def evaluate_stateful_flow(
    model: StatefulFlowFDG,
    evaluation_split: TemporalSplit,
    *,
    train_warmup_days: Sequence[TemporalDay],
    device: torch.device,
    loss_config: Mapping[str, Any],
    split_name: str = "valid",
) -> EvaluationResult:
    """Evaluate every requested day after feature-only chronological warm-up.

    Training keeps the default ``valid`` split.  The explicit ``test`` value is
    reserved for the separate, evaluation-only reporting runner so opening the
    test period cannot silently change the checkpoint-selection path.
    """

    split_name = str(split_name)
    if split_name not in {"valid", "test"}:
        raise ValueError("evaluation split_name must be 'valid' or 'test'")
    if evaluation_split.split != split_name:
        raise ValueError(
            "evaluation split identity mismatch: "
            f"object={evaluation_split.split!r}, requested={split_name!r}"
        )

    model.eval()
    state: StatefulFlowState | None = None
    previous_instruments: tuple[str, ...] | None = None
    seen: set[str] = set()

    for day in train_warmup_days:
        state, _ = align_state_by_instrument(
            state,
            previous_instruments,
            day.instruments,
            seen_instruments=seen,
        )
        output = model(day.sequences.to(device), state=state)
        _assert_finite_evaluation_output(
            output, split_name=f"{split_name}_warmup", date=day.date
        )
        state = output.state
        previous_instruments = day.instruments
        seen.update(day.instruments)

    full_daily_rows: list[dict[str, Any]] = []
    local_daily_rows: list[dict[str, Any]] = []
    full_prediction_rows: list[dict[str, Any]] = []
    local_prediction_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    full_losses: list[float] = []
    local_losses: list[float] = []
    rows_seen = 0

    for day in evaluation_split.days:
        aligned_state, audit = align_state_by_instrument(
            state,
            previous_instruments,
            day.instruments,
            seen_instruments=seen,
        )
        target = day.target.to(device)
        if not torch.isfinite(target).all() or not torch.isfinite(day.raw_target).all():
            raise FloatingPointError(
                f"non-finite {split_name} evaluation target on {day.date.date()}"
            )
        output = model(day.sequences.to(device), state=aligned_state)
        _assert_finite_evaluation_output(
            output, split_name=split_name, date=day.date
        )
        _, components = stateful_flow_loss(output, target, loss_config)
        full_losses.append(float(components["main_point_loss"].cpu()))
        local_losses.append(float(components["local_point_loss"].cpu()))

        full_score = output.prediction.detach().cpu().numpy()
        local_score = output.p_local.detach().cpu().numpy()
        raw_target = day.raw_target.numpy()
        full_metrics = daily_signal_metrics(full_score, raw_target)
        local_metrics = daily_signal_metrics(local_score, raw_target)
        if int(full_metrics["n"]) != day.row_count or int(local_metrics["n"]) != day.row_count:
            raise AssertionError(
                f"{split_name} metric coverage dropped predictions on {day.date.date()}"
            )
        full_daily_rows.append({"split": split_name, "variant": "full", "date": day.date, **full_metrics})
        local_daily_rows.append({"split": split_name, "variant": "local", "date": day.date, **local_metrics})
        for position, instrument in enumerate(day.instruments):
            base = {
                "split": split_name,
                "date": day.date,
                "instrument": instrument,
                "label": float(raw_target[position]),
            }
            full_prediction_rows.append({**base, "score": float(full_score[position])})
            local_prediction_rows.append({**base, "score": float(local_score[position])})

        sender = output.diagnostics["S"]
        receiver = output.diagnostics["R"]
        lag_weights = output.diagnostics["pi"]
        transport = output.diagnostics["T"]
        gate = output.diagnostics["g"]
        sender_entropy = -(sender * sender.clamp_min(1e-12).log()).sum(dim=-1)
        receiver_entropy = -(receiver * receiver.clamp_min(1e-12).log()).sum(dim=-1)
        lag_entropy = -(lag_weights * lag_weights.clamp_min(1e-12).log()).sum(dim=-1)
        diagnostic_rows.append(
            {
                "split": split_name,
                "date": day.date,
                "n": day.row_count,
                "rho": _diagnostic_scalar(output.diagnostics["rho"]),
                "lambda": _diagnostic_scalar(output.diagnostics["lambda"]),
                "eta": float(model.eta.detach().cpu()),
                "v_std": _diagnostic_scalar(output.v, "std"),
                "p_local_std": _diagnostic_scalar(output.p_local, "std"),
                "prediction_std": _diagnostic_scalar(output.prediction, "std"),
                "flow_contribution_std": _diagnostic_scalar(
                    output.diagnostics["lambda"] * output.v, "std"
                ),
                "source_std": _diagnostic_scalar(output.diagnostics["u"], "std"),
                "payload_std": _diagnostic_scalar(output.diagnostics["b"], "std"),
                "gate_mean": _diagnostic_scalar(gate),
                "gate_std": _diagnostic_scalar(gate, "std"),
                "gate_min": _diagnostic_scalar(gate, "min"),
                "gate_max": _diagnostic_scalar(gate, "max"),
                "reliability_mean": _diagnostic_scalar(output.diagnostics["r"]),
                "direction_abs_mean": _diagnostic_scalar(output.diagnostics["a"].abs()),
                "sender_role_entropy": _diagnostic_scalar(sender_entropy),
                "receiver_role_entropy": _diagnostic_scalar(receiver_entropy),
                "sender_role_zero_fraction": _diagnostic_scalar((sender == 0).float()),
                "receiver_role_zero_fraction": _diagnostic_scalar((receiver == 0).float()),
                "lag_entropy": _diagnostic_scalar(lag_entropy),
                "transport_abs_mean": _diagnostic_scalar(transport.abs()),
                "transport_fro_norm": float(transport.detach().float().norm().cpu()),
                "transport_row_l1_max": _diagnostic_scalar(transport.abs().sum(dim=-1), "max"),
                "alignment_carried": audit.carried,
                "alignment_entered": audit.entered,
                "alignment_reentered": audit.reentered,
                "alignment_exited": audit.exited,
            }
        )
        rows_seen += day.row_count
        state = output.state
        previous_instruments = day.instruments
        seen.update(day.instruments)

    if (
        len(full_daily_rows) != evaluation_split.coverage.emitted_days
        or rows_seen != evaluation_split.coverage.emitted_rows
    ):
        raise AssertionError(
            f"{split_name} evaluation did not cover every locked day and row"
        )
    daily_full = pd.DataFrame(full_daily_rows)
    daily_local = pd.DataFrame(local_daily_rows)
    summary_full = summarize_daily_metrics(daily_full)
    summary_local = summarize_daily_metrics(daily_local)
    summary_full.update(
        {
            "point_loss": float(np.mean(full_losses)),
            "rows": rows_seen,
            "days_expected": evaluation_split.coverage.emitted_days,
            "rows_expected": evaluation_split.coverage.emitted_rows,
            "coverage_complete": True,
            "warmup_days": len(train_warmup_days),
        }
    )
    summary_local.update(
        {
            "point_loss": float(np.mean(local_losses)),
            "rows": rows_seen,
            "days_expected": evaluation_split.coverage.emitted_days,
            "rows_expected": evaluation_split.coverage.emitted_rows,
            "coverage_complete": True,
            "warmup_days": len(train_warmup_days),
        }
    )
    return EvaluationResult(
        daily_full=daily_full,
        daily_local=daily_local,
        predictions_full=pd.DataFrame(full_prediction_rows),
        predictions_local=pd.DataFrame(local_prediction_rows),
        diagnostics_daily=pd.DataFrame(diagnostic_rows),
        summary_full=summary_full,
        summary_local=summary_local,
    )


def checkpoint_payload(
    model: StatefulFlowFDG,
    *,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    feature_dim: int,
    lookback: int,
    stage: str,
    stage_epoch: int,
    global_epoch: int,
    valid_full: Mapping[str, Any],
    valid_local: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    payload: dict[str, Any] = {
        "schema_version": "stateful_flow_fdg_checkpoint_v2",
        "saved_at_utc": _utc_now(),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": dict(config),
        "model_config": dict(model_config),
        "feature_dim": int(feature_dim),
        "lookback": int(lookback),
        "stage": str(stage),
        "stage_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "selection_split": "valid",
        "selection_metric": "IC",
        "valid_full": dict(valid_full),
        "valid_local": dict(valid_local),
        "test_touched": False,
    }
    if optimizer is not None:
        payload.update(
            {
                "optimizer_state": optimizer.state_dict(),
                "rng_state": {
                    "python": random.getstate(),
                    "numpy_algorithm": str(numpy_state[0]),
                    "numpy_keys": torch.from_numpy(numpy_state[1].copy()),
                    "numpy_position": int(numpy_state[2]),
                    "numpy_has_gauss": int(numpy_state[3]),
                    "numpy_cached_gaussian": float(numpy_state[4]),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else [],
                },
                "snapshot_boundary": "after_complete_train_and_validation_epoch_before_any_stage_best_restore",
                "automatic_resume_supported": False,
            }
        )
    return payload


def save_training_history(history: Sequence[Mapping[str, Any]], run_root: Path) -> None:
    frame = pd.DataFrame(list(history))
    frame.to_parquet(run_root / "training_history.parquet", index=False)
    frame.to_csv(run_root / "logs" / "epochs.csv", index=False)


def write_attempt_plan(
    run_root: Path,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    plan = {
        "schema_version": "stateful_flow_attempt_plan_v1",
        "created_at_utc": _utc_now(),
        "attempt": dict(config["attempt"]),
        "job_id": config["job_id"],
        "protocol_id": config.get("protocol_id"),
        "fixed_splits": FIXED_SPLITS,
        "evaluation_permission": ["valid"],
        "test_touched": False,
        "full_epoch_contract": {
            "sampling": False,
            "shuffle_dates": False,
            "tbptt_days": int(config["train"]["tbptt_days"]),
            "state_boundary": "detach_not_clear",
            "validation_after_every_complete_epoch": True,
            "numeric_guards": [
                "preclip_named_gradients",
                "clip_error_if_nonfinite",
                "poststep_named_parameters",
                "poststep_adam_state",
            ],
            "numeric_chunk_log": "logs/numeric_chunks.jsonl",
        },
        "stages": list(config["train"]["stages"]),
        "loss": dict(config["loss"]),
        "model": dict(model_config),
        "implementation": dict(config.get("implementation", {})),
        "success_gates": dict(config.get("success_gates", {})),
    }
    atomic_json(run_root / "attempt_plan.json", plan)
    digest = sha256_file(run_root / "attempt_plan.json")
    (run_root / "attempt_plan.sha256").write_text(f"{digest}  attempt_plan.json\n", encoding="utf-8")
    return plan


def write_model_structure(run_root: Path, model: StatefulFlowFDG) -> dict[str, Any]:
    groups = model.parameter_groups()
    structure = {
        **model.structure_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count_initial": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "parameter_groups": {
            name: {
                "tensor_count": len(parameters),
                "parameter_count": sum(parameter.numel() for parameter in parameters),
            }
            for name, parameters in groups.items()
        },
        "state_alignment": "instrument identity, contiguous carry only, zero every re-entry",
        "tbptt_boundary": "detach numeric state, never clear it",
    }
    atomic_json(run_root / "model_structure.json", structure)
    return structure


def _gate_result(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def evaluate_success_gates(
    config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    evaluation: EvaluationResult,
) -> tuple[bool, list[dict[str, Any]]]:
    requested = config.get("success_gates", {})
    gates: list[dict[str, Any]] = []
    completed_epochs = len(history)
    coverage_complete = bool(history) and all(bool(row.get("coverage_complete")) for row in history)
    complete_requirement = requested.get("complete_full_epochs", True)
    if isinstance(complete_requirement, bool):
        passed = coverage_complete if complete_requirement else True
        required: Any = complete_requirement
    else:
        required = int(complete_requirement)
        passed = coverage_complete and completed_epochs >= required
    gates.append(_gate_result("complete_full_epochs", passed, completed_epochs, required))

    numeric_values: list[float] = []
    for summary in (evaluation.summary_full, evaluation.summary_local):
        numeric_values.extend(float(value) for value in summary.values() if isinstance(value, (int, float)))
    finite = all(math.isfinite(value) for value in numeric_values)
    gates.append(_gate_result("finite_metrics", finite, finite, True))
    gates.append(_gate_result("test_touched", True, False, False))

    if "lambda_or_eta_gradient_nonzero" in requested:
        observed = max(float(row.get("eta_gradient_abs_max", 0.0)) for row in history)
        gates.append(_gate_result("lambda_or_eta_gradient_nonzero", observed > 0.0, observed, "> 0"))
    if "flow_score_std_min" in requested:
        observed = float(evaluation.diagnostics_daily["v_std"].mean())
        required_value = float(requested["flow_score_std_min"])
        gates.append(_gate_result("flow_score_std_min", observed >= required_value, observed, required_value))
    if "valid_full_ic_min" in requested:
        observed = float(evaluation.summary_full["IC"])
        required_value = float(requested["valid_full_ic_min"])
        gates.append(_gate_result("valid_full_ic_min", observed >= required_value, observed, required_value))
    if "valid_delta_over_local_min" in requested:
        observed = float(evaluation.summary_full["IC"]) - float(evaluation.summary_local["IC"])
        required_value = float(requested["valid_delta_over_local_min"])
        gates.append(
            _gate_result("valid_delta_over_local_min", observed >= required_value, observed, required_value)
        )
    return all(item["passed"] for item in gates), gates


def classify_failure(
    error: BaseException | None,
    *,
    gates: Sequence[Mapping[str, Any]] | None = None,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, str, str]:
    """Return taxonomy, evidence-based cause, and the next testable hypothesis."""

    if error is not None:
        message = str(error)
        if isinstance(error, (ValueError, RuntimeError)) and any(
            token in message.lower()
            for token in (
                "non-finite temporal features",
                "sampler",
                "data contract",
                "provider",
            )
        ):
            return (
                "DATA_CONTRACT_FAILURE",
                message,
                "Audit the causal sampler/window preprocessing and row coverage; do not change model structure or loss.",
            )
        if isinstance(error, (ValueError, KeyError)) and any(
            token in message.lower() for token in ("contract", "split", "sampling", "test", "stage")
        ):
            return (
                "CONTRACT_VIOLATION",
                message,
                "Correct only the violated resolved-config contract, then rerun the same attempt once.",
            )
        if isinstance(error, AssertionError) and "coverage" in message.lower():
            return (
                "FULL_EPOCH_INCOMPLETE",
                message,
                "Audit sampler day/row identities against the full-epoch iterator before any model change.",
            )
        if isinstance(error, FloatingPointError) or any(
            token in message.lower()
            for token in (
                "non-finite",
                "must contain only finite values",
                "entmax",
            )
        ):
            return (
                "NUMERICAL_INSTABILITY",
                message,
                "Identify the first affected parameter with the pre-clip and post-step finite guards; test a lower flow learning rate only if the stable analytical Entmax backward still fails.",
            )
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower():
            return (
                "RESOURCE_EXHAUSTION",
                message,
                "Keep all days/rows but reduce diagnostic retention or model width; do not introduce sampling.",
            )
        return (
            "EXECUTION_CRASH",
            f"{type(error).__name__}: {message}",
            "Reproduce the crash on the same config and isolate the failing boundary before changing the model.",
        )

    failed = {str(item["name"]): item for item in gates or () if not bool(item.get("passed"))}
    if "complete_full_epochs" in failed:
        return (
            "FULL_EPOCH_INCOMPLETE",
            f"coverage gate failed: {failed['complete_full_epochs']}",
            "Audit the full chronological iterator; model tuning is not authorized until coverage passes.",
        )
    if "finite_metrics" in failed:
        return (
            "NUMERICAL_INSTABILITY",
            "one or more reported metrics are non-finite",
            "Test Huber loss and a lower flow learning rate with the structure unchanged.",
        )
    if "lambda_or_eta_gradient_nonzero" in failed:
        return (
            "OPTIMIZATION_NO_SIGNAL",
            f"eta/lambda gradient gate failed: {failed['lambda_or_eta_gradient_nonzero']}",
            "Audit zero-lambda gradient routing; if correct, test a recorded lambda initialization of 0.01.",
        )
    if "flow_score_std_min" in failed or "valid_delta_over_local_min" in failed:
        return (
            "FLOW_COLLAPSE",
            json.dumps(failed, ensure_ascii=False, default=str),
            "Inspect role/gate/lag diagnostics, then vary one of source mode, rho initialization, or flow loss at a time.",
        )
    if "valid_full_ic_min" in failed:
        return (
            "METRIC_SHORTFALL",
            json.dumps(failed["valid_full_ic_min"], ensure_ascii=False, default=str),
            "Compare local/full daily deltas and choose the next single structural or loss hypothesis on validation only.",
        )
    if failed:
        return (
            "METRIC_SHORTFALL",
            json.dumps(failed, ensure_ascii=False, default=str),
            "Use validation diagnostics to define one recorded next hypothesis; do not inspect test.",
        )
    return ("METRIC_SHORTFALL", "no explicit gate evidence", "Inspect validation-only diagnostics.")


def write_failure_analysis(
    run_root: Path,
    *,
    error: BaseException | None,
    gates: Sequence[Mapping[str, Any]] | None,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if error is None and gates is not None and all(bool(item.get("passed")) for item in gates):
        payload = {
            "schema_version": "stateful_flow_failure_analysis_v1",
            "status": "not_required",
            "taxonomy": None,
            "evidence": "all execution and attempt gates passed",
            "next_hypothesis": None,
            "test_touched": False,
        }
    else:
        taxonomy, evidence, next_hypothesis = classify_failure(
            error, gates=gates, history=history
        )
        if taxonomy not in FAILURE_TAXONOMY:
            raise AssertionError(f"unregistered failure taxonomy {taxonomy}")
        payload = {
            "schema_version": "stateful_flow_failure_analysis_v1",
            "status": "analyzed",
            "taxonomy": taxonomy,
            "evidence": evidence,
            "next_hypothesis": next_hypothesis,
            "exception_type": None if error is None else type(error).__name__,
            "traceback": None if error is None else traceback.format_exc(),
            "test_touched": False,
        }
    atomic_json(run_root / "failure_analysis.json", payload)
    return payload


def write_manifest(run_root: Path) -> dict[str, Any]:
    excluded = {"manifest.json", "MANIFEST.sha256"}
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in run_root.rglob("*") if value.is_file()):
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded:
            continue
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": "stateful_flow_run_manifest_v1",
        "created_at_utc": _utc_now(),
        "run_root": str(run_root.resolve()),
        "files": files,
    }
    atomic_json(run_root / "manifest.json", manifest)
    rows = [
        f"{sha256_file(run_root / relative)}  {relative}\n"
        for relative in sorted(files)
    ]
    rows.append(f"{sha256_file(run_root / 'manifest.json')}  manifest.json\n")
    (run_root / "MANIFEST.sha256").write_text("".join(rows), encoding="utf-8")
    return manifest


def campaign_ledger_path(repo_root: Path, config: Mapping[str, Any]) -> Path:
    campaign_id = str(config["attempt"]["campaign_id"])
    return repo_root / "runs" / str(config["tier"]) / "_campaigns" / campaign_id / "ledger.jsonl"


def append_campaign_ledger(
    repo_root: Path,
    config: Mapping[str, Any],
    run_root: Path,
    outcome: Mapping[str, Any],
) -> Path:
    path = campaign_ledger_path(repo_root, config)
    artifact_paths = {
        "resolved_config": run_root / "resolved_config.json",
        "attempt_plan": run_root / "attempt_plan.json",
        "model_structure": run_root / "model_structure.json",
        "attempt_outcome": run_root / "attempt_outcome.json",
        "manifest": run_root / "manifest.json",
        "manifest_chain": run_root / "MANIFEST.sha256",
    }
    artifact_hashes = {
        name: sha256_file(artifact) if artifact.is_file() else None
        for name, artifact in artifact_paths.items()
    }
    append_jsonl(
        path,
        {
            "recorded_at_utc": _utc_now(),
            "campaign_id": config["attempt"]["campaign_id"],
            "attempt_id": config["attempt"]["attempt_id"],
            "job_id": config["job_id"],
            "run_root": str(run_root.resolve()),
            "status": outcome.get("status"),
            "gates_passed": outcome.get("gates_passed"),
            "best_valid_full_ic": outcome.get("best_valid_full_ic"),
            "valid_full_ic": outcome.get("valid_full", {}).get("IC"),
            "valid_local_ic": outcome.get("valid_local", {}).get("IC"),
            "failure_taxonomy": outcome.get("failure_taxonomy"),
            "next_hypothesis": outcome.get("next_hypothesis"),
            "artifact_sha256": artifact_hashes,
            "core_hash_evidence_complete": all(
                artifact_hashes[name] is not None
                for name in (
                    "resolved_config",
                    "attempt_plan",
                    "attempt_outcome",
                    "manifest",
                    "manifest_chain",
                )
            ),
            "test_touched": False,
        },
    )
    return path


__all__ = [
    "AlignmentAudit",
    "EpochResult",
    "EvaluationResult",
    "FAILURE_TAXONOMY",
    "FIXED_SPLITS",
    "align_state_by_instrument",
    "append_campaign_ledger",
    "append_jsonl",
    "build_stage_optimizer",
    "build_stateful_flow_model",
    "campaign_ledger_path",
    "checkpoint_payload",
    "classify_failure",
    "evaluate_stateful_flow",
    "evaluate_success_gates",
    "model_kwargs_from_config",
    "negative_pearson_ic",
    "save_training_history",
    "set_stage_trainability",
    "sha256_file",
    "stateful_flow_loss",
    "train_full_epoch",
    "validate_stateful_flow_config",
    "write_attempt_plan",
    "write_failure_analysis",
    "write_manifest",
    "write_model_structure",
]
