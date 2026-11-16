#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import torch

from meta_fdg.data import build_temporal_qlib_source
from meta_fdg.training.flow_fdg import (
    build_flow_fdg_model,
    build_optimizer,
    checkpoint_payload,
    evaluate_flow_fdg,
    train_flow_epoch,
    validate_flow_fdg_config,
)
from meta_fdg.training.run_io import RunDirectory, atomic_json
from meta_fdg.training.stateful_flow import (
    append_campaign_ledger,
    append_jsonl,
    write_manifest,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lock_owner(handle: Any) -> str:
    handle.seek(0)
    return handle.read().strip() or "owner metadata unavailable"


def _write_lock_owner(handle: Any, *, slot: int | None, parallel_slots: int) -> None:
    handle.seek(0)
    handle.truncate()
    json.dump(
        {
            "pid": os.getpid(),
            "hostname": os.uname().nodename,
            "acquired_at_utc": _utc_now(),
            "argv": sys.argv,
            "slot": slot,
            "parallel_slots": parallel_slots,
        },
        handle,
        ensure_ascii=False,
        sort_keys=True,
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


@contextmanager
def _gpu_run_lock(repo_root: Path, *, parallel_slots: int = 1) -> Iterator[int]:
    """Acquire the default exclusive lock or one explicitly configured GPU slot.

    The default remains fail-closed and single-process.  A campaign may opt into
    a small fixed number of slots after profiling an exclusive server.  Slot
    holders take a shared gate lock, so a default exclusive run and a slotted
    campaign can never overlap on the same workspace.
    """

    if parallel_slots < 1 or parallel_slots > 8:
        raise ValueError("resource_limits.gpu_parallel_slots must be in [1, 8]")
    lock_root = repo_root / "runs"
    lock_root.mkdir(parents=True, exist_ok=True)
    gate_path = lock_root / ".flow_fdg_gpu.lock"
    gate = gate_path.open("a+", encoding="utf-8")
    gate_mode = fcntl.LOCK_EX if parallel_slots == 1 else fcntl.LOCK_SH
    slot_handle = None
    slot_index = 0
    try:
        try:
            fcntl.flock(gate.fileno(), gate_mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            owner = _lock_owner(gate)
            raise RuntimeError(
                f"another Flow-FDG campaign owns the GPU gate: {owner}"
            ) from error

        if parallel_slots == 1:
            _write_lock_owner(gate, slot=None, parallel_slots=1)
        else:
            owners: list[str] = []
            for candidate in range(parallel_slots):
                path = lock_root / f".flow_fdg_gpu.slot{candidate}.lock"
                handle = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    owners.append(_lock_owner(handle))
                    handle.close()
                    continue
                slot_handle = handle
                slot_index = candidate
                _write_lock_owner(
                    slot_handle,
                    slot=slot_index,
                    parallel_slots=parallel_slots,
                )
                break
            if slot_handle is None:
                raise RuntimeError(
                    "all configured Flow-FDG GPU slots are occupied: "
                    + " | ".join(owners)
                )
        yield slot_index
    finally:
        if slot_handle is not None:
            try:
                fcntl.flock(slot_handle.fileno(), fcntl.LOCK_UN)
            finally:
                slot_handle.close()
        try:
            fcntl.flock(gate.fileno(), fcntl.LOCK_UN)
        finally:
            gate.close()


@contextmanager
def _test_reporting_lock(repo_root: Path) -> Iterator[None]:
    """Serialize memory-heavy Test materialization across parallel workers."""

    lock_root = repo_root / "runs"
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_root / ".flow_fdg_test_reporting.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _write_lock_owner(handle, slot=None, parallel_slots=1)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one no-residual Flow-FDG and directly report Test."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--defer-test-report",
        action="store_true",
        help=(
            "Stop after the best-Valid checkpoint and Valid artifacts are "
            "frozen. Test can later be reported with "
            "report_flow_fdg_test_only.py without rerunning training."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _resolve_config(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    for key in ("provider_path", "cache_root", "materialized_cache_root"):
        if key not in resolved["data"]:
            continue
        value = Path(resolved["data"][key]).expanduser()
        if not value.is_absolute():
            value = repo_root / value
        resolved["data"][key] = str(value.resolve())
    model_path = Path(resolved["model_config"]).expanduser()
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    resolved["model_config"] = str(model_path.resolve())
    return resolved


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _safe_torch_load(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint root must be a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _archive_epoch_checkpoint(
    payload: Mapping[str, Any],
    *,
    run_root: Path,
    global_epoch: int,
    stage: str,
    stage_epoch: int,
    valid_full_ic: float,
    valid_local_ic: float | None,
    selection_metric: float | None,
    selection_eligible: bool,
    stage_best_after_epoch: bool,
    best_valid_after_epoch: bool,
) -> Path:
    relative_path = (
        Path("checkpoints")
        / "epochs"
        / f"epoch_{global_epoch:03d}__{stage}_{stage_epoch:03d}.pt"
    )
    checkpoint_path = run_root / relative_path
    _atomic_torch_save(payload, checkpoint_path)
    append_jsonl(
        run_root / "checkpoints" / "epoch_index.jsonl",
        {
            "schema": "flow_fdg_epoch_checkpoint_index_v1",
            "created_at_utc": _utc_now(),
            "path": str(relative_path),
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "global_epoch": global_epoch,
            "stage": stage,
            "stage_epoch": stage_epoch,
            "valid_full_IC": valid_full_ic,
            "valid_local_IC": valid_local_ic,
            "selection_metric": selection_metric,
            "selection_eligible": selection_eligible,
            "stage_best_after_epoch": stage_best_after_epoch,
            "best_valid_after_epoch": best_valid_after_epoch,
            "best_valid_selection_metric": (
                valid_full_ic if best_valid_after_epoch else None
            ),
            "selection_split": "valid",
            "test_touched": False,
        },
    )
    return checkpoint_path


def _initialize_tra_backbone(
    model: Any,
    train_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    initialization = train_config.get("backbone_initialization")
    if initialization is None:
        return None
    if not isinstance(initialization, Mapping):
        raise TypeError("train.backbone_initialization must be a mapping")
    if str(initialization.get("kind")) != "official_qlib_tra":
        raise ValueError("unsupported backbone initialization kind")
    checkpoint = Path(str(initialization["checkpoint"])).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_sha = _sha256(checkpoint)
    expected_sha = str(initialization["sha256"])
    if observed_sha != expected_sha:
        raise ValueError(
            f"TRA checkpoint SHA mismatch: expected {expected_sha}, observed {observed_sha}"
        )
    payload = _safe_torch_load(checkpoint, map_location="cpu")
    model_state = payload.get("model")
    tra_state = payload.get("tra")
    if not isinstance(model_state, Mapping) or not isinstance(tra_state, Mapping):
        raise TypeError("official TRA checkpoint must contain model and tra states")
    details = model.initialize_tra_core(model_state, tra_state)
    return {
        "kind": "official_qlib_tra",
        "checkpoint": str(checkpoint),
        "sha256": observed_sha,
        "trainable_after_initialization": bool(
            initialization.get("trainable_after_initialization", True)
        ),
        "details": details,
    }


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _local_semantic_contract(
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    data = config["data"]
    semantic_data_keys = (
        "context_start_time",
        "data_id",
        "handler",
        "handler_class",
        "handler_kwargs",
        "label",
        "market",
        "preprocessing_version",
        "region",
        "splits",
    )
    train = config["train"]
    local_stage = next(
        (
            dict(stage)
            for stage in train.get("stages", [])
            if str(stage.get("name")) == "local"
        ),
        None,
    )
    if local_stage is None:
        raise ValueError("local checkpoint transplant requires a local stage")
    train_keys = (
        "encoder_learning_rate",
        "full_epoch_required",
        "grad_clip",
        "optimizer",
        "sampling",
        "shuffle_dates",
        "tbptt_days",
        "validation_warmup_days",
        "weight_decay",
    )
    model_keys = (
        "encoder",
        "flow",
        "model_family",
        "output",
        "schema_version",
        "single_model",
    )
    return {
        "seed": int(config["seed"]),
        "data": {key: data.get(key) for key in semantic_data_keys},
        "dataset": config["dataset"],
        "loss": config["loss"],
        "train": {key: train.get(key) for key in train_keys},
        "local_stage": local_stage,
        "model": {key: model_config.get(key) for key in model_keys},
    }


def _local_optimizer_parameter_names(
    model: torch.nn.Module,
    train_config: Mapping[str, Any],
) -> list[str]:
    optimizer = build_optimizer(model, train_config, stage="local")
    names_by_identity = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    names = sorted(
        {
            names_by_identity[id(parameter)]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
    )
    del optimizer
    if not names:
        raise RuntimeError("local optimizer exposes no parameters")
    return names


def _rng_state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_rng_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_rng_state_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def _restore_transplanted_local_rng(rng_state: Mapping[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "local checkpoint transplant is GPU-only because target CPU RNG "
            "must remain at its capacity-specific post-initialization state"
        )
    target_cpu_before = torch.get_rng_state().clone()
    random.setstate(rng_state["python"])
    numpy_state = rng_state["numpy"]
    numpy_keys = numpy_state["keys"]
    if isinstance(numpy_keys, torch.Tensor):
        numpy_keys = numpy_keys.cpu().numpy()
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_keys, dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    cuda_states = list(rng_state["torch_cuda"])
    if len(cuda_states) != torch.cuda.device_count():
        raise ValueError(
            "source CUDA RNG device count does not match the target runtime"
        )
    torch.cuda.set_rng_state_all(cuda_states)
    if not torch.equal(torch.get_rng_state(), target_cpu_before):
        raise AssertionError("target CPU RNG changed during local transplant")
    return {
        "policy": {
            "python": "restore_source_local_completion",
            "numpy": "restore_source_local_completion",
            "torch_cuda": "restore_source_local_completion",
            "torch_cpu": "preserve_target_capacity_post_initialization",
        },
        "target_torch_cpu_sha256": hashlib.sha256(
            target_cpu_before.cpu().numpy().tobytes()
        ).hexdigest(),
        "source_torch_cpu_sha256": hashlib.sha256(
            rng_state["torch_cpu"].cpu().numpy().tobytes()
        ).hexdigest(),
        "cuda_device_count": len(cuda_states),
    }


def _initialize_local_stage_transplant(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    initialization = config["train"].get("local_stage_transplant")
    if initialization is None:
        return None
    if not isinstance(initialization, Mapping):
        raise TypeError("train.local_stage_transplant must be a mapping")
    if str(initialization.get("kind")) != "flow_fdg_local_stage_transplant_v1":
        raise ValueError("unsupported local_stage_transplant kind")
    if initialization.get("skip_stage") != "local":
        raise ValueError("local_stage_transplant must skip exactly local")
    best_path = Path(str(initialization["local_best_checkpoint"])).expanduser().resolve()
    completion_path = Path(
        str(initialization["local_completion_checkpoint"])
    ).expanduser().resolve()
    for path in (best_path, completion_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    best_sha = _sha256(best_path)
    completion_sha = _sha256(completion_path)
    if best_sha != str(initialization["local_best_sha256"]):
        raise ValueError("local_best checkpoint SHA mismatch")
    if completion_sha != str(initialization["local_completion_sha256"]):
        raise ValueError("local completion checkpoint SHA mismatch")
    best = _safe_torch_load(best_path, map_location="cpu")
    completion = _safe_torch_load(completion_path, map_location="cpu")
    if best.get("stage") != "local" or completion.get("stage") != "local":
        raise ValueError("transplant sources must both belong to the local stage")
    if int(best.get("epoch", -1)) != int(
        initialization["expected_local_best_global_epoch"]
    ):
        raise ValueError("unexpected source local-best epoch")
    if int(completion.get("epoch", -1)) != int(
        initialization["expected_local_completion_global_epoch"]
    ):
        raise ValueError("unexpected source local-completion epoch")
    source_config = best.get("resolved_config")
    source_model_config = best.get("model_config")
    if not isinstance(source_config, Mapping) or not isinstance(
        source_model_config, Mapping
    ):
        raise TypeError("source checkpoint lacks its resolved configuration")
    if str(source_config.get("job_id")) != str(initialization["source_job_id"]):
        raise ValueError("source local checkpoint job identity mismatch")
    source_contract = _local_semantic_contract(
        source_config, source_model_config
    )
    target_contract = _local_semantic_contract(config, model_config)
    source_contract_sha = _canonical_sha(source_contract)
    target_contract_sha = _canonical_sha(target_contract)
    expected_contract_sha = str(initialization["semantic_contract_sha256"])
    if (
        source_contract != target_contract
        or source_contract_sha != target_contract_sha
        or source_contract_sha != expected_contract_sha
    ):
        raise ValueError("source and target local-stage semantic contracts differ")
    local_names = _local_optimizer_parameter_names(model, config["train"])
    expected_names_sha = str(initialization["local_parameter_names_sha256"])
    observed_names_sha = _canonical_sha(local_names)
    if observed_names_sha != expected_names_sha:
        raise ValueError("local optimizer parameter allowlist drift")
    source_state = best.get("model_state")
    if not isinstance(source_state, Mapping):
        raise TypeError("source local-best checkpoint lacks model_state")
    target_state = model.state_dict()
    for name in local_names:
        if name not in source_state or name not in target_state:
            raise KeyError(f"missing transplanted local parameter: {name}")
        if source_state[name].shape != target_state[name].shape:
            raise ValueError(f"local parameter shape mismatch: {name}")
        target_state[name] = source_state[name].to(
            device=target_state[name].device,
            dtype=target_state[name].dtype,
        )
    model.load_state_dict(target_state, strict=True)
    rng_state = completion.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise TypeError("source local-completion checkpoint lacks rng_state")
    rng_report = _restore_transplanted_local_rng(rng_state)
    return {
        "schema": "flow_fdg_local_stage_transplant_report_v1",
        "created_at_utc": _utc_now(),
        "kind": "flow_fdg_local_stage_transplant_v1",
        "source_job_id": str(source_config["job_id"]),
        "source_local_best_checkpoint": str(best_path),
        "source_local_best_sha256": best_sha,
        "source_local_completion_checkpoint": str(completion_path),
        "source_local_completion_sha256": completion_sha,
        "source_local_best_global_epoch": int(best["epoch"]),
        "source_local_completion_global_epoch": int(completion["epoch"]),
        "semantic_contract_sha256": source_contract_sha,
        "loaded_parameter_names": local_names,
        "loaded_parameter_names_sha256": observed_names_sha,
        "loaded_parameter_count": len(local_names),
        "rng": rng_report,
        "skip_stage": "local",
        "test_touched": False,
    }


def _write_plan(
    run_root: Path, config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> None:
    benchmark_classification = str(
        config["test_report"].get(
            "classification", "test-tuned benchmark optimization"
        )
    )
    atomic_json(
        run_root / "attempt_plan.json",
        {
            "schema_version": "flow_fdg_attempt_plan_v1",
            "created_at_utc": _utc_now(),
            "attempt": dict(config["attempt"]),
            "fixed_split": dict(config["data"]["splits"]),
            "selection": "best Valid IC checkpoint",
            "test_policy": {
                "direct_report_every_candidate": True,
                "test_labels_in_gradient": False,
                "test_checkpoint_selection": False,
                "slow_state_updates": False,
                "classification": benchmark_classification,
            },
            "chronology": {
                "full_epoch": True,
                "shuffle_dates": False,
                "tbptt_days": int(config["train"]["tbptt_days"]),
                "stock_state": "identity bank with suspension decay/reset",
            },
            "stages": list(config["train"].get("stages", [])),
            "local_stage_transplant": config["train"].get(
                "local_stage_transplant"
            ),
            "model": dict(model_config),
            "loss": dict(config["loss"]),
            "success_gates": dict(config.get("success_gates", {})),
        },
    )


def _write_model_structure(run_root: Path, model: torch.nn.Module) -> None:
    groups = model.parameter_groups()
    atomic_json(
        run_root / "model_structure.json",
        {
            **model.structure_dict(),
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "trainable_parameter_count": sum(
                value.numel() for value in model.parameters() if value.requires_grad
            ),
            "parameter_groups": {
                name: {
                    "tensor_count": len(values),
                    "parameter_count": sum(value.numel() for value in values),
                }
                for name, values in groups.items()
            },
        },
    )


def _write_history(history: list[dict[str, Any]], run_root: Path) -> None:
    frame = pd.DataFrame(history)
    frame.to_parquet(run_root / "training_history.parquet", index=False)
    frame.to_csv(run_root / "logs" / "epochs.csv", index=False)


def _write_evaluation(
    run_root: Path, evaluation: Any, *, split: str
) -> None:
    evaluation.daily_full.to_parquet(
        run_root / f"metrics_{split}_full_daily.parquet", index=False
    )
    evaluation.predictions_full.to_parquet(
        run_root / f"predictions_{split}_full.parquet", index=False
    )
    if evaluation.summary_local.get("available", True):
        evaluation.daily_local.to_parquet(
            run_root / f"metrics_{split}_local_diagnostic_daily.parquet", index=False
        )
        evaluation.daily_flow.to_parquet(
            run_root / f"metrics_{split}_flow_diagnostic_daily.parquet", index=False
        )
        evaluation.predictions_local.to_parquet(
            run_root / f"predictions_{split}_local_diagnostic.parquet", index=False
        )
        evaluation.predictions_flow.to_parquet(
            run_root / f"predictions_{split}_flow_diagnostic.parquet", index=False
        )
    evaluation.diagnostics_daily.to_parquet(
        run_root / f"flow_diagnostics_{split}_daily.parquet", index=False
    )
    atomic_json(
        run_root / f"metrics_{split}_summary.json",
        {
            "split": split,
            "full": evaluation.summary_full,
            "local_diagnostic": evaluation.summary_local,
            "flow_diagnostic": evaluation.summary_flow,
            "primary_selection_metric": "IC" if split == "valid" else None,
            "rankic_reporting_only": True,
            "prediction_residual": False,
        },
    )


def _history_row(
    *,
    epoch: int,
    stage: str,
    stage_epoch: int,
    elapsed_seconds: float,
    train_result: Any,
    evaluation: Any,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "stage": str(stage),
        "stage_epoch": int(stage_epoch),
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": float(elapsed_seconds),
        **train_result.metrics,
        **{
            f"train_{name}": value
            for name, value in train_result.component_means.items()
        },
        "valid_full_IC": float(evaluation.summary_full["IC"]),
        "valid_full_ICIR": float(evaluation.summary_full["ICIR"]),
        "valid_full_RankIC": float(evaluation.summary_full["RankIC"]),
        **(
            {
                "valid_local_diagnostic_IC": float(evaluation.summary_local["IC"]),
                "valid_flow_diagnostic_IC": float(evaluation.summary_flow["IC"]),
            }
            if evaluation.summary_local.get("available", True)
            else {}
        ),
        "valid_days_seen": int(evaluation.summary_full["n_days"]),
        "valid_rows_seen": int(evaluation.summary_full["rows"]),
        "valid_coverage_complete": bool(evaluation.summary_full["coverage_complete"]),
    }


def _reporting_config(config: Mapping[str, Any]) -> dict[str, Any]:
    reporting = copy.deepcopy(dict(config))
    reporting["data"]["materialization_end_time"] = reporting["data"]["splits"][
        "test"
    ][1]
    reporting["data"]["qlib_kernels"] = 1
    return reporting


def _evaluate_test_report(
    *,
    repo_root: Path,
    run: RunDirectory,
    config: Mapping[str, Any],
    model: Any,
    expected_feature_dim: int,
    splits: Mapping[str, Any],
    data_status: dict[str, Any],
    device: torch.device,
    best_valid_epoch: int,
    best_valid_ic: float,
) -> Any:
    run.set_status(
        "waiting_test_report_lock",
        best_valid_epoch=int(best_valid_epoch),
        best_valid_IC=float(best_valid_ic),
        test_touched=False,
    )
    with _test_reporting_lock(repo_root):
        run.set_status(
            "reporting_test",
            best_valid_epoch=int(best_valid_epoch),
            best_valid_IC=float(best_valid_ic),
            test_touched=False,
        )
        reporting_source = build_temporal_qlib_source(
            _reporting_config(config),
            purpose="reporting",
            requested_splits=("test",),
        )
        test_split = reporting_source.materialize_split("test")
        if test_split.feature_dim != expected_feature_dim:
            raise ValueError(
                "reporting feature identity mismatch: "
                f"handler={config['data'].get('handler')} "
                f"expected={expected_feature_dim} "
                f"observed={test_split.feature_dim}"
            )
        data_status.update(
            {
                "test_touched": True,
                "test_materialized_at_utc": _utc_now(),
                "reporting_compact_materialization": getattr(
                    reporting_source.handler,
                    "materialization_metadata",
                    None,
                ),
                "test_split": test_split.coverage.to_dict(),
            }
        )
        atomic_json(run.root / "data_status.json", data_status)
        test_warmup_count = int(
            config["test_report"].get("valid_warmup_days", 20)
        )
        test_warmup = splits["valid"].days[-test_warmup_count:]
        final_test = evaluate_flow_fdg(
            model,
            test_split,
            warmup_days=test_warmup,
            device=device,
            loss_config=config["loss"],
            split_name="test",
        )
        _write_evaluation(run.root, final_test, split="test")
        return final_test


def _success_gates(
    config: Mapping[str, Any], valid: Any, test: Any
) -> tuple[bool, dict[str, Any]]:
    required = config.get("success_gates", {})
    gates = {
        "valid_ic": {
            "observed": float(valid.summary_full["IC"]),
            "required": float(required.get("valid_ic_min", -math.inf)),
        },
        "test_ic": {
            "observed": float(test.summary_full["IC"]),
            "required": float(required.get("test_ic_min", -math.inf)),
        },
        "test_coverage": {
            "observed": bool(test.summary_full["coverage_complete"]),
            "required": True,
        },
    }
    for value in gates.values():
        value["passed"] = bool(value["observed"] >= value["required"])
    return all(bool(value["passed"]) for value in gates.values()), gates


def run_attempt(
    repo_root: Path,
    config: dict[str, Any],
    *,
    defer_test_report: bool = False,
) -> int:
    run = RunDirectory.create(repo_root, config, allow_existing=False)
    history: list[dict[str, Any]] = []
    try:
        run.set_status("validating", test_touched=False)
        validate_flow_fdg_config(config)
        model_config = _read_json(Path(config["model_config"]))
        if model_config["output"].get("residual_readout") is not False:
            raise ValueError("final prediction residual must be deleted")
        _write_plan(run.root, config, model_config)

        cpu_threads = int(config.get("resource_limits", {}).get("cpu_threads", 0))
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
        if torch.cuda.is_available():
            fraction = float(
                config.get("resource_limits", {}).get("gpu_memory_fraction", 1.0)
            )
            if 0.0 < fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(fraction)
        requested = str(config.get("device", "cuda:0"))
        device = torch.device(
            requested
            if requested.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        _set_seed(int(config["seed"]))

        run.set_status("materializing_train_valid", test_touched=False)
        development_source = build_temporal_qlib_source(
            config, purpose="development", requested_splits=("train", "valid")
        )
        splits = development_source.materialize_all()
        if tuple(splits) != ("train", "valid"):
            raise AssertionError("development materialized a forbidden split")
        if splits["train"].feature_dim != splits["valid"].feature_dim:
            raise ValueError("train/Valid feature dimensions differ")
        expected_feature_dim = development_source.expected_feature_dim
        if expected_feature_dim is None or (
            splits["train"].feature_dim != expected_feature_dim
        ):
            raise ValueError(
                "development feature identity mismatch: "
                f"handler={config['data'].get('handler')} "
                f"expected={expected_feature_dim} "
                f"observed={splits['train'].feature_dim}"
            )
        if splits["train"].lookback != int(model_config["encoder"]["lookback"]):
            raise ValueError("dataset/model lookback mismatch")
        data_status = {
            "schema_version": "flow_fdg_data_status_v2",
            "captured_at_utc": _utc_now(),
            "development_permission": ["train", "valid"],
            "test_touched": False,
            "handler": config["data"].get("handler"),
            "expected_feature_dim": expected_feature_dim,
            "feature_dim": splits["train"].feature_dim,
            "feature_identity_verified": True,
            "lookback": splits["train"].lookback,
            "compact_materialization": getattr(
                development_source.handler,
                "materialization_metadata",
                None,
            ),
            "splits": {
                name: split.coverage.to_dict() for name, split in splits.items()
            },
        }
        atomic_json(run.root / "data_status.json", data_status)

        model = build_flow_fdg_model(
            splits["train"].feature_dim, model_config, device
        )
        backbone_initialization = _initialize_tra_backbone(
            model,
            config["train"],
        )
        local_stage_transplant = _initialize_local_stage_transplant(
            model,
            config,
            model_config,
        )
        if (
            backbone_initialization is not None
            and local_stage_transplant is not None
        ):
            raise ValueError(
                "TRA backbone initialization and local-stage transplant "
                "cannot be combined"
            )
        if backbone_initialization is not None:
            atomic_json(
                run.root / "backbone_initialization.json",
                backbone_initialization,
            )
        if local_stage_transplant is not None:
            atomic_json(
                run.root / "local_stage_transplant.json",
                local_stage_transplant,
            )
            data_status["local_stage_transplant"] = {
                "source_job_id": local_stage_transplant["source_job_id"],
                "semantic_contract_sha256": local_stage_transplant[
                    "semantic_contract_sha256"
                ],
                "test_touched": False,
            }
            atomic_json(run.root / "data_status.json", data_status)
        _write_model_structure(run.root, model)
        train_log = run.root / "logs" / "train.jsonl"
        append_jsonl(
            train_log,
            {
                "event": "training_started",
                "at_utc": _utc_now(),
                "device": str(device),
                "parameter_count": sum(value.numel() for value in model.parameters()),
                "train_days": splits["train"].coverage.emitted_days,
                "valid_days": splits["valid"].coverage.emitted_days,
                "prediction_residual": False,
                "test_touched": False,
                "backbone_initialization": backbone_initialization,
                "local_stage_transplant": local_stage_transplant,
            },
        )
        run.set_status("running", epoch=0, test_touched=False)
        train_config = config["train"]
        save_every_epoch = train_config.get("save_every_epoch", False)
        if not isinstance(save_every_epoch, bool):
            raise TypeError("train.save_every_epoch must be a boolean")
        allow_local_full_selection = train_config.get(
            "allow_local_full_checkpoint_selection",
            False,
        )
        if not isinstance(allow_local_full_selection, bool):
            raise TypeError(
                "train.allow_local_full_checkpoint_selection must be boolean"
            )
        if allow_local_full_selection and (
            backbone_initialization is None
            or model.readout_mode != "strict_norm_bounded_shared"
            or not isinstance(
                backbone_initialization.get("details", {}).get(
                    "strict_source_anchor"
                ),
                Mapping,
            )
        ):
            raise ValueError(
                "local full checkpoint selection requires a verified strict "
                "TRA source anchor"
            )
        warmup_count = int(train_config.get("validation_warmup_days", 20))
        valid_warmup = splits["train"].days[-warmup_count:]
        best_valid_ic = -math.inf
        best_path = run.root / "checkpoints" / "best_valid.pt"
        configured_stages = train_config.get("stages")
        if configured_stages is None:
            stage_configs = [
                {
                    "name": "joint",
                    "epochs": int(train_config["epochs"]),
                    "early_stop_patience": int(
                        train_config.get("early_stop_patience", 0)
                    ),
                    "minimum_epochs_before_stop": int(
                        train_config.get("minimum_epochs_before_stop", 0)
                    ),
                }
            ]
        else:
            stage_configs = [dict(value) for value in configured_stages]
        if local_stage_transplant is not None:
            stage_configs = [
                stage
                for stage in stage_configs
                if str(stage["name"])
                != str(local_stage_transplant["skip_stage"])
            ]
            if not stage_configs or any(
                str(stage["name"]) == "local" for stage in stage_configs
            ):
                raise ValueError(
                    "local-stage transplant did not remove exactly the local stage"
                )
            append_jsonl(
                train_log,
                {
                    "event": "stage_transplanted",
                    "at_utc": _utc_now(),
                    "stage": "local",
                    "source_job_id": local_stage_transplant["source_job_id"],
                    "source_local_best_global_epoch": local_stage_transplant[
                        "source_local_best_global_epoch"
                    ],
                    "source_local_completion_global_epoch": (
                        local_stage_transplant[
                            "source_local_completion_global_epoch"
                        ]
                    ),
                    "loaded_parameter_names_sha256": local_stage_transplant[
                        "loaded_parameter_names_sha256"
                    ],
                    "semantic_contract_sha256": local_stage_transplant[
                        "semantic_contract_sha256"
                    ],
                    "test_touched": False,
                },
            )
        global_epoch = 0

        for stage_config in stage_configs:
            stage = str(stage_config["name"])
            optimizer = build_optimizer(model, train_config, stage=stage)
            selection_variant = "local" if stage == "local" else "full"
            stage_best_metric = -math.inf
            stage_best_path = run.root / "checkpoints" / f"{stage}_best.pt"
            no_improvement = 0
            append_jsonl(
                train_log,
                {
                    "event": "stage_started",
                    "at_utc": _utc_now(),
                    "stage": stage,
                    "requested_epochs": int(stage_config["epochs"]),
                    "selection_variant": selection_variant,
                    "optimizer_groups": [
                        {
                            "name": group.get("name"),
                            "lr": group["lr"],
                            "parameter_tensors": len(group["params"]),
                        }
                        for group in optimizer.param_groups
                    ],
                },
            )
            for stage_epoch in range(1, int(stage_config["epochs"]) + 1):
                global_epoch += 1
                started = time.monotonic()
                uniform_router_epochs = int(
                    train_config.get("pattern_router_uniform_epochs", 0)
                )
                model.set_pattern_router_uniform(
                    global_epoch <= uniform_router_epochs
                )
                temporal_only_epochs = int(
                    train_config.get("coupled_temporal_only_epochs", 0)
                )
                model.set_coupled_temporal_only(
                    (model.fully_coupled and stage == "local")
                    or global_epoch <= temporal_only_epochs
                )
                model.set_detach_lagged_flow_roles(
                    model.fully_coupled and stage == "flow_warm"
                )
                if (
                    temporal_only_epochs > 0
                    and global_epoch == temporal_only_epochs + 1
                ):
                    changed_rates: dict[str, float] = {}
                    for group in optimizer.param_groups:
                        name = str(group.get("name"))
                        key = f"post_warmup_{name}_learning_rate"
                        if key not in train_config:
                            continue
                        rate = float(train_config[key])
                        if rate <= 0.0:
                            raise ValueError(f"train.{key} must be positive")
                        group["lr"] = rate
                        changed_rates[name] = rate
                    append_jsonl(
                        train_log,
                        {
                            "event": "coupled_temporal_warmup_completed",
                            "at_utc": _utc_now(),
                            "global_epoch": global_epoch,
                            "warmup_epochs": temporal_only_epochs,
                            "post_warmup_learning_rates": changed_rates,
                            "single_prediction_head_preserved": True,
                        },
                    )
                train_result = train_flow_epoch(
                    model,
                    splits["train"],
                    optimizer,
                    device=device,
                    loss_config=config["loss"],
                    tbptt_days=int(train_config["tbptt_days"]),
                    grad_clip=float(train_config["grad_clip"]),
                    stage=stage,
                )
                valid_evaluation = evaluate_flow_fdg(
                    model,
                    splits["valid"],
                    warmup_days=valid_warmup,
                    device=device,
                    loss_config=config["loss"],
                    split_name="valid",
                )
                row = _history_row(
                    epoch=global_epoch,
                    stage=stage,
                    stage_epoch=stage_epoch,
                    elapsed_seconds=time.monotonic() - started,
                    train_result=train_result,
                    evaluation=valid_evaluation,
                )
                row["coupled_temporal_only"] = model.force_coupled_temporal_only
                row["lagged_flow_roles_detached"] = (
                    model.detach_lagged_flow_roles
                )
                history.append(row)
                _write_history(history, run.root)
                append_jsonl(train_log, {"event": "epoch_complete", **row})
                run.set_status(
                    "running",
                    epoch=global_epoch,
                    stage=stage,
                    stage_epoch=stage_epoch,
                    valid_full_IC=row["valid_full_IC"],
                    valid_local_IC=row.get("valid_local_diagnostic_IC"),
                    train_days_seen=row["train_days_seen"],
                    coupled_temporal_only=model.force_coupled_temporal_only,
                    test_touched=False,
                )
                payload = checkpoint_payload(
                    model,
                    config=config,
                    model_config=model_config,
                    feature_dim=splits["train"].feature_dim,
                    lookback=splits["train"].lookback,
                    epoch=global_epoch,
                    valid_summary=valid_evaluation.summary_full,
                    optimizer=optimizer,
                )
                payload.update(
                    {
                        "stage": stage,
                        "stage_epoch": stage_epoch,
                        "coupled_temporal_only": model.force_coupled_temporal_only,
                        "valid_local_summary": dict(
                            valid_evaluation.summary_local
                        ),
                    }
                )
                selection_eligible = (
                    not model.force_coupled_temporal_only
                    or (model.fully_coupled and stage == "local")
                )
                stage_metric: float | None = None
                stage_best_after_epoch = False
                best_valid_after_epoch = False
                if selection_eligible:
                    stage_metric = float(
                        valid_evaluation.summary_local["IC"]
                        if stage == "local"
                        and valid_evaluation.summary_local.get("available", True)
                        else valid_evaluation.summary_full["IC"]
                    )
                    stage_best_after_epoch = (
                        stage_metric > stage_best_metric + 1e-12
                    )
                    if stage_best_after_epoch:
                        stage_best_metric = stage_metric
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    valid_ic = float(valid_evaluation.summary_full["IC"])
                    best_valid_after_epoch = (
                        (
                            stage != "local"
                            or allow_local_full_selection
                        )
                        and valid_ic > best_valid_ic + 1e-12
                    )
                    if best_valid_after_epoch:
                        best_valid_ic = valid_ic
                else:
                    valid_ic = float(valid_evaluation.summary_full["IC"])
                payload.update(
                    {
                        "rng_state": _capture_rng_state(),
                        "resume_state": {
                            "schema": "flow_fdg_epoch_resume_state_v1",
                            "global_epoch": global_epoch,
                            "stage": stage,
                            "stage_epoch": stage_epoch,
                            "stage_best_metric": stage_best_metric,
                            "best_valid_ic": best_valid_ic,
                            "no_improvement": no_improvement,
                            "selection_metric": stage_metric,
                            "selection_eligible": selection_eligible,
                            "coupled_temporal_only": (
                                model.force_coupled_temporal_only
                            ),
                            "lagged_flow_roles_detached": (
                                model.detach_lagged_flow_roles
                            ),
                            "test_touched": False,
                        },
                    }
                )
                if stage_best_after_epoch:
                    _atomic_torch_save(payload, stage_best_path)
                if best_valid_after_epoch:
                        _atomic_torch_save(payload, best_path)
                _atomic_torch_save(
                    payload, run.root / "checkpoints" / "last_complete.pt"
                )
                if save_every_epoch:
                    valid_local_value = valid_evaluation.summary_local.get("IC")
                    _archive_epoch_checkpoint(
                        payload,
                        run_root=run.root,
                        global_epoch=global_epoch,
                        stage=stage,
                        stage_epoch=stage_epoch,
                        valid_full_ic=valid_ic,
                        valid_local_ic=(
                            float(valid_local_value)
                            if valid_local_value is not None
                            else None
                        ),
                        selection_metric=stage_metric,
                        selection_eligible=selection_eligible,
                        stage_best_after_epoch=stage_best_after_epoch,
                        best_valid_after_epoch=best_valid_after_epoch,
                    )
                patience = int(stage_config.get("early_stop_patience", 0))
                minimum = int(
                    stage_config.get("minimum_epochs_before_stop", 0)
                )
                if (
                    selection_eligible
                    and patience > 0
                    and stage_epoch >= minimum
                    and no_improvement >= patience
                ):
                    append_jsonl(
                        train_log,
                        {
                            "event": "early_stop",
                            "at_utc": _utc_now(),
                            "stage": stage,
                            "stage_epoch": stage_epoch,
                            "global_epoch": global_epoch,
                            "patience": patience,
                        },
                    )
                    break
            model.set_pattern_router_uniform(False)
            model.set_coupled_temporal_only(False)
            model.set_detach_lagged_flow_roles(False)
            if not stage_best_path.is_file():
                raise RuntimeError(f"stage {stage} produced no finite checkpoint")
            stage_best = _safe_torch_load(stage_best_path, map_location=device)
            model.load_state_dict(stage_best["model_state"])
            append_jsonl(
                train_log,
                {
                    "event": "stage_completed",
                    "at_utc": _utc_now(),
                    "stage": stage,
                    "best_selection_metric": stage_best_metric,
                    "selection_variant": selection_variant,
                    "restored_stage_best": True,
                },
            )

        if not best_path.is_file():
            raise RuntimeError("training produced no finite Valid checkpoint")
        best = _safe_torch_load(best_path, map_location=device)
        model.load_state_dict(best["model_state"])
        final_valid = evaluate_flow_fdg(
            model,
            splits["valid"],
            warmup_days=valid_warmup,
            device=device,
            loss_config=config["loss"],
            split_name="valid",
        )
        _write_evaluation(run.root, final_valid, split="valid")

        if defer_test_report:
            valid_only_outcome = {
                "schema": "flow_fdg_valid_only_outcome_v1",
                "completed_at_utc": _utc_now(),
                "status": "valid_complete_test_deferred",
                "epochs_completed": len(history),
                "best_valid_epoch": int(best["epoch"]),
                "best_valid_full_ic": float(final_valid.summary_full["IC"]),
                "valid_full": final_valid.summary_full,
                "valid_local": final_valid.summary_local,
                "valid_flow": final_valid.summary_flow,
                "checkpoint": str(best_path.resolve()),
                "selection_split": "valid",
                "test_used_for_gradient": False,
                "test_used_for_checkpoint_selection": False,
                "test_touched": False,
                "test_report_deferred": True,
                "allow_local_full_checkpoint_selection": (
                    allow_local_full_selection
                ),
                "prediction_residual": False,
            }
            atomic_json(
                run.root / "valid_only_outcome.json",
                valid_only_outcome,
            )
            run.set_status(
                "valid_complete_test_deferred",
                epochs_completed=len(history),
                best_valid_epoch=int(best["epoch"]),
                final_valid_full_IC=float(final_valid.summary_full["IC"]),
                test_touched=False,
                test_report_deferred=True,
            )
            atomic_json(
                run.root / "status_valid_only.json",
                _read_json(run.root / "status.json"),
            )
            write_manifest(run.root)
            print(
                json.dumps(
                    valid_only_outcome,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0

        # Test is opened only after the best-Valid checkpoint is fixed. The
        # shared lock prevents concurrent Test materialization from exceeding
        # the campaign's cgroup memory headroom.
        final_test = _evaluate_test_report(
            repo_root=repo_root,
            run=run,
            config=config,
            model=model,
            expected_feature_dim=expected_feature_dim,
            splits=splits,
            data_status=data_status,
            device=device,
            best_valid_epoch=int(best["epoch"]),
            best_valid_ic=float(final_valid.summary_full["IC"]),
        )
        gates_passed, gates = _success_gates(config, final_valid, final_test)
        outcome = {
            "schema_version": "flow_fdg_attempt_outcome_v1",
            "completed_at_utc": _utc_now(),
            "status": "passed" if gates_passed else "completed_below_gates",
            "gates_passed": gates_passed,
            "gates": gates,
            "epochs_completed": len(history),
            "best_valid_epoch": int(best["epoch"]),
            "best_valid_full_ic": float(final_valid.summary_full["IC"]),
            "valid_full": final_valid.summary_full,
            "valid_local": final_valid.summary_local,
            "valid_flow": final_valid.summary_flow,
            "test_full": final_test.summary_full,
            "test_local": final_test.summary_local,
            "test_flow": final_test.summary_flow,
            "checkpoint": str(best_path.resolve()),
            "selection_split": "valid",
            "test_used_for_gradient": False,
            "test_used_for_checkpoint_selection": False,
            "test_touched": True,
            "prediction_residual": False,
            "local_stage_transplant": local_stage_transplant,
            "trained_epochs_completed": len(history),
            "inherited_local_epochs": (
                int(
                    local_stage_transplant[
                        "source_local_completion_global_epoch"
                    ]
                )
                if local_stage_transplant is not None
                else 0
            ),
            "benchmark_classification": str(
                config["test_report"].get(
                    "classification", "test-tuned benchmark optimization"
                )
            ),
        }
        atomic_json(run.root / "attempt_outcome.json", outcome)
        run.set_status(
            "done" if gates_passed else "completed_below_gates",
            epochs_completed=len(history),
            best_valid_epoch=int(best["epoch"]),
            final_valid_full_IC=float(final_valid.summary_full["IC"]),
            final_test_full_IC=float(final_test.summary_full["IC"]),
            gates_passed=gates_passed,
            test_touched=True,
        )
        write_manifest(run.root)
        append_campaign_ledger(repo_root, config, run.root, outcome)
        print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))
        return 0
    except BaseException as error:
        if history:
            _write_history(history, run.root)
        outcome = {
            "schema_version": "flow_fdg_attempt_outcome_v1",
            "completed_at_utc": _utc_now(),
            "status": "crashed",
            "exception_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "epochs_completed": len(history),
        }
        atomic_json(run.root / "attempt_outcome.json", outcome)
        run.fail(error)
        write_manifest(run.root)
        append_campaign_ledger(repo_root, config, run.root, outcome)
        raise


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = _resolve_config(_read_json(args.config.resolve()), repo_root)
    parallel_slots = int(
        config.get("resource_limits", {}).get("gpu_parallel_slots", 1)
    )
    with _gpu_run_lock(repo_root, parallel_slots=parallel_slots):
        return run_attempt(
            repo_root,
            config,
            defer_test_report=bool(args.defer_test_report),
        )


if __name__ == "__main__":
    raise SystemExit(main())
