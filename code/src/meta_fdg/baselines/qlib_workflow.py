from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Any, Iterator, TextIO

import numpy as np
import pandas as pd
import qlib
import torch
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

from meta_fdg.data.compact_qlib import (
    CompactQlibHandler,
    compact_handler_from_config,
    expected_feature_dim,
)
from meta_fdg.training.metrics import daily_signal_metrics, summarize_daily_metrics
from meta_fdg.training.run_io import RunDirectory, atomic_json


QLIB_BASELINE_CACHE_SCHEMA = "meta_fdg_qlib_baseline_handler_v2"
RUN_ROOT_PLACEHOLDER = "__RUN_ROOT__"
PAPER_LOCK_ID = "paper-fixed-split-v1-20260716"
PAPER_SPLITS = {
    "train": ["2016-01-01", "2020-12-31"],
    "valid": ["2021-01-01", "2021-12-31"],
    "test": ["2022-01-01", "2025-12-31"],
}
US_EXTENSION_TIER = "us_headline"
US_PROTOCOL_ID = "us-sp500-pit-alpha158-ohlcv-v1"
US_LOCK_ID = "us-sp500-pit-alpha158-ohlcv-v1-20260724"
US_METHODS = {
    "lightgbm",
    "lstm",
    "hist",
    "master",
    "tra",
    "doubleadapt",
}
US_FACTOR_UNIVERSE = "alpha158_minus_vwap0_fixed_157"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _source_file_record(path: str | Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source), "sha256": digest.hexdigest()}


def _method_native_asset_records(
    model: Any, config: dict[str, Any]
) -> dict[str, dict[str, str] | None]:
    """Fingerprint method-native files using the paths held by the live model."""
    if str(config.get("method", "")).lower() != "hist":
        return {}
    return {
        "concept_matrix": _source_file_record(getattr(model, "stock2concept", None)),
        "stock_index": _source_file_record(getattr(model, "stock_index", None)),
        "pretrained_lstm_checkpoint": _source_file_record(
            getattr(model, "pretrained_model_path", None)
        ),
    }


def _runtime_provenance(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Record what was actually imported, not only declared provenance."""
    model_type = type(model)
    try:
        model_source = inspect.getsourcefile(model_type)
    except (OSError, TypeError):
        model_source = None
    provenance = {
        "packages": {
            "pyqlib": _distribution_version("pyqlib"),
            "torch": _distribution_version("torch"),
            "numpy": _distribution_version("numpy"),
            "pandas": _distribution_version("pandas"),
            "lightgbm": _distribution_version("lightgbm"),
        },
        "qlib_import": {
            "version": getattr(qlib, "__version__", None),
            "package_file": _source_file_record(getattr(qlib, "__file__", None)),
        },
        "model_import": {
            "class": model_type.__qualname__,
            "module": model_type.__module__,
            "source_file": _source_file_record(model_source),
        },
        "declared_implementation": copy.deepcopy(config.get("implementation", {})),
    }
    method_native_assets = _method_native_asset_records(model, config)
    if method_native_assets:
        provenance["method_native_assets"] = method_native_assets
    return provenance


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            try:
                stream.write(value)
                stream.flush()
            except ValueError:
                # Qlib retains logging handlers until interpreter shutdown.
                # The per-run file can already be closed by that point.
                continue
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            try:
                stream.flush()
            except ValueError:
                continue


@contextmanager
def _capture_process_log(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        tee_out = _Tee(sys.stdout, handle)
        tee_err = _Tee(sys.stderr, handle)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            yield


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_resource_limits(config: dict[str, Any]) -> dict[str, Any]:
    limits = config.get("resource_limits")
    if not isinstance(limits, dict):
        raise ValueError("paper baseline config requires resource_limits")
    cpu_threads = int(limits.get("cpu_threads", 0))
    if cpu_threads < 1 or cpu_threads > 10:
        raise ValueError("paper baseline CPU limit must be within 1..10 threads")
    torch.set_num_threads(cpu_threads)
    fraction = float(limits.get("gpu_memory_fraction", 0.0))
    if fraction < 0.0 or fraction > 0.5:
        raise ValueError("paper baseline GPU memory fraction must be within 0..0.5")
    applied_fraction = fraction
    override_id = os.environ.get(
        "META_FDG_RUNTIME_GPU_MEMORY_OVERRIDE_ID"
    )
    override_value = os.environ.get(
        "META_FDG_RUNTIME_GPU_MEMORY_FRACTION"
    )
    override_applied = False
    if override_id is not None or override_value is not None:
        if (
            override_id
            != "csi800-alpha360-doubleadapt-single-run-v1"
            or override_value is None
        ):
            raise ValueError("invalid runtime GPU memory override contract")
        is_csi800_doubleadapt = (
            config.get("tier") == "rolling_development"
            and config.get("lock_id")
            == "csi800-baseline-fixed-transfer-paper4-20260728-v1"
            and config.get("method") == "doubleadapt"
            and config.get("data", {}).get("market") == "csi800"
            and int(config.get("seed", -1)) in {0, 2, 5, 9}
        )
        if not is_csi800_doubleadapt:
            raise ValueError(
                "runtime GPU memory override escaped the CSI800 "
                "DoubleAdapt campaign"
            )
        if config.get("data", {}).get("handler") == "Alpha360":
            applied_fraction = float(override_value)
            if applied_fraction != 0.9 or fraction != 0.5:
                raise ValueError(
                    "CSI800 Alpha360 DoubleAdapt recovery requires the "
                    "locked 0.50 config and exact 0.90 runtime ceiling"
                )
            override_applied = True
            print(
                "runtime_gpu_memory_override="
                f"{fraction:.2f}->{applied_fraction:.2f} "
                f"contract={override_id}",
                flush=True,
            )
        elif config.get("data", {}).get("handler") != "Alpha158":
            raise ValueError(
                "runtime GPU memory override found an unexpected handler"
            )
    if applied_fraction > 0.0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            applied_fraction, device=0
        )
    return {
        "configured_gpu_memory_fraction": fraction,
        "applied_gpu_memory_fraction": applied_fraction,
        "gpu_memory_override_id": (
            override_id if override_applied else None
        ),
        "gpu_memory_override_applied": override_applied,
        "cuda_allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "statistical_configuration_changed": False,
    }


def _resolve_runtime_placeholders(value: Any, run_root: Path) -> Any:
    """Resolve paths that cannot be known when source configs are generated."""
    if isinstance(value, dict):
        return {
            key: _resolve_runtime_placeholders(item, run_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_runtime_placeholders(item, run_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_runtime_placeholders(item, run_root) for item in value)
    if isinstance(value, str):
        return value.replace(RUN_ROOT_PLACEHOLDER, str(run_root.resolve()))
    return value


def _handler_identity(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    return {
        "schema": QLIB_BASELINE_CACHE_SCHEMA,
        "data_id": data["data_id"],
        "handler": data["handler"],
        "handler_class": data.get("handler_class", data["handler"]),
        "handler_module_path": data.get(
            "handler_module_path", "qlib.contrib.data.handler"
        ),
        "market": data["market"],
        "label": data["label"],
        "splits": data["splits"],
        "context_start_time": data.get("context_start_time"),
        "materialization_end_time": data.get("materialization_end_time"),
        "label_maturity": data.get("label_maturity"),
        "preprocessing_version": data["preprocessing_version"],
        "handler_kwargs": data.get("handler_kwargs", {}),
        "code_revision": config["implementation"]["runner_revision"],
    }


def _handler_config(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    splits = data["splits"]
    kwargs = {
        "instruments": data["market"],
        "start_time": data.get("context_start_time", splits["train"][0]),
        "end_time": data.get(
            "materialization_end_time",
            max(str(period[1]) for period in splits.values()),
        ),
        "fit_start_time": splits["train"][0],
        "fit_end_time": splits["train"][1],
        "label": ([data["label"]], ["LABEL0"]),
        **data.get("handler_kwargs", {}),
    }
    return {
        "class": data.get("handler_class", data["handler"]),
        "module_path": data.get(
            "handler_module_path", "qlib.contrib.data.handler"
        ),
        "kwargs": kwargs,
    }


def _validate_execution_contract(config: dict[str, Any]) -> None:
    evaluation_splits = list(config.get("evaluation_splits", ["valid", "test"]))
    splits = config["data"]["splits"]
    unknown = set(evaluation_splits) - set(splits)
    if unknown:
        raise ValueError(f"evaluation_splits are missing from data.splits: {sorted(unknown)}")
    tier = config.get("tier")
    if tier == US_EXTENSION_TIER:
        data = config["data"]
        if config.get("protocol_id") != US_PROTOCOL_ID:
            raise ValueError("US baseline has the wrong protocol id")
        if config.get("lock_id") != US_LOCK_ID:
            raise ValueError("US baseline has the wrong extension lock id")
        if splits != PAPER_SPLITS:
            raise ValueError("US baseline has the wrong fixed split")
        if (
            data.get("handler") != "Alpha158USOHLCV"
            or data.get("handler_class") != "Alpha158USOHLCV"
            or data.get("handler_module_path")
            != "meta_fdg.data.us_ohlcv_handler"
            or data.get("market") != "sp500_pit"
            or data.get("region") != "us"
        ):
            raise ValueError(
                "US baseline must use the audited Alpha158USOHLCV/sp500_pit "
                "data contract"
            )
        if evaluation_splits != ["valid", "test"]:
            raise ValueError("US baseline must evaluate valid and test")
        if data.get("materialization_end_time") != "2025-12-31":
            raise ValueError("US baseline must materialize through fixed test end")
        if data.get("context_start_time") != "2015-07-01":
            raise ValueError("US baseline requires the locked causal context")
        if data.get("use_compact_materialization") is not True:
            raise ValueError("US baseline must consume the audited compact cache")
        if not Path(str(data.get("materialized_cache_root", ""))).is_absolute():
            raise ValueError("US compact cache root must be absolute at execution")
        method = str(config.get("method", "")).lower()
        if method not in US_METHODS:
            raise ValueError(f"unsupported US baseline method: {method}")
        policy = config.get("missing_data_policy", {})
        if policy.get("factor_universe") != US_FACTOR_UNIVERSE:
            raise ValueError("US baseline factor universe is not locked to 157")
        if policy.get("vwap") != "VWAP0_deleted_without_proxy":
            raise ValueError("US baseline must delete VWAP0 without a proxy")
        if policy.get("pit_price_gaps") != "exclude_uncovered_rows_without_proxy":
            raise ValueError("US baseline PIT price-gap policy is not locked")
        limits = config.get("resource_limits", {})
        expected_cpu = 10 if method == "lightgbm" else 4
        if int(limits.get("cpu_threads", 0)) != expected_cpu:
            raise ValueError("US baseline CPU budget differs from A-share parity")
        expected_fraction = 0.0 if method == "lightgbm" else 0.5
        if float(limits.get("gpu_memory_fraction", -1.0)) != expected_fraction:
            raise ValueError(
                "US baseline GPU memory budget differs from A-share parity"
            )
        if int(limits.get("memory_gib", 0)) != 45:
            raise ValueError("US baseline memory budget must be 45 GiB")
        if method == "tra":
            width = (
                config.get("model", {})
                .get("kwargs", {})
                .get("model_config", {})
                .get("input_size")
            )
            if int(width or 0) != 157:
                raise ValueError("US TRA input width must be 157")
        if method == "master":
            kwargs = config.get("model", {}).get("kwargs", {})
            if int(kwargs.get("d_feat", 0)) != 157:
                raise ValueError("US MASTER input width must be 157")
            if kwargs.get("alpha_only") is not True:
                raise ValueError("US MASTER must use the audited Alpha-only gate")
        return
    if tier == "headline":
        if config.get("lock_id") != PAPER_LOCK_ID:
            raise ValueError("headline baseline has the wrong paper lock id")
        if splits != PAPER_SPLITS:
            raise ValueError("headline baseline has the wrong fixed split")
        if config["data"].get("handler") != "Alpha158" or config["data"].get(
            "market"
        ) != "csi300":
            raise ValueError("headline baseline must use Alpha158/csi300")
        if evaluation_splits != ["valid", "test"]:
            raise ValueError("headline baseline must evaluate valid and test")
        if config["data"].get("materialization_end_time") != "2025-12-31":
            raise ValueError("headline baseline must materialize through fixed test end")
        method = str(config.get("method", "")).lower()
        limits = config.get("resource_limits", {})
        if int(limits.get("memory_gib", 0)) != 45:
            raise ValueError("headline baseline memory budget must be 45 GiB")
        expected_cpu = 10 if method == "lightgbm" else 4
        if int(limits.get("cpu_threads", 0)) != expected_cpu:
            raise ValueError("headline baseline CPU budget differs from its method class")
        expected_fraction = 0.0 if method == "lightgbm" else 0.5
        if float(limits.get("gpu_memory_fraction", -1.0)) != expected_fraction:
            raise ValueError("headline baseline GPU memory budget differs from its method class")
        if method != "lightgbm" and config["data"].get("context_start_time") != (
            "2015-07-01"
        ):
            raise ValueError("sequence baseline requires pre-train causal context")
        return
    if tier == "test":
        if evaluation_splits != ["test"]:
            raise ValueError("test tier must evaluate only ['test']")
        if set(splits) != {"train", "valid", "test"}:
            raise ValueError("test tier requires train, valid, and test splits")
        final_test = config.get("final_test")
        if not isinstance(final_test, dict):
            raise ValueError("test tier requires final_test lock metadata")
        if final_test.get("allows_reselection") is not False:
            raise ValueError("test tier must forbid reselection")
        if final_test.get("reporting_splits") != ["test"]:
            raise ValueError("test tier reporting must be test-only")
        selection_sha = str(final_test.get("selection_manifest_sha256", ""))
        if len(selection_sha) != 64 or any(
            character not in "0123456789abcdef" for character in selection_sha
        ):
            raise ValueError("test tier requires a lowercase SHA-256 selection lock")
        if list(final_test.get("test_window", [])) != list(splits["test"]):
            raise ValueError("test tier window must match data.splits.test")
        maturity = config["data"].get("label_maturity")
        if not isinstance(maturity, dict) or maturity.get("rule") != "strictly_before_next_split_start":
            raise ValueError("test tier requires strict label-maturity metadata")
        materialization_end = config["data"].get("materialization_end_time")
        if materialization_end is None or pd.Timestamp(materialization_end) != pd.Timestamp(
            splits["test"][1]
        ):
            raise ValueError("test tier materialization must end at the locked test end")
        return
    if tier != "selection":
        return
    if evaluation_splits != ["valid"]:
        raise ValueError("selection tier must evaluate only ['valid']")
    maturity = config["data"].get("label_maturity")
    if not isinstance(maturity, dict):
        raise ValueError("selection tier requires data.label_maturity")
    if maturity.get("rule") != "strictly_before_next_split_start":
        raise ValueError("selection tier requires strict label-maturity purge")
    if int(maturity.get("horizon_trading_days", 0)) < 1:
        raise ValueError("selection tier requires a positive label horizon")
    materialization_end = config["data"].get("materialization_end_time")
    if materialization_end is None:
        raise ValueError("selection tier requires data.materialization_end_time")
    if pd.Timestamp(materialization_end) > pd.Timestamp(splits["valid"][1]):
        raise ValueError("selection tier may not materialize beyond validation end")


def validate_test_selection_manifest(config: dict[str, Any], repo_root: Path) -> None:
    if config.get("tier") != "test":
        return
    final_test = config["final_test"]
    path = Path(final_test["selection_manifest"]).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen selection manifest: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != final_test["selection_manifest_sha256"]:
        raise ValueError("frozen selection manifest SHA-256 mismatch")


def strict_maturity_safe_end(
    calendar: Any,
    next_split_start: str | pd.Timestamp,
    *,
    horizon_trading_days: int,
) -> pd.Timestamp:
    """Latest sample whose forward label matures before the next split starts."""
    if int(horizon_trading_days) < 1:
        raise ValueError("horizon_trading_days must be positive")
    values = pd.DatetimeIndex(pd.to_datetime(list(calendar))).unique().sort_values()
    next_position = int(
        values.searchsorted(pd.Timestamp(next_split_start), side="left")
    )
    safe_position = next_position - int(horizon_trading_days) - 1
    if safe_position < 0:
        raise ValueError("calendar has insufficient history for strict label maturity")
    return pd.Timestamp(values[safe_position])


def _initialize_qlib(config: dict[str, Any]) -> None:
    data = config["data"]
    provider = str(Path(data["provider_path"]).expanduser().resolve())
    cpu_threads = int(config["resource_limits"]["cpu_threads"])
    qlib.init(
        provider_uri=provider,
        region=data.get("region", "cn"),
        kernels=cpu_threads,
    )


def build_cached_handler(config: dict[str, Any]) -> tuple[Path, bool]:
    _initialize_qlib(config)
    data = config["data"]

    identity = _handler_identity(config)
    directory = Path(data["cache_root"]).expanduser() / data["data_id"]
    cache_path = directory / (
        f"qlib__{data['handler'].lower()}__{data['market']}__{_fingerprint(identity)}.pkl"
    )
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path, True

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with _exclusive_lock(lock_path):
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path, True
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        handler = init_instance_by_config(_handler_config(config))
        handler.to_pickle(temporary, dump_all=True)
        os.replace(temporary, cache_path)
    return cache_path, False


def _dataset_config(
    config: dict[str, Any],
    handler_source: Path | DataHandlerLP,
    *,
    segments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = config["data"]
    dataset = config.get("dataset", {})
    handler: str | DataHandlerLP
    if isinstance(handler_source, Path):
        handler = f"file://{handler_source}"
    else:
        handler = handler_source
    kwargs = {
        "handler": handler,
        "segments": data["splits"] if segments is None else segments,
        **dataset.get("kwargs", {}),
    }
    dataset_class = dataset.get("class", "DatasetH")
    dataset_module = dataset.get("module_path", "qlib.data.dataset")
    if isinstance(handler_source, CompactQlibHandler) and dataset_class == "TSDatasetH":
        dataset_class = "CompactTSDatasetH"
        dataset_module = "meta_fdg.data.compact_qlib"
    return {
        "class": dataset_class,
        "module_path": dataset_module,
        "kwargs": kwargs,
    }


def _development_segments(config: dict[str, Any]) -> dict[str, Any]:
    splits = config["data"]["splits"]
    # Qlib's row-wise LSTM eagerly requests train, valid, and test during fit,
    # although it never consumes the returned test frame. Alias that fit-time
    # slot to validation so compact development handlers never load held-out
    # test data while still satisfying the upstream model interface.
    return {
        "train": splits["train"],
        "valid": splits["valid"],
        "test": splits["valid"],
    }


def _frame_from_sampler(value: Any, column: str) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        return value.to_frame(column)
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.columns = [column] if frame.shape[1] == 1 else list(frame.columns)
        return frame
    if hasattr(value, "data_arr") and hasattr(value, "data_index"):
        array = np.asarray(value.data_arr)
        if array.ndim == 2:
            array = array[:, 0]
        size = min(len(array), len(value.data_index))
        return pd.DataFrame({column: array[:size]}, index=value.data_index[:size])
    raise TypeError(f"unsupported Qlib output type: {type(value).__name__}")


def _predict(
    model: Any,
    dataset: DatasetH,
    split: str,
    *,
    config: dict[str, Any] | None = None,
    handler_source: Path | DataHandlerLP | None = None,
) -> pd.DataFrame:
    signature = inspect.signature(model.predict)
    if "segment" in signature.parameters:
        value = model.predict(dataset, segment=split)
    else:
        prediction_dataset = dataset
        if split != "test":
            if config is None or handler_source is None:
                raise ValueError(f"{type(model).__name__}.predict cannot select split {split!r}")
            alias_segments = {
                **_development_segments(config),
                "test": config["data"]["splits"][split],
            }
            alias_config = _dataset_config(
                config, handler_source, segments=alias_segments
            )
            prediction_dataset = init_instance_by_config(alias_config)
        value = model.predict(prediction_dataset)
    frame = _frame_from_sampler(value, "score")
    if "score" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "score"})
    return frame[["score"]]


def _raw_label(dataset: DatasetH, split: str) -> pd.DataFrame:
    # TSDataSampler keeps pre-segment history in data_arr.  Pairing that array
    # with its shorter data_index can silently shift labels, so fetch the raw
    # split directly from the handler whenever segment metadata is available.
    handler = getattr(dataset, "handler", None)
    segments = getattr(dataset, "segments", None)
    if handler is not None and isinstance(segments, dict) and split in segments:
        segment = segments[split]
        selector = slice(*segment) if isinstance(segment, (list, tuple)) else segment
        value = handler.fetch(
            selector,
            col_set=["label"],
            data_key=DataHandlerLP.DK_R,
        )
    else:
        value = dataset.prepare(split, col_set=["label"], data_key=DataHandlerLP.DK_R)
    frame = _frame_from_sampler(value, "label")
    if "label" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "label"})
    return frame[["label"]]


def _evaluate_split(
    model: Any,
    dataset: DatasetH,
    split: str,
    *,
    config: dict[str, Any] | None = None,
    handler_source: Path | DataHandlerLP | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score = _predict(
        model,
        dataset,
        split,
        config=config,
        handler_source=handler_source,
    )
    label = _raw_label(dataset, split)
    joined = score.join(label, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        raise RuntimeError(f"no finite aligned predictions for {split}")

    date_level = "datetime" if "datetime" in joined.index.names else joined.index.names[0]
    instrument_level = "instrument" if "instrument" in joined.index.names else joined.index.names[1]
    daily_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for date, frame in joined.groupby(level=date_level, sort=True):
        metrics = daily_signal_metrics(
            frame["score"].to_numpy(dtype=np.float64),
            frame["label"].to_numpy(dtype=np.float64),
        )
        daily_rows.append({"split": split, "date": pd.Timestamp(date), **metrics})
        instruments = frame.index.get_level_values(instrument_level).astype(str)
        prediction_rows.extend(
            {
                "split": split,
                "date": pd.Timestamp(date),
                "instrument": instrument,
                "label": float(target),
                "score": float(prediction),
            }
            for instrument, target, prediction in zip(
                instruments,
                frame["label"].to_numpy(),
                frame["score"].to_numpy(),
                strict=True,
            )
        )
    daily = pd.DataFrame(daily_rows)
    predictions = pd.DataFrame(prediction_rows)
    return daily, predictions, summarize_daily_metrics(daily)


def _save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _close_runtime_writers(model: Any) -> None:
    """Detach non-pickleable log writers after training has finished."""
    writer = getattr(model, "_writer", None)
    if writer is None:
        return
    try:
        writer.flush()
    finally:
        writer.close()
        model._writer = None


def run_qlib_baseline(repo_root: Path, config: dict[str, Any], *, allow_existing: bool = False) -> Path:
    config = copy.deepcopy(config)
    _validate_execution_contract(config)
    validate_test_selection_manifest(config, repo_root)
    run_root = repo_root / "runs" / str(config["tier"]) / str(config["job_id"])
    config = _resolve_runtime_placeholders(config, run_root)
    run = RunDirectory.create(repo_root, config, allow_existing=allow_existing)
    process_log = run.root / "logs" / "train.log"
    with _capture_process_log(process_log):
        try:
            started = _now()
            run.set_status("running", started_at_utc=started, pid=os.getpid())
            print(f"[{started}] start {config['job_id']}", flush=True)
            runtime_resource_status = _apply_resource_limits(config)
            _set_seed(int(config["seed"]))

            use_compact = bool(config["data"].get("use_compact_materialization", False))
            if use_compact:
                # The legacy handler-cache path initializes Qlib while loading
                # or building the pickle.  Compact mmap handlers bypass that
                # path, but downstream workflow/model helpers still require
                # Qlib's global provider and recorder configuration.
                _initialize_qlib(config)
                development_handler = compact_handler_from_config(
                    config, include_test=False
                )
                expected_width = expected_feature_dim(config)
                if development_handler.feature_dim != expected_width:
                    raise ValueError(
                        "compact handler feature dimension mismatch before fit: "
                        f"expected={expected_width} "
                        f"observed={development_handler.feature_dim}"
                    )
                handler_source: Path | DataHandlerLP = development_handler
                cache_path = None
                cache_hit = True
            else:
                cache_path, cache_hit = build_cached_handler(config)
                handler_source = cache_path
            from qlib.workflow import R

            # Qlib model helpers may create an implicit recorder. Keep every
            # such runtime artifact inside the owning run directory.
            R.set_uri(str((run.root / "mlruns").resolve()))
            dataset_config = _dataset_config(
                config,
                handler_source,
                segments=(
                    _development_segments(config) if use_compact else None
                ),
            )
            dataset = init_instance_by_config(dataset_config)
            data_status: dict[str, Any] = {
                "data_id": config["data"]["data_id"],
                "provider_path": config["data"]["provider_path"],
                "cache_hit": cache_hit,
                "handler_cache_identity": _handler_identity(config),
                "resource_limits": config["resource_limits"],
                "runtime_resource_status": runtime_resource_status,
                "qlib_kernels": int(config["resource_limits"]["cpu_threads"]),
            }
            if use_compact:
                data_status.update(
                    {
                        "cache_backend": "compact_float32_mmap_v1",
                        "development_materialization": (
                            development_handler.materialization_metadata
                        ),
                        "test_materialization_loaded_during_fit": False,
                        "expected_feature_dim": expected_width,
                        "feature_identity_verified": True,
                    }
                )
            else:
                data_status.update(
                    {
                        "cache_backend": "qlib_handler_pickle",
                        "handler_cache": str(cache_path),
                    }
                )
            model = init_instance_by_config(config["model"])
            data_status["runtime_provenance"] = _runtime_provenance(model, config)
            atomic_json(run.root / "data_status.json", data_status)
            if use_compact:
                print(
                    "handler_cache=compact_float32_mmap_v1 "
                    f"factor_id={development_handler.factor_id} "
                    f"label_id={development_handler.label_id}",
                    flush=True,
                )
            else:
                print(f"handler_cache={cache_path} hit={cache_hit}", flush=True)

            evals_result: dict[str, Any] = {}
            fit_kwargs = dict(config.get("fit_kwargs", {}))
            fit_signature = inspect.signature(model.fit)
            if "evals_result" in fit_signature.parameters:
                fit_kwargs["evals_result"] = evals_result
            if "save_path" in fit_signature.parameters and "save_path" not in fit_kwargs:
                fit_kwargs["save_path"] = str(run.root / "checkpoints" / "native_best.pt")
            model.fit(dataset, **fit_kwargs)
            data_status["fit_completed_at_utc"] = _now()
            atomic_json(run.root / "data_status.json", data_status)
            atomic_json(run.root / "logs" / "fit_history.json", evals_result)
            _close_runtime_writers(model)
            _save_model(model, run.root / "checkpoints" / "model.pkl")

            daily_frames: list[pd.DataFrame] = []
            prediction_frames: list[pd.DataFrame] = []
            summaries: dict[str, dict[str, Any]] = {}
            for split in config.get("evaluation_splits", ["valid", "test"]):
                evaluation_dataset = dataset
                evaluation_handler_source = handler_source
                if use_compact and split == "test":
                    reporting_handler = compact_handler_from_config(
                        config, include_test=True
                    )
                    if reporting_handler.feature_dim != expected_width:
                        raise ValueError(
                            "compact reporting handler feature dimension mismatch: "
                            f"expected={expected_width} "
                            f"observed={reporting_handler.feature_dim}"
                        )
                    evaluation_handler_source = reporting_handler
                    evaluation_dataset = init_instance_by_config(
                        _dataset_config(config, reporting_handler)
                    )
                    data_status["reporting_materialization"] = (
                        reporting_handler.materialization_metadata
                    )
                    data_status["test_materialization_loaded_at_utc"] = _now()
                    atomic_json(run.root / "data_status.json", data_status)
                daily, predictions, summary = _evaluate_split(
                    model,
                    evaluation_dataset,
                    split,
                    config=config,
                    handler_source=evaluation_handler_source,
                )
                daily_frames.append(daily)
                prediction_frames.append(predictions)
                summaries[split] = summary
                print(f"{split}: {json.dumps(summary, sort_keys=True)}", flush=True)

            runtime_audit = getattr(model, "runtime_audit", None)
            if isinstance(runtime_audit, dict):
                atomic_json(run.root / "runtime_audit.json", runtime_audit)

            pd.concat(daily_frames, ignore_index=True).to_parquet(
                run.root / "metrics_daily.parquet", index=False
            )
            pd.concat(prediction_frames, ignore_index=True).to_parquet(
                run.root / "predictions.parquet", index=False
            )
            ended = _now()
            atomic_json(
                run.root / "metrics_summary.json",
                {
                    "method": config["method"],
                    "seed": config["seed"],
                    "splits": summaries,
                    "started_at_utc": started,
                    "ended_at_utc": ended,
                },
            )
            run.set_status("done", started_at_utc=started, ended_at_utc=ended)
            print(f"[{ended}] done {config['job_id']}", flush=True)
            return run.root
        except BaseException as error:
            run.fail(error)
            print(f"[{_now()}] failed {config['job_id']}: {type(error).__name__}: {error}", flush=True)
            raise
