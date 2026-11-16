from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.data.dataset.utils import fetch_df_by_col
from qlib.utils import init_instance_by_config


COMPACT_QLIB_SCHEMA = "meta_fdg_compact_qlib_v1"
DEVELOPMENT_PARTITION = "development"
TEST_PARTITION = "test"
COMPLETE_MARKER = "MATERIALIZATION_COMPLETE"
EXPECTED_BASE_FEATURE_DIMS = {
    "Alpha158": 158,
    "Alpha158USOHLCV": 157,
    "Alpha360": 360,
    "MasterAlpha158": 221,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_processors(values: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    return [json.loads(_canonical_json(dict(value))) for value in (values or [])]


def _normalise_splits(
    splits: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    return {
        str(name): [str(period[0]), str(period[1])]
        for name, period in sorted(splits.items())
    }


def expected_feature_dim(config: Mapping[str, Any]) -> int:
    """Resolve the post-inference-processor feature width fail closed."""

    data = copy.deepcopy(dict(config["data"]))
    handler = str(data.get("handler_class") or data.get("handler"))
    if handler not in EXPECTED_BASE_FEATURE_DIMS:
        raise ValueError(f"unsupported compact handler feature contract: {handler}")
    processors = data.get("handler_kwargs", {}).get("infer_processors", [])
    filtered: int | None = None
    for processor in processors:
        if str(processor.get("class")) != "FilterCol":
            continue
        kwargs = dict(processor.get("kwargs", {}))
        if kwargs.get("fields_group", "feature") != "feature":
            continue
        columns = kwargs.get("col_list")
        if not isinstance(columns, list) or not columns:
            raise ValueError("feature FilterCol requires a non-empty col_list")
        width = len(columns)
        if filtered is not None and filtered != width:
            raise ValueError("multiple feature FilterCol processors disagree on width")
        filtered = width
    return int(filtered if filtered is not None else EXPECTED_BASE_FEATURE_DIMS[handler])


def compact_identities(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return seed/model-independent identities for factors and label views."""

    data = dict(config["data"])
    splits = data["splits"]
    handler_kwargs = copy.deepcopy(dict(data.get("handler_kwargs", {})))
    context_start = str(data.get("context_start_time", splits["train"][0]))
    provider = str(Path(data["provider_path"]).expanduser().resolve())
    factor_handler_kwargs = {
        key: value
        for key, value in handler_kwargs.items()
        if key not in {"learn_processors", "process_type"}
    }
    factor = {
        "schema": COMPACT_QLIB_SCHEMA,
        "data_id": str(data["data_id"]),
        "provider_path": provider,
        "region": str(data.get("region", "cn")),
        "handler": str(data.get("handler_class") or data.get("handler")),
        "handler_module_path": str(
            data.get("handler_module_path") or "qlib.contrib.data.handler"
        ),
        "market": str(data["market"]),
        "label_expression": str(data["label"]),
        "context_start_time": context_start,
        "splits": _normalise_splits(splits),
        "fit_start_time": str(splits["train"][0]),
        "fit_end_time": str(splits["train"][1]),
        "handler_kwargs": json.loads(_canonical_json(factor_handler_kwargs)),
        "expected_feature_dim": expected_feature_dim(config),
    }
    label = {
        "schema": COMPACT_QLIB_SCHEMA,
        "factor_id": _fingerprint(factor),
        "process_type": str(handler_kwargs.get("process_type", DataHandlerLP.PTYPE_A)),
        "learn_processors": _normalise_processors(
            handler_kwargs.get("learn_processors")
        ),
    }
    return factor, label


def compact_store_ids(config: Mapping[str, Any]) -> tuple[str, str]:
    factor, label = compact_identities(config)
    return _fingerprint(factor), _fingerprint(label)


def compact_cache_root(config: Mapping[str, Any]) -> Path:
    data = config["data"]
    value = data.get("materialized_cache_root")
    if not value:
        raise ValueError("compact Qlib mode requires data.materialized_cache_root")
    root = Path(str(value)).expanduser()
    if not root.is_absolute():
        raise ValueError("data.materialized_cache_root must be absolute")
    return root.resolve()


def _factor_partition_path(root: Path, data_id: str, factor_id: str, partition: str) -> Path:
    return root / str(data_id) / "factors" / factor_id / partition


def _label_partition_path(root: Path, data_id: str, label_id: str, partition: str) -> Path:
    return root / str(data_id) / "labels" / label_id / partition


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


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_directory(target: Path, writer: Callable[[Path], None]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        writer(temporary)
        (temporary / COMPLETE_MARKER).write_text(
            f"schema={COMPACT_QLIB_SCHEMA}\ncreated_at_utc={_now()}\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    marker = path / COMPLETE_MARKER
    if not manifest_path.is_file() or not marker.is_file():
        raise FileNotFoundError(f"incomplete compact materialization: {path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _normalise_frame_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise TypeError("compact Qlib materialization requires a two-level MultiIndex")
    names = list(frame.index.names)
    if "datetime" not in names or "instrument" not in names:
        frame = frame.copy()
        frame.index = frame.index.set_names(["datetime", "instrument"])
    elif names != ["datetime", "instrument"]:
        frame = frame.reorder_levels(["datetime", "instrument"])
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("compact Qlib source index contains duplicates")
    return frame


def _column_group(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        raise TypeError("compact Qlib source requires grouped feature/label columns")
    if group not in set(frame.columns.get_level_values(0)):
        raise KeyError(f"compact Qlib source is missing column group {group!r}")
    value = frame.loc[:, group]
    if isinstance(value, pd.Series):
        value = value.to_frame()
    return value


def _partition_mask(index: pd.MultiIndex, partition: str, splits: Mapping[str, Sequence[str]]) -> np.ndarray:
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    valid_end = pd.Timestamp(splits["valid"][1])
    if partition == DEVELOPMENT_PARTITION:
        return np.asarray(dates <= valid_end, dtype=bool)
    if partition == TEST_PARTITION:
        test_end = pd.Timestamp(splits["test"][1])
        return np.asarray((dates > valid_end) & (dates <= test_end), dtype=bool)
    raise ValueError(f"unknown compact partition: {partition}")


def _save_array(path: Path, value: np.ndarray) -> None:
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)


def _array_record(path: Path) -> dict[str, Any]:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _equal_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def _instrument_array(index: pd.MultiIndex) -> np.ndarray:
    values = [str(value) for value in index.get_level_values("instrument")]
    width = max(1, max((len(value) for value in values), default=1))
    return np.asarray(values, dtype=f"<U{width}")


def _processor_records(processors: Sequence[Any] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for processor in processors or []:
        cls = type(processor)
        try:
            state = pickle.dumps(processor, protocol=pickle.HIGHEST_PROTOCOL)
            state_sha256 = hashlib.sha256(state).hexdigest()
        except Exception as error:  # pragma: no cover - defensive provenance path
            state_sha256 = None
            pickle_error = f"{type(error).__name__}: {error}"
        else:
            pickle_error = None
        records.append(
            {
                "class": cls.__qualname__,
                "module": cls.__module__,
                "state_sha256": state_sha256,
                "pickle_error": pickle_error,
            }
        )
    return records


def _normalization_provenance(
    handler: DataHandlerLP, config: Mapping[str, Any]
) -> dict[str, Any]:
    splits = config["data"]["splits"]
    return {
        "fit_scope": "train_only",
        "fit_start_time": str(splits["train"][0]),
        "fit_end_time": str(splits["train"][1]),
        "infer_processors": _processor_records(
            getattr(handler, "infer_processors", None)
        ),
        "learn_processors": _processor_records(
            getattr(handler, "learn_processors", None)
        ),
    }


def _validate_existing_factor(
    path: Path,
    *,
    identity: Mapping[str, Any],
    index: pd.MultiIndex,
    feature_columns: Sequence[str],
    features: np.ndarray,
    infer_label: np.ndarray,
    raw_label: np.ndarray,
) -> dict[str, Any]:
    manifest = _read_manifest(path)
    if manifest.get("identity") != dict(identity):
        raise ValueError(f"factor identity mismatch at {path}")
    if list(manifest.get("feature_columns", [])) != list(feature_columns):
        raise ValueError(f"factor columns mismatch at {path}")
    dates = np.asarray(index.get_level_values("datetime"), dtype="datetime64[ns]").astype(np.int64)
    instruments = _instrument_array(index)
    comparisons = {
        "dates": _equal_nan(np.load(path / "dates.npy", mmap_mode="r"), dates),
        "instruments": np.array_equal(
            np.load(path / "instruments.npy", mmap_mode="r"), instruments
        ),
        "features": _equal_nan(
            np.load(path / "features.npy", mmap_mode="r"), features
        ),
        "infer_label": _equal_nan(
            np.load(path / "infer_label.npy", mmap_mode="r"), infer_label
        ),
        "raw_label": _equal_nan(
            np.load(path / "raw_label.npy", mmap_mode="r"), raw_label
        ),
    }
    if not all(comparisons.values()):
        raise ValueError(f"factor parity mismatch at {path}: {comparisons}")
    return {"path": str(path), "cache_hit": True, "comparisons": comparisons}


def _validate_existing_label(
    path: Path,
    *,
    identity: Mapping[str, Any],
    positions: np.ndarray,
    learn_label: np.ndarray,
) -> dict[str, Any]:
    manifest = _read_manifest(path)
    if manifest.get("identity") != dict(identity):
        raise ValueError(f"label identity mismatch at {path}")
    comparisons = {
        "learn_positions": np.array_equal(
            np.load(path / "learn_positions.npy", mmap_mode="r"), positions
        ),
        "learn_label": _equal_nan(
            np.load(path / "learn_label.npy", mmap_mode="r"), learn_label
        ),
    }
    if not all(comparisons.values()):
        raise ValueError(f"label parity mismatch at {path}: {comparisons}")
    return {"path": str(path), "cache_hit": True, "comparisons": comparisons}


def materialize_compact_handler(
    config: Mapping[str, Any],
    handler: DataHandlerLP,
    *,
    cache_root: Path | None = None,
    source: Mapping[str, Any] | None = None,
    partitions: Sequence[str] = (DEVELOPMENT_PARTITION, TEST_PARTITION),
) -> dict[str, Any]:
    """Persist factor rows once and small model-specific label views separately."""

    root = compact_cache_root(config) if cache_root is None else Path(cache_root).resolve()
    data = dict(config["data"])
    factor_identity, label_identity = compact_identities(config)
    factor_id, label_id = _fingerprint(factor_identity), _fingerprint(label_identity)
    infer = _normalise_frame_index(handler._get_df_by_key(DataHandlerLP.DK_I))
    raw = _normalise_frame_index(handler._get_df_by_key(DataHandlerLP.DK_R))
    learn = _normalise_frame_index(handler._get_df_by_key(DataHandlerLP.DK_L))
    feature_frame = _column_group(infer, "feature")
    infer_label_frame = _column_group(infer, "label")
    raw_label_frame = _column_group(raw, "label").reindex(infer.index)
    learn_feature_frame = _column_group(learn, "feature")
    learn_label_frame = _column_group(learn, "label")
    if infer_label_frame.shape[1] != 1 or raw_label_frame.shape[1] != 1 or learn_label_frame.shape[1] != 1:
        raise ValueError("compact Qlib materialization supports exactly one label")
    feature_columns = [str(value) for value in feature_frame.columns]
    expected_width = expected_feature_dim(config)
    if len(feature_columns) != expected_width:
        raise ValueError(
            "compact feature identity mismatch: "
            f"handler={data.get('handler_class') or data.get('handler')} "
            f"expected={expected_width} observed={len(feature_columns)}"
        )
    if [str(value) for value in learn_feature_frame.columns] != feature_columns:
        raise ValueError("learn and infer feature columns differ")
    requested_partitions = tuple(str(value) for value in partitions)
    if not requested_partitions or len(set(requested_partitions)) != len(
        requested_partitions
    ):
        raise ValueError("compact partitions must be non-empty and unique")
    if set(requested_partitions) - {DEVELOPMENT_PARTITION, TEST_PARTITION}:
        raise ValueError(f"unsupported compact partitions: {requested_partitions}")
    normalization = _normalization_provenance(handler, config)
    audit: dict[str, Any] = {
        "schema": COMPACT_QLIB_SCHEMA,
        "created_at_utc": _now(),
        "factor_id": factor_id,
        "label_id": label_id,
        "source": dict(source or {}),
        "expected_feature_dim": expected_width,
        "normalization_provenance": normalization,
        "partitions": {},
    }

    for partition in requested_partitions:
        infer_mask = _partition_mask(infer.index, partition, data["splits"])
        part_index = infer.index[infer_mask]
        if len(part_index) == 0:
            raise ValueError(f"empty compact factor partition: {partition}")
        features = feature_frame.loc[part_index].to_numpy(dtype=np.float32, copy=True)
        infer_label = infer_label_frame.loc[part_index].to_numpy(
            dtype=np.float32, copy=True
        )[:, 0]
        raw_label = raw_label_frame.loc[part_index].to_numpy(
            dtype=np.float32, copy=True
        )[:, 0]
        factor_path = _factor_partition_path(
            root, str(data["data_id"]), factor_id, partition
        )
        factor_lock = factor_path.with_name(factor_path.name + ".lock")
        with _exclusive_lock(factor_lock):
            if factor_path.exists():
                factor_audit = _validate_existing_factor(
                    factor_path,
                    identity=factor_identity,
                    index=part_index,
                    feature_columns=feature_columns,
                    features=features,
                    infer_label=infer_label,
                    raw_label=raw_label,
                )
            else:
                def write_factor(directory: Path) -> None:
                    dates = np.asarray(
                        part_index.get_level_values("datetime"), dtype="datetime64[ns]"
                    ).astype(np.int64)
                    instruments = _instrument_array(part_index)
                    _save_array(directory / "dates.npy", dates)
                    _save_array(directory / "instruments.npy", instruments)
                    _save_array(directory / "features.npy", features)
                    _save_array(directory / "infer_label.npy", infer_label)
                    _save_array(directory / "raw_label.npy", raw_label)
                    files = {
                        name: _array_record(directory / name)
                        for name in (
                            "dates.npy",
                            "instruments.npy",
                            "features.npy",
                            "infer_label.npy",
                            "raw_label.npy",
                        )
                    }
                    _atomic_json(
                        directory / "manifest.json",
                        {
                            "schema": COMPACT_QLIB_SCHEMA,
                            "kind": "factor_partition",
                            "partition": partition,
                            "created_at_utc": _now(),
                            "identity": factor_identity,
                            "factor_id": factor_id,
                            "feature_columns": feature_columns,
                            "label_column": str(infer_label_frame.columns[0]),
                            "rows": int(len(part_index)),
                            "date_min": str(pd.Timestamp(part_index[0][0]).date()),
                            "date_max": str(pd.Timestamp(part_index[-1][0]).date()),
                            "files": files,
                            "source": dict(source or {}),
                            "normalization_provenance": {
                                **normalization,
                                "learn_processors": [],
                            },
                        },
                    )

                _atomic_directory(factor_path, write_factor)
                factor_audit = {
                    "path": str(factor_path),
                    "cache_hit": False,
                    "comparisons": {
                        "dates": True,
                        "instruments": True,
                        "features": True,
                        "infer_label": True,
                        "raw_label": True,
                    },
                }

        learn_mask = _partition_mask(learn.index, partition, data["splits"])
        learn_part = learn.loc[learn.index[learn_mask]]
        positions = part_index.get_indexer(learn_part.index).astype(np.int64)
        if (positions < 0).any():
            raise ValueError(f"learn index is not a subset of infer index: {partition}")
        observed_learn_features = _column_group(learn_part, "feature").to_numpy(
            dtype=np.float32, copy=False
        )
        if not _equal_nan(observed_learn_features, features[positions]):
            raise ValueError(
                "learn processors changed features; a separate factor identity is required"
            )
        learn_label = _column_group(learn_part, "label").to_numpy(
            dtype=np.float32, copy=True
        )[:, 0]
        label_path = _label_partition_path(
            root, str(data["data_id"]), label_id, partition
        )
        label_lock = label_path.with_name(label_path.name + ".lock")
        with _exclusive_lock(label_lock):
            if label_path.exists():
                label_audit = _validate_existing_label(
                    label_path,
                    identity=label_identity,
                    positions=positions,
                    learn_label=learn_label,
                )
            else:
                def write_label(directory: Path) -> None:
                    _save_array(directory / "learn_positions.npy", positions)
                    _save_array(directory / "learn_label.npy", learn_label)
                    files = {
                        name: _array_record(directory / name)
                        for name in ("learn_positions.npy", "learn_label.npy")
                    }
                    _atomic_json(
                        directory / "manifest.json",
                        {
                            "schema": COMPACT_QLIB_SCHEMA,
                            "kind": "label_partition",
                            "partition": partition,
                            "created_at_utc": _now(),
                            "identity": label_identity,
                            "factor_id": factor_id,
                            "label_id": label_id,
                            "rows": int(len(positions)),
                            "label_column": str(learn_label_frame.columns[0]),
                            "files": files,
                            "source": dict(source or {}),
                            "normalization_provenance": normalization,
                        },
                    )

                _atomic_directory(label_path, write_label)
                label_audit = {
                    "path": str(label_path),
                    "cache_hit": False,
                    "comparisons": {"learn_positions": True, "learn_label": True},
                }
        audit["partitions"][partition] = {
            "factor": factor_audit,
            "label": label_audit,
            "factor_rows": int(len(part_index)),
            "learn_rows": int(len(positions)),
        }
    return audit


def build_source_handler(
    config: Mapping[str, Any],
    *,
    partitions: Sequence[str] = (DEVELOPMENT_PARTITION, TEST_PARTITION),
) -> DataHandlerLP:
    """Build one full-period Qlib handler for one-time compact materialization."""

    # Qlib's recursive config instantiation may mutate nested processor
    # dictionaries.  Keep the locked resolved config immutable because compact
    # factor/label identities are computed from it before and after handler
    # construction.
    data = copy.deepcopy(dict(config["data"]))
    splits = data["splits"]
    provider = str(Path(data["provider_path"]).expanduser().resolve())
    import qlib

    qlib.init(
        provider_uri=provider,
        region=str(data.get("region", "cn")),
        kernels=int(config.get("resource_limits", {}).get("cpu_threads", 4)),
    )
    handler_kwargs = copy.deepcopy(dict(data.get("handler_kwargs", {})))
    forbidden = {
        "instruments",
        "start_time",
        "end_time",
        "fit_start_time",
        "fit_end_time",
        "label",
    } & set(handler_kwargs)
    if forbidden:
        raise ValueError(f"handler_kwargs overrides locked fields: {sorted(forbidden)}")
    requested_partitions = tuple(str(value) for value in partitions)
    if set(requested_partitions) - {DEVELOPMENT_PARTITION, TEST_PARTITION}:
        raise ValueError(f"unsupported compact partitions: {requested_partitions}")
    end_time = (
        splits["test"][1]
        if TEST_PARTITION in requested_partitions
        else splits["valid"][1]
    )
    handler_config = {
        "class": data.get("handler_class") or data["handler"],
        "module_path": data.get("handler_module_path")
        or "qlib.contrib.data.handler",
        "kwargs": {
            "instruments": data["market"],
            "start_time": data.get("context_start_time", splits["train"][0]),
            "end_time": end_time,
            "fit_start_time": splits["train"][0],
            "fit_end_time": splits["train"][1],
            "label": ([data["label"]], ["LABEL0"]),
            **handler_kwargs,
        },
    }
    handler = init_instance_by_config(handler_config)
    if not isinstance(handler, DataHandlerLP):
        raise TypeError("compact materialization source must be DataHandlerLP")
    return handler


def load_source_handler(path: str | Path) -> DataHandlerLP:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        handler = pickle.load(handle)
    if not isinstance(handler, DataHandlerLP):
        raise TypeError(f"cached source is not DataHandlerLP: {source}")
    return handler


@dataclass
class _CompactPartition:
    factor_path: Path
    label_path: Path
    factor_manifest: dict[str, Any]
    label_manifest: dict[str, Any]
    dates: np.ndarray
    instruments: np.ndarray
    features: np.ndarray
    infer_label: np.ndarray
    raw_label: np.ndarray
    learn_positions: np.ndarray
    learn_label: np.ndarray


class CompactQlibHandler(DataHandlerLP):
    """Read-only Qlib handler backed by shared float32 memory maps."""

    def __init__(
        self,
        *,
        cache_root: str | Path,
        data_id: str,
        factor_id: str,
        label_id: str,
        partitions: Sequence[str] = (DEVELOPMENT_PARTITION,),
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.data_id = str(data_id)
        self.factor_id = str(factor_id)
        self.label_id = str(label_id)
        self.partition_names = tuple(str(value) for value in partitions)
        if self.partition_names not in (
            (DEVELOPMENT_PARTITION,),
            (DEVELOPMENT_PARTITION, TEST_PARTITION),
        ):
            raise ValueError("compact handler partitions must be development or development+test")
        self._parts = tuple(self._load_partition(name) for name in self.partition_names)
        label_expression = str(
            self._parts[0].factor_manifest["identity"]["label_expression"]
        )
        self.data_loader = SimpleNamespace(
            fields={"label": ([label_expression], ["LABEL0"])}
        )
        self.instruments = None
        self.start_time = pd.Timestamp(self._parts[0].dates[0], unit="ns")
        self.end_time = pd.Timestamp(self._parts[-1].dates[-1], unit="ns")
        self.fetch_orig = True
        self.drop_raw = False
        self.from_cast = True

    def _load_partition(self, partition: str) -> _CompactPartition:
        factor_path = _factor_partition_path(
            self.cache_root, self.data_id, self.factor_id, partition
        )
        label_path = _label_partition_path(
            self.cache_root, self.data_id, self.label_id, partition
        )
        factor_manifest = _read_manifest(factor_path)
        label_manifest = _read_manifest(label_path)
        if factor_manifest.get("factor_id") != self.factor_id:
            raise ValueError(f"factor id mismatch at {factor_path}")
        if label_manifest.get("label_id") != self.label_id:
            raise ValueError(f"label id mismatch at {label_path}")
        if label_manifest.get("factor_id") != self.factor_id:
            raise ValueError(f"label/factor link mismatch at {label_path}")
        dates = np.load(factor_path / "dates.npy", mmap_mode="r", allow_pickle=False)
        instruments = np.load(
            factor_path / "instruments.npy", mmap_mode="r", allow_pickle=False
        )
        features = np.load(
            factor_path / "features.npy", mmap_mode="r", allow_pickle=False
        )
        infer_label = np.load(
            factor_path / "infer_label.npy", mmap_mode="r", allow_pickle=False
        )
        raw_label = np.load(
            factor_path / "raw_label.npy", mmap_mode="r", allow_pickle=False
        )
        learn_positions = np.load(
            label_path / "learn_positions.npy", mmap_mode="r", allow_pickle=False
        )
        learn_label = np.load(
            label_path / "learn_label.npy", mmap_mode="r", allow_pickle=False
        )
        rows = int(factor_manifest.get("rows", -1))
        if rows < 1 or not (
            len(dates)
            == len(instruments)
            == len(features)
            == len(infer_label)
            == len(raw_label)
            == rows
        ):
            raise ValueError(f"factor array row contract mismatch at {factor_path}")
        if instruments.dtype.hasobject:
            raise ValueError(f"instrument array must be non-object Unicode at {factor_path}")
        expected_width = int(
            factor_manifest["identity"].get("expected_feature_dim", -1)
        )
        if features.ndim != 2 or features.shape[1] != expected_width:
            raise ValueError(
                f"factor feature dimension mismatch at {factor_path}: "
                f"expected={expected_width} observed={features.shape}"
            )
        if features.dtype != np.dtype("float32"):
            raise ValueError(f"factor features must be float32 at {factor_path}")
        if len(learn_positions) != len(learn_label) or (
            len(learn_positions) and int(np.max(learn_positions)) >= rows
        ):
            raise ValueError(f"label position contract mismatch at {label_path}")
        return _CompactPartition(
            factor_path=factor_path,
            label_path=label_path,
            factor_manifest=factor_manifest,
            label_manifest=label_manifest,
            dates=dates,
            instruments=instruments,
            features=features,
            infer_label=infer_label,
            raw_label=raw_label,
            learn_positions=learn_positions,
            learn_label=learn_label,
        )

    @property
    def feature_dim(self) -> int:
        return int(self._parts[0].features.shape[1])

    @property
    def calendar(self) -> list[pd.Timestamp]:
        values = np.concatenate(
            [np.asarray(part.dates, dtype=np.int64) for part in self._parts]
        )
        return list(pd.to_datetime(np.unique(values)))

    @property
    def _learn(self) -> pd.DataFrame:
        """Compatibility view for Qlib MTSDatasetH; still row-wise, never windows."""

        value = self.fetch(
            slice(None, None),
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if not isinstance(value, pd.DataFrame):  # pragma: no cover - type guard
            raise TypeError("compact learn view must be a DataFrame")
        return value

    @property
    def materialization_metadata(self) -> dict[str, Any]:
        return {
            "schema": COMPACT_QLIB_SCHEMA,
            "cache_root": str(self.cache_root),
            "data_id": self.data_id,
            "factor_id": self.factor_id,
            "label_id": self.label_id,
            "partitions": list(self.partition_names),
            "feature_dim": self.feature_dim,
            "test_partition_loaded": TEST_PARTITION in self.partition_names,
            "factor_paths": [str(part.factor_path) for part in self._parts],
            "label_paths": [str(part.label_path) for part in self._parts],
        }

    def config(self, **kwargs: Any) -> None:
        unsupported = set(kwargs) - {
            "instruments",
            "start_time",
            "end_time",
            "processor_kwargs",
            "dump_all",
            "exclude",
            "include",
        }
        if unsupported:
            raise ValueError(f"compact handler cannot be reconfigured: {sorted(unsupported)}")

    def setup_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise ValueError("compact handler is immutable and already materialized")

    @staticmethod
    def _bounds(dates: np.ndarray, selector: Any) -> tuple[int, int]:
        if isinstance(selector, (tuple, list)):
            selector = slice(*selector)
        if isinstance(selector, slice):
            if selector.step is not None:
                raise ValueError("compact datetime slices do not support a step")
            start = None if selector.start is None else pd.Timestamp(selector.start).value
            stop = None if selector.stop is None else pd.Timestamp(selector.stop).value
        elif isinstance(selector, (str, pd.Timestamp, np.datetime64)):
            value = pd.Timestamp(selector).value
            start = stop = value
        else:
            raise TypeError(f"unsupported compact selector: {type(selector).__name__}")
        left = 0 if start is None else int(np.searchsorted(dates, start, side="left"))
        right = len(dates) if stop is None else int(np.searchsorted(dates, stop, side="right"))
        return left, right

    @staticmethod
    def _index(part: _CompactPartition, positions: np.ndarray) -> pd.MultiIndex:
        dates = pd.to_datetime(np.asarray(part.dates[positions], dtype=np.int64))
        instruments = np.asarray(part.instruments[positions]).astype(str)
        return pd.MultiIndex.from_arrays(
            [dates, instruments], names=["datetime", "instrument"]
        )

    @staticmethod
    def _full_frame(
        part: _CompactPartition,
        positions: np.ndarray,
        *,
        data_key: str,
    ) -> pd.DataFrame:
        if data_key == DataHandlerLP.DK_L:
            source_positions = part.learn_positions[positions]
            feature_values = part.features[source_positions]
            label_values = part.learn_label[positions]
        else:
            source_positions = positions
            feature_values = part.features[source_positions]
            label_values = (
                part.raw_label[source_positions]
                if data_key == DataHandlerLP.DK_R
                else part.infer_label[source_positions]
            )
        index = CompactQlibHandler._index(part, source_positions)
        feature_columns = pd.MultiIndex.from_product(
            [["feature"], part.factor_manifest["feature_columns"]]
        )
        label_columns = pd.MultiIndex.from_tuples(
            [("label", part.factor_manifest["label_column"])]
        )
        features = pd.DataFrame(
            feature_values, index=index, columns=feature_columns, copy=False
        )
        labels = pd.DataFrame(
            np.asarray(label_values).reshape(-1, 1),
            index=index,
            columns=label_columns,
            copy=False,
        )
        if data_key == DataHandlerLP.DK_R:
            return labels
        return pd.concat([features, labels], axis=1, copy=False)

    def fetch(
        self,
        selector: Any = slice(None, None),
        level: str | int = "datetime",
        col_set: str | list[str] = DataHandler.CS_ALL,
        data_key: str = DataHandlerLP.DK_I,
        squeeze: bool = False,
        proc_func: Callable | None = None,
    ) -> pd.DataFrame | pd.Series:
        if level not in ("datetime", 0):
            raise ValueError("compact handler supports datetime selection only")
        if data_key not in (DataHandlerLP.DK_R, DataHandlerLP.DK_I, DataHandlerLP.DK_L):
            raise ValueError(f"unknown Qlib data key: {data_key}")
        frames: list[pd.DataFrame] = []
        for part in self._parts:
            left, right = self._bounds(part.dates, selector)
            if data_key == DataHandlerLP.DK_L:
                learn_positions = np.arange(len(part.learn_positions), dtype=np.int64)
                source = part.learn_positions
                keep = (source >= left) & (source < right)
                positions = learn_positions[keep]
            else:
                positions = np.arange(left, right, dtype=np.int64)
            if len(positions):
                frames.append(self._full_frame(part, positions, data_key=data_key))
        if frames:
            frame = frames[0] if len(frames) == 1 else pd.concat(frames, axis=0, copy=False)
        else:
            columns = pd.MultiIndex.from_tuples([])
            frame = pd.DataFrame(columns=columns)
            frame.index = pd.MultiIndex.from_arrays(
                [pd.DatetimeIndex([]), pd.Index([], dtype=str)],
                names=["datetime", "instrument"],
            )
        if proc_func is not None:
            frame = proc_func(frame.copy())
        frame = fetch_df_by_col(frame, col_set)
        if squeeze:
            value: pd.DataFrame | pd.Series = frame.squeeze()
            if isinstance(selector, (str, pd.Timestamp)) and isinstance(
                value.index, pd.MultiIndex
            ):
                value = value.reset_index(level=level, drop=True)
            return value
        return frame

    def get_cols(
        self,
        col_set: str | list[str] = DataHandler.CS_ALL,
        data_key: str = DataHandlerLP.DK_I,
    ) -> list[Any]:
        part = self._parts[0]
        feature_columns = pd.MultiIndex.from_product(
            [["feature"], part.factor_manifest["feature_columns"]]
        )
        label_columns = pd.MultiIndex.from_tuples(
            [("label", part.factor_manifest["label_column"])]
        )
        columns = label_columns if data_key == DataHandlerLP.DK_R else feature_columns.append(label_columns)
        frame = pd.DataFrame(columns=columns)
        frame = fetch_df_by_col(frame, col_set)
        return list(frame.columns)


class CompactTSDatasetH(TSDatasetH):
    """TSDatasetH variant that obtains its calendar without fetching all factors."""

    def setup_data(self, **kwargs: Any) -> None:
        DatasetH.setup_data(self, **kwargs)
        if not isinstance(self.handler, CompactQlibHandler):
            raise TypeError("CompactTSDatasetH requires CompactQlibHandler")
        self.cal = self.handler.calendar


def compact_handler_from_config(
    config: Mapping[str, Any],
    *,
    include_test: bool,
) -> CompactQlibHandler:
    root = compact_cache_root(config)
    factor_id, label_id = compact_store_ids(config)
    partitions = (
        (DEVELOPMENT_PARTITION, TEST_PARTITION)
        if include_test
        else (DEVELOPMENT_PARTITION,)
    )
    return CompactQlibHandler(
        cache_root=root,
        data_id=str(config["data"]["data_id"]),
        factor_id=factor_id,
        label_id=label_id,
        partitions=partitions,
    )


def materialize_configs(
    configs: Sequence[Mapping[str, Any]],
    *,
    cache_root: Path,
    source_handlers: Sequence[str | Path | None] | None = None,
    partitions: Sequence[str] = (DEVELOPMENT_PARTITION, TEST_PARTITION),
) -> list[dict[str, Any]]:
    if source_handlers is None:
        source_handlers = [None] * len(configs)
    if len(source_handlers) != len(configs):
        raise ValueError("source handler list must align with configs")
    rows: list[dict[str, Any]] = []
    for config, source_path in zip(configs, source_handlers, strict=True):
        if source_path is None:
            handler = build_source_handler(config, partitions=partitions)
            provider = Path(str(config["data"]["provider_path"])).expanduser().resolve()
            source = {
                "kind": "built_from_raw_provider",
                "provider_path": str(provider),
                "materialized_partitions": list(partitions),
            }
        else:
            handler = load_source_handler(source_path)
            source_file = Path(source_path).expanduser().resolve()
            source = {
                "kind": "existing_qlib_handler",
                "path": str(source_file),
                "bytes": int(source_file.stat().st_size),
            }
        audit = materialize_compact_handler(
            config,
            handler,
            cache_root=cache_root,
            source=source,
            partitions=partitions,
        )
        rows.append(
            {
                "job_id": config.get("job_id"),
                "seed": config.get("seed"),
                **audit,
            }
        )
        del handler
    return rows
