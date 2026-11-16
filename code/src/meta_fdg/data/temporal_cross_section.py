from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


PRETRAIN_CONTEXT_START = "2015-07-01"
DEVELOPMENT_SPLITS = ("train", "valid")
REPORTING_SPLITS = ("test",)
FIXED_SPLITS = {
    "train": ("2016-01-01", "2020-12-31"),
    "valid": ("2021-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2025-12-31"),
}
SHIFT2Y_SHORTTEST_CONTRACT = "shift2y_shorttest_v1"
SHIFT2Y_SHORTTEST_CONTEXT_START = "2017-07-01"
SHIFT2Y_SHORTTEST_SPLITS = {
    "train": ("2018-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-12-31"),
}
EXPECTED_TEMPORAL_FEATURE_DIMS = {
    "Alpha158": 158,
    "Alpha158USOHLCV": 157,
    "Alpha360": 360,
    "MasterAlpha158": 221,
}


def expected_temporal_feature_dim(handler_name: str) -> int:
    try:
        return EXPECTED_TEMPORAL_FEATURE_DIMS[str(handler_name)]
    except KeyError as error:
        raise ValueError(
            f"unsupported temporal handler feature contract: {handler_name}"
        ) from error


@dataclass(frozen=True)
class DailyPositions:
    """True public sampler positions for one date, in instrument order."""

    date: pd.Timestamp
    positions: np.ndarray
    instruments: tuple[str, ...]


@dataclass(frozen=True)
class DayCoverage:
    sampler_rows: int
    emitted_rows: int
    dropped_rows: int
    nonfinite_processed_labels: int
    nonfinite_raw_labels: int


@dataclass(frozen=True)
class TemporalDay:
    """One exact daily cross-section emitted by ``TSDatasetH``.

    ``sequences`` is always ``[N, L, F]``.  Rows with a non-finite processed
    or raw target are excluded explicitly and accounted for in ``coverage``;
    feature rows are never silently repaired here.
    """

    split: str
    date: pd.Timestamp
    sequences: torch.Tensor
    target: torch.Tensor
    raw_target: torch.Tensor
    instruments: tuple[str, ...]
    sample_positions: tuple[int, ...]
    coverage: DayCoverage

    @property
    def features(self) -> torch.Tensor:
        return self.sequences

    @property
    def row_count(self) -> int:
        return int(self.sequences.shape[0])

    @property
    def lookback(self) -> int:
        return int(self.sequences.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.sequences.shape[2])


@dataclass(frozen=True)
class SplitCoverage:
    split: str
    sampler_days: int
    emitted_days: int
    sampler_rows: int
    emitted_rows: int
    dropped_rows: int
    nonfinite_processed_labels: int
    nonfinite_raw_labels: int
    date_min: str
    date_max: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalSplit:
    split: str
    days: tuple[TemporalDay, ...]
    coverage: SplitCoverage

    @property
    def feature_dim(self) -> int:
        return self.days[0].feature_dim

    @property
    def lookback(self) -> int:
        return self.days[0].lookback


@dataclass(frozen=True)
class TemporalChunk:
    """Chronological padded days, suitable for one TBPTT chunk."""

    split: str
    sequences: torch.Tensor
    target: torch.Tensor
    raw_target: torch.Tensor
    mask: torch.Tensor
    dates: tuple[pd.Timestamp, ...]
    instruments: tuple[tuple[str, ...], ...]
    sample_positions: tuple[tuple[int, ...], ...]
    row_count: int

    @property
    def features(self) -> torch.Tensor:
        return self.sequences

    @property
    def day_count(self) -> int:
        return int(self.sequences.shape[0])


def _index_levels(index: pd.MultiIndex) -> tuple[str | int, str | int]:
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise TypeError("TSDatasetH sampler index must be a two-level MultiIndex")
    date_level: str | int
    instrument_level: str | int
    if "datetime" in index.names:
        date_level = "datetime"
    else:
        datetime_levels = [
            level
            for level in range(index.nlevels)
            if pd.api.types.is_datetime64_any_dtype(index.get_level_values(level))
        ]
        if len(datetime_levels) != 1:
            raise TypeError("cannot identify the sampler datetime index level")
        date_level = datetime_levels[0]
    if "instrument" in index.names:
        instrument_level = "instrument"
    else:
        date_number = (
            index.names.index(date_level) if isinstance(date_level, str) else date_level
        )
        instrument_level = 1 - date_number
    return date_level, instrument_level


def exact_daily_positions(sampler: Any) -> tuple[DailyPositions, ...]:
    """Group by ``get_index`` positions, never by Qlib's backing offsets.

    Qlib stores time-series arrays instrument-major.  Its internal index-frame
    values are backing-array offsets, not public sample positions.  Enumerating
    ``get_index`` is therefore the only safe way to fetch daily cross-sections.
    """

    index = sampler.get_index()
    date_level, instrument_level = _index_levels(index)
    dates = pd.DatetimeIndex(index.get_level_values(date_level))
    instruments = index.get_level_values(instrument_level).astype(str)
    frame = pd.DataFrame(
        {
            "position": np.arange(len(index), dtype=np.int64),
            "date": dates,
            "instrument": instruments,
        }
    ).sort_values(["date", "instrument"], kind="stable")
    if frame.empty:
        raise ValueError("TSDatasetH sampler contains no public samples")
    groups: list[DailyPositions] = []
    for date, day in frame.groupby("date", sort=True):
        groups.append(
            DailyPositions(
                date=pd.Timestamp(date),
                positions=day["position"].to_numpy(dtype=np.int64, copy=True),
                instruments=tuple(day["instrument"].astype(str)),
            )
        )
    return tuple(groups)


def _single_label(values: pd.Series | pd.DataFrame, *, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        result = values.copy()
    elif isinstance(values, pd.DataFrame) and values.shape[1] == 1:
        result = values.iloc[:, 0].copy()
    else:
        width = values.shape[1] if isinstance(values, pd.DataFrame) else "unknown"
        raise ValueError(f"{name} must contain exactly one label column, found {width}")
    if not isinstance(result.index, pd.MultiIndex) or result.index.nlevels != 2:
        raise TypeError(f"{name} must use the sampler's two-level MultiIndex")
    if not result.index.is_unique:
        raise ValueError(f"{name} index contains duplicate stock-date keys")
    return result


def _align_label(
    label: pd.Series, target_index: pd.MultiIndex, *, name: str
) -> np.ndarray:
    source = label
    if (
        all(level is not None for level in source.index.names)
        and all(level is not None for level in target_index.names)
        and set(source.index.names) == set(target_index.names)
        and source.index.names != target_index.names
    ):
        source = source.reorder_levels(target_index.names)
    try:
        aligned = source.reindex(target_index)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"cannot align {name} to sampler public keys") from exc
    return aligned.to_numpy(dtype=np.float32, copy=True)


def materialize_temporal_split(
    sampler: Any,
    raw_labels: pd.Series | pd.DataFrame,
    *,
    split: str,
    lookback: int,
    feature_dim: int | None = None,
    min_stocks: int = 1,
    expected_period: Sequence[str] | None = None,
) -> TemporalSplit:
    """Materialize one sampler into exact chronological daily tensors."""

    if int(lookback) < 1:
        raise ValueError("lookback must be positive")
    if int(min_stocks) < 1:
        raise ValueError("min_stocks must be positive")
    index = sampler.get_index()
    groups = exact_daily_positions(sampler)
    raw = _single_label(raw_labels, name="raw_labels")
    days: list[TemporalDay] = []
    resolved_feature_dim = None if feature_dim is None else int(feature_dim)

    for group in groups:
        values = np.asarray(sampler[group.positions])
        if values.ndim != 3:
            raise ValueError(
                "TSDatasetH must emit [sample, lookback, feature+label], "
                f"found shape {values.shape}"
            )
        if values.shape[0] != len(group.positions) or values.shape[1] != int(lookback):
            raise ValueError(
                f"TSDatasetH shape differs from exact position/lookback contract: {values.shape}"
            )
        if resolved_feature_dim is None:
            resolved_feature_dim = int(values.shape[2]) - 1
        if resolved_feature_dim < 1 or values.shape[2] != resolved_feature_dim + 1:
            raise ValueError(
                "sampler must contain exactly F feature columns followed by one label"
            )

        sequences = values[:, :, :resolved_feature_dim].astype(np.float32, copy=False)
        processed = values[:, -1, resolved_feature_dim].astype(np.float32, copy=False)
        day_index = index.take(group.positions)
        raw_values = _align_label(raw, day_index, name="raw_labels")
        finite_processed = np.isfinite(processed)
        finite_raw = np.isfinite(raw_values)
        keep = finite_processed & finite_raw
        if not np.isfinite(sequences[keep]).all():
            raise ValueError(f"non-finite temporal features remain on {group.date.date()}")
        emitted = int(keep.sum())
        if emitted < int(min_stocks):
            raise RuntimeError(
                f"{split} {group.date.date()} has {emitted} valid stocks, below {min_stocks}"
            )
        kept_positions = group.positions[keep]
        kept_instruments = tuple(
            instrument
            for instrument, retain in zip(group.instruments, keep, strict=True)
            if retain
        )
        days.append(
            TemporalDay(
                split=str(split),
                date=group.date,
                sequences=torch.from_numpy(np.ascontiguousarray(sequences[keep])),
                target=torch.from_numpy(np.ascontiguousarray(processed[keep])),
                raw_target=torch.from_numpy(np.ascontiguousarray(raw_values[keep])),
                instruments=kept_instruments,
                sample_positions=tuple(int(value) for value in kept_positions),
                coverage=DayCoverage(
                    sampler_rows=len(group.positions),
                    emitted_rows=emitted,
                    dropped_rows=int((~keep).sum()),
                    nonfinite_processed_labels=int((~finite_processed).sum()),
                    nonfinite_raw_labels=int((~finite_raw).sum()),
                ),
            )
        )

    if not days:
        raise RuntimeError(f"TSDatasetH produced no materialized days for {split}")
    if expected_period is not None:
        start, end = (pd.Timestamp(expected_period[0]), pd.Timestamp(expected_period[1]))
        if days[0].date < start or days[-1].date > end:
            raise ValueError(
                f"{split} sampler emitted dates outside [{start.date()}, {end.date()}]"
            )
    coverage = SplitCoverage(
        split=str(split),
        sampler_days=len(groups),
        emitted_days=len(days),
        sampler_rows=sum(day.coverage.sampler_rows for day in days),
        emitted_rows=sum(day.row_count for day in days),
        dropped_rows=sum(day.coverage.dropped_rows for day in days),
        nonfinite_processed_labels=sum(
            day.coverage.nonfinite_processed_labels for day in days
        ),
        nonfinite_raw_labels=sum(day.coverage.nonfinite_raw_labels for day in days),
        date_min=str(days[0].date.date()),
        date_max=str(days[-1].date.date()),
    )
    return TemporalSplit(split=str(split), days=tuple(days), coverage=coverage)


def _pad_chunk(days: Sequence[TemporalDay]) -> TemporalChunk:
    if not days:
        raise ValueError("cannot create an empty temporal chunk")
    split = days[0].split
    if any(day.split != split for day in days):
        raise ValueError("one temporal chunk cannot cross split boundaries")
    lookback = days[0].lookback
    feature_dim = days[0].feature_dim
    if any(day.lookback != lookback or day.feature_dim != feature_dim for day in days):
        raise ValueError("temporal feature shape changed within one chunk")
    day_count = len(days)
    max_stocks = max(day.row_count for day in days)
    sequences = torch.zeros(
        day_count,
        max_stocks,
        lookback,
        feature_dim,
        dtype=days[0].sequences.dtype,
    )
    target = torch.full(
        (day_count, max_stocks), torch.nan, dtype=days[0].target.dtype
    )
    raw_target = torch.full_like(target, torch.nan)
    mask = torch.zeros(day_count, max_stocks, dtype=torch.bool)
    for offset, day in enumerate(days):
        count = day.row_count
        sequences[offset, :count] = day.sequences
        target[offset, :count] = day.target
        raw_target[offset, :count] = day.raw_target
        mask[offset, :count] = True
    return TemporalChunk(
        split=split,
        sequences=sequences,
        target=target,
        raw_target=raw_target,
        mask=mask,
        dates=tuple(day.date for day in days),
        instruments=tuple(day.instruments for day in days),
        sample_positions=tuple(day.sample_positions for day in days),
        row_count=sum(day.row_count for day in days),
    )


class FullEpochIterator(Iterable[TemporalChunk]):
    """Deterministic, no-sampling traversal of every row in one split."""

    shuffle = False
    max_steps_per_epoch = None
    sampling = None

    def __init__(
        self,
        days: Sequence[TemporalDay] | TemporalSplit,
        *,
        chunk_size: int = 20,
        expected_days: int | None = None,
        expected_rows: int | None = None,
    ) -> None:
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive")
        self.days = tuple(days.days if isinstance(days, TemporalSplit) else days)
        if not self.days:
            raise ValueError("full epoch requires at least one day")
        self.chunk_size = int(chunk_size)
        dates = [day.date for day in self.days]
        if dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError("full epoch days must be unique and chronological")
        split = self.days[0].split
        if any(day.split != split for day in self.days):
            raise ValueError("full epoch cannot cross split boundaries")
        for day in self.days:
            if day.row_count != len(day.instruments) or day.row_count != len(
                day.sample_positions
            ):
                raise ValueError("day row metadata is not aligned")
            if len(set(day.instruments)) != len(day.instruments):
                raise ValueError(f"duplicate instruments on {day.date.date()}")
        self.split = split
        self.expected_days = (
            len(self.days) if expected_days is None else int(expected_days)
        )
        actual_rows = sum(day.row_count for day in self.days)
        self.expected_rows = (
            actual_rows if expected_rows is None else int(expected_rows)
        )
        if len(self.days) != self.expected_days:
            raise AssertionError(
                f"full epoch day coverage differs: {len(self.days)} != {self.expected_days}"
            )
        if actual_rows != self.expected_rows:
            raise AssertionError(
                f"full epoch row coverage differs: {actual_rows} != {self.expected_rows}"
            )

    def __len__(self) -> int:
        return (self.expected_days + self.chunk_size - 1) // self.chunk_size

    def __iter__(self) -> Iterator[TemporalChunk]:
        emitted_days = 0
        emitted_rows = 0
        for start in range(0, len(self.days), self.chunk_size):
            chunk = _pad_chunk(self.days[start : start + self.chunk_size])
            emitted_days += chunk.day_count
            emitted_rows += chunk.row_count
            yield chunk
        if emitted_days != self.expected_days or emitted_rows != self.expected_rows:
            raise AssertionError(
                "full epoch iteration did not emit the locked day/row coverage"
            )


def _normalise_splits(splits: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, str]]:
    if set(splits) != {"train", "valid", "test"}:
        raise ValueError("splits must contain exactly train, valid, and test")
    result = {
        str(name): (str(period[0]), str(period[1])) for name, period in splits.items()
    }
    if not (
        pd.Timestamp(result["train"][1]) < pd.Timestamp(result["valid"][0])
        <= pd.Timestamp(result["valid"][1]) < pd.Timestamp(result["test"][0])
    ):
        raise ValueError("splits must be non-overlapping and chronological")
    return result


def _materialized_split_names(
    purpose: str,
    requested_splits: Sequence[str] | None,
) -> tuple[str, ...]:
    purpose = str(purpose)
    if purpose == "development":
        requested = (
            DEVELOPMENT_SPLITS if requested_splits is None else tuple(requested_splits)
        )
        if "test" in requested:
            raise ValueError("development materialization must never include test")
        if tuple(requested) != DEVELOPMENT_SPLITS:
            raise ValueError("development materialization must be exactly train+valid")
        return DEVELOPMENT_SPLITS
    if purpose == "reporting":
        requested = (
            REPORTING_SPLITS if requested_splits is None else tuple(requested_splits)
        )
        if tuple(requested) != REPORTING_SPLITS:
            raise ValueError("reporting materialization must be exactly test")
        return REPORTING_SPLITS
    raise ValueError("purpose must be 'development' or 'reporting'")


def configure_temporal_sampler(
    sampler: Any,
    fillna_type: str | None,
) -> None:
    """Apply the declared within-window fill policy at the sampler boundary.

    Qlib's feature-level ``Fillna`` processor does not fill the synthetic
    history slots emitted by ``TSDataSampler`` for listings or suspensions.
    ``ffill+bfill`` operates only inside each already-causal lookback window;
    it never reads a date later than the sample date and preserves every
    public sampler row instead of silently dropping stocks.
    """

    if fillna_type is None:
        return
    if str(fillna_type) != "ffill+bfill":
        raise ValueError("temporal sampler fillna_type must be 'ffill+bfill' or null")
    if not hasattr(sampler, "config"):
        raise TypeError("temporal sampler does not expose Qlib's config API")
    sampler.config(fillna_type="ffill+bfill")


@dataclass
class TemporalQlibSource:
    """A lazily imported Qlib ``TSDatasetH`` plus strict split permissions."""

    dataset: Any
    handler: Any
    splits: dict[str, tuple[str, str]]
    materialized_splits: tuple[str, ...]
    lookback: int
    handler_start: str
    handler_end: str
    sampler_fillna_type: str | None = None
    min_stocks: int = 1
    expected_feature_dim: int | None = None

    def materialize_split(self, split: str) -> TemporalSplit:
        if split not in self.materialized_splits:
            raise ValueError(
                f"split {split!r} is outside this source's materialization permission"
            )
        # Qlib remains an optional runtime dependency for all sampler-only unit
        # tests; it is imported only at the actual provider boundary.
        from qlib.data.dataset.handler import DataHandlerLP

        sampler = self.dataset.prepare(
            split,
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if not hasattr(sampler, "get_index"):
            raise TypeError("temporal source requires TSDatasetH/TSDataSampler output")
        configure_temporal_sampler(sampler, self.sampler_fillna_type)
        selector = slice(*self.splits[split])
        raw_labels = self.handler.fetch(
            selector,
            col_set=["label"],
            data_key=DataHandlerLP.DK_R,
        )
        return materialize_temporal_split(
            sampler,
            raw_labels,
            split=split,
            lookback=self.lookback,
            min_stocks=self.min_stocks,
            expected_period=self.splits[split],
            feature_dim=self.expected_feature_dim,
        )

    def materialize_all(self) -> dict[str, TemporalSplit]:
        return {name: self.materialize_split(name) for name in self.materialized_splits}


def build_temporal_qlib_source(
    resolved_config: Mapping[str, Any],
    *,
    purpose: str = "development",
    requested_splits: Sequence[str] | None = None,
) -> TemporalQlibSource:
    """Build the exact ``TSDatasetH(step_len=lookback)`` provider wrapper.

    Importing this module does not import Qlib.  Development construction also
    refuses any request containing ``test`` before Qlib initialization occurs.
    """

    data = dict(resolved_config.get("data", resolved_config))
    dataset_config = dict(resolved_config.get("dataset", {}))
    splits = _normalise_splits(data["splits"])
    split_contract = str(resolved_config.get("split_contract", "paper_fixed"))
    if split_contract == "paper_fixed":
        expected_splits = FIXED_SPLITS
        expected_context_start = PRETRAIN_CONTEXT_START
        if splits != expected_splits:
            raise ValueError("temporal source must use the public fixed paper split")
    elif split_contract == SHIFT2Y_SHORTTEST_CONTRACT:
        expected_splits = SHIFT2Y_SHORTTEST_SPLITS
        expected_context_start = SHIFT2Y_SHORTTEST_CONTEXT_START
        if splits != expected_splits:
            raise ValueError(
                "temporal source split does not match shift2y_shorttest_v1"
            )
    else:
        raise ValueError(f"unsupported temporal split contract: {split_contract}")
    names = _materialized_split_names(purpose, requested_splits)
    context_start = str(data.get("context_start_time", expected_context_start))
    if context_start != expected_context_start:
        raise ValueError(
            "temporal handler context must start at "
            f"{expected_context_start}, found {context_start}"
        )
    handler_name = str(data.get("handler", "Alpha158"))
    expected_feature_dim = expected_temporal_feature_dim(handler_name)
    market = str(data.get("market"))
    fair_alpha_markets = (
        resolved_config.get("protocol_id") == "fair-alpha-markets-shift2y-v1"
        and resolved_config.get("tier") == "rolling_development"
        and split_contract == SHIFT2Y_SHORTTEST_CONTRACT
    )
    if fair_alpha_markets:
        if handler_name not in {"Alpha158", "Alpha360"} or market not in {
            "csi300",
            "csi500",
            "csi800",
        }:
            raise ValueError(
                "Stateful fair Alpha-market data requires Alpha158 or Alpha360 "
                "on CSI300, CSI500, or CSI800"
            )
        if (
            data.get("handler_class") != handler_name
            or data.get("handler_module_path") != "qlib.contrib.data.handler"
        ):
            raise ValueError(
                "Stateful fair Alpha-market data must use the declared official "
                "Qlib Alpha handler"
            )
    elif handler_name not in {
        "Alpha158",
        "Alpha158USOHLCV",
        "MasterAlpha158",
    }:
        raise ValueError(
            "Stateful Flow-FDG requires Alpha158, Alpha158USOHLCV, or its "
            "audited MASTER extension"
        )
    if handler_name == "Alpha158USOHLCV" and (
        data.get("handler_class") != "Alpha158USOHLCV"
        or data.get("handler_module_path")
        != "meta_fdg.data.us_ohlcv_handler"
    ):
        raise ValueError(
            "Alpha158USOHLCV must use the audited no-VWAP handler adapter"
        )
    if handler_name == "MasterAlpha158" and (
        data.get("handler_class") != "MasterAlpha158"
        or data.get("handler_module_path")
        != "meta_fdg.baselines.master_qlib"
    ):
        raise ValueError(
            "MasterAlpha158 must use the audited in-repository handler adapter"
        )
    dataset_kwargs = dict(dataset_config.get("kwargs", {}))
    lookback = int(dataset_kwargs.get("step_len", data.get("lookback", 20)))
    if lookback < 1:
        raise ValueError("TSDatasetH step_len/lookback must be positive")
    if dataset_config and dataset_config.get("class", "TSDatasetH") != "TSDatasetH":
        raise ValueError("temporal source must wrap Qlib TSDatasetH")
    sampler_fillna_type = dataset_config.get("sampler_fillna_type")
    if sampler_fillna_type not in (None, "ffill+bfill"):
        raise ValueError(
            "dataset.sampler_fillna_type must be 'ffill+bfill' or null"
        )
    expected_handler_end = (
        splits["valid"][1] if purpose == "development" else splits["test"][1]
    )
    configured_handler_end = str(data.get("materialization_end_time", ""))
    if configured_handler_end != expected_handler_end:
        raise ValueError(
            f"{purpose} materialization_end_time must be {expected_handler_end}, "
            f"found {configured_handler_end or 'missing'}"
        )

    segments = {name: splits[name] for name in names}
    if bool(data.get("use_compact_materialization", False)):
        from meta_fdg.data.compact_qlib import (
            CompactTSDatasetH,
            compact_handler_from_config,
        )

        handler = compact_handler_from_config(
            resolved_config, include_test=purpose == "reporting"
        )
        if handler.feature_dim != expected_feature_dim:
            raise ValueError(
                "compact temporal handler feature identity mismatch: "
                f"handler={handler_name} expected={expected_feature_dim} "
                f"observed={handler.feature_dim}"
            )
        dataset = CompactTSDatasetH(
            handler=handler, segments=segments, step_len=lookback
        )
        if int(getattr(dataset, "step_len", -1)) != lookback:
            raise RuntimeError("Qlib TSDatasetH did not preserve the locked lookback")
        return TemporalQlibSource(
            dataset=dataset,
            handler=handler,
            splits=splits,
            materialized_splits=names,
            lookback=lookback,
            handler_start=context_start,
            handler_end=configured_handler_end,
            sampler_fillna_type=sampler_fillna_type,
            min_stocks=int(data.get("min_stocks", 1)),
            expected_feature_dim=expected_feature_dim,
        )

    import qlib
    from qlib.data.dataset import TSDatasetH

    provider = Path(data["provider_path"]).expanduser().resolve()
    region = str(data.get("region", "cn"))
    qlib_kernels = data.get("qlib_kernels")
    init_kwargs: dict[str, Any] = {}
    if qlib_kernels is not None:
        if int(qlib_kernels) < 1:
            raise ValueError("data.qlib_kernels must be positive when provided")
        init_kwargs["kernels"] = int(qlib_kernels)
    qlib.init(provider_uri=str(provider), region=region, **init_kwargs)
    if handler_name == "Alpha158":
        from qlib.contrib.data.handler import Alpha158

        handler_type = Alpha158
    elif handler_name == "Alpha158USOHLCV":
        from meta_fdg.data.us_ohlcv_handler import Alpha158USOHLCV

        handler_type = Alpha158USOHLCV
    elif handler_name == "Alpha360":
        from qlib.contrib.data.handler import Alpha360

        handler_type = Alpha360
    else:
        from meta_fdg.baselines.master_qlib import MasterAlpha158

        handler_type = MasterAlpha158
    handler_kwargs = dict(data.get("handler_kwargs", {}))
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
    # Development stops at validation by construction.  Qlib's expression
    # engine obtains the small future window needed by Ref labels internally;
    # no test sampler or test feature frame is processed for selection.
    handler_end = configured_handler_end
    handler = handler_type(
        instruments=str(data["market"]),
        start_time=context_start,
        end_time=handler_end,
        fit_start_time=splits["train"][0],
        fit_end_time=splits["train"][1],
        label=([str(data["label"])], ["LABEL0"]),
        **handler_kwargs,
    )
    observed_feature_dim = len(handler.get_cols(col_set="feature"))
    if observed_feature_dim != expected_feature_dim:
        raise ValueError(
            "temporal handler feature identity mismatch: "
            f"handler={handler_name} expected={expected_feature_dim} "
            f"observed={observed_feature_dim}"
        )
    dataset = TSDatasetH(handler=handler, segments=segments, step_len=lookback)
    if int(getattr(dataset, "step_len", -1)) != lookback:
        raise RuntimeError("Qlib TSDatasetH did not preserve the locked lookback")
    return TemporalQlibSource(
        dataset=dataset,
        handler=handler,
        splits=splits,
        materialized_splits=names,
        lookback=lookback,
        handler_start=context_start,
        handler_end=handler_end,
        sampler_fillna_type=sampler_fillna_type,
        min_stocks=int(data.get("min_stocks", 1)),
        expected_feature_dim=expected_feature_dim,
    )
