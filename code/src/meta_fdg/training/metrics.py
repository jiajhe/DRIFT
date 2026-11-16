from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid].astype(np.float64, copy=False)
    right = right[valid].astype(np.float64, copy=False)
    if left.size < 2:
        return math.nan
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right / denominator) if denominator > 1e-12 else math.nan


def daily_signal_metrics(score: np.ndarray, label: np.ndarray, long_fraction: float = 0.2) -> dict[str, float]:
    valid = np.isfinite(score) & np.isfinite(label)
    score = score[valid]
    label = label[valid]
    count = int(score.size)
    if count < 2:
        return {"IC": math.nan, "RankIC": math.nan, "LongShortReturn": math.nan, "n": count}
    score_rank = pd.Series(score).rank(method="average").to_numpy(dtype=np.float64)
    label_rank = pd.Series(label).rank(method="average").to_numpy(dtype=np.float64)
    tail = max(1, int(math.floor(count * long_fraction)))
    order = np.argsort(score)
    long_short = float(label[order[-tail:]].mean() - label[order[:tail]].mean())
    return {
        "IC": _correlation(score, label),
        "RankIC": _correlation(score_rank, label_rank),
        "LongShortReturn": long_short,
        "n": count,
    }


def summarize_daily_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    payload: dict[str, float | int] = {"n_days": int(len(frame))}
    for column, prefix in (("IC", "IC"), ("RankIC", "RankIC")):
        values = frame[column].dropna().to_numpy(dtype=np.float64)
        mean = float(values.mean()) if values.size else math.nan
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        payload[prefix] = mean
        payload[f"{prefix}_std"] = std
        payload[f"{prefix}IR"] = float(mean / std) if std > 1e-12 else 0.0
    returns = frame["LongShortReturn"].dropna().to_numpy(dtype=np.float64)
    payload["LongShortReturn"] = float(np.prod(1.0 + returns) - 1.0) if returns.size else math.nan
    payload["n_obs_mean"] = float(frame["n"].mean()) if len(frame) else 0.0
    return payload
