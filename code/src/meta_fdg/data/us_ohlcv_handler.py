from __future__ import annotations

import hashlib
import json
from typing import Sequence

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL


HANDLER_NAME = "Alpha158USOHLCV"
FEATURE_SET_ID = "qlib-alpha158-minus-vwap0-v1"
EXPECTED_FEATURE_DIM = 157
REMOVED_FEATURES = ("VWAP0",)


def alpha158_us_ohlcv_feature_config() -> dict[str, object]:
    """Return the locked Alpha158-minus-VWAP feature recipe.

    Qlib's default Alpha158 recipe contains exactly one VWAP-backed feature,
    ``VWAP0``.  The US OHLCV protocol removes that feature rather than
    synthesising a proxy from close, typical price, or another observable.
    """

    return {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": ["OPEN", "HIGH", "LOW"],
        },
        "rolling": {},
    }


def alpha158_us_ohlcv_features() -> tuple[list[str], list[str]]:
    fields, names = Alpha158DL.get_feature_config(
        alpha158_us_ohlcv_feature_config()
    )
    fields = [str(value) for value in fields]
    names = [str(value) for value in names]
    validate_alpha158_us_ohlcv_features(fields, names)
    return fields, names


def alpha158_us_ohlcv_feature_sha256() -> str:
    fields, names = alpha158_us_ohlcv_features()
    payload = json.dumps(
        {"fields": fields, "names": names},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_alpha158_us_ohlcv_features(
    fields: Sequence[str], names: Sequence[str]
) -> None:
    if len(fields) != EXPECTED_FEATURE_DIM or len(names) != EXPECTED_FEATURE_DIM:
        raise ValueError(
            "Alpha158-US-OHLCV must contain exactly "
            f"{EXPECTED_FEATURE_DIM} features; observed fields={len(fields)} "
            f"names={len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("Alpha158-US-OHLCV feature names must be unique")
    forbidden_names = [
        name for name in names if "VWAP" in str(name).upper()
    ]
    forbidden_fields = [
        field for field in fields if "$vwap" in str(field).lower()
    ]
    if forbidden_names or forbidden_fields:
        raise ValueError(
            "Alpha158-US-OHLCV must not contain VWAP features or expressions: "
            f"names={forbidden_names} fields={forbidden_fields}"
        )
    if any(name in names for name in REMOVED_FEATURES):
        raise ValueError(f"removed features reappeared: {REMOVED_FEATURES}")


class Alpha158USOHLCV(Alpha158):
    """Qlib Alpha158 with ``VWAP0`` deleted and no proxy substitution."""

    FEATURE_SET_ID = FEATURE_SET_ID
    EXPECTED_FEATURE_DIM = EXPECTED_FEATURE_DIM
    REMOVED_FEATURES = REMOVED_FEATURES

    def get_feature_config(self) -> tuple[list[str], list[str]]:
        return alpha158_us_ohlcv_features()
