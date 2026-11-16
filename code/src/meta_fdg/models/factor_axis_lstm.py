"""Qlib-compatible factor-axis LSTM building blocks for Flow-FDG.

The existing Flow-FDG encoder treats the outer rolling window as the recurrent
axis.  Qlib's strong Alpha360 and Alpha158 LSTM baselines use a different
orientation:

* Alpha360 reshapes the latest 360-vector into 60 steps with 6 inputs.
* Alpha158 reshapes the 20-by-158 rolling window into 158 steps with 20 inputs.

This module captures those layouts without changing the default Flow-FDG
architecture.  The integration remains opt-in so an experiment must state the
layout explicitly in its resolved configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn


ALPHA360_LAST_SNAPSHOT = "alpha360_last_snapshot"
ALPHA158_TEMPORAL_FACTOR = "alpha158_temporal_factor"
SUPPORTED_FACTOR_AXIS_LAYOUTS = {
    ALPHA360_LAST_SNAPSHOT,
    ALPHA158_TEMPORAL_FACTOR,
}


def orient_factor_axis_features(
    features: torch.Tensor,
    *,
    layout: str,
    input_dim: int,
    outer_lookback: int,
    alpha360_field_count: int = 6,
) -> torch.Tensor:
    """Apply one locked Qlib factor-axis tensor orientation."""

    if layout not in SUPPORTED_FACTOR_AXIS_LAYOUTS:
        raise ValueError(
            f"unsupported factor-axis layout {layout!r}; "
            f"expected one of {sorted(SUPPORTED_FACTOR_AXIS_LAYOUTS)}"
        )
    if features.ndim != 3:
        raise ValueError("features must be [stock, lookback, feature]")
    if features.shape[1] != int(outer_lookback):
        raise ValueError(
            f"lookback {features.shape[1]} != configured {outer_lookback}"
        )
    if features.shape[2] != int(input_dim):
        raise ValueError(
            f"feature dimension {features.shape[2]} != configured {input_dim}"
        )
    if not torch.isfinite(features).all():
        raise FloatingPointError("factor-axis features must be finite")
    if layout == ALPHA360_LAST_SNAPSHOT:
        if int(input_dim) % int(alpha360_field_count) != 0:
            raise ValueError(
                "Alpha360 feature dimension must be divisible by the field count"
            )
        latest = features[:, -1, :]
        return latest.reshape(
            latest.shape[0], int(alpha360_field_count), -1
        ).permute(0, 2, 1)
    return features.permute(0, 2, 1)


@dataclass(frozen=True)
class FactorAxisContract:
    """Resolved tensor contract for an opt-in factor-axis encoder."""

    layout: str
    input_dim: int
    outer_lookback: int
    recurrent_steps: int
    recurrent_input_size: int
    hidden_dim: int
    num_layers: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


class FactorAxisLSTMEncoder(nn.Module):
    """LSTM encoder matching the tensor orientations used by Qlib baselines."""

    def __init__(
        self,
        *,
        input_dim: int,
        outer_lookback: int,
        layout: str,
        hidden_dim: int = 64,
        num_layers: int = 2,
        alpha360_field_count: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(outer_lookback) <= 0:
            raise ValueError("outer_lookback must be positive")
        if layout not in SUPPORTED_FACTOR_AXIS_LAYOUTS:
            raise ValueError(
                f"unsupported factor-axis layout {layout!r}; "
                f"expected one of {sorted(SUPPORTED_FACTOR_AXIS_LAYOUTS)}"
            )
        if int(hidden_dim) <= 0 or int(num_layers) <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if int(alpha360_field_count) <= 0:
            raise ValueError("alpha360_field_count must be positive")

        self.input_dim = int(input_dim)
        self.outer_lookback = int(outer_lookback)
        self.layout = str(layout)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.alpha360_field_count = int(alpha360_field_count)
        self.dropout = float(dropout)

        if self.layout == ALPHA360_LAST_SNAPSHOT:
            if self.input_dim % self.alpha360_field_count != 0:
                raise ValueError(
                    "Alpha360 feature dimension must be divisible by the field count"
                )
            recurrent_steps = self.input_dim // self.alpha360_field_count
            recurrent_input_size = self.alpha360_field_count
        else:
            recurrent_steps = self.input_dim
            recurrent_input_size = self.outer_lookback

        self.contract = FactorAxisContract(
            layout=self.layout,
            input_dim=self.input_dim,
            outer_lookback=self.outer_lookback,
            recurrent_steps=recurrent_steps,
            recurrent_input_size=recurrent_input_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.recurrent = nn.LSTM(
            input_size=recurrent_input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

    def orient(self, features: torch.Tensor) -> torch.Tensor:
        """Return the exact recurrent tensor consumed by the Qlib LSTM."""

        return orient_factor_axis_features(
            features,
            layout=self.layout,
            input_dim=self.input_dim,
            outer_lookback=self.outer_lookback,
            alpha360_field_count=self.alpha360_field_count,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        oriented = self.orient(features)
        _, (hidden, _) = self.recurrent(oriented)
        return hidden[-1]

    def structure_dict(self) -> dict[str, int | float | str]:
        return {
            **self.contract.to_dict(),
            "architecture": "lstm",
            "dropout": self.dropout,
            "latest_outer_snapshot_only": (
                self.layout == ALPHA360_LAST_SNAPSHOT
            ),
        }


def load_qlib_lstm_state(
    *,
    encoder: FactorAxisLSTMEncoder,
    local_head: nn.Linear,
    qlib_state: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Strictly import a Qlib LSTM ``rnn`` and ``fc_out`` state.

    The function accepts an already loaded state mapping so the caller remains
    responsible for file hashing and safe checkpoint deserialization.  Only
    the recurrent encoder and one linear local head are whitelisted.
    """

    expected_recurrent = {
        f"rnn.{name}" for name in encoder.recurrent.state_dict()
    }
    expected_head = {f"fc_out.{name}" for name in local_head.state_dict()}
    expected = expected_recurrent | expected_head
    observed = set(qlib_state)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            "Qlib LSTM state violates the strict rnn/fc_out whitelist: "
            f"missing={missing}, unexpected={unexpected}"
        )

    recurrent_state = {
        key.removeprefix("rnn."): qlib_state[key]
        for key in sorted(expected_recurrent)
    }
    head_state = {
        key.removeprefix("fc_out."): qlib_state[key]
        for key in sorted(expected_head)
    }
    try:
        encoder.recurrent.load_state_dict(recurrent_state, strict=True)
        local_head.load_state_dict(head_state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Qlib LSTM checkpoint shapes do not match the factor-axis contract"
        ) from error

    return {
        "schema": "factor_axis_qlib_lstm_import_v1",
        "layout": encoder.layout,
        "contract": encoder.contract.to_dict(),
        "loaded_keys": sorted(expected),
        "strict": True,
        "local_head": "linear",
    }


__all__ = [
    "ALPHA158_TEMPORAL_FACTOR",
    "ALPHA360_LAST_SNAPSHOT",
    "SUPPORTED_FACTOR_AXIS_LAYOUTS",
    "FactorAxisContract",
    "FactorAxisLSTMEncoder",
    "load_qlib_lstm_state",
    "orient_factor_axis_features",
]
