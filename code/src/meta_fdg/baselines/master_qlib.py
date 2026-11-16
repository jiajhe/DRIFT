from __future__ import annotations

import copy
import importlib.util
import itertools
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP


MASTER_REPOSITORY = "https://github.com/SJTU-DMTai/MASTER"
MASTER_REVISION = "de8f58557096abde4216a701b35fc4368158d111"
DEFAULT_MASTER_SOURCE_ROOT = Path(
    os.environ.get("META_FDG_MASTER_SOURCE", str(Path(__file__).resolve().parents[4] / "third_party" / "MASTER"))
)
MARKET_CODES = ("SH000300", "SH000905", "SH000906")
MARKET_WINDOWS = (5, 10, 20, 30, 60)


def _source_revision(source_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot verify MASTER source checkout: {source_root}") from error


def verify_master_source(
    source_root: str | Path = DEFAULT_MASTER_SOURCE_ROOT,
    expected_revision: str = MASTER_REVISION,
) -> Path:
    root = Path(source_root).expanduser().resolve()
    source_file = root / "qlib-update" / "pytorch_master_ts.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"MASTER author source is missing: {source_file}")
    actual_revision = _source_revision(root)
    if actual_revision != expected_revision:
        raise RuntimeError(
            "MASTER source revision mismatch: "
            f"expected {expected_revision}, got {actual_revision} at {root}"
        )
    return source_file


def _load_author_module() -> Any:
    source_file = verify_master_source()
    module_name = "qlib.contrib.model._meta_fdg_master_de8f585"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MASTER author source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def market_feature_config() -> tuple[list[str], list[str]]:
    fields: list[str] = []
    names: list[str] = []
    for code in MARKET_CODES:
        fields.append(f'Mask($close/Ref($close,1)-1, "{code}")')
        names.append(f"MKT_{code}_RET1")
        for window in MARKET_WINDOWS:
            fields.extend(
                [
                    f'Mask(Mean($close/Ref($close,1)-1,{window}), "{code}")',
                    f'Mask(Std($close/Ref($close,1)-1,{window}), "{code}")',
                    f'Mask(Mean($amount,{window})/$amount, "{code}")',
                    f'Mask(Std($amount,{window})/$amount, "{code}")',
                ]
            )
            names.extend(
                [
                    f"MKT_{code}_RET_MEAN{window}",
                    f"MKT_{code}_RET_STD{window}",
                    f"MKT_{code}_AMOUNT_MEAN{window}",
                    f"MKT_{code}_AMOUNT_STD{window}",
                ]
            )
    if len(fields) != 63 or len(names) != 63:
        raise AssertionError("MASTER market feature contract must contain 63 fields")
    return fields, names


class MasterAlpha158(Alpha158):
    """Author-compatible Alpha158 handler with 63 current-provider market fields."""

    def get_feature_config(self) -> tuple[list[str], list[str]]:
        fields, names = super().get_feature_config()
        market_fields, market_names = market_feature_config()
        return list(fields) + market_fields, list(names) + market_names


_AUTHOR_MODULE = _load_author_module()
_AuthorMASTERModel = _AUTHOR_MODULE.MASTERModel
_AuthorMASTERNetwork = _AUTHOR_MODULE.MASTER


class _AlphaOnlyMASTER(_AuthorMASTERNetwork):
    """MASTER core with a trainable global gate when market fields are absent.

    The author's gate consumes 63 benchmark-market fields.  The US provider is
    intentionally restricted to the common Alpha158-minus-VWAP factor universe,
    so feeding current or synthetic market fields would break the comparison
    contract.  This variant keeps the temporal and cross-sectional MASTER
    blocks unchanged and replaces only that unavailable conditional gate with a
    seed-trained, sample-independent feature gate.
    """

    def __init__(
        self,
        *,
        d_feat: int,
        d_model: int,
        t_nhead: int,
        s_nhead: int,
        T_dropout_rate: float,
        S_dropout_rate: float,
        beta: float,
    ) -> None:
        super().__init__(
            d_feat=d_feat,
            d_model=d_model,
            t_nhead=t_nhead,
            s_nhead=s_nhead,
            T_dropout_rate=T_dropout_rate,
            S_dropout_rate=S_dropout_rate,
            gate_input_start_index=d_feat,
            gate_input_end_index=d_feat + 1,
            beta=beta,
        )
        del self.feature_gate
        self.alpha_feature_dim = int(d_feat)
        self.alpha_gate_temperature = float(beta)
        self.alpha_gate_logits = torch.nn.Parameter(torch.zeros(d_feat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or int(x.shape[-1]) != self.alpha_feature_dim:
            raise ValueError(
                "Alpha-only MASTER input width mismatch: "
                f"expected {self.alpha_feature_dim}, observed {tuple(x.shape)}"
            )
        gate = self.alpha_feature_dim * torch.softmax(
            self.alpha_gate_logits / self.alpha_gate_temperature,
            dim=-1,
        )
        value = x * gate.view(1, 1, -1)
        value = self.x2y(value)
        value = self.pe(value)
        value = self.tatten(value)
        value = self.satten(value)
        value = self.temporalatten(value)
        return self.decoder(value).squeeze(-1)


class _ExactDailySampler(Sampler[np.ndarray]):
    """Yield exact sample positions for each date in an instrument-major sampler."""

    def __init__(self, data_source: Any, *, shuffle: bool) -> None:
        index = data_source.get_index()
        date_level = "datetime" if "datetime" in index.names else index.names[0]
        positions = pd.Series(np.arange(len(index)), index=index)
        self.batches = [
            frame.to_numpy(dtype=np.int64)
            for _, frame in positions.groupby(level=date_level, sort=True)
        ]
        self.shuffle = shuffle

    def __iter__(self):
        order = np.arange(len(self.batches))
        if self.shuffle:
            np.random.shuffle(order)
        for index in order:
            yield self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)


class MasterModelAdapter(_AuthorMASTERModel):
    """Thin runner adapter around the author's corrected Qlib MASTER model.

    The network, loss, optimizer, daily batching, label trimming, and label
    normalization remain author code.  This adapter repairs the author's
    checkpoint path by selecting the epoch with the highest validation IC and
    exposes Qlib's conventional ``segment`` prediction argument.
    """

    def __init__(
        self,
        *args: Any,
        source_root: str | Path = DEFAULT_MASTER_SOURCE_ROOT,
        source_revision: str = MASTER_REVISION,
        max_steps_per_epoch: int | None = None,
        alpha_only: bool = False,
        **kwargs: Any,
    ) -> None:
        verify_master_source(source_root, source_revision)
        if max_steps_per_epoch is not None and int(max_steps_per_epoch) < 1:
            raise ValueError("max_steps_per_epoch must be positive")
        requested_beta = kwargs.get("beta")
        if requested_beta is not None and float(requested_beta) <= 0:
            raise ValueError("MASTER beta must be positive")
        self.max_steps_per_epoch = (
            None if max_steps_per_epoch is None else int(max_steps_per_epoch)
        )
        super().__init__(*args, **kwargs)
        # The released author wrapper overwrites every explicit CSI300 beta
        # with 5.  That makes the documented beta search a no-op.  Preserve
        # the public architecture while restoring the caller's explicit
        # temperature on the already-created feature gate.
        if requested_beta is not None:
            self.beta = float(requested_beta)
            self.model.feature_gate.t = float(requested_beta)
        self.alpha_only = bool(alpha_only)
        if self.alpha_only:
            if int(self.d_feat) < 1:
                raise ValueError("Alpha-only MASTER requires a positive feature width")
            self.model = _AlphaOnlyMASTER(
                d_feat=int(self.d_feat),
                d_model=int(self.d_model),
                t_nhead=int(self.t_nhead),
                s_nhead=int(self.s_nhead),
                T_dropout_rate=float(self.T_dropout_rate),
                S_dropout_rate=float(self.S_dropout_rate),
                beta=float(self.beta),
            ).to(self.device)
            self.train_optimizer = torch.optim.Adam(
                self.model.parameters(), self.lr
            )

    def _init_data_loader(
        self, data: Any, shuffle: bool = True, drop_last: bool = True
    ) -> DataLoader:
        # The author's sampler assumes Qlib's TSDataSampler is date-major, but
        # it is instrument-major.  Preserve daily cross-sectional batching by
        # yielding the actual positions for each date.
        return DataLoader(
            data,
            sampler=_ExactDailySampler(data, shuffle=shuffle),
            drop_last=drop_last,
        )

    def _mean_daily_ic(self, sampler: Any) -> float:
        """Return the evaluator-compatible mean daily Pearson IC."""
        daily_ics: list[float] = []
        self.model.eval()
        for positions in _ExactDailySampler(sampler, shuffle=False):
            data = torch.from_numpy(np.asarray(sampler[positions]))
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].detach().cpu().numpy().reshape(-1)
            if not torch.isfinite(feature).all():
                raise RuntimeError(
                    "MASTER received non-finite validation features after sampler fill"
                )
            with torch.no_grad():
                score = (
                    self.model(feature.float())
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
            finite = np.isfinite(score) & np.isfinite(label)
            if int(finite.sum()) < 2:
                continue
            finite_score = score[finite]
            finite_label = label[finite]
            if np.std(finite_score) == 0.0 or np.std(finite_label) == 0.0:
                continue
            daily_ic = float(np.corrcoef(finite_score, finite_label)[0, 1])
            if np.isfinite(daily_ic):
                daily_ics.append(daily_ic)
        if not daily_ics:
            raise RuntimeError("MASTER produced no finite daily validation IC values")
        return float(np.mean(daily_ics))

    def fit(
        self,
        dataset: Any,
        evals_result: dict[str, Any] | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        train.config(fillna_type="ffill+bfill")
        valid = dataset.prepare(
            "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
        )
        valid.config(fillna_type="ffill+bfill")
        train_loader = self._init_data_loader(train, shuffle=True, drop_last=True)
        self.fitted = True
        train_losses: list[float] = []
        valid_ics: list[float] = []
        best_param: dict[str, Any] | None = None
        best_valid_ic = -np.inf
        best_epoch = 0

        for epoch in range(1, self.n_epochs + 1):
            epoch_loader = (
                train_loader
                if self.max_steps_per_epoch is None
                else itertools.islice(train_loader, self.max_steps_per_epoch)
            )
            train_loss = float(self.train_epoch(epoch_loader))
            if not np.isfinite(train_loss):
                raise RuntimeError(f"MASTER produced non-finite train loss: {train_loss}")
            train_losses.append(train_loss)
            valid_ic = self._mean_daily_ic(valid)
            valid_ics.append(valid_ic)
            if valid_ic > best_valid_ic:
                best_valid_ic = valid_ic
                best_epoch = epoch
                best_param = copy.deepcopy(self.model.state_dict())
            print(
                f"MASTER epoch={epoch} train_loss={train_loss:.9f} "
                f"valid_ic={valid_ic:.9f} best_epoch={best_epoch}",
                flush=True,
            )
            if (
                self.train_stop_loss_thred is not None
                and train_loss <= self.train_stop_loss_thred
            ):
                break

        if best_param is None:
            raise RuntimeError("MASTER failed to select a validation checkpoint")
        self.model.load_state_dict(best_param)
        if evals_result is not None:
            evals_result["train"] = train_losses
            evals_result["valid_ic"] = valid_ics
            evals_result["best_epoch"] = best_epoch
            evals_result["best_valid_ic"] = best_valid_ic
            evals_result["selection_metric"] = "valid.IC"
        if save_path is not None:
            checkpoint = Path(save_path).expanduser()
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_param, checkpoint)

    def predict(self, dataset: Any, segment: str = "test") -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("model is not fitted yet")
        sampler = dataset.prepare(
            segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
        )
        sampler.config(fillna_type="ffill+bfill")
        predictions: list[pd.Series] = []
        index = sampler.get_index()
        self.model.eval()
        for positions in _ExactDailySampler(sampler, shuffle=False):
            data = torch.from_numpy(np.asarray(sampler[positions]))
            feature = data[:, :, 0:-1].to(self.device)
            if not torch.isfinite(feature).all():
                raise RuntimeError(
                    "MASTER received non-finite time-series features after sampler fill"
                )
            with torch.no_grad():
                prediction = self.model(feature.float()).detach().cpu().numpy()
            predictions.append(
                pd.Series(prediction.reshape(-1), index=index[positions])
            )
        if not predictions:
            raise RuntimeError(f"MASTER produced no predictions for {segment}")
        return pd.concat(predictions).sort_index().to_frame("score")
