"""MarketFlow v1.0 task rendering and lazy access to real daily DRIFT snapshots."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
CELL = "alpha158us_sp500_pit"
DATA = ROOT / "data/prompt_data/local"
WINDOW = 20
VERSION = "marketflow-v1.0"
MEMORY = ("RETENTION", "NEW_TRANSMISSION", "RETRANSMISSION")


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=2)
def load_returns(path, digest):
    if file_digest(path) != digest:
        raise ValueError(f"Snapshot changed since sample generation: {path}")
    with np.load(path, allow_pickle=False) as data:
        return data["returns"], data["return_mask"]


class MarketFlowDataset(Dataset):
    def __init__(self, index, split):
        self.index = Path(index).resolve()
        self.rows = []
        sources = {}
        with self.index.open() as stream:
            for n, line in enumerate(stream):
                row = json.loads(line)
                if n == 0 and row.get("format") == "marketflow-index-v2":
                    sources = row["snapshots"]
                    continue
                if row["split"] == split:
                    if row["template_version"] != VERSION or row["lookback"] != WINDOW:
                        raise ValueError("Incompatible training sample")
                    if sources:
                        row["snapshot_sha256"] = sources[row["snapshot"]]
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        values, valid = load_returns(self.index.parent / row["snapshot"], row["snapshot_sha256"])
        positions = row["positions"]
        if values.shape[1] != WINDOW or not valid[positions].all():
            raise ValueError(f"Invalid return window: {row['sample_id']}")
        sample = render_inputs(row, values[positions])
        return dict(sample, answer=row["answer"])


def render_inputs(row, values):
    """Render only history, task, and present-day evidence. Gold labels are a separate argument to loss."""
    if values.shape != (6, WINDOW):
        raise ValueError("Expected six historical return windows")
    series = torch.from_numpy(values)
    if not torch.isfinite(series).all():
        raise ValueError("Non-finite historical input")
    scale = series.std(-1, keepdim=True, unbiased=False)
    series = (series - series.mean(-1, keepdim=True)) / torch.where(scale > 0, scale, 1)
    ticker = row["target"]
    task = row["task_type"]
    if task == "FORECAST":
        post = (
            "TASK=FORECAST\nConsider the target's return from the close one trading day after "
            "the observation date to the close two trading days after it.\nPredict its category "
            "among stocks in the observation-day universe: TOP_20, MIDDLE_60, or BOTTOM_20.\n"
            "Reply with exactly one category."
        )
    elif task == "FLOW_QA":
        edges = "\n".join(f"Signed operator weight from {s} to {ticker}: {v:.9g}."
                          for s, v in zip(row["sender_ids"], row["edges"], strict=True))
        post = (f"{edges}\nTASK=FLOW_QA\nWhich listed sender has the largest absolute directed "
                f"operator weight into {ticker} on the observation date?\n"
                "Reply with exactly one listed sender ticker.")
    elif task == "FLOW_MEMORY":
        norms = "\n".join(f"{label} {v:.9g}" for label, v in zip(MEMORY, row["norms"], strict=True))
        post = (f"Flow component L2 norms for {ticker} on {row['date']}:\n{norms}\nTASK=FLOW_MEMORY\n"
                "Which component has the largest L2 norm in the target's DRIFT flow update on the "
                "observation date?\nReply with exactly one of RETENTION, NEW_TRANSMISSION, or RETRANSMISSION.")
    else:
        raise ValueError(f"Unknown task: {task}")
    return {
        "pre_prompt": (f"You are MarketFlow. Market: SP500. Observation date: {row['date']}. Target: {ticker}.\n"
                       f"Each return series supplied to the language model contains {WINDOW} daily close-to-close "
                       "returns through the observation date, in trading-day order, normalized within that window."),
        "time_series_text": [f"Target {ticker}. Historical daily returns."] +
                            [f"Sender {s}. Historical daily returns." for s in row["sender_ids"]],
        "time_series": list(series.unbind()), "post_prompt": post,
        "task_type": task, "sample_id": row["sample_id"], "sender_ids": row["sender_ids"],
    }
