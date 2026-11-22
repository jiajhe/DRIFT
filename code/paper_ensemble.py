#!/usr/bin/env python3
"""Run the fixed four-seed ensemble used in the paper's main comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from meta_fdg.data import build_temporal_qlib_source
from meta_fdg.training.flow_fdg import build_flow_fdg_model, evaluate_flow_fdg

MANIFEST = ROOT / "checkpoints" / "paper_ensemble" / "manifest.json"


def combine_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Standardize each seed's daily cross-section and equally average scores."""
    indexed = [frame.set_index(["date", "instrument"], verify_integrity=True) for frame in frames]
    labels = indexed[0]["label"]
    for frame in indexed[1:]:
        if len(frame) != len(labels) or not frame["label"].reindex(labels.index).equals(labels):
            raise ValueError("Seed predictions must cover the same dates, stocks and labels")
    scores = pd.concat([frame["score"] for frame in indexed], axis=1).astype("float64")
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Seed predictions contain nonfinite scores")
    daily = scores.groupby(level="date", sort=False)
    scale = daily.transform("std", ddof=0).replace(0.0, 1.0)
    normalized = (scores - daily.transform("mean")) / scale
    return pd.DataFrame({"label": labels, "score": normalized.mean(axis=1)}).reset_index()


def load_members(setting: dict, seeds: list[int], device: torch.device):
    if [member["seed"] for member in setting["members"]] != seeds:
        raise ValueError("Checkpoint seeds do not match the ensemble manifest")
    for member in setting["members"]:
        path = ROOT / member["checkpoint"]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != member["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["seed"], checkpoint["epoch"]) != (member["seed"], member["epoch"]):
            raise ValueError(f"Checkpoint seed or epoch mismatch: {path}")
        model = build_flow_fdg_model(checkpoint["feature_dim"], checkpoint["model_config"], device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        yield model.eval(), checkpoint


@torch.no_grad()
def smoke_test(setting: dict, seeds: list[int], device: torch.device) -> int:
    frames = []
    for model, checkpoint in load_members(setting, seeds, device):
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(8, checkpoint["lookback"], checkpoint["feature_dim"], generator=generator)
        output = model(features.to(device), compute_graph_diagnostics=False).prediction.cpu()
        if output.shape != (8,):
            raise ValueError(f"Unexpected prediction shape: {output.shape}")
        frames.append(pd.DataFrame({
            "date": pd.Timestamp("2024-01-02"),
            "instrument": [f"stock_{i}" for i in range(8)],
            "label": 0.0,
            "score": output.numpy(),
        }))
    return len(combine_predictions(frames))


def predict_test(setting: dict, seeds: list[int], device: torch.device) -> pd.DataFrame:
    config = json.loads((ROOT / setting["runtime_config"]).read_text(encoding="utf-8"))
    for field in ("provider_path", "cache_root", "materialized_cache_root"):
        if field in config["data"]:
            config["data"][field] = str((ROOT / "code" / config["data"][field]).resolve())
    warmup_count = int(config["test_report"]["valid_warmup_days"])
    valid = build_temporal_qlib_source(config).materialize_split("valid")
    warmup = valid.days[-warmup_count:] if warmup_count else ()
    del valid
    reporting = copy.deepcopy(config)
    reporting["data"]["materialization_end_time"] = config["data"]["splits"]["test"][1]
    reporting["data"]["qlib_kernels"] = 1
    test = build_temporal_qlib_source(reporting, purpose="reporting").materialize_split("test")

    frames = []
    for model, checkpoint in load_members(setting, seeds, device):
        if (test.feature_dim, test.lookback) != (checkpoint["feature_dim"], checkpoint["lookback"]):
            raise ValueError("Test data shape does not match the checkpoint")
        result = evaluate_flow_fdg(model, test, warmup_days=warmup, device=device,
                                   loss_config=config["loss"], split_name="test")
        frames.append(result.predictions_full)
    return combine_predictions(frames)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=manifest["settings"], required=True)
    parser.add_argument("--device", default="cpu")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true", help="Run one synthetic cross-section")
    mode.add_argument("--output", type=Path, help="Write full test predictions to Parquet")
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)
    setting = manifest["settings"][args.cell]
    seeds = manifest["seeds"]
    if args.smoke_test:
        count = smoke_test(setting, seeds, device)
    else:
        predictions = predict_test(setting, seeds, device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(args.output, index=False)
        count = len(predictions)
    print(json.dumps({"cell": args.cell, "seeds": seeds, "device": str(device),
                      "smoke_test": args.smoke_test, "rows": count}))


if __name__ == "__main__":
    main()
