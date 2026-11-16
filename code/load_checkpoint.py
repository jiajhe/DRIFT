#!/usr/bin/env python3
"""Load the compact DRIFT checkpoint without starting training."""
from pathlib import Path
import argparse, copy, json, sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))
def load_model(cell="alpha158_csi300", device="cpu"):
    from meta_fdg.training.flow_fdg import build_flow_fdg_model
    index = json.loads((ROOT / "MANIFEST.json").read_text())
    entry = next((r for r in index["models"] if r["cell"] == cell), None)
    if entry is None: raise ValueError(f"Unknown cell: {cell}")
    checkpoint = torch.load(ROOT / entry["checkpoint"], map_location="cpu", weights_only=True)
    model = build_flow_fdg_model(int(checkpoint["feature_dim"]), checkpoint["model_config"], torch.device(device))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    config = copy.deepcopy(checkpoint["resolved_config"])
    code_root = ROOT / "code"
    for key in ("provider_path", "cache_root", "materialized_cache_root"):
        if key in config["data"]:
            config["data"][key] = str((code_root / config["data"][key]).resolve())
    config["model_config"] = str((code_root / config["model_config"]).resolve())
    return model, config, checkpoint
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="alpha158_csi300")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, config, checkpoint = load_model(args.cell, args.device)
    print(json.dumps({"cell": args.cell, "epoch": checkpoint["epoch"], "feature_dim": checkpoint["feature_dim"],
        "lookback": checkpoint["lookback"], "parameters": sum(p.numel() for p in model.parameters()),
        "provider": str(Path(config["data"]["provider_path"]).relative_to(ROOT)), "loaded": True}, indent=2))
if __name__ == "__main__": main()
