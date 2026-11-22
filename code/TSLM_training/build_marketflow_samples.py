"""Index F/Q/M tasks after date partitioning; price lookahead is confined to supervision."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from marketflow_data import CELL, DATA, MEMORY, ROOT, VERSION, WINDOW, file_digest


def partition(calendar, splits, date):
    pos = calendar.index(date)
    for split, (start, end) in splits.items():
        if start <= date <= end and pos + 2 < len(calendar) and calendar[pos + 2] <= end:
            return split, pos
    return None, pos


def forecast_labels(provider, date, pos):
    intervals = (provider / "instruments/sp500_pit.txt").read_text().splitlines()
    members = {s for s, start, end in map(str.split, intervals) if start <= date <= end}
    y = {}
    for symbol in sorted(members):
        path = provider / "features" / symbol.lower() / "close.day.bin"
        if not path.is_file():
            continue
        raw = np.fromfile(path, dtype="<f4")
        j = pos + 2 - int(raw[0])  # offset header, then price for t+1
        if 1 <= j < len(raw) - 1:
            a, b = map(float, raw[j:j+2])
            if np.isfinite([a, b]).all() and min(a, b) > 0:
                y[symbol] = b / a - 1
    ranks = (pd.Series(y, dtype=float).rank(method="average") - 0.5) / max(len(y), 1)
    return {s: ("BOTTOM_20" if rank < 0.2 else "TOP_20" if rank >= 0.8 else "MIDDLE_60")
            for s, rank in ranks.items()}


def observations(snapshot, destination, config, seed, checksum=None):
    provider = ROOT / "code" / config["provider_path"]
    calendar = (provider / "calendars/day.txt").read_text().splitlines()
    checksum = checksum or file_digest(snapshot)
    with np.load(snapshot, allow_pickle=False) as data:
        meta = json.loads(data["metadata"].item())
        date = meta["date"]
        split, pos = partition(calendar, config["splits"], date)
        if split is None:
            return
        if (meta["cell"] != CELL or meta["release"] != provider.name or meta.get("lookback") != WINDOW
                or data["return_dates"].tolist() != calendar[pos-WINDOW+1:pos+1]):
            raise ValueError(f"Invalid source or dates in {snapshot}")
        names = data["instruments"]
        weights = data["K"]
        norms = np.linalg.norm(data["components"], axis=2).T
        valid = data["return_mask"].all(axis=1)
        if data["returns"].shape != (len(names), WINDOW) or len(np.unique(names)) != len(names):
            raise ValueError("Invalid universe or window")
    labels = forecast_labels(provider, date, pos)
    for i, ticker in enumerate(names.tolist()):
        order = np.lexsort((names, -np.abs(weights[i])))
        senders = order[order != i][:5]
        if len(senders) != 5 or not valid[i] or not valid[senders].all():
            continue
        sample_id = f"SP500|{date}|{ticker}"
        entropy = int.from_bytes(sha256(f"{seed}/{sample_id}".encode()).digest()[:8], "big")
        senders = np.random.default_rng(entropy).permutation(senders)
        edges = weights[i, senders]
        base = dict(sample_id=sample_id, date=date, split=split, target=ticker, sender_ids=names[senders].tolist(),
                    snapshot=os.path.relpath(snapshot, destination.parent), snapshot_sha256=checksum,
                    positions=[i, *senders.tolist()], lookback=WINDOW, template_version=VERSION,
                    checkpoint_sha256=meta["checkpoint_sha256"], release=meta["release"])
        if ticker in labels:
            yield dict(base, task_type="FORECAST", answer=labels[ticker])
        if (abs(edges) == abs(edges).max()).sum() == 1:
            yield dict(base, task_type="FLOW_QA", answer=str(names[senders[abs(edges).argmax()]]),
                       edges=edges.tolist())
        if (norms[i] == norms[i].max()).sum() == 1:
            yield dict(base, task_type="FLOW_MEMORY", answer=MEMORY[norms[i].argmax()], norms=norms[i].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=DATA / "samples-v1.jsonl")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()
    config = json.loads((ROOT / "code/configs" / CELL / "runtime.json").read_text())["data"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    seen = set()
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "index"
        sources = {os.path.relpath(p.resolve(), args.out.resolve().parent): file_digest(p) for p in args.snapshots}
        with stage.open("w") as f:
            print(json.dumps({"format": "marketflow-index-v2", "snapshots": sources}), file=f)
            for path in sorted(args.snapshots):
                relative = os.path.relpath(path.resolve(), args.out.resolve().parent)
                for row in observations(path.resolve(), args.out.resolve(), config, args.seed, sources[relative]):
                    key = row["sample_id"], row["task_type"]
                    if key in seen:
                        raise ValueError(f"Duplicate task: {key}")
                    seen.add(key)
                    counts[(row["split"], row["task_type"])] += 1
                    row.pop("snapshot_sha256")
                    print(json.dumps(row, allow_nan=False), file=f)
        if not counts:
            raise ValueError("No eligible observations in requested date partitions")
        stage.replace(args.out)
    print(dict(counts))


if __name__ == "__main__":
    main()
