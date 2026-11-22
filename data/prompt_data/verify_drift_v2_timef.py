"""Check the official SDK matrices, Qlib factors and split labels of the v2 release."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data/prompt_data/local/drift-alpha158-flow-input"
REGISTRY = BUNDLE.parent / "drift-alpha158-registry"
REFERENCE = BUNDLE.parent / "drift-alpha158-qlib-reference.npz"
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent
        / "timenet_connectors/datasets/10eit/drift_alpha158_flow"
    ),
)
from alpha158 import ORDER, compute_alpha158_ohlcv


def qlib_reference(records):
    import qlib
    from qlib.data import D

    sys.path.insert(0, str(ROOT / "code/src"))
    from meta_fdg.data.us_ohlcv_handler import alpha158_us_ohlcv_features

    qlib.init(
        provider_uri=str(ROOT / "data/us_yahoo_sp500_pit_ohlcv_20260723"),
        region="us",
        kernels=2,
        expression_cache=None,
        dataset_cache=None,
    )
    cols, names = alpha158_us_ohlcv_features()
    assert names == ORDER
    by = defaultdict(list)
    for i, r in enumerate(records):
        by[r["target"]].append(i)
    arrays = np.empty((len(records), 20, 157), dtype="float32")
    for ticker, indices in by.items():
        dates = sorted({d for i in indices for d in records[i]["dates"]})
        q = D.features(
            [ticker], cols, start_time=dates[0], end_time=dates[-1]
        ).droplevel("instrument")
        for i in indices:
            arrays[i] = q.loc[records[i]["dates"]].to_numpy("float32")
    np.savez_compressed(REFERENCE, values=arrays)


def audit(records):
    from timenet import TimeNet
    from timenet.types import Split
    import pandas as pd

    dataset = TimeNet(sources=[str(REGISTRY)]).load("10eit/drift-alpha158-flow@1.0.0")
    by = {(r["date"], r["target"]): i for i, r in enumerate(records)}
    refs = np.load(REFERENCE)["values"]
    err = 0.0
    with np.load(BUNDLE / "ohlcv.npz", allow_pickle=False) as f:
        dates = f["dates"].tolist()
        tickers = f["tickers"].tolist()
        bars = f["values"]
        cols = f["columns"].tolist()
    sources = defaultdict(list)
    for r in records:
        for s in {r["target"], r["other"], *r["senders"]}:
            sources[s].append(dates.index(r["date"]) + 1)
    raw = {}
    for s in {r["target"] for r in records}:
        first = max(0, min(sources[s]) - 81)
        last = max(sources[s])
        raw[s] = compute_alpha158_ohlcv(
            pd.DataFrame(
                bars[tickers.index(s), first:last],
                index=dates[first:last],
                columns=cols,
            )
        )
    for rec in dataset.records:
        ann = {a.key: a.value.read() for a in rec.annotations}
        i = by[ann["date"], ann["target_ticker"]]
        r = records[i]
        ss = {s.name: s for s in rec.sources}
        order = json.loads(ann["factor_order"])
        axis = json.loads(ann["trading_dates"])
        assert order == ORDER and axis == r["dates"]
        signals = {sig.name: sig.to_numpy() for sig in ss[r["target"]].signals}
        matrix = np.column_stack([signals[k] for k in order])
        exact = raw[r["target"]].loc[axis, order].to_numpy("float32")
        assert matrix.dtype == np.float32 and np.array_equal(
            matrix.view("uint32"), exact.view("uint32")
        )
        assert np.array_equal(np.isfinite(matrix), np.isfinite(refs[i]))
        err = max(err, float(np.nanmax(np.abs(matrix - refs[i]))))
        assert set(order) <= set(sig.name for sig in ss[r["other"]].signals)
    assert err < 1e-5 and len(dataset.records) == len(records)
    counts = {}
    for key, split in [
        ("train", Split.TRAIN),
        ("valid", Split.VALIDATION),
        ("test", Split.TEST),
    ]:
        expected = Counter(
            (r["date"] + "|" + r["target"], t["prompt"], t["target"])
            for r in records
            if r["split"] == key
            for t in r["tasks"]
        )
        actual = Counter()
        for task in dataset.iter_tasks(split=split):
            ann = {a.key: a.value.read() for a in task.inputs[0].annotations}
            actual[ann["sample_id"], task.prompt.read(), task.targets[0].read()] += 1
        assert actual == expected
        counts[key] = sum(actual.values())
    size = sum(p.stat().st_size for p in REGISTRY.rglob("*") if p.is_file())
    assert size < 300_000_000
    report = dict(
        records=len(records),
        tasks=counts,
        core_stocks=len(raw),
        sectors=len({r["sector"] for r in records}),
        max_abs_factor_error=err,
        checked_factor_values=refs.size,
        registry_bytes=size,
        exact_sdk_roundtrip=True,
    )
    (ROOT / "reports/v2_timenet.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qlib-reference", action="store_true")
    args = p.parse_args()
    records = json.loads((BUNDLE / "observations.json").read_text())
    (qlib_reference if args.qlib_reference else audit)(records)
