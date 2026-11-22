from collections import Counter
from pathlib import Path
import argparse, gzip, json
import numpy as np
from timenet import TimeNet
from timenet.types import Split

ROOT = Path(__file__).resolve().parents[2]


def verify(public, offline):
    root = ROOT
    name = "10eit/drift-alpha158-flow-public@1.0.0"
    datasets = [TimeNet(sources=[str(src)]).load(name) for src in (public, offline)]
    tag = lambda rec: {a.key: a.value.read() for a in rec.annotations}

    def key(rec):
        return tag(rec)["sample_id"]

    public_rows = {key(r): r for r in datasets[0].records}
    offline_rows = {key(r): r for r in datasets[1].records}
    cohort = json.loads(
        gzip.decompress(
            (
                root
                / "data/prompt_data/timenet_connectors/datasets/10eit/drift_alpha158_flow_public/observation_index.json.gz"
            ).read_bytes()
        )
    )
    assert (
        len(cohort) == len(public_rows) == 98
        and public_rows.keys() == offline_rows.keys()
    )
    signals = 0
    for r in cohort:
        k = r["date"] + "|" + r["target"]
        rs = public_rows[k], offline_rows[k]
        assert tag(rs[0]) == tag(rs[1])
        order = json.loads(tag(rs[0])["factor_order"])
        assert len(order) == 157 and tag(rs[0])["date"] == r["date"]
        left, right = (
            {s.name: {x.name: x for x in s.signals} for s in rec.sources} for rec in rs
        )
        assert left.keys() == right.keys()
        for n in left:
            assert left[n].keys() == right[n].keys()
            assert set(left[n]) == set(
                ["daily_return"]
                + (
                    order
                    if n in (r["target"], r["other"])
                    else ["KMID", "KLEN", "ROC5", "STD5", "WVMA5"]
                )
            )
            a, b = left[n], right[n]
            for s in a:
                x, y = a[s].to_numpy(), b[s].to_numpy()
                assert x.shape == y.shape == (20,) and np.array_equal(
                    x.view("uint32"), y.view("uint32")
                )
                signals += 1
    counts = {}
    for split, label in [
        ("train", Split.TRAIN),
        ("valid", Split.VALIDATION),
        ("test", Split.TEST),
    ]:
        expected = Counter(
            (r["date"] + "|" + r["target"], t["prompt"], t["target"])
            for r in cohort
            if r["split"] == split
            for t in r["tasks"]
        )
        for ds in datasets:
            actual = Counter(
                (key(task.inputs[0]), task.prompt.read(), task.targets[0].read())
                for task in ds.iter_tasks(split=label)
            )
            assert actual == expected
        counts[split] = sum(expected.values())
    report = dict(
        records=len(cohort),
        core_stocks=len({r["target"] for r in cohort}),
        sectors=len({r["sector"] for r in cohort}),
        signals=signals,
        values_compared=20 * signals,
        split_tasks=counts,
        anonymous_offline_equal=True,
        source_sha256="1400038a83f5ba2702adf059829d4e1d2be0f977170fea5455e10bbd22b2c9a2",
        registry_bytes=sum(p.stat().st_size for p in public.rglob("*") if p.is_file()),
    )
    print(report)
    (root / "reports/v2_timenet_public.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two TimeNet builds, including every signal and task."
    )
    parser.add_argument(
        "--online",
        type=Path,
        default=ROOT / "data/prompt_data/local/drift-alpha158-public-registry",
    )
    parser.add_argument(
        "--offline",
        type=Path,
        default=ROOT / "cache/marketflow-public-offline-registry",
    )
    args = parser.parse_args()
    verify(args.online, args.offline)
