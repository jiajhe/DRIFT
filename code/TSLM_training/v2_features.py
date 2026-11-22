"""Day-t inputs and train-only scaling for the six DRIFT projector families."""

from functools import lru_cache
import json
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/prompt_data/local/history-v3"
WIDTHS = {
    "temporal": 128,
    "local": 72,
    "component": 18,
    "role": 42,
    "sender_temporal": 128,
    "edge": 115,
}
EPS = 1e-8


def slots(d):
    z = d["Z_last"]
    js = d["senders"].astype(int)
    k = d["edges"][..., None]
    norms = d["norms"]
    features = {
        "temporal": d["Z_patches"].astype("float32"),
        "local": np.column_stack(
            [
                z,
                *[
                    d[key]
                    for key in (
                        "p_local",
                        "prediction",
                        "flow_delta",
                        "ens_std",
                        "ens_score",
                    )
                ],
                *shares(d),
            ]
        ),
        "component": np.concatenate(
            [
                d["components"].transpose(1, 0, 2),
                np.log1p(norms)[..., None],
                (norms / (norms.sum(1, keepdims=True) + EPS))[..., None],
            ],
            axis=-1,
        ),
        "role": np.column_stack(
            [
                d["M_s"],
                d["M_r"],
                d["g_s"],
                d["g_r"],
                d["g_s"] - d["g_r"],
                d["role_tv_20d"],
                np.broadcast_to(d["regime_scalars"], (len(z), 5)),
            ]
        ),
        "sender_temporal": d["sender_halves"].astype("float32"),
        "edge": np.concatenate(
            [
                z[js],
                d["U"][js],
                k,
                abs(k),
                np.sign(k),
                k * d["U"][js],
                k * d["F_minus"][js],
            ],
            axis=-1,
        ),
    }
    if any(
        v.shape[-1] != WIDTHS[k] or not np.isfinite(v).all()
        for k, v in features.items()
    ):
        raise ValueError("Invalid DRIFT slot")
    return features


def shares(d):
    n = d["norms"]
    flow = abs(d["flow_delta"]) / (abs(d["p_local"]) + abs(d["flow_delta"]) + EPS)
    return flow, n[:, 2] / (n.sum(1) + EPS), n[:, 0] / (n[:, :2].sum(1) + EPS)


def train_stats(root=BASE):
    manifest = json.loads((root / "manifest.json").read_text())
    sums = {}
    squares = {}
    counts = {}
    samples = {
        k: [] for k in ("ratio", "relay", "keep", "cf", "net", "tv", "regime", "spread")
    }
    for rec in manifest["days"]:
        if rec["split"] != "train":
            continue
        with (
            np.load(root / (rec["date"] + ".npz")) as snap,
            np.load(root / "labels" / (rec["date"] + ".npz")) as l,
        ):
            d = dict(snap)
            for key, value in slots(d).items():
                if key == "role":
                    value = value[d["role_valid"]]
                if key == "sender_temporal":
                    value = value[d["senders"]]
                # Each component keeps a distinct 18-column distribution.
                v = (
                    value if key == "component" else value.reshape(-1, WIDTHS[key])
                ).astype("float64")
                sums[key] = sums.get(key, 0) + v.sum(axis=0)
                squares[key] = squares.get(key, 0) + (v * v).sum(axis=0)
                counts[key] = counts.get(key, 0) + len(v)
            for k, v in zip(("ratio", "relay", "keep"), shares(d), strict=True):
                samples[k].append(v)
            samples["cf"].append(abs(l["sender_cf_delta"]).ravel())
            samples["net"].append(d["g_s"] - d["g_r"])
            samples["tv"].append(d["role_tv_20d"][d["role_valid"]].mean(1))
            samples["regime"].append([d["regime_scalars"][3] * d["k_frobenius"]])
            pct = l["future_pct"]
            hi = np.flatnonzero(pct >= 0.8)
            lo = np.flatnonzero(pct < 0.2)
            samples["spread"].append(
                abs(d["ens_score"][hi, None] - d["ens_score"][lo]).ravel()
            )
    values = {k: np.concatenate(v) for k, v in samples.items()}
    q = lambda key, prob: np.quantile(values[key], prob).tolist()
    relay = q("relay", 0.75)
    thresholds = dict(
        ratio=q("ratio", 0.33),
        relay=relay,
        keep=float(np.median(values["keep"][values["relay"] < relay])),
        cf=q("cf", [0.33, 0.67]),
        net=q("net", [1 / 3, 2 / 3]),
        tv=q("tv", 0.5),
        regime=q("regime", [1 / 3, 2 / 3]),
        spread=q("spread", 0.5),
    )
    moments = {}
    for k, n in counts.items():
        m = sums[k] / n
        scale = np.sqrt(np.maximum(squares[k] / n - m * m, 0))
        moments[k] = dict(
            mean=m.tolist(), scale=np.where(scale > 1e-10, scale, 1).tolist(), n=n
        )
    data = dict(
        schema="drift-v2.1", fit_split="train", thresholds=thresholds, slots=moments
    )
    temp = root / "train_stats.json.tmp"
    temp.write_text(json.dumps(data, allow_nan=False))
    temp.replace(root / "train_stats.json")
    return data


class Inputs:
    def __init__(self, root=BASE):
        self.root = root
        raw_stats = (root / "train_stats.json").read_bytes()
        self.stats = json.loads(raw_stats)
        self.stats_digest = hashlib.sha256(raw_stats).hexdigest()
        if (
            self.stats.get("schema") != "drift-v2.1"
            or self.stats["fit_split"] != "train"
        ):
            raise ValueError(
                "Regenerate the train-only v2.1 statistics with v2_features.py"
            )
        self.days = {
            r["date"]: r
            for r in json.loads((root / "manifest.json").read_text())["days"]
        }
        self.moments = {
            k: tuple(np.asarray(v[p], dtype="float32") for p in ("mean", "scale"))
            for k, v in self.stats["slots"].items()
        }

    @lru_cache(maxsize=2010)
    def day(self, date):
        with np.load(self.root / (date + ".npz"), allow_pickle=False) as f:
            d = dict(f)
        raw = slots(d)
        scaled = {
            k: (v - m) / sd for k, v in raw.items() for m, sd in [self.moments[k]]
        }
        return d, scaled

    @lru_cache(maxsize=2010)
    def return_day(self, date):
        with np.load(self.root / (date + ".npz"), allow_pickle=False) as day:
            return {k: day[k] for k in ("returns_20d", "senders")}

    def stock(self, row, donor=None):
        d, features = self.day(row["date"])
        actual = row if donor is None else donor
        if actual["date"] != row["date"]:
            raise ValueError("Donor must be from the same date")
        i = actual["i"]

        def target(i):
            return [
                ("temporal", j, x) for j, x in enumerate(features["temporal"][i])
            ] + [
                ("local", 0, features["local"][i]),
                *[("component", j, x) for j, x in enumerate(features["component"][i])],
                ("role", 0, features["role"][i]),
            ]

        blocks = [target(i)]
        if row["task"] == 3:
            blocks.append(target(actual["b"]))
        else:
            for pos, j in enumerate(d["senders"][i]):
                blocks.append(
                    [
                        ("sender_temporal", h, x)
                        for h, x in enumerate(features["sender_temporal"][j])
                    ]
                    + [("edge", pos, features["edge"][i, pos])]
                )
        return blocks


if __name__ == "__main__":
    print(train_stats()["thresholds"])
