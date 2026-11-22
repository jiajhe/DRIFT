"""Verify the overlap of the archived experiment with a pinned public OHLCV snapshot."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "local/drift-alpha158-flow-input"
OUT = (
    HERE
    / "timenet_connectors/datasets/10eit/drift_alpha158_flow_public/observation_index.json.gz"
)
PIN = "1400038a83f5ba2702adf059829d4e1d2be0f977170fea5455e10bbd22b2c9a2"


def write_index(archive):
    with archive.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != PIN:
            raise ValueError("Wrong reference price archive")
    records = json.loads((BUNDLE / "observations.json").read_text())
    with np.load(BUNDLE / "ohlcv.npz") as f:
        dates = f["dates"].tolist()
        ref = f["values"][:, :, f["columns"].tolist().index("close")]
        ids = {s: i for i, s in enumerate(f["tickers"].tolist())}
    matched = []
    errors = []
    with zipfile.ZipFile(archive) as z:
        members = {
            Path(n).stem: n
            for n in z.namelist()
            if n.startswith("sp500_cache/") and n.endswith(".csv")
        }
        prices = {}
        for r in records:
            syms = [r["target"], r["other"], *r["senders"]]
            if any(s not in members for s in syms):
                continue
            t = dates.index(r["date"])
            window = dates[t - 20 : t + 3]
            for s in syms:
                if s not in prices:
                    with z.open(members[s]) as f:
                        data = pd.read_csv(f, usecols=["Date", "Adj Close"])
                    prices[s] = data.drop_duplicates("Date", keep="last").set_index(
                        "Date"
                    )["Adj Close"]
            pub = np.array([prices[s].reindex(window).values for s in syms])
            native = ref[[ids[s] for s in syms], t - 20 : t + 3]
            if (
                not np.isfinite(pub).all()
                or (pub <= 0).any()
                or not np.isfinite(native).all()
            ):
                continue
            error = float(
                np.abs(
                    pub[:, 1:21] / pub[:, :20] - native[:, 1:21] / native[:, :20]
                ).max()
            )
            winner = "A" if pub[0, -1] / pub[0, -2] > pub[1, -1] / pub[1, -2] else "B"
            gold = json.loads(
                r["tasks"][3]["target"]
                .partition("<answer>")[2]
                .partition("</answer>")[0]
            )
            if error > 1e-5 or winner != gold["winner"]:
                continue
            matched.append(r)
            errors.append(error)
    if len(matched) != 98 or len({r["target"] for r in matched}) != 30:
        raise ValueError("Public observation index changed")
    OUT.write_bytes(
        gzip.compress(json.dumps(matched, separators=(",", ":")).encode(), mtime=0)
    )
    print(
        dict(
            records=len(matched),
            max_daily_return_error=max(errors),
            splits=Counter(r["split"] for r in matched),
            sectors=len({r["sector"] for r in matched}),
            price_series=len(
                {s for r in matched for s in [r["target"], r["other"], *r["senders"]]}
            ),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    write_index(parser.parse_args().archive)
