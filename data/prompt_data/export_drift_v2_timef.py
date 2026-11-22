"""Prepare source inputs for the local licensed TimeNet connector from history-v3."""

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/TSLM_training"))
from v2_features import BASE
from prompt_templates_v2 import (
    DRIVER,
    FLOW,
    SHIFT,
    SENS,
    ROLE,
    REGIME,
    STABILITY,
    evidence,
    response,
    user_parts,
)
from v2_build_tasks import labels

SECTORS = {
    "Information Technology": "AAPL MSFT NVDA",
    "Communication Services": "GOOGL META NFLX",
    "Consumer Discretionary": "AMZN HD MCD",
    "Consumer Staples": "PG KO PEP",
    "Health Care": "JNJ UNH MRK",
    "Financials": "JPM BAC GS",
    "Industrials": "CAT HON UNP",
    "Energy": "XOM CVX",
    "Materials": "LIN APD",
    "Utilities": "NEE DUK SO",
    "Real Estate": "AMT PLD",
}
BUNDLE = BASE.parent / "drift-alpha158-flow-input"
PROVIDER = ROOT / "data/us_yahoo_sp500_pit_ohlcv_20260723"


def build():
    rng = np.random.default_rng(71)
    manifest = json.loads((BASE / "manifest.json").read_text())
    h = json.loads((BASE / "train_stats.json").read_text())["thresholds"]
    sector = {k: s for s, tickers in SECTORS.items() for k in tickers.split()}
    assert len(sector) == 30
    calendar = (PROVIDER / "calendars/day.txt").read_text().splitlines()
    days = manifest["days"]
    selection = []
    used = set()
    for ix in rng.permutation(len(days)):
        day = days[ix]
        date = day["date"]
        with (
            np.load(BASE / (date + ".npz")) as f,
            np.load(BASE / "labels" / (date + ".npz")) as y,
        ):
            names = f["instruments"]
            ids = {v: i for i, v in enumerate(names)}
            pct = y["future_pct"]
            mask = f["role_valid"] & f["return_mask"].all(1)
            available = [
                (s, ids[s])
                for s in sector
                if s in ids
                and (s, date[:4]) not in used
                and mask[ids[s]]
                and (pct[ids[s]] < 0.2 or pct[ids[s]] >= 0.8)
            ]
            if not available:
                continue
            d = dict(f)
            l = dict(y)
        at, cf, roles = labels(d, l, h)
        for symbol, i in available:
            js = d["senders"][i]
            if not mask[js].all():
                continue
            other = np.flatnonzero(mask & (pct < 0.2 if pct[i] >= 0.8 else pct >= 0.8))
            if not len(other):
                continue
            b = int(rng.choice(other))
            q = int(rng.integers(5))
            send = names[js].tolist()
            ans = [
                dict(
                    driver=DRIVER[at[0][i]],
                    flow_mode=FLOW[at[1][i]],
                    top_sender=send[at[2][i]],
                ),
                dict(
                    queried_sender=send[q],
                    score_shift=SHIFT[cf[0][i, q]],
                    sensitivity=SENS[cf[1][i, q]],
                ),
                dict(
                    net_role=ROLE[roles[0][i]],
                    role_stability=STABILITY[roles[1][i]],
                    market_regime=REGIME[roles[2]],
                ),
                dict(
                    winner="A" if l["future_ret"][i] > l["future_ret"][b] else "B",
                    confidence="HIGH"
                    if abs(d["ens_score"][i] - d["ens_score"][b]) > h["spread"]
                    else "MODERATE",
                ),
            ]
            tasks = []
            for t, gold in enumerate(ans):
                row = dict(
                    date=date,
                    task=t,
                    ticker=symbol,
                    other=str(names[b]),
                    senders=send,
                    query=send[q],
                    answer=gold,
                    evidence=evidence(
                        t, gold if t != 3 else None, d, i, b, rng.integers(9)
                    ),
                )
                headers, question = user_parts(row)
                tasks.append(
                    dict(prompt="".join(headers) + question, target=response(row))
                )
            end = calendar.index(date) + 1
            selection.append(
                dict(
                    date=date,
                    target=symbol,
                    senders=send,
                    other=str(names[b]),
                    sector=sector[symbol],
                    dates=calendar[end - 20 : end],
                    split=day["split"],
                    tasks=tasks,
                )
            )
            used.add((symbol, date[:4]))
        if len(used) == 30 * 8:
            break
    if {s for s, y in used} != set(sector):
        raise ValueError("Missing core stock")
    tickers = sorted(
        {s for r in selection for s in [r["target"], r["other"], *r["senders"]]}
    )
    cols = ("open", "high", "low", "close", "volume")
    bars = np.full((len(tickers), len(calendar), 5), np.nan, dtype="float32")
    for i, s in enumerate(tickers):
        for j, k in enumerate(cols):
            x = np.fromfile(
                PROVIDER / "features" / s.lower() / (k + ".day.bin"), dtype="<f4"
            )
            offset = int(x[0])
            bars[i, offset : offset + len(x) - 1, j] = x[1:]
    BUNDLE.mkdir(exist_ok=True)
    np.savez_compressed(
        BUNDLE / "ohlcv.npz", dates=calendar, tickers=tickers, values=bars, columns=cols
    )
    (BUNDLE / "observations.json").write_text(
        json.dumps(
            sorted(selection, key=lambda r: (r["date"], r["target"])),
            separators=(",", ":"),
        )
    )
    print(
        "observations",
        len(selection),
        "sectors",
        len(set(sector.values())),
        "source stocks",
        len(tickers),
        flush=True,
    )


if __name__ == "__main__":
    build()
