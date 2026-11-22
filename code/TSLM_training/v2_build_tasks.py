"""Balanced v2 supervision. Read future outcomes here, never in the model input pipeline."""

from collections import Counter, defaultdict
import json
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from v2_features import BASE, shares, train_stats
from prompt_templates_v2 import (
    DRIVER,
    FLOW,
    SHIFT,
    SENS,
    ROLE,
    STABILITY,
    REGIME,
    evidence,
)


def labels(d, l, h):
    ratio, relay, keep = shares(d)
    driver = np.where(
        ratio < h["ratio"],
        0,
        np.where(np.sign(d["flow_delta"]) == np.sign(d["p_local"]), 1, 2),
    )
    mode = np.where(relay >= h["relay"], 2, np.where(keep >= h["keep"], 0, 1))
    cf = l["sender_cf_delta"]
    sens = np.digitize(abs(cf), h["cf"])
    shift = np.where(sens == 0, 0, np.where(cf > 0, 1, 2))
    regime = int(np.digitize(d["regime_scalars"][3] * d["k_frobenius"], h["regime"]))
    return (
        (driver, mode, abs(d["edges"]).argmax(1)),
        (shift, sens),
        (
            np.digitize(d["g_s"] - d["g_r"], h["net"]),
            (d["role_tv_20d"].mean(1) >= h["tv"]).astype(int),
            regime,
        ),
    )


def eligible(d):
    return (
        d["role_valid"]
        & d["return_mask"].all(1)
        & d["return_mask"][d["senders"]].all((1, 2))
    )


def pools(root, days, h, rng):
    pools = [defaultdict(list) for _ in range(3)]
    for di, r in enumerate(days):
        with (
            np.load(root / (r["date"] + ".npz")) as f,
            np.load(root / "labels" / (r["date"] + ".npz")) as y,
        ):
            d = dict(f)
            at, cf, role = labels(d, y, h)
        query = rng.integers(0, 5, len(d["instruments"]))
        per = [Counter() for _ in range(3)]
        for i in rng.permutation(np.flatnonzero(eligible(d))):
            q = query[i]
            kr = [
                (at[0][i], at[1][i], at[2][i]),
                (cf[0][i, q], cf[1][i, q]),
                (role[0][i], role[1][i], role[2]),
            ]
            for t, key in enumerate(kr):
                if per[t][key] < 8:
                    pools[t][key].append((di, int(i), int(q)))
                    per[t][key] += 1
    return pools


def balanced(pool, n, days, rng, require_days=True):
    keys = list(pool)
    for items in pool.values():
        rng.shuffle(items)
    dims = [max(k[c] for k in keys) + 1 for c in range(len(keys[0]))]
    if dims not in ([3, 3, 5], [3, 3], [3, 2, 3]):
        raise ValueError(f"Missing task classes: {dims}")
    # A date is represented by its observed set of available strata. Grouping identical
    # sets keeps the integer allocation small while guaranteeing coverage of every date.
    by_day = defaultdict(set)
    lookup = defaultdict(list)
    for s, k in enumerate(keys):
        for item in pool[k]:
            by_day[item[0]].add(s)
            lookup[item[0], s].append(item)
    patterns = defaultdict(list)
    if require_days:
        for i in range(days):
            patterns[tuple(sorted(by_day[i]))].append(i)
    mandatory = [(pi, s) for pi, pattern in enumerate(patterns) for s in pattern]
    size = len(keys) + len(mandatory)
    rows = []
    lows = []
    highs = []

    def constrain(vec, lo, hi):
        rows.append(vec)
        lows.append(lo)
        highs.append(hi)

    constrain([1] * len(keys) + [0] * len(mandatory), n, n)
    for c, dim in enumerate(dims):
        for v in range(dim):
            constrain(
                [int(k[c] == v) for k in keys] + [0] * len(mandatory),
                n // dim,
                -(-n // dim),
            )
    for pi, dates in enumerate(patterns.values()):
        constrain(
            [0] * len(keys) + [int(a == pi) for a, s in mandatory],
            len(dates),
            len(dates),
        )
    for s in range(len(keys)):
        constrain(
            [int(j == s) for j in range(len(keys))]
            + [-int(j == s) for _, j in mandatory],
            0,
            np.inf,
        )
    # Rake available joint cells to balanced marginal targets, then choose the
    # closest feasible integer counts instead of inducing arbitrary label correlations.
    desired = np.array([len(pool[k]) for k in keys], dtype=float)
    for _ in range(80):
        for c, dim in enumerate(dims):
            for v in range(dim):
                mask = np.array([key[c] == v for key in keys])
                desired[mask] *= (n / dim) / desired[mask].sum()
    base_size = size
    rows = [r + [0] * len(keys) for r in rows]
    for s, target in enumerate(desired):
        u = np.zeros(base_size + len(keys))
        u[s] = 1
        u[base_size + s] = -1
        constrain(u, -np.inf, target)
        u = u.copy()
        u[s] = -1
        constrain(u, -np.inf, -target)
    m = np.array(rows, dtype=float)
    out = milp(
        np.r_[rng.uniform(0, 1e-7, size), np.ones(len(keys))],
        integrality=np.r_[np.ones(size), np.zeros(len(keys))],
        bounds=Bounds(
            np.zeros(size + len(keys)),
            [len(pool[k]) for k in keys]
            + [days] * len(mandatory)
            + [np.inf] * len(keys),
        ),
        constraints=LinearConstraint(m, lows, highs),
    )
    if not out.success:
        raise ValueError(
            f"Exact class balance with requested day coverage is infeasible: {out.message}"
        )
    x = np.rint(out.x[:base_size]).astype(int)
    mandatory_rows = set()
    for pi, dates in enumerate(patterns.values()):
        slots = [
            s
            for v, (p, s) in zip(x[len(keys) :], mandatory, strict=True)
            if p == pi
            for _ in range(v)
        ]
        rng.shuffle(slots)
        for di, s in zip(dates, slots, strict=True):
            mandatory_rows.add(lookup[di, s][0])
    picked = set(mandatory_rows)
    for s, k in enumerate(keys):
        used = sum(it in mandatory_rows for it in pool[k])
        remaining = [it for it in pool[k] if it not in mandatory_rows]
        picked.update(remaining[: x[s] - used])
    if len(picked) != n:
        raise ValueError("Duplicate or missing balanced observations")
    return sorted(picked)


def pair_candidates(d, l, n, rng, random_pairs, bands, threshold):
    ids = np.flatnonzero(
        d["role_valid"] & d["return_mask"].all(1) & np.isfinite(l["future_ret"])
    )
    p = l["future_pct"]
    hi, lo = ids[p[ids] >= 0.8], ids[p[ids] < 0.2]
    pairs = {}
    for _ in range(10000):
        if len(pairs) == n:
            break
        a, b = (
            rng.choice(ids, 2, replace=False)
            if random_pairs
            else (rng.choice(hi), rng.choice(lo))
        )
        if l["future_ret"][a] == l["future_ret"][b]:
            continue
        high, low = (a, b) if l["future_ret"][a] > l["future_ret"][b] else (b, a)
        if (
            not random_pairs
            and (abs(d["ens_score"][high] - d["ens_score"][low]) > threshold)
            != bands[len(pairs)]
        ):
            continue
        pairs[(int(high), int(low))] = True
    if len(pairs) != n:
        raise ValueError("Insufficient distinct eligible pairs for confidence stratum")
    return list(pairs)


def make_split(root, days, split, n, h, rng):
    pp = pools(root, days, h, rng)
    byday = defaultdict(list)
    for task, pool in enumerate(pp):
        for di, i, q in balanced(pool, n, len(days), rng):
            byday[di].append((task, i, q))
    pair_counts = np.full(len(days), n // len(days))
    pair_counts[rng.permutation(len(days))[: n % len(days)]] += 1
    winners = np.arange(n) % 2
    rng.shuffle(winners)
    bonus = np.arange(n) % 2
    rng.shuffle(bonus)
    bands = np.arange(n) % 2
    rng.shuffle(bands)
    index = 0
    rows = []
    random_rows = []
    for di, r in enumerate(days):
        with (
            np.load(root / (r["date"] + ".npz")) as f,
            np.load(root / "labels" / (r["date"] + ".npz")) as y,
        ):
            d = dict(f)
            l = dict(y)
        at, cf, role = labels(d, l, h)
        names = d["instruments"]

        def row(i, t):
            return dict(date=r["date"], i=int(i), ticker=str(names[i]), task=t)

        for t, i, q in byday[di]:
            record = row(i, t)
            send = names[d["senders"][i]].tolist()
            record["senders"] = send
            if t == 0:
                ans = dict(
                    driver=DRIVER[at[0][i]],
                    flow_mode=FLOW[at[1][i]],
                    top_sender=send[at[2][i]],
                )
            elif t == 1:
                record["query"] = send[q]
                ans = dict(
                    queried_sender=send[q],
                    score_shift=SHIFT[cf[0][i, q]],
                    sensitivity=SENS[cf[1][i, q]],
                )
            else:
                ans = dict(
                    net_role=ROLE[role[0][i]],
                    role_stability=STABILITY[role[1][i]],
                    market_regime=REGIME[role[2]],
                )
            record.update(
                answer=ans, evidence=evidence(t, ans, d, i, None, rng.integers(9))
            )
            rows.append(record)
        count = int(pair_counts[di])
        for random in [False, True] if split == "test" else [False]:
            pairs = pair_candidates(
                d, l, count, rng, random, bands[index : index + count], h["spread"]
            )
            for k, (high, low) in enumerate(pairs):
                flip = (bonus if random else winners)[index + k]
                i, b = (low, high) if flip else (high, low)
                winner = "B" if flip else "A"
                ans = dict(
                    winner=winner,
                    confidence="HIGH"
                    if abs(d["ens_score"][i] - d["ens_score"][b]) > h["spread"]
                    else "MODERATE",
                )
                record = row(i, 3)
                record.update(
                    b=b,
                    other=str(names[b]),
                    answer=ans,
                    evidence=evidence(3, None, d, i, b, rng.integers(9)),
                )
                (random_rows if random else rows).append(record)
        index += count
    rng.shuffle(rows)
    return rows, random_rows


def build(root=BASE):
    stats = (
        json.loads((root / "train_stats.json").read_text())
        if (root / "train_stats.json").exists()
        else train_stats(root)
    )
    manifest = json.loads((root / "manifest.json").read_text())
    summary = {}
    folder = root / "tasks"
    folder.mkdir(exist_ok=True)
    for idx, (split, n) in enumerate([("train", 6000), ("valid", 500), ("test", 1000)]):
        days = [d for d in manifest["days"] if d["split"] == split]
        rows, extra = make_split(
            root, days, split, n, stats["thresholds"], np.random.default_rng(71 + idx)
        )
        assert all(sum(r["task"] == t for r in rows) == n for t in range(4))
        for name, records in [(split, rows), ("test_random_pairs", extra)]:
            if not records:
                continue
            (folder / (name + ".jsonl")).write_text(
                "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
            )
            summary[name] = {}
            for t in set(r["task"] for r in records):
                xs = [r for r in records if r["task"] == t]
                summary[name][t] = dict(
                    n=len(xs),
                    days=len({r["date"] for r in xs}),
                    classes={
                        k: dict(Counter(r["answer"][k] for r in xs))
                        for k in xs[0]["answer"]
                        if k not in ("top_sender", "queried_sender")
                    },
                )
        print(split, len(rows), len(extra), flush=True)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    build()
