"""Check every persisted answer and evidence statement against the dated release."""

from collections import defaultdict
import json
import numpy as np
from prompt_templates_v2 import (
    DRIVER,
    FLOW,
    ROLE,
    REGIME,
    STABILITY,
    SHIFT,
    SENS,
    evidence,
    user_parts,
)
from v2_features import BASE
from v2_build_tasks import labels


def verify():
    h = json.loads((BASE / "train_stats.json").read_text())["thresholds"]
    summary = {}
    for split in ("train", "valid", "test", "test_random_pairs"):
        byday = defaultdict(list)
        for line in (BASE / "tasks" / (split + ".jsonl")).read_text().splitlines():
            r = json.loads(line)
            byday[r["date"]].append(r)
        total = 0
        for day, rows in byday.items():
            with (
                np.load(BASE / (day + ".npz")) as f,
                np.load(BASE / "labels" / (day + ".npz")) as y,
            ):
                d = dict(f)
                l = dict(y)
            at, cf, role = labels(d, l, h)
            for r in rows:
                i, t = r["i"], r["task"]
                b = r.get("b")
                assert d["instruments"][i] == r["ticker"]
                if t != 3:
                    ss = d["instruments"][d["senders"][i]].tolist()
                    assert ss == r["senders"]
                if t == 0:
                    truth = dict(
                        driver=DRIVER[at[0][i]],
                        flow_mode=FLOW[at[1][i]],
                        top_sender=ss[at[2][i]],
                    )
                elif t == 1:
                    j = ss.index(r["query"])
                    truth = dict(
                        queried_sender=ss[j],
                        score_shift=SHIFT[cf[0][i, j]],
                        sensitivity=SENS[cf[1][i, j]],
                    )
                elif t == 2:
                    truth = dict(
                        net_role=ROLE[role[0][i]],
                        role_stability=STABILITY[role[1][i]],
                        market_regime=REGIME[role[2]],
                    )
                else:
                    truth = dict(
                        winner="A" if l["future_ret"][i] > l["future_ret"][b] else "B",
                        confidence="HIGH"
                        if abs(d["ens_score"][i] - d["ens_score"][b]) > h["spread"]
                        else "MODERATE",
                    )
                    if split != "test_random_pairs":
                        assert (
                            max(l["future_pct"][[i, b]]) >= 0.8
                            and min(l["future_pct"][[i, b]]) < 0.2
                        )
                assert truth == r["answer"]
                assert r["evidence"] in {
                    evidence(t, truth if t != 3 else None, d, i, b, k) for k in range(9)
                }
                assert user_parts(r) == user_parts(
                    dict(
                        r,
                        answer={},
                        evidence="",
                        future_ret=123,
                        p_local=456,
                        edges=[],
                        sender_cf_delta=[],
                    )
                )
                total += 1
        summary[split] = total
    print(summary)


if __name__ == "__main__":
    verify()
