"""Prepare actual model outputs for 96 rich historical stock-days (12 dates × 8 stocks) in the static demo."""

import hashlib
import json
from pathlib import Path
import re

import torch
from api.main import snapshot, intervene, EdgeRequest
from prompt_templates_v2 import TASKS, legal_answer
from v2_features import ROOT
from v2_drift_tslm import DriftTSLM, WEIGHTS, parse
from v2_train import merged

DATES = (
    "2024-01-02",
    "2024-01-03",
    "2024-03-15",
    "2024-07-01",
    "2024-07-02",
    "2024-09-18",
    "2024-11-06",
    "2025-01-30",
    "2025-04-09",
    "2025-08-15",
    "2025-12-26",
    "2025-12-29",
)
TARGETS = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM")


def _r(val, digits=6):
    return round(float(val), digits)


@torch.inference_mode()
def export():
    path = WEIGHTS / "drift/best_valid.pt"
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    result = json.loads((ROOT / "reports/v2_evaluation.json").read_text())
    if result["checkpoint_sha256"] != digest:
        raise ValueError("The published evaluation and the demo weights differ")
    model = DriftTSLM()
    model.restore(torch.load(path, map_location="cpu", weights_only=True))
    archive = dict(
        schema=1,
        dataset_version="history-v3",
        checkpoint_sha256=digest,
        gate=result["gate"],
        days={},
    )
    jobs = []
    for day in DATES:
        d, _ = model.source.day(day)
        ids = d["instruments"].tolist()
        if model.source.days[day]["split"] != "test":
            raise ValueError(day)
        observations = archive["days"][day] = {}
        for target in TARGETS:
            i = ids.index(target)
            sender_idxs = d["senders"][i].tolist()
            senders = [ids[idx] for idx in sender_idxs]
            snap = snapshot(day, target)
            # Attach strictly day-t observed DRIFT telemetry & 20-day return curves for rich visualization
            snap["telemetry"] = dict(
                p_local=_r(d["p_local"][i]),
                flow_delta=_r(d["flow_delta"][i]),
                seed0_score=_r(d["prediction"][i]),
                ens_std=_r(d["ens_std"][i]),
                g_s=_r(d["g_s"][i], 4),
                g_r=_r(d["g_r"][i], 4),
                role_tv_20d=_r(float(d["role_tv_20d"][i].mean()), 4),
                gamma_k=_r(float(d["regime_scalars"][3] * d["k_frobenius"]), 5),
            )
            snap["returns_20d"] = {
                target: [round(float(x), 5) for x in d["returns_20d"][i]],
                **{
                    ids[idx]: [round(float(x), 5) for x in d["returns_20d"][idx]]
                    for idx in sender_idxs
                },
            }
            record = dict(snapshot=snap, interventions={}, answers={})
            observations[target] = record
            base = dict(date=day, ticker=target, i=i, senders=senders)
            for t, task in enumerate(TASKS):
                params = (
                    senders
                    if t == 1
                    else [s for s in TARGETS if s != target]
                    if t == 3
                    else [""]
                )
                for key in params:
                    row = dict(base, task=t)
                    if t == 1:
                        row["query"] = key
                        record["interventions"][key] = intervene(
                            EdgeRequest(date=day, target=target, sender=key)
                        )
                    if t == 3:
                        row.update(other=key, b=ids.index(key))
                    jobs.append((row, record["answers"].setdefault(task, {}), key))
    print(f"Running real GPU inference for {len(jobs)} prompts across {len(DATES) * len(TARGETS)} stock-days...", flush=True)
    with merged(model), torch.autocast("cuda", dtype=torch.bfloat16):
        for j in range(0, len(jobs), 32):
            group = jobs[j : j + 32]
            texts = model.generate([row for row, _, _ in group])
            for (row, bucket, key), text in zip(group, texts, strict=True):
                answer = parse(text)
                body = (
                    text.partition("<reasoning>")[2]
                    .partition("</reasoning>")[0]
                    .strip()
                )
                if (
                    not legal_answer(row, answer)
                    or not 2 <= len(re.findall(r"[.!?](?:\s|$)", body)) <= 4
                ):
                    raise ValueError(
                        f"Malformed real generation for {row['date']} {row['ticker']} {row['task']} {key}: {text}"
                    )
                bucket[key] = dict(reasoning=body, answer=answer)
            if (j // 32) % 8 == 0:
                print(f"  completed {min(j + 32, len(jobs))}/{len(jobs)}", flush=True)
    out = ROOT / "web/historical-demo.json"
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(archive, separators=(",", ":"), allow_nan=False) + "\n")
    temp.replace(out)
    print(
        f"DONE: {len(DATES) * len(TARGETS)} stock-days, {len(jobs)} responses, {len(DATES) * len(TARGETS) * 5} interventions, {out.stat().st_size} bytes",
        flush=True,
    )


if __name__ == "__main__":
    export()
