"""Two-stage DRIFT-TSLM v2 SFT, validation selection, and locked free-generation comparisons."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import random
import re
from time import monotonic
import numpy as np
import torch
from v2_drift_tslm import DriftTSLM, LoRA, ROOT, WEIGHTS, REV, parse
from v2_features import BASE
from prompt_templates_v2 import TASKS, response, legal_answer

RUN = ROOT / "runs/drift-tslm-v2.1"
SEED = 71


def dataset(split):
    return [
        json.loads(s)
        for s in (BASE / "tasks" / (split + ".jsonl")).read_text().splitlines()
    ]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(r):
    return {
        k: r[k]
        for k in ("date", "i", "b", "task", "ticker", "other", "query", "senders")
        if k in r
    }


@contextmanager
def merged(model):
    saved = []
    try:
        # Form the adapter update in FP32 in both the API and batched evaluation.
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            for m in model.decoder.modules():
                if isinstance(m, LoRA) and m.active:
                    saved.append((m, m.base.weight))
                    w = m.base.weight.float() + 2 * m.up.weight @ m.down.weight
                    m.base.weight = torch.nn.Parameter(
                        w.bfloat16(), requires_grad=False
                    )
                    m.active = False
        yield
    finally:
        for m, w in saved:
            m.base.weight = w
            m.active = True


def donor(model, row, rng):
    d, _ = model.source.day(row["date"])
    excluded = [row["i"]]
    if "b" in row:
        excluded.append(row["b"])
    avail = np.flatnonzero(d["role_valid"] & d["return_mask"].all(1))
    picks = rng.choice(np.setdiff1d(avail, excluded), size=2, replace=False)
    return dict(date=row["date"], i=int(picks[0]), b=int(picks[1]))


@torch.inference_mode()
def evaluate(model, rows, mode, path, batch=32):
    model.eval()
    rng = np.random.default_rng(173)
    totals = {i: dict(n=0, exact=0, format=0, fields={}) for i in range(4)}
    records = []
    with merged(model), torch.autocast("cuda", dtype=torch.bfloat16):
        for lo in range(0, len(rows), batch):
            xs = rows[lo : lo + batch]
            obs = [observed(r) for r in xs]
            ds = [donor(model, r, rng) for r in obs] if mode == "shuffled" else None
            texts = model.generate(obs, mode, ds)
            for r, txt in zip(xs, texts, strict=True):
                gold = r["answer"]
                pred = parse(txt)
                t = totals[r["task"]]
                t["n"] += 1
                body = (
                    txt.partition("<reasoning>")[2].partition("</reasoning>")[0].strip()
                )
                sentences = len(re.findall(r"[.!?](?:\s|$)", body))
                valid = legal_answer(r, pred) and 2 <= sentences <= 4
                t["format"] += int(valid)
                t["exact"] += int(valid and pred == gold)
                for key in gold:
                    t["fields"][key] = t["fields"].get(key, 0) + int(
                        valid and pred.get(key) == gold[key]
                    )
                records.append(
                    dict(
                        date=r["date"], i=r["i"], task=r["task"], text=txt, answer=gold
                    )
                )
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return {
        TASKS[i]: dict(**v, accuracy=v["exact"] / v["n"])
        for i, v in totals.items()
        if v["n"]
    }


def atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(variant, micro, eval_batch):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    rows = dataset("train")
    valid = dataset("valid")
    if len(rows) != 24000 or len(valid) != 2000:
        raise ValueError("Incomplete v2 cohorts")
    model = DriftTSLM(variant)
    run = RUN / variant
    run.mkdir(parents=True, exist_ok=True)
    out = WEIGHTS / variant
    last = out / "last.pt"
    bestpath = out / "best_valid.pt"
    saved = (
        torch.load(last, map_location="cpu", weights_only=True)
        if last.exists()
        else None
    )
    contract = {
        str(k): digest(BASE / "tasks" / (k + ".jsonl")) for k in ("train", "valid")
    }
    contract["stats"] = digest(BASE / "train_stats.json")
    if saved:
        if saved["data"] != contract:
            raise ValueError("Checkpoint belongs to another task release")
        model.restore(saved)
    best = saved["best"] if saved else -1.0
    total = sum(x.numel() for x in model.learned().values())
    print(variant, "trainable parameters in joint stage", total, flush=True)
    start = monotonic()
    for stage, steps in [(1, 750), (2, 5000)]:
        if saved and (
            stage < saved["stage"]
            or (stage == saved["stage"] and saved["step"] == steps)
        ):
            continue
        model.stage(stage, True)
        p = list(model.project.parameters())
        groups = [dict(params=p, lr=5e-4 if stage == 1 else 1e-4)]
        if stage == 2:
            groups.append(
                dict(
                    params=[
                        v
                        for k, v in model.learned().items()
                        if not k.startswith("project.")
                    ],
                    lr=2e-4,
                )
            )
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
        base_lrs = [g["lr"] for g in groups]
        rng = np.random.default_rng(SEED + stage)
        order = np.concatenate(
            [
                rng.permutation(len(rows))
                for _ in range(math.ceil(steps * 16 / len(rows)))
            ]
        )
        first = 0
        if saved and stage == saved["stage"]:
            opt.load_state_dict(saved["optimizer"])
            first = saved["step"]
            torch.cuda.set_rng_state(saved["cuda_rng"])
        losses = []
        for step in range(first + 1, steps + 1):
            chosen = [rows[j] for j in order[(step - 1) * 16 : step * 16]]
            scale = (
                step / max(1, int(0.05 * steps))
                if step <= 0.05 * steps
                else 0.5
                * (1 + math.cos(math.pi * (step - 0.05 * steps) / (0.95 * steps)))
            )
            for g, lr in zip(opt.param_groups, base_lrs, strict=True):
                g["lr"] = lr * scale
            opt.zero_grad(set_to_none=True)
            n = sum(model.loss_mass(r) for r in chosen)
            value = 0.0
            for k in range(0, 16, micro):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, count = model.loss(chosen[k : k + micro])
                    loss = loss / n
                loss.backward()
                value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]],
                1.0,
                error_if_nonfinite=True,
            )
            opt.step()
            losses.append(value)
            val = stage == 2 and step in (3000, 4000, 5000)
            if step % 50 == 0:
                print(
                    variant,
                    stage,
                    step,
                    "loss",
                    round(float(np.mean(losses[-50:])), 4),
                    "elapsed",
                    round(monotonic() - start),
                    flush=True,
                )
                with (run / "loss.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            dict(
                                stage=stage,
                                step=step,
                                loss=float(np.mean(losses[-50:])),
                            )
                        )
                        + "\n"
                    )
            if val or step % 100 == 0 or step == steps:
                core = dict(
                    variant=variant,
                    recipe="v2.1-weighted-answer",
                    revision=REV,
                    stage=stage,
                    step=step,
                    data=contract,
                    parameters={
                        k: v.detach().cpu() for k, v in model.learned().items()
                    },
                )
                if val:
                    model.stage(2, False)
                    scores = evaluate(
                        model, valid, "real", run / f"valid-{step}.jsonl", eval_batch
                    )
                    metric = np.mean([m["accuracy"] for m in scores.values()]).item()
                    (run / f"valid-{step}.json").write_text(
                        json.dumps(scores, indent=2)
                    )
                    print("validation", variant, step, scores, flush=True)
                    if metric > best:
                        best = metric
                        atomic(dict(**core, valid=scores), bestpath)
                    model.stage(2, True)
                atomic(
                    dict(
                        **core,
                        best=best,
                        optimizer=opt.state_dict(),
                        cuda_rng=torch.cuda.get_rng_state(),
                    ),
                    last,
                )
    print("training finished", variant, flush=True)


def test(variant, eval_batch):
    model = DriftTSLM("drift" if variant == "text" else variant)
    if variant != "text":
        model.restore(
            torch.load(
                WEIGHTS / variant / "best_valid.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
    else:
        model.stage(1, False)
    run = RUN / variant
    run.mkdir(parents=True, exist_ok=True)
    modes = {
        "drift": ["real", "shuffled", "zero", "local"],
        "returns": ["real"],
        "text": ["text"],
    }[variant]
    for mode in modes:
        for split in ("test", "test_random_pairs"):
            out = run / (split + "-" + mode + ".json")
            if out.exists():
                continue
            scores = evaluate(
                model, dataset(split), mode, out.with_suffix(".jsonl"), eval_batch
            )
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(scores, indent=2))
            tmp.replace(out)
            print(variant, mode, split, scores, flush=True)

    if variant == "drift":
        from v2_pairwise import evaluate_pairs

        with (
            merged(model),
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            evaluate_pairs(model, run, eval_batch)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("drift", "returns", "text"), default="drift")
    p.add_argument("--phase", choices=("train", "test"), default="train")
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=32)
    a = p.parse_args()
    if a.phase == "train":
        train(a.variant, a.microbatch, a.eval_batch)
    else:
        test(a.variant, a.eval_batch)
