"""Read the A-vs-B margin from the selected Qwen model on the locked forecast pairs."""

import json

import numpy as np
from scipy.stats import rankdata
import torch

from v2_features import BASE

WINNER_PREFIX = '<answer>{"winner":"'


def binary_metrics(gold, scores):
    gold = np.asarray(gold, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if not scores.shape == gold.shape or not np.isfinite(scores).all():
        raise ValueError("Missing winner scores")
    pos, neg = int(gold.sum()), int((~gold).sum())
    if not pos or not neg:
        raise ValueError("AUC requires both directions")
    auc = (rankdata(scores)[gold].sum() - pos * (pos + 1) / 2) / (pos * neg)
    correct = int(np.sum(((scores > 0) == gold) & (scores != 0)))
    ties = int(np.sum(scores == 0))
    return dict(
        n=len(gold),
        correct=correct,
        ties=ties,
        win_rate=(correct + ties / 2) / len(gold),
        auc=float(auc),
    )


def winner_prefix(output):
    # Only reuse the model's own explanation. Never read an answer token.
    reasoning, close, _ = output.partition("</reasoning>")
    if close and reasoning.startswith("<reasoning>") and "<answer>" not in reasoning:
        return reasoning + close + "\n" + WINNER_PREFIX, True
    return "<reasoning></reasoning>\n" + WINNER_PREFIX, False


def evaluate_pairs(model, run, batch):
    tokens = [model.encode(x) for x in ("A", "B")]
    if not all(len(t) == 1 for t in tokens):
        raise ValueError("Winner choices must be single tokens")
    choices = [x[0] for x in tokens]
    summary = {}
    for split in ("test", "test_random_pairs"):
        truth = [
            json.loads(l)
            for l in (BASE / "tasks" / f"{split}.jsonl").read_text().splitlines()
        ]
        preds = [
            json.loads(l)
            for l in (run / f"{split}-real.jsonl").read_text().splitlines()
        ]
        pairs = []
        for row, pred in zip(truth, preds, strict=True):
            if (row["date"], row["i"], row["task"]) != (
                pred["date"],
                pred["i"],
                pred["task"],
            ):
                raise ValueError("Misaligned prediction")
            if row["task"] == 3:
                prefix, complete = winner_prefix(pred["text"])
                # An allow-list keeps labels and reference explanations out of the scoring path.
                obs = {k: row[k] for k in ("date", "task", "i", "b", "ticker", "other")}
                pairs.append((obs, prefix, complete, row["answer"]["winner"] == "A"))
        if len(pairs) != 1000:
            raise ValueError("Wrong pair count")
        records = []
        for start in range(0, len(pairs), batch):
            part = pairs[start : start + batch]
            rows, prefixes, _, _ = zip(*part, strict=True)
            for prefix in prefixes:
                if any(
                    model.encode(prefix + choice) != model.encode(prefix) + [token]
                    for choice, token in zip(("A", "B"), choices, strict=True)
                ):
                    raise ValueError(
                        "Tokenizer boundary differs at the winner position"
                    )
            em, mask, _ = model.prepare(
                rows, "real", None, False, continuations=prefixes
            )
            pos = (mask.cumsum(1) - 1).clamp_min(0)
            h = model.decoder(
                inputs_embeds=em, attention_mask=mask, position_ids=pos, use_cache=False
            ).last_hidden_state[:, -1]
            logits = model.head(h).float()[:, choices]
            margins = (logits[:, 0] - logits[:, 1]).tolist()
            for (row, _, complete, actual), margin in zip(part, margins, strict=True):
                d, _ = model.source.day(row["date"])
                ens = d["ens_score"]
                records.append(
                    dict(
                        date=row["date"],
                        a=row["ticker"],
                        b=row["other"],
                        gold_A=actual,
                        ensemble_margin=float(ens[row["i"]]) - float(ens[row["b"]]),
                        logit_margin=margin,
                        generated_reasoning=complete,
                    )
                )
        target = [r["gold_A"] for r in records]
        summary[split] = dict(
            native=binary_metrics(target, [r["ensemble_margin"] for r in records]),
            logits=binary_metrics(target, [r["logit_margin"] for r in records]),
            missing_reasoning=sum(not r["generated_reasoning"] for r in records),
        )
        (run / f"{split}-winner-scores.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
    (run / "pairwise_scores.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print("paired forecasts", summary, flush=True)
