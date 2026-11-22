"""Historical DRIFT v3 explorer and v2 Qwen analyst."""

from datetime import date as Date
from functools import lru_cache
import json
import os
import re
from threading import Lock
from time import perf_counter
from typing import Literal
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from v2_features import ROOT, BASE
from prompt_templates_v2 import TASKS, legal_answer

app = FastAPI(title="DRIFT-TSLM v2 / MarketFlow")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MARKETFLOW_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
lock = Lock()
COMPONENTS = ("RETENTION", "NEW_TRANSMISSION", "RETRANSMISSION")


class EdgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    market: Literal["SP500"] = "SP500"
    date: Date
    target: str
    sender: str


class CopilotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    market: Literal["SP500"] = "SP500"
    date: Date
    target: str
    task: Literal[
        "ATTRIBUTION_COT",
        "COUNTERFACTUAL_COT",
        "REGIME_ROLE_COT",
        "PAIRWISE_FORECAST_COT",
    ]
    sender: str | None = None
    other: str | None = None


@lru_cache(maxsize=1)
def days():
    path = BASE / "manifest.json"
    if not path.exists():
        raise HTTPException(503, "history-v3 is unavailable")
    return {r["date"]: r for r in json.loads(path.read_text())["days"]}


@lru_cache(maxsize=10)
def day(date):
    if date not in days():
        raise HTTPException(404, "Unknown historical observation date")
    with np.load(BASE / (date + ".npz"), allow_pickle=False) as f:
        return {
            k: f[k]
            for k in (
                "instruments",
                "senders",
                "edges",
                "prediction",
                "ens_score",
                "ens_std",
                "norms",
                "role_valid",
                "return_mask",
            )
        }


def lookup(date, target):
    d = day(str(date))
    ids = d["instruments"].tolist()
    if target.upper() not in ids:
        raise HTTPException(404, "Ticker outside this day's universe")
    i = ids.index(target.upper())
    return d, ids, i


@app.get("/api/catalog")
def catalog(date: Date | None = None):
    ds = days()
    selected = (
        str(date) if date else next(d for d, r in ds.items() if r["split"] == "test")
    )
    a = day(selected)
    return dict(
        date=selected,
        dates=list(ds),
        targets=a["instruments"].tolist(),
        dataset_version="history-v3",
    )


@app.get("/api/snapshot")
def snapshot(date: Date, target: str, market: Literal["SP500"] = "SP500"):
    a, ids, i = lookup(date, target)
    s = a["senders"][i]
    return dict(
        date=str(date),
        target=ids[i],
        market=market,
        window_start=days()[str(date)]["window_start"],
        dataset_version="history-v3",
        universe_size=len(ids),
        nodes=[ids[i], *[ids[j] for j in s]],
        edges=[
            dict(source=ids[j], target=ids[i], weight=float(k))
            for j, k in zip(s, a["edges"][i], strict=True)
        ],
        forecast=dict(
            drift_score=float(a["ens_score"][i]),
            seed0_score=float(a["prediction"][i]),
            ensemble_std=float(a["ens_std"][i]),
        ),
        components=dict(zip(COMPONENTS, a["norms"][i].tolist(), strict=True)),
        intervention_available=True,
    )


@app.post("/api/intervene")
def intervene(req: EdgeRequest):
    a, ids, i = lookup(req.date, req.target)
    candidates = [ids[j] for j in a["senders"][i]]
    if req.sender.upper() not in candidates:
        raise HTTPException(422, "Select a displayed upstream edge")
    j = candidates.index(req.sender.upper())
    delta = float(
        np.load(
            BASE / "interventions" / (str(req.date) + ".npy"),
            allow_pickle=False,
            mmap_mode="r",
        )[i, j]
    )
    original = float(a["prediction"][i])
    return dict(
        status="ok",
        model_id="paper-drift-seed0",
        dataset_version="history-v3",
        target=ids[i],
        sender=candidates[j],
        original_score=original,
        intervened_score=original + delta,
        delta=delta,
        original_edge=float(a["edges"][i, j]),
        masked_edge=0.0,
        method="Frozen model, one normalized incoming edge masked, both cross-sectional centerings recomputed",
    )


@lru_cache(maxsize=1)
def decoder():
    from v2_drift_tslm import DriftTSLM

    return DriftTSLM("drift")


@lru_cache(maxsize=1)
def model_at(path, modified):
    import torch

    # The caller holds the inference lock. Reload only the trained adapters when
    # validation selects new weights, reusing the existing Qwen allocation.
    model = decoder()
    model.restore(torch.load(path, map_location="cpu", weights_only=True))
    return model


@app.post("/api/copilot")
def copilot(req: CopilotRequest):
    a, ids, i = lookup(req.date, req.target)
    row = dict(
        date=str(req.date),
        task=TASKS.index(req.task),
        ticker=ids[i],
        i=i,
        senders=[ids[j] for j in a["senders"][i]],
    )
    if not a["role_valid"][i]:
        raise HTTPException(422, "A full role lookback is unavailable on this date")
    if row["task"] == 1:
        row["query"] = (req.sender or "").upper()
        if row["query"] not in row["senders"]:
            raise HTTPException(422, "Choose one of the displayed upstream stocks")
    if row["task"] == 3:
        other = (req.other or "").upper()
        if other not in ids or other == ids[i]:
            raise HTTPException(422, "Choose a different stock from this date")
        row.update(other=other, b=ids.index(other))
        if not a["role_valid"][row["b"]]:
            raise HTTPException(422, "Comparison stock has incomplete role history")
    path = ROOT / "checkpoints/TSLM_training/v2.1/drift/best_valid.pt"
    if not path.exists():
        raise HTTPException(503, "The v2 validation-selected checkpoint is pending")
    import torch
    from v2_drift_tslm import parse
    from v2_train import merged

    start = perf_counter()
    with lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = model_at(path, path.stat().st_mtime_ns)
        with merged(model):
            text = model.generate([row])[0]
    ans = parse(text)
    body = text.partition("<reasoning>")[2].partition("</reasoning>")[0].strip()
    if (
        not legal_answer(row, ans)
        or not 2 <= len(re.findall(r"[.!?](?:\s|$)", body)) <= 4
    ):
        raise HTTPException(502, "The model produced an invalid structured answer")
    status = ROOT / "reports/v2_evaluation.json"
    result = json.loads(status.read_text()) if status.is_file() else {}
    gate = result["gate"] if result.get("model_revision") == "v2.1" else "pending"
    return dict(
        status="ok",
        model_id="DRIFT-TSLM-v2.1-Qwen3.5-4B",
        dataset_version="history-v3",
        grounding_gate=gate,
        task=req.task,
        sample_id="SP500|" + str(req.date) + "|" + ids[i],
        reasoning=body,
        answer=ans,
        inference_ms=round((perf_counter() - start) * 1000, 1),
    )


@app.get("/api/evaluation")
def evaluation():
    p = ROOT / "reports/v2_evaluation.json"
    result = json.loads(p.read_text()) if p.is_file() else {}
    return result if result.get("model_revision") == "v2.1" else dict(model_revision="v2.1", gate="pending")


app.mount(
    "/",
    StaticFiles(directory=ROOT / "web/dist", html=True, check_dir=False),
    name="demo",
)
