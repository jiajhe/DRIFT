"""Integration checks against a real historical observation and a trained checkpoint."""

import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)


OBS = dict(market="SP500", date="2023-01-03", target="AAPL")


def test_observed_routes_and_directed_mask(client):
    response = client.get("/api/snapshot", params=OBS)
    assert response.status_code == 200
    snap = response.json()
    assert (
        snap["date"] == OBS["date"]
        and len(snap["nodes"]) == 6
        and len(snap["edges"]) == 5
    )
    assert all(edge["target"] == "AAPL" for edge in snap["edges"])
    assert not ({"answer", "label", "label_end"} & snap.keys())
    edge = snap["edges"][0]
    r = client.post("/api/intervene", json={**OBS, "sender": edge["source"]})
    assert r.status_code == 200
    changed = r.json()
    assert changed["original_edge"] == edge["weight"] and changed["masked_edge"] == 0.0
    assert abs(changed["original_score"] - snap["forecast"]["seed0_score"]) < 2e-6
    assert np.isclose(
        changed["delta"], changed["intervened_score"] - changed["original_score"]
    )
    assert changed["delta"] != 0
    assert client.get("/api/snapshot", params=OBS).json() == snap


def test_catalog_and_missing_inputs(client):
    catalog = client.get("/api/catalog", params={"date": OBS["date"]}).json()
    assert "AAPL" in catalog["targets"] and catalog["dates"] == sorted(
        set(catalog["dates"])
    )
    assert (
        client.get("/api/snapshot", params={**OBS, "date": "2023-01-01"}).status_code
        == 404
    )
    assert (
        client.get("/api/snapshot", params={**OBS, "target": "MISSING"}).status_code
        == 404
    )
    assert (
        client.post("/api/intervene", json={**OBS, "sender": "AAPL"}).status_code == 422
    )
    assert client.post("/api/copilot", json={**OBS, "task": "OTHER"}).status_code == 422
    assert (
        client.post(
            "/api/copilot", json={**OBS, "task": "FORECAST", "answer": "TOP_20"}
        ).status_code
        == 422
    )


@pytest.mark.skipif(
    os.environ.get("MARKETFLOW_GPU_TEST") != "1",
    reason="Set MARKETFLOW_GPU_TEST=1 to load the GPU checkpoint",
)
def test_loaded_copilot(client):
    snap = client.get("/api/snapshot", params=OBS).json()
    for task in (
        "ATTRIBUTION_COT",
        "COUNTERFACTUAL_COT",
        "REGIME_ROLE_COT",
        "PAIRWISE_FORECAST_COT",
    ):
        r = client.post(
            "/api/copilot",
            json={
                **OBS,
                "task": task,
                "sender": snap["edges"][0]["source"],
                "other": "MSFT",
            },
        )
        assert r.status_code == 200, r.text
        answer = r.json()
        assert answer["model_id"] == "DRIFT-TSLM-v2.1-Qwen3.5-4B"
        assert (
            answer["reasoning"]
            and isinstance(answer["answer"], dict)
            and answer["inference_ms"] > 0
        )
