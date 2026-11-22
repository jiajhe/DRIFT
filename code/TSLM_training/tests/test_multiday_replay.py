"""Independently check every released trajectory using the native low-rank propagation."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from v2_build_multiday_replay import ROOT, MANIFEST, MARKETS, TOL, fork, panel, validate
from paper_ensemble import load_members
from meta_fdg.models import flow_fdg
from meta_fdg.training.flow_fdg import FlowStateBank


@pytest.mark.parametrize("market", MARKETS)
@torch.inference_mode()
def test_native_masked_replay(market):
    # Subtract one sender's payload inside the original implicit operator. This oracle
    # never builds a masked K or uses the exporter's explicit recurrence/readout.
    archive = json.loads((ROOT / "web/multiday-replay.json").read_bytes())
    validate(archive)
    export = archive["markets"][market]
    cell = export["cell"]
    setting = json.loads(MANIFEST.read_bytes())["settings"][cell]
    (seed,) = [m for m in setting["members"] if m["seed"] == 0]
    assert seed["sha256"] == export["checkpoint_sha256"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    model, ck = next(load_members({"members": [seed]}, [0], device))
    config = json.loads((ROOT / setting["runtime_config"]).read_bytes())
    dates, names, valid, cube, factor_id = panel(config, ck["feature_dim"])
    assert factor_id == export["factor_id"]
    cases = {c["window_dates"][0]: c for c in export["cases"]}
    start = int(np.flatnonzero(dates == export["replay_start"])[0])
    end = int(dates.searchsorted(export["cases"][-1]["date"]))
    bank = FlowStateBank(
        flow_dim=model.flow_dim, suspension_max_days=model.suspension_max_days
    )
    native_apply = flow_fdg.implicit_offdiag_apply
    active = None
    max_error = 0.0
    checked = 0
    for t in range(start, end + 1):
        hist = cube[t - 19 : t + 1]
        ids = np.flatnonzero(valid[t] & np.isfinite(hist).all(axis=(0, 2)))
        stocks = names[ids].tolist()
        x = torch.as_tensor(
            np.ascontiguousarray(hist[:, ids].transpose(1, 0, 2)), device=device
        )
        if dates[t] in cases:
            active = cases[dates[t]]
            channels = (active["sender_a"], active["sender_b"])
            branches = {
                (s, mode): fork(bank)
                for s in channels
                for mode in ("today", "five", "restored")
            }
            first = t
        state, _ = bank.prepare(stocks, day_index=t, reference=x)
        last = active is not None and t - first == 4
        ref = model(x, state=state, compute_graph_diagnostics=last)
        bank.update(stocks, ref.flow, day_index=t)
        if active is None:
            continue
        slot = t - first
        assert active["window_dates"][slot] == dates[t]
        i = stocks.index(active["target"])
        baseline = ref.prediction[i].item()
        assert abs(baseline - active["base_scores"][slot]) < TOL
        for (sender, mode), branch in branches.items():
            j = stocks.index(sender)
            if last:
                edges = ref.diagnostics["K"][i].cpu().numpy()
                ranked = [r for r in np.argsort(-abs(edges), kind="stable") if r != i]
                assert (
                    ranked.index(j) + 1 == active["final_abs_edge_ranks"][sender] <= 10
                )
                assert abs(float(edges[j]) - active["edges_on_final_day"][sender]) < TOL
            masked = mode == "five" or (slot == 4 if mode == "today" else slot < 4)

            def mask_one(receiver, operator, source, payload, gamma, *, eps):
                mixed, diag, norm, scale = native_apply(
                    receiver, operator, source, payload, gamma, eps=eps
                )
                weight = ((receiver[i] @ operator) * source[j]).sum() * scale
                mixed[i] -= weight * payload[j]
                return mixed, diag, norm, scale

            state, _ = branch.prepare(stocks, day_index=t, reference=x)
            with patch.object(
                flow_fdg, "implicit_offdiag_apply", mask_one if masked else native_apply
            ):
                result = model(x, state=state)
            branch.update(stocks, result.flow, day_index=t)
            expected = active["trajectories"][sender][mode]
            score = result.prediction[i].item()
            max_error = max(
                max_error,
                abs(expected["scores"][slot] - score),
                abs(expected["delta"][slot] - (score - baseline)),
            )
            checked += 1
        if last:
            active = None
    assert checked == len(cases) * 2 * 3 * 5 and max_error < TOL
    print(
        market,
        checked,
        "native masked score/delta checks; maximum absolute error",
        max_error,
    )
