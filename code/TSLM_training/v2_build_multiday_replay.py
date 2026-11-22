"""Replay two paper seed-0 DRIFT models with one normalized incoming edge masked across sessions."""

import argparse
import bisect
import copy
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/src")]
from paper_ensemble import MANIFEST, load_members
from meta_fdg.data.compact_qlib import compact_store_ids, materialize_configs
from meta_fdg.models.flow_fdg import center_stocks
from meta_fdg.training.flow_fdg import (
    FlowStateBank,
    bounded_flow_counterfactual_prediction,
)

MARKETS = {
    "SP500": (
        "alpha158us_sp500_pit",
        (
            "AAPL",
            "MSFT",
            "NVDA",
            "JPM",
            "XOM",
            "AMZN",
            "LLY",
            "GOOGL",
            "WMT",
            "AVGO",
            "META",
            "JNJ",
            "CAT",
            "HD",
        ),
    ),
    "CSI300": (
        "alpha158_csi300",
        (
            "SH600519",
            "SZ300750",
            "SH601318",
            "SH600036",
            "SZ000858",
            "SZ000333",
            "SH601012",
            "SH600276",
            "SH601899",
            "SH600900",
            "SZ002594",
            "SH600030",
            "SH600887",
            "SH600309",
        ),
    ),
}
TOL = 2e-6


def panel(config, width):
    data = config["data"]
    for key in ("provider_path", "cache_root", "materialized_cache_root"):
        path = (ROOT / "code" / data[key]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Expected a repository-local {key}")
        data[key] = str(path)
    provider = Path(data["provider_path"])
    cache = Path(data["materialized_cache_root"])
    factor_id, _ = compact_store_ids(config)
    folder = cache / data["data_id"] / "factors" / factor_id
    parts = ("development", "test")
    if any(not (folder / part / "MATERIALIZATION_COMPLETE").exists() for part in parts):
        materialize_configs([config], cache_root=cache, partitions=parts)

    all_dates = (provider / "calendars/day.txt").read_text().splitlines()
    begin = bisect.bisect_left(all_dates, data["context_start_time"])
    end = bisect.bisect_right(all_dates, data["splits"]["test"][1])
    dates = np.array(all_dates[begin:end])
    intervals = [
        line.split()
        for line in (provider / "instruments" / (data["market"] + ".txt"))
        .read_text()
        .splitlines()
    ]
    names = np.array(sorted({s for s, _, _ in intervals}))
    positions = {s: j for j, s in enumerate(names)}
    eligible = np.zeros((len(dates), len(names)), dtype=bool)
    for s, first, last in intervals:
        eligible[
            dates.searchsorted(first) : dates.searchsorted(last, side="right"),
            positions[s],
        ] = True
    for s, j in positions.items():
        path = provider / "features" / s.lower() / "close.day.bin"
        valid = np.zeros(len(dates), dtype=bool)
        if path.exists():
            raw = np.fromfile(path, dtype="<f4")
            start = int(raw[0])
            left, right = max(begin, start), min(end, start + len(raw) - 1)
            if left < right:
                close = raw[1 + left - start : 1 + right - start]
                valid[left - begin : right - begin] = np.isfinite(close) & (close > 0)
        eligible[:, j] &= valid

    cube = np.full((*eligible.shape, width), np.nan, dtype=np.float32)
    columns = None
    for part in parts:
        base = folder / part
        meta = json.loads((base / "manifest.json").read_bytes())
        fit = meta["normalization_provenance"]
        if [fit["fit_start_time"], fit["fit_end_time"]] != data["splits"][
            "train"
        ] or fit["fit_scope"] != "train_only":
            raise ValueError("Unexpected normalization period")
        if columns is not None and columns != meta["feature_columns"]:
            raise ValueError("Development and test factor orders differ")
        columns = meta["feature_columns"]
        ds = (
            np.load(base / "dates.npy")
            .astype("datetime64[ns]")
            .astype("datetime64[D]")
            .astype(str)
        )
        ss = np.load(base / "instruments.npy")
        i, j = dates.searchsorted(ds), names.searchsorted(ss)
        if not np.array_equal(dates[i], ds) or not np.array_equal(names[j], ss):
            raise ValueError("Unaligned Qlib factor rows")
        cube[i, j] = np.load(base / "features.npy", mmap_mode="r")
    return dates, names, eligible, cube, factor_id


def fork(bank):
    # update() replaces each stored tensor. Fork the maps; previous immutable rows are shared.
    branch = copy.copy(bank)
    branch.values = bank.values.copy()
    branch.last_seen = bank.last_seen.copy()
    branch.pattern_values = bank.pattern_values.copy()
    return branch


@dataclass
class Day:
    index: int
    date: str
    names: tuple
    x: torch.Tensor
    reference: torch.Tensor | None = None


def forward(model, bank, day, *, dense=False, edge=None, operator=False):
    state, _ = bank.prepare(day.names, day_index=day.index, reference=day.x)
    out = model(day.x, state=state, compute_graph_diagnostics=dense or operator)
    flow, scores = out.flow, out.prediction
    if dense:
        d = out.diagnostics
        k = d["K"]
        if edge is not None:
            i, j = map(day.names.index, edge)
            k = k.clone()
            k[i, j] = (
                0  # Mask the already normalized operator; preserve all other entries.
            )
        fwd = d["effective_rho"] * (k @ d["F_minus"])
        flow = center_stocks(d["P_keep"] + k @ d["U"] + fwd)
        scores = bounded_flow_counterfactual_prediction(
            model, d["Z"], flow, forwarding=fwd
        )
    bank.update(day.names, flow, day_index=day.index)
    return scores, out.diagnostics.get("K")


def channel(model, initial, before_last, days, target, sender, base):
    bank = fork(initial)
    edge = (target, sender)
    first = []
    for d in days[:-1]:
        scores, _ = forward(model, bank, d, dense=True, edge=edge)
        first.append(scores[d.names.index(target)].item())
    final = days[-1]
    i = final.names.index(target)
    # Branch BEFORE updating day five. Both long masks inherit the identical four-day history.
    restored, _ = forward(model, fork(bank), final)
    five, _ = forward(model, bank, final, dense=True, edge=edge)
    today, _ = forward(model, fork(before_last), final, dense=True, edge=edge)
    # The today-only branch shares the four observed baseline forwards up to its fork.
    scores = dict(
        today=[*base[:4], today[i].item()],
        five=[*first, five[i].item()],
        restored=[*first, restored[i].item()],
    )
    return {
        mode: {"scores": ys, "delta": [a - b for a, b in zip(ys, base, strict=True)]}
        for mode, ys in scores.items()
    }


def preference(a, b):
    def contrast(mode):
        x, y = [abs(s[mode]["delta"][-1]) for s in (a, b)]
        return (x - y) / max(x, y) if x or y else 0.0

    today, five = contrast("today"), contrast("five")
    reversal = today * five < 0 and min(abs(today), abs(five)) >= 0.05
    gain = max(
        abs(s["five"]["delta"][-1]) / abs(s["today"]["delta"][-1]) for s in (a, b)
    )
    memory = min(abs(s["restored"]["delta"][-1]) for s in (a, b))
    return reversal, gain >= 1.25, memory


def select_case(model, initial, days, target_pool, used):
    raw, dense = fork(initial), fork(initial)
    max_native = max_dense = 0.0
    baseline = []
    for day in days:
        if day is days[-1]:
            before_last = fork(raw)
        native, k = forward(model, raw, day, operator=day is days[-1])
        explicit, _ = forward(model, dense, day, dense=True)
        native_err = (native - day.reference).abs().max().item()
        dense_err = (explicit - day.reference).abs().max().item()
        if not np.isfinite([native_err, dense_err]).all():
            raise ValueError(f"Nonfinite flow replay on {day.date}")
        max_native = max(max_native, native_err)
        max_dense = max(max_dense, dense_err)
        baseline.append(native)
    if max(max_native, max_dense) >= TOL:
        raise ValueError(
            f"Independent unmasked replay mismatch {max_native}, {max_dense}"
        )
    final = days[-1]
    common = set.intersection(*(set(d.names) for d in days))
    targets = dict.fromkeys([*target_pool, *sorted(common)])
    for target in targets:
        if target not in common or target in used:
            continue
        i = final.names.index(target)
        edges = k[i].cpu().numpy()
        # Rank against the entire final market before requiring five-session membership.
        ranked = [j for j in np.argsort(-abs(edges), kind="stable") if j != i][:10]
        base = [
            s[d.names.index(target)].item() for s, d in zip(baseline, days, strict=True)
        ]
        candidates = {}
        for rank, j in enumerate(ranked, 1):
            s = final.names[j]
            if s not in common:
                continue
            paths = channel(model, initial, before_last, days, target, s, base)
            if any(not np.isfinite(paths[m]["scores"]).all() for m in paths):
                raise ValueError("Nonfinite counterfactual")
            if (
                abs(paths["restored"]["delta"][-1]) > 5 * TOL
                and paths["today"]["delta"][-1] != 0
            ):
                candidates[s] = (rank, float(edges[j]), paths)
        if len(candidates) < 2:
            continue
        a, b = max(
            combinations(candidates, 2),
            key=lambda pair: preference(candidates[pair[0]][2], candidates[pair[1]][2]),
        )
        ra, ka, pa = candidates[a]
        rb, kb, pb = candidates[b]
        return dict(
            date=final.date,
            window_dates=[d.date for d in days],
            target=target,
            sender_a=a,
            sender_b=b,
            edges_on_final_day={a: ka, b: kb},
            final_abs_edge_ranks={a: ra, b: rb},
            base_scores=base,
            trajectories={a: pa, b: pb},
            unmasked_max_abs_error=max_native,
            explicit_unmasked_max_abs_error=max_dense,
        )
    raise ValueError(f"No eligible pair on {final.date}")


@torch.inference_mode()
def build_market(market, device):
    cell, targets = MARKETS[market]
    setting = json.loads(MANIFEST.read_bytes())["settings"][cell]
    (member,) = (m for m in setting["members"] if m["seed"] == 0)
    model, ck = next(load_members({"members": [member]}, [0], device))
    model.requires_grad_(False)
    assert ck["lookback"] == 20 and model.readout_mode == "bounded_shared_gated"
    cfg = json.loads((ROOT / setting["runtime_config"]).read_bytes())
    dates, names, eligible, cube, factor_id = panel(cfg, ck["feature_dim"])
    splits = cfg["data"]["splits"]
    test = np.flatnonzero((dates >= splits["test"][0]) & (dates <= splits["test"][1]))
    finals = test[np.linspace(4, len(test) - 3, len(targets), dtype=int)]
    windows = {int(end) - 4: j for j, end in enumerate(finals)}
    start = int(dates.searchsorted(splits["train"][0])) - max(
        21, cfg["train"]["validation_warmup_days"]
    )
    bank = FlowStateBank(
        flow_dim=model.flow_dim, suspension_max_days=model.suspension_max_days
    )
    cases, used = [], set()
    for t in range(start, int(finals[-1]) + 1):
        hist = cube[t - 19 : t + 1]
        idx = np.flatnonzero(eligible[t] & np.isfinite(hist).all(axis=(0, 2)))
        x = torch.from_numpy(np.ascontiguousarray(hist[:, idx].transpose(1, 0, 2))).to(
            device
        )
        instruments = tuple(names[idx].tolist())
        if t in windows:
            initial, window = fork(bank), []
        day = Day(t, str(dates[t]), instruments, x)
        day.reference, _ = forward(model, bank, day)
        if any(t - offset in windows for offset in range(5)):
            window.append(day)
            if len(window) == 5:
                j = windows[t - 4]
                case = select_case(
                    model, initial, window, targets[j:] + targets[:j], used
                )
                used.add(case["target"])
                cases.append(dict(market=market, **case))
                print(
                    market,
                    case["date"],
                    case["target"],
                    case["sender_a"],
                    case["sender_b"],
                    flush=True,
                )
    return dict(
        cell=cell,
        seed=0,
        checkpoint_sha256=member["sha256"],
        factor_id=factor_id,
        factor_count=ck["feature_dim"],
        replay_start=str(dates[start]),
        cases=cases,
    )


def validate(archive):
    if archive["version"] != 1 or archive["markets"].keys() != MARKETS.keys():
        raise ValueError("Expected both frozen markets in the replay archive")
    for market, data in archive["markets"].items():
        cases = data["cases"]
        if not 12 <= len(cases) <= 15 or len({c["target"] for c in cases}) != len(
            cases
        ):
            raise ValueError("Invalid market coverage")
        for c in cases:
            dates = c["window_dates"]
            if (
                c["market"] != market
                or len(set(dates)) != 5
                or sorted(dates) != dates
                or dates[-1] != c["date"]
                or dates[0] < "2024-01-01"
                or dates[-1] > "2025-12-31"
                or not all(
                    0 <= c[k] < TOL
                    for k in (
                        "unmasked_max_abs_error",
                        "explicit_unmasked_max_abs_error",
                    )
                )
            ):
                raise ValueError("Invalid unmasked replay or test window")
            a, b = c["sender_a"], c["sender_b"]
            if len({a, b, c["target"]}) != 3 or any(
                not 1 <= c["final_abs_edge_ranks"][s] <= 10 for s in (a, b)
            ):
                raise ValueError("Invalid directed channels")
            for s in (a, b):
                modes = c["trajectories"][s]
                for trajectory in modes.values():
                    expected = np.asarray(trajectory["scores"]) - c["base_scores"]
                    if (
                        expected.shape != (5,)
                        or not np.isfinite(expected).all()
                        or not np.array_equal(expected, trajectory["delta"])
                    ):
                        raise ValueError("Invalid daily score difference")
                if (
                    modes["today"]["delta"][:4] != [0.0] * 4
                    or modes["today"]["delta"][4] == 0
                    or modes["restored"]["delta"][4] == 0
                    or modes["five"]["scores"][:4] != modes["restored"]["scores"][:4]
                ):
                    raise ValueError("Invalid mask schedule")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the released JSON without replaying the models",
    )
    args = parser.parse_args()
    out = ROOT / "web/multiday-replay.json"
    if args.check:
        archive = json.loads(out.read_bytes())
        validate(archive)
    else:
        torch.set_num_threads(1)
        archive = {
            "version": 1,
            "markets": {m: build_market(m, torch.device(args.device)) for m in MARKETS},
        }
        validate(archive)
        with NamedTemporaryFile(mode="w", dir=out.parent, delete=False) as temp:
            json.dump(archive, temp, separators=(",", ":"), allow_nan=False)
            temp.write("\n")
        Path(temp.name).chmod(0o644)
        Path(temp.name).replace(out)
    for market, data in archive["markets"].items():
        cases = data["cases"]
        print(
            market,
            len(cases),
            "cases; maximum native / explicit unmasked error:",
            max(c["unmasked_max_abs_error"] for c in cases),
            max(c["explicit_unmasked_max_abs_error"] for c in cases),
        )
