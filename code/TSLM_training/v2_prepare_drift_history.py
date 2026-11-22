"""Export observed paper DRIFT outputs separately from supervision, in trading-day order."""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.stats import rankdata

from marketflow_data import CELL, DATA, ROOT

sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/src")]
from paper_ensemble import MANIFEST, load_members
from prepare_history import label_source, routing
from build_marketflow_samples import partition
from meta_fdg.training.flow_fdg import (
    FlowStateBank,
    bounded_flow_counterfactual_prediction,
)
from meta_fdg.models.flow_fdg import center_stocks

BASE = DATA / "history-v3"


def save_npz(path, **arrays):
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def panels(cfg):
    provider = ROOT / "code" / cfg["data"]["provider_path"]
    dates, idmap, prices, pit = label_source(provider)
    ids = np.array(sorted(idmap))
    cache = ROOT / "cache/compact" / cfg["data"]["data_id"] / "factors"
    (store,) = cache.glob("*/development/manifest.json")
    blocks = []
    for split in ("development", "test"):
        folder = store.parents[1] / split
        meta = json.loads((folder / "manifest.json").read_text())
        fit = meta["normalization_provenance"]
        if (fit["fit_scope"], fit["fit_start_time"], fit["fit_end_time"]) != (
            "train_only",
            "2018-01-01",
            "2022-12-31",
        ):
            raise ValueError("Unexpected Alpha158 normalization")
        d = (
            np.load(folder / "dates.npy")
            .astype("datetime64[ns]")
            .astype("datetime64[D]")
            .astype(str)
        )
        s = np.load(folder / "instruments.npy")
        blocks.append((d, s, np.load(folder / "features.npy", mmap_mode="r")))
    start = dates.index(min(blocks[0][0]))
    end = dates.index(max(blocks[-1][0])) + 1
    dates = dates[start:end]
    cube = np.full((len(dates), len(ids), 157), np.nan, dtype="float32")
    calendar = np.array(dates)
    for ds, ss, x in blocks:
        i, j = np.searchsorted(calendar, ds), np.searchsorted(ids, ss)
        if not np.array_equal(calendar[i], ds) or not np.array_equal(ids[j], ss):
            raise ValueError("Factor identifiers do not align with provider")
        cube[i, j] = x
    return dates, ids, prices[start:end], pit[start:end], cube, meta["feature_columns"]


class Trajectory:
    def __init__(self, model):
        gru = model.temporal_encoder
        if (gru.input_size, gru.hidden_size, gru.num_layers) != (157, 64, 2):
            raise ValueError(gru)
        self.lower = torch.nn.GRU(157, 64, batch_first=True).to(gru.weight_ih_l0.device)
        self.lower.load_state_dict(
            {k: gru.state_dict()[k] for k in self.lower.state_dict()}
        )
        self.lower.eval().requires_grad_(False)
        self.handle = gru.register_forward_hook(self.capture)

    def capture(self, module, args, result):
        top, last = result
        low, _ = self.lower(args[0])
        torch.testing.assert_close(low[:, -1], last[0], rtol=2e-5, atol=2e-6)
        if top.shape[1:] != (20, 64):
            raise ValueError(top.shape)
        self.values = torch.cat([low, top], -1)


def counterfactuals(model, out, senders, check):
    d = out.diagnostics
    n = len(senders)
    rows = torch.arange(n, device=senders.device).repeat_interleave(5)
    cols = senders.flatten()
    edge = d["K"][rows, cols]
    change = -edge[:, None] * (d["U"][cols] + d["effective_rho"] * d["F_minus"][cols])
    if (
        model.forwarding_projection is not None
        or model.flow_reliability_gate is not None
    ):
        raise ValueError("Unsupported counterfactual readout")

    def read(flow):
        return bounded_flow_counterfactual_prediction(model, d["Z"], flow)

    result = []
    # Both centering operations matter: every other stock's nonlinear readout also changes.
    for r, delta in zip(rows.split(128), change.split(128)):
        i = torch.arange(len(r), device=r.device)
        flows = out.flow[None] - delta[:, None, :] / n
        flows[i, r] += delta
        z = d["Z"].expand(len(r), -1, -1).reshape(-1, d["Z"].shape[-1])
        # The native function subtracts one scalar mean. Re-centering each scenario
        # removes that common offset and restores its independent market mean.
        joint = bounded_flow_counterfactual_prediction(
            model, z, flows.flatten(0, 1)
        ).view(len(r), n)
        alternative = joint - joint.mean(1, keepdim=True)
        result.append(alternative[i, r] - out.prediction[r])
    effects = torch.cat(result).reshape(n, 5)
    error = 0.0
    if check:
        for i, j in ((n // 3, 0), (n // 2, 3), (n - 1, 4)):
            k = d["K"].clone()
            k[i, senders[i, j]] = 0
            cf = center_stocks(
                d["P_keep"] + k @ d["U"] + d["effective_rho"] * (k @ d["F_minus"])
            )
            error = max(
                error, abs(float(read(cf)[i] - out.prediction[i] - effects[i, j]))
            )
        if error >= 1e-4:
            raise ValueError(f"counterfactual error {error}")
    return effects, error


def supervision(prices, pit, t, idx):
    p, q = prices[t + 1 : t + 3]
    mask = pit[t] & np.isfinite(p) & np.isfinite(q) & (p > 0) & (q > 0)
    y = np.full(len(p), np.nan)
    ranks = y.copy()
    y[mask] = q[mask] / p[mask] - 1
    ranks[mask] = (rankdata(y[mask]) - 0.5) / mask.sum()
    return y[idx].astype("float32"), ranks[idx].astype("float32")


@torch.inference_mode()
def build(root, limit, device):
    from time import monotonic

    root.mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    (root / "interventions").mkdir(exist_ok=True)
    spec = json.loads(MANIFEST.read_text())
    setting = spec["settings"][CELL]
    cfg = json.loads((ROOT / setting["runtime_config"]).read_text())
    models = [
        m.requires_grad_(False)
        for m, _ in load_members(setting, [0, 2, 5, 9], torch.device(device))
    ]
    primary = models[0]
    bank = [
        FlowStateBank(flow_dim=m.flow_dim, suspension_max_days=m.suspension_max_days)
        for m in models
    ]
    traj = Trajectory(primary)
    dates, names, prices, pit, cube, columns = panels(cfg)
    splits = cfg["data"]["splits"]
    keepdays = [i for i, d in enumerate(dates) if partition(dates, splits, d)[0]]
    checks = set(
        np.array(keepdays)[np.linspace(0, len(keepdays) - 1, 10).astype(int)].tolist()
    )
    # Initialize role history and flow before the first training observation.
    start = keepdays[0] - max(21, cfg["train"]["validation_warmup_days"])
    past = deque(maxlen=20)
    records = []
    metrics = []
    errors = {}
    maxflow = 0.0
    begin = monotonic()
    for t in range(start, keepdays[-1] + 1):
        hist = cube[t - 19 : t + 1]
        valid = (
            pit[t]
            & np.isfinite(hist).all(axis=(0, 2))
            & np.isfinite(prices[t])
            & (prices[t] > 0)
        )
        idx = np.flatnonzero(valid)
        stocks = names[idx]
        x = torch.as_tensor(
            np.ascontiguousarray(hist[:, idx].transpose(1, 0, 2)), device=device
        )
        preds = []
        for j, (model, states) in enumerate(zip(models, bank, strict=True)):
            state, _ = states.prepare(stocks, day_index=t, reference=x)
            output = model(x, state=state, compute_graph_diagnostics=(j == 0))
            states.update(stocks, output.flow, day_index=t)
            preds.append(output.prediction.double())
            if j == 0:
                out = output
        d = out.diagnostics
        to_np = lambda v: v.detach().cpu().numpy()
        role = np.stack([to_np(d["S"]), to_np(d["R"])], axis=1)
        full = np.full((len(names), 2, 16), np.nan, dtype="float32")
        full[idx] = role
        tv = (
            0.5 * np.abs(role - past[0][idx]).sum(-1)
            if len(past) == 20
            else np.full((len(idx), 2), np.nan)
        )
        past.append(full)
        split, _ = partition(dates, splits, dates[t])
        if not split:
            continue
        traj20 = traj.values
        parts = torch.stack([center_stocks(d[k]) for k in ("P_keep", "P_new", "P_fwd")])
        err = (parts.sum(0) - out.flow).abs().max().item()
        if err > 2e-5:
            raise ValueError(f"Flow decomposition error {err}")
        maxflow = max(maxflow, err)
        send, edge = routing({"K": to_np(d["K"])}, dates[t], 71)
        cf, err = counterfactuals(
            primary,
            out,
            torch.as_tensor(send.astype("int64"), device=device),
            t in checks,
        )
        if t in checks:
            errors[dates[t]] = err
        ps = torch.stack(preds)
        sd = ps.std(1, keepdim=True, unbiased=False)
        stdps = (ps - ps.mean(1, keepdim=True)) / torch.where(sd > 0, sd, 1.0)
        ensemble = to_np(stdps.mean(0)).astype("float32")
        prices20 = prices[t - 20 : t + 1, idx].T
        mask = np.isfinite(prices20) & (prices20 > 0)
        rm = mask[:, 1:] & mask[:, :-1]
        ret = np.zeros(rm.shape, dtype="float32")
        ret[rm] = prices20[:, 1:][rm] / prices20[:, :-1][rm] - 1
        datum = dict(
            instruments=stocks,
            Z_patches=to_np(traj20.reshape(-1, 5, 4, 128).mean(2).half()),
            sender_halves=to_np(traj20.reshape(-1, 2, 10, 128).mean(2).half()),
            Z_last=to_np(d["Z"]),
            p_local=to_np(out.p_local),
            prediction=to_np(out.prediction),
            flow_delta=to_np(out.prediction - out.p_local),
            ens_score=ensemble,
            ens_std=to_np(stdps.std(0, unbiased=False)).astype("float32"),
            F=to_np(out.flow),
            F_minus=to_np(d["F_minus"]),
            U=to_np(d["U"]),
            components=to_np(parts),
            norms=to_np(parts.norm(dim=-1).T),
            M_s=role[:, 0],
            M_r=role[:, 1],
            g_s=to_np(d["sender_activity"]).reshape(-1),
            g_r=to_np(d["receiver_activity"]).reshape(-1),
            role_tv_20d=np.nan_to_num(tv).astype("float32"),
            role_valid=np.isfinite(tv).all(1),
            regime_scalars=to_np(
                torch.stack(
                    [
                        d[k].reshape(())
                        for k in (
                            "alpha",
                            "rho",
                            "delta",
                            "gamma",
                            "operator_frobenius",
                        )
                    ]
                )
            ),
            k_frobenius=to_np(d["K"].norm()),
            senders=send,
            edges=edge,
            returns_20d=ret,
            return_mask=rm,
        )
        save_npz(root / (dates[t] + ".npz"), **datum)
        future, pct = supervision(prices, pit, t, idx)
        np.save(
            root / "interventions" / (dates[t] + ".npy"), to_np(cf), allow_pickle=False
        )
        save_npz(
            root / "labels" / (dates[t] + ".npz"),
            future_ret=future,
            future_pct=pct,
            sender_cf_delta=to_np(cf),
        )
        if split == "test":
            good = np.isfinite(future)
            metrics.append(
                [
                    np.corrcoef(ensemble[good], future[good])[0, 1],
                    np.corrcoef(rankdata(ensemble[good]), rankdata(future[good]))[0, 1],
                    good.sum(),
                ]
            )
        records.append(
            dict(
                date=dates[t],
                split=split,
                n=len(idx),
                window_start=dates[t - 19],
                label_end=dates[t + 2],
            )
        )
        if len(records) % 50 == 0 or len(records) == 1:
            print(
                len(records),
                dates[t],
                len(idx),
                round(monotonic() - begin, 1),
                flush=True,
            )
        if limit and len(records) >= limit:
            break
    traj.handle.remove()
    summary = dict(days=len(records), max_flow_error=maxflow, full_replay_errors=errors)
    if metrics:
        m = np.array(metrics)
        summary.update(
            test_ic=float(m[:, 0].mean()),
            test_rankic=float(m[:, 1].mean()),
            test_rows=int(m[:, 2].sum()),
        )
    manifest = dict(
        format="drift-history-v3",
        seeds=[0, 2, 5, 9],
        structural_seed=0,
        checkpoints=setting["members"],
        factor_order=columns,
        days=records,
        summary=summary,
    )
    temp = root / "manifest.tmp"
    temp.write_text(json.dumps(manifest, separators=(",", ":")))
    temp.replace(root / "manifest.json")
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=BASE)
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.limit and args.out == BASE:
        p.error("Use a separate output for a limited export")
    build(args.out, args.limit, args.device)
