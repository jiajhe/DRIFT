"""Build the full chronological release with physically separate observations and supervision."""
import argparse
import bisect
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.data.data import ExpressionD
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.processor import RobustZScoreNorm, Fillna
import torch

from load_checkpoint import load_model
from meta_fdg.data.us_ohlcv_handler import alpha158_us_ohlcv_features
from meta_fdg.models.flow_fdg import center_stocks
from meta_fdg.training.flow_fdg import FlowStateBank
from build_marketflow_samples import partition
from marketflow_data import CELL, DATA, MEMORY, ROOT, VERSION, WINDOW, file_digest

TASKS = ("FORECAST", "FLOW_QA", "FLOW_MEMORY")
CATEGORIES = ("BOTTOM_20", "MIDDLE_60", "TOP_20")
INDEX_DTYPE = np.dtype([("day", "<u2"), ("target", "<u2"), ("task", "u1"), ("label", "u1")])


def historical_panels(cfg, out):
    data = cfg["data"]
    provider = Path(data["provider_path"])
    qlib.init(provider_uri=str(provider), region="us", kernels=4, expression_cache=None, dataset_cache=None)
    train_start, train_end = data["splits"]["train"]
    calendar = pd.DatetimeIndex(D.calendar(start_time=data["context_start_time"], end_time=data["splits"]["test"][1]))
    first_train = calendar.searchsorted(train_start)
    first_replay = first_train - cfg["train"]["validation_warmup_days"]
    first_row = first_replay - WINDOW
    recipe = alpha158_us_ohlcv_features()
    rights = [ExpressionD.get_expression_instance(expr).get_extended_window_size()[1] for expr in recipe[0]]
    if max(rights) != 0:
        raise ValueError("A DRIFT factor requires a future observation")
    loader = QlibDataLoader(config={"feature": recipe})
    market = D.instruments(data["market"])
    raw = loader.load(market, calendar[0], calendar[-1])
    # Verify truncating the source before the next session leaves the endpoint unchanged.
    cut = pd.Timestamp("2023-01-03")
    probe = loader.load(market, cut, cut)
    pd.testing.assert_frame_equal(raw.loc[cut:cut], probe, check_exact=True)
    prices = D.features(market, ["$close"], start_time=calendar[0], end_time=calendar[-1])
    prices = prices["$close"].unstack("instrument").sort_index(axis=1).reindex(calendar)
    names = prices.columns
    cube = np.full((len(calendar),len(names),len(recipe[0])), np.nan, dtype=np.float32)
    fits = []
    # Warm-up and the first training year use only the pre-replay calibration prefix.
    # Thereafter refit annually using strictly prior training years, freezing at train_end.
    for year in range(calendar[first_row].year, calendar[-1].year + 1):
        part = calendar[(calendar.year == year) & (calendar >= calendar[first_row])]
        if year <= pd.Timestamp(train_start).year:
            lo, hi = calendar[0], calendar[first_row-1]
        else:
            lo = pd.Timestamp(train_start)
            hi = min(pd.Timestamp(train_end), pd.Timestamp(year-1, 12, 31))
        if hi >= part[0]:
            raise ValueError("Normalizer overlaps its inference period")
        norm = RobustZScoreNorm(lo, hi, fields_group="feature", clip_outlier=True)
        norm.fit(raw.loc[lo:hi])
        normalized = Fillna(fields_group="feature")(norm(raw.loc[part[0]:part[-1]].copy()))["feature"]
        dates = calendar.get_indexer(normalized.index.get_level_values("datetime"))
        stocks = names.get_indexer(normalized.index.get_level_values("instrument"))
        if min(dates.min(), stocks.min()) < 0:
            raise ValueError("Factor index outside price panel")
        cube[dates, stocks] = normalized.to_numpy(dtype=np.float32)
        filename = f"normalizer-{year}.npz"
        np.savez(out/filename, mean=norm.mean_train, scale=norm.std_train, columns=np.asarray(recipe[1], dtype=str))
        fits.append(dict(year=year, fit_from=str(lo.date()), fit_through=str(hi.date()),
                         apply_from=str(part[0].date()), file=filename, sha256=file_digest(out/filename)))
    return calendar, names.to_numpy(dtype=str), prices.to_numpy(dtype=np.float64), cube, first_replay, fits


@torch.inference_mode()
def forward_day(model, bank, step, names, history, closes, materialize):
    # Only these 20 factor rows and 21 historical closes can reach the model.
    assert history.shape[0] == WINDOW and closes.shape[0] == WINDOW + 1
    keep = np.isfinite(history).all(axis=(0,2)) & np.isfinite(closes[-1]) & (closes[-1] > 0)
    ids = names[keep]
    x = torch.from_numpy(np.ascontiguousarray(history[:,keep].transpose(1,0,2))).to(next(model.parameters()).device)
    state, _ = bank.prepare(ids, day_index=step, reference=x)
    result = model(x, state=state, compute_graph_diagnostics=materialize)
    bank.update(ids, result.flow, day_index=step)
    if not materialize:
        return None
    d = result.diagnostics
    parts = torch.stack([center_stocks(d[k]) for k in ("P_keep","P_new","P_fwd")])
    comparisons = ((parts.sum(0), result.flow), (d["K"] @ d["U"], d["P_new"]),
                   (d["effective_alpha"]*d["F_minus"], d["P_keep"]),
                   (d["effective_rho"]*(d["K"] @ d["F_minus"]), d["P_fwd"]))
    errors = []
    for actual, expected in comparisons:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        errors.append((actual - expected).abs().max().item())
    tensors = {k:d[k] for k in ("K","Z","U","F","F_minus","S","R","alpha","rho","delta",
                                "effective_alpha","effective_rho")}
    tensors.update(components=parts, norms=parts.norm(dim=-1).T, prediction=result.prediction,
                   aligned_F_prev=state.flow_prev)
    arrays = {k:v.cpu().numpy() for k,v in tensors.items()}
    p = closes[:,keep].T
    valid = np.isfinite(p) & (p > 0)
    mask = valid[:,1:] & valid[:,:-1]
    returns = np.zeros(mask.shape, dtype=np.float32)
    returns[mask] = p[:,1:][mask] / p[:,:-1][mask] - 1
    arrays.update(instruments=ids, returns=returns, return_mask=mask)
    return arrays, errors


def routing(arrays, date, seed):
    # Universe is already ticker-sorted. A stable argsort breaks ties by ticker.
    weights = arrays["K"]
    strength = abs(weights)
    np.fill_diagonal(strength, -np.inf)
    top = np.argsort(-strength, kind="stable", axis=1)[:,:5]
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(date.replace("-", ""))]))
    perm = np.argsort(rng.random(top.shape), axis=1)
    senders = np.take_along_axis(top, perm, axis=1)
    edges = np.take_along_axis(weights, senders, axis=1)
    if (senders == np.arange(len(senders))[:,None]).any():
        raise ValueError("Self edge selected")
    return senders.astype(np.uint16), edges


def label_source(provider):
    cal = (provider/"calendars/day.txt").read_text().splitlines()
    intervals = [r.split() for r in (provider/"instruments/sp500_pit.txt").read_text().splitlines()]
    ids = sorted({r[0] for r in intervals})
    idmap = {s:i for i,s in enumerate(ids)}
    prices = np.full((len(cal),len(ids)), np.nan)
    pit = np.zeros(prices.shape, dtype=bool)
    for s,start,end in intervals:
        pit[bisect.bisect_left(cal,start):bisect.bisect_right(cal,end), idmap[s]] = True
    for s,col in idmap.items():
        path = provider/"features"/s.lower()/"close.day.bin"
        if path.exists():
            v = np.fromfile(path, dtype="<f4")
            offset = int(v[0])
            prices[offset:offset+len(v)-1,col] = v[1:]
    return cal, idmap, prices, pit


def supervision(arrays, day, future_prices, members, symbol_pos):
    names, edges, norms = (arrays[k] for k in ("instruments","edges","norms"))
    history_ok = arrays["return_mask"].all(1)
    eligible = history_ok & history_ok[arrays["senders"]].all(1)
    p1,p2 = future_prices
    has_y = members & np.isfinite(p1) & np.isfinite(p2) & (p1 > 0) & (p2 > 0)
    y = p2[has_y] / p1[has_y] - 1
    pct = (pd.Series(y).rank(method="average").to_numpy() - 0.5) / len(y)
    classes = np.full(len(members), -1, dtype=np.int8)
    classes[has_y] = np.select([pct < 0.2, pct >= 0.8],[0,2],default=1)
    cols = np.asarray([symbol_pos[s] for s in names])
    if not members[cols].all():
        raise ValueError("Non-PIT model input")
    labels = [classes[cols], abs(edges).argmax(1), norms.argmax(1)]
    unique = [labels[0] >= 0, (abs(edges)==abs(edges).max(1,keepdims=True)).sum(1)==1,
              (norms==norms.max(1,keepdims=True)).sum(1)==1]
    blocks = []
    for task in range(3):
        positions = np.flatnonzero(eligible & unique[task])
        chunk = np.empty(len(positions),dtype=INDEX_DTYPE)
        chunk["day"],chunk["target"],chunk["task"],chunk["label"] = day,positions,task,labels[task][positions]
        blocks.append(chunk)
    return blocks, int(eligible.sum())


def build(out, seed):
    model, cfg, _ = load_model(CELL, "cuda")
    model.requires_grad_(False)
    provider = Path(cfg["data"]["provider_path"])
    dates, names, prices, factors, first, fits = historical_panels(cfg, out)
    calendar = dates.strftime("%Y-%m-%d").tolist()
    splits = cfg["data"]["splits"]
    bank = FlowStateBank(flow_dim=model.flow_dim, suspension_max_days=model.suspension_max_days)
    records, maxerr = [], np.zeros(4)
    print("Causal factor panels ready; replaying chronological model state", flush=True)
    for i in range(first,len(dates)):
        date = calendar[i]
        split,_ = partition(calendar, splits, date)
        result = forward_day(model, bank, i, names, factors[i-WINDOW+1:i+1], prices[i-WINDOW:i+1], split is not None)
        if split is None:
            continue
        arrays, errors = result
        maxerr = np.maximum(maxerr,errors)
        arrays["senders"],arrays["edges"] = routing(arrays,date,seed)
        filename = date+".npz"
        np.savez(out/filename, **arrays)
        records.append(dict(date=date,split=split,file=filename,sha256=file_digest(out/filename),
                            window_start=calendar[i-WINDOW+1], preceding_close=calendar[i-WINDOW],
                            label_start=calendar[i+1],label_end=calendar[i+2]))
        if len(records)%250 == 0:
            print(f"sealed {len(records)} daily feature files through {date}",flush=True)
    del factors, prices
    # Second pass: forecast supervision can never affect the feature writers above.
    cal,idmap,raw_close,pit = label_source(provider)
    indexes = {s:[] for s in splits}
    aggregates = {s:Counter() for s in splits}
    for day,rec in enumerate(records):
        i = cal.index(rec["date"])
        with np.load(out/rec["file"],allow_pickle=False) as snap:
            a = {k:snap[k] for k in ("instruments","edges","norms","return_mask","senders")}
        blocks,n = supervision(a,day,raw_close[i+1:i+3],pit[i],idmap)
        indexes[rec["split"]].extend(blocks)
        counts = dict(zip(TASKS,map(len,blocks)))
        rec.update(observations=n,counts=counts)
        aggregates[rec["split"]].update(dict(days=1,observations=n,**counts))
    index_files = {}
    for split,blocks in indexes.items():
        index = np.concatenate(blocks)
        path = out/(split+".npy")
        np.save(path,index,allow_pickle=False)
        sub = [r for r in records if r["split"] == split]
        index_files[split] = dict(file=path.name,sha256=file_digest(path),tasks=len(index),
                                  first_date=sub[0]["date"],last_date=sub[-1]["date"],**aggregates[split])
    manifest = dict(format="marketflow-daily-v2",created=datetime.now(timezone.utc).isoformat(),
                    seed=seed,template_version=VERSION,lookback=WINDOW,cell=CELL,
                    checkpoint_sha256=file_digest(ROOT/"checkpoints"/f"{CELL}.pt"),
                    provider_release=provider.name,provider_manifest_sha256=file_digest(provider/"release_manifest.json"),
                    build_source_sha256=file_digest(Path(__file__)),splits=splits,normalizers=fits,days=records,
                    max_abs_decomposition_errors=maxerr.tolist(),feature_right_extension=0,
                    truncated_factor_probe="2023-01-03",supervision_dtype=INDEX_DTYPE.descr,index=index_files)
    (out/"manifest.json").write_text(json.dumps(manifest, separators=(",",":"))+"\n")
    print(json.dumps(index_files,indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=DATA/"history-v2")
    p.add_argument("--seed",type=int,default=12)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Choose a new release directory: {args.out}")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
        stage = Path(temp)/"release"
        stage.mkdir()
        build(stage,args.seed)
        stage.rename(args.out)


if __name__=="__main__":
    main()
