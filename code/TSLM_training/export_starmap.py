"""Export a compact single-day S&P 500 Market Star Map payload from history-v3.

Embeds all S&P 500 stocks inside a 3D volumetric sphere (x^2 + y^2 + z^2 <= R^2,
filling the 3D interior rather than a hollow shell) using 3D spring relaxation
so that geometric 3D edge lengths stay inversely proportional to |K[i, j]|.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HISTORY_V3 = ROOT / "data/prompt_data/local/history-v3"
DEFAULT_OUT = ROOT / "web/starmap-demo.json"
DEFAULT_DATE = "2024-01-02"


def _rank_corr(a, b):
    """Spearman rank correlation without SciPy dependency."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra @ rb) / den) if den > 1e-12 else 0.0


def compute_sphere_volume_coords(z_last, senders, edges, steps=200):
    """Embed all stocks inside a 3D volumetric sphere where length tracks |K[i, j]|.

    - Starts from deterministic 3D PCA of DRIFT's 64-dim latent state Z_last
      scaled isotropically into the interior of the unit sphere.
    - Runs full 3D Hookean spring relaxation on all top-5 directed edges so
      stronger |K[i, j]| pairs sit proportionally closer in 3D space.
    - Uses 3D short/mid-range pairwise repulsion so stocks populate the full
      interior volume evenly without clumping or collapsing onto a hollow shell.
    """
    n = z_last.shape[0]
    x_feat = z_last.astype(np.float64)
    x_feat = x_feat - x_feat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_feat, full_matrices=False)
    pca3 = x_feat @ vt[:3].T
    for axis in range(3):
        idx = int(np.argmax(np.abs(pca3[:, axis])))
        if pca3[idx, axis] < 0:
            pca3[:, axis] = -pca3[:, axis]

    # Normalize each axis cleanly into an isotropic 3D interior cloud.
    axis_scale = np.percentile(np.abs(pca3), 95, axis=0)
    axis_scale = np.where(axis_scale > 1e-8, axis_scale, 1.0)
    anchor = np.tanh(pca3 / (axis_scale * 1.15)) * 0.78

    # Flatten directed top-5 edges into (dst, src, |K|).
    dst = np.repeat(np.arange(n, dtype=np.int64), senders.shape[1])
    src = senders.reshape(-1).astype(np.int64)
    w_abs = np.abs(edges.reshape(-1).astype(np.float64))

    # Proportional inverse mapping from |K| to 3D target distance in [0.16, 0.96].
    lo, hi = np.percentile(w_abs, [2, 98])
    span = max(float(hi - lo), 1e-8)
    w_norm = np.clip((w_abs - lo) / span, 0.0, 1.0)
    d_target = 0.96 - 0.80 * (w_norm ** 0.72)
    stiffness = 0.35 + 0.95 * w_norm

    pos = anchor.copy()
    repel_radius = 0.28
    for step in range(steps):
        lr = 0.14 * (1.0 - 0.65 * (step / steps))
        grad = np.zeros_like(pos)

        # 1) 3D weighted spring force matching edge length to d_target(|K[i, j]|).
        diff = pos[dst] - pos[src]
        dist = np.linalg.norm(diff, axis=1) + 1e-6
        force_mag = stiffness * (dist - d_target) / dist
        f_vec = diff * force_mag[:, None]
        np.add.at(grad, dst, -f_vec)
        np.add.at(grad, src, f_vec)

        # 2) Full 3D volumetric repulsion so stars fill the sphere interior cleanly.
        pdiff = pos[:, None, :] - pos[None, :, :]
        pdist = np.linalg.norm(pdiff, axis=2) + 1e-6
        close = (pdist < repel_radius) & (pdist > 1e-5)
        rep_mag = np.where(close, 0.045 * ((repel_radius - pdist) / repel_radius) ** 2 / pdist, 0.0)
        grad += np.einsum("ij,ijk->ik", rep_mag, pdiff)

        # 3) Gentle 3D PCA anchor + soft outer spherical boundary (interior stays free).
        grad += 0.085 * (anchor - pos)
        rad = np.linalg.norm(pos, axis=1) + 1e-6
        outside = np.maximum(0.0, rad - 0.92)
        grad -= (0.50 * outside / rad)[:, None] * pos

        g_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        pos += lr * grad / np.maximum(1.0, g_norm / 0.28)

    # Recenter and scale gently so 99% of stocks sit cleanly inside radius 0.94.
    pos -= pos.mean(axis=0, keepdims=True)
    r_all = np.linalg.norm(pos, axis=1)
    r_scale = float(np.percentile(r_all, 99.0)) or 1.0
    pos = pos / r_scale * 0.92
    r_clipped = np.linalg.norm(pos, axis=1)
    over = r_clipped > 0.98
    if np.any(over):
        pos[over] *= (0.98 / r_clipped[over])[:, None]

    final_dist = np.linalg.norm(pos[dst] - pos[src], axis=1)
    corr = _rank_corr(w_abs, final_dist)
    radii = np.linalg.norm(pos, axis=1)
    return pos, corr, np.percentile(radii, [10, 50, 90])


def build_starmap_payload(date=DEFAULT_DATE, history_dir=HISTORY_V3):
    npz_path = Path(history_dir) / f"{date}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {npz_path}")

    manifest = json.loads((Path(history_dir) / "manifest.json").read_text())
    rec = next((d for d in manifest["days"] if d["date"] == date), None)
    window_start = rec["window_start"] if rec else ""

    with np.load(npz_path, allow_pickle=False) as d:
        instruments = d["instruments"].tolist()
        z_last = d["Z_last"]
        senders = d["senders"].astype(int)
        edges = d["edges"].astype(np.float64)
        ens_score = d["ens_score"].astype(np.float64)
        p_local = d["p_local"].astype(np.float64)
        flow_delta = d["flow_delta"].astype(np.float64)
        g_s = d["g_s"].astype(np.float64)
        g_r = d["g_r"].astype(np.float64)

    n = len(instruments)
    net_role = g_s - g_r
    coords, edge_corr, rad_pcts = compute_sphere_volume_coords(z_last, senders, edges)

    out_deg = np.bincount(senders.reshape(-1), minlength=n)

    # Role x Score alternative coordinates normalized to [-0.92, 0.92].
    rx = net_role - np.median(net_role)
    rx_scale = float(np.percentile(np.abs(rx), 98)) or 1.0
    rx = np.clip(rx / rx_scale, -1.0, 1.0) * 0.90

    ry = ens_score - np.median(ens_score)
    ry_scale = float(np.percentile(np.abs(ry), 98)) or 1.0
    ry = np.clip(ry / ry_scale, -1.0, 1.0) * 0.88

    nodes = []
    for i in range(n):
        in_edges = [
            [int(senders[i, k]), round(float(edges[i, k]), 6)]
            for k in range(senders.shape[1])
        ]
        in_edges.sort(key=lambda pair: abs(pair[1]), reverse=True)
        nodes.append(
            dict(
                id=instruments[i],
                x=round(float(coords[i, 0]), 4),
                y=round(float(coords[i, 1]), 4),
                z=round(float(coords[i, 2]), 4),
                rx=round(float(rx[i]), 4),
                ry=round(float(ry[i]), 4),
                s=round(float(ens_score[i]), 4),
                loc=round(float(p_local[i]), 4),
                flw=round(float(flow_delta[i]), 4),
                gs=round(float(g_s[i]), 4),
                gr=round(float(g_r[i]), 4),
                out=int(out_deg[i]),
                in_edges=in_edges,
            )
        )

    flat = []
    for i in range(n):
        for k in range(senders.shape[1]):
            flat.append((abs(float(edges[i, k])), int(senders[i, k]), i, float(edges[i, k])))
    flat.sort(key=lambda t: t[0], reverse=True)
    skeleton = [[src, dst, round(w, 6)] for _, src, dst, w in flat[:180]]

    return dict(
        dataset_version="history-v3",
        market="SP500",
        date=date,
        window_start=window_start,
        universe_size=n,
        edge_count=int(n * senders.shape[1]),
        edge_length_spearman=round(edge_corr, 4),
        radius_percentiles=[round(float(v), 3) for v in rad_pcts],
        layout_note="3D volumetric sphere from DRIFT Z_last with |K[i,j]| proportional 3D spring layout",
        skeleton=skeleton,
        nodes=nodes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = build_starmap_payload(date=args.date)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    args.out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.chmod(args.out, 0o664)
    print(
        f"Wrote {args.out} ({args.out.stat().st_size:,} bytes, "
        f"{payload['universe_size']} stocks, {payload['edge_count']} edges, "
        f"Spearman(|K|, dist3D)={payload['edge_length_spearman']:+.4f}, "
        f"radius p10/p50/p90={payload['radius_percentiles']}, date={payload['date']})"
    )


if __name__ == "__main__":
    main()
