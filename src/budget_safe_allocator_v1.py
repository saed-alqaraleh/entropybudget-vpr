#!/usr/bin/env python3
"""
Q1-safe matching-budget reconstruction and query-adaptive allocation.

This script fixes two methodological issues in the original budget simulation:

1) A budget m may use local matching evidence from ONLY the top-m global candidates.
   It must not use local reliability, RRF ranks, or top-k support computed from
   candidates m+1 ... K.

2) A query-budget allocator may not use local evidence from candidates that were
   not paid for. The retrieval-only allocator below uses only global retrieval
   statistics. The optional two-stage allocator first pays for candidate 1, then
   uses only that candidate's geometric evidence to decide whether deeper matching
   is worthwhile.

The script operates entirely on compact cached retrieval/scoring NPZ files.
It does not need datasets, descriptor caches, or the 6 GB raw matching directory.

Dependencies:
    numpy, pandas, scikit-learn
    project source package: src/vprbudget

Outputs:
    q1_safe_uniform_budget_curve.csv
    q1_safe_allocator_lodo.csv
    q1_safe_anytime_stage1_lodo.csv
    q1_safe_audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def row_minmax(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    lo = a.min(axis=1, keepdims=True)
    hi = a.max(axis=1, keepdims=True)
    span = hi - lo
    out = np.divide(a - lo, span, out=np.zeros_like(a), where=span >= eps)
    return np.where(span < eps, 0.5, out)


def row_ranks_desc(a: np.ndarray) -> np.ndarray:
    order = np.argsort(-a, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rr = np.broadcast_to(np.arange(1, a.shape[1] + 1), a.shape)
    np.put_along_axis(ranks, order, rr, axis=1)
    return ranks


def row_softmax_entropy(score: np.ndarray, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(score, float) / max(float(temperature), 1e-12)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    if score.shape[1] <= 1:
        h = np.zeros(score.shape[0], dtype=float)
    else:
        h = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1) / np.log(score.shape[1])
    return p, h


# ---------------------------------------------------------------------------
# Budget-faithful frozen fusion
# ---------------------------------------------------------------------------

def budget_safe_top1(
    G: np.ndarray,
    inlier_count: np.ndarray,
    match_count: np.ndarray,
    spatial: np.ndarray,
    global_reliability: np.ndarray,
    m: int,
    cfg,
) -> np.ndarray:
    """Return the selected top-1 candidate index under budget m.

    Only top-m globally retrieved candidates are allowed to contribute local
    evidence to local score normalization, local reliability, RRF, and top-k
    support. All retrieval evidence remains available because retrieval has
    already been performed.
    """
    G = np.asarray(G, float)
    n = np.asarray(inlier_count, float)
    mc = np.asarray(match_count, float)
    sp = np.asarray(spatial, float)
    cg = np.asarray(global_reliability, float)

    N, K = G.shape
    m = int(np.clip(m, 0, K))
    order_g = np.argsort(-G, axis=1, kind="stable")

    if m <= 1:
        return order_g[:, 0]

    head = order_g[:, :m]
    Gh = np.take_along_axis(G, head, axis=1)
    nh = np.take_along_axis(n, head, axis=1)
    mh = np.take_along_axis(mc, head, axis=1)
    sh = np.take_along_axis(sp, head, axis=1)

    rho = nh / np.maximum(mh, 1.0)
    L = (
        cfg.w_inlier_count * row_minmax(np.log1p(nh))
        + cfg.w_inlier_ratio * rho
    )
    SL = L * (cfg.gamma_spatial + (1.0 - cfg.gamma_spatial) * sh)

    p, ent = row_softmax_entropy(SL, cfg.softmax_temperature)
    order_l = np.argsort(-SL, axis=1, kind="stable")
    q1 = np.take_along_axis(p, order_l[:, :1], axis=1)[:, 0]
    sorted_sl = np.take_along_axis(SL, order_l, axis=1)
    span = sorted_sl[:, 0] - sorted_sl[:, -1]
    dl = np.divide(
        sorted_sl[:, 0] - sorted_sl[:, 1],
        span,
        out=np.zeros(N, dtype=float),
        where=span > 1e-12,
    )

    il_norm = row_minmax(np.log1p(nh))
    il = np.take_along_axis(il_norm, order_l[:, :1], axis=1)[:, 0]
    rl = np.take_along_axis(rho, order_l[:, :1], axis=1)[:, 0]

    b = cfg.local_reliability_weights
    cl = b[0] * q1 + b[1] * dl + b[2] * il + b[3] * rl + b[4] * (1.0 - ent)
    lam = np.clip(
        cg / (cg + cl + cfg.epsilon),
        cfg.lambda_min,
        cfg.lambda_max,
    )

    methods = [
        Gh,
        L,
        0.7 * Gh + 0.3 * L,
        0.7 * Gh + 0.3 * SL,
        lam[:, None] * Gh + (1.0 - lam)[:, None] * L,
        lam[:, None] * Gh + (1.0 - lam)[:, None] * SL,
    ]

    ranks = [row_ranks_desc(x) for x in methods]
    rrf = sum(1.0 / (cfg.rrf_k + r) for r in ranks)

    tk = np.zeros_like(Gh)
    for r in ranks:
        for cutoff in cfg.topk_support_cutoffs:
            tk += (r <= cutoff) * (1.0 - (r - 1.0) / cutoff)

    adaptive = methods[-1]
    if cfg.normalize_fusion_terms:
        adaptive = row_minmax(adaptive)
        rrf = row_minmax(rrf)
        tk = row_minmax(tk)

    w = cfg.fuf_weights
    fused = w[0] * adaptive + w[1] * rrf + w[2] * tk
    best_in_head = np.argmax(fused, axis=1)
    return head[np.arange(N), best_in_head]


# ---------------------------------------------------------------------------
# Leakage-free allocator features
# ---------------------------------------------------------------------------

def retrieval_only_features(dist: np.ndarray, G: np.ndarray) -> np.ndarray:
    X = []
    K = dist.shape[1]

    for d, g in zip(dist, G):
        d = np.asarray(d, float)
        span = max(float(d.max() - d.min()), 1e-12)
        dn = (d - d.min()) / span
        s = np.sort(dn)
        gs = np.sort(np.asarray(g, float))[::-1]

        concentrations = []
        for T in (0.03, 0.05, 0.10, 0.20):
            z = -dn / T
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            h = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum() / np.log(K)
            concentrations.append(1.0 - h)

        z = -dn / 0.10
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        ps = np.sort(p)[::-1]

        k10 = min(10, K)
        slope = np.polyfit(np.arange(k10), s[:k10], 1)[0] if k10 > 1 else 0.0

        X.append([
            s[1] - s[0] if K > 1 else 0.0,
            s[min(4, K - 1)] - s[0],
            s[min(9, K - 1)] - s[0],
            s[min(19, K - 1)] - s[0],
            np.mean(s[:5]),
            np.mean(s[:10]),
            np.mean(s[:20]),
            np.std(s),
            slope,
            gs[0] - gs[1] if K > 1 else 0.0,
            gs[0] - gs[min(4, K - 1)],
            gs[0] - gs[min(9, K - 1)],
            np.mean(gs[:5]) - np.mean(gs[5:20]) if K >= 20 else 0.0,
            *concentrations,
            ps[0],
            ps[:5].sum(),
            ps[:10].sum(),
        ])

    return np.asarray(X, dtype=float)


def stage1_features(
    dist: np.ndarray,
    G: np.ndarray,
    inlier_count: np.ndarray,
    match_count: np.ndarray,
    spatial: np.ndarray,
) -> np.ndarray:
    """Retrieval evidence + evidence from exactly one paid-for top-1 match."""
    Xr = retrieval_only_features(dist, G)
    order_g = np.argsort(-G, axis=1, kind="stable")
    i1 = order_g[:, 0]
    row = np.arange(len(i1))

    n = inlier_count[row, i1].astype(float)
    m = match_count[row, i1].astype(float)
    s = spatial[row, i1].astype(float)
    rho = n / np.maximum(m, 1.0)

    Xg = np.c_[
        np.log1p(n),
        np.log1p(m),
        rho,
        s,
        np.log1p(n) * s,
        (n >= 8).astype(float),
        (n >= 15).astype(float),
        (rho >= 0.2).astype(float),
    ]
    return np.c_[Xr, Xg]


# ---------------------------------------------------------------------------
# Expected-value allocator
# ---------------------------------------------------------------------------

def fit_value_models(X: np.ndarray, Y: np.ndarray, levels: np.ndarray, base_budget: int):
    base_idx = int(np.where(levels == base_budget)[0][0])
    models = []
    for j, m in enumerate(levels):
        if m == base_budget:
            models.append(None)
            continue
        model = HistGradientBoostingRegressor(
            max_iter=120,
            max_leaf_nodes=7,
            learning_rate=0.05,
            l2_regularization=3.0,
            random_state=0,
        )
        model.fit(X, Y[:, j] - Y[:, base_idx])
        models.append(model)
    return models


def predict_value(models, X: np.ndarray, levels: np.ndarray, base_budget: int) -> np.ndarray:
    P = np.zeros((len(X), len(levels)), dtype=float)
    for j, m in enumerate(levels):
        if m != base_budget:
            P[:, j] = models[j].predict(X)
    return P


def choose_under_lambda(pred_value: np.ndarray, levels: np.ndarray, lam: float) -> np.ndarray:
    utility = pred_value - lam * levels[None, :]
    return levels[np.argmax(utility, axis=1)]


def enforce_mean_budget(
    pred_value: np.ndarray,
    levels: np.ndarray,
    target: float,
) -> np.ndarray:
    """Use predictions only; no test labels enter resource calibration."""
    lambdas = np.r_[0.0, np.logspace(-8, 1, 2000)]
    best = None
    for lam in lambdas:
        m = choose_under_lambda(pred_value, levels, lam)
        mean = float(m.mean())
        if mean <= target + 1e-12:
            gap = target - mean
            if best is None or gap < best[0]:
                best = (gap, lam, m)
    if best is None:
        return np.full(len(pred_value), int(levels.min()), dtype=int)
    return best[2]


def evaluate_budgets(m: np.ndarray, Y: np.ndarray, levels: np.ndarray) -> float:
    lookup = {int(v): j for j, v in enumerate(levels)}
    return float(np.mean([Y[i, lookup[int(v)]] for i, v in enumerate(m)]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace-root", required=True)
    ap.add_argument("--run-id", default="run02")
    ap.add_argument("--config", default="config_run02_with_sf_partial_final_v2.json")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    W = Path(args.workspace_root).expanduser().resolve()
    RUN = W / "runs" / args.run_id
    OUT = Path(args.output_dir).expanduser().resolve() if args.output_dir else W / "q1_safe_results"
    OUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(W / "src"))
    from vprbudget.config import load_config
    from vprbudget.groundtruth import get_spec, positives

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = W / cfg_path
    cfg = load_config(str(cfg_path))

    datasets = list(cfg.datasets)
    levels = np.asarray(cfg.budget_levels, dtype=int)
    target_budgets = [b for b in cfg.mean_budgets if b <= max(levels)]

    data = {}
    uniform_rows = []

    for ds in datasets:
        ret = np.load(RUN / "retrieval" / f"{ds}.npz")
        sc = np.load(RUN / "scoring" / f"{ds}.npz")
        methods = json.loads((RUN / "scoring" / f"{ds}_methods.json").read_text())
        gi = methods.index("global_only")
        G = sc["scores"][:, gi, :].astype(float)
        pos = positives(ret["geo_distances"], get_spec(ds, cfg.dataset_gt_overrides))

        Y = np.zeros((len(G), len(levels)), dtype=np.int8)
        for j, m in enumerate(levels):
            top = budget_safe_top1(
                G,
                sc["inlier_count"],
                sc["match_count"],
                sc["spatial"],
                sc["global_reliability"],
                int(m),
                cfg,
            )
            Y[:, j] = pos[np.arange(len(pos)), top].astype(np.int8)
            uniform_rows.append({
                "dataset": ds,
                "budget": int(m),
                "n": int(len(pos)),
                "recall@1": float(Y[:, j].mean()),
            })

        X_retr = retrieval_only_features(ret["descriptor_distances"], G)
        X_stage1 = stage1_features(
            ret["descriptor_distances"],
            G,
            sc["inlier_count"],
            sc["match_count"],
            sc["spatial"],
        )
        data[ds] = {"Y": Y, "X_retr": X_retr, "X_stage1": X_stage1}

    pd.DataFrame(uniform_rows).to_csv(OUT / "q1_safe_uniform_budget_curve.csv", index=False)

    # Leave-one-dataset-out retrieval-only allocator.
    allocator_rows = []
    for held in datasets:
        train = [d for d in datasets if d != held]
        Xtr = np.vstack([data[d]["X_retr"] for d in train])
        Ytr = np.vstack([data[d]["Y"] for d in train])
        Xte = data[held]["X_retr"]
        Yte = data[held]["Y"]

        models = fit_value_models(Xtr, Ytr, levels, base_budget=0)
        pred = predict_value(models, Xte, levels, base_budget=0)

        for B in target_budgets:
            m = enforce_mean_budget(pred, levels, float(B))
            r1 = evaluate_budgets(m, Yte, levels)
            j = int(np.where(levels == int(B))[0][0])
            uniform = float(Yte[:, j].mean())
            allocator_rows.append({
                "heldout_dataset": held,
                "target_mean_budget": float(B),
                "actual_mean_budget": float(m.mean()),
                "recall@1": r1,
                "uniform_recall@1": uniform,
                "delta_vs_uniform_pp": (r1 - uniform) * 100.0,
                "fraction_queries_matched": float((m > 0).mean()),
            })

    pd.DataFrame(allocator_rows).to_csv(OUT / "q1_safe_allocator_lodo.csv", index=False)

    # Two-stage anytime allocator: pay for top-1, then selectively expand.
    anytime_rows = []
    stage_levels = levels[levels >= 1]

    for held in datasets:
        train = [d for d in datasets if d != held]
        Xtr = np.vstack([data[d]["X_stage1"] for d in train])
        # Select only stage-compatible budget columns.
        idx = [int(np.where(levels == m)[0][0]) for m in stage_levels]
        Ytr = np.vstack([data[d]["Y"][:, idx] for d in train])
        Xte = data[held]["X_stage1"]
        Yte = data[held]["Y"][:, idx]

        models = fit_value_models(Xtr, Ytr, stage_levels, base_budget=1)
        pred = predict_value(models, Xte, stage_levels, base_budget=1)

        for B in [b for b in target_budgets if b >= 1]:
            m = enforce_mean_budget(pred, stage_levels, float(B))
            r1 = evaluate_budgets(m, Yte, stage_levels)
            j = int(np.where(stage_levels == int(B))[0][0])
            uniform = float(Yte[:, j].mean())
            anytime_rows.append({
                "heldout_dataset": held,
                "target_mean_budget": float(B),
                "actual_mean_budget": float(m.mean()),
                "recall@1": r1,
                "uniform_recall@1": uniform,
                "delta_vs_uniform_pp": (r1 - uniform) * 100.0,
                "fraction_queries_expanded_beyond_top1": float((m > 1).mean()),
            })

    pd.DataFrame(anytime_rows).to_csv(OUT / "q1_safe_anytime_stage1_lodo.csv", index=False)

    # Compact audit summary.
    uni = pd.DataFrame(uniform_rows)
    wide = uni.pivot(index="dataset", columns="budget", values="recall@1")
    effect_m5 = {}
    if 5 in wide.columns and 100 in wide.columns:
        effect_m5 = ((wide[5] - wide[100]) * 100).to_dict()

    summary = {
        "budget_simulation": "MATCHED_CANDIDATES_ONLY",
        "allocator_features": "RETRIEVAL_ONLY",
        "anytime_stage1_features": "RETRIEVAL_PLUS_EXACTLY_ONE_PAID_TOP1_MATCH",
        "leave_one_dataset_out": True,
        "datasets": datasets,
        "budget_levels": levels.tolist(),
        "m5_minus_m100_effect_pp": effect_m5,
        "important_note": (
            "These analyses are appropriate for method development. Freeze the "
            "method before the final new-backbone confirmatory experiment."
        ),
    }
    (OUT / "q1_safe_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote Q1-safe results to: {OUT}")
    print("No raw matching directory was required.")


if __name__ == "__main__":
    main()
