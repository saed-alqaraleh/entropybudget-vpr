"""Candidate scores, reliability estimates, fusion rules, budget policies.

Every function here is pure and deterministic: given the same arrays it returns
the same output. That is what makes the unit tests meaningful.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max to [0,1] over the candidate axis. A constant vector maps to all
    0.5 (not 0 and not NaN), so ties stay ties."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < eps:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def normalized_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    h = -(p * np.log(p)).sum()
    return float(h / np.log(len(p))) if len(p) > 1 else 0.0


# ----------------------------------------------------------- candidate scores

def global_score(distances: np.ndarray, beta: float) -> np.ndarray:
    """G_i = beta * norm(-d_i) + (1-beta) * rank prior."""
    K = len(distances)
    g_dist = minmax(-np.asarray(distances, dtype=np.float64))
    ranks = np.arange(1, K + 1, dtype=np.float64)
    g_rank = 1.0 - (ranks - 1.0) / max(K - 1, 1)
    return beta * g_dist + (1.0 - beta) * g_rank


def local_score(inlier_count: np.ndarray, match_count: np.ndarray,
                w_n: float, w_rho: float) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(inlier_count, dtype=np.float64)
    m = np.asarray(match_count, dtype=np.float64)
    rho = n / np.maximum(m, 1.0)
    return w_n * minmax(np.log1p(n)) + w_rho * rho, rho


def spatial_coverage(xy: np.ndarray, wh: Sequence[int], grid: Sequence[int]) -> Tuple[float, float]:
    """(normalized bounding-box area, fraction of grid cells occupied).

    Zero inliers -> (0,0). One inlier -> (0, 1/cells): a single point spans no
    area but does occupy one cell, which is the intended behaviour.
    """
    xy = np.asarray(xy, dtype=np.float64)
    w, h = float(wh[0]), float(wh[1])
    gx, gy = int(grid[0]), int(grid[1])
    if xy.shape[0] == 0 or w <= 0 or h <= 0:
        return 0.0, 0.0
    x, y = np.clip(xy[:, 0], 0, w - 1e-9), np.clip(xy[:, 1], 0, h - 1e-9)
    bbox = ((x.max() - x.min()) * (y.max() - y.min())) / (w * h)
    cx = np.clip((x / w * gx).astype(int), 0, gx - 1)
    cy = np.clip((y / h * gy).astype(int), 0, gy - 1)
    cells = len(set(zip(cx.tolist(), cy.tolist())))
    return float(np.clip(bbox, 0.0, 1.0)), float(cells / (gx * gy))


def spatial_distribution_score(bq: float, bc: float, cq: float, cc: float) -> float:
    """s_i = 0.5*sqrt(Bq*Bc) + 0.5*sqrt(Cq*Cc); geometric means penalise evidence
    that is spread in one image but concentrated in the other."""
    return 0.5 * float(np.sqrt(max(bq, 0) * max(bc, 0))) + 0.5 * float(np.sqrt(max(cq, 0) * max(cc, 0)))


def spatial_local_score(L: np.ndarray, s: np.ndarray, gamma: float) -> np.ndarray:
    return np.asarray(L, np.float64) * (gamma + (1.0 - gamma) * np.asarray(s, np.float64))


# --------------------------------------------------------- query reliability

def global_reliability(distances: np.ndarray, T: float, weights: Sequence[float]) -> float:
    d = np.asarray(distances, dtype=np.float64)
    z = -d / max(T, 1e-12)
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    order = np.sort(p)[::-1]
    p1 = float(order[0])
    dp = float(order[0] - order[1]) if len(order) > 1 else float(order[0])
    dsorted = np.sort(d)
    gap = float(dsorted[1] - dsorted[0]) if len(dsorted) > 1 else 0.0
    span = float(dsorted[-1] - dsorted[0]) if len(dsorted) > 1 else 0.0
    dgap = gap / span if span > 1e-12 else 0.0
    a1, a2, a3, a4 = weights
    return float(a1 * p1 + a2 * dp + a3 * dgap + a4 * (1.0 - normalized_entropy(p)))


def local_reliability(SL: np.ndarray, inlier_count: np.ndarray, inlier_ratio: np.ndarray,
                      T: float, weights: Sequence[float]) -> float:
    sl = np.asarray(SL, dtype=np.float64)
    z = sl / max(T, 1e-12)
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    order = np.argsort(sl)[::-1]
    q1 = float(p[order[0]])
    ssorted = np.sort(sl)[::-1]
    span = float(ssorted[0] - ssorted[-1])
    dl = float((ssorted[0] - ssorted[1]) / span) if len(ssorted) > 1 and span > 1e-12 else 0.0
    n = np.asarray(inlier_count, dtype=np.float64)
    il = float(minmax(np.log1p(n))[order[0]])
    rl = float(np.asarray(inlier_ratio, dtype=np.float64)[order[0]])
    b1, b2, b3, b4, b5 = weights
    return float(b1 * q1 + b2 * dl + b3 * il + b4 * rl + b5 * (1.0 - normalized_entropy(p)))


def lambda_q(cg: float, cl: float, lmin: float, lmax: float, eps: float) -> float:
    return float(np.clip(cg / (cg + cl + eps), lmin, lmax))


# ------------------------------------------------------------- rank fusion

def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """1-based competition ranks (descending score). Deterministic ties broken by
    candidate index, so results do not depend on sort implementation."""
    s = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(len(s)), -s))
    r = np.empty(len(s), dtype=np.int64)
    r[order] = np.arange(1, len(s) + 1)
    return r


def rrf_score(rank_lists: Sequence[np.ndarray], k: int) -> np.ndarray:
    out = np.zeros(len(rank_lists[0]), dtype=np.float64)
    for r in rank_lists:
        out += 1.0 / (k + np.asarray(r, dtype=np.float64))
    return out


def topk_support(rank_lists: Sequence[np.ndarray], cutoffs: Sequence[int]) -> np.ndarray:
    out = np.zeros(len(rank_lists[0]), dtype=np.float64)
    for r in rank_lists:
        r = np.asarray(r, dtype=np.float64)
        for ks in cutoffs:
            inside = r <= ks
            out += inside * (1.0 - (r - 1.0) / ks)
    return out


def fuse(adaptive: np.ndarray, rrf: np.ndarray, tk: np.ndarray,
         weights: Sequence[float], normalize: bool) -> np.ndarray:
    """Audit issue B3: raw AS, RRF and TK span ~[0,1], ~[0,0.1] and ~[0,12].
    Combining them at 0.6/0.2/0.2 without normalisation does NOT give the adaptive
    term the dominant role the weights imply. We normalise per query and record
    the choice explicitly."""
    if normalize:
        adaptive, rrf, tk = minmax(adaptive), minmax(rrf), minmax(tk)
    w = np.asarray(weights, dtype=np.float64)
    return w[0] * adaptive + w[1] * rrf + w[2] * tk


# ------------------------------------------------- per-query score assembly

def score_query(distances: np.ndarray, inlier_count: np.ndarray, match_count: np.ndarray,
                spatial: np.ndarray, cfg) -> Dict[str, np.ndarray]:
    """Compute every ranking method's scores for one query. `spatial` is the
    precomputed s_i vector (K,). Returns method-name -> score vector."""
    G = global_score(distances, cfg.beta_global)
    L, rho = local_score(inlier_count, match_count, cfg.w_inlier_count, cfg.w_inlier_ratio)
    SL = spatial_local_score(L, spatial, cfg.gamma_spatial)

    cg = global_reliability(distances, cfg.softmax_temperature, cfg.global_reliability_weights)
    cl = local_reliability(SL, inlier_count, rho, cfg.softmax_temperature,
                           cfg.local_reliability_weights)
    lam = lambda_q(cg, cl, cfg.lambda_min, cfg.lambda_max, cfg.epsilon)

    out: Dict[str, np.ndarray] = {}
    out["global_only"] = G
    out["inlier_count_only"] = np.asarray(inlier_count, np.float64)      # true prior-work baseline
    out["inlier_ratio_only"] = rho
    out["local_composite"] = L                                           # the manuscript's baseline
    out["fixed_global_local"] = 0.7 * G + 0.3 * L
    out["fixed_global_spatial"] = 0.7 * G + 0.3 * SL
    out["adaptive_global_local"] = lam * G + (1 - lam) * L
    out["adaptive_global_spatial"] = lam * G + (1 - lam) * SL

    intermediate = [ranks_from_scores(out[m]) for m in
                    ["global_only", "local_composite", "fixed_global_local",
                     "fixed_global_spatial", "adaptive_global_local",
                     "adaptive_global_spatial"]]
    rrf = rrf_score(intermediate, cfg.rrf_k)
    tk = topk_support(intermediate, cfg.topk_support_cutoffs)
    out["rrf_consensus"] = rrf
    out["frozen_universal_fusion"] = fuse(out["adaptive_global_spatial"], rrf, tk,
                                          cfg.fuf_weights, cfg.normalize_fusion_terms)
    out["_lambda_q"] = np.array([lam])
    out["_global_reliability"] = np.array([cg])
    out["_local_reliability"] = np.array([cl])
    return out


# ------------------------------------------------------------ budget policies

def apply_budget(global_scores: np.ndarray, fused_scores: np.ndarray, m: int) -> np.ndarray:
    """Rank with matching applied to only the top-`m` global candidates.

    The top-m block is reordered by the fused score; positions m..K keep their
    global order and stay strictly below the block. m=0 is pure retrieval,
    m=K is exhaustive re-ranking. This is what makes budget curves computable
    offline from the cached top-K matches.
    """
    K = len(global_scores)
    m = int(np.clip(m, 0, K))
    order_g = np.argsort(-np.asarray(global_scores, np.float64), kind="stable")
    if m == 0:
        return order_g
    head = order_g[:m]
    head = head[np.argsort(-np.asarray(fused_scores, np.float64)[head], kind="stable")]
    return np.concatenate([head, order_g[m:]])


def allocate_budget(risk: np.ndarray, mean_budget: float, levels: Sequence[int]) -> np.ndarray:
    """Greedy allocation: spend the budget on the riskiest queries first.

    Returns m_q per query with mean(m_q) <= mean_budget. Queries are sorted by
    descending risk and given the largest affordable level until the pot runs out.
    """
    risk = np.asarray(risk, dtype=np.float64)
    n = len(risk)
    levels = sorted(set(int(x) for x in levels))
    pot = float(mean_budget) * n
    m = np.zeros(n, dtype=np.int64)
    order = np.argsort(-risk, kind="stable")
    for i in order:
        afford = [lv for lv in levels if lv <= pot]
        if not afford:
            break
        lv = max(afford)
        m[i] = lv
        pot -= lv
    return m
