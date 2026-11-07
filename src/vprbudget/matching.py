"""Local matching + geometric verification, with a resumable shard cache.

Cache contract (the single most valuable artifact this pipeline produces):
for every (query, candidate) pair we persist inlier COUNT, match COUNT, and the
inlier KEYPOINT COORDINATES in both images. The coordinates are what make
contribution C1 (spatial inlier evidence) and C2 (offline budget simulation)
possible without re-running any matching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io_utils import Heartbeat, atomic_write_json, atomic_write_npz, read_json


def shard_path(stage_dir: Path, dataset: str, shard_id: int) -> Path:
    return stage_dir / dataset / f"shard_{shard_id:05d}.npz"


def shard_marker(stage_dir: Path, dataset: str, shard_id: int) -> Path:
    return stage_dir / dataset / f"shard_{shard_id:05d}.done.json"


def plan_shards(n_queries: int, shard_size: int) -> List[Tuple[int, int, int]]:
    """-> [(shard_id, start, stop)] over query positions."""
    return [(i, s, min(s + shard_size, n_queries))
            for i, s in enumerate(range(0, n_queries, shard_size))]


def completed_shards(stage_dir: Path, dataset: str, shards) -> set:
    """A shard counts as done only if BOTH its marker and its data file exist."""
    done = set()
    for sid, _, _ in shards:
        if shard_marker(stage_dir, dataset, sid).exists() and shard_path(stage_dir, dataset, sid).exists():
            done.add(sid)
    return done


# ------------------------------------------------------------------ geometry

# BLOCKER B-6. The run01 hotfix wrapped cv2 in a bare except and returned zero
# inliers, so an unknown fraction of pairs silently produced no evidence. These
# counters make that rate reportable; the pipeline writes them per dataset.
GV_STATS = {"attempted": 0, "no_model": 0, "cv2_error": 0, "degenerate": 0}


def reset_gv_stats() -> None:
    for k in GV_STATS:
        GV_STATS[k] = 0


def geometric_verify(kpts0: np.ndarray, kpts1: np.ndarray, model: str,
                     thresh: float, conf: float, max_iters: int,
                     min_matches: int) -> np.ndarray:
    """Return a boolean inlier mask over the putative correspondences.

    Below `min_matches` the estimate is not identifiable, so we return all-False
    rather than an arbitrary model. This is the documented zero-match policy
    (audit issue: 'handling of zero matches' was unspecified in the manuscript).
    """
    import cv2
    n = len(kpts0)
    if n < min_matches:
        return np.zeros(n, dtype=bool)
    GV_STATS["attempted"] += 1
    p0 = np.ascontiguousarray(kpts0, dtype=np.float64)
    p1 = np.ascontiguousarray(kpts1, dtype=np.float64)
    if model not in ("homography", "fundamental"):
        raise ValueError(f"unknown geometric_model {model!r}")
    try:
        if model == "homography":
            M, mask = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, thresh,
                                         maxIters=max_iters, confidence=conf)
        else:
            M, mask = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC, thresh,
                                             conf, max_iters)
    except cv2.error:
        # Degenerate correspondence sets (collinear or coincident points) make
        # the USAC solver raise. Counted, not hidden.
        GV_STATS["cv2_error"] += 1
        return np.zeros(n, dtype=bool)
    if M is None or mask is None:
        GV_STATS["no_model"] += 1
        return np.zeros(n, dtype=bool)
    return mask.ravel().astype(bool)


def load_image_tensor(path: Path, size: int, device) -> "torch.Tensor":
    import torch
    from lightglue.utils import load_image, resize_image  # noqa: F401
    img = load_image(str(path))                      # (3,H,W) float in [0,1]
    import torch.nn.functional as F
    _, h, w = img.shape
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    img = F.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
    return img.to(device), (nw, nh)


@ (lambda f: f)   # no-op decorator; keeps signature obvious for readers
def match_pair(extractor, lg, feats_q, feats_c, device):
    """Run LightGlue on two pre-extracted feature dicts -> (kpts0, kpts1)."""
    import torch
    from lightglue.utils import rbd
    with torch.inference_mode():
        out = lg({"image0": feats_q, "image1": feats_c})
    f0, f1, o = rbd(feats_q), rbd(feats_c), rbd(out)
    m = o["matches"]
    k0 = f0["keypoints"][m[:, 0]].detach().cpu().numpy()
    k1 = f1["keypoints"][m[:, 1]].detach().cpu().numpy()
    return k0, k1


def extract_features(extractor, path: Path, size: int, device):
    import torch
    img, wh = load_image_tensor(path, size, device)
    with torch.inference_mode():
        feats = extractor.extract(img[None])
    return feats, wh


def run_matching_for_dataset(cfg, ds, retrieval, stage_dir: Path, logger) -> Dict[str, int]:
    """Match every (query, top-K candidate) pair, checkpointing per shard.

    Resume semantics: completed shards are detected and skipped. An interrupted
    run therefore loses at most `matching_shard_size` queries of work.
    """
    import torch
    from .models import load_matcher, resolve_device

    device = resolve_device(cfg.device, cfg.require_cuda)
    extractor, lg = load_matcher(cfg.detector, cfg.matcher, cfg.max_keypoints, device)

    q_idx = retrieval["query_indices"]          # (Nq,) into ds.query_paths
    cand = retrieval["candidates"]              # (Nq, K) into ds.database_paths
    n_q, K = cand.shape
    shards = plan_shards(n_q, cfg.matching_shard_size)
    done = completed_shards(stage_dir, ds.name, shards)
    logger.info(f"[{ds.name}] {len(shards)} shards, {len(done)} already complete "
                f"-> {len(shards)-len(done)} to run ({(len(shards)-len(done))*cfg.matching_shard_size*K} pairs)")

    (stage_dir / ds.name).mkdir(parents=True, exist_ok=True)
    hb = Heartbeat(logger, f"matching[{ds.name}]", total=len(shards) - len(done))

    # Database features are reused heavily across queries within a shard.
    for sid, start, stop in shards:
        if sid in done:
            continue
        n_local = stop - start
        inl_count = np.zeros((n_local, K), dtype=np.int32)
        mat_count = np.zeros((n_local, K), dtype=np.int32)
        q_wh = np.zeros((n_local, 2), dtype=np.int32)
        c_wh = np.zeros((n_local, K, 2), dtype=np.int32)
        # Ragged inlier coordinates -> flat arrays + offsets (npz-friendly).
        coords_q: List[np.ndarray] = []
        coords_c: List[np.ndarray] = []
        offsets = np.zeros((n_local, K, 2), dtype=np.int64)   # (start, length)
        cursor = 0

        db_cache: Dict[int, tuple] = {}
        for li in range(n_local):
            gi = start + li
            fq, wq = extract_features(extractor, ds.query_paths[q_idx[gi]],
                                      cfg.matching_image_size, device)
            q_wh[li] = wq
            for k in range(K):
                dbi = int(cand[gi, k])
                if dbi not in db_cache:
                    if len(db_cache) > 2 * K:
                        db_cache.clear()
                    db_cache[dbi] = extract_features(
                        extractor, ds.database_paths[dbi], cfg.matching_image_size, device)
                fc, wc = db_cache[dbi]
                c_wh[li, k] = wc
                k0, k1 = match_pair(extractor, lg, fq, fc, device)
                mat_count[li, k] = len(k0)
                mask = geometric_verify(k0, k1, cfg.geometric_model,
                                        cfg.ransac_reproj_threshold, cfg.ransac_confidence,
                                        cfg.ransac_max_iters, cfg.min_matches_for_gv)
                n_in = int(mask.sum())
                inl_count[li, k] = n_in
                offsets[li, k] = (cursor, n_in)
                if n_in:
                    coords_q.append(k0[mask].astype(np.float32))
                    coords_c.append(k1[mask].astype(np.float32))
                cursor += n_in
            del db_cache
            db_cache = {}
            if device.type == "cuda":
                torch.cuda.empty_cache()

        cq = np.concatenate(coords_q) if coords_q else np.zeros((0, 2), np.float32)
        cc = np.concatenate(coords_c) if coords_c else np.zeros((0, 2), np.float32)
        atomic_write_npz(shard_path(stage_dir, ds.name, sid),
                         inlier_count=inl_count, match_count=mat_count,
                         inlier_xy_query=cq, inlier_xy_cand=cc, offsets=offsets,
                         query_wh=q_wh, cand_wh=c_wh,
                         query_positions=np.arange(start, stop, dtype=np.int64))
        atomic_write_json(shard_marker(stage_dir, ds.name, sid),
                          {"shard": sid, "start": start, "stop": stop, "K": K,
                           "total_inliers": int(cursor)})
        hb.update(1, f"shard {sid}")
    hb.close()
    return {"n_shards": len(shards), "n_queries": n_q, "K": K}


def load_matching(stage_dir: Path, dataset: str, n_queries: int, K: int,
                  shard_size: int) -> Dict[str, np.ndarray]:
    """Reassemble all shards into one in-memory structure."""
    shards = plan_shards(n_queries, shard_size)
    inl = np.zeros((n_queries, K), np.int32)
    mat = np.zeros((n_queries, K), np.int32)
    q_wh = np.zeros((n_queries, 2), np.int32)
    c_wh = np.zeros((n_queries, K, 2), np.int32)
    coords_q: List[np.ndarray] = []
    coords_c: List[np.ndarray] = []
    offs = np.zeros((n_queries, K, 2), np.int64)
    base = 0
    for sid, start, stop in shards:
        p = shard_path(stage_dir, dataset, sid)
        if not p.exists():
            raise FileNotFoundError(f"missing matching shard {p}; rerun stage 'matching'")
        z = np.load(p)
        inl[start:stop] = z["inlier_count"]
        mat[start:stop] = z["match_count"]
        q_wh[start:stop] = z["query_wh"]
        c_wh[start:stop] = z["cand_wh"]
        o = z["offsets"].copy()
        o[..., 0] += base
        offs[start:stop] = o
        coords_q.append(z["inlier_xy_query"])
        coords_c.append(z["inlier_xy_cand"])
        base += z["inlier_xy_query"].shape[0]
    return {"inlier_count": inl, "match_count": mat, "offsets": offs,
            "inlier_xy_query": np.concatenate(coords_q) if coords_q else np.zeros((0, 2), np.float32),
            "inlier_xy_cand": np.concatenate(coords_c) if coords_c else np.zeros((0, 2), np.float32),
            "query_wh": q_wh, "cand_wh": c_wh}
