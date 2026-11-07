#!/usr/bin/env python3
"""
Q1 Confirmatory Multi-Retriever Experiment — Fast Shared-Cache Edition — Fast Implementation
==========================================

Purpose
-------
Run the FROZEN matching-allocation policy on new global VPR retrievers while
holding the query cohort, local matcher, geometric verification, ground truth,
and fusion rule fixed.

Frozen primary policy (selected BEFORE this confirmatory experiment)
--------------------------------------------------------------------
- Retrieval uncertainty: normalized entropy of the top-100 descriptor-distance
  distribution, temperature T=0.10.
- Distribution calibration: within each unlabeled query workload, compute
  uncertainty quintiles. No ground-truth label is used.
- Allocation: bottom 60% uncertainty -> m=0; top 40% -> m=50.
- Mean candidate-verification budget: approximately 20/query.
- Local evidence: SuperPoint + LightGlue + MAGSAC, exactly as in run02.
- Budget scoring: MATCHED CANDIDATES ONLY (Q1 budget-safe scorer).
- Primary comparisons:
    (a) uniform m=20,
    (b) same-cost top1/top2-margin scheduler (top 40% -> m=50),
    (c) retrieval-only m=0.
- Secondary comparison: uniform m=50.

New retrievers
--------------
Default: SALAD (CVPR 2024) and BoQ-DINOv2 (CVPR 2024), using their official
PyTorch Hub entry points.

Efficiency
----------
The script does NOT read the 6 GB raw run02 matching cache. It first builds a
compact pair-evidence dictionary from run02/retrieval + run02/scoring. Any
query-candidate pair already verified under MegaLoc is reused exactly.
Only new pairs introduced by SALAD/BoQ are matched. Missing-pair evidence is
saved in resumable shards.

This is a CONFIRMATORY script. Do not tune T, the 60/40 threshold, m=50,
fusion weights, or matching settings after inspecting these results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Frozen constants: DO NOT TUNE AFTER CONFIRMATORY RESULTS
# ---------------------------------------------------------------------------

FROZEN_ENTROPY_T = 0.10
FROZEN_HIGH_UNCERTAINTY_FRACTION = 0.40
FROZEN_MATCH_BUDGET = 50
PRIMARY_UNIFORM_BUDGET = 20
RETRIEVAL_TOP_K = 100
CONFIRMATORY_VERSION = "Q1_EXTERNAL_CONFIRMATORY_V3_RESTARTSAFE"


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def stable_top_fraction(x: np.ndarray, fraction: float, largest: bool) -> np.ndarray:
    """Boolean mask selecting an exact top fraction by stable rank.

    This is label-free workload calibration. For n not divisible by 5, the
    selected count is round(fraction*n), bounded to [0,n].
    """
    x = np.asarray(x, float)
    n = len(x)
    k = int(round(float(fraction) * n))
    k = min(max(k, 0), n)
    order = np.argsort(x, kind="stable")
    if largest:
        order = order[::-1]
    mask = np.zeros(n, dtype=bool)
    if k:
        mask[order[:k]] = True
    return mask


def retrieval_entropy(dist: np.ndarray, temperature: float = FROZEN_ENTROPY_T) -> np.ndarray:
    d = np.asarray(dist, dtype=float)
    lo = d.min(axis=1, keepdims=True)
    hi = d.max(axis=1, keepdims=True)
    span = hi - lo
    dn = np.divide(d - lo, span, out=np.zeros_like(d), where=span > 1e-12)
    dn = np.where(span <= 1e-12, 0.5, dn)
    z = -dn / float(temperature)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    if d.shape[1] <= 1:
        return np.zeros(len(d), dtype=float)
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1) / np.log(d.shape[1])


def normalized_top12_margin(dist: np.ndarray) -> np.ndarray:
    """Larger = more confident; small margin receives matching in the baseline."""
    d = np.asarray(dist, float)
    lo = d.min(axis=1, keepdims=True)
    hi = d.max(axis=1, keepdims=True)
    span = np.maximum(hi - lo, 1e-12)
    dn = (d - lo) / span
    s = np.sort(dn, axis=1)
    return s[:, 1] - s[:, 0]


def paired_bootstrap_pp(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired arrays differ in length")
    d = a - b
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    step = 250
    n = len(d)
    for s in range(0, n_boot, step):
        e = min(s + step, n_boot)
        idx = rng.integers(0, n, size=(e - s, n))
        vals[s:e] = d[idx].mean(axis=1) * 100.0
    return {
        "difference_pp": float(d.mean() * 100.0),
        "ci_low_pp": float(np.quantile(vals, 0.025)),
        "ci_high_pp": float(np.quantile(vals, 0.975)),
    }


# ---------------------------------------------------------------------------
# Official retrievers
# ---------------------------------------------------------------------------

class DescriptorWrapper:
    """Normalize heterogeneous official model outputs to a descriptor tensor."""
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        import torch
        out = self.model(x)

        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            for key in ("global_descriptor", "descriptor", "descriptors",
                        "global_desc", "embedding", "embeddings", "feat"):
                if key in out:
                    out = out[key]
                    break
            else:
                tensors = [v for v in out.values() if torch.is_tensor(v)]
                if len(tensors) != 1:
                    raise RuntimeError(
                        f"Cannot identify descriptor tensor from model dict keys {list(out)}"
                    )
                out = tensors[0]

        if not torch.is_tensor(out):
            raise TypeError(f"Retriever output is {type(out)}, expected tensor")
        if out.ndim > 2:
            out = out.flatten(1)
        return torch.nn.functional.normalize(out.float(), p=2, dim=1)


def load_official_retriever(name: str, device):
    import torch

    key = name.lower().replace("_", "-")
    if key == "salad":
        model = torch.hub.load(
            "serizba/salad", "dinov2_salad", trust_repo=True
        )
        source = {
            "repository": "serizba/salad",
            "entry_point": "dinov2_salad",
            "paper": "Optimal Transport Aggregation for Visual Place Recognition (CVPR 2024)",
        }
    elif key in ("boq", "boq-dinov2", "boqdinov2"):
        model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="dinov2",
            output_dim=12288,
            trust_repo=True,
        )
        source = {
            "repository": "amaralibey/bag-of-queries",
            "entry_point": "get_trained_boq(backbone_name='dinov2', output_dim=12288)",
            "paper": "BoQ: A Place is Worth a Bag of Learnable Queries (CVPR 2024)",
        }
    else:
        raise ValueError(
            f"Unsupported confirmatory retriever {name!r}; use salad and/or boq-dinov2."
        )

    model = model.eval().to(device)
    return DescriptorWrapper(model), source


def official_input_transform():
    """322x322 ImageNet-normalized input; bicubic for BoQ-DINOv2 compatibility."""
    import torchvision.transforms as T

    return T.Compose([
        T.ToTensor(),
        T.Resize(
            (322, 322),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def gallery_signature(paths: Sequence[Path]) -> str:
    """Fingerprint an exact ordered gallery by resolved image path.

    Two conditions reuse descriptors only when every gallery path is identical
    and in the same order. This changes no scientific computation.
    """
    h = hashlib.sha256()
    h.update(f"n={len(paths)}\n".encode())
    for p in paths:
        h.update(str(Path(p).expanduser().resolve()).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def extract_descriptor_shards(
    model,
    paths: Sequence[Path],
    out_dir: Path,
    split: str,
    device,
    batch_size: int = 16,
    shard_images: int = 1000,
    loader_workers: int = 4,
    concatenate: bool = True,
):
    """Extract deterministic descriptor shards with parallel CPU image loading.

    `concatenate=False` avoids materializing huge galleries (e.g. San Francisco)
    as one multi-GB NumPy array. In that case this returns the ordered shard paths
    and exact L2 retrieval is performed shard-by-shard.
    """
    import torch
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    tf = official_input_transform()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(paths)
    arrays = []
    shard_paths: List[Path] = []

    def load_one(pth: Path):
        with Image.open(pth) as im:
            return tf(im.convert("RGB"))

    pool = ThreadPoolExecutor(max_workers=max(1, int(loader_workers)))
    try:
        for sid, start in enumerate(range(0, n, shard_images)):
            stop = min(start + shard_images, n)
            p = out_dir / f"{split}_{sid:05d}.npz"
            shard_paths.append(p)
            if p.exists():
                z = np.load(p)
                arr = z["descriptors"]
                if len(arr) != stop - start:
                    raise RuntimeError(f"Bad cached descriptor shard: {p}")
                if concatenate:
                    arrays.append(arr)
                print(f"    {split} shard {sid}: cached", flush=True)
                continue

            chunks = []
            for b in range(start, stop, batch_size):
                batch_paths = paths[b:min(b + batch_size, stop)]
                # Executor.map preserves input order exactly.
                imgs_cpu = list(pool.map(load_one, batch_paths))
                imgs = torch.stack(imgs_cpu).to(device, non_blocking=True)
                with torch.inference_mode():
                    desc = model(imgs)
                chunks.append(desc.detach().cpu().numpy().astype(np.float32))
                del imgs, imgs_cpu, desc

            arr = np.concatenate(chunks, axis=0)
            atomic_npz(p, descriptors=arr)
            if concatenate:
                arrays.append(arr)
            print(f"    {split} shard {sid}: wrote {len(arr)}", flush=True)
    finally:
        pool.shutdown(wait=True)

    if not concatenate:
        return shard_paths

    out = np.concatenate(arrays, axis=0).astype(np.float32)
    if len(out) != n or not np.isfinite(out).all():
        raise RuntimeError(f"Descriptor extraction failed for {split}")
    return out


def exact_topk_search_shards(
    db_shards: Sequence[Path],
    q_desc: np.ndarray,
    k: int,
):
    """Exact L2 top-k over descriptor shards with bounded memory.

    This is mathematically the same exhaustive L2 retrieval as IndexFlatL2;
    only the gallery is searched in blocks and the blockwise exact top-k results
    are merged. It prevents the San Francisco gallery from requiring a 5-8 GB
    contiguous descriptor matrix.
    """
    q = np.ascontiguousarray(q_desc.astype(np.float32, copy=False))
    nq = len(q)
    best_d2 = np.empty((nq, 0), dtype=np.float32)
    best_idx = np.empty((nq, 0), dtype=np.int64)
    offset = 0

    try:
        import faiss
        have_faiss = True
    except ImportError:
        have_faiss = False

    for si, p in enumerate(db_shards):
        db = np.load(p)["descriptors"].astype(np.float32, copy=False)
        kk = min(int(k), len(db))
        if kk <= 0:
            continue

        if have_faiss:
            index = faiss.IndexFlatL2(db.shape[1])
            index.add(np.ascontiguousarray(db))
            d2, idx = index.search(q, kk)
        else:
            # Exact squared-L2 without a (query,db,dim) tensor.
            qn = np.sum(q * q, axis=1, keepdims=True)
            dn = np.sum(db * db, axis=1)[None, :]
            d2full = np.maximum(qn + dn - 2.0 * (q @ db.T), 0.0)
            idx = np.argpartition(d2full, kk - 1, axis=1)[:, :kk]
            d2 = np.take_along_axis(d2full, idx, axis=1).astype(np.float32)

        idx = idx.astype(np.int64) + offset
        cand_d = np.concatenate([best_d2, d2.astype(np.float32)], axis=1)
        cand_i = np.concatenate([best_idx, idx], axis=1)
        keep = min(int(k), cand_d.shape[1])
        sel = np.argpartition(cand_d, keep - 1, axis=1)[:, :keep]
        best_d2 = np.take_along_axis(cand_d, sel, axis=1)
        best_idx = np.take_along_axis(cand_i, sel, axis=1)
        order = np.argsort(best_d2, axis=1, kind="stable")
        best_d2 = np.take_along_axis(best_d2, order, axis=1)
        best_idx = np.take_along_axis(best_idx, order, axis=1)
        offset += len(db)
        print(f"    exact retrieval shard {si}: searched {len(db)} gallery images", flush=True)

    if best_d2.shape[1] != int(k):
        raise RuntimeError(f"Only {best_d2.shape[1]} retrieval candidates available, expected {k}")
    return np.sqrt(np.maximum(best_d2, 0.0)).astype(np.float32), best_idx.astype(np.int64)


# ---------------------------------------------------------------------------
# Compact baseline pair-evidence reuse
# ---------------------------------------------------------------------------

def baseline_pair_evidence(
    baseline_run: Path,
    dataset: str,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, int, float]]]:
    ret = np.load(baseline_run / "retrieval" / f"{dataset}.npz")
    sc = np.load(baseline_run / "scoring" / f"{dataset}.npz")

    qidx = ret["query_indices"].astype(np.int64)
    cand = ret["candidates"].astype(np.int64)
    inl = sc["inlier_count"].astype(np.int32)
    mat = sc["match_count"].astype(np.int32)
    spatial = sc["spatial"].astype(np.float32)

    if cand.shape != inl.shape or cand.shape != mat.shape or cand.shape != spatial.shape:
        raise RuntimeError(f"{dataset}: baseline retrieval/scoring shape mismatch")

    cache: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
    for i, q in enumerate(qidx):
        for k, c in enumerate(cand[i]):
            cache[(int(q), int(c))] = (
                int(inl[i, k]), int(mat[i, k]), float(spatial[i, k])
            )
    return qidx, cache


def augment_reuse_from_prior_retrievers(
    out_root: Path,
    dataset: str,
    reuse: Dict[Tuple[int, int], Tuple[int, int, float]],
    current_retriever: str,
    local_top_k: int,
) -> int:
    """Reuse pair-level local evidence already computed by another retriever.

    SuperPoint/LightGlue/MAGSAC evidence is a deterministic property of the image
    pair under the frozen local pipeline, so recomputing an identical pair for
    BoQ after SALAD is unnecessary. This also works after a resumed session.
    """
    added = 0
    if not out_root.exists():
        return 0
    for rdir in sorted(out_root.iterdir()):
        if not rdir.is_dir() or rdir.name == current_retriever:
            continue
        ret_path = rdir / "retrieval" / f"{dataset}.npz"
        cdir = rdir / "compact_matching" / dataset
        if not ret_path.exists() or not cdir.is_dir():
            continue
        ret = np.load(ret_path)
        qidx = ret["query_indices"].astype(np.int64)
        cand = ret["candidates"].astype(np.int64)
        for marker in sorted(cdir.glob("compact_*.done.json")):
            try:
                md = json.loads(marker.read_text())
                start, stop = int(md["start"]), int(md["stop"])
                data_path = marker.with_name(marker.name.replace(".done.json", ".npz"))
                if not data_path.exists():
                    continue
                z = np.load(data_path)
                inl = z["inlier_count"]
                mat = z["match_count"]
                sp = z["spatial"]
                for li, gi in enumerate(range(start, stop)):
                    q = int(qidx[gi])
                    for k in range(min(int(local_top_k), cand.shape[1])):
                        key = (q, int(cand[gi, k]))
                        if key not in reuse:
                            reuse[key] = (
                                int(inl[li, k]), int(mat[li, k]), float(sp[li, k])
                            )
                            added += 1
            except Exception as exc:
                print(f"    warning: could not reuse {marker}: {type(exc).__name__}: {exc}")
    return added



def load_matcher_flexible(detector: str, matcher: str, max_keypoints: int, device):
    """Frozen local-matcher loader for Q1 external validation.

    Supports the manuscript backend (SuperPoint+LightGlue) and the pre-frozen
    secondary robustness backend (ALIKED+LightGlue). No other pairing is accepted.
    """
    if matcher.lower() != "lightglue":
        raise ValueError(f"Unsupported matcher {matcher!r}; expected 'lightglue'.")

    try:
        from lightglue import LightGlue, SuperPoint, ALIKED
    except ImportError as exc:
        raise RuntimeError(
            "Official cvg/LightGlue package is required. "
            "Install from https://github.com/cvg/LightGlue"
        ) from exc

    key = detector.lower()
    if key == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        lg = LightGlue(features="superpoint").eval().to(device)
    elif key == "aliked":
        extractor = ALIKED(max_num_keypoints=max_keypoints).eval().to(device)
        lg = LightGlue(features="aliked").eval().to(device)
    else:
        raise ValueError(
            f"Unsupported detector {detector!r}; expected 'superpoint' or 'aliked'."
        )
    return extractor, lg


# ---------------------------------------------------------------------------
# Missing-pair local verification
# ---------------------------------------------------------------------------

def compact_match_shards(
    cfg,
    ds,
    qidx: np.ndarray,
    candidates: np.ndarray,
    reuse: Dict[Tuple[int, int], Tuple[int, int, float]],
    out_dir: Path,
    local_top_k: int,
    shard_queries: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return K-wide arrays; evidence beyond local_top_k remains zero.

    Each shard is independently resumable.
    """
    import torch
    from vprbudget.models import resolve_device
    from vprbudget.matching import (
        extract_features,
        match_pair,
        geometric_verify,
        reset_gv_stats,
        GV_STATS,
    )
    from vprbudget.scoring import (
        spatial_coverage,
        spatial_distribution_score,
    )

    device = resolve_device(cfg.device, cfg.require_cuda)
    extractor, lg = load_matcher_flexible(
        cfg.detector, cfg.matcher, cfg.max_keypoints, device
    )

    # FAST IMPLEMENTATION, SCIENTIFICALLY IDENTICAL:
    # Cache SuperPoint features on local Colab disk and keep a small CPU LRU.
    # Local features depend only on the image + frozen detector settings, not on
    # the global retriever, so SALAD and BoQ can safely reuse them.  This avoids
    # repeatedly running SuperPoint for database images that recur across queries.
    from collections import OrderedDict

    cache_tag = f"{cfg.detector}_kp{cfg.max_keypoints}_sz{cfg.matching_image_size}"
    feature_cache_root = Path(os.environ.get("Q1_LOCAL_FEATURE_CACHE", "/content/q1_local_feature_cache"))
    feature_cache_dir = feature_cache_root / cache_tag
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    cpu_lru = OrderedDict()
    cpu_lru_max = 128
    path_key_cache = {}

    def _to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_to_cpu(v) for v in obj)
        return obj

    def _to_device(obj):
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_to_device(v) for v in obj)
        return obj

    def _lru_put(key, value):
        if key in cpu_lru:
            cpu_lru.pop(key)
        cpu_lru[key] = value
        while len(cpu_lru) > cpu_lru_max:
            cpu_lru.popitem(last=False)

    def cached_extract(kind: str, image_id: int, path: Path):
        # Feature identity is the resolved physical image path, not dataset role.
        # Thus an identical shared SVOX gallery image is extracted once and can
        # be reused in Night/Rain and across SALAD/BoQ.
        pstr = str(Path(path).expanduser().resolve())
        key = path_key_cache.get(pstr)
        if key is None:
            key = hashlib.sha1(pstr.encode("utf-8")).hexdigest()
            path_key_cache[pstr] = key
        if key in cpu_lru:
            feats_cpu, wh = cpu_lru.pop(key)
            cpu_lru[key] = (feats_cpu, wh)
            return _to_device(feats_cpu), wh

        sub = feature_cache_dir / key[:2]
        sub.mkdir(parents=True, exist_ok=True)
        cp = sub / f"{key}.pt"
        if cp.exists():
            try:
                obj = torch.load(cp, map_location="cpu", weights_only=True)
            except TypeError:
                obj = torch.load(cp, map_location="cpu")
            feats_cpu = obj["feats"]
            wh = tuple(int(x) for x in obj["wh"])
            _lru_put(key, (feats_cpu, wh))
            return _to_device(feats_cpu), wh

        feats_gpu, wh = extract_features(
            extractor, path, cfg.matching_image_size, device
        )
        feats_cpu = _to_cpu(feats_gpu)
        tmp = cp.with_suffix(cp.suffix + ".tmp")
        torch.save({"feats": feats_cpu, "wh": tuple(wh)}, tmp)
        os.replace(tmp, cp)
        _lru_put(key, (feats_cpu, tuple(wh)))
        return feats_gpu, tuple(wh)

    nq, K = candidates.shape
    if local_top_k > K:
        raise ValueError("local_top_k > retrieval K")

    out_dir.mkdir(parents=True, exist_ok=True)
    reset_gv_stats()
    total_reused = 0
    total_new = 0

    for sid, start in enumerate(range(0, nq, shard_queries)):
        stop = min(start + shard_queries, nq)
        p = out_dir / f"compact_{sid:05d}.npz"
        marker = out_dir / f"compact_{sid:05d}.done.json"
        if p.exists() and marker.exists():
            print(f"    matching shard {sid}: cached")
            continue

        nloc = stop - start
        inl = np.zeros((nloc, K), dtype=np.int32)
        mat = np.zeros((nloc, K), dtype=np.int32)
        spatial = np.zeros((nloc, K), dtype=np.float32)
        reused = 0
        newly = 0

        for li, gi in enumerate(range(start, stop)):
            qid = int(qidx[gi])
            fq = None
            wq = None

            for k in range(local_top_k):
                cid = int(candidates[gi, k])
                key = (qid, cid)

                if key in reuse:
                    ni, mi, si = reuse[key]
                    inl[li, k] = ni
                    mat[li, k] = mi
                    spatial[li, k] = si
                    reused += 1
                    continue

                if fq is None:
                    fq, wq = cached_extract(
                        "q", qid, ds.query_paths[qid]
                    )

                fc, wc = cached_extract(
                    "db", cid, ds.database_paths[cid]
                )
                k0, k1 = match_pair(extractor, lg, fq, fc, device)
                mat[li, k] = len(k0)

                mask = geometric_verify(
                    k0, k1,
                    cfg.geometric_model,
                    cfg.ransac_reproj_threshold,
                    cfg.ransac_confidence,
                    cfg.ransac_max_iters,
                    cfg.min_matches_for_gv,
                )
                ni = int(mask.sum())
                inl[li, k] = ni

                if ni:
                    bq, cq = spatial_coverage(
                        k0[mask], wq, cfg.spatial_grid
                    )
                    bc, cc = spatial_coverage(
                        k1[mask], wc, cfg.spatial_grid
                    )
                    spatial[li, k] = spatial_distribution_score(
                        bq, bc, cq, cc
                    )
                newly += 1
                del fc

            if fq is not None:
                del fq
            # Do NOT call torch.cuda.empty_cache() after every query: that forces
            # allocator synchronization and can substantially slow LightGlue.
            # The feature LRU is CPU-only, so GPU live memory remains bounded.

        if device.type == "cuda":
            torch.cuda.empty_cache()
        atomic_npz(p, inlier_count=inl, match_count=mat, spatial=spatial)
        atomic_json(marker, {
            "shard": sid,
            "start": start,
            "stop": stop,
            "retrieval_K": int(K),
            "local_top_k": int(local_top_k),
            "pairs_reused_from_megaloc": int(reused),
            "pairs_newly_verified": int(newly),
        })
        total_reused += reused
        total_new += newly
        print(
            f"    matching shard {sid}: wrote; reused={reused}, new={newly}"
        )

    # Reassemble
    all_inl, all_mat, all_sp = [], [], []
    reused_sum = 0
    new_sum = 0
    for sid, start in enumerate(range(0, nq, shard_queries)):
        p = out_dir / f"compact_{sid:05d}.npz"
        marker = out_dir / f"compact_{sid:05d}.done.json"
        if not p.exists() or not marker.exists():
            raise RuntimeError(f"Missing compact matching shard {sid}")
        z = np.load(p)
        all_inl.append(z["inlier_count"])
        all_mat.append(z["match_count"])
        all_sp.append(z["spatial"])
        md = json.loads(marker.read_text())
        reused_sum += int(md["pairs_reused_from_megaloc"])
        new_sum += int(md["pairs_newly_verified"])

    return (
        np.concatenate(all_inl),
        np.concatenate(all_mat),
        np.concatenate(all_sp),
        {
            "pairs_reused_from_megaloc": reused_sum,
            "pairs_newly_verified": new_sum,
            "gv_stats_new_pairs": dict(GV_STATS),
        },
    )


# ---------------------------------------------------------------------------
# Frozen budget-safe evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    cfg,
    dataset: str,
    retrieval_npz: Path,
    inl: np.ndarray,
    mat: np.ndarray,
    spatial: np.ndarray,
    safe,
):
    from vprbudget.groundtruth import get_spec, positives
    from vprbudget.scoring import global_score, global_reliability

    ret = np.load(retrieval_npz)
    dist = ret["descriptor_distances"].astype(float)
    geo = ret["geo_distances"]
    N, K = dist.shape

    if K != RETRIEVAL_TOP_K:
        raise RuntimeError(f"Expected retrieval top-{RETRIEVAL_TOP_K}, got {K}")

    G = np.stack([
        global_score(dist[i], cfg.beta_global) for i in range(N)
    ]).astype(float)
    cg = np.asarray([
        global_reliability(
            dist[i],
            cfg.softmax_temperature,
            cfg.global_reliability_weights,
        )
        for i in range(N)
    ])

    pos = positives(
        geo,
        get_spec(dataset, cfg.dataset_gt_overrides)
    )

    # Frozen entropy policy: top 40% uncertainty -> m=50.
    ent = retrieval_entropy(dist, FROZEN_ENTROPY_T)
    high_u = stable_top_fraction(
        ent, FROZEN_HIGH_UNCERTAINTY_FRACTION, largest=True
    )
    policy_budget = np.where(high_u, FROZEN_MATCH_BUDGET, 0).astype(int)

    # Same-cost margin baseline: smallest top1/top2 margins receive m=50.
    margin = normalized_top12_margin(dist)
    high_margin_risk = stable_top_fraction(
        margin, FROZEN_HIGH_UNCERTAINTY_FRACTION, largest=False
    )
    margin_budget = np.where(
        high_margin_risk, FROZEN_MATCH_BUDGET, 0
    ).astype(int)

    # Precompute safe top-1 at budgets actually needed.
    needed = [0, PRIMARY_UNIFORM_BUDGET, FROZEN_MATCH_BUDGET]
    top_by_m = {}
    correct_by_m = {}
    for m in needed:
        top = safe.budget_safe_top1(
            G, inl, mat, spatial, cg, m, cfg
        )
        top_by_m[m] = top
        correct_by_m[m] = pos[np.arange(N), top].astype(np.int8)

    y_policy = np.where(
        policy_budget == 0,
        correct_by_m[0],
        correct_by_m[FROZEN_MATCH_BUDGET],
    ).astype(np.int8)

    y_margin = np.where(
        margin_budget == 0,
        correct_by_m[0],
        correct_by_m[FROZEN_MATCH_BUDGET],
    ).astype(np.int8)

    perq = pd.DataFrame({
        "dataset": dataset,
        "query_position": np.arange(N),
        "query_index": ret["query_indices"],
        "entropy_uncertainty": ent,
        "top12_margin": margin,
        "frozen_policy_budget": policy_budget,
        "margin_policy_budget": margin_budget,
        "retrieval_correct": correct_by_m[0],
        "uniform_m20_correct": correct_by_m[PRIMARY_UNIFORM_BUDGET],
        "uniform_m50_correct": correct_by_m[FROZEN_MATCH_BUDGET],
        "frozen_policy_correct": y_policy,
        "margin_policy_correct": y_margin,
    })

    return perq


# ---------------------------------------------------------------------------
# Restart-safe San Francisco local alias
# ---------------------------------------------------------------------------

def ensure_sf_local_alias(workspace_root: Path, sf_data_root: Path) -> Path:
    """Rebuild the certified local SF alias after a Colab runtime restart."""
    import json
    import os
    import shutil

    target_dataset = sf_data_root / "san_francisco_partial_embedded_gps" / "images" / "test"
    local_db = target_dataset / "database"
    local_q = target_dataset / "queries"

    try:
        if local_db.is_dir() and local_q.is_dir():
            n_db = sum(1 for _ in local_db.iterdir())
            n_q = sum(1 for _ in local_q.iterdir())
            if n_db == 154_890 and n_q == 233:
                print("    SF local alias: existing certified 154890/233 alias reused", flush=True)
                return sf_data_root
    except OSError:
        pass

    sf = workspace_root / "data" / "san_francisco"
    source_db = sf / "images" / "test" / "database"
    source_q = sf / "images" / "test" / "queries_embedded_gps"

    manifest_root = sf / "_partial_gallery_api_local_v11_3"
    db_manifest = manifest_root / "database_manifest.json"
    q_manifest = manifest_root / "query_manifest.json"

    repair_root = sf / "_strict_image_repair_v11_7"
    strict_state = repair_root / "strict_decode_state.json"
    coverage_state = repair_root / "coverage_state.json"

    required = [db_manifest, q_manifest, strict_state, coverage_state]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(
            "Cannot auto-restore SF local alias; missing persistent artifact(s):\n"
            + "\n".join(missing)
        )
    if not source_db.is_dir():
        raise RuntimeError(f"Missing persistent SF source database: {source_db}")
    if not source_q.is_dir():
        raise RuntimeError(f"Missing persistent SF source queries: {source_q}")

    dbm = json.loads(db_manifest.read_text())
    qm = json.loads(q_manifest.read_text())
    strict = json.loads(strict_state.read_text())
    coverage = json.loads(coverage_state.read_text())

    if isinstance(dbm, dict):
        for key in ("files", "items", "records", "images"):
            if key in dbm:
                dbm = dbm[key]
                break
    if isinstance(qm, dict):
        for key in ("files", "items", "records", "images"):
            if key in qm:
                qm = qm[key]
                break

    excluded_db = set(strict.get("excluded_database_names", []))
    excluded_q = set(strict.get("excluded_query_names", []))
    selected_q = list(coverage.get("selected_query_names", []))

    db_names = [item["name"] for item in dbm if item["name"] not in excluded_db]
    q_manifest_names = {
        item["name"] for item in qm if item["name"] not in excluded_q
    }

    missing_selected = [name for name in selected_q if name not in q_manifest_names]
    if missing_selected:
        raise RuntimeError(
            f"SF alias restoration failed: {len(missing_selected)} selected queries "
            "are absent from the frozen query manifest."
        )

    if len(db_names) != 154_890:
        raise RuntimeError(
            f"SF alias restoration failed: expected 154890 DB names, got {len(db_names)}"
        )
    if len(selected_q) != 233:
        raise RuntimeError(
            f"SF alias restoration failed: expected 233 selected queries, got {len(selected_q)}"
        )

    print(
        "    SF local alias missing after runtime restart; rebuilding from frozen manifests...",
        flush=True,
    )

    for root in (local_db, local_q):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    # Do not enumerate the giant mounted Drive directory.
    for name in db_names:
        os.symlink(source_db / name, local_db / name)
    for name in selected_q:
        os.symlink(source_q / name, local_q / name)

    n_db = sum(1 for _ in local_db.iterdir())
    n_q = sum(1 for _ in local_q.iterdir())
    if n_db != 154_890 or n_q != 233:
        raise RuntimeError(
            f"SF alias integrity failure after rebuild: DB={n_db}, Q={n_q}"
        )

    print("    SF local alias rebuilt: 154890 database / 233 queries", flush=True)
    return sf_data_root

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace-root", required=True)
    ap.add_argument("--baseline-run-id", default="run02")
    ap.add_argument(
        "--config",
        default="config_run02_with_sf_partial_final_v2.json",
    )
    ap.add_argument(
        "--safe-script",
        default="/content/Q1_Budget_Safe_Allocator_v1.py",
    )
    ap.add_argument(
        "--retrievers",
        nargs="+",
        default=["salad", "boq-dinov2"],
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Default: all datasets in config.",
    )
    ap.add_argument("--local-top-k", type=int, default=50)
    ap.add_argument("--descriptor-batch-size", type=int, default=16)
    ap.add_argument("--descriptor-shard-images", type=int, default=1000)
    ap.add_argument("--descriptor-loader-workers", type=int, default=4)
    ap.add_argument(
        "--max-concat-gallery-images", type=int, default=0,
        help="0 forces exact shard-wise L2 retrieval for every gallery; this is the safe default."
    )
    ap.add_argument("--matching-shard-queries", type=int, default=25)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--output-root", default=None)
    ap.add_argument(
        "--seed-retrieval-root",
        default=None,
        help=(
            "Optional prior output root from which to seed exact global retrieval "
            "NPZs. Expected layout: <root>/<retriever>/retrieval/<dataset>.npz. "
            "Used only when the current output retrieval file is absent. This is "
            "intended for a second local-backend robustness run, where global "
            "retrieval is scientifically identical and must not be recomputed."
        ),
    )
    ap.add_argument(
        "--sf-data-root",
        default=None,
        help=(
            "Optional data root containing san_francisco_partial_embedded_gps. "
            "Used only for that dataset; all other datasets continue to use cfg.data_root."
        ),
    )
    args = ap.parse_args()

    # Hard safeguard against accidental post-freeze tuning.
    if args.local_top_k != FROZEN_MATCH_BUDGET:
        raise SystemExit(
            f"CONFIRMATORY FREEZE: --local-top-k must remain {FROZEN_MATCH_BUDGET}"
        )

    W = Path(args.workspace_root).expanduser().resolve()
    baseline_run = W / "runs" / args.baseline_run_id
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = W / config_path
    safe_path = Path(args.safe_script).expanduser().resolve()

    if not baseline_run.is_dir():
        raise SystemExit(f"Baseline run not found: {baseline_run}")
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    if not safe_path.is_file():
        raise SystemExit(f"Budget-safe script not found: {safe_path}")

    OUT = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else W / "q1_confirmatory_multi_retriever_v1"
    )
    OUT.mkdir(parents=True, exist_ok=True)

    # Project + safe scorer.
    sys.path.insert(0, str(W / "src"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("q1_safe_v1", str(safe_path))
    safe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safe)

    from vprbudget.config import load_config
    from vprbudget.data import load_dataset, candidate_distances
    from vprbudget.retrieval import topk_search
    from vprbudget.models import resolve_device

    cfg = load_config(str(config_path))
    datasets = list(args.datasets) if args.datasets else list(cfg.datasets)

    # We must use the same query cohort as run02.
    device = resolve_device(cfg.device, cfg.require_cuda)

    all_condition_rows = []
    all_perq = []
    provenance = {
        "analysis": CONFIRMATORY_VERSION,
        "status": "FROZEN_CONFIRMATORY",
        "created_unix": time.time(),
        "workspace": str(W),
        "baseline_run_id": args.baseline_run_id,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "safe_script": str(safe_path),
        "safe_script_sha256": sha256_file(safe_path),
        "frozen_method": {
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "entropy_temperature": FROZEN_ENTROPY_T,
            "uncertainty_calibration": "UNLABELED_WITHIN_WORKLOAD_PERCENTILE",
            "high_uncertainty_fraction": FROZEN_HIGH_UNCERTAINTY_FRACTION,
            "high_uncertainty_budget": FROZEN_MATCH_BUDGET,
            "low_uncertainty_budget": 0,
            "nominal_mean_budget": (
                FROZEN_HIGH_UNCERTAINTY_FRACTION * FROZEN_MATCH_BUDGET
            ),
            "primary_uniform_comparator": PRIMARY_UNIFORM_BUDGET,
            "same_cost_margin_comparator": "LOWEST_40_PERCENT_TOP1_TOP2_MARGIN_GET_M50",
        },
        "retrievers": {},
        "datasets": datasets,
        "seed_retrieval_root": (
            str(Path(args.seed_retrieval_root).expanduser().resolve())
            if args.seed_retrieval_root else None
        ),
        "local_backend": {
            "detector": str(cfg.detector),
            "matcher": str(cfg.matcher),
            "geometric_model": str(cfg.geometric_model),
            "max_keypoints": int(cfg.max_keypoints),
            "matching_image_size": int(cfg.matching_image_size),
        },
        "confirmatory_rule": (
            "Do not alter method constants after inspecting these results."
        ),
    }

    for retriever_name in args.retrievers:
        print()
        print("=" * 78)
        print("RETRIEVER:", retriever_name)
        print("=" * 78)

        model, source = load_official_retriever(retriever_name, device)
        provenance["retrievers"][retriever_name] = source
        # In-memory per-model cache: exact ordered galleries are descriptor-identical.
        gallery_desc_cache = {}

        for ds_name in datasets:
            print()
            print(f"[{retriever_name}] DATASET: {ds_name}")
            if (
                ds_name == "san_francisco_partial_embedded_gps"
                and args.sf_data_root is not None
            ):
                ds_root = Path(args.sf_data_root).expanduser().resolve()
                ds_root = ensure_sf_local_alias(W, ds_root)
                print(f"    dataset root override: {ds_root}", flush=True)
            else:
                ds_root = Path(cfg.data_root)
            ds = load_dataset(ds_name, ds_root)

            base_ret_path = baseline_run / "retrieval" / f"{ds_name}.npz"
            base_sc_path = baseline_run / "scoring" / f"{ds_name}.npz"
            have_baseline_pair_cache = base_ret_path.is_file() and base_sc_path.is_file()

            if have_baseline_pair_cache:
                base_ret = np.load(base_ret_path)
                baseline_qidx = base_ret["query_indices"].astype(np.int64)
                query_cache_name = "queries_existing_cohort"
                print(
                    f"    query cohort: reused {len(baseline_qidx)} baseline query identities",
                    flush=True,
                )
            else:
                baseline_qidx = np.arange(ds.n_queries, dtype=np.int64)
                query_cache_name = "queries_external_all"
                print(
                    f"    query cohort: unseen dataset; using all {len(baseline_qidx)} test queries",
                    flush=True,
                )

            # -------- exact global retrieval (restart-safe and optionally seedable)
            # Retrieval depends only on the global retriever and dataset, never on
            # the local detector.  Therefore a second-local-backend run may copy
            # the exact frozen retrieval NPZ from the primary run.  Likewise, on
            # ordinary resume we check retrieval BEFORE descriptor extraction, so
            # completed global work is never repeated after a runtime interruption.
            ret_out = OUT / retriever_name / "retrieval" / f"{ds_name}.npz"

            if (not ret_out.exists()) and args.seed_retrieval_root:
                seed_root = Path(args.seed_retrieval_root).expanduser().resolve()
                seed_ret = seed_root / retriever_name / "retrieval" / f"{ds_name}.npz"
                if seed_ret.is_file():
                    seed = np.load(seed_ret)
                    required = {"query_indices", "candidates", "descriptor_distances", "geo_distances"}
                    if not required.issubset(seed.files):
                        raise RuntimeError(
                            f"Seed retrieval missing arrays for {retriever_name}/{ds_name}: "
                            f"{sorted(required - set(seed.files))}"
                        )
                    sq = seed["query_indices"].astype(np.int64)
                    sc = seed["candidates"].astype(np.int64)
                    sd = seed["descriptor_distances"]
                    sg = seed["geo_distances"]
                    if not np.array_equal(sq, baseline_qidx):
                        raise RuntimeError(
                            f"Seed retrieval query cohort mismatch for {retriever_name}/{ds_name}"
                        )
                    if sc.shape != (len(baseline_qidx), RETRIEVAL_TOP_K):
                        raise RuntimeError(
                            f"Seed retrieval candidate shape mismatch for {retriever_name}/{ds_name}: {sc.shape}"
                        )
                    if sd.shape != sc.shape or sg.shape != sc.shape:
                        raise RuntimeError(
                            f"Seed retrieval array shape mismatch for {retriever_name}/{ds_name}"
                        )
                    if sc.size and (sc.min() < 0 or sc.max() >= ds.n_database):
                        raise RuntimeError(
                            f"Seed retrieval candidate index outside database for {retriever_name}/{ds_name}"
                        )
                    ret_out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(seed_ret, ret_out)
                    print(
                        f"    retrieval: SEEDED exact frozen NPZ from {seed_ret}",
                        flush=True,
                    )

            if ret_out.exists():
                ret_new = np.load(ret_out)
                required = {"query_indices", "candidates", "descriptor_distances", "geo_distances"}
                if not required.issubset(ret_new.files):
                    raise RuntimeError(
                        f"Cached retrieval is incomplete for {retriever_name}/{ds_name}"
                    )
                if not np.array_equal(
                    ret_new["query_indices"].astype(np.int64), baseline_qidx
                ):
                    raise RuntimeError(
                        f"Cached retrieval query cohort mismatch for {retriever_name}/{ds_name}"
                    )
                candidates = ret_new["candidates"].astype(np.int64)
                if candidates.shape != (len(baseline_qidx), RETRIEVAL_TOP_K):
                    raise RuntimeError(
                        f"Cached retrieval shape mismatch for {retriever_name}/{ds_name}: {candidates.shape}"
                    )
                print(
                    "    retrieval: cached/seeded; descriptor extraction skipped",
                    flush=True,
                )
            else:
                # -------- global descriptors, frozen query cohort
                desc_dir = OUT / retriever_name / "descriptors" / ds_name
                q_paths = [ds.query_paths[int(i)] for i in baseline_qidx]
                q_desc = extract_descriptor_shards(
                    model,
                    q_paths,
                    desc_dir,
                    query_cache_name,
                    device,
                    batch_size=args.descriptor_batch_size,
                    shard_images=args.descriptor_shard_images,
                    loader_workers=args.descriptor_loader_workers,
                    concatenate=True,
                )

                # Exact ordered gallery reuse (not merely same dataset-family name).
                gsig = gallery_signature(ds.database_paths)
                if gsig in gallery_desc_cache:
                    db_repr = gallery_desc_cache[gsig]["repr"]
                    db_mode = gallery_desc_cache[gsig]["mode"]
                    src_ds = gallery_desc_cache[gsig]["dataset"]
                    print(
                        f"    database descriptors: REUSED exact ordered gallery from {src_ds}"
                    )
                else:
                    concatenate_db = (
                        args.max_concat_gallery_images > 0
                        and len(ds.database_paths) <= args.max_concat_gallery_images
                    )
                    db_repr = extract_descriptor_shards(
                        model,
                        ds.database_paths,
                        desc_dir,
                        "database",
                        device,
                        batch_size=args.descriptor_batch_size,
                        shard_images=args.descriptor_shard_images,
                        loader_workers=args.descriptor_loader_workers,
                        concatenate=concatenate_db,
                    )
                    db_mode = "concat" if concatenate_db else "shards"
                    gallery_desc_cache[gsig] = {
                        "repr": db_repr, "mode": db_mode, "dataset": ds_name
                    }
                    if db_mode == "shards":
                        print(
                            f"    database descriptors: kept as {len(db_repr)} shards "
                            f"to avoid multi-GB RAM concatenation"
                        )

                # -------- exact retrieval
                print(
                    f"    retrieval phase: starting exact top-{RETRIEVAL_TOP_K} search",
                    flush=True,
                )
                if db_mode == "concat":
                    dist, candidates = topk_search(
                        db_repr, q_desc, RETRIEVAL_TOP_K
                    )
                else:
                    dist, candidates = exact_topk_search_shards(
                        db_repr, q_desc, RETRIEVAL_TOP_K
                    )
                geo = np.stack([
                    candidate_distances(
                        ds.query_xy[int(q)],
                        ds.database_xy[candidates[i]],
                    )
                    for i, q in enumerate(baseline_qidx)
                ]).astype(np.float32)
                atomic_npz(
                    ret_out,
                    query_indices=baseline_qidx,
                    candidates=candidates.astype(np.int64),
                    descriptor_distances=dist.astype(np.float32),
                    geo_distances=geo,
                )
                print(
                    f"    retrieval: wrote {len(baseline_qidx)} x top-{RETRIEVAL_TOP_K}",
                    flush=True,
                )

            # -------- pair evidence reuse + missing matching
            if have_baseline_pair_cache:
                qidx_check, reuse = baseline_pair_evidence(
                    baseline_run, ds_name
                )
                if not np.array_equal(qidx_check, baseline_qidx):
                    raise RuntimeError(
                        f"{ds_name}: baseline query cohort changed unexpectedly"
                    )
            else:
                # Truly unseen external dataset: there is no MegaLoc local cache to
                # borrow from. Start empty. A later retriever may reuse identical
                # image-pair evidence computed by an earlier retriever under the
                # SAME local backend.
                reuse = {}

            prior_added = augment_reuse_from_prior_retrievers(
                OUT, ds_name, reuse, retriever_name, args.local_top_k
            )
            if prior_added:
                print(
                    f"    pair evidence: added {prior_added} cached pairs from prior retriever(s)"
                )

            compact_dir = (
                OUT / retriever_name / "compact_matching" / ds_name
            )
            inl, mat, spatial, match_stats = compact_match_shards(
                cfg,
                ds,
                baseline_qidx,
                candidates,
                reuse,
                compact_dir,
                local_top_k=args.local_top_k,
                shard_queries=args.matching_shard_queries,
            )

            # -------- frozen confirmatory evaluation
            perq = evaluate_dataset(
                cfg, ds_name, ret_out, inl, mat, spatial, safe
            )
            perq.insert(0, "retriever", retriever_name)
            all_perq.append(perq)

            policy = perq["frozen_policy_correct"].to_numpy()
            uniform20 = perq["uniform_m20_correct"].to_numpy()
            margin = perq["margin_policy_correct"].to_numpy()
            retrieval = perq["retrieval_correct"].to_numpy()
            uniform50 = perq["uniform_m50_correct"].to_numpy()

            ci_u = paired_bootstrap_pp(
                policy, uniform20, args.n_bootstrap,
                cfg.bootstrap_seed + len(all_condition_rows) * 17 + 1,
            )
            ci_m = paired_bootstrap_pp(
                policy, margin, args.n_bootstrap,
                cfg.bootstrap_seed + len(all_condition_rows) * 17 + 2,
            )

            row = {
                "retriever": retriever_name,
                "dataset": ds_name,
                "n": len(perq),
                "retrieval_r1": float(retrieval.mean()),
                "uniform_m20_r1": float(uniform20.mean()),
                "uniform_m50_r1": float(uniform50.mean()),
                "frozen_policy_mean_budget": float(
                    perq["frozen_policy_budget"].mean()
                ),
                "frozen_policy_r1": float(policy.mean()),
                "delta_policy_vs_uniform_m20_pp": ci_u["difference_pp"],
                "delta_policy_vs_uniform_m20_ci_low_pp": ci_u["ci_low_pp"],
                "delta_policy_vs_uniform_m20_ci_high_pp": ci_u["ci_high_pp"],
                "margin_same_cost_r1": float(margin.mean()),
                "delta_policy_vs_margin_pp": ci_m["difference_pp"],
                "delta_policy_vs_margin_ci_low_pp": ci_m["ci_low_pp"],
                "delta_policy_vs_margin_ci_high_pp": ci_m["ci_high_pp"],
                **match_stats,
            }
            all_condition_rows.append(row)

            pd.DataFrame(all_condition_rows).to_csv(
                OUT / "confirmatory_condition_results.csv",
                index=False,
            )
            pd.concat(all_perq, ignore_index=True).to_csv(
                OUT / "confirmatory_per_query.csv",
                index=False,
            )
            atomic_json(
                OUT / "confirmatory_manifest.json",
                provenance,
            )

            print(
                f"    R@1 retrieval={row['retrieval_r1']:.4f}, "
                f"uniform20={row['uniform_m20_r1']:.4f}, "
                f"policy={row['frozen_policy_r1']:.4f}, "
                f"margin={row['margin_same_cost_r1']:.4f}"
            )

        # Free retriever before the next model.
        del model
        try:
            import torch
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------ pooled descriptive results by retriever
    cond = pd.DataFrame(all_condition_rows)
    pooled = []
    for rname, g in cond.groupby("retriever"):
        total = int(g["n"].sum())

        def wt(col):
            return float(np.sum(g[col] * g["n"]) / total)

        pooled.append({
            "retriever": rname,
            "n_query_conditions": total,
            "n_datasets": len(g),
            "retrieval_r1": wt("retrieval_r1"),
            "uniform_m20_r1": wt("uniform_m20_r1"),
            "frozen_policy_r1": wt("frozen_policy_r1"),
            "margin_same_cost_r1": wt("margin_same_cost_r1"),
            "uniform_m50_r1": wt("uniform_m50_r1"),
            "frozen_policy_mean_budget": wt("frozen_policy_mean_budget"),
            "delta_policy_vs_uniform_m20_pp": (
                wt("frozen_policy_r1") - wt("uniform_m20_r1")
            ) * 100.0,
            "delta_policy_vs_margin_pp": (
                wt("frozen_policy_r1") - wt("margin_same_cost_r1")
            ) * 100.0,
        })

    pd.DataFrame(pooled).to_csv(
        OUT / "confirmatory_pooled_by_retriever.csv",
        index=False,
    )

    # Machine-readable freeze reminder.
    provenance["completed"] = True
    provenance["outputs"] = [
        "confirmatory_condition_results.csv",
        "confirmatory_pooled_by_retriever.csv",
        "confirmatory_per_query.csv",
    ]
    atomic_json(OUT / "confirmatory_manifest.json", provenance)

    print()
    print("CONFIRMATORY RUN COMPLETE:", OUT)
    print("Do not tune the frozen allocation policy from these results.")


if __name__ == "__main__":
    main()
