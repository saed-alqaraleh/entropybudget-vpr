"""Descriptor extraction (resumable, sharded) and FAISS top-K retrieval."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .io_utils import Heartbeat, atomic_write_json, atomic_write_npz


def _desc_shard(stage_dir: Path, dataset: str, split: str, sid: int) -> Path:
    return stage_dir / dataset / f"{split}_{sid:05d}.npz"


def extract_descriptors(model, paths: List[Path], image_size, batch_size: int,
                        device, stage_dir: Path, dataset: str, split: str,
                        logger, shard_images: int = 2000) -> np.ndarray:
    """Extract global descriptors, checkpointing every `shard_images` images."""
    import torch
    from PIL import Image
    import torchvision.transforms as T

    tf = T.Compose([T.Resize(tuple(image_size)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    (stage_dir / dataset).mkdir(parents=True, exist_ok=True)
    n = len(paths)
    bounds = [(i, s, min(s + shard_images, n))
              for i, s in enumerate(range(0, n, shard_images))]
    hb = Heartbeat(logger, f"descriptors[{dataset}/{split}]", total=len(bounds))
    out: List[np.ndarray] = []
    for sid, start, stop in bounds:
        p = _desc_shard(stage_dir, dataset, split, sid)
        if p.exists():
            out.append(np.load(p)["descriptors"])
            hb.update(1, "cached")
            continue
        chunk: List[np.ndarray] = []
        for b in range(start, stop, batch_size):
            batch = paths[b:min(b + batch_size, stop)]
            imgs = torch.stack([tf(Image.open(q).convert("RGB")) for q in batch]).to(device)
            with torch.inference_mode():
                d = model(imgs)
            chunk.append(torch.nn.functional.normalize(d, dim=1).float().cpu().numpy())
        arr = np.concatenate(chunk).astype(np.float32)
        atomic_write_npz(p, descriptors=arr)
        out.append(arr)
        hb.update(1)
    hb.close()
    desc = np.concatenate(out).astype(np.float32)
    if desc.shape[0] != n:
        raise RuntimeError(f"descriptor count {desc.shape[0]} != image count {n}")
    if not np.isfinite(desc).all():
        raise RuntimeError(f"non-finite descriptors for {dataset}/{split}")
    return desc


def topk_search(db_desc: np.ndarray, q_desc: np.ndarray, k: int):
    """Exact L2 search. Exact, not approximate: an ANN index would add a second
    source of variation between methods that we cannot control for."""
    try:
        import faiss
        index = faiss.IndexFlatL2(db_desc.shape[1])
        index.add(np.ascontiguousarray(db_desc))
        dist, idx = index.search(np.ascontiguousarray(q_desc), k)
        return np.sqrt(np.maximum(dist, 0)).astype(np.float32), idx.astype(np.int64)
    except ImportError:
        d2 = ((q_desc[:, None, :] - db_desc[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d2, axis=1)[:, :k]
        dist = np.sqrt(np.take_along_axis(d2, idx, axis=1))
        return dist.astype(np.float32), idx.astype(np.int64)
