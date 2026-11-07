"""Dataset discovery, ground-truth coordinates, deterministic query subsets.

Layout assumed (the convention produced by gmberton/VPR-datasets-downloader):

    <data_root>/<dataset>/images/test/database/*.jpg
    <data_root>/<dataset>/images/test/queries/*.jpg

with UTM coordinates encoded in the filename as

    @utm_east@utm_north@...@.jpg

VERIFY THIS AGAINST YOUR ACTUAL DOWNLOAD. If the convention differs, either
(a) point `coords_csv` at a CSV with columns [filename, east, north], or
(b) adjust `FILENAME_COORD_FIELDS`. The parser fails loudly with a real example
rather than silently producing wrong ground truth.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
FILENAME_COORD_FIELDS = (1, 2)   # 0-based fields between '@' holding (east, north)


class DatasetError(RuntimeError):
    pass


@dataclass
class VPRDataset:
    name: str
    root: Path
    database_paths: List[Path]
    query_paths: List[Path]
    database_xy: np.ndarray      # (Nd, 2) float64, metres
    query_xy: np.ndarray         # (Nq, 2) float64, metres
    query_frame_idx: Optional[np.ndarray] = None   # for frame-index ground truth
    coord_diagnostic: Optional[dict] = None        # populated by _sanity_check

    @property
    def n_database(self) -> int:
        return len(self.database_paths)

    @property
    def n_queries(self) -> int:
        return len(self.query_paths)


def _list_images(d: Path) -> List[Path]:
    if not d.is_dir():
        raise DatasetError(f"not a directory: {d}")
    files = sorted(p for p in d.iterdir() if p.suffix in IMAGE_EXTS)
    if not files:
        raise DatasetError(f"no images found in {d}")
    return files


def parse_xy_from_filename(path: Path) -> Tuple[float, float]:
    parts = path.name.split("@")
    i, j = FILENAME_COORD_FIELDS
    if len(parts) <= max(i, j):
        raise DatasetError(
            f"cannot parse coordinates from filename {path.name!r}: expected '@'-separated "
            f"fields with east at index {i} and north at index {j}, found {len(parts)} fields. "
            f"Supply a coords CSV instead, or fix FILENAME_COORD_FIELDS in data.py.")
    try:
        return float(parts[i]), float(parts[j])
    except ValueError as e:
        raise DatasetError(
            f"non-numeric coordinate in filename {path.name!r}: fields "
            f"{parts[i]!r}, {parts[j]!r}") from e


def _load_coords(paths: List[Path], coords_csv: Optional[Path]) -> np.ndarray:
    if coords_csv is not None and coords_csv.exists():
        import csv
        table: Dict[str, Tuple[float, float]] = {}
        with open(coords_csv, newline="") as f:
            for row in csv.DictReader(f):
                table[row["filename"]] = (float(row["east"]), float(row["north"]))
        missing = [p.name for p in paths if p.name not in table]
        if missing:
            raise DatasetError(f"{len(missing)} images missing from {coords_csv}; "
                               f"first: {missing[:3]}")
        return np.array([table[p.name] for p in paths], dtype=np.float64)
    return np.array([parse_xy_from_filename(p) for p in paths], dtype=np.float64)


def _paths_from_manifest(root: Path, base: Path, manifest: Path):
    """Use existing sorted historical filename manifest; never readdir a huge Drive folder.

    In particular, Nordland contains 27,592 items *per folder*. Colab DriveFS
    can fail with errno 5 on os.scandir even when individual image files exist.
    Filename-only metadata preserves all original gallery and query indices.
    """
    root=Path(root);base=Path(base)
    entries={'database':[], 'queries':[]}
    prefix=base.relative_to(root).as_posix()+'/'
    seen=set()
    with Path(manifest).open('r',encoding='utf8') as f:
        for lineno,raw in enumerate(f,1):
            rel=raw.strip().replace('\\','/')
            if not rel:continue
            if rel.startswith('/') or '..' in Path(rel).parts or not rel.startswith(prefix):
                raise DatasetError(f'Unsafe/unexpected manifest path in {manifest}, line {lineno}: {rel}')
            tail=rel[len(prefix):].split('/')
            if len(tail)!=2 or tail[0] not in entries or Path(tail[1]).suffix not in IMAGE_EXTS:
                raise DatasetError(f'Unexpected image manifest record {lineno}: {rel}')
            if rel in seen:raise DatasetError(f'Duplicate manifest filename: {rel}')
            seen.add(rel)
            entries[tail[0]].append(base/tail[0]/tail[1])
    for kind in entries:
        if not entries[kind]:raise DatasetError(f'{manifest}: missing {kind} records')
        entries[kind].sort(key=lambda p:p.name)
    return entries['database'],entries['queries']


def _save_manifest_once(root: Path, base: Path, db_paths, q_paths, manifest: Path):
    """Persist a one-time Baidu manifest to Drive; refuse to alter an existing one."""
    manifest=Path(manifest)
    lines=[]
    for paths in (db_paths,q_paths):
        lines.extend(p.relative_to(root).as_posix()+'\n' for p in paths)
    text=''.join(lines)
    if manifest.exists():
        if manifest.read_text(encoding='utf8')!=text:
            raise DatasetError('Persistent manifest differs from live listing: '+str(manifest))
        return
    manifest.parent.mkdir(parents=True,exist_ok=True)
    tmp=manifest.with_name(manifest.name+'.incomplete')
    tmp.write_text(text,encoding='utf8')
    os.replace(tmp,manifest)
    if manifest.read_text(encoding='utf8')!=text:
        raise DatasetError('Manifest Drive write/read mismatch: '+str(manifest))


def load_dataset(name: str, data_root: Path,
                 split_subdir: str = "images/test", manifest_dir: Optional[Path] = None) -> VPRDataset:
    root = Path(data_root) / name
    if not root.is_dir():
        raise DatasetError(f'dataset directory not found: {root}')
    base=root/split_subdir
    if not base.is_dir():
        cands=[p for p in root.rglob('database') if p.is_dir()]
        if not cands:raise DatasetError(f'could not locate a database folder under {root}')
        base=cands[0].parent
    # First-party data export, not a reconstructed approximation.
    native=root/'all_images_paths.txt'
    portable=Path(manifest_dir)/f'{name}_all_images_paths.txt' if manifest_dir else None
    if native.is_file():
        db_paths,q_paths=_paths_from_manifest(root,base,native)
        source=str(native)
    elif portable is not None and portable.is_file():
        db_paths,q_paths=_paths_from_manifest(root,base,portable)
        source=str(portable)
    else:
        db_paths=_list_images(base/'database')
        q_paths=_list_images(base/'queries')
        source='single verified live listing'
        if portable is not None:
            _save_manifest_once(root,base,db_paths,q_paths,portable)
    expected={'nordland':(27592,27592),'baidu':(5207,3208),'st_lucia':(1549,None)}
    if name in expected:
        nd,nq=expected[name]
        if len(db_paths)!=nd or (nq is not None and len(q_paths)!=nq):
            raise DatasetError(f'{name} image-count mismatch from {source}: database={len(db_paths)},query={len(q_paths)}')
    db_xy=_load_coords(db_paths, root/'database_coords.csv')
    q_xy=_load_coords(q_paths, root/'query_coords.csv')
    ds=VPRDataset(name=name,root=root,database_paths=db_paths,query_paths=q_paths,
                  database_xy=db_xy,query_xy=q_xy)
    _sanity_check(ds)
    ds.manifest_source=source
    return ds


def _sanity_check(ds: VPRDataset) -> None:
    if not np.isfinite(ds.database_xy).all() or not np.isfinite(ds.query_xy).all():
        raise DatasetError(f"{ds.name}: non-finite coordinates")
    spread = ds.database_xy.max(0) - ds.database_xy.min(0)
    if float(spread.max()) < 1.0:
        raise DatasetError(
            f"{ds.name}: database coordinates span {spread} m, which is implausibly small. "
            f"The filename coordinate convention is probably wrong -- verify before trusting "
            f"any ground truth built from it.")
    # BLOCKER B-1. Spread alone is NOT sufficient: index placeholders produce a
    # huge spread and passed this check in run01. What distinguishes real metric
    # coordinates is a populated band of small non-zero nearest-neighbour
    # distances. Record the diagnostic; groundtruth.validate_spec enforces it.
    sub = ds.database_xy[:min(len(ds.database_xy), 500)]
    dd = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    off = dd[~np.eye(len(sub), dtype=bool)]
    ds.coord_diagnostic = {
        "n_pairs_sampled": int(off.size),
        "frac_zero": float((off == 0).mean()),
        "n_distinct_under_25m": int(np.unique(off[(off > 0) & (off <= 25.0)]).size),
        "median_m": float(np.median(off)),
    }


def deterministic_query_subset(n_queries: int, subset_size: Optional[int],
                               seed: int) -> np.ndarray:
    """Reproducible query subset. Returns SORTED indices so downstream shard
    boundaries are stable across runs."""
    if subset_size is None or subset_size >= n_queries:
        return np.arange(n_queries, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_queries, size=subset_size, replace=False)
    return np.sort(idx).astype(np.int64)


def positives_mask(query_xy: np.ndarray, cand_xy: np.ndarray,
                   threshold_m: float) -> np.ndarray:
    """Boolean (K,) array: is each candidate within `threshold_m` of the query?"""
    d = np.linalg.norm(cand_xy - query_xy[None, :], axis=1)
    return d <= threshold_m


def candidate_distances(query_xy: np.ndarray, cand_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(cand_xy - query_xy[None, :], axis=1)
