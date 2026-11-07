"""Atomic IO, stage markers, config hashing, logging. No science here."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory, then os.replace (atomic on POSIX).

    A crash mid-write therefore leaves either the old file or nothing -- never a
    truncated file that a later run would mistake for valid output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default).encode())


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    import io as _io
    buf = _io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_write_bytes(path, buf.getvalue())


def config_hash(cfg: Dict[str, Any], keys: Iterable[str]) -> str:
    """Stable hash over only the config keys that affect a given stage's output.

    Restricting the key set means changing an unrelated setting does not
    needlessly invalidate expensive completed stages.
    """
    sub = {k: cfg[k] for k in sorted(keys) if k in cfg}
    blob = json.dumps(sub, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- stage markers

MARKER_NAME = "_STAGE_SUCCESS.json"


def marker_path(stage_dir: Path) -> Path:
    return stage_dir / MARKER_NAME


def stage_is_complete(stage_dir: Path, expected_hash: str,
                      required: Iterable[Path] = ()) -> bool:
    """A stage counts as complete only if the marker exists, the config hash
    matches, and every required output is still present on disk."""
    mp = marker_path(stage_dir)
    if not mp.exists():
        return False
    try:
        rec = read_json(mp)
    except (json.JSONDecodeError, OSError):
        return False
    if rec.get("config_hash") != expected_hash:
        return False
    for p in required:
        if not Path(p).exists():
            return False
    return True


def write_stage_marker(stage_dir: Path, expected_hash: str, stage: str,
                       required: Iterable[Path] = (), extra: Optional[Dict] = None) -> None:
    """Validate outputs FIRST, then write the marker. Never the other way round."""
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise RuntimeError(
            f"refusing to write success marker for stage '{stage}': missing outputs {missing}")
    rec = {"stage": stage, "config_hash": expected_hash,
           "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "outputs": [str(p) for p in required]}
    if extra:
        rec.update(extra)
    atomic_write_json(marker_path(stage_dir), rec)


class Heartbeat:
    """Prints progress at most every `every_s` seconds. Never invents a percentage
    when the denominator is unknown."""

    def __init__(self, logger: logging.Logger, label: str, total: Optional[int] = None,
                 every_s: float = 30.0):
        self.log, self.label, self.total, self.every_s = logger, label, total, every_s
        self.t0 = time.time()
        self._last = 0.0
        self.n = 0

    def update(self, n: int = 1, detail: str = "") -> None:
        self.n += n
        now = time.time()
        if now - self._last < self.every_s:
            return
        self._last = now
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        if self.total:
            eta = (self.total - self.n) / rate if rate > 0 else float("nan")
            self.log.info(f"{self.label}: {self.n}/{self.total} "
                          f"({100.0*self.n/self.total:.1f}%) elapsed={el/60:.1f}m "
                          f"eta={eta/60:.1f}m {detail}")
        else:
            self.log.info(f"{self.label}: {self.n} done elapsed={el/60:.1f}m {detail}")

    def close(self) -> None:
        el = time.time() - self.t0
        self.log.info(f"{self.label}: FINISHED {self.n} units in {el/60:.1f}m")
