"""Ground-truth derivation, per dataset, with explicit protocol modes.

BLOCKER B-1. In run01 the pipeline assumed every dataset carried metric UTM
coordinates and applied a 25 m radius uniformly. Verification showed that for
AmsterTime and SPED every candidate distance was either exactly 0 or 34-1288 km:
those are index placeholders, not metres. The 25 m threshold was inert for 62%
of queries, and the reported "cross-threshold robustness" was an artefact.

This module makes the protocol explicit and refuses to guess.

Modes
-----
`metric`      : positive iff geodesic distance <= threshold_m. Requires genuinely
                continuous coordinates (SVOX, St Lucia, Eynsham, Baidu, ...).
`pair`        : positive iff the candidate is THE corresponding image. Native
                protocol for AmsterTime (historical<->modern 1-to-1) and for any
                dataset whose coordinates collapse to {0, huge}.
`frame_index` : positive iff |frame(q) - frame(c)| <= tolerance. Native protocol
                for Nordland.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

VALID_MODES = ("metric", "pair", "frame_index")


class GroundTruthError(RuntimeError):
    pass


@dataclass(frozen=True)
class GTSpec:
    mode: str
    threshold_m: Optional[float] = None
    frame_tolerance: Optional[int] = None
    note: str = ""

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise GroundTruthError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.mode == "metric" and self.threshold_m is None:
            raise GroundTruthError("metric mode requires threshold_m")
        if self.mode == "frame_index" and self.frame_tolerance is None:
            raise GroundTruthError("frame_index mode requires frame_tolerance")

    def describe(self) -> str:
        if self.mode == "metric":
            return f"metric radius, tau = {self.threshold_m:g} m"
        if self.mode == "pair":
            return "exact 1-to-1 pair match"
        return f"frame index within +/-{self.frame_tolerance}"


# Native protocol per dataset. Deliberately explicit: an unknown dataset raises
# rather than silently defaulting to a 25 m radius.
DEFAULT_GT: Dict[str, GTSpec] = {
    "svox":        GTSpec("metric", threshold_m=25.0, note="SVOX gallery has real UTM"),
    "svox_sun":    GTSpec("metric", threshold_m=25.0),
    "svox_night":  GTSpec("metric", threshold_m=25.0),
    "svox_rain":   GTSpec("metric", threshold_m=25.0),
    "svox_overcast": GTSpec("metric", threshold_m=25.0),
    "st_lucia":    GTSpec("metric", threshold_m=25.0),
    "eynsham":     GTSpec("metric", threshold_m=25.0),
    "baidu":       GTSpec("metric", threshold_m=25.0, note="indoor; consider 10 m"),
    "san_francisco": GTSpec("metric", threshold_m=25.0),
    "pitts30k":    GTSpec("metric", threshold_m=25.0),
    "tokyo247":    GTSpec("metric", threshold_m=25.0),
    "msls":        GTSpec("metric", threshold_m=25.0),
    "amstertime":  GTSpec("pair", note="historical<->modern 1-to-1; native protocol"),
    "sped":        GTSpec("pair", note="coordinates collapse to {0, huge}"),
    "nordland":    GTSpec("frame_index", frame_tolerance=10,
                          note="native protocol is frame tolerance, not metres"),
}


# --------------------------------------------------------------- diagnostics

def diagnose_coordinates(geo_distances: np.ndarray,
                         metric_tol_m: float = 25.0) -> Dict[str, float]:
    """Decide, from the candidate-distance matrix alone, whether coordinates are
    genuinely metric. This is the check that `data._sanity_check` failed to make:
    it tested coordinate SPREAD, which passes trivially on huge placeholders.

    The diagnostic that matters is whether any candidate distance falls strictly
    inside the (0, tol] band. Real GPS produces hundreds of such values; index
    placeholders produce exactly zero.
    """
    d = np.asarray(geo_distances, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise GroundTruthError("no finite distances to diagnose")
    inside = (d > 0) & (d <= metric_tol_m)
    return {
        "n": int(d.size),
        "frac_exactly_zero": float((d == 0).mean()),
        "frac_in_open_band": float(inside.mean()),
        "n_distinct_in_band": int(np.unique(d[inside]).size),
        "median": float(np.median(d)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
    }


def infer_mode(geo_distances: np.ndarray, metric_tol_m: float = 25.0) -> str:
    """-> 'metric' or 'pair'. Used to CROSS-CHECK the declared spec, never to
    silently override it."""
    s = diagnose_coordinates(geo_distances, metric_tol_m)
    if s["n_distinct_in_band"] >= 20 and s["frac_exactly_zero"] < 0.01:
        return "metric"
    return "pair"


def validate_spec(dataset: str, spec: GTSpec, geo_distances: np.ndarray,
                  strict: bool = True) -> Dict[str, object]:
    """Confirm the declared protocol is consistent with the data. Raises in
    strict mode on a mismatch -- which is the whole point of this module."""
    stats = diagnose_coordinates(geo_distances,
                                 spec.threshold_m if spec.mode == "metric" else 25.0)
    inferred = infer_mode(geo_distances)
    ok = (spec.mode != "metric") or (inferred == "metric")
    report = {"dataset": dataset, "declared_mode": spec.mode, "inferred_mode": inferred,
              "consistent": bool(ok), **stats}
    if strict and not ok:
        raise GroundTruthError(
            f"{dataset}: declared mode 'metric' with tau={spec.threshold_m} m, but the "
            f"distances contain {stats['n_distinct_in_band']} distinct values in (0, tau] "
            f"and {stats['frac_exactly_zero']:.1%} exact zeros. These are index "
            f"placeholders, not metres. Use GTSpec('pair') for this dataset, or supply "
            f"real coordinates via <dataset>/database_coords.csv.")
    return report


# ------------------------------------------------------------------ labelling

def positives(geo_distances: np.ndarray, spec: GTSpec,
              query_frames: Optional[np.ndarray] = None,
              cand_frames: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean (Nq, K) matrix of positives under the declared protocol."""
    d = np.asarray(geo_distances, dtype=np.float64)
    if spec.mode == "metric":
        return d <= float(spec.threshold_m)
    if spec.mode == "pair":
        return d == 0.0
    if query_frames is None or cand_frames is None:
        raise GroundTruthError("frame_index mode requires query_frames and cand_frames")
    return np.abs(np.asarray(cand_frames) - np.asarray(query_frames)[:, None]) <= int(spec.frame_tolerance)


def get_spec(dataset: str, overrides: Optional[Dict[str, dict]] = None) -> GTSpec:
    if overrides and dataset in overrides:
        return GTSpec(**overrides[dataset])
    if dataset not in DEFAULT_GT:
        raise GroundTruthError(
            f"no ground-truth protocol registered for dataset {dataset!r}. Add it to "
            f"DEFAULT_GT or pass dataset_gt_overrides, e.g. "
            f"{{'{dataset}': {{'mode': 'metric', 'threshold_m': 25.0}}}}. "
            f"Refusing to assume a 25 m radius.")
    return DEFAULT_GT[dataset]


def threshold_grid_for(spec: GTSpec, grid: Sequence[float]) -> Tuple[float, ...]:
    """Only metric datasets have a meaningful threshold sweep. For pair and
    frame-index datasets a sweep is meaningless and reporting one as evidence of
    robustness is the artefact we are removing."""
    return tuple(grid) if spec.mode == "metric" else (float("nan"),)
