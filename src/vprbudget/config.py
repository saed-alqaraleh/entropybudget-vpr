"""Single source of truth for every constant. Nothing numeric lives in the code."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Stage names, in dependency order.
STAGES = ["validate", "descriptors", "retrieval", "matching", "scoring", "metrics",
          "stats", "anchor", "c1", "c2", "report"]

# Which config keys invalidate which stage. Editing a stats setting must not
# invalidate 40 GPU-hours of matching.
STAGE_KEYS: Dict[str, List[str]] = {
    "validate":    ["datasets"],
    "descriptors": ["datasets", "global_model", "retrieval_image_size", "descriptor_batch_size"],
    "retrieval":   ["datasets", "global_model", "retrieval_image_size", "top_k",
                    "query_subset_size", "query_subset_seed"],
    "matching":    ["datasets", "global_model", "retrieval_image_size", "top_k",
                    "query_subset_size", "query_subset_seed", "matching_image_size",
                    "max_keypoints", "detector", "matcher", "geometric_model",
                    "ransac_reproj_threshold", "ransac_confidence", "ransac_max_iters",
                    "min_matches_for_gv"],
    "scoring":     ["*"],   # cheap; always recompute
    "metrics":     ["*"],
    "stats":       ["*"],
    "anchor":      ["*"],
    "c1":          ["*"],
    "c2":          ["*"],
    "report":      ["*"],
}


@dataclass
class Config:
    # ---- workspace ---------------------------------------------------------
    workspace_root: str = "./workspace"
    run_id: str = "run01"
    data_root: str = "./datasets"
    scratch_root: str = "/content/scratch"     # fast local disk; safe to lose

    # ---- datasets ----------------------------------------------------------
    # DEVELOPMENT group: everything is tuned here. TEST group stays untouched
    # until the protocol is frozen (see research_program_v01.md, Session S4).
    dev_datasets: List[str] = field(default_factory=lambda: ["svox_sun", "baidu", "st_lucia"])
    test_datasets: List[str] = field(default_factory=lambda: [
        "svox_night", "amstertime", "eynsham", "sped", "nordland"])
    # Stages run on this list. Keep TEST out of it until S4 sign-off.
    datasets: List[str] = field(default_factory=lambda: ["svox_sun", "baidu", "st_lucia"])

    # ---- ground truth ------------------------------------------------------
    positive_threshold_m: float = 25.0
    # Sensitivity grid. Audit issue MJ10: a single threshold is not defensible
    # across indoor / route / historical datasets.
    threshold_grid_m: List[float] = field(default_factory=lambda: [5.0, 10.0, 25.0])
    # Per-dataset override, e.g. {"nordland": {"mode": "frame_index", "tolerance": 10}}
    dataset_gt_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ---- retrieval ---------------------------------------------------------
    global_model: str = "megaloc"
    retrieval_image_size: List[int] = field(default_factory=lambda: [322, 322])
    descriptor_batch_size: int = 32
    top_k: int = 100
    query_subset_size: Optional[int] = 1000
    query_subset_seed: int = 12345
    # Audit issue MD2: one subsample gives no variance estimate.
    n_query_subsets: int = 5
    subset_seed_offsets: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # ---- local matching ----------------------------------------------------
    detector: str = "superpoint"
    matcher: str = "lightglue"
    matching_image_size: int = 512
    max_keypoints: int = 2048
    geometric_model: str = "fundamental"   # {"fundamental","homography"}
    ransac_reproj_threshold: float = 3.0
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 10000
    min_matches_for_gv: int = 8            # below this, GV is undefined -> 0 inliers
    matching_shard_size: int = 50          # queries per checkpoint shard

    # ---- scoring -----------------------------------------------------------
    beta_global: float = 0.5
    w_inlier_count: float = 0.7
    w_inlier_ratio: float = 0.3
    gamma_spatial: float = 0.5
    spatial_grid: List[int] = field(default_factory=lambda: [4, 4])
    softmax_temperature: float = 0.1
    lambda_min: float = 0.05
    lambda_max: float = 0.95
    epsilon: float = 1e-8
    rrf_k: int = 60
    topk_support_cutoffs: List[int] = field(default_factory=lambda: [5, 10])
    # Global-reliability weights a1..a4 and local-reliability weights b1..b5.
    # Audit issue B6: these were never reported in the manuscript.
    global_reliability_weights: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    local_reliability_weights: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2])
    fuf_weights: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    # Audit issue B3: terms have incompatible ranges. Normalize per query before fusion.
    normalize_fusion_terms: bool = True

    # ---- matching budget (contribution C2) --------------------------------
    budget_levels: List[int] = field(default_factory=lambda: [0, 1, 2, 5, 10, 20, 50, 100])
    mean_budgets: List[float] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])

    # Methods whose budget curves are computed. Keep this short: the Pareto
    # sweep is O(queries x budgets x policies) and most methods are not of interest.
    budget_methods: List[str] = field(default_factory=lambda: [
        "frozen_universal_fusion", "adaptive_global_spatial", "adaptive_global_local",
        "inlier_count_only", "local_composite", "inlier_ratio_only"])

    # ---- conformal risk control (contribution C3) -------------------------
    conformal_alpha: float = 0.1
    conformal_m_on: int = 10          # budget applied when the gate opens

    # ---- statistics --------------------------------------------------------
    n_bootstrap: int = 10000
    bootstrap_seed: int = 20260801
    confidence_level: float = 0.95
    multiplicity_method: str = "holm"
    # Audit issue MD1: route datasets have non-independent consecutive frames.
    cluster_bootstrap_datasets: List[str] = field(default_factory=lambda: [
        "st_lucia", "eynsham", "nordland"])
    cluster_size: int = 20

    # ---- runtime -----------------------------------------------------------
    device: str = "cuda"
    require_cuda: bool = True
    num_workers: int = 4

    # ---- provenance --------------------------------------------------------
    primary_metric: str = "recall@1"   # preregistered; see audit issue MJ5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -- derived paths -------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.workspace_root) / "runs" / self.run_id

    @property
    def log_dir(self) -> Path:
        return Path(self.workspace_root) / "logs"

    def stage_dir(self, stage: str) -> Path:
        return self.run_dir / stage

    def dataset_dir(self, dataset: str) -> Path:
        return Path(self.data_root) / dataset

    def stage_key_hash(self, stage: str) -> str:
        from .io_utils import config_hash
        keys = STAGE_KEYS.get(stage, ["*"])
        d = self.to_dict()
        if keys == ["*"]:
            keys = sorted(d.keys() - {"workspace_root", "data_root", "scratch_root",
                                      "run_id", "device", "num_workers"})
        return config_hash(d, keys)


def load_config(path: Optional[str] = None, **overrides: Any) -> Config:
    import json
    base = Config()
    if path:
        with open(path, "r") as f:
            user = json.load(f)
        d = base.to_dict()
        unknown = set(user) - set(d)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        d.update(user)
        base = Config(**d)
    if overrides:
        d = base.to_dict()
        unknown = set(overrides) - set(d)
        if unknown:
            raise ValueError(f"unknown override keys: {sorted(unknown)}")
        d.update({k: v for k, v in overrides.items() if v is not None})
        base = Config(**d)
    _validate(base)
    return base


def _validate(cfg: Config) -> None:
    if cfg.geometric_model not in {"fundamental", "homography"}:
        raise ValueError(f"geometric_model must be 'fundamental' or 'homography', got {cfg.geometric_model!r}")
    if not (0.0 <= cfg.lambda_min < cfg.lambda_max <= 1.0):
        raise ValueError("require 0 <= lambda_min < lambda_max <= 1")
    if abs(sum(cfg.fuf_weights) - 1.0) > 1e-9:
        raise ValueError(f"fuf_weights must sum to 1, got {cfg.fuf_weights}")
    if abs(cfg.w_inlier_count + cfg.w_inlier_ratio - 1.0) > 1e-9:
        raise ValueError("w_inlier_count + w_inlier_ratio must equal 1")
    if len(cfg.global_reliability_weights) != 4:
        raise ValueError("global_reliability_weights must have 4 entries (a1..a4)")
    if len(cfg.local_reliability_weights) != 5:
        raise ValueError("local_reliability_weights must have 5 entries (b1..b5)")
    if max(cfg.budget_levels) > cfg.top_k:
        raise ValueError("budget_levels cannot exceed top_k")
    leak = set(cfg.dev_datasets) & set(cfg.test_datasets)
    if leak:
        raise ValueError(f"datasets appear in BOTH dev and test groups: {sorted(leak)}")
    unassigned = set(cfg.datasets) - set(cfg.dev_datasets) - set(cfg.test_datasets)
    if unassigned:
        raise ValueError(
            f"datasets {sorted(unassigned)} belong to neither dev_datasets nor "
            f"test_datasets. BLOCKER B-7: every evaluated dataset must be assigned "
            f"to a group before running, so the held-out claim is checkable.")
    if cfg.conformal_m_on not in cfg.budget_levels:
        raise ValueError(f"conformal_m_on={cfg.conformal_m_on} must be one of budget_levels")
