"""Model loading, isolated so that a change in an upstream API touches one file.

The MegaLoc entry point below follows the loading pattern documented in the
official repository (https://github.com/gmberton/MegaLoc). CONFIRM IT before a
long run: `python -m vprbudget.pipeline --stage validate` exercises it cheaply.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch


def resolve_device(requested: str, require_cuda: bool) -> torch.device:
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            if require_cuda:
                raise RuntimeError(
                    "CUDA requested but unavailable. In Colab: Runtime > Change runtime "
                    "type > GPU. Set require_cuda=false only for CPU smoke tests -- never "
                    "for reported runtime measurements.")
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device(requested)


def load_global_model(name: str, device: torch.device) -> torch.nn.Module:
    if name.lower() != "megaloc":
        raise ValueError(f"unsupported global model {name!r}")
    try:
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
    except Exception as e:
        raise RuntimeError(
            "Failed to load MegaLoc via torch.hub. Check network access and the current "
            "entry-point name in https://github.com/gmberton/MegaLoc . This is the one "
            "external API in the pipeline that must be confirmed against upstream."
        ) from e
    return model.eval().to(device)


def load_matcher(detector: str, matcher: str, max_keypoints: int,
                 device: torch.device) -> Tuple[Any, Any]:
    if detector.lower() != "superpoint" or matcher.lower() != "lightglue":
        raise ValueError(f"unsupported pair: {detector}+{matcher}")
    try:
        from lightglue import LightGlue, SuperPoint
    except ImportError as e:
        raise RuntimeError("pip install lightglue (see requirements.txt)") from e
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    lg = LightGlue(features="superpoint").eval().to(device)
    return extractor, lg


def set_global_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_metadata(device: torch.device) -> dict:
    md = {"torch": torch.__version__, "device": str(device),
          "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        md["gpu_name"] = torch.cuda.get_device_name(0)
        md["cuda"] = torch.version.cuda
    return md
