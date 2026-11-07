# Official model sources and dependency provenance

The original evaluation uses MegaLoc for allocator development and SALAD and BoQ-DINOv2 for frozen-policy confirmation. The primary local detector/matcher is SuperPoint–LightGlue, with MAGSAC robust geometric estimation; a detector-transfer analysis uses ALIKED–LightGlue. Third-party pretrained retrievers and local-match models are not committed here.

| External component | Recorded entrypoint / provenance | Upstream license |
| --- | --- | --- |
| [BoQ-DINOv2](https://github.com/amaralibey/Bag-of-Queries) | `get_trained_boq(backbone_name='dinov2', output_dim=12288)`; A100 source checkout `1a4965ea7dfd9bd0dd846adf7a0e430f68101d12`; `hubconf.py` SHA-256 `26d1cd9f1bd12b1f407b2441b7dae37223e7d0e59435ad88abebef813d7b1d2b` | MIT |
| [SALAD](https://github.com/serizba/salad) | Official `dinov2_salad` source entrypoint; exact source commit not documented in this package | GPL-3.0 |
| [LightGlue](https://github.com/cvg/LightGlue) | A100 source checkout `eb42fee2d71449efb0aa5c10549752b5d75384d8` | Apache-2.0 |
| [MegaLoc](https://github.com/gmberton/MegaLoc) | Development retrievals; historical source revision not established by the A100 profiling record | MIT |

The A100 environment records PyTorch `2.11.0+cu130`, CUDA `13.0`, FAISS `1.15.1`, NumPy `2.4.6`, OpenCV `4.14.0`, and Python `3.13.16`. Other dependencies were not recorded as a complete lock; `requirements_unlocked.txt` lists imports and `requirements_recorded_scope.md` documents known gaps. The pinned CPU verification requirements are **not** a complete GPU environment specification.

Recorded A100 model-weight hashes:

| File | SHA-256 |
| --- | --- |
| `dinov2_12288.pth` | `d72ee0ce899e2790be6cf6d1d57f75dfad0c76475bd2198e6dc0c3f0221b3583` |
| `dinov2_vitb14_pretrain.pth` | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |
| `superpoint_lightglue_v0-1_arxiv.pth` | `6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a` |
| `superpoint_v1.pth` | `52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e` |

These values are transcribed from `profiling/a100/RECORDED_ENVIRONMENT_NO_PRIVATE_PATHS.json`. The BoQ source commit belongs to the A100 recorded checkout and, by itself, does **not** establish the revision of every earlier retriever execution. The original model weights, image datasets, upstream code, and source license files are not redistributed here. `src/boq_hub_recovery.py` is a local source-loading utility, not a copied BoQ model. Observe upstream licenses for any separately downloaded, adapted, or redistributed third-party code.
