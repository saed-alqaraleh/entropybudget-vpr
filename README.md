# EntropyBudget: Uncertainty-Guided Query-Adaptive Geometric Verification for Visual Place Recognition under Computational Budgets

**Official implementation and reproducibility materials** for *Uncertainty-Guided Query-Adaptive Geometric Verification for Visual Place Recognition under Computational Budgets*.

EntropyBudget allocates geometric verification across visual place recognition (VPR) queries using retrieval uncertainty measured before local matching. This repository provides the implementation, experimental configurations, evaluation scripts, integrity tests, per-query evidence, retrieval comparisons, and GPU timing records. The verification scripts reproduce the reported numerical summaries from the supplied evidence.

## Quick start

Use Python 3.12 to verify the stored evidence on CPU.

```bash
git clone https://github.com/saed-alqaraleh/entropybudget-vpr.git
cd entropybudget-vpr
python -m pip install -r requirements_cpu_verified.txt
python scripts/verify_release.py --check-inventory
python tests/smoke_authentic_subset.py
python scripts/reproduce_a100_table.py
python scripts/reproduce_compact_retrieval_audit.py
```

These commands verify the release files, run a scoring check, and reconstruct the A100 timing and retrieval comparisons. See [Reproduce the stored evidence](#reproduce-the-stored-evidence) for the full verification procedure.

## Method

Geometric verification can rescue an incorrect retrieved top-1 result or overturn a correct one. The study formalizes this *signed verification utility* as the change in top-1 correctness relative to retrieval alone and optimizes expected correctness under a mean candidate-verification budget. **Budget fidelity** requires every local normalization, reliability estimate, rank, and support statistic at depth `m` to use only the first `m` verified candidates; the global top-100 retrieval evidence remains available.

For each query, EntropyBudget min–max normalizes its top-100 retrieval distances, maps a constant distance vector to `0.5`, and applies a softmax to the negated normalized distances using a fixed temperature **`T = 0.10`**. The uncertainty score is the Shannon entropy of that distribution divided by `log(100)`. Queries are ranked by entropy **within each unlabeled workload** (dataset or condition). The **most uncertain 40%** receive geometric verification at depth **50**; the remaining **60%** retain global retrieval ranking at depth **0**. The nominal mean depth is **20 verified candidates per query**, matching uniform depth 20. Stable query ordering resolves ties; actual realized costs are used in evaluation.

The primary rule is a workload-relative **batch** policy. A **past-only streaming** variant uses 50 warm-up queries at depth 20 and subsequently compares each incoming query's entropy with the empirical 60th percentile of *previously observed* entropies. Its realized cost depends on arrival order.

For verified candidates, the fixed spatial rank-fusion scorer combines global similarity, SuperPoint–LightGlue matching evidence, MAGSAC inliers, inlier spatial support, adaptive global/local reliability, six ranking vectors, reciprocal-rank fusion (offset 60), and top-rank support (cutoffs 5 and 10). All local-dependent operations are restricted to the verified head. For depths 0 or 1, retrieval order is returned. The same scoring rule is used across allocation policies.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/` | Research implementation and GPU entrypoints; five protected files retain recorded source fingerprints |
| `configs/` | Recorded experimental configuration values |
| `tests/` | CPU source, scorer, geometry, and execution-path checks |
| `scripts/` | Timing, precomputed/live retrieval, checksum, and compatibility-path utilities |
| `analysis/calibration_and_utility/` | Utility decomposition, sensitivity analysis scripts, input tables, and generated figures |
| `evidence/` | Stored per-query decisions, condition-level outputs, streaming, and scorer-ablation evidence |
| `profiling/a100/` | A100 stage times, per-query service records, environment details, and summary tables |
| `retrieval_replay_inputs/` | Precomputed/live retrieval comparison records and checksums |
| `validation_inputs/`, `validation_results/` | Budget-fidelity and spatial-scoring validation inputs and outcomes |
| `notebooks/`, `docs/` | GPU launch template, timing scope, dependency provenance, and reproduction guidance |
| `source_fingerprints.json`, `SHA256_FILE_INVENTORY.json` | Two source hash sets and released-file SHA-256 inventory |

## GPU execution

A clean GPU run requires PyTorch/CUDA, FAISS, OpenCV, SuperPoint–LightGlue, the official global retriever implementations and checkpoints, and the original benchmark images. The A100 profiling record identifies **PyTorch 2.11.0+cu130, CUDA 13.0, FAISS 1.15.1, NumPy 2.4.6, and OpenCV 4.14.0**. The recorded dependency versions and the scope of the partial environment record are documented in `requirements_recorded_scope.md`, `requirements_unlocked.txt`, and `docs/dependencies.md`.

## Data access

No benchmark images, full third-party datasets, or pretrained model weights are redistributed. Obtain datasets under their providers' terms and reproduce the query/gallery and ground-truth protocols documented in the study and configuration files:

| Evaluation workload | Positive definition / protocol |
| --- | --- |
| SVOX, SVOX Night, SVOX Rain | 25 m radius |
| AmsterTime, SPED | Native exact-pair correspondence |
| San Francisco (partial gallery) | 50 m radius; 154,890-image gallery and 233 evaluable queries |
| St Lucia, Nordland, Baidu | External evaluation with a frozen 25 m positive radius |

Dataset provenance is documented by the original benchmark papers cited in the study: SVOX (Berton et al., 2021), AmsterTime (Yildiz et al., 2022), SPED (Chen et al., 2017), San Francisco (Chen et al., 2011), St Lucia (Milford and Wyeth, 2008), Nordland (Sünderhauf et al., 2013), and Baidu (Sun et al., 2017). Access the original providers' releases and follow their licenses. The included stored-evidence tests operate on the supplied derived records and **do not require downloading the image datasets**.

## Models and dependency provenance

The study uses MegaLoc for allocator development, with **SALAD** and **BoQ-DINOv2** as confirmatory retrievers on the same physical queries. The primary local backend is **SuperPoint–LightGlue** with MAGSAC; the detector-transfer study uses **ALIKED–LightGlue**. The retrievers are used in their officially released pretrained form, without fine-tuning.

| External source | Entrypoint or recorded revision | Licensing / provenance |
| --- | --- | --- |
| [BoQ](https://github.com/amaralibey/Bag-of-Queries) | `get_trained_boq(backbone_name='dinov2', output_dim=12288)`; A100 source checkout `1a4965ea7dfd9bd0dd846adf7a0e430f68101d12` | MIT; the original historical revision is **not proven** by the profiling checkout alone |
| [SALAD](https://github.com/serizba/salad) | `dinov2_salad` via the official source | GPL-3.0; a source revision is not recorded here |
| [LightGlue](https://github.com/cvg/LightGlue) | A100 source checkout `eb42fee2d71449efb0aa5c10549752b5d75384d8` | Apache-2.0 |
| [MegaLoc](https://github.com/gmberton/MegaLoc) | Development retriever | MIT; use the original provider release |

Recorded A100 model-weight **SHA-256** values (the weight files themselves are not supplied):

| Checkpoint filename | SHA-256 |
| --- | --- |
| `dinov2_12288.pth` | `d72ee0ce899e2790be6cf6d1d57f75dfad0c76475bd2198e6dc0c3f0221b3583` |
| `dinov2_vitb14_pretrain.pth` | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |
| `superpoint_lightglue_v0-1_arxiv.pth` | `6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a` |
| `superpoint_v1.pth` | `52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e` |

The retained `src/boq_hub_recovery.py` loads **official upstream BoQ code** from a local checkout; it does not vendor an upstream model implementation. Code here imports official SALAD, BoQ, and LightGlue packages rather than embedding those projects. Original third-party license notices and license obligations remain with the upstream projects. **MIT licensing of this repository applies only to original EntropyBudget code**, not to upstream code, weights, or dataset content.

## Reproduce the stored evidence

The commands below validate the supplied records on CPU from the repository root. Check the release inventory on a clean checkout, then use a separate working copy to regenerate tables and figures.

| Command | Expected outcome |
| --- | --- |
| `python scripts/verify_release.py --check-inventory` | `PASS:` for five source fingerprints and released-file hashes |
| `python tests/smoke_authentic_subset.py` | Final JSON includes `"status": "PASS"` |
| `python tests/test_drivefs_manifest_and_local_staging.py` | Prints `PASS:` |
| `python scripts/reproduce_a100_table.py` | Reconstructs all **six** A100 policy summaries from 1,200 per-query records and prints `PASS:`; Nordland EntropyBudget mean **654.877 ms**, median **107.405 ms**, p95 **1568.048 ms** in the stored records |
| `python scripts/reproduce_compact_retrieval_audit.py` | Two 200-query precomputed/live retrieval comparisons and `PASS:` |
| `python tests/audit_authentic.py --package-root .` | JSON output with `"status": "PASS"` for stored scorer and budget-fidelity checks |
| `python analysis/calibration_and_utility/analysis/run_phase_a.py` | Reproduces the stored Phase A CSV tables |
| `python analysis/calibration_and_utility/analysis/run_phase_b.py` | Reproduces the stored utility and fraction–depth analysis CSV tables |
| `python analysis/calibration_and_utility/analysis/run_phase_b_temperature.py` | Reproduces temperature-sensitivity and ordering CSV tables |
| `python scripts/verify_release.py --check-tables` | `PASS:` for nine stored analysis CSVs, byte-for-byte |

These checks do not amount to fresh GPU inference. For image-level execution, obtain the original inputs and checkpoints under their respective licenses, then use `python scripts/prepare_execution_paths.py` to create hash-checked, untracked compatibility paths needed by the immutable runner. Inspect `python src/benchmark_e2e_resilient.py --help` for arguments and `docs/execution.md` for scope. The notebook in `notebooks/` never rewrites protected source files.

## Key results reported in the manuscript

The primary cohort comprises **4,600 physical queries** across six conditions. The allocation is frozen before matched SALAD and BoQ-DINOv2 confirmation. All Recall@1 values below are percentages (manuscript Table 3):

| Retriever | Retrieval only | Uniform depth 20 | Margin, cost 20 | EntropyBudget, cost 20 | Uniform depth 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SALAD | 87.35 | 89.24 | 89.24 | **89.65** | 89.59 |
| BoQ-DINOv2 | 89.00 | 89.85 | 90.61 | **90.72** | 90.52 |

Averaged within matched physical queries across retrievers, the frozen policy exceeds **uniform depth 20 by +0.64 percentage points (pp; 95% CI [+0.27, +1.02])**, exceeds equal-budget margin allocation by **+0.26 pp ([+0.01, +0.52])**, and differs from **uniform depth 50 by +0.13 pp ([−0.18, +0.45])** while using **60% fewer verifications**.

**External evaluation:** The three new datasets comprise **32,264 queries** (St Lucia 1,464; Nordland 27,592; Baidu 3,208). Entropy-based query ordering exceeds equal-budget margin allocation by **+1.52 pp ([+1.39, +1.64])**. Crucially, the unchanged depth map **underperforms uniform depth 20 by −0.31 pp ([−0.45, −0.16])** and differs from uniform depth 50 by **−0.74 pp** (one-sided 95% lower bound **−0.845 pp**). The two prespecified external criteria, H1 and H2, are **not supported**. The negative equal-budget contrast concentrates on Baidu, where verification benefits queries across the uncertainty spectrum (SALAD **−2.49 pp**, BoQ-DINOv2 **−2.52 pp** versus uniform depth 20). These results distinguish transfer of the *ordering* from calibration of the fixed depth map to each workload (manuscript Sections 5.7–5.9).

## Timing scope and A100 service profile

Reported **query service time** sums image decoding and global descriptor extraction, top-100 search, and local processing through final top-1 scoring. Timed stages include reads of staged query images and, at positive depth, query SuperPoint extraction, gallery-feature loading and transfer, LightGlue matching, MAGSAC estimation, and final scoring. Offline gallery and index preparation, input staging, checkpoint I/O, batch waiting, and entropy and allocation overhead fall outside the sum. The retrieval prepass wall time and allocation overhead are reported separately. Uninterrupted image-to-ranking latency and sustained throughput remain unmeasured (manuscript Section 4.5).

| Dataset, 200-query A100 sample | Policy | Mean depth | Mean service ms | Median ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Nordland | Uniform depth 20 | 20 | 676.1 | 677.7 | 746.7 |
| Nordland | **EntropyBudget** | 20 | **654.9** | **107.4** | **1568.0** |
| Nordland | Uniform depth 50 | 50 | 1518.2 | 1517.3 | 1748.4 |
| Baidu | Uniform depth 20 | 20 | 634.7 | 638.0 | 704.5 |
| Baidu | **EntropyBudget** | 20 | **612.6** | **50.4** | **1611.6** |
| Baidu | Uniform depth 50 | 50 | 1446.0 | 1444.1 | 1620.2 |

Against uniform depth 50, mean A100 service demand decreases by **56.9%** on Nordland and **57.6%** on Baidu. At equal mean depth, the verification-heavy branch increases p95 service time to **2.10×** and **2.29×** that of uniform depth 20. Values above are exactly the rounded values in manuscript Table 9; the reconstruction script checks unrounded stored records. See `docs/timing_scope.md`.

## Integrity and licensing

The five protected source files retain the released SHA-256 values in `source_fingerprints.json`. **Four** byte-match their recorded executed counterparts; `src/benchmark_e2e_resilient.py` differs from the executed source only in a command-line help string, with no change to computation or numerical settings. Renamed protected files have an explicit source-to-release path mapping. Every released file is included in `SHA256_FILE_INVENTORY.json`, excluding the inventory itself.

Original code: [MIT License](LICENSE), copyright © 2026 The Authors. Third-party repositories, datasets, and pretrained weights have independent copyright and licensing terms.

## Citation

The software citation and author affiliations are in [CITATION.cff](CITATION.cff).
