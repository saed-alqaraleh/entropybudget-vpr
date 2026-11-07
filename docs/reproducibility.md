# Reproducibility scope

All verification commands start from the repository root. Install `requirements_cpu_verified.txt` for the CPU checks. Verify the inventory on a clean checkout before regenerating analysis outputs in a separate working copy.

| Command | Expected outcome |
| --- | --- |
| `python scripts/verify_release.py --check-inventory` | Five source fingerprints and all release file hashes match |
| `python tests/smoke_authentic_subset.py` | JSON containing `"status": "PASS"` |
| `python tests/test_drivefs_manifest_and_local_staging.py` | Line beginning `PASS:` |
| `python scripts/reproduce_a100_table.py` | Six reconstructed rows, then `PASS:` |
| `python scripts/reproduce_compact_retrieval_audit.py` | Two dataset rows, then `PASS:` |
| `python tests/audit_authentic.py --package-root .` | Final JSON containing `"status": "PASS"` |
| Three scripts under `analysis/calibration_and_utility/analysis/`, then `python scripts/verify_release.py --check-tables` | Nine regenerated CSVs match the stored SHA-256 fingerprints |

For independent GPU execution, consult `docs/dependencies.md`. Stored-evidence checks are not a substitute for fresh model execution. The original benchmark datasets and pretrained weights are excluded from this repository.

GitHub Actions runs these checks with Python 3.12 and the pinned CPU dependencies. The temperature check allows an absolute difference of `1e-12` in recomputed entropy to account for CPU rounding, and requires the archived query ordering at `T = 0.10` to match exactly.
