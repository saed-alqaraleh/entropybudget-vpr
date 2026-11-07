# Reproducibility scope

All verification commands start from the repository root. No compatibility copies are needed for the CPU checks. Install `requirements_cpu_verified.txt` before using the numerical evidence checkers.

| Command | Expected outcome |
| --- | --- |
| `python tests/smoke_authentic_subset.py` | JSON containing `"status": "PASS"` |
| `python tests/test_drivefs_manifest_and_local_staging.py` | Line beginning `PASS:` |
| `python scripts/reproduce_a100_table.py` | Six reconstructed rows, then `PASS:` |
| `python scripts/reproduce_compact_retrieval_audit.py` | Two dataset rows, then `PASS:` |
| `python tests/audit_authentic.py --package-root .` | Final JSON containing `"status": "PASS"` |
| Three scripts under `analysis/calibration_and_utility/analysis/` | Outputs match stored input-derived CSVs by SHA-256 after execution |

For independent GPU execution, consult `docs/dependencies.md`. Stored-evidence checks are not a substitute for fresh model execution. The original benchmark datasets and pretrained weights are excluded from this repository.

The CPU checks were also run under Python 3.13.5 with the installed numerical library versions. The pinned CPU requirement file represents the earlier verified Python 3.12.14 environment; a GitHub Actions run remains necessary to verify the hosted environment.
