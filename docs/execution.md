# GPU entrypoint

The immutable benchmark entrypoint is `src/benchmark_e2e_resilient.py`. Use `--help` to list options and supply original external data inputs and model-weight checkpoints before GPU work. The `notebooks/gpu_execution_template.ipynb` file only inspects launch arguments; it does not modify source files or reproduce the GPU experiment by itself. The precomputed/live retrieval audit is available in `scripts/audit_precomputed_live_retrieval.py`.

Before image-level GPU execution, call `python scripts/prepare_execution_paths.py` to construct and verify temporary paths referenced by the immutable source. These compatibility copies are never committed.
