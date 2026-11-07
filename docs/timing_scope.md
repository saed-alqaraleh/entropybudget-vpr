# Timing scope and measured service demand

The manuscript's query-service profile measures **stage-summed service demand** for BoQ-DINOv2 and SuperPoint–LightGlue on an NVIDIA A100-SXM4-40GB GPU. On each of two fixed 200-query cohorts (Nordland and Baidu), the policy verifies 80 queries at depth 50 and serves the other 120 at retrieval-only depth 0; uniform comparators use depths 20 and 50. Each dataset uses 20 warm-up queries followed by one measured pass per policy in fixed order.

For every measured query, service time is the sum of: (1) image decoding plus global descriptor extraction; (2) top-100 nearest-neighbor search; and (3) the local path through final scoring. At positive depth, the third stage includes query SuperPoint extraction, gallery local-feature load/transfer, LightGlue matching, MAGSAC geometry estimation, and budget-faithful scoring. Reported per-query means, medians, and p95 values are computed from these stage sums.

**Outside** the summed service boundary: gallery-descriptor and exact FAISS L2 index preparation, local input staging and storage I/O, batch waiting, global retrieval prepass, entropy computation, and workload allocation. Retrieval prepass and allocation overhead are reported separately. These measurements characterize query service demand rather than full deployment throughput or sustained-load wall-clock response time.

The timing reconstruction reads `profiling/a100/{nordland,baidu}/e2e_raw_query_timings.csv` and verifies all six summary rows. See manuscript Sections 4.5 and 5.10 and Table 9 for the reported, rounded values.
