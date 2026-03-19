# Architecture Snapshot

## Runtime

- IO/orchestration tasks run in a `ThreadPoolExecutor`.
- CPU-heavy tasks can run in a `ProcessPoolExecutor`.
- This matches the migration target of hybrid concurrency and Python 3.14t readiness.

## Pipeline

- Stage definitions mirror StaMPS stages 1-8.
- Patch-scoped stages: 1-5.
- Merged-scoped stages: 6-8.
- Stages 1-8 are implemented in `pystamps.pipeline.ported` with Python-native logic.
- Stage 5 includes patch promotion and merged aggregation (`ps2/ph2/pm2`, `ifgstd2`).
- Stages 6-8 currently use migration-baseline numerical methods and are not yet parity-equivalent to legacy StaMPS on full datasets.
- Each stage has an expected artifact map used for progress/status and execution dispatch.

## Output checks

- Comparison and parity checks are documented in the dedicated verification guide.
