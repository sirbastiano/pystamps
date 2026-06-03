# pySTAMPS Stages and Code Paths

This page maps the numbered pySTAMPS stages to the files, artifacts, Python entrypoints, and Rust entrypoints that implement them.

![pySTAMPS stage map](assets/pystamps-stage-map.svg)

## Runtime Flow

The normal execution path is:

1. `pystamps.cli` parses `run`, loads configuration, and creates a `PipelineContext`.
2. `pystamps.io.dataset.discover_dataset()` resolves the dataset root and orders patch folders.
3. `pystamps.pipeline.stages.run_pipeline()` selects stage definitions for the requested range.
4. Patch stages run once per discovered `PATCH_*` directory.
5. Merged stages run once at the dataset root.
6. Expected artifacts decide whether a result is `planned`, `completed`, `skipped_existing`, `skipped`, or `failed`.

## Files To Read First

| File | Why it matters |
| --- | --- |
| `pystamps/pipeline/stages.py` | Stage definitions, artifact skips, patch/merged dispatch, strict reference replay, and result timing |
| `pystamps/pipeline/ported.py` | Python stage entrypoints and scientific helpers for stages 1 through 8 |
| `pystamps/io/dataset.py` | Dataset discovery plus expected artifact mapping used by status and scheduling |
| `pystamps/kernels/accelerated.py` | Backend dispatch for hot kernels in stages 2, 4, 7, and 8 |
| `crates/pystamps-core/src/bin/pystamps-native.rs` | Standalone Rust CLI argument parsing and native stage dispatch |
| `crates/pystamps-stages/src/lib.rs` | Native readiness inventory and parity certification details for each stage scope |

## Stage Summary

| Stage | Scope | Intent | Python entrypoint | Rust entrypoint | Main artifacts |
| --- | --- | --- | --- | --- | --- |
| 1 | patch | Prepare candidates and early metadata | `stage1_load_initial()` | `native_stage1::run_stage1_native` | `ps1.mat`, `ph1.mat`, `bp1.mat`, `da1.mat`, `hgt1.mat`, `la1.mat` |
| 2 | patch | Estimate gamma/coherence-like model terms | `stage2_estimate_gamma()` | `native_stage2::run_stage2_native` | `pm1.mat` |
| 3 | patch | Select persistent-scatterer candidates | `stage3_select_ps()` | `native_stage3::run_stage3_native` | `select1.mat` |
| 4 | patch | Weed weak or redundant selected candidates | `stage4_weed_ps()` | `native_stage4::run_stage4_native` | `weed1.mat` |
| 5 | patch and merged | Promote patch outputs and merge dataset products | `stage5_correct_and_promote()`, `stage5_merge_and_ifgstd()` | `native_stage5::run_stage5_patch_native`, `native_stage5::run_stage5_merge_native` | patch `ps2.mat`/`ph2.mat`; root `ifgstd2.mat` |
| 6 | merged | Unwrap merged phase | `stage6_unwrap()` | `native_stage6::run_stage6_native` | `phuw2.mat`, `uw_phaseuw.mat`, `uw_grid.mat`, `uw_interp.mat` |
| 7 | merged | Estimate SCLA correction terms | `stage7_calc_scla()` | `native_stage7::run_stage7_native` | `scla2.mat`, `scla_smooth2.mat` |
| 8 | merged | Apply final space-time filtering | `stage8_filter_scn()` | `native_stage8::run_stage8_native` | `mean_v.mat`, `uw_space_time.mat` |

## Direct Native Stage Commands

Patch stages use `--patch`; merged stages use `--dataset`.

```bash
target/release/pystamps-native stage 1 --patch "$RUN_DATASET/PATCH_1"
target/release/pystamps-native stage 2 --patch "$RUN_DATASET/PATCH_1"
target/release/pystamps-native stage 5 --patch "$RUN_DATASET/PATCH_1"
target/release/pystamps-native stage 5 --dataset "$RUN_DATASET"
target/release/pystamps-native stage 8 --dataset "$RUN_DATASET"
```

## Python API Example

```python
from pathlib import Path

from pystamps.config import RunConfig
from pystamps.pipeline.stages import run_pipeline
from pystamps.pipeline.types import PipelineContext

context = PipelineContext(
    dataset_root=Path("/path/to/run_dataset"),
    run_config=RunConfig(),
    start_step=1,
    end_step=8,
    dry_run=True,
)
report = run_pipeline(context)
```

## Optimized Kernels

Stage 2 owns the most kernel-heavy path:

- `stage2_grid_accumulate`
- `stage2_histogram`
- `stage2_topofit`
- `stage2_topofit_row_invariant`
- `stage2_topofit_coh_row_invariant`

Other optimized kernels:

- `stage4_edge_stats`
- `stage7_scla`
- `stage8_edge_noise`

Use `uv run pystamps describe-backends` to inspect local availability.
