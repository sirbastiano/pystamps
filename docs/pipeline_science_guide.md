# pySTAMPS Pipeline and Science Guide

This guide teaches the two things a new user needs at the same time:

- the scientific meaning of the StaMPS-style persistent-scatterer workflow
- the practical pySTAMPS commands used to inspect, run, accelerate, and verify that workflow

If you only need commands, start with [../howtorun.md](../howtorun.md) or [quickstart.html](quickstart.html). If you need to understand why the commands exist and what each stage produces, read this guide first.

## What pySTAMPS is

pySTAMPS is a Python-first runtime for StaMPS-style InSAR processing. It works on a dataset directory, discovers patch folders and stage artifacts, runs selected stages from 1 to 8, and writes new `.mat` products back into that same dataset tree.

The project has three roles:

- **Program role:** provide a CLI, Python API, config model, runtime scheduler, kernel registry, and verification tools.
- **Science role:** reproduce the persistent-scatterer processing stages used by legacy StaMPS-style workflows.
- **Migration role:** make parity against trusted MATLAB/StaMPS outputs explicit through golden datasets, audit manifests, and numerical tolerances.

The most important operational rule is simple: **run on a copy of your dataset**. pySTAMPS writes stage outputs in place.

## Minimal science background

### SAR observations

Synthetic Aperture Radar satellites observe the same area repeatedly. Each acquisition records complex radar values. The phase part of those values is sensitive to geometry, atmosphere, motion, topography, and noise.

### Interferograms

An interferogram compares two radar acquisitions. In StaMPS-style single-master processing, many slave acquisitions are compared to one master acquisition. In small-baseline processing, pairs can be arranged differently. pySTAMPS carries both timing and baseline metadata through the pipeline so later stages know how each interferogram relates to the acquisition stack.

### Wrapped phase

Radar phase is periodic. Values repeat every `2*pi`, so a raw phase stack contains wrapped measurements. Wrapped phase is useful, but later products need an unwrapped phase estimate that is continuous enough for time-series analysis.

### Persistent scatterers

A persistent scatterer is a point that remains stable across many radar acquisitions. Buildings, exposed rocks, and other stable reflectors are typical examples. The early pySTAMPS stages organize candidate points, estimate quality terms, select candidates, and remove noisy or redundant points.

### Coherence and gamma-like quality

Coherence is a practical reliability signal. High-coherence candidates are more stable and are usually better inputs for later phase analysis. Stage 2 estimates per-candidate model and coherence terms used by selection and weeding.

### Baseline, topographic error, and look-angle error

Perpendicular baseline describes the imaging geometry between acquisitions. Baseline-dependent phase terms can be mistaken for displacement if they are not estimated and corrected. Later stages estimate correction products such as SCLA, the spatially correlated look-angle error term.

### Unwrapping and filtering

Stage 6 unwraps merged phase products. Stage 7 estimates slow correction terms. Stage 8 applies final space-time filtering and writes the final filtered products used for interpretation and comparison.

## The dataset mental model

A pySTAMPS run points at one dataset root.

```text
DATASET/
  patch.list                 # optional explicit patch order
  PATCH_1/
    ps1.mat
    ph1.mat
    pm1.mat
    select1.mat
    weed1.mat
  PATCH_2/
  diff0/
  geo/
  rslc/
  ps2.mat
  ph2.mat
  phuw2.mat
  scla2.mat
  uw_space_time.mat
```

The exact contents depend on where the dataset is in the workflow. pySTAMPS can start at a later stage when the required earlier artifacts already exist.

Use this command to see what pySTAMPS discovers:

```bash
uv run pystamps status --dataset DATASET
```

The status output reports:

- the inferred merged stage for dataset-level products
- each discovered patch and its inferred patch stage
- whether the dataset looks ready for the selected stage range

## Artifact-driven execution

pySTAMPS is artifact driven. Each stage has an expected output artifact. If that artifact, or the stage bundle for merged stages, already exists, the pipeline reports `skipped_existing` instead of recomputing the stage.

This is intentional and useful:

- completed datasets can be inspected without rerunning expensive work
- failed runs can be resumed from the missing stage range
- benchmark runs can be isolated to a copy that actually needs the target outputs

It also means a copied golden dataset may not execute kernels if all expected outputs are already present. For speed tests, use `make benchmark`, the direct kernel API, or a dataset copy with the target outputs absent.

## Install pySTAMPS

Recommended local setup:

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
uv sync
uv run pystamps describe-backends
```

Editable `pip` setup:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

Source and editable installs compile the native Rust/CPU extension. Install a Rust toolchain first and confirm:

```bash
cargo --version
```

Optional GPU support uses the `gpu` extra:

```bash
python -m pip install -e ".[gpu]"
```

Some full workflows also require external executables:

- `triangle`
- `snaphu`

You can override their paths in config:

```yaml
tools:
  triangle: /usr/local/bin/triangle
  snaphu: /usr/local/bin/snaphu
```

## First end-to-end run

Use a copy, not your only original dataset:

```bash
cp -a /path/to/source_dataset /tmp/pystamps_run
uv run pystamps status --dataset /tmp/pystamps_run
uv run pystamps run --dataset /tmp/pystamps_run --start-step 1 --end-step 8 --dry-run
uv run pystamps run --dataset /tmp/pystamps_run --start-step 1 --end-step 8
```

If you are using the repo diagnostic data, a smaller late-stage run is often more practical:

```bash
cp -a inputs_and_outputs/InSAR_dataset_test_stage8diag /tmp/pystamps_stage8diag_run
uv run pystamps status --dataset /tmp/pystamps_stage8diag_run
uv run pystamps run --dataset /tmp/pystamps_stage8diag_run --start-step 6 --end-step 8
```

If the output says `skipped_existing`, the requested stage already has its expected artifacts in that copied dataset.

## CLI command map

| Command | Purpose | Typical use |
| --- | --- | --- |
| `status` | Inspect dataset and inferred progress | First command on any dataset |
| `run` | Execute or dry-run stages | Normal processing |
| `verify` | Compare a run tree against a golden tree | Trust but verify |
| `describe-inputs` | Print logical input contracts for stages | Learning and debugging |
| `describe-backends` | Print kernel/backend availability | Backend setup and speed work |
| `list-legacy` | List StaMPS legacy scripts under a checkout | Migration/debugging support |

Useful examples:

```bash
uv run pystamps describe-inputs --stage all
uv run pystamps describe-inputs --stage 1 --dataset DATASET --patch PATCH_1
uv run pystamps describe-backends
```

## Stage-by-stage science and outputs

| Stage | Scope | Science question | Main inputs | Main outputs |
| --- | --- | --- | --- | --- |
| 1 | patch | What candidate points and metadata are available? | candidate indices, complex phase stack, lon/lat, width, length, days, master day, baseline | `ps1.mat`, `ph1.mat`, `bp1.mat`, `da1.mat`, `hgt1.mat`, `la1.mat` |
| 2 | patch | How well does each candidate fit the phase model? | `ps1.mat`, `ph1.mat`, `bp1.mat` | `pm1.mat` |
| 3 | patch | Which candidates are good persistent-scatterer candidates? | `pm1.mat`, `ps1.mat` | `select1.mat` |
| 4 | patch | Which selected points are noisy or redundant? | `select1.mat`, stage-1 and stage-2 quality products | `weed1.mat` |
| 5 | patch and merged | How do patch results become one dataset view? | retained patch candidates | patch `ps2.mat`/`ph2.mat` and merged `ps2.mat`, `ph2.mat`, `ifgstd2.mat` |
| 6 | merged | What is the unwrapped phase estimate? | merged phase, geometry, graph-support arrays | `phuw2.mat`, `uw_phaseuw.mat`, `uw_grid.mat`, `uw_interp.mat` |
| 7 | merged | What slow correction terms should be estimated? | unwrapped phase products and merged geometry | `scla2.mat`, `scla_smooth2.mat` |
| 8 | merged | What are the final filtered space-time products? | stage-7 correction products and unwrapped phase | `mean_v.mat`, `uw_space_time.mat` |

Patch stages run once per `PATCH_*` directory. Merged stages run once at the dataset root. Stage 5 is special because it completes patch promotion and also writes merged products when the selected range includes stage 5.

## Stage 1: load and organize candidates

Stage 1 prepares raw candidate data for the rest of the workflow. It turns exported StaMPS/SNAP2StaMPS-style inputs into structured `.mat` artifacts that later stages can read consistently.

Typical logical inputs:

- candidate index array `ij`
- complex phase stack `ph`
- candidate `lonlat`
- patch raster `width` and `length`
- acquisition `day` and `master_day`
- perpendicular baseline `bperp`
- optional stability metric `D_A`
- optional height prior `hgt`

Stage 1 writes `ps1.mat`, `ph1.mat`, `bp1.mat`, and side products. If these files exist in a patch, pySTAMPS considers that patch ready for stage 2.

## Stage 2: estimate phase model and coherence

Stage 2 estimates phase-model and coherence-like terms per candidate. These terms are the bridge between raw candidate stacks and persistent-scatterer selection.

The optimized native kernels are especially important here. Stage 2 has native implementations for:

- `stage2_grid_accumulate`
- `stage2_histogram`
- `stage2_topofit`
- `stage2_topofit_row_invariant`
- `stage2_topofit_coh_row_invariant`

Use the Python backend for reference behavior and the native backend for compiled Rust/CPU execution.

## Stage 3: select persistent scatterers

Stage 3 uses stage-2 model and coherence outputs to choose candidate points worth keeping. It writes `select1.mat`, which records selected candidates and keep/reject decisions used by later patch stages.

## Stage 4: weed noisy or redundant candidates

Stage 4 removes poor or redundant selections. It uses candidate geometry and quality metrics to avoid carrying unstable points into the merged products.

The `stage4_edge_stats` kernel supports Python and native backends. Where registered and available, the native backend is the optimized Rust/CPU path.

## Stage 5: promote and merge patches

Stage 5 is the transition from patch-local processing to dataset-level processing. It writes per-patch `ps2.mat`, `ph2.mat`, `pm2.mat`, `bp2.mat`, and related products. It also creates merged root-level products such as `ps2.mat`, `ph2.mat`, and `ifgstd2.mat`.

After stage 5, later stages work at the merged dataset root rather than independently inside each patch.

## Stage 6: unwrap merged phase

Stage 6 produces unwrapped phase products. It can call external tools such as `triangle` and `snaphu`, depending on the dataset and execution path.

Important outputs include:

- `phuw2.mat`
- `uw_phaseuw.mat`
- `uw_grid.mat`
- `uw_interp.mat`

If stage 6 fails, check external tool availability, merged stage-5 products, and whether the dataset copy has enough disk space for generated artifacts.

## Stage 7: estimate SCLA and corrections

Stage 7 estimates SCLA and related correction terms from unwrapped phase and merged geometry.

The `stage7_scla` kernel supports Python, native, and CUDA providers where registered. Use native for the compiled Rust/CPU implementation and CUDA only when CuPy and that backend are available.

Stage 7 writes:

- `scla2.mat`
- `scla_smooth2.mat`

## Stage 8: final space-time filtering

Stage 8 applies the final space-time filtering and writes the products usually used for final inspection and parity comparisons.

The `stage8_edge_noise` kernel supports Python, native, and CUDA providers where registered.

Stage 8 writes:

- `mean_v.mat`
- `uw_space_time.mat`

## Switching kernel modality

The CLI command stays the same. You switch modality through config.

Reference Python config:

```yaml
runtime:
  backend: auto
  stage2_kernel_backend: python
  kernel_backend_overrides:
    stage2_grid_accumulate: python
    stage2_histogram: python
    stage2_topofit: python
    stage2_topofit_row_invariant: python
    stage2_topofit_coh_row_invariant: python
    stage4_edge_stats: python
    stage7_scla: python
    stage8_edge_noise: python
```

Optimized native Rust/CPU config:

```yaml
runtime:
  backend: auto
  stage2_kernel_backend: native
  stage2_native_threads: 0
  kernel_backend_overrides:
    stage2_grid_accumulate: native
    stage2_histogram: native
    stage2_topofit: native
    stage2_topofit_row_invariant: native
    stage2_topofit_coh_row_invariant: native
    stage4_edge_stats: native
    stage7_scla: native
    stage8_edge_noise: native
  io_workers: 8
  cpu_workers: 0
  stage7_chunk_ps: 100000
  stage8_chunk_edges: 200000
```

Run with the config:

```bash
uv run pystamps --config native-kernels.yaml run \
  --dataset /tmp/pystamps_run \
  --start-step 2 --end-step 8
```

Inspect actual backend availability first:

```bash
uv run pystamps describe-backends
```

`native` can only run when the compiled extension exports that kernel. `cuda` can only run where CuPy and the CUDA kernel are available.

## Python API examples

The CLI is the recommended path for users. The Python API is useful for notebooks, automation, and kernel development.

Inspect dataset status:

```python
from pystamps.status import collect_status

status = collect_status("/tmp/pystamps_run")
print(status.merged_stage)
for patch in status.patch_statuses:
    print(patch.patch, patch.stage)
```

Run a pipeline range:

```python
from pathlib import Path

from pystamps.config import RunConfig
from pystamps.pipeline.stages import run_pipeline
from pystamps.pipeline.types import PipelineContext

context = PipelineContext(
    dataset_root=Path("/tmp/pystamps_run"),
    run_config=RunConfig(),
    start_step=6,
    end_step=8,
    dry_run=False,
)
report = run_pipeline(context)
for result in report.results:
    print(result.stage_id, result.scope, result.target, result.status)
```

Call one kernel directly:

```python
import numpy as np

from pystamps.kernels import run_stage8_edge_noise_kernel

out = run_stage8_edge_noise_kernel(
    np.ones((100, 4), dtype=np.complex64),
    np.array([0, 1, 2], dtype=np.int64),
    np.array([1, 2, 3], dtype=np.int64),
    backend="native",
)
print(out["dph_noise"].shape)
```

## Verification and parity

Verification answers a practical question: does this run match a golden dataset within the configured tolerance?

```bash
uv run pystamps verify \
  --run /tmp/pystamps_run \
  --golden inputs_and_outputs/InSAR_dataset_test_stage8diag
```

The repo-level parity audit is broader:

```bash
make audit
```

The audited dataset list is maintained in:

```text
pystamps/data/audited_workflow_manifest.json
```

Do not replace the audit with an older two-dataset command. The manifest is the contract for full validation.

The oracle precedence is documented in:

```text
pystamps/data/oracle_contract.json
```

In practical terms, pySTAMPS compares against trusted wrapper/MATLAB/manual reference outputs and records which oracle source was used. Use audit artifacts for parity claims, not notebook screenshots alone.

## Benchmarking speed

Use repeatable benchmarks for speed claims:

```bash
make benchmark
```

Or customize the benchmark:

```bash
uv run python scripts/benchmark_backends.py \
  --dataset inputs_and_outputs/InSAR_dataset_test_stage8diag \
  --start-step 1 --end-step 8 \
  --repeat 3 --warmup 1
```

Benchmark outputs are written under:

```text
inputs_and_outputs/benchmarks/
```

Notebook cell timings are useful for teaching, but benchmark JSON/CSV files are the stronger evidence because they use repeated runs and a consistent command surface.

## Choosing stage ranges

Use the narrowest range that matches your goal.

| Goal | Command pattern |
| --- | --- |
| Learn dataset state | `uv run pystamps status --dataset DATASET` |
| Rehearse without writing | `uv run pystamps run --dataset DATASET --start-step 1 --end-step 8 --dry-run` |
| Continue after stage 5 | `uv run pystamps run --dataset DATASET --start-step 6 --end-step 8` |
| Recompute selection after stage 2 | `uv run pystamps run --dataset DATASET --start-step 3 --end-step 5` |
| Benchmark native late-stage kernels | `make benchmark` or direct kernel API |
| Compare with a reference | `uv run pystamps verify --run RUN --golden GOLDEN` |

## Troubleshooting

### The stage was skipped

`skipped_existing` means the expected output artifact already exists. Use a fresh run copy that still needs the target stage if you want execution rather than inspection.

### Native backend is unavailable

Run:

```bash
uv run pystamps describe-backends
```

Then confirm the editable install was built with Rust available:

```bash
cargo --version
uv sync
```

If you installed from source without Rust, rebuild after installing the toolchain.

### Stage 6 or stage 8 fails around unwrapping

Check `triangle` and `snaphu`:

```bash
which triangle
which snaphu
```

If they are installed in non-standard paths, set them in config under `tools`.

### Verification fails

Check that `--run` and `--golden` point to comparable dataset states. A stage-6-only run should not be expected to match a fully post-processed stage-8 golden tree for files it did not produce.

### Full audit is slow

That is expected. The audit runs every dataset in the maintained manifest and may perform expensive late-stage processing. Use targeted pytest tests and `verify` during development, then `make audit` for release-quality parity evidence.

## Recommended learning path

1. Read this guide once for the science and program model.
2. Run `uv run pystamps describe-backends`.
3. Run `uv run pystamps status --dataset DATASET` on your dataset copy.
4. Run a dry-run for your target stage range.
5. Execute the stage range on a copy.
6. Verify against a golden dataset when one exists.
7. Use `make benchmark` before making speed claims.
8. Use `make audit` before making broad parity claims.
