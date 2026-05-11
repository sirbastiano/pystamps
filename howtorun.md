# How To Run pySTAMPS

This guide explains how to run pySTAMPS if you are new both to the package and to interferometry.

If you want a slower, more tutorial-style introduction first, read [docs/pipeline_science_guide.md](docs/pipeline_science_guide.md), then [docs/getting_started.md](docs/getting_started.md), and then open `notebooks/00_pystamps_beginner_walkthrough.ipynb`.

## What pySTAMPS is doing

pySTAMPS works on a dataset directory. You point the tool at that directory, pySTAMPS inspects what files exist, then it runs one or more processing stages and writes results back into that same directory.

You can think of it like this:
- the dataset directory is your project folder
- stage files are intermediate checkpoints
- later stages depend on outputs from earlier stages
- verification compares your run folder with a reference folder that you trust

You do not need to understand radar physics to start using the tool. For a first run, the main things to know are:
- run on a copy of your data
- check the dataset layout first
- start with `status`, then `run`, then `verify`

## Before you start

You need:
- this repository checked out locally
- `uv` installed
- the Python environment synced with `uv sync` or `make setup`
- Rust installed when building from source or editable mode, because the optimized native kernels are compiled locally

External tools are also needed for some workflows:
- `triangle`
- `snaphu`

These tools are required for parts of the full processing flow, especially later stages and parity workflows.

Recommended setup:

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

## A few plain-language terms

- interferogram: a file that stores phase differences between radar observations taken at different times
- phase unwrap: the process of converting repeating wrapped phase values into a smoother continuous estimate
- coherence: a rough measure of how stable or reliable a signal is
- persistent scatterer: a point on the ground that stays stable enough across acquisitions to be useful for time-series analysis
- golden dataset: a reference output directory that you treat as the expected answer

That is enough vocabulary for operating pySTAMPS as a user.

## Your dataset folder

Your dataset directory must look like a StaMPS-style dataset root.

At minimum, pySTAMPS expects:
- a dataset root directory
- either a `patch.list` file or one or more `PATCH_*` directories

Depending on how early you start, the dataset also needs the stage inputs that those stages read.

Typical example datasets already in this repo are:
- `inputs_and_outputs/InSAR_dataset_test`
- `inputs_and_outputs/InSAR_dataset_test_stage8diag`

## The safest way to run

Run on a copy of your input dataset, not your only original.

Example:

```bash
cp -a /path/to/my_dataset /path/to/my_dataset_run
uv run pystamps status --dataset /path/to/my_dataset_run
uv run pystamps run --dataset /path/to/my_dataset_run --start-step 1 --end-step 8
```

Why this matters:
- pySTAMPS writes outputs into the dataset directory
- reruns are easier when you keep one untouched input copy
- comparing two run folders is easier when the original is still clean

For repo maintenance commands such as setup, test, implementation tests, build, audit, verify, and benchmark, you can also use the root `Makefile` targets (`make setup`, `make test`, `make test-impl`, `make build`, `make audit`, `make verify`, `make benchmark`).

## Step 1: inspect the dataset

Before running, inspect what pySTAMPS sees:

```bash
uv run pystamps status --dataset /path/to/dataset
```

This helps confirm that:
- the dataset root is valid
- patch folders were discovered
- pySTAMPS can infer the current stage of the dataset

## Step 2: preview a run

If you want to see what pySTAMPS would do without doing heavy work, use a dry-run:

```bash
uv run pystamps run --dataset /path/to/dataset --start-step 1 --end-step 8 --dry-run
```

This is useful when you are still learning the stage ranges.

## Step 3: run the pipeline

If your dataset has the required inputs for stages 1 through 8, run:

```bash
uv run pystamps run --dataset /path/to/dataset --start-step 1 --end-step 8
```

If you only want part of the pipeline, change the stage range.

Examples:

```bash
uv run pystamps run --dataset /path/to/dataset --start-step 2 --end-step 6
uv run pystamps run --dataset /path/to/dataset --start-step 6 --end-step 8
```

A practical interpretation:
- lower stage numbers are earlier preparation and selection steps
- higher stage numbers are later merged products and filtering steps
- partial runs are useful when you already have earlier artifacts and only want to continue from a later point
- if an expected output already exists, pySTAMPS reports `skipped_existing` for that stage instead of recomputing it

## Step 4: optional config file

If you want to control backend choice or chunk sizes, create a config file like this:

```yaml
runtime:
  backend: auto
  stage2_kernel_backend: auto
  stage2_native_threads: 0
  io_workers: 8
  cpu_workers: 0
  stage7_chunk_ps: 100000
  stage8_chunk_edges: 200000
  enable_mat_stage_cache: true
  stage2_checkpoint_mode: final
  stage2_checkpoint_interval: 1
```

Use `stage2_kernel_backend: native` to require the compiled stage-2 kernels, or `python` to keep the reference implementation even when the extension is installed. Leave `stage2_native_threads: 0` to let pySTAMPS give stage 2 the full detected CPU budget by default; it will then run stage-2 patches one at a time to avoid oversubscription. `cpu_workers: 0` now uses all detected CPU workers by default. Set a positive value to force a fixed OpenMP team size instead.

Then run:

```bash
uv run pystamps --config accel.yaml run --dataset /path/to/dataset --start-step 1 --end-step 8
```

## Step 5: run the optimized Rust/native kernels

The easiest way to see which optimized kernels are available is:

```bash
uv run pystamps describe-backends
```

The important backend names are:
- `python`: reference NumPy/Python implementation
- `native`: compiled Rust/CPU implementation
- `cuda`: CuPy implementation where that kernel and dependency are available

Create a native-kernel config:

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

Run it on a copy of the golden dataset:

```bash
cp -a inputs_and_outputs/InSAR_dataset_test_stage8diag /tmp/pystamps_native_demo
uv run pystamps --config native-kernels.yaml run \
  --dataset /tmp/pystamps_native_demo \
  --start-step 2 --end-step 8
```

If the copied dataset already contains the expected stage outputs, the CLI reports `skipped_existing` for those stages. That is correct. To exercise the optimized kernels through the pipeline, use a dataset copy that still needs those stage outputs. To exercise a kernel directly on the repo golden data, use the direct API example in the next step.

For debugging parity, switch any one kernel back to the reference implementation:

```yaml
runtime:
  stage2_kernel_backend: python
  kernel_backend_overrides:
    stage8_edge_noise: python
```

## Step 6: call an optimized kernel directly on repo data

Use the CLI for normal workflows. Use the direct kernel API when you are developing or benchmarking one numerical kernel in isolation.

This example loads real arrays from the golden stage-8 diagnostic dataset, selects a small valid edge subset, and runs the native stage-8 edge-noise kernel:

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np

from pystamps.io.mat import read_mat
from pystamps.kernels import run_stage8_edge_noise_kernel

root = Path("inputs_and_outputs/InSAR_dataset_test_stage8diag")
uw_grid = read_mat(root / "uw_grid.mat")
uw_interp = read_mat(root / "uw_interp.mat")

uw_ph = np.asarray(uw_grid["ph"][:1000, :8], dtype=np.complex64)
edges = np.asarray(uw_interp["edgs"], dtype=np.int64)
node_a = edges[:, 1] - 1
node_b = edges[:, 2] - 1
valid = (
    (node_a >= 0)
    & (node_b >= 0)
    & (node_a < uw_ph.shape[0])
    & (node_b < uw_ph.shape[0])
)

out = run_stage8_edge_noise_kernel(
    uw_ph,
    node_a[valid][:2000],
    node_b[valid][:2000],
    backend="native",
)
print(out["dph_noise"].shape)
print(out["dph_space_uw"].shape)
PY
```

Use `backend="python"` for reference behavior, `backend="native"` for Rust/CPU, and `backend="cuda"` when CuPy and that backend are available.

## Step 7: measure speed

To benchmark the maintained dataset:

```bash
make benchmark
```

Or customize the comparison:

```bash
uv run python scripts/benchmark_backends.py \
  --dataset inputs_and_outputs/InSAR_dataset_test_stage8diag \
  --start-step 6 --end-step 8 \
  --backends threads native \
  --repeat 3 --warmup 1
```

The benchmark writes JSON and CSV results under `inputs_and_outputs/benchmarks/`. Use those files for speed claims rather than eyeballing notebook cell runtimes.

## Step 8: understand what gets written

pySTAMPS writes stage outputs into the dataset.

Common merged outputs include:
- `ps2.mat`
- `ph2.mat`
- `pm2.mat`
- `ifgstd2.mat`
- `phuw2.mat`
- `uw_grid.mat`
- `uw_interp.mat`
- `scla2.mat`
- `mean_v.mat`
- `mv2.mat`
- `uw_space_time.mat`

You do not need to open these files manually for a first pass. In practice, you usually:
- run stages
- confirm the expected files were written
- compare the run against a reference dataset if one is available

## Step 9: verify against a reference dataset

Use:

```bash
uv run pystamps verify --run /path/to/run_dataset --golden /path/to/golden_dataset
```

Use `make verify` only for the repo's maintained reference-path check:
- run: `inputs_and_outputs/RUN_FULL_GATE_1e10`
- golden: `inputs_and_outputs/InSAR_dataset_test`

Example:

```bash
uv run pystamps verify \
  --run inputs_and_outputs/InSAR_dataset_test \
  --golden inputs_and_outputs/InSAR_dataset_test
```

Use `verify` when you already know which run folder and which reference folder you want to compare.

## Step 10: repo parity audit

The repo has a stricter audit command for the maintained validation datasets:

```bash
make audit
```

`make audit` reads the required dataset list from `pystamps/data/audited_workflow_manifest.json`. Do not replace it with a shorter hand-written list.

Important:
- this is slower than a normal run
- it creates fresh run copies under `inputs_and_outputs/validation_runs/<timestamp>/`
- in this workspace it can take around an hour
- it is stricter than a simple `make verify` check because it records a fresh audit artifact first
- the follow-up verify step must use the fresh `run_root` from `latest_audit.json`

## Troubleshooting

If a run fails:
1. Check the dataset layout with `uv run pystamps status --dataset /path/to/dataset`
2. Try a smaller stage range first, for example `--start-step 6 --end-step 8`
3. Confirm `triangle` and `snaphu` are installed and on `PATH`
4. Run on a fresh dataset copy if old outputs may be interfering
5. Use the notebook `notebooks/00_pystamps_beginner_walkthrough.ipynb` to compare your local steps with the guided flow

## Short version

For most cases, this is all you need:

```bash
uv sync
cp -a /path/to/input_dataset /path/to/input_dataset_run
uv run pystamps status --dataset /path/to/input_dataset_run
uv run pystamps run --dataset /path/to/input_dataset_run --start-step 1 --end-step 8
```
