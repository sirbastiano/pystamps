# How To Run pySTAMPS

This guide explains how to run pySTAMPS if you are new both to the package and to interferometry.

If you want a slower, more tutorial-style introduction first, read [docs/getting_started.md](docs/getting_started.md) and then open `notebooks/00_pystamps_beginner_walkthrough.ipynb`.

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

External tools are also needed for some workflows:
- `triangle`
- `snaphu`

These tools are required for parts of the full processing flow, especially later stages and parity workflows.

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

## Step 5: understand what gets written

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

## Step 6: verify against a reference dataset

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

## Step 7: repo parity audit

The repo has a stricter audit command for the maintained validation datasets:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
uv run python scripts/validate_audit.py \
  --datasets \
    inputs_and_outputs/InSAR_dataset_test_stage8diag \
    inputs_and_outputs/InSAR_dataset_test \
  --output inputs_and_outputs/validation_runs/latest_audit.json
# or
make audit
```

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
