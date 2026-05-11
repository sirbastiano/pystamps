# Getting Started With pySTAMPS

This document is for readers who want to use pySTAMPS without already knowing interferometry.

For the complete program-and-science tutorial, read [pipeline_science_guide.md](pipeline_science_guide.md). This page is the shorter beginner path.

## What problem pySTAMPS solves

pySTAMPS processes a radar time-series dataset through a sequence of stages and writes a collection of `.mat` outputs that later stages and validation tools can use.

You can think of pySTAMPS as a workflow engine for a scientific dataset folder.

It helps you:
- inspect whether a dataset looks valid
- run part or all of the processing chain
- validate your outputs before trusting a new run
- tune runtime settings through a config file
- switch between reference Python kernels and optimized Rust/native kernels

## The minimum interferometry background you need

For pySTAMPS usage, you only need a few working ideas.

### Radar observation
A radar satellite revisits the same area multiple times and records signals from the ground.

### Interferogram
An interferogram stores how the signal phase changed between observations. For pySTAMPS, you can treat it as one of the important intermediate data products used to estimate motion and noise.

### Wrapped phase and unwrapping
Phase values often repeat in cycles. Unwrapping is the step that turns those repeating values into a more continuous estimate that later analysis can use.

### Persistent scatterer
A persistent scatterer is a location that behaves consistently enough across acquisitions to be useful for time-series analysis.

### Golden dataset
A golden dataset is a reference output directory that you trust. Compare new runs against it in the dedicated verification workflow.

That vocabulary is enough to start operating the package.

## The mental model for using pySTAMPS

Work with three ideas:
- dataset root: the folder you point pySTAMPS at
- stages: numbered processing steps from 1 to 8
- outputs: files written back into the dataset root and patch directories

The safest workflow is:
1. make a copy of the dataset
2. inspect the copy with `status`
3. run the stage range you want
4. move to verification if you have a baseline output

## What a dataset folder usually contains

A pySTAMPS dataset is usually organized around:
- `PATCH_*` directories
- optional `patch.list`
- stage outputs such as `ps2.mat`, `phuw2.mat`, `uw_space_time.mat`
- supporting directories such as `rslc`, `diff0`, `dem`, or `geo`

The exact contents depend on where you start in the workflow.

Reference datasets in this repo:
- `inputs_and_outputs/InSAR_dataset_test`
- `inputs_and_outputs/InSAR_dataset_test_stage8diag`

## Install and check the runtime

From a local checkout, use `uv`:

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
uv sync
uv run pystamps describe-backends
```

If you use editable `pip` installs, source builds require Rust because the optimized native extension is compiled locally:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

`describe-backends` tells you whether the `python`, `native`, and optional `cuda` kernel backends are registered on your machine.

## First commands to learn

### Inspect a dataset

```bash
uv run pystamps status --dataset inputs_and_outputs/InSAR_dataset_test
```

Use this first when you are not sure whether the dataset layout is ready.

### Preview a run

```bash
uv run pystamps run \
  --dataset inputs_and_outputs/InSAR_dataset_test \
  --start-step 1 --end-step 8 --dry-run
```

Use this when you want to see the requested range without doing the expensive work yet.

### Run selected stages

```bash
cp -a inputs_and_outputs/InSAR_dataset_test_stage8diag /tmp/pystamps_first_run
uv run pystamps run \
  --dataset /tmp/pystamps_first_run \
  --start-step 6 --end-step 8
```

Use a partial range when earlier products already exist. Always run on a copy because pySTAMPS writes artifacts into the dataset tree.

The checked-in reference datasets already contain many outputs, so some stages can report `skipped_existing`. That is expected for completed examples.

### Use optimized Rust/native kernels

Create `native-kernels.yaml`:

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

Run the same dataset copy with that config:

```bash
uv run pystamps --config native-kernels.yaml run \
  --dataset /tmp/pystamps_first_run \
  --start-step 2 --end-step 8
```

The CLI skips stages whose expected output artifacts already exist. Use a dataset copy that still needs those stages if you want the pipeline to execute the optimized kernels, or use the direct kernel/benchmark examples in [howtorun.md](../howtorun.md) on the repo golden data.

Use `stage2_kernel_backend: python` or a per-kernel override such as `stage8_edge_noise: python` when you need the reference implementation for debugging.

### Verify and benchmark

```bash
uv run pystamps verify \
  --run /tmp/pystamps_first_run \
  --golden inputs_and_outputs/InSAR_dataset_test_stage8diag
```

Use `make benchmark` to measure backend speed on the maintained benchmark dataset. Use `make audit` for the full repo parity audit; it reads the maintained dataset list from `pystamps/data/audited_workflow_manifest.json`.

### Verification workflow

Move to the dedicated [Verification](verification.html) guide for the compare command and parity-audit flow.

## What the stages mean in plain language

- stage 1: read and organize the earliest patch-level inputs
- stage 2: estimate coherence or gamma-like quality terms
- stage 3: choose useful persistent scatterer candidates
- stage 4: weed out poor candidates
- stage 5: promote selected patch outputs into merged artifacts
- stage 6: unwrap temporal phase products
- stage 7: estimate raw SCLA and write `scla2.mat` plus the smoothed `scla_smooth2.mat` artifact used by the final replay
- stage 8: rerun the final unwrap-backed products and write `mean_v.mat` plus `uw_space_time.mat`

For legacy single-master parity runs, the wrapper-backed post flow maps to `stamps(5,7)` followed by `stamps(6,6)` and the `ps_plot('v-do', ...)` export, so pySTAMPS keeps SCLA in stage 7 and the final unwrap-backed outputs in stage 8.

You do not need to master the mathematics of each stage to operate the pipeline responsibly. You mainly need to know which range you intend to run and whether the required inputs already exist.

## Recommended learning path

1. Read [howtorun.md](../howtorun.md)
2. Open `notebooks/00_pystamps_beginner_walkthrough.ipynb`
3. Use `status` on `inputs_and_outputs/InSAR_dataset_test`
4. Try a `--dry-run`
5. Run a small stage range on a copy of a dataset
6. Continue in `verification` when you have a reference output tree

## Where to go next

- [docs/pipeline_science_guide.md](pipeline_science_guide.md): full program and science guide
- [howtorun.md](../howtorun.md): operational run guide
- [docs/function_reference.md](function_reference.md): module and function reference
- [docs/architecture.md](architecture.md): package structure and design
