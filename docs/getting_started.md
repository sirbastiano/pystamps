# Getting Started With pySTAMPS

This document is for readers who want to use pySTAMPS without already knowing interferometry.

## What problem pySTAMPS solves

pySTAMPS processes a radar time-series dataset through a sequence of stages and writes a collection of `.mat` outputs that later stages and validation tools can use.

You can think of pySTAMPS as a workflow engine for a scientific dataset folder.

It helps you:
- inspect whether a dataset looks valid
- run part or all of the processing chain
- validate your outputs before trusting a new run
- tune runtime settings through a config file

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

## First commands to learn

### Inspect a dataset

```bash
pystamps status --dataset inputs_and_outputs/InSAR_dataset_test
```

Use this first when you are not sure whether the dataset layout is ready.

### Preview a run

```bash
pystamps run \
  --dataset inputs_and_outputs/InSAR_dataset_test \
  --start-step 1 --end-step 8 --dry-run
```

Use this when you want to see the requested range without doing the expensive work yet.

### Run selected stages

```bash
pystamps run \
  --dataset inputs_and_outputs/InSAR_dataset_test \
  --start-step 6 --end-step 8
```

Use a partial range when earlier products already exist.

### Verification workflow

Move to the dedicated [Verification](verification.html) guide for the compare command and parity-audit flow.

## What the stages mean in plain language

- stage 1: read and organize the earliest patch-level inputs
- stage 2: estimate coherence or gamma-like quality terms
- stage 3: choose useful persistent scatterer candidates
- stage 4: weed out poor candidates
- stage 5: promote selected patch outputs into merged artifacts
- stage 6: unwrap temporal phase products
- stage 7: estimate SCLA and velocity-related products
- stage 8: perform final space-time filtering outputs

You do not need to master the mathematics of each stage to operate the pipeline responsibly. You mainly need to know which range you intend to run and whether the required inputs already exist.

## Recommended learning path

1. Read [howtorun.md](../howtorun.md)
2. Open `examples/00_pystamps_beginner_walkthrough.ipynb`
3. Use `status` on `inputs_and_outputs/InSAR_dataset_test`
4. Try a `--dry-run`
5. Run a small stage range on a copy of a dataset
6. Continue in `verification` when you have a reference output tree

## Where to go next

- [howtorun.md](../howtorun.md): operational run guide
- [docs/function_reference.md](function_reference.md): module and function reference
- [docs/architecture.md](architecture.md): package structure and design
