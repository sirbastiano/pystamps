<div align="center">

<img src="docs/assets/pystamps-logo.svg" alt="pySTAMPS" style="width: 200px; height: auto; max-width: 100%;" />

# pySTAMPS

Python-first StaMPS migration runtime for structured InSAR processing, verification, and reproducible parity workflows.

Run pipeline stages, inspect dataset state, and verify outputs against reference datasets.

<p align="center">
  <a href="https://sirbastiano.github.io/pystamps/"><img src="https://img.shields.io/badge/-Documentation-0f172a?style=for-the-badge&logo=readme&logoColor=white&labelColor=0f172a" alt="Documentation" style="height: 34px;" /></a>
  <a href="https://sirbastiano.github.io/pystamps/quickstart.html"><img src="https://img.shields.io/badge/-Quick%20Start-0f172a?style=for-the-badge&logo=firefoxbrowser&logoColor=white&labelColor=0f172a" alt="Quick Start" style="height: 34px;" /></a>
  <a href="https://sirbastiano.github.io/pystamps/api/pystamps.html"><img src="https://img.shields.io/badge/-API%20Reference-0f172a?style=for-the-badge&logo=python&logoColor=white&labelColor=0f172a" alt="API Reference" style="height: 34px;" /></a>
  <a href="notebooks/start_here.ipynb"><img src="https://img.shields.io/badge/-Start%20Here-0f172a?style=for-the-badge&logo=jupyter&logoColor=white&labelColor=0f172a" alt="Start Here Notebook" style="height: 34px;" /></a>
  <a href="notebooks/00_pystamps_beginner_walkthrough.ipynb"><img src="https://img.shields.io/badge/-Beginner%20Notebook-0f172a?style=for-the-badge&logo=jupyter&logoColor=white&labelColor=0f172a" alt="Beginner Notebook" style="height: 34px;" /></a>
</p>

</div>

**Author:** Roberto Del Prete

pySTAMPS works with StaMPS-style dataset folders by orchestrating stage execution and optional parity verification in a reproducible way.

## Install

Recommended local setup from a checkout:

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
uv sync
uv run pystamps describe-backends
```

Editable `pip` install:

```bash
python -m pip install -e .
# Developer tools, tests, and notebooks:
python -m pip install -e ".[dev]"
```

Source and editable installs compile the Rust-backed native extension. Install a Rust toolchain first, for example with `rustup`, and confirm `cargo --version` works. Supported PyPI installs use platform wheels for the Rust extension when available; source builds still require Rust locally.

Optional GPU dependencies:

```bash
python -m pip install -e ".[gpu]"
```

## First run

Always run on a copy because pySTAMPS writes outputs into the dataset directory:

```bash
cp -a inputs_and_outputs/InSAR_dataset_test_stage8diag /tmp/pystamps_stage8diag_run
uv run pystamps status --dataset /tmp/pystamps_stage8diag_run
uv run pystamps run --dataset /tmp/pystamps_stage8diag_run --start-step 6 --end-step 8
uv run pystamps verify \
  --run /tmp/pystamps_stage8diag_run \
  --golden inputs_and_outputs/InSAR_dataset_test_stage8diag
```

The checked-in diagnostic datasets already contain many outputs, so some stages can report `skipped_existing`. That is expected; use your own incomplete dataset copy for real processing.

Use `--dry-run` first if you only want to see the selected stage range:

```bash
uv run pystamps run --dataset /tmp/pystamps_stage8diag_run --start-step 1 --end-step 8 --dry-run
```

## Optimized kernels

Built-in kernel backends are `python`, `native`, and `cuda`. `native` is the compiled Rust/CPU path. Inspect what is available on your machine:

```bash
uv run pystamps describe-backends
```

Force the optimized native kernels from a config file:

```bash
cat > native-kernels.yaml <<'YAML'
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
YAML

uv run pystamps --config native-kernels.yaml run \
  --dataset /tmp/pystamps_stage8diag_run \
  --start-step 2 --end-step 8
```

Normal CLI execution skips a stage when that stage's expected output artifact is already present. To exercise optimized kernels through the pipeline, run this config on a dataset copy that still needs those stages. To exercise kernels on the checked-in golden data without changing stage artifacts, use the direct kernel API example in `howtorun.md` or the benchmark script below.

Use `stage2_kernel_backend: python` or per-kernel overrides such as `stage8_edge_noise: python` when you need the reference path for debugging. Stage 2 accepts `auto`, `python`, or `native`; stage 4, 7, and 8 can use `python`, `native`, and `cuda` where registered. With `stage2_native_threads: 0`, pySTAMPS gives each stage-2 patch the detected CPU budget and runs stage-2 patches one at a time to avoid oversubscription. `cpu_workers: 0` uses all detected CPU workers.

Benchmark the configured backends on the maintained dataset:

```bash
make benchmark
# or customize directly:
uv run python scripts/benchmark_backends.py \
  --dataset inputs_and_outputs/InSAR_dataset_test_stage8diag \
  --start-step 6 --end-step 8 \
  --backends threads native \
  --repeat 3 --warmup 1
```

## Fresh-clone validation commands:

```bash
uv run pytest -q
uv run --with build python -m build --sdist --wheel
uv run --with twine python -m twine check dist/*
```

The local parity datasets under `inputs_and_outputs/InSAR_dataset_test_stage8diag`,
`inputs_and_outputs/InSAR_dataset_test`, `inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag`,
and `inputs_and_outputs/InSAR_dataset_small_baseline_stage7` are optional repo assets.
Keep the fresh-clone validation surface separate from parity workflows that need those optional repo assets.

## Repo entrypoints

The tracked local entrypoints match the root `Makefile`:

```bash
make setup
make test
make build
make twine-check
make audit
make parity-loop
make verify
make benchmark
```

## Oracle-backed parity contract

The supported audit driver is `scripts/validate_audit.py`, and `make audit` is the repo-local wrapper for that same command surface. The required audited dataset set is owned by `pystamps/data/audited_workflow_manifest.json`; do not replace it with a reduced hand-written dataset list.

Oracle precedence is owned by `pystamps/data/oracle_contract.json`: `cpp_wrapper` first, then `matlab_source`, then `manual_references`. When the pinned StaMPS wrapper behavior intentionally differs from plain MATLAB, pySTAMPS treats the wrapper-backed path as the practical parity oracle and records that source in the audit evidence.

## What pySTAMPS does

- Inspect dataset status and stage progress (`status`)
- Dry-run and execute targeted stage ranges (`run`)
- Validate outputs with explicit comparison flows (`verify`)
- Track compatibility/replay mode for controlled reproducibility

For merged post-processing, pySTAMPS now keeps the StaMPS stage boundary aligned with the legacy single-master flow: stage 7 writes both the raw `scla2.mat` result and the smoothed `scla_smooth2.mat` envelope, while stage 8 only performs the final space-time filtering and writes `uw_space_time.mat`.

The internal parity-audit regeneration path for `RUN_FULL_GATE_1e10` also mirrors the legacy merged-post refinement loop `6 -> 7 -> 6 -> 7 -> 8`. This is used for audit reproducibility; the normal CLI stage range model stays unchanged.

## Read the full docs

- [Introduction (docs index)](https://sirbastiano.github.io/pystamps/)
- [Pipeline and science guide](https://sirbastiano.github.io/pystamps/pipeline-science-guide.html)
- [Quick Start](https://sirbastiano.github.io/pystamps/quickstart.html)
- [Getting Started](https://sirbastiano.github.io/pystamps/getting-started.html)
- [Usage and command patterns](https://sirbastiano.github.io/pystamps/usage.html)
- [Configuration](https://sirbastiano.github.io/pystamps/configuration.html)
- [Architecture](https://sirbastiano.github.io/pystamps/architecture.html)
- [Verification](https://sirbastiano.github.io/pystamps/verification.html)
- [Parity contract](parity.md)
- [API Reference](https://sirbastiano.github.io/pystamps/api/pystamps.html)
- [Release workflow](https://sirbastiano.github.io/pystamps/release.md)

## Notebooks

- `notebooks/start_here.ipynb`
- `notebooks/00_pystamps_beginner_walkthrough.ipynb`
- `howtorun.md`

## Governance

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [License](LICENSE) (Apache 2.0)
