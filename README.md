<div align="center">

<img src="docs/assets/pystamps-logo.svg" alt="pySTAMPS" style="width: 200px; height: auto; max-width: 100%;" />

# pySTAMPS

Python-first STA(MPS)-style runtime for staged InSAR/PS processing, verification, and deterministic audit checks.

Run staged pipelines, inspect dataset progress, and validate outputs against a reference dataset.

<p align="center">
  <a href="https://sirbastiano.github.io/pystamps/"><img src="https://img.shields.io/badge/-Documentation-0f172a?style=for-the-badge&logo=readme&logoColor=white&labelColor=0f172a" alt="Documentation" style="height: 34px;" /></a>
  <a href="https://sirbastiano.github.io/pystamps/quickstart.html"><img src="https://img.shields.io/badge/-Quick%20Start-0f172a?style=for-the-badge&logo=firefoxbrowser&logoColor=white&labelColor=0f172a" alt="Quick Start" style="height: 34px;" /></a>
  <a href="https://sirbastiano.github.io/pystamps/api/pystamps.html"><img src="https://img.shields.io/badge/-API%20Reference-0f172a?style=for-the-badge&logo=python&logoColor=white&labelColor=0f172a" alt="API Reference" style="height: 34px;" /></a>
</p>

</div>

## Install

From source:

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
make deps
uv run pystamps describe-backends
```

`make deps` installs or verifies the Rust toolchain pieces needed by native execution
(`cargo`, `rustfmt`, and `clippy`) and syncs the Python environment with `uv`.
On a fresh Ubuntu VM, install system build packages first:

```bash
make deps-ubuntu
make deps-check
```

Required local tools for source/native development are:
- Python 3.12 or newer
- `uv`
- Rust via `rustup`, including `rustfmt` and `clippy`
- a C/C++ build toolchain, `curl`, `pkg-config`, and Python development headers

Editable install:

```bash
python -m pip install -e .
python -m pip install -e "[dev]"
```

`cargo` is required for editable/source installs that build the Rust extension, the
native Rust CLI, and the Rust HTML frontend. Wheels from PyPI may avoid local
compilation.

## Run by stage

Set a local dataset path and always work on a writeable copy:

```bash
export DATASET_SOURCE=/path/to/original_dataset
export DATASET_COPY=/path/to/dataset_copy
cp -a "$DATASET_SOURCE" "$DATASET_COPY"
```

First, check status and verify what can execute:

```bash
uv run pystamps status --dataset "$DATASET_COPY"
```

Run a single stage or stage range:

```bash
uv run pystamps run --dataset "$DATASET_COPY" --start-step 1 --end-step 1      # stage 1 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 2 --end-step 2      # stage 2 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 3 --end-step 3      # stage 3 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 4 --end-step 4      # stage 4 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 5 --end-step 5      # stage 5 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 6 --end-step 6      # stage 6 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 7 --end-step 7      # stage 7 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 8 --end-step 8      # stage 8 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 1 --end-step 8          # full pipeline
```

Use `--dry-run` to preview actions without writing:

```bash
uv run pystamps run --dataset "$DATASET_COPY" --start-step 1 --end-step 8 --dry-run
```

## Native execution

### Run the full native chain with Python compatibility mode

The Python CLI stays the compatibility entrypoint, but it only delegates to Rust when
`runtime.backend` is set to `native` in config.

```bash
cat > native-rust.yaml <<'YAML'
runtime:
  backend: native
  stage2_kernel_backend: auto
  io_workers: 8
  cpu_workers: 0
  stage2_native_threads: 0
YAML

uv run pystamps --config native-rust.yaml run --dataset "$DATASET_COPY" --start-step 1 --end-step 8
```

Expected output is a JSON array of stage reports from the Rust pipeline driver:

```json
[
  {
    "stage": 1,
    "scope": "patch",
    "target": "PATCH_1",
    "status": "completed",
    "details": "written: ps1.mat, ph1.mat, bp1.mat, psver.mat",
    "duration_sec": 0.84
  },
  {
    "stage": 1,
    "scope": "patch",
    "target": "PATCH_2",
    "status": "completed",
    "details": "written: ps1.mat, ph1.mat, bp1.mat, psver.mat",
    "duration_sec": 0.81
  }
]
```

`uv run pystamps run` with `--dry-run` returns the same schema with each entry in `planned` status.

### Native CLI and web console inspection

Start the Rust HTML frontend for manual runs and coverage checks:

```bash
make web
```

Then open `http://127.0.0.1:8787`.

Inspect raw coverage from the Rust core:

```bash
cargo run -p pystamps-core --bin pystamps-native -- coverage --start-step 1 --end-step 8
```

Use the coverage HTTP API:

```bash
curl http://127.0.0.1:8787/api/native-coverage
```

Both paths return `StageCoverage[]` objects with:
- `rust_driver`: the chain can be planned/launched from Rust for that scope
- `native_stage`: Python execution is not needed for that scope
- `parity_certified`: the native implementation has passed its story gate
- `disabled` / `disabled_reason`: whether native coverage was explicitly disabled for that scope
- `not_native_reason` / `not_parity_certified_reason`: why a scope is not currently native-certified
- `unsupported_modes`: non-native execution modes rejected by the native-only gate
- `native_kernels`: accelerated kernel labels used inside the Rust stage

You can also exercise direct stage entry points (for debugging/validation):

```bash
cargo run -p pystamps-core --bin pystamps-native -- stage 1 --patch "$DATASET_COPY/PATCH_1"
cargo run -p pystamps-core --bin pystamps-native -- stage5-merge --dataset "$DATASET_COPY"
```

### Unsupported native configurations

Both wrappers fail fast when a requested mode is not supported:

- Rust wrapper backend values are limited to `auto`, `threads`, `processes`, `gpu`, or `native`.
- Rust `--stage2-kernel-backend` accepts only `auto`, `python`, or `native`.
- Rust `--native-only` requires `--backend native` and `--stage2-kernel-backend native`.
- Python config normalizer also rejects `runtime.stage2_kernel_backend: cuda` because stage-2 native execution does not expose CUDA.

```bash
cargo run -p pystamps-core --bin pystamps-native -- run --dataset "$DATASET_COPY" --backend bogus
cargo run -p pystamps-core --bin pystamps-native -- run --dataset "$DATASET_COPY" --native-only --backend auto
```

returns exit code 2 with:

```text
error: unsupported runtime backend 'bogus'
```

## Verify a run

```bash
export RUN_COPY=/path/to/run_copy
export GOLDEN_DATASET=/path/to/golden_dataset
uv run pystamps verify --run "$RUN_COPY" --golden "$GOLDEN_DATASET"
```

## Stage-backend profile (optional)

```bash
uv run pystamps describe-backends
```

Create `native-kernels.yaml` and pass it with `--config`:

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

uv run pystamps --config native-kernels.yaml run --dataset "$DATASET_COPY" --start-step 2 --end-step 8
```

Use `python` backends for reference behavior in debugging, and `native` for the compiled Rust/CPU path.

## Benchmarking and audit checkpoints

```bash
make benchmark
make audit
```

`make audit` reads the manifest in `pystamps/data/audited_workflow_manifest.json`.

## Notes

- Do not point docs or examples at a fixed repository dataset path.
- Always treat outputs in your run tree as authoritative; avoid running on your only source copy.
- Optional repo assets are kept for parity and offline reproducibility, not required for runtime usage.

## Read the docs

- [Pipeline and science guide](https://sirbastiano.github.io/pystamps/pipeline-science-guide.html)
- [Quick Start](https://sirbastiano.github.io/pystamps/quickstart.html)
- [Getting Started](https://sirbastiano.github.io/pystamps/getting-started.html)
- [Usage](https://sirbastiano.github.io/pystamps/usage.html)
- [Configuration](https://sirbastiano.github.io/pystamps/configuration.html)
- [Architecture](https://sirbastiano.github.io/pystamps/architecture.html)
- [Verification](https://sirbastiano.github.io/pystamps/verification.html)
- [API Reference](https://sirbastiano.github.io/pystamps/api/pystamps.html)
- [Release workflow](https://sirbastiano.github.io/pystamps/release.md)

## Notebooks

- `notebooks/start_here.ipynb`
- `notebooks/00_pystamps_beginner_walkthrough.ipynb`
