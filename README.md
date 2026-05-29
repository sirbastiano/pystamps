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

## Install the bundled native Rust CLI

Platform wheels bundle the standalone `pystamps-native` executable under the
Python package and expose a `pystamps-native` console entry point that immediately
launches that executable. After installing a wheel, run the native chain without
building from source:

```bash
python -m pip install pystamps-insar

pystamps-native run \
  --native-only \
  --dataset /path/to/writable_dataset_copy \
  --start-step 1 \
  --end-step 8 \
  --backend native \
  --stage2-kernel-backend native \
  --cpu-workers 0 \
  --stage2-native-threads 0
```

The wheel build matrix targets CPython 3.12-3.14 on Linux x86_64/aarch64,
macOS Intel, macOS Apple Silicon, and Windows AMD64. Other platforms can still
install from source when a Rust toolchain and platform build tools are available.
Alpine/musllinux, PyPy, Windows ARM64, and other architectures are not currently
part of the wheel matrix.

## Fresh VM native validation

Use this path when reproducing the fast native full-chain gate on a new VM. It
documents both the implementation intent and the test evidence expected from the
release runner: build the release Rust binary, create a clean validation run
copy, execute native-only stages 1-8, then compare the generated MAT artifacts
against the golden dataset.

Fresh-clone validation commands:

```bash
make deps-ubuntu
make deps-check
uv run python -c "import h5py, mat73"
cargo build --release -p pystamps-core --bin pystamps-native
make native-full-chain-verify
```

For non-Ubuntu hosts, install the equivalent of `build-essential`, `curl`,
`pkg-config`, Python development headers, `uv`, and a Rust toolchain through
`rustup`. The vendored Rust HDF5 reader/writer is pure Rust and does not require
a system `libhdf5` for native execution. Python verification still uses `h5py`
and `mat73` for MATLAB v7.3/HDF5 MAT support, so keep the `uv run python -c
"import h5py, mat73"` check in the setup path. If that import fails, run
`uv sync` through `make deps` or install the platform HDF5 build headers needed
by your Python wheels before running the gate.

The default validation dataset is `inputs_and_outputs/InSAR_dataset_test`.
Either copy it into the checkout or mount it read-only and pass explicit paths:

```bash
mkdir -p inputs_and_outputs/validation_runs
cp -a /mnt/pystamps-validation/InSAR_dataset_test inputs_and_outputs/

make native-full-chain-verify \
  DATASET=/mnt/pystamps-validation/InSAR_dataset_test \
  GOLDEN=/mnt/pystamps-validation/InSAR_dataset_test \
  RUN=inputs_and_outputs/validation_runs/native-full-chain-vm \
  THREADS=8
```

`DATASET` is the source tree copied into a clean run directory. `GOLDEN` is the
reference tree used by the verifier. `RUN` is deleted and recreated by the gate,
so do not point it at `DATASET`, inside `DATASET`, the repo root, `/`, or your
home directory. `THREADS=0` is the default and lets the native runner use all
available processing threads; set `THREADS=N` to pin both `--cpu-workers` and
`--stage2-native-threads`. `START_STEP` and `END_STEP` can scope focused
verification while preserving the same copy/clean/report behavior.

The gate writes machine-readable evidence under `RUN/_native_gate_reports/`:

- `native-coverage-report.json`: native-only coverage precheck
- `native-run-report.json`: command, stdout/stderr, stage results, budget status
- `native-run-timings.json`: per-stage durations and performance waivers
- `native-verify-report.json`: parity verifier payload and failed comparisons

Artifact parity uses `pystamps/data/artifact_tolerances.json`. The accepted
defaults are exact structural and sparse parity, f64 `atol=1e-6`/`rtol=1e-8`,
f32 `atol=1e-4`/`rtol=1e-5`, and wrapped phase modulo `2*pi` with
`atol=1e-4` radians. Shape, required-key, sparse-structure, and PS ordering
mismatches are not waived by the verifier. Runtime and stage performance
waivers live in `pystamps/data/native_performance_budgets.json` as
`temporary_waiver` objects with non-empty `reason`, `owner`, and future
`expires_at_utc`; expired or incomplete waivers are ignored by the gate.

Expected setup failures are ordinary errors, not panics. An absent validation
dataset exits with status 2 and an error like `dataset root is not a directory:
...`. Missing Python HDF5/MAT support is caught by the import check above or by
the verifier as an environment/setup error; repair the `uv` environment before
interpreting parity results.

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

### Run the full native chain without the Python wrapper

Use the standalone Rust binary when you want the processing chain to execute
without the Python CLI wrapper. Build the optimized binary once, copy the
dataset to a writable run directory, then launch `pystamps-native` directly:

```bash
export DATASET_SOURCE=/path/to/original_dataset
export RUN_COPY=/path/to/native_run_copy

rm -rf "$RUN_COPY"
cp -a "$DATASET_SOURCE" "$RUN_COPY"
cargo build --release -p pystamps-core --bin pystamps-native

target/release/pystamps-native run \
  --native-only \
  --dataset "$RUN_COPY" \
  --start-step 1 \
  --end-step 8 \
  --backend native \
  --stage2-kernel-backend native \
  --cpu-workers 0 \
  --stage2-native-threads 0
```

`--native-only` rejects bridge execution and requires `--backend native` with
`--stage2-kernel-backend native`. `--cpu-workers 0` and
`--stage2-native-threads 0` mean "use all CPUs visible to the process"; set
positive values to cap processing threads for reproducible resource limits.

Run a focused native range by changing `--start-step` and `--end-step`:

```bash
target/release/pystamps-native run \
  --native-only \
  --dataset "$RUN_COPY" \
  --start-step 2 \
  --end-step 2 \
  --backend native \
  --stage2-kernel-backend native \
  --cpu-workers 0 \
  --stage2-native-threads 0
```

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

`cpu_workers: 0` and `stage2_native_threads: 0` select max-throughput auto mode:
the runner uses all CPUs visible to the process. Set positive values to cap CPU
process workers and Stage 2 native threads.

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

### Native Rust benchmark evidence

The following benchmark is a single-run native-only measurement on the bundled
`inputs_and_outputs/InSAR_dataset_test` dataset. Stage totals are measured inside
the Rust native runner and exclude the release build time. Gate elapsed time
includes setup and report generation. Treat these values as host-specific
evidence, not a cross-machine performance guarantee.

Benchmark command:

```bash
make native-full-chain-run \
  DATASET=inputs_and_outputs/InSAR_dataset_test \
  RUN=inputs_and_outputs/validation_runs/native-rust-benchmark-20260529-maxthreads \
  THREADS=0 \
  START_STEP=1 \
  END_STEP=8
```

The gate above prepares a clean run copy, but the command it validates is the
standalone Rust binary:

```bash
target/release/pystamps-native run \
  --native-only \
  --dataset inputs_and_outputs/validation_runs/native-rust-benchmark-20260529-maxthreads \
  --start-step 1 \
  --end-step 8 \
  --backend native \
  --stage2-kernel-backend native \
  --cpu-workers 0 \
  --stage2-native-threads 0
```

Benchmark environment:

| Field | Value |
|---|---|
| Date generated | 2026-05-29T13:15:38Z |
| Git revision | `12c3e4b` |
| OS/kernel | Ubuntu Linux, `6.8.0-107-generic`, x86_64 |
| Processor reported by `lscpu` | Intel Xeon Processor (Cascadelake) |
| Virtualization | KVM full virtualization |
| Online CPUs visible to process | 16 (`0-15`) |
| VM CPU topology reported by `lscpu` | 16 sockets x 1 core/socket x 1 thread/core |
| NUMA nodes | 1 |
| Rust toolchain | `rustc 1.94.0`, `cargo 1.94.0` |
| Native thread setting | `THREADS=0`, resolved to 16 Stage 2 native threads |
| Native execution elapsed | 330.419 s |
| Gate elapsed | 353.663 s |
| Release build time before run | 33.76 s |
| Performance budget status | OK, with existing waived Stage 5 merged local-budget overrun |

Stage totals:

| Stage | Scope | Total duration |
|---|---|---:|
| 1 | patch | 31.535 s |
| 2 | patch | 135.497 s |
| 3 | patch | 18.963 s |
| 4 | patch | 17.365 s |
| 5 | patch + merged | 73.555 s |
| 6 | merged | 21.841 s |
| 7 | merged | 14.849 s |
| 8 | merged | 11.211 s |

Patch and merged breakdown:

| Stage | PATCH_1 | PATCH_2 | PATCH_3 | PATCH_4 | Merged |
|---|---:|---:|---:|---:|---:|
| 1 | 6.373 s | 9.440 s | 8.220 s | 7.503 s | - |
| 2 | 36.450 s | 38.546 s | 28.911 s | 31.590 s | - |
| 3 | 2.464 s | 6.799 s | 4.762 s | 4.938 s | - |
| 4 | 2.041 s | 6.139 s | 4.446 s | 4.739 s | - |
| 5 | 4.794 s | 14.288 s | 9.468 s | 11.149 s | 33.855 s |
| 6 | - | - | - | - | 21.841 s |
| 7 | - | - | - | - | 14.849 s |
| 8 | - | - | - | - | 11.211 s |

Timing evidence is written to
`inputs_and_outputs/validation_runs/native-rust-benchmark-20260529-maxthreads/_native_gate_reports/native-run-timings.json`.

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
