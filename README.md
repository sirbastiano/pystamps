# pySTAMPS

Python-first migration workspace for StaMPS with hybrid threaded/process execution and golden-dataset verification.

## Install

Install the published package:

```bash
pip install pystamps
```

Optional GPU support:

```bash
pip install "pystamps[gpu]"
```

External prerequisites are not bundled in the package:
- `triangle` for stage-4 triangulation workflows
- `snaphu` for unwrap workflows
- a separate `StaMPS` checkout only when using `pystamps list-legacy`
- external datasets and golden artifacts for parity validation

The wheel ships the `pystamps` Python package and package metadata. The sdist ships the Python source tree and tracked docs needed to rebuild that package, while generated release outputs under `dist/`, `build/`, and local parity datasets under `inputs_and_outputs/` remain outside the release artifacts.

## Development Setup

```bash
uv sync
```

Fresh-clone validation commands:

```bash
uv run pytest -q
uv run --with build python -m build --sdist --wheel
uv run --with twine python -m twine check dist/*
```

Optional local-dataset parity validation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  uv run python scripts/validate_audit.py \
    --datasets \
      inputs_and_outputs/InSAR_dataset_test_stage8diag \
      inputs_and_outputs/InSAR_dataset_test \
    --output inputs_and_outputs/validation_runs/latest_audit.json
```

Manual release uploads:

```bash
uv run --with twine python -m twine upload --repository testpypi dist/*
uv run --with twine python -m twine upload dist/*
```

The standalone repo uses direct `uv` commands; there is no tracked `Makefile` in this snapshot.

## What It Does

- `pystamps` CLI with commands:
  - `status`: inspect patch and merged stage progress
  - `run`: execute stage orchestration (supports `--dry-run`)
  - `verify`: compare run artifacts against a golden dataset
  - `list-legacy`: enumerate legacy `StaMPS/bin` scripts
- Dataset discovery for `patch.list` and `PATCH_*` folders
- Hybrid runtime (`threads` + `processes`) abstraction
- MAT loader and numeric tolerance comparison utilities
- Verification harness wired to `inputs_and_outputs/InSAR_dataset_test`
- Ported Python implementations for patch stages 1-5:
  - stage 1: candidate load from `pscands.*` + metadata (`day.1.in`, `bperp.1.in`, ...)
  - stage 2: coherence/gamma proxy estimation and `pm1.mat` creation
  - stage 3: PS selection and `select1.mat` generation
  - stage 4: weed filtering and `weed1.mat` generation
  - stage 5: promote selected/weeded PS into version-2 artifacts (`ps2/ph2/pm2`)
- Ported Python implementations for merged stages 6-8:
  - stage 6: temporal unwrap to `phuw2.mat`
  - stage 7: SCLA and velocity proxy estimation (`scla2.mat`, `mean_v.mat`, `mv2.mat`)
  - stage 8: space-time filtering payload (`uw_space_time.mat`)

## CLI Usage

```bash
pystamps status --dataset inputs_and_outputs/InSAR_dataset_test
pystamps run \
  --dataset inputs_and_outputs/InSAR_dataset_test \
  --start-step 1 --end-step 8 --dry-run
pystamps verify \
  --run inputs_and_outputs/InSAR_dataset_test \
  --golden inputs_and_outputs/InSAR_dataset_test
pystamps list-legacy --stamps-root /path/to/StaMPS
```

Environment-based legacy discovery:

```bash
STAMPS_ROOT=/path/to/StaMPS pystamps list-legacy
```

Strict legacy parity replay mode:

```yaml
# config.yaml
compat:
  strict_reference: true
  reference_root: /abs/path/to/original/outputs
```

```bash
pystamps --config config.yaml run --dataset /path/to/run_copy --start-step 2 --end-step 8
```

Acceleration backend selection:

```yaml
# accel.yaml
runtime:
  backend: auto   # auto | threads | processes | gpu | native
  io_workers: 8
  cpu_workers: 0
  stage7_chunk_ps: 100000
  stage8_chunk_edges: 200000
  enable_mat_stage_cache: true
```

```bash
pystamps --config accel.yaml run --dataset /path/to/dataset --start-step 1 --end-step 8
```

## Validation and Benchmarking

Strict parity audit:

```bash
uv run python scripts/validate_audit.py \
  --datasets \
    inputs_and_outputs/InSAR_dataset_test_stage8diag \
    inputs_and_outputs/InSAR_dataset_test \
  --output inputs_and_outputs/validation_runs/latest_audit.json
```

- `scripts/validate_audit.py` is the supported unattended audit entrypoint.
- The audit validates both required datasets before verification begins and exits non-zero with a missing-dataset report if either path is absent.
- `latest_audit.json` records the contract, per-dataset audits, `failed_workflows`, `completed`, `interrupted`, and `ok`.

Benchmark runner:

```bash
uv run python scripts/benchmark_backends.py \
  --dataset inputs_and_outputs/InSAR_dataset_test_stage8diag \
  --start-step 1 --end-step 8 \
  --repeat 3 --warmup 1
```

- Each measured run now executes on a dedicated dataset copy under `inputs_and_outputs/benchmarks`.
- Benchmark subprocesses pin `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` for reproducible CPU timings.

## Build and Distribution

Create release artifacts:

```bash
uv run --with build python -m build --sdist --wheel
```

Check release artifacts:

```bash
uv run --with twine python -m twine check dist/*
```

Manual upload targets:

```bash
uv run --with twine python -m twine upload --repository testpypi dist/*
uv run --with twine python -m twine upload dist/*
```

Build outputs are written to `dist/` as one wheel and one sdist. Generated release directories such as `dist/` and `build/` are excluded from future source builds so packaging validation stays reproducible. The release process is manual and tag-driven; see [docs/release.md](docs/release.md) for the full checklist.

## Notes

- Stages 1-8 now execute in Python if artifacts are missing.
- The checked-in parity workflow is expected to reproduce the golden StaMPS artifacts exactly on the required datasets.
- Local parity datasets under `inputs_and_outputs/` are optional repo assets used for audit and verify workflows; a fresh clone can still run the unit/build/package gates without them.
- Repo-only developer workflows such as `scripts/validate_audit.py` require the full source tree, not just an installed wheel.
