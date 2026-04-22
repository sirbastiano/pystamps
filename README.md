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

```bash
git clone https://github.com/sirbastiano/pystamps.git
cd pystamps
python -m pip install -e .
```

Source and editable installs now compile the stage-2 native extension with Rust. Install a Rust toolchain first, for example with `rustup`.

For local development (including docs and tests), use:

```bash
python -m pip install -e ".[dev]"
```

Supported PyPI installs use platform wheels for the Rust extension. Source builds still require a local Rust toolchain.

Stage 2 can use the compiled native kernels automatically when they are available. To force the reference path or require the native path, set `runtime.stage2_kernel_backend` to `python` or `native` in your run config. With the default `runtime.stage2_native_threads: 0`, pySTAMPS now gives each stage-2 patch the full detected CPU budget and runs stage-2 patches one at a time to avoid oversubscription. `runtime.cpu_workers: 0` now means “use all detected CPU workers” rather than reserving one core. Set a positive `runtime.stage2_native_threads` value to force a fixed native worker count instead.

Built-in kernel backends are `python`, `native`, and `cuda`. Use `runtime.kernel_backend_overrides` to pin individual kernels without changing the runtime-wide backend, for example `stage7_scla: gpu` or `stage8_edge_noise: python`. Run `pystamps describe-backends` to print the registered backend providers and the current per-kernel coverage matrix.

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
