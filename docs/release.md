# Release Process

`pystamps` releases are manual and tag-driven. This standalone repo uses direct Python packaging commands rather than a tracked `Makefile`.

## Prerequisites

- Python 3.12+ and `pip`
- Rust toolchain (`cargo`, `rustc`) available on any machine that builds source or wheel artifacts
- PyPI credentials available to `twine`
- a clean Git worktree
- local access to the validation datasets required by the parity audit
- the maintained run-copy seed `inputs_and_outputs/RUN_FULL_GATE_1e10` for the `InSAR_dataset_test` refresh
- Docker available on the Linux release host for manylinux wheel builds via `cibuildwheel`

## Release Steps

1. Sync the maintainer environment:

   ```bash
   pip install -e .[dev]
   ```

2. Run the test gate:

   ```bash
   uv run pytest -q
   ```

   Fresh-clone release prep stops here if the required local parity datasets are unavailable. Do not substitute a Makefile target, a hidden CI workflow, or a one-dataset audit command.

3. Run the strict parity gate:

   ```bash
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
     uv run python scripts/validate_audit.py \
       --datasets \
         inputs_and_outputs/InSAR_dataset_test_stage8diag \
         inputs_and_outputs/InSAR_dataset_test \
         inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag \
         inputs_and_outputs/InSAR_dataset_small_baseline_stage7 \
       --output inputs_and_outputs/validation_runs/latest_audit.json
   ```

   This command must finish unattended. It now creates fresh run copies under `inputs_and_outputs/validation_runs/<timestamp>/` before the parity validation step. See [Verification](verification.html) for the exact run-copy and compare flow.

4. Resolve the required run copy from the fresh audit artifact and run verification as described in the dedicated guide:

   See [Verification](verification.html).

5. Create and push a release tag using the version form `vX.Y.Z`.

6. Build the release artifacts from the tagged commit:

   ```bash
   uv run --with build python -m build --sdist
   ```

7. Build Linux wheels on the Linux host:

   ```bash
   uv run --with cibuildwheel python -m cibuildwheel --platform linux --output-dir dist
   ```

8. Build macOS wheels on the macOS host:

   ```bash
   uv run --with cibuildwheel python -m cibuildwheel --platform macos --output-dir dist
   ```

9. Build Windows wheels on the Windows host:

   ```bash
   uv run --with cibuildwheel python -m cibuildwheel --platform windows --output-dir dist
   ```

10. Validate the gathered artifacts:

   ```bash
   uv run --with twine python -m twine check dist/*
   ```

11. Upload to TestPyPI for rehearsal when needed:

   ```bash
   uv run --with twine python -m twine upload --repository testpypi dist/*
   ```

12. Upload the final artifacts to PyPI:

   ```bash
   uv run --with twine python -m twine upload dist/*
   ```

## Release Requirements

- `pytest` must pass.
- `latest_audit.json` must report no failed parity workflows.
- The explicit verification command must pass using the `run_root` recorded in `latest_audit.json` (documented in verification).
- `python -m build --sdist` must emit the release sdist.
- `cibuildwheel` must emit the expected platform wheels for Linux, macOS, and Windows.
- `twine check` must pass on every file gathered in `dist/`.
- Any interrupted audit, manual restart, or stale run-copy reuse leaves the release gate closed.

## Distribution Scope

- The wheel set contains the `pystamps` Python package, the compiled Rust stage-2 native extension, and package metadata.
- The sdist contains the tracked Python source tree, Rust sources, and release docs needed to rebuild those wheels.
- Release artifacts do not include `inputs_and_outputs/`, `tmp/`, or the vendored `StaMPS/` tree.
- Generated directories such as `dist/` and `build/` are excluded from the source distribution so repeated build validation does not recurse on prior outputs.
- External binaries such as `triangle` and `snaphu` remain user-managed prerequisites.
