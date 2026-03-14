# Release Process

`pystamps` releases are manual and tag-driven. This standalone repo uses direct `uv`/Python packaging commands rather than a tracked `Makefile`.

## Prerequisites

- Python 3.12+
- `uv`
- PyPI credentials available to `twine`
- a clean Git worktree
- local access to the validation datasets required by the parity audit

## Release Steps

1. Sync the maintainer environment:

   ```bash
   uv sync
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
       --output inputs_and_outputs/validation_runs/latest_audit.json
   ```

4. Create and push a release tag using the version form `vX.Y.Z`.

5. Build the release artifacts from the tagged commit:

   ```bash
   uv run --with build python -m build --sdist --wheel
   ```

6. Validate the built artifacts:

   ```bash
   uv run --with twine python -m twine check dist/*
   ```

7. Upload to TestPyPI for rehearsal when needed:

   ```bash
   uv run --with twine python -m twine upload --repository testpypi dist/*
   ```

8. Upload the final artifacts to PyPI:

   ```bash
   uv run --with twine python -m twine upload dist/*
   ```

## Release Requirements

- `pytest` must pass.
- `latest_audit.json` must report no failed parity workflows.
- `python -m build` must emit exactly one wheel and one sdist.
- `twine check` must pass on all files in `dist/`.

## Distribution Scope

- The wheel contains the `pystamps` Python package and package metadata.
- The sdist contains the tracked Python source tree and release docs needed to rebuild that wheel.
- Release artifacts do not include `inputs_and_outputs/`, `tmp/`, or the vendored `StaMPS/` tree.
- Generated directories such as `dist/` and `build/` are excluded from the source distribution so repeated build validation does not recurse on prior outputs.
- External binaries such as `triangle` and `snaphu` remain user-managed prerequisites.
