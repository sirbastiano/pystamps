# ExecPlan

## Goal
- Make the standalone validation and release guidance truthful from a clean source checkout by aligning tracked docs, optional dataset-backed tests, and packaging metadata with the real repo command surface.

## Scope / Non-goals
- In scope: tracked contributor/release docs, packaging metadata and manifests, and focused regression tests that lock the standalone validation contract in place.
- Out of scope: parity math fixes, dataset contents, workflow orchestration changes, or adding new task runners / CI surfaces that do not already exist in the repo.

## Invariants and contracts to preserve
- `uv run pytest -q`, `uv run --with build python -m build --sdist --wheel`, and `uv run --with twine python -m twine check dist/*` remain the standalone unit/build validation commands.
- `scripts/validate_audit.py` remains the supported full-validation audit entrypoint, and it requires the local parity datasets named in `pystamps.parity_contract`.
- Fresh-clone validation must not imply that `inputs_and_outputs/*` parity datasets are committed or required for the default unit-test path.
- Release artifacts must continue to package the `pystamps` source tree without recursive inclusion of generated `dist/`, `build/`, or egg-info outputs.

## Files / layers likely to change
- `README.md`
- `docs/release.md`
- `docs/testing.html`
- `pyproject.toml`
- `MANIFEST.in`
- `tests/` coverage for docs/packaging contract checks

## Ordered steps
1. Confirm the exact standalone validation commands and dataset requirements from repo code/config rather than from stale docs.
2. Update tracked docs so they separate fresh-clone unit/build validation from optional local-dataset parity validation and do not reference nonexistent task runners or unsupported fallback behavior.
3. Tighten packaging metadata/manifests so generated release outputs cannot be re-ingested into future builds.
4. Add focused regression tests for the documented validation commands, dataset-test guarding, and manifest exclusions.
5. Run targeted checks first, then the required repo quality gates for this story, followed by a brief security/performance/regression review before commit.

## Validation plan
- `uv run pytest -q tests/test_parity_contract.py tests/test_validate_audit.py tests/test_dataset.py`
- `uv run pytest -q`
- `uv run --with build python -m build --sdist --wheel`
- `uv run --with twine python -m twine check dist/*`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run python scripts/validate_audit.py --datasets inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test --output inputs_and_outputs/validation_runs/latest_audit.json`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run pystamps verify --run <run-copy> --golden ./inputs_and_outputs/InSAR_dataset_test`

## Rollback / recovery
- Revert the docs/packaging/test hunks together. No stateful rollback is required beyond deleting rebuilt artifacts from `dist/` or `build/` if validation regenerates them.

## Risks / blockers
- The repo already has unrelated uncommitted changes from the story loop; edits must stay tightly scoped so this story does not absorb earlier work accidentally.
- Some tracked docs are generated/static HTML rather than Markdown, so command-surface fixes may need to be applied in multiple files to avoid contradictory guidance.
- Full audit and verify commands depend on local datasets and possibly a prepared run copy; if those assets are absent, that remains validation evidence rather than something this story can fake.

## Definition of done
- Tracked docs reference only the real standalone validation and packaging commands and clearly distinguish optional local-dataset gates from fresh-clone unit/build validation.
- Dataset-backed tests are explicitly optional/guarded so a clean checkout does not imply committed parity datasets.
- Packaging metadata and manifests exclude generated or recursive release artifacts from the release surface.
- Focused regression coverage and required validation commands provide evidence for the updated contract.
