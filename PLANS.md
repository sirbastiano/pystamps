# ExecPlan

## Goal
- Reproduce the current full-loop parity failures with the supported audit and verify surfaces, then classify the remaining mismatches by stage cluster with concrete artifact names and mismatch keys.

## Scope / Non-goals
- In scope: the audit/verify reporting path, a truthful full-loop run-root selection strategy for the required datasets, focused regression tests, and the recorded run outputs for US-003.
- Out of scope: parity math fixes, dataset content changes, pipeline-stage algorithm changes, or broad validation-contract/doc rewrites beyond what this story needs to make the classification truthful.

## Invariants and contracts to preserve
- `scripts/validate_audit.py` remains the supported audit entrypoint and still validates the required dataset contract from `pystamps.parity_contract`.
- The required datasets remain `inputs_and_outputs/InSAR_dataset_test_stage8diag` and `inputs_and_outputs/InSAR_dataset_test`; the audit must compare concrete run outputs against those golden datasets rather than silently self-comparing them.
- `pystamps verify` remains the underlying comparator and continues to emit artifact-level mismatches derived from `pystamps.verify`.
- Fresh-clone unit/build validation remains `uv run pytest -q`, `uv run --with build python -m build --sdist --wheel`, and `uv run --with twine python -m twine check dist/*`.

## Files / layers likely to change
- `scripts/validate_audit.py`
- `tests/test_validate_audit.py`
- `PLANS.md`
- Story run records under `.ralph/` and `inputs_and_outputs/validation_runs/`

## Ordered steps
1. Replace the audit driver's self-compare default with deterministic full-loop run-root resolution for the required datasets and surface the chosen `run_root` in the audit payload.
2. Add focused tests for the new run-root selection and keep the existing missing-dataset/interruption coverage passing.
3. Run the required audit and concrete verify commands, plus a machine-readable failure classification artifact, to capture current parity evidence.
4. Run the repo quality gates, perform a brief security/performance/regression review, then record progress and commit the story output.

## Validation plan
- `uv run pytest -q tests/test_validate_audit.py tests/test_verify.py`
- `uv run pytest -q`
- `uv run --with build python -m build --sdist --wheel`
- `uv run --with twine python -m twine check dist/*`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run python scripts/validate_audit.py --datasets inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test --output inputs_and_outputs/validation_runs/latest_audit.json`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run pystamps verify --run inputs_and_outputs/RUN_FULL_GATE_1e10 --golden ./inputs_and_outputs/InSAR_dataset_test`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run python scripts/classify_verify_failures.py --run inputs_and_outputs/RUN_FULL_GATE_1e10 --golden ./inputs_and_outputs/InSAR_dataset_test --output inputs_and_outputs/validation_runs/us003_verify_classification.json`

## Rollback / recovery
- Revert the audit-driver/test/logging hunks together. Generated `dist/`, `build/`, and `inputs_and_outputs/validation_runs/*` artifacts can be removed if this story needs to be backed out.

## Risks / blockers
- The repo already contains unrelated loop-managed changes in `.agents/tasks/prd-full-parity-loop.json` and `.ralph/activity.log`; edits and commit review must stay tight so US-003 does not rewrite unrelated state.
- The concrete full-loop run roots are repo-local assets. If they disappear or are renamed, the audit should fail loudly instead of silently self-comparing a golden dataset against itself.
- Full parity comparisons are materially slower than unit tests, so targeted checks should run before the repo-wide gates.

## Definition of done
- The supported audit command produces a current failure artifact that points at concrete run roots and groups failures by stage cluster with artifact paths and mismatch keys.
- The required verify command is run against a concrete full-loop run copy and its failing output is recorded.
- Required validation commands and repo-wide gates complete with outcomes captured in the story logs.
