# ExecPlan

## Goal
- Investigate the US-004 stage 5-6 parity blockers with current audit evidence, confirm whether a source-of-truth stage-5/6 Python fix exists on this branch, and avoid speculative downstream drift.

## Scope / Non-goals
- In scope: current stage8diag audit evidence, patch-level stage-5 promotion and merged stage-5/6 inputs, required validation commands, and story-progress documentation.
- Out of scope: stage-3/4 fixes, stage-7/8 numerical changes, dataset edits, or unverifiable stage-6 algorithm guesses.

## Invariants and contracts to preserve
- Do not change stage-7/8 code to mask unchanged stage-5/6 mismatches.
- Do not leave speculative parity code in place if the verify/audit evidence does not improve.
- Keep the supported audit and verify surfaces unchanged: `scripts/validate_audit.py` and `pystamps verify`.

## Files / layers likely to change
- `PLANS.md`
- `.ralph/progress.md`
- `.ralph/activity.log`

## Ordered steps
1. Reproduce the current stage8diag stage-5/6 verify failure on the concrete run copy named by `latest_audit.json`.
2. Trace the first shape or value divergence back through `uw_interp.mat` / `uw_grid.mat` to the earliest upstream artifact.
3. Only edit source-of-truth Python if the root cause is confirmed within stage-5/6 code and the fix measurably improves the targeted artifacts.
4. If the earliest divergence is upstream of stage 5, leave product code unchanged, record the blocker, and keep the iteration documentation truthful.

## Validation plan
- `uv run pytest -q`
- `uv run --with build python -m build --sdist --wheel`
- `uv run --with twine python -m twine check dist/*`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run python scripts/validate_audit.py --datasets inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test --output inputs_and_outputs/validation_runs/latest_audit.json`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run pystamps verify --run inputs_and_outputs/validation_runs/20260313_035019/InSAR_dataset_test_stage8diag_stage2_8 --golden ./inputs_and_outputs/InSAR_dataset_test_stage8diag`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run pystamps verify --run inputs_and_outputs/RUN_FULL_GATE_1e10 --golden ./inputs_and_outputs/InSAR_dataset_test`

## Rollback / recovery
- This iteration should remain documentation-only unless a verified stage-5/6 root cause is proven. Roll back plan/progress updates together if they need to be reverted.

## Risks / blockers
- Current stage8diag failures classified under stage 5-6 are fed by earlier patch-level divergence:
  - `PATCH_1/select1.mat`: `C_ps2` shape `(80929,) != (80938,)`
  - `PATCH_1/weed1.mat`: `ix_weed` shape `(79132,) != (79227,)`
  - Derived patch stage-5 output count: `71671` vs golden `77888`
- Because stage-5 promotion already receives the wrong selected/weeded population, a stage-5/6-only code change on this branch would be speculative and would not satisfy US-004 acceptance.

## Definition of done
- Either a confirmed stage-5/6 source bug is fixed with measurable audit/verify improvement, or the story is explicitly recorded as blocked by upstream evidence with no speculative code left behind.
