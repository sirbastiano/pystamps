# ExecPlan

## Goal
- Resolve US-005 by rerunning the current parity surfaces, confirming whether stage 7-8 still have independent numerical drift, and refusing speculative downstream edits when the contained full-run copy already matches golden.

## Scope / Non-goals
- In scope: refreshed audit evidence for `scla2.mat`, `mean_v.mat`, and `uw_space_time.mat`; required repo gates; and truthful story-progress documentation.
- Out of scope: stage-3/4 fixes, stage-5/6 fixes, dataset edits, or any stage-7/8 source change that is not supported by fresh before/after evidence on the targeted keys.

## Invariants and contracts to preserve
- Do not modify stage-7/8 code unless a rerun shows an actual stage-7/8 mismatch on a contained run copy.
- Do not treat stage8diag shape drift as standalone stage-7/8 evidence when upstream patch and unwrap artifacts already diverge.
- Keep the supported validation contract unchanged: `scripts/validate_audit.py` and `pystamps verify` remain the source-of-truth gates.

## Files / layers likely to change
- `PLANS.md`
- `.ralph/progress.md`
- `.ralph/activity.log`

## Ordered steps
1. Refresh the full audit and direct verify evidence on the current branch.
2. Compare `RUN_FULL_GATE_1e10` against `InSAR_dataset_test` for `scla2.mat`, `mean_v.mat`, and `uw_space_time.mat` to confirm whether stage-7/8 parity is already green once upstream blockers are contained.
3. Only edit source-of-truth Python if a fresh rerun shows a real stage-7/8 mismatch and the change measurably improves the exact failing key.
4. If stage-7/8 already match on the contained run, leave product code unchanged and record that the remaining gate failures are upstream/out-of-scope for US-005.

## Validation plan
- `uv run pytest -q`
- `uv run --with build python -m build --sdist --wheel`
- `uv run --with twine python -m twine check dist/*`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run python scripts/validate_audit.py --datasets inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test --output inputs_and_outputs/validation_runs/latest_audit.json`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run pystamps verify --run inputs_and_outputs/RUN_FULL_GATE_1e10 --golden ./inputs_and_outputs/InSAR_dataset_test`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run python - <<'PY' ... _compare_mat(...) for scla2.mat, mean_v.mat, uw_space_time.mat ... PY`

## Rollback / recovery
- This iteration should remain evidence-only unless fresh contained-run evidence proves a stage-7/8 source bug. Roll back plan/progress/activity updates together if the documentation needs correction.

## Risks / blockers
- The refreshed audit still shows stage8diag downstream file failures, but they are coupled to upstream stage-3/4 and stage-5/6 divergence:
  - `PATCH_1/select1.mat`: `C_ps2` shape `(80929,) != (80938,)`
  - `PATCH_1/weed1.mat`: `ix_weed` shape `(79132,) != (79227,)`
  - `pm2.mat`: `C_ps` shape `(71671,) != (69009,)`
  - `uw_interp.mat`: `Z` shape `(931, 2355) != (1773, 4378)`
- On the contained full-run copy `RUN_FULL_GATE_1e10`, the direct stage-7/8 artifact comparisons already match golden and the only failing verify artifact is upstream: `PATCH_3/weed1.mat.ps_max` with `max_abs=0.000147104`.

## Definition of done
- Either a confirmed stage-7/8 source bug is fixed with measurable improvement on the exact failing key, or the story is truthfully recorded as requiring no stage-7/8 code change because fresh contained-run evidence shows those artifacts already match golden.
