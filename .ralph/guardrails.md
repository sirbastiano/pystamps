# Guardrails (Signs)

> Lessons learned from failures. Read before acting.

## Core Signs

### Sign: Read Before Writing
- **Trigger**: Before modifying any file
- **Instruction**: Read the file first
- **Added after**: Core principle

### Sign: Test Before Commit
- **Trigger**: Before committing changes
- **Instruction**: Run required tests and verify outputs
- **Added after**: Core principle

---

## Learned Signs

### Sign: Configure Audit External Tools
- **Trigger**: Before running `make audit` or `scripts/validate_audit.py` on datasets that reach Stage 6+
- **Instruction**: Ensure `triangle` and `snaphu` resolve in the audit path, either through an explicit run config or by exposing `.cache/pystamps-tools/bin` on `PATH`.
- **Added after**: Iteration 5 - repeated full-audit interruption because exact `make audit` used default tool names and could not find bundled `snaphu`.

### Sign: Bound Notebook Stage 2 Hangs
- **Trigger**: When the stage-by-stage notebook stays in Stage 2 without new artifacts for more than 30 minutes
- **Instruction**: Check `PATCH_*/pm1.mat` mtimes and kernel CPU. If no artifact changes after another interval, terminate the notebook gate and record it as a Stage 2 execution blocker instead of waiting indefinitely.
- **Added after**: Iteration 7 - repeated notebook execution did not advance past Stage 2 and ended with DeadKernelError.

### Sign: Bound Notebook Post-Stage2 Stalls
- **Trigger**: When `notebooks/03_stage_by_stage_oracle.ipynb` creates `PATCH_*/pm1.mat` but no `select1.mat`, `weed1.mat`, or merged artifacts appear for more than 30 minutes.
- **Instruction**: Treat the run as stuck in Stage 2 verification/plotting or Stage 3 setup. Check artifact mtimes and kernel CPU once more, then terminate and record the notebook gate as blocked rather than waiting indefinitely.
- **Added after**: Iteration 8 - notebook execution produced `pm1.mat`, then made no later artifact progress for roughly 45 minutes and ended with DeadKernelError after termination.

### Sign: Bound Full Audit Stage 2 Stalls
- **Trigger**: When `make audit` spends more than 30 minutes in an `*_stage2_8` validation run with no new `PATCH_*/pm1.mat` or later stage artifacts after the initial run-copy setup.
- **Instruction**: Check the current validation run mtimes and process CPU once more, then terminate the run and record the audit gate as blocked unless a scoped performance fix is being made before rerunning the exact `make audit` command.
- **Added after**: Iteration 9 - repeated full-audit Stage 2 run stayed CPU-bound for ~30 minutes with only `patch.list` written.

### Sign: Bound Full Audit Post-Stage2 Stalls
- **Trigger**: When `make audit` writes `PATCH_*/pm1.mat` but no `select1.mat`, `weed1.mat`, or merged artifacts appear for more than 30 minutes.
- **Instruction**: Check artifact mtimes and process CPU once more, then terminate and record the audit gate as blocked unless a scoped Stage 3+ performance fix is being made before rerunning the exact `make audit` command.
- **Added after**: Iteration 9 - full-audit Stage 2 completed, then Stage 3 selection/re-estimation stayed CPU-bound without writing `select1.mat`.

### Sign: Bound Uv Editable Build Stalls
- **Trigger**: When an exact `uv run ...` verification command remains in `setuptools.build_meta` with a child process in `D` state before the target script or test starts.
- **Instruction**: Capture the stalled process state, terminate the wrapper, record the exact command as failed, and use the repo `.venv/bin/python` fallback only to gather diagnostic evidence.
- **Added after**: Iteration 4 - repeated `uv run` editable-build stalls blocked Stage 2 precondition commands before `stage2_patch1_probe.py` or `narrow_compare.py` executed.

### Sign: Count Wildcard Patch Comparisons
- **Trigger**: Before accepting a `PATCH_*` wildcard parity compare as all-patch evidence.
- **Instruction**: Confirm the compare enumerated the authoritative patch list, such as `patch.list_old` or the audited manifest source, and that the reported checked file count covers every audited patch.
- **Added after**: Iteration 3 - `narrow_compare --patterns 'PATCH_*/pm1.mat'` previously checked only `PATCH_1` because the golden `patch.list` was shortened, hiding PATCH_2+ Stage 2 drift.

### Sign: Prove Stage 2 Before Downstream Repair
- **Trigger**: Before making Stage 3+ changes for downstream parity repair.
- **Instruction**: Run the authoritative all-patch Stage 2 probe and `PATCH_*/pm1.mat` compare first; if any Stage 2 artifact/key still fails, stop and record the blocker instead of changing downstream stages.
- **Added after**: Iteration 4 - US-004 was blocked because all 4 Stage 2 `pm1.mat` comparisons still failed `C_ps`, starting with PATCH_1 max_abs=0.0295872.
