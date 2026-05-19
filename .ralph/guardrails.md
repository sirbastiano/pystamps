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

### Sign: Prove Stage 2 Before Notebook Proof
- **Trigger**: Before claiming `notebooks/03_stage_by_stage_oracle.ipynb` proves full parity.
- **Instruction**: Run the authoritative all-patch Stage 2 probe and `PATCH_*/pm1.mat` compare first; if Stage 2 still fails, execute the notebook only to record the blocked proof and do not tune notebook outputs to hide downstream failures.
- **Added after**: Iteration 5 - US-005 exact notebook execution completed, but the parity assertion failed for Stages 2-8 because Stage 2 `C_ps` drift still starts the failure chain.

### Sign: Stop Notebook Audit On Red Stage 2
- **Trigger**: When a US-009 notebook proof run has just failed the immediate manifest-backed `PATCH_*/pm1.mat` Stage 2 compare.
- **Instruction**: Do not run the notebook, notebook parity assertion, or `make audit` as completion proof. Record the Stage 2 blocker, run only scoped regression checks for any local changes, and return to US-008.
- **Added after**: Iteration 4 - US-009 was blocked again because the all-patch Stage 2 compare checked 4 files and failed `C_ps` for every patch.

### Sign: Block Push On Stage 2 Drift
- **Trigger**: Before pushing `main` for a release/audit story.
- **Instruction**: Confirm the manifest-backed all-patch Stage 2 compare passed in the same run. If it failed or was skipped, do not run `git push`; record the blocked push outcome instead.
- **Added after**: Iteration 5 - US-010 release push was correctly blocked because `narrow_compare --patterns 'PATCH_*/pm1.mat'` checked 4 patches and still failed `C_ps`.

### Sign: Validate Notebook Oracle Provenance
- **Trigger**: When a notebook/reference dataset fails all-patch Stage 2 parity after source-level Stage 2 diagnostics point back to the target artifact.
- **Instruction**: Check `pm1.mat` file format/header provenance, `i_loop`, bundled `STAMPS.log`, and the audited workflow manifest before treating the reference as a source-code parity target. Record a fixture blocker if the target is hybrid; do not loosen tolerances, special-case rows, or overwrite oracle values to make the compare pass.
- **Added after**: Iteration 2 - US-008 found `stage8diag_hl/PATCH_2..4/pm1.mat` are March 2026 MATLAB 5 artifacts while the bundled logs and manifest oracle outputs are December 2025 MATLAB 7.3 artifacts.

### Sign: Prove Candidate Fix Is Exercised
- **Trigger**: Before spending a long Stage 2 probe on a candidate C_ps fix.
- **Instruction**: Confirm the changed branch is exercised by the manifest-backed failing patches, using debug counts or a focused probe. If the branch is not exercised, keep the change as diagnostic/correctness work only and continue root-cause analysis instead of expecting the all-patch compare to turn green.
- **Added after**: Iteration 1 - US-008 aligned partial-zero row handling with StaMPS, but all manifest patches had full valid row counts and the all-patch `C_ps` compare stayed red.

### Sign: Do Not Treat Seed Nr Drift As Causal
- **Trigger**: When Stage 2 `C_ps` parity is failing and `Nr` differs by only one random sample bin or a saved seed `pm1.mat` has a tempting random histogram.
- **Instruction**: Do not spend a full all-patch rerun on `Nr` reuse unless a focused probe first changes the failing `C_ps` magnitude. The manifest `C_ps` blocker persists even when PATCH_1 uses the seed `Nr` distribution; continue investigating CLAP/topofit prior-state drift instead.
- **Added after**: Iteration 3 - US-008 seed-`Nr` PATCH_1 rerun completed but still failed `C_ps` with max_abs=0.0295872.
