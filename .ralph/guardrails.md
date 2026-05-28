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

### Sign: Preserve Stage 2 Iteration Snapshots
- **Trigger**: When changing Stage 2 CLAP, topofit candidate selection, random histograms, or P-square weighting
- **Instruction**: Compare per-iteration `pm1_iter_##.mat` and weighting snapshots before relying on the final `pm1.mat`; small early CLAP/topofit drift can be amplified by P-square weighting into different candidate rows.
- **Added after**: Iteration 5 - US-005 parity gates repeatedly narrowed to PATCH_1 iteration-7 weighting count drift and PATCH_2/PATCH_3 one-bin `Nr` shifts.
- **Example**: Use `PYSTAMPS_STAGE2_NATIVE_DEBUG_PM=1` on a single PATCH_1 run, then compare against `inputs_and_outputs/validation_runs/stage2_manifest_probe_final/PATCH_1/pm1_iter_*.mat` and `inputs_and_outputs/validation_runs/stage2_weighting_snapshot.json`.

### Sign: Isolate Stage 6 From Stale Upstream Artifacts
- **Trigger**: When validating Stage 6 unwrap parity or performance.
- **Instruction**: Confirm `rc2.mat`, `ps2.mat`, `ph2.mat`, `pm2.mat`, and `bp2.mat` come from an upstream-parity source before treating Stage 6 value or shape mismatches as Stage 6 bugs. The checked-in validation tree can contain stale merged artifacts, so record Stage 6 budget evidence separately from full-chain upstream failures.
- **Added after**: Iteration 9 - US-009 Stage 6-only runs on the checked-in validation tree used an incompatible `rc2.mat` row count and failed fallback-path parity/performance, while full-chain Stage 6 completed under 30s but inherited out-of-scope Stage 5 drift.
- **Example**: Check `rc2.ph_rc` orientation/row count against `ps2.n_ps` before debugging unwrap math; use the full-chain timing report to distinguish Stage 6 budget status from Stage 5/7/8 budget failures.

### Sign: Treat Stage 8 Noise As Time-Smoothing Parity
- **Trigger**: When changing `uw_space_time.mat/dph_noise` or claiming Stage 8 output parity.
- **Instruction**: Verify against the STAMPS `uw_unwrap_space_time` time-smoothing semantics, not only the accelerated placeholder kernel. Sparse structure and direct edge phase parity are insufficient; `dph_noise` must pass the manifest verifier before the story is complete.
- **Added after**: Iteration 11 - US-011 produced sparse `spread` and budget-compliant focused Stage 8 runs, but repeated verifier failures remained for `uw_space_time.mat/dph_noise`.
- **Example**: Run `make native-full-chain-verify START_STEP=8 END_STEP=8 RUN=inputs_and_outputs/validation_runs/us011_stage8_sparse_verify` and inspect `_native_gate_reports/native-verify-report.json` for `merged_uw_space_time.dph_noise.phase_modulo_f32`.

### Sign: Separate Focused Stage Gates From Full-Chain Upstream Drift
- **Trigger**: When a selected downstream story passes its focused stage verifier but exact `make native-full-chain-verify` fails in earlier stages or total runtime.
- **Instruction**: Record the exact full-chain failure, but do not treat upstream Stage 5/full-run budget drift as evidence against the focused downstream implementation. Use a stage-scoped run with `START_STEP`/`END_STEP` and a dedicated `RUN` path for the story acceptance signal.
- **Added after**: Iteration 1 - US-011 focused Stage 8 verification passed parity and the 25s budget, while exact full-chain verification still failed out-of-scope release runtime and Stage 5 merged budgets.
- **Example**: For Stage 8, use `make native-full-chain-verify START_STEP=8 END_STEP=8 RUN=inputs_and_outputs/validation_runs/us011_stage8_final_verify6` and compare its Stage 8 timing/parity report against the full-chain timing report.

### Sign: Keep Coverage Stories Out Of Stage Performance Repairs
- **Trigger**: When a coverage/native-only enforcement story passes its coverage gate but exact full-chain verification fails on a stage-local performance budget.
- **Instruction**: Record the stage budget failure and coverage result separately; do not widen the coverage story into Stage 5/6/7/8 performance work unless that stage is explicitly selected.
- **Added after**: Iteration 2 - US-012 native coverage precheck passed twice, but exact full-chain verification failed before parity comparison on existing Stage 5 merged budget drift.
- **Example**: For US-012, cite `_native_gate_reports/native-coverage-report.json` as the coverage signal and record the separate Stage 5 merged duration budget failure from `_native_gate_reports/native-run-timings.json`.

### Sign: Keep Documentation Stories Out Of Native Budget Repairs
- **Trigger**: When a documentation/setup workflow story completes its docs changes and local validation, but exact `make native-full-chain-verify` fails on an existing native runtime or stage budget.
- **Instruction**: Record the exact budget failure and report paths, but do not change native stage code, performance budgets, or tolerance manifests unless that repair is the selected story.
- **Added after**: Iteration 4 - US-014 documented VM setup and native run workflow, while exact full-chain verification again failed on release runtime and Stage 5 merged duration budgets.
- **Example**: For docs-only workflow stories, cite `README.md`, `Makefile`, and `RUN/_native_gate_reports/native-run-timings.json` as evidence, then leave Stage 5 performance work to a Stage 5 story.
