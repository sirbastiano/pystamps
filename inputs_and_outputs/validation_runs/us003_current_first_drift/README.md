# US-003 First-Drift Baseline

- Source HEAD: `da3bfa9f631756f684377c2415ae7d2e49e7bb1e`
- Fresh run root: `inputs_and_outputs/validation_runs/20260421_153559/InSAR_dataset_test_stage2_8`
- Golden root: `inputs_and_outputs/InSAR_dataset_test`
- Oracle source: `cpp_wrapper` pinned to `c159eb81b16c446e0e8fdef7dd435eb22e0240ed`

## Conclusion

The current first material boundary is stage 2. The fresh stopped run root never emitted `PATCH_1/pm1.mat`, so the saved stage-boundary probes classify `PATCH_1/pm1.mat` as the first failing artifact with `failure_kind=missing_run_artifact`.

`PATCH_1/select1.mat` and `PATCH_1/weed1.mat` are also missing in the same stopped run root, but those stage-3 and stage-4 probe failures are downstream of the missing `pm1.mat` boundary rather than earlier drift.

## Canonical Artifacts

- `first_drift_probe_summary.json`
- `InSAR_dataset_test_first_boundary_trace.json`
- `InSAR_dataset_test_stage2_boundary_probe.json`
- `InSAR_dataset_test_stage3_boundary_probe.json`
- `InSAR_dataset_test_stage4_boundary_probe.json`
- `parity_loop_interrupted.json`
- `validate_audit_interrupted.json`

## Notes

The baseline parity-loop command was started against the current workspace state and then interrupted after the fresh run root remained pre-stage2 for more than twenty minutes. The saved probe JSON was emitted only after the run was stopped so the evidence reflects a stable filesystem snapshot rather than an in-progress run.

The separate repo gate `scripts/validate_audit.py --datasets inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test --output inputs_and_outputs/validation_runs/latest_audit.json` was also run and interrupted after it showed the same early-stage pattern. Its interrupted JSON snapshot is copied here as `validate_audit_interrupted.json`.

See `commands.sh` for the exact command set used to produce this evidence bundle.
