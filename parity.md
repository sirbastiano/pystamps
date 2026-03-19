# Full parity audit status

Status: blocked by environment prerequisites.

## Requested audit scope
- Full parity audit across:
  - `inputs_and_outputs/InSAR_dataset_test_stage8diag`
  - `inputs_and_outputs/InSAR_dataset_test`
- Expected generated evidence artifact:
  - `inputs_and_outputs/validation_runs/latest_audit.json`
- Expected follow-up verify step:
  - `uv run pystamps verify --run <run_root_from_latest_audit> --golden ./inputs_and_outputs/InSAR_dataset_test`

## Prerequisite checks run
```bash
test -d inputs_and_outputs/InSAR_dataset_test_stage8diag && test -d inputs_and_outputs/InSAR_dataset_test && printf 'datasets:ok\n' || printf 'datasets:missing\n'
command -v triangle >/dev/null && printf 'triangle:ok\n' || printf 'triangle:missing\n'
command -v snaphu >/dev/null && printf 'snaphu:ok\n' || printf 'snaphu:missing\n'
```

Observed outcomes:
- `datasets:ok`
- `triangle:missing`
- `snaphu:missing`

## Why the audit was not run
The approved parity plan requires the documented local prerequisites to be present before starting the long-running audit flow. Both external tools required by the repo documentation, `triangle` and `snaphu`, are missing from `PATH`, so the parity workflow is blocked by environment setup.

Because the prerequisite gate failed, the supported audit command was intentionally not started:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  uv run python scripts/validate_audit.py \
    --datasets \
      inputs_and_outputs/InSAR_dataset_test_stage8diag \
      inputs_and_outputs/InSAR_dataset_test \
    --output inputs_and_outputs/validation_runs/latest_audit.json
```

The explicit verify step was also intentionally not run, because it depends on a fresh `latest_audit.json` and the recorded `run_root` for `InSAR_dataset_test`.

## Exact blocked status
- Result: `blocked`
- Classification: `environment/prerequisites`
- Parity state: not executed
- Audit artifact: not generated in this run
- Verify state: not executed

## Unblock conditions
Install or expose both `triangle` and `snaphu` on `PATH`, then rerun the documented audit flow to generate a fresh `latest_audit.json` and use its recorded `run_root` for the verify step.
