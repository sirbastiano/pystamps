# Oracle-backed parity contract

This document records the final repo-tracked contract for parity work. Future changes should start here instead of re-deriving command ownership or keeping story-specific debugging notes around.

## Authoritative files

- `pystamps/data/oracle_contract.json` owns the pinned oracle sources and precedence rule.
- `pystamps/data/audited_workflow_manifest.json` owns the required done-gate datasets, run seeds, and workflow profiles.
- `scripts/validate_audit.py` is the authoritative parity-audit driver and writes `inputs_and_outputs/validation_runs/latest_audit.json`.
- `scripts/parity_bug_loop.py` is the follow-up triage surface that consumes audit output and selects the next first-boundary target.
- `make audit` and `make parity-loop` are convenience wrappers around those script entrypoints, not separate contracts.

## Workflow ownership

- `inputs_and_outputs/InSAR_dataset_test_stage8diag`
  - Purpose: compact single-master diagnostic audit target
  - Workflow profile: `default`
  - Run seed: dataset root itself
- `inputs_and_outputs/InSAR_dataset_test`
  - Purpose: full single-master done-gate comparison target
  - Workflow profile: `legacy_post`
  - Run seed: `inputs_and_outputs/RUN_FULL_GATE_1e10`
- `inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag`
  - Purpose: compact small-baseline stage-7 audit target
  - Workflow profile: `small_baseline`
  - Run seed: dataset root itself
- `inputs_and_outputs/InSAR_dataset_small_baseline_stage7`
  - Purpose: full small-baseline stage-7 done-gate target
  - Workflow profile: `small_baseline`
  - Run seed: dataset root itself

The audited workflow manifest is the source of truth for these mappings. Do not replace it with a shorter ad hoc dataset list.

## Audit process

Run the local wrapper:

```bash
make audit
```

Or run the authoritative driver directly:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  uv run python scripts/validate_audit.py \
    --output inputs_and_outputs/validation_runs/latest_audit.json
```

Key rules:

- Omitting `--datasets` makes `scripts/validate_audit.py` use the manifest-backed required dataset set.
- `latest_audit.json` is the active audit artifact and records `run_root`, `workflow_profile`, `failed_workflows`, and first-boundary trace metadata.
- Explicit verification must use the `run_root` recorded in that fresh audit artifact.
- Interrupted audits, manual restart paths, or stale run-copy reuse are not valid evidence.

Example verify step:

```bash
RUN_COPY="$(python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path('inputs_and_outputs/validation_runs/latest_audit.json').read_text(encoding='utf-8'))
print(next(audit['run_root'] for audit in payload['audits'] if audit['dataset'] == 'InSAR_dataset_test'))
PY
)"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  uv run pystamps verify --run "$RUN_COPY" --golden ./inputs_and_outputs/InSAR_dataset_test
```

## Oracle precedence

`pystamps/data/oracle_contract.json` defines the precedence order:

1. `cpp_wrapper`
2. `matlab_source`
3. `manual_references`

When the pinned StaMPS C/C++ helper behavior intentionally differs from the pinned MATLAB scripts, the wrapper-backed path is the practical parity oracle. Audit traces should record that source instead of treating plain MATLAB as authoritative in those cases.

## Rules for future parity changes

- Keep only oracle-backed diffs that improve the first materially divergent boundary.
- Do not keep speculative downstream fixes that merely move the failure to a later artifact.
- Use the trace from `latest_audit.json` or `latest_parity_loop.json` to justify every parity edit.
- Remove temporary story-specific debugging notes once the audited contract is understood and documented here.
