#!/usr/bin/env bash

# Fresh baseline parity-loop command.
# This run was interrupted after the fresh run root remained pre-stage2
# for more than twenty minutes and still had no PATCH_1/pm1.mat.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
uv run python scripts/parity_bug_loop.py \
  --datasets inputs_and_outputs/InSAR_dataset_test \
  --allow-subset \
  --output inputs_and_outputs/validation_runs/us003_current_first_drift/parity_loop.json \
  --audit-output inputs_and_outputs/validation_runs/us003_current_first_drift/parity_audit.json

# Stable stage-boundary probes emitted from the stopped fresh run root.
PYTHONPATH=. uv run python - <<'PY'
import json
from pathlib import Path
from pystamps.config import RunConfig
import scripts.validate_audit as va

run_root = Path('inputs_and_outputs/validation_runs/20260421_153559/InSAR_dataset_test_stage2_8').resolve()
golden_root = Path('inputs_and_outputs/InSAR_dataset_test').resolve()
output_dir = Path('inputs_and_outputs/validation_runs/us003_current_first_drift').resolve()
contract = va._resolve_contract()
probes, first_trace, first_trace_path = va._emit_stage_boundary_traces(
    run_root,
    golden_root,
    contract,
    RunConfig().tolerance,
    audit_stamp='us003_current_first_drift',
    generation={'validation_run_dir': str(output_dir)},
)
summary = {
    'generated_at_utc': va._now_utc(),
    'run_root': str(run_root),
    'golden_root': str(golden_root),
    'stage_boundary_probes': [probe['output_path'] for probe in probes],
    'first_divergent_boundary': first_trace,
    'first_divergent_boundary_output_path': first_trace_path,
}
(output_dir / 'first_drift_probe_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
PY

# Required repo gate run for the maintained audit entrypoint.
# This run was interrupted after the fresh 20260421_160012 stage8diag run root
# showed the same pre-stage2 pattern, and its output was copied to
# validate_audit_interrupted.json.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
uv run python scripts/validate_audit.py \
  --datasets inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test \
  --output inputs_and_outputs/validation_runs/latest_audit.json
