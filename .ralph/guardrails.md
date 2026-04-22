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

### Sign: Follow The Current Audit Contract
- **Trigger**: Running `scripts/validate_audit.py`, `scripts/parity_bug_loop.py`, or heavy `tests/test_validate_audit.py` coverage after parity-manifest changes
- **Instruction**: Read the required dataset set from `pystamps/data/audited_workflow_manifest.json` or `make audit` instead of reusing an older hard-coded dataset pair. If `/tmp` is tight, move `TMPDIR` to a repo-local directory before running the heavy audit pytest groups.
- **Added after**: Iteration 10 - US-010 expanded the audit contract to four required datasets, the stale two-dataset full-validation command failed immediately, and `/tmp` exhaustion broke repeated `validate_audit` pytest copies
