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
