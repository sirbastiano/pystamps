# AGENTS.md

## Goal
Ship correct, minimal, verifiable changes. Prefer root-cause fixes. Keep diffs and words small.

## Brevity
- Keep replies, plans, and handoffs short.
- Prefer the shortest answer that fully satisfies the request; minimize token usage.
- Do not restate the request or repo context unless needed to act.
- Read only the files and lines needed for the current step.
- Stop searching once there is enough evidence to act.
- Summarize tool output; avoid long quotes and log dumps.

## Fast Repo Work
- Discover commands before editing from CI, task runners, and local docs.
- Prefer `rg`, `fd` or `rg --files`, and `jq`.
- Avoid `grep -R`, `find`, `ls -R`, and broad reads.
- For full-validation parity runs, take the required dataset list from `pystamps/data/audited_workflow_manifest.json` or `make audit`; the older two-dataset command is stale once new audited targets are added.
- When running pytest groups that include `tests/test_validate_audit.py`, prefer a repo-local `TMPDIR` if `/tmp` is space-constrained.

## Routing
- `implement`: code, tests, config, or behavior changes.
- `plan`: non-trivial, cross-file, risky, or unclear work.
- `docs`: docs, examples, changelog, or migration notes only.
- `publish`: final gate and any change to `AGENTS.md`, `.agents/skills/**`, `.github/skills/**`, `.github/agents/**`, or `.codex/**`.
- Use one primary skill. Add downstream skills only when needed.

## Done
- requested behavior is correct
- diff is minimal and fits local patterns
- validation matches blast radius
- skipped checks and residual risk are disclosed

## Instruction Changes
- For instruction-file changes, include one positive and one negative routing check in the handoff.
- If new catalog artifacts are added, validate required contract fields.

## Precedence
- deeper `AGENTS.md` or `AGENTS.override.md` override this file for their subtree
- user instructions override repo instructions
- code, tests, CI, and build config beat stale docs
