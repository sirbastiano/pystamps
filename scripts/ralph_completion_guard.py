#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


MANDATORY_COMMAND_FRAGMENTS_FOR_US010 = [
    "timeout 180 uv run python -c \"print('uv-smoke-ok')\"",
    "uv run pytest -q tests/test_stage2_ported.py tests/test_stage2_trial_wraps.py tests/test_kernels_accelerated.py",
    "uv run python scripts/stage2_patch1_probe.py --dataset inputs_and_outputs/InSAR_dataset_test_stage8diag --run-root inputs_and_outputs/validation_runs/stage2_manifest_probe",
    "uv run python scripts/narrow_compare.py --run inputs_and_outputs/validation_runs/stage2_manifest_probe --golden inputs_and_outputs/InSAR_dataset_test_stage8diag",
    "TMPDIR=\"$PWD/.tmp_pytest\" uv run pytest -q",
    "uv run jupyter execute --inplace --timeout=-1 notebooks/03_stage_by_stage_oracle.ipynb",
    "uv run python scripts/assert_notebook_parity.py --notebook notebooks/03_stage_by_stage_oracle.ipynb",
    "make audit",
]


def _iter_json_payloads(log_text: str):
    decoder = json.JSONDecoder()
    index = 0
    length = len(log_text)
    while index < length:
        start = log_text.find("{", index)
        if start < 0:
            return
        try:
            payload, end = decoder.raw_decode(log_text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        yield payload
        index = end


def has_successful_stage2_manifest_compare(
    log_text: str,
    *,
    run_root_fragment: str = "inputs_and_outputs/validation_runs/stage2_manifest_probe",
    golden_root_fragment: str = "inputs_and_outputs/InSAR_dataset_test_stage8diag",
    required_pattern: str = "PATCH_*/pm1.mat",
    min_checked: int = 4,
) -> bool:
    for payload in _iter_json_payloads(log_text):
        if not isinstance(payload, dict):
            continue
        if payload.get("label") != "narrow_compare":
            continue

        run_root = str(payload.get("run_root") or "")
        golden_root = str(payload.get("golden_root") or "")
        if run_root_fragment not in run_root:
            continue
        if golden_root_fragment not in golden_root:
            continue

        patterns = payload.get("patterns")
        if not isinstance(patterns, list):
            continue
        if required_pattern not in patterns:
            continue

        checked = payload.get("checked")
        if not isinstance(checked, int) or checked < min_checked:
            continue
        if payload.get("ok") is not True:
            continue
        if payload.get("failures"):
            continue

        return True
    return False


def _line_has_passed_command(log_line: str, command_fragment: str) -> bool:
    if not log_line.lstrip().startswith("- Command: "):
        return False
    if command_fragment not in log_line:
        return False
    if "-> PASS" in log_line:
        return True
    return False


def _line_has_failed_or_skipped_command(log_line: str, command_fragment: str) -> bool:
    if not log_line.lstrip().startswith("- Command: "):
        return False
    if command_fragment not in log_line:
        return False
    return "-> FAIL" in log_line or "-> SKIPPED" in log_line


def completion_is_allowed_for_us010(log_text: str) -> bool:
    for command_fragment in MANDATORY_COMMAND_FRAGMENTS_FOR_US010:
        if not any(_line_has_passed_command(line, command_fragment) for line in log_text.splitlines()):
            return False
        if any(_line_has_failed_or_skipped_command(line, command_fragment) for line in log_text.splitlines()):
            return False
    return True


def completion_is_allowed(story_id: str, log_text: str) -> bool:
    if story_id not in {"US-008", "US-009", "US-010"}:
        return True
    if not has_successful_stage2_manifest_compare(log_text):
        return False
    if story_id == "US-010":
        return completion_is_allowed_for_us010(log_text)
    return True


def _load_log(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate completion allowance from a loop log.")
    parser.add_argument("log_path")
    parser.add_argument("--story-id", required=True)
    args = parser.parse_args(argv)
    log_text = _load_log(args.log_path)
    allowed = completion_is_allowed(args.story_id, log_text)
    return 0 if allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
