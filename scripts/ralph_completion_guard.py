#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def completion_is_allowed(story_id: str, log_text: str) -> bool:
    if story_id != "US-008":
        return True
    return has_successful_stage2_manifest_compare(log_text)


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
