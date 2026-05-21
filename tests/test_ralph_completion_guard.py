from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_guard_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "ralph_completion_guard.py"
    )
    spec = importlib.util.spec_from_file_location("ralph_completion_guard", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("story_id", ["US-008", "US-009"])
def test_ralph_completion_guard_accepts_green_stage2_manifest_compare(story_id: str) -> None:
    module = _load_guard_module()
    log_text = "\n".join(
        [
            "INFO: starting",
            '{"label":"narrow_compare","run_root":"/tmp/inputs_and_outputs/validation_runs/stage2_manifest_probe","golden_root":"/tmp/inputs_and_outputs/InSAR_dataset_test_stage8diag","patterns":["PATCH_*/pm1.mat"],"ok":true,"checked":4,"failures":[]}',
            "INFO: complete",
            "DONE",
            "",
        ]
    )
    assert module.completion_is_allowed(story_id, log_text)


@pytest.mark.parametrize("story_id", ["US-008", "US-009"])
def test_ralph_completion_guard_rejects_green_compare_with_wrong_artifacts(story_id: str) -> None:
    module = _load_guard_module()
    log_text = (
        '{"label":"narrow_compare","run_root":"/tmp/inputs_and_outputs/validation_runs/stage2_parity_probe","golden_root":"/tmp/inputs_and_outputs/InSAR_dataset_test_stage8diag_hl","patterns":["PATCH_1/pm1.mat"],"ok":true,"checked":1,"failures":[]}'
    )
    assert not module.completion_is_allowed(story_id, log_text)


@pytest.mark.parametrize("story_id", ["US-008", "US-009"])
def test_ralph_completion_guard_rejects_red_stage2_manifest_compare(story_id: str) -> None:
    module = _load_guard_module()
    log_text = (
        '{"label":"narrow_compare","run_root":"/tmp/inputs_and_outputs/validation_runs/stage2_manifest_probe","golden_root":"/tmp/inputs_and_outputs/InSAR_dataset_test_stage8diag","patterns":["PATCH_*/pm1.mat"],"ok":false,"checked":4,"failures":[{"path":"PATCH_1/pm1.mat","message":"C_ps mismatch"}]}'
    )
    assert not module.completion_is_allowed(story_id, log_text)


@pytest.mark.parametrize("story_id", ["US-008", "US-009"])
def test_ralph_completion_guard_rejects_short_stage2_manifest_compare(story_id: str) -> None:
    module = _load_guard_module()
    log_text = (
        '{"label":"narrow_compare","run_root":"/tmp/inputs_and_outputs/validation_runs/stage2_manifest_probe","golden_root":"/tmp/inputs_and_outputs/InSAR_dataset_test_stage8diag","patterns":["PATCH_*/pm1.mat"],"ok":true,"checked":2,"failures":[]}'
    )
    assert not module.completion_is_allowed(story_id, log_text)
