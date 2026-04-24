from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace


def _load_validate_audit_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_audit.py"
    spec = importlib.util.spec_from_file_location("validate_audit", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_test_tree(src: Path, dst: Path) -> None:
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def test_validate_audit_writes_contract_and_passes(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset_a = inputs_root / "InSAR_dataset_test_stage8diag"
    dataset_b = inputs_root / "InSAR_dataset_test"
    dataset_a.mkdir(parents=True)
    dataset_b.mkdir(parents=True)
    output = tmp_path / "latest_audit.json"

    contract = {
        "required_dataset_paths": [
            "inputs_and_outputs/InSAR_dataset_test_stage8diag",
            "inputs_and_outputs/InSAR_dataset_test",
        ]
    }

    monkeypatch.setattr(module, "_resolve_contract", lambda: contract)
    monkeypatch.setattr(module, "capture_code_state", lambda repo_root: {"git_commit": "abc123", "git_dirty": False})
    monkeypatch.setattr(
        module,
        "_repo_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            datasets=None,
            golden_root=None,
            output=str(output),
        ),
    )
    monkeypatch.setattr(
        module,
        "_prepare_run_selection",
        lambda dataset_root, golden_base, audit_stamp: (dataset_root, dataset_root, "test_run_root", {"start_step": 2}),
    )

    def fake_verify(run_root: Path, golden_root: Path, tolerance) -> SimpleNamespace:
        return SimpleNamespace(ok=True, comparisons=[object(), object()])

    monkeypatch.setattr(module, "verify_run_against_golden", fake_verify)
    monkeypatch.setattr(
        module,
        "summarize_failures",
        lambda report: {"failed": 0, "failures": [], "groups": [], "trace": {"stage3_4_residual_present": False}},
    )

    exit_code = module.main()

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["completed"] is True
    assert payload["interrupted"] is False
    assert payload["failed_workflows"] == []
    assert [audit["run_source"] for audit in payload["audits"]] == ["test_run_root", "test_run_root"]
    assert [audit["run_generation"]["start_step"] for audit in payload["audits"]] == [2, 2]
    assert [audit["workflow"] for audit in payload["audits"]] == [
        "InSAR_dataset_test_stage8diag_audit",
        "InSAR_dataset_test_audit",
    ]
    assert payload["code_state"] == {"git_commit": "abc123", "git_dirty": False}
    assert payload["contract"] == contract


def test_validate_audit_fails_fast_when_required_dataset_missing(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset_a = inputs_root / "InSAR_dataset_test_stage8diag"
    dataset_a.mkdir(parents=True)
    dataset_b = inputs_root / "InSAR_dataset_test"
    output = tmp_path / "latest_audit.json"

    contract = {
        "required_dataset_paths": [
            "inputs_and_outputs/InSAR_dataset_test_stage8diag",
            "inputs_and_outputs/InSAR_dataset_test",
        ]
    }

    monkeypatch.setattr(module, "_resolve_contract", lambda: contract)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            datasets=None,
            golden_root=None,
            output=str(output),
        ),
    )

    called = {"verify": 0}

    def fake_verify(run_root: Path, golden_root: Path, tolerance) -> SimpleNamespace:
        called["verify"] += 1
        return SimpleNamespace(ok=True, comparisons=[])

    monkeypatch.setattr(module, "verify_run_against_golden", fake_verify)

    exit_code = module.main()

    assert exit_code == 1
    assert called["verify"] == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["completed"] is False
    assert payload["interrupted"] is False
    assert payload["audits"] == []
    assert payload["missing_datasets"] == [str(dataset_b.resolve())]
    assert payload["failed_workflows"] == ["full_validation"]
    assert payload["interruption"]["kind"] == "missing_dataset"


def test_validate_audit_records_interruption(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset_a = inputs_root / "InSAR_dataset_test_stage8diag"
    dataset_b = inputs_root / "InSAR_dataset_test"
    dataset_a.mkdir(parents=True)
    dataset_b.mkdir(parents=True)
    output = tmp_path / "latest_audit.json"

    contract = {
        "required_dataset_paths": [
            "inputs_and_outputs/InSAR_dataset_test_stage8diag",
            "inputs_and_outputs/InSAR_dataset_test",
        ]
    }

    monkeypatch.setattr(module, "_resolve_contract", lambda: contract)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_prepare_run_selection",
        lambda dataset_root, golden_base, audit_stamp: (dataset_root, dataset_root, "test_run_root", None),
    )
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            datasets=None,
            golden_root=None,
            output=str(output),
        ),
    )

    calls = {"count": 0}

    def fake_verify(run_root: Path, golden_root: Path, tolerance) -> SimpleNamespace:
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt
        return SimpleNamespace(ok=True, comparisons=[object()])

    monkeypatch.setattr(module, "verify_run_against_golden", fake_verify)
    monkeypatch.setattr(
        module,
        "summarize_failures",
        lambda report: {"failed": 0, "failures": [], "groups": [], "trace": {}},
    )

    exit_code = module.main()

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["completed"] is False
    assert payload["interrupted"] is True
    assert payload["failed_workflows"] == ["full_validation"]
    assert payload["interruption"]["kind"] == "keyboard_interrupt"
    assert [audit["workflow"] for audit in payload["audits"]] == ["InSAR_dataset_test_stage8diag_audit"]


def test_validate_audit_rejects_unsupported_dataset_selection(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    output = tmp_path / "latest_audit.json"
    rogue_dataset = tmp_path / "rogue_dataset"
    rogue_dataset.mkdir()

    contract = {
        "required_dataset_paths": [
            "inputs_and_outputs/InSAR_dataset_test_stage8diag",
            "inputs_and_outputs/InSAR_dataset_test",
        ]
    }

    monkeypatch.setattr(module, "_resolve_contract", lambda: contract)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            datasets=[str(rogue_dataset)],
            golden_root=None,
            output=str(output),
        ),
    )

    exit_code = module.main()

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["failed_workflows"] == ["full_validation"]
    assert payload["interruption"]["kind"] == "unsupported_dataset_selection"
    assert payload["interruption"]["extra_datasets"] == ["rogue_dataset"]


def test_validate_audit_allows_required_subset_when_flag_enabled(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset_a = inputs_root / "InSAR_dataset_test_stage8diag"
    dataset_b = inputs_root / "InSAR_dataset_test"
    dataset_a.mkdir(parents=True)
    dataset_b.mkdir(parents=True)
    output = tmp_path / "latest_audit.json"

    contract = {
        "required_dataset_paths": [
            "inputs_and_outputs/InSAR_dataset_test_stage8diag",
            "inputs_and_outputs/InSAR_dataset_test",
        ]
    }

    monkeypatch.setattr(module, "_resolve_contract", lambda: contract)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            datasets=[str(dataset_a)],
            golden_root=None,
            output=str(output),
            allow_subset=True,
        ),
    )
    monkeypatch.setattr(
        module,
        "_prepare_run_selection",
        lambda dataset_root, golden_base, audit_stamp: (dataset_root, dataset_root, "test_run_root", None),
    )
    monkeypatch.setattr(
        module,
        "verify_run_against_golden",
        lambda run_root, golden_root, tolerance: SimpleNamespace(ok=True, comparisons=[object()]),
    )
    monkeypatch.setattr(
        module,
        "summarize_failures",
        lambda report: {"failed": 0, "failures": [], "groups": [], "trace": {}},
    )

    exit_code = module.main()

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["completed"] is True
    assert payload["missing_datasets"] == []
    assert payload["interrupted"] is False
    assert [audit["dataset"] for audit in payload["audits"]] == ["InSAR_dataset_test_stage8diag"]


def test_validate_audit_fails_when_required_run_copy_is_missing(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset_a = inputs_root / "InSAR_dataset_test_stage8diag"
    dataset_b = inputs_root / "InSAR_dataset_test"
    dataset_a.mkdir(parents=True)
    dataset_b.mkdir(parents=True)
    output = tmp_path / "latest_audit.json"

    contract = {
        "required_dataset_paths": [
            "inputs_and_outputs/InSAR_dataset_test_stage8diag",
            "inputs_and_outputs/InSAR_dataset_test",
        ]
    }

    monkeypatch.setattr(module, "_resolve_contract", lambda: contract)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_prepare_run_selection",
        lambda dataset_root, golden_base, audit_stamp: (_ for _ in ()).throw(FileNotFoundError("missing run copy")),
    )
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            datasets=None,
            golden_root=None,
            output=str(output),
        ),
    )

    exit_code = module.main()

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["completed"] is False
    assert payload["interrupted"] is True
    assert payload["failed_workflows"] == ["full_validation"]
    assert payload["interruption"]["kind"] == "missing_run_copy"


def test_validate_audit_records_run_copy_generation_failure(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset_a = inputs_root / "InSAR_dataset_test_stage8diag"
    dataset_b = inputs_root / "InSAR_dataset_test"
    dataset_a.mkdir(parents=True)
    dataset_b.mkdir(parents=True)
    output = tmp_path / "latest_audit.json"

    contract = {
        "required_dataset_paths": [
            "inputs_and_outputs/InSAR_dataset_test_stage8diag",
            "inputs_and_outputs/InSAR_dataset_test",
        ]
    }

    monkeypatch.setattr(module, "_resolve_contract", lambda: contract)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_prepare_run_selection",
        lambda dataset_root, golden_base, audit_stamp: (_ for _ in ()).throw(
            module.RunCopyGenerationError(
                "generation failed",
                debug_artifacts={"stage6_debug": {"phase": "snaphu_loop", "ifg_completed": 3}},
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            datasets=None,
            golden_root=None,
            output=str(output),
        ),
    )

    exit_code = module.main()

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["completed"] is False
    assert payload["interrupted"] is True
    assert payload["failed_workflows"] == ["full_validation"]
    assert payload["interruption"]["kind"] == "run_copy_generation_failed"
    assert payload["interruption"]["debug_artifacts"]["stage6_debug"]["phase"] == "snaphu_loop"


def test_resolve_run_selection_prefers_explicit_and_latest_validation_copy(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    validation_runs = inputs_root / "validation_runs"
    dataset_test = inputs_root / "InSAR_dataset_test"
    dataset_stage8 = inputs_root / "InSAR_dataset_test_stage8diag"
    dataset_test.mkdir(parents=True)
    dataset_stage8.mkdir(parents=True)
    explicit = inputs_root / "RUN_FULL_GATE_1e10"
    explicit.mkdir()
    older_stage1 = validation_runs / "20260306_112145" / "InSAR_dataset_test_stage8diag_stage1_8"
    latest_stage1 = validation_runs / "20260313_010004" / "InSAR_dataset_test_stage8diag_stage1_8"
    older_stage1.mkdir(parents=True)
    latest_stage1.mkdir(parents=True)

    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)

    run_root, golden_root, run_source = module._resolve_run_selection(dataset_test, None)
    assert run_root == explicit.resolve()
    assert golden_root == dataset_test.resolve()
    assert run_source == "resolved_full_loop_run_copy"

    run_root, golden_root, run_source = module._resolve_run_selection(dataset_stage8, None)
    assert run_root == latest_stage1.resolve()
    assert golden_root == dataset_stage8.resolve()
    assert run_source == "resolved_full_loop_run_copy"


def test_build_run_copy_uses_stage2_when_stage1_artifacts_exist(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    dataset = tmp_path / "inputs_and_outputs" / "InSAR_dataset_test_stage8diag"
    patch = dataset / "PATCH_1"
    patch.mkdir(parents=True)
    for filename in ("ps1.mat", "ph1.mat", "bp1.mat", "da1.mat", "hgt1.mat", "pm1.mat", "select1.mat", "weed1.mat"):
        (patch / filename).write_text("stub", encoding="utf-8")

    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_inputs_root", lambda: tmp_path / "inputs_and_outputs")
    monkeypatch.setattr(module, "_copy_dataset", _copy_test_tree)

    captured: dict[str, object] = {}

    def fake_run_pipeline(context):
        captured["dataset_root"] = context.dataset_root
        captured["start_step"] = context.start_step
        captured["end_step"] = context.end_step
        captured["workflow_profile"] = context.workflow_profile
        stage6_debug_path = Path(os.environ[module._STAGE6_DEBUG_ENV])
        stage6_debug_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "phase": "completed",
                    "timings_sec": {"snaphu_loop": 12.5},
                    "ifg_completed": 4,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(failures=[])

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)

    run_root, generation = module._build_run_copy(dataset, "20260314_120000")

    assert run_root.name == "InSAR_dataset_test_stage8diag_stage2_8"
    assert generation["start_step"] == 2
    assert generation["end_step"] == 8
    assert generation["workflow_profile"] == "default"
    assert "PATCH_*/pm1.mat" in generation["clean_patterns"]
    assert captured["dataset_root"] == run_root
    assert captured["start_step"] == 2
    assert captured["end_step"] == 8
    assert captured["workflow_profile"] == "default"
    assert not (run_root / "PATCH_1" / "pm1.mat").exists()
    assert not (run_root / "PATCH_1" / "select1.mat").exists()
    assert not (run_root / "PATCH_1" / "weed1.mat").exists()
    assert generation["stage6_debug"]["status"] == "completed"
    assert generation["stage6_debug"]["timings_sec"]["snaphu_loop"] == 12.5


def test_build_run_copy_uses_run_full_gate_seed_for_dataset_test(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset = inputs_root / "InSAR_dataset_test"
    dataset.mkdir(parents=True)
    seed = inputs_root / "RUN_FULL_GATE_1e10"
    patch = seed / "PATCH_1"
    patch.mkdir(parents=True)
    for name in ("PATCH_2", "PATCH_3", "PATCH_4"):
        (seed / name).mkdir(parents=True)
    (seed / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    (seed / "patch.list_old").write_text("PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n", encoding="utf-8")
    for filename in ("select1.mat", "weed1.mat", "pm2.mat", "ps2.mat", "phuw2.mat", "scla2.mat", "mean_v.mat"):
        (patch / filename).write_text("stub", encoding="utf-8")
    (seed / "pm2.mat").write_text("stub", encoding="utf-8")
    (seed / "phuw2.mat").write_text("stub", encoding="utf-8")
    (seed / "scla2.mat").write_text("stub", encoding="utf-8")
    (seed / "mean_v.mat").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(module, "_inputs_root", lambda: inputs_root)
    monkeypatch.setattr(module, "_seed_root_for_dataset", lambda dataset_root: seed.resolve())
    monkeypatch.setattr(module, "_copy_dataset", _copy_test_tree)
    captured: dict[str, object] = {}

    def fake_run_pipeline(context):
        captured["workflow_profile"] = context.workflow_profile
        return SimpleNamespace(failures=[])

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)

    run_root, generation = module._build_run_copy(dataset, "20260314_120000")

    assert run_root.name == "InSAR_dataset_test_stage4_8"
    assert generation["start_step"] == 4
    assert generation["end_step"] == 8
    assert generation["seed_name"] == "RUN_FULL_GATE_1e10"
    assert generation["workflow_profile"] == "legacy_post"
    assert "scla_smooth2.mat" not in generation["clean_patterns"]
    assert captured["workflow_profile"] == "legacy_post"
    assert Path(generation["seed_root"]) == seed.resolve()
    assert (run_root / "patch.list").read_text(encoding="utf-8") == "PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n"
    assert (run_root / "PATCH_2").exists()
    assert not (run_root / "PATCH_1" / "weed1.mat").exists()
    assert not (run_root / "PATCH_1" / "pm2.mat").exists()
    assert not (run_root / "pm2.mat").exists()
    assert not (run_root / "phuw2.mat").exists()


def test_build_run_copy_prefers_stage5_when_run_full_gate_seed_has_stage1_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset = inputs_root / "InSAR_dataset_test"
    dataset.mkdir(parents=True)
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    seed = inputs_root / "RUN_FULL_GATE_1e10"
    patch = seed / "PATCH_1"
    patch.mkdir(parents=True)
    (seed / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    (seed / "patch.list_old").write_text("PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n", encoding="utf-8")
    for name in ("PATCH_2", "PATCH_3", "PATCH_4"):
        (seed / name).mkdir(parents=True)
    for filename in (
        "ps1.mat",
        "ph1.mat",
        "bp1.mat",
        "da1.mat",
        "hgt1.mat",
        "pm1.mat",
        "select1.mat",
        "weed1.mat",
        "ps2.mat",
        "ph2.mat",
        "pm2.mat",
    ):
        (patch / filename).write_text("stub", encoding="utf-8")

    monkeypatch.setattr(module, "_inputs_root", lambda: inputs_root)
    monkeypatch.setattr(module, "_seed_root_for_dataset", lambda dataset_root: seed.resolve())
    monkeypatch.setattr(module, "_copy_dataset", _copy_test_tree)
    captured: dict[str, object] = {}

    def fake_run_pipeline(context):
        captured["workflow_profile"] = context.workflow_profile
        return SimpleNamespace(failures=[])

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)

    run_root, generation = module._build_run_copy(dataset, "20260314_120000")

    assert run_root.name == "InSAR_dataset_test_stage5_8"
    assert generation["start_step"] == 5
    assert generation["end_step"] == 8
    assert generation["seed_name"] == "RUN_FULL_GATE_1e10"
    assert generation["workflow_profile"] == "legacy_post"
    assert captured["workflow_profile"] == "legacy_post"
    assert Path(generation["seed_root"]) == seed.resolve()
    assert (run_root / "patch.list").read_text(encoding="utf-8") == "PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n"
    assert (run_root / "PATCH_2").exists()
    assert (run_root / "PATCH_1" / "pm1.mat").exists()
    assert (run_root / "PATCH_1" / "select1.mat").exists()
    assert (run_root / "PATCH_1" / "weed1.mat").exists()
    assert (run_root / "PATCH_1" / "ps2.mat").exists()
    assert (run_root / "PATCH_1" / "ph2.mat").exists()
    assert (run_root / "PATCH_1" / "pm2.mat").exists()
    assert not (run_root / "ps2.mat").exists()
    assert not (run_root / "ph2.mat").exists()
    assert not (run_root / "pm2.mat").exists()


def test_build_run_copy_uses_manifest_stage7_small_baseline_profile(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset = inputs_root / "InSAR_dataset_small_baseline_stage7diag"
    dataset.mkdir(parents=True)
    (dataset / "patch.list").write_text("", encoding="utf-8")
    for filename in ("ps2.mat", "phuw2.mat", "ifgstd2.mat", "parms.mat", "scla2.mat", "scla_smooth2.mat"):
        (dataset / filename).write_text("stub", encoding="utf-8")

    contract = {
        "audited_workflow_manifest": {
            "workflow_targets": [
                {
                    "id": "small_baseline_diagnostic",
                    "local_dataset_path": "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag",
                    "run_seed_path": "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag",
                    "workflow_profile": "small_baseline",
                    "audit_start_step": 7,
                    "audit_end_step": 7,
                }
            ]
        }
    }

    monkeypatch.setattr(module, "_resolve_contract", lambda: contract)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_inputs_root", lambda: inputs_root)
    monkeypatch.setattr(module, "_copy_dataset", _copy_test_tree)
    captured: dict[str, object] = {}

    def fake_run_pipeline(context):
        captured["dataset_root"] = context.dataset_root
        captured["start_step"] = context.start_step
        captured["end_step"] = context.end_step
        captured["workflow_profile"] = context.workflow_profile
        return SimpleNamespace(failures=[])

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)

    run_root, generation = module._build_run_copy(dataset, "20260422_150000")

    assert run_root.name == "InSAR_dataset_small_baseline_stage7diag_stage7_7"
    assert generation["start_step"] == 7
    assert generation["end_step"] == 7
    assert generation["workflow_profile"] == "small_baseline"
    assert generation["seed_name"] == "InSAR_dataset_small_baseline_stage7diag"
    assert generation["clean_patterns"] == ["scla2.mat", "scla_smooth2.mat"]
    assert captured["dataset_root"] == run_root
    assert captured["start_step"] == 7
    assert captured["end_step"] == 7
    assert captured["workflow_profile"] == "small_baseline"
    assert (run_root / "phuw2.mat").exists()
    assert not (run_root / "scla2.mat").exists()
    assert not (run_root / "scla_smooth2.mat").exists()


def test_align_run_copy_replaces_hardlinked_patch_list(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    dataset = tmp_path / "dataset"
    run_root = tmp_path / "run"
    (dataset / "PATCH_1").mkdir(parents=True)
    (dataset / "PATCH_2").mkdir(parents=True)
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    shutil.copytree(dataset, run_root, copy_function=os.link)

    assert os.path.samefile(dataset / "patch.list", run_root / "patch.list")

    module._align_run_copy_with_dataset(run_root, dataset)

    assert (run_root / "patch.list").read_text(encoding="utf-8") == "PATCH_1\n"
    assert not os.path.samefile(dataset / "patch.list", run_root / "patch.list")
    assert not (run_root / "PATCH_2").exists()


def test_align_run_copy_removes_dataset_patch_dirs_missing_from_patch_list(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    dataset = tmp_path / "dataset"
    run_root = tmp_path / "run"
    for name in ("PATCH_1", "PATCH_2", "PATCH_3", "PATCH_4"):
        (dataset / name).mkdir(parents=True, exist_ok=True)
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    shutil.copytree(dataset, run_root)

    module._align_run_copy_with_dataset(run_root, dataset)

    assert (run_root / "patch.list").read_text(encoding="utf-8") == "PATCH_1\n"
    assert not (run_root / "PATCH_2").exists()
    assert not (run_root / "PATCH_3").exists()
    assert not (run_root / "PATCH_4").exists()


def test_align_run_copy_legacy_post_restores_patch_list_old(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    dataset = tmp_path / "dataset"
    run_root = tmp_path / "run"
    for name in ("PATCH_1", "PATCH_2", "PATCH_3", "PATCH_4"):
        (dataset / name).mkdir(parents=True, exist_ok=True)
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    shutil.copytree(dataset, run_root)
    (run_root / "patch.list_old").write_text("PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n", encoding="utf-8")

    module._align_run_copy_with_dataset(run_root, dataset, "legacy_post")

    assert (run_root / "patch.list").read_text(encoding="utf-8") == "PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n"
    assert (run_root / "PATCH_2").exists()
    assert (run_root / "PATCH_3").exists()
    assert (run_root / "PATCH_4").exists()


def test_dataset_audit_emits_saved_stage_boundary_trace(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    run_root = tmp_path / "run"
    golden_root = tmp_path / "golden"
    run_root.mkdir()
    golden_root.mkdir()
    generation = {"validation_run_dir": str(tmp_path / "validation_runs"), "start_step": 2}
    contract = {
        "oracle_contract": {
            "cpp_wrapper": {
                "repository_url": "https://example.invalid/stamps",
                "pinned_revision": "abc123",
            },
            "precedence_rule": {"ordered_sources": ["cpp_wrapper"]},
        }
    }

    def fake_verify(run_root_arg, golden_root_arg, tolerance, patterns=None):
        pattern_tuple = tuple(patterns) if patterns is not None else None
        if pattern_tuple == module.STAGE2_VERIFY_PATTERNS:
            comparisons = []
        elif pattern_tuple == module.STAGE3_VERIFY_PATTERNS:
            comparisons = [
                SimpleNamespace(
                    ok=False,
                    relative_path="PATCH_1/select1.mat",
                    message="Value mismatch for key 'C_ps2', max_abs=2.5",
                )
            ]
        elif pattern_tuple == module.STAGE4_VERIFY_PATTERNS:
            comparisons = []
        else:
            comparisons = [
                SimpleNamespace(
                    ok=False,
                    relative_path="PATCH_1/select1.mat",
                    message="Value mismatch for key 'C_ps2', max_abs=2.5",
                ),
                SimpleNamespace(
                    ok=False,
                    relative_path="uw_space_time.mat",
                    message="Shape mismatch for key 'dph_noise': (3, 4) != (5, 4)",
                ),
            ]
        return SimpleNamespace(ok=not comparisons, comparisons=comparisons)

    def fake_summarize(report):
        failures = []
        for comparison in report.comparisons:
            if comparison.relative_path == "PATCH_1/select1.mat":
                failures.append(
                    {
                        "path": "PATCH_1/select1.mat",
                        "message": comparison.message,
                        "stage_scope": "stage3",
                        "failure_class": "stage3_patch_boundary",
                        "label": "Stage 3 patch boundary",
                        "failing_key": "C_ps2",
                        "failure_kind": "value_mismatch",
                        "shape_run": [10],
                        "shape_oracle": [10],
                        "max_abs": 2.5,
                        "guidance": "fix stage 3",
                    }
                )
            elif comparison.relative_path == "uw_space_time.mat":
                failures.append(
                    {
                        "path": "uw_space_time.mat",
                        "message": comparison.message,
                        "stage_scope": "stage7_8",
                        "failure_class": "unwrapped_noise_statistics",
                        "label": "Unwrapped-noise / statistics",
                        "failing_key": "dph_noise",
                        "failure_kind": "shape_mismatch",
                        "shape_run": [3, 4],
                        "shape_oracle": [5, 4],
                        "max_abs": None,
                        "guidance": "fix stage 7/8",
                    }
                )
        return {
            "ok": report.ok,
            "checked": len(report.comparisons),
            "failed": len(failures),
            "failures": failures,
            "groups": [],
            "first_boundary_failure": failures[0] if failures else None,
            "trace": {},
        }

    monkeypatch.setattr(module, "verify_run_against_golden", fake_verify)
    monkeypatch.setattr(module, "summarize_failures", fake_summarize)

    payload = module._dataset_audit(
        run_root,
        golden_root,
        "generated_full_loop_run_copy",
        contract,
        "20260421_120000",
        module.RunConfig(),
        generation,
    )

    trace = payload["trace"]["first_divergent_boundary"]
    assert trace["artifact_path"] == "PATCH_1/select1.mat"
    assert trace["failing_key"] == "C_ps2"
    assert trace["oracle_source"]["name"] == "cpp_wrapper"
    assert [item["artifact_path"] for item in trace["artifact_lineage"]] == [
        "PATCH_1/ps1.mat",
        "PATCH_1/ph1.mat",
        "PATCH_1/bp1.mat",
        "PATCH_1/da1.mat",
        "PATCH_1/pm1.mat",
        "PATCH_1/select1.mat",
    ]
    assert Path(payload["trace"]["first_divergent_boundary_output_path"]).exists()
    assert len(payload["trace"]["stage_boundary_probes"]) == 3
    stage3_probe = payload["trace"]["stage_boundary_probes"][1]
    assert stage3_probe["stage_boundary"] == 3
    assert Path(stage3_probe["output_path"]).exists()
