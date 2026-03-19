from __future__ import annotations

import importlib.util
import json
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
        lambda dataset_root, golden_base, audit_stamp: (_ for _ in ()).throw(RuntimeError("generation failed")),
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

    monkeypatch.setattr(module, "_inputs_root", lambda: tmp_path / "inputs_and_outputs")
    monkeypatch.setattr(module, "_copy_dataset", lambda src, dst: shutil.copytree(src, dst))

    captured: dict[str, object] = {}

    def fake_run_pipeline(context):
        captured["dataset_root"] = context.dataset_root
        captured["start_step"] = context.start_step
        captured["end_step"] = context.end_step
        return SimpleNamespace(failures=[])

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)

    run_root, generation = module._build_run_copy(dataset, "20260314_120000")

    assert run_root.name == "InSAR_dataset_test_stage8diag_stage2_8"
    assert generation["start_step"] == 2
    assert generation["end_step"] == 8
    assert "PATCH_*/pm1.mat" in generation["clean_patterns"]
    assert captured["dataset_root"] == run_root
    assert captured["start_step"] == 2
    assert captured["end_step"] == 8
    assert not (run_root / "PATCH_1" / "pm1.mat").exists()
    assert not (run_root / "PATCH_1" / "select1.mat").exists()
    assert not (run_root / "PATCH_1" / "weed1.mat").exists()


def test_build_run_copy_uses_run_full_gate_seed_for_dataset_test(monkeypatch, tmp_path: Path) -> None:
    module = _load_validate_audit_module()
    inputs_root = tmp_path / "inputs_and_outputs"
    dataset = inputs_root / "InSAR_dataset_test"
    dataset.mkdir(parents=True)
    seed = inputs_root / "RUN_FULL_GATE_1e10"
    patch = seed / "PATCH_1"
    patch.mkdir(parents=True)
    for filename in ("select1.mat", "weed1.mat", "pm2.mat", "ps2.mat", "phuw2.mat", "scla2.mat", "mean_v.mat"):
        (patch / filename).write_text("stub", encoding="utf-8")
    (seed / "pm2.mat").write_text("stub", encoding="utf-8")
    (seed / "phuw2.mat").write_text("stub", encoding="utf-8")
    (seed / "scla2.mat").write_text("stub", encoding="utf-8")
    (seed / "mean_v.mat").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(module, "_inputs_root", lambda: inputs_root)
    monkeypatch.setattr(module, "_seed_root_for_dataset", lambda dataset_root: seed.resolve())
    monkeypatch.setattr(module, "_copy_dataset", lambda src, dst: shutil.copytree(src, dst))
    monkeypatch.setattr(module, "run_pipeline", lambda context: SimpleNamespace(failures=[]))

    run_root, generation = module._build_run_copy(dataset, "20260314_120000")

    assert run_root.name == "InSAR_dataset_test_stage4_8"
    assert generation["start_step"] == 4
    assert generation["end_step"] == 8
    assert generation["seed_name"] == "RUN_FULL_GATE_1e10"
    assert Path(generation["seed_root"]) == seed.resolve()
    assert not (run_root / "PATCH_1" / "weed1.mat").exists()
    assert not (run_root / "PATCH_1" / "pm2.mat").exists()
    assert not (run_root / "pm2.mat").exists()
    assert not (run_root / "phuw2.mat").exists()
