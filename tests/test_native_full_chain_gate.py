from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_gate_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "native_full_chain_gate.py"
    spec = importlib.util.spec_from_file_location("native_full_chain_gate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_run_copy_restores_legacy_patch_list_and_cleans_outputs(tmp_path: Path) -> None:
    module = _load_gate_module()
    dataset = tmp_path / "dataset"
    run_root = tmp_path / "run"
    for name in ("PATCH_1", "PATCH_2", "PATCH_3", "PATCH_4"):
        patch = dataset / name
        patch.mkdir(parents=True)
        (patch / "pscands.1.ij").write_text("input", encoding="utf-8")
        (patch / "ps1.mat").write_text("generated", encoding="utf-8")
        (patch / "pm1.mat").write_text("generated", encoding="utf-8")
        (patch / "select1.mat").write_text("generated", encoding="utf-8")
        (patch / "weed1.mat").write_text("generated", encoding="utf-8")
        (patch / "ps2.mat").write_text("generated", encoding="utf-8")
    (dataset / "phuw2.mat").write_text("generated", encoding="utf-8")
    (dataset / "scla2.mat").write_text("generated", encoding="utf-8")
    (dataset / "mean_v.mat").write_text("generated", encoding="utf-8")
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    (dataset / "patch.list_old").write_text("PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n", encoding="utf-8")

    setup = module.prepare_run_copy(dataset, run_root, 1, 8)

    assert setup["patch_manifest_source"] == "patch.list_old"
    assert (run_root / "patch.list").read_text(encoding="utf-8") == "PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n"
    assert (dataset / "patch.list").read_text(encoding="utf-8") == "PATCH_1\n"
    assert (run_root / "PATCH_2").exists()
    assert (run_root / "PATCH_1" / "pscands.1.ij").exists()
    assert not (run_root / "PATCH_1" / "ps1.mat").exists()
    assert not (run_root / "PATCH_1" / "pm1.mat").exists()
    assert not (run_root / "PATCH_1" / "select1.mat").exists()
    assert not (run_root / "PATCH_1" / "weed1.mat").exists()
    assert not (run_root / "PATCH_1" / "ps2.mat").exists()
    assert not (run_root / "phuw2.mat").exists()
    assert not (run_root / "scla2.mat").exists()
    assert not (run_root / "mean_v.mat").exists()


def test_authoritative_patch_manifest_rejects_subset_without_legacy_manifest(tmp_path: Path) -> None:
    module = _load_gate_module()
    dataset = tmp_path / "dataset"
    for name in ("PATCH_1", "PATCH_2", "PATCH_3", "PATCH_4"):
        (dataset / name).mkdir(parents=True)
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")

    with pytest.raises(module.GateError, match="patch.list lists 1 patch"):
        module.authoritative_patch_manifest(dataset)
