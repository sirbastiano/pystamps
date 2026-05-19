import importlib.util
from pathlib import Path


def _load_stage2_probe_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "stage2_patch1_probe.py"
    spec = importlib.util.spec_from_file_location("stage2_patch1_probe", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage2_probe_default_uses_authoritative_patch_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    (dataset / "patch.list_old").write_text("PATCH_1\nPATCH_2\n", encoding="utf-8")
    (dataset / "PATCH_1").mkdir()
    (dataset / "PATCH_2").mkdir()

    module = _load_stage2_probe_module()

    assert module._selected_patch_names(dataset, None) == ["PATCH_1", "PATCH_2"]
    assert module._selected_patch_names(dataset, "PATCH_2") == ["PATCH_2"]
