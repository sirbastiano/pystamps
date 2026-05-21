import importlib.util
from pathlib import Path
import shutil


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


def test_stage2_probe_default_uses_all_patch_dirs_without_patch_list_old(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    for name in ("PATCH_1", "PATCH_2", "PATCH_3"):
        (dataset / name).mkdir()

    module = _load_stage2_probe_module()

    assert module._selected_patch_names(dataset, None) == ["PATCH_1", "PATCH_2", "PATCH_3"]


def test_stage2_probe_copy_fallback_retries_after_partial_hardlink_copy(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "PATCH_1").mkdir()
    (dataset / "patch.list_old").write_text("PATCH_1\n", encoding="utf-8")
    (dataset / "PATCH_1" / "seed.mat").write_text("seed", encoding="utf-8")

    run_root = tmp_path / "run_root"
    call_log: list[bool] = []
    real_copytree = shutil.copytree
    module = _load_stage2_probe_module()

    def fake_copytree(src, dst, *args, **kwargs):
        if not call_log:
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "partial.txt").write_text("partial", encoding="utf-8")
            call_log.append(True)
            raise OSError("forced hardlink copy failure")
        call_log.append(False)
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "copytree", fake_copytree)
    module._copy_dataset_for_probe(dataset, run_root)

    assert not (run_root / "partial.txt").exists()
    assert call_log and call_log[0] is True
    assert any(not flag for flag in call_log[1:])
    assert (run_root / "PATCH_1").is_dir()
    assert (run_root / "PATCH_1" / "seed.mat").read_text(encoding="utf-8") == "seed"
