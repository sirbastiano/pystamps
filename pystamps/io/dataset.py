from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PATCH_PREFIX = "PATCH_"

PATCH_STAGE_ARTIFACTS: dict[int, str] = {
    1: "ps1.mat",
    2: "pm1.mat",
    3: "select1.mat",
    4: "weed1.mat",
    5: "ph2.mat",
}

MERGED_STAGE_ARTIFACTS: dict[int, str] = {
    5: "ifgstd2.mat",
    6: "phuw2.mat",
    7: "scla2.mat",
    8: "uw_space_time.mat",
}


@dataclass(slots=True)
class DatasetLayout:
    root: Path
    patches: list[Path]
    patch_list_file: Path | None


class DatasetError(ValueError):
    """Raised for invalid dataset layouts."""


def _patch_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.replace(PATCH_PREFIX, "", 1)
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (10**9, path.name)


def read_patch_manifest(path: str | Path) -> list[str]:
    manifest = Path(path)
    if not manifest.exists():
        return []
    return [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def discover_authoritative_patch_paths(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise DatasetError(f"Dataset root does not exist: {root_path}")

    discovered = sorted([p for p in root_path.iterdir() if p.is_dir() and p.name.startswith(PATCH_PREFIX)], key=_patch_sort_key)
    for manifest_name in ("patch.list_old", "patch.list"):
        patch_names = read_patch_manifest(root_path / manifest_name)
        patches = [root_path / name for name in patch_names if (root_path / name).is_dir()]
        if patches and manifest_name == "patch.list":
            # Some single-master fixture datasets keep a legacy shortened patch.list for compatibility.
            # When patch.list is incomplete, fall back to the actual patch directories so wildcard
            # comparisons cannot accidentally hide unverified audited patches.
            if len(patches) < len(discovered):
                continue
            return patches
        if patches:
            return patches
    return discovered


def discover_dataset(root: str | Path) -> DatasetLayout:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise DatasetError(f"Dataset root does not exist: {root_path}")

    patch_list = root_path / "patch.list"
    patches: list[Path]

    if patch_list.exists():
        names = read_patch_manifest(patch_list)
        patches = [root_path / name for name in names if (root_path / name).is_dir()]
    else:
        patches = sorted([p for p in root_path.iterdir() if p.is_dir() and p.name.startswith(PATCH_PREFIX)], key=_patch_sort_key)

    return DatasetLayout(root=root_path, patches=patches, patch_list_file=patch_list if patch_list.exists() else None)


def infer_patch_stage(patch_dir: str | Path) -> int:
    patch = Path(patch_dir)
    stage = 0
    for candidate_stage, artifact in sorted(PATCH_STAGE_ARTIFACTS.items()):
        if (patch / artifact).exists():
            stage = candidate_stage
    return stage


def infer_merged_stage(root_dir: str | Path) -> int:
    root = Path(root_dir)
    stage = 0
    for candidate_stage, artifact in sorted(MERGED_STAGE_ARTIFACTS.items()):
        if (root / artifact).exists():
            stage = max(stage, candidate_stage)
    return stage


def expected_stage_artifact(stage: int, scope: str) -> str | None:
    if scope == "patch":
        return PATCH_STAGE_ARTIFACTS.get(stage)
    if scope == "merged":
        return MERGED_STAGE_ARTIFACTS.get(stage)
    return None
