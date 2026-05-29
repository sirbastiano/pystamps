from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def native_binary_name() -> str:
    return "pystamps-native.exe" if sys.platform == "win32" else "pystamps-native"


def bundled_native_binary() -> Path:
    return Path(__file__).resolve().parent / "bin" / native_binary_name()


def local_native_binary_candidates(repo_root: Path) -> list[Path]:
    return [
        repo_root / "target" / "debug" / native_binary_name(),
        repo_root / "target" / "release" / native_binary_name(),
    ]


def native_binary_command(*, skip_path: str | None = None) -> tuple[list[str], Path | None]:
    repo_root = Path(__file__).resolve().parents[1]
    if value := os.environ.get("PYSTAMPS_NATIVE_BIN"):
        return [value], None

    bundled = bundled_native_binary()
    if bundled.exists():
        return [str(bundled)], None

    binary = shutil.which("pystamps-native")
    if binary:
        if skip_path is None or Path(binary).resolve() != Path(skip_path).resolve():
            return [binary], None

    for candidate in local_native_binary_candidates(repo_root):
        if candidate.exists():
            return [str(candidate)], None

    fallback_command = ["cargo", "run", "-p", "pystamps-core", "--bin", "pystamps-native", "--"]
    return fallback_command, repo_root
