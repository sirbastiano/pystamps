from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools_rust import Binding, RustExtension


class build_py(_build_py):
    """Build and bundle the standalone Rust CLI into platform wheels."""

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = super().get_outputs(include_bytecode)
        bundled = Path(self.build_lib) / "pystamps" / "bin"
        if bundled.is_dir():
            outputs.extend(str(path) for path in bundled.iterdir() if path.is_file())
        return outputs

    def run(self) -> None:
        super().run()
        if os.environ.get("PYSTAMPS_SKIP_NATIVE_BIN") or getattr(self, "editable_mode", False):
            return
        self._bundle_native_binary()

    def _bundle_native_binary(self) -> None:
        root = Path(__file__).resolve().parent
        subprocess.run(
            ["cargo", "build", "--release", "-p", "pystamps-core", "--bin", "pystamps-native"],
            cwd=root,
            check=True,
        )
        suffix = ".exe" if os.name == "nt" else ""
        source = root / "target" / "release" / f"pystamps-native{suffix}"
        if not source.is_file():
            raise RuntimeError(f"native binary build did not produce {source}")

        target_dir = Path(self.build_lib) / "pystamps" / "bin"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        target.chmod(target.stat().st_mode | 0o755)


setup(
    cmdclass={"build_py": build_py},
    rust_extensions=[
        RustExtension(
            "pystamps.kernels._stage2_native",
            path="Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    zip_safe=False,
)
