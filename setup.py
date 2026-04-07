from __future__ import annotations

from setuptools import setup
from setuptools_rust import Binding, RustExtension


setup(
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
