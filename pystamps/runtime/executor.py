from __future__ import annotations

import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Literal


TaskKind = Literal["io", "cpu"]


def _default_cpu_workers() -> int:
    return max(1, os.cpu_count() or 4)


class HybridExecutor:
    """Hybrid execution model: threads for IO/orchestration, processes for CPU kernels."""

    def __init__(self, io_workers: int = 8, cpu_workers: int = 0) -> None:
        self.io_workers = max(1, int(io_workers))
        self.cpu_workers = int(cpu_workers) if cpu_workers and cpu_workers > 0 else _default_cpu_workers()
        self._thread_pool: ThreadPoolExecutor | None = None
        self._process_pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "HybridExecutor":
        self._thread_pool = ThreadPoolExecutor(max_workers=self.io_workers, thread_name_prefix="pystamps-io")
        self._process_pool = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True, cancel_futures=False)
            self._thread_pool = None
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True, cancel_futures=False)
            self._process_pool = None

    def submit(self, kind: TaskKind, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        if kind == "io":
            if self._thread_pool is None:
                raise RuntimeError("HybridExecutor not started")
            return self._thread_pool.submit(fn, *args, **kwargs)
        if kind == "cpu":
            if self._thread_pool is None:
                raise RuntimeError("HybridExecutor not started")
            if self._process_pool is None:
                self._process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
            return self._process_pool.submit(fn, *args, **kwargs)
        raise ValueError(f"Unknown task kind: {kind}")
