from pathlib import Path

import pytest

from pystamps.config import CompatibilityConfig, RunConfig, RuntimeConfig
from pystamps.pipeline.stages import (
    STAGE_DEFS,
    StageExecutionError,
    _kernel_backend_for_name,
    _run_merged_stage,
    _effective_stage2_native_threads,
    _normalize_backend,
    _stage2_kernel_backend_for_patch,
    _stage2_uses_full_cpu_default,
    _task_kind_for_stage,
    run_pipeline,
)
from pystamps.pipeline.types import PipelineContext, StageResult
from pystamps.runtime.executor import HybridExecutor


def _ctx(
    backend: str = "auto",
    strict_reference: bool = False,
    *,
    stage2_kernel_backend: str = "auto",
    stage2_native_threads: int = 0,
    io_workers: int = 8,
    cpu_workers: int = 0,
) -> PipelineContext:
    return PipelineContext(
        dataset_root=Path("."),
        run_config=RunConfig(
            runtime=RuntimeConfig(
                backend=backend,
                stage2_kernel_backend=stage2_kernel_backend,
                stage2_native_threads=stage2_native_threads,
                io_workers=io_workers,
                cpu_workers=cpu_workers,
            ),
            compat=CompatibilityConfig(strict_reference=strict_reference),
        ),
        start_step=1,
        end_step=8,
        dry_run=False,
    )


def test_backend_normalization_aliases() -> None:
    assert _normalize_backend("auto") == "auto"
    assert _normalize_backend("threads") == "threads"
    assert _normalize_backend("thread") == "threads"
    assert _normalize_backend("io") == "threads"
    assert _normalize_backend("processes") == "processes"
    assert _normalize_backend("process") == "processes"
    assert _normalize_backend("cpu") == "processes"
    assert _normalize_backend("gpu") == "gpu"
    assert _normalize_backend("native") == "native"


def test_backend_normalization_rejects_invalid() -> None:
    with pytest.raises(StageExecutionError, match="Unsupported runtime backend"):
        _normalize_backend("bogus")


def test_task_kind_auto_mode() -> None:
    context = _ctx("auto")
    assert _task_kind_for_stage(STAGE_DEFS[0], context, patch_count=4) == "io"
    assert _task_kind_for_stage(STAGE_DEFS[1], context, patch_count=4) == "cpu"
    assert _task_kind_for_stage(STAGE_DEFS[1], context, patch_count=1) == "io"
    assert _task_kind_for_stage(STAGE_DEFS[5], context, patch_count=4) == "io"


def test_task_kind_threads_mode() -> None:
    context = _ctx("threads")
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "io"


def test_task_kind_processes_mode() -> None:
    context = _ctx("processes")
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "cpu"


def test_task_kind_gpu_mode() -> None:
    context = _ctx("gpu")
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "io"


def test_task_kind_native_mode() -> None:
    context = _ctx("native")
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "cpu"


def test_task_kind_strict_reference_forces_io() -> None:
    context = _ctx("processes", strict_reference=True)
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "io"


def test_effective_stage2_native_threads_uses_explicit_override() -> None:
    context = _ctx(stage2_kernel_backend="native", stage2_native_threads=6)
    assert _effective_stage2_native_threads(STAGE_DEFS[1], context, patch_count=4) == 6


def test_effective_stage2_native_threads_default_uses_all_cpu_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pystamps.pipeline.stages.os.cpu_count", lambda: 9)
    context = _ctx("auto", stage2_kernel_backend="native", cpu_workers=0)

    assert _effective_stage2_native_threads(STAGE_DEFS[1], context, patch_count=4) == 9
    assert _effective_stage2_native_threads(STAGE_DEFS[1], context, patch_count=1) == 9


def test_effective_stage2_native_threads_is_disabled_for_python_backend() -> None:
    context = _ctx(stage2_kernel_backend="python")
    assert _effective_stage2_native_threads(STAGE_DEFS[1], context, patch_count=4) == 0


def test_effective_stage2_native_threads_uses_patch_override_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pystamps.pipeline.stages.os.cpu_count", lambda: 7)
    context = _ctx(stage2_kernel_backend="python")

    assert (
        _effective_stage2_native_threads(
            STAGE_DEFS[1],
            context,
            patch_count=4,
            stage2_kernel_backend="native",
        )
        == 7
    )


def test_stage2_patch_backend_override_routes_by_patch_name() -> None:
    context = _ctx(stage2_kernel_backend="python")
    context.run_config.runtime.stage2_patch_backend_overrides = {"PATCH_2": "native"}

    assert _stage2_kernel_backend_for_patch(context, Path("PATCH_1")) == "python"
    assert _stage2_kernel_backend_for_patch(context, Path("PATCH_2")) == "native"


def test_stage2_full_cpu_default_detects_only_auto_native_backend() -> None:
    assert _stage2_uses_full_cpu_default(STAGE_DEFS[1], _ctx(stage2_kernel_backend="native")) is True
    assert _stage2_uses_full_cpu_default(STAGE_DEFS[1], _ctx(stage2_kernel_backend="auto")) is True
    assert _stage2_uses_full_cpu_default(STAGE_DEFS[1], _ctx(stage2_kernel_backend="python")) is False
    assert _stage2_uses_full_cpu_default(STAGE_DEFS[2], _ctx(stage2_kernel_backend="native")) is False


def test_stage2_full_cpu_default_honors_patch_override_native_backend() -> None:
    context = _ctx(stage2_kernel_backend="python")
    context.run_config.runtime.stage2_patch_backend_overrides = {"PATCH_2": "native"}

    assert _stage2_uses_full_cpu_default(STAGE_DEFS[1], context) is True


def test_kernel_backend_override_routes_by_kernel_name() -> None:
    context = _ctx(backend="processes")
    context.run_config.runtime.kernel_backend_overrides = {"stage7_scla": "cuda"}

    assert _kernel_backend_for_name(context, "stage7_scla", "processes") == "cuda"
    assert _kernel_backend_for_name(context, "stage8_edge_noise", "processes") == "processes"


def test_run_merged_stage_uses_kernel_backend_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _ctx(backend="processes")
    context.run_config.runtime.kernel_backend_overrides = {"stage7_scla": "cuda"}
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "pystamps.pipeline.stages.stage7_calc_scla",
        lambda dataset_root, backend, chunk_ps, enable_mat_cache, io_workers: captured.setdefault("backend", backend)
        or "ok",
    )

    result = _run_merged_stage(STAGE_DEFS[6], tmp_path, context)

    assert result.status == "completed"
    assert captured == {"backend": "cuda"}


def test_run_pipeline_serializes_stage2_patches_when_default_uses_full_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_a = tmp_path / "PATCH_1"
    patch_b = tmp_path / "PATCH_2"
    patch_a.mkdir()
    patch_b.mkdir()

    context = PipelineContext(
        dataset_root=tmp_path,
        run_config=RunConfig(runtime=RuntimeConfig(stage2_kernel_backend="native")),
        start_step=2,
        end_step=2,
        dry_run=False,
    )

    class _Dataset:
        root = tmp_path
        patches = [patch_a, patch_b]

    calls: list[str] = []

    class _FakeExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_FakeExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, kind: str, fn: object, *args: object, **kwargs: object):
            calls.append(f"submit:{kind}")
            raise AssertionError("stage-2 default should not submit patch jobs to the executor")

    def fake_run_patch_stage_timed(stage, patch_dir, run_context, patch_count):
        calls.append(f"run:{patch_dir.name}")
        return StageResult(
            stage_id=stage.stage_id,
            scope="patch",
            target=patch_dir.name,
            status="completed",
            details="ok",
        )

    monkeypatch.setattr("pystamps.pipeline.stages.discover_dataset", lambda path: _Dataset())
    monkeypatch.setattr("pystamps.pipeline.stages.HybridExecutor", _FakeExecutor)
    monkeypatch.setattr("pystamps.pipeline.stages._run_patch_stage_timed", fake_run_patch_stage_timed)

    report = run_pipeline(context)

    assert [result.target for result in report.results] == ["PATCH_1", "PATCH_2"]
    assert calls == ["run:PATCH_1", "run:PATCH_2"]


def test_run_pipeline_serializes_stage2_patches_when_patch_override_uses_native(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_a = tmp_path / "PATCH_1"
    patch_b = tmp_path / "PATCH_2"
    patch_a.mkdir()
    patch_b.mkdir()

    context = PipelineContext(
        dataset_root=tmp_path,
        run_config=RunConfig(
            runtime=RuntimeConfig(
                stage2_kernel_backend="python",
                stage2_patch_backend_overrides={"PATCH_2": "native"},
            )
        ),
        start_step=2,
        end_step=2,
        dry_run=False,
    )

    class _Dataset:
        root = tmp_path
        patches = [patch_a, patch_b]

    calls: list[str] = []

    class _FakeExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_FakeExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, kind: str, fn: object, *args: object, **kwargs: object):
            calls.append(f"submit:{kind}")
            raise AssertionError("stage-2 native patch overrides should serialize patch execution")

    def fake_run_patch_stage_timed(stage, patch_dir, run_context, patch_count):
        calls.append(f"run:{patch_dir.name}")
        return StageResult(
            stage_id=stage.stage_id,
            scope="patch",
            target=patch_dir.name,
            status="completed",
            details="ok",
        )

    monkeypatch.setattr("pystamps.pipeline.stages.discover_dataset", lambda path: _Dataset())
    monkeypatch.setattr("pystamps.pipeline.stages.HybridExecutor", _FakeExecutor)
    monkeypatch.setattr("pystamps.pipeline.stages._run_patch_stage_timed", fake_run_patch_stage_timed)

    report = run_pipeline(context)

    assert [result.target for result in report.results] == ["PATCH_1", "PATCH_2"]
    assert calls == ["run:PATCH_1", "run:PATCH_2"]


def test_hybrid_executor_lazily_creates_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeFuture:
        def __init__(self, value: object) -> None:
            self._value = value

        def result(self) -> object:
            return self._value

    class _FakeThreadPool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("thread_init")

        def submit(self, fn: object, *args: object, **kwargs: object) -> _FakeFuture:
            return _FakeFuture(fn(*args, **kwargs))

        def shutdown(self, **kwargs: object) -> None:
            calls.append("thread_shutdown")

    class _FakeProcessPool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("process_init")

        def submit(self, fn: object, *args: object, **kwargs: object) -> _FakeFuture:
            return _FakeFuture(fn(*args, **kwargs))

        def shutdown(self, **kwargs: object) -> None:
            calls.append("process_shutdown")

    monkeypatch.setattr("pystamps.runtime.executor.ThreadPoolExecutor", _FakeThreadPool)
    monkeypatch.setattr("pystamps.runtime.executor.ProcessPoolExecutor", _FakeProcessPool)

    with HybridExecutor(io_workers=2, cpu_workers=2) as executor:
        assert "process_init" not in calls
        assert executor.submit("io", lambda: 1).result() == 1
        assert "process_init" not in calls
        assert executor.submit("cpu", lambda: 2).result() == 2
        assert calls.count("process_init") == 1
