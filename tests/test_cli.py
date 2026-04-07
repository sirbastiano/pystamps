from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pystamps import cli


def _payload(capsys: pytest.CaptureFixture[str]) -> object:
    return json.loads(capsys.readouterr().out)


def _run_config() -> SimpleNamespace:
    return SimpleNamespace(
        runtime=SimpleNamespace(io_workers=2, cpu_workers=4),
        tolerance=SimpleNamespace(rtol=1e-6, atol=1e-8),
    )


def test_cmd_status_prints_dataset_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    status = SimpleNamespace(
        dataset=Path("/tmp/run"),
        merged_stage=8,
        patch_statuses=[
            SimpleNamespace(patch="PATCH_1", stage=4),
            SimpleNamespace(patch="PATCH_2", stage=6),
        ],
    )
    monkeypatch.setattr(cli, "collect_status", lambda dataset: status)

    exit_code = cli._cmd_status("ignored")

    assert exit_code == 0
    assert _payload(capsys) == {
        "dataset": "/tmp/run",
        "merged_stage": 8,
        "patches": [
            {"patch": "PATCH_1", "stage": 4},
            {"patch": "PATCH_2", "stage": 6},
        ],
    }


def test_cmd_run_overrides_workers_and_reports_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        dataset=str(tmp_path / "dataset"),
        start_step=2,
        end_step=5,
        dry_run=True,
        io_workers=7,
        cpu_workers=3,
    )
    run_config = _run_config()
    captured: dict[str, object] = {}

    def fake_run_pipeline(context):
        captured["context"] = context
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    stage_id=2,
                    scope="patch",
                    target="PATCH_1",
                    status="ok",
                    details={"files": 4},
                    duration_sec=1.25,
                )
            ],
            failures=[],
        )

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    exit_code = cli._cmd_run(args, run_config)

    context = captured["context"]
    assert exit_code == 0
    assert run_config.runtime.io_workers == 7
    assert run_config.runtime.cpu_workers == 3
    assert context.dataset_root == (tmp_path / "dataset").resolve()
    assert context.start_step == 2
    assert context.end_step == 5
    assert context.dry_run is True
    assert _payload(capsys) == [
        {
            "stage": 2,
            "scope": "patch",
            "target": "PATCH_1",
            "status": "ok",
            "details": {"files": 4},
            "duration_sec": 1.25,
        }
    ]


def test_cmd_run_returns_failure_when_pipeline_reports_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        dataset=str(tmp_path / "dataset"),
        start_step=1,
        end_step=8,
        dry_run=False,
        io_workers=None,
        cpu_workers=None,
    )

    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda context: SimpleNamespace(
            results=[
                SimpleNamespace(
                    stage_id=6,
                    scope="merged",
                    target="dataset",
                    status="failed",
                    details={"reason": "unwrap"},
                    duration_sec=9.5,
                )
            ],
            failures=["stage6"],
        ),
    )

    exit_code = cli._cmd_run(args, _run_config())

    assert exit_code == 1
    assert _payload(capsys)[0]["status"] == "failed"


def test_cmd_verify_prints_success_payload(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    report = SimpleNamespace(
        ok=True,
        comparisons=[SimpleNamespace(ok=True, relative_path="PATCH_1/ps2.mat", message="matched")],
    )
    monkeypatch.setattr(cli, "verify_run_against_golden", lambda run, golden, tolerance: report)

    exit_code = cli._cmd_verify("run", "golden", _run_config())

    assert exit_code == 0
    assert _payload(capsys) == {"ok": True, "checked": 1, "failed": []}


def test_cmd_verify_prints_failed_comparisons(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    report = SimpleNamespace(
        ok=False,
        comparisons=[
            SimpleNamespace(ok=False, relative_path="PATCH_1/select1.mat", message="Value mismatch"),
            SimpleNamespace(ok=True, relative_path="PATCH_1/ph1.mat", message="matched"),
        ],
    )
    monkeypatch.setattr(cli, "verify_run_against_golden", lambda run, golden, tolerance: report)

    exit_code = cli._cmd_verify("run", "golden", _run_config())

    assert exit_code == 1
    assert _payload(capsys) == {
        "ok": False,
        "checked": 2,
        "failed": [{"path": "PATCH_1/select1.mat", "message": "Value mismatch"}],
    }


def test_resolve_stamps_root_prefers_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPS_ROOT", "/env/stamps")

    assert cli._resolve_stamps_root("/explicit/stamps") == "/explicit/stamps"


def test_resolve_stamps_root_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPS_ROOT", "/env/stamps")

    assert cli._resolve_stamps_root(None) == "/env/stamps"


def test_resolve_stamps_root_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STAMPS_ROOT", raising=False)

    with pytest.raises(SystemExit, match="Config error: list-legacy requires --stamps-root or STAMPS_ROOT"):
        cli._resolve_stamps_root(None)


def test_cmd_list_legacy_prints_discovered_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "discover_legacy_commands", lambda root: [Path(root) / "mt_prep", Path(root) / "ps_plot"])

    exit_code = cli._cmd_list_legacy("/opt/stamps")

    assert exit_code == 0
    assert _payload(capsys) == ["/opt/stamps/mt_prep", "/opt/stamps/ps_plot"]


def test_cmd_describe_inputs_prints_stage_contracts(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli._cmd_describe_inputs("1", dataset=None, patch="PATCH_1")

    payload = _payload(capsys)
    assert exit_code == 0
    assert [stage["stage"] for stage in payload["stages"]] == [1]
    assert any(item["array_name"] == "ph" for item in payload["stages"][0]["inputs"])


def test_cmd_describe_inputs_includes_dataset_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "inspect_stage1_inputs",
        lambda dataset, patch_name: {
            "metadata_mode": "direct patch metadata files",
            "overview_rows": [{"metric": "candidate count", "value": 10, "meaning": "count"}],
            "consistency_rows": [{"check": "candidate rows in pscands.1.ij", "status": "ok"}],
            "warnings": ["example warning"],
        },
    )

    exit_code = cli._cmd_describe_inputs("1", dataset="/tmp/dataset", patch="PATCH_2")

    payload = _payload(capsys)
    assert exit_code == 0
    assert payload["stage1_dataset_check"] == {
        "dataset": "dataset",
        "patch": "PATCH_2",
        "metadata_mode": "direct patch metadata files",
        "overview": [{"metric": "candidate count", "value": 10, "meaning": "count"}],
        "consistency": [{"check": "candidate rows in pscands.1.ij", "status": "ok"}],
        "warnings": ["example warning"],
    }


def test_cmd_describe_inputs_rejects_unknown_stage() -> None:
    with pytest.raises(SystemExit, match="Config error: Unsupported stage: 99"):
        cli._cmd_describe_inputs("99", dataset=None, patch="PATCH_1")


def test_cmd_describe_backends_prints_backend_matrix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "describe_backend_matrix",
        lambda: {
            "providers": {"python": {"available": True}},
            "kernels": {"stage8_edge_noise": {"supported_backends": ["python", "cuda"]}},
        },
    )

    exit_code = cli._cmd_describe_backends()

    assert exit_code == 0
    assert _payload(capsys) == {
        "providers": {"python": {"available": True}},
        "kernels": {"stage8_edge_noise": {"supported_backends": ["python", "cuda"]}},
    }


def test_main_dispatches_verify_command(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(command="verify", config="cfg.yml", run="/run", golden="/golden")
    run_config = _run_config()
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "_parse_args", lambda: args)
    monkeypatch.setattr(cli, "_load_run_config", lambda path: run_config)

    def fake_cmd_verify(run: str, golden: str, loaded_config) -> int:
        captured["run"] = run
        captured["golden"] = golden
        captured["config"] = loaded_config
        return 17

    monkeypatch.setattr(cli, "_cmd_verify", fake_cmd_verify)

    exit_code = cli.main()

    assert exit_code == 17
    assert captured == {"run": "/run", "golden": "/golden", "config": run_config}


def test_main_dispatches_describe_inputs_command(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(command="describe-inputs", config="cfg.yml", stage="all", dataset=None, patch="PATCH_1")
    run_config = _run_config()
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "_parse_args", lambda: args)
    monkeypatch.setattr(cli, "_load_run_config", lambda path: run_config)

    def fake_cmd_describe_inputs(stage: str, dataset: str | None, patch: str) -> int:
        captured["stage"] = stage
        captured["dataset"] = dataset
        captured["patch"] = patch
        return 23

    monkeypatch.setattr(cli, "_cmd_describe_inputs", fake_cmd_describe_inputs)

    exit_code = cli.main()

    assert exit_code == 23
    assert captured == {"stage": "all", "dataset": None, "patch": "PATCH_1"}


def test_main_dispatches_describe_backends_command(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(command="describe-backends", config="cfg.yml")
    run_config = _run_config()
    called: list[str] = []

    monkeypatch.setattr(cli, "_parse_args", lambda: args)
    monkeypatch.setattr(cli, "_load_run_config", lambda path: run_config)
    monkeypatch.setattr(cli, "_cmd_describe_backends", lambda: called.append("describe") or 29)

    exit_code = cli.main()

    assert exit_code == 29
    assert called == ["describe"]
