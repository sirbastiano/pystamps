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
