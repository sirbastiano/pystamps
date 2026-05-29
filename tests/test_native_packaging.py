from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from pystamps import native, native_cli


def test_native_binary_command_prefers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSTAMPS_NATIVE_BIN", "/opt/pystamps-native")

    assert native.native_binary_command() == (["/opt/pystamps-native"], None)


def test_native_binary_command_prefers_bundled_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundled = tmp_path / native.native_binary_name()
    bundled.write_text("#!/bin/sh\n", encoding="utf-8")
    bundled.chmod(bundled.stat().st_mode | stat.S_IXUSR)

    monkeypatch.delenv("PYSTAMPS_NATIVE_BIN", raising=False)
    monkeypatch.setattr(native, "bundled_native_binary", lambda: bundled)
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/pystamps-native")

    assert native.native_binary_command() == ([str(bundled)], None)


def test_native_binary_command_falls_back_to_cargo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PYSTAMPS_NATIVE_BIN", raising=False)
    monkeypatch.setattr(native, "bundled_native_binary", lambda: tmp_path / native.native_binary_name())
    monkeypatch.setattr(native.shutil, "which", lambda name: None)
    monkeypatch.setattr(native, "local_native_binary_candidates", lambda repo_root: [])

    command, cwd = native.native_binary_command()

    assert command == ["cargo", "run", "-p", "pystamps-core", "--bin", "pystamps-native", "--"]
    assert cwd is not None


def test_native_binary_command_skips_entrypoint_to_avoid_recursion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entrypoint = tmp_path / "pystamps-native"
    local_binary = tmp_path / "target" / native.native_binary_name()
    local_binary.parent.mkdir()
    local_binary.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.delenv("PYSTAMPS_NATIVE_BIN", raising=False)
    monkeypatch.setattr(native, "bundled_native_binary", lambda: tmp_path / "missing")
    monkeypatch.setattr(native.shutil, "which", lambda name: str(entrypoint))
    monkeypatch.setattr(native, "local_native_binary_candidates", lambda repo_root: [local_binary])

    assert native.native_binary_command(skip_path=str(entrypoint)) == ([str(local_binary)], None)


def test_native_cli_execs_binary_when_possible(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(native_cli, "native_binary_command", lambda **kwargs: (["/bin/pystamps-native"], None))
    monkeypatch.setattr(native_cli.sys, "argv", ["pystamps-native", "coverage"])

    def fake_execv(program: str, command: list[str]) -> None:
        calls["program"] = program
        calls["command"] = command
        raise SystemExit(0)

    monkeypatch.setattr(native_cli.os, "execv", fake_execv)

    with pytest.raises(SystemExit):
        native_cli.main()

    assert calls == {
        "program": "/bin/pystamps-native",
        "command": ["/bin/pystamps-native", "coverage"],
    }


def test_native_cli_returns_subprocess_status_for_cwd_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(native_cli, "native_binary_command", lambda **kwargs: (["cargo", "run", "--"], tmp_path))
    monkeypatch.setattr(native_cli.sys, "argv", ["pystamps-native", "coverage"])

    def fake_run(command: list[str], *, cwd: Path | None = None, check: bool = False) -> SimpleNamespace:
        calls["command"] = command
        calls["cwd"] = cwd
        calls["check"] = check
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(native_cli.subprocess, "run", fake_run)

    assert native_cli.main() == 7
    assert calls == {
        "command": ["cargo", "run", "--", "coverage"],
        "cwd": tmp_path,
        "check": False,
    }
