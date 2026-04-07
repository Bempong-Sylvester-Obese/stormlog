from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import Sequence

import pytest

from examples.cli import capability_matrix


def test_run_stormlog_diagnose_uses_absolute_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    observed_cmd: list[str] = []

    def _fake_run_command(
        cmd: Sequence[str], **kwargs: object
    ) -> CompletedProcess[str]:
        nonlocal observed_cmd
        observed_cmd = list(cmd)
        assert kwargs.get("cwd") == capability_matrix.REPO_ROOT
        return CompletedProcess(list(cmd), 0, "", "")

    monkeypatch.setattr(capability_matrix, "run_command", _fake_run_command)

    result = capability_matrix._run_stormlog_diagnose(Path("relative-artifacts"))

    output_path = Path(observed_cmd[observed_cmd.index("--output") + 1])
    assert output_path.is_absolute()
    assert result["artifact_dir"] == str(output_path)


def test_run_benchmark_check_uses_absolute_artifact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    observed_cmd: list[str] = []

    def _fake_run_command(
        cmd: Sequence[str], **kwargs: object
    ) -> CompletedProcess[str]:
        nonlocal observed_cmd
        observed_cmd = list(cmd)
        assert kwargs.get("cwd") == capability_matrix.REPO_ROOT
        return CompletedProcess(list(cmd), 0, "", "")

    monkeypatch.setattr(capability_matrix, "run_command", _fake_run_command)

    result = capability_matrix._run_benchmark_check(Path("relative-artifacts"), "smoke")

    output_path = Path(observed_cmd[observed_cmd.index("--output") + 1])
    artifact_root = Path(observed_cmd[observed_cmd.index("--artifact-root") + 1])
    assert output_path.is_absolute()
    assert artifact_root.is_absolute()
    assert result["output"] == str(output_path)
    assert "--profile" in observed_cmd
    assert observed_cmd[observed_cmd.index("--profile") + 1] == "pr"
    assert "--mode" in observed_cmd
    assert observed_cmd[observed_cmd.index("--mode") + 1] == "overhead"
    assert observed_cmd[observed_cmd.index("--budgets") + 1] == str(
        capability_matrix.REPO_ROOT
        / "docs"
        / "benchmarks"
        / "v0.4_operating_budget.json"
    )


def test_run_benchmark_check_skips_when_tensorflow_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    result = capability_matrix._run_benchmark_check(Path("relative-artifacts"), "smoke")

    assert result["status"] == "SKIP"
    assert "tensorflow" in str(result["reason"]).lower()


def test_run_tui_smoke_skips_when_tui_extras_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSpawnError(Exception):
        pass

    class FakeChild:
        def __init__(self) -> None:
            self.before = (
                "The Stormlog TUI requires optional dependencies. "
                "Install with `pip install 'stormlog[tui,torch]'`."
            )

        def expect(self, _pattern: object, timeout: float | None = None) -> None:
            _ = timeout
            raise FakeSpawnError("missing textual")

        def send(self, _chars: str) -> None:
            raise AssertionError("send should not be reached when the TUI exits early")

        def isalive(self) -> bool:
            return False

        def terminate(self, force: bool = False) -> None:
            return None

        def close(self) -> None:
            return None

    def _fake_spawn(*args: object, **kwargs: object) -> FakeChild:
        _ = args, kwargs
        return FakeChild()

    fake_pexpect = SimpleNamespace(spawn=_fake_spawn, EOF=FakeSpawnError)
    monkeypatch.setitem(sys.modules, "pexpect", fake_pexpect)

    result = capability_matrix._run_tui_smoke()

    assert result["status"] == "SKIP"
    assert result["reason"] == "missing TUI extras"
