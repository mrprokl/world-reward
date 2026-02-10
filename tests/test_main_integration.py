from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import worldreward.cli as cli
import worldreward.main as app_main
import worldreward.repl as repl


def test_main_dispatches_generate_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(app_main, "_load_env_files", lambda: captured.setdefault("env_loaded", True))
    monkeypatch.setattr(
        app_main,
        "ensure_runtime_layout",
        lambda copy_builtin_configs=True: captured.setdefault("layout_arg", copy_builtin_configs),
    )
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(command="generate", domain="public_safety", count=3, model="m"),
    )
    monkeypatch.setattr(
        cli,
        "run_generate",
        lambda domain, count, model: captured.setdefault(
            "run_generate_args", (domain, count, model)
        ),
    )
    monkeypatch.setattr(sys, "argv", ["worldreward", "generate"])

    app_main.main()

    assert captured["env_loaded"] is True
    assert captured["layout_arg"] is True
    assert captured["run_generate_args"] == ("public_safety", 3, "m")


def test_main_dispatches_config_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(app_main, "_load_env_files", lambda: None)
    monkeypatch.setattr(app_main, "ensure_runtime_layout", lambda copy_builtin_configs=True: None)
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(command="config", set_api_key=True, show_api_key=False),
    )
    monkeypatch.setattr(
        cli,
        "run_config",
        lambda set_api_key=False, show_api_key=False: captured.setdefault(
            "run_config_args", (set_api_key, show_api_key)
        ),
    )
    monkeypatch.setattr(sys, "argv", ["worldreward", "config"])

    app_main.main()

    assert captured["run_config_args"] == (True, False)


def test_main_without_args_launches_repl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(app_main, "_load_env_files", lambda: None)
    monkeypatch.setattr(app_main, "ensure_runtime_layout", lambda copy_builtin_configs=True: None)
    monkeypatch.setattr(repl, "run_repl", lambda: captured.setdefault("run_repl_called", True))
    monkeypatch.setattr(sys, "argv", ["worldreward"])

    app_main.main()

    assert captured["run_repl_called"] is True


def test_main_unknown_command_exits_with_usage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(app_main, "_load_env_files", lambda: None)
    monkeypatch.setattr(app_main, "ensure_runtime_layout", lambda copy_builtin_configs=True: None)
    monkeypatch.setattr(cli, "parse_args", lambda: SimpleNamespace(command="unknown"))
    monkeypatch.setattr(sys, "argv", ["worldreward", "unknown"])

    with pytest.raises(SystemExit) as exc_info:
        app_main.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert "Usage: worldreward" in output
