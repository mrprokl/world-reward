from __future__ import annotations

from pathlib import Path

from worldreward import setup_wizard


def test_run_setup_wizard_saves_valid_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORLDREWARD_HOME", str(tmp_path))
    messages: list[str] = []

    ok = setup_wizard.run_setup_wizard(
        api_key="test-key",
        print_fn=messages.append,
        validator=lambda _k: (True, None),
    )

    assert ok is True
    config_file = tmp_path / "config.toml"
    assert config_file.exists()
    assert "gemini_api_key" in config_file.read_text(encoding="utf-8")


def test_run_setup_wizard_rejects_invalid_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORLDREWARD_HOME", str(tmp_path))
    messages: list[str] = []

    ok = setup_wizard.run_setup_wizard(
        api_key="bad-key",
        print_fn=messages.append,
        validator=lambda _k: (False, "unauthorized"),
    )

    assert ok is False
    assert any("validation failed" in message for message in messages)


def test_configure_api_key_interactive_uses_prompt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORLDREWARD_HOME", str(tmp_path))
    messages: list[str] = []

    ok = setup_wizard.configure_api_key_interactive(
        print_fn=messages.append,
        secret_prompt_fn=lambda _prompt: "prompt-key",
        validator=lambda _k: (True, None),
    )

    assert ok is True
    assert (tmp_path / "config.toml").exists()


def test_render_config_summary_masks_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORLDREWARD_HOME", str(tmp_path))
    setup_wizard.run_setup_wizard(
        api_key="verysecretkey1234",
        print_fn=lambda _m: None,
        validator=lambda _k: (True, None),
    )

    summary = setup_wizard.render_config_summary(show_api_key=False)
    assert "API key source: config" in summary
    assert "very...1234" in summary


def test_render_config_summary_can_show_plain_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORLDREWARD_HOME", str(tmp_path))
    setup_wizard.run_setup_wizard(
        api_key="visible-key",
        print_fn=lambda _m: None,
        validator=lambda _k: (True, None),
    )

    summary = setup_wizard.render_config_summary(show_api_key=True)
    assert "visible-key" in summary
