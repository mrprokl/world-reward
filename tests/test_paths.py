from __future__ import annotations

from pathlib import Path

import pytest

from worldreward import paths


def test_resolve_api_key_prefers_explicit_value(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert paths.resolve_api_key("explicit-key") == "explicit-key"


def test_resolve_api_key_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert paths.resolve_api_key() == "env-key"


def test_resolve_api_key_reads_toml_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("WORLDREWARD_HOME", str(tmp_path))
    config_file = tmp_path / "config.toml"
    config_file.write_text('gemini_api_key = "toml-key"\n', encoding="utf-8")

    assert paths.resolve_api_key() == "toml-key"


def test_output_dir_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom-output"
    monkeypatch.setenv("WORLDREWARD_OUTPUT_DIR", str(override))
    assert paths.get_output_dir() == override


def test_save_api_key_persists_value_for_subsequent_resolution(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("WORLDREWARD_HOME", str(tmp_path))

    config_path = paths.save_api_key("stored-key")

    assert config_path == tmp_path / "config.toml"
    assert config_path.exists()
    assert paths.resolve_api_key() == "stored-key"


def test_save_api_key_updates_existing_auth_section(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORLDREWARD_HOME", str(tmp_path))
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[auth]\ngemini_api_key = "old-key"\nother_value = "kept"\n',
        encoding="utf-8",
    )

    paths.save_api_key("new-key")
    loaded = paths.load_user_config()

    auth = loaded.get("auth")
    assert isinstance(auth, dict)
    assert auth.get("gemini_api_key") == "new-key"
    assert auth.get("other_value") == "kept"


def test_save_api_key_rejects_empty_value(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORLDREWARD_HOME", str(tmp_path))
    with pytest.raises(ValueError):
        paths.save_api_key("   ")
