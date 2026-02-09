"""Path and local configuration resolution for World Reward."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    tomllib = None  # type: ignore[assignment]


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
APP_DIRNAME = ".worldreward"

_ENV_HOME = "WORLDREWARD_HOME"
_ENV_CONFIGS = "WORLDREWARD_CONFIGS_DIR"
_ENV_OUTPUT = "WORLDREWARD_OUTPUT_DIR"


def get_app_dir() -> Path:
    """Return the app home directory used outside repository mode."""
    env_home = os.getenv(_ENV_HOME)
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / APP_DIRNAME


def get_config_search_dirs() -> list[Path]:
    """Return config directories ordered by precedence."""
    env_configs = os.getenv(_ENV_CONFIGS)
    if env_configs:
        return [Path(env_configs).expanduser()]

    dirs: list[Path] = []
    repo_configs = REPO_ROOT / "configs"
    user_configs = get_app_dir() / "configs"
    package_configs = PACKAGE_DIR / "builtin_configs"

    for directory in (repo_configs, user_configs, package_configs):
        if directory.exists() and directory not in dirs:
            dirs.append(directory)

    # Keep deterministic fallback targets even before first run.
    if not dirs:
        dirs = [user_configs, package_configs]

    return dirs


def get_primary_configs_dir() -> Path:
    """Return the first config directory used for reads/writes."""
    return get_config_search_dirs()[0]


def get_output_dir() -> Path:
    """Return output root directory preserving repository compatibility."""
    env_output = os.getenv(_ENV_OUTPUT)
    if env_output:
        return Path(env_output).expanduser()

    repo_output = REPO_ROOT / "output"
    repo_configs = REPO_ROOT / "configs"
    if repo_configs.exists():
        return repo_output
    return get_app_dir() / "output"


def get_datasets_dir() -> Path:
    """Return dataset output directory."""
    return get_output_dir() / "datasets"


def get_videos_dir() -> Path:
    """Return videos output directory."""
    return get_output_dir() / "videos"


def get_results_dir() -> Path:
    """Return verification results output directory."""
    return get_output_dir() / "results"


def get_user_config_file() -> Path:
    """Return user config file path (`~/.worldreward/config.toml`)."""
    return get_app_dir() / "config.toml"


def save_api_key(api_key: str) -> Path:
    """Persist API key to user config file and return file path."""
    normalized_key = api_key.strip()
    if not normalized_key:
        raise ValueError("API key cannot be empty.")

    config_file = get_user_config_file()
    config_file.parent.mkdir(parents=True, exist_ok=True)

    config = load_user_config()
    auth_section = config.get("auth")
    if not isinstance(auth_section, dict):
        auth_section = {}
    auth_section["gemini_api_key"] = normalized_key
    config["auth"] = auth_section

    serialized = _dump_toml(config)
    temp_path = config_file.with_suffix(".tmp")
    temp_path.write_text(serialized, encoding="utf-8")
    os.replace(temp_path, config_file)

    if os.name == "posix":
        os.chmod(config_file, 0o600)

    return config_file


def load_user_config() -> dict[str, Any]:
    """Load user TOML configuration if it exists."""
    config_file = get_user_config_file()
    if not config_file.exists():
        return {}
    if tomllib is not None:
        with open(config_file, "rb") as f:
            data = tomllib.load(f)
        return data if isinstance(data, dict) else {}
    return _load_user_config_fallback(config_file)


def resolve_api_key(explicit_api_key: str | None = None) -> str | None:
    """Resolve API key from explicit arg, env var, or user config."""
    if explicit_api_key:
        return explicit_api_key

    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key

    config = load_user_config()

    # Support flat and sectioned TOML styles.
    for key in ("gemini_api_key", "GEMINI_API_KEY"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    auth = config.get("auth")
    if isinstance(auth, dict):
        value = auth.get("gemini_api_key")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _load_user_config_fallback(config_file: Path) -> dict[str, Any]:
    """Best-effort parser for simple TOML key/value files on Python <3.11."""
    data: dict[str, Any] = {}
    current_section: str | None = None

    for line in config_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            if current_section and current_section not in data:
                data[current_section] = {}
            continue
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip('"').strip("'")

        if current_section:
            section = data.get(current_section)
            if not isinstance(section, dict):
                section = {}
                data[current_section] = section
            section[key] = value
        else:
            data[key] = value

    return data


def _dump_toml(data: dict[str, Any]) -> str:
    """Serialize a limited TOML subset for local configuration writes."""
    lines = ["# World Reward local configuration", ""]

    scalar_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, dict[str, Any]]] = []

    for key in sorted(data.keys()):
        value = data[key]
        if isinstance(value, dict):
            table_items.append((key, value))
        else:
            scalar_items.append((key, value))

    for key, value in scalar_items:
        lines.append(f"{key} = {_toml_scalar(value)}")

    if scalar_items and table_items:
        lines.append("")

    for idx, (table, values) in enumerate(table_items):
        lines.append(f"[{table}]")
        for item_key in sorted(values.keys()):
            lines.append(f"{item_key} = {_toml_scalar(values[item_key])}")
        if idx != len(table_items) - 1:
            lines.append("")

    lines.append("")
    return "\n".join(lines)


def _toml_scalar(value: Any) -> str:
    """Serialize simple scalar types used in config."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value)
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'
