"""Path and local configuration resolution for World Reward."""

from __future__ import annotations

import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
APP_DIRNAME = ".worldreward"

_ENV_HOME = "WORLDREWARD_HOME"
_ENV_CONFIGS = "WORLDREWARD_CONFIGS_DIR"
_ENV_OUTPUT = "WORLDREWARD_OUTPUT_DIR"


@dataclass(frozen=True)
class RuntimeLayout:
    """Resolved and provisioned runtime paths."""

    app_dir: Path
    config_file: Path
    user_configs_dir: Path
    user_output_dir: Path
    datasets_dir: Path
    videos_dir: Path
    results_dir: Path
    copied_builtin_configs: tuple[Path, ...] = ()


def get_app_dir() -> Path:
    """Return the app home directory used outside repository mode."""
    env_home = os.getenv(_ENV_HOME)
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / APP_DIRNAME


def get_builtin_configs_dir() -> Path:
    """Return package-bundled config directory."""
    return PACKAGE_DIR / "builtin_configs"


def get_user_configs_dir() -> Path:
    """Return user config directory (`~/.worldreward/configs`)."""
    return get_app_dir() / "configs"


def get_user_output_dir() -> Path:
    """Return user output directory (`~/.worldreward/output`)."""
    return get_app_dir() / "output"


def is_repo_checkout_mode() -> bool:
    """Whether this execution context is a repository checkout."""
    return (REPO_ROOT / ".git").exists() and (REPO_ROOT / "configs").exists()


def get_config_search_dirs() -> list[Path]:
    """Return config directories ordered by precedence."""
    env_configs = os.getenv(_ENV_CONFIGS)
    if env_configs:
        return [Path(env_configs).expanduser()]

    dirs: list[Path] = []
    repo_configs = REPO_ROOT / "configs"
    if repo_configs.exists():
        dirs.append(repo_configs)
    dirs.append(get_user_configs_dir())

    package_configs = get_builtin_configs_dir()
    if package_configs.exists():
        dirs.append(package_configs)

    return _unique_paths(dirs)


def get_primary_configs_dir() -> Path:
    """Return the first config directory used for reads/writes."""
    return get_config_search_dirs()[0]


def get_output_dir() -> Path:
    """Return output root directory preserving repository compatibility."""
    env_output = os.getenv(_ENV_OUTPUT)
    if env_output:
        return Path(env_output).expanduser()

    if is_repo_checkout_mode():
        return REPO_ROOT / "output"
    return get_user_output_dir()


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


def ensure_runtime_layout(copy_builtin_configs: bool = True) -> RuntimeLayout:
    """Create runtime directories and optionally seed user configs."""
    app_dir = get_app_dir()
    user_configs = get_user_configs_dir()
    user_output = get_user_output_dir()
    datasets_dir = user_output / "datasets"
    videos_dir = user_output / "videos"
    results_dir = user_output / "results"
    config_file = get_user_config_file()

    for directory in (app_dir, user_configs, user_output, datasets_dir, videos_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if os.name == "posix":
        try:
            os.chmod(app_dir, 0o700)
        except OSError:
            pass

    copied_builtin: list[Path] = []
    if copy_builtin_configs:
        builtin_configs = get_builtin_configs_dir()
        if builtin_configs.exists():
            for source in sorted(builtin_configs.glob("*.yaml")):
                destination = user_configs / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)
                    copied_builtin.append(destination)

    return RuntimeLayout(
        app_dir=app_dir,
        config_file=config_file,
        user_configs_dir=user_configs,
        user_output_dir=user_output,
        datasets_dir=datasets_dir,
        videos_dir=videos_dir,
        results_dir=results_dir,
        copied_builtin_configs=tuple(copied_builtin),
    )


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
    try:
        with open(config_file, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def resolve_api_key(explicit_api_key: str | None = None) -> str | None:
    """Resolve API key from explicit arg, env var, or user config."""
    api_key, _source = resolve_api_key_with_source(explicit_api_key)
    return api_key


def resolve_api_key_with_source(explicit_api_key: str | None = None) -> tuple[str | None, str]:
    """Resolve API key and return `(value, source)`."""
    if explicit_api_key:
        return explicit_api_key, "explicit"

    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key, "env"

    config = load_user_config()

    # Support flat and sectioned TOML styles.
    for key in ("gemini_api_key", "GEMINI_API_KEY"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), "config"

    auth = config.get("auth")
    if isinstance(auth, dict):
        value = auth.get("gemini_api_key")
        if isinstance(value, str) and value.strip():
            return value.strip(), "config"

    return None, "missing"


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


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Keep path order while removing duplicates."""
    deduped: list[Path] = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return deduped
