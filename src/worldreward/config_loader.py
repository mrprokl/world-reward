"""Load and validate domain configuration from YAML files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from worldreward.exceptions import ConfigLoadError
from worldreward.models import CategoryConfig, DomainConfig


def load_domain_config(config_path: Path) -> DomainConfig:
    """Load a domain configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated DomainConfig instance.

    Raises:
        ConfigLoadError: If the file cannot be read or is missing required fields.
    """
    if not config_path.exists():
        raise ConfigLoadError(str(config_path), "File not found")

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(str(config_path), f"Invalid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigLoadError(str(config_path), "YAML root must be a mapping/object")

    _validate_required_fields(raw, config_path)

    categories = [
        CategoryConfig(
            name=cat["name"],
            description=cat.get("description", ""),
            example_scenarios=cat.get("example_scenarios", []),
        )
        for cat in raw["categories"]
    ]

    return DomainConfig(
        domain_id=raw["domain_id"],
        domain_name=raw["domain_name"],
        description=raw["description"],
        context_prompt=raw["context_prompt"],
        categories=categories,
        id_prefix=raw["id_prefix"],
    )


def list_available_domains(configs_dir: Path | Iterable[Path]) -> list[str]:
    """List available domain config files (without extension)."""
    paths = [configs_dir] if isinstance(configs_dir, Path) else list(configs_dir)
    domain_names: set[str] = set()
    for directory in paths:
        if not directory.exists():
            continue
        for file_path in directory.glob("*.yaml"):
            domain_names.add(file_path.stem)
    return sorted(domain_names)


def resolve_domain_config_path(domain: str, configs_dir: Path | Iterable[Path]) -> Path | None:
    """Resolve the YAML file path for a given domain name."""
    paths = [configs_dir] if isinstance(configs_dir, Path) else list(configs_dir)
    for directory in paths:
        config_path = directory / f"{domain}.yaml"
        if config_path.exists():
            return config_path
    return None


_REQUIRED_FIELDS = ["domain_id", "domain_name", "description", "context_prompt", "categories", "id_prefix"]


def _validate_required_fields(raw: dict, config_path: Path) -> None:
    """Validate that all required fields are present in the config."""
    for field in _REQUIRED_FIELDS:
        if field not in raw:
            raise ConfigLoadError(str(config_path), f"Missing required field: '{field}'")
