from __future__ import annotations

from pathlib import Path

import pytest

from worldreward.config_loader import (
    list_available_domains,
    load_domain_config,
    resolve_domain_config_path,
)
from worldreward.exceptions import ConfigLoadError


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_domain_config_success(tmp_path: Path) -> None:
    config_path = tmp_path / "demo.yaml"
    _write_file(
        config_path,
        """
domain_id: demo
domain_name: Demo Domain
description: Demo
context_prompt: Build scenarios
id_prefix: DM
categories:
  - name: collisions
    description: Collision cases
""".strip(),
    )

    config = load_domain_config(config_path)

    assert config.domain_id == "demo"
    assert config.id_prefix == "DM"
    assert config.category_names == ["collisions"]


def test_load_domain_config_missing_required_field(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    _write_file(
        broken,
        """
domain_id: demo
domain_name: Demo Domain
description: Missing context prompt
id_prefix: DM
categories: []
""".strip(),
    )

    with pytest.raises(ConfigLoadError):
        load_domain_config(broken)


def test_list_available_domains_and_resolve_across_multiple_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_file(first / "alpha.yaml", "domain_id: a")
    _write_file(second / "beta.yaml", "domain_id: b")
    _write_file(second / "alpha.yaml", "domain_id: a2")

    domains = list_available_domains([first, second])
    resolved_alpha = resolve_domain_config_path("alpha", [first, second])
    resolved_beta = resolve_domain_config_path("beta", [first, second])
    resolved_missing = resolve_domain_config_path("gamma", [first, second])

    assert domains == ["alpha", "beta"]
    assert resolved_alpha == first / "alpha.yaml"
    assert resolved_beta == second / "beta.yaml"
    assert resolved_missing is None
