from __future__ import annotations

from worldreward.models import CategoryConfig, DomainConfig
from worldreward.prompt_builder import build_generation_prompt


def test_build_generation_prompt_contains_core_sections() -> None:
    config = DomainConfig(
        domain_id="public_safety",
        domain_name="Public Safety",
        description="Urban hazards",
        context_prompt="Generate physically verifiable scenarios.",
        id_prefix="PS",
        categories=[
            CategoryConfig(
                name="structural_failure",
                description="Failures under load",
                example_scenarios=["Bridge collapse under overload"],
            )
        ],
    )

    prompt = build_generation_prompt(config, count=7)

    assert "DOMAIN: Public Safety" in prompt
    assert "Generate exactly 7 diverse physics-verifiable scenarios" in prompt
    assert "### structural_failure" in prompt
    assert "Bridge collapse under overload" in prompt
    assert "Return ONLY a valid JSON array" in prompt
