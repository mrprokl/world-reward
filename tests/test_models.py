from __future__ import annotations

from worldreward.models import (
    CategoryConfig,
    Confidence,
    DomainConfig,
    RewardScore,
    Scenario,
    VerificationResult,
)


def test_scenario_to_dict_serializes_confidence() -> None:
    scenario = Scenario(
        scenario_id="AD-001",
        category="vehicle_collision",
        world_prompt="scene",
        action="impact",
        verification_question="Does it deform?",
        expected_answer="no",
        confidence=Confidence.HIGH,
        video_prompt="cinematic prompt",
    )

    payload = scenario.to_dict()

    assert payload["scenario_id"] == "AD-001"
    assert payload["confidence"] == "high"
    assert payload["video_prompt"] == "cinematic prompt"


def test_verification_result_to_dict_serializes_reward_value() -> None:
    result = VerificationResult(
        scenario_id="AD-001",
        category="vehicle_collision",
        verification_question="Does it deform?",
        expected_answer="no",
        vlm_answer="no",
        vlm_reasoning="Observed deformation.",
        reward=RewardScore.CORRECT,
        video_path="/tmp/AD-001.mp4",
    )

    payload = result.to_dict()

    assert payload["reward"] == "1"
    assert payload["vlm_answer"] == "no"


def test_domain_config_category_names_property() -> None:
    config = DomainConfig(
        domain_id="autonomous_driving",
        domain_name="Autonomous Driving",
        description="desc",
        context_prompt="context",
        id_prefix="AD",
        categories=[
            CategoryConfig(name="vehicle_collision", description="x"),
            CategoryConfig(name="weather_physics", description="y"),
        ],
    )

    assert config.category_names == ["vehicle_collision", "weather_physics"]
