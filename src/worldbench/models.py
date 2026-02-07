"""Data models for WorldBench scenarios and domain configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Confidence(Enum):
    """How unambiguous the expected answer is."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RewardScore(Enum):
    """Ternary reward signal for physics verification.

    +1: World model output matches expected physical outcome.
     0: Insufficient visual evidence to determine the outcome.
    -1: World model output violates expected physical outcome.
    """

    CORRECT = 1
    UNDETERMINED = 0
    INCORRECT = -1


@dataclass(frozen=True)
class Scenario:
    """A single physics-verifiable scenario for world model evaluation.

    Each scenario describes a 3D scene, an action that triggers a physical event,
    and a yes/no verification question with a known ground-truth answer.
    """

    scenario_id: str
    category: str
    world_prompt: str
    action: str
    verification_question: str
    expected_answer: str
    confidence: Confidence
    video_prompt: str = ""

    def to_dict(self) -> dict[str, str]:
        """Convert to a flat dictionary for CSV export."""
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "world_prompt": self.world_prompt,
            "action": self.action,
            "verification_question": self.verification_question,
            "expected_answer": self.expected_answer,
            "confidence": self.confidence.value,
            "video_prompt": self.video_prompt,
        }


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a single scenario against world model output.

    Contains the VLM's assessment, the comparison with ground truth,
    and the resulting reward score.
    """

    scenario_id: str
    category: str
    verification_question: str
    expected_answer: str
    vlm_answer: str
    vlm_reasoning: str
    reward: RewardScore
    video_path: str

    def to_dict(self) -> dict[str, str]:
        """Convert to a flat dictionary for CSV export."""
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "verification_question": self.verification_question,
            "expected_answer": self.expected_answer,
            "vlm_answer": self.vlm_answer,
            "vlm_reasoning": self.vlm_reasoning,
            "reward": str(self.reward.value),
            "video_path": self.video_path,
        }


@dataclass(frozen=True)
class CategoryConfig:
    """Configuration for a single physics category within a domain."""

    name: str
    description: str
    example_scenarios: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DomainConfig:
    """Configuration for a physics domain (e.g., autonomous driving, public safety).

    Loaded from YAML config files. Defines the categories, context,
    and generation parameters for a specific evaluation domain.
    """

    domain_id: str
    domain_name: str
    description: str
    context_prompt: str
    categories: list[CategoryConfig]
    id_prefix: str

    @property
    def category_names(self) -> list[str]:
        """Return list of category names in this domain."""
        return [cat.name for cat in self.categories]
