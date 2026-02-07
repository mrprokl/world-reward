"""Scenario generation orchestrator for World Reward."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from worldreward.config_loader import load_domain_config
from worldreward.dataset_writer import write_dataset
from worldreward.exceptions import DatasetGenerationError
from worldreward.gemini_client import GeminiClient
from worldreward.models import Confidence, DomainConfig, Scenario
from worldreward.prompt_builder import build_generation_prompt


class ScenarioGenerator:
    """Orchestrates the end-to-end scenario generation pipeline.

    Loads domain config, builds prompts, calls Gemini, parses results,
    and writes the final CSV dataset.
    """

    def __init__(self, gemini_client: GeminiClient) -> None:
        self._client = gemini_client

    def generate(
        self,
        config_path: Path,
        count: int,
        output_dir: Path,
    ) -> Path:
        """Generate a dataset of physics-verifiable scenarios.

        Args:
            config_path: Path to the domain YAML config file.
            count: Number of scenarios to generate.
            output_dir: Directory where the CSV will be saved.

        Returns:
            Path to the generated CSV file.

        Raises:
            DatasetGenerationError: If generation fails at any step.
        """
        config = load_domain_config(config_path)
        prompt = build_generation_prompt(config, count)

        print(f"🔧 Domain: {config.domain_name}")
        print(f"📊 Requesting {count} scenarios across {len(config.categories)} categories...")
        print(f"🤖 Calling Gemini API...")

        try:
            raw_scenarios = self._client.generate_scenarios_json(prompt)
        except Exception as e:
            raise DatasetGenerationError(config.domain_id, str(e)) from e

        scenarios = self._parse_raw_scenarios(raw_scenarios, config)

        print(f"✅ Generated {len(scenarios)} valid scenarios")

        output_path = self._build_output_path(config, output_dir)
        write_dataset(scenarios, output_path)

        print(f"💾 Dataset saved to: {output_path}")
        return output_path

    @staticmethod
    def _parse_raw_scenarios(
        raw_scenarios: list[dict],
        config: DomainConfig,
    ) -> list[Scenario]:
        """Convert raw JSON dictionaries into validated Scenario instances.

        Args:
            raw_scenarios: List of dicts from Gemini response.
            config: Domain config for ID prefix generation.

        Returns:
            List of validated Scenario instances.
        """
        scenarios: list[Scenario] = []
        for idx, raw in enumerate(raw_scenarios, start=1):
            try:
                scenario = Scenario(
                    scenario_id=f"{config.id_prefix}-{idx:03d}",
                    category=raw["category"],
                    world_prompt=raw["world_prompt"],
                    action=raw["action"],
                    verification_question=raw["verification_question"],
                    expected_answer=raw["expected_answer"].lower(),
                    confidence=Confidence(raw.get("confidence", "medium").lower()),
                    video_prompt=raw.get("video_prompt", ""),
                )
                scenarios.append(scenario)
            except (KeyError, ValueError) as e:
                print(f"⚠️  Skipping invalid scenario {idx}: {e}")
                continue

        return scenarios

    @staticmethod
    def _build_output_path(config: DomainConfig, output_dir: Path) -> Path:
        """Build a timestamped output file path."""
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{config.domain_id}_{timestamp}.csv"
        return output_dir / filename
