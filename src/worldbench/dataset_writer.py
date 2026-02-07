"""CSV dataset writer for generated scenarios."""

from __future__ import annotations

import csv
from pathlib import Path

from worldbench.models import Scenario

FIELDNAMES = [
    "scenario_id",
    "category",
    "world_prompt",
    "action",
    "video_prompt",
    "verification_question",
    "expected_answer",
    "confidence",
]


def write_dataset(scenarios: list[Scenario], output_path: Path) -> Path:
    """Write a list of scenarios to a CSV file.

    Args:
        scenarios: List of Scenario instances to export.
        output_path: Path for the output CSV file.

    Returns:
        The path to the written CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for scenario in scenarios:
            writer.writerow(scenario.to_dict())

    return output_path
