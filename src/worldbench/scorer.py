"""Scoring and reporting for WorldBench verification results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from worldbench.models import RewardScore, VerificationResult


RESULT_FIELDNAMES = [
    "scenario_id",
    "category",
    "verification_question",
    "expected_answer",
    "vlm_answer",
    "vlm_reasoning",
    "reward",
    "video_path",
]


def write_results(results: list[VerificationResult], output_path: Path) -> Path:
    """Write verification results to a CSV file.

    Args:
        results: List of VerificationResult instances.
        output_path: Path for the output CSV file.

    Returns:
        The path to the written CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    return output_path


def print_score_report(results: list[VerificationResult]) -> None:
    """Print a formatted scoring report to the terminal.

    Shows per-category and overall scores with reward distribution.

    Args:
        results: List of VerificationResult instances.
    """
    if not results:
        print("No results to report.")
        return

    print("\n" + "=" * 70)
    print("  WorldBench — Verification Score Report")
    print("=" * 70)

    by_category: dict[str, list[VerificationResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)

    for category, cat_results in sorted(by_category.items()):
        _print_category_score(category, cat_results)

    _print_overall_score(results)
    print("=" * 70)


def _print_category_score(category: str, results: list[VerificationResult]) -> None:
    """Print score summary for a single category."""
    correct = sum(1 for r in results if r.reward == RewardScore.CORRECT)
    incorrect = sum(1 for r in results if r.reward == RewardScore.INCORRECT)
    undetermined = sum(1 for r in results if r.reward == RewardScore.UNDETERMINED)
    total = len(results)
    total_reward = sum(r.reward.value for r in results)

    evaluable = correct + incorrect
    accuracy = (correct / evaluable * 100) if evaluable > 0 else 0.0

    print(f"\n  📂 {category}")
    print(f"     Scenarios: {total} | ✅ {correct} | ❌ {incorrect} | ❓ {undetermined}")
    print(f"     Accuracy (evaluable only): {accuracy:.0f}%")
    print(f"     Total reward: {total_reward:+d}")


def _print_overall_score(results: list[VerificationResult]) -> None:
    """Print overall score summary across all categories."""
    correct = sum(1 for r in results if r.reward == RewardScore.CORRECT)
    incorrect = sum(1 for r in results if r.reward == RewardScore.INCORRECT)
    undetermined = sum(1 for r in results if r.reward == RewardScore.UNDETERMINED)
    total = len(results)
    total_reward = sum(r.reward.value for r in results)

    evaluable = correct + incorrect
    accuracy = (correct / evaluable * 100) if evaluable > 0 else 0.0

    print(f"\n  {'─' * 50}")
    print(f"  📊 OVERALL")
    print(f"     Total scenarios: {total}")
    print(f"     ✅ Correct:      {correct}")
    print(f"     ❌ Incorrect:    {incorrect}")
    print(f"     ❓ Undetermined: {undetermined}")
    print(f"     Accuracy (evaluable only): {accuracy:.0f}%")
    print(f"     Total reward: {total_reward:+d} / {total}")
