from __future__ import annotations

from pathlib import Path

from worldreward.models import RewardScore, VerificationResult
from worldreward.scorer import print_score_report, write_results


def _make_result(scenario_id: str, category: str, reward: RewardScore) -> VerificationResult:
    return VerificationResult(
        scenario_id=scenario_id,
        category=category,
        verification_question="question",
        expected_answer="yes",
        vlm_answer="yes" if reward == RewardScore.CORRECT else "no",
        vlm_reasoning="reasoning",
        reward=reward,
        video_path=f"/tmp/{scenario_id}.mp4",
    )


def test_write_results_creates_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "results.csv"
    results = [
        _make_result("A-001", "cat_a", RewardScore.CORRECT),
        _make_result("A-002", "cat_a", RewardScore.INCORRECT),
    ]

    written = write_results(results, output_path)

    assert written.exists()
    content = written.read_text(encoding="utf-8")
    assert "scenario_id,category,verification_question" in content
    assert "A-001" in content


def test_print_score_report_outputs_summary(capsys) -> None:
    results = [
        _make_result("A-001", "cat_a", RewardScore.CORRECT),
        _make_result("A-002", "cat_a", RewardScore.INCORRECT),
        _make_result("B-001", "cat_b", RewardScore.UNDETERMINED),
    ]

    print_score_report(results)
    captured = capsys.readouterr().out

    assert "OVERALL" in captured
    assert "cat_a" in captured
    assert "Total scenarios: 3" in captured
