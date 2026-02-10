from __future__ import annotations

from pathlib import Path

import pytest

import worldreward.cli as cli
from worldreward.exceptions import WorldRewardError
from worldreward.models import RewardScore, VerificationResult


def test_run_generate_happy_path_invokes_generator_with_expected_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "configs" / "public_safety.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("domain_id: public_safety\n", encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, model: str | None = None) -> None:
            captured["model"] = model

    class FakeGenerator:
        def __init__(self, _client: FakeClient) -> None:
            captured["client_initialized"] = True

        def generate(self, config_path: Path, count: int, output_dir: Path) -> Path:
            captured["config_path"] = config_path
            captured["count"] = count
            captured["output_dir"] = output_dir
            return output_dir / "public_safety_20260210_120000.csv"

    monkeypatch.setattr(
        cli,
        "resolve_domain_config_path",
        lambda domain, _dirs: config_path if domain == "public_safety" else None,
    )
    monkeypatch.setattr(cli, "GeminiClient", FakeClient)
    monkeypatch.setattr(cli, "ScenarioGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(cli, "CONFIG_SEARCH_DIRS", [tmp_path / "configs"])

    cli.run_generate("public_safety", count=4, model="gemini-test")
    output = capsys.readouterr().out

    assert "Done! Dataset" in output
    assert captured["model"] == "gemini-test"
    assert captured["config_path"] == config_path
    assert captured["count"] == 4
    assert captured["output_dir"] == tmp_path / "datasets"


def test_run_generate_handles_domain_not_found(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_domain_config_path", lambda _d, _s: None)
    monkeypatch.setattr(cli, "list_available_domains", lambda _s: ["autonomous_driving"])
    monkeypatch.setattr(cli, "CONFIG_SEARCH_DIRS", [])

    cli.run_generate("missing_domain")
    output = capsys.readouterr().out

    assert "Domain 'missing_domain' not found" in output
    assert "autonomous_driving" in output


def test_run_generate_handles_worldreward_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "configs" / "public_safety.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("x", encoding="utf-8")

    class FakeClient:
        def __init__(self, model: str | None = None) -> None:
            _ = model

    class FakeGenerator:
        def __init__(self, _client: FakeClient) -> None:
            pass

        def generate(self, config_path: Path, count: int, output_dir: Path) -> Path:
            _ = (config_path, count, output_dir)
            raise WorldRewardError("boom")

    monkeypatch.setattr(cli, "resolve_domain_config_path", lambda _d, _s: config_path)
    monkeypatch.setattr(cli, "GeminiClient", FakeClient)
    monkeypatch.setattr(cli, "ScenarioGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(cli, "CONFIG_SEARCH_DIRS", [tmp_path / "configs"])

    cli.run_generate("public_safety")
    output = capsys.readouterr().out

    assert "Error: boom" in output


def test_run_videos_happy_path_uses_run_stem_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "datasets" / "public_safety_20260210_120000.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("scenario_id\nPS-001\n", encoding="utf-8")
    monkeypatch.setattr(cli, "VIDEOS_DIR", tmp_path / "videos")

    captured: dict[str, object] = {}

    class FakeVideoGenerator:
        def generate_from_dataset(self, dataset_path: Path, output_dir: Path) -> list[dict]:
            captured["dataset_path"] = dataset_path
            captured["output_dir"] = output_dir
            return []

    monkeypatch.setattr(cli, "VideoGenerator", FakeVideoGenerator)

    cli.run_videos(str(dataset_path))
    output = capsys.readouterr().out

    assert "Done! Videos saved to" in output
    assert captured["dataset_path"] == dataset_path
    assert captured["output_dir"] == (tmp_path / "videos" / dataset_path.stem)


def test_run_verify_happy_path_writes_results_and_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "public_safety_20260210_120000"
    dataset_path = tmp_path / "datasets" / f"{run_id}.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("scenario_id\nPS-001\n", encoding="utf-8")

    videos_dir = tmp_path / "videos" / run_id
    videos_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cli, "VIDEOS_DIR", tmp_path / "videos")
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")

    results = [
        VerificationResult(
            scenario_id="PS-001",
            category="cat",
            verification_question="q?",
            expected_answer="yes",
            vlm_answer="yes",
            vlm_reasoning="ok",
            reward=RewardScore.CORRECT,
            video_path=str(videos_dir / "PS-001.mp4"),
        )
    ]
    captured: dict[str, object] = {}

    class FakeVerifier:
        def verify_dataset(self, dataset_path: Path, videos_dir: Path) -> list[VerificationResult]:
            captured["dataset_path"] = dataset_path
            captured["videos_dir"] = videos_dir
            return results

    def fake_write_results(payload: list[VerificationResult], output_path: Path) -> Path:
        captured["write_results_payload"] = payload
        captured["write_results_path"] = output_path
        return output_path

    def fake_print_score_report(payload: list[VerificationResult]) -> None:
        captured["score_payload"] = payload

    monkeypatch.setattr(cli, "Verifier", FakeVerifier)
    monkeypatch.setattr(cli, "write_results", fake_write_results)
    monkeypatch.setattr(cli, "print_score_report", fake_print_score_report)

    cli.run_verify(str(dataset_path))
    output = capsys.readouterr().out

    assert "Results saved to" in output
    assert captured["dataset_path"] == dataset_path
    assert captured["videos_dir"] == videos_dir
    assert captured["write_results_payload"] == results
    assert captured["score_payload"] == results
    assert captured["write_results_path"] == (tmp_path / "results" / f"results_{run_id}.csv")


def test_run_setup_and_run_config_wire_to_setup_module(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run_setup_wizard() -> bool:
        captured["setup_called"] = True
        return True

    def fake_configure_api_key_interactive() -> bool:
        captured["set_api_key_called"] = True
        return True

    def fake_render_config_summary(show_api_key: bool = False) -> str:
        captured["show_api_key"] = show_api_key
        return "summary-output"

    monkeypatch.setattr(cli, "run_setup_wizard", fake_run_setup_wizard)
    monkeypatch.setattr(cli, "configure_api_key_interactive", fake_configure_api_key_interactive)
    monkeypatch.setattr(cli, "render_config_summary", fake_render_config_summary)

    cli.run_setup()
    cli.run_config(set_api_key=True, show_api_key=True)
    output = capsys.readouterr().out

    assert captured["setup_called"] is True
    assert captured["set_api_key_called"] is True
    assert captured["show_api_key"] is True
    assert "summary-output" in output
