from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import worldreward.verifier as verifier
import worldreward.video_generator as video_generator


def test_video_generator_marks_pending_operations_as_failed_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        (
            "scenario_id,category,world_prompt,action,video_prompt,verification_question,"
            "expected_answer,confidence\n"
            "PS-001,cat,world,action,video prompt,q?,yes,high\n"
        ),
        encoding="utf-8",
    )

    class FakeModels:
        def generate_videos(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(done=False)

    class FakeOperations:
        def get(self, operation: SimpleNamespace) -> SimpleNamespace:
            return operation

    generator = cast(Any, video_generator.VideoGenerator.__new__(video_generator.VideoGenerator))
    generator._model = "veo-test"
    generator._client = SimpleNamespace(
        models=FakeModels(),
        operations=FakeOperations(),
    )
    monkeypatch.setattr(video_generator.VideoGenerator, "MAX_POLL_SECONDS", 0)

    results = generator.generate_from_dataset(dataset_path, tmp_path / "videos")

    assert results == [{"scenario_id": "PS-001", "video_path": ""}]


def test_verifier_fails_fast_when_uploaded_video_stays_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoOpSpinner:
        def __init__(self, _message: str) -> None:
            pass

        def __enter__(self) -> NoOpSpinner:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeFiles:
        def __init__(self) -> None:
            self.deleted_names: list[str] = []

        def upload(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                name="file-1",
                state=SimpleNamespace(name="PROCESSING"),
            )

        def get(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                name="file-1",
                state=SimpleNamespace(name="PROCESSING"),
            )

        def delete(self, *, name: str, **_kwargs: object) -> None:
            self.deleted_names.append(name)

    monkeypatch.setattr(verifier, "Spinner", NoOpSpinner)
    monkeypatch.setattr(verifier.Verifier, "MAX_VIDEO_PROCESS_SECONDS", 0)

    fake_files = FakeFiles()
    test_verifier = cast(Any, verifier.Verifier.__new__(verifier.Verifier))
    test_verifier._model = "gemini-test"
    test_verifier._client = SimpleNamespace(files=fake_files)

    scenario = {
        "scenario_id": "PS-001",
        "category": "cat",
        "verification_question": "q?",
        "expected_answer": "yes",
    }
    video_path = tmp_path / "PS-001.mp4"
    video_path.write_bytes(b"fake")

    with pytest.raises(verifier.VerificationError, match="Timed out after"):
        test_verifier._verify_single(scenario, video_path)
    assert fake_files.deleted_names == ["file-1"]


def test_verifier_deletes_uploaded_file_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoOpSpinner:
        def __init__(self, _message: str) -> None:
            pass

        def __enter__(self) -> NoOpSpinner:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeFiles:
        def __init__(self) -> None:
            self.deleted_names: list[str] = []

        def upload(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                name="file-2",
                state=SimpleNamespace(name="ACTIVE"),
            )

        def delete(self, *, name: str, **_kwargs: object) -> None:
            self.deleted_names.append(name)

    class FakeModels:
        def generate_content(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(text='{"answer":"yes","reasoning":"looks correct"}')

    monkeypatch.setattr(verifier, "Spinner", NoOpSpinner)

    fake_files = FakeFiles()
    test_verifier = cast(Any, verifier.Verifier.__new__(verifier.Verifier))
    test_verifier._model = "gemini-test"
    test_verifier._client = SimpleNamespace(files=fake_files, models=FakeModels())

    scenario = {
        "scenario_id": "PS-002",
        "category": "cat",
        "verification_question": "q?",
        "expected_answer": "yes",
    }
    video_path = tmp_path / "PS-002.mp4"
    video_path.write_bytes(b"fake")

    result = test_verifier._verify_single(scenario, video_path)

    assert result.scenario_id == "PS-002"
    assert result.vlm_answer == "yes"
    assert fake_files.deleted_names == ["file-2"]
