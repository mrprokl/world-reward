"""Veo 3.1 video generation from scenario prompts.

Supports parallel generation: all API operations are launched concurrently,
then polled together until completion.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from worldreward.dataset_writer import load_scenarios_csv
from worldreward.exceptions import GeminiAPIError, VideoGenerationError


NEGATIVE_PROMPT = "cartoon, drawing, animation, low quality, blurry, watermark, text overlay"


@dataclass
class _PendingVideo:
    """Tracks an in-flight Veo generation operation."""

    scenario_id: str
    operation: object  # google.genai Operation
    output_path: Path
    done: bool = False
    error: str | None = None


class VideoGenerator:
    """Generates videos from scenario prompts using Veo 3.1.

    Reads scenarios from a CSV dataset, sends each video_prompt to Veo 3.1,
    and saves the resulting videos to an output directory.
    All generation requests are launched in parallel, then polled together.
    """

    DEFAULT_MODEL = "veo-3.1-generate-preview"
    POLL_INTERVAL_SECONDS = 10

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Initialize the Veo client.

        Args:
            api_key: Google AI Studio API key. Falls back to GEMINI_API_KEY env var.
            model: Veo model name. Defaults to veo-3.1-generate-preview.

        Raises:
            GeminiAPIError: If no API key is provided or found in environment.
        """
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise GeminiAPIError("No API key provided. Set GEMINI_API_KEY in .env or pass api_key.")
        self._client = genai.Client(api_key=resolved_key)
        self._model = model or self.DEFAULT_MODEL

    def generate_from_dataset(self, dataset_path: Path, output_dir: Path) -> list[dict]:
        """Generate videos for all scenarios in a CSV dataset (parallel).

        All Veo operations are launched first, then polled concurrently.

        Args:
            dataset_path: Path to the scenario CSV file.
            output_dir: Directory to save generated videos.

        Returns:
            List of dicts mapping scenario_id to video file path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        scenarios = load_scenarios_csv(dataset_path)
        results: list[dict] = []
        pending: list[_PendingVideo] = []

        print(f"🎬 Generating {len(scenarios)} videos with Veo 3.1 (parallel)...")
        print(f"   Model: {self._model} | Duration: 8s | Resolution: 720p")

        # --- Phase 1: launch all operations ---
        for idx, scenario in enumerate(scenarios, start=1):
            scenario_id = scenario["scenario_id"]
            video_prompt = scenario.get("video_prompt", "")

            if not video_prompt:
                print(f"⚠️  [{idx}/{len(scenarios)}] {scenario_id}: no video_prompt, skipping")
                continue

            video_path = output_dir / f"{scenario_id}.mp4"
            if video_path.exists():
                print(f"⏭️  [{idx}/{len(scenarios)}] {scenario_id}: already exists, skipping")
                results.append({"scenario_id": scenario_id, "video_path": str(video_path)})
                continue

            try:
                print(f"🚀 [{idx}/{len(scenarios)}] {scenario_id}: launching...")
                operation = self._client.models.generate_videos(
                    model=self._model,
                    prompt=video_prompt,
                    config=types.GenerateVideosConfig(
                        negative_prompt=NEGATIVE_PROMPT,
                        aspect_ratio="16:9",
                        person_generation="allow_all",
                    ),
                )
                pending.append(_PendingVideo(
                    scenario_id=scenario_id,
                    operation=operation,
                    output_path=video_path,
                ))
            except Exception as e:
                print(f"❌ [{idx}/{len(scenarios)}] {scenario_id}: launch failed — {e}")
                results.append({"scenario_id": scenario_id, "video_path": ""})

        if not pending:
            print("\n🎬 No videos to generate.")
            return results

        print(f"\n⏳ {len(pending)} operations launched — polling for completion...")

        # --- Phase 2: poll all operations concurrently ---
        while any(not p.done for p in pending):
            time.sleep(self.POLL_INTERVAL_SECONDS)
            for p in pending:
                if p.done:
                    continue
                try:
                    p.operation = self._client.operations.get(p.operation)
                    if p.operation.done:
                        p.done = True
                        self._save_video(p)
                        print(f"✅ {p.scenario_id}: saved to {p.output_path}")
                except Exception as e:
                    p.done = True
                    p.error = str(e)
                    print(f"❌ {p.scenario_id}: poll failed — {e}")

            remaining = sum(1 for p in pending if not p.done)
            if remaining:
                print(f"   ⏳ {remaining} still generating...")

        # --- Collect results ---
        for p in pending:
            if p.error:
                results.append({"scenario_id": p.scenario_id, "video_path": ""})
            else:
                results.append({"scenario_id": p.scenario_id, "video_path": str(p.output_path)})

        successful = sum(1 for r in results if r["video_path"])
        total = len(scenarios)
        print(f"\n🎬 Video generation complete: {successful}/{total} succeeded")
        return results

    def _save_video(self, pending: _PendingVideo) -> None:
        """Download and save a completed video.

        Raises:
            VideoGenerationError: If download or save fails.
        """
        try:
            generated_video = pending.operation.response.generated_videos[0]
            self._client.files.download(file=generated_video.video)
            generated_video.video.save(str(pending.output_path))
        except Exception as e:
            pending.error = str(e)
            raise VideoGenerationError(pending.scenario_id, str(e)) from e
