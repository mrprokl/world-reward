"""Veo 3.1 video generation from scenario prompts."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

from worldbench.exceptions import VideoGenerationError


VEO_MODEL = "veo-3.1-generate-preview"
POLL_INTERVAL_SECONDS = 10
NEGATIVE_PROMPT = "cartoon, drawing, animation, low quality, blurry, watermark, text overlay"


class VideoGenerator:
    """Generates videos from scenario prompts using Veo 3.1.

    Reads scenarios from a CSV dataset, sends each video_prompt to Veo 3.1,
    and saves the resulting videos to an output directory.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Veo client.

        Args:
            api_key: Google AI Studio API key. Falls back to GEMINI_API_KEY env var.
        """
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise VideoGenerationError("N/A", "No API key. Set GEMINI_API_KEY.")
        self._client = genai.Client(api_key=resolved_key)

    def generate_from_dataset(self, dataset_path: Path, output_dir: Path) -> list[dict]:
        """Generate videos for all scenarios in a CSV dataset.

        Args:
            dataset_path: Path to the scenario CSV file.
            output_dir: Directory to save generated videos.

        Returns:
            List of dicts mapping scenario_id to video file path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        scenarios = self._load_scenarios(dataset_path)
        results: list[dict] = []

        print(f"🎬 Generating {len(scenarios)} videos with Veo 3.1...")
        print(f"   Model: {VEO_MODEL} | Duration: 8s | Resolution: 720p")

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

            print(f"🎥 [{idx}/{len(scenarios)}] {scenario_id}: generating...")

            try:
                video_path = self._generate_single(scenario_id, video_prompt, output_dir)
                results.append({"scenario_id": scenario_id, "video_path": str(video_path)})
                print(f"✅ [{idx}/{len(scenarios)}] {scenario_id}: saved to {video_path}")
            except VideoGenerationError as e:
                print(f"❌ [{idx}/{len(scenarios)}] {scenario_id}: {e}")
                results.append({"scenario_id": scenario_id, "video_path": ""})

        successful = sum(1 for r in results if r["video_path"])
        print(f"\n🎬 Video generation complete: {successful}/{len(scenarios)} succeeded")
        return results

    def _generate_single(self, scenario_id: str, prompt: str, output_dir: Path) -> Path:
        """Generate a single video from a prompt.

        Args:
            scenario_id: ID for the filename.
            prompt: Veo 3.1 video prompt.
            output_dir: Directory to save the video.

        Returns:
            Path to the saved video file.

        Raises:
            VideoGenerationError: If generation or download fails.
        """
        try:
            operation = self._client.models.generate_videos(
                model=VEO_MODEL,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    negative_prompt=NEGATIVE_PROMPT,
                    aspect_ratio="16:9",
                    person_generation="allow_adult",
                ),
            )

            while not operation.done:
                time.sleep(POLL_INTERVAL_SECONDS)
                operation = self._client.operations.get(operation)

            generated_video = operation.response.generated_videos[0]
            video_path = output_dir / f"{scenario_id}.mp4"

            self._client.files.download(file=generated_video.video)
            generated_video.video.save(str(video_path))

            return video_path

        except Exception as e:
            raise VideoGenerationError(scenario_id, str(e)) from e

    @staticmethod
    def _load_scenarios(dataset_path: Path) -> list[dict]:
        """Load scenarios from a CSV file."""
        with open(dataset_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
