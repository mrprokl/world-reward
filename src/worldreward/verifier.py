"""Physics verification of world model outputs using Gemini 3 Pro as VLM judge."""

from __future__ import annotations

import json
import time
from pathlib import Path

from google import genai

from worldreward.dataset_writer import load_scenarios_csv
from worldreward.exceptions import GeminiAPIError, VerificationError
from worldreward.models import RewardScore, VerificationResult
from worldreward.paths import resolve_api_key
from worldreward.spinner import Spinner


class Verifier:
    """Verifies world model video outputs against physics ground truth.

    Uses Gemini 3 Pro to analyze generated videos and answer
    the verification question. Compares the VLM answer against
    the expected ground truth to produce a ternary reward score.

    Reward scores:
        +1 (CORRECT):      VLM answer matches expected answer.
         0 (UNDETERMINED):  VLM cannot determine the answer from the video.
        -1 (INCORRECT):     VLM answer contradicts expected answer.
    """

    DEFAULT_MODEL = "gemini-3-pro-preview"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Initialize the verifier with a Gemini client.

        Args:
            api_key: Google AI Studio API key. Falls back to GEMINI_API_KEY env var.
            model: Gemini model name. Defaults to gemini-3-pro-preview.

        Raises:
            GeminiAPIError: If no API key is provided or found in environment.
        """
        resolved_key = resolve_api_key(api_key)
        if not resolved_key:
            raise GeminiAPIError(
                "No API key provided. Set GEMINI_API_KEY, use ~/.worldreward/config.toml, or pass api_key."
            )
        self._client = genai.Client(api_key=resolved_key)
        self._model = model or self.DEFAULT_MODEL

    def verify_dataset(
        self,
        dataset_path: Path,
        videos_dir: Path,
    ) -> list[VerificationResult]:
        """Verify all scenarios in a dataset against their generated videos.

        Args:
            dataset_path: Path to the scenario CSV file.
            videos_dir: Directory containing generated video files (named {scenario_id}.mp4).

        Returns:
            List of VerificationResult instances.
        """
        scenarios = load_scenarios_csv(dataset_path)
        results: list[VerificationResult] = []

        print(f"🔍 Verifying {len(scenarios)} scenarios with Gemini 3 Pro...")

        for idx, scenario in enumerate(scenarios, start=1):
            scenario_id = scenario["scenario_id"]
            video_path = videos_dir / f"{scenario_id}.mp4"

            if not video_path.exists():
                print(f"⚠️  [{idx}/{len(scenarios)}] {scenario_id}: video not found, skipping")
                continue

            print(f"🔎 [{idx}/{len(scenarios)}] {scenario_id}: verifying...")

            try:
                result = self._verify_single(scenario, video_path)
                results.append(result)
                reward_icon = {1: "✅", 0: "❓", -1: "❌"}[result.reward.value]
                print(f"{reward_icon} [{idx}/{len(scenarios)}] {scenario_id}: "
                      f"reward={result.reward.value:+d} | "
                      f"expected={result.expected_answer} | "
                      f"vlm={result.vlm_answer}")
            except VerificationError as e:
                print(f"💥 [{idx}/{len(scenarios)}] {scenario_id}: {e}")

        return results

    def _verify_single(self, scenario: dict, video_path: Path) -> VerificationResult:
        """Verify a single scenario against its video.

        Args:
            scenario: Scenario dict from CSV.
            video_path: Path to the video file.

        Returns:
            VerificationResult with reward score.

        Raises:
            VerificationError: If verification fails.
        """
        prompt = _build_verification_prompt(scenario["verification_question"])

        try:
            with Spinner("Uploading video"):
                video_file = self._client.files.upload(file=str(video_path))

            # Wait for file to become ACTIVE (processing takes a few seconds)
            state_name = video_file.state.name if video_file.state else None
            with Spinner("Processing video"):
                while state_name == "PROCESSING":
                    time.sleep(2)
                    if not video_file.name:
                        raise VerificationError(
                            scenario["scenario_id"],
                            "Uploaded video has no retrievable file name.",
                        )
                    video_file = self._client.files.get(name=video_file.name)
                    state_name = video_file.state.name if video_file.state else None

            if state_name != "ACTIVE":
                raise VerificationError(
                    scenario["scenario_id"],
                    f"File upload failed — state: {state_name or 'UNKNOWN'}",
                )

            with Spinner("Analyzing with Gemini"):
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[
                        video_file,
                        prompt,
                    ],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "enum": ["yes", "no", "undetermined"],
                                },
                                "reasoning": {
                                    "type": "string",
                                },
                            },
                            "required": ["answer", "reasoning"],
                        },
                    ),
                )
            response_text = response.text
            if not response_text:
                raise VerificationError(
                    scenario["scenario_id"],
                    "Gemini returned an empty verification response.",
                )
            vlm_answer, vlm_reasoning = _parse_verification_response(response_text)
        except VerificationError:
            raise
        except Exception as e:
            raise VerificationError(scenario["scenario_id"], str(e)) from e

        reward = _compute_reward(vlm_answer, scenario["expected_answer"])

        return VerificationResult(
            scenario_id=scenario["scenario_id"],
            category=scenario["category"],
            verification_question=scenario["verification_question"],
            expected_answer=scenario["expected_answer"],
            vlm_answer=vlm_answer,
            vlm_reasoning=vlm_reasoning,
            reward=reward,
            video_path=str(video_path),
        )


def _build_verification_prompt(verification_question: str) -> str:
    """Build the VLM verification prompt.

    The prompt asks Gemini to watch the video and answer the verification
    question WITHOUT knowing the expected answer (no cheating).

    Args:
        verification_question: The yes/no physics question.

    Returns:
        Formatted verification prompt.
    """
    return f"""You are a physics verification judge. Watch this video carefully and answer the following question based ONLY on what you observe in the video.

QUESTION: {verification_question}

INSTRUCTIONS:
1. Watch the entire video carefully.
2. Focus on the physical outcome shown in the video.
3. Answer based ONLY on what is visually observable.
4. If the video does not show enough information to answer the question, say "undetermined".

Respond with ONLY a valid JSON object with these exact keys:
- "answer": either "yes", "no", or "undetermined"
- "reasoning": a brief explanation (1-2 sentences) of what you observed that led to your answer

Example: {{"answer": "no", "reasoning": "The car shows significant deformation after impact, it did not retain its original shape."}}

Your response (JSON only):"""


def _parse_verification_response(text: str) -> tuple[str, str]:
    """Parse the VLM verification response.

    Args:
        text: Raw response from Gemini.

    Returns:
        Tuple of (answer, reasoning).

    Raises:
        VerificationError: If response cannot be parsed.
    """
    cleaned = text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if line.strip() != "```"]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise VerificationError("unknown", f"Invalid JSON from VLM: {e}") from e

    answer = parsed.get("answer", "undetermined").lower().strip()
    reasoning = parsed.get("reasoning", "No reasoning provided.")

    if answer not in ("yes", "no", "undetermined"):
        answer = "undetermined"

    return answer, reasoning


def _compute_reward(vlm_answer: str, expected_answer: str) -> RewardScore:
    """Compute the ternary reward by comparing VLM answer to ground truth.

    Args:
        vlm_answer: The VLM's answer ("yes", "no", or "undetermined").
        expected_answer: The ground truth answer ("yes" or "no").

    Returns:
        RewardScore: +1 if match, -1 if mismatch, 0 if undetermined.
    """
    if vlm_answer == "undetermined":
        return RewardScore.UNDETERMINED
    if vlm_answer == expected_answer.lower():
        return RewardScore.CORRECT
    return RewardScore.INCORRECT
