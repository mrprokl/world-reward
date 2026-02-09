"""Gemini API client wrapper for World Reward."""

from __future__ import annotations

import json

from google import genai

from worldreward.exceptions import GeminiAPIError, ParsingError
from worldreward.paths import resolve_api_key


class GeminiClient:
    """Wrapper around the Google Gemini API for structured scenario generation.

    Uses the google-genai SDK to call Gemini models and parse JSON responses.
    """

    DEFAULT_MODEL = "gemini-3-pro-preview"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Initialize the Gemini client.

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

    def generate_scenarios_json(self, prompt: str) -> list[dict]:
        """Send a prompt to Gemini and parse the JSON array response.

        Args:
            prompt: The full generation prompt.

        Returns:
            List of scenario dictionaries parsed from Gemini's JSON response.

        Raises:
            GeminiAPIError: If the API call fails.
            ParsingError: If the response cannot be parsed as JSON.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.8,
                    top_p=0.95,
                ),
            )
        except Exception as e:
            raise GeminiAPIError(str(e)) from e

        response_text = response.text
        if not response_text:
            raise ParsingError("Empty response body from Gemini API.")

        return self._parse_json_response(response_text)

    @staticmethod
    def _parse_json_response(text: str) -> list[dict]:
        """Parse Gemini response text as a JSON array.

        Handles common issues like markdown code fences wrapping the JSON.

        Args:
            text: Raw response text from Gemini.

        Returns:
            Parsed list of dictionaries.

        Raises:
            ParsingError: If parsing fails.
        """
        cleaned = text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [line for line in lines[1:] if line.strip() != "```"]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ParsingError(f"Invalid JSON: {e}\nRaw response:\n{text[:500]}") from e

        if not isinstance(parsed, list):
            raise ParsingError(f"Expected JSON array, got {type(parsed).__name__}")

        return parsed
