"""Build structured prompts for Gemini scenario generation and Veo video generation."""

from __future__ import annotations

from worldbench.models import DomainConfig


def build_generation_prompt(config: DomainConfig, count: int) -> str:
    """Build a structured prompt for Gemini to generate physics-verifiable scenarios.

    The generated scenarios include a `video_prompt` field: a cinematic prompt
    designed for Veo 3.1 that describes the full scene and physical event,
    WITHOUT revealing the expected outcome (to avoid biasing the video model).

    Args:
        config: Domain configuration with categories and context.
        count: Number of scenarios to generate.

    Returns:
        Formatted prompt string ready for Gemini API.
    """
    categories_block = _build_categories_block(config)
    return f"""{config.context_prompt.strip()}

---

DOMAIN: {config.domain_name}
DESCRIPTION: {config.description.strip()}

CATEGORIES AND EXAMPLES:
{categories_block}

---

TASK: Generate exactly {count} diverse physics-verifiable scenarios spread across the categories above.

OUTPUT FORMAT: Return ONLY a valid JSON array. Each element must have these exact keys:
- "category": one of the category names listed above
- "world_prompt": a vivid, detailed scene description (location, weather, lighting, objects present). 2-3 sentences.
- "action": the specific physical event or action that triggers the test. 1 sentence.
- "video_prompt": a CINEMATIC video generation prompt for Veo 3.1 that combines the scene and action into a single continuous shot description. Must describe the FULL EVENT unfolding visually (setup → action → aftermath) so the physical outcome is VISIBLE in the generated video. Do NOT mention the expected result — just describe what happens cinematically. 2-4 sentences. Use film language (camera angles, shot types, lighting).
- "verification_question": a precise yes/no question about the physical outcome. Must be answerable by WATCHING the generated video.
- "expected_answer": either "yes" or "no" — the physically correct answer.
- "confidence": "high" if the answer is near-certain based on physics, "medium" if very likely but edge cases exist, "low" if debatable.

RULES:
1. Each scenario must test a SPECIFIC, UNAMBIGUOUS physical law or principle.
2. The verification_question must be answerable by WATCHING the video output — the physical outcome must be VISIBLE.
3. The video_prompt must show enough of the event's aftermath for a viewer to judge the outcome. Include the consequence in the shot description.
4. The video_prompt must NEVER mention the expected answer or hint at correctness. It is a neutral cinematic description.
5. Distribute scenarios roughly evenly across categories.
6. Vary the settings (different locations, times of day, weather, camera angles).
7. Prefer "high" confidence scenarios — these are unit tests for reality.
8. Do NOT include any markdown formatting, code fences, or explanations — ONLY the JSON array.

EXAMPLE OUTPUT:
[
  {{
    "category": "vehicle_collision",
    "world_prompt": "Realistic highway, clear weather, daytime. A silver supercar travels at high speed on a straight section with a metal guardrail on the right side.",
    "action": "The car veers right and hits the metal guardrail at 300 km/h.",
    "video_prompt": "Cinematic wide shot of a silver supercar racing down a sunlit highway at extreme speed. The car suddenly veers right and slams into a metal guardrail. Close-up tracking shot captures the moment of impact and the immediate aftermath, showing the state of both the car body and the guardrail. Slow motion, photorealistic, shot on ARRI Alexa.",
    "verification_question": "Does the car retain its original undamaged shape after the impact?",
    "expected_answer": "no",
    "confidence": "high"
  }}
]

Generate {count} scenarios now:"""


def _build_categories_block(config: DomainConfig) -> str:
    """Build the categories section of the prompt."""
    lines: list[str] = []
    for cat in config.categories:
        lines.append(f"\n### {cat.name}")
        lines.append(f"Description: {cat.description.strip()}")
        if cat.example_scenarios:
            lines.append("Examples:")
            for example in cat.example_scenarios:
                lines.append(f"  - {example}")
    return "\n".join(lines)
