"""Build structured prompts for Gemini scenario generation and Veo video generation."""

from __future__ import annotations

from worldreward.models import DomainConfig


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
- "video_prompt": a CINEMATIC video generation prompt for Veo 3.1. Describes the scene setup and the action unfolding, then the camera LINGERS on the scene so the physical outcome is visible. CRITICAL: do NOT describe the physical result or aftermath — only describe the setup, the action happening, and the camera staying to observe. Let the video model decide what the outcome looks like. 2-4 sentences. Use film language (camera angles, shot types, lighting).
- "verification_question": a precise yes/no question about the physical outcome. Must be answerable by WATCHING the generated video.
- "expected_answer": either "yes" or "no" — the physically correct answer.
- "confidence": "high" if the answer is near-certain based on physics, "medium" if very likely but edge cases exist, "low" if debatable.

RULES:
1. Each scenario must test a SPECIFIC, UNAMBIGUOUS physical law or principle.
2. The verification_question must be answerable by WATCHING the video output — the physical outcome must be VISIBLE.
3. The video_prompt must describe ONLY the setup and the action. The camera must stay on the scene long enough for the outcome to be observable, but the prompt must NOT describe what the outcome looks like. No words like "crumples", "shatters", "breaks", "remains intact", "bounces", "sinks", "floats" etc. in the video_prompt. The video model must render the physics on its own.
4. The video_prompt must NEVER leak the expected answer, the physical result, or any hint about correctness.
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
    "video_prompt": "Cinematic wide shot of a silver supercar racing down a sunlit highway at extreme speed. The car suddenly veers right and slams into a metal guardrail. The camera holds on the scene in the seconds following the collision. Slow motion, photorealistic, shot on ARRI Alexa.",
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
