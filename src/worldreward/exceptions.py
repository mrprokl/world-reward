"""Custom exceptions for World Reward."""


class WorldRewardError(Exception):
    """Base exception for all World Reward errors."""


class ConfigLoadError(WorldRewardError):
    """Raised when a domain configuration file cannot be loaded or is invalid."""

    def __init__(self, config_path: str, reason: str) -> None:
        super().__init__(f"Failed to load config '{config_path}': {reason}")
        self.config_path = config_path
        self.reason = reason


class GeminiAPIError(WorldRewardError):
    """Raised when the Gemini API call fails."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Gemini API error: {message}")


class DatasetGenerationError(WorldRewardError):
    """Raised when scenario generation fails."""

    def __init__(self, domain: str, reason: str) -> None:
        super().__init__(f"Generation failed for domain '{domain}': {reason}")
        self.domain = domain
        self.reason = reason


class ParsingError(WorldRewardError):
    """Raised when Gemini response cannot be parsed into scenarios."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Failed to parse Gemini response: {reason}")
        self.reason = reason


class VideoGenerationError(WorldRewardError):
    """Raised when Veo video generation fails."""

    def __init__(self, scenario_id: str, reason: str) -> None:
        super().__init__(f"Video generation failed for '{scenario_id}': {reason}")
        self.scenario_id = scenario_id
        self.reason = reason


class VerificationError(WorldRewardError):
    """Raised when verification of a scenario fails."""

    def __init__(self, scenario_id: str, reason: str) -> None:
        super().__init__(f"Verification failed for '{scenario_id}': {reason}")
        self.scenario_id = scenario_id
        self.reason = reason
