"""First-run setup and configuration helpers."""

from __future__ import annotations

from getpass import getpass
from typing import Callable

from google import genai

from worldreward.paths import (
    ensure_runtime_layout,
    get_config_search_dirs,
    get_output_dir,
    is_repo_checkout_mode,
    load_user_config,
    resolve_api_key_with_source,
    save_api_key,
)

ValidationFn = Callable[[str], tuple[bool, str | None]]
PrintFn = Callable[[str], None]
InputFn = Callable[[str], str]


def validate_api_key(api_key: str) -> tuple[bool, str | None]:
    """Validate API key by performing a lightweight authenticated request."""
    normalized_key = api_key.strip()
    if not normalized_key:
        return False, "API key cannot be empty."

    try:
        client = genai.Client(api_key=normalized_key)
        _ = next(iter(client.models.list(config={"page_size": 1})), None)
    except Exception as exc:  # pragma: no cover - network/auth behavior is integration-level
        return False, str(exc)

    return True, None


def run_setup_wizard(
    *,
    api_key: str | None = None,
    print_fn: PrintFn = print,
    secret_prompt_fn: InputFn = getpass,
    validator: ValidationFn = validate_api_key,
    max_attempts: int = 3,
) -> bool:
    """Run interactive first-time setup for API key + runtime layout."""
    layout = ensure_runtime_layout(copy_builtin_configs=True)

    print_fn("Welcome to World Reward setup.")
    print_fn(f"Runtime home: {layout.app_dir}")

    if layout.copied_builtin_configs:
        print_fn(f"Copied {len(layout.copied_builtin_configs)} builtin domain config(s).")

    provided_key = api_key.strip() if api_key else None
    candidate_key = ""

    for attempt in range(1, max_attempts + 1):
        if provided_key is not None and attempt == 1:
            candidate_key = provided_key
        else:
            try:
                candidate_key = secret_prompt_fn("Gemini API key: ").strip()
            except (EOFError, KeyboardInterrupt):
                print_fn("Setup cancelled.")
                return False

        if not candidate_key:
            print_fn("Empty key. Please try again.")
            continue

        ok, error = validator(candidate_key)
        if ok:
            config_path = save_api_key(candidate_key)
            print_fn(f"API key saved to: {config_path}")
            print_fn("Setup complete. You can now run: worldreward")
            return True

        print_fn(f"API key validation failed: {error or 'Unknown error'}")
        if provided_key is not None:
            return False

    print_fn("Setup aborted: unable to validate API key.")
    return False


def configure_api_key_interactive(
    *,
    print_fn: PrintFn = print,
    secret_prompt_fn: InputFn = getpass,
    validator: ValidationFn = validate_api_key,
) -> bool:
    """Prompt, validate, and persist a Gemini API key."""
    try:
        api_key = secret_prompt_fn("Gemini API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        print_fn("API key update cancelled.")
        return False

    ok, error = validator(api_key)
    if not ok:
        print_fn(f"API key validation failed: {error or 'Unknown error'}")
        return False

    config_path = save_api_key(api_key)
    print_fn(f"API key saved to: {config_path}")
    return True


def render_config_summary(show_api_key: bool = False) -> str:
    """Render a human-readable configuration summary."""
    ensure_runtime_layout(copy_builtin_configs=False)
    api_key, api_key_source = resolve_api_key_with_source()
    config = load_user_config()

    mode = "repository-checkout" if is_repo_checkout_mode() else "user-local"
    key_display = _mask_secret(api_key) if api_key and not show_api_key else (api_key or "(missing)")
    defaults = config.get("defaults")

    lines = [
        "World Reward configuration",
        f"- Mode: {mode}",
        f"- API key source: {api_key_source}",
        f"- API key: {key_display}",
        f"- Output dir: {get_output_dir()}",
        "- Config search dirs:",
    ]
    lines.extend([f"  - {path}" for path in get_config_search_dirs()])

    if isinstance(defaults, dict):
        model = defaults.get("model")
        if isinstance(model, str) and model.strip():
            lines.append(f"- Default model: {model}")

    return "\n".join(lines)


def _mask_secret(value: str) -> str:
    """Mask secret values for safe terminal display."""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
