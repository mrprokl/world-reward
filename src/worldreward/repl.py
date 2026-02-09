"""Interactive REPL with step-by-step wizards for World Reward."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory

from worldreward.cli import (
    CONFIG_SEARCH_DIRS,
    DATASETS_DIR,
    VIDEOS_DIR,
    run_generate,
    run_list_domains,
    run_verify,
    run_videos,
)
from worldreward.config_loader import list_available_domains
from worldreward.paths import resolve_api_key, save_api_key

ReplHandler = Callable[[list[str], PromptSession], None]


# ─── Banner & help ───────────────────────────────────────────────────

BANNER = """
\033[1;36m╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗                 ║
║   ██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗                ║
║   ██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║                ║
║   ██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║                ║
║   ╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝                ║
║    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝                ║
║                                                              ║
║   ██████╗ ███████╗██╗    ██╗ █████╗ ██████╗ ██████╗          ║
║   ██╔══██╗██╔════╝██║    ██║██╔══██╗██╔══██╗██╔══██╗         ║
║   ██████╔╝█████╗  ██║ █╗ ██║███████║██████╔╝██║  ██║         ║
║   ██╔══██╗██╔══╝  ██║███╗██║██╔══██║██╔══██╗██║  ██║         ║
║   ██║  ██║███████╗╚███╔███╔╝██║  ██║██║  ██║██████╔╝         ║
║   ╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝         ║
║                                                              ║
║   Experimentation towards scalable evaluation                ║
║   environments for 3D World Models.                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝\033[0m

  Type \033[1m/help\033[0m to see available commands, \033[1mCtrl+C\033[0m to exit.
"""

HELP_TEXT = """
\033[1mAvailable commands:\033[0m

  \033[1;32m/generate\033[0m
      Interactive wizard to generate a physics scenario dataset.
      Guides you step-by-step: domain → count → model.

  \033[1;32m/videos\033[0m
      Interactive wizard to render videos from a dataset.
      Lists available datasets, you pick by number.

  \033[1;32m/verify\033[0m
      Interactive wizard to verify videos against physics ground truth.
      Lists datasets with generated videos, you pick by number.

  \033[1;32m/domains\033[0m
      List available domain configurations.

  \033[1;32m/help\033[0m
      Show this help message.

  \033[1;32m/quit\033[0m or \033[1mCtrl+C\033[0m
      Exit World Reward.

\033[1mPipeline:\033[0m  /generate → /videos → /verify
"""


# ─── Prompt helpers ──────────────────────────────────────────────────

def _select_from_list(title: str, items: list[str], session: PromptSession) -> str | None:
    """Present a numbered list and let user pick by number.

    Returns:
        Selected item, or None if cancelled.
    """
    print(f"\n\033[1m{title}\033[0m\n")
    for idx, item in enumerate(items, start=1):
        print(f"  \033[1;36m[{idx}]\033[0m {item}")
    print()

    try:
        choice = session.prompt(
            HTML("<b><ansigreen>  ❯ </ansigreen></b>Select (number): "),
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print("  Cancelled.")
        return None

    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(items):
        print(f"  Invalid choice: {choice}")
        return None

    return items[int(choice) - 1]


def _prompt_input(label: str, default: str, session: PromptSession) -> str:
    """Prompt for a single value with a default shown in brackets.

    Returns:
        User input or default if empty.
    """
    try:
        value = session.prompt(
            HTML(f"<b><ansigreen>  ❯ </ansigreen></b>{label} <ansigray>[{default}]</ansigray>: "),
        ).strip()
    except (KeyboardInterrupt, EOFError):
        return default

    return value if value else default


def _list_files(directory: Path, pattern: str = "*.csv") -> list[Path]:
    """List files in a directory matching a glob pattern, sorted."""
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def _maybe_setup_api_key(session: PromptSession) -> None:
    """Prompt for API key on REPL startup when no key is configured."""
    if resolve_api_key():
        return

    print("\n\033[1;33mNo Gemini API key detected.\033[0m")
    print("Set GEMINI_API_KEY in your environment or configure one now.")
    print("A configured key is stored at ~/.worldreward/config.toml.")

    try:
        should_configure = session.prompt(
            HTML("<b><ansigreen>  ❯ </ansigreen></b>Configure API key now? <ansigray>[Y/n]</ansigray>: "),
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("  Skipping setup.")
        return

    if should_configure in {"n", "no"}:
        print("  Setup skipped. Commands requiring API access will fail until a key is configured.")
        return

    try:
        api_key = session.prompt(
            HTML("<b><ansigreen>  ❯ </ansigreen></b>Gemini API key: "),
            is_password=True,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print("  Setup cancelled.")
        return

    if not api_key:
        print("  Empty key ignored.")
        return

    try:
        config_path = save_api_key(api_key)
    except Exception as e:
        print(f"  Failed to save API key: {e}")
        return

    print(f"  ✅ API key saved to: {config_path}")


# ─── Wizards ─────────────────────────────────────────────────────────

def _wizard_generate(session: PromptSession) -> None:
    """Step-by-step wizard for /generate."""
    domains = list_available_domains(CONFIG_SEARCH_DIRS)
    if not domains:
        print("  No domain configs found.")
        return

    domain = _select_from_list("Select a domain:", domains, session)
    if not domain:
        return

    count_str = _prompt_input("Number of scenarios", "5", session)
    try:
        count = int(count_str)
    except ValueError:
        print(f"  Invalid number: {count_str}")
        return

    model_str = _prompt_input("Gemini model", "gemini-3-pro-preview", session)
    model = model_str if model_str != "gemini-3-pro-preview" else None

    print()
    run_generate(domain, count, model)


def _wizard_videos(session: PromptSession) -> None:
    """Step-by-step wizard for /videos."""
    datasets = _list_files(DATASETS_DIR)
    if not datasets:
        print("  No datasets found in output/datasets/. Run /generate first.")
        return

    display_names = [f.name for f in datasets]
    choice = _select_from_list("Select a dataset:", display_names, session)
    if not choice:
        return

    dataset_path = DATASETS_DIR / choice
    print()
    run_videos(str(dataset_path))


def _wizard_verify(session: PromptSession) -> None:
    """Step-by-step wizard for /verify."""
    datasets = _list_files(DATASETS_DIR)
    if not datasets:
        print("  No datasets found in output/datasets/. Run /generate first.")
        return

    datasets_with_videos = [
        d for d in datasets
        if (VIDEOS_DIR / d.stem).exists()
        and any((VIDEOS_DIR / d.stem).glob("*.mp4"))
    ]

    if not datasets_with_videos:
        print("  No datasets with generated videos found. Run /videos first.")
        return

    display_names = []
    for d in datasets_with_videos:
        video_count = len(list((VIDEOS_DIR / d.stem).glob("*.mp4")))
        display_names.append(f"{d.name}  ({video_count} videos)")

    choice = _select_from_list("Select a dataset to verify:", display_names, session)
    if not choice:
        return

    filename = choice.split("  (")[0]
    dataset_path = DATASETS_DIR / filename
    print()
    run_verify(str(dataset_path))


# ─── Command dispatch ────────────────────────────────────────────────

def _make_handler(wizard_fn: Callable[[PromptSession], None]) -> ReplHandler:
    """Wrap a wizard function into a REPL command handler."""
    def handler(_tokens: list[str], session: PromptSession) -> None:
        wizard_fn(session)
    return handler


REPL_COMMANDS: dict[str, ReplHandler] = {
    "/generate": _make_handler(_wizard_generate),
    "/videos": _make_handler(_wizard_videos),
    "/verify": _make_handler(_wizard_verify),
    "/domains": lambda _t, _s: run_list_domains(),
    "/help": lambda _t, _s: print(HELP_TEXT),
}


# ─── REPL loop ───────────────────────────────────────────────────────

def run_repl() -> None:
    """Launch the interactive REPL with step-by-step wizards."""
    print(BANNER)

    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        bottom_toolbar=HTML(
            "<ansigray>  /help · /generate → /videos → /verify · /quit</ansigray>"
        ),
    )
    _maybe_setup_api_key(session)

    while True:
        try:
            user_input = session.prompt(
                HTML("\n<b><ansigreen>worldreward</ansigreen></b><b> ❯ </b>"),
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit", "/q"):
            print("👋 Goodbye!")
            break

        try:
            tokens = shlex.split(user_input)
        except ValueError as e:
            print(f"Parse error: {e}")
            continue

        cmd = tokens[0]
        cmd_args = tokens[1:]

        handler = REPL_COMMANDS.get(cmd)
        if handler is not None:
            try:
                handler(cmd_args, session)
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")
