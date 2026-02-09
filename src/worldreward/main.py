"""Entrypoint for CLI and interactive REPL modes."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv


def _load_env_files() -> None:
    """Load .env from cwd and repository-style locations when available."""
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parent.parent
    candidates = (
        Path.cwd() / ".env",
        repo_root / ".env",
        repo_root.parent / ".env",
    )
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def main() -> None:
    """Dispatch to direct CLI subcommands or interactive REPL."""
    # Suppress noisy warnings from third-party HTTP stack.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*urllib3.*")

    _load_env_files()

    from worldreward.cli import (  # Lazy import so env vars are loaded first.
        parse_args,
        run_generate,
        run_list_domains,
        run_verify,
        run_videos,
    )

    if len(sys.argv) > 1:
        args = parse_args()
        if args.command == "generate":
            run_generate(args.domain, args.count, args.model)
        elif args.command == "videos":
            run_videos(args.dataset)
        elif args.command == "verify":
            run_verify(args.dataset, args.videos_dir)
        elif args.command == "list-domains":
            run_list_domains()
        else:
            print("Usage: worldreward {generate|videos|verify|list-domains} --help")
            sys.exit(1)
        return

    try:
        from worldreward.repl import run_repl
    except ImportError:
        print("Error: prompt_toolkit is required for interactive mode.")
        print("Install it with: pip install prompt_toolkit")
        sys.exit(1)

    run_repl()
