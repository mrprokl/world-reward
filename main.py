#!/usr/bin/env python3
"""World Reward — Physics-verifiable evaluation pipeline for video/3D world models.

Supports two modes:
  - Direct CLI:  python main.py generate --domain X --count N
  - Interactive:  python main.py   (launches REPL)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Suppress noisy Python 3.9 deprecation warnings from third-party libs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

from dotenv import load_dotenv

from worldreward.cli import (
    parse_args,
    run_generate,
    run_list_domains,
    run_verify,
    run_videos,
)

PROJECT_ROOT = Path(__file__).parent


def main() -> None:
    """Entry point — dispatches to CLI or interactive REPL."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT.parent / ".env")

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
            print("Usage: python main.py {generate|videos|verify|list-domains} --help")
            sys.exit(1)
    else:
        try:
            from worldreward.repl import run_repl  # noqa: lazy import — prompt_toolkit optional
        except ImportError:
            print("Error: prompt_toolkit is required for interactive mode.")
            print("Install it with:  pip install prompt_toolkit")
            print("Or activate the venv:  source .venv/bin/activate")
            sys.exit(1)
        run_repl()


if __name__ == "__main__":
    main()
