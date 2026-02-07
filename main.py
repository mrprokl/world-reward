#!/usr/bin/env python3
"""World Reward CLI — Physics-verifiable evaluation pipeline for video/3D world models.

Supports two modes:
  - Direct CLI:  python main.py generate --domain X --count N
  - Interactive:  python main.py   (launches REPL)
"""

from __future__ import annotations

import argparse
import shlex
import sys
import warnings
from pathlib import Path

# Suppress noisy Python 3.9 deprecation warnings from third-party libs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory

from worldreward.config_loader import list_available_domains
from worldreward.gemini_client import GeminiClient
from worldreward.generator import ScenarioGenerator
from worldreward.scorer import print_score_report, write_results
from worldreward.verifier import Verifier
from worldreward.video_generator import VideoGenerator

PROJECT_ROOT = Path(__file__).parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATASETS_DIR = OUTPUT_DIR / "datasets"
VIDEOS_DIR = OUTPUT_DIR / "videos"
RESULTS_DIR = OUTPUT_DIR / "results"


# ─── Shared logic (used by both CLI and REPL) ────────────────────────

def _extract_run_id(dataset_path: Path) -> str:
    """Extract run ID from dataset filename (e.g. 'autonomous_driving_20260207_131855')."""
    return dataset_path.stem


def run_generate(domain: str, count: int = 10, model: str | None = None) -> None:
    """Generate a physics scenario dataset."""
    config_path = CONFIGS_DIR / f"{domain}.yaml"
    if not config_path.exists():
        available = list_available_domains(CONFIGS_DIR)
        print(f"Error: Domain '{domain}' not found. Available: {', '.join(available)}")
        return

    print("=" * 60)
    print("  World Reward — Dataset Generator")
    print("=" * 60)

    client = GeminiClient(model=model)
    generator = ScenarioGenerator(client)
    output_path = generator.generate(
        config_path=config_path,
        count=count,
        output_dir=DATASETS_DIR,
    )

    print("=" * 60)
    print(f"  Done! Dataset: {output_path}")
    print("=" * 60)


def run_videos(dataset: str) -> None:
    """Generate videos from a dataset using Veo 3.1."""
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return

    run_id = _extract_run_id(dataset_path)
    videos_out = VIDEOS_DIR / run_id

    print("=" * 60)
    print("  World Reward — Video Generator (Veo 3.1)")
    print("=" * 60)

    gen = VideoGenerator()
    gen.generate_from_dataset(dataset_path, videos_out)

    print("=" * 60)
    print(f"  Done! Videos saved to: {videos_out}")
    print("=" * 60)


def run_verify(dataset: str, videos_dir: str | None = None) -> None:
    """Verify generated videos against physics ground truth."""
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return

    run_id = _extract_run_id(dataset_path)
    vdir = Path(videos_dir) if videos_dir else VIDEOS_DIR / run_id

    if not vdir.exists():
        print(f"Error: Videos directory not found: {vdir}")
        return

    print("=" * 60)
    print("  World Reward — Physics Verifier (Gemini 3 Pro)")
    print("=" * 60)

    verifier = Verifier()
    results = verifier.verify_dataset(dataset_path, vdir)

    if results:
        results_path = RESULTS_DIR / f"results_{run_id}.csv"
        write_results(results, results_path)
        print(f"\n💾 Results saved to: {results_path}")

    print_score_report(results)


def run_list_domains() -> None:
    """List available domain configurations."""
    domains = list_available_domains(CONFIGS_DIR)
    if not domains:
        print("No domain configs found in configs/")
        return
    print("Available domains:")
    for domain in domains:
        print(f"  - {domain}")


# ─── CLI mode (argparse) ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="World Reward — Physics-verifiable evaluation pipeline for world models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    available = list_available_domains(CONFIGS_DIR)
    domains_help = f"Available: {', '.join(available)}" if available else "No configs found"

    gen_parser = subparsers.add_parser("generate", help="Generate a physics scenario dataset using Gemini")
    gen_parser.add_argument("--domain", type=str, required=True, help=f"Domain config. {domains_help}")
    gen_parser.add_argument("--count", type=int, default=10, help="Number of scenarios (default: 10)")
    gen_parser.add_argument("--model", type=str, default=None, help="Gemini model override")

    vid_parser = subparsers.add_parser("videos", help="Generate videos from dataset using Veo 3.1")
    vid_parser.add_argument("--dataset", type=str, required=True, help="Path to scenario CSV dataset")

    ver_parser = subparsers.add_parser("verify", help="Verify videos against physics ground truth")
    ver_parser.add_argument("--dataset", type=str, required=True, help="Path to scenario CSV dataset")
    ver_parser.add_argument("--videos-dir", type=str, default=None, help="Directory with video files")

    subparsers.add_parser("list-domains", help="List available domain configurations")

    return parser.parse_args()


# ─── Interactive REPL mode ───────────────────────────────────────────

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
║   Unit tests for reality — verifiable physics rewards        ║
║   for 3D World Models.                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝\033[0m

  Type \033[1m/help\033[0m to see available commands, \033[1mCtrl+C\033[0m to exit.
"""

HELP_TEXT = """
\033[1mAvailable commands:\033[0m

  \033[1;32m/generate\033[0m --domain <name> [--count N] [--model <model>]
      Generate a physics scenario dataset using Gemini 3 Pro.
      Scenarios are saved to output/datasets/.
      Example: /generate --domain autonomous_driving --count 5

  \033[1;32m/videos\033[0m --dataset <path>
      Render videos from a dataset using Veo 3.1 (parallel).
      Videos are saved to output/videos/<run_id>/.
      Example: /videos --dataset output/datasets/autonomous_driving_20260207_131855.csv

  \033[1;32m/verify\033[0m --dataset <path> [--videos-dir <dir>]
      Verify generated videos against physics ground truth.
      Uses Gemini 3 Pro as VLM judge. Outputs ternary rewards (+1/0/-1).
      Example: /verify --dataset output/datasets/autonomous_driving_20260207_131855.csv

  \033[1;32m/domains\033[0m
      List available domain configurations.

  \033[1;32m/help\033[0m
      Show this help message.

  \033[1;32m/quit\033[0m or \033[1mCtrl+C\033[0m
      Exit World Reward.

\033[1mPipeline:\033[0m  /generate → /videos → /verify
"""


def _parse_repl_generate(tokens: list[str]) -> None:
    """Parse and run /generate command."""
    parser = argparse.ArgumentParser(prog="/generate", add_help=False)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--model", type=str, default=None)
    try:
        args = parser.parse_args(tokens)
        run_generate(args.domain, args.count, args.model)
    except SystemExit:
        print("Usage: /generate --domain <name> [--count N] [--model <model>]")


def _parse_repl_videos(tokens: list[str]) -> None:
    """Parse and run /videos command."""
    parser = argparse.ArgumentParser(prog="/videos", add_help=False)
    parser.add_argument("--dataset", type=str, required=True)
    try:
        args = parser.parse_args(tokens)
        run_videos(args.dataset)
    except SystemExit:
        print("Usage: /videos --dataset <path>")


def _parse_repl_verify(tokens: list[str]) -> None:
    """Parse and run /verify command."""
    parser = argparse.ArgumentParser(prog="/verify", add_help=False)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--videos-dir", type=str, default=None)
    try:
        args = parser.parse_args(tokens)
        run_verify(args.dataset, args.videos_dir)
    except SystemExit:
        print("Usage: /verify --dataset <path> [--videos-dir <dir>]")


REPL_COMMANDS = {
    "/generate": _parse_repl_generate,
    "/videos": _parse_repl_videos,
    "/verify": _parse_repl_verify,
    "/domains": lambda _: run_list_domains(),
    "/help": lambda _: print(HELP_TEXT),
}


def run_repl() -> None:
    """Launch the interactive REPL."""
    print(BANNER)

    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
    )

    while True:
        try:
            user_input = session.prompt(
                HTML("<b><ansigreen>worldreward</ansigreen></b><b> ❯ </b>"),
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
        if handler:
            try:
                handler(cmd_args)
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")


# ─── Entry point ─────────────────────────────────────────────────────

def main() -> None:
    """Entry point for World Reward CLI."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT.parent / ".env")

    # If subcommand provided → direct CLI mode, otherwise → REPL
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
        run_repl()


if __name__ == "__main__":
    main()
