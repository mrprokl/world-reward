#!/usr/bin/env python3
"""WorldBench CLI — Physics-verifiable evaluation pipeline for video/3D world models."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from worldbench.config_loader import list_available_domains
from worldbench.gemini_client import GeminiClient
from worldbench.generator import ScenarioGenerator
from worldbench.scorer import print_score_report, write_results
from worldbench.verifier import Verifier
from worldbench.video_generator import VideoGenerator

PROJECT_ROOT = Path(__file__).parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEOS_DIR = OUTPUT_DIR / "videos"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WorldBench — Physics-verifiable evaluation pipeline for world models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- generate ---
    available = list_available_domains(CONFIGS_DIR)
    domains_help = f"Available: {', '.join(available)}" if available else "No configs found"

    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a physics scenario dataset using Gemini",
        epilog=f"  {domains_help}",
    )
    gen_parser.add_argument("--domain", type=str, required=True, help=f"Domain config. {domains_help}")
    gen_parser.add_argument("--count", type=int, default=10, help="Number of scenarios (default: 10)")
    gen_parser.add_argument("--model", type=str, default=None, help="Gemini model override")

    # --- videos ---
    vid_parser = subparsers.add_parser("videos", help="Generate videos from dataset using Veo 3.1")
    vid_parser.add_argument("--dataset", type=str, required=True, help="Path to scenario CSV dataset")

    # --- verify ---
    ver_parser = subparsers.add_parser("verify", help="Verify videos against physics ground truth")
    ver_parser.add_argument("--dataset", type=str, required=True, help="Path to scenario CSV dataset")
    ver_parser.add_argument("--videos-dir", type=str, default=None, help="Directory with video files (default: output/videos/)")

    # --- list-domains ---
    subparsers.add_parser("list-domains", help="List available domain configurations")

    return parser.parse_args()


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a physics scenario dataset."""
    config_path = CONFIGS_DIR / f"{args.domain}.yaml"
    if not config_path.exists():
        available = list_available_domains(CONFIGS_DIR)
        print(f"Error: Domain '{args.domain}' not found. Available: {', '.join(available)}")
        sys.exit(1)

    print("=" * 60)
    print("  WorldBench — Dataset Generator")
    print("=" * 60)

    client = GeminiClient(model=args.model)
    generator = ScenarioGenerator(client)
    output_path = generator.generate(
        config_path=config_path,
        count=args.count,
        output_dir=OUTPUT_DIR,
    )

    print("=" * 60)
    print(f"  Done! Dataset: {output_path}")
    print("=" * 60)


def cmd_videos(args: argparse.Namespace) -> None:
    """Generate videos from a dataset using Veo 3.1."""
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    print("=" * 60)
    print("  WorldBench — Video Generator (Veo 3.1)")
    print("=" * 60)

    generator = VideoGenerator()
    generator.generate_from_dataset(dataset_path, VIDEOS_DIR)

    print("=" * 60)
    print(f"  Done! Videos saved to: {VIDEOS_DIR}")
    print("=" * 60)


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify generated videos against physics ground truth."""
    dataset_path = Path(args.dataset)
    videos_dir = Path(args.videos_dir) if args.videos_dir else VIDEOS_DIR

    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)
    if not videos_dir.exists():
        print(f"Error: Videos directory not found: {videos_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  WorldBench — Physics Verifier (Gemini 3 Pro)")
    print("=" * 60)

    verifier = Verifier()
    results = verifier.verify_dataset(dataset_path, videos_dir)

    if results:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_path = OUTPUT_DIR / f"results_{timestamp}.csv"
        write_results(results, results_path)
        print(f"\n💾 Results saved to: {results_path}")

    print_score_report(results)


def cmd_list_domains() -> None:
    """List available domain configurations."""
    domains = list_available_domains(CONFIGS_DIR)
    if not domains:
        print("No domain configs found in configs/")
        sys.exit(1)
    print("Available domains:")
    for domain in domains:
        print(f"  - {domain}")


def main() -> None:
    """Entry point for WorldBench CLI."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT.parent / ".env")

    args = parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "videos":
        cmd_videos(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "list-domains":
        cmd_list_domains()
    else:
        print("Usage: python main.py {generate|videos|verify|list-domains} --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
