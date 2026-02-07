"""Shared pipeline commands and CLI argument parsing for World Reward."""

from __future__ import annotations

import argparse
from pathlib import Path

from worldreward.config_loader import list_available_domains
from worldreward.gemini_client import GeminiClient
from worldreward.generator import ScenarioGenerator
from worldreward.scorer import print_score_report, write_results
from worldreward.verifier import Verifier
from worldreward.video_generator import VideoGenerator


# ─── Directory layout ────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/worldreward/ → project root
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATASETS_DIR = OUTPUT_DIR / "datasets"
VIDEOS_DIR = OUTPUT_DIR / "videos"
RESULTS_DIR = OUTPUT_DIR / "results"


# ─── Pipeline commands (used by both CLI and REPL) ───────────────────

def extract_run_id(dataset_path: Path) -> str:
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

    run_id = extract_run_id(dataset_path)
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

    run_id = extract_run_id(dataset_path)
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


# ─── Argparse (direct CLI mode) ─────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for direct CLI mode."""
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
