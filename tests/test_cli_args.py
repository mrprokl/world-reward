from __future__ import annotations

from worldreward.cli import parse_args


def test_parse_args_generate() -> None:
    args = parse_args(["generate", "--domain", "public_safety", "--count", "3"])
    assert args.command == "generate"
    assert args.domain == "public_safety"
    assert args.count == 3
    assert args.model is None


def test_parse_args_verify_with_optional_videos_dir() -> None:
    args = parse_args(
        ["verify", "--dataset", "output/datasets/demo.csv", "--videos-dir", "output/videos/demo"]
    )
    assert args.command == "verify"
    assert args.dataset.endswith("demo.csv")
    assert args.videos_dir.endswith("demo")
