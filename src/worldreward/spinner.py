"""Animated terminal spinner for long-running operations."""

from __future__ import annotations

import sys
import threading
import time

_DOTS = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_CYAN = "\033[1;36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


class Spinner:
    """Context-manager spinner that animates while a block executes.

    Usage::

        with Spinner("Calling Gemini API"):
            result = client.generate(...)
    """

    def __init__(self, message: str = "Loading") -> None:
        self._message = message
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Spinner:
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _animate(self) -> None:
        idx = 0
        while self._running:
            # Three dots at different phases for a wave effect
            d1 = _DOTS[idx % len(_DOTS)]
            d2 = _DOTS[(idx + 3) % len(_DOTS)]
            d3 = _DOTS[(idx + 6) % len(_DOTS)]
            sys.stdout.write(
                f"\r  {_CYAN}{d1} {d2} {d3}{_RESET}  {_BOLD}{self._message}{_RESET}"
            )
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)


class ProgressBar:
    """Minimal terminal progress bar with elapsed + ETA."""

    def __init__(self, total: int, label: str = "Progress", width: int = 28) -> None:
        self._total = max(0, int(total))
        self._label = label
        self._width = max(10, int(width))
        self._start = time.monotonic()
        self._last_len = 0

    def update(self, current: int, label: str | None = None) -> None:
        """Render or re-render the progress bar in place."""
        if self._total <= 0:
            return

        current = max(0, min(int(current), self._total))
        label = label or self._label

        ratio = current / self._total if self._total else 1.0
        filled = int(self._width * ratio)
        bar = f"{'█' * filled}{'░' * (self._width - filled)}"

        elapsed = time.monotonic() - self._start
        if current > 0:
            eta = elapsed * (self._total - current) / current
            eta_str = f"ETA {eta:.1f}s"
        else:
            eta_str = "ETA --"

        line = (
            f"\r  {_CYAN}{bar}{_RESET}  {_BOLD}{label}{_RESET} "
            f"{current}/{self._total} ({ratio * 100:>3.0f}%) | "
            f"{elapsed:.1f}s | {eta_str}"
        )
        pad = " " * max(0, self._last_len - len(line))
        sys.stdout.write(line + pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def clear(self) -> None:
        """Clear the current progress bar line."""
        if self._last_len:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self._last_len = 0

    def finish(self) -> None:
        """Render completion and move to the next line."""
        if self._total <= 0:
            return
        self.update(self._total)
        sys.stdout.write("\n")
        sys.stdout.flush()
