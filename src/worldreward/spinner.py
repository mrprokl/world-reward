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
