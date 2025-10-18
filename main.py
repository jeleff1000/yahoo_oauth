"""
main.py (NEUTRALIZED)

This file was intentionally replaced with a small README-style stub to avoid confusion.
The original Streamlit application that provided the interactive OAuth + import UI
has been removed from this file to keep the repository tidy. If you need to
restore the original behavior, recover the file from version control or a
backup copy.

Why neutralize?
- This prevents accidental execution in environments that aren't configured for
  the app (missing secrets, Streamlit, network access, etc.).

Behavior of this stub:
- When executed, it prints a short informational message and exits with code 0.
"""

from __future__ import annotations

import sys

STUB_MESSAGE = (
    "main.py has been neutralized.\n"
    "This file was replaced with a stub to avoid accidental runs.\n"
    "Restore the original file from version control to re-enable the Streamlit app."
)


def info() -> None:
    """Print a brief informational message."""
    print(STUB_MESSAGE)


if __name__ == "__main__":
    info()
    sys.exit(0)

