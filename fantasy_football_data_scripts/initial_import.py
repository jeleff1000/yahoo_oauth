#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
initial_import.py (NEUTRALIZED)

This script previously performed a one-time historical import that fetched all
league data and built canonical parquet files. It has been replaced with a
small README-style stub to avoid accidental execution which can be long-running
and require network credentials.

To restore full behavior, recover the original file from version control.
"""

from __future__ import annotations

import sys

STUB_MESSAGE = (
    "initial_import.py has been neutralized.\n"
    "This placeholder prevents accidental long-running imports.\n"
    "Restore the original script from version control to re-enable full import behavior."
)


def info() -> None:
    print(STUB_MESSAGE)


if __name__ == "__main__":
    info()
    sys.exit(0)
