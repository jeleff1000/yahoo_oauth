#!/usr/bin/env python3
from __future__ import annotations

try:
    # Import the new UI module written to streamlit_ui.py
    from streamlit_ui import main as app_main
except Exception as e:
    import sys
    print("Failed to import streamlit_ui:", e, file=sys.stderr)
    raise

if __name__ == "__main__":
    app_main()
