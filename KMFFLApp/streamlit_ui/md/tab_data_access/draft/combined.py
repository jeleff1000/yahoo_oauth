#!/usr/bin/env python3
"""
Combined data loader for Draft tab.

Main entry point for loading all data needed by the draft tab.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from .draft_data import load_draft_data


@st.cache_data(show_spinner=True, ttl=600)
def load_optimized_draft_data() -> Dict[str, Any]:
    """
    Load all data for draft tab in one optimized call.

    Returns:
        Dict with "Draft History" key containing DataFrame, or "error" key on failure
    """
    try:
        # Load draft data with column selection
        draft_data = load_draft_data()

        # Check for errors
        if "error" in draft_data:
            return {"error": draft_data["error"]}

        # Return the data
        return draft_data

    except Exception as e:
        st.error(f"Failed to load draft tab data: {e}")
        return {"error": str(e)}
