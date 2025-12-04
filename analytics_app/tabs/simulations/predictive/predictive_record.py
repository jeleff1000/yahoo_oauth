#!/usr/bin/env python3
"""
predictive_record.py - Predicted final records visualization
"""
import re
from typing import Optional
import pandas as pd
import streamlit as st
from .table_styles import render_modern_table
from ..shared.simulation_styles import render_section_header


@st.fragment
def _render_expected_record(base_df: pd.DataFrame, year: int, week: int):
    """Render predicted records table."""
    week_slice = base_df[(base_df["year"] == year) & (base_df["week"] == week)]

    if week_slice.empty:
        st.info("No data available for selected year/week.")
        return

    # Find all xN_win columns
    all_cols = week_slice.columns
    win_meta = []
    pattern = re.compile(r"x(\d+)_win")

    for c in all_cols:
        m = pattern.fullmatch(c)
        if m:
            win_meta.append((int(m.group(1)), c))

    if not win_meta:
        st.info("No predictive win columns found in data.")
        return

    season_len = max(k for k, _ in win_meta)
    win_meta.sort(key=lambda t: t[0], reverse=True)
    ordered_win_cols = [c for _, c in win_meta]

    # Build dataframe
    needed = ["manager"] + [c for c in ordered_win_cols if c in week_slice.columns]
    df = (
        week_slice[needed]
        .drop_duplicates(subset=["manager"])
        .set_index("manager")
        .sort_index()
    )

    # Rename columns to W-L format (reversed so best records come first)
    rename_map = {c: f"{k}-{season_len - k}" for k, c in win_meta}
    df = df.rename(columns=rename_map)
    df = df[list(rename_map.values())]

    # Sort by total probability in best records (descending)
    # Sum probabilities for top half of records
    best_records = [
        col for col in df.columns if int(col.split("-")[0]) > season_len // 2
    ]
    if best_records:
        df["_sort_prob"] = df[best_records].sum(axis=1)
        df = df.sort_values("_sort_prob", ascending=False).drop(columns=["_sort_prob"])

    render_section_header("Predicted Final Records", "")
    st.caption(f"Season: {season_len} games | Probability of each win-loss record")

    # Identify numeric columns for gradient (all columns)
    numeric_cols = list(df.columns)

    # Create column rename map - just the record
    column_names = {}

    # Create format specs
    format_specs = {c: "{:.1f}" for c in numeric_cols}

    render_modern_table(
        df,
        title="",
        color_columns=numeric_cols,
        reverse_columns=[],
        format_specs=format_specs,
        column_names=column_names,
    )


@st.fragment
def display_predicted_record(
    matchup_data_df: pd.DataFrame,
    year: Optional[int] = None,
    week: Optional[int] = None,
):
    """Main entry point for predicted records.

    Args:
        matchup_data_df: Matchup data
        year: Selected year (from unified header)
        week: Selected week (from unified header)
    """
    if matchup_data_df is None or matchup_data_df.empty:
        st.info("No data available")
        return

    # Filter to regular season only
    base_df = matchup_data_df[
        (matchup_data_df["is_playoffs"] == 0) & (matchup_data_df["is_consolation"] == 0)
    ].copy()

    if base_df.empty:
        st.info("No regular season data available")
        return

    # Type conversion
    base_df["year"] = base_df["year"].astype(int)
    base_df["week"] = base_df["week"].astype(int)

    # Use provided year/week or default to latest
    if year is None:
        year = int(base_df["year"].max())
    if week is None:
        week = int(base_df[base_df["year"] == year]["week"].max())

    # Centered constrained-width layout
    st.markdown(
        '<div class="sim-centered-content">',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        _render_expected_record(base_df, year, week)

    st.markdown("</div>", unsafe_allow_html=True)
