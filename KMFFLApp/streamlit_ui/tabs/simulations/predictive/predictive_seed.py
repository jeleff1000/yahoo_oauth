#!/usr/bin/env python3
"""
predictive_seed.py - Predicted playoff seeding visualization
"""
import re
import pandas as pd
import streamlit as st
from .table_styles import render_modern_table


def _select_week_for_seed(base_df: pd.DataFrame):
    """Week selection with auto-load option."""
    mode = st.radio(
        "Selection Mode",
        ["Today's Date", "Specific Week"],
        horizontal=True,
        key="pred_mode_seed",
        index=0
    )

    if mode == "Today's Date":
        year = int(base_df['year'].max())
        week = int(base_df[base_df['year'] == year]['week'].max())
        st.caption(f"📅 Auto-selected Year {year}, Week {week}")
        return year, week, True
    else:
        years = sorted(base_df['year'].astype(int).unique())

        col1, col2 = st.columns(2)

        with col1:
            year_choice = st.selectbox(
                "Year",
                ["Select Year"] + [str(y) for y in years],
                key="pred_year_seed"
            )

        if year_choice == "Select Year":
            return None, None, False

        year = int(year_choice)
        weeks = sorted(base_df[base_df['year'] == year]['week'].astype(int).unique())

        with col2:
            week_choice = st.selectbox(
                "Week",
                ["Select Week"] + [str(w) for w in weeks],
                key="pred_week_seed"
            )

        if week_choice == "Select Week":
            return None, None, False

        week = int(week_choice)
        return year, week, False


@st.fragment
def _render_expected_seed(base_df: pd.DataFrame, year: int, week: int):
    """Render predicted seeding table."""
    week_slice = base_df[(base_df['year'] == year) & (base_df['week'] == week)]

    if week_slice.empty:
        st.info("No data available for selected year/week.")
        return

    # Find all xN_seed columns
    all_cols = week_slice.columns
    seed_meta = []
    pattern = re.compile(r"x(\d+)_seed")

    for c in all_cols:
        m = pattern.fullmatch(c)
        if m:
            seed_meta.append((int(m.group(1)), c))

    if not seed_meta:
        st.info("No predictive seed columns found in data.")
        return

    seed_meta.sort(key=lambda t: t[0])
    ordered_seed_cols = [c for _, c in seed_meta]

    # Build dataframe
    needed = ['manager'] + [c for c in ordered_seed_cols if c in week_slice.columns]
    df = (
        week_slice[needed]
        .drop_duplicates(subset=['manager'])
        .set_index('manager')
        .sort_index()
    )

    # Rename columns - just the seed number
    rename_map = {c: str(k) for k, c in seed_meta}
    df = df.rename(columns=rename_map)
    df = df[list(rename_map.values())]

    # Sort by probability of top seeds (best to worst)
    # Sum probabilities for seeds 1-3
    top_seeds = [col for col in df.columns if int(col) <= 3]
    if top_seeds:
        df['_sort_prob'] = df[top_seeds].sum(axis=1)
        df = df.sort_values('_sort_prob', ascending=False).drop(columns=['_sort_prob'])

    st.markdown("---")
    st.subheader("🎯 Predicted Playoff Seeding")
    st.caption("Showing probability of finishing in each playoff seed position")

    # Identify numeric columns for gradient (all seed columns)
    numeric_cols = list(df.columns)

    # Create column rename map - no changes
    column_names = {}

    # Create format specs
    format_specs = {c: "{:.1f}" for c in numeric_cols}

    render_modern_table(
        df,
        title="",
        color_columns=numeric_cols,
        reverse_columns=[],
        format_specs=format_specs,
        column_names=column_names
    )


@st.fragment
def display_predicted_seed(matchup_data_df: pd.DataFrame):
    """Main entry point for predicted seeding."""
    if matchup_data_df is None or matchup_data_df.empty:
        st.info("No data available")
        return

    # Filter to regular season only
    base_df = matchup_data_df[
        (matchup_data_df['is_playoffs'] == 0) &
        (matchup_data_df['is_consolation'] == 0)
        ].copy()

    if base_df.empty:
        st.info("No regular season data available")
        return

    # Type conversion
    base_df['year'] = base_df['year'].astype(int)
    base_df['week'] = base_df['week'].astype(int)

    year, week, auto_display = _select_week_for_seed(base_df)

    if year is None or week is None:
        return

    # Auto-display or show button
    if auto_display:
        _render_expected_seed(base_df, year, week)
    else:
        if st.button("Go", key="pred_seed_go"):
            _render_expected_seed(base_df, year, week)