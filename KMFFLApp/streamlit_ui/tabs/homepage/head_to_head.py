#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure streamlit_ui directory is in path for imports
_streamlit_ui_dir = Path(__file__).parent.parent.parent.resolve()
if str(_streamlit_ui_dir) not in sys.path:
    sys.path.insert(0, str(_streamlit_ui_dir))

# Reuse the proven viewer from player_stats
from ..player_stats.weekly_player_subprocesses.head_to_head import H2HViewer, filter_h2h_data
from md.data_access import (
    list_player_seasons,
    list_player_weeks,
    list_optimal_seasons,
    list_optimal_weeks,
    load_player_week,
    load_optimal_week,
)
from shared.dataframe_utils import as_dataframe, get_matchup_df

@st.fragment
def display_head_to_head(df_dict: Dict[str, Any]):
    """
    Home → Head-to-Head tab:
      • Let user pick ANY year/week (back to earliest available, e.g. 1999)
      • Offer 'Optimal' plus actual matchups for that week
      • Render using the same H2H viewer you use in player_stats
    """
    # We don't actually need the matchup dataframe for this viewer,
    # but we’ll use it to help infer available seasons/weeks if needed.
    mdf = get_matchup_df(df_dict)

    st.markdown("### Head-to-Head")

    # ----- Build full season list (include earliest like 1999) -----
    seasons_opt = list_optimal_seasons() or []
    seasons_player = list_player_seasons() or []
    seasons_from_mdf = []
    if mdf is not None and not mdf.empty and "year" in mdf.columns:
        try:
            seasons_from_mdf = pd.to_numeric(mdf["year"], errors="coerce").dropna().astype(int).unique().tolist()
        except Exception:
            seasons_from_mdf = []
    all_seasons = sorted(set(map(int, seasons_opt + seasons_player + seasons_from_mdf)))
    if not all_seasons:
        st.error("No seasons available.")
        return

    default_year = max(all_seasons)

    # ----- All weeks for default year first (to populate week dropdown) -----
    weeks_opt = list_optimal_weeks(int(default_year)) or []
    weeks_player = list_player_weeks(int(default_year)) or []
    weeks_from_mdf = []
    if mdf is not None and not mdf.empty and {"year", "week"}.issubset(mdf.columns):
        try:
            weeks_from_mdf = (
                mdf.loc[pd.to_numeric(mdf["year"], errors="coerce") == int(default_year), "week"]
                .pipe(pd.to_numeric, errors="coerce")
                .dropna().astype(int).unique().tolist()
            )
        except Exception:
            weeks_from_mdf = []
    all_weeks = sorted(set(map(int, weeks_opt + weeks_player + weeks_from_mdf)))

    # Year and Week on one row
    col1, col2 = st.columns(2)
    with col1:
        sel_year = st.selectbox(
            "Year",
            all_seasons,
            index=all_seasons.index(default_year),
            key="home_h2h_year",
        )

    # Refresh weeks for selected year
    if sel_year != default_year:
        weeks_opt = list_optimal_weeks(int(sel_year)) or []
        weeks_player = list_player_weeks(int(sel_year)) or []
        weeks_from_mdf = []
        if mdf is not None and not mdf.empty and {"year", "week"}.issubset(mdf.columns):
            try:
                weeks_from_mdf = (
                    mdf.loc[pd.to_numeric(mdf["year"], errors="coerce") == int(sel_year), "week"]
                    .pipe(pd.to_numeric, errors="coerce")
                    .dropna().astype(int).unique().tolist()
                )
            except Exception:
                weeks_from_mdf = []
        all_weeks = sorted(set(map(int, weeks_opt + weeks_player + weeks_from_mdf)))

    if not all_weeks:
        st.error("No weeks available for the selected year.")
        return

    default_week = max(all_weeks)
    with col2:
        sel_week = st.selectbox(
            "Week",
            all_weeks,
            index=all_weeks.index(default_week),
            key="home_h2h_week",
        )

    # ----- Load both data views for the chosen Year/Week -----
    optimal_df = load_optimal_week(int(sel_year), int(sel_week))
    player_week_df = load_player_week(int(sel_year), int(sel_week))

    if (optimal_df is None or optimal_df.empty) and (player_week_df is None or player_week_df.empty):
        st.warning("No player rows for that Year/Week.")
        return

    # ----- Build Matchup dropdown: "Optimal" first, then actual matchups (if any) -----
    matchup_options = ["Optimal"]
    if player_week_df is not None and not player_week_df.empty:
        if "matchup_name" not in player_week_df.columns:
            st.error("Column 'matchup_name' is missing in player data.")
            return
        raw_matchup_names = (
            player_week_df["matchup_name"]
            .dropna().astype(str).unique().tolist()
        )
        # Format matchup names: replace "__vs__" with " vs "
        formatted_matchups = [m.replace("__vs__", " vs ") for m in raw_matchup_names if m.lower() not in ['none vs none', 'nan vs nan', 'none', 'nan', '']]
        matchup_options.extend(sorted(formatted_matchups))

    # ✅ Default selection index = 0 ("Optimal")
    sel_matchup = st.selectbox(
        "Matchup",
        matchup_options,
        index=0,  # <— ensures "Optimal" is open by default
        key="home_h2h_matchup",
        help="Choose 'Optimal' to see the league-wide optimal lineup for this week, or pick a specific matchup.",
    )

    # ----- Render -----
    if sel_matchup == "Optimal":
        if optimal_df is None or optimal_df.empty:
            st.warning("No Optimal lineup available for this Week/Year.")
        else:
            H2HViewer(optimal_df).display_league_optimal(prefix="home_h2h")
    else:
        if player_week_df is None or player_week_df.empty:
            st.warning("No Head-to-Head data available for this Week/Year.")
        else:
            # Calculate and show score summary
            raw_matchup = sel_matchup.replace(" vs ", "__vs__")
            matchup_data = player_week_df[player_week_df["matchup_name"] == raw_matchup]
            if not matchup_data.empty and "team_1" in matchup_data.columns and "team_2" in matchup_data.columns:
                team_1 = matchup_data["team_1"].dropna().iloc[0] if matchup_data["team_1"].notna().any() else "Team 1"
                team_2 = matchup_data["team_2"].dropna().iloc[0] if matchup_data["team_2"].notna().any() else "Team 2"

                # Get starting lineup points only (exclude BN, IR)
                if "lineup_position" in matchup_data.columns:
                    starters = matchup_data[~matchup_data["lineup_position"].astype(str).str.startswith(("BN", "IR"))]
                else:
                    starters = matchup_data

                team_1_pts = starters[starters["manager"] == team_1]["points"].sum() if "points" in starters.columns else 0
                team_2_pts = starters[starters["manager"] == team_2]["points"].sum() if "points" in starters.columns else 0

                # Score summary
                col1, col2, col3 = st.columns([2, 1, 2])
                with col1:
                    st.markdown(f"<div style='text-align: center; font-size: 1.3rem;'><b>{team_1}</b><br><span style='font-size: 2rem; color: {'#4ade80' if team_1_pts > team_2_pts else '#f87171' if team_1_pts < team_2_pts else 'white'};'>{team_1_pts:.2f}</span></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div style='text-align: center; font-size: 1.5rem; padding-top: 1rem;'>vs</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div style='text-align: center; font-size: 1.3rem;'><b>{team_2}</b><br><span style='font-size: 2rem; color: {'#4ade80' if team_2_pts > team_1_pts else '#f87171' if team_2_pts < team_1_pts else 'white'};'>{team_2_pts:.2f}</span></div>", unsafe_allow_html=True)

            H2HViewer(player_week_df).display(prefix="home_h2h", matchup_name=sel_matchup)