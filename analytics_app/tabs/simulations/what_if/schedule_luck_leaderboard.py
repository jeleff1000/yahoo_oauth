#!/usr/bin/env python3
"""
schedule_luck_leaderboard.py - Quick Schedule Luck Summary

Simple glanceable dashboard showing luckiest/unluckiest managers.
For detailed analysis, use the Schedule Shuffles tab.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd


@st.fragment
def display_schedule_luck_leaderboard(matchup_data_df: pd.DataFrame = None):
    """Quick-glance schedule luck summary."""

    st.subheader("📊 Schedule Luck Summary")
    st.caption(
        "Quick snapshot of who's been luckiest/unluckiest. See Schedule Shuffles tab for details."
    )

    if matchup_data_df is None or matchup_data_df.empty:
        st.info("No data available.")
        return

    # Filter to regular season with luck data
    df = matchup_data_df[
        (matchup_data_df["is_playoffs"] == 0)
        & (matchup_data_df["is_consolation"] == 0)
        & (matchup_data_df["wins_vs_shuffle_wins"].notna())
    ].copy()

    if df.empty:
        st.info("No schedule luck data available.")
        return

    # Type conversion
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype(int)
    df["wins_vs_shuffle_wins"] = pd.to_numeric(
        df["wins_vs_shuffle_wins"], errors="coerce"
    )

    # Get current season data
    current_year = df["year"].max()
    latest_week = df[df["year"] == current_year]["week"].max()
    season_data = df[(df["year"] == current_year) & (df["week"] == latest_week)]

    luck_df = (
        season_data.groupby("manager")
        .agg(
            {
                "wins_vs_shuffle_wins": "mean",
                "wins_to_date": "max",
                "shuffle_avg_wins": "mean",
            }
        )
        .reset_index()
    )
    luck_df.columns = ["Manager", "Luck", "Actual", "Expected"]
    luck_df = luck_df.sort_values("Luck", ascending=False).reset_index(drop=True)

    # Top 3 luckiest and unluckiest
    st.markdown(f"**{current_year} Season (Week {latest_week})**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🍀 Luckiest")
        for i, row in luck_df.head(3).iterrows():
            luck_val = row["Luck"]
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            st.markdown(f"{emoji} **{row['Manager']}** — {luck_val:+.1f} wins")

    with col2:
        st.markdown("##### 😢 Unluckiest")
        for i, row in luck_df.tail(3).iloc[::-1].iterrows():
            luck_val = row["Luck"]
            rank = len(luck_df) - list(luck_df.tail(3).iloc[::-1].index).index(i)
            emoji = (
                "😭"
                if rank == len(luck_df)
                else "😔" if rank == len(luck_df) - 1 else "😕"
            )
            st.markdown(f"{emoji} **{row['Manager']}** — {luck_val:+.1f} wins")

    # Simple explanation
    st.markdown("---")
    st.caption(
        "**Luck** = Actual Wins − Expected Wins from 100K simulations. Positive = lucky schedule, Negative = tough schedule."
    )
