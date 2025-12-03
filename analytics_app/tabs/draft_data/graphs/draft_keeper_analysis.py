#!/usr/bin/env python3
"""
draft_keeper_analysis.py - Keeper vs drafted player analysis
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.core import T, run_query


@st.fragment
def display_draft_keeper_analysis(prefix=""):
    """Analyze keeper vs drafted player performance"""
    st.subheader("⚡ Keeper vs Drafted Analysis")

    with st.spinner("Loading draft data..."):
        draft_data = run_query(
            f"""
            SELECT
                year, manager, player, yahoo_position,
                round, pick, cost,
                COALESCE(TRY_CAST(is_keeper_status AS INTEGER), 0) as is_keeper,
                COALESCE(total_fantasy_points, 0) as points,
                COALESCE(season_ppg, 0) as season_ppg,
                COALESCE(spar, 0) as spar
            FROM {T['draft']}
            WHERE cost > 0
            ORDER BY year DESC, pick
        """
        )

    if draft_data.empty:
        st.warning("No draft data available.")
        return

    # Calculate SPAR metrics uniformly for all players (keepers and drafted)
    # Keepers are just a subset of drafted players with is_keeper == 1
    draft_data["value"] = draft_data["spar"]

    # Use draft_roi from database if available, otherwise calculate with safeguards
    if "draft_roi" in draft_data.columns:
        draft_data["roi"] = draft_data["draft_roi"].fillna(0)
    else:
        # Safe division: replace 0 cost with small number to avoid divide-by-zero
        cost_safe = draft_data["cost"].replace(0, 0.1).fillna(0.1)
        draft_data["roi"] = (draft_data["spar"] / cost_safe).fillna(0)

    draft_data["points_per_dollar"] = draft_data["roi"]  # Same metric

    # Year selection
    years = sorted(draft_data["year"].unique(), reverse=True)
    selected_year = st.selectbox(
        "Select Year",
        options=["All Years"] + list(years),
        index=0,
        key=f"{prefix}_keeper_year",
    )

    # Filter data
    if selected_year != "All Years":
        data = draft_data[draft_data["year"] == selected_year].copy()
    else:
        data = draft_data.copy()

    keeper_data = data[data["is_keeper"] == 1]
    drafted_data = data[data["is_keeper"] == 0]

    if keeper_data.empty:
        st.info("No keeper data available for this selection.")
        return

    # Summary metrics (with safeguards for empty dataframes)
    col1, col2, col3, col4 = st.columns(4)

    # Calculate averages safely (handle empty dataframes)
    keeper_avg_roi = keeper_data["roi"].mean() if not keeper_data.empty else 0
    drafted_avg_roi = drafted_data["roi"].mean() if not drafted_data.empty else 0
    keeper_avg_ppd = (
        keeper_data["points_per_dollar"].mean() if not keeper_data.empty else 0
    )
    drafted_avg_ppd = (
        drafted_data["points_per_dollar"].mean() if not drafted_data.empty else 0
    )
    keeper_avg_value = keeper_data["value"].mean() if not keeper_data.empty else 0
    drafted_avg_value = drafted_data["value"].mean() if not drafted_data.empty else 0

    keeper_count = len(keeper_data)
    total_count = len(data)
    keeper_pct = (keeper_count / total_count * 100) if total_count > 0 else 0

    with col1:
        st.metric(
            "Keeper Avg SPAR/$",
            f"{keeper_avg_roi:.2f}",
            delta=f"{keeper_avg_roi - drafted_avg_roi:+.2f} vs drafted",
        )

    with col2:
        st.metric(
            "Drafted Avg SPAR/$",
            f"{drafted_avg_ppd:.2f}",
            delta=f"{keeper_avg_ppd - drafted_avg_ppd:+.2f} difference",
        )

    with col3:
        st.metric(
            "Keeper %", f"{keeper_pct:.1f}%", delta=f"{keeper_count} of {total_count}"
        )

    with col4:
        st.metric(
            "Keeper Avg SPAR",
            f"{keeper_avg_value:.1f}",
            delta=f"{keeper_avg_value - drafted_avg_value:+.1f} vs drafted",
        )

    # Comparison visualizations
    col1, col2 = st.columns(2)

    with col1:
        # SPAR/$ box plot comparison
        fig = go.Figure()

        fig.add_trace(
            go.Box(
                y=keeper_data["roi"],
                name="Keepers",
                marker_color="lightblue",
                boxmean="sd",
                hovertemplate="SPAR/$: %{y:.2f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Box(
                y=drafted_data["roi"],
                name="Drafted",
                marker_color="lightcoral",
                boxmean="sd",
                hovertemplate="SPAR/$: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"SPAR/$ Distribution: Keepers vs Drafted ({selected_year})",
            yaxis_title="SPAR per Dollar",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_keeper_roi")

    with col2:
        # SPAR value comparison
        fig = go.Figure()

        fig.add_trace(
            go.Box(
                y=keeper_data["value"],
                name="Keepers",
                marker_color="lightgreen",
                boxmean="sd",
                hovertemplate="SPAR: %{y:.1f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Box(
                y=drafted_data["value"],
                name="Drafted",
                marker_color="lightyellow",
                boxmean="sd",
                hovertemplate="SPAR: %{y:.1f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"SPAR Distribution: Keepers vs Drafted ({selected_year})",
            yaxis_title="SPAR",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_keeper_ppd")

    # Position breakdown
    st.markdown("### 📊 Keeper Performance by Position")

    pos_comparison = pd.DataFrame()
    for pos in keeper_data["yahoo_position"].unique():
        keeper_pos = keeper_data[keeper_data["yahoo_position"] == pos]
        drafted_pos = drafted_data[drafted_data["yahoo_position"] == pos]

        pos_comparison = pd.concat(
            [
                pos_comparison,
                pd.DataFrame(
                    {
                        "Position": [pos],
                        "Keeper Avg SPAR/$": [keeper_pos["roi"].mean()],
                        "Drafted Avg SPAR/$": [drafted_pos["roi"].mean()],
                        "Keeper Count": [len(keeper_pos)],
                        "Drafted Count": [len(drafted_pos)],
                        "Keeper Avg SPAR": [keeper_pos["value"].mean()],
                        "Drafted Avg SPAR": [drafted_pos["value"].mean()],
                    }
                ),
            ],
            ignore_index=True,
        )

    pos_comparison = pos_comparison.round(2)
    st.dataframe(pos_comparison, hide_index=True, use_container_width=True)

    # Top keepers table
    st.markdown("### 🌟 Best Keeper Values (by SPAR)")
    top_keepers = keeper_data.nlargest(15, "value")[
        ["player", "manager", "yahoo_position", "cost", "points", "value", "roi"]
    ].copy()
    top_keepers = top_keepers.rename(
        columns={"value": "spar", "roi": "spar_per_dollar"}
    )
    top_keepers = top_keepers.round(1)

    column_config = {
        "spar": st.column_config.NumberColumn("SPAR", format="%.1f"),
        "spar_per_dollar": st.column_config.NumberColumn("SPAR/$", format="%.2f"),
    }
    st.dataframe(
        top_keepers,
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
    )

    # Worst keepers table
    st.markdown("### 📉 Worst Keeper Values (by SPAR)")
    worst_keepers = keeper_data.nsmallest(10, "value")[
        ["player", "manager", "yahoo_position", "cost", "points", "value", "roi"]
    ].copy()
    worst_keepers = worst_keepers.rename(
        columns={"value": "spar", "roi": "spar_per_dollar"}
    )
    worst_keepers = worst_keepers.round(1)

    st.dataframe(
        worst_keepers,
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
    )
