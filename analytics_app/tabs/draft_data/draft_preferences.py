#!/usr/bin/env python3
"""
Enhanced Draft Preferences with heatmaps and trend analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


@st.fragment
def display_draft_preferences(draft_data):
    """Enhanced draft preferences with visualizations"""

    st.header("📈 Draft Trends & Preferences")

    # Info banner
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;">
        <h3 style="margin: 0 0 0.5rem 0;">📊 Understand Draft Patterns</h3>
        <p style="margin: 0; opacity: 0.95; color: #1a1a1a;">
        Analyze spending habits, position preferences, and historical trends to make smarter draft decisions.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Prepare data
    draft_data = draft_data.copy()
    # Coerce year to numeric for safe comparisons and plotting
    draft_data["year"] = pd.to_numeric(draft_data.get("year"), errors="coerce")
    # Keep manager as string but ensure not-na
    draft_data["manager"] = draft_data["manager"].astype(str)

    # Remove invalid managers and years with no costs
    draft_data = draft_data[draft_data["manager"].notna()]
    draft_data = draft_data[draft_data["manager"].str.lower() != "nan"]

    # Ensure cost numeric for grouping
    draft_data["cost"] = pd.to_numeric(draft_data.get("cost"), errors="coerce").fillna(
        0
    )

    nonzero_years = draft_data.groupby("year")["cost"].sum()
    valid_years = nonzero_years[nonzero_years != 0].index.tolist()
    # Filter out NaN years
    valid_years = [y for y in valid_years if y == y]
    draft_data = draft_data[draft_data["year"].isin(valid_years)]

    allowed_positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

    # Get options
    years = sorted(
        list({int(y) for y in draft_data["year"].dropna().unique().tolist()})
    )
    if not years:
        st.warning("No valid draft years with cost data available to analyze.")
        return

    managers = ["League Average"] + sorted(draft_data["manager"].unique().tolist())

    # === FILTERS ===
    st.markdown("### 🔍 Analysis Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Ensure default index safe
        start_idx = 0
        start_year = st.selectbox(
            "Start Year", years, index=start_idx, key="pref_start"
        )
    with col2:
        end_idx = max(0, len(years) - 1)
        end_year = st.selectbox("End Year", years, index=end_idx, key="pref_end")
    with col3:
        selected_manager = st.selectbox("Manager", managers, index=0, key="pref_mgr")

    # Ensure start/end are ints
    try:
        start_year = int(start_year)
        end_year = int(end_year)
    except Exception:
        st.warning("Invalid start/end year selection")
        return

    # Ensure start <= end
    if start_year > end_year:
        st.warning("Start year must be less than or equal to end year")
        return

    selected_years = [y for y in years if start_year <= y <= end_year]

    st.markdown("---")

    # === TABS ===
    viz_tab, tables_tab, trends_tab = st.tabs(
        ["📊 Visualizations", "📋 Data Tables", "📈 Trends Over Time"]
    )

    with viz_tab:
        display_preference_charts(
            draft_data, selected_years, selected_manager, allowed_positions
        )

    with tables_tab:
        display_preference_tables(
            draft_data, selected_years, selected_manager, allowed_positions
        )

    with trends_tab:
        display_preference_trends(
            draft_data, selected_years, selected_manager, allowed_positions
        )


@st.fragment
def display_preference_charts(
    draft_data, selected_years, selected_manager, allowed_positions
):
    """Create interactive preference visualizations"""

    st.subheader("📊 Visual Analysis")

    # Filter data
    filtered = draft_data[draft_data["year"].isin(selected_years)].copy()

    if selected_manager != "League Average":
        filtered = filtered[filtered["manager"] == selected_manager]

    # === SAFELY HANDLE KEEPER MASK ===
    # Create an explicit boolean mask for non-keepers (drafted only). This avoids ambiguous
    # chaining methods that some static analyzers may misinterpret.
    if "is_keeper_status" in filtered.columns:
        keeper_mask = filtered["is_keeper_status"] == 1
        # Treat NaNs as not a keeper (i.e., drafted)
        keeper_mask = keeper_mask.fillna(False)
        drafted_only = filtered[~keeper_mask].copy()
    else:
        drafted_only = filtered.copy()

    # Ensure numeric columns exist and coerce safely
    if "cost" in drafted_only.columns:
        drafted_only["cost"] = pd.to_numeric(drafted_only["cost"], errors="coerce")
    else:
        drafted_only["cost"] = pd.NA

    if "season_ppg" in drafted_only.columns:
        drafted_only["season_ppg"] = pd.to_numeric(
            drafted_only["season_ppg"], errors="coerce"
        )
    else:
        drafted_only["season_ppg"] = pd.NA

    # Chart 1: Spending by Position
    st.markdown("#### 💰 Spending Distribution by Position")

    pos_spending = (
        drafted_only[drafted_only["cost"] > 0]
        .groupby("yahoo_position")
        .agg({"cost": ["sum", "mean", "count"]})
        .reset_index()
    )
    pos_spending.columns = ["Position", "Total Spent", "Avg Cost", "Count"]
    pos_spending = pos_spending[pos_spending["Position"].isin(allowed_positions)]

    fig1 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Total Spending", "Average Cost per Pick"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    fig1.add_trace(
        go.Bar(
            x=pos_spending["Position"],
            y=pos_spending["Total Spent"],
            name="Total",
            marker_color="#3b82f6",
        ),
        row=1,
        col=1,
    )

    fig1.add_trace(
        go.Bar(
            x=pos_spending["Position"],
            y=pos_spending["Avg Cost"],
            name="Average",
            marker_color="#10b981",
        ),
        row=1,
        col=2,
    )

    fig1.update_layout(height=400, showlegend=False)
    fig1.update_xaxes(title_text="Position", row=1, col=1)
    fig1.update_xaxes(title_text="Position", row=1, col=2)
    fig1.update_yaxes(title_text="Total $", row=1, col=1)
    fig1.update_yaxes(title_text="Avg $", row=1, col=2)

    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Cost vs Performance Heatmap
    st.markdown("#### 🔥 Position Rankings: Cost vs Performance")
    st.caption("Compare draft spending to actual performance by position tier")

    # Calculate position ranks
    drafted_only["pos_rank"] = drafted_only.groupby(["year", "yahoo_position"])[
        "cost"
    ].rank(method="first", ascending=False)
    drafted_only["pos_rank"] = (
        drafted_only["pos_rank"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(int)
    )

    # Aggregate by position and rank
    heatmap_data = (
        drafted_only[
            (drafted_only["cost"] > 0)
            & (drafted_only["yahoo_position"].isin(allowed_positions))
        ]
        .groupby(["yahoo_position", "pos_rank"])
        .agg({"cost": "mean", "season_ppg": "mean"})
        .reset_index()
    )

    # Pivot for heatmap
    cost_pivot = heatmap_data.pivot(
        index="yahoo_position", columns="pos_rank", values="cost"
    ).fillna(0)

    ppg_pivot = heatmap_data.pivot(
        index="yahoo_position", columns="pos_rank", values="season_ppg"
    ).fillna(0)

    # Create dual heatmaps
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Average Cost by Rank", "Average PPG by Rank"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
    )

    fig2.add_trace(
        go.Heatmap(
            z=cost_pivot.values,
            x=[f"#{i}" for i in cost_pivot.columns],
            y=cost_pivot.index,
            colorscale="Blues",
            text=cost_pivot.values.round(1),
            texttemplate="$%{text}",
            textfont={"size": 10},
            colorbar=dict(title="$", x=0.46),
        ),
        row=1,
        col=1,
    )

    fig2.add_trace(
        go.Heatmap(
            z=ppg_pivot.values,
            x=[f"#{i}" for i in ppg_pivot.columns],
            y=ppg_pivot.index,
            colorscale="Greens",
            text=ppg_pivot.values.round(1),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="PPG", x=1.02),
        ),
        row=1,
        col=2,
    )

    fig2.update_layout(height=400)
    fig2.update_xaxes(title_text="Position Rank", row=1, col=1)
    fig2.update_xaxes(title_text="Position Rank", row=1, col=2)

    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Draft Capital Allocation
    st.markdown("#### 🥧 Draft Budget Allocation")

    allocation = pos_spending.copy()
    allocation["Percentage"] = (
        allocation["Total Spent"] / allocation["Total Spent"].sum() * 100
    ).round(1)

    fig3 = px.pie(
        allocation,
        values="Total Spent",
        names="Position",
        title=f"Budget Distribution{' - ' + selected_manager if selected_manager != 'League Average' else ' - League Average'}",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig3.update_traces(textposition="inside", textinfo="percent+label")
    fig3.update_layout(height=500)

    st.plotly_chart(fig3, use_container_width=True)


@st.fragment
def display_preference_tables(
    draft_data, selected_years, selected_manager, allowed_positions
):
    """Display detailed preference tables"""

    st.subheader("📋 Detailed Statistics")

    # Filter data
    filtered = draft_data[draft_data["year"].isin(selected_years)].copy()

    if selected_manager != "League Average":
        filtered = filtered[filtered["manager"] == selected_manager]

    # === DRAFTED PLAYERS ===
    st.markdown("### Drafted Players (Non-Keepers)")

    drafted = filtered[filtered["is_keeper_status"].ne(1).fillna(True)].copy()
    drafted = prepare_preference_table(
        drafted, selected_manager, allowed_positions, is_keeper=False
    )

    if not drafted.empty:
        st.dataframe(drafted, hide_index=True, use_container_width=True)
    else:
        st.info("No drafted players found.")

    st.markdown("---")

    # === KEEPERS ===
    st.markdown("### Kept Players")

    kept = filtered[filtered["is_keeper_status"].eq(1).fillna(False)].copy()
    kept = prepare_preference_table(
        kept, selected_manager, allowed_positions, is_keeper=True
    )

    if not kept.empty:
        st.dataframe(kept, hide_index=True, use_container_width=True)
    else:
        st.info("No keeper data found.")


def prepare_preference_table(
    data, selected_manager, allowed_positions, is_keeper=False
):
    """Prepare aggregated preference table"""

    data = data[data["yahoo_position"].isin(allowed_positions)].copy()
    data = data[data["cost"] > 0]

    # Ensure numeric
    data["cost"] = pd.to_numeric(data.get("cost"), errors="coerce")
    data["season_ppg"] = pd.to_numeric(data.get("season_ppg"), errors="coerce")
    data["points"] = pd.to_numeric(data.get("points"), errors="coerce")

    if not is_keeper:
        # Calculate position ranks
        data["pos_rank"] = data.groupby(["year", "yahoo_position"])["cost"].rank(
            method="first", ascending=False
        )
        data["pos_rank"] = (
            data["pos_rank"].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
        )
        data["pos_label"] = data["yahoo_position"] + data["pos_rank"].astype(str)

        group_cols = ["pos_label", "yahoo_position"]
        if selected_manager != "League Average":
            group_cols.append("manager")

        agg_data = (
            data.groupby(group_cols)
            .agg(
                {
                    "cost": ["mean", "max", "min", "median"],
                    "season_ppg": "mean",
                    "points": "mean",
                    "pos_label": "count",
                }
            )
            .reset_index()
        )

        agg_data.columns = (
            [
                "Position Rank",
                "Position",
                "Manager",
                "Avg Cost",
                "Max Cost",
                "Min Cost",
                "Median Cost",
                "Avg PPG",
                "Avg Points",
                "Times Drafted",
            ]
            if selected_manager != "League Average"
            else [
                "Position Rank",
                "Position",
                "Avg Cost",
                "Max Cost",
                "Min Cost",
                "Median Cost",
                "Avg PPG",
                "Avg Points",
                "Times Drafted",
            ]
        )
    else:
        group_cols = ["yahoo_position"]
        if selected_manager != "League Average":
            group_cols.append("manager")

        agg_data = (
            data.groupby(group_cols)
            .agg(
                {
                    "cost": ["mean", "max", "min", "median"],
                    "season_ppg": "mean",
                    "points": "mean",
                    "player": "count",
                }
            )
            .reset_index()
        )

        agg_data.columns = (
            [
                "Position",
                "Manager",
                "Avg Cost",
                "Max Cost",
                "Min Cost",
                "Median Cost",
                "Avg PPG",
                "Avg Points",
                "Times Kept",
            ]
            if selected_manager != "League Average"
            else [
                "Position",
                "Avg Cost",
                "Max Cost",
                "Min Cost",
                "Median Cost",
                "Avg PPG",
                "Avg Points",
                "Times Kept",
            ]
        )

    # Format numbers
    for col in ["Avg Cost", "Max Cost", "Min Cost", "Median Cost"]:
        if col in agg_data.columns:
            agg_data[col] = agg_data[col].round(2)

    for col in ["Avg PPG", "Avg Points"]:
        if col in agg_data.columns:
            agg_data[col] = agg_data[col].round(2)

    # Sort
    position_order = pd.CategoricalDtype(allowed_positions, ordered=True)
    agg_data["Position"] = agg_data["Position"].astype(position_order)

    if not is_keeper and "Position Rank" in agg_data.columns:
        agg_data["rank_num"] = (
            agg_data["Position Rank"].str.extract(r"(\d+)").fillna(0).astype(int)
        )
        agg_data = agg_data.sort_values(["Position", "rank_num"])
        agg_data = agg_data.drop(columns=["rank_num"])
    else:
        agg_data = agg_data.sort_values("Position")

    return agg_data


@st.fragment
def display_preference_trends(
    draft_data, selected_years, selected_manager, allowed_positions
):
    """Show trends over time"""

    st.subheader("📈 Historical Trends")

    # Filter data
    filtered = draft_data[draft_data["year"].isin(selected_years)].copy()

    if selected_manager != "League Average":
        filtered = filtered[filtered["manager"] == selected_manager]

    drafted = filtered[filtered["is_keeper_status"].ne(1).fillna(True)].copy()
    drafted = drafted[drafted["yahoo_position"].isin(allowed_positions)]
    drafted["cost"] = pd.to_numeric(drafted.get("cost"), errors="coerce")
    drafted["season_ppg"] = pd.to_numeric(drafted.get("season_ppg"), errors="coerce")

    if len(selected_years) < 2:
        st.warning("⚠️ Select at least 2 years to see trends.")
        return

    # Trend 1: Average cost by position over time
    st.markdown("#### 💵 Average Draft Cost Trends")

    cost_trend = (
        drafted[drafted["cost"] > 0]
        .groupby(["year", "yahoo_position"])
        .agg({"cost": "mean"})
        .reset_index()
    )

    fig1 = px.line(
        cost_trend,
        x="year",
        y="cost",
        color="yahoo_position",
        markers=True,
        labels={
            "cost": "Average Cost ($)",
            "year": "Year",
            "yahoo_position": "Position",
        },
        title="Position Cost Trends Over Time",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # Trend 2: Draft picks by position over time
    st.markdown("#### 📊 Position Popularity Trends")

    pick_trend = (
        drafted.groupby(["year", "yahoo_position"]).size().reset_index(name="count")
    )

    fig2 = px.bar(
        pick_trend,
        x="year",
        y="count",
        color="yahoo_position",
        labels={
            "count": "Number of Picks",
            "year": "Year",
            "yahoo_position": "Position",
        },
        title="Picks by Position Over Time",
        color_discrete_sequence=px.colors.qualitative.Bold,
        barmode="group",
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Trend 3: Value over time
    st.markdown("#### 💎 Value Efficiency Trends")
    st.caption("Manager SPAR ROI (actual value captured per dollar) - higher is better")

    # Use manager_spar (actual value while rostered) with fallback to spar then draft_roi
    value_data = drafted[drafted["cost"] > 0].copy()
    if "manager_spar" in value_data.columns:
        value_data["manager_spar_num"] = pd.to_numeric(
            value_data["manager_spar"], errors="coerce"
        ).fillna(0)
        value_data["value"] = value_data["manager_spar_num"] / value_data["cost"]
    elif "draft_roi" in value_data.columns:
        value_data["value"] = pd.to_numeric(
            value_data["draft_roi"], errors="coerce"
        ).fillna(0)
    elif "spar" in value_data.columns:
        value_data["spar_num"] = pd.to_numeric(
            value_data["spar"], errors="coerce"
        ).fillna(0)
        value_data["value"] = value_data["spar_num"] / value_data["cost"]
    else:
        value_data["value"] = 0

    value_trend = (
        value_data.groupby(["year", "yahoo_position"])
        .agg({"value": "mean"})
        .reset_index()
    )

    fig3 = px.line(
        value_trend,
        x="year",
        y="value",
        color="yahoo_position",
        markers=True,
        labels={
            "value": "Manager SPAR ROI",
            "year": "Year",
            "yahoo_position": "Position",
        },
        title="Value Efficiency (Manager SPAR ROI) by Position Over Time",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # Summary stats
    st.markdown("---")
    st.markdown("### 📊 Summary Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_cost = drafted[drafted["cost"] > 0]["cost"].mean()
        st.metric("Avg Draft Cost", f"${avg_cost:.2f}")

    with col2:
        total_spent = drafted[drafted["cost"] > 0]["cost"].sum()
        st.metric("Total Spent", f"${total_spent:.2f}")

    with col3:
        total_picks = len(drafted[drafted["cost"] > 0])
        st.metric("Total Picks", f"{total_picks}")
