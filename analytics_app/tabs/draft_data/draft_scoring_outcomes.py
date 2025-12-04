#!/usr/bin/env python3
"""
Enhanced Scoring Outcomes with visualizations and better filtering
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


@st.fragment
def display_scoring_outcomes(draft_data):
    """Enhanced scoring outcomes with charts and advanced filtering"""

    st.markdown("### Player Performance Analysis")
    st.caption("Compare draft cost vs actual performance using Manager SPAR")

    # Prepare data
    draft_data = draft_data.copy()
    draft_data["year"] = draft_data["year"].astype(str)
    draft_data["manager"] = draft_data["manager"].astype(str)

    # Filter out undrafted players
    draft_data = draft_data[draft_data["manager"].notna()]
    draft_data = draft_data[draft_data["manager"] != ""]
    draft_data = draft_data[draft_data["manager"].str.lower() != "nan"]
    draft_data = draft_data[draft_data["manager"].str.lower() != "none"]

    if "cost" in draft_data.columns:
        draft_data = draft_data[
            pd.to_numeric(draft_data["cost"], errors="coerce").fillna(0) > 0
        ]

    # Get filter options
    years = sorted(draft_data["year"].unique().tolist(), reverse=True)
    managers = sorted(draft_data["manager"].unique().tolist())
    allowed_positions = ["QB", "RB", "WR", "TE", "DEF", "K"]
    positions = [p for p in allowed_positions if p in draft_data["yahoo_position"].dropna().unique()]

    # Summary metrics - standardized 6-column layout
    total_picks = len(draft_data)
    avg_cost = draft_data["cost"].mean() if "cost" in draft_data.columns else 0
    avg_pts = draft_data["points"].mean() if "points" in draft_data.columns else 0
    avg_ppg = draft_data["season_ppg"].mean() if "season_ppg" in draft_data.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Picks", f"{total_picks:,}")
    with c2:
        st.metric("Avg Cost", f"${avg_cost:.0f}" if avg_cost > 0 else "—")
    with c3:
        st.metric("Avg Points", f"{avg_pts:.0f}" if avg_pts > 0 else "—")
    with c4:
        st.metric("Avg PPG", f"{avg_ppg:.1f}" if avg_ppg > 0 else "—")
    with c5:
        st.metric("Seasons", draft_data["year"].nunique())
    with c6:
        st.metric("Managers", draft_data["manager"].nunique())

    st.markdown("---")

    # Count active filters
    def count_active():
        count = 0
        if st.session_state.get("perf_year"):
            count += 1
        if st.session_state.get("perf_mgr"):
            count += 1
        if st.session_state.get("perf_pos"):
            count += 1
        if st.session_state.get("perf_player"):
            count += 1
        if not st.session_state.get("perf_inc_drafted", True):
            count += 1
        if not st.session_state.get("perf_inc_keepers", True):
            count += 1
        return count

    num_filters = count_active()
    filter_label = "Filters" + (f" ({num_filters} active)" if num_filters > 0 else "")

    with st.expander(filter_label, expanded=num_filters > 0):
        # Row 1: Year → Manager → Position → Player (consistent order)
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            selected_years = st.multiselect("Year", years, default=[], key="perf_year")

        with f2:
            selected_managers = st.multiselect("Manager", managers, default=[], key="perf_mgr")

        with f3:
            selected_positions = st.multiselect("Position", positions, default=[], key="perf_pos")

        with f4:
            search_players = st.multiselect("Player", sorted(draft_data["player"].unique().tolist()), default=[], key="perf_player")

        # Row 2: Include toggles
        st.markdown("**Include**")
        inc1, inc2, inc3 = st.columns([1, 1, 2])
        with inc1:
            include_drafted = st.checkbox("Drafted", value=True, key="perf_inc_drafted")
        with inc2:
            include_keepers = st.checkbox("Keepers", value=True, key="perf_inc_keepers")

    # Apply filters
    filtered_data = apply_scoring_filters(
        draft_data, selected_years, selected_managers, selected_positions,
        search_players, include_drafted, include_keepers, allowed_positions,
    )

    if filtered_data.empty:
        st.warning("No data matches your filters. Try adjusting them.")
        return

    st.caption(f"Showing {len(filtered_data):,} of {len(draft_data):,} picks")

    # === VISUALIZATION TABS ===
    viz_tab, table_tab, insights_tab = st.tabs(
        ["📈 Visualizations", "📋 Data Table", "💡 Insights"]
    )

    with viz_tab:
        display_performance_charts(filtered_data)

    with table_tab:
        display_performance_table(filtered_data, allowed_positions)

    with insights_tab:
        display_performance_insights(filtered_data)


def apply_scoring_filters(
    draft_data,
    selected_years,
    selected_managers,
    selected_positions,
    search_players,
    include_drafted,
    include_keepers,
    allowed_positions,
):
    """Apply all filters to draft data"""

    df = draft_data.copy()

    # Year filter
    if selected_years:
        df = df[df["year"].isin(selected_years)]

    # Manager filter
    if selected_managers:
        df = df[df["manager"].isin(selected_managers)]

    # Keeper status filter
    if include_drafted and not include_keepers:
        df = df[df["is_keeper_status"] != 1]
    elif not include_drafted and include_keepers:
        df = df[df["is_keeper_status"] == 1]
    elif not include_drafted and not include_keepers:
        return pd.DataFrame()

    # Position filter
    df = df[df["yahoo_position"].isin(allowed_positions)]
    if selected_positions:
        df = df[df["yahoo_position"].isin(selected_positions)]

    # Player search
    if search_players:
        df = df[df["player"].isin(search_players)]

    # Only include players with performance data
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["season_ppg"] = pd.to_numeric(df["season_ppg"], errors="coerce")
    df = df[(df["points"].notna()) | (df["season_ppg"].notna())]

    return df


def calculate_rankings(df):
    """Calculate performance rankings"""

    # Ensure numeric
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    df["season_ppg"] = pd.to_numeric(df["season_ppg"], errors="coerce").fillna(0)

    # Cost rank (handling 2014-2015 draft pick logic and undrafted players)
    def calc_cost_rank(group):
        year = group["year"].iloc[0]
        if year in ["2014", "2015"]:
            # For 2014-2015, use pick number (lower pick = better)
            # Undrafted players get last rank in group + 1
            has_pick = group["pick"].notna()
            if has_pick.any():
                has_pick.sum()
                group_copy = group.copy()
                group_copy.loc[~has_pick, "pick"] = group["pick"].max() + 1
                return group_copy["pick"].rank(method="first", ascending=True)
            else:
                # No players have picks - assign sequential ranks
                return pd.Series(range(1, len(group) + 1), index=group.index)
        else:
            # For auction years, use cost (higher cost = better rank)
            # Undrafted players (cost=0 or NaN) get worst rank
            return group["cost"].rank(
                method="first", ascending=False, na_option="bottom"
            )

    cost_rank_series = df.groupby(["year", "yahoo_position"], group_keys=False).apply(
        calc_cost_rank
    )
    # Replace infinite values before converting
    cost_rank_series = cost_rank_series.replace([np.inf, -np.inf], np.nan)
    # For any remaining NaNs, assign last rank + 1 (shouldn't happen with na_option='bottom')
    max_rank = cost_rank_series.max()
    cost_rank_series = cost_rank_series.fillna(
        max_rank + 1 if not pd.isna(max_rank) else 1
    ).astype(int)
    df["cost_rank"] = cost_rank_series

    # Points rank (players with 0 points get worst ranks via na_option='bottom')
    points_rank_series = df.groupby(["year", "yahoo_position"])["points"].transform(
        lambda x: x.rank(method="first", ascending=False, na_option="bottom")
    )
    points_rank_series = points_rank_series.replace([np.inf, -np.inf], np.nan)
    # For any remaining NaNs, assign last rank + 1
    max_rank = points_rank_series.max()
    points_rank_series = points_rank_series.fillna(
        max_rank + 1 if not pd.isna(max_rank) else 1
    ).astype(int)
    df["points_rank"] = points_rank_series

    # PPG rank (players with 0 ppg get worst ranks via na_option='bottom')
    ppg_rank_series = df.groupby(["year", "yahoo_position"])["season_ppg"].transform(
        lambda x: x.rank(method="first", ascending=False, na_option="bottom")
    )
    ppg_rank_series = ppg_rank_series.replace([np.inf, -np.inf], np.nan)
    # For any remaining NaNs, assign last rank + 1
    max_rank = ppg_rank_series.max()
    ppg_rank_series = ppg_rank_series.fillna(
        max_rank + 1 if not pd.isna(max_rank) else 1
    ).astype(int)
    df["ppg_rank"] = ppg_rank_series

    # Calculate value metrics using Manager SPAR (actual value while rostered)
    df["rank_diff"] = df["cost_rank"] - df["points_rank"]  # Positive = outperformed

    # Use manager_spar for draft value (actual value captured)
    # Priority: manager_spar > spar (legacy) > calculate from draft_roi
    if "manager_spar" in df.columns:
        df["manager_spar"] = pd.to_numeric(df["manager_spar"], errors="coerce").fillna(
            0
        )
        df["value_score"] = df["manager_spar"] / df["cost"].replace(0, np.nan)
        df["value_score"] = (
            df["value_score"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
    elif "draft_roi" in df.columns:
        df["value_score"] = pd.to_numeric(df["draft_roi"], errors="coerce").fillna(0)
    elif "spar" in df.columns:
        df["spar"] = pd.to_numeric(df["spar"], errors="coerce").fillna(0)
        df["value_score"] = df["spar"] / df["cost"].replace(0, np.nan)
        df["value_score"] = (
            df["value_score"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
    else:
        df["value_score"] = 0

    return df


@st.fragment
def display_performance_charts(filtered_data):
    """Create interactive visualizations"""

    # Calculate rankings
    plot_data = calculate_rankings(filtered_data)

    # Chart 1: Cost Rank vs Performance Rank
    st.markdown("#### Draft Position vs Actual Performance")
    st.caption("Points below the diagonal = outperformed draft cost")

    fig1 = px.scatter(
        plot_data,
        x="cost_rank",
        y="points_rank",
        color="yahoo_position",
        hover_data=["player", "year", "manager", "cost", "points", "season_ppg"],
        labels={
            "cost_rank": "Draft Position (Lower = Higher Cost)",
            "points_rank": "Performance Rank (Lower = Better)",
        },
        title="Draft Position vs Performance",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    # Add diagonal line (perfect prediction)
    max_rank = max(plot_data["cost_rank"].max(), plot_data["points_rank"].max())
    fig1.add_trace(
        go.Scatter(
            x=[0, max_rank],
            y=[0, max_rank],
            mode="lines",
            name="Perfect Prediction",
            line=dict(dash="dash", color="gray", width=2),
        )
    )

    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Value Distribution by Position
    st.markdown("#### Points Per Dollar by Position")

    value_data = plot_data[plot_data["cost"] > 0].copy()

    fig2 = px.box(
        value_data,
        x="yahoo_position",
        y="value_score",
        color="yahoo_position",
        labels={"value_score": "Points per Dollar", "yahoo_position": "Position"},
        title="Value Distribution by Position",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Top Performers vs Draft Position
    st.markdown("#### Best Value Picks")
    st.caption("Players who far exceeded their draft position")

    # Find biggest positive surprises
    top_surprises = plot_data.nlargest(20, "rank_diff")[
        [
            "player",
            "year",
            "yahoo_position",
            "cost_rank",
            "points_rank",
            "rank_diff",
            "manager",
        ]
    ]

    fig3 = px.bar(
        top_surprises,
        x="rank_diff",
        y="player",
        color="yahoo_position",
        hover_data=["year", "manager", "cost_rank", "points_rank"],
        labels={"rank_diff": "Spots Outperformed", "player": "Player"},
        title="Top 20 Value Picks (Outperformed Draft Position)",
        orientation="h",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)


@st.fragment
def display_performance_table(filtered_data, allowed_positions):
    """Display data table with rankings"""

    st.subheader("📋 Performance Data Table")

    # Calculate rankings
    table_data = calculate_rankings(filtered_data)

    # Sort by position and rank
    position_order = {pos: i for i, pos in enumerate(allowed_positions)}
    table_data["pos_order"] = table_data["yahoo_position"].map(position_order)
    table_data = table_data.sort_values(
        ["year", "pos_order", "cost_rank"], ascending=[False, True, True]
    )

    # Select columns for display
    display_cols = [
        "year",
        "player",
        "yahoo_position",
        "manager",
        "cost",
        "cost_rank",
        "points",
        "points_rank",
        "season_ppg",
        "ppg_rank",
        "rank_diff",
    ]

    # Add Manager SPAR (actual value while rostered) if available
    if "manager_spar" in table_data.columns:
        display_cols.append("manager_spar")
        if "value_score" in table_data.columns:
            display_cols.append("value_score")
    elif "spar" in table_data.columns:
        display_cols.append("spar")
        if "draft_roi" in table_data.columns:
            display_cols.append("draft_roi")
        elif "value_score" in table_data.columns:
            display_cols.append("value_score")

    display_df = table_data[
        [col for col in display_cols if col in table_data.columns]
    ].copy()

    # Build column names dynamically
    col_names = [
        "Year",
        "Player",
        "Pos",
        "Manager",
        "Cost",
        "Cost Rank",
        "Total Pts",
        "Pts Rank",
        "PPG",
        "PPG Rank",
        "Value (+/-)",
    ]

    # Add SPAR column names based on what's available
    if "manager_spar" in display_df.columns:
        col_names.append("Manager SPAR")
        if "value_score" in display_df.columns:
            col_names.append("SPAR/$")
    elif "spar" in display_df.columns:
        col_names.append("SPAR")
        if "draft_roi" in display_df.columns:
            col_names.append("SPAR/$")
        elif "value_score" in display_df.columns:
            col_names.append("SPAR/$")

    display_df.columns = col_names[: len(display_df.columns)]

    # Format numbers
    display_df["Cost"] = display_df["Cost"].apply(
        lambda x: f"${x:.0f}" if pd.notna(x) else "-"
    )
    display_df["PPG"] = display_df["PPG"].round(2)
    display_df["Total Pts"] = display_df["Total Pts"].round(1)

    # Format SPAR columns if they exist
    if "Manager SPAR" in display_df.columns:
        display_df["Manager SPAR"] = display_df["Manager SPAR"].round(1)
    elif "SPAR" in display_df.columns:
        display_df["SPAR"] = display_df["SPAR"].round(1)
    if "SPAR/$" in display_df.columns:
        display_df["SPAR/$"] = display_df["SPAR/$"].round(2)

    # Color code value column
    def color_value(val):
        if pd.isna(val):
            return ""
        if val > 5:
            return "background-color: #d1fae5; color: #065f46"
        elif val < -5:
            return "background-color: #fee2e2; color: #991b1b"
        return ""

    # Color code SPAR/$ column
    def color_spar_roi(val):
        if pd.isna(val):
            return ""
        if val > 2:
            return "background-color: #d1fae5; color: #065f46"
        elif val < 0:
            return "background-color: #fee2e2; color: #991b1b"
        return ""

    st.caption(
        "💡 **Value (+/-)**: Positive = outperformed draft position, Negative = underperformed | **Manager SPAR**: Actual value while rostered | **SPAR/$**: Manager SPAR per dollar (higher = better value)"
    )

    # Apply styling to both Value and SPAR/$ columns if they exist
    styled_df = display_df.style.applymap(color_value, subset=["Value (+/-)"])
    if "SPAR/$" in display_df.columns:
        styled_df = styled_df.applymap(color_spar_roi, subset=["SPAR/$"])

    st.dataframe(styled_df, hide_index=True, use_container_width=True, height=600)


@st.fragment
def display_performance_insights(filtered_data):
    """Generate insights from the data"""

    # Calculate metrics
    analysis_data = calculate_rankings(filtered_data)
    analysis_data = analysis_data[analysis_data["cost"] > 0]

    if analysis_data.empty:
        st.info("Not enough data for insights.")
        return

    # Top performers - card style
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Best Performances")
        st.caption("Top 5 value picks by SPAR/$")
        top_value = analysis_data.nlargest(5, "value_score")[
            ["player", "year", "yahoo_position", "cost", "points", "rank_diff", "value_score"]
        ]
        for idx, row in top_value.iterrows():
            st.markdown(f"**{row['player']}** ({row['year']}, {row['yahoo_position']})")
            st.caption(f"${row['cost']:.0f} → {row['points']:.0f} pts | +{row['rank_diff']:.0f} spots | SPAR/$: {row['value_score']:.2f}")
            st.markdown("---")

    with col2:
        st.markdown("#### Biggest Busts")
        st.caption("Worst value picks by rank difference")
        top_busts = analysis_data.nsmallest(5, "rank_diff")[
            ["player", "year", "yahoo_position", "cost", "points", "rank_diff", "value_score"]
        ]
        for idx, row in top_busts.iterrows():
            st.markdown(f"**{row['player']}** ({row['year']}, {row['yahoo_position']})")
            st.caption(f"${row['cost']:.0f} → {row['points']:.0f} pts | {row['rank_diff']:.0f} spots | SPAR/$: {row['value_score']:.2f}")
            st.markdown("---")

    # Position analysis
    st.markdown("#### Position Analysis")
    st.caption("How each position performed relative to draft expectations")

    pos_stats = (
        analysis_data.groupby("yahoo_position")
        .agg(
            {
                "rank_diff": ["mean", "std"],
                "value_score": "mean",
                "cost": "mean",
                "points": "mean",
            }
        )
        .round(2)
    )

    pos_stats.columns = [
        "Avg Rank Diff",
        "Std Dev",
        "Avg Value",
        "Avg Cost",
        "Avg Points",
    ]
    pos_stats = pos_stats.sort_values("Avg Value", ascending=False)

    st.dataframe(pos_stats, use_container_width=True)

    st.caption(
        """
    **Avg Rank Diff**: Average spots outperformed (positive) or underperformed (negative)
    **Avg Value**: Average Manager SPAR per dollar (actual value captured)
    **Std Dev**: Consistency (lower = more predictable)
    """
    )

    # Manager analysis
    if "manager" in analysis_data.columns and analysis_data["manager"].nunique() > 1:
        st.markdown("---")
        st.markdown("### 👤 Manager Performance")

        mgr_stats = (
            analysis_data.groupby("manager")
            .agg({"rank_diff": "mean", "value_score": "mean", "player": "count"})
            .round(2)
        )

        mgr_stats.columns = ["Avg Rank Diff", "Avg Value", "Picks"]
        mgr_stats = mgr_stats[mgr_stats["Picks"] >= 3]  # Min 3 picks
        mgr_stats = mgr_stats.sort_values("Avg Value", ascending=False)

        st.dataframe(mgr_stats, use_container_width=True)
        st.caption("*Managers with at least 3 picks shown*")
