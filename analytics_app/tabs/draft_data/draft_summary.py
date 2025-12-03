from __future__ import annotations
import streamlit as st
import pandas as pd

# Grade color mapping for styling
GRADE_COLORS = {
    "A": "#28a745",  # Green
    "B": "#6c9a1f",  # Yellow-green
    "C": "#ffc107",  # Yellow
    "D": "#fd7e14",  # Orange
    "F": "#dc3545",  # Red
}

VALUE_TIER_COLORS = {
    "Steal": "#28a745",  # Green
    "Good Value": "#6c9a1f",  # Yellow-green
    "Fair": "#6c757d",  # Gray
    "Overpay": "#fd7e14",  # Orange
    "Bust": "#dc3545",  # Red
}


@st.fragment
def display_draft_summary(draft_data: pd.DataFrame) -> None:
    """Enhanced draft summary with draft grades, value tiers, and dual SPAR metrics."""

    st.markdown("### 📊 Draft Pick Summary")
    st.markdown(
        "*Complete draft history with grades, value tiers, and performance stats*"
    )

    # Prepare data
    df = draft_data.copy()

    # Normalize manager/year columns
    if "manager" in df.columns:
        df["manager"] = df["manager"].fillna("").astype(str).str.strip()
        # Remove invalid managers
        df = df[
            (df["manager"] != "") & (df["manager"] != "None") & (df["manager"].notna())
        ]

    if "year" in df.columns:
        df["year"] = df["year"].astype(str)

    # Ensure numeric columns
    numeric_cols = [
        "cost",
        "pick",
        "round",
        "total_fantasy_points",
        "season_ppg",
        "games_played",
        "weeks_rostered",
        "weeks_started",
        "manager_spar",
        "season_overall_rank",
        "season_position_rank",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Summary Metrics
    total_picks = len(df)
    total_years = df["year"].nunique() if "year" in df.columns else 0
    avg_cost = df[df["cost"] > 0]["cost"].mean() if "cost" in df.columns else 0
    avg_ppg = df["season_ppg"].mean() if "season_ppg" in df.columns else 0
    keeper_count = (
        (pd.to_numeric(df["is_keeper_status"], errors="coerce") == 1).sum()
        if "is_keeper_status" in df.columns
        else 0
    )
    breakout_count = (
        (pd.to_numeric(df["is_breakout"], errors="coerce") == 1).sum()
        if "is_breakout" in df.columns
        else 0
    )
    bust_count = (
        (pd.to_numeric(df["is_bust"], errors="coerce") == 1).sum()
        if "is_bust" in df.columns
        else 0
    )

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("Total Picks", f"{total_picks:,}")
    with col2:
        st.metric("Seasons", total_years)
    with col3:
        st.metric("Avg Cost", f"${avg_cost:.1f}" if avg_cost > 0 else "N/A")
    with col4:
        st.metric("Avg PPG", f"{avg_ppg:.2f}" if avg_ppg > 0 else "N/A")
    with col5:
        st.metric("Keepers", f"{keeper_count:,}")
    with col6:
        st.metric(
            "Breakouts",
            f"{breakout_count:,}",
            help="Late-round picks with top finishes",
        )
    with col7:
        st.metric(
            "Busts", f"{bust_count:,}", help="Early-round picks with bottom finishes"
        )

    st.divider()

    # Filters
    with st.expander("🔍 Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            years = (
                sorted(df["year"].unique().tolist(), reverse=True)
                if "year" in df.columns
                else []
            )
            selected_years = st.multiselect(
                "Year", years, default=[], key="draft_summary_year"
            )

        with col2:
            managers = (
                sorted(df["manager"].unique().tolist())
                if "manager" in df.columns
                else []
            )
            selected_managers = st.multiselect(
                "Manager", managers, default=[], key="draft_summary_mgr"
            )

        with col3:
            desired_order = ["QB", "RB", "WR", "TE", "DEF", "K"]
            all_positions = (
                df["yahoo_position"].dropna().unique().tolist()
                if "yahoo_position" in df.columns
                else []
            )
            positions = [pos for pos in desired_order if pos in all_positions] + sorted(
                [pos for pos in all_positions if pos not in desired_order]
            )
            selected_positions = st.multiselect(
                "Position", positions, default=[], key="draft_summary_pos"
            )

        with col4:
            players = (
                sorted(df["player"].unique().tolist()) if "player" in df.columns else []
            )
            selected_players = st.multiselect(
                "Player", players, default=[], key="draft_summary_player"
            )

        # Second row of filters: Grade, Value Tier, and quick filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            grades = ["A", "B", "C", "D", "F"]
            available_grades = [
                g
                for g in grades
                if "draft_grade" in df.columns
                and g in df["draft_grade"].dropna().unique()
            ]
            selected_grades = st.multiselect(
                "Draft Grade",
                available_grades if available_grades else grades,
                default=[],
                key="draft_summary_grade",
            )

        with col2:
            tiers = ["Steal", "Good Value", "Fair", "Overpay", "Bust"]
            available_tiers = [
                t
                for t in tiers
                if "value_tier" in df.columns
                and t in df["value_tier"].dropna().unique()
            ]
            selected_tiers = st.multiselect(
                "Value Tier",
                available_tiers if available_tiers else tiers,
                default=[],
                key="draft_summary_tier",
            )

        with col3:
            show_breakouts = st.checkbox(
                "Breakouts Only",
                value=False,
                key="draft_summary_breakouts",
                help="Late-round picks with top finishes",
            )
        with col4:
            show_busts = st.checkbox(
                "Busts Only",
                value=False,
                key="draft_summary_busts",
                help="Early-round picks with bottom finishes",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            include_keepers = st.checkbox(
                "Include Keepers", value=True, key="draft_summary_keepers"
            )
        with col2:
            min_cost = st.number_input(
                "Min Cost", min_value=0, value=0, key="draft_summary_min_cost"
            )
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                [
                    "Recent",
                    "Points ↓",
                    "PPG ↓",
                    "Cost ↓",
                    "Overall Rank",
                    "Grade ↓",
                    "SPAR ↓",
                ],
                key="draft_summary_sort",
            )

    # Apply filters
    filtered = df.copy()
    if selected_years:
        filtered = filtered[filtered["year"].isin(selected_years)]
    if selected_managers:
        filtered = filtered[filtered["manager"].isin(selected_managers)]
    if selected_positions:
        filtered = filtered[filtered["yahoo_position"].isin(selected_positions)]
    if selected_players:
        filtered = filtered[filtered["player"].isin(selected_players)]
    if not include_keepers and "is_keeper_status" in filtered.columns:
        filtered = filtered[
            pd.to_numeric(filtered["is_keeper_status"], errors="coerce") != 1
        ]
    if min_cost > 0 and "cost" in filtered.columns:
        filtered = filtered[filtered["cost"] >= min_cost]

    # New filters: grade, tier, breakouts, busts
    if selected_grades and "draft_grade" in filtered.columns:
        filtered = filtered[filtered["draft_grade"].isin(selected_grades)]
    if selected_tiers and "value_tier" in filtered.columns:
        filtered = filtered[filtered["value_tier"].isin(selected_tiers)]
    if show_breakouts and "is_breakout" in filtered.columns:
        filtered = filtered[
            pd.to_numeric(filtered["is_breakout"], errors="coerce") == 1
        ]
    if show_busts and "is_bust" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["is_bust"], errors="coerce") == 1]

    # Apply sorting
    if sort_by == "Points ↓" and "total_fantasy_points" in filtered.columns:
        filtered = filtered.sort_values("total_fantasy_points", ascending=False)
    elif sort_by == "PPG ↓" and "season_ppg" in filtered.columns:
        filtered = filtered.sort_values("season_ppg", ascending=False)
    elif sort_by == "Cost ↓" and "cost" in filtered.columns:
        filtered = filtered.sort_values("cost", ascending=False)
    elif sort_by == "Overall Rank" and "season_overall_rank" in filtered.columns:
        filtered = filtered.sort_values("season_overall_rank", ascending=True)
    elif sort_by == "Grade ↓" and "draft_grade" in filtered.columns:
        # Sort A->F (A is best)
        grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
        filtered["_grade_sort"] = filtered["draft_grade"].map(grade_order).fillna(5)
        filtered = filtered.sort_values("_grade_sort", ascending=True)
        filtered = filtered.drop(columns=["_grade_sort"])
    elif sort_by == "SPAR ↓" and "manager_spar" in filtered.columns:
        filtered = filtered.sort_values("manager_spar", ascending=False)
    else:
        # Recent: sort by year desc, then round/pick
        if "year" in filtered.columns:
            sort_cols = ["year"]
            sort_order = [False]
            if "round" in filtered.columns:
                sort_cols.append("round")
                sort_order.append(True)
            if "pick" in filtered.columns:
                sort_cols.append("pick")
                sort_order.append(True)
            filtered = filtered.sort_values(sort_cols, ascending=sort_order)

    st.markdown(f"**Showing {len(filtered)} of {len(df)} picks**")

    # Build display columns - include new grade and tier columns
    display_cols = [
        "year",
        "round",
        "pick",
        "manager",
        "player",
        "yahoo_position",
        "nfl_team",
        "cost",
        "draft_grade",
        "value_tier",
        "total_fantasy_points",
        "season_ppg",
        "games_played",
        "manager_spar",
        "season_position_rank",
        "is_keeper_status",
    ]

    # Only include columns that exist
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Column configuration
    column_config = {
        "year": st.column_config.TextColumn("Year", width="small"),
        "round": st.column_config.NumberColumn("Rnd", format="%d", width="small"),
        "pick": st.column_config.NumberColumn("Pick", format="%d", width="small"),
        "manager": st.column_config.TextColumn("Manager", width="medium"),
        "player": st.column_config.TextColumn("Player", width="medium"),
        "yahoo_position": st.column_config.TextColumn("Pos", width="small"),
        "nfl_team": st.column_config.TextColumn("Team", width="small"),
        "cost": st.column_config.NumberColumn("Cost", format="$%d", width="small"),
        "draft_grade": st.column_config.TextColumn(
            "Grade", width="small", help="A-F based on SPAR percentile"
        ),
        "value_tier": st.column_config.TextColumn(
            "Value", width="small", help="Steal/Good Value/Fair/Overpay/Bust"
        ),
        "total_fantasy_points": st.column_config.NumberColumn(
            "Points", format="%.1f", width="small"
        ),
        "season_ppg": st.column_config.NumberColumn(
            "PPG", format="%.2f", width="small"
        ),
        "games_played": st.column_config.NumberColumn(
            "Games", format="%d", width="small"
        ),
        "weeks_rostered": st.column_config.NumberColumn(
            "Rostered", format="%.0f", help="Weeks on roster", width="small"
        ),
        "weeks_started": st.column_config.NumberColumn(
            "Started", format="%.0f", help="Weeks started", width="small"
        ),
        "manager_spar": st.column_config.NumberColumn(
            "SPAR", format="%.1f", help="SPAR while rostered", width="small"
        ),
        "season_overall_rank": st.column_config.NumberColumn(
            "Overall Rank", format="%d", width="small"
        ),
        "season_position_rank": st.column_config.NumberColumn(
            "Pos Rank", format="%d", width="small"
        ),
        "is_keeper_status": st.column_config.CheckboxColumn(
            "Keeper", help="Keeper pick", width="small"
        ),
    }

    # Only include configs for columns that exist
    column_config = {k: v for k, v in column_config.items() if k in display_cols}

    st.dataframe(
        filtered[display_cols],
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
    )

    # Grade distribution summary (if available)
    if "draft_grade" in filtered.columns and filtered["draft_grade"].notna().any():
        st.markdown("---")
        st.markdown("#### Grade Distribution")
        grade_counts = (
            filtered["draft_grade"]
            .value_counts()
            .reindex(["A", "B", "C", "D", "F"])
            .fillna(0)
        )
        cols = st.columns(5)
        for i, (grade, count) in enumerate(grade_counts.items()):
            with cols[i]:
                color = GRADE_COLORS.get(grade, "#6c757d")
                st.markdown(
                    f"<div style='text-align:center;'><span style='font-size:2em;color:{color};font-weight:bold;'>{grade}</span><br/>{int(count)} picks</div>",
                    unsafe_allow_html=True,
                )

    # Export
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button(
        "📥 Export", csv, "draft_summary.csv", "text/csv", key="draft_summary_export"
    )
