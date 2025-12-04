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

    st.markdown("### Draft Pick Summary")
    st.caption("Complete draft history with grades, value tiers, and performance stats")

    # Prepare data
    df = draft_data.copy()

    # Normalize manager/year columns
    if "manager" in df.columns:
        df["manager"] = df["manager"].fillna("").astype(str).str.strip()
        df = df[
            (df["manager"] != "") & (df["manager"] != "None") & (df["manager"].notna())
        ]

    if "year" in df.columns:
        df["year"] = df["year"].astype(str)

    # Ensure numeric columns
    numeric_cols = [
        "cost", "pick", "round", "total_fantasy_points", "season_ppg",
        "games_played", "weeks_rostered", "weeks_started", "manager_spar",
        "season_overall_rank", "season_position_rank",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Summary Metrics - standardized 6-column layout
    total_picks = len(df)
    total_years = df["year"].nunique() if "year" in df.columns else 0
    avg_cost = df[df["cost"] > 0]["cost"].mean() if "cost" in df.columns else 0
    avg_ppg = df["season_ppg"].mean() if "season_ppg" in df.columns else 0
    keeper_count = (
        (pd.to_numeric(df["is_keeper_status"], errors="coerce") == 1).sum()
        if "is_keeper_status" in df.columns else 0
    )
    breakout_count = (
        (pd.to_numeric(df["is_breakout"], errors="coerce") == 1).sum()
        if "is_breakout" in df.columns else 0
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Picks", f"{total_picks:,}")
    with c2:
        st.metric("Seasons", total_years)
    with c3:
        st.metric("Avg Cost", f"${avg_cost:.0f}" if avg_cost > 0 else "—")
    with c4:
        st.metric("Avg PPG", f"{avg_ppg:.1f}" if avg_ppg > 0 else "—")
    with c5:
        st.metric("Keepers", f"{keeper_count:,}")
    with c6:
        st.metric("Breakouts", f"{breakout_count:,}")

    st.markdown("---")

    # Count active filters for label
    def count_active_filters():
        count = 0
        if st.session_state.get("draft_summary_year"):
            count += 1
        if st.session_state.get("draft_summary_mgr"):
            count += 1
        if st.session_state.get("draft_summary_pos"):
            count += 1
        if st.session_state.get("draft_summary_player"):
            count += 1
        if st.session_state.get("draft_summary_grade"):
            count += 1
        if st.session_state.get("draft_summary_tier"):
            count += 1
        if st.session_state.get("draft_summary_breakouts"):
            count += 1
        if st.session_state.get("draft_summary_busts"):
            count += 1
        if not st.session_state.get("draft_summary_keepers", True):
            count += 1
        if st.session_state.get("draft_summary_min_cost", 0) > 0:
            count += 1
        return count

    num_filters = count_active_filters()
    filter_label = "Filters" + (f" ({num_filters} active)" if num_filters > 0 else "")

    # Filters
    with st.expander(filter_label, expanded=num_filters > 0):
        # Row 1: Year → Manager → Position → Player (consistent order)
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            years = sorted(df["year"].unique().tolist(), reverse=True) if "year" in df.columns else []
            selected_years = st.multiselect("Year", years, default=[], key="draft_summary_year")

        with f2:
            managers = sorted(df["manager"].unique().tolist()) if "manager" in df.columns else []
            selected_managers = st.multiselect("Manager", managers, default=[], key="draft_summary_mgr")

        with f3:
            desired_order = ["QB", "RB", "WR", "TE", "DEF", "K"]
            all_positions = df["yahoo_position"].dropna().unique().tolist() if "yahoo_position" in df.columns else []
            positions = [pos for pos in desired_order if pos in all_positions] + sorted(
                [pos for pos in all_positions if pos not in desired_order]
            )
            selected_positions = st.multiselect("Position", positions, default=[], key="draft_summary_pos")

        with f4:
            players = sorted(df["player"].unique().tolist()) if "player" in df.columns else []
            selected_players = st.multiselect("Player", players, default=[], key="draft_summary_player")

        # Row 2: Grade, Value Tier, Sort
        f5, f6, f7 = st.columns(3)

        with f5:
            grades = ["A", "B", "C", "D", "F"]
            available_grades = [g for g in grades if "draft_grade" in df.columns and g in df["draft_grade"].dropna().unique()]
            selected_grades = st.multiselect("Grade", available_grades if available_grades else grades, default=[], key="draft_summary_grade")

        with f6:
            tiers = ["Steal", "Good Value", "Fair", "Overpay", "Bust"]
            available_tiers = [t for t in tiers if "value_tier" in df.columns and t in df["value_tier"].dropna().unique()]
            selected_tiers = st.multiselect("Value Tier", available_tiers if available_tiers else tiers, default=[], key="draft_summary_tier")

        with f7:
            sort_by = st.selectbox("Sort by", ["Recent", "Points ↓", "PPG ↓", "Cost ↓", "Overall Rank", "Grade ↓", "SPAR ↓"], key="draft_summary_sort")

        # Row 3: Include toggles and min cost
        st.markdown("**Include**")
        inc1, inc2, inc3, inc4, inc5 = st.columns(5)
        with inc1:
            include_keepers = st.checkbox("Keepers", value=True, key="draft_summary_keepers")
        with inc2:
            show_breakouts = st.checkbox("Breakouts Only", value=False, key="draft_summary_breakouts")
        with inc3:
            show_busts = st.checkbox("Busts Only", value=False, key="draft_summary_busts")
        with inc4:
            min_cost = st.number_input("Min Cost", min_value=0, value=0, key="draft_summary_min_cost")

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

    # Table header row
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.caption(f"Showing {len(filtered):,} of {len(df):,} picks")
    with header_right:
        csv = filtered[[c for c in ["year", "round", "pick", "manager", "player", "yahoo_position", "cost", "draft_grade", "value_tier", "total_fantasy_points", "season_ppg", "manager_spar"] if c in filtered.columns]].to_csv(index=False)
        st.download_button("Download CSV", csv, "draft_summary.csv", "text/csv", key="draft_summary_export_top", use_container_width=True)

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
