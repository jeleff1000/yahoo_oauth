#!/usr/bin/env python3
from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import datetime

from .weekly_player_subprocesses.weekly_player_basic_stats import get_basic_stats
from .weekly_player_subprocesses.weekly_player_advanced_stats import get_advanced_stats
from .weekly_player_subprocesses.weekly_player_matchup_stats import CombinedMatchupStatsViewer
from .base.pagination import PaginationManager
from .weekly_player_subprocesses.head_to_head import H2HViewer
from md.data_access import (
    load_filtered_weekly_data,
    load_players_weekly_data,
    list_player_seasons,
    list_player_weeks,
    list_optimal_seasons,
    list_optimal_weeks,
    load_player_week,
    load_optimal_week,
)


class StreamlitWeeklyPlayerDataViewer:
    def __init__(self, player_data: pd.DataFrame):
        self.initial_data = player_data.copy()
        self.player_data = player_data.copy()

        # Normalize types
        for col in ["year", "week"]:
            if col in self.player_data.columns:
                self.player_data[col] = pd.to_numeric(self.player_data[col], errors="coerce")

        # Pagination managers for each tab
        self.pagination_basic = PaginationManager("weekly_basic")
        self.pagination_advanced = PaginationManager("weekly_advanced")
        self.pagination_matchup = PaginationManager("weekly_matchup")

    # ---------------------- helpers ----------------------

    def get_unique_values(self, column, filters=None):
        filtered_data = self.player_data if not filters else self.apply_filters(filters)
        if column not in filtered_data.columns:
            return []
        series = filtered_data[column].dropna()
        if column in ["year", "week"]:
            try:
                return sorted(series.astype(float).astype(int).unique().tolist())
            except Exception:
                return sorted(series.astype(str).unique().tolist())
        else:
            return sorted(series.astype(str).unique().tolist())

    def apply_filters(self, filters):
        df = self.player_data
        for column, values in (filters or {}).items():
            if values and column in df.columns:
                if column in ["year", "week"]:
                    df = df[df[column].astype("Int64").isin(pd.Series(values, dtype="Int64"))]
                else:
                    df = df[df[column].astype(str).isin([str(v) for v in values])]
        return df

    def has_active_filters(self, filters):
        return any([
            filters.get('player_query'),
            filters.get('manager'),  # list
            filters.get('opp_manager'),  # list
            filters.get('nfl_position'),  # list
            filters.get('fantasy_position'),  # list
            filters.get('nfl_team'),  # list
            filters.get('opponent_nfl_team'),  # list
            filters.get('week'),  # list
            filters.get('year'),  # list (empty if using defaults)
            filters.get('rostered_only') is True,  # only True if toggled on
            filters.get('started_only') is True,  # only True if toggled on
        ])

    def _reset_pagination_on_sort_change(self, key_prefix: str):
        if key_prefix == "basic":
            st.session_state["weekly_basic_offset"] = 0
        elif key_prefix == "advanced":
            st.session_state["weekly_advanced_offset"] = 0
        elif key_prefix == "matchup":
            st.session_state["weekly_matchup_offset"] = 0

    # ---------------------- sort UI ----------------------

    @st.fragment
    def display_sort_controls(self, df: pd.DataFrame, key_prefix: str):
        """Collapsed sort controls: one row per column with ▲ (ASC) / ▼ (DESC)."""
        if df.empty:
            return

        sort_col_key = f"{key_prefix}_sort_col"
        sort_dir_key = f"{key_prefix}_sort_dir"

        # sensible defaults
        if sort_col_key not in st.session_state:
            st.session_state[sort_col_key] = "points" if "points" in df.columns else df.columns[0]
        if sort_dir_key not in st.session_state:
            st.session_state[sort_dir_key] = "DESC"

        with st.expander("Sort options", expanded=False):
            # Header row
            hdr_cat, hdr_up, hdr_down, hdr_curr = st.columns([6, 2, 2, 3])
            with hdr_cat:
                st.markdown("**Category**")
            with hdr_up:
                st.markdown("**Sort Ascend**")
            with hdr_down:
                st.markdown("**Sort Descend**")
            with hdr_curr:
                st.markdown("**Current**")

            for col in df.columns:
                c_cat, c_up, c_down, c_curr = st.columns([6, 2, 2, 3])

                with c_cat:
                    st.write(col)

                with c_up:
                    if st.button("▲", key=f"{key_prefix}_{col}_asc", help=f"Sort {col} ascending"):
                        st.session_state[sort_col_key] = col
                        st.session_state[sort_dir_key] = "ASC"
                        self._reset_pagination_on_sort_change(key_prefix)
                        st.rerun()

                with c_down:
                    if st.button("▼", key=f"{key_prefix}_{col}_desc", help=f"Sort {col} descending"):
                        st.session_state[sort_col_key] = col
                        st.session_state[sort_dir_key] = "DESC"
                        self._reset_pagination_on_sort_change(key_prefix)
                        st.rerun()

                with c_curr:
                    if st.session_state[sort_col_key] == col:
                        arrow = "↑" if st.session_state[sort_dir_key] == "ASC" else "↓"
                        st.write(f"**{arrow} active**")
                    else:
                        st.write("-")

            # Optional: quick clear / default buttons
            cc1, cc2, _ = st.columns([2, 2, 6])
            with cc1:
                if st.button("Reset to Points ↓", key=f"{key_prefix}_reset_points"):
                    if "points" in df.columns:
                        st.session_state[sort_col_key] = "points"
                        st.session_state[sort_dir_key] = "DESC"
                    else:
                        st.session_state[sort_col_key] = df.columns[0]
                        st.session_state[sort_dir_key] = "DESC"
                    self._reset_pagination_on_sort_change(key_prefix)
                    st.rerun()
            with cc2:
                if st.button("Clear (first col ↑)", key=f"{key_prefix}_reset_first"):
                    st.session_state[sort_col_key] = df.columns[0]
                    st.session_state[sort_dir_key] = "ASC"
                    self._reset_pagination_on_sort_change(key_prefix)
                    st.rerun()

    @st.fragment
    def display_sortable_dataframe(self, df: pd.DataFrame, key_prefix: str):
        """
        Show the collapsed sort controls (▲/▼ per column) above the dataframe.
        Sorting state is tracked in session_state under <prefix>_sort_col/_sort_dir.
        """
        if df.empty:
            st.warning("No data available.")
            return

        # draw the collapsed control
        self.display_sort_controls(df, key_prefix)

        # Show the dataframe (no clickable headers)
        st.dataframe(df, hide_index=True, use_container_width=True)

    # ---------------------- main UI ----------------------

    @st.fragment
    def display(self):
        st.title("Weekly Player Data Viewer")
        st.info("Use the **Sort options** expander to sort any column ascending/descending. "
                "Data is fetched from the server with your sort order.")

        tabs = st.tabs(["Basic Stats", "Advanced Stats", "Matchup Stats", "Head-to-Head"])

        @st.fragment
        def display_filters(tab_index):
            selected_filters = {}
            current_year = datetime.now().year
            nfl_order = [
                "QB", "RB", "WR", "TE", "DEF", "K",
                "C", "CB", "DB", "DE", "DL", "DT", "FB", "FS", "G", "ILB", "LB", "LS", "MLB", "NT", "OL", "OLB", "OT", "P", "S", "SAF", ""
            ]
            # NFL Position ordering
            nfl_positions = self.get_unique_values("nfl_position")
            nfl_ordered = [p for p in nfl_order if p in nfl_positions] + [p for p in nfl_positions if p not in nfl_order]
            # Fantasy Position ordering (QB, RB, WR, TE, W/R/T, DEF, K, then others)
            fantasy_order = ["QB", "RB", "WR", "TE", "W/R/T", "DEF", "K", "BN", ""]
            fantasy_positions = self.get_unique_values("fantasy_position")
            fantasy_ordered = [p for p in fantasy_order if p in fantasy_positions] + [p for p in fantasy_positions if p not in fantasy_order]

            # Row 1: Player search, Manager, Opp Manager
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                player_search = st.text_input("Search Player", key=f"player_search_{tab_index}")
            with col2:
                manager_values = st.multiselect(
                    "Manager", self.get_unique_values("manager"),
                    key=f"manager_value_{tab_index}"
                )
            with col3:
                opp_manager_values = st.multiselect(
                    "Opp Manager", self.get_unique_values("opponent"),
                    key=f"opp_manager_value_{tab_index}"
                )

            # Row 2: NFL/Fantasy Position, NFL Team, Opponent NFL Team
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                nfl_position_values = st.multiselect(
                    "NFL Position", nfl_ordered,
                    key=f"nfl_position_value_{tab_index}"
                )
            with col2:
                fantasy_position_values = st.multiselect(
                    "Fantasy Position", fantasy_ordered,
                    key=f"fantasy_position_value_{tab_index}"
                )
            with col3:
                nfl_team_values = st.multiselect(
                    "NFL Team", self.get_unique_values("nfl_team"),
                    key=f"nfl_team_value_{tab_index}"
                )
            with col4:
                opponent_nfl_team_values = st.multiselect(
                    "Opponent NFL Team", self.get_unique_values("opponent_nfl_team"),
                    key=f"opponent_nfl_team_value_{tab_index}"
                )

            # Row 3: Week, Start Year, End Year
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                week_values = st.multiselect(
                    "Week", list(range(1, 22)),
                    key=f"week_value_{tab_index}"
                )
            with col2:
                start_year = st.number_input(
                    "Start Year", min_value=1999, max_value=current_year, value=1999, step=1, key=f"start_year_{tab_index}"
                )
            with col3:
                end_year = st.number_input(
                    "End Year", min_value=1999, max_value=current_year, value=current_year, step=1, key=f"end_year_{tab_index}"
                )

            # Row 4: Rostered Only, Started Only (toggles)
            col1, col2 = st.columns([1, 1])
            with col1:
                rostered_only = st.toggle("Rostered Only", value=False, key=f"show_rostered_{tab_index}")
            with col2:
                started_only = st.toggle("Started Only", value=False, key=f"show_started_{tab_index}")

            selected_filters["player_query"] = (player_search or "").strip() or None
            selected_filters["manager"] = manager_values
            selected_filters["opp_manager"] = opp_manager_values
            selected_filters["nfl_position"] = nfl_position_values
            selected_filters["fantasy_position"] = fantasy_position_values
            selected_filters["nfl_team"] = nfl_team_values
            selected_filters["opponent_nfl_team"] = opponent_nfl_team_values
            selected_filters["week"] = week_values
            # Only include year filter if user changed from defaults (don't filter on full range)
            if start_year != 1999 or end_year != current_year:
                selected_filters["year"] = list(range(int(start_year), int(end_year) + 1)) if start_year <= end_year else []
            else:
                selected_filters["year"] = []
            selected_filters["rostered_only"] = bool(rostered_only)
            selected_filters["started_only"] = bool(started_only)

            return selected_filters

        # -------------------- Basic Stats --------------------
        with tabs[0]:
            st.header("Basic Stats")
            filters = display_filters(tab_index=0)

            # Get current sort state for Basic Stats
            sort_col = st.session_state.get("basic_sort_col", "points")
            sort_dir = st.session_state.get("basic_sort_dir", "DESC")

            if self.has_active_filters(filters):
                with st.spinner("Loading filtered data..."):
                    filtered_data = load_filtered_weekly_data(
                        filters,
                        limit=500,
                        sort_column=sort_col,
                        sort_direction=sort_dir
                    )
                    if filtered_data is not None:
                        # Apply Rostered Only/Started Only filters
                        if filters["rostered_only"] and "rostered" in filtered_data.columns:
                            filtered_data = filtered_data[filtered_data["rostered"] == True]
                        if filters["started_only"] and "started" in filtered_data.columns:
                            filtered_data = filtered_data[filtered_data["started"] == True]
                        basic_stats_df = get_basic_stats(filtered_data, "All")
                        self.display_sortable_dataframe(basic_stats_df, "basic")
            else:
                # Load data with server-side sorting
                offset = st.session_state.get("weekly_basic_offset", 0)
                limit = st.session_state.get("weekly_basic_limit", 100)

                sorted_data = load_players_weekly_data(
                    year=None,
                    week=None,
                    limit=limit,
                    offset=offset,
                    sort_column=sort_col,
                    sort_direction=sort_dir
                )

                if sorted_data is not None:
                    # Apply Rostered Only/Started Only filters
                    if filters["rostered_only"] and "rostered" in sorted_data.columns:
                        sorted_data = sorted_data[sorted_data["rostered"] == True]
                    if filters["started_only"] and "started" in sorted_data.columns:
                        sorted_data = sorted_data[sorted_data["started"] == True]
                    self.pagination_basic.display_controls(sorted_data)
                    basic_stats_df = get_basic_stats(sorted_data, "All")
                    self.display_sortable_dataframe(basic_stats_df, "basic")

        # -------------------- Advanced Stats -----------------
        with tabs[1]:
            st.header("Advanced Stats")
            filters = display_filters(tab_index=1)

            # Get current sort state for Advanced Stats
            sort_col = st.session_state.get("advanced_sort_col", "points")
            sort_dir = st.session_state.get("advanced_sort_dir", "DESC")

            if self.has_active_filters(filters):
                with st.spinner("Loading filtered data..."):
                    filtered_data = load_filtered_weekly_data(
                        filters,
                        limit=500,
                        sort_column=sort_col,
                        sort_direction=sort_dir
                    )
                    if filtered_data is not None:
                        advanced_stats_df = get_advanced_stats(filtered_data)
                        self.display_sortable_dataframe(advanced_stats_df, "advanced")
            else:
                offset = st.session_state.get("weekly_advanced_offset", 0)
                limit = st.session_state.get("weekly_advanced_limit", 100)

                sorted_data = load_players_weekly_data(
                    year=None,
                    week=None,
                    limit=limit,
                    offset=offset,
                    sort_column=sort_col,
                    sort_direction=sort_dir
                )

                if sorted_data is not None:
                    self.pagination_advanced.display_controls(sorted_data)
                    advanced_stats_df = get_advanced_stats(sorted_data)
                    self.display_sortable_dataframe(advanced_stats_df, "advanced")

        # -------------------- Matchup Stats ------------------
        with tabs[2]:
            st.header("Matchup Stats")
            filters = display_filters(tab_index=2)

            # Get current sort state for Matchup Stats
            sort_col = st.session_state.get("matchup_sort_col", "points")
            sort_dir = st.session_state.get("matchup_sort_dir", "DESC")

            if self.has_active_filters(filters):
                with st.spinner("Loading filtered data..."):
                    filtered_data = load_filtered_weekly_data(
                        filters,
                        limit=500,
                        sort_column=sort_col,
                        sort_direction=sort_dir
                    )
                    if filtered_data is not None:
                        viewer = CombinedMatchupStatsViewer(filtered_data)
                        viewer.display(prefix="matchup_stats")
            else:
                offset = st.session_state.get("weekly_matchup_offset", 0)
                limit = st.session_state.get("weekly_matchup_limit", 100)

                sorted_data = load_players_weekly_data(
                    year=None,
                    week=None,
                    limit=limit,
                    offset=offset,
                    sort_column=sort_col,
                    sort_direction=sort_dir
                )

                if sorted_data is not None:
                    self.pagination_matchup.display_controls(sorted_data)
                    viewer = CombinedMatchupStatsViewer(sorted_data)
                    viewer.display(prefix="matchup_stats")

        # -------------------- Head-to-Head / League-Optimal -------------
        with tabs[3]:
            st.header("Head-to-Head")

            # Build the season list from both sources; default to the max available year
            seasons_opt = list_optimal_seasons() or []
            seasons_player = list_player_seasons() or []
            seasons_from_df = self.get_unique_values("year") or []
            all_seasons = sorted(set(map(int, seasons_opt + seasons_player + seasons_from_df)))
            if not all_seasons:
                st.error("No seasons available.")
                st.stop()

            default_year = max(all_seasons)
            sel_year = st.selectbox(
                "Year",
                all_seasons,
                index=all_seasons.index(default_year),
                key="h2h_year",
            )

            # Build the week list for the selected year from both sources; default to max week
            weeks_opt = list_optimal_weeks(int(sel_year)) or []
            weeks_player = list_player_weeks(int(sel_year)) or []

            # Fallback: infer from loaded dataframe if present
            weeks_df = []
            try:
                weeks_df = sorted(
                    set(
                        self.player_data.loc[self.player_data["year"] == int(sel_year), "week"]
                        .dropna()
                        .astype(int)
                        .tolist()
                    )
                )
            except Exception:
                pass

            all_weeks = sorted(set(map(int, weeks_opt + weeks_player + weeks_df)))
            if not all_weeks:
                st.error("No weeks available for selected year.")
                st.stop()

            default_week = max(all_weeks)
            sel_week = st.selectbox(
                "Week",
                all_weeks,
                index=all_weeks.index(default_week),
                key="h2h_week",
            )

            # Load both views for the chosen Year/Week
            optimal_df = load_optimal_week(int(sel_year), int(sel_week))
            player_week_df = load_player_week(int(sel_year), int(sel_week))

            if (optimal_df is None or optimal_df.empty) and (player_week_df is None or player_week_df.empty):
                st.warning("No player rows for that Year/Week.")
                st.stop()

            # Build Matchup dropdown: "Optimal" first, then actual matchups (if any)
            matchup_options = ["Optimal"]
            if player_week_df is not None and not player_week_df.empty:
                if "matchup_name" not in player_week_df.columns:
                    st.error("Column 'matchup_name' is missing in player data.")
                    st.stop()
                matchup_names = (
                    player_week_df["matchup_name"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                matchup_names = sorted(matchup_names)
                matchup_options.extend(matchup_names)

            # Default to "Optimal"
            sel_matchup = st.selectbox(
                "Matchup",
                matchup_options,
                index=0,
                key="h2h_matchup",
                help="Choose 'Optimal' to see the league-wide optimal lineup for this week, or pick a specific matchup.",
            )

            # Render based on selection
            if sel_matchup == "Optimal":
                if optimal_df is None or optimal_df.empty:
                    st.warning("No Optimal lineup available for this Week/Year.")
                else:
                    H2HViewer(optimal_df).display_league_optimal(prefix="h2h")
            else:
                # Standard H2H for the selected matchup
                if player_week_df is None or player_week_df.empty:
                    st.warning("No Head-to-Head data available for this Week/Year.")
                else:
                    H2HViewer(player_week_df).display(prefix="h2h", matchup_name=sel_matchup)
