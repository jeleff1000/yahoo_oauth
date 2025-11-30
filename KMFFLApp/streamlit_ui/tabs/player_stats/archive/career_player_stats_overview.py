from __future__ import annotations

import pandas as pd
import streamlit as st
from datetime import datetime
from md.data_access import load_players_career_data, run_query
from .base.pagination import PaginationManager
from .career_player_subprocesses.career_player_basic_stats import get_basic_stats as career_basic
from .career_player_subprocesses.career_player_advanced_stats import get_advanced_stats as career_adv
from .career_player_subprocesses.career_player_matchup_stats import CombinedMatchupStatsViewer as CareerMatchups

def _ensure_df(x) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

def _split_unique(series: pd.Series) -> list[str]:
    vals: set[str] = set()
    if series is None:
        return []
    for v in series.dropna().astype(str):
        for part in v.split(","):
            s = part.strip()
            if s:
                vals.add(s)
    return sorted(vals)

def _list_player_positions_fallback() -> list[str]:
    try:
        df = run_query("""
            SELECT DISTINCT nfl_position
            FROM kmffl.player
            WHERE nfl_position IS NOT NULL
            ORDER BY nfl_position
        """)
        if df is not None and not df.empty:
            return df["nfl_position"].dropna().astype(str).tolist()
    except Exception:
        pass
    return []

def _list_player_positions() -> list[str]:
    try:
        from md.data_access import list_player_positions as _real
        positions = _real()
        if isinstance(positions, (list, tuple)):
            return list(positions)
    except Exception:
        pass
    return _list_player_positions_fallback()

class StreamlitCareerPlayerDataViewer:
    """
    Career Player Data Viewer (Aggregated across years).

    - Player/Manager search bars and all selectors filter **before** aggregation.
    - No automatic position filtering; all filtering happens via the UI above the tabs.
    """

    def __init__(self) -> None:
        # Pagination managers for each tab
        self.pagination_basic = PaginationManager("career_basic")
        self.pagination_advanced = PaginationManager("career_advanced")
        self.pagination_matchup = PaginationManager("career_matchup")

    def _first_query(self, player_q: str | None, manager_q: str | None) -> pd.DataFrame:
        return _ensure_df(
            load_players_career_data(
                player_query=(player_q or None),
                manager_query=(manager_q or None),
            )
        )

    def _full_query(
        self,
        player_q: str | None,
        manager_q: str | None,
        sel_manager: list[str] | None,
        sel_nfl_pos: list[str] | None,
        sel_fantasy_pos: list[str] | None,
        sel_team: list[str] | None,
        sel_opponent: list[str] | None,
        sel_opp_team: list[str] | None,
        sel_year: list[int] | None,
        rostered_only: bool = False,
        started_only: bool = False,
    ) -> pd.DataFrame:
        """
        Query career data with all filters applied at the SQL level.
        """
        df = load_players_career_data(
            player_query=(player_q or None),
            manager_query=(manager_q or None),
            manager=(sel_manager or None),
            nfl_team=(sel_team or None),
            opponent=(sel_opponent or None),
            opponent_nfl_team=(sel_opp_team or None),
            year=(sel_year or None),
            rostered_only=rostered_only,
            started_only=started_only,
        )
        df = _ensure_df(df)

        # Apply only frontend-specific filters that can't be done in SQL
        if not df.empty:
            # NFL Position filter (list)
            if sel_nfl_pos and "nfl_position" in df.columns:
                df = df[df["nfl_position"].isin(sel_nfl_pos)]

            # Fantasy Position filter (list) - if the column exists
            if sel_fantasy_pos and "fantasy_position" in df.columns:
                df = df[df["fantasy_position"].isin(sel_fantasy_pos)]

        return df

    def _reset_pagination_on_sort_change(self, key_prefix: str):
        """Reset pagination when sort changes."""
        if key_prefix == "career_basic":
            st.session_state["career_basic_offset"] = 0
        elif key_prefix == "career_advanced":
            st.session_state["career_advanced_offset"] = 0
        elif key_prefix == "career_matchup":
            st.session_state["career_matchup_offset"] = 0

    @st.fragment
    def display_sort_controls(self, df: pd.DataFrame, key_prefix: str):
        """Collapsed sort controls: one row per column with ▲ (ASC) / ▼ (DESC)."""
        if df.empty:
            return

        sort_col_key = f"{key_prefix}_sort_col"
        sort_dir_key = f"{key_prefix}_sort_dir"

        # sensible defaults
        if sort_col_key not in st.session_state:
            st.session_state[sort_col_key] = "total_points" if "total_points" in df.columns else df.columns[0]
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
                    if "total_points" in df.columns:
                        st.session_state[sort_col_key] = "total_points"
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

        # Apply sorting based on session state
        sort_col = st.session_state.get(f"{key_prefix}_sort_col", "total_points")
        sort_dir = st.session_state.get(f"{key_prefix}_sort_dir", "DESC")

        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, ascending=(sort_dir == "ASC"), na_position="last")

        # draw the collapsed control
        self.display_sort_controls(df, key_prefix)

        # Show the dataframe (no clickable headers)
        st.dataframe(df, hide_index=True, use_container_width=True)

    @st.fragment
    def display(self) -> None:
        st.title("Career Player Data Viewer (Aggregated Across Years)")
        st.info("Search bars and selectors filter **source rows before aggregation**. "
                "Use the **Sort options** expander to sort any column ascending/descending.")

        tabs = st.tabs(["Basic Stats", "Advanced Stats", "Matchup Stats", "Optimal Lineup"])

        @st.fragment
        def display_filters(tab_index):
            selected_filters = {}
            current_year = datetime.now().year
            nfl_order = [
                "QB", "RB", "WR", "TE", "DEF", "K",
                "C", "CB", "DB", "DE", "DL", "DT", "FB", "FS", "G", "ILB", "LB", "LS", "MLB", "NT", "OL", "OLB", "OT", "P", "S", "SAF", ""
            ]
            fantasy_order = [
                "QB", "RB", "WR", "TE", "W/R/T", "DEF", "K", "BN", ""
            ]

            # Get initial data for filter options
            base_df = self._first_query(None, None)

            # NFL Position ordering
            nfl_positions = _split_unique(base_df["nfl_position"]) if "nfl_position" in base_df.columns else []
            nfl_ordered = [p for p in nfl_order if p in nfl_positions] + [p for p in nfl_positions if p not in nfl_order]

            # Fantasy Position ordering
            fantasy_positions = _split_unique(base_df["fantasy_position"]) if "fantasy_position" in base_df.columns else []
            fantasy_ordered = [p for p in fantasy_order if p in fantasy_positions] + [p for p in fantasy_positions if p not in fantasy_order]

            # Row 1: Player search, Manager, Opp Manager
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                player_search = st.text_input("Search Player", key=f"career_player_search_{tab_index}")
            with col2:
                manager_values = st.multiselect(
                    "Manager", _split_unique(base_df["manager"]) if "manager" in base_df.columns else [],
                    key=f"career_manager_value_{tab_index}"
                )
            with col3:
                opp_manager_values = st.multiselect(
                    "Opp Manager", _split_unique(base_df["opponent"]) if "opponent" in base_df.columns else [],
                    key=f"career_opp_manager_value_{tab_index}"
                )

            # Row 2: NFL/Fantasy Position, NFL Team, Opponent NFL Team
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                nfl_position_values = st.multiselect(
                    "NFL Position", nfl_ordered,
                    key=f"career_nfl_position_value_{tab_index}"
                )
            with col2:
                fantasy_position_values = st.multiselect(
                    "Fantasy Position", fantasy_ordered,
                    key=f"career_fantasy_position_value_{tab_index}"
                )
            with col3:
                nfl_team_values = st.multiselect(
                    "NFL Team", _split_unique(base_df["nfl_team"]) if "nfl_team" in base_df.columns else [],
                    key=f"career_nfl_team_value_{tab_index}"
                )
            with col4:
                opponent_nfl_team_values = st.multiselect(
                    "Opponent NFL Team", _split_unique(base_df["opponent_nfl_team"]) if "opponent_nfl_team" in base_df.columns else [],
                    key=f"career_opponent_nfl_team_value_{tab_index}"
                )

            # Row 3: Start Year, End Year
            col1, col2 = st.columns([1, 1])
            with col1:
                start_year = st.number_input(
                    "Start Year", min_value=1999, max_value=current_year, value=1999, step=1, key=f"career_start_year_{tab_index}"
                )
            with col2:
                end_year = st.number_input(
                    "End Year", min_value=1999, max_value=current_year, value=current_year, step=1, key=f"career_end_year_{tab_index}"
                )

            # Row 4: Rostered Only, Started Only (toggles)
            col1, col2 = st.columns([1, 1])
            with col1:
                rostered_only = st.toggle("Rostered Only", value=False, key=f"career_show_rostered_{tab_index}")
            with col2:
                started_only = st.toggle("Started Only", value=False, key=f"career_show_started_{tab_index}")

            selected_filters["player_query"] = (player_search or "").strip() or None
            selected_filters["manager"] = manager_values
            selected_filters["opp_manager"] = opp_manager_values
            selected_filters["nfl_position"] = nfl_position_values
            selected_filters["fantasy_position"] = fantasy_position_values
            selected_filters["nfl_team"] = nfl_team_values
            selected_filters["opponent_nfl_team"] = opponent_nfl_team_values
            selected_filters["year"] = list(range(int(start_year), int(end_year) + 1)) if start_year <= end_year else []
            selected_filters["rostered_only"] = bool(rostered_only)
            selected_filters["started_only"] = bool(started_only)

            return selected_filters

        # -------------------- Basic Stats --------------------
        with tabs[0]:
            st.header("Basic Stats")
            filters = display_filters(tab_index=0)

            view_df = self._full_query(
                filters["player_query"],
                None,
                filters["manager"],
                filters["nfl_position"],
                filters["fantasy_position"],
                filters["nfl_team"],
                filters["opp_manager"],
                filters["opponent_nfl_team"],
                filters["year"],
                filters["rostered_only"],
                filters["started_only"],
            )

            if view_df is None or view_df.empty:
                st.warning("No career data matches your filters.")
            else:
                st.info(f"📊 Showing {len(view_df):,} player careers (aggregated across all years)")
                basic_df = career_basic(view_df)
                self.display_sortable_dataframe(basic_df, "career_basic")

        # -------------------- Advanced Stats -----------------
        with tabs[1]:
            st.header("Advanced Stats")
            filters = display_filters(tab_index=1)

            view_df = self._full_query(
                filters["player_query"],
                None,
                filters["manager"],
                filters["nfl_position"],
                filters["fantasy_position"],
                filters["nfl_team"],
                filters["opp_manager"],
                filters["opponent_nfl_team"],
                filters["year"],
                filters["rostered_only"],
                filters["started_only"],
            )

            if view_df is None or view_df.empty:
                st.warning("No career data matches your filters.")
            else:
                st.info(f"📊 Showing {len(view_df):,} player careers (aggregated across all years)")
                adv_df = career_adv(view_df, position="All")
                self.display_sortable_dataframe(adv_df, "career_advanced")

        # -------------------- Matchup Stats ------------------
        with tabs[2]:
            st.header("Matchup Stats")
            filters = display_filters(tab_index=2)

            view_df = self._full_query(
                filters["player_query"],
                None,
                filters["manager"],
                filters["nfl_position"],
                filters["fantasy_position"],
                filters["nfl_team"],
                filters["opp_manager"],
                filters["opponent_nfl_team"],
                filters["year"],
                filters["rostered_only"],
                filters["started_only"],
            )

            if view_df is None or view_df.empty:
                st.warning("No career data matches your filters.")
            else:
                st.info(f"📊 Showing {len(view_df):,} player careers (aggregated across all years)")
                CareerMatchups(view_df).display(prefix="career_matchup", show_per_game=False)

        # Optimal Lineup Tab
        with tabs[3]:
            # Import locally to avoid affecting top-level imports unless tab is used
            from .career_player_subprocesses.optimal_lineup_career import OptimalLineupCareerViewer
            OptimalLineupCareerViewer().display()
