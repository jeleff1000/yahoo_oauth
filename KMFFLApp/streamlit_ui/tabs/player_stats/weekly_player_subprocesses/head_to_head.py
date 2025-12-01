# tabs/player_stats/head_to_head.py
#!/usr/bin/env python3
from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st


MAIN_POSITIONS = ["QB", "RB", "WR", "TE", "W/R/T", "K", "DEF"]
BENCH_IR = ["BN", "IR"]
ALL_POSITIONS = ["QB", "RB", "WR", "TE", "W/R/T", "K", "DEF", "BN", "IR"]  # Includes bench positions
DEFAULT_HEADSHOT = "https://static.www.nfl.com/image/private/f_auto,q_auto/league/mdrlzgankwwjldxllgcx"


class H2HViewer:
    """
    Player-only H2H viewer.

    Expects all columns to be present in the provided player_df:
      - matchup_name, team_1, team_2
      - player, points, fantasy_position, headshot_url
      - manager, opponent
      - year, week
      - (for league-optimal) league_wide_optimal_player, position
    """

    def __init__(self, player_df: pd.DataFrame):
        self.player_df = player_df.copy()

        # FIX: Use lineup_position (QB1, RB1, WR2, etc.) for H2H display instead of fantasy_position (QB, RB, WR)
        # The data has lineup_position with numbered slots, which is what we need for proper display
        if "lineup_position" in self.player_df.columns:
            # Store original fantasy_position as base_position for reference
            self.player_df["base_position"] = self.player_df["fantasy_position"].astype(str)
            # Use lineup_position as the display position
            self.player_df["fantasy_position"] = self.player_df["lineup_position"].astype(str)

        # Normalize types early
        for col in ("year", "week"):
            if col in self.player_df.columns:
                self.player_df[col] = pd.to_numeric(self.player_df[col], errors="coerce")
        # Safety: fill required string cols to avoid duckdb NULL string issues
        for col in ("player", "fantasy_position", "position", "manager", "opponent", "matchup_name", "team_1", "team_2"):
            if col in self.player_df.columns:
                self.player_df[col] = self.player_df[col].astype(str)

        if "points" in self.player_df.columns:
            self.player_df["points"] = pd.to_numeric(self.player_df["points"], errors="coerce")

        if "headshot_url" in self.player_df.columns:
            self.player_df["headshot_url"] = self.player_df["headshot_url"].fillna("").astype(str)

    # ---- utilities -----------------------------------------------------------

    def get_matchup_names(self) -> list[str]:
        if "matchup_name" not in self.player_df.columns:
            raise KeyError("The required column 'matchup_name' is missing in player data.")

        # Get unique matchup names, filter out invalid ones, and format them
        raw_matchups = self.player_df["matchup_name"].dropna().astype(str).unique().tolist()
        formatted_matchups = []

        for matchup in raw_matchups:
            # Skip "None vs None" and similar invalid matchups
            if matchup.lower() in ['none vs none', 'nan vs nan', 'none', 'nan', '']:
                continue

            # Format matchup: replace "__vs__" with " vs "
            formatted = matchup.replace("__vs__", " vs ")
            formatted_matchups.append(formatted)

        return sorted(formatted_matchups)

    def _prepare_team_duckdb(
        self,
        df: pd.DataFrame,
        manager: str,
        team_col: str,
        positions: list[str],
        filter_col: str = "manager",
    ) -> pd.DataFrame:
        """
        Use DuckDB to add a slot per fantasy_position (ROW_NUMBER()) and
        to order by our custom position order and points desc.
        filter_col: column to filter by (default 'manager')

        Note: positions can be exact matches (e.g., "QB", "RB") or patterns that will match
        positions with numbers (e.g., "QB" matches "QB1", "QB2", etc.)

        For bench positions (BN, IR), we sort by the player's actual NFL position (position column)
        in the order QB, RB, WR, TE, K, DEF while still displaying "BN" or "IR".
        """
        con = duckdb.connect()
        con.register("team_df", df)

        # Check if we have the position column (NFL position) for sorting bench
        has_position_col = "position" in df.columns

        # Build position filter - handle both exact matches and position prefixes
        # For positions like "QB", "RB", we want to match "QB1", "QB2", "RB1", "RB2", etc.
        # For positions like "BN", "IR", we match exactly
        pos_conditions = []
        for pos in positions:
            if pos in ["BN", "IR"]:
                # Exact match for bench/IR
                pos_conditions.append(f"fantasy_position LIKE '{pos}%'")
            else:
                # Pattern match for main positions (QB, RB, WR, etc. match QB1, RB1, WR1, etc.)
                pos_conditions.append(f"fantasy_position LIKE '{pos}%'")

        pos_filter = " OR ".join(pos_conditions)

        # For bench/IR, sort by actual NFL position (QB, RB, WR, TE, K, DEF) then by fantasy_position (BN before IR)
        is_bench = positions == BENCH_IR

        if is_bench and has_position_col:
            # Sort bench by actual position: QB, RB, WR, TE, K, DEF, then BN before IR
            pos_order_case = """
                CASE
                    WHEN position = 'QB' THEN 0
                    WHEN position = 'RB' THEN 1
                    WHEN position = 'WR' THEN 2
                    WHEN position = 'TE' THEN 3
                    WHEN position IN ('W/R/T', 'FLEX') THEN 4
                    WHEN position = 'K' THEN 5
                    WHEN position = 'DEF' THEN 6
                    ELSE 999
                END
            """
            bn_ir_order = """
                CASE
                    WHEN fantasy_position LIKE 'BN%' THEN 0
                    WHEN fantasy_position LIKE 'IR%' THEN 1
                    ELSE 2
                END
            """
            order_clause = f"{pos_order_case}, {bn_ir_order}, points DESC"
        else:
            # For main lineup, sort by position order map
            pos_order_map = {pos: i for i, pos in enumerate(positions)}
            pos_case_parts = []
            for pos, order in pos_order_map.items():
                pos_case_parts.append(f"WHEN fantasy_position LIKE '{pos}%' THEN {order}")
            pos_case = "CASE " + " ".join(pos_case_parts) + " ELSE 999 END"
            order_clause = f"{pos_case}, fantasy_position, points DESC"

        # FIX: Add deduplication to prevent same player appearing multiple times
        # First, get the best row for each player (highest points, then prefer actual fantasy_position)
        # Include position column if available for bench sorting
        position_select = "position," if has_position_col else ""

        q = f"""
            WITH deduplicated AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY player, {filter_col}
                        ORDER BY points DESC,
                                 CASE WHEN fantasy_position ~ '[0-9]$' THEN 0 ELSE 1 END,
                                 fantasy_position
                    ) AS player_rank
                FROM team_df
                WHERE {filter_col} = '{manager.replace("'", "''")}'
                  AND ({pos_filter})
            )
            SELECT
                {team_col},
                fantasy_position,
                {position_select}
                player,
                points,
                COALESCE(NULLIF(headshot_url, ''), '{DEFAULT_HEADSHOT}') AS headshot_url,
                ROW_NUMBER() OVER (PARTITION BY fantasy_position ORDER BY points DESC) - 1 AS slot
            FROM deduplicated
            WHERE player_rank = 1
            ORDER BY {order_clause}
        """
        out = con.execute(q).df()
        con.close()
        return out

    def _merge_two_teams(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """
        Full outer join on (fantasy_position, slot) to align each side by slot.
        Preserves position column (NFL position) if present for color coding bench players.
        """
        # Check if position column exists in either dataframe
        has_position_left = "position" in left.columns
        has_position_right = "position" in right.columns

        l = left.rename(
            columns={
                "player": "player_1",
                "points": "points_1",
                "headshot_url": "headshot_url_1",
                "team_1": "team_1",
            }
        )
        r = right.rename(
            columns={
                "player": "player_2",
                "points": "points_2",
                "headshot_url": "headshot_url_2",
                "team_2": "team_2",
            }
        )

        # Rename position columns separately to preserve them
        if has_position_left:
            l = l.rename(columns={"position": "position_1"})
        if has_position_right:
            r = r.rename(columns={"position": "position_2"})

        on = ["fantasy_position", "slot"]
        merged = pd.merge(l, r, on=on, how="outer", sort=True)

        # Keep team names if present (might be NaN on one side)
        if "team_1" not in merged.columns:
            merged["team_1"] = None
        if "team_2" not in merged.columns:
            merged["team_2"] = None

        # Coalesce position columns to single position column for rendering
        # Use position_1 if available, otherwise position_2
        if has_position_left or has_position_right:
            if has_position_left and has_position_right:
                merged["position"] = merged["position_1"].fillna(merged["position_2"])
            elif has_position_left:
                merged["position"] = merged["position_1"]
            else:
                merged["position"] = merged["position_2"]

        return merged

    def _postprocess_and_totals(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
        df["points_1"] = pd.to_numeric(df.get("points_1", 0), errors="coerce").fillna(0).round(2)
        df["points_2"] = pd.to_numeric(df.get("points_2", 0), errors="coerce").fillna(0).round(2)
        df["margin_1"] = (df["points_1"] - df["points_2"]).round(2)
        df["margin_2"] = -df["margin_1"]

        # Try to read team names from any non-null value
        team_1_name = df.get("team_1")
        team_2_name = df.get("team_2")
        team_1_name = str(team_1_name.dropna().iloc[0]) if team_1_name is not None and team_1_name.notna().any() else "Team 1"
        team_2_name = str(team_2_name.dropna().iloc[0]) if team_2_name is not None and team_2_name.notna().any() else "Team 2"

        # Position sort: extract base position and slot number for proper ordering
        # E.g., "QB1" -> base="QB", slot=1; "W/R/T1" -> base="W/R/T", slot=1
        # For bench (BN/IR), sort by BN first then IR, within each sort by position
        pos_order_map = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "W/R/T": 4, "K": 5, "DEF": 6, "BN": 7, "IR": 8}

        def get_position_order(pos_str):
            """Get sort order for a position string"""
            if pd.isna(pos_str) or str(pos_str).strip() in ["", "nan", "None"]:
                return 999
            return pos_order_map.get(str(pos_str).strip(), 999)

        def extract_position_parts(row):
            """Extract sorting info from position string"""
            pos_str = row.get("fantasy_position", "")
            if pd.isna(pos_str) or pos_str == "":
                return ("", 999, 0)  # (base_pos, slot, bn_ir_order)
            pos_str = str(pos_str).strip()

            # For bench/IR positions, return (actual_position, 0, bn_ir_order)
            # bn_ir_order: 0 for BN, 1 for IR (so BN comes before IR)
            if pos_str.startswith("BN") or pos_str.startswith("IR"):
                bn_ir_order = 0 if pos_str.startswith("BN") else 1
                actual_pos = row.get("position", "")
                if pd.notna(actual_pos) and str(actual_pos).strip() not in ["", "nan", "None"]:
                    return (str(actual_pos).strip(), 0, bn_ir_order)
                return (pos_str, 999, bn_ir_order)

            # Handle special positions like W/R/T
            if pos_str.startswith("W/R/T"):
                base = "W/R/T"
                try:
                    slot = int(pos_str[5:]) if len(pos_str) > 5 else 1
                except:
                    slot = 1
                return (base, slot, 0)

            # Extract base position (letters) and slot number (digits at end)
            import re
            match = re.match(r'^([A-Z]+)(\d*)$', pos_str)
            if match:
                base = match.group(1)
                slot_str = match.group(2)
                slot = int(slot_str) if slot_str else 1
                return (base, slot, 0)

            return (pos_str, 1, 0)

        if not df.empty:
            result = list(df.apply(extract_position_parts, axis=1))
            df["__base_pos"] = [r[0] for r in result]
            df["__slot_num"] = [r[1] for r in result]
            df["__bn_ir_order"] = [r[2] for r in result]

            # For bench, get position order for BOTH teams and use the minimum (earliest position)
            # This ensures rows are sorted by the best position on either side
            df["__pos_order"] = df["__base_pos"].map(pos_order_map).fillna(999).astype(int)

            # Also get position order for the other team's player (position_1 and position_2)
            if "position_1" in df.columns:
                df["__pos_order_1"] = df["position_1"].apply(get_position_order)
            else:
                df["__pos_order_1"] = 999
            if "position_2" in df.columns:
                df["__pos_order_2"] = df["position_2"].apply(get_position_order)
            else:
                df["__pos_order_2"] = 999

            # Use minimum of both position orders for sorting (so QB on either side comes first)
            df["__min_pos_order"] = df[["__pos_order_1", "__pos_order_2"]].min(axis=1)
        else:
            df["__base_pos"] = ""
            df["__slot_num"] = 999
            df["__bn_ir_order"] = 0
            df["__pos_order"] = 999
            df["__min_pos_order"] = 999

        # Sort by: BN/IR order first (BN before IR), then min position order (for bench), then slot, then points
        df = df.sort_values(["__bn_ir_order", "__min_pos_order", "__slot_num", "points_1"], ascending=[True, True, True, False])

        # Clean up temp columns
        cols_to_drop = ["__pos_order", "__base_pos", "__slot_num", "__bn_ir_order", "__min_pos_order"]
        if "__pos_order_1" in df.columns:
            cols_to_drop.append("__pos_order_1")
        if "__pos_order_2" in df.columns:
            cols_to_drop.append("__pos_order_2")
        df = df.drop(columns=cols_to_drop).reset_index(drop=True)

        # Totals row - ensure all required columns are present
        total_row = pd.DataFrame(
            [{
                "player_1": "TOTAL",
                "points_1": round(float(df["points_1"].sum()), 2),
                "fantasy_position": "",
                "points_2": round(float(df["points_2"].sum()), 2),
                "player_2": "TOTAL",
                "headshot_url_1": "",
                "headshot_url_2": "",
                "margin_1": 0,
                "margin_2": 0,
                "slot": 999,
            }]
        )
        df = pd.concat([df, total_row], ignore_index=True)
        return df, team_1_name, team_2_name

    # -------- League-wide Optimal (single-table) ------------------------------

    def display_league_optimal(self, prefix: str):
        """
        Show the league-wide optimal lineup for the already-filtered week slice
        (self.player_df should be filtered to one year/week upstream).
        Uses the *position* column (not fantasy_position) and requires
        'league_wide_optimal_player' (1/True for chosen players).
        """
        dfw = self.player_df.copy()

        if "league_wide_optimal_player" not in dfw.columns:
            st.error("Column 'league_wide_optimal_player' is missing in player data.")
            return

        # Coerce to numeric and filter to flagged rows
        dfw["league_wide_optimal_player"] = pd.to_numeric(
            dfw["league_wide_optimal_player"], errors="coerce"
        ).fillna(0)
        opt = dfw[dfw["league_wide_optimal_player"] > 0].copy()
        if opt.empty:
            st.warning("No league-wide optimal players found for this week.")
            return

        # CRITICAL: Deduplicate players (in case source data has duplicate rows)
        # Keep the row with highest points for each unique player name
        # Use ONLY player name as key since IDs might be inconsistent
        before_dedup = len(opt)

        # Sort by points descending, then drop duplicates keeping first (highest points)
        opt = opt.sort_values("points", ascending=False).drop_duplicates(subset=["player"], keep="first").copy()

        after_dedup = len(opt)

        if before_dedup != after_dedup:
            duplicates_removed = before_dedup - after_dedup
            st.warning(f"⚠️ Removed {duplicates_removed} duplicate player rows. This indicates duplicate data in source file - consider re-running the pipeline.")

        # ---- Use the 'league_wide_optimal_position' column for display & ordering
        # This column shows the roster slot each player fills (QB, RB1, W/R/T, etc.)
        if "league_wide_optimal_position" in opt.columns:
            pos_col = "league_wide_optimal_position"
        elif "position" in opt.columns:
            pos_col = "position"
            st.info("ℹ️ 'league_wide_optimal_position' column not found; falling back to 'position' for ordering/display.")
        else:
            pos_col = "fantasy_position"
            st.info("ℹ️ Neither 'league_wide_optimal_position' nor 'position' found; falling back to 'fantasy_position' for ordering/display.")
        opt[pos_col] = opt[pos_col].astype(str)

        # Normalize key columns
        for c in ("player", "manager", "headshot_url"):
            if c in opt.columns:
                opt[c] = opt[c].astype(str)
        opt["points"] = pd.to_numeric(opt.get("points", 0), errors="coerce").fillna(0)

        # Fill missing headshots
        if "headshot_url" in opt.columns:
            opt["headshot_url"] = opt["headshot_url"].fillna(DEFAULT_HEADSHOT)
            opt["headshot_url"] = opt["headshot_url"].replace("", DEFAULT_HEADSHOT)
        else:
            opt["headshot_url"] = DEFAULT_HEADSHOT

        # Order by *position* (stable) then points DESC
        # Extract base position from numbered positions (e.g., "RB1" -> "RB", "W/R/T" -> "W/R/T")
        def extract_base_position(pos_str):
            """Extract base position from position string like 'QB1', 'WR2', 'W/R/T'"""
            if pd.isna(pos_str) or pos_str == "":
                return ""
            pos_str = str(pos_str).strip()

            # Handle special positions like W/R/T (may have numbers like W/R/T1)
            if pos_str.startswith("W/R/T"):
                return "W/R/T"

            # Extract base position (letters only, remove trailing digits)
            import re
            match = re.match(r'^([A-Z]+)', pos_str)
            if match:
                return match.group(1)

            return pos_str

        opt["__base_pos"] = opt[pos_col].apply(extract_base_position)
        pos_order = {p: i for i, p in enumerate(MAIN_POSITIONS)}
        opt["__pos_order"] = opt["__base_pos"].map(pos_order).fillna(999).astype(int)
        opt = opt.sort_values(["__pos_order", "points"], ascending=[True, False]).drop(columns=["__base_pos"]).reset_index(drop=True)

        # Calculate total
        total_points = opt["points"].sum()

        # Render with spinner
        with st.spinner("Loading optimal lineup..."):
            # Build HTML table with dark/light mode support
            st.markdown(
                """
                <style>
                /* Fixed dark colors that work in both light and dark mode */
                .optimal-table-wrapper {
                    width: 100%;
                    max-width: 1000px;
                    margin: 20px auto;
                    overflow-x: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                }

                table.optimal-visual {
                    width: 100%;
                    border-collapse: collapse;
                    background-color: #1e293b;
                }

                table.optimal-visual th {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 12px 8px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 0.95em;
                    border: none;
                }

                /* Sticky headers for scrolling */
                table.optimal-visual thead {
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }

                table.optimal-visual td {
                    border: 1px solid #334155;
                    padding: 10px 8px;
                    text-align: center;
                    vertical-align: middle;
                    background-color: #1e293b;
                    color: white;
                }

                table.optimal-visual tbody tr:nth-child(even) {
                    background-color: #334155;
                }

                table.optimal-visual tbody tr:nth-child(even) td {
                    background-color: #334155;
                }

                table.optimal-visual tbody tr:hover:not(.total-row) td {
                    background-color: #475569;
                    transition: background-color 0.2s;
                }

                .opt-player-stack {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 6px;
                    min-height: 75px;
                    justify-content: center;
                }

                .opt-player-img {
                    width: 45px;
                    height: 45px;
                    border-radius: 50%;
                    object-fit: cover;
                    border: 2px solid #667eea;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                    flex-shrink: 0;
                    background: #334155;
                    image-rendering: -webkit-optimize-contrast;
                    image-rendering: crisp-edges;
                }

                /* Defense logos need different object-fit to show full logo */
                .opt-player-img.def-logo {
                    object-fit: contain;
                    padding: 3px;
                }

                .opt-player-name {
                    font-weight: 600;
                    font-size: 0.9em;
                    color: white;
                    text-align: center;
                    line-height: 1.2;
                    max-width: 160px;
                    word-wrap: break-word;
                }

                .opt-pos-badge {
                    display: inline-block;
                    color: white;
                    padding: 6px 10px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 0.85em;
                    min-width: 50px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }

                /* Position-specific colors for optimal lineup */
                .opt-pos-badge.pos-QB { background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); }
                .opt-pos-badge.pos-RB { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
                .opt-pos-badge.pos-WR { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); }
                .opt-pos-badge.pos-TE { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); }
                .opt-pos-badge.pos-K { background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%); }
                .opt-pos-badge.pos-DEF { background: linear-gradient(135deg, #6B7280 0%, #4B5563 100%); }
                .opt-pos-badge.pos-FLEX { background: linear-gradient(135deg, #EC4899 0%, #DB2777 100%); }

                .opt-points-cell {
                    color: #4ade80;
                    font-weight: bold;
                    font-size: 1.15em;
                    position: relative;
                }

                /* Points visualization bar */
                .opt-points-bar {
                    position: absolute;
                    left: 0;
                    top: 0;
                    bottom: 0;
                    background: linear-gradient(90deg, rgba(16, 185, 129, 0.25) 0%, transparent 100%);
                    z-index: 0;
                    border-radius: 4px;
                }

                .opt-points-value {
                    position: relative;
                    z-index: 1;
                }

                .total-row {
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
                    color: white !important;
                    font-weight: bold;
                    font-size: 1.15em;
                }

                .total-row td {
                    padding: 12px !important;
                    border-color: #f093fb !important;
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
                }

                /* Mobile responsive styles for optimal table */
                @media (max-width: 768px) {
                    table.optimal-visual {
                        font-size: 0.85em;
                    }
                    table.optimal-visual th,
                    table.optimal-visual td {
                        padding: 8px 4px;
                    }
                    .opt-player-img {
                        width: 38px;
                        height: 38px;
                    }
                    .opt-player-name {
                        font-size: 0.85em;
                        max-width: 120px;
                    }
                    .opt-pos-badge {
                        padding: 5px 8px;
                        font-size: 0.8em;
                        min-width: 45px;
                    }
                    .opt-player-stack {
                        min-height: 65px;
                    }
                }

                @media (max-width: 480px) {
                    table.optimal-visual {
                        font-size: 0.75em;
                    }
                    table.optimal-visual th,
                    table.optimal-visual td {
                        padding: 6px 2px;
                    }
                    .opt-player-img {
                        width: 32px;
                        height: 32px;
                    }
                    .opt-player-name {
                        font-size: 0.75em;
                        max-width: 90px;
                    }
                    .opt-pos-badge {
                        padding: 4px 6px;
                        font-size: 0.7em;
                        min-width: 35px;
                    }
                    .opt-player-stack {
                        min-height: 55px;
                        gap: 4px;
                    }
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            rows = []
            rows.append("<div class='optimal-table-wrapper'>")
            rows.append("<table class='optimal-visual'><thead><tr>")
            rows.append("<th style='width:12%'>Position</th>")
            rows.append("<th style='width:35%'>Player</th>")
            rows.append("<th style='width:12%'>Points</th>")
            rows.append("<th style='width:25%'>Manager</th>")
            rows.append("</tr></thead><tbody>")

            # Calculate max points for visualization bars
            max_points = opt["points"].max() if not opt.empty else 1

            for _, r in opt.iterrows():
                headshot = str(r.get("headshot_url") or DEFAULT_HEADSHOT)
                if not headshot or headshot.lower() == 'nan':
                    headshot = DEFAULT_HEADSHOT
                player = str(r.get("player") or "")
                pts = float(r.get('points', 0))
                pos = str(r.get(pos_col) or "")

                # Extract base position for color coding
                import re
                match = re.match(r'^([A-Z]+(?:/[A-Z]+(?:/[A-Z]+)?)?)', pos)
                if match:
                    base_pos = match.group(1)
                else:
                    base_pos = pos

                # Map to position class
                if 'W/R/T' in base_pos:
                    pos_class = 'pos-FLEX'
                else:
                    pos_class = f'pos-{base_pos}'

                # Add def-logo class for defense team logos
                img_class = "opt-player-img def-logo" if base_pos == 'DEF' else "opt-player-img"

                # Calculate percentage for visualization bar
                bar_pct = (pts / max_points * 100) if max_points > 0 else 0

                # Handle unrostered players (manager is None, NA, empty, or "nan" string)
                # If manager contains "Unrostered" AND other names, show only the other names
                mgr_raw = r.get("manager")
                if pd.isna(mgr_raw) or str(mgr_raw).strip().lower() in ["", "nan", "none", "null"]:
                    mgr = "Unrostered"
                else:
                    mgr = str(mgr_raw)
                    # Filter out "Unrostered" if there are other managers
                    if "," in mgr and "Unrostered" in mgr:
                        managers_list = [m.strip() for m in mgr.split(",") if m.strip().lower() != "unrostered"]
                        if managers_list:
                            mgr = ", ".join(managers_list)
                        else:
                            mgr = "Unrostered"

                rows.append("<tr>")
                rows.append(f"<td><span class='opt-pos-badge {pos_class}'>{pos}</span></td>")
                # Player with photo stacked above name
                rows.append("<td><div class='opt-player-stack'>")
                rows.append(f"<img src='{headshot}' class='{img_class}' alt='{player}' loading='lazy'>")
                rows.append(f"<span class='opt-player-name'>{player}</span>")
                rows.append("</div></td>")
                # Points with visualization bar
                rows.append(f"<td><div class='opt-points-cell'>")
                rows.append(f"<div class='opt-points-bar' style='width:{bar_pct}%'></div>")
                rows.append(f"<span class='opt-points-value'>{pts:,.2f}</span>")
                rows.append("</div></td>")
                rows.append(f"<td>{mgr}</td>")
                rows.append("</tr>")

            # Total row
            rows.append("<tr class='total-row'>")
            rows.append("<td colspan='2'><strong>TOTAL</strong></td>")
            rows.append(f"<td><strong>{total_points:,.2f}</strong></td>")
            rows.append("<td></td>")
            rows.append("</tr>")

            rows.append("</tbody></table></div>")
            st.markdown("".join(rows), unsafe_allow_html=True)

    # ---- public API (H2H) ----------------------------------------------------

    def display(self, prefix: str, matchup_name: str):
        """
        Show H2H table for a single matchup_name within the player_df slice (year/week already filtered upstream).
        Includes radio button to switch between actual and optimal lineups.
        """
        if "matchup_name" not in self.player_df.columns:
            st.error("`matchup_name` column is missing from player data.")
            return

        # Convert formatted matchup name back to raw format for filtering
        raw_matchup_name = matchup_name.replace(" vs ", "__vs__")

        dfm = self.player_df[self.player_df["matchup_name"] == raw_matchup_name].copy()
        if dfm.empty:
            st.warning("No rows for that matchup.")
            return

        # Identify each side from columns present in the player_df
        # team_1 and team_2 ARE the manager names (alphabetically ordered)
        if "team_1" in dfm.columns and "team_2" in dfm.columns:
            team_1 = str(dfm["team_1"].dropna().astype(str).iloc[0]) if dfm["team_1"].notna().any() else None
            team_2 = str(dfm["team_2"].dropna().astype(str).iloc[0]) if dfm["team_2"].notna().any() else None
        else:
            # Fallback: derive names from managers (best-effort)
            mgrs = dfm["manager"].dropna().astype(str).unique().tolist()
            team_1 = mgrs[0] if mgrs else "Team 1"
            team_2 = mgrs[1] if len(mgrs) > 1 else "Team 2"

        # Tabs to switch between actual and optimal lineups
        lineup_tabs = st.tabs(["Actual", "Optimal"])

        # Actual Lineups tab
        with lineup_tabs[0]:
            team_1_main = self._prepare_team_duckdb(dfm, team_1, "team_1", MAIN_POSITIONS)
            team_2_main = self._prepare_team_duckdb(dfm, team_2, "team_2", MAIN_POSITIONS)
            team_1_bench = self._prepare_team_duckdb(dfm, team_1, "team_1", BENCH_IR)
            team_2_bench = self._prepare_team_duckdb(dfm, team_2, "team_2", BENCH_IR)

            main_df = self._merge_two_teams(team_1_main, team_2_main)
            main_df, team_1_name, team_2_name = self._postprocess_and_totals(main_df)
            bench_df = self._merge_two_teams(team_1_bench, team_2_bench)
            bench_df, _, _ = self._postprocess_and_totals(bench_df)

            self.render_table(main_df, team_1_name, team_2_name, color_coding=True)
            if not bench_df.empty and len(bench_df) > 1:
                self.render_table(bench_df, team_1_name, team_2_name, color_coding=False)

        # Optimal Lineups tab
        with lineup_tabs[1]:
            if "optimal_position" not in dfm.columns:
                st.warning("Optimal lineup data not available for this matchup.")
            else:
                opt_check = dfm[dfm["optimal_position"].notna() &
                               ~dfm["optimal_position"].isin(["BN", "IR", ""])]
                if opt_check.empty:
                    st.warning("No optimal lineup positions found for this matchup.")
                else:
                    dfm_optimal = dfm.copy()
                    dfm_optimal["fantasy_position"] = dfm_optimal["optimal_position"]

                    team_1_main = self._prepare_team_duckdb(dfm_optimal, team_1, "team_1", MAIN_POSITIONS)
                    team_2_main = self._prepare_team_duckdb(dfm_optimal, team_2, "team_2", MAIN_POSITIONS)
                    team_1_bench = self._prepare_team_duckdb(dfm_optimal, team_1, "team_1", BENCH_IR)
                    team_2_bench = self._prepare_team_duckdb(dfm_optimal, team_2, "team_2", BENCH_IR)

                    main_df = self._merge_two_teams(team_1_main, team_2_main)
                    main_df, team_1_name, team_2_name = self._postprocess_and_totals(main_df)
                    bench_df = self._merge_two_teams(team_1_bench, team_2_bench)
                    bench_df, _, _ = self._postprocess_and_totals(bench_df)

                    self.render_table(main_df, team_1_name, team_2_name, color_coding=True)
                    if not bench_df.empty and len(bench_df) > 1:
                        self.render_table(bench_df, team_1_name, team_2_name, color_coding=False)

    # ---- rendering -----------------------------------------------------------

    def render_table(self, df: pd.DataFrame, team_1_name: str, team_2_name: str, color_coding: bool):
        st.markdown(
            """
            <style>
            /* Fixed dark colors that work in both light and dark mode - H2H Table */
            .h2h-table-wrapper {
                width: 100%;
                max-width: 1000px;
                margin: 0 auto 20px auto;
                overflow-x: auto;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }

            table.h2h-visual {
                width: 100%;
                border-collapse: collapse;
                background-color: #1e293b;
            }

            table.h2h-visual th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 8px;
                text-align: center;
                font-weight: bold;
                font-size: 0.95em;
                border: none;
            }

            table.h2h-visual td {
                border: 1px solid #334155;
                padding: 10px 8px;
                text-align: center;
                vertical-align: middle;
                background-color: #1e293b;
                color: white;
            }

            table.h2h-visual tbody tr:nth-child(even):not(.h2h-total-row) {
                background-color: #334155;
            }

            table.h2h-visual tbody tr:nth-child(even):not(.h2h-total-row) td {
                background-color: #334155;
            }

            table.h2h-visual tbody tr:hover:not(.h2h-total-row) td {
                background-color: #475569;
                transition: background-color 0.2s;
            }

            .h2h-player-stack {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 6px;
                min-height: 75px;
                justify-content: center;
            }

            .h2h-player-img {
                width: 48px;
                height: 48px;
                border-radius: 50%;
                object-fit: cover;
                border: 2px solid #667eea;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                flex-shrink: 0;
                background: #334155;
                image-rendering: -webkit-optimize-contrast;
            }

            /* Defense logos need different object-fit to show full logo */
            .h2h-player-img.def-logo {
                object-fit: contain;
                padding: 3px;
            }

            .h2h-player-name {
                font-weight: 600;
                font-size: 0.9em;
                color: white;
                text-align: center;
                line-height: 1.2;
                max-width: 160px;
                word-wrap: break-word;
            }

            .h2h-points-cell {
                font-weight: bold;
                font-size: 1.15em;
                color: #1a1a1a;
            }

            .h2h-pos-badge {
                display: inline-block;
                color: white;
                padding: 6px 10px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 0.85em;
                min-width: 50px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }

            /* Position-specific colors for H2H */
            .h2h-pos-badge.pos-QB { background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); }
            .h2h-pos-badge.pos-RB { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
            .h2h-pos-badge.pos-WR { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); }
            .h2h-pos-badge.pos-TE { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); }
            .h2h-pos-badge.pos-K { background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%); }
            .h2h-pos-badge.pos-DEF { background: linear-gradient(135deg, #6B7280 0%, #4B5563 100%); }
            .h2h-pos-badge.pos-FLEX { background: linear-gradient(135deg, #EC4899 0%, #DB2777 100%); }
            .h2h-pos-badge.pos-BN { background: linear-gradient(135deg, #64748B 0%, #475569 100%); }
            .h2h-pos-badge.pos-IR { background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%); }

            .h2h-total-row {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
                color: white !important;
                font-weight: bold;
                font-size: 1.15em;
            }

            .h2h-total-row td {
                padding: 12px !important;
                border-color: #f093fb !important;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
            }

            /* Mobile responsive styles for H2H table */
            @media (max-width: 768px) {
                table.h2h-visual {
                    font-size: 0.85em;
                }
                table.h2h-visual th,
                table.h2h-visual td {
                    padding: 8px 4px;
                }
                .h2h-player-img {
                    width: 40px;
                    height: 40px;
                }
                .h2h-player-name {
                    font-size: 0.85em;
                    max-width: 120px;
                }
                .h2h-pos-badge {
                    padding: 5px 8px;
                    font-size: 0.8em;
                    min-width: 45px;
                }
                .h2h-player-stack {
                    min-height: 65px;
                }
            }

            @media (max-width: 480px) {
                table.h2h-visual {
                    font-size: 0.75em;
                }
                table.h2h-visual th,
                table.h2h-visual td {
                    padding: 6px 2px;
                }
                .h2h-player-img {
                    width: 35px;
                    height: 35px;
                }
                .h2h-player-name {
                    font-size: 0.75em;
                    max-width: 90px;
                }
                .h2h-pos-badge {
                    padding: 4px 6px;
                    font-size: 0.7em;
                    min-width: 35px;
                }
                .h2h-player-stack {
                    min-height: 55px;
                    gap: 4px;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Precompute global margins for color scaling (excluding total row)
        if color_coding and "margin_1" in df.columns and "margin_2" in df.columns:
            df_no_total = df[df.get("player_1", "") != "TOTAL"]
            if not df_no_total.empty:
                m1_min, m1_max = df_no_total["margin_1"].min(skipna=True), df_no_total["margin_1"].max(skipna=True)
                m2_min, m2_max = df_no_total["margin_2"].min(skipna=True), df_no_total["margin_2"].max(skipna=True)
                global_min = min(m for m in [m1_min, m2_min] if pd.notna(m)) if any(pd.notna([m1_min, m2_min])) else 0
                global_max = max(m for m in [m1_max, m2_max] if pd.notna(m)) if any(pd.notna([m1_max, m2_max])) else 1
                rng = max(global_max - global_min, 1)
            else:
                global_min, rng = 0, 1
        else:
            global_min, rng = 0, 1

        def heat(c):
            # green-ish as value increases, lighter near min
            val = max(c - global_min, 0) / rng
            r = 255 - int(180 * val)
            g = 200 + int(55 * val)
            b = 180 - int(130 * val)
            return f"rgb({r},{g},{b})"

        html = []
        html.append("<div class='h2h-table-wrapper'>")
        html.append("<table class='h2h-visual'><thead><tr>")
        # Team 1 name spanning player + points columns
        html.append(f"<th colspan='2' style='width:42%'>{team_1_name}</th>")
        html.append(f"<th style='width:16%'>Position</th>")
        # Team 2 name spanning player + points columns
        html.append(f"<th colspan='2' style='width:42%'>{team_2_name}</th>")
        html.append("</tr></thead><tbody>")

        for _, row in df.iterrows():
            is_total = str(row.get("player_1", "")).strip().upper() == "TOTAL"

            if is_total:
                # Total row - special formatting
                pts1 = row.get('points_1', 0)
                pts2 = row.get('points_2', 0)
                html.append("<tr class='h2h-total-row'>")
                html.append("<td><strong>TOTAL</strong></td>")
                html.append(f"<td><strong>{pts1:,.2f}</strong></td>")
                html.append("<td></td>")
                html.append(f"<td><strong>{pts2:,.2f}</strong></td>")
                html.append("<td><strong>TOTAL</strong></td>")
                html.append("</tr>")
            else:
                # Regular player row
                player1 = str(row.get('player_1', '')).strip() if pd.notna(row.get('player_1')) else ''
                player2 = str(row.get('player_2', '')).strip() if pd.notna(row.get('player_2')) else ''
                headshot_1 = str(row.get('headshot_url_1', '')).strip() if pd.notna(row.get('headshot_url_1')) else ''
                headshot_2 = str(row.get('headshot_url_2', '')).strip() if pd.notna(row.get('headshot_url_2')) else ''
                pts1 = row.get('points_1', 0)
                pts2 = row.get('points_2', 0)
                pos = str(row.get('fantasy_position', '')).strip() if pd.notna(row.get('fantasy_position')) else ''

                # Extract base position and determine display
                # For bench/IR, show "BN" or "IR" but color-code by actual position
                if pos.startswith('BN') or pos.startswith('IR'):
                    is_bench = True
                    pos_display = 'BN' if pos.startswith('BN') else 'IR'

                    # Get actual NFL position for color coding from position column
                    actual_position = str(row.get('position', '')).strip() if pd.notna(row.get('position')) else ''
                    if actual_position and actual_position not in ['nan', '', 'None']:
                        # Use actual position for color coding
                        if actual_position == 'W/R/T' or actual_position == 'FLEX':
                            pos_class = 'pos-FLEX'
                        else:
                            pos_class = f'pos-{actual_position}'
                    else:
                        # Fallback to BN color if no position info
                        pos_class = 'pos-BN' if pos.startswith('BN') else 'pos-IR'
                else:
                    is_bench = False
                    # Extract base position (remove trailing numbers)
                    import re
                    match = re.match(r'^([A-Z]+(?:/[A-Z]+(?:/[A-Z]+)?)?)', pos)
                    if match:
                        pos_display = match.group(1)
                    else:
                        pos_display = pos
                    # Map to position class
                    if 'W/R/T' in pos_display:
                        pos_class = 'pos-FLEX'
                    else:
                        pos_class = f'pos-{pos_display}'

                # Add def-logo class for defense team logos
                img_class = "h2h-player-img def-logo" if pos_display == 'DEF' else "h2h-player-img"

                # Use default headshot if missing
                if not headshot_1 or headshot_1.lower() == 'nan':
                    headshot_1 = DEFAULT_HEADSHOT
                if not headshot_2 or headshot_2.lower() == 'nan':
                    headshot_2 = DEFAULT_HEADSHOT

                # Color coding
                if color_coding:
                    c1 = heat(float(row.get("margin_1", 0) or 0))
                    c2 = heat(float(row.get("margin_2", 0) or 0))
                    text_color = "#1a1a1a"  # Dark text on light backgrounds
                else:
                    c1 = c2 = "transparent"
                    text_color = "white"  # White text on dark backgrounds

                html.append("<tr>")

                # Team 1 player cell: photo on top, name below
                html.append("<td style='width:30%;'>")
                if player1:
                    html.append("<div class='h2h-player-stack'>")
                    html.append(f"<img src='{headshot_1}' class='{img_class}' alt='{player1}' loading='lazy'>")
                    html.append(f"<span class='h2h-player-name'>{player1}</span>")
                    html.append("</div>")
                else:
                    html.append("<div style='min-height:75px;'></div>")
                html.append("</td>")

                # Team 1 points cell
                html.append(f"<td style='background-color:{c1}; width:12%;'><div class='h2h-points-cell' style='color:{text_color};'>{pts1:,.2f}</div></td>")

                # Position badge with color coding
                html.append("<td>")
                if pos:
                    html.append(f"<span class='h2h-pos-badge {pos_class}'>{pos_display}</span>")
                html.append("</td>")

                # Team 2 points cell
                html.append(f"<td style='background-color:{c2}; width:12%;'><div class='h2h-points-cell' style='color:{text_color};'>{pts2:,.2f}</div></td>")

                # Team 2 player cell: photo on top, name below
                html.append("<td style='width:30%;'>")
                if player2:
                    html.append("<div class='h2h-player-stack'>")
                    html.append(f"<img src='{headshot_2}' class='{img_class}' alt='{player2}' loading='lazy'>")
                    html.append(f"<span class='h2h-player-name'>{player2}</span>")
                    html.append("</div>")
                else:
                    html.append("<div style='min-height:75px;'></div>")
                html.append("</td>")

                html.append("</tr>")

        html.append("</tbody></table></div>")
        st.markdown("".join(html), unsafe_allow_html=True)


# ---------- small helpers & standalone viewer --------------------------------

def filter_h2h_data(player_df: pd.DataFrame, year: int, week: int, matchup_name: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("player_data", player_df)
    q = f"""
        SELECT *
        FROM player_data
        WHERE year = {int(year)}
          AND week = {int(week)}
          AND matchup_name = '{matchup_name.replace("'", "''")}'
    """
    out = con.execute(q).df()
    con.close()
    return out


@st.fragment
def display_head_to_head(player_df: pd.DataFrame):
    """
    Optional standalone section if you want to render H2H from a raw player_df.
    """
    if player_df is None or player_df.empty:
        st.write("No player data available.")
        return

    df = player_df.copy()
    for c in ("year", "week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Year/Week from player_df only
    pairs = df.loc[df["year"].notna() & df["week"].notna(), ["year", "week"]].drop_duplicates()
    pairs = pairs.sort_values(["year", "week"]).astype({"year": int, "week": int})
    if pairs.empty:
        st.write("No year/week rows available.")
        return

    years = sorted(pairs["year"].unique().tolist())
    c1, c2, c3, c4 = st.columns([1, 1, 2, 0.6])

    with c1:
        y = st.selectbox("Year", years, index=len(years) - 1, key="h2h_year")
    with c2:
        weeks = sorted(pairs.loc[pairs["year"] == y, "week"].unique().tolist())
        w = st.selectbox("Week", weeks, index=len(weeks) - 1, key="h2h_week")
    with c3:
        matchups = sorted(df.query("year==@y and week==@w")["matchup_name"].dropna().astype(str).unique().tolist())
        m = st.selectbox("Matchup", matchups, index=0 if matchups else None, key="h2h_matchup")
    with c4:
        go = st.button("Go", key="h2h_go")

    if go and m:
        sub = filter_h2h_data(df, y, w, m)
        H2HViewer(sub).display(prefix="h2h", matchup_name=m)
    elif not go:
        st.write("Select a year, week, and matchup, then click Go.")

