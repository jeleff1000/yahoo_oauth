"""
Optimal Lineup view for Season Player Stats
Shows best possible lineups based on different criteria
"""
import streamlit as st
import pandas as pd
import duckdb
from typing import Optional


DEFAULT_HEADSHOT = "https://static.www.nfl.com/image/private/f_auto,q_auto/league/mdrlzgankwwjldxllgcx"


class OptimalLineupSeasonViewer:
    """Display optimal lineups for a season based on various scoring criteria"""

    def __init__(self):
        self.con = duckdb.connect(database=":memory:")

    @st.fragment
    def display(self):
        """Main display method"""
        st.markdown("""
            <div style='background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
                        padding: 1.2rem; border-radius: 10px; margin-bottom: 1.5rem;
                        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.25);'>
                <h3 style='margin: 0; color: white; font-size: 1.5rem;'>
                    🏆 Optimal Lineup Builder
                </h3>
                <p style='margin: 0.3rem 0 0 0; color: rgba(255,255,255,0.95); font-size: 0.9rem;'>
                    Build the best possible lineup based on different scoring criteria
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Year range controls and filters
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            st.markdown("**Start Year**")
            start_year = st.number_input(
                "start_year_label",
                min_value=1999,
                max_value=2024,
                value=1999,
                step=1,
                key="optimal_season_start_year",
                label_visibility="collapsed"
            )

        with col2:
            st.markdown("**End Year**")
            end_year = st.number_input(
                "end_year_label",
                min_value=1999,
                max_value=2024,
                value=2024,
                step=1,
                key="optimal_season_end_year",
                label_visibility="collapsed"
            )

        with col3:
            st.markdown("**Include NFL Playoffs**")
            include_playoffs = st.checkbox(
                "Include NFL Playoffs (POST)",
                value=False,
                key="optimal_season_include_playoffs",
                help="Include games from NFL playoff weeks (season_type = 'POST')"
            )

        # Validate year range
        if start_year > end_year:
            st.warning("⚠️ Start year must be less than or equal to end year")
            return

        # Scoring criteria selection
        st.markdown("### 📊 Select Scoring Criteria")
        scoring_criteria = st.radio(
            "How should we rank players?",
            options=[
                "Total Points",
                "Points Per Game (PPG)",
                "Max Single Game",
                "Times in Optimal Lineup"
            ],
            horizontal=True,
            key="optimal_season_criteria"
        )

        # Load and display optimal lineup
        self._render_optimal_lineup(start_year, end_year, scoring_criteria, include_playoffs)

    def _load_player_data(self, start_year: int, end_year: int, include_playoffs: bool = False) -> Optional[pd.DataFrame]:
        """Load player season data from database

        NOTE: group by NFL_player_id + year (source column `player` holds the display name).
        """
        try:
            from md.core import T, run_query

            # Season type filter
            season_type_filter = "" if include_playoffs else "AND season_type != 'POST'"

            query = f"""
                WITH max_week_data AS (
                    SELECT
                        NFL_player_id,
                        year,
                        position,
                        points,
                        week,
                        manager,
                        ROW_NUMBER() OVER (PARTITION BY NFL_player_id, year, position ORDER BY points DESC) as rn
                    FROM {T['player']}
                    WHERE year >= {start_year}
                      AND year <= {end_year}
                      AND position IS NOT NULL
                      AND position != ''
                      {season_type_filter}
                )
                SELECT
                    p.NFL_player_id,
                    MAX(p.player) as player_name,
                    p.position,
                    p.year,
                    SUM(p.points) as total_points,
                    AVG(p.points) as season_ppg,
                    MAX(p.points) as max_points,
                    SUM(CASE WHEN p.league_wide_optimal_player = 1 THEN 1 ELSE 0 END) as times_optimal,
                    COUNT(DISTINCT p.week) as games_played,
                    COALESCE(MAX(p.headshot_url), '{DEFAULT_HEADSHOT}') as headshot_url,
                    MAX(CASE WHEN mw.rn = 1 THEN mw.week END) as max_week,
                    MAX(CASE WHEN mw.rn = 1 THEN mw.year END) as max_year,
                    STRING_AGG(DISTINCT p.manager, ', ' ORDER BY p.manager) as all_managers,
                    MAX(CASE WHEN mw.rn = 1 THEN mw.manager END) as max_week_manager
                FROM {T['player']} p
                LEFT JOIN max_week_data mw ON p.NFL_player_id = mw.NFL_player_id
                    AND p.year = mw.year
                    AND p.position = mw.position
                    AND p.week = mw.week
                    AND mw.rn = 1
                WHERE p.year >= {start_year}
                  AND p.year <= {end_year}
                  AND p.position IS NOT NULL
                  AND p.position != ''
                  {season_type_filter}
                GROUP BY p.NFL_player_id, p.year, p.position
                HAVING COUNT(DISTINCT p.week) > 0
            """

            df = run_query(query, ttl=300)
            return df

        except Exception as e:
            st.error(f"Error loading player data: {e}")
            return None

    def _build_optimal_lineup(
        self,
        df: pd.DataFrame,
        criteria: str
    ) -> dict:
        """Build optimal lineup based on selected criteria"""

        # Map criteria to column name
        criteria_map = {
            "Total Points": "total_points",
            "Points Per Game (PPG)": "season_ppg",
            "Max Single Game": "max_points",
            "Times in Optimal Lineup": "times_optimal"
        }

        sort_column = criteria_map[criteria]

        # Sort by criteria
        df_sorted = df.sort_values(by=sort_column, ascending=False)

        lineup = {}
        used_players = set()  # track by NFL_player_id to avoid duplicates across rows/years

        # Position requirements
        positions_needed = {
            "QB": 1,
            "RB": 2,
            "WR": 3,
            "TE": 1,
            "DEF": 1,
            "K": 1
        }

        # Fill standard positions
        for pos, count in positions_needed.items():
            pos_players = df_sorted[
                (df_sorted['position'] == pos) &
                (~df_sorted['NFL_player_id'].isin(used_players))
            ].head(count)

            for idx, player in pos_players.iterrows():
                if pos not in lineup:
                    lineup[pos] = []
                lineup[pos].append(player)
                used_players.add(player['NFL_player_id'])

        # Fill W/R/T flex spot with highest remaining WR/RB/TE
        flex_candidates = df_sorted[
            (df_sorted['position'].isin(['WR', 'RB', 'TE'])) &
            (~df_sorted['NFL_player_id'].isin(used_players))
        ].head(1)

        if not flex_candidates.empty:
            flex_player = flex_candidates.iloc[0]
            lineup['W/R/T'] = [flex_player]
            used_players.add(flex_player['NFL_player_id'])

        return lineup

    @st.fragment
    def _render_optimal_lineup(self, start_year: int, end_year: int, criteria: str, include_playoffs: bool = False):
        """Render the optimal lineup display"""

        # Load data
        with st.spinner("Building optimal lineup..."):
            df = self._load_player_data(start_year, end_year, include_playoffs)

        if df is None or df.empty:
            st.info("No player data available for the selected year range")
            return

        # Build lineup
        lineup = self._build_optimal_lineup(df, criteria)

        # Display year range info
        year_text = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"
        st.markdown(f"#### 🎯 Optimal Lineup for {year_text}")
        st.caption(f"Based on: **{criteria}**")

        # Calculate total lineup stats
        total_points = 0
        total_ppg = 0
        total_max = 0
        total_optimal_count = 0
        player_count = 0

        for pos_players in lineup.values():
            for player in pos_players:
                total_points += player['total_points']
                total_ppg += player['season_ppg']
                total_max += player['max_points']
                total_optimal_count += player['times_optimal']
                player_count += 1

        # Show lineup summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Points", f"{total_points:,.1f}")
        with col2:
            st.metric("Avg PPG", f"{total_ppg / player_count:.2f}" if player_count > 0 else "0.0")
        with col3:
            st.metric("Best Week Total", f"{total_max:,.1f}")
        with col4:
            st.metric("Times Optimal", f"{total_optimal_count:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Render lineup as visual formation
        self._render_lineup_visual(lineup, criteria)

    @st.fragment
    def _render_lineup_visual(self, lineup: dict, criteria: str):
        """Render lineup as a table with headshots (like H2H optimal view)"""

        criteria_map = {
            "Total Points": "total_points",
            "Points Per Game (PPG)": "season_ppg",
            "Max Single Game": "max_points",
            "Times in Optimal Lineup": "times_optimal"
        }
        value_column = criteria_map[criteria]

        # CSS styling with fixed dark colors for universal dark/light mode support
        st.markdown(
            """
            <style>
            .optimal-table-wrapper {
                width: 100%;
                max-width: 1000px;
                margin: 20px auto;
                overflow-x: auto;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
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
                border: 1px solid #475569;
                padding: 10px 8px;
                text-align: center;
                vertical-align: middle;
                background-color: #1e293b;
                color: white;
            }

            table.optimal-visual tbody tr:nth-child(even) td {
                background-color: #334155;
            }

            table.optimal-visual tbody tr:hover td {
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

            /* Position-specific colors */
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
                background: linear-gradient(90deg, rgba(74, 222, 128, 0.2) 0%, transparent 100%);
                z-index: 0;
                border-radius: 4px;
            }

            .opt-points-value {
                position: relative;
                z-index: 1;
            }

            /* Tablet responsive styles */
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
                .opt-points-cell {
                    font-size: 1.05em;
                }
                .opt-player-stack {
                    min-height: 65px;
                }
            }

            /* Mobile responsive styles */
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
                .opt-points-cell {
                    font-size: 0.95em;
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

        # Determine which columns to show based on criteria
        criteria_map = {
            "Total Points": "total_points",
            "Points Per Game (PPG)": "season_ppg",
            "Max Single Game": "max_points",
            "Times in Optimal Lineup": "times_optimal"
        }

        # Column headers based on criteria
        if criteria == "Total Points":
            stat_header = "Total Pts"
            show_max_week = False
        elif criteria == "Points Per Game (PPG)":
            stat_header = "PPG"
            show_max_week = False
        elif criteria == "Max Single Game":
            stat_header = "Max Pts"
            show_max_week = True
        else:  # Times in Optimal Lineup
            stat_header = "Times Optimal"
            show_max_week = False

        # Build lineup table with wrapper for proper scrolling
        rows = []
        rows.append("<div class='optimal-table-wrapper'>")
        rows.append("<table class='optimal-visual'><thead><tr>")

        # Fixed column widths that add up to 100% regardless of show_max_week
        if show_max_week:
            rows.append("<th style='width:12%'>Position</th>")
            rows.append("<th style='width:28%'>Player</th>")
            rows.append(f"<th style='width:12%'>{stat_header}</th>")
            rows.append("<th style='width:10%'>Year</th>")
            rows.append("<th style='width:10%'>Games</th>")
            rows.append("<th style='width:16%'>Managers</th>")
            rows.append("<th style='width:12%'>Max Week</th>")
        else:
            rows.append("<th style='width:12%'>Position</th>")
            rows.append("<th style='width:32%'>Player</th>")
            rows.append(f"<th style='width:14%'>{stat_header}</th>")
            rows.append("<th style='width:12%'>Year</th>")
            rows.append("<th style='width:12%'>Games</th>")
            rows.append("<th style='width:18%'>Managers</th>")

        rows.append("</tr></thead><tbody>")

        # Position order
        position_order = ['QB', 'RB', 'WR', 'TE', 'W/R/T', 'K', 'DEF']

        # Calculate max value for visualization bars
        max_value = 0
        for pos_players in lineup.values():
            for player in pos_players:
                val = player.get(value_column, 0)
                if val > max_value:
                    max_value = val

        for pos in position_order:
            if pos in lineup:
                for i, player in enumerate(lineup[pos]):
                    headshot = str(player.get("headshot_url") or DEFAULT_HEADSHOT)
                    if not headshot or headshot.lower() == 'nan' or headshot == '':
                        headshot = DEFAULT_HEADSHOT
                    player_name = str(player.get("player_name") or "")
                    value = player.get(value_column, 0)
                    games = int(player.get('games_played', 0))
                    year = int(player.get('year', 0))
                    max_week = player.get('max_week')
                    max_year = player.get('max_year')

                    # Use appropriate manager field based on criteria
                    if criteria == "Max Single Game":
                        managers = str(player.get('max_week_manager') or "-")
                    else:
                        managers = str(player.get('all_managers') or "-")

                    # Filter out "Unrostered" if there are other managers
                    if "," in managers and "Unrostered" in managers:
                        managers_list = [m.strip() for m in managers.split(",") if m.strip().lower() != "unrostered"]
                        if managers_list:
                            managers = ", ".join(managers_list)
                        else:
                            managers = "Unrostered"

                    # Format value based on criteria
                    if criteria == "Points Per Game (PPG)":
                        value_display = f"{value:.2f}"
                    elif criteria == "Times in Optimal Lineup":
                        value_display = f"{int(value)}"
                    else:
                        value_display = f"{value:.1f}"

                    # Format max week display
                    if pd.notna(max_week) and pd.notna(max_year):
                        max_week_display = f"Wk {int(max_week)} ({int(max_year)})"
                    else:
                        max_week_display = "-"

                    # Position label with slot number for multi-position slots
                    if pos in ['RB', 'WR'] and len(lineup[pos]) > 1:
                        pos_label = f"{pos}{i+1}"
                    else:
                        pos_label = pos if pos != 'W/R/T' else 'FLEX'

                    # Determine position class for color coding
                    base_pos = pos if pos != 'W/R/T' else 'FLEX'
                    pos_class = f"pos-{base_pos}"

                    # Add def-logo class for defense team logos
                    img_class = "opt-player-img def-logo" if pos == 'DEF' else "opt-player-img"

                    # Calculate percentage for visualization bar
                    bar_pct = (value / max_value * 100) if max_value > 0 else 0

                    rows.append("<tr>")
                    rows.append(f"<td><span class='opt-pos-badge {pos_class}'>{pos_label}</span></td>")
                    # Player with photo stacked above name
                    rows.append("<td><div class='opt-player-stack'>")
                    rows.append(f"<img src='{headshot}' class='{img_class}' alt='{player_name}' loading='lazy'>")
                    rows.append(f"<span class='opt-player-name'>{player_name}</span>")
                    rows.append("</div></td>")
                    # Stat value with visualization bar
                    rows.append(f"<td><div class='opt-points-cell'>")
                    rows.append(f"<div class='opt-points-bar' style='width:{bar_pct}%'></div>")
                    rows.append(f"<span class='opt-points-value'>{value_display}</span>")
                    rows.append("</div></td>")
                    rows.append(f"<td>{year}</td>")
                    rows.append(f"<td>{games}</td>")
                    rows.append(f"<td>{managers}</td>")
                    if show_max_week:
                        rows.append(f"<td>{max_week_display}</td>")
                    rows.append("</tr>")

        rows.append("</tbody></table></div>")
        st.markdown("".join(rows), unsafe_allow_html=True)
