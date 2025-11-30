#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import streamlit as st
from md.data_access import run_query, T


@st.fragment
def display_optimal_season_lineups(year_filter: str = "All"):
    """
    Display optimal lineups for a season with year dropdown.
    Enforces proper roster: 1 QB, 2 RB, 3 WR, 1 TE, 1 DEF, 1 K, 1 W/R/T
    """
    st.subheader("🏆 Optimal Season Lineups")

    # Get available years
    years_df = run_query(f"""
        SELECT DISTINCT year 
        FROM {T['player_season']} 
        WHERE nfl_position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')
        ORDER BY year DESC
    """)

    if years_df.empty:
        st.warning("No optimal lineup data available.")
        return

    years = years_df['year'].tolist()

    # Year selection + View type
    col1, col2 = st.columns([2, 1])

    with col1:
        year_options = ["All"] + [str(y) for y in years]
        selected_year = st.selectbox(
            "Select Year",
            options=year_options,
            index=0,
            key="optimal_season_year"
        )

    with col2:
        view_type = st.radio(
            "View Type",
            ["Total Points", "Head-to-Head"],
            key="optimal_season_view",
            horizontal=True
        )

    if view_type == "Total Points":
        _display_total_points_optimal(selected_year)
    else:
        _display_head_to_head_optimal(selected_year)


@st.fragment
def _display_total_points_optimal(selected_year: str):
    """Show optimal lineup based on total points with proper roster constraints"""

    if selected_year == "All":
        st.markdown("**Best Total Season Score (Any Year)**")
        # Get best season performance for each position across all years
        data = _get_optimal_roster_all_years()
    else:
        st.markdown(f"**Optimal Season Lineup - {selected_year}**")
        data = _get_optimal_roster_single_year(int(selected_year))

    if data.empty:
        st.info("No optimal lineup data for the selected year.")
        return

    _display_optimal_lineup_table(data, selected_year)


@st.fragment
def _display_head_to_head_optimal(selected_year: str):
    """Show head-to-head optimal lineup (best weekly performers)"""

    st.markdown(f"**Head-to-Head Optimal Lineup - {selected_year}**")

    if selected_year == "All":
        st.warning("Please select a specific year for Head-to-Head view.")
        return

    # Get best weekly performers for the selected year
    data = _get_h2h_optimal_roster(int(selected_year))

    if data.empty:
        st.info("No head-to-head optimal data for the selected year.")
        return

    _display_optimal_lineup_table(data, selected_year, is_h2h=True)


def _get_optimal_roster_all_years() -> pd.DataFrame:
    """Get optimal roster across all years with proper position constraints"""

    query = f"""
        WITH season_totals AS (
            SELECT 
                year,
                player,
                nfl_position,
                manager,
                headshot_url,
                total_points,
                games_played as games
            FROM {T['player_season']}
            WHERE nfl_position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')
              AND total_points IS NOT NULL
        ),
        qb AS (
            SELECT *, 'QB' as position, 1 as pos_order
            FROM season_totals WHERE nfl_position = 'QB'
            ORDER BY total_points DESC LIMIT 1
        ),
        rb AS (
            SELECT *, 'RB' as position, 2 as pos_order
            FROM season_totals WHERE nfl_position = 'RB'
            ORDER BY total_points DESC LIMIT 2
        ),
        wr AS (
            SELECT *, 'WR' as position, 3 as pos_order
            FROM season_totals WHERE nfl_position = 'WR'
            ORDER BY total_points DESC LIMIT 3
        ),
        te AS (
            SELECT *, 'TE' as position, 4 as pos_order
            FROM season_totals WHERE nfl_position = 'TE'
            ORDER BY total_points DESC LIMIT 1
        ),
        k AS (
            SELECT *, 'K' as position, 6 as pos_order
            FROM season_totals WHERE nfl_position = 'K'
            ORDER BY total_points DESC LIMIT 1
        ),
        def AS (
            SELECT *, 'DEF' as position, 7 as pos_order
            FROM season_totals WHERE nfl_position = 'DEF'
            ORDER BY total_points DESC LIMIT 1
        ),
        already_selected AS (
            SELECT player FROM qb UNION ALL
            SELECT player FROM rb UNION ALL
            SELECT player FROM wr UNION ALL
            SELECT player FROM te UNION ALL
            SELECT player FROM k UNION ALL
            SELECT player FROM def
        ),
        flex AS (
            SELECT *, 'W/R/T' as position, 5 as pos_order
            FROM season_totals 
            WHERE nfl_position IN ('WR', 'RB', 'TE')
              AND player NOT IN (SELECT player FROM already_selected)
            ORDER BY total_points DESC LIMIT 1
        )
        SELECT position, player, manager, year, total_points, games, headshot_url, pos_order
        FROM (
            SELECT * FROM qb UNION ALL
            SELECT * FROM rb UNION ALL
            SELECT * FROM wr UNION ALL
            SELECT * FROM te UNION ALL
            SELECT * FROM flex UNION ALL
            SELECT * FROM k UNION ALL
            SELECT * FROM def
        )
        ORDER BY pos_order, total_points DESC
    """

    return run_query(query)


def _get_optimal_roster_single_year(year: int) -> pd.DataFrame:
    """Get optimal roster for a specific year with proper position constraints"""

    query = f"""
        WITH season_totals AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                total_points,
                games_played as games,
                ppg
            FROM {T['player_season']}
            WHERE year = {year}
              AND nfl_position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')
              AND total_points IS NOT NULL
        ),
        qb AS (
            SELECT *, 'QB' as position, 1 as pos_order
            FROM season_totals WHERE nfl_position = 'QB'
            ORDER BY total_points DESC LIMIT 1
        ),
        rb AS (
            SELECT *, 'RB' as position, 2 as pos_order
            FROM season_totals WHERE nfl_position = 'RB'
            ORDER BY total_points DESC LIMIT 2
        ),
        wr AS (
            SELECT *, 'WR' as position, 3 as pos_order
            FROM season_totals WHERE nfl_position = 'WR'
            ORDER BY total_points DESC LIMIT 3
        ),
        te AS (
            SELECT *, 'TE' as position, 4 as pos_order
            FROM season_totals WHERE nfl_position = 'TE'
            ORDER BY total_points DESC LIMIT 1
        ),
        k AS (
            SELECT *, 'K' as position, 6 as pos_order
            FROM season_totals WHERE nfl_position = 'K'
            ORDER BY total_points DESC LIMIT 1
        ),
        def AS (
            SELECT *, 'DEF' as position, 7 as pos_order
            FROM season_totals WHERE nfl_position = 'DEF'
            ORDER BY total_points DESC LIMIT 1
        ),
        already_selected AS (
            SELECT player FROM qb UNION ALL
            SELECT player FROM rb UNION ALL
            SELECT player FROM wr UNION ALL
            SELECT player FROM te UNION ALL
            SELECT player FROM k UNION ALL
            SELECT player FROM def
        ),
        flex AS (
            SELECT *, 'W/R/T' as position, 5 as pos_order
            FROM season_totals 
            WHERE nfl_position IN ('WR', 'RB', 'TE')
              AND player NOT IN (SELECT player FROM already_selected)
            ORDER BY total_points DESC LIMIT 1
        )
        SELECT position, player, manager, total_points, games, ppg as avg_points, headshot_url, pos_order
        FROM (
            SELECT * FROM qb UNION ALL
            SELECT * FROM rb UNION ALL
            SELECT * FROM wr UNION ALL
            SELECT * FROM te UNION ALL
            SELECT * FROM flex UNION ALL
            SELECT * FROM k UNION ALL
            SELECT * FROM def
        )
        ORDER BY pos_order, total_points DESC
    """

    return run_query(query)


def _get_h2h_optimal_roster(year: int) -> pd.DataFrame:
    """Get head-to-head optimal roster (best single game performances) with proper position constraints"""

    query = f"""
        WITH weekly_performances AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                week,
                points,
                points as max_points
            FROM {T['player']}
            WHERE year = {year}
              AND league_wide_optimal_player = 1
              AND nfl_position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')
        ),
        best_weeks AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                MAX(points) as max_points,
                COUNT(*) as games
            FROM weekly_performances
            GROUP BY player, nfl_position, manager, headshot_url
        ),
        qb AS (
            SELECT *, 'QB' as position, 1 as pos_order
            FROM best_weeks WHERE nfl_position = 'QB'
            ORDER BY max_points DESC LIMIT 1
        ),
        rb AS (
            SELECT *, 'RB' as position, 2 as pos_order
            FROM best_weeks WHERE nfl_position = 'RB'
            ORDER BY max_points DESC LIMIT 2
        ),
        wr AS (
            SELECT *, 'WR' as position, 3 as pos_order
            FROM best_weeks WHERE nfl_position = 'WR'
            ORDER BY max_points DESC LIMIT 3
        ),
        te AS (
            SELECT *, 'TE' as position, 4 as pos_order
            FROM best_weeks WHERE nfl_position = 'TE'
            ORDER BY max_points DESC LIMIT 1
        ),
        k AS (
            SELECT *, 'K' as position, 6 as pos_order
            FROM best_weeks WHERE nfl_position = 'K'
            ORDER BY max_points DESC LIMIT 1
        ),
        def AS (
            SELECT *, 'DEF' as position, 7 as pos_order
            FROM best_weeks WHERE nfl_position = 'DEF'
            ORDER BY max_points DESC LIMIT 1
        ),
        already_selected AS (
            SELECT player FROM qb UNION ALL
            SELECT player FROM rb UNION ALL
            SELECT player FROM wr UNION ALL
            SELECT player FROM te UNION ALL
            SELECT player FROM k UNION ALL
            SELECT player FROM def
        ),
        flex AS (
            SELECT *, 'W/R/T' as position, 5 as pos_order
            FROM best_weeks 
            WHERE nfl_position IN ('WR', 'RB', 'TE')
              AND player NOT IN (SELECT player FROM already_selected)
            ORDER BY max_points DESC LIMIT 1
        )
        SELECT position, player, manager, max_points as total_points, games, headshot_url, pos_order
        FROM (
            SELECT * FROM qb UNION ALL
            SELECT * FROM rb UNION ALL
            SELECT * FROM wr UNION ALL
            SELECT * FROM te UNION ALL
            SELECT * FROM flex UNION ALL
            SELECT * FROM k UNION ALL
            SELECT * FROM def
        )
        ORDER BY pos_order, total_points DESC
    """

    return run_query(query)


@st.fragment
def display_optimal_career_lineups(start_year: int = 1999, end_year: int = 2025, metric: str = "total_points"):
    """
    Display optimal career lineups with year range and metric selection.
    Enforces proper roster: 1 QB, 2 RB, 3 WR, 1 TE, 1 DEF, 1 K, 1 W/R/T
    """
    st.subheader("🏆 Optimal Career Lineups")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        start = st.number_input("Start Year", min_value=1999, max_value=2025, value=start_year, key="career_opt_start")
    with col2:
        end = st.number_input("End Year", min_value=1999, max_value=2025, value=end_year, key="career_opt_end")
    with col3:
        metric_choice = st.radio(
            "Metric",
            ["Total Points", "PPG All-Time", "Max Single Game"],
            horizontal=False,
            key="career_opt_metric"
        )

    # Map metric choice to column name in the data (not display name)
    if metric_choice == "Total Points":
        data = _get_career_optimal_total_points(start, end)
        metric_col = "total_points"  # actual column name in data
        display_name = "Total Points"  # display name for UI
    elif metric_choice == "PPG All-Time":
        data = _get_career_optimal_ppg(start, end)
        metric_col = "ppg"  # actual column name in data
        display_name = "PPG All-Time"
    else:  # Max Single Game
        data = _get_career_optimal_max_game(start, end)
        metric_col = "max_points"  # actual column name in data
        display_name = "Max Single Game"

    if data.empty:
        st.info("No optimal career data for the selected range.")
        return

    st.markdown(f"**Optimal Career Lineup by {display_name} ({start} - {end})**")
    _display_optimal_career_table(data, metric_col, display_name, start, end)


def _get_career_optimal_total_points(start_year: int, end_year: int) -> pd.DataFrame:
    """Get career optimal based on total points"""

    query = f"""
        WITH career_stats AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                SUM(total_points) as total_points,
                SUM(games_played) as games
            FROM {T['player_season']}
            WHERE year BETWEEN {int(start_year)} AND {int(end_year)}
              AND nfl_position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')
              AND total_points IS NOT NULL
            GROUP BY player, nfl_position, manager, headshot_url
        ),
        qb AS (
            SELECT *, 'QB' as position, 1 as pos_order
            FROM career_stats WHERE nfl_position = 'QB'
            ORDER BY total_points DESC LIMIT 1
        ),
        rb AS (
            SELECT *, 'RB' as position, 2 as pos_order
            FROM career_stats WHERE nfl_position = 'RB'
            ORDER BY total_points DESC LIMIT 2
        ),
        wr AS (
            SELECT *, 'WR' as position, 3 as pos_order
            FROM career_stats WHERE nfl_position = 'WR'
            ORDER BY total_points DESC LIMIT 3
        ),
        te AS (
            SELECT *, 'TE' as position, 4 as pos_order
            FROM career_stats WHERE nfl_position = 'TE'
            ORDER BY total_points DESC LIMIT 1
        ),
        k AS (
            SELECT *, 'K' as position, 6 as pos_order
            FROM career_stats WHERE nfl_position = 'K'
            ORDER BY total_points DESC LIMIT 1
        ),
        def AS (
            SELECT *, 'DEF' as position, 7 as pos_order
            FROM career_stats WHERE nfl_position = 'DEF'
            ORDER BY total_points DESC LIMIT 1
        ),
        already_selected AS (
            SELECT player FROM qb UNION ALL
            SELECT player FROM rb UNION ALL
            SELECT player FROM wr UNION ALL
            SELECT player FROM te UNION ALL
            SELECT player FROM k UNION ALL
            SELECT player FROM def
        ),
        flex AS (
            SELECT *, 'W/R/T' as position, 5 as pos_order
            FROM career_stats 
            WHERE nfl_position IN ('WR', 'RB', 'TE')
              AND player NOT IN (SELECT player FROM already_selected)
            ORDER BY total_points DESC LIMIT 1
        )
        SELECT position, player, manager, total_points, games, headshot_url, pos_order
        FROM (
            SELECT * FROM qb UNION ALL
            SELECT * FROM rb UNION ALL
            SELECT * FROM wr UNION ALL
            SELECT * FROM te UNION ALL
            SELECT * FROM flex UNION ALL
            SELECT * FROM k UNION ALL
            SELECT * FROM def
        )
        ORDER BY pos_order, total_points DESC
    """

    return run_query(query)


def _get_career_optimal_ppg(start_year: int, end_year: int) -> pd.DataFrame:
    """Get career optimal based on PPG (points per game)"""

    query = f"""
        WITH career_stats AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                SUM(total_points) / SUM(games_played) as ppg,
                SUM(games_played) as games,
                SUM(total_points) as total_points
            FROM {T['player_season']}
            WHERE year BETWEEN {int(start_year)} AND {int(end_year)}
              AND nfl_position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')
              AND total_points IS NOT NULL
              AND games_played > 0
            GROUP BY player, nfl_position, manager, headshot_url
            HAVING SUM(games_played) >= 10
        ),
        qb AS (
            SELECT *, 'QB' as position, 1 as pos_order
            FROM career_stats WHERE nfl_position = 'QB'
            ORDER BY ppg DESC LIMIT 1
        ),
        rb AS (
            SELECT *, 'RB' as position, 2 as pos_order
            FROM career_stats WHERE nfl_position = 'RB'
            ORDER BY ppg DESC LIMIT 2
        ),
        wr AS (
            SELECT *, 'WR' as position, 3 as pos_order
            FROM career_stats WHERE nfl_position = 'WR'
            ORDER BY ppg DESC LIMIT 3
        ),
        te AS (
            SELECT *, 'TE' as position, 4 as pos_order
            FROM career_stats WHERE nfl_position = 'TE'
            ORDER BY ppg DESC LIMIT 1
        ),
        k AS (
            SELECT *, 'K' as position, 6 as pos_order
            FROM career_stats WHERE nfl_position = 'K'
            ORDER BY ppg DESC LIMIT 1
        ),
        def AS (
            SELECT *, 'DEF' as position, 7 as pos_order
            FROM career_stats WHERE nfl_position = 'DEF'
            ORDER BY ppg DESC LIMIT 1
        ),
        already_selected AS (
            SELECT player FROM qb UNION ALL
            SELECT player FROM rb UNION ALL
            SELECT player FROM wr UNION ALL
            SELECT player FROM te UNION ALL
            SELECT player FROM k UNION ALL
            SELECT player FROM def
        ),
        flex AS (
            SELECT *, 'W/R/T' as position, 5 as pos_order
            FROM career_stats 
            WHERE nfl_position IN ('WR', 'RB', 'TE')
              AND player NOT IN (SELECT player FROM already_selected)
            ORDER BY ppg DESC LIMIT 1
        )
        SELECT position, player, manager, ppg, total_points, games, headshot_url, pos_order
        FROM (
            SELECT * FROM qb UNION ALL
            SELECT * FROM rb UNION ALL
            SELECT * FROM wr UNION ALL
            SELECT * FROM te UNION ALL
            SELECT * FROM flex UNION ALL
            SELECT * FROM k UNION ALL
            SELECT * FROM def
        )
        ORDER BY pos_order, ppg DESC
    """

    return run_query(query)


def _get_career_optimal_max_game(start_year: int, end_year: int) -> pd.DataFrame:
    """Get career optimal based on max single game points"""

    query = f"""
        WITH career_stats AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                MAX(points) as max_points,
                COUNT(*) as games,
                SUM(points) as total_points
            FROM {T['player']}
            WHERE year BETWEEN {int(start_year)} AND {int(end_year)}
              AND nfl_position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')
            GROUP BY player, nfl_position, manager, headshot_url
        ),
        qb AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                max_points,
                games,
                total_points,
                'QB' as position, 
                1 as pos_order
            FROM career_stats WHERE nfl_position = 'QB'
            ORDER BY max_points DESC LIMIT 1
        ),
        rb AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                max_points,
                games,
                total_points,
                'RB' as position, 
                2 as pos_order
            FROM career_stats WHERE nfl_position = 'RB'
            ORDER BY max_points DESC LIMIT 2
        ),
        wr AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                max_points,
                games,
                total_points,
                'WR' as position, 
                3 as pos_order
            FROM career_stats WHERE nfl_position = 'WR'
            ORDER BY max_points DESC LIMIT 3
        ),
        te AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                max_points,
                games,
                total_points,
                'TE' as position, 
                4 as pos_order
            FROM career_stats WHERE nfl_position = 'TE'
            ORDER BY max_points DESC LIMIT 1
        ),
        k AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                max_points,
                games,
                total_points,
                'K' as position, 
                6 as pos_order
            FROM career_stats WHERE nfl_position = 'K'
            ORDER BY max_points DESC LIMIT 1
        ),
        def AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                max_points,
                games,
                total_points,
                'DEF' as position, 
                7 as pos_order
            FROM career_stats WHERE nfl_position = 'DEF'
            ORDER BY max_points DESC LIMIT 1
        ),
        already_selected AS (
            SELECT player FROM qb UNION ALL
            SELECT player FROM rb UNION ALL
            SELECT player FROM wr UNION ALL
            SELECT player FROM te UNION ALL
            SELECT player FROM k UNION ALL
            SELECT player FROM def
        ),
        flex AS (
            SELECT 
                player,
                nfl_position,
                manager,
                headshot_url,
                max_points,
                games,
                total_points,
                'W/R/T' as position, 
                5 as pos_order
            FROM career_stats 
            WHERE nfl_position IN ('WR', 'RB', 'TE')
              AND player NOT IN (SELECT player FROM already_selected)
            ORDER BY max_points DESC LIMIT 1
        )
        SELECT position, player, manager, max_points, total_points, games, headshot_url, pos_order
        FROM (
            SELECT * FROM qb UNION ALL
            SELECT * FROM rb UNION ALL
            SELECT * FROM wr UNION ALL
            SELECT * FROM te UNION ALL
            SELECT * FROM flex UNION ALL
            SELECT * FROM k UNION ALL
            SELECT * FROM def
        )
        ORDER BY pos_order, max_points DESC
    """

    return run_query(query)


@st.fragment
def _display_optimal_lineup_table(data: pd.DataFrame, year_label: str, is_h2h: bool = False):
    """Render optimal lineup table for season view"""

    # Create styled HTML table
    html = ["<style>"]
    html.append("table.optimal {width: 100%; border-collapse: collapse; margin: 20px 0;}")
    html.append("table.optimal th {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; text-align: left; font-weight: bold;}")
    html.append("table.optimal td {border: 1px solid #ddd; padding: 10px;}")
    html.append("table.optimal tr:nth-child(even) {background-color: #f9f9f9;}")
    html.append("table.optimal tr:hover {background-color: #f0f0f0;}")
    html.append(".pos-badge {display: inline-block; background: #667eea; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; min-width: 50px; text-align: center;}")
    html.append(".points-highlight {color: #2e7d32; font-weight: bold; font-size: 1.1em;}")
    html.append("</style>")

    html.append("<table class='optimal'>")
    html.append("<thead><tr>")
    html.append("<th>Position</th><th>Player</th><th>Manager</th>")

    if 'year' in data.columns:
        html.append("<th>Year</th>")

    html.append("<th>Total Points</th>")

    if 'games' in data.columns:
        html.append("<th>Games</th>")
    if 'avg_points' in data.columns:
        html.append("<th>Avg Points</th>")

    html.append("</tr></thead><tbody>")

    total_points = 0
    for _, row in data.iterrows():
        html.append("<tr>")
        html.append(f"<td><span class='pos-badge'>{row['position']}</span></td>")
        html.append(f"<td><strong>{row['player']}</strong></td>")
        html.append(f"<td>{row.get('manager', '')}</td>")

        if 'year' in data.columns:
            html.append(f"<td>{row['year']}</td>")

        pts = float(row['total_points'])
        total_points += pts
        html.append(f"<td class='points-highlight'>{pts:,.1f}</td>")

        if 'games' in data.columns:
            html.append(f"<td>{int(row['games'])}</td>")
        if 'avg_points' in data.columns:
            html.append(f"<td>{float(row['avg_points']):.1f}</td>")

        html.append("</tr>")

    # Total row
    html.append("<tr style='background: #f0f0f0; font-weight: bold;'>")
    colspan = 3 + (1 if 'year' in data.columns else 0)
    html.append(f"<td colspan='{colspan}' style='text-align: right;'>TOTAL:</td>")
    html.append(f"<td class='points-highlight' style='font-size: 1.2em;'>{total_points:,.1f}</td>")

    if 'games' in data.columns:
        html.append("<td></td>")
    if 'avg_points' in data.columns:
        html.append("<td></td>")

    html.append("</tr>")
    html.append("</tbody></table>")

    st.markdown("".join(html), unsafe_allow_html=True)


@st.fragment
def _display_optimal_career_table(data: pd.DataFrame, metric_col: str, metric_name: str, start_year: int, end_year: int):
    """Render optimal career lineup table"""

    # Create styled HTML table
    html = ["<style>"]
    html.append("table.optimal {width: 100%; border-collapse: collapse; margin: 20px 0;}")
    html.append("table.optimal th {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 12px; text-align: left; font-weight: bold;}")
    html.append("table.optimal td {border: 1px solid #ddd; padding: 10px;}")
    html.append("table.optimal tr:nth-child(even) {background-color: #f9f9f9;}")
    html.append("table.optimal tr:hover {background-color: #f0f0f0;}")
    html.append(".pos-badge {display: inline-block; background: #f5576c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; min-width: 50px; text-align: center;}")
    html.append(".metric-highlight {color: #c62828; font-weight: bold; font-size: 1.1em;}")
    html.append("</style>")

    html.append("<table class='optimal'>")
    html.append("<thead><tr>")
    html.append(f"<th>Position</th><th>Player</th><th>Manager</th><th>{metric_name}</th><th>Games</th><th>Total Points</th>")
    html.append("</tr></thead><tbody>")

    total_metric = 0
    total_games = 0
    total_points = 0

    for _, row in data.iterrows():
        html.append("<tr>")
        html.append(f"<td><span class='pos-badge'>{row['position']}</span></td>")
        html.append(f"<td><strong>{row['player']}</strong></td>")
        html.append(f"<td>{row.get('manager', '')}</td>")

        metric_val = float(row[metric_col])
        total_metric += metric_val
        html.append(f"<td class='metric-highlight'>{metric_val:,.1f}</td>")

        games = int(row['games'])
        total_games += games
        html.append(f"<td>{games}</td>")

        pts = float(row['total_points'])
        total_points += pts
        html.append(f"<td>{pts:,.1f}</td>")

        html.append("</tr>")

    # Total row
    html.append("<tr style='background: #f0f0f0; font-weight: bold;'>")
    html.append("<td colspan='3' style='text-align: right;'>TOTAL:</td>")
    html.append(f"<td class='metric-highlight' style='font-size: 1.2em;'>{total_metric:,.1f}</td>")
    html.append(f"<td>{total_games}</td>")
    html.append(f"<td>{total_points:,.1f}</td>")
    html.append("</tr>")

    html.append("</tbody></table>")

    st.markdown("".join(html), unsafe_allow_html=True)

    st.caption(f"Career stats from {start_year} to {end_year}")

