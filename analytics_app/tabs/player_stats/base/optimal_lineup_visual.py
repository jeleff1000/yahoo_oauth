#!/usr/bin/env python3
"""
Optimal Lineup Visual Display
Shows true optimal lineup with roster constraints (1 QB, 2 RB, 3 WR, 1 TE, 1 W/R/T, 1 K, 1 DEF)
Uses the same efficient season aggregation as Basic Stats tab

OPTIMIZATIONS:
- Theme-aware styling (light/dark mode support)
- Mobile responsive layouts
- Lazy loading for player images
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from md.core import T, run_query
from md.tab_data_access.players import load_players_career_data, load_players_season_data

# Import theme system
from shared.themes import inject_theme_css

DEFAULT_HEADSHOT = "https://static.www.nfl.com/image/private/f_auto,q_auto/league/mdrlzgankwwjldxllgcx"


def _get_top_players_by_position(df: pd.DataFrame, position: str, limit: int, criteria: str, use_season_ppg: bool = True) -> pd.DataFrame:
    """Get top N players by position based on criteria."""
    pos_df = df[df['nfl_position'] == position].copy()

    if pos_df.empty:
        return pd.DataFrame()

    # Map criteria to column name
    sort_column_map = {
        "Total Points": "points",
        "PPG Season": "season_ppg",
        "PPG Career": "ppg_all_time",
        "Max Week": "max_week",
        "Times Optimal": "times_optimal"
    }

    sort_col = sort_column_map.get(criteria, "points")

    # Make sure column exists, fallback to points
    if sort_col not in pos_df.columns:
        sort_col = "points"

    # Sort and get top N
    result = pos_df.nlargest(limit, sort_col).copy()
    result['position'] = position

    return result


def _enrich_with_max_week_and_optimal(df: pd.DataFrame, year_list=None) -> pd.DataFrame:
    """
    Enrich season/career data with max_week and times_optimal from weekly data.
    This queries weekly data efficiently for just the players we have.
    FOR SEASON VIEW: Filters by the specific year for each player
    """
    if df.empty:
        return df
    
    # Check if we have year column (season view) or not (career view)
    has_year = 'year' in df.columns
    
    if has_year:
        # Season view - we need to enrich EACH player-year combination separately
        enriched_rows = []
        
        for _, row in df.iterrows():
            player_key = row.get('player_key', row.get('player', ''))
            year = row.get('year')
            
            if not player_key or not year:
                enriched_rows.append(row)
                continue
            
            # Escape player key for SQL
            escaped_player = str(player_key).replace("'", "''").lower()
            
            # Build player column expression
            if 'player_key' in df.columns:
                player_col = 'COALESCE(LOWER(TRIM(CAST(yahoo_player_id AS VARCHAR))), LOWER(TRIM(CAST(NFL_player_id AS VARCHAR))), LOWER(TRIM(player)))'
            else:
                player_col = 'LOWER(TRIM(player))'
            
            # Query weekly data for THIS SPECIFIC PLAYER AND YEAR
            weekly_stats = run_query(f"""
                SELECT 
                    MAX(points) AS max_week,
                    COUNT(CASE WHEN league_wide_optimal_player = 1 THEN 1 END) AS times_optimal,
                    MAX(headshot_url) AS headshot_url
                FROM {T['player']}
                WHERE year = {int(year)}
                  AND {player_col} = '{escaped_player}'
            """)
            
            if weekly_stats is not None and not weekly_stats.empty:
                row['max_week'] = float(weekly_stats.iloc[0]['max_week']) if pd.notna(weekly_stats.iloc[0]['max_week']) else 0.0
                row['times_optimal'] = int(weekly_stats.iloc[0]['times_optimal']) if pd.notna(weekly_stats.iloc[0]['times_optimal']) else 0
                row['headshot_url'] = str(weekly_stats.iloc[0]['headshot_url']) if pd.notna(weekly_stats.iloc[0]['headshot_url']) else DEFAULT_HEADSHOT
            else:
                row['max_week'] = 0.0
                row['times_optimal'] = 0
                row['headshot_url'] = DEFAULT_HEADSHOT
            
            enriched_rows.append(row)
        
        return pd.DataFrame(enriched_rows)
    
    else:
        # Career view - get totals across all years in the range
        # Build WHERE clause for year filter
        if year_list is None:
            year_clause = "1 = 1"
        elif isinstance(year_list, list) and len(year_list) > 0:
            year_nums = ", ".join(str(int(y)) for y in year_list)
            year_clause = f"year IN ({year_nums})"
        else:
            year_clause = "1 = 1"
        
        # Get player keys to query
        if 'player_key' in df.columns:
            players = df['player_key'].unique().tolist()
            player_col = 'COALESCE(LOWER(TRIM(CAST(yahoo_player_id AS VARCHAR))), LOWER(TRIM(CAST(NFL_player_id AS VARCHAR))), LOWER(TRIM(player)))'
        else:
            players = df['player'].unique().tolist()
            player_col = 'LOWER(TRIM(player))'
        
        if not players:
            df['max_week'] = 0.0
            df['times_optimal'] = 0
            df['headshot_url'] = DEFAULT_HEADSHOT
            return df
        
        # Escape and quote player names for SQL IN clause (limit to 500 to avoid huge queries)
        escaped_players = []
        for p in players[:500]:
            escaped = str(p).replace("'", "''").lower()
            escaped_players.append(f"'{escaped}'")
        players_sql = ", ".join(escaped_players)
        
        # Query weekly data for max week and times optimal across career
        weekly_stats = run_query(f"""
            SELECT 
                {player_col} AS player_key,
                MAX(points) AS max_week,
                COUNT(CASE WHEN league_wide_optimal_player = 1 THEN 1 END) AS times_optimal,
                MAX(headshot_url) AS headshot_url
            FROM {T['player']}
            WHERE {year_clause}
              AND {player_col} IN ({players_sql})
            GROUP BY {player_col}
        """)
        
        if weekly_stats is None or weekly_stats.empty:
            df['max_week'] = 0.0
            df['times_optimal'] = 0
            df['headshot_url'] = DEFAULT_HEADSHOT
            return df
        
        # Merge the weekly stats into our dataframe
        merge_key = 'player_key' if 'player_key' in df.columns else 'player'
        
        # Create a temporary key in weekly_stats that matches our df
        if merge_key == 'player':
            weekly_stats['player'] = weekly_stats['player_key']
        
        df = df.merge(
            weekly_stats[['player_key' if merge_key == 'player_key' else 'player', 'max_week', 'times_optimal', 'headshot_url']],
            left_on=merge_key,
            right_on='player_key' if merge_key == 'player_key' else 'player',
            how='left',
            suffixes=('', '_weekly')
        )
        
        # Fill missing values
        df['max_week'] = df['max_week'].fillna(0.0)
        df['times_optimal'] = df['times_optimal'].fillna(0).astype(int)
        df['headshot_url'] = df['headshot_url'].fillna(DEFAULT_HEADSHOT)
        
        return df


@st.fragment
def display_optimal_season_visual():
    """
    Visual display of optimal season lineup - enforces roster constraints
    Uses efficient season aggregation (same as Basic Stats tab)
    """
    st.subheader("🏆 Optimal Season Lineup")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Optional year filter - default to "All Years"
        year_filter = st.selectbox(
            "Filter by Season",
            options=["All Years"] + [str(y) for y in range(2025, 1998, -1)],
            index=0,
            key="opt_season_year_visual"
        )

    with col2:
        # Selection criteria
        criteria = st.radio(
            "Rank Players By:",
            options=["Total Points", "PPG Season", "Max Week", "Times Optimal"],
            horizontal=True,
            key="opt_season_criteria"
        )

    # Load data using the efficient function
    year_list = None if year_filter == "All Years" else [int(year_filter)]
    df = load_players_season_data(year=year_list)

    if df.empty:
        st.info("No data available.")
        return

    # Enrich with max_week and times_optimal
    with st.spinner("Loading weekly stats..."):
        df = _enrich_with_max_week_and_optimal(df, year_list)

    # Build optimal lineup with roster constraints
    lineup_parts = []

    # 1 QB
    qb = _get_top_players_by_position(df, 'QB', 1, criteria, use_season_ppg=True)
    lineup_parts.append(qb)

    # 2 RB
    rb = _get_top_players_by_position(df, 'RB', 2, criteria, use_season_ppg=True)
    lineup_parts.append(rb)

    # 3 WR
    wr = _get_top_players_by_position(df, 'WR', 3, criteria, use_season_ppg=True)
    lineup_parts.append(wr)

    # 1 TE
    te = _get_top_players_by_position(df, 'TE', 1, criteria, use_season_ppg=True)
    lineup_parts.append(te)

    # 1 FLEX (W/R/T) - exclude players already selected
    selected_players = pd.concat([rb, wr, te], ignore_index=True)
    flex_eligible = df[df['nfl_position'].isin(['WR', 'RB', 'TE'])].copy()

    if not selected_players.empty and 'player_key' in selected_players.columns:
        flex_eligible = flex_eligible[~flex_eligible['player_key'].isin(selected_players['player_key'])]
    elif not selected_players.empty and 'player' in selected_players.columns:
        flex_eligible = flex_eligible[~flex_eligible['player'].isin(selected_players['player'])]

    if not flex_eligible.empty:
        flex = _get_top_players_by_position(flex_eligible, flex_eligible['nfl_position'].iloc[0], 1, criteria, use_season_ppg=True)
        if not flex.empty:
            flex['position'] = 'W/R/T'
        lineup_parts.append(flex)

    # 1 K
    k = _get_top_players_by_position(df, 'K', 1, criteria, use_season_ppg=True)
    lineup_parts.append(k)

    # 1 DEF
    def_team = _get_top_players_by_position(df, 'DEF', 1, criteria, use_season_ppg=True)
    lineup_parts.append(def_team)

    # Combine all parts
    lineup = pd.concat([p for p in lineup_parts if not p.empty], ignore_index=True)

    if lineup.empty:
        st.info("No data available.")
        return

    # Display visual table
    year_label = "All-Time" if year_filter == "All Years" else f"Season {year_filter}"
    _render_optimal_visual_table(lineup, year_label, criteria, use_season_ppg=True)


@st.fragment
def display_optimal_career_visual():
    """
    Visual display of optimal career lineup - enforces roster constraints
    Uses efficient career aggregation across year range
    """
    st.subheader("🏆 Optimal Career Lineup")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        start_year = st.number_input("Start Year", min_value=1999, max_value=2025, value=1999, key="career_opt_start_visual")

    with col2:
        end_year = st.number_input("End Year", min_value=1999, max_value=2025, value=2025, key="career_opt_end_visual")

    with col3:
        # Selection criteria
        criteria = st.radio(
            "Rank Players By:",
            options=["Total Points", "PPG Career", "Max Week", "Times Optimal"],
            horizontal=True,
            key="opt_career_criteria"
        )

    # Load data using the efficient function
    year_list = list(range(int(start_year), int(end_year) + 1))
    df = load_players_career_data(year=year_list)

    if df.empty:
        st.info("No data available for this range.")
        return

    # Enrich with max_week and times_optimal
    with st.spinner("Loading weekly stats..."):
        df = _enrich_with_max_week_and_optimal(df, year_list)

    # Build optimal lineup with roster constraints (same as season)
    lineup_parts = []

    # 1 QB
    qb = _get_top_players_by_position(df, 'QB', 1, criteria, use_season_ppg=False)
    lineup_parts.append(qb)

    # 2 RB
    rb = _get_top_players_by_position(df, 'RB', 2, criteria, use_season_ppg=False)
    lineup_parts.append(rb)

    # 3 WR
    wr = _get_top_players_by_position(df, 'WR', 3, criteria, use_season_ppg=False)
    lineup_parts.append(wr)

    # 1 TE
    te = _get_top_players_by_position(df, 'TE', 1, criteria, use_season_ppg=False)
    lineup_parts.append(te)

    # 1 FLEX (W/R/T) - exclude players already selected
    selected_players = pd.concat([rb, wr, te], ignore_index=True)
    flex_eligible = df[df['nfl_position'].isin(['WR', 'RB', 'TE'])].copy()

    if not selected_players.empty and 'player_key' in selected_players.columns:
        flex_eligible = flex_eligible[~flex_eligible['player_key'].isin(selected_players['player_key'])]
    elif not selected_players.empty and 'player' in selected_players.columns:
        flex_eligible = flex_eligible[~flex_eligible['player'].isin(selected_players['player'])]

    if not flex_eligible.empty:
        flex = _get_top_players_by_position(flex_eligible, flex_eligible['nfl_position'].iloc[0], 1, criteria, use_season_ppg=False)
        if not flex.empty:
            flex['position'] = 'W/R/T'
        lineup_parts.append(flex)

    # 1 K
    k = _get_top_players_by_position(df, 'K', 1, criteria, use_season_ppg=False)
    lineup_parts.append(k)

    # 1 DEF
    def_team = _get_top_players_by_position(df, 'DEF', 1, criteria, use_season_ppg=False)
    lineup_parts.append(def_team)

    # Combine all parts
    lineup = pd.concat([p for p in lineup_parts if not p.empty], ignore_index=True)

    if lineup.empty:
        st.info("No data available for this range.")
        return

    # Display visual table
    _render_optimal_visual_table(lineup, f"{start_year}-{end_year}", criteria, use_season_ppg=False)


@st.fragment
def _render_optimal_visual_table(data: pd.DataFrame, year_label: str, criteria: str, use_season_ppg: bool = True):
    """
    Render optimal lineup in visual style with headshots
    Shows all metrics including max_week and times_optimal
    """
    # Ensure headshot_url column exists
    if 'headshot_url' not in data.columns:
        data['headshot_url'] = DEFAULT_HEADSHOT
    else:
        data['headshot_url'] = data['headshot_url'].fillna(DEFAULT_HEADSHOT)

    # Calculate totals
    total_points = float(data['points'].sum()) if 'points' in data.columns else 0.0
    ppg_col = 'season_ppg' if use_season_ppg else 'ppg_all_time'
    avg_ppg = float(data[ppg_col].mean()) if ppg_col in data.columns else 0.0
    total_max_week = float(data['max_week'].sum()) if 'max_week' in data.columns else 0.0
    total_times_optimal = int(data['times_optimal'].sum()) if 'times_optimal' in data.columns else 0

    # Inject theme-aware CSS (replaces hardcoded styles with theme variables)
    inject_theme_css()

    st.markdown(f"**Optimal Lineup - {year_label}** (Ranked by: *{criteria}*)")

    ppg_label = "PPG Season" if use_season_ppg else "PPG Career"

    # Determine which column to highlight based on criteria
    highlight_map = {
        "Total Points": 2,
        "PPG Season": 3,
        "PPG Career": 3,
        "Max Week": 4,
        "Times Optimal": 5
    }
    highlight_col = highlight_map.get(criteria, -1)

    # Build HTML table
    html = []
    html.append("<table class='optimal-visual'><thead><tr>")
    html.append("<th style='width:8%'>Pos</th>")
    html.append("<th style='width:20%'>Player</th>")

    # Build header with conditional highlighting (extract conditions outside f-string)
    th_class_2 = ' class="highlight-col"' if highlight_col == 2 else ''
    html.append(f"<th style='width:12%'{th_class_2}>Total<br>Points</th>")

    th_class_3 = ' class="highlight-col"' if highlight_col == 3 else ''
    html.append(f"<th style='width:10%'{th_class_3}>{ppg_label}</th>")

    th_class_4 = ' class="highlight-col"' if highlight_col == 4 else ''
    html.append(f"<th style='width:10%'{th_class_4}>Max<br>Week</th>")

    th_class_5 = ' class="highlight-col"' if highlight_col == 5 else ''
    html.append(f"<th style='width:10%'{th_class_5}>Times<br>Optimal</th>")

    html.append("<th style='width:30%'>Manager(s)</th>")
    html.append("</tr></thead><tbody>")

    # Render rows
    for _, row in data.iterrows():
        position = str(row.get('position', row.get('nfl_position', '')))
        headshot = str(row.get('headshot_url', DEFAULT_HEADSHOT))
        if not headshot or headshot.lower() in ['nan', 'none', '']:
            headshot = DEFAULT_HEADSHOT

        player_name = str(row.get('player', 'Unknown'))
        year = int(row.get('year', 0)) if 'year' in row and pd.notna(row.get('year')) else ''
        manager = str(row.get('manager', ''))
        points = float(row.get('points', 0))
        ppg = float(row.get(ppg_col, 0)) if ppg_col in row else 0.0
        max_week = float(row.get('max_week', 0))
        times_optimal = int(row.get('times_optimal', 0))

        html.append("<tr>")
        html.append(f"<td><span class='opt-pos-badge'>{position}</span></td>")

        # Player with photo stacked above name and year - WITH LAZY LOADING
        html.append("<td><div class='opt-player-stack'>")
        html.append(f"<img src='{headshot}' class='opt-player-img' alt='{player_name}' loading='lazy' decoding='async'>")
        html.append(f"<span class='opt-player-name'>{player_name}</span>")
        if year:
            html.append(f"<span class='opt-player-year'>({year})</span>")
        html.append("</div></td>")

        # Build data cells with conditional highlighting (extract conditions outside f-string)
        td_class_2 = ' class="highlight-col"' if highlight_col == 2 else ''
        html.append(f"<td{td_class_2}><div class='opt-stat-cell'>{points:,.1f}</div></td>")

        td_class_3 = ' class="highlight-col"' if highlight_col == 3 else ''
        html.append(f"<td{td_class_3}><div class='opt-stat-cell'>{ppg:.2f}</div></td>")

        td_class_4 = ' class="highlight-col"' if highlight_col == 4 else ''
        html.append(f"<td{td_class_4}><div class='opt-stat-cell'>{max_week:.1f}</div></td>")

        td_class_5 = ' class="highlight-col"' if highlight_col == 5 else ''
        html.append(f"<td{td_class_5}><div class='opt-stat-cell'>{times_optimal}</div></td>")
        html.append(f"<td>{manager}</td>")
        html.append("</tr>")

    # Total row
    html.append("<tr class='total-row'>")
    html.append("<td colspan='2'><strong>LINEUP TOTAL</strong></td>")
    html.append(f"<td><strong>{total_points:,.1f}</strong></td>")
    html.append(f"<td><strong>{avg_ppg:.2f}</strong></td>")
    html.append(f"<td><strong>{total_max_week:.1f}</strong></td>")
    html.append(f"<td><strong>{total_times_optimal}</strong></td>")
    html.append("<td></td>")
    html.append("</tr>")

    html.append("</tbody></table>")

    st.markdown("".join(html), unsafe_allow_html=True)
