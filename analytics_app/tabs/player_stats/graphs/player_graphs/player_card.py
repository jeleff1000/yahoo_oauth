#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from md.core import list_player_seasons
from md.tab_data_access.players import (
    load_players_career_data,
    load_players_season_data,
)
from md.tab_data_access.players.weekly_player_data import (
    load_filtered_weekly_player_data,
)


# Position colors (vintage 1970s inspired)
POSITION_COLORS = {
    "QB": {"primary": "#C41E3A", "secondary": "#8B1A2E", "name": "QUARTERBACK"},
    "RB": {"primary": "#006B3F", "secondary": "#004D2E", "name": "RUNNING BACK"},
    "WR": {"primary": "#FF6B35", "secondary": "#D4551B", "name": "WIDE RECEIVER"},
    "TE": {"primary": "#C41E3A", "secondary": "#8B1A2E", "name": "TIGHT END"},
    "K": {"primary": "#5B2C6F", "secondary": "#3E1D4C", "name": "KICKER"},
    "DEF": {"primary": "#003366", "secondary": "#001A33", "name": "DEFENSE"},
}

# Team logos
TEAM_LOGO_MAP = {
    "ARI": "https://upload.wikimedia.org/wikipedia/en/thumb/7/72/Arizona_Cardinals_logo.svg/179px-Arizona_Cardinals_logo.svg.png",
    "ATL": "https://upload.wikimedia.org/wikipedia/en/thumb/c/c5/Atlanta_Falcons_logo.svg/192px-Atlanta_Falcons_logo.svg.png",
    "BAL": "https://upload.wikimedia.org/wikipedia/en/thumb/1/16/Baltimore_Ravens_logo.svg/193px-Baltimore_Ravens_logo.svg.png",
    "BUF": "https://upload.wikimedia.org/wikipedia/en/thumb/7/77/Buffalo_Bills_logo.svg/189px-Buffalo_Bills_logo.svg.png",
    "CAR": "https://upload.wikimedia.org/wikipedia/en/thumb/1/1c/Carolina_Panthers_logo.svg/100px-Carolina_Panthers_logo.svg.png",
    "CHI": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Chicago_Bears_logo.svg/100px-Chicago_Bears_logo.svg.png",
    "CIN": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Cincinnati_Bengals_logo.svg/100px-Cincinnati_Bengals_logo.svg.png",
    "CLE": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d9/Cleveland_Browns_logo.svg/100px-Cleveland_Browns_logo.svg.png",
    "DAL": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Dallas_Cowboys.svg/100px-Dallas_Cowboys.svg.png",
    "DEN": "https://upload.wikimedia.org/wikipedia/en/thumb/4/44/Denver_Broncos_logo.svg/100px-Denver_Broncos_logo.svg.png",
    "DET": "https://upload.wikimedia.org/wikipedia/en/thumb/7/71/Detroit_Lions_logo.svg/100px-Detroit_Lions_logo.svg.png",
    "GB": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Green_Bay_Packers_logo.svg/100px-Green_Bay_Packers_logo.svg.png",
    "HOU": "https://upload.wikimedia.org/wikipedia/en/thumb/2/28/Houston_Texans_logo.svg/100px-Houston_Texans_logo.svg.png",
    "IND": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Indianapolis_Colts_logo.svg/100px-Indianapolis_Colts_logo.svg.png",
    "JAX": "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/Jacksonville_Jaguars_logo.svg/100px-Jacksonville_Jaguars_logo.svg.png",
    "JAC": "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/Jacksonville_Jaguars_logo.svg/100px-Jacksonville_Jaguars_logo.svg.png",
    "KC": "https://upload.wikimedia.org/wikipedia/en/thumb/e/e1/Kansas_City_Chiefs_logo.svg/100px-Kansas_City_Chiefs_logo.svg.png",
    "LAC": "https://upload.wikimedia.org/wikipedia/en/thumb/7/72/NFL_Chargers_logo.svg/100px-NFL_Chargers_logo.svg.png",
    "LAR": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8a/Los_Angeles_Rams_logo.svg/100px-Los_Angeles_Rams_logo.svg.png",
    "MIA": "https://upload.wikimedia.org/wikipedia/en/thumb/3/37/Miami_Dolphins_logo.svg/100px-Miami_Dolphins_logo.svg.png",
    "MIN": "https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Minnesota_Vikings_logo.svg/98px-Minnesota_Vikings_logo.svg.png",
    "NE": "https://upload.wikimedia.org/wikipedia/en/thumb/b/b9/New_England_Patriots_logo.svg/100px-New_England_Patriots_logo.svg.png",
    "NO": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/New_Orleans_Saints_logo.svg/98px-New_Orleans_Saints_logo.svg.png",
    "NYG": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/New_York_Giants_logo.svg/100px-New_York_Giants_logo.svg.png",
    "NYJ": "https://upload.wikimedia.org/wikipedia/en/thumb/6/6b/New_York_Jets_logo.svg/100px-New_York_Jets_logo.svg.png",
    "LV": "https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Las_Vegas_Raiders_logo.svg/150px-Las_Vegas_Raiders_logo.svg.png",
    "PHI": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/Philadelphia_Eagles_logo.svg/100px-Philadelphia_Eagles_logo.svg.png",
    "PIT": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Pittsburgh_Steelers_logo.svg/100px-Pittsburgh_Steelers_logo.svg.png",
    "SF": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/San_Francisco_49ers_logo.svg/100px-San_Francisco_49ers_logo.svg.png",
    "SEA": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/Seattle_Seahawks_logo.svg/100px-Seattle_Seahawks_logo.svg.png",
    "TB": "https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/Tampa_Bay_Buccaneers_logo.svg/100px-Tampa_Bay_Buccaneers_logo.svg.png",
    "TEN": "https://upload.wikimedia.org/wikipedia/en/thumb/c/c1/Tennessee_Titans_logo.svg/100px-Tennessee_Titans_logo.svg.png",
    "WAS": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Washington_football_team_wlogo.svg/1024px-Washington_football_team_wlogo.svg.png",
    "WSH": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Washington_football_team_wlogo.svg/1024px-Washington_football_team_wlogo.svg.png",
    "STL": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8a/Los_Angeles_Rams_logo.svg/100px-Los_Angeles_Rams_logo.svg.png",
    "SD": "https://upload.wikimedia.org/wikipedia/en/thumb/7/72/NFL_Chargers_logo.svg/100px-NFL_Chargers_logo.svg.png",
    "OAK": "https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Las_Vegas_Raiders_logo.svg/150px-Las_Vegas_Raiders_logo.svg.png",
}


def _safe_int(val, default=0):
    """Safely convert value to int, handling NaN and None."""
    if val is None:
        return default
    try:
        if isinstance(val, float) and pd.isna(val):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0):
    """Safely convert value to float, handling NaN and None."""
    if val is None:
        return default
    try:
        if isinstance(val, float) and pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


@st.fragment
def display_player_card(prefix=""):
    """
    1970s-style football trading card with vintage aesthetic.
    """
    st.header("🏈 Player Card")

    st.markdown(
        """
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Classic football card:</strong> Vintage 1970s-style player card.
    Search for any player to see their card.
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Year selection
    available_years = list_player_seasons()
    if not available_years:
        st.error("No player data found.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox(
            "Select Season",
            options=sorted(available_years, reverse=True),
            key=f"{prefix}_card_year",
        )

    with col2:
        view_mode = st.radio(
            "View Mode",
            ["Season Stats", "Career Stats"],
            horizontal=True,
            key=f"{prefix}_card_mode",
        )

    # Player search
    player_search = st.text_input(
        "🔍 Search for a player:",
        placeholder="e.g., Lamar Jackson",
        key=f"{prefix}_card_search",
    ).strip()

    if not player_search:
        st.info("💡 Enter a player name to generate their card")
        return

    # Load data based on mode
    with st.spinner("Loading player data..."):
        if view_mode == "Season Stats":
            player_data = load_players_season_data(
                year=[int(selected_year)],
                rostered_only=False,
                sort_column="points",
                sort_direction="DESC",
            )
        else:
            player_data = load_players_career_data(
                year=None,
                position=None,
                rostered_only=False,
                started_only=False,
                sort_column="points",
                sort_direction="DESC",
            )

        if player_data is None or player_data.empty:
            st.warning("No data found")
            return

    # Filter by player name
    player_data["player_lower"] = player_data["player"].str.lower()
    search_lower = player_search.lower()

    filtered = player_data[
        player_data["player_lower"].str.contains(search_lower)
    ].copy()
    filtered = filtered.drop(columns=["player_lower"])

    if filtered.empty:
        st.warning(f"No player found matching: {player_search}")
        return

    # If multiple matches, let user select
    if len(filtered) > 1:
        filtered["display"] = (
            filtered["player"] + " (" + filtered["nfl_position"].astype(str) + ")"
        )
        selected_player_display = st.selectbox(
            "Multiple players found - select one:",
            filtered["display"].unique(),
            key=f"{prefix}_card_player_select",
        )
        selected_player_name = selected_player_display.split(" (")[0]
        player_row = filtered[filtered["player"] == selected_player_name].iloc[0]
    else:
        player_row = filtered.iloc[0]

    # Extract player info
    player_name = str(player_row["player"])
    position = str(player_row.get("nfl_position", "N/A"))
    team = str(player_row.get("nfl_team", "N/A"))
    headshot = str(player_row.get("headshot_url", ""))
    if not headshot or headshot == "nan":
        headshot = "https://static.www.nfl.com/image/private/f_auto,q_auto/league/mdrlzgankwwjldxllgcx"

    # Get position colors
    pos_colors = POSITION_COLORS.get(position, POSITION_COLORS["QB"])

    # Get team logo
    team_logo = TEAM_LOGO_MAP.get(team, "")

    # Load weekly data to calculate matchup stats
    matchup_stats = {}
    try:
        filters = {
            "player_query": player_name,
            "rostered_only": False,  # Get all games to find best game
        }
        if view_mode == "Season Stats":
            filters["year"] = [int(selected_year)]

        weekly_data = load_filtered_weekly_player_data(filters, limit=1000)

        if weekly_data is not None and not weekly_data.empty:
            # Filter to only this player's data (in case of partial name matches)
            # Use contains instead of exact match to be more forgiving
            weekly_data = weekly_data[
                weekly_data["player"]
                .str.lower()
                .str.contains(player_name.lower(), regex=False)
            ].copy()

            if not weekly_data.empty:
                # Convert to numeric
                weekly_data["points"] = pd.to_numeric(
                    weekly_data["points"], errors="coerce"
                )
                weekly_data["win"] = pd.to_numeric(weekly_data["win"], errors="coerce")
                # Column is 'is_started' not 'started'
                if "is_started" in weekly_data.columns:
                    weekly_data["is_started"] = pd.to_numeric(
                        weekly_data["is_started"], errors="coerce"
                    )
                elif "started" in weekly_data.columns:
                    weekly_data["is_started"] = pd.to_numeric(
                        weekly_data["started"], errors="coerce"
                    )
                else:
                    weekly_data["is_started"] = (
                        1  # Assume all games count if no started column
                    )
                weekly_data["is_playoffs"] = pd.to_numeric(
                    weekly_data["is_playoffs"], errors="coerce"
                )
                weekly_data["optimal_player"] = pd.to_numeric(
                    weekly_data["optimal_player"], errors="coerce"
                )
                weekly_data["team_points"] = pd.to_numeric(
                    weekly_data["team_points"], errors="coerce"
                )
                weekly_data["opponent_points"] = pd.to_numeric(
                    weekly_data["opponent_points"], errors="coerce"
                )

                # Only count started games for W-L record
                started_games = weekly_data[weekly_data["is_started"] == 1].copy()

                if not started_games.empty:
                    # Overall W-L record
                    wins = int(started_games["win"].sum())
                    losses = len(started_games) - wins
                    matchup_stats["record"] = f"{wins}-{losses}"
                    matchup_stats["win_pct"] = (
                        (wins / len(started_games) * 100)
                        if len(started_games) > 0
                        else 0
                    )

                    # Playoff W-L record
                    playoff_games = started_games[started_games["is_playoffs"] == 1]
                    if not playoff_games.empty:
                        playoff_wins = int(playoff_games["win"].sum())
                        playoff_losses = len(playoff_games) - playoff_wins
                        matchup_stats["playoff_record"] = (
                            f"{playoff_wins}-{playoff_losses}"
                        )
                    else:
                        matchup_stats["playoff_record"] = "0-0"

                    # Average margin when started
                    started_games["margin"] = (
                        started_games["team_points"] - started_games["opponent_points"]
                    )
                    matchup_stats["avg_margin"] = started_games["margin"].mean()
                else:
                    matchup_stats["record"] = "0-0"
                    matchup_stats["win_pct"] = 0
                    matchup_stats["playoff_record"] = "0-0"
                    matchup_stats["avg_margin"] = 0

                # Best game (all games, not just started)
                matchup_stats["best_game"] = weekly_data["points"].max()

                # Championships (rings) - check for championship column
                if "championship" in weekly_data.columns:
                    weekly_data["championship"] = pd.to_numeric(
                        weekly_data["championship"], errors="coerce"
                    )
                    # Championship + win = championship won
                    championship_wins = int(
                        weekly_data[
                            (weekly_data["championship"] == 1)
                            & (weekly_data["win"] == 1)
                        ]["win"].sum()
                    )
                    matchup_stats["rings"] = championship_wins
                else:
                    matchup_stats["rings"] = 0
    except Exception:
        # If matchup stats fail, just continue without them
        pass

    # Calculate stats
    if view_mode == "Season Stats":
        total_points = _safe_float(player_row.get("points", 0))
        ppg = _safe_float(player_row.get("season_ppg", 0))
        games = _safe_int(player_row.get("fantasy_games", 0))
        year_label = f"{selected_year}"

        # Calculate optimal % from season data (only counts actual games played)
        optimal_count = _safe_int(player_row.get("optimal_player", 0))
        optimal_pct = (optimal_count / games * 100) if games > 0 else 0
    else:
        total_points = _safe_float(player_row.get("points", 0))
        ppg = (
            _safe_float(player_row.get("ppg", 0))
            if "ppg" in player_row
            else (
                total_points / _safe_int(player_row.get("games_started", 1))
                if _safe_int(player_row.get("games_started", 0)) > 0
                else 0
            )
        )
        games = _safe_int(player_row.get("games_started", 0))
        year_label = "CAREER"

        # Calculate optimal % from career data (only counts actual games played)
        optimal_count = _safe_int(player_row.get("optimal_player", 0))
        games_played = (
            _safe_int(player_row.get("games_played", 0))
            if "games_played" in player_row
            else games
        )
        optimal_pct = (optimal_count / games_played * 100) if games_played > 0 else 0

    # Build vintage 1970s card HTML - FRONT AND BACK
    card_html = f"""
    <style>
    .cards-wrapper {{
        display: flex;
        flex-wrap: wrap;
        gap: 2rem;
        justify-content: center;
        margin: 2rem auto;
        max-width: 1100px;
        font-family: 'Courier New', monospace;
    }}

    .vintage-card-container {{
        flex: 1;
        min-width: 380px;
        max-width: 450px;
    }}

    .vintage-card {{
        background: #E8D5B7;
        border: 8px solid #8B4513;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        padding: 20px;
        position: relative;
        height: 100%;
        z-index: 1;
    }}

    .vintage-card::before {{
        content: '';
        position: absolute;
        top: 12px;
        left: 12px;
        right: 12px;
        bottom: 12px;
        border: 2px solid #CD853F;
        border-radius: 4px;
        pointer-events: none;
    }}

    .card-label {{
        text-align: center;
        font-size: 0.75rem;
        font-weight: bold;
        color: #5D4037;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .photo-frame {{
        width: 320px;
        height: 240px;
        margin: 0 auto 12px auto;
        position: relative;
        background: {pos_colors['primary']};
        border-radius: 50% / 45%;
        overflow: visible;
        border: 6px solid white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }}

    .photo-frame img.player-photo {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center 20%;
        border-radius: 50% / 55%;
    }}

    .team-logo-corner {{
        position: absolute;
        top: -15px;
        left: -15px;
        width: 60px;
        height: 60px;
        background: white;
        border-radius: 50%;
        padding: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 3px solid {pos_colors['primary']};
        z-index: 10;
    }}

    .team-logo-corner img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
    }}

    .position-badge {{
        position: absolute;
        top: 16px;
        right: 16px;
        background: {pos_colors['secondary']};
        color: white;
        width: 65px;
        height: 65px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1rem;
        border: 4px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        z-index: 10;
    }}

    .team-banner {{
        position: absolute;
        bottom: 20px;
        left: -10px;
        right: -10px;
        background: {pos_colors['primary']};
        color: white;
        text-align: center;
        padding: 8px 20px;
        font-weight: bold;
        font-size: 1.2rem;
        letter-spacing: 2px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        border: 3px solid white;
        transform: skewY(-1deg);
    }}

    .player-name-vintage {{
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        color: #2C1810;
        margin: 12px 0 4px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
    }}

    .year-label {{
        text-align: center;
        font-size: 0.9rem;
        color: #5D4037;
        margin: 0 0 16px 0;
        font-weight: bold;
    }}

    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 16px 0;
    }}

    .stat-box-vintage {{
        background: white;
        border: 3px solid {pos_colors['primary']};
        padding: 12px 8px;
        text-align: center;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }}

    .stat-value-vintage {{
        font-size: 1.6rem;
        font-weight: bold;
        color: {pos_colors['primary']};
        margin: 0;
        line-height: 1;
    }}

    .stat-label-vintage {{
        font-size: 0.7rem;
        color: #5D4037;
        margin: 6px 0 0 0;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .position-stat-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-top: 12px;
        background: rgba(255,255,255,0.5);
        padding: 12px;
        border-radius: 8px;
        border: 2px dashed {pos_colors['primary']};
    }}

    .position-stat {{
        text-align: center;
    }}

    .position-stat-value {{
        font-size: 1.3rem;
        font-weight: bold;
        color: {pos_colors['secondary']};
        margin: 0;
    }}

    .position-stat-label {{
        font-size: 0.65rem;
        color: #5D4037;
        margin: 4px 0 0 0;
        font-weight: bold;
    }}

    .card-footer {{
        text-align: center;
        margin-top: 12px;
        padding: 8px;
        background: rgba(139, 69, 19, 0.2);
        border-radius: 6px;
        font-size: 0.75rem;
        color: #5D4037;
        font-weight: bold;
    }}

    .manager-section {{
        text-align: center;
        margin: 12px 0;
        padding: 8px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 6px;
        border: 2px solid {pos_colors['primary']};
    }}

    .manager-label {{
        font-size: 0.65rem;
        color: #5D4037;
        font-weight: bold;
        text-transform: uppercase;
        margin: 0;
    }}

    .manager-value {{
        font-size: 0.85rem;
        color: {pos_colors['secondary']};
        font-weight: bold;
        margin: 4px 0 0 0;
    }}

    .highlights-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin: 12px 0;
    }}

    .highlight-box {{
        background: rgba(255, 255, 255, 0.7);
        border: 2px solid {pos_colors['primary']};
        padding: 8px 6px;
        text-align: center;
        border-radius: 6px;
    }}

    .highlight-value {{
        font-size: 1.1rem;
        font-weight: bold;
        color: {pos_colors['secondary']};
        margin: 0;
        line-height: 1;
    }}

    .highlight-label {{
        font-size: 0.6rem;
        color: #5D4037;
        margin: 4px 0 0 0;
        font-weight: bold;
        text-transform: uppercase;
    }}

    /* BACK CARD STYLES */
    .back-card-header {{
        text-align: center;
        padding: 12px;
        background: {pos_colors['primary']};
        color: white;
        border-radius: 8px;
        margin-bottom: 16px;
        border: 3px solid white;
    }}

    .back-card-header h3 {{
        margin: 0;
        font-size: 1.4rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .back-card-header p {{
        margin: 4px 0 0 0;
        font-size: 0.8rem;
        opacity: 0.9;
    }}

    .stat-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        margin: 6px 0;
        background: white;
        border-left: 4px solid {pos_colors['primary']};
        border-radius: 4px;
    }}

    .stat-row-label {{
        font-size: 0.75rem;
        color: #5D4037;
        font-weight: bold;
        text-transform: uppercase;
    }}

    .stat-row-value {{
        font-size: 1.1rem;
        color: {pos_colors['secondary']};
        font-weight: bold;
    }}

    .stat-section {{
        margin: 16px 0;
    }}

    .stat-section-title {{
        font-size: 0.8rem;
        color: {pos_colors['primary']};
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 8px;
        padding: 6px;
        background: rgba(255,255,255,0.6);
        border-radius: 4px;
        text-align: center;
    }}

    @media (max-width: 768px) {{
        .cards-wrapper {{
            flex-direction: column;
            align-items: center;
        }}
        .vintage-card-container {{
            max-width: 100%;
        }}
    }}
    </style>

    <div class="cards-wrapper">
        <!-- FRONT CARD -->
        <div class="vintage-card-container">
            <div class="card-label">⚡ FRONT ⚡</div>
            <div class="vintage-card">
                <div class="position-badge">{position}</div>

                <div class="photo-frame">
                    {f'<div class="team-logo-corner"><img src="{team_logo}" alt="{team}"></div>' if team_logo else ''}
                    <img class="player-photo" src="{headshot}" alt="{player_name}" onerror="this.src='https://static.www.nfl.com/image/private/f_auto,q_auto/league/mdrlzgankwwjldxllgcx'">
                    <div class="team-banner">{team}</div>
                </div>

                <div class="player-name-vintage">{player_name}</div>
                <div class="year-label">{year_label}</div>

                <div class="stats-grid">
                    <div class="stat-box-vintage">
                        <p class="stat-value-vintage">{total_points:.0f}</p>
                        <p class="stat-label-vintage">Points</p>
                    </div>
                    <div class="stat-box-vintage">
                        <p class="stat-value-vintage">{ppg:.1f}</p>
                        <p class="stat-label-vintage">PPG</p>
                    </div>
                    <div class="stat-box-vintage">
                        <p class="stat-value-vintage">{games}</p>
                        <p class="stat-label-vintage">Games</p>
                    </div>
                </div>
    """

    # Position-specific stats for FRONT card
    # Column names differ between season and career data
    if position == "QB":
        if view_mode == "Season Stats":
            pass_yds = _safe_int(player_row.get("passing_yards", 0))
            pass_tds = _safe_int(player_row.get("passing_tds", 0))
            pass_ints = _safe_int(player_row.get("passing_interceptions", 0))
        else:  # Career Stats
            pass_yds = _safe_int(player_row.get("pass_yds", 0))
            pass_tds = _safe_int(player_row.get("pass_td", 0))
            pass_ints = _safe_int(player_row.get("passing_interceptions", 0))

        card_html += f"""
            <div class="position-stat-grid">
                <div class="position-stat">
                    <p class="position-stat-value">{pass_yds:,}</p>
                    <p class="position-stat-label">PASS YDS</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{pass_tds}</p>
                    <p class="position-stat-label">PASS TD</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{pass_ints}</p>
                    <p class="position-stat-label">INT</p>
                </div>
            </div>
        """

    elif position in ["RB", "WR", "TE"]:
        if view_mode == "Season Stats":
            rush_yds = _safe_int(player_row.get("rushing_yards", 0))
            rush_tds = _safe_int(player_row.get("rushing_tds", 0))
            rec_yds = _safe_int(player_row.get("receiving_yards", 0))
            rec_tds = _safe_int(player_row.get("receiving_tds", 0))
            receptions = _safe_int(player_row.get("receptions", 0))
        else:  # Career Stats
            rush_yds = _safe_int(player_row.get("rush_yds", 0))
            rush_tds = _safe_int(player_row.get("rush_td", 0))
            rec_yds = _safe_int(player_row.get("rec_yds", 0))
            rec_tds = _safe_int(player_row.get("rec_td", 0))
            receptions = _safe_int(player_row.get("rec", 0))

        if position == "RB":
            card_html += f"""
            <div class="position-stat-grid">
                <div class="position-stat">
                    <p class="position-stat-value">{rush_yds:,}</p>
                    <p class="position-stat-label">RUSH YDS</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{rush_tds}</p>
                    <p class="position-stat-label">RUSH TD</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{receptions}</p>
                    <p class="position-stat-label">REC</p>
                </div>
            </div>
            """
        else:  # WR/TE
            card_html += f"""
            <div class="position-stat-grid">
                <div class="position-stat">
                    <p class="position-stat-value">{receptions}</p>
                    <p class="position-stat-label">REC</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{rec_yds:,}</p>
                    <p class="position-stat-label">REC YDS</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{rec_tds}</p>
                    <p class="position-stat-label">REC TD</p>
                </div>
            </div>
            """

    elif position == "K":
        fgm = _safe_int(player_row.get("fg_made", 0))
        fga = _safe_int(player_row.get("fg_att", 0))
        xpm = _safe_int(player_row.get("pat_made", 0))

        card_html += f"""
            <div class="position-stat-grid">
                <div class="position-stat">
                    <p class="position-stat-value">{fgm}/{fga}</p>
                    <p class="position-stat-label">FG M/A</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{xpm}</p>
                    <p class="position-stat-label">XPM</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{(fgm/fga*100) if fga > 0 else 0:.0f}%</p>
                    <p class="position-stat-label">FG PCT</p>
                </div>
            </div>
        """

    elif position == "DEF":
        sacks = _safe_int(player_row.get("def_sacks", 0))
        ints = _safe_int(player_row.get("def_interceptions", 0))
        def_tds = _safe_int(player_row.get("def_tds", 0))

        card_html += f"""
            <div class="position-stat-grid">
                <div class="position-stat">
                    <p class="position-stat-value">{sacks}</p>
                    <p class="position-stat-label">SACKS</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{ints}</p>
                    <p class="position-stat-label">INT</p>
                </div>
                <div class="position-stat">
                    <p class="position-stat-value">{def_tds}</p>
                    <p class="position-stat-label">TD</p>
                </div>
            </div>
        """

    # Add manager info and highlights to front card
    # Get manager name from player data
    manager_name = str(player_row.get("manager", "N/A"))
    if manager_name == "nan" or not manager_name or manager_name == "":
        manager_name = "Free Agent"

    card_html += f"""
            <div class="manager-section">
                <p class="manager-label">Rostered By</p>
                <p class="manager-value">{manager_name}</p>
            </div>
    """

    # Add fantasy highlights - get stats from player_row data
    # Extract win/loss/champion data from player_row (aggregated data)
    wins_from_data = _safe_int(player_row.get("win", 0))
    losses_from_data = _safe_int(player_row.get("loss", 0))
    championships_from_data = _safe_int(player_row.get("championships", 0))
    position_rank = player_row.get("position_season_rank", None)

    # Pre-compute best game display - use matchup_stats if available, otherwise show PPG as fallback
    if matchup_stats and "best_game" in matchup_stats:
        best_game_val = matchup_stats.get("best_game", 0)
        best_game_display = f"{best_game_val:.1f}" if best_game_val else "N/A"
    else:
        # Fallback: use total points / games as estimate, or just show N/A
        best_game_display = "N/A"

    if view_mode == "Career Stats":
        # Career: show W-L record, Championships, Best Game, Optimal %
        champ_display = (
            "🏆" * championships_from_data if championships_from_data > 0 else "0"
        )
        card_html += f"""
            <div class="highlights-grid">
                <div class="highlight-box">
                    <p class="highlight-value">{wins_from_data}-{losses_from_data}</p>
                    <p class="highlight-label">W-L Record</p>
                </div>
                <div class="highlight-box">
                    <p class="highlight-value">{champ_display}</p>
                    <p class="highlight-label">Championships</p>
                </div>
                <div class="highlight-box">
                    <p class="highlight-value">{best_game_display}</p>
                    <p class="highlight-label">Best Game</p>
                </div>
                <div class="highlight-box">
                    <p class="highlight-value">{optimal_pct:.0f}%</p>
                    <p class="highlight-label">Optimal %</p>
                </div>
            </div>
        """
    else:
        # Season: show W-L record, Position Rank, Best Game, Optimal %
        # Handle position_rank - could be None, NaN, or a valid number
        try:
            if (
                position_rank is not None
                and not pd.isna(position_rank)
                and float(position_rank) > 0
            ):
                rank_display = f"#{int(position_rank)}"
            else:
                rank_display = "N/A"
        except (ValueError, TypeError):
            rank_display = "N/A"
        card_html += f"""
            <div class="highlights-grid">
                <div class="highlight-box">
                    <p class="highlight-value">{wins_from_data}-{losses_from_data}</p>
                    <p class="highlight-label">W-L Record</p>
                </div>
                <div class="highlight-box">
                    <p class="highlight-value">{rank_display}</p>
                    <p class="highlight-label">Pos Rank</p>
                </div>
                <div class="highlight-box">
                    <p class="highlight-value">{best_game_display}</p>
                    <p class="highlight-label">Best Game</p>
                </div>
                <div class="highlight-box">
                    <p class="highlight-value">{optimal_pct:.0f}%</p>
                    <p class="highlight-label">Optimal %</p>
                </div>
            </div>
        """

    # Close the FRONT card
    card_html += """
            <div class="card-footer">
                ★ FANTASY FOOTBALL CARD ★
            </div>
        </div>
    </div>
    """

    # Build BACK card with detailed stats
    card_html += f"""
        <!-- BACK CARD -->
        <div class="vintage-card-container">
            <div class="card-label">⚡ BACK ⚡</div>
            <div class="vintage-card">
                <div class="back-card-header">
                    <h3>{player_name}</h3>
                    <p>{pos_colors['name']} • {team} • {year_label}</p>
                </div>
    """

    # Add comprehensive stats based on position
    # Column names differ between season and career data
    if position == "QB":
        if view_mode == "Season Stats":
            pass_yds = _safe_int(player_row.get("passing_yards", 0))
            pass_tds = _safe_int(player_row.get("passing_tds", 0))
            pass_ints = _safe_int(player_row.get("passing_interceptions", 0))
            pass_att = _safe_int(player_row.get("attempts", 0))
            pass_cmp = _safe_int(player_row.get("completions", 0))
            rush_yds = _safe_int(player_row.get("rushing_yards", 0))
            rush_tds = _safe_int(player_row.get("rushing_tds", 0))
        else:  # Career Stats
            pass_yds = _safe_int(player_row.get("pass_yds", 0))
            pass_tds = _safe_int(player_row.get("pass_td", 0))
            pass_ints = _safe_int(player_row.get("passing_interceptions", 0))
            pass_att = _safe_int(player_row.get("attempts", 0))
            pass_cmp = _safe_int(player_row.get("completions", 0))
            rush_yds = _safe_int(player_row.get("rush_yds", 0))
            rush_tds = _safe_int(player_row.get("rush_td", 0))

        cmp_pct = (pass_cmp / pass_att * 100) if pass_att > 0 else 0

        card_html += f"""
                <div class="stat-section">
                    <div class="stat-section-title">⚡ Passing Stats</div>
                    <div class="stat-row">
                        <span class="stat-row-label">Passing Yards</span>
                        <span class="stat-row-value">{pass_yds:,}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Touchdowns</span>
                        <span class="stat-row-value">{pass_tds}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Interceptions</span>
                        <span class="stat-row-value">{pass_ints}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Completion %</span>
                        <span class="stat-row-value">{cmp_pct:.1f}%</span>
                    </div>
                </div>
                <div class="stat-section">
                    <div class="stat-section-title">🏃 Rushing Stats</div>
                    <div class="stat-row">
                        <span class="stat-row-label">Rushing Yards</span>
                        <span class="stat-row-value">{rush_yds:,}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Rush TDs</span>
                        <span class="stat-row-value">{rush_tds}</span>
                    </div>
                </div>
        """

    elif position == "RB":
        if view_mode == "Season Stats":
            rush_yds = _safe_int(player_row.get("rushing_yards", 0))
            rush_tds = _safe_int(player_row.get("rushing_tds", 0))
            rush_att = _safe_int(player_row.get("carries", 0))
            rec_yds = _safe_int(player_row.get("receiving_yards", 0))
            rec_tds = _safe_int(player_row.get("receiving_tds", 0))
            receptions = _safe_int(player_row.get("receptions", 0))
            targets = _safe_int(player_row.get("targets", 0))
        else:  # Career Stats
            rush_yds = _safe_int(player_row.get("rush_yds", 0))
            rush_tds = _safe_int(player_row.get("rush_td", 0))
            rush_att = _safe_int(player_row.get("rush_att", 0))
            rec_yds = _safe_int(player_row.get("rec_yds", 0))
            rec_tds = _safe_int(player_row.get("rec_td", 0))
            receptions = _safe_int(player_row.get("rec", 0))
            targets = _safe_int(player_row.get("targets", 0))

        ypc = (rush_yds / rush_att) if rush_att > 0 else 0
        catch_rate = (receptions / targets * 100) if targets > 0 else 0

        card_html += f"""
                <div class="stat-section">
                    <div class="stat-section-title">🏃 Rushing Stats</div>
                    <div class="stat-row">
                        <span class="stat-row-label">Rushing Yards</span>
                        <span class="stat-row-value">{rush_yds:,}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Rush TDs</span>
                        <span class="stat-row-value">{rush_tds}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Yards/Carry</span>
                        <span class="stat-row-value">{ypc:.1f}</span>
                    </div>
                </div>
                <div class="stat-section">
                    <div class="stat-section-title">🎯 Receiving Stats</div>
                    <div class="stat-row">
                        <span class="stat-row-label">Receptions</span>
                        <span class="stat-row-value">{receptions}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Rec Yards</span>
                        <span class="stat-row-value">{rec_yds:,}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Rec TDs</span>
                        <span class="stat-row-value">{rec_tds}</span>
                    </div>
                </div>
        """

    elif position in ["WR", "TE"]:
        if view_mode == "Season Stats":
            rec_yds = _safe_int(player_row.get("receiving_yards", 0))
            rec_tds = _safe_int(player_row.get("receiving_tds", 0))
            receptions = _safe_int(player_row.get("receptions", 0))
            targets = _safe_int(player_row.get("targets", 0))
            rush_yds = _safe_int(player_row.get("rushing_yards", 0))
            rush_tds = _safe_int(player_row.get("rushing_tds", 0))
        else:  # Career Stats
            rec_yds = _safe_int(player_row.get("rec_yds", 0))
            rec_tds = _safe_int(player_row.get("rec_td", 0))
            receptions = _safe_int(player_row.get("rec", 0))
            targets = _safe_int(player_row.get("targets", 0))
            rush_yds = _safe_int(player_row.get("rush_yds", 0))
            rush_tds = _safe_int(player_row.get("rush_td", 0))

        catch_rate = (receptions / targets * 100) if targets > 0 else 0
        ypr = (rec_yds / receptions) if receptions > 0 else 0

        card_html += f"""
                <div class="stat-section">
                    <div class="stat-section-title">🎯 Receiving Stats</div>
                    <div class="stat-row">
                        <span class="stat-row-label">Receptions</span>
                        <span class="stat-row-value">{receptions}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Rec Yards</span>
                        <span class="stat-row-value">{rec_yds:,}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Rec TDs</span>
                        <span class="stat-row-value">{rec_tds}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Targets</span>
                        <span class="stat-row-value">{targets}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Catch Rate</span>
                        <span class="stat-row-value">{catch_rate:.1f}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Yards/Rec</span>
                        <span class="stat-row-value">{ypr:.1f}</span>
                    </div>
                </div>
        """

    elif position == "K":
        fgm = _safe_int(player_row.get("fg_made", 0))
        fga = _safe_int(player_row.get("fg_att", 0))
        xpm = _safe_int(player_row.get("pat_made", 0))
        xpa = _safe_int(player_row.get("pat_att", 0))
        fg_pct = (fgm / fga * 100) if fga > 0 else 0
        xp_pct = (xpm / xpa * 100) if xpa > 0 else 0

        card_html += f"""
                <div class="stat-section">
                    <div class="stat-section-title">🎯 Kicking Stats</div>
                    <div class="stat-row">
                        <span class="stat-row-label">Field Goals</span>
                        <span class="stat-row-value">{fgm}/{fga}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">FG Percentage</span>
                        <span class="stat-row-value">{fg_pct:.1f}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Extra Points</span>
                        <span class="stat-row-value">{xpm}/{xpa}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">XP Percentage</span>
                        <span class="stat-row-value">{xp_pct:.1f}%</span>
                    </div>
                </div>
        """

    elif position == "DEF":
        sacks = _safe_float(player_row.get("def_sacks", 0))
        ints = _safe_int(player_row.get("def_interceptions", 0))
        def_tds = _safe_int(player_row.get("def_tds", 0))
        fumbles = _safe_int(
            player_row.get("fumble_recovery_opp", 0)
        )  # Or could use def_fumbles
        safeties = _safe_int(player_row.get("def_safeties", 0))
        pts_allowed = _safe_int(player_row.get("points_allowed", 0))

        card_html += f"""
                <div class="stat-section">
                    <div class="stat-section-title">🛡️ Defensive Stats</div>
                    <div class="stat-row">
                        <span class="stat-row-label">Sacks</span>
                        <span class="stat-row-value">{sacks:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Interceptions</span>
                        <span class="stat-row-value">{ints}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Def TDs</span>
                        <span class="stat-row-value">{def_tds}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Fumbles Rec</span>
                        <span class="stat-row-value">{fumbles}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Safeties</span>
                        <span class="stat-row-value">{safeties}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Pts Allowed</span>
                        <span class="stat-row-value">{pts_allowed}</span>
                    </div>
                </div>
        """

    # Fantasy League Stats (if available) - UNIQUE STATS ONLY
    if matchup_stats:
        card_html += f"""
                <div class="stat-section">
                    <div class="stat-section-title">🏆 Fantasy League Stats</div>
                    <div class="stat-row">
                        <span class="stat-row-label">Championships</span>
                        <span class="stat-row-value">{'🏆 ' * matchup_stats.get('rings', 0) if matchup_stats.get('rings', 0) > 0 else '0'}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Playoff Record</span>
                        <span class="stat-row-value">{matchup_stats.get('playoff_record', 'N/A')}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-row-label">Avg Team Margin</span>
                        <span class="stat-row-value">{matchup_stats.get('avg_margin', 0):+.1f} pts</span>
                    </div>
                </div>
        """

    # Close the BACK card
    card_html += """
                <div class="card-footer">
                    ★ FANTASY FOOTBALL CARD ★
                </div>
            </div>
        </div>
    </div>
    """

    # Render the card using components.html for proper rendering
    components.html(card_html, height=800, scrolling=True)

    # Additional details
    with st.expander("📋 Full Stats Breakdown", expanded=False):
        exclude_cols = ["player_lower", "headshot_url", "display"]
        available_cols = [c for c in filtered.columns if c not in exclude_cols]
        detail_df = filtered[available_cols].head(1).T.reset_index()
        detail_df.columns = ["Stat", "Value"]
        st.dataframe(detail_df, hide_index=True, use_container_width=True)
