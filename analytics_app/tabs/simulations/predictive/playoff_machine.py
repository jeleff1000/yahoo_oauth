#!/usr/bin/env python3
"""
playoff_machine.py - Interactive Playoff Scenario Generator

ESPN-style "Playoff Machine" where users can:
1. Pick winners for remaining games
2. See instant standings/seeding updates
3. Simulate unpicked games using team power ratings
4. See live playoff odds updates based on picks

Fast because:
- No Monte Carlo for user-picked scenarios (deterministic math)
- Slim Monte Carlo (1000 sims) for unpicked games
- All calculation happens client-side in milliseconds

Data Sources:
- Matchup table: Current standings (wins, losses, team_mu, team_sigma)
- Schedule table: Remaining games (future matchups with 0 points)

Responsive Design:
- Works on mobile and desktop
- Supports dark and light mode
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# Import data access
from md.core import T, run_query
from ..shared.simulation_styles import (
    render_section_header,
    render_odds_card,
    close_card,
)

# Slim simulation settings
SLIM_N_SIMS = 1000  # Fast enough for real-time updates


def _simulate_season_slim(
    current_standings: pd.DataFrame,
    remaining_games: pd.DataFrame,
    picks: Dict[Tuple[int, str, str], str],
    num_playoff_teams: int = 6,
    num_bye_teams: int = 2,
    n_sims: int = SLIM_N_SIMS,
) -> pd.DataFrame:
    """
    Run slim Monte Carlo simulation to get playoff odds.

    This is a fast version that:
    - Uses user picks as locked outcomes
    - Simulates unpicked games using power ratings
    - Runs fewer iterations (1000 vs 10000) for speed

    Returns DataFrame with columns:
    - manager, p_playoffs, p_bye, p_champ, avg_seed, exp_final_wins
    """
    if current_standings.empty:
        return pd.DataFrame()

    managers = current_standings["manager"].tolist()
    len(managers)
    {m: i for i, m in enumerate(managers)}

    # Get power ratings (mu) from standings
    mu_map = dict(
        zip(current_standings["manager"], current_standings["team_mu"].fillna(100))
    )

    # Get current wins/losses
    base_wins = current_standings.set_index("manager")["wins_to_date"].to_dict()
    base_losses = current_standings.set_index("manager")["losses_to_date"].to_dict()
    base_pf = (
        current_standings.set_index("manager")["points_for_to_date"].fillna(0).to_dict()
    )

    # Identify picked vs unpicked games
    picked_games = []
    unpicked_games = []

    for _, game in remaining_games.iterrows():
        week = int(game["week"])
        team_a = game["manager"]
        team_b = game["opponent"]
        key = (week, team_a, team_b)
        alt_key = (week, team_b, team_a)

        winner = picks.get(key) or picks.get(alt_key)
        if winner:
            picked_games.append((team_a, team_b, winner))
        else:
            unpicked_games.append((team_a, team_b))

    # Track simulation results
    playoff_counts = {m: 0 for m in managers}
    bye_counts = {m: 0 for m in managers}
    champ_counts = {m: 0 for m in managers}
    seed_totals = {m: 0.0 for m in managers}
    wins_totals = {m: 0.0 for m in managers}

    rng = np.random.default_rng(42)

    for sim in range(n_sims):
        # Start with base wins/losses
        sim_wins = base_wins.copy()
        sim_losses = base_losses.copy()
        sim_pf = base_pf.copy()

        # Apply picked games (deterministic)
        for team_a, team_b, winner in picked_games:
            loser = team_b if winner == team_a else team_a
            sim_wins[winner] = sim_wins.get(winner, 0) + 1
            sim_losses[loser] = sim_losses.get(loser, 0) + 1

        # Simulate unpicked games
        for team_a, team_b in unpicked_games:
            mu_a = mu_map.get(team_a, 100)
            mu_b = mu_map.get(team_b, 100)

            # Logistic win probability
            p_a_wins = 1 / (1 + 10 ** ((mu_b - mu_a) / 25.0))
            if rng.random() < p_a_wins:
                sim_wins[team_a] = sim_wins.get(team_a, 0) + 1
                sim_losses[team_b] = sim_losses.get(team_b, 0) + 1
            else:
                sim_wins[team_b] = sim_wins.get(team_b, 0) + 1
                sim_losses[team_a] = sim_losses.get(team_a, 0) + 1

        # Rank teams by wins (tiebreaker: points for)
        standings_list = [(m, sim_wins.get(m, 0), sim_pf.get(m, 0)) for m in managers]
        standings_list.sort(key=lambda x: (-x[1], -x[2]))

        # Assign seeds and track stats
        for seed, (mgr, wins, pf) in enumerate(standings_list, 1):
            seed_totals[mgr] += seed
            wins_totals[mgr] += wins

            if seed <= num_playoff_teams:
                playoff_counts[mgr] += 1
            if seed <= num_bye_teams:
                bye_counts[mgr] += 1

        # Simple championship simulation (top seeds have advantage)
        # Weight by inverse seed (seed 1 = weight 6, seed 6 = weight 1)
        playoff_teams = [
            standings_list[i][0]
            for i in range(min(num_playoff_teams, len(standings_list)))
        ]
        if playoff_teams:
            weights = [num_playoff_teams - i for i in range(len(playoff_teams))]
            weights = np.array(weights) / sum(weights)
            champ_idx = rng.choice(len(playoff_teams), p=weights)
            champ_counts[playoff_teams[champ_idx]] += 1

    # Build results DataFrame
    results = []
    for mgr in managers:
        results.append(
            {
                "manager": mgr,
                "p_playoffs": round(100 * playoff_counts[mgr] / n_sims, 1),
                "p_bye": round(100 * bye_counts[mgr] / n_sims, 1),
                "p_champ": round(100 * champ_counts[mgr] / n_sims, 1),
                "avg_seed": round(seed_totals[mgr] / n_sims, 2),
                "exp_final_wins": round(wins_totals[mgr] / n_sims, 1),
            }
        )

    return pd.DataFrame(results).sort_values("avg_seed")


def _render_scenario_odds_table(
    odds_df: pd.DataFrame,
    num_playoff_teams: int = 6,
    num_bye_teams: int = 2,
    color_scheme: str = "adjusted",
    table_key: str = "odds_table",
):
    """
    Render playoff odds table using AgGrid.

    Args:
        color_scheme: "original" for blue-green, "adjusted" for purple-orange
    """
    if odds_df.empty:
        st.info("No odds data to display.")
        return

    # Prepare DataFrame for display
    display_df = odds_df.copy()
    display_df = display_df.set_index("manager")

    # Define columns to show and color
    color_columns = ["p_playoffs", "p_bye", "p_champ"]
    reverse_columns = ["avg_seed"]  # Lower is better

    # Column display names
    column_names = {
        "avg_seed": "Avg Seed",
        "p_playoffs": "Playoff %",
        "p_bye": "Bye %",
        "p_champ": "Champ %",
        "exp_final_wins": "Exp Wins",
    }

    # Format specs
    format_specs = {
        "avg_seed": "{:.2f}",
        "p_playoffs": "{:.1f}",
        "p_bye": "{:.1f}",
        "p_champ": "{:.1f}",
        "exp_final_wins": "{:.1f}",
    }

    # Filter to relevant columns
    cols_to_show = ["avg_seed", "p_playoffs", "p_bye", "p_champ", "exp_final_wins"]
    cols_available = [c for c in cols_to_show if c in display_df.columns]
    display_df = display_df[cols_available]

    # Reset index for AgGrid
    display_df = display_df.reset_index()

    # Calculate min/max for gradients
    gradients = {}
    all_gradient_cols = list(set(color_columns + reverse_columns))
    for col in all_gradient_cols:
        if col in display_df.columns:
            numeric_vals = pd.to_numeric(display_df[col], errors="coerce").dropna()
            if len(numeric_vals) > 0:
                gradients[col] = {
                    "min": numeric_vals.min(),
                    "max": numeric_vals.max(),
                    "reverse": col in reverse_columns,
                }

    # Rename columns for display
    renamed_cols = {}
    for orig_col in all_gradient_cols:
        renamed_cols[orig_col] = column_names.get(orig_col, orig_col)

    display_df = display_df.rename(columns=column_names)
    [column_names.get(c, c) for c in color_columns]
    [column_names.get(c, c) for c in reverse_columns]

    # Build GridOptions
    gb = GridOptionsBuilder.from_dataframe(display_df)

    for idx, col in enumerate(display_df.columns):
        col_config = {"sortable": True, "filter": False}

        # Get original column name
        original_col = col
        for orig, display in column_names.items():
            if display == col:
                original_col = orig
                break

        # Column width
        if idx == 0:
            col_config.update({"width": 100, "minWidth": 80, "maxWidth": 140})
        else:
            col_config.update({"width": 70, "minWidth": 50, "maxWidth": 90})

        # Number formatting
        if original_col in format_specs:
            fmt = format_specs[original_col]
            decimals = 2 if ".2f" in fmt else 1
            no_percent = original_col in ["avg_seed", "exp_final_wins"]
            col_config["type"] = "numericColumn"
            col_config["valueFormatter"] = JsCode(
                f"(params) => params.value !== null ? params.value.toFixed({decimals}) + "
                f"({'\"\"' if no_percent else '\"%\"'}) : '—'"
            )

        # Gradient coloring based on color scheme
        if original_col in gradients:
            is_reverse = gradients[original_col]["reverse"]
            col_min = gradients[original_col]["min"]
            col_max = gradients[original_col]["max"]

            if color_scheme == "original":
                # Blue-green gradient (matching Championship Odds table)
                gradient_js = """
                // Blue-green gradient
                var r, g, b;
                if (normalized < 0.33) {
                    // Light blue to medium blue
                    var t = normalized * 3;
                    r = Math.floor(220 - (100 * t));
                    g = Math.floor(240 - (70 * t));
                    b = Math.floor(255 - (30 * t));
                } else if (normalized < 0.67) {
                    // Medium blue to teal
                    var t = (normalized - 0.33) * 3;
                    r = Math.floor(120 - (70 * t));
                    g = Math.floor(170 - (30 * t));
                    b = Math.floor(225 - (45 * t));
                } else {
                    // Teal to dark green
                    var t = (normalized - 0.67) * 3;
                    r = Math.floor(50 - (30 * t));
                    g = Math.floor(140 - (20 * t));
                    b = Math.floor(180 - (60 * t));
                }
                """
            else:
                # Purple-orange gradient
                gradient_js = """
                // Purple-orange gradient
                var r, g, b;
                if (normalized < 0.33) {
                    var t = normalized * 3;
                    r = Math.floor(147 + (180 - 147) * t);
                    g = Math.floor(112 + (100 - 112) * t);
                    b = Math.floor(219 + (180 - 219) * t);
                } else if (normalized < 0.67) {
                    var t = (normalized - 0.33) * 3;
                    r = Math.floor(180 + (255 - 180) * t);
                    g = Math.floor(100 + (160 - 100) * t);
                    b = Math.floor(180 + (50 - 180) * t);
                } else {
                    var t = (normalized - 0.67) * 3;
                    r = Math.floor(255);
                    g = Math.floor(160 + (120 - 160) * t);
                    b = Math.floor(50 + (30 - 50) * t);
                }
                """

            cell_style_js = f"""
            function(params) {{
                if (params.value === null || params.value === undefined || isNaN(params.value)) {{
                    return {{}};
                }}

                var value = parseFloat(params.value);
                var min = {col_min};
                var max = {col_max};

                if (min === max) {{
                    return {{}};
                }}

                var normalized = (value - min) / (max - min);
                {'normalized = 1 - normalized;' if is_reverse else ''}

                {gradient_js}

                var brightness = (r * 299 + g * 587 + b * 114) / 1000;
                var textColor = brightness < 130 ? '#ffffff' : '#000000';

                return {{
                    'backgroundColor': 'rgb(' + r + ',' + g + ',' + b + ')',
                    'color': textColor,
                    'fontWeight': '500'
                }};
            }}
            """
            col_config["cellStyle"] = JsCode(cell_style_js)

        gb.configure_column(col, **col_config)

    gb.configure_default_column(
        resizable=True, filterable=False, sortable=True, editable=False
    )
    gb.configure_grid_options(
        domLayout="normal",
        enableCellTextSelection=True,
        rowHeight=36,
        headerHeight=38,
        suppressColumnVirtualisation=True,
        suppressHorizontalScroll=False,
    )

    # Custom CSS based on color scheme
    if color_scheme == "original":
        header_bg = "linear-gradient(135deg, #3498db, #1abc9c)"
        shadow_color = "rgba(52,152,219,0.15)"
    else:
        header_bg = "linear-gradient(135deg, #6c5ce7, #a55eea)"
        shadow_color = "rgba(108,92,231,0.15)"

    custom_css = {
        ".ag-header-cell": {
            "font-weight": "600",
            "font-size": "0.85em",
            "padding": "4px 6px",
            "background": header_bg,
            "color": "white",
        },
        ".ag-cell": {"font-size": "0.85em", "padding": "4px 6px"},
        ".ag-root-wrapper": {
            "border-radius": "8px",
            "overflow": "hidden",
            "box-shadow": f"0 2px 8px {shadow_color}",
        },
    }

    grid_options = gb.build()
    table_height = min(400, 40 + len(display_df) * 36 + 10)

    AgGrid(
        display_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        custom_css=custom_css,
        height=table_height,
        theme="streamlit",
        fit_columns_on_grid_load=True,
        key=table_key,
    )


def _get_remaining_schedule_from_schedule_table(year: int) -> pd.DataFrame:
    """
    Get remaining regular season games from schedule table.

    Remaining games have team_points = 0 or NULL.
    """
    try:
        query = f"""
            SELECT manager, opponent, year, week, team_points
            FROM {T['schedule']}
            WHERE year = {year}
              AND (team_points = 0 OR team_points IS NULL)
              AND is_playoffs = 0
              AND is_consolation = 0
            ORDER BY week ASC
        """
        remaining = run_query(query)

        if remaining.empty:
            return remaining

        # Deduplicate matchups (each game appears twice - once per team)
        remaining["matchup_key"] = remaining.apply(
            lambda r: tuple(sorted([str(r["manager"]), str(r["opponent"])])), axis=1
        )
        remaining = remaining.drop_duplicates(subset=["year", "week", "matchup_key"])

        return remaining

    except Exception as e:
        st.warning(f"Could not load schedule: {e}")
        return pd.DataFrame()


def _get_current_standings(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    """Get current standings as of specified week."""
    current = df[
        (df["year"] == year)
        & (df["week"] == week)
        & (df["is_playoffs"] == 0)
        & (df["is_consolation"] == 0)
    ].copy()

    if current.empty:
        return pd.DataFrame()

    # Determine points column name (could be points_for_to_date or points_scored_to_date)
    pts_col = (
        "points_scored_to_date"
        if "points_scored_to_date" in current.columns
        else "points_for_to_date"
    )

    # Build aggregation dict dynamically based on available columns
    agg_dict = {
        "wins_to_date": "max",
        "losses_to_date": "max",
    }

    if pts_col in current.columns:
        agg_dict[pts_col] = "max"
    if "team_mu" in current.columns:
        agg_dict["team_mu"] = "mean"
    if "team_sigma" in current.columns:
        agg_dict["team_sigma"] = "mean"

    standings = current.groupby("manager").agg(agg_dict).reset_index()

    # Normalize column name to points_for_to_date for consistency
    if pts_col in standings.columns and pts_col != "points_for_to_date":
        standings["points_for_to_date"] = standings[pts_col]

    # Fill missing columns
    if "points_for_to_date" not in standings.columns:
        standings["points_for_to_date"] = 0
    if "team_mu" not in standings.columns:
        standings["team_mu"] = 100
    if "team_sigma" not in standings.columns:
        standings["team_sigma"] = 20

    return standings.sort_values(
        ["wins_to_date", "points_for_to_date"], ascending=[False, False]
    ).reset_index(drop=True)


def _simulate_game(team_a_mu: float, team_b_mu: float, scale: float = 25.0) -> str:
    """
    Simulate a single game using power ratings.

    Uses logistic function: P(A wins) = 1 / (1 + 10^((mu_B - mu_A) / scale))
    """
    p_a_wins = 1 / (1 + 10 ** ((team_b_mu - team_a_mu) / scale))
    return "A" if np.random.random() < p_a_wins else "B"


def _simulate_all_games(
    remaining_games: pd.DataFrame, standings: pd.DataFrame, mode: str = "simulate"
) -> Dict[Tuple[int, str, str], str]:
    """
    Simulate ALL remaining games (replaces any existing picks).

    Args:
        mode: "simulate" (probabilistic), "favorites" (higher mu wins), "underdogs" (lower mu wins)

    Returns new picks dict with all games simulated.
    """
    new_picks = {}

    # Get team power ratings
    mu_map = dict(zip(standings["manager"], standings["team_mu"].fillna(100)))

    for _, row in remaining_games.iterrows():
        week = row["week"]
        team_a = row["manager"]
        team_b = row["opponent"]

        key = (week, team_a, team_b)

        mu_a = mu_map.get(team_a, 100)
        mu_b = mu_map.get(team_b, 100)

        if mode == "favorites":
            winner = team_a if mu_a >= mu_b else team_b
        elif mode == "underdogs":
            winner = team_a if mu_a < mu_b else team_b
        else:
            result = _simulate_game(mu_a, mu_b)
            winner = team_a if result == "A" else team_b

        new_picks[key] = winner

    return new_picks


@st.fragment
def display_playoff_machine(
    matchup_data_df: pd.DataFrame = None,
    year: int = None,
    week: int = None,
):
    """
    Display the interactive Playoff Machine.

    Users can:
    - Pick winners for remaining games
    - See instant standings updates
    - Simulate unpicked games

    Args:
        matchup_data_df: Optional pre-loaded matchup data
        year: Selected year (from unified header)
        week: Selected week (from unified header)
    """
    st.caption("Pick winners to see playoff picture changes.")

    # Load data if not provided
    if matchup_data_df is None:
        with st.spinner("Loading data..."):
            matchup_data_df = run_query(
                f"""
                SELECT * FROM {T['matchup']}
                WHERE is_playoffs = 0 AND is_consolation = 0
                ORDER BY year DESC, week DESC
            """
            )

    if matchup_data_df.empty:
        st.info("No matchup data available.")
        return

    # Ensure numeric types
    matchup_data_df["year"] = pd.to_numeric(
        matchup_data_df["year"], errors="coerce"
    ).astype(int)
    matchup_data_df["week"] = pd.to_numeric(
        matchup_data_df["week"], errors="coerce"
    ).astype(int)

    # Use provided year/week or default to latest
    years = sorted(matchup_data_df["year"].unique(), reverse=True)
    if year is None:
        selected_year = years[0] if years else None
    else:
        selected_year = year

    year_data = matchup_data_df[matchup_data_df["year"] == selected_year]

    if week is None:
        current_week = int(year_data["week"].max())
    else:
        current_week = week

    # Get data
    current_standings = _get_current_standings(year_data, selected_year, current_week)

    # Get remaining games from SCHEDULE table (has future matchups)
    remaining_games = _get_remaining_schedule_from_schedule_table(selected_year)

    if current_standings.empty:
        st.warning("No standings data available for this week.")
        return

    # Initialize session state for picks and active mode
    if "playoff_picks" not in st.session_state:
        st.session_state.playoff_picks = {}
    if "pm_active_mode" not in st.session_state:
        st.session_state.pm_active_mode = None

    # Prominent action buttons row
    active_mode = st.session_state.pm_active_mode
    btn_cols = st.columns([1.5, 1, 1, 1, 3])

    with btn_cols[0]:
        if st.button(
            "Simulate",
            key="pm_simulate",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.playoff_picks = _simulate_all_games(
                remaining_games, current_standings, "simulate"
            )
            st.session_state.pm_active_mode = "simulate"
            st.rerun(scope="fragment")

    with btn_cols[1]:
        fav_type = "primary" if active_mode == "favorites" else "secondary"
        if st.button("Favorites", key="pm_favorites", type=fav_type, use_container_width=True):
            st.session_state.playoff_picks = _simulate_all_games(
                remaining_games, current_standings, "favorites"
            )
            st.session_state.pm_active_mode = "favorites"
            st.rerun(scope="fragment")

    with btn_cols[2]:
        und_type = "primary" if active_mode == "underdogs" else "secondary"
        if st.button("Underdogs", key="pm_underdogs", type=und_type, use_container_width=True):
            st.session_state.playoff_picks = _simulate_all_games(
                remaining_games, current_standings, "underdogs"
            )
            st.session_state.pm_active_mode = "underdogs"
            st.rerun(scope="fragment")

    with btn_cols[3]:
        if st.button("Reset", key="pm_reset", use_container_width=True):
            st.session_state.playoff_picks = {}
            st.session_state.pm_active_mode = None
            st.rerun(scope="fragment")

    # 2-Panel Layout: Left (picker) | Right (results)
    left_panel, right_panel = st.columns([2, 1])

    with left_panel:
        with st.container(border=True):
            _render_remaining_games(remaining_games)

    with right_panel:
        with st.container(border=True):
            st.markdown("**Results**")
            _render_standings(
                current_standings, remaining_games, st.session_state.playoff_picks
            )


def _render_remaining_games(remaining_games: pd.DataFrame):
    """Render remaining games with compact styled matchup cards."""
    st.markdown("**Pick Winners**")

    if remaining_games.empty:
        st.info("No remaining games - regular season complete!")
        return

    # Inject CSS for compact matchup styling
    st.markdown(
        """
    <style>
    .week-label {
        font-size: 0.9em;
        font-weight: 600;
        color: #6c5ce7;
        margin: 0.5rem 0 0.375rem 0;
    }
    .vs-text {
        text-align: center;
        font-size: 0.7em;
        font-weight: 700;
        color: #888;
        padding: 4px 0;
    }

    /* Compact buttons inside matchup-picks */
    .matchup-picks [data-testid="stVerticalBlockBorderWrapper"] {
        margin-bottom: 0.375rem !important;
    }
    .matchup-picks [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] > button {
        background-color: #f1c40f !important;
        color: #000 !important;
        border: 1px solid #d4ac0d !important;
        font-weight: 600 !important;
        min-height: 32px !important;
        height: 32px !important;
        border-radius: 4px !important;
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        padding: 0 6px !important;
        font-size: 0.75em !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .matchup-picks [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] > button:hover {
        background-color: #f4d03f !important;
        border-color: #b7950b !important;
    }

    /* Winner button - green */
    .matchup-picks [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] > button[kind="primary"] {
        background-color: #27ae60 !important;
        color: #fff !important;
        border-color: #1e8449 !important;
    }
    .matchup-picks [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #2ecc71 !important;
    }

    /* Make button columns equal width */
    .matchup-picks [data-testid="stHorizontalBlock"] > div {
        flex: 1 1 0 !important;
        min-width: 0 !important;
    }

    /* More compact bordered containers */
    .matchup-picks [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.375rem !important;
    }

    @media (prefers-color-scheme: dark) {
        .week-label { color: #a55eea; }
    }
    @media (max-width: 600px) {
        .matchup-picks [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] > button {
            min-height: 28px !important;
            height: 28px !important;
            font-size: 0.7em !important;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Wrap all matchups in matchup-picks class for CSS scoping
    st.markdown('<div class="matchup-picks">', unsafe_allow_html=True)

    # Group by week
    weeks = sorted(remaining_games["week"].unique())
    first_week = weeks[0] if weeks else None

    for week in weeks:
        week_games = remaining_games[remaining_games["week"] == week]
        games_list = week_games.to_dict("records")

        # First week expanded, future weeks in expander
        if week == first_week:
            st.markdown(
                f"<div class='week-label'>Week {int(week)}</div>",
                unsafe_allow_html=True,
            )
            _render_week_matchups(week, games_list)
        else:
            with st.expander(f"Week {int(week)}", expanded=False):
                _render_week_matchups(week, games_list)

    st.markdown("</div>", unsafe_allow_html=True)

    # Inject JavaScript to style loser buttons after all matchups rendered
    _inject_loser_button_styles()


def _render_week_matchups(week: int, games_list: list):
    """Render matchups for a single week."""
    num_games = len(games_list)
    matchups_per_row = 3

    for row_start in range(0, num_games, matchups_per_row):
        row_games = games_list[row_start : row_start + matchups_per_row]

        # Always create 3 columns for consistent widths (some may be empty)
        matchup_cols = st.columns(3)

        for game_idx, game in enumerate(row_games):
            team_a = game["manager"]
            team_b = game["opponent"]

            key = (int(week), team_a, team_b)
            alt_key = (int(week), team_b, team_a)
            current_pick = st.session_state.playoff_picks.get(
                key
            ) or st.session_state.playoff_picks.get(alt_key)

            with matchup_cols[game_idx]:
                # Framed matchup container
                with st.container(border=True):
                    # Inner columns for Team A | vs | Team B
                    inner_cols = st.columns([3, 1, 3])

                    # Determine button states: winner=green, loser=red, unpicked=yellow
                    a_is_winner = current_pick == team_a
                    b_is_winner = current_pick == team_b
                    a_is_loser = current_pick == team_b  # A loses if B was picked
                    b_is_loser = current_pick == team_a  # B loses if A was picked

                    with inner_cols[0]:
                        # Winner: ✓ prefix, primary type (green)
                        # Loser: ✗ prefix, secondary type (will be styled red via JS)
                        # Unpicked: no prefix, secondary type (yellow)
                        if a_is_winner:
                            label_a = f"✓ {team_a}"
                            type_a = "primary"
                        elif a_is_loser:
                            label_a = f"✗ {team_a}"
                            type_a = "secondary"
                        else:
                            label_a = team_a
                            type_a = "secondary"

                        if st.button(
                            label_a,
                            key=f"pm_{week}_{team_a}_{team_b}_a",
                            type=type_a,
                            use_container_width=True,
                        ):
                            st.session_state.playoff_picks[key] = team_a
                            if alt_key in st.session_state.playoff_picks:
                                del st.session_state.playoff_picks[alt_key]
                            st.session_state.pm_active_mode = (
                                None  # Clear active mode on manual pick
                            )
                            st.rerun(scope="fragment")

                    with inner_cols[1]:
                        st.markdown(
                            "<div class='vs-text'>vs</div>", unsafe_allow_html=True
                        )

                    with inner_cols[2]:
                        if b_is_winner:
                            label_b = f"✓ {team_b}"
                            type_b = "primary"
                        elif b_is_loser:
                            label_b = f"✗ {team_b}"
                            type_b = "secondary"
                        else:
                            label_b = team_b
                            type_b = "secondary"

                        if st.button(
                            label_b,
                            key=f"pm_{week}_{team_a}_{team_b}_b",
                            type=type_b,
                            use_container_width=True,
                        ):
                            st.session_state.playoff_picks[key] = team_b
                            if alt_key in st.session_state.playoff_picks:
                                del st.session_state.playoff_picks[alt_key]
                            st.session_state.pm_active_mode = (
                                None  # Clear active mode on manual pick
                            )
                            st.rerun(scope="fragment")


def _inject_loser_button_styles():
    """Inject JavaScript to style loser buttons (✗) as red."""
    components.html(
        """
    <script>
    function styleLoserButtons() {
        const buttons = parent.document.querySelectorAll('button');
        buttons.forEach(btn => {
            const text = btn.innerText || btn.textContent;
            if (text.includes('✗')) {
                btn.style.setProperty('background-color', '#e74c3c', 'important');
                btn.style.setProperty('color', '#fff', 'important');
                btn.style.setProperty('border-color', '#c0392b', 'important');
            }
        });
    }
    // Run immediately and after short delay for Streamlit render
    styleLoserButtons();
    setTimeout(styleLoserButtons, 100);
    setTimeout(styleLoserButtons, 300);
    </script>
    """,
        height=0,
    )


def _render_standings(
    current_standings: pd.DataFrame, remaining_games: pd.DataFrame, picks: Dict
):
    """Render original and adjusted playoff odds tables side by side."""
    st.markdown("**Playoff Odds Comparison**")

    # Get playoff config (default 6 teams, 2 byes)
    num_playoff_teams = 6
    num_bye_teams = 2

    # Stats
    picks_made = len([p for p in picks.values() if p])
    total_games = len(remaining_games)
    st.caption(f"Picks: {picks_made}/{total_games} games")

    # Calculate both original (no picks) and adjusted (with picks) odds
    with st.spinner("Calculating odds..."):
        # Original odds - no picks applied
        original_odds = _simulate_season_slim(
            current_standings,
            remaining_games,
            {},  # Empty picks = original odds
            num_playoff_teams=num_playoff_teams,
            num_bye_teams=num_bye_teams,
        )

        # Adjusted odds - with user picks
        adjusted_odds = _simulate_season_slim(
            current_standings,
            remaining_games,
            picks,
            num_playoff_teams=num_playoff_teams,
            num_bye_teams=num_bye_teams,
        )

    if original_odds.empty or adjusted_odds.empty:
        st.info("Unable to calculate odds.")
        return

    # Display side by side with compact headers
    col1, col2 = st.columns(2)

    with col1:
        render_odds_card("Original Odds", "", "Before picks")
        _render_scenario_odds_table(
            original_odds,
            num_playoff_teams,
            num_bye_teams,
            color_scheme="original",
            table_key="pm_original_odds",
        )
        close_card()

    with col2:
        render_odds_card("Adjusted Odds", "", "With selections")
        _render_scenario_odds_table(
            adjusted_odds,
            num_playoff_teams,
            num_bye_teams,
            color_scheme="original",
            table_key="pm_adjusted_odds",
        )
        close_card()


def display_playoff_machine_compact(matchup_data_df: pd.DataFrame = None):
    """Compact version for embedding in other views."""
    display_playoff_machine(matchup_data_df)
