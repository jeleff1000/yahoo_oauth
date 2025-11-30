"""
WEEKLY RECAP DISPLAY
====================
Displays the weekly matchup recap using config-driven narratives.
All text/criteria are defined in weekly_recap_config.py for easy editing.
"""

from typing import Any, Dict, Optional, List
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure streamlit_ui directory is in path for imports
_streamlit_ui_dir = Path(__file__).parent.parent.parent.parent.parent.resolve()
if str(_streamlit_ui_dir) not in sys.path:
    sys.path.insert(0, str(_streamlit_ui_dir))

from ..weekly_recap_config import (
    RESULT_TEMPLATES,
    RESULT_FLAVOR,
    LEAGUE_CONTEXT,
    PROJECTION_CONTEXT,
    STREAK_CONTEXT,
    OPTIMAL_CONTEXT,
    GRADE_CONTEXT,
    MILESTONE_CONTEXT,
    HISTORICAL_CONTEXT,
    evaluate_criteria,
    format_template,
    _safe_get,
    _to_float,
    _to_int,
    _is_win,
)
from shared.dataframe_utils import as_dataframe, get_matchup_df


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower().strip())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _val(row: pd.Series, col: Optional[str], default=None):
    if not col:
        return default
    try:
        v = row[col]
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def _fmt_points(v) -> str:
    f = _to_float(v, None)
    if f is None:
        return "N/A"
    return f"{f:.1f}"


# Bolding helpers for markdown
_UNIT_WORDS = (
    "point", "points", "win", "wins", "seed", "seeds",
    "chance", "favorite", "favorites", "spread", "margin",
    "victory", "loss", "losses", "team", "teams", "game", "games",
)
_ADJ_WORDS = ("extra", "total", "more", "fewer", "additional", "straight")

_RE_NUM_UNIT = re.compile(
    rf"(?<!\w)("
    rf"(?:\d{{1,3}}(?:,\d{{3}})*|\d+)(?:\.\d+)?%?"
    rf"(?:\s+(?:{'|'.join(_ADJ_WORDS)}))?"
    rf"(?:\s+(?:{'|'.join(_UNIT_WORDS)}))"
    rf")(?!\w)",
    re.IGNORECASE
)


def _bold_parentheticals(s: str) -> str:
    def repl(m: re.Match) -> str:
        return f"**({m.group(1)})**"
    return re.sub(r"\(([^()]*)\)", repl, str(s or ""))


def _apply_bolding(s: str) -> str:
    """Bold parentheticals for emphasis."""
    return _bold_parentheticals(s)


# =============================================================================
# MAIN DISPLAY FUNCTION
# =============================================================================

@st.fragment
def display_weekly_recap(
    df_dict: Optional[Dict[Any, Any]] = None,
    *,
    year: int,
    week: int,
    manager: str,
) -> None:
    """
    Display a flowing weekly recap narrative.
    All criteria and text are defined in weekly_recap_config.py.
    """
    matchup_df = get_matchup_df(df_dict)
    if matchup_df is None:
        st.info("No `Matchup Data` dataset available.")
        return

    df = matchup_df.copy()

    # Find columns
    col_year = _find_col(df, ["year"])
    col_week = _find_col(df, ["week"])
    col_manager = _find_col(df, ["manager"])

    # Filter to selected row
    if col_year:
        df = df[pd.to_numeric(df[col_year], errors="coerce").astype("Int64") == year]
    if col_week:
        df = df[pd.to_numeric(df[col_week], errors="coerce").astype("Int64") == week]
    if col_manager:
        df = df[df[col_manager].astype(str).str.strip() == str(manager).strip()]

    if df.empty:
        st.warning("No record found for the selected Manager, Week, and Year.")
        return

    row = df.iloc[0]

    # Build canonical row with all needed fields
    row_dict = _build_canonical_row(df, row)

    # =========================================================================
    # BUILD THE NARRATIVE - Group related concepts into flowing paragraphs
    # =========================================================================

    paragraphs = []

    # --- Paragraph 1: Result + Flavor + League Context ---
    # "What happened and how lucky/unlucky was it?"
    para1_parts = []

    # Core result + flavor
    result_sentence = _build_result_sentence(row_dict)
    flavor_matches = evaluate_criteria(RESULT_FLAVOR, row_dict)
    if flavor_matches:
        flavor_text = format_template(flavor_matches[0].get("text", ""), row_dict)
        result_sentence += f" {flavor_text}"
    para1_parts.append(result_sentence)

    # League context (luck/teams beaten)
    league_matches = evaluate_criteria(LEAGUE_CONTEXT, row_dict)
    if league_matches:
        luck_text = format_template(league_matches[0].get("text", ""), row_dict)
        para1_parts.append(luck_text)

    if para1_parts:
        paragraphs.append(" ".join(para1_parts))

    # --- Paragraph 2: Projection + Optimal Lineup ---
    # "How did you perform vs expectations and did you make the right calls?"
    para2_parts = []

    proj_matches = evaluate_criteria(PROJECTION_CONTEXT, row_dict)
    if proj_matches:
        proj_text = format_template(proj_matches[0].get("text", ""), row_dict)
        para2_parts.append(proj_text)

    optimal_matches = evaluate_criteria(OPTIMAL_CONTEXT, row_dict)
    if optimal_matches:
        optimal_text = format_template(optimal_matches[0].get("text", ""), row_dict)
        para2_parts.append(optimal_text)

    if para2_parts:
        paragraphs.append(" ".join(para2_parts))

    # --- Paragraph 3: Streak + Grade + Milestones + Historical ---
    # "Where do you stand now?"
    para3_parts = []

    streak_matches = evaluate_criteria(STREAK_CONTEXT, row_dict)
    if streak_matches:
        streak_text = format_template(streak_matches[0].get("text", ""), row_dict)
        para3_parts.append(streak_text)

    grade_matches = evaluate_criteria(GRADE_CONTEXT, row_dict)
    if grade_matches:
        grade_text = format_template(grade_matches[0].get("text", ""), row_dict)
        para3_parts.append(grade_text)

    # Milestones woven in
    milestone_matches = evaluate_criteria(MILESTONE_CONTEXT, row_dict)
    for m in milestone_matches:
        text = format_template(m.get("text", ""), row_dict)
        para3_parts.append(text)

    # Historical achievements woven in
    historical_matches = evaluate_criteria(HISTORICAL_CONTEXT, row_dict)
    for h in historical_matches:
        text = format_template(h.get("text", ""), row_dict)
        para3_parts.append(text)

    if para3_parts:
        paragraphs.append(" ".join(para3_parts))

    # --- Output paragraphs ---
    for para in paragraphs:
        st.markdown(_apply_bolding(para))


def _build_canonical_row(df: pd.DataFrame, row: pd.Series) -> Dict:
    """
    Build a canonical dict with all the fields the config expects.
    Maps various column names to standard keys.
    """
    # Column mappings
    col_mappings = {
        'manager': ['manager', 'manager_name', 'owner'],
        'opponent': ['opponent', 'opponent_name'],
        'team_points': ['team_points', 'points', 'score'],
        'opponent_points': ['opponent_points', 'opp_points', 'opponent_score'],
        'margin': ['margin', 'point_diff'],
        'win': ['win', 'did_win', 'is_win'],
        'week': ['week'],
        'year': ['year', 'season'],
        'is_playoffs': ['is_playoffs', 'playoff_game', 'playoffs'],
        'is_championship': ['is_championship', 'championship_game'],
        'is_sacko_game': ['is_sacko_game', 'sacko_game'],
        'is_consolation': ['is_consolation', 'consolation_game'],
        'weekly_mean': ['weekly_mean', 'league_mean', 'avg_score'],
        'weekly_median': ['weekly_median', 'league_median'],
        'teams_beat_this_week': ['teams_beat_this_week', 'teams_beaten'],
        'above_league_median': ['above_league_median', 'beat_median'],
        'above_proj_score': ['above_proj_score', 'beat_projection'],
        'proj_wins': ['proj_wins', 'projected_win', 'was_favorite'],
        'win_vs_spread': ['win_vs_spread', 'covered_spread', 'beat_spread'],
        'expected_odds': ['expected_odds', 'win_probability', 'win_odds'],
        'expected_spread': ['expected_spread', 'spread', 'point_spread'],
        'proj_score_error': ['proj_score_error', 'projection_error'],
        'winning_streak': ['winning_streak', 'win_streak'],
        'losing_streak': ['losing_streak', 'loss_streak'],
        'optimal_points': ['optimal_points', 'optimal_score', 'best_possible'],
        'grade': ['grade', 'weekly_grade'],
        'gpa': ['gpa', 'weekly_gpa'],
        'wins_to_date': ['wins_to_date', 'wins', 'season_wins'],
        'losses_to_date': ['losses_to_date', 'losses', 'season_losses'],
        'playoff_seed_to_date': ['playoff_seed_to_date', 'seed', 'current_seed'],
        'champion': ['champion', 'is_champion'],
        'sacko': ['sacko', 'is_sacko'],
        'team_made_playoffs': ['team_made_playoffs', 'made_playoffs'],
        'manager_all_time_ranking': ['manager_all_time_ranking'],
        'manager_all_time_percentile': ['manager_all_time_percentile'],
        'manager_season_ranking': ['manager_season_ranking'],
        'prev_losing_streak': ['prev_losing_streak', 'previous_losing_streak'],
        'prev_winning_streak': ['prev_winning_streak', 'previous_winning_streak'],
        'prev_loss': ['prev_loss', 'previous_loss'],
        'prev_win': ['prev_win', 'previous_win'],
        'prev_playoff_seed': ['prev_playoff_seed', 'previous_playoff_seed'],
        'must_win': ['must_win', 'elimination_game'],
        'eliminated': ['eliminated', 'out_of_contention'],
        'clinched_playoffs': ['clinched_playoffs', 'playoff_clinch'],
        'runner_up': ['runner_up', 'is_runner_up'],
        'third_place': ['third_place', 'is_third_place'],
    }

    result = {}

    for canon_key, candidates in col_mappings.items():
        col = _find_col(df, candidates)
        if col:
            val = _val(row, col, None)
            result[canon_key] = val
        else:
            result[canon_key] = None

    return result


def _build_result_sentence(row_dict: Dict) -> str:
    """Build the core result sentence."""
    opponent = _safe_get(row_dict, 'opponent', 'Unknown')
    team_pts = _to_float(_safe_get(row_dict, 'team_points'), 0)
    opp_pts = _to_float(_safe_get(row_dict, 'opponent_points'), 0)

    if _is_win(row_dict):
        template = RESULT_TEMPLATES.get("win", "")
    else:
        template = RESULT_TEMPLATES.get("loss", "")

    return format_template(template, {
        'opponent': opponent,
        'team_points': team_pts,
        'opponent_points': opp_pts,
        **row_dict
    })
