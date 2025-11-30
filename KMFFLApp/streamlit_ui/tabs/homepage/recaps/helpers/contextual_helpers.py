"""
Contextual Message Helpers
===========================
Helper functions to generate contextual messages based on game data.
These work with the dialogue in recap_dialogue.py
"""

from typing import Optional, Dict, Any, List
import pandas as pd
from .recap_dialogue import (
    CONTEXTUAL_MESSAGES,
    PLAYER_PERFORMANCE,
    format_dialogue,
    SEASON_ANALYSIS,
    get_outcome_dialogue,
)


# ============================================================================
# HELPER FUNCTION FOR SAFE VALUE EXTRACTION
# ============================================================================

def _safe_get(row: pd.Series, key: str, default=None):
    """Safely get a value from a Series, handling NA/NaN/None."""
    val = row.get(key, default)
    if pd.isna(val):
        return default
    return val


def _safe_compare(val, compare_val) -> bool:
    """Safely compare values, returning False if either is NA/NaN."""
    if pd.isna(val) or pd.isna(compare_val):
        return False
    return val == compare_val


def _is_win(v) -> Optional[bool]:
    """Coerce various win representations into True/False/None.

    Returns:
    - True for wins (1, True, 'W', 'win', 'won', 'true', etc.)
    - False for losses (0, False, 'L', 'loss', 'lost', 'false', etc.)
    - None if unknown/uninterpretable
    """
    if v is None:
        return None
    # Handle pandas NA / NaN
    try:
        if isinstance(v, float) and pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, bool):
        return v

    # Numeric-like
    if isinstance(v, (int, float)) and not (isinstance(v, float) and pd.isna(v)):
        try:
            return int(v) == 1
        except Exception:
            pass

    s = str(v).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y', 'win', 'w', 'won'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n', 'loss', 'l', 'lost'}:
        return False

    return None


# Small numeric helpers used by multiple features
_DEF_POINT_CANDS = (
    'points', 'Points', 'computed_points', 'points_original',
    'fantasy_points_ppr', 'fantasy_points_half_ppr', 'fantasy_points_zero_ppr',
    'rolling_point_total'
)

def _safe_float(v, default=None) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    except Exception:
        return default


def _detect_points_value(row: pd.Series) -> Optional[float]:
    for c in _DEF_POINT_CANDS:
        if c in row.index:
            f = _safe_float(row.get(c))
            if f is not None:
                return f
    # last resort: any column containing 'point'
    for c, v in row.items():
        if isinstance(c, str) and 'point' in c.lower():
            f = _safe_float(v)
            if f is not None:
                return f
    return None


def _to_percent_0_100(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    # Normalize common encodings: 0-1 or 0-100
    if 0 <= f <= 1:
        return f * 100.0
    return f


def _top_bucket_from_percentile(pct_0_100: Optional[float]) -> Optional[str]:
    """Map a percentile (higher is better) to a concise 'top X%' bucket string.
    Only return a bucket for standout values (>= 75th percentile)."""
    if pct_0_100 is None:
        return None
    try:
        p = float(pct_0_100)
    except Exception:
        return None
    if p >= 99:
        return '1%'
    if p >= 95:
        return '5%'
    if p >= 90:
        return '10%'
    if p >= 75:
        return '25%'
    return None


# ============================================================================
# OPTIMAL LINEUP MESSAGES
# ============================================================================

def get_optimal_lineup_message(row: pd.Series) -> Optional[str]:
    """Generate message about optimal lineup performance."""
    team_points = _safe_get(row, 'team_points')
    optimal_points = _safe_get(row, 'optimal_points')

    if team_points is None or optimal_points is None:
        return None

    optimal_diff = optimal_points - team_points
    optimal_pct = (optimal_diff / optimal_points * 100) if optimal_points > 0 else 0

    # Swing impact calculation
    margin = _safe_get(row, 'margin')
    opponent_points = _safe_get(row, 'opponent_points')
    could_have_won = False
    near_flip = False
    if margin is not None and opponent_points is not None:
        potential_swing = optimal_points - team_points
        could_have_won = (margin < 0) and (team_points + potential_swing > opponent_points)
        near_flip = (margin < 0) and (team_points + potential_swing >= opponent_points - 3)

    # Determine severity
    left_points = CONTEXTUAL_MESSAGES['optimal_lineup']['left_points']
    win_flag_raw = _safe_get(row, 'win')
    win_flag = _is_win(win_flag_raw)

    # If win is ambiguous/missing, try to infer from margin (positive margin = win)
    if win_flag is None and margin is not None:
        try:
            m_val = float(margin)
            if m_val > 0:
                win_flag = True
            elif m_val < 0:
                win_flag = False
        except Exception:
            pass

    # Check for swing impact first (only for actual losses)
    if win_flag is False and could_have_won:
        template = left_points.get('swing_loss')
        margin_pos_str = f"{abs(margin):.1f}"
        return format_dialogue(template, optimal_diff=optimal_diff, margin_pos_str=margin_pos_str)
    elif win_flag is False and near_flip:
        template = left_points.get('near_swing')
        margin_pos_str = f"{abs(margin):.1f}"
        return format_dialogue(template, optimal_diff=optimal_diff, margin_pos_str=margin_pos_str)
    else:
        # Existing logic
        if optimal_diff == 0:
            template = left_points['perfect']
        elif optimal_diff < 6.1:
            # Treat anything under 6.1 as minor (covers <6 and 6.0)
            if win_flag is True:
                template = left_points.get('minor_win', left_points.get('minor'))
            else:
                template = left_points.get('minor_loss', left_points.get('minor'))
        elif optimal_diff <= 15:
            template = left_points.get('significant', left_points.get('minor'))
        else:
            template = left_points.get('major', left_points.get('massive'))

        return format_dialogue(template, optimal_diff=optimal_diff, optimal_pct=optimal_pct)


def get_optimal_record_message(row: pd.Series) -> Optional[str]:
    """Compare actual vs optimal win-loss record."""
    optimal_win = _safe_get(row, 'optimal_win')
    win_raw = _safe_get(row, 'win')
    win = _is_win(win_raw)
    wins = _safe_get(row, 'wins_to_date', 0)
    losses = _safe_get(row, 'losses_to_date', 0)

    if optimal_win is None or win is None:
        return None

    # This would need season-wide aggregation - placeholder for now
    template = CONTEXTUAL_MESSAGES['optimal_lineup']['optimal_record']

    # Determine comment based on difference
    optimal_comment = "You're maximizing your roster!" if optimal_win == win else "Better lineup decisions could improve your record."

    return format_dialogue(template,
                          optimal_wins=wins,  # Placeholder - would need actual optimal wins
                          optimal_losses=losses,
                          wins=wins,
                          losses=losses,
                          optimal_comment=optimal_comment)


# ============================================================================
# STREAK MESSAGES
# ============================================================================

def get_streak_message(row: pd.Series) -> Optional[str]:
    """Generate message about winning or losing streaks."""
    winning_streak = _safe_get(row, 'winning_streak', 0)
    losing_streak = _safe_get(row, 'losing_streak', 0)

    if winning_streak >= 5:
        template = CONTEXTUAL_MESSAGES['streaks']['winning_streak']['hot']
        return format_dialogue(template, winning_streak=winning_streak)
    elif winning_streak >= 3:
        template = CONTEXTUAL_MESSAGES['streaks']['winning_streak']['rolling']
        return format_dialogue(template, winning_streak=winning_streak)
    elif winning_streak == 2:
        return CONTEXTUAL_MESSAGES['streaks']['winning_streak']['started']

    if losing_streak >= 4:
        template = CONTEXTUAL_MESSAGES['streaks']['losing_streak']['crisis']
        return format_dialogue(template, losing_streak=losing_streak)
    elif losing_streak >= 3:
        template = CONTEXTUAL_MESSAGES['streaks']['losing_streak']['slump']
        return format_dialogue(template, losing_streak=losing_streak)
    elif losing_streak == 2:
        return CONTEXTUAL_MESSAGES['streaks']['losing_streak']['trouble']

    return None


# ============================================================================
# WEEKLY RANKING MESSAGES
# ============================================================================

def get_weekly_ranking_message(row: pd.Series) -> Optional[str]:
    """Generate message about weekly scoring percentile."""
    opp_pts_week_pct = _safe_get(row, 'opp_pts_week_pct')
    team_points = _safe_get(row, 'team_points')

    if opp_pts_week_pct is None or team_points is None:
        return None

    # opp_pts_week_pct is % of teams you BEAT, so higher = better
    percentile = opp_pts_week_pct * 100 if opp_pts_week_pct <= 1 else opp_pts_week_pct

    # Don't show message - this is already covered in weekly recap narrative
    return None


# ============================================================================
# PLAYOFF RACE MESSAGES
# ============================================================================

def get_playoff_race_message(row: pd.Series) -> Optional[str]:
    """Generate message about playoff standing."""
    playoff_seed = _safe_get(row, 'playoff_seed_to_date')
    p_playoffs = _safe_get(row, 'p_playoffs')
    p_bye = _safe_get(row, 'p_bye')

    if playoff_seed is None:
        return None

    # Don't add extra messages - the season recap already covers this
    # These create contradictory statements
    return None


# ============================================================================
# SCHEDULE LUCK MESSAGES
# ============================================================================

def get_schedule_luck_message(row: pd.Series) -> Optional[str]:
    """Generate message about schedule luck vs expected wins."""
    wins_to_date = _safe_get(row, 'wins_to_date')
    exp_final_wins = _safe_get(row, 'exp_final_wins')

    if wins_to_date is None or exp_final_wins is None:
        return None

    # Don't show - this contradicts the narrative already written
    return None


# ============================================================================
# ALTERNATE SCHEDULE MESSAGES
# ============================================================================

def get_alternate_schedule_message(row: pd.Series) -> Optional[str]:
    """Generate message about performance across alternate schedules."""
    shuffle_avg_playoffs = _safe_get(row, 'shuffle_avg_playoffs')
    shuffle_avg_wins = _safe_get(row, 'shuffle_avg_wins')
    wins_to_date = _safe_get(row, 'wins_to_date', 0)

    if shuffle_avg_playoffs is None:
        return None

    # Convert to percentage if needed
    playoff_pct = shuffle_avg_playoffs * 100 if shuffle_avg_playoffs <= 1 else shuffle_avg_playoffs

    if playoff_pct >= 80:
        template = CONTEXTUAL_MESSAGES['alternate_schedules']['playoffs_most']
    elif playoff_pct >= 40:
        template = CONTEXTUAL_MESSAGES['alternate_schedules']['playoffs_some']
    else:
        template = CONTEXTUAL_MESSAGES['alternate_schedules']['playoffs_few']

    msg = format_dialogue(template, shuffle_avg_playoffs=shuffle_avg_playoffs)

    # Add wins context
    if shuffle_avg_wins is not None:
        wins_template = CONTEXTUAL_MESSAGES['alternate_schedules']['avg_wins']
        msg += " " + format_dialogue(wins_template,
                                     shuffle_avg_wins=shuffle_avg_wins,
                                     wins_to_date=wins_to_date)

    return msg


# ============================================================================
# PERFORMANCE GRADE MESSAGES
# ============================================================================

def get_grade_message(row: pd.Series) -> Optional[str]:
    """Generate message about weekly performance grade."""
    grade = _safe_get(row, 'grade')
    gpa = _safe_get(row, 'gpa')

    if grade is None or gpa is None:
        return None

    grade_str = str(grade).upper()

    if grade_str in ['A+', 'A']:
        if grade_str == 'A+':
            template = CONTEXTUAL_MESSAGES['performance_grade']['a_plus']
        else:
            template = CONTEXTUAL_MESSAGES['performance_grade']['a_range']
    elif grade_str.startswith('B'):
        template = CONTEXTUAL_MESSAGES['performance_grade']['b_range']
    elif grade_str.startswith('C'):
        template = CONTEXTUAL_MESSAGES['performance_grade']['c_range']
    elif grade_str.startswith('D'):
        template = CONTEXTUAL_MESSAGES['performance_grade']['d_range']
    else:
        template = CONTEXTUAL_MESSAGES['performance_grade']['f_range']

    return format_dialogue(template, grade=grade, gpa=gpa)


# ============================================================================
# MILESTONE DETECTION
# ============================================================================

def get_milestone_message(row: pd.Series, prev_row: Optional[pd.Series] = None) -> Optional[str]:
    """Detect and generate milestone messages."""
    wins = _safe_get(row, 'wins_to_date', 0)
    losses = _safe_get(row, 'losses_to_date', 0)
    champion = _safe_get(row, 'champion')
    sacko = _safe_get(row, 'sacko')
    team_made_playoffs = _safe_get(row, 'team_made_playoffs')

    # Championship
    if _safe_compare(champion, 1):
        return CONTEXTUAL_MESSAGES['milestones']['championship']

    # Sacko
    if _safe_compare(sacko, 1):
        return CONTEXTUAL_MESSAGES['milestones']['sacko_claim']

    # First win
    if wins == 1 and losses >= 0:
        return CONTEXTUAL_MESSAGES['milestones']['first_win']

    # .500 record
    if wins == losses and wins > 0:
        template = CONTEXTUAL_MESSAGES['milestones']['500_record']
        return format_dialogue(template, wins_to_date=wins, losses_to_date=losses)

    # Playoff clinch (would need prev_row to detect change)
    if _safe_compare(team_made_playoffs, 1) and prev_row is not None:
        prev_playoffs = _safe_get(prev_row, 'team_made_playoffs')
        if not _safe_compare(prev_playoffs, 1):
            return CONTEXTUAL_MESSAGES['milestones']['clinch_playoffs']

    return None


# ============================================================================
# POWER RANKING MESSAGES
# ============================================================================

def get_power_ranking_message(row: pd.Series) -> Optional[str]:
    """Generate message about FELO/power rankings."""
    felo_score = _safe_get(row, 'felo_score')
    felo_tier = _safe_get(row, 'felo_tier')
    power_rating = _safe_get(row, 'power_rating')

    if power_rating is None:
        return None

    if felo_score is not None and felo_tier is not None:
        template = CONTEXTUAL_MESSAGES['power_rankings']['elite']
        return format_dialogue(template,
                              felo_score=felo_score,
                              felo_tier=felo_tier,
                              power_rating=power_rating)

    # Could add rising/falling detection with prev_row comparison
    return None


# ============================================================================
# PLAYER PERCENTILE MESSAGES
# ============================================================================

def get_player_percentile_message(player_row: pd.Series) -> Optional[str]:
    """Generate message about player's percentile performance."""
    # Look for percentile columns in player data
    percentile = None
    position = _safe_get(player_row, 'fantasy_position', _safe_get(player_row, 'position', 'PLAYER'))
    player_name = _safe_get(player_row, 'player', _safe_get(player_row, 'player_name', 'Player'))

    # Check for various percentile columns
    for col in player_row.index:
        if 'percentile' in str(col).lower():
            val = _safe_get(player_row, col)
            if val is not None:
                percentile = val
                break

    if percentile is None:
        return None

    # Convert to 0-100 scale if needed
    pct_value = percentile * 100 if percentile <= 1 else percentile

    # Select message based on percentile
    if pct_value >= 90:
        template = PLAYER_PERFORMANCE['percentile_performance']['elite_week']
    elif pct_value >= 75:
        template = PLAYER_PERFORMANCE['percentile_performance']['great_week']
    elif pct_value >= 50:
        template = PLAYER_PERFORMANCE['percentile_performance']['solid_week']
    elif pct_value >= 30:
        template = PLAYER_PERFORMANCE['percentile_performance']['mediocre_week']
    elif pct_value >= 10:
        template = PLAYER_PERFORMANCE['percentile_performance']['poor_week']
    else:
        template = PLAYER_PERFORMANCE['percentile_performance']['awful_week']

    return format_dialogue(template,
                          player=player_name,
                          position=position,
                          percentile=pct_value)


# ============================================================================
# PLAYER SPOTLIGHT: History + #1 flags -> robust short lines
# ============================================================================

def build_player_spotlight_lines(player_row: pd.Series) -> List[str]:
    """Return concise lines highlighting standout context for a player's week.

    Focuses on columns provided in the request, especially the *_percentile
    variants. Only surfaces notable (>=75th percentile) items and #1 flags.
    """
    if player_row is None or not isinstance(player_row, pd.Series):
        return []

    name = _safe_get(player_row, 'player', _safe_get(player_row, 'player_name', 'Player'))
    position = _safe_get(player_row, 'fantasy_position', _safe_get(player_row, 'position', 'PLAYER'))
    points = _detect_points_value(player_row)

    # Map: percentile column -> (template_key, extra context)
    pct_map = {
        'manager_player_all_time_history_percentile': ('mgr_player_all_time_top', {}),
        'manager_player_season_history_percentile': ('mgr_player_season_top', {}),
        'manager_position_all_time_history_percentile': ('mgr_pos_all_time_top', {'position': position}),
        'manager_position_season_history_percentile': ('mgr_pos_season_top', {'position': position}),
        'player_personal_all_time_history_percentile': ('player_personal_all_time_top', {}),
        'player_personal_season_history_percentile': ('player_personal_season_top', {}),
        'position_all_time_history_percentile': ('league_pos_all_time_top', {'position': position}),
        'position_season_history_percentile': ('league_pos_season_top', {'position': position}),
        'player_all_time_history_percentile': ('league_player_all_time_top', {}),
        'player_season_history_percentile': ('league_player_season_top', {}),
    }

    lines: List[str] = []

    # Percentile-driven highlights
    for col, (tpl_key, extra) in pct_map.items():
        if col in player_row.index:
            pct = _to_percent_0_100(_safe_get(player_row, col))
            bucket = _top_bucket_from_percentile(pct)
            if bucket:
                template = PLAYER_PERFORMANCE['history_highlights'][tpl_key]
                params = {'player': name, 'position': position, 'bucket': bucket}
                params.update(extra)
                lines.append(format_dialogue(template, **params))

    # #1 flags
    def _truthy(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and not pd.isna(v):
            return int(v) == 1
        if isinstance(v, str):
            return v.strip().lower() in {'1', 'true', 't', 'yes', 'y'}
        return False

    n1_templates = PLAYER_PERFORMANCE['number_one_player']
    if _truthy(_safe_get(player_row, 'number_one_player')):
        # Only include points if we have a numeric
        if points is not None:
            lines.append(format_dialogue(n1_templates['league_week'], player=name, points=points))
        else:
            lines.append(format_dialogue(n1_templates['team_week'], player=name))
    if _truthy(_safe_get(player_row, 'number_one_player_season')):
        lines.append(format_dialogue(n1_templates['league_season'], player=name))
    if _truthy(_safe_get(player_row, 'number_one_position')):
        lines.append(format_dialogue(n1_templates['position_week'], player=name, position=position))
    if _truthy(_safe_get(player_row, 'number_one_position_season')):
        lines.append(format_dialogue(n1_templates['position_season'], player=name, position=position))

    # If nothing yet, try a generic percentile line
    if not lines:
        gen = get_player_percentile_message(player_row)
        if gen:
            lines.append(gen)

    return lines


def _pick_best_percentile(player_row: pd.Series) -> Optional[float]:
    """Return the highest available percentile across known player percentile columns (0-100 scale)."""
    cand_cols = [
        'manager_player_all_time_history_percentile',
        'manager_player_season_history_percentile',
        'manager_position_all_time_history_percentile',
        'manager_position_season_history_percentile',
        'player_personal_all_time_history_percentile',
        'player_personal_season_history_percentile',
        'position_all_time_history_percentile',
        'position_season_history_percentile',
        'player_all_time_history_percentile',
        'player_season_history_percentile',
    ]
    best = None
    for c in cand_cols:
        if c in player_row.index:
            pct = _to_percent_0_100(_safe_get(player_row, c))
            if pct is not None:
                best = max(best, pct) if best is not None else pct
    return best


# ============================================================================
# AGGREGATE ALL CONTEXTUAL MESSAGES
# ============================================================================

def generate_all_contextual_messages(row: pd.Series, prev_row: Optional[pd.Series] = None) -> Dict[str, Optional[str]]:
    """Generate all available contextual messages for a row."""
    return {
        'optimal_lineup': get_optimal_lineup_message(row),
        'streak': get_streak_message(row),
        'weekly_ranking': get_weekly_ranking_message(row),
        'playoff_race': get_playoff_race_message(row),
        'schedule_luck': get_schedule_luck_message(row),
        'alternate_schedule': get_alternate_schedule_message(row),
        'grade': get_grade_message(row),
        'milestone': get_milestone_message(row, prev_row),
        'power_ranking': get_power_ranking_message(row),
    }


# -------------------------
# Weekly fine-tuned paragraph
# -------------------------

def compose_weekly_paragraph(row: pd.Series) -> Optional[str]:
    """Compose a richer weekly narrative using margin, spread, projections, totals, etc."""
    if row is None:
        return None

    def g(k, d=None):
        return _safe_get(row, k, d)

    team = g('team_name', g('manager', 'Your team'))
    opp  = g('opponent', 'your opponent')
    tp   = _safe_float(g('team_points'))
    op   = _safe_float(g('opponent_points'))
    margin = _safe_float(g('margin'))
    spread = _safe_float(g('expected_spread'))
    odds   = _safe_float(g('expected_odds'))
    proj   = _safe_float(g('team_projected_points'))
    abs_err = _safe_float(g('abs_proj_score_error'))
    total  = _safe_float(g('total_matchup_score', (tp or 0) + (op or 0)))
    league_mean = _safe_float(g('league_weekly_mean', g('weekly_mean')))
    league_med  = _safe_float(g('league_weekly_median', g('weekly_median')))
    underdog_w  = g('underdog_wins')
    favorite_l  = g('favorite_losses')
    covered     = g('win_vs_spread')

    bits: List[str] = []

    # Game style
    if total is not None and league_mean is not None:
        if total >= league_mean + 30:
            bits.append("It turned into a full-on shootout.")
        elif total <= max(0.0, league_mean - 25):
            bits.append("This one was a low-scoring slog.")

    # Margin context
    if margin is not None:
        am = abs(margin)
        if am >= 40:
            bits.append(f"An absolute demolition by {am:.1f}.")
        elif am >= 25:
            bits.append(f"A comfortable {am:.1f}-point decision.")
        elif am >= 10:
            bits.append(f"A solid {am:.1f}-point result.")
        else:
            bits.append(f"Tight one decided by {am:.1f}.")

    # Spread/upset context
    if spread is not None:
        if (spread < 0 and margin is not None and margin < 0) or (spread > 0 and margin is not None and margin > 0):
            # Favorite won or underdog lost
            if covered in (1, True, '1', 'true'):
                bits.append("You covered the spread.")
            else:
                bits.append("You failed to cover.")
        else:
            # Opposite of expectation
            if underdog_w in (1, True, '1', 'true'):
                bits.append("You pulled the upset.")
            elif favorite_l in (1, True, '1', 'true'):
                bits.append("A favorite went down.")

    # Projection context
    if tp is not None and proj is not None:
        delta = tp - proj
        if delta >= 20:
            bits.append(f"Crushed projections by {delta:.1f}.")
        elif delta >= 8:
            bits.append(f"Beat projections by {delta:.1f}.")
        elif delta <= -20:
            bits.append(f"Way under projections by {abs(delta):.1f}.")
        elif delta <= -8:
            bits.append(f"Missed projections by {abs(delta):.1f}.")

    # Median context
    abl = g('above_league_median')
    if abl in (1, True, '1', 'true'):
        bits.append("Scored above the league median.")
    elif abl in (0, False, '0', 'false'):
        bits.append("Fell below the league median.")

    # Teams beaten metric
    tb = _safe_float(g('teams_beat_this_week'))
    if tb is not None:
        if tb >= 10:
            bits.append(f"Would've beaten {int(tb)} other teams.")
        elif tb >= 6:
            bits.append(f"Better than {int(tb)} peers this week.")
        elif tb <= 1:
            bits.append("Only topped a team or two.")

    # GPA/grade flair
    grade = g('grade'); gpa = _safe_float(g('gpa'))
    if isinstance(grade, str) and gpa is not None:
        if grade.upper() == 'A+':
            bits.append("A+ execution.")
        elif grade.upper().startswith('A'):
            bits.append("Honor-roll effort.")
        elif grade.upper().startswith('B'):
            bits.append("Respectable work.")
        elif grade.upper().startswith('C'):
            bits.append("Middle of the pack.")
        elif grade.upper().startswith('D'):
            bits.append("On thin ice.")
        else:
            bits.append("That'll leave a mark.")

    # Collapse into sentence
    core = []
    if tp is not None and op is not None:
        core.append(f"Final was {tp:.1f}-{op:.1f} vs {opp}.")
    if odds is not None:
        pct = odds*100 if 0 <= odds <= 1 else odds
        core.append(f"Pre-game win odds were {pct:.0f}%.")

    sentence = " ".join(core + bits)
    return sentence or None


# -------------------------
# Season fine-tuned paragraph
# -------------------------

def compose_season_paragraph(row: pd.Series) -> Optional[str]:
    if row is None:
        return None
    def g(k, d=None): return _safe_get(row, k, d)

    wins = _safe_float(g('wins_to_date')); losses = _safe_float(g('losses_to_date'))
    seed = _safe_float(g('playoff_seed_to_date'))
    p_playoffs = _safe_float(g('p_playoffs'))
    p_bye      = _safe_float(g('p_bye'))
    p_champ    = _safe_float(g('p_champ'))
    exp_wins   = _safe_float(g('exp_final_wins'))
    shuffle_pct = _safe_float(g('shuffle_avg_playoffs'))
    wins_vs_shuffle = _safe_float(g('wins_vs_shuffle_wins'))
    power = _safe_float(g('power_rating'))
    felo  = _safe_float(g('felo_score'))
    tier  = g('felo_tier')

    lines: List[str] = []
    if wins is not None and losses is not None:
        if seed is not None:
            lines.append(f"You're {int(wins)}-{int(losses)} and sit at seed #{int(seed)} right now.")
        else:
            lines.append(f"You're {int(wins)}-{int(losses)} so far.")

    def pct_str(v):
        if v is None: return None
        p = v*100 if 0 <= v <= 1 else v
        return f"{p:.0f}%"

    # Odds & power
    if p_playoffs is not None:
        lines.append(f"Playoff odds sit at {pct_str(p_playoffs)}{'' if p_bye is None else f', with a {pct_str(p_bye)} shot at a bye' }.")
    if p_champ is not None:
        lines.append(f"Title odds are {pct_str(p_champ)}.")
    if power is not None:
        if power >= 80:
            lines.append("Power rating says you're a juggernaut.")
        elif power >= 60:
            lines.append("Firmly above average by power rating.")
        elif power < 40:
            lines.append("Power rating flags some concerns.")
    if felo is not None and tier is not None:
        lines.append(f"FELO {felo:.0f} ({tier} tier).")

    # Schedule luck vs shuffled
    if wins_vs_shuffle is None:
        # compute if possible
        sav = _safe_float(g('shuffle_avg_wins'))
        if sav is not None and wins is not None:
            wins_vs_shuffle = wins - sav
    if wins_vs_shuffle is not None:
        if wins_vs_shuffle >= 1.0:
            lines.append(f"Schedule has gifted about {wins_vs_shuffle:.1f} extra wins.")
        elif wins_vs_shuffle <= -1.0:
            lines.append(f"Schedule has cost you about {abs(wins_vs_shuffle):.1f} wins.")
        else:
            lines.append("Record aligns with schedule-adjusted expectation.")

    if shuffle_pct is not None:
        lines.append(f"Across shuffled schedules, you'd make the playoffs {pct_str(shuffle_pct)} of the time.")

    return " ".join(lines) or None


# -------------------------
# Player spotlight paragraph
# -------------------------

def build_player_spotlight_paragraph(
    week_rows: pd.DataFrame,
    *,
    points_col: str,
    improved_row: Optional[pd.Series] = None,
    max_players: int = 3,
) -> Optional[str]:
    """Compose a short paragraph highlighting standout player performances.

    Inputs:
    - week_rows: DataFrame of the selected manager's players for the chosen week.
    - points_col: Column name containing player points for the week.
    - improved_row: Optional Series for "Most Improved" computed across two-week slice.
    - max_players: Max number of top performers to include in the intro.

    Returns a Markdown-friendly paragraph or None if insufficient data.
    """
    if week_rows is None or not isinstance(week_rows, pd.DataFrame) or week_rows.empty:
        return None
    if not points_col or points_col not in week_rows.columns:
        return None

    df = week_rows.copy()
    # Coerce points to numeric and drop NA
    df[points_col] = pd.to_numeric(df[points_col], errors="coerce")
    df = df[pd.notna(df[points_col])]
    if df.empty:
        return None

    # Try to find a reasonable player-name column
    name_candidates = [
        'player', 'Player', 'player_name', 'Player_Name', 'name', 'Name'
    ]
    name_col = next((c for c in name_candidates if c in df.columns), None)
    if not name_col:
        # Fallback: first object dtype column
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        name_col = obj_cols[0] if obj_cols else df.columns[0]

    # Try to exclude non-starters if lineup position is available
    pos_cols = [
        'fantasy_position', 'fantasy pos', 'fantasyposition', 'fantasy_pos',
        'lineup_slot', 'lineup_position', 'slot',
        'roster_slot', 'Roster_Slot', 'roster_position', 'player_fantasy_position'
    ]
    lineup_col = next((c for c in pos_cols if c in df.columns), None)
    if lineup_col:
        df = df[~df[lineup_col].astype(str).str.upper().isin({"BN", "IR"})]

    if df.empty:
        return None

    # Pick top N performers by points
    top_df = df.sort_values(points_col, ascending=False).head(max(1, int(max_players)))

    # Build intro list: Name (X pts)
    intro_parts: List[str] = []
    per_player_lines: List[str] = []

    for idx, row in top_df.iterrows():
        name = str(_safe_get(row, name_col, 'Player'))
        pts = _detect_points_value(row)
        if pts is None:
            pts = _safe_float(row.get(points_col))
        pts_txt = f"{pts:.1f}" if isinstance(pts, (int, float)) and not pd.isna(pts) else "—"
        intro_parts.append(f"{name} ({pts_txt} pts)")

        # Gather concise context lines for this player
        lines = build_player_spotlight_lines(row)
        if lines:
            # Give up to 2 lines for the very top player, else 1
            keep = 2 if len(per_player_lines) == 0 else 1
            per_player_lines.extend(lines[:keep])

    # Compose intro sentence
    if not intro_parts:
        return None

    if len(intro_parts) == 1:
        intro = f"Top performer this week: {intro_parts[0]}."
    elif len(intro_parts) == 2:
        intro = f"Top performers: {intro_parts[0]} and {intro_parts[1]}."
    else:
        intro = f"Top performers: {', '.join(intro_parts[:-1])}, and {intro_parts[-1]}."

    # Add Most Improved context if available
    tail_bits: List[str] = []
    if isinstance(improved_row, pd.Series) and not improved_row.empty:
        imp_name = str(_safe_get(improved_row, name_col, _safe_get(improved_row, 'player', 'A player')))
        delta = _safe_float(improved_row.get('__improvement_delta__'))
        if delta is None:
            # Try to infer from two-week points if present
            cur_pts = _safe_float(improved_row.get(points_col))
            prev_pts = _safe_float(improved_row.get('prev_points'))
            if cur_pts is not None and prev_pts is not None:
                delta = cur_pts - prev_pts
        if delta is not None and delta > 0:
            tail_bits.append(f"Most Improved: {imp_name} (+{delta:.1f} pts week-over-week).")

    # Stitch everything
    parts: List[str] = [intro]
    if per_player_lines:
        parts.append(" ".join(per_player_lines))
    if tail_bits:
        parts.append(" ".join(tail_bits))

    paragraph = " ".join(p.strip() for p in parts if p and p.strip())
    return paragraph or None

