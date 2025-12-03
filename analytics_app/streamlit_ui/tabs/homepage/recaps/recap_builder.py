"""
RECAP PARAGRAPH BUILDER
=======================
Builds flowing paragraph text from the config tables.
No colored boxes, no bubbles - just clean text.
"""

from typing import Dict, List, Any, Optional
import pandas as pd

from .recap_config import (
    WEEKLY_RESULT_TEMPLATES,
    WEEKLY_CONTEXT_CRITERIA,
    WEEKLY_MILESTONE_CRITERIA,
    SEASON_RECORD_TEMPLATE,
    SEASON_CONTEXT_CRITERIA,
    PLAYER_INTRO_TEMPLATE,
    PLAYER_HIGHLIGHT_CRITERIA,
)


def _safe_get(row: Dict, key: str, default=None):
    """Safely get a value from row dict or pandas Series."""
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default) if row else default


def _format_template(template: str, row: Dict) -> str:
    """Format a template string with row values, handling missing keys gracefully."""
    # Build a safe dict with computed values
    safe_dict = dict(row) if not isinstance(row, pd.Series) else row.to_dict()

    # Add computed helper values
    safe_dict['margin'] = abs(_safe_get(row, 'margin', 0) or 0)
    safe_dict['winning_streak'] = _safe_get(row, 'winning_streak') or _safe_get(row, 'win_streak') or 0
    safe_dict['losing_streak'] = _safe_get(row, 'losing_streak') or _safe_get(row, 'loss_streak') or 0
    safe_dict['p_playoffs_pct'] = int((_safe_get(row, 'p_playoffs') or 0) * 100)
    safe_dict['p_champ_pct'] = int((_safe_get(row, 'p_champ') or 0) * 100)
    safe_dict['wins_vs_shuffle_wins_abs'] = abs(_safe_get(row, 'wins_vs_shuffle_wins') or 0)

    try:
        return template.format(**safe_dict)
    except KeyError as e:
        # Return template with missing key noted
        return template.replace("{" + str(e).strip("'") + "}", "[?]")
    except Exception:
        return template


def _evaluate_criteria(criteria_list: List[Dict], row: Dict) -> List[str]:
    """Evaluate a list of criteria against a row and return matching text snippets."""
    matches = []
    seen_categories = set()

    for criterion in criteria_list:
        try:
            condition = criterion.get("condition")
            if condition and condition(row):
                category = criterion.get("category")
                # Only take first match per category (avoid redundant statements)
                if category and category in seen_categories:
                    continue
                if category:
                    seen_categories.add(category)

                text = _format_template(criterion["text"], row)
                matches.append({
                    "text": text,
                    "standalone": criterion.get("standalone", False),
                })
        except Exception:
            # Skip criteria that fail to evaluate
            continue

    return matches


# =============================================================================
# WEEKLY RECAP BUILDER
# =============================================================================

def build_weekly_recap(row: Dict) -> str:
    """
    Build a weekly recap paragraph from matchup data.

    Returns a flowing paragraph like:
    "John defeated Mike 142.5 to 128.3, a comfortable 14.2-point victory while
    outperforming projections. This extends their winning streak to 4 games.
    They currently sit in playoff position at the #2 seed."
    """
    parts = []

    # 1. Core result sentence
    win = _safe_get(row, 'win')
    if win == 1:
        result = _format_template(WEEKLY_RESULT_TEMPLATES["win"], row)
    else:
        result = _format_template(WEEKLY_RESULT_TEMPLATES["loss"], row)

    # 2. Gather context snippets
    context_matches = _evaluate_criteria(WEEKLY_CONTEXT_CRITERIA, row)

    # Separate inline context from standalone sentences
    inline_context = [m["text"] for m in context_matches if not m.get("standalone")]
    standalone_context = [m["text"] for m in context_matches if m.get("standalone")]

    # 3. Build the main sentence with inline context
    if inline_context:
        # Join context with commas and "and"
        if len(inline_context) == 1:
            context_str = inline_context[0]
        elif len(inline_context) == 2:
            context_str = f"{inline_context[0]} and {inline_context[1]}"
        else:
            context_str = ", ".join(inline_context[:-1]) + f", and {inline_context[-1]}"

        main_sentence = f"{result}, {context_str}."
    else:
        main_sentence = f"{result}."

    parts.append(main_sentence)

    # 4. Add standalone context as separate sentences
    for ctx in standalone_context:
        if not ctx.endswith("."):
            ctx += "."
        parts.append(ctx)

    # 5. Add milestone sentences
    milestone_matches = _evaluate_criteria(WEEKLY_MILESTONE_CRITERIA, row)
    for m in milestone_matches:
        text = m["text"]
        if not text.endswith("."):
            text += "."
        parts.append(text)

    return " ".join(parts)


# =============================================================================
# SEASON RECAP BUILDER
# =============================================================================

def build_season_recap(row: Dict) -> str:
    """
    Build a season recap paragraph from matchup data.

    Returns a flowing paragraph like:
    "John sits at 8-4, currently in playoff position at #2 with a 85% chance
    to make playoffs and a 12% shot at the title. They've benefited from a
    favorable schedule, winning 1.5 more games than expected."
    """
    parts = []

    # 1. Record sentence
    record = _format_template(SEASON_RECORD_TEMPLATE, row)

    # 2. Gather context snippets
    context_matches = _evaluate_criteria(SEASON_CONTEXT_CRITERIA, row)

    # Separate inline from standalone
    inline_context = [m["text"] for m in context_matches if not m.get("standalone")]
    standalone_context = [m["text"] for m in context_matches if m.get("standalone")]

    # 3. Build main sentence
    if inline_context:
        context_str = ", ".join(inline_context)
        main_sentence = f"{record}, {context_str}."
    else:
        main_sentence = f"{record}."

    parts.append(main_sentence)

    # 4. Add standalone sentences
    for ctx in standalone_context:
        text = _format_template(ctx, row) if "{" in ctx else ctx
        if not text.endswith("."):
            text += "."
        parts.append(text)

    return " ".join(parts)


# =============================================================================
# PLAYER RECAP BUILDER
# =============================================================================

def build_player_recap(players_df: pd.DataFrame, top_n: int = 3) -> str:
    """
    Build a player recap paragraph from player data.

    Args:
        players_df: DataFrame with player stats for the week
        top_n: Number of top performers to highlight

    Returns a flowing paragraph like:
    "Top performers this week: Patrick Mahomes led the way with 28.5 points.
    Derrick Henry (RB) contributed 22.3, and Davante Adams (WR) added 18.7.
    QB: 28.5 pts, RB: 38.2 pts, WR: 52.1 pts, TE: 8.4 pts."
    """
    if players_df is None or players_df.empty:
        return "No player data available."

    parts = [PLAYER_INTRO_TEMPLATE]

    # Get top scorers
    points_col = None
    for col in ['points', 'player_points', 'fantasy_points']:
        if col in players_df.columns:
            points_col = col
            break

    if not points_col:
        return "No player scoring data available."

    # Sort by points and get top performers
    sorted_df = players_df.sort_values(points_col, ascending=False)
    top_players = sorted_df.head(top_n)

    # Build top performer sentences
    player_sentences = []
    for i, (_, player) in enumerate(top_players.iterrows()):
        name = player.get('player', player.get('player_name', 'Unknown'))
        points = player.get(points_col, 0)
        position = player.get('fantasy_position', player.get('position', ''))

        if i == 0:
            player_sentences.append(f"{name} led the way with {points:.2f} points")
        else:
            if position:
                player_sentences.append(f"{name} ({position}) contributed {points:.2f}")
            else:
                player_sentences.append(f"{name} added {points:.2f}")

    if player_sentences:
        # Join with commas and period
        if len(player_sentences) == 1:
            parts.append(f"{player_sentences[0]}.")
        elif len(player_sentences) == 2:
            parts.append(f"{player_sentences[0]}, and {player_sentences[1]}.")
        else:
            parts.append(f"{player_sentences[0]}, {', '.join(player_sentences[1:-1])}, and {player_sentences[-1]}.")

    # Add position totals if available
    position_col = None
    for col in ['fantasy_position', 'position', 'pos']:
        if col in players_df.columns:
            position_col = col
            break

    if position_col:
        position_totals = []
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_df = players_df[players_df[position_col] == pos]
            if not pos_df.empty:
                total = pos_df[points_col].sum()
                position_totals.append(f"{pos}: {total:.1f} pts")

        if position_totals:
            parts.append(f"Position totals: {', '.join(position_totals)}.")

    return " ".join(parts)


# =============================================================================
# COMBINED RECAP BUILDER
# =============================================================================

def build_full_recap(matchup_row: Dict, players_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Build all three recap paragraphs.

    Returns dict with keys: 'weekly', 'season', 'players'
    """
    return {
        'weekly': build_weekly_recap(matchup_row),
        'season': build_season_recap(matchup_row),
        'players': build_player_recap(players_df) if players_df is not None else "",
    }
