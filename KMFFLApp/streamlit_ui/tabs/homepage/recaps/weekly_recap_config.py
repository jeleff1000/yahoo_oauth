"""
WEEKLY RECAP CONFIGURATION
===========================
Edit this file to customize the weekly recap narrative.

HOW IT WORKS:
- Each section below is a list of criteria
- The system checks each criterion's "condition" function
- If True, that text gets added to the recap
- Only ONE item per "category" is used (first match wins)
- Items are processed in the order they appear

To edit text: Just change the "text" field
To change when something appears: Edit the "condition" lambda
To add new criteria: Add a new dict to the appropriate section
"""

from typing import Dict, Any, Optional, List
import pandas as pd


# =============================================================================
# HELPER FUNCTIONS FOR CONDITIONS
# =============================================================================

def _safe_get(row, key: str, default=None):
    """Safely get value from row."""
    if row is None:
        return default
    if isinstance(row, pd.Series):
        val = row.get(key, default)
        return default if pd.isna(val) else val
    if isinstance(row, dict):
        val = row.get(key, default)
        return default if (isinstance(val, float) and pd.isna(val)) else val
    return default


def _to_float(v, default=None) -> Optional[float]:
    """Convert to float safely."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    except:
        return default


def _to_int(v, default=None) -> Optional[int]:
    """Convert to int safely."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return int(float(v))
    except:
        return default


def _is_win(row) -> bool:
    """Check if this was a win."""
    win = _safe_get(row, 'win')
    if win in (1, True, '1', 'true', 'True'):
        return True
    return False


def _is_loss(row) -> bool:
    """Check if this was a loss."""
    return not _is_win(row)


def _margin(row) -> float:
    """Get absolute margin."""
    return abs(_to_float(_safe_get(row, 'margin'), 0))


def _teams_beat(row) -> int:
    """Get teams beat this week."""
    return _to_int(_safe_get(row, 'teams_beat_this_week'), 0)


# =============================================================================
# SECTION 1: CORE RESULT
# =============================================================================
# The opening sentence about the final score

RESULT_TEMPLATES = {
    "win": "The final score of your matchup against {opponent} was ({team_points:.1f} - {opponent_points:.1f}).",
    "loss": "The final score of your matchup against {opponent} was ({team_points:.1f} - {opponent_points:.1f}).",
}

# What to add after the score based on game type
RESULT_FLAVOR = [
    # ===================
    # CHAMPIONSHIP GAME
    # ===================
    {
        "condition": lambda r: _safe_get(r, 'champion') == 1,
        "text": "You did it! LEAGUE CHAMPION!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_championship') == 1 and _is_loss(r),
        "text": "So close to glory. Runner-up isn't bad, but it's not the trophy.",
        "category": "result_flavor",
    },

    # ===================
    # PLAYOFFS - CLOSE GAMES
    # ===================
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_win(r) and _margin(r) < 1,
        "text": "A playoff win by less than a point?! Your heart must've stopped!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_win(r) and _margin(r) < 3,
        "text": "Playoff thriller! Won by the skin of your teeth!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_win(r) and _margin(r) < 5,
        "text": "Way to win a close one in the playoffs! On to the next round!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_loss(r) and _margin(r) < 1,
        "text": "Lost your playoff game by less than a point. That's gonna sting for a while.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_loss(r) and _margin(r) < 3,
        "text": "Playoff heartbreaker. Lost by just {margin_abs:.1f} points. So close.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_loss(r) and _margin(r) < 5,
        "text": "Lost your playoff game in a nail-biter. Better luck next year!",
        "category": "result_flavor",
    },

    # ===================
    # PLAYOFFS - BLOWOUTS
    # ===================
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_win(r) and _margin(r) >= 40,
        "text": "Playoff demolition! You made a statement with that one!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_win(r) and _margin(r) >= 25,
        "text": "Dominant playoff win! Never in doubt!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_loss(r) and _margin(r) >= 40,
        "text": "Got absolutely demolished in the playoffs. Ouch.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_loss(r) and _margin(r) >= 25,
        "text": "Rough playoff exit. They had your number.",
        "category": "result_flavor",
    },

    # ===================
    # PLAYOFFS - DEFAULT
    # ===================
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_win(r),
        "text": "Playoff W! Moving on to the next round!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_playoffs') == 1 and _is_loss(r),
        "text": "Tough playoff loss. Season's over, but you made it this far!",
        "category": "result_flavor",
    },

    # ===================
    # REGULAR SEASON - SUPER CLOSE (< 1 point)
    # ===================
    {
        "condition": lambda r: _is_win(r) and _margin(r) < 1,
        "text": "Won by less than a point! That's the kind of win you remember forever!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and _margin(r) < 1,
        "text": "Lost by less than a point. That one's gonna haunt you.",
        "category": "result_flavor",
    },

    # ===================
    # REGULAR SEASON - CLOSE (1-3 points)
    # ===================
    {
        "condition": lambda r: _is_win(r) and _margin(r) < 3,
        "text": "Survived by the slimmest of margins! A win's a win!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and _margin(r) < 3,
        "text": "So close! Lost by just {margin_abs:.1f} points. Brutal.",
        "category": "result_flavor",
    },

    # ===================
    # REGULAR SEASON - NAIL-BITER (3-5 points)
    # ===================
    {
        "condition": lambda r: _is_win(r) and _margin(r) < 5,
        "text": "Way to win a nail-biter!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and _margin(r) < 5,
        "text": "Tough loss. You'll get 'em next time!",
        "category": "result_flavor",
    },

    # ===================
    # REGULAR SEASON - COMFORTABLE (5-15 points)
    # ===================
    {
        "condition": lambda r: _is_win(r) and 5 <= _margin(r) < 10,
        "text": "Solid win! Never really in danger.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_win(r) and 10 <= _margin(r) < 15,
        "text": "Comfortable victory! You handled business.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and 5 <= _margin(r) < 10,
        "text": "Clear loss but not embarrassing. Regroup and move on.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and 10 <= _margin(r) < 15,
        "text": "Got outplayed this week. Back to the drawing board.",
        "category": "result_flavor",
    },

    # ===================
    # REGULAR SEASON - DOMINANT (15-30 points)
    # ===================
    {
        "condition": lambda r: _is_win(r) and 15 <= _margin(r) < 20,
        "text": "Dominant performance! They didn't stand a chance!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_win(r) and 20 <= _margin(r) < 30,
        "text": "You rolled over them! That's a statement win!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and 15 <= _margin(r) < 20,
        "text": "Got beat pretty handily. Shake it off.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and 20 <= _margin(r) < 30,
        "text": "Rough week. They had your number from the jump.",
        "category": "result_flavor",
    },

    # ===================
    # REGULAR SEASON - BLOWOUTS (30+ points)
    # ===================
    {
        "condition": lambda r: _is_win(r) and 30 <= _margin(r) < 40,
        "text": "What a blowout! Total domination!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_win(r) and 40 <= _margin(r) < 50,
        "text": "Absolute destruction! That's gotta be one for the record books!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_win(r) and _margin(r) >= 50,
        "text": "MASSACRE! A 50+ point win?! Did they even field a team?!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and 30 <= _margin(r) < 40,
        "text": "Yikes, that was rough. Got blown out.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and 40 <= _margin(r) < 50,
        "text": "Oof. Got absolutely steamrolled this week.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and _margin(r) >= 50,
        "text": "A 50+ point loss? Let's just pretend this week didn't happen.",
        "category": "result_flavor",
    },

    # ===================
    # SACKO GAME
    # ===================
    {
        "condition": lambda r: _safe_get(r, 'sacko') == 1,
        "text": "The Sacko is yours. Congrats... I guess?",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_sacko_game') == 1 and _is_win(r),
        "text": "You escaped the Sacko! Someone else gets to hold that shame!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_sacko_game') == 1 and _is_loss(r),
        "text": "Lost the Sacko game. Enjoy last place all offseason!",
        "category": "result_flavor",
    },

    # ===================
    # CONSOLATION BRACKET
    # ===================
    {
        "condition": lambda r: _safe_get(r, 'is_consolation') == 1 and _is_win(r) and _margin(r) >= 20,
        "text": "Dominated in the consolation bracket! Playing for pride!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_consolation') == 1 and _is_win(r),
        "text": "Consolation W! At least you're winning something!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _safe_get(r, 'is_consolation') == 1 and _is_loss(r),
        "text": "Lost in the consolation bracket. There's always next year.",
        "category": "result_flavor",
    },

    # ===================
    # WEEK 1 SPECIAL
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'week'), 0) == 1 and _is_win(r) and _margin(r) >= 20,
        "text": "What a way to start the season! A dominant Week 1 W!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'week'), 0) == 1 and _is_win(r),
        "text": "Starting the season with a W! Let's go!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'week'), 0) == 1 and _is_loss(r) and _margin(r) >= 20,
        "text": "Rough way to kick off the season. Getting blown out in Week 1 is never fun.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'week'), 0) == 1 and _is_loss(r),
        "text": "Starting the season 0-1. Plenty of time to turn it around.",
        "category": "result_flavor",
    },

    # ===================
    # HIGH SCORING GAMES
    # ===================
    {
        "condition": lambda r: _is_win(r) and _to_float(_safe_get(r, 'team_points'), 0) >= 180,
        "text": "180+ points?! That's a video game score!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_win(r) and _to_float(_safe_get(r, 'team_points'), 0) >= 160,
        "text": "160+ points! Your team went OFF!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_win(r) and _to_float(_safe_get(r, 'team_points'), 0) >= 150,
        "text": "150+ points! Everyone showed up!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and _to_float(_safe_get(r, 'team_points'), 0) >= 150 and _to_float(_safe_get(r, 'opponent_points'), 0) >= 150,
        "text": "A 150+ point shootout and you came up short! What a game!",
        "category": "result_flavor",
    },

    # ===================
    # LOW SCORING GAMES
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _to_float(_safe_get(r, 'team_points'), 0) < 80,
        "text": "Under 80 points? Did your team even show up?",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r) and _to_float(_safe_get(r, 'team_points'), 0) < 90,
        "text": "Under 90 points. Rough week all around.",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_win(r) and _to_float(_safe_get(r, 'team_points'), 0) < 100 and _to_float(_safe_get(r, 'opponent_points'), 0) < 100,
        "text": "Neither team hit 100 but you survived! A win's a win!",
        "category": "result_flavor",
    },

    # ===================
    # DEFAULT FALLBACKS
    # ===================
    {
        "condition": lambda r: _is_win(r),
        "text": "Nice win!",
        "category": "result_flavor",
    },
    {
        "condition": lambda r: _is_loss(r),
        "text": "Better luck next week.",
        "category": "result_flavor",
    },
]


# =============================================================================
# SECTION 2: LEAGUE CONTEXT
# =============================================================================
# How you compared to the rest of the league this week

LEAGUE_CONTEXT = [
    # ===================
    # TOP SCORER
    # ===================
    {
        "condition": lambda r: _is_win(r) and _teams_beat(r) >= 9,
        "text": "You were the top scorer in the league this week! Would've beaten everyone!",
        "category": "luck",
    },
    {
        "condition": lambda r: _is_loss(r) and _teams_beat(r) >= 9,
        "text": "You had the highest score in the league and STILL LOST?! That's the unluckiest thing I've ever seen!",
        "category": "luck",
    },

    # ===================
    # VERY LUCKY WINS (beat 0-2 teams)
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_league_median') == 0 and _teams_beat(r) == 0,
        "text": "Luckiest win of all time! You had the lowest score in the league and still won! Buy a lottery ticket!",
        "category": "luck",
    },
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_league_median') == 0 and _teams_beat(r) == 1,
        "text": "Lucky win! You would've lost to 8 other teams this week. Somehow it still worked out!",
        "category": "luck",
    },
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_league_median') == 0 and _teams_beat(r) == 2,
        "text": "Lucky win! You would've only beaten 2 teams this week. The schedule was kind to you!",
        "category": "luck",
    },
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_league_median') == 0 and _teams_beat(r) <= 3,
        "text": "Lucky win! You would've only beaten {teams_beat_this_week} teams this week and still won!",
        "category": "luck",
    },

    # ===================
    # VERY UNLUCKY LOSSES (beat 7-8 teams)
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'above_league_median') == 1 and _teams_beat(r) >= 8,
        "text": "Heartbreaking! You would've beaten 8 other teams this week and STILL lost! The schedule gods hate you.",
        "category": "luck",
    },
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'above_league_median') == 1 and _teams_beat(r) == 7,
        "text": "Unlucky! You would've beaten 7 teams this week and still lost! Fantasy is cruel sometimes.",
        "category": "luck",
    },
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'above_league_median') == 1 and _teams_beat(r) >= 6,
        "text": "Unlucky! You would have beaten {teams_beat_this_week} teams this week and still lost!",
        "category": "luck",
    },

    # ===================
    # DOMINANT WINS (beat 7-8 teams)
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_league_median') == 1 and _teams_beat(r) >= 8,
        "text": "You dominated the league this week! Would have beaten {teams_beat_this_week} other teams!",
        "category": "luck",
    },
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_league_median') == 1 and _teams_beat(r) == 7,
        "text": "Great week! You would've beaten 7 teams. You earned this W!",
        "category": "luck",
    },

    # ===================
    # SOLID WINS (beat 5-6 teams)
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_league_median') == 1 and _teams_beat(r) >= 5,
        "text": "You deserved this one! Would have beaten {teams_beat_this_week} other teams.",
        "category": "luck",
    },

    # ===================
    # AVERAGE WINS (beat 4-5 teams, above median)
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_league_median') == 1,
        "text": "Above the league median this week. Solid performance.",
        "category": "luck",
    },

    # ===================
    # DESERVED LOSSES (beat 0-3 teams)
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'above_league_median') == 0 and _teams_beat(r) == 0,
        "text": "Can't really complain about this one. You had the lowest score in the league.",
        "category": "luck",
    },
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'above_league_median') == 0 and _teams_beat(r) <= 2,
        "text": "Can't blame the schedule. You only would've beaten {teams_beat_this_week} teams this week.",
        "category": "luck",
    },
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'above_league_median') == 0,
        "text": "Below the league median this week. Need to find more points.",
        "category": "luck",
    },

    # ===================
    # SOMEWHAT UNLUCKY LOSSES (beat 4-5 teams)
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _teams_beat(r) >= 4,
        "text": "A bit unlucky. You would've beaten {teams_beat_this_week} teams but ran into the wrong opponent.",
        "category": "luck",
    },

    # ===================
    # SECOND HIGHEST SCORER
    # ===================
    {
        "condition": lambda r: _is_win(r) and _teams_beat(r) == 8,
        "text": "Second highest score in the league! Great week all around.",
        "category": "luck",
    },

    # ===================
    # BOTTOM DWELLER WINS
    # ===================
    {
        "condition": lambda r: _is_win(r) and _teams_beat(r) <= 3 and _margin(r) >= 10,
        "text": "Won by double digits despite a mediocre score. Your opponent had a ROUGH week.",
        "category": "luck",
    },

    # ===================
    # HIGH SCORER LOSSES
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _teams_beat(r) >= 5,
        "text": "Would've beaten {teams_beat_this_week} other teams. You just ran into a buzzsaw this week.",
        "category": "luck",
    },
]


# =============================================================================
# SECTION 3: PROJECTION ANALYSIS
# =============================================================================
# How you performed vs projections and spread

PROJECTION_CONTEXT = [
    # ===================
    # DOMINANT WIN - Beat projection, favorite, covered spread
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_proj_score') == 1 and _safe_get(r, 'proj_wins') == 1 and _safe_get(r, 'win_vs_spread') == 1 and _margin(r) >= 20,
        "text": "Flawless execution! You crushed your projection, covered the spread, and won by {margin_abs:.1f}. This is what dominance looks like.",
        "category": "projection",
    },
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_proj_score') == 1 and _safe_get(r, 'proj_wins') == 1 and _safe_get(r, 'win_vs_spread') == 1,
        "text": "You exceeded expectations. You had a {expected_odds_pct}% chance of winning and delivered, beating your projection by {proj_score_error_abs:.1f} points.",
        "category": "projection",
    },

    # ===================
    # FAVORITE WIN - But didn't cover spread
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_proj_score') == 1 and _safe_get(r, 'proj_wins') == 1 and _safe_get(r, 'win_vs_spread') == 0,
        "text": "Got the W but didn't cover the spread. You beat your projection by {proj_score_error_abs:.1f} but {opponent} kept it closer than expected.",
        "category": "projection",
    },

    # ===================
    # UNDERDOG UPSET - Big upset
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'proj_wins') == 0 and _safe_get(r, 'above_proj_score') == 1 and _to_float(_safe_get(r, 'expected_odds'), 50) < 30,
        "text": "GIANT SLAYER! You only had a {expected_odds_pct}% chance and pulled off a massive upset! Exceeded projections by {proj_score_error_abs:.1f} too!",
        "category": "projection",
    },
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'proj_wins') == 0 and _safe_get(r, 'above_proj_score') == 1,
        "text": "Everyone doubted you! You pulled off the upset despite only having a {expected_odds_pct}% chance, exceeding your projection by {proj_score_error_abs:.1f}.",
        "category": "projection",
    },

    # ===================
    # UNDERDOG UPSET - Didn't even beat projection
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'proj_wins') == 0 and _safe_get(r, 'above_proj_score') == 0,
        "text": "Stole one! You missed your projection by {proj_score_error_abs:.1f} but {opponent} was even worse. Sometimes it's better to be lucky than good.",
        "category": "projection",
    },

    # ===================
    # FAVORITE WIN - But underperformed projection
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_proj_score') == 0 and _safe_get(r, 'proj_wins') == 1 and _margin(r) < 3,
        "text": "Did you and {opponent} agree to take it easy this week? You missed your projected score by {proj_score_error_abs:.1f} but still squeaked out a {margin_abs:.2f} point victory. Way to survive and advance.",
        "category": "projection",
    },
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'above_proj_score') == 0 and _safe_get(r, 'proj_wins') == 1,
        "text": "Ugly win but a win nonetheless. Missed your projection by {proj_score_error_abs:.1f} but got the job done.",
        "category": "projection",
    },

    # ===================
    # HEARTBREAKER LOSS - Favorite who choked
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'proj_wins') == 1 and _safe_get(r, 'above_proj_score') == 0 and _to_float(_safe_get(r, 'expected_odds'), 50) > 70,
        "text": "Devastating. You were a {expected_odds_pct}% favorite and completely collapsed, missing your projection by {proj_score_error_abs:.1f}. That's a tough pill to swallow.",
        "category": "projection",
    },
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'proj_wins') == 1 and _safe_get(r, 'above_proj_score') == 0,
        "text": "This one hurts. You had a {expected_odds_pct}% chance of winning but missed your projection by {proj_score_error_abs:.1f} and lost by {margin_abs:.2f}.",
        "category": "projection",
    },

    # ===================
    # FAVORITE LOSS - Beat projection but still lost
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'proj_wins') == 1 and _safe_get(r, 'above_proj_score') == 1,
        "text": "You beat your projection by {proj_score_error_abs:.1f} and STILL lost?! {opponent} went nuclear. Nothing you could do.",
        "category": "projection",
    },

    # ===================
    # UNDERDOG LOSS - Fought hard
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'proj_wins') == 0 and _safe_get(r, 'above_proj_score') == 1 and _margin(r) < 5,
        "text": "Noble effort! You exceeded your projection by {proj_score_error_abs:.1f} and almost pulled off the upset. So close!",
        "category": "projection",
    },
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'proj_wins') == 0 and _safe_get(r, 'above_proj_score') == 1,
        "text": "Good showing! You exceeded your projection by {proj_score_error_abs:.1f} but {opponent}'s best was still better.",
        "category": "projection",
    },

    # ===================
    # UNDERDOG LOSS - Didn't show up
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'proj_wins') == 0 and _safe_get(r, 'above_proj_score') == 0 and _margin(r) >= 20,
        "text": "Never had a chance. You were the underdog, missed your projection by {proj_score_error_abs:.1f}, and got blown out. On to next week.",
        "category": "projection",
    },
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'proj_wins') == 0 and _safe_get(r, 'above_proj_score') == 0,
        "text": "It was an uphill battle and you missed your projection by {proj_score_error_abs:.1f}. Back to the drawing board.",
        "category": "projection",
    },
]


# =============================================================================
# SECTION 4: STREAKS
# =============================================================================
# Winning/losing streak context

STREAK_CONTEXT = [
    # ===================
    # WINNING STREAKS
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'winning_streak'), 0) >= 8,
        "text": "{winning_streak} straight wins! Are you cheating?! This is insane!",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'winning_streak'), 0) == 7,
        "text": "7 in a row! You're officially the hottest team in the league!",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'winning_streak'), 0) == 6,
        "text": "6 straight wins! Nobody can stop you right now!",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'winning_streak'), 0) == 5,
        "text": "5 straight wins! You're on FIRE!",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'winning_streak'), 0) == 4,
        "text": "4-game winning streak! You're rolling!",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'winning_streak'), 0) == 3,
        "text": "You're riding a 3-game winning streak. Keep it rolling!",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'winning_streak'), 0) == 2,
        "text": "Back-to-back wins! A streak is born.",
        "category": "streak",
    },

    # ===================
    # LOSING STREAKS
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'losing_streak'), 0) >= 7,
        "text": "{losing_streak} straight losses?! This is a nightmare. Wake up!",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'losing_streak'), 0) == 6,
        "text": "6 losses in a row. At this point, what's one more? (Please make it stop)",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'losing_streak'), 0) == 5,
        "text": "5 straight losses. Rock bottom has to be close... right?",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'losing_streak'), 0) == 4,
        "text": "{losing_streak} straight losses. This is a full-blown crisis!",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'losing_streak'), 0) == 3,
        "text": "You've lost 3 in a row. Time to shake things up.",
        "category": "streak",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'losing_streak'), 0) == 2,
        "text": "Two straight losses. Time to stop the bleeding.",
        "category": "streak",
    },

    # ===================
    # STREAK BREAKERS
    # ===================
    {
        "condition": lambda r: _is_win(r) and _to_int(_safe_get(r, 'prev_losing_streak'), 0) >= 5,
        "text": "FINALLY! You snapped a brutal {prev_losing_streak}-game losing streak! Freedom!",
        "category": "streak",
    },
    {
        "condition": lambda r: _is_win(r) and _to_int(_safe_get(r, 'prev_losing_streak'), 0) >= 3,
        "text": "Streak breaker! You snapped a {prev_losing_streak}-game losing streak!",
        "category": "streak",
    },
    {
        "condition": lambda r: _is_win(r) and _to_int(_safe_get(r, 'prev_losing_streak'), 0) == 2,
        "text": "Good to get back in the win column after 2 straight losses.",
        "category": "streak",
    },
    {
        "condition": lambda r: _is_loss(r) and _to_int(_safe_get(r, 'prev_winning_streak'), 0) >= 5,
        "text": "The dream is over! Your incredible {prev_winning_streak}-game winning streak comes to an end.",
        "category": "streak",
    },
    {
        "condition": lambda r: _is_loss(r) and _to_int(_safe_get(r, 'prev_winning_streak'), 0) >= 3,
        "text": "Ouch! Your {prev_winning_streak}-game winning streak is over.",
        "category": "streak",
    },
    {
        "condition": lambda r: _is_loss(r) and _to_int(_safe_get(r, 'prev_winning_streak'), 0) == 2,
        "text": "Back-to-back wins came to an end. Can't win 'em all.",
        "category": "streak",
    },

    # ===================
    # BOUNCE BACK / COLLAPSE
    # ===================
    {
        "condition": lambda r: _is_win(r) and _safe_get(r, 'prev_loss') == 1 and _margin(r) >= 20,
        "text": "Talk about a bounce-back week! After last week's loss, you dominated!",
        "category": "streak",
    },
    {
        "condition": lambda r: _is_loss(r) and _safe_get(r, 'prev_win') == 1 and _margin(r) >= 20,
        "text": "After winning last week, you came crashing back to earth. Humbling.",
        "category": "streak",
    },
]


# =============================================================================
# SECTION 5: OPTIMAL LINEUP
# =============================================================================
# Points left on bench analysis

def _optimal_diff(row) -> float:
    """Calculate points left on bench."""
    optimal = _to_float(_safe_get(row, 'optimal_points'), 0)
    team = _to_float(_safe_get(row, 'team_points'), 0)
    return optimal - team if optimal and team else 0


OPTIMAL_CONTEXT = [
    # ===================
    # PERFECT LINEUP
    # ===================
    {
        "condition": lambda r: _optimal_diff(r) == 0,
        "text": "PERFECT LINEUP! You started your absolute best team. Chef's kiss!",
        "category": "optimal",
    },
    {
        "condition": lambda r: _optimal_diff(r) < 1,
        "text": "Basically perfect lineup. Less than a point left on the bench. Excellent management!",
        "category": "optimal",
    },

    # ===================
    # LOSS - COULD HAVE WON WITH OPTIMAL
    # ===================
    {
        "condition": lambda r: _is_loss(r) and _optimal_diff(r) > _margin(r) and _margin(r) < 3,
        "text": "BRUTAL! You lost by {margin_abs:.1f} with {optimal_diff:.1f} points on your bench. One different call and you win this!",
        "category": "optimal",
    },
    {
        "condition": lambda r: _is_loss(r) and _optimal_diff(r) > _margin(r),
        "text": "You lost by {margin_abs:.1f} but had {optimal_diff:.1f} points on your bench. A lineup change would've swung the matchup. That stings.",
        "category": "optimal",
    },

    # ===================
    # WIN - MINOR BENCH POINTS (1-5)
    # ===================
    {
        "condition": lambda r: _is_win(r) and 1 <= _optimal_diff(r) < 3,
        "text": "Great lineup management! Only {optimal_diff:.1f} points left on the bench and you got the W!",
        "category": "optimal",
    },
    {
        "condition": lambda r: _is_win(r) and 3 <= _optimal_diff(r) < 5,
        "text": "Pretty close to optimal! Only {optimal_diff:.1f} points left on the bench - and you still came away with the W!",
        "category": "optimal",
    },
    {
        "condition": lambda r: _is_win(r) and 5 <= _optimal_diff(r) < 8,
        "text": "Left {optimal_diff:.1f} on the bench but it didn't matter. A win's a win!",
        "category": "optimal",
    },

    # ===================
    # LOSS - MINOR BENCH POINTS (1-5)
    # ===================
    {
        "condition": lambda r: _is_loss(r) and 1 <= _optimal_diff(r) < 3,
        "text": "Only {optimal_diff:.1f} points left on the bench. You did everything right, just got beat.",
        "category": "optimal",
    },
    {
        "condition": lambda r: _is_loss(r) and 3 <= _optimal_diff(r) < 5,
        "text": "Pretty close to optimal - only {optimal_diff:.1f} points left on the bench. Sometimes you just run into a buzzsaw.",
        "category": "optimal",
    },
    {
        "condition": lambda r: _is_loss(r) and 5 <= _optimal_diff(r) < 8,
        "text": "Left {optimal_diff:.1f} on the bench. Might not have mattered, but every point counts.",
        "category": "optimal",
    },

    # ===================
    # SIGNIFICANT BENCH POINTS (8-15)
    # ===================
    {
        "condition": lambda r: _is_win(r) and 8 <= _optimal_diff(r) < 15,
        "text": "You left {optimal_diff:.1f} points on your bench but still won. Imagine if you'd gotten that lineup right!",
        "category": "optimal",
    },
    {
        "condition": lambda r: _is_loss(r) and 8 <= _optimal_diff(r) < 15,
        "text": "You left {optimal_diff:.1f} points on your bench. Better lineup decisions could've changed the outcome.",
        "category": "optimal",
    },

    # ===================
    # MAJOR BENCH POINTS (15-25)
    # ===================
    {
        "condition": lambda r: _is_win(r) and 15 <= _optimal_diff(r) < 25,
        "text": "Whoa - {optimal_diff:.1f} points left on the bench?! Good thing you won anyway!",
        "category": "optimal",
    },
    {
        "condition": lambda r: _is_loss(r) and 15 <= _optimal_diff(r) < 25,
        "text": "You left {optimal_diff:.1f} points on the bench this week! That's a lot of missed potential.",
        "category": "optimal",
    },

    # ===================
    # MASSIVE BENCH POINTS (25+)
    # ===================
    {
        "condition": lambda r: _is_win(r) and _optimal_diff(r) >= 25,
        "text": "{optimal_diff:.1f} POINTS left on the bench?! How did you even win this game?!",
        "category": "optimal",
    },
    {
        "condition": lambda r: _is_loss(r) and _optimal_diff(r) >= 25,
        "text": "You left {optimal_diff:.1f} points on the bench! That's inexcusable. Gotta pay more attention to those lineups!",
        "category": "optimal",
    },
]


# =============================================================================
# SECTION 6: PERFORMANCE GRADE
# =============================================================================
# Weekly grade/GPA context

GRADE_CONTEXT = [
    # A+
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'A+',
        "text": "A+ grade this week with a {gpa:.2f} GPA. Absolute perfection!",
        "category": "grade",
    },
    # A
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'A',
        "text": "A grade this week (GPA: {gpa:.2f}). Dean's list performance!",
        "category": "grade",
    },
    # A-
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'A-',
        "text": "A- grade (GPA: {gpa:.2f}). Excellent work, minor room for improvement.",
        "category": "grade",
    },
    # B+
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'B+',
        "text": "B+ grade this week. GPA of {gpa:.2f}. Solid effort, almost great!",
        "category": "grade",
    },
    # B
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'B',
        "text": "B grade this week (GPA: {gpa:.2f}). Above average but not elite.",
        "category": "grade",
    },
    # B-
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'B-',
        "text": "B- grade. GPA of {gpa:.2f}. Room to improve but respectable.",
        "category": "grade",
    },
    # C+
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'C+',
        "text": "C+ grade (GPA: {gpa:.2f}). Slightly above average. Not bad, not great.",
        "category": "grade",
    },
    # C
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'C',
        "text": "C grade. GPA of {gpa:.2f}. Perfectly average. Meh.",
        "category": "grade",
    },
    # C-
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'C-',
        "text": "C- grade (GPA: {gpa:.2f}). Below average. You can do better.",
        "category": "grade",
    },
    # D+
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'D+',
        "text": "D+ grade. GPA of {gpa:.2f}. Barely passing. Step it up!",
        "category": "grade",
    },
    # D
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'D',
        "text": "D grade (GPA: {gpa:.2f}). You're on thin ice.",
        "category": "grade",
    },
    # D-
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'D-',
        "text": "D- grade. GPA of {gpa:.2f}. Academic probation territory.",
        "category": "grade",
    },
    # F
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper() == 'F',
        "text": "F. GPA: {gpa:.2f}. Complete failure. Time to hit the books!",
        "category": "grade",
    },
    # Generic fallbacks
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper().startswith('A'),
        "text": "{grade} grade this week (GPA: {gpa:.2f}). Excellent!",
        "category": "grade",
    },
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper().startswith('B'),
        "text": "{grade} grade this week. GPA of {gpa:.2f}. Good work!",
        "category": "grade",
    },
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper().startswith('C'),
        "text": "{grade} grade. GPA of {gpa:.2f}. Average performance.",
        "category": "grade",
    },
    {
        "condition": lambda r: str(_safe_get(r, 'grade', '')).upper().startswith('D'),
        "text": "{grade} grade. GPA of {gpa:.2f}. Below expectations.",
        "category": "grade",
    },
]


# =============================================================================
# SECTION 7: MILESTONES & ACHIEVEMENTS
# =============================================================================
# Special achievements to highlight

MILESTONE_CONTEXT = [
    # ===================
    # MAJOR ACHIEVEMENTS
    # ===================
    {
        "condition": lambda r: _safe_get(r, 'champion') == 1,
        "text": "LEAGUE CHAMPION! You did it! The trophy is yours!",
        "category": "milestone",
        "emoji": "🏆",
        "style": "success",
    },
    {
        "condition": lambda r: _safe_get(r, 'sacko') == 1,
        "text": "Congratulations? You've secured last place and the Sacko. Wear it with... shame.",
        "category": "milestone",
        "emoji": "💩",
        "style": "error",
    },
    {
        "condition": lambda r: _safe_get(r, 'runner_up') == 1,
        "text": "Runner-up! So close to the championship. Silver medal isn't bad!",
        "category": "milestone",
        "emoji": "🥈",
        "style": "info",
    },
    {
        "condition": lambda r: _safe_get(r, 'third_place') == 1,
        "text": "Third place finish! You made it to the podium!",
        "category": "milestone",
        "emoji": "🥉",
        "style": "info",
    },

    # ===================
    # PLAYOFF CLINCH
    # ===================
    {
        "condition": lambda r: _safe_get(r, 'clinched_playoffs') == 1 and _to_int(_safe_get(r, 'playoff_seed_to_date'), 99) <= 2,
        "text": "You've clinched a playoff spot AND a first-round bye! Rest up for the semis!",
        "category": "playoff_clinch",
        "emoji": "🎫",
        "style": "success",
    },
    {
        "condition": lambda r: _safe_get(r, 'clinched_playoffs') == 1,
        "text": "Playoffs clinched! You're officially in the postseason!",
        "category": "playoff_clinch",
        "emoji": "🎫",
        "style": "success",
    },

    # ===================
    # FIRST WIN
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'wins_to_date'), 0) == 1 and _is_win(r),
        "text": "Your first win of the season! Feels good, doesn't it?",
        "category": "first_win",
        "emoji": "🎉",
        "style": "success",
    },

    # ===================
    # .500 MILESTONES
    # ===================
    {
        "condition": lambda r: _is_win(r) and _to_int(_safe_get(r, 'wins_to_date'), 0) == _to_int(_safe_get(r, 'losses_to_date'), -1) and _to_int(_safe_get(r, 'wins_to_date'), 0) > 1,
        "text": "Back to .500! All square at {wins_to_date}-{losses_to_date}.",
        "category": "record_milestone",
        "emoji": "⚖️",
        "style": "info",
    },
    {
        "condition": lambda r: _is_win(r) and _to_int(_safe_get(r, 'wins_to_date'), 0) == _to_int(_safe_get(r, 'losses_to_date'), -1) + 1,
        "text": "Above .500 for the first time! {wins_to_date}-{losses_to_date} looks good on you!",
        "category": "record_milestone",
        "emoji": "📈",
        "style": "success",
    },

    # ===================
    # PLAYOFF POSITION
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'playoff_seed_to_date'), 99) == 1,
        "text": "You're the #1 seed! Top of the mountain!",
        "category": "playoff_position",
        "emoji": "👑",
        "style": "success",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'playoff_seed_to_date'), 99) == 2,
        "text": "Holding the #2 seed. Bye week territory!",
        "category": "playoff_position",
        "emoji": "🎯",
        "style": "info",
    },
    {
        "condition": lambda r: _safe_get(r, 'team_made_playoffs') == 1 and _to_int(_safe_get(r, 'playoff_seed_to_date'), 99) <= 6,
        "text": "You're in playoff position at the #{playoff_seed_to_date} seed!",
        "category": "playoff_position",
        "emoji": "✅",
        "style": "info",
    },

    # ===================
    # BUBBLE/OUTSIDE
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'playoff_seed_to_date'), 99) == 7,
        "text": "On the outside looking in at #7. One spot away from the playoffs!",
        "category": "playoff_position",
        "emoji": "😰",
        "style": "warning",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'playoff_seed_to_date'), 99) >= 8,
        "text": "Currently at #{playoff_seed_to_date}. Gonna need to make a push for the playoffs.",
        "category": "playoff_position",
        "emoji": "📉",
        "style": "warning",
    },

    # ===================
    # ELIMINATION / MUST-WIN
    # ===================
    {
        "condition": lambda r: _safe_get(r, 'must_win') == 1 and _is_win(r),
        "text": "A must-win game, and you delivered! Season saved... for now.",
        "category": "must_win",
        "emoji": "💪",
        "style": "success",
    },
    {
        "condition": lambda r: _safe_get(r, 'must_win') == 1 and _is_loss(r),
        "text": "This was a must-win and you lost. Playoff hopes are on life support.",
        "category": "must_win",
        "emoji": "💔",
        "style": "error",
    },
    {
        "condition": lambda r: _safe_get(r, 'eliminated') == 1,
        "text": "Mathematically eliminated from playoff contention. It's over.",
        "category": "elimination",
        "emoji": "❌",
        "style": "error",
    },

    # ===================
    # LATE SEASON POSITIONING
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'week'), 0) >= 11 and _to_int(_safe_get(r, 'playoff_seed_to_date'), 99) <= 2 and _is_win(r),
        "text": "Holding strong in bye week position with the regular season winding down!",
        "category": "late_season",
        "emoji": "🎯",
        "style": "success",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'week'), 0) >= 11 and _to_int(_safe_get(r, 'playoff_seed_to_date'), 99) <= 6 and _safe_get(r, 'prev_playoff_seed') and _to_int(_safe_get(r, 'prev_playoff_seed'), 99) > 6 and _is_win(r),
        "text": "Climbed into playoff position in the final weeks! Perfect timing!",
        "category": "late_season",
        "emoji": "📈",
        "style": "success",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'week'), 0) >= 11 and _to_int(_safe_get(r, 'playoff_seed_to_date'), 99) > 6 and _safe_get(r, 'prev_playoff_seed') and _to_int(_safe_get(r, 'prev_playoff_seed'), 99) <= 6 and _is_loss(r),
        "text": "Dropped out of playoff position in the final weeks. Terrible timing!",
        "category": "late_season",
        "emoji": "📉",
        "style": "error",
    },

    # ===================
    # WINLESS / UNDEFEATED
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'losses_to_date'), 0) == 0 and _to_int(_safe_get(r, 'week'), 0) >= 4 and _is_win(r),
        "text": "Still undefeated! The perfect season continues!",
        "category": "perfect_season",
        "emoji": "🏅",
        "style": "success",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'wins_to_date'), 0) == 0 and _to_int(_safe_get(r, 'week'), 0) >= 4 and _is_loss(r),
        "text": "Still searching for that elusive first win...",
        "category": "winless",
        "emoji": "😢",
        "style": "error",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'losses_to_date'), 0) == 1 and _is_loss(r) and _to_int(_safe_get(r, 'week'), 0) >= 5,
        "text": "First loss of the season! The perfect record is gone.",
        "category": "first_loss",
        "emoji": "😤",
        "style": "warning",
    },
]


# =============================================================================
# SECTION 8: HISTORICAL PERFORMANCE
# =============================================================================
# All-time/season rankings

HISTORICAL_CONTEXT = [
    # ===================
    # CAREER BESTS
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'manager_all_time_ranking'), 0) == 1,
        "text": "CAREER HIGH! This was your #1 highest-scoring week of all time!",
        "category": "historical",
        "emoji": "🔥",
        "style": "success",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'manager_all_time_ranking'), 0) == 2,
        "text": "This was your 2nd highest-scoring week EVER! Almost set the record!",
        "category": "historical",
        "emoji": "🔥",
        "style": "success",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'manager_all_time_ranking'), 0) <= 5 and _to_int(_safe_get(r, 'manager_all_time_ranking'), 0) > 0,
        "text": "Top 5 all-time! This was your #{manager_all_time_ranking} best week ever!",
        "category": "historical",
        "emoji": "🔥",
        "style": "success",
    },

    # ===================
    # TOP PERCENTILE
    # ===================
    {
        "condition": lambda r: _to_float(_safe_get(r, 'manager_all_time_percentile'), 0) >= 99,
        "text": "Elite performance! Top 1% of all your career weeks!",
        "category": "historical",
        "emoji": "💎",
        "style": "success",
    },
    {
        "condition": lambda r: _to_float(_safe_get(r, 'manager_all_time_percentile'), 0) >= 95,
        "text": "Outstanding! This was a top 5% performance in your career!",
        "category": "historical",
        "emoji": "⭐",
        "style": "success",
    },
    {
        "condition": lambda r: _to_float(_safe_get(r, 'manager_all_time_percentile'), 0) >= 90,
        "text": "Top 10% of your career weeks! Great performance!",
        "category": "historical",
        "emoji": "📈",
        "style": "info",
    },

    # ===================
    # BOTTOM PERCENTILE
    # ===================
    {
        "condition": lambda r: _to_float(_safe_get(r, 'manager_all_time_percentile'), 100) <= 1,
        "text": "Career low! This was your worst week of all time. Nowhere to go but up!",
        "category": "historical",
        "emoji": "💀",
        "style": "error",
    },
    {
        "condition": lambda r: _to_float(_safe_get(r, 'manager_all_time_percentile'), 100) <= 5,
        "text": "Rough week! This was in the bottom 5% of your career. Ouch!",
        "category": "historical",
        "emoji": "😬",
        "style": "error",
    },
    {
        "condition": lambda r: _to_float(_safe_get(r, 'manager_all_time_percentile'), 100) <= 10,
        "text": "Bottom 10% of your career weeks. Not your finest moment.",
        "category": "historical",
        "emoji": "😐",
        "style": "warning",
    },

    # ===================
    # SEASON RANKINGS
    # ===================
    {
        "condition": lambda r: _to_int(_safe_get(r, 'manager_season_ranking'), 0) == 1,
        "text": "This was your BEST week of the season! New season high!",
        "category": "season_rank",
        "emoji": "🌟",
        "style": "success",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'manager_season_ranking'), 0) == 2,
        "text": "Second-best week of your season! You're heating up!",
        "category": "season_rank",
        "emoji": "📈",
        "style": "info",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'manager_season_ranking'), 0) <= 3,
        "text": "Top 3 scoring week of your season! Keep it rolling!",
        "category": "season_rank",
        "emoji": "✨",
        "style": "info",
    },
    {
        "condition": lambda r: _to_int(_safe_get(r, 'manager_season_ranking'), 99) >= 13,
        "text": "This was one of your worst weeks of the season (#{manager_season_ranking}). Shake it off.",
        "category": "season_rank",
        "emoji": "📉",
        "style": "warning",
    },
]


# =============================================================================
# BUILD FUNCTION - Assembles the recap from config
# =============================================================================

def evaluate_criteria(criteria_list: List[Dict], row: Dict, max_per_category: int = 1) -> List[Dict]:
    """
    Evaluate criteria against row data.
    Returns list of matching criteria (first match per category).
    """
    matches = []
    seen_categories = set()

    for criterion in criteria_list:
        try:
            condition = criterion.get("condition")
            if condition and condition(row):
                category = criterion.get("category", "default")

                # Check category limit
                cat_count = sum(1 for m in matches if m.get("category") == category)
                if cat_count >= max_per_category:
                    continue

                seen_categories.add(category)
                matches.append(criterion)
        except Exception:
            continue

    return matches


def format_template(template: str, row: Dict) -> str:
    """Format template with row values."""
    # Build safe dict with all values
    safe_dict = {}

    if isinstance(row, pd.Series):
        safe_dict = row.to_dict()
    elif isinstance(row, dict):
        safe_dict = dict(row)

    # Add computed values
    margin = _to_float(_safe_get(row, 'margin'), 0)
    safe_dict['margin_abs'] = abs(margin) if margin else 0

    proj_err = _to_float(_safe_get(row, 'proj_score_error'), 0)
    safe_dict['proj_score_error_abs'] = abs(proj_err) if proj_err else 0

    optimal = _to_float(_safe_get(row, 'optimal_points'), 0)
    team = _to_float(_safe_get(row, 'team_points'), 0)
    safe_dict['optimal_diff'] = optimal - team if optimal and team else 0

    # Expected odds as percentage
    odds = _to_float(_safe_get(row, 'expected_odds'), 0)
    if odds and 0 <= odds <= 1:
        safe_dict['expected_odds_pct'] = int(odds * 100)
    else:
        safe_dict['expected_odds_pct'] = int(odds) if odds else 0

    # Manager all-time top percentage
    pct = _to_float(_safe_get(row, 'manager_all_time_percentile'), 0)
    safe_dict['manager_all_time_top_pct'] = 100 - pct if pct else 0

    try:
        return template.format(**safe_dict)
    except KeyError as e:
        # Return with placeholder for missing key
        return template.replace("{" + str(e).strip("'") + "}", "[N/A]")
    except Exception:
        return template


# =============================================================================
# QUICK REFERENCE: View all criteria in a table format
# =============================================================================

def print_criteria_table():
    """Print a readable table of all criteria for quick reference."""
    sections = [
        ("RESULT FLAVOR", RESULT_FLAVOR),
        ("LEAGUE CONTEXT", LEAGUE_CONTEXT),
        ("PROJECTION", PROJECTION_CONTEXT),
        ("STREAKS", STREAK_CONTEXT),
        ("OPTIMAL LINEUP", OPTIMAL_CONTEXT),
        ("GRADE", GRADE_CONTEXT),
        ("MILESTONES", MILESTONE_CONTEXT),
        ("HISTORICAL", HISTORICAL_CONTEXT),
    ]

    print("\n" + "=" * 100)
    print("WEEKLY RECAP CRITERIA REFERENCE")
    print("=" * 100)

    total = 0
    for section_name, criteria_list in sections:
        print(f"\n{'-' * 100}")
        print(f"  {section_name} ({len(criteria_list)} criteria)")
        print(f"{'-' * 100}")

        for i, c in enumerate(criteria_list, 1):
            category = c.get('category', 'default')
            text = c.get('text', '')[:65]
            print(f"  {i:2}. [{category}] {text}...")

        total += len(criteria_list)

    print("\n" + "=" * 100)
    print(f"TOTAL: {total} configurable criteria")
    print("=" * 100)
