#!/usr/bin/env python3
"""
Weekly Matchup Data V2 - Multi-League Edition

Fetches weekly matchup data from Yahoo Fantasy Football API including:
- Manager scores and opponents
- Win/loss records
- Projection accuracy
- Team metadata (FAAB, moves, trades, grades, etc.)

NOTE: Head-to-head records (w_vs_X, l_vs_X) and playoff flags are NOT calculated here.
They are added by the transformation pipeline (cumulative_stats_v2.py) to maintain
separation of concerns between data fetching and transformation.

Key improvements over V1:
- Multi-league support via LeagueContext
- Manager name overrides from context (no hardcoded names)
- League-specific output paths
- RunLogger integration for structured logging
- Modular helper functions
- Better error handling

Usage:
    # With LeagueContext
    from multi_league.core.league_context import LeagueContext
    ctx = LeagueContext.load("leagues/kmffl/league_context.json")
    df = weekly_matchup_data(ctx, year=2024, week=5)

    # Standalone (backward compatible)
    df = weekly_matchup_data(year=2024, week=5, oauth_file=Path("Oauth.json"))

    # CLI with context
    python weekly_matchup_data_v2.py --context leagues/kmffl/league_context.json --year 2024 --week 5

    # CLI standalone
    python weekly_matchup_data_v2.py --year 2024 --week 5 --oauth Oauth.json
"""

import sys
import os
import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import time

try:
    import yahoo_fantasy_api as yfa
    YFA_AVAILABLE = True
except ImportError:
    YFA_AVAILABLE = False
    yfa = None

# Add paths for imports
_script_file = Path(__file__).resolve()
_multi_league_dir = _script_file.parent.parent  # multi_league directory
sys.path.insert(0, str(_multi_league_dir / "core"))
sys.path.insert(0, str(_multi_league_dir / "utils"))

# Multi-league infrastructure
try:
    from league_context import LeagueContext
    LEAGUE_CONTEXT_AVAILABLE = True
except ImportError:
    LeagueContext = None
    LEAGUE_CONTEXT_AVAILABLE = False

try:
    from run_metadata import RunLogger
    RUN_LOGGER_AVAILABLE = True
except ImportError:
    RunLogger = None
    RUN_LOGGER_AVAILABLE = False

try:
    from oauth_utils import create_oauth2
except ImportError:
    create_oauth2 = None

# Default paths (for standalone mode)
THIS_FILE = Path(__file__).resolve()
SCRIPT_ROOT = THIS_FILE.parent.parent.parent  # Back to scripts root
DEFAULT_DATA_ROOT = SCRIPT_ROOT.parent / "fantasy_football_data" / "matchup_data"

# GPA scale for matchup grades
GPA_SCALE = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "F": 0.0, "F-": 0.0,
}


# =============================================================================
# Helper Functions
# =============================================================================

def norm_manager(nickname: str, overrides: Dict[str, str] = None) -> str:
    """
    Normalize manager name with optional overrides.

    Args:
        nickname: Raw manager nickname from Yahoo API
        overrides: Dictionary mapping old names to new names

    Returns:
        Normalized manager name (title case for consistency)
    """
    if not nickname:
        return "N/A"

    s = str(nickname).strip()

    # Apply overrides if provided (check both original case and title case)
    if overrides:
        if s in overrides:
            return overrides[s]
        if s.title() in overrides:
            return overrides[s.title()]
        if s.lower() in overrides:
            return overrides[s.lower()]

    # Default fallback for --hidden--
    if s == "--hidden--":
        return "Unknown"

    # Normalize to title case for consistency (e.g., "ezra" -> "Ezra")
    return s.title()


def safe_float(text, default=np.nan):
    """Convert text to float, returning default on error."""
    try:
        if text is None:
            return default
        s = str(text).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _first_text(node: ET.Element, paths: List[str]) -> str:
    """Get first non-empty text from list of XML paths."""
    for p in paths:
        v = node.findtext(p)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def _first_float(node: ET.Element, paths: List[str]) -> float:
    """Get first valid float from list of XML paths."""
    for p in paths:
        v = node.findtext(p)
        fv = safe_float(v, np.nan)
        if not (isinstance(fv, float) and np.isnan(fv)):
            return fv
    return np.nan


def derive_playoff_structure(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive playoff structure from league settings.

    Args:
        settings: League settings dictionary

    Returns:
        Dictionary with playoff structure details
    """
    start_week = int(settings.get('start_week') or 1)
    end_week = int(settings.get('end_week') or 17)
    playoff_start_week = int(settings.get('playoff_start_week') or max(end_week - 2, start_week))
    matchup_len = int(settings.get('playoff_matchup_length') or 1)
    num_teams = int(settings.get('num_playoff_teams') or 6)

    if num_teams <= 4:
        rounds = 2
    elif num_teams in (6, 8):
        rounds = 3
    else:
        rounds = 3 if num_teams >= 6 else 2

    qf_weeks, sf_weeks, final_weeks = [], [], []
    wk = playoff_start_week
    if rounds == 3:
        qf_weeks = list(range(wk, wk + matchup_len))
        wk += matchup_len
        sf_weeks = list(range(wk, wk + matchup_len))
        wk += matchup_len
        final_weeks = list(range(wk, wk + matchup_len))
    else:
        sf_weeks = list(range(wk, wk + matchup_len))
        wk += matchup_len
        final_weeks = list(range(wk, wk + matchup_len))

    champion_week = final_weeks[-1] if final_weeks else end_week

    return {
        'start_week': start_week,
        'end_week': end_week,
        'playoff_start_week': playoff_start_week,
        'matchup_len': matchup_len,
        'num_playoff_teams': num_teams,
        'rounds': rounds,
        'qf_weeks': qf_weeks,
        'sf_weeks': sf_weeks,
        'final_weeks': final_weeks,
        'champion_week': champion_week,
    }


def get_weeks_to_fetch(league, year: int, week_input: Optional[int], settings: Dict[str, Any]) -> List[int]:
    """
    Determine which weeks to fetch.

    Args:
        league: Yahoo league object
        year: Season year
        week_input: Specific week (None or 0 = all weeks)
        settings: League settings

    Returns:
        List of week numbers to fetch
    """
    s = derive_playoff_structure(settings)
    start_week, end_week = s['start_week'], s['end_week']

    if week_input and week_input != 0:
        return [week_input]

    # For current year, only fetch completed weeks
    try:
        cw_attr = getattr(league, 'current_week', None)
        current_week = int(cw_attr() if callable(cw_attr) else cw_attr) if cw_attr is not None else None
    except Exception:
        current_week = None

    if (year == datetime.now().year) and current_week and current_week > start_week:
        last = min(end_week, current_week - 1)
        return list(range(start_week, last + 1))

    return list(range(start_week, end_week + 1))


def extract_team(team_node: ET.Element, matchup_node: ET.Element = None, manager_overrides: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Extract team data from XML node.

    Args:
        team_node: XML Element for team
        matchup_node: XML Element for matchup (for grade/felo data)
        manager_overrides: Dictionary of manager name overrides

    Returns:
        Dictionary with team data
    """
    nickname = (
        team_node.findtext(".//managers/manager/nickname")
        or team_node.findtext(".//managers/manager/name")
        or team_node.findtext(".//managers/manager/guid")
        or ""
    )
    manager = norm_manager(nickname, manager_overrides)
    team_name = team_node.findtext("name") or manager

    points = safe_float(team_node.findtext("team_points/total"), 0.0)
    projected = safe_float(team_node.findtext("team_projected_points/total"), np.nan)

    # Try to find grade in multiple locations
    grade = ""
    if matchup_node is not None:
        # Check matchup_grades at matchup level
        grade = _first_text(matchup_node, [
            ".//matchup_grades/matchup_grade/grade",
            "matchup_grades/matchup_grade/grade",
        ])

    # If not found at matchup level, try team level
    if not grade:
        grade = _first_text(team_node, [
            ".//matchup_grade/grade",
            "matchup_grade/grade",
            ".//matchup_grades/matchup_grade/grade",
            "matchup_grades/matchup_grade/grade",
            "grade",
        ])

    gpa = GPA_SCALE.get(grade, np.nan)

    url_ = team_node.findtext("url") or ""
    image_url = team_node.findtext("team_logos/team_logo/url") or ""
    division_id = team_node.findtext("division_id") or ""
    waiver_priority = safe_float(team_node.findtext("waiver_priority"), np.nan)
    faab_balance = safe_float(team_node.findtext("faab_balance"), np.nan)
    number_of_moves = safe_float(team_node.findtext("number_of_moves"), np.nan)
    number_of_trades = safe_float(team_node.findtext("number_of_trades"), np.nan)
    coverage_value = safe_float(team_node.findtext("coverage_value"), np.nan)
    value = safe_float(team_node.findtext("value"), np.nan)
    has_draft_grade = (team_node.findtext("has_draft_grade") or "").strip()

    # Try multiple locations for felo_score and felo_tier
    felo_score = _first_float(team_node, [
        "felo_score",
        ".//team_standings/felo_score",
        ".//team_stats/felo_score",
        "team_standings/felo_score",
        "team_stats/felo_score",
    ])

    felo_tier = _first_text(team_node, [
        "felo_tier",
        ".//team_standings/felo_tier",
        ".//team_stats/felo_tier",
        "team_standings/felo_tier",
        "team_stats/felo_tier",
    ])

    win_probability = safe_float(team_node.findtext("win_probability"), np.nan)

    auction_budget_total = _first_float(team_node, ["draft_results/auction_budget/total", "auction_budget_total"])
    auction_budget_spent = _first_float(team_node, ["draft_results/auction_budget/spent", "auction_budget_spent"])

    return {
        'manager': manager,
        'team_name': team_name,
        'team_points': points,
        'team_projected_points': projected,
        'grade': grade,
        'gpa': gpa,
        'url': url_,
        'image_url': image_url,
        'division_id': division_id,
        'waiver_priority': waiver_priority,
        'faab_balance': faab_balance,
        'number_of_moves': number_of_moves,
        'number_of_trades': number_of_trades,
        'coverage_value': coverage_value,
        'value': value,
        'has_draft_grade': has_draft_grade,
        'auction_budget_total': auction_budget_total,
        'auction_budget_spent': auction_budget_spent,
        'felo_score': felo_score,
        'felo_tier': felo_tier,
        'win_probability': win_probability,
    }


def parse_matchups_for_week(
    oauth,
    yearid: str,
    year_val: int,
    week: int,
    manager_overrides: Dict[str, str] = None,
    debug_xml: bool = False
) -> List[Dict[str, Any]]:
    """
    Parse matchups for a specific week.

    Args:
        oauth: OAuth2 instance
        yearid: Yahoo league ID
        year_val: Season year
        week: Week number
        manager_overrides: Dictionary of manager name overrides
        debug_xml: If True, save raw XML response to file for debugging

    Returns:
        List of matchup row dictionaries
    """
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{yearid}/scoreboard;week={week}"
    print(f"[week {week}] Fetching matchups...")

    resp = oauth.session.get(url)
    resp.raise_for_status()

    # Save raw XML for debugging if requested
    if debug_xml:
        debug_file = Path(f"debug_matchup_week_{week}.xml")
        debug_file.write_text(resp.text, encoding='utf-8')
        print(f"[debug] Saved XML to {debug_file}")

    xmlstring = re.sub(r' xmlns=\"[^\"]+\"', '', resp.text, count=1)
    root = ET.fromstring(xmlstring)

    # Skip incomplete weeks (all totals zero)
    try:
        totals = [safe_float(tp.text, 0.0) for tp in root.findall(".//team_points/total")]
        if totals and all((p or 0.0) == 0.0 for p in totals):
            print(f"[week {week}] Skipping (no scores yet)")
            return []
    except Exception:
        pass

    def build_row(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """Build matchup row for manager a vs manager b."""
        a_points = safe_float(a['team_points'], 0.0)
        b_points = safe_float(b['team_points'], 0.0)
        margin = a_points - b_points
        total_matchup_score = a_points + b_points
        close_margin = int(abs(margin) <= 10.0)
        win = 1 if a_points > b_points else 0
        loss = 1 if a_points < b_points else 0
        tie = 1 if a_points == b_points else 0

        # Projection-based metrics
        a_proj = safe_float(a['team_projected_points'], np.nan)
        b_proj = safe_float(b['team_projected_points'], np.nan)
        proj_score_error = a_points - a_proj if not np.isnan(a_proj) else np.nan
        abs_proj_score_error = abs(proj_score_error) if not np.isnan(proj_score_error) else np.nan
        above_proj_score = int(proj_score_error > 0) if not np.isnan(proj_score_error) else 0
        below_proj_score = int(proj_score_error < 0) if not np.isnan(proj_score_error) else 0

        expected_spread = (a_proj - b_proj) if not np.isnan(a_proj) and not np.isnan(b_proj) else np.nan
        win_vs_spread = int(margin > expected_spread) if not np.isnan(expected_spread) else 0
        lose_vs_spread = int(margin < expected_spread) if not np.isnan(expected_spread) else 0

        # Expected win probability
        if not np.isnan(expected_spread):
            expected_odds = 1 / (1 + np.exp(-expected_spread / 10))
        else:
            expected_odds = np.nan

        underdog_wins = int(win and expected_spread < 0) if not np.isnan(expected_spread) else 0
        favorite_losses = int(loss and expected_spread > 0) if not np.isnan(expected_spread) else 0

        proj_wins = int(a_proj > b_proj) if not np.isnan(a_proj) and not np.isnan(b_proj) else 0
        proj_losses = int(a_proj < b_proj) if not np.isnan(a_proj) and not np.isnan(b_proj) else 0

        return {
            'week': int(week),
            'year': int(year_val),
            'manager': a['manager'],
            'team_name': a['team_name'],
            'team_points': a_points,
            'opponent': b['manager'],
            'opponent_points': b_points,
            'opponent_projected_points': b_proj,
            'margin': margin,
            'total_matchup_score': total_matchup_score,
            'close_margin': close_margin,
            'win': win,
            'loss': loss,
            'tie': tie,
            'proj_wins': proj_wins,
            'proj_losses': proj_losses,
            'proj_score_error': proj_score_error,
            'abs_proj_score_error': abs_proj_score_error,
            'above_proj_score': above_proj_score,
            'below_proj_score': below_proj_score,
            'expected_spread': expected_spread,
            'expected_odds': expected_odds,
            'win_vs_spread': win_vs_spread,
            'lose_vs_spread': lose_vs_spread,
            'underdog_wins': underdog_wins,
            'favorite_losses': favorite_losses,
            **a,  # Include all team-specific data
        }

    rows: List[Dict[str, Any]] = []
    for matchup in root.findall(".//matchup"):
        wk_node = matchup.find("week")
        if wk_node is None or not wk_node.text:
            continue
        week_num = int(wk_node.text)

        # NOTE: is_playoffs and is_consolation are NOT fetched here
        # They are created by the transformation pipeline (cumulative_stats_v2.py, playoff_flags.py)
        # Yahoo's API flags are unreliable (mark ALL postseason as playoffs)
        week_start = matchup.findtext("week_start", default="") or ""
        week_end = matchup.findtext("week_end", default="") or ""
        matchup_recap_url = matchup.findtext("matchup_recap_url", default="") or ""
        matchup_recap_title = matchup.findtext("matchup_recap_title", default="") or ""

        teams = matchup.findall(".//teams/team")
        if len(teams) != 2:
            teams = matchup.findall(".//team")
        if len(teams) != 2:
            continue

        # Pass matchup node to extract_team for grade/felo data
        t1 = extract_team(teams[0], matchup, manager_overrides)
        t2 = extract_team(teams[1], matchup, manager_overrides)

        # Create row for each manager
        row1 = build_row(t1, t2)
        row1.update({
            'week_start': week_start,
            'week_end': week_end,
            'matchup_recap_url': matchup_recap_url,
            'matchup_recap_title': matchup_recap_title,
        })

        row2 = build_row(t2, t1)
        row2.update({
            'week_start': week_start,
            'week_end': week_end,
            'matchup_recap_url': matchup_recap_url,
            'matchup_recap_title': matchup_recap_title,
        })

        rows.append(row1)
        rows.append(row2)

    return rows


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived metrics to matchup DataFrame.

    Args:
        df: Raw matchup DataFrame

    Returns:
        DataFrame with added metrics
    """
    # Weekly mean/median per manager
    weekly_stats = df.groupby(['manager', 'year', 'week'])['team_points'].first()
    df['weekly_mean'] = df.apply(lambda r: weekly_stats.loc[(r['manager'], r['year'], slice(None))].mean(), axis=1)
    df['weekly_median'] = df.apply(lambda r: weekly_stats.loc[(r['manager'], r['year'], slice(None))].median(), axis=1)

    # League-wide weekly mean/median
    league_weekly = df.groupby(['year', 'week'])['team_points'].agg(['mean', 'median']).reset_index()
    league_weekly.columns = ['year', 'week', 'league_weekly_mean', 'league_weekly_median']
    df = df.merge(league_weekly, on=['year', 'week'], how='left')

    df['above_league_median'] = (df['team_points'] > df['league_weekly_median']).astype(int)
    df['below_league_median'] = (df['team_points'] < df['league_weekly_median']).astype(int)

    # Teams beat this week (how many other teams you would have beaten)
    df['teams_beat_this_week'] = 0
    df['opponent_teams_beat_this_week'] = 0

    for (y, w), g in df.groupby(['year', 'week'], sort=False):
        scores = g.groupby('manager')['team_points'].first().to_dict()
        for idx, row in g.iterrows():
            my_score = row['team_points']
            opp_score = row['opponent_points']
            teams_beat = sum(1 for s in scores.values() if my_score > s)
            opp_teams_beat = sum(1 for s in scores.values() if opp_score > s)
            df.at[idx, 'teams_beat_this_week'] = teams_beat
            df.at[idx, 'opponent_teams_beat_this_week'] = opp_teams_beat

    return df


def add_head_to_head_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add head-to-head W/L records vs each manager.

    Args:
        df: Matchup DataFrame

    Returns:
        DataFrame with W/L columns vs each manager
    """
    mgr_all = sorted(df['manager'].unique())
    mgr_tokens = {m: re.sub(r'\W+', '_', m.strip().lower()) for m in mgr_all}

    # Create W/L vs each manager columns
    for m in mgr_all:
        tok = mgr_tokens[m]
        w_col = f"w_vs_{tok}"
        l_col = f"l_vs_{tok}"
        df[w_col] = 0
        df[l_col] = 0

    # Fill head-to-head W/L by week
    for (y, w), g in df.groupby(['year', 'week'], sort=False):
        scores = g.groupby('manager')['team_points'].first()
        mgrs = scores.index.to_list()
        vals = scores.to_numpy()

        # Comparison matrix
        gt = (vals[:, None] > vals[None, :]).astype(int)
        lt = (vals[:, None] < vals[None, :]).astype(int)

        for i, mgr in enumerate(mgrs):
            idx = g.index[g['manager'] == mgr]
            for j, opp in enumerate(mgrs):
                if mgr == opp:
                    continue
                tok = mgr_tokens[opp]
                df.loc[idx, f"w_vs_{tok}"] += gt[i, j]
                df.loc[idx, f"l_vs_{tok}"] += lt[i, j]

    # W/L vs each manager's schedule
    opp_points_map = df.set_index(['manager', 'year', 'week'])['opponent_points'].to_dict()

    for m in mgr_all:
        tok = mgr_tokens[m]
        aligned_sched = np.array([
            opp_points_map.get((m, y, w), np.nan)
            for y, w in zip(df['year'], df['week'])
        ])
        df[f"w_vs_{tok}_sched"] = (df['team_points'].to_numpy() > aligned_sched).astype(int)
        df[f"l_vs_{tok}_sched"] = (df['team_points'].to_numpy() < aligned_sched).astype(int)

    return df


# =============================================================================
# Main API Function
# =============================================================================

def weekly_matchup_data(
    ctx: Optional[LeagueContext] = None,
    year: Optional[int] = None,
    week: Optional[int] = None,
    oauth_file: Optional[Path] = None,
    league_key: Optional[str] = None,
    data_dir: Optional[Path] = None,
    logger: Optional[RunLogger] = None
) -> pd.DataFrame:
    """
    Fetch weekly matchup data from Yahoo Fantasy API.

    Args:
        ctx: Optional LeagueContext for league-specific configuration
        year: Season year
        week: Week number (None or 0 = all weeks)
        oauth_file: Path to OAuth credentials (if no context)
        league_key: Yahoo league_key (if no context)
        data_dir: Custom data directory (overrides context)
        logger: Optional RunLogger instance

    Returns:
        DataFrame with matchup data

    Raises:
        ValueError: If required parameters missing
        RuntimeError: If API calls fail
    """
    if not YFA_AVAILABLE:
        raise RuntimeError("yahoo_fantasy_api not available. Install with: pip install yahoo_fantasy_api")

    # Determine data directory
    if data_dir:
        matchup_data_dir = Path(data_dir)
    elif ctx:
        matchup_data_dir = ctx.matchup_data_directory
    else:
        matchup_data_dir = DEFAULT_DATA_ROOT

    matchup_data_dir.mkdir(parents=True, exist_ok=True)

    if year is None:
        raise ValueError("year is required")

    # Get OAuth
    if ctx:
        oauth = ctx.get_oauth_session()
        manager_overrides = ctx.manager_name_overrides
    elif oauth_file:
        if create_oauth2:
            oauth = create_oauth2(str(oauth_file))
        else:
            # Fallback to yahoo_oauth if create_oauth2 not available
            from yahoo_oauth import OAuth2
            oauth = OAuth2(None, None, from_file=str(oauth_file))
        manager_overrides = {}
    else:
        raise ValueError("Either ctx or oauth_file is required")

    # Create logger if context provided
    if logger is None and ctx and RUN_LOGGER_AVAILABLE:
        logger = RunLogger("weekly_matchup_data", year=year, week=week, league_id=ctx.league_id)
        logger.__enter__()
        close_logger = True
    else:
        close_logger = False

    try:
        # Get league
        if logger:
            logger.start_step("get_league")

        gm = yfa.Game(oauth, 'nfl')
        # Attempt to fetch league IDs with one retry after refreshing the token.
        league_ids = None
        for _attempt in (1, 2):
            try:
                league_ids = gm.league_ids(year=year)
                if league_ids:
                    break
                else:
                    raise RuntimeError(f"No leagues found for {year}")
            except Exception as e:
                # On first failure attempt to refresh access token and retry
                if _attempt == 1 and hasattr(oauth, 'refresh_access_token'):
                    try:
                        oauth.refresh_access_token()
                    except Exception:
                        pass
                    # Small backoff before retrying
                    time.sleep(1)
                    continue
                # If second attempt fails, rethrow
                raise RuntimeError(f"Failed to fetch league_ids for {year}: {e}")

        if not league_ids:
            raise RuntimeError(f"No leagues found for {year}")

        # CRITICAL: Use specific league_id from context to avoid data mixing
        # Priority: 1) ctx.get_league_id_for_year(), 2) league_key param, 3) league_ids[-1] (warn)
        yearid = None

        # Debug: Show what league_ids are available
        if ctx and hasattr(ctx, 'league_ids'):
            print(f"[league] Context has {len(ctx.league_ids)} league_ids: {list(ctx.league_ids.keys())}")

        # Try context first (safest - ensures league isolation)
        if ctx and hasattr(ctx, 'get_league_id_for_year'):
            yearid = ctx.get_league_id_for_year(year)
            if yearid:
                print(f"[league] Using league_id from context for {year}: {yearid}")

        # Fallback to explicit league_key parameter
        if not yearid and league_key:
            yearid = league_key
            print(f"[league] Using explicit league_key parameter: {yearid}")

        # Last resort: use API discovery (may mix leagues if user is in multiple!)
        if not yearid:
            if len(league_ids) > 1:
                print(f"[league] WARNING: Multiple leagues found for {year}: {league_ids}")
                print(f"[league] WARNING: Using last one ({league_ids[-1]}) - this may cause data mixing!")
                print(f"[league] TIP: Populate ctx.league_ids to ensure correct league isolation")
            yearid = league_ids[-1]

        league = gm.to_league(yearid)

        print(f"[league] Using league: {yearid}")

        if logger:
            logger.complete_step()

        # Get league settings
        if logger:
            logger.start_step("get_league_settings")

        settings = league.settings() if hasattr(league, 'settings') else {}

        if logger:
            logger.complete_step()

        # Determine weeks to fetch
        weeks = get_weeks_to_fetch(league, year, week, settings)
        print(f"[weeks] Fetching weeks: {weeks}")

        # Fetch matchups for each week
        if logger:
            logger.start_step("fetch_matchups")

        # Check for debug mode via environment variable
        debug_xml = os.environ.get('DEBUG_MATCHUP_XML', '').lower() in ('1', 'true', 'yes')

        all_rows = []
        for w in weeks:
            rows = parse_matchups_for_week(oauth, yearid, year, w, manager_overrides, debug_xml)
            all_rows.extend(rows)

        if not all_rows:
            raise RuntimeError("No matchup data found")

        df = pd.DataFrame(all_rows)
        print(f"[matchups] Fetched {len(df)} matchup rows")

        # VALIDATION: Ensure all league_ids in fetched data match expected league
        if 'league_id' in df.columns and ctx:
            unique_league_ids = df['league_id'].unique()
            expected_league_id = yearid
            if len(unique_league_ids) > 1:
                print(f"[matchups] WARNING: Multiple league_ids found in data: {unique_league_ids}")
                print(f"[matchups] WARNING: Expected only: {expected_league_id}")
            elif len(unique_league_ids) == 1 and unique_league_ids[0] != expected_league_id:
                print(f"[matchups] WARNING: League ID mismatch - expected {expected_league_id}, got {unique_league_ids[0]}")

        if logger:
            logger.complete_step(rows_read=len(df))

        # Add derived metrics
        if logger:
            logger.start_step("add_derived_metrics")

        df = add_derived_metrics(df)

        if logger:
            logger.complete_step()

        # REMOVED: Head-to-head records calculation
        # This is now handled by cumulative_stats_v2.py (using head_to_head.py module)
        # to avoid duplication and ensure transformations are separated from fetching
        # if logger:
        #     logger.start_step("add_head_to_head_records")
        # df = add_head_to_head_records(df)
        # if logger:
        #     logger.complete_step()

        # Column ordering
        # NOTE: is_playoffs and is_consolation are NOT included here
        # They will be added by transformation pipeline (cumulative_stats_v2.py)
        # NOTE: w_vs_X and l_vs_X head-to-head columns are also NOT included here
        # They will be added by cumulative_stats_v2.py (using head_to_head.py module)
        KEEP = [
            'week', 'year', 'manager', 'team_name',
            'team_points', 'team_projected_points', 'opponent', 'opponent_points',
            'opponent_projected_points', 'margin', 'total_matchup_score', 'close_margin',
            'weekly_mean', 'weekly_median',
            'league_weekly_mean', 'league_weekly_median', 'above_league_median', 'below_league_median',
            'win', 'loss', 'tie',
            'proj_wins', 'proj_losses',
            'teams_beat_this_week', 'opponent_teams_beat_this_week',
            'proj_score_error', 'abs_proj_score_error', 'above_proj_score', 'below_proj_score',
            'expected_spread', 'expected_odds', 'win_vs_spread', 'lose_vs_spread',
            'underdog_wins', 'favorite_losses',
            'gpa', 'grade', 'matchup_recap_title', 'matchup_recap_url', 'url', 'image_url',
            'division_id', 'week_start', 'week_end',
            'waiver_priority', 'has_draft_grade', 'faab_balance', 'number_of_moves', 'number_of_trades',
            'auction_budget_spent', 'auction_budget_total', 'win_probability', 'coverage_value', 'value',
            'felo_score', 'felo_tier'
        ]

        # REMOVED: W/L vs each manager columns (w_vs_X, l_vs_X, w_vs_X_sched, l_vs_X_sched)
        # These are now calculated by cumulative_stats_v2.py to avoid duplication
        # KEEP.extend(sorted([c for c in df.columns if c.startswith("w_vs_") and not c.endswith("_sched")]))
        # KEEP.extend(sorted([c for c in df.columns if c.startswith("l_vs_") and not c.endswith("_sched")]))
        # KEEP.extend(sorted([c for c in df.columns if c.startswith("w_vs_") and c.endswith("_sched")]))
        # KEEP.extend(sorted([c for c in df.columns if c.startswith("l_vs_") and c.endswith("_sched")]))

        # Ensure all columns exist
        for col in KEEP:
            if col not in df.columns:
                df[col] = np.nan

        df = df[KEEP].copy()

        # Add league_id for multi-league isolation (use year-specific league_id)
        df["league_id"] = yearid

        # Save outputs
        if logger:
            logger.start_step("save_outputs")

        week_tag = "all" if week in (None, 0) else str(week).zfill(2)
        csv_path = matchup_data_dir / f"matchup_data_week_{week_tag}_year_{year}.csv"
        parquet_path = matchup_data_dir / f"matchup_data_week_{week_tag}_year_{year}.parquet"

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        try:
            df.to_parquet(parquet_path, engine='pyarrow', index=False)
        except Exception:
            df.to_parquet(parquet_path, engine='fastparquet', index=False)

        print(f"\n[output] Saved CSV: {csv_path}")
        print(f"[output] Saved Parquet: {parquet_path}")
        print(f"[output] Rows: {len(df):,}, Columns: {len(df.columns)}")

        if logger:
            logger.complete_step(files_written=2, rows_written=len(df))

        return df

    finally:
        if close_logger and logger:
            logger.__exit__(None, None, None)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Fetch Yahoo Fantasy Football weekly matchup data (multi-league compliant)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--context', type=Path, help='Path to league_context.json')
    parser.add_argument('--year', type=int, required=True, help='Season year')
    parser.add_argument('--week', type=int, help='Week number (0 or omit = all weeks)')
    parser.add_argument('--oauth', type=Path, help='Path to OAuth credentials (if no context)')
    parser.add_argument('--league-key', help='Yahoo league_key (if no context)')
    parser.add_argument('--data-dir', type=Path, help='Custom data directory')

    args = parser.parse_args()

    # Load context if provided
    ctx = None
    if args.context:
        if not LEAGUE_CONTEXT_AVAILABLE:
            print("Error: league_context module not available", file=sys.stderr)
            sys.exit(1)

        try:
            ctx = LeagueContext.load(args.context)
            print(f"Loaded context: {ctx.league_name}")
        except Exception as e:
            print(f"Error loading context: {e}", file=sys.stderr)
            sys.exit(1)

    # Run
    try:
        weekly_matchup_data(
            ctx=ctx,
            year=args.year,
            week=args.week,
            oauth_file=args.oauth,
            league_key=args.league_key,
            data_dir=args.data_dir
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
