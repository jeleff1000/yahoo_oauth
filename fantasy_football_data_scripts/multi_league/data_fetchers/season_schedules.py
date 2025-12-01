#!/usr/bin/env python3
"""
Season Schedule Data Fetcher

Fetches fantasy football schedule data from Yahoo API.

Key Features:
- Multi-league support via LeagueContext
- Ensures is_consolation=1 implies is_playoffs=0 (mutually exclusive)
- Data dictionary compliant output
- Backward compatible with legacy mode

Data Dictionary Compliance:
- Primary keys: (manager, year, week)
- Foreign keys: opponent → manager
- Proper playoff/consolation flags
"""
from __future__ import annotations

import os
import re
import sys
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yahoo_fantasy_api as yfa

# Add paths for imports (matching draft_data_v2.py approach)
_script_file = Path(__file__).resolve()
_multi_league_dir = _script_file.parent.parent  # multi_league directory
sys.path.insert(0, str(_multi_league_dir / "core"))
sys.path.insert(0, str(_multi_league_dir / "utils"))

# Import OAuth2 for legacy mode
try:
    from yahoo_oauth import OAuth2
except ImportError:
    OAuth2 = None

# Try to import LeagueContext for multi-league support
try:
    from league_context import LeagueContext
    MULTI_LEAGUE_AVAILABLE = True
except ImportError:
    MULTI_LEAGUE_AVAILABLE = False
    LeagueContext = None

# =============================================================================
# Paths (ALL RELATIVE)
# =============================================================================
try:
    THIS_FILE = __file__
    BASE_DIR = os.path.dirname(os.path.abspath(THIS_FILE))
except NameError:
    BASE_DIR = os.getcwd()

OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'fantasy_football_data', 'schedule_data'))
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# OAuth discovery: Will be set from LeagueContext when using --context mode
# =============================================================================
oauth = None
gm = None  # Will be initialized in main() with proper oauth

# =============================================================================
# Helpers
# =============================================================================
REQ_COLUMNS = [
    "is_playoffs", "is_consolation",
    "manager", "manager_guid", "team_name",
    "cumulative_week", "manager_week", "manager_year",
    "opponent", "opponent_week", "opponent_year",
    "week", "year",
    "team_points", "opponent_points",
    "win", "loss",
]

def norm_manager(nickname: str, overrides: dict = None, team_name_fallback: str = None) -> str:
    """
    Normalize manager name with optional overrides from context.

    Args:
        nickname: Raw manager nickname from Yahoo API
        overrides: Dictionary mapping old names to new names (from LeagueContext)
        team_name_fallback: Team name to use if manager is hidden

    Returns:
        Normalized manager name
    """
    if not nickname:
        if team_name_fallback:
            fallback = str(team_name_fallback).strip().title()
            if overrides and fallback in overrides:
                return overrides[fallback]
            return fallback
        return "N/A"

    s = str(nickname).strip()

    # Apply overrides if provided
    if overrides:
        if s in overrides:
            return overrides[s]
        if s.title() in overrides:
            return overrides[s.title()]

    # Handle --hidden-- managers by using team name
    if s == "--hidden--":
        if team_name_fallback:
            fallback = str(team_name_fallback).strip().title()
            if overrides and fallback in overrides:
                return overrides[fallback]
            return fallback
        return "Unknown"

    return s.title()

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def get_league_for_year(y: int, ctx=None):
    """Return (league_key, league_obj) for a given year.

    CRITICAL: Uses context.league_ids if available to avoid data mixing
    when user is in multiple leagues.
    """
    league_key = None

    # Try context first (safest - ensures league isolation)
    if ctx and hasattr(ctx, 'get_league_id_for_year'):
        league_key = ctx.get_league_id_for_year(y)
        if league_key:
            print(f"[schedule] Using league_id from context for {y}: {league_key}")

    # Fallback to API discovery (may mix leagues!)
    if not league_key:
        league_ids = gm.league_ids(year=y)
        if not league_ids:
            raise RuntimeError(f"No Yahoo leagues found for {y}")
        if len(league_ids) > 1:
            print(f"[schedule] WARNING: Multiple leagues found for {y}: {league_ids}")
            print(f"[schedule] WARNING: Using last one - this may cause data mixing!")
        league_key = league_ids[-1]

    league = gm.to_league(league_key)
    return league_key, league

def league_weeks(league) -> list[int]:
    """Return full fantasy schedule range (start_week..end_week) for the league."""
    try:
        settings = league.settings()
    except Exception:
        settings = {}
    start_week = int(settings.get('start_week') or 1)
    end_week = int(settings.get('end_week') or 18)  # safe default if missing
    return list(range(start_week, end_week + 1))

def extract_team(team_node: ET.Element, manager_overrides: dict = None) -> dict:
    # Get team name first so it can be used as fallback for hidden managers
    team_name_raw = team_node.findtext("name") or ""

    # Get manager guid (persistent identifier across years)
    manager_guid = team_node.findtext(".//managers/manager/guid") or ""

    nickname = (
        team_node.findtext(".//managers/manager/nickname")
        or team_node.findtext(".//managers/manager/name")
        or manager_guid
        or ""
    )
    # Pass team_name as fallback for hidden managers
    manager = norm_manager(nickname, manager_overrides, team_name_fallback=team_name_raw)
    team_name = team_name_raw or manager
    points = safe_float(team_node.findtext("team_points/total"), 0.0)
    return {'manager': manager, 'manager_guid': manager_guid, 'team_name': team_name, 'team_points': points}

def parse_week_schedule(league_key: str, season_year: int, week: int, manager_overrides: dict = None) -> list[dict]:
    """
    Pull one week's scoreboard and emit TWO rows per matchup
    (one per manager perspective) with the exact requested columns.
    Includes unplayed/incomplete weeks (points may be 0.0).

    IMPORTANT: Ensures is_consolation=1 implies is_playoffs=0 (mutually exclusive).

    Args:
        league_key: Yahoo league key
        season_year: Season year
        week: Week number
        manager_overrides: Optional manager name overrides from LeagueContext
    """
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/scoreboard;week={week}"
    print("API:", url)
    resp = oauth.session.get(url); resp.raise_for_status()

    # strip xmlns
    xmlstring = re.sub(r' xmlns=\"[^\"]+\"', '', resp.text, count=1)
    root = ET.fromstring(xmlstring)

    rows: list[dict] = []
    for matchup in root.findall(".//matchup"):
        week_node = matchup.find("week")
        if week_node is None or not week_node.text:
            continue
        week_num = int(week_node.text)

        is_playoffs = int((matchup.findtext("is_playoffs") or "0").strip() or "0")
        is_consolation = int((matchup.findtext("is_consolation") or "0").strip() or "0")

        # CRITICAL: Ensure is_consolation=1 means is_playoffs=0 (mutually exclusive)
        if is_consolation == 1:
            is_playoffs = 0

        teams = matchup.findall(".//teams/team")
        if len(teams) != 2:
            teams = matchup.findall(".//team")
        if len(teams) != 2:
            # malformed matchup; skip
            continue

        t1 = extract_team(teams[0], manager_overrides)
        t2 = extract_team(teams[1], manager_overrides)

        a_pts = safe_float(t1['team_points'], 0.0)
        b_pts = safe_float(t2['team_points'], 0.0)

        # Calculate cumulative_week (YYYYWW format: e.g., 202401 for 2024 week 1)
        cumulative_week = season_year * 100 + week_num

        # Create manager composite keys (manager without spaces + identifier)
        manager1_no_spaces = t1['manager'].replace(" ", "")
        manager2_no_spaces = t2['manager'].replace(" ", "")

        manager1_week = f"{manager1_no_spaces}{cumulative_week}"
        manager1_year = f"{manager1_no_spaces}{season_year}"
        manager2_week = f"{manager2_no_spaces}{cumulative_week}"
        manager2_year = f"{manager2_no_spaces}{season_year}"

        # ties => win=0, loss=0
        rows.append({
            'is_playoffs': is_playoffs,
            'is_consolation': is_consolation,
            'manager': t1['manager'],
            'manager_guid': t1['manager_guid'],
            'team_name': t1['team_name'],
            'cumulative_week': cumulative_week,
            'manager_week': manager1_week,
            'manager_year': manager1_year,
            'opponent': t2['manager'],
            'opponent_week': week_num,
            'opponent_year': season_year,
            'week': week_num,
            'year': season_year,
            'team_points': a_pts,
            'opponent_points': b_pts,
            'win': 1 if a_pts > b_pts else 0,
            'loss': 1 if a_pts < b_pts else 0,
        })
        rows.append({
            'is_playoffs': is_playoffs,
            'is_consolation': is_consolation,
            'manager': t2['manager'],
            'manager_guid': t2['manager_guid'],
            'team_name': t2['team_name'],
            'cumulative_week': cumulative_week,
            'manager_week': manager2_week,
            'manager_year': manager2_year,
            'opponent': t1['manager'],
            'opponent_week': week_num,
            'opponent_year': season_year,
            'week': week_num,
            'year': season_year,
            'team_points': b_pts,
            'opponent_points': a_pts,
            'win': 1 if b_pts > a_pts else 0,
            'loss': 1 if b_pts < a_pts else 0,
        })
    return rows

def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Integer columns
    ints = [
        "is_playoffs", "is_consolation",
        "cumulative_week", "opponent_week", "opponent_year",
        "week", "year", "win", "loss",
    ]
    for c in ints:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    # String columns (manager_week and manager_year are composite keys)
    for c in ["manager_week", "manager_year"]:
        if c in d.columns:
            d[c] = d[c].astype(str)

    # Float columns
    for c in ["team_points", "opponent_points"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # preserve column order exactly
    return d[REQ_COLUMNS]

def discover_all_years(min_year: int = 2008, max_year: int | None = None) -> list[int]:
    """
    Try to discover all years where the user has an NFL league.
    Strategy: ping Yahoo for each year in a reasonable range and keep those with leagues.
    """
    if max_year is None:
        max_year = datetime.now().year
    years: list[int] = []
    for y in range(min_year, max_year + 1):
        try:
            ids = gm.league_ids(year=y)
            if ids:
                years.append(y)
        except Exception:
            # ignore years that error out
            pass
    return years

def build_and_save_for_year(year_val: int, manager_overrides: dict = None, ctx=None) -> tuple[str, str, int]:
    """Fetch full schedule for a year, write CSV/Parquet, return (csv_path, parquet_path, nrows)."""
    league_key, league = get_league_for_year(year_val, ctx=ctx)
    all_rows: list[dict] = []
    for w in league_weeks(league):
        try:
            all_rows.extend(parse_week_schedule(league_key, year_val, w, manager_overrides))
        except Exception as e:
            print(f"Warning: {year_val} week {w} skipped: {e}", file=sys.stderr)

    if not all_rows:
        print(f"[ERROR] No schedule rows found for {year_val}. Verify API response and mapping for league week -> NFL week.")
        raise SystemExit(2)

    df = coerce_dtypes(pd.DataFrame(all_rows))
    csv_path = os.path.join(OUT_DIR, f"schedule_data_year_{year_val}.csv")
    parquet_path = os.path.join(OUT_DIR, f"schedule_data_year_{year_val}.parquet")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
    except Exception:
        df.to_parquet(parquet_path, engine="fastparquet", index=False)

    print(f"[OK] [{year_val}] Saved CSV:     {csv_path}")
    print(f"[OK] [{year_val}] Saved Parquet: {parquet_path}")
    print(f"[OK] [{year_val}] Rows/Cols:     {df.shape}")
    return csv_path, parquet_path, len(df)

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fetch Yahoo Fantasy Football schedule data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-league mode (NEW - recommended)
  python season_schedules.py --context /path/to/league_context.json --year 2024
  python season_schedules.py --context /path/to/league_context.json --all-years

  # Legacy mode (backward compatible)
  python season_schedules.py --year 2024
  python season_schedules.py --year 0  # All years
        """
    )
    parser.add_argument('--context', type=Path, help='Path to league_context.json (multi-league mode)')
    parser.add_argument('--year', type=int, default=None, help='Season year (0 for all years)')
    parser.add_argument('--all-years', action='store_true', help='Fetch all years (alternative to --year 0)')
    parser.add_argument('--week', type=int, default=None, help='Week number (ignored for schedule, which fetches full season)')
    args = parser.parse_args()

    # Multi-league mode (preferred)
    manager_overrides = {}  # Will be set from context if available
    ctx = None  # Will be set from context if available

    if args.context:
        if not MULTI_LEAGUE_AVAILABLE:
            print("ERROR: LeagueContext not available. Ensure multi_league package is installed.", file=sys.stderr)
            sys.exit(1)

        if not args.context.exists():
            print(f"ERROR: Context file not found: {args.context}", file=sys.stderr)
            sys.exit(1)

        try:
            ctx = LeagueContext.load(args.context)
            print(f"Processing league: {ctx.league_name} ({ctx.league_id})")

            # Override OUT_DIR with context schedule directory
            global OUT_DIR
            OUT_DIR = str(ctx.schedule_data_directory)
            os.makedirs(OUT_DIR, exist_ok=True)

            # Override oauth with context oauth
            global oauth, gm
            oauth = ctx.get_oauth_session()
            gm = yfa.Game(oauth, ctx.game_code)

            # Get manager name overrides from context
            manager_overrides = ctx.manager_name_overrides if ctx.manager_name_overrides else {}

            # Determine years from context
            if args.all_years or args.year == 0:
                current_year = datetime.now().year
                start_year = ctx.start_year if ctx.start_year else 2014
                end_year = ctx.end_year if ctx.end_year else current_year
                years = list(range(start_year, end_year + 1))
            elif args.year:
                years = [args.year]
            else:
                # Default to current year from context
                years = [ctx.end_year if ctx.end_year else datetime.now().year]

        except Exception as e:
            print(f"ERROR: Failed to load league context: {e}", file=sys.stderr)
            sys.exit(1)

    # Legacy mode (backward compatible)
    else:
        # Use CLI arg if provided, otherwise prompt
        if args.all_years:
            year_input = 0
        elif args.year is not None:
            year_input = args.year
        else:
            try:
                prompt = "Enter fantasy season year (e.g., 2025) or 0 for ALL available years: "
                year_input = int(input(prompt).strip())
            except Exception:
                print("Invalid input. Please enter a number (e.g., 2025 or 0).", file=sys.stderr)
                sys.exit(1)

        if year_input == 0:
            years = discover_all_years(min_year=2008)
            if not years:
                print("No available years were discovered for your account.", file=sys.stderr)
                sys.exit(0)
        else:
            years = [year_input]

    # Process years
    print(f"Processing {len(years)} year(s): {years}")
    total_rows = 0

    for y in years:
        try:
            _, parquet_path, n = build_and_save_for_year(y, manager_overrides, ctx=ctx)
            total_rows += n
        except Exception as e:
            print(f"[ERROR] Failed to process year {y}: {e}", file=sys.stderr)
            continue

    print(f"[OK] Wrote schedule files for {len(years)} year(s). Total rows: {total_rows}.")
    print(f"[INFO] Combined schedule.parquet will be created by initial_import_v2.py")

if __name__ == "__main__":
    main()
