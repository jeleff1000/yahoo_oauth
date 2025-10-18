#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import yahoo_fantasy_api as yfa

# Add parent directory to path for oauth_utils import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from oauth_utils import ensure_oauth_path, create_oauth2
except Exception:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from oauth_utils import ensure_oauth_path, create_oauth2

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
# OAuth discovery: centralized helper
# =============================================================================
try:
    oauth_path = ensure_oauth_path()
    oauth = create_oauth2(oauth_path)
except SystemExit:
    raise
except Exception as e:
    raise SystemExit(f"OAuth initialization failed: {e}")

gm = yfa.Game(oauth, 'nfl')

# =============================================================================
# Helpers
# =============================================================================
REQ_COLUMNS = [
    "is_playoffs", "is_consolation",
    "manager", "team_name",
    "manager_week", "manager_year",
    "opponent", "opponent_week", "opponent_year",
    "week", "year",
    "team_points", "opponent_points",
    "win", "loss",
]

def norm_manager(nickname: str) -> str:
    if not nickname:
        return "N/A"
    s = str(nickname).strip()
    # Preserve your prior special case
    if s == "--hidden--":
        return "Ilan"
    return s

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

def get_league_for_year(y: int):
    """Return (league_key, league_obj) for a given year (choose last if multiple)."""
    league_ids = gm.league_ids(year=y)
    if not league_ids:
        raise RuntimeError(f"No Yahoo leagues found for {y}")
    league_key = league_ids[-1]  # if multiple, pick the last
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

def extract_team(team_node: ET.Element) -> dict:
    nickname = (
        team_node.findtext(".//managers/manager/nickname")
        or team_node.findtext(".//managers/manager/name")
        or team_node.findtext(".//managers/manager/guid")
        or ""
    )
    manager = norm_manager(nickname)
    team_name = team_node.findtext("name") or manager
    points = safe_float(team_node.findtext("team_points/total"), 0.0)
    return {'manager': manager, 'team_name': team_name, 'team_points': points}

def parse_week_schedule(league_key: str, season_year: int, week: int) -> list[dict]:
    """
    Pull one week's scoreboard and emit TWO rows per matchup
    (one per manager perspective) with the exact requested columns.
    Includes unplayed/incomplete weeks (points may be 0.0).
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

        teams = matchup.findall(".//teams/team")
        if len(teams) != 2:
            teams = matchup.findall(".//team")
        if len(teams) != 2:
            # malformed matchup; skip
            continue

        t1 = extract_team(teams[0])
        t2 = extract_team(teams[1])

        a_pts = safe_float(t1['team_points'], 0.0)
        b_pts = safe_float(t2['team_points'], 0.0)

        # ties => win=0, loss=0
        rows.append({
            'is_playoffs': is_playoffs,
            'is_consolation': is_consolation,
            'manager': t1['manager'],
            'team_name': t1['team_name'],
            'manager_week': week_num,
            'manager_year': season_year,
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
            'team_name': t2['team_name'],
            'manager_week': week_num,
            'manager_year': season_year,
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
    ints = [
        "is_playoffs", "is_consolation",
        "manager_week", "manager_year", "opponent_week", "opponent_year",
        "week", "year", "win", "loss",
    ]
    for c in ints:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
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

def build_and_save_for_year(year_val: int) -> tuple[str, str, int]:
    """Fetch full schedule for a year, write CSV/Parquet, return (csv_path, parquet_path, nrows)."""
    league_key, league = get_league_for_year(year_val)
    all_rows: list[dict] = []
    for w in league_weeks(league):
        try:
            all_rows.extend(parse_week_schedule(league_key, year_val, w))
        except Exception as e:
            print(f"Warning: {year_val} week {w} skipped: {e}", file=sys.stderr)

    if not all_rows:
        print(f"No schedule rows found for {year_val}.")
        return "", "", 0

    df = coerce_dtypes(pd.DataFrame(all_rows))
    csv_path = os.path.join(OUT_DIR, f"schedule_data_year_{year_val}.csv")
    parquet_path = os.path.join(OUT_DIR, f"schedule_data_year_{year_val}.parquet")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
    except Exception:
        df.to_parquet(parquet_path, engine="fastparquet", index=False)

    print(f"[{year_val}] Saved CSV:     {csv_path}")
    print(f"[{year_val}] Saved Parquet: {parquet_path}")
    print(f"[{year_val}] Rows/Cols:     {df.shape}")
    return csv_path, parquet_path, len(df)

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fetch Yahoo Fantasy Football schedule data")
    parser.add_argument('--year', type=int, default=None, help='Season year (0 for all years)')
    parser.add_argument('--week', type=int, default=None, help='Week number (ignored for schedule, which fetches full season)')
    args = parser.parse_args()

    # Use CLI arg if provided, otherwise prompt
    if args.year is not None:
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
        print(f"Discovered years: {years}")
        total_rows = 0
        all_dfs = []
        for y in years:
            _, parquet_path, n = build_and_save_for_year(y)
            if parquet_path and n > 0:
                try:
                    df = pd.read_parquet(parquet_path)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not read parquet for year {y}: {e}", file=sys.stderr)
            total_rows += n
        if all_dfs:
            big_df = pd.concat(all_dfs, ignore_index=True)
            big_parquet_path = os.path.join(OUT_DIR, "schedule_data_all_years.parquet")
            try:
                big_df.to_parquet(big_parquet_path, engine="pyarrow", index=False)
            except Exception:
                big_df.to_parquet(big_parquet_path, engine="fastparquet", index=False)
            print(f"Saved combined Parquet: {big_parquet_path} ({len(big_df)} rows)")
        print(f"Done. Wrote schedule files for {len(years)} year(s). Total rows: {total_rows}.")
    else:
        build_and_save_for_year(year_input)
if __name__ == "__main__":
    main()
