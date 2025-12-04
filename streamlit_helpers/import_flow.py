#!/usr/bin/env python3
"""
Import Flow Management

This module handles import-related logic including:
- Import protection (prevent double-clicks)
- Data templates
- Import flow coordination

Note: Most import execution logic remains in main.py due to tight
integration with Streamlit session state and UI.
"""

import os
import time
from typing import Optional


def check_active_import_in_motherduck(league_name: str) -> tuple[bool, Optional[str]]:
    """
    Check MotherDuck for active imports for this league.
    Returns (has_active_import, job_id_if_active).
    """
    try:
        import duckdb

        token = os.environ.get("MOTHERDUCK_TOKEN")
        if not token:
            try:
                import streamlit as st
                token = st.secrets.get("MOTHERDUCK_TOKEN")
            except:
                pass

        if not token:
            return False, None

        con = duckdb.connect(f"md:?motherduck_token={token}")

        # Check for running imports for this league (within last 3 hours)
        result = con.execute("""
            SELECT user_id, workflow_run_id, started_at
            FROM ops.import_jobs
            WHERE league_name = ?
            AND status = 'running'
            AND started_at > NOW() - INTERVAL 3 HOUR
            ORDER BY started_at DESC
            LIMIT 1
        """, [league_name]).fetchone()

        con.close()

        if result:
            return True, result[0]  # Return user_id
        return False, None

    except Exception:
        # If we can't check, allow the import (fail open)
        return False, None


def can_start_import(session_state, league_name: str = None) -> tuple[bool, str]:
    """
    Check if a new import can be started.
    Returns (can_start, reason_if_blocked).

    Args:
        session_state: Streamlit session state object.
        league_name: Optional league name to check for server-side active imports.
    """
    # Check if import is flagged as in progress (local session state)
    if session_state.get("import_in_progress", False):
        return False, "Import already in progress"

    # Check cooldown - prevent rapid clicks (30 second cooldown)
    last_import_time = session_state.get("last_import_triggered_at", 0)
    elapsed = time.time() - last_import_time
    if elapsed < 30:
        remaining = int(30 - elapsed)
        return False, f"Please wait {remaining}s before starting another import"

    # Check if we already have a job for this league (local session state)
    if session_state.get("import_job_id"):
        return False, "An import job was already started. Check the status below."

    # Server-side check: Query MotherDuck for active imports for this league
    # This prevents duplicates even if user refreshes or opens multiple tabs
    if league_name:
        has_active, active_job_id = check_active_import_in_motherduck(league_name)
        if has_active:
            return False, f"An import is already running for this league (job: {active_job_id})"

    return True, ""


def mark_import_started(session_state):
    """Mark that an import has been started (for cooldown tracking)."""
    session_state.import_in_progress = True
    session_state.last_import_triggered_at = time.time()


# =========================
# Data Validation & Cleaning
# =========================

def clean_dataframe(df):
    """
    Clean a DataFrame by:
    - Stripping whitespace from string columns
    - Converting empty strings to None
    - Parsing numeric strings with commas (1,234 -> 1234)
    - Normalizing boolean values

    Returns cleaned DataFrame.
    """
    import pandas as pd

    df = df.copy()

    for col in df.columns:
        # Strip whitespace from string columns
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            # Convert empty strings to None
            df[col] = df[col].replace('', None)
            df[col] = df[col].replace('N/A', None)
            df[col] = df[col].replace('n/a', None)
            df[col] = df[col].replace('NA', None)
            df[col] = df[col].replace('null', None)
            df[col] = df[col].replace('NULL', None)

    return df


def parse_numeric_column(series):
    """
    Parse a column that should be numeric but may have formatting issues.
    Handles: commas (1,234), dollar signs ($50), percentages (50%).

    Returns parsed series and list of unparseable values.
    """
    import pandas as pd
    import re

    errors = []

    def parse_value(x):
        if pd.isna(x) or x is None:
            return None
        if isinstance(x, (int, float)):
            return x
        if isinstance(x, str):
            # Remove common formatting
            cleaned = x.strip()
            cleaned = cleaned.replace(',', '')  # 1,234 -> 1234
            cleaned = cleaned.replace('$', '')  # $50 -> 50
            cleaned = cleaned.replace('%', '')  # 50% -> 50
            cleaned = cleaned.strip()

            if cleaned == '' or cleaned.lower() in ('n/a', 'na', 'null', '-', '--'):
                return None

            try:
                # Try int first, then float
                if '.' in cleaned:
                    return float(cleaned)
                else:
                    return int(cleaned)
            except ValueError:
                errors.append(x)
                return None
        return x

    result = series.apply(parse_value)
    return result, errors


def validate_dataframe(df, data_type: str, templates: dict) -> dict:
    """
    Validate a DataFrame against the template requirements.

    Returns dict with:
    - valid: bool
    - errors: list of error messages
    - warnings: list of warning messages
    - stats: dict of column statistics
    """
    import pandas as pd

    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }

    template = templates.get(data_type, {})
    required_cols = set(template.get("required_columns", []))

    # Check required columns exist
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        result["valid"] = False
        result["errors"].append(f"Missing required columns: {missing_cols}")

    # Validate data types for key columns
    type_checks = {
        "year": ("numeric", 2000, 2100),
        "week": ("numeric", 0, 25),
        "points": ("numeric", -100, 500),
        "team_points": ("numeric", 0, 300),
        "opponent_points": ("numeric", 0, 300),
        "win": ("boolean_int", 0, 1),
        "loss": ("boolean_int", 0, 1),
        "round": ("numeric", 1, 30),
        "pick": ("numeric", 1, 500),
        "cost": ("numeric", 0, 1000),
    }

    for col, (check_type, min_val, max_val) in type_checks.items():
        if col not in df.columns:
            continue

        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        if check_type == "numeric":
            # Check if values are numeric
            non_numeric = []
            for val in col_data:
                if not isinstance(val, (int, float)):
                    try:
                        float(str(val).replace(',', '').replace('$', ''))
                    except (ValueError, TypeError):
                        non_numeric.append(val)

            if non_numeric:
                result["warnings"].append(
                    f"Column '{col}' has non-numeric values: {non_numeric[:3]}{'...' if len(non_numeric) > 3 else ''}"
                )

            # Check range
            try:
                numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(numeric_vals) > 0:
                    if numeric_vals.min() < min_val or numeric_vals.max() > max_val:
                        result["warnings"].append(
                            f"Column '{col}' has values outside expected range ({min_val}-{max_val}): "
                            f"min={numeric_vals.min()}, max={numeric_vals.max()}"
                        )
            except Exception:
                pass

        elif check_type == "boolean_int":
            unique_vals = set(col_data.unique())
            valid_vals = {0, 1, '0', '1', True, False, 'true', 'false', 'True', 'False', 'TRUE', 'FALSE'}
            invalid = unique_vals - valid_vals
            if invalid:
                result["warnings"].append(
                    f"Column '{col}' should be 0/1 but has: {invalid}"
                )

    # Check for required string columns not being empty
    string_required = {"manager", "player", "opponent", "team_name", "transaction_type"}
    for col in string_required:
        if col in df.columns and col in required_cols:
            null_count = df[col].isna().sum()
            empty_count = (df[col] == '').sum() if df[col].dtype == 'object' else 0
            total_bad = null_count + empty_count
            if total_bad > 0:
                result["warnings"].append(
                    f"Column '{col}' has {total_bad} empty/null values out of {len(df)} rows"
                )

    # Generate stats
    result["stats"] = {
        "row_count": len(df),
        "columns": list(df.columns),
    }

    if "year" in df.columns:
        try:
            years = pd.to_numeric(df["year"], errors='coerce').dropna()
            if len(years) > 0:
                result["stats"]["years"] = sorted(years.unique().astype(int).tolist())
                result["stats"]["year_min"] = int(years.min())
                result["stats"]["year_max"] = int(years.max())
        except Exception:
            pass

    if "week" in df.columns:
        try:
            weeks = pd.to_numeric(df["week"], errors='coerce').dropna()
            if len(weeks) > 0:
                result["stats"]["week_min"] = int(weeks.min())
                result["stats"]["week_max"] = int(weeks.max())
        except Exception:
            pass

    if "manager" in df.columns:
        result["stats"]["managers"] = df["manager"].dropna().unique().tolist()

    return result


def check_year_conflicts(uploaded_years: list[int], yahoo_years: list[int]) -> dict:
    """
    Check if uploaded data years conflict with Yahoo-covered years.

    Args:
        uploaded_years: Years in the uploaded data
        yahoo_years: Years that Yahoo API will fetch

    Returns dict with:
    - conflicts: list of years that conflict
    - allowed: list of years that are OK to upload
    - blocked: list of years that should be blocked
    """
    uploaded_set = set(uploaded_years)
    yahoo_set = set(yahoo_years)

    conflicts = uploaded_set & yahoo_set
    allowed = uploaded_set - yahoo_set
    blocked = conflicts

    return {
        "conflicts": sorted(list(conflicts)),
        "allowed": sorted(list(allowed)),
        "blocked": sorted(list(blocked)),
        "has_conflicts": len(conflicts) > 0
    }


def check_duplicate_uploads(uploaded_files: dict) -> dict:
    """
    Check for duplicate data within uploaded files.

    Returns dict with:
    - duplicates: list of (data_type, year) tuples that appear multiple times
    - warnings: list of warning messages
    """
    from collections import defaultdict

    year_counts = defaultdict(lambda: defaultdict(list))  # data_type -> year -> [filenames]

    for data_type, files in uploaded_files.items():
        for file_info in files:
            filename = file_info.get("filename", "unknown")

            if file_info.get("file_type") == "json":
                year = file_info.get("year")
                if year:
                    year_counts[data_type][year].append(filename)
            else:
                # Check years in tabular data
                data = file_info.get("data", [])
                years_in_file = set()
                for row in data:
                    if "year" in row and row["year"]:
                        try:
                            years_in_file.add(int(row["year"]))
                        except (ValueError, TypeError):
                            pass
                for year in years_in_file:
                    year_counts[data_type][year].append(filename)

    duplicates = []
    warnings = []

    for data_type, years in year_counts.items():
        for year, filenames in years.items():
            if len(filenames) > 1:
                duplicates.append((data_type, year))
                warnings.append(
                    f"Year {year} appears in multiple {data_type} files: {', '.join(filenames)}"
                )

    return {
        "duplicates": duplicates,
        "warnings": warnings,
        "has_duplicates": len(duplicates) > 0
    }


def get_column_aliases() -> dict:
    """
    Return a comprehensive mapping of column name aliases to canonical names.

    This allows users to upload files with different column naming conventions
    (e.g., "owner" instead of "manager", "season" instead of "year").

    Covers common exports from: Yahoo, ESPN, Sleeper, NFL.com, CBS, Fleaflicker
    """
    return {
        # ===================
        # Manager/Owner variations (required for: matchup, player, draft, transactions, schedule)
        # ===================
        "owner": "manager",
        "Owner": "manager",
        "OWNER": "manager",
        "manager_name": "manager",
        "Manager": "manager",
        "MANAGER": "manager",
        "Manager Name": "manager",
        "manager name": "manager",
        "ManagerName": "manager",
        "owner_name": "manager",
        "Owner Name": "manager",
        "owner name": "manager",
        "OwnerName": "manager",
        "fantasy_owner": "manager",
        "team_owner": "manager",
        "Team Owner": "manager",
        "team owner": "manager",
        "TeamOwner": "manager",
        "user": "manager",
        "User": "manager",
        "USER": "manager",
        "username": "manager",
        "Username": "manager",
        "user_name": "manager",
        "display_name": "manager",
        "displayName": "manager",
        "Display Name": "manager",
        "roster_owner": "manager",
        "Roster Owner": "manager",
        "team_manager": "manager",
        "Team Manager": "manager",

        # ===================
        # Player variations (required for: player, draft, transactions)
        # ===================
        "player_name": "player",
        "Player": "player",
        "PLAYER": "player",
        "Player Name": "player",
        "player name": "player",
        "PLAYER_NAME": "player",
        "playerName": "player",
        "PlayerName": "player",
        "athlete": "player",
        "Athlete": "player",
        "ATHLETE": "player",
        "athlete_name": "player",
        "Athlete Name": "player",
        "name": "player",
        "Name": "player",
        "NAME": "player",
        "full_name": "player",
        "Full Name": "player",
        "fullName": "player",
        "player_full_name": "player",
        "Player Full Name": "player",
        "rostered_player": "player",
        "drafted_player": "player",
        "Drafted Player": "player",
        "acquired_player": "player",
        "dropped_player": "player",

        # ===================
        # Year/Season variations (required for: ALL tables)
        # ===================
        "season": "year",
        "Season": "year",
        "SEASON": "year",
        "Year": "year",
        "YEAR": "year",
        "yr": "year",
        "Yr": "year",
        "YR": "year",
        "fantasy_year": "year",
        "Fantasy Year": "year",
        "league_year": "year",
        "League Year": "year",
        "draft_year": "year",
        "Draft Year": "year",
        "nfl_season": "year",
        "NFL Season": "year",
        "season_year": "year",
        "Season Year": "year",

        # ===================
        # Week variations (required for: matchup, player, transactions, schedule)
        # ===================
        "Week": "week",
        "WEEK": "week",
        "week_num": "week",
        "week_number": "week",
        "Week Number": "week",
        "week number": "week",
        "WeekNumber": "week",
        "weekNum": "week",
        "wk": "week",
        "Wk": "week",
        "WK": "week",
        "gameweek": "week",
        "Gameweek": "week",
        "game_week": "week",
        "Game Week": "week",
        "matchup_week": "week",
        "Matchup Week": "week",
        "scoring_period": "week",
        "Scoring Period": "week",
        "period": "week",
        "Period": "week",
        "nfl_week": "week",
        "NFL Week": "week",

        # ===================
        # Points variations (required for: player)
        # ===================
        "Points": "points",
        "POINTS": "points",
        "fantasy_points": "points",
        "Fantasy Points": "points",
        "fantasy points": "points",
        "FANTASY_POINTS": "points",
        "fantasyPoints": "points",
        "FantasyPoints": "points",
        "fpts": "points",
        "FPTS": "points",
        "Fpts": "points",
        "f_pts": "points",
        "pts": "points",
        "Pts": "points",
        "PTS": "points",
        "score": "points",
        "Score": "points",
        "SCORE": "points",
        "total_points": "points",
        "Total Points": "points",
        "total points": "points",
        "totalPoints": "points",
        "TotalPoints": "points",
        "actual_points": "points",
        "Actual Points": "points",
        "actual": "points",
        "Actual": "points",
        "weekly_points": "points",
        "Weekly Points": "points",
        "player_points": "points",
        "Player Points": "points",

        # ===================
        # Team name variations (required for: matchup)
        # ===================
        "team": "team_name",
        "Team": "team_name",
        "TEAM": "team_name",
        "Team Name": "team_name",
        "team name": "team_name",
        "TEAM_NAME": "team_name",
        "teamName": "team_name",
        "TeamName": "team_name",
        "fantasy_team": "team_name",
        "Fantasy Team": "team_name",
        "fantasy team": "team_name",
        "fantasyTeam": "team_name",
        "my_team": "team_name",
        "My Team": "team_name",
        "roster_name": "team_name",
        "Roster Name": "team_name",
        "franchise": "team_name",
        "Franchise": "team_name",
        "franchise_name": "team_name",
        "Franchise Name": "team_name",

        # ===================
        # Team points variations (required for: matchup)
        # ===================
        "team_score": "team_points",
        "Team Score": "team_points",
        "team score": "team_points",
        "teamScore": "team_points",
        "TeamScore": "team_points",
        "Team Points": "team_points",
        "team points": "team_points",
        "TeamPoints": "team_points",
        "teamPoints": "team_points",
        "team_pts": "team_points",
        "Team Pts": "team_points",
        "my_points": "team_points",
        "My Points": "team_points",
        "my_score": "team_points",
        "My Score": "team_points",
        "my_pts": "team_points",
        "score_for": "team_points",
        "Score For": "team_points",
        "points_for": "team_points",
        "Points For": "team_points",
        "points for": "team_points",
        "pointsFor": "team_points",
        "PointsFor": "team_points",
        "pts_for": "team_points",
        "Pts For": "team_points",
        "PF": "team_points",
        "pf": "team_points",
        "for": "team_points",
        "For": "team_points",
        "fpts_for": "team_points",
        "FPTS For": "team_points",
        "your_score": "team_points",
        "Your Score": "team_points",
        "your_points": "team_points",
        "Your Points": "team_points",
        "roster_points": "team_points",
        "Roster Points": "team_points",

        # ===================
        # Opponent variations (required for: matchup, schedule)
        # ===================
        "Opponent": "opponent",
        "OPPONENT": "opponent",
        "opp": "opponent",
        "Opp": "opponent",
        "OPP": "opponent",
        "opponent_name": "opponent",
        "Opponent Name": "opponent",
        "opponent name": "opponent",
        "opponentName": "opponent",
        "OpponentName": "opponent",
        "opp_name": "opponent",
        "Opp Name": "opponent",
        "vs": "opponent",
        "VS": "opponent",
        "Vs": "opponent",
        "versus": "opponent",
        "Versus": "opponent",
        "opponent_manager": "opponent",
        "Opponent Manager": "opponent",
        "opp_manager": "opponent",
        "Opp Manager": "opponent",
        "against": "opponent",
        "Against": "opponent",
        "matchup_opponent": "opponent",
        "Matchup Opponent": "opponent",
        "opposing_team": "opponent",
        "Opposing Team": "opponent",
        "other_team": "opponent",
        "Other Team": "opponent",

        # ===================
        # Opponent points variations (required for: matchup)
        # ===================
        "opp_points": "opponent_points",
        "Opp Points": "opponent_points",
        "opp points": "opponent_points",
        "oppPoints": "opponent_points",
        "OppPoints": "opponent_points",
        "opp_pts": "opponent_points",
        "Opp Pts": "opponent_points",
        "opponent_score": "opponent_points",
        "Opponent Score": "opponent_points",
        "opponent score": "opponent_points",
        "opponentScore": "opponent_points",
        "OpponentScore": "opponent_points",
        "Opponent Points": "opponent_points",
        "opponent points": "opponent_points",
        "opponentPoints": "opponent_points",
        "OpponentPoints": "opponent_points",
        "score_against": "opponent_points",
        "Score Against": "opponent_points",
        "points_against": "opponent_points",
        "Points Against": "opponent_points",
        "points against": "opponent_points",
        "pointsAgainst": "opponent_points",
        "PointsAgainst": "opponent_points",
        "pts_against": "opponent_points",
        "Pts Against": "opponent_points",
        "PA": "opponent_points",
        "pa": "opponent_points",
        "against_points": "opponent_points",
        "Against Points": "opponent_points",
        "fpts_against": "opponent_points",
        "FPTS Against": "opponent_points",
        "their_score": "opponent_points",
        "Their Score": "opponent_points",
        "their_points": "opponent_points",
        "Their Points": "opponent_points",

        # ===================
        # Win/Loss variations (required for: matchup)
        # ===================
        "Win": "win",
        "WIN": "win",
        "W": "win",
        "w": "win",
        "won": "win",
        "Won": "win",
        "WON": "win",
        "wins": "win",
        "Wins": "win",
        "is_win": "win",
        "Is Win": "win",
        "isWin": "win",
        "victory": "win",
        "Victory": "win",
        "winner": "win",
        "Winner": "win",
        "Loss": "loss",
        "LOSS": "loss",
        "L": "loss",
        "l": "loss",
        "lost": "loss",
        "Lost": "loss",
        "LOST": "loss",
        "losses": "loss",
        "Losses": "loss",
        "is_loss": "loss",
        "Is Loss": "loss",
        "isLoss": "loss",
        "defeat": "loss",
        "Defeat": "loss",
        "loser": "loss",
        "Loser": "loss",

        # ===================
        # Round variations (required for: draft)
        # ===================
        "Round": "round",
        "ROUND": "round",
        "draft_round": "round",
        "Draft Round": "round",
        "draft round": "round",
        "draftRound": "round",
        "DraftRound": "round",
        "rd": "round",
        "Rd": "round",
        "RD": "round",
        "rnd": "round",
        "Rnd": "round",
        "RND": "round",
        "round_num": "round",
        "Round Number": "round",
        "round_number": "round",

        # ===================
        # Pick variations (required for: draft)
        # ===================
        "Pick": "pick",
        "PICK": "pick",
        "draft_pick": "pick",
        "Draft Pick": "pick",
        "draft pick": "pick",
        "draftPick": "pick",
        "DraftPick": "pick",
        "overall_pick": "pick",
        "Overall Pick": "pick",
        "overall pick": "pick",
        "overallPick": "pick",
        "OverallPick": "pick",
        "overall": "pick",
        "Overall": "pick",
        "selection": "pick",
        "Selection": "pick",
        "SELECTION": "pick",
        "pick_num": "pick",
        "Pick Number": "pick",
        "pick_number": "pick",
        "pick_in_round": "pick",
        "Pick In Round": "pick",
        "slot": "pick",
        "Slot": "pick",
        "draft_slot": "pick",
        "Draft Slot": "pick",
        "draft_position": "pick",
        "Draft Position": "pick",

        # ===================
        # Cost/Price variations (optional for: draft)
        # ===================
        "Cost": "cost",
        "COST": "cost",
        "price": "cost",
        "Price": "cost",
        "PRICE": "cost",
        "auction_price": "cost",
        "Auction Price": "cost",
        "auction price": "cost",
        "auctionPrice": "cost",
        "AuctionPrice": "cost",
        "draft_cost": "cost",
        "Draft Cost": "cost",
        "draft_price": "cost",
        "Draft Price": "cost",
        "salary": "cost",
        "Salary": "cost",
        "SALARY": "cost",
        "bid": "cost",
        "Bid": "cost",
        "BID": "cost",
        "bid_amount": "cost",
        "Bid Amount": "cost",
        "winning_bid": "cost",
        "Winning Bid": "cost",
        "amount": "cost",
        "Amount": "cost",
        "value": "cost",
        "Value": "cost",
        "dollar_value": "cost",
        "Dollar Value": "cost",
        "$": "cost",

        # ===================
        # Keeper variations (optional for: draft)
        # ===================
        "Keeper": "keeper",
        "KEEPER": "keeper",
        "is_keeper": "keeper",
        "Is Keeper": "keeper",
        "isKeeper": "keeper",
        "kept": "keeper",
        "Kept": "keeper",
        "KEPT": "keeper",
        "keeper_player": "keeper",
        "Keeper Player": "keeper",
        "is_kept": "keeper",
        "Is Kept": "keeper",
        "retained": "keeper",
        "Retained": "keeper",

        # ===================
        # Transaction type variations (required for: transactions)
        # ===================
        "type": "transaction_type",
        "Type": "transaction_type",
        "TYPE": "transaction_type",
        "trans_type": "transaction_type",
        "Trans Type": "transaction_type",
        "transType": "transaction_type",
        "TransType": "transaction_type",
        "transaction": "transaction_type",
        "Transaction": "transaction_type",
        "TRANSACTION": "transaction_type",
        "action": "transaction_type",
        "Action": "transaction_type",
        "ACTION": "transaction_type",
        "move": "transaction_type",
        "Move": "transaction_type",
        "MOVE": "transaction_type",
        "move_type": "transaction_type",
        "Move Type": "transaction_type",
        "moveType": "transaction_type",
        "MoveType": "transaction_type",
        "activity": "transaction_type",
        "Activity": "transaction_type",
        "activity_type": "transaction_type",
        "Activity Type": "transaction_type",
        "trans": "transaction_type",
        "Trans": "transaction_type",

        # ===================
        # Position variations (optional for: player)
        # ===================
        "position": "nfl_position",
        "Position": "nfl_position",
        "POSITION": "nfl_position",
        "pos": "nfl_position",
        "Pos": "nfl_position",
        "POS": "nfl_position",
        "player_position": "nfl_position",
        "Player Position": "nfl_position",
        "playerPosition": "nfl_position",
        "PlayerPosition": "nfl_position",
        "nfl_pos": "nfl_position",
        "NFL Position": "nfl_position",
        "eligible_position": "nfl_position",
        "Eligible Position": "nfl_position",
        "primary_position": "nfl_position",
        "Primary Position": "nfl_position",
        "slot": "lineup_position",
        "Slot": "lineup_position",
        "roster_slot": "lineup_position",
        "Roster Slot": "lineup_position",
        "lineup_slot": "lineup_position",
        "Lineup Slot": "lineup_position",
        "roster_position": "lineup_position",
        "Roster Position": "lineup_position",
        "starting_position": "lineup_position",
        "Starting Position": "lineup_position",

        # ===================
        # NFL team variations (optional for: player)
        # ===================
        "team_abbr": "nfl_team",
        "Team Abbr": "nfl_team",
        "team abbr": "nfl_team",
        "teamAbbr": "nfl_team",
        "TeamAbbr": "nfl_team",
        "pro_team": "nfl_team",
        "Pro Team": "nfl_team",
        "proTeam": "nfl_team",
        "ProTeam": "nfl_team",
        "nfl_team_abbr": "nfl_team",
        "NFL Team": "nfl_team",
        "nfl team": "nfl_team",
        "nflTeam": "nfl_team",
        "NFLTeam": "nfl_team",
        "real_team": "nfl_team",
        "Real Team": "nfl_team",
        "team_abbreviation": "nfl_team",
        "Team Abbreviation": "nfl_team",
        "pro_team_abbr": "nfl_team",
        "Pro Team Abbr": "nfl_team",

        # ===================
        # Projected points variations (optional for: player, matchup)
        # ===================
        "proj_points": "projected_points",
        "Proj Points": "projected_points",
        "proj points": "projected_points",
        "projPoints": "projected_points",
        "ProjPoints": "projected_points",
        "proj_pts": "projected_points",
        "Proj Pts": "projected_points",
        "Projected Points": "projected_points",
        "projected points": "projected_points",
        "projectedPoints": "projected_points",
        "ProjectedPoints": "projected_points",
        "projection": "projected_points",
        "Projection": "projected_points",
        "PROJECTION": "projected_points",
        "proj": "projected_points",
        "Proj": "projected_points",
        "PROJ": "projected_points",
        "expected_points": "projected_points",
        "Expected Points": "projected_points",
        "exp_points": "projected_points",
        "Exp Points": "projected_points",
        "forecast": "projected_points",
        "Forecast": "projected_points",

        # ===================
        # FAAB variations (optional for: transactions)
        # ===================
        "faab": "faab_bid",
        "FAAB": "faab_bid",
        "Faab": "faab_bid",
        "faab_bid": "faab_bid",
        "FAAB Bid": "faab_bid",
        "faab bid": "faab_bid",
        "faabBid": "faab_bid",
        "FAABBid": "faab_bid",
        "waiver_bid": "faab_bid",
        "Waiver Bid": "faab_bid",
        "waiver bid": "faab_bid",
        "waiverBid": "faab_bid",
        "WaiverBid": "faab_bid",
        "faab_amount": "faab_bid",
        "FAAB Amount": "faab_bid",
        "faab_spent": "faab_bid",
        "FAAB Spent": "faab_bid",
        "bid_amount": "faab_bid",
        "Bid Amount": "faab_bid",
        "waiver_amount": "faab_bid",
        "Waiver Amount": "faab_bid",

        # ===================
        # League settings specific
        # ===================
        "league_key": "league_key",
        "League Key": "league_key",
        "leagueKey": "league_key",
        "LeagueKey": "league_key",
        "league_id": "league_id",
        "League ID": "league_id",
        "league id": "league_id",
        "leagueId": "league_id",
        "LeagueId": "league_id",
        "num_teams": "num_teams",
        "Num Teams": "num_teams",
        "numTeams": "num_teams",
        "NumTeams": "num_teams",
        "number_of_teams": "num_teams",
        "Number Of Teams": "num_teams",
        "team_count": "num_teams",
        "Team Count": "num_teams",
        "teams": "num_teams",
        "Teams": "num_teams",
        "size": "num_teams",
        "Size": "num_teams",
        "league_size": "num_teams",
        "League Size": "num_teams",
        "playoff_teams": "playoff_teams",
        "Playoff Teams": "playoff_teams",
        "playoffTeams": "playoff_teams",
        "PlayoffTeams": "playoff_teams",
        "num_playoff_teams": "playoff_teams",
        "Num Playoff Teams": "playoff_teams",
        "playoff_spots": "playoff_teams",
        "Playoff Spots": "playoff_teams",

        # ===================
        # Tie variations (optional for: matchup)
        # ===================
        "Tie": "tie",
        "TIE": "tie",
        "T": "tie",
        "tied": "tie",
        "Tied": "tie",
        "ties": "tie",
        "Ties": "tie",
        "is_tie": "tie",
        "Is Tie": "tie",
        "isTie": "tie",
        "draw": "tie",
        "Draw": "tie",

        # ===================
        # Margin variations (optional for: matchup)
        # ===================
        "Margin": "margin",
        "MARGIN": "margin",
        "point_diff": "margin",
        "Point Diff": "margin",
        "point_differential": "margin",
        "Point Differential": "margin",
        "diff": "margin",
        "Diff": "margin",
        "spread": "margin",
        "Spread": "margin",
        "margin_of_victory": "margin",
        "Margin Of Victory": "margin",
        "mov": "margin",
        "MOV": "margin",
    }


def get_file_name_patterns() -> dict:
    """
    Return patterns for auto-detecting data type from file names.

    Each key is a data type, and the value is a list of regex patterns
    that would indicate that data type.
    """
    # Common file extensions pattern
    ext = r"\.(csv|xlsx|xls|parquet|json)$"

    return {
        "settings": [
            r"league_settings.*\.(json|csv|xlsx)$",
            r"settings.*\.(json|csv|xlsx)$",
            r"league_config.*\.(json|csv|xlsx)$",
            r"config.*\.(json|csv|xlsx)$",
            r"scoring.*\.(json|csv|xlsx)$",
            r"scoring_rules.*\.(json|csv|xlsx)$",
            r"league_rules.*\.(json|csv|xlsx)$",
            r"rules.*\.(json|csv|xlsx)$",
            r"roster_config.*\.(json|csv|xlsx)$",
            r"league_info.*\.(json|csv|xlsx)$",
            r"playoff.*config.*\.(json|csv|xlsx)$",
        ],
        "matchup": [
            r"matchup" + ext,
            r"matchups" + ext,
            r"match_up" + ext,
            r"match_ups" + ext,
            r"results" + ext,
            r"game_results" + ext,
            r"weekly_results" + ext,
            r"scores" + ext,
            r"weekly_scores" + ext,
            r"standings" + ext,
            r"head.*head" + ext,
            r"h2h" + ext,
            r"records" + ext,
            r"win.*loss" + ext,
            r"wins.*losses" + ext,
            r"weekly_matchup" + ext,
            r"season_results" + ext,
            r"box_scores" + ext,
            r"boxscores" + ext,
        ],
        "player": [
            r"player" + ext,
            r"players" + ext,
            r"player_stats" + ext,
            r"player_points" + ext,
            r"roster" + ext,
            r"rosters" + ext,
            r"weekly_player" + ext,
            r"weekly_players" + ext,
            r"lineup" + ext,
            r"lineups" + ext,
            r"starting_lineup" + ext,
            r"starters" + ext,
            r"bench" + ext,
            r"player_data" + ext,
            r"fantasy_points" + ext,
            r"weekly_lineup" + ext,
            r"team_roster" + ext,
            r"active_roster" + ext,
            r"player_scores" + ext,
            r"weekly_stats" + ext,
        ],
        "draft": [
            r"draft" + ext,
            r"drafts" + ext,
            r"draft_results" + ext,
            r"draft_picks" + ext,
            r"draft_board" + ext,
            r"auction" + ext,
            r"auction_results" + ext,
            r"auction_draft" + ext,
            r"draft_order" + ext,
            r"picks" + ext,
            r"selections" + ext,
            r"draft_data" + ext,
            r"keeper" + ext,
            r"keepers" + ext,
            r"keeper_draft" + ext,
            r"snake_draft" + ext,
        ],
        "transactions": [
            r"transaction" + ext,
            r"transactions" + ext,
            r"waiver" + ext,
            r"waivers" + ext,
            r"waiver_wire" + ext,
            r"trade" + ext,
            r"trades" + ext,
            r"moves" + ext,
            r"adds" + ext,
            r"drops" + ext,
            r"add_drop" + ext,
            r"adds_drops" + ext,
            r"pickups" + ext,
            r"free_agent" + ext,
            r"free_agents" + ext,
            r"fa_" + ext,
            r"claims" + ext,
            r"waiver_claims" + ext,
            r"activity" + ext,
            r"league_activity" + ext,
            r"roster_moves" + ext,
            r"faab" + ext,
        ],
        "schedule": [
            r"schedule" + ext,
            r"schedules" + ext,
            r"matchup_schedule" + ext,
            r"weekly_schedule" + ext,
            r"season_schedule" + ext,
            r"opponents" + ext,
            r"weekly_opponents" + ext,
            r"matchup_list" + ext,
            r"fixture" + ext,
            r"fixtures" + ext,
            r"calendar" + ext,
            r"game_schedule" + ext,
        ],
    }


def extract_year_from_filename(filename: str) -> Optional[int]:
    """
    Try to extract a year from a filename.

    Looks for 4-digit years between 2000-2099 in various formats:
    - league_settings_2014_xxx.json
    - matchup_2023.csv
    - draft-2019.xlsx
    - 2015_player_data.csv

    Returns the year as int or None if not found.
    """
    import re

    # Common patterns for years in filenames
    patterns = [
        r'[_\-\s](\d{4})[_\-\s\.]',  # _2014_, -2014-, 2014.
        r'^(\d{4})[_\-\s]',           # 2014_ at start
        r'[_\-\s](\d{4})$',           # _2014 at end (before extension)
        r'(\d{4})',                   # Any 4-digit sequence as fallback
    ]

    # Remove extension first
    name_without_ext = re.sub(r'\.[^.]+$', '', filename)

    for pattern in patterns:
        matches = re.findall(pattern, name_without_ext)
        for match in matches:
            year = int(match)
            # Validate it's a reasonable fantasy football year (2000-2099)
            if 2000 <= year <= 2099:
                return year

    return None


def detect_data_type_from_filename(filename: str) -> Optional[str]:
    """
    Auto-detect the data type based on the filename.

    Returns the detected data type or None if no match.
    """
    import re

    filename_lower = filename.lower()
    patterns = get_file_name_patterns()

    for data_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, filename_lower, re.IGNORECASE):
                return data_type

    return None


def normalize_columns(df, column_aliases: dict = None):
    """
    Normalize DataFrame column names using the alias mapping.

    Args:
        df: pandas DataFrame
        column_aliases: Optional custom aliases (uses get_column_aliases() if None)

    Returns:
        DataFrame with normalized column names
    """
    if column_aliases is None:
        column_aliases = get_column_aliases()

    # Create a new column mapping
    new_columns = {}
    for col in df.columns:
        # First try exact match
        if col in column_aliases:
            new_columns[col] = column_aliases[col]
        # Then try case-insensitive match by stripping whitespace
        elif col.strip() in column_aliases:
            new_columns[col] = column_aliases[col.strip()]
        else:
            # Keep original
            new_columns[col] = col

    df = df.rename(columns=new_columns)
    return df


def get_data_templates() -> dict:
    """
    Return templates for external data file imports.

    These templates define the expected columns for different data types
    when importing from external sources (ESPN, older Yahoo data, etc.)
    """
    return {
        "settings": {
            "description": "League settings (scoring rules, roster positions, playoff config)",
            "required_columns": ["year"],
            "optional_columns": [
                "league_key", "league_id", "league_name", "num_teams",
                "scoring_type", "draft_type", "playoff_teams", "playoff_start_week",
                "regular_season_weeks", "roster_positions", "scoring_rules",
                "stat_modifiers", "metadata"
            ],
            "file_types": ["json"],
            "example_row": {
                "year": 2014,
                "league_key": "331.l.381581",
                "metadata": {"name": "My League", "num_teams": 10},
                "roster_positions": [{"position": "QB", "count": 1}],
                "scoring_rules": [{"stat_id": "4", "name": "Pass Yds", "points": 0.04}]
            },
            "notes": "JSON files with league configuration. Can be Yahoo export format or custom."
        },
        "matchup": {
            "description": "Weekly matchup results (scores, wins/losses)",
            "required_columns": ["week", "year", "manager", "team_name", "team_points", "opponent", "opponent_points", "win", "loss"],
            "optional_columns": ["team_projected_points", "opponent_projected_points", "margin", "division_id", "is_playoffs", "is_consolation"],
            "file_types": ["csv", "xlsx", "parquet"],
            "example_row": {
                "week": 1, "year": 2013, "manager": "John", "team_name": "Team Awesome",
                "team_points": 125.5, "opponent": "Jane", "opponent_points": 110.2,
                "win": 1, "loss": 0, "division_id": "", "is_playoffs": 0, "is_consolation": 0
            }
        },
        "player": {
            "description": "Weekly player stats (points scored per player per week)",
            "required_columns": ["year", "week", "manager", "player", "points"],
            "optional_columns": ["nfl_position", "lineup_position", "nfl_team", "projected_points", "percent_started", "percent_owned"],
            "file_types": ["csv", "xlsx", "parquet"],
            "example_row": {
                "year": 2013, "week": 1, "manager": "John", "player": "Patrick Mahomes",
                "points": 28.5, "nfl_position": "QB", "lineup_position": "QB", "nfl_team": "KC"
            }
        },
        "draft": {
            "description": "Draft results (picks and costs)",
            "required_columns": ["year", "round", "pick", "manager", "player"],
            "optional_columns": ["cost", "keeper", "draft_type"],
            "file_types": ["csv", "xlsx", "parquet"],
            "example_row": {
                "year": 2013, "round": 1, "pick": 1, "manager": "John",
                "player": "Adrian Peterson", "cost": 65, "keeper": 0, "draft_type": "auction"
            }
        },
        "transactions": {
            "description": "Waiver/FA pickups, drops, and trades",
            "required_columns": ["year", "week", "manager", "player", "transaction_type"],
            "optional_columns": ["faab_bid", "source_type", "destination_type", "trade_partner"],
            "file_types": ["csv", "xlsx", "parquet"],
            "example_row": {
                "year": 2013, "week": 3, "manager": "John", "player": "Tyreek Hill",
                "transaction_type": "add", "faab_bid": 15, "source_type": "waivers"
            }
        },
        "schedule": {
            "description": "Season schedule (who plays who each week)",
            "required_columns": ["year", "week", "manager", "opponent"],
            "optional_columns": ["is_playoffs", "is_consolation", "week_start", "week_end"],
            "file_types": ["csv", "xlsx", "parquet"],
            "example_row": {
                "year": 2013, "week": 1, "manager": "John", "opponent": "Jane",
                "is_playoffs": 0, "is_consolation": 0
            }
        }
    }
