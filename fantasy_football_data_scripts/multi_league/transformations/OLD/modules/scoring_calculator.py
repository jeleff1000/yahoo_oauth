"""
Scoring Calculator Module

Handles fantasy points calculation from scoring rules.

This module:
- Loads year-specific scoring rules from JSON
- Builds Polars expressions from scoring rules
- Applies scoring to player stats
- Handles stat column detection and normalization
- Uses position-aware column mapping to handle ambiguous stats
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json


def load_scoring_rules(scoring_dir: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load scoring rules from JSON files.

    Args:
        scoring_dir: Directory containing scoring_rules_YYYY.json or league_settings_YYYY_*.json files

    Returns:
        Dict mapping year to scoring rules (with "scoring" key containing list of rules)
    """
    rules_by_year = {}

    if not scoring_dir.exists():
        return rules_by_year

    # First try to load scoring_rules_*.json files
    scoring_files = list(scoring_dir.glob("scoring_rules_*.json"))

    # If no scoring_rules files found, try league_settings_*.json files
    if not scoring_files:
        scoring_files = list(scoring_dir.glob("league_settings_*.json"))
        print(f"[scoring] No scoring_rules_*.json found, using {len(scoring_files)} league_settings files")

    for file_path in sorted(scoring_files):
        try:
            # Extract year from filename
            # Format: scoring_rules_2014.json OR league_settings_2014_449_l_198278.json
            parts = file_path.stem.split("_")
            if "league" in file_path.stem:
                # league_settings_YYYY_... format
                year = int(parts[2])
            else:
                # scoring_rules_YYYY format
                year = int(parts[-1])

            with open(file_path, 'r') as f:
                data = json.load(f)

                # If this is a league_settings file, extract scoring_rules and wrap in "scoring" key
                if "scoring_rules" in data:
                    rules_by_year[year] = {"scoring": data["scoring_rules"]}
                # If data already has "scoring" key, use as-is
                elif "scoring" in data:
                    rules_by_year[year] = data
                else:
                    # Unknown format, skip
                    print(f"[scoring] Warning: {file_path} has no 'scoring_rules' or 'scoring' key")
                    continue

        except (ValueError, json.JSONDecodeError, KeyError) as e:
            print(f"[scoring] Warning: Could not load {file_path}: {e}")
            continue

    return rules_by_year


def load_roster_settings(roster_dir: Path) -> Dict[int, Dict[str, int]]:
    """
    Load roster settings from JSON files.

    Args:
        roster_dir: Directory containing roster_settings_YYYY.json files

    Returns:
        Dict mapping year to roster settings
    """
    roster_by_year = {}

    if not roster_dir.exists():
        return roster_by_year

    for file_path in sorted(roster_dir.glob("roster_settings_*.json")):
        try:
            year = int(file_path.stem.split("_")[-1])
            with open(file_path, 'r') as f:
                settings = json.load(f)
                roster_by_year[year] = settings
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue

    return roster_by_year


def detect_column_patterns(df_columns: List[str]) -> Dict[str, bool]:
    """
    Detect column naming patterns in DataFrame.

    This helps determine which naming convention the data uses.

    Args:
        df_columns: List of column names in DataFrame

    Returns:
        Dict with pattern detection results
    """
    cols_lower = {c.lower() for c in df_columns}

    return {
        "has_def_prefix": any(c.startswith('def_') for c in cols_lower),
        "has_pass_prefix": any(c.startswith('pass_') for c in cols_lower),
        "has_rush_prefix": any(c.startswith('rush') for c in cols_lower),
        "has_rec_prefix": any(c.startswith('rec') for c in cols_lower),
        "has_offensive_stats": any(s in cols_lower for s in ['passing_yards', 'rushing_yards', 'receiving_yards']),
        "has_defensive_stats": any(s in cols_lower for s in ['def_sacks', 'def_interceptions', 'def_tds']),
        "has_kicker_stats": any(s in cols_lower for s in ['fg_made', 'pat_made']),
    }


def build_position_aware_mappings(
    df_columns: List[str],
    position_types: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Build column mappings based on actual DataFrame columns and position context.

    This resolves ambiguous stat names by:
    1. Detecting what columns actually exist in the DataFrame
    2. Using position context to disambiguate (e.g., "Int" for QB vs DEF)
    3. Preferring exact matches over fuzzy matches

    Args:
        df_columns: List of column names available in DataFrame
        position_types: Position types from scoring rule (e.g., ['O'], ['D'], ['K'])

    Returns:
        Dict mapping stat names to actual DataFrame columns
    """
    cols_lower = {c.lower(): c for c in df_columns}
    patterns = detect_column_patterns(df_columns)

    is_defense = position_types and any(pt in ['D', 'DT'] for pt in position_types)
    is_offense = position_types and 'O' in position_types
    is_kicker = position_types and 'K' in position_types

    mappings = {}

    # Offensive mappings with intelligent fallbacks
    if is_offense or not position_types:
        offensive_stats = {
            # Passing stats
            "Passing Touchdowns": ["passing_tds", "pass_td", "passing_touchdowns"],
            "Pass TD": ["passing_tds", "pass_td"],
            "Passing Yards": ["passing_yards", "pass_yds", "passing_yds"],
            "Pass Yds": ["passing_yards", "pass_yds"],
            "Interceptions": ["pass_int", "interceptions", "passing_int"],
            "Int": ["pass_int", "interceptions"] if patterns["has_pass_prefix"] else ["interceptions", "pass_int"],

            # Rushing stats
            "Rushing Yards": ["rushing_yards", "rush_yds", "rushing_yds"],
            "Rush Yds": ["rushing_yards", "rush_yds"],
            "Rushing Touchdowns": ["rushing_tds", "rush_td", "rushing_touchdowns"],
            "Rush TD": ["rushing_tds", "rush_td"],
            "Rushing Attempts": ["carries", "rushing_attempts", "rush_att"],
            "Rush Att": ["carries", "rushing_attempts"],

            # Receiving stats
            "Receptions": ["receptions", "rec", "receiving_rec"],
            "Rec": ["receptions", "rec"],
            "Receiving Yards": ["receiving_yards", "rec_yds", "receiving_yds"],
            "Rec Yds": ["receiving_yards", "rec_yds"],
            "Receiving Touchdowns": ["receiving_tds", "rec_td", "receiving_touchdowns"],
            "Rec TD": ["receiving_tds", "rec_td"],

            # Special teams / other
            "Return Touchdowns": ["special_teams_tds", "return_tds", "ret_td"],
            "Ret TD": ["special_teams_tds", "return_tds"],
            "2-Point Conversions": ["passing_2pt_conversions", "two_pt_conversions", "2pt_conversions"],
            "2-PT": ["passing_2pt_conversions", "two_pt_conversions"],
            "Fumbles Lost": ["sack_fumbles_lost", "fumbles_lost", "fum_lost"],
            "Fum Lost": ["sack_fumbles_lost", "fumbles_lost"],
            "Offensive Fumble Return TD": ["fum_ret_td", "fumble_return_td"],
            "Fum Ret TD": ["fum_ret_td", "fumble_return_td"],
        }

        for stat_name, candidates in offensive_stats.items():
            for candidate in candidates:
                if candidate.lower() in cols_lower:
                    mappings[stat_name] = cols_lower[candidate.lower()]
                    break

    # Kicker mappings
    if is_kicker or not position_types:
        kicker_stats = {
            "PAT Made": ["pat_made", "xp_made", "extra_point_made"],
            "Point After Attempt Made": ["pat_made", "xp_made"],
            "FG Yds": ["fg_made_distance", "fg_yds", "field_goal_yards"],
            "Field Goals Total Yards": ["fg_made_distance", "fg_yds"],
            "FG Made": ["fg_made", "field_goals_made"],
            "Field Goals Made": ["fg_made", "field_goals_made"],
            "FG Miss": ["fg_missed", "field_goals_missed"],
            "PAT Miss": ["pat_missed", "xp_missed"],
        }

        for stat_name, candidates in kicker_stats.items():
            if stat_name not in mappings:  # Don't override offensive mappings
                for candidate in candidates:
                    if candidate.lower() in cols_lower:
                        mappings[stat_name] = cols_lower[candidate.lower()]
                        break

    # Defensive/DST mappings with prefix awareness
    if is_defense or not position_types:
        defensive_stats = {
            "Sack": ["def_sacks", "sacks", "defensive_sacks"],
            "Int": ["def_interceptions", "def_int", "interceptions"] if patterns["has_def_prefix"] else ["interceptions", "def_interceptions"],
            "Fum Rec": ["fum_rec", "fumble_recoveries", "fumbles_recovered"],
            "Fumble Recovery": ["fum_rec", "fumble_recoveries"],
            "TD": ["def_tds", "defensive_tds", "touchdowns"],
            "Touchdown": ["def_tds", "defensive_tds"],
            "Safe": ["def_safeties", "safeties", "safety"],
            "Safety": ["def_safeties", "safeties"],
            "Blk Kick": ["fg_blocked", "blocked_kicks", "blocks"],
            "Block Kick": ["fg_blocked", "blocked_kicks"],
            "Ret TD": ["fum_ret_td", "return_tds", "kickoff_return_td"],
            "Kickoff and Punt Return Touchdowns": ["fum_ret_td", "return_tds"],
            "TFL": ["def_tackles_for_loss", "tackles_for_loss", "tfl"],
            "Tackles for Loss": ["def_tackles_for_loss", "tackles_for_loss"],
            "4 Dwn Stops": ["def_fourth_down_stops", "fourth_down_stops"],
            "4th Down Stops": ["def_fourth_down_stops", "fourth_down_stops"],
            "3 and Outs": ["def_three_out", "three_out"],
            "Three and Outs Forced": ["def_three_out", "three_out"],
            "XPR": ["extra_point_return_tds", "xp_return_td"],
            "Extra Point Returned": ["extra_point_return_tds", "xp_return_td"],

            # Points Allowed buckets
            "Pts Allow 0": ["pts_allow_0", "points_allowed_0"],
            "Points Allowed 0 points": ["pts_allow_0", "points_allowed_0"],
            "Pts Allow 1-6": ["pts_allow_1_6", "points_allowed_1_6"],
            "Points Allowed 1-6 points": ["pts_allow_1_6", "points_allowed_1_6"],
            "Pts Allow 7-13": ["pts_allow_7_13", "points_allowed_7_13"],
            "Points Allowed 7-13 points": ["pts_allow_7_13", "points_allowed_7_13"],
            "Pts Allow 14-20": ["pts_allow_14_20", "points_allowed_14_20"],
            "Points Allowed 14-20 points": ["pts_allow_14_20", "points_allowed_14_20"],
            "Pts Allow 21-27": ["pts_allow_21_27", "points_allowed_21_27"],
            "Points Allowed 21-27 points": ["pts_allow_21_27", "points_allowed_21_27"],
            "Pts Allow 28-34": ["pts_allow_28_34", "points_allowed_28_34"],
            "Points Allowed 28-34 points": ["pts_allow_28_34", "points_allowed_28_34"],
            "Pts Allow 35+": ["pts_allow_35_plus", "points_allowed_35_plus"],
            "Points Allowed 35+ points": ["pts_allow_35_plus", "points_allowed_35_plus"],
            "Pts Allow Neg": ["pts_allow_neg", "points_allowed_neg"],

            # Yards Allowed buckets
            "Yds Allow Neg": ["yds_allow_neg", "yards_allowed_neg"],
            "Defensive Yards Allowed - Negative": ["yds_allow_neg", "yards_allowed_neg"],
            "Yds Allow 0-99": ["yds_allow_0_99", "yards_allowed_0_99"],
            "Defensive Yards Allowed 0-99": ["yds_allow_0_99", "yards_allowed_0_99"],
            "Yds Allow 100-199": ["yds_allow_100_199", "yards_allowed_100_199"],
            "Defensive Yards Allowed 100-199": ["yds_allow_100_199", "yards_allowed_100_199"],
            "Yds Allow 200-299": ["yds_allow_200_299", "yards_allowed_200_299"],
            "Defensive Yards Allowed 200-299": ["yds_allow_200_299", "yards_allowed_200_299"],
            "Yds Allow 300-399": ["yds_allow_300_399", "yards_allowed_300_399"],
            "Defensive Yards Allowed 300-399": ["yds_allow_300_399", "yards_allowed_300_399"],
            "Yds Allow 400-499": ["yds_allow_400_499", "yards_allowed_400_499"],
            "Defensive Yards Allowed 400-499": ["yds_allow_400_499", "yards_allowed_400_499"],
            "Yds Allow 500+": ["yds_allow_500_plus", "yards_allowed_500_plus"],
            "Defensive Yards Allowed 500+": ["yds_allow_500_plus", "yards_allowed_500_plus"],
        }

        for stat_name, candidates in defensive_stats.items():
            if stat_name not in mappings:  # Don't override kicker/offensive mappings
                for candidate in candidates:
                    if candidate.lower() in cols_lower:
                        mappings[stat_name] = cols_lower[candidate.lower()]
                        break

    return mappings


def normalize_stat_name(stat: str, position_types: Optional[List[str]] = None) -> str:
    """
    Normalize stat names to match DataFrame columns.

    DEPRECATED: Use build_position_aware_mappings() instead for better accuracy.
    This function is kept for backward compatibility.

    Handles common variations and disambiguates based on position_types:
    - Passing Touchdowns -> passing_tds
    - Rushing Yards -> rushing_yds
    - Sack (defense) -> def_sacks
    - Int (offense vs defense) -> interceptions vs def_interceptions
    - etc.

    Args:
        stat: Stat name from scoring rules (e.g., "Int", "Sack", "Pass TD")
        position_types: List of position types from scoring rule (e.g., ['O'], ['D'], ['DT'])
                       'O' = Offense, 'D' = Defense, 'DT' = Defense Tackles

    Returns:
        Normalized column name
    """
    is_defense = position_types and any(pt in ['D', 'DT'] for pt in position_types)
    is_offense = position_types and 'O' in position_types

    # Offensive stats
    offensive_mappings = {
        "Passing Touchdowns": "passing_tds",
        "Pass TD": "passing_tds",
        "Passing Yards": "passing_yards",
        "Pass Yds": "passing_yards",
        "Interceptions": "pass_int",  # QB throwing INT (negative)
        "Int": "pass_int",  # Offensive INT
        "Rushing Yards": "rushing_yards",
        "Rush Yds": "rushing_yards",
        "Rushing Touchdowns": "rushing_tds",
        "Rush TD": "rushing_tds",
        "Rushing Attempts": "carries",
        "Rush Att": "carries",
        "Receptions": "receptions",
        "Rec": "receptions",
        "Receiving Yards": "receiving_yards",
        "Rec Yds": "receiving_yards",
        "Receiving Touchdowns": "receiving_tds",
        "Rec TD": "receiving_tds",
        "Return Touchdowns": "special_teams_tds",
        "Ret TD": "special_teams_tds",
        "2-Point Conversions": "passing_2pt_conversions",  # Will handle multiple types below
        "2-PT": "passing_2pt_conversions",
        "Fumbles Lost": "sack_fumbles_lost",  # Will sum multiple fumble types
        "Fum Lost": "sack_fumbles_lost",
        "Offensive Fumble Return TD": "fum_ret_td",
        "Fum Ret TD": "fum_ret_td",
    }

    # Kicker stats
    kicker_mappings = {
        "PAT Made": "pat_made",
        "Point After Attempt Made": "pat_made",
        "FG Yds": "fg_made_distance",
        "Field Goals Total Yards": "fg_made_distance",
        "FG Made": "fg_made",
        "Field Goals Made": "fg_made",
    }

    # Defensive/DST stats (Yahoo stat names → NFL defense columns)
    defensive_mappings = {
        "Sack": "def_sacks",
        "Int": "def_interceptions",  # Defense catching INT (positive)
        "Fum Rec": "fum_rec",
        "Fumble Recovery": "fum_rec",
        "TD": "def_tds",
        "Touchdown": "def_tds",
        "Safe": "def_safeties",
        "Safety": "def_safeties",
        "Blk Kick": "fg_blocked",
        "Block Kick": "fg_blocked",
        "Ret TD": "fum_ret_td",  # Defensive/special teams return TD
        "Kickoff and Punt Return Touchdowns": "fum_ret_td",
        "TFL": "def_tackles_for_loss",
        "Tackles for Loss": "def_tackles_for_loss",
        "4 Dwn Stops": "def_fourth_down_stops",
        "4th Down Stops": "def_fourth_down_stops",
        "3 and Outs": "def_three_out",
        "Three and Outs Forced": "def_three_out",
        "XPR": "extra_point_return_tds",
        "Extra Point Returned": "extra_point_return_tds",

        # Points Allowed buckets
        "Pts Allow 0": "pts_allow_0",
        "Points Allowed 0 points": "pts_allow_0",
        "Pts Allow 1-6": "pts_allow_1_6",
        "Points Allowed 1-6 points": "pts_allow_1_6",
        "Pts Allow 7-13": "pts_allow_7_13",
        "Points Allowed 7-13 points": "pts_allow_7_13",
        "Pts Allow 14-20": "pts_allow_14_20",
        "Points Allowed 14-20 points": "pts_allow_14_20",
        "Pts Allow 21-27": "pts_allow_21_27",
        "Points Allowed 21-27 points": "pts_allow_21_27",
        "Pts Allow 28-34": "pts_allow_28_34",
        "Points Allowed 28-34 points": "pts_allow_28_34",
        "Pts Allow 35+": "pts_allow_35_plus",
        "Points Allowed 35+ points": "pts_allow_35_plus",
        "Pts Allow Neg": "pts_allow_neg",

        # Yards Allowed buckets
        "Yds Allow Neg": "yds_allow_neg",
        "Defensive Yards Allowed - Negative": "yds_allow_neg",
        "Yds Allow 0-99": "yds_allow_0_99",
        "Defensive Yards Allowed 0-99": "yds_allow_0_99",
        "Yds Allow 100-199": "yds_allow_100_199",
        "Defensive Yards Allowed 100-199": "yds_allow_100_199",
        "Yds Allow 200-299": "yds_allow_200_299",
        "Defensive Yards Allowed 200-299": "yds_allow_200_299",
        "Yds Allow 300-399": "yds_allow_300_399",
        "Defensive Yards Allowed 300-399": "yds_allow_300_399",
        "Yds Allow 400-499": "yds_allow_400_499",
        "Defensive Yards Allowed 400-499": "yds_allow_400_499",
        "Yds Allow 500+": "yds_allow_500_plus",
        "Defensive Yards Allowed 500+": "yds_allow_500_plus",
    }

    # Determine if this is a kicker stat
    is_kicker = position_types and 'K' in position_types

    # Choose the right mapping based on position_types
    if is_defense:
        mapped = defensive_mappings.get(stat)
        if mapped:
            return mapped
    elif is_kicker:
        mapped = kicker_mappings.get(stat)
        if mapped:
            return mapped
    elif is_offense:
        mapped = offensive_mappings.get(stat)
        if mapped:
            return mapped

    # Fallback: check all mappings (offensive first, then kicker, then defensive)
    mapped = offensive_mappings.get(stat) or kicker_mappings.get(stat) or defensive_mappings.get(stat)
    if mapped:
        return mapped

    # Final fallback: normalize by replacing spaces with underscores
    return stat.lower().replace(" ", "_")


def build_points_expression(
    rules: List[Dict[str, Any]],
    df_columns: List[str]
) -> pl.Expr:
    """
    Build Polars expression for fantasy points calculation.

    Uses position-aware column mapping to handle ambiguous stat names correctly.

    Args:
        rules: List of scoring rules [{"stat": "Passing Yards", "points": 0.04, "position_types": ["O"]}, ...]
        df_columns: Available columns in DataFrame

    Returns:
        Polars expression that calculates total fantasy points
    """
    expr = pl.lit(0.0)

    # Create case-insensitive column lookup
    df_cols_lower = {col.lower(): col for col in df_columns}

    # Track which stats were successfully mapped
    mapped_stats = []
    unmapped_stats = []

    for rule in rules:
        stat_name = rule.get("stat", "") or rule.get("name", "")
        points_per = rule.get("points", 0.0)
        position_types = rule.get("position_types", [])

        # Use position-aware mapping system
        position_mappings = build_position_aware_mappings(df_columns, position_types)

        # Check if this stat has a direct mapping
        if stat_name in position_mappings:
            mapped_col = position_mappings[stat_name]
            expr = expr + (pl.col(mapped_col).cast(pl.Float64, strict=False).fill_null(0) * points_per)
            mapped_stats.append(f"{stat_name} -> {mapped_col}")
            continue

        # Special handling for stats that sum multiple columns
        if stat_name in ["2-Point Conversions", "2-PT"]:
            # Sum all 2-point conversion types
            two_pt_cols = ["passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"]
            two_pt_expr = pl.lit(0.0)
            found_any = False
            for col_name in two_pt_cols:
                if col_name.lower() in df_cols_lower:
                    two_pt_expr = two_pt_expr + pl.col(df_cols_lower[col_name.lower()]).cast(pl.Float64, strict=False).fill_null(0)
                    found_any = True
            if found_any:
                expr = expr + (two_pt_expr * points_per)
                mapped_stats.append(f"{stat_name} -> [2PT conversions sum]")
            else:
                unmapped_stats.append(stat_name)
            continue

        if stat_name in ["Fumbles Lost", "Fum Lost"]:
            # Sum all fumble types
            fumble_cols = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost", "fumbles_lost"]
            fumble_expr = pl.lit(0.0)
            found_any = False
            for col_name in fumble_cols:
                if col_name.lower() in df_cols_lower:
                    fumble_expr = fumble_expr + pl.col(df_cols_lower[col_name.lower()]).cast(pl.Float64, strict=False).fill_null(0)
                    found_any = True
            if found_any:
                expr = expr + (fumble_expr * points_per)
                mapped_stats.append(f"{stat_name} -> [fumbles sum]")
            else:
                unmapped_stats.append(stat_name)
            continue

        # Fallback: try legacy normalize_stat_name
        norm_stat = normalize_stat_name(stat_name, position_types)
        matching_col = df_cols_lower.get(norm_stat.lower())

        if matching_col:
            expr = expr + (pl.col(matching_col).cast(pl.Float64, strict=False).fill_null(0) * points_per)
            mapped_stats.append(f"{stat_name} -> {matching_col} (fallback)")
        else:
            unmapped_stats.append(stat_name)

    # Log mapping results for debugging
    if unmapped_stats:
        print(f"[scoring] Warning: {len(unmapped_stats)} stats could not be mapped: {', '.join(unmapped_stats[:5])}")

    return expr


def detect_stat_columns(df: pl.DataFrame) -> Dict[str, str]:
    """
    Detect available stat columns in DataFrame.

    Returns mapping of normalized stat names to actual column names.
    """
    stat_mapping = {}
    df_cols_lower = {c.lower(): c for c in df.columns}

    # Common stats to look for
    common_stats = [
        "passing_yds", "passing_tds", "interceptions",
        "rushing_yds", "rushing_tds",
        "receptions", "receiving_yds", "receiving_tds",
        "return_tds", "two_pt_conversions", "fumbles_lost",
        "fumble_return_tds"
    ]

    for stat in common_stats:
        if stat in df_cols_lower:
            stat_mapping[stat] = df_cols_lower[stat]

    return stat_mapping


def calculate_fantasy_points(
    df: pl.DataFrame,
    scoring_rules_by_year: Dict[int, Dict[str, Any]],
    year_col: str = "year",
    league_start_year: Optional[int] = None
) -> pl.DataFrame:
    """
    Calculate fantasy points for each player based on year-specific scoring rules.

    Args:
        df: DataFrame with player stats
        scoring_rules_by_year: Dict mapping year to scoring rules
        year_col: Name of year column
        league_start_year: The year the league started (from context). If provided,
                          years before this will use earliest available scoring rules.

    Returns:
        DataFrame with fantasy_points column added
    """
    if not scoring_rules_by_year:
        # No scoring rules - preserve existing fantasy_points if present, otherwise set to 0
        if "fantasy_points" not in df.columns:
            return df.with_columns(pl.lit(0.0).alias("fantasy_points"))
        else:
            # Keep existing fantasy_points column (from Yahoo data)
            return df

    df_cols = df.columns
    result_frames = []

    # Get all unique years in the data
    all_years = df.select(pl.col(year_col).unique()).to_series().to_list()
    available_rule_years = sorted(scoring_rules_by_year.keys())

    # Use league_start_year from context if provided, otherwise try to detect from data
    league_start = league_start_year

    if league_start is None and 'manager' in df.columns:
        # Fallback: try to detect from data (for backwards compatibility)
        # Filter for ACTUAL managers (not null and not "Unrostered")
        league_years = df.filter(
            pl.col('manager').is_not_null() &
            (pl.col('manager') != "Unrostered")
        ).select(year_col).unique().sort(year_col)

        if len(league_years) > 0:
            league_start = league_years[year_col].min()
            print(f"[scoring] League years detected from data: {league_start} to {league_years[year_col].max()}")
        else:
            # No rostered players found - all years are pre-league
            print(f"[scoring] No rostered players found - treating all years as pre-league")

    if league_start is not None:
        print(f"[scoring] League start year: {league_start}")

    # Process each year in the data
    for year in all_years:
        df_year = df.filter(pl.col(year_col) == year)

        if df_year.is_empty():
            continue

        # Find rules for this year (exact match or nearest year)
        if year in scoring_rules_by_year:
            rules = scoring_rules_by_year[year]
            settings_source = year
        else:
            # Fallback: use nearest year's rules (handles pre-league and missing years)
            if available_rule_years:
                # For pre-league years (or when league_start unknown), prefer earliest available settings
                if league_start is None or year < league_start:
                    settings_source = min(available_rule_years)
                    print(f"[scoring] Year {year} (pre-league): Using {settings_source} rules")
                else:
                    # Find closest year
                    closest_year = min(available_rule_years, key=lambda y: abs(y - year))
                    settings_source = closest_year
                    if settings_source != year:
                        print(f"[scoring] Using {closest_year} rules for year {year} (exact rules not found)")

                rules = scoring_rules_by_year[settings_source]
            else:
                # No rules at all, set to 0
                df_year = df_year.with_columns(pl.lit(0.0).alias("fantasy_points"))
                result_frames.append(df_year)
                continue

        # Build points expression for this year
        scoring_list = rules.get("scoring", [])
        points_expr = build_points_expression(scoring_list, df_cols)

        # Add fantasy_points column
        df_year = df_year.with_columns(points_expr.alias("fantasy_points"))
        result_frames.append(df_year)

    if not result_frames:
        return df.with_columns(pl.lit(0.0).alias("fantasy_points"))

    # Combine all years
    return pl.concat(result_frames, how="vertical")


def calculate_all_player_points(
    df: pl.DataFrame,
    scoring_rules_by_year: Dict[int, Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    year_col: str = "year"
) -> pl.DataFrame:
    """
    Calculate fantasy points for ALL players including unrostered and pre-league.

    This method ensures:
    1. Pre-league players use earliest available scoring settings
    2. Unrostered players get points calculated
    3. Uses nearest year's scoring when exact year unavailable

    Args:
        df: DataFrame with player stats
        scoring_rules_by_year: Dict mapping year to scoring rules
        context: Optional context dict with league info (for league_start detection)
        year_col: Name of year column

    Returns:
        DataFrame with fantasy_points column populated for all players
    """
    # This is actually the same as calculate_fantasy_points since we don't filter
    # The key is that the caller doesn't filter by manager/roster before calling this
    print("[scoring] Calculating points for ALL players (no roster filtering)")

    # Log player counts
    if 'manager' in df.columns:
        total = len(df)
        rostered = df.filter(pl.col('manager').is_not_null()).shape[0]
        unrostered = total - rostered
        print(f"[scoring]   Total: {total:,} players")
        print(f"[scoring]   Rostered: {rostered:,}")
        print(f"[scoring]   Unrostered: {unrostered:,}")

    return calculate_fantasy_points(df, scoring_rules_by_year, year_col)
