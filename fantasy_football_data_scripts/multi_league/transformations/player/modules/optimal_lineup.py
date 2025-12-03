"""
Optimal Lineup Module

Calculates league-wide optimal players and lineup metrics.

This module:
- Determines optimal players by position based on roster settings
- Calculates optimal lineup points for each manager
- Compares actual vs optimal performance
- Handles flex positions (W/R/T, FLEX, etc.) correctly
"""

import polars as pl
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


def load_roster_settings_from_json(settings_dir: Path, league_id: str = "449.l.198278") -> Dict[int, Dict[str, int]]:
    """
    Load roster settings from yahoo_roster_*.json files.

    Args:
        settings_dir: Directory containing yahoo_roster_*.json files
        league_id: League ID to filter for (e.g., "449.l.198278")

    Returns:
        Dict mapping year to position counts (e.g., {2024: {"QB": 1, "WR": 3, ...}})
    """
    roster_by_year = {}

    if not settings_dir.exists():
        print(f"[roster] Settings directory not found: {settings_dir}")
        return roster_by_year

    # Normalize league_id for filename matching (replace dots with underscores)
    normalized_league_id = league_id.replace(".", "_")

    # Find all roster/league settings files for this league
    # Try both naming conventions: yahoo_roster_* and league_settings_*
    file_pattern_1 = f"yahoo_roster_*_{normalized_league_id}.json"
    file_pattern_2 = f"league_settings_*_{normalized_league_id}.json"

    files = list(settings_dir.glob(file_pattern_1)) + list(settings_dir.glob(file_pattern_2))

    for file_path in sorted(files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            year = data.get("year")
            roster_positions = data.get("roster_positions", [])

            if not year or not roster_positions:
                continue

            # Build position count dict (exclude BN and IR)
            position_counts = {}
            for pos_info in roster_positions:
                position = pos_info.get("position", "")
                count = pos_info.get("count", 0)

                # Skip bench and IR positions
                if position in ("BN", "IR"):
                    continue

                if position and count > 0:
                    position_counts[position] = count

            # Validate that flex positions are separate from dedicated position counts
            # CRITICAL: Ensure roster structure is correct for optimal lineup calculation
            # Example: "WR": 3 + "W/R/T": 1, NOT "WR": 4
            has_flex = any(pos in position_counts for pos in ["W/R/T", "FLEX", "W/R", "W/T", "R/T", "Q/W/R/T", "OP"])

            roster_by_year[year] = position_counts

            if has_flex:
                flex_slots = [f"{pos}:{position_counts[pos]}" for pos in ["W/R/T", "FLEX", "W/R", "W/T", "R/T", "Q/W/R/T", "OP"] if pos in position_counts]
                print(f"[roster] Loaded {year}: {position_counts} (Flex: {', '.join(flex_slots)})")
            else:
                print(f"[roster] Loaded {year}: {position_counts} (No flex positions)")

        except (ValueError, json.JSONDecodeError, KeyError) as e:
            print(f"[roster] Warning: Could not load {file_path}: {e}")
            continue

    return roster_by_year


def identify_flex_positions(roster_settings: Dict[str, int]) -> List[tuple]:
    """
    Identify flex positions and their eligible positions.

    Args:
        roster_settings: Dict of position -> count

    Returns:
        List of (flex_position, eligible_positions, count) tuples
        e.g., [("W/R/T", ["WR", "RB", "TE"], 1), ("FLEX", ["RB", "WR", "TE"], 2)]
    """
    flex_positions = []

    # Common flex position patterns
    flex_patterns = {
        "W/R/T": ["WR", "RB", "TE"],
        "FLEX": ["WR", "RB", "TE"],
        "W/R": ["WR", "RB"],
        "W/T": ["WR", "TE"],
        "R/T": ["RB", "TE"],
        "Q/W/R/T": ["QB", "WR", "RB", "TE"],
        "OP": ["QB", "WR", "RB", "TE"],  # Offensive Player
    }

    for position, count in roster_settings.items():
        if position in flex_patterns:
            eligible = flex_patterns[position]
            flex_positions.append((position, eligible, count))

    return flex_positions


def compute_league_wide_optimal_players(
    df: pl.DataFrame,
    roster_by_year: Dict[int, Dict[str, int]],
    year_col: str = "year",
    week_col: str = "week",
    position_col: str = "position",
    points_col: str = "fantasy_points",
    player_id_col: str = "player_id"
) -> pl.DataFrame:
    """
    Mark the league-wide optimal players for each week.

    This determines which players SHOULD have been started based on:
    - Roster slot counts (e.g., 1 QB, 3 WR, 2 RB, 1 TE, 1 W/R/T, 1 K, 1 DEF)
    - Flex position handling (W/R/T takes best remaining WR/RB/TE)
    - ALL players who had stats (not just rostered players)

    Args:
        df: DataFrame with player stats (must have position, year, week, points)
        roster_by_year: Dict mapping year to roster settings
        year_col: Name of year column
        week_col: Name of week column
        position_col: Name of position column (unified 'position')
        points_col: Name of points column (default: fantasy_points)
        player_id_col: Name of player ID column (default: player_id - unified yahoo/NFL ID)

    Returns:
        DataFrame with league_wide_optimal_player flag and league_wide_optimal_position added
    """
    if not roster_by_year:
        print("[optimal] No roster settings found, marking all as not optimal")
        return df.with_columns([
            pl.lit(0).cast(pl.Int64).alias("league_wide_optimal_player"),
            pl.lit(None).cast(pl.Utf8).alias("league_wide_optimal_position")
        ])

    result_frames = []

    # Get all unique years in the data
    all_years = df.select(pl.col(year_col).unique()).to_series().to_list()
    available_roster_years = sorted(roster_by_year.keys())

    for year in all_years:
        df_year = df.filter(pl.col(year_col) == year)

        if df_year.is_empty():
            continue

        # Find roster settings for this year (exact match or nearest year)
        if year in roster_by_year:
            roster_settings = roster_by_year[year]
            settings_source = year
        else:
            # Fallback: use nearest year's roster settings (works for historical years like 1999-2013)
            if available_roster_years:
                closest_year = min(available_roster_years, key=lambda y: abs(y - year))
                roster_settings = roster_by_year[closest_year]
                settings_source = closest_year
                print(f"[optimal] Using {closest_year} roster settings for year {year} (exact settings not found)")
            else:
                # No roster settings at all, mark all as not optimal
                df_year = df_year.with_columns([
                    pl.lit(0).cast(pl.Int64).alias("league_wide_optimal_player"),
                    pl.lit(None).cast(pl.Utf8).alias("league_wide_optimal_position")
                ])
                result_frames.append(df_year)
                continue

        # Identify flex positions
        flex_positions = identify_flex_positions(roster_settings)

        # Debug: Show roster structure
        dedicated_positions = {pos: count for pos, count in roster_settings.items()
                             if pos not in ["W/R/T", "FLEX", "W/R", "W/T", "R/T", "Q/W/R/T", "OP"]}
        if flex_positions and year == sorted(df_year[year_col].unique().to_list())[0]:  # Print once per year
            print(f"[optimal] {year} roster: Dedicated={dedicated_positions}, Flex={[(fp[0], fp[2]) for fp in flex_positions]}")

        # Process each week
        weeks = sorted(df_year[week_col].unique().to_list())

        week_frames = []
        for week in weeks:
            df_week = df_year.filter(pl.col(week_col) == week)

            # Initialize optimal columns
            df_week = df_week.with_columns([
                pl.lit(0).cast(pl.Int64).alias("league_wide_optimal_player"),
                pl.lit(None).cast(pl.Utf8).alias("league_wide_optimal_position")
            ])

            # Track optimal player assignments: {nfl_player_id: position_label}
            optimal_assignments = {}

            # Step 1: Fill dedicated position slots (QB, RB, WR, TE, K, DEF)
            position_counters = {}
            for position, slots in roster_settings.items():
                # Skip flex positions (handle them separately)
                if position in ["W/R/T", "FLEX", "W/R", "W/T", "R/T", "Q/W/R/T", "OP"]:
                    continue

                if slots <= 0:
                    continue

                # Get top N players at this position by points
                # Deduplicate on player_id first to handle duplicate rows
                top_players = (
                    df_week
                    .filter(pl.col(position_col) == position)
                    .sort(points_col, descending=True)
                    .unique(subset=[player_id_col], keep="first", maintain_order=True)
                    .head(slots)
                )

                if not top_players.is_empty():
                    # Get player IDs and their points in descending order
                    player_data = top_players.select([player_id_col, points_col]).to_dicts()

                    # Assign position labels (e.g., QB1, WR1, WR2, WR3, RB1, RB2)
                    # Numbers are based on points ranking (1 = highest points)
                    for idx, row in enumerate(player_data, start=1):
                        player_id = row[player_id_col]
                        if slots == 1:
                            # Single slot position (e.g., QB, TE, K, DEF)
                            position_label = position
                        else:
                            # Multiple slots (e.g., WR1, WR2, WR3, RB1, RB2)
                            position_label = f"{position}{idx}"

                        optimal_assignments[player_id] = position_label
                        position_counters[position] = idx

            # Step 2: Fill flex positions with best remaining players
            for flex_pos, eligible_positions, flex_count in flex_positions:
                # Get all players eligible for flex
                flex_eligible = df_week.filter(pl.col(position_col).is_in(eligible_positions))

                # Exclude players already selected for dedicated positions
                already_selected_ids = list(optimal_assignments.keys())
                if already_selected_ids:
                    flex_eligible = flex_eligible.filter(~pl.col(player_id_col).is_in(already_selected_ids))

                # Get top flex_count remaining players
                # Deduplicate on player_id first to handle duplicate rows
                flex_players = (
                    flex_eligible
                    .sort(points_col, descending=True)
                    .unique(subset=[player_id_col], keep="first", maintain_order=True)
                    .head(flex_count)
                )

                if not flex_players.is_empty():
                    # Get player IDs and their points in descending order
                    player_data = flex_players.select([player_id_col, points_col]).to_dicts()

                    # Assign flex position labels based on points ranking
                    for idx, row in enumerate(player_data, start=1):
                        player_id = row[player_id_col]
                        if flex_count == 1:
                            # Single flex slot (e.g., W/R/T, FLEX)
                            position_label = flex_pos
                        else:
                            # Multiple flex slots (e.g., FLEX1, FLEX2)
                            position_label = f"{flex_pos}{idx}"

                        optimal_assignments[player_id] = position_label

            # Step 3: Apply optimal assignments to dataframe
            if optimal_assignments:
                # Convert to polars-compatible format
                optimal_ids = list(optimal_assignments.keys())
                optimal_positions = [optimal_assignments[pid] for pid in optimal_ids]

                # Get the dtype of player_id_col from the dataframe to ensure type matching
                player_id_dtype = df_week.schema[player_id_col]

                # Create a mapping dataframe with explicit type casting
                mapping_df = pl.DataFrame({
                    player_id_col: optimal_ids,
                    "league_wide_optimal_position": optimal_positions
                }).with_columns([
                    pl.col(player_id_col).cast(player_id_dtype)
                ])

                # Join to add optimal positions
                df_week = df_week.join(
                    mapping_df,
                    on=player_id_col,
                    how="left",
                    suffix="_optimal"
                )

                # Update the columns (coalesce handles the join)
                df_week = df_week.with_columns([
                    pl.col("league_wide_optimal_position_optimal").alias("league_wide_optimal_position"),
                    pl.when(pl.col("league_wide_optimal_position_optimal").is_not_null())
                      .then(pl.lit(1))
                      .otherwise(pl.lit(0))
                      .cast(pl.Int64)
                      .alias("league_wide_optimal_player")
                ]).drop("league_wide_optimal_position_optimal")

            week_frames.append(df_week)

        if week_frames:
            df_year_marked = pl.concat(week_frames, how="vertical")
        else:
            df_year_marked = df_year.with_columns([
                pl.lit(0).cast(pl.Int64).alias("league_wide_optimal_player"),
                pl.lit(None).cast(pl.Utf8).alias("league_wide_optimal_position")
            ])

        result_frames.append(df_year_marked)

    if not result_frames:
        return df.with_columns([
            pl.lit(0).cast(pl.Int64).alias("league_wide_optimal_player"),
            pl.lit(None).cast(pl.Utf8).alias("league_wide_optimal_position")
        ])

    return pl.concat(result_frames, how="vertical")


def compute_manager_optimal_players(
    df: pl.DataFrame,
    roster_by_year: Dict[int, Dict[str, int]],
    manager_col: str = "manager",
    year_col: str = "year",
    week_col: str = "week",
    position_col: str = "position",
    points_col: str = "fantasy_points",
    rostered_col: str = "is_rostered",
    player_id_col: str = "player_id"
) -> pl.DataFrame:
    """
    Mark the manager-specific optimal players for each manager-week.

    This determines the best lineup each manager could have started from their
    actual roster (not including free agents).

    Args:
        df: DataFrame with player stats (must have manager, position, year, week, points, is_rostered)
        roster_by_year: Dict mapping year to roster settings
        manager_col: Name of manager column
        year_col: Name of year column
        week_col: Name of week column
        position_col: Name of position column (unified 'position')
        points_col: Name of points column (default: fantasy_points)
        rostered_col: Name of rostered flag column (default: is_rostered)
        player_id_col: Name of player ID column (default: yahoo_player_id)

    Returns:
        DataFrame with optimal_player flag and optimal_position added
    """
    if not roster_by_year:
        print("[manager_optimal] No roster settings found, marking all as not optimal")
        return df.with_columns([
            pl.lit(0).cast(pl.Int64).alias("optimal_player"),
            pl.lit(None).cast(pl.Utf8).alias("optimal_position")
        ])

    # Check required columns
    required_cols = [manager_col, year_col, week_col, position_col, points_col, rostered_col, player_id_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[manager_optimal] Missing required columns: {missing}")
        return df.with_columns([
            pl.lit(0).cast(pl.Int64).alias("optimal_player"),
            pl.lit(None).cast(pl.Utf8).alias("optimal_position")
        ])

    result_frames = []

    # Get all unique years in the data
    all_years = df.select(pl.col(year_col).unique()).to_series().to_list()
    available_roster_years = sorted(roster_by_year.keys())

    for year in all_years:
        df_year = df.filter(pl.col(year_col) == year)

        if df_year.is_empty():
            continue

        # Find roster settings for this year
        if year in roster_by_year:
            roster_settings = roster_by_year[year]
            settings_source = year
        else:
            # Fallback: use nearest year's roster settings
            if available_roster_years:
                closest_year = min(available_roster_years, key=lambda y: abs(y - year))
                roster_settings = roster_by_year[closest_year]
                settings_source = closest_year
                print(f"[manager_optimal] Using {closest_year} roster settings for year {year}")
            else:
                df_year = df_year.with_columns([
                    pl.lit(0).cast(pl.Int64).alias("optimal_player"),
                    pl.lit(None).cast(pl.Utf8).alias("optimal_position")
                ])
                result_frames.append(df_year)
                continue

        # Identify flex positions
        flex_positions = identify_flex_positions(roster_settings)

        # Debug: Show roster structure (once per year)
        dedicated_positions = {pos: count for pos, count in roster_settings.items()
                             if pos not in ["W/R/T", "FLEX", "W/R", "W/T", "R/T", "Q/W/R/T", "OP"]}
        if flex_positions and year == sorted(all_years)[0]:  # Print once for first year
            print(f"[manager_optimal] {year} roster: Dedicated={dedicated_positions}, Flex={[(fp[0], fp[2]) for fp in flex_positions]}")

        # Process each manager-week combination
        managers = df_year.select(pl.col(manager_col).unique()).to_series().to_list()
        managers = [m for m in managers if m is not None and str(m).strip()]

        weeks = sorted(df_year[week_col].unique().to_list())

        manager_week_frames = []
        for manager in managers:
            for week in weeks:
                df_mw = df_year.filter(
                    (pl.col(manager_col) == manager) &
                    (pl.col(week_col) == week) &
                    (pl.col(rostered_col) == 1)  # Only consider rostered players
                )

                if df_mw.is_empty():
                    continue

                # Initialize optimal columns
                df_mw = df_mw.with_columns([
                    pl.lit(0).cast(pl.Int64).alias("optimal_player"),
                    pl.lit(None).cast(pl.Utf8).alias("optimal_position")
                ])

                # Track optimal player assignments
                optimal_assignments = {}

                # Step 1: Fill dedicated position slots
                for position, slots in roster_settings.items():
                    # Skip flex positions
                    if position in ["W/R/T", "FLEX", "W/R", "W/T", "R/T", "Q/W/R/T", "OP"]:
                        continue

                    if slots <= 0:
                        continue

                    # Get top N players at this position from manager's roster
                    # First, deduplicate on player_id (keep row with max points in case of duplicates)
                    # Then sort by points descending and take top N slots
                    top_players = (
                        df_mw
                        .filter(pl.col(position_col) == position)
                        .sort(points_col, descending=True)
                        .unique(subset=[player_id_col], keep="first", maintain_order=True)
                        .head(slots)
                    )

                    if not top_players.is_empty():
                        # Get player IDs and their points in descending order
                        player_data = top_players.select([player_id_col, points_col]).to_dicts()

                        # Assign position numbers (1, 2, 3...) based on points ranking
                        for idx, row in enumerate(player_data, start=1):
                            player_id = row[player_id_col]
                            if slots == 1:
                                position_label = position
                            else:
                                position_label = f"{position}{idx}"

                            optimal_assignments[player_id] = position_label

                # Step 2: Fill flex positions with best remaining rostered players
                for flex_pos, eligible_positions, flex_count in flex_positions:
                    flex_eligible = df_mw.filter(pl.col(position_col).is_in(eligible_positions))

                    # Exclude players already selected
                    already_selected_ids = list(optimal_assignments.keys())
                    if already_selected_ids:
                        flex_eligible = flex_eligible.filter(~pl.col(player_id_col).is_in(already_selected_ids))

                    # Get top flex_count remaining players
                    # Deduplicate on player_id first, then sort by points and take top N
                    flex_players = (
                        flex_eligible
                        .sort(points_col, descending=True)
                        .unique(subset=[player_id_col], keep="first", maintain_order=True)
                        .head(flex_count)
                    )

                    if not flex_players.is_empty():
                        # Get player IDs and their points in descending order
                        player_data = flex_players.select([player_id_col, points_col]).to_dicts()

                        # Assign flex position numbers based on points ranking
                        for idx, row in enumerate(player_data, start=1):
                            player_id = row[player_id_col]
                            if flex_count == 1:
                                position_label = flex_pos
                            else:
                                position_label = f"{flex_pos}{idx}"

                            optimal_assignments[player_id] = position_label

                # Step 3: Apply optimal assignments
                if optimal_assignments:
                    optimal_ids = list(optimal_assignments.keys())
                    optimal_positions = [optimal_assignments[pid] for pid in optimal_ids]

                    player_id_dtype = df_mw.schema[player_id_col]

                    mapping_df = pl.DataFrame({
                        player_id_col: optimal_ids,
                        "optimal_position": optimal_positions
                    }).with_columns([
                        pl.col(player_id_col).cast(player_id_dtype)
                    ])

                    # Join to add optimal positions
                    df_mw = df_mw.join(
                        mapping_df,
                        on=player_id_col,
                        how="left",
                        suffix="_opt"
                    )

                    # Update the columns
                    df_mw = df_mw.with_columns([
                        pl.col("optimal_position_opt").alias("optimal_position"),
                        pl.when(pl.col("optimal_position_opt").is_not_null())
                          .then(pl.lit(1))
                          .otherwise(pl.lit(0))
                          .cast(pl.Int64)
                          .alias("optimal_player")
                    ]).drop("optimal_position_opt")

                manager_week_frames.append(df_mw)

        # Get non-rostered rows from this year (they get 0 flags)
        df_non_rostered = df_year.filter(
            (pl.col(rostered_col) != 1) | pl.col(rostered_col).is_null()
        ).with_columns([
            pl.lit(0).cast(pl.Int64).alias("optimal_player"),
            pl.lit(None).cast(pl.Utf8).alias("optimal_position")
        ])

        if manager_week_frames:
            df_year_marked = pl.concat(manager_week_frames + [df_non_rostered], how="vertical")
        else:
            df_year_marked = df_year.with_columns([
                pl.lit(0).cast(pl.Int64).alias("optimal_player"),
                pl.lit(None).cast(pl.Utf8).alias("optimal_position")
            ])

        result_frames.append(df_year_marked)

    if not result_frames:
        return df.with_columns([
            pl.lit(0).cast(pl.Int64).alias("optimal_player"),
            pl.lit(None).cast(pl.Utf8).alias("optimal_position")
        ])

    return pl.concat(result_frames, how="vertical")


def calculate_optimal_lineup_metrics(
    df: pl.DataFrame,
    manager_col: str = "manager",
    year_col: str = "year",
    week_col: str = "week",
    points_col: str = "fantasy_points",
    actual_points_col: str = "team_points"
) -> pl.DataFrame:
    """
    Calculate optimal lineup points and efficiency for each manager-week.

    IMPORTANT: optimal_points is the MANAGER-SPECIFIC optimal (best lineup from their roster),
    NOT league-wide optimal. This represents the best score they could have achieved with
    their actual roster.

    Args:
        df: DataFrame with player data (must have optimal_player flag)
        manager_col: Name of manager column
        year_col: Name of year column
        week_col: str = "week",
        points_col: Name of points column (default: fantasy_points)
        actual_points_col: Name of actual points column (points scored)

    Returns:
        DataFrame with optimal_points and lineup_efficiency columns
    """
    # Drop existing optimal columns if they exist (from previous runs)
    cols_to_drop = ["optimal_points", "lineup_efficiency"]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    if existing_cols:
        df = df.drop(existing_cols)

    if "optimal_player" not in df.columns:
        # No optimal flags, can't calculate
        print("[optimal_metrics] optimal_player column not found, skipping")
        return df.with_columns([
            pl.lit(None).alias("optimal_points"),
            pl.lit(None).alias("lineup_efficiency")
        ])

    # Create manager_week key if not exists
    if "manager_week" not in df.columns:
        df = df.with_columns(
            (pl.col(manager_col).cast(pl.Utf8) + "_" +
             pl.col(year_col).cast(pl.Utf8) + "_" +
             pl.col(week_col).cast(pl.Utf8)).alias("manager_week")
        )

    # Calculate manager-specific optimal points per manager-week
    # Sum points from rows flagged as optimal_player==1 for each manager's roster
    manager_not_null = pl.col(manager_col).is_not_null() & (pl.col(manager_col) != "")

    optimal_totals = (
        df.filter(manager_not_null & (pl.col("optimal_player") == 1))
        .group_by("manager_week")
        .agg(pl.col(points_col).cast(pl.Float64, strict=False).sum().alias("__optimal_total"))
    )

    # Join back so EVERY row in that manager_week gets the same team optimal total
    df = df.join(optimal_totals, on="manager_week", how="left", coalesce=True)

    df = df.with_columns(
        pl.when(manager_not_null)
        .then(pl.col("__optimal_total").fill_null(0.0))
        .otherwise(None)
        .alias("optimal_points")
    ).drop("__optimal_total")

    # Calculate efficiency (actual / optimal)
    if actual_points_col in df.columns:
        df = df.with_columns(
            (pl.col(actual_points_col) / pl.col("optimal_points") * 100.0)
            .round(2)
            .alias("lineup_efficiency")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("lineup_efficiency"))

    return df


def calculate_bench_points(
    df: pl.DataFrame,
    manager_col: str = "manager",
    year_col: str = "year",
    week_col: str = "week",
    points_col: str = "fantasy_points",
    rostered_col: str = "is_rostered",
    started_col: str = "is_started"
) -> pl.DataFrame:
    """
    Calculate bench points (rostered but not started).

    Args:
        df: DataFrame with player data
        manager_col: Name of manager column
        year_col: Name of year column
        week_col: Name of week column
        points_col: Name of points column (default: fantasy_points)
        rostered_col: Name of rostered flag column
        started_col: Name of started flag column

    Returns:
        DataFrame with bench_points column added
    """
    # Drop existing bench_points column if it exists (from previous runs)
    if "bench_points" in df.columns:
        df = df.drop("bench_points")

    required_cols = [rostered_col, started_col]
    if not all(c in df.columns for c in required_cols):
        return df.with_columns(pl.lit(None).alias("bench_points"))

    # Bench = rostered but not started
    # Filter out rows where join keys are null (e.g., NFL-only years with no manager)
    bench_agg = (
        df.filter(
            (pl.col(rostered_col) == 1) &
            (pl.col(started_col) == 0)
        )
        .filter(pl.col(manager_col).is_not_null())
        .filter(pl.col(year_col).is_not_null())
        .filter(pl.col(week_col).is_not_null())
        .group_by([manager_col, year_col, week_col])
        .agg(pl.col(points_col).sum().alias("bench_points"))
    )

    # Join back
    df = df.join(
        bench_agg,
        on=[manager_col, year_col, week_col],
        how="left",
        coalesce=True
    )

    return df
