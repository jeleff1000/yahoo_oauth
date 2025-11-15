"""
Player Stats Transformation

Multi-league player statistics enrichment pipeline.

RECALCULATE WEEKLY: All columns must be recalculated every week.

This transformation:
- Calculates fantasy points from scoring rules
- Determines league-wide optimal players (includes free agents)
- Determines manager-specific optimal players (from each manager's roster)
- Creates player rankings (personal, position, manager-based)
- Calculates PPG metrics and consistency scores
- Computes optimal lineup efficiency (actual vs manager-optimal)

Usage:
    python player_stats_v2.py --context path/to/league_context.json
    python player_stats_v2.py --context path/to/league_context.json --current-week 5 --current-year 2024
"""

import argparse
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional
import sys



# Add parent directories to path for imports
_script_file = Path(__file__).resolve()

# Auto-detect if we're in modules/ subdirectory or transformations/ directory
if _script_file.parent.name == 'modules':
    # We're in multi_league/transformations/modules/
    _modules_dir = _script_file.parent
    _transformations_dir = _modules_dir.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'transformations':
    # We're in multi_league/transformations/
    _transformations_dir = _script_file.parent
    _multi_league_dir = _transformations_dir.parent
else:
    # Fallback: assume we're somewhere in the tree, navigate up to find multi_league
    _current = _script_file.parent
    while _current.name != 'multi_league' and _current.parent != _current:
        _current = _current.parent
    _multi_league_dir = _current

_scripts_dir = _multi_league_dir.parent  # fantasy_football_data_scripts directory
sys.path.insert(0, str(_scripts_dir))  # Allows: from multi_league.core.XXX
sys.path.insert(0, str(_multi_league_dir))  # Allows: from core.XXX

from core.league_context import LeagueContext
from multi_league.transformations.modules.scoring_calculator import (
    load_scoring_rules,
    load_roster_settings,
    calculate_fantasy_points
)
from multi_league.transformations.modules.optimal_lineup import (
    load_roster_settings_from_json,
    compute_league_wide_optimal_players,
    compute_manager_optimal_players,
    calculate_optimal_lineup_metrics,
    calculate_bench_points
)
from multi_league.transformations.modules.player_rankings import (
    add_manager_player_ranks,
    add_lineup_position,
    add_optimal_lineup_position,
    add_manager_player_history_ranks,
    add_manager_position_history_ranks,
    add_player_personal_history_ranks,
    add_player_personal_ranks,
    add_position_history_ranks,
    add_manager_all_player_history_ranks,
    add_league_wide_position_ranks,
    add_all_players_alltime_ranks
)
from multi_league.transformations.modules.ppg_calculator import (
    calculate_season_ppg,
    calculate_alltime_ppg,
    calculate_rolling_avg,
    calculate_weighted_ppg,
    calculate_ppg_trend,
    calculate_consistency_score,
    calculate_rolling_point_total
)


# ------------------------------------------------------------------
# Helper: unify player id across yahoo_player_id / nfl_player_id
# ------------------------------------------------------------------
def unify_player_id(df):
    """
    Combine yahoo_player_id and nfl_player_id into a single canonical player_id.
    Prefers yahoo_player_id when both are present. Ensures player_week and
    player_year keys exist downstream. Works with polars or pandas DataFrames.
    """
    # Polars DataFrame
    try:
        import polars as _pl
    except Exception:
        _pl = None

    # If polars DataFrame, convert to pandas for convenience, then convert back
    is_polars = (_pl is not None) and isinstance(df, _pl.DataFrame)
    if is_polars:
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    # unified player_id: prefer yahoo_player_id, else NFL_player_id (case-insensitive)
    # ALWAYS recalculate to ensure historical years get NFL_player_id
    yahoo = pdf.get('yahoo_player_id') if 'yahoo_player_id' in pdf.columns else None
    # Check both lowercase and uppercase NFL_player_id
    nfl = pdf.get('NFL_player_id') if 'NFL_player_id' in pdf.columns else (
          pdf.get('nfl_player_id') if 'nfl_player_id' in pdf.columns else None)

    if yahoo is not None:
        pdf['player_id'] = yahoo.fillna(nfl if nfl is not None else pd.NA)
    else:
        # No yahoo column: use nfl if present, otherwise set to NaN
        if nfl is not None:
            pdf['player_id'] = nfl.fillna(pd.NA)
        else:
            if 'player_id' not in pdf.columns:
                pdf['player_id'] = pd.NA

    # Build composite keys consistently
    if 'player_week' not in pdf.columns and {'player_id','year','week'}.issubset(pdf.columns):
        pdf['player_week'] = pdf['player_id'].astype(str) + "_" + pdf['year'].astype(str) + "_" + pdf['week'].astype(str)

    if 'player_year' not in pdf.columns and {'player_id','year'}.issubset(pdf.columns):
        pdf['player_year'] = pdf['player_id'].astype(str) + "_" + pdf['year'].astype(str)

    if is_polars:
        return _pl.from_pandas(pdf)
    return pdf


# =========================================================
# Main Transformation Function
# =========================================================
def calculate_player_stats(
    player_df: pl.DataFrame,
    ctx: LeagueContext,
    current_week: Optional[int] = None,
    current_year: Optional[int] = None
) -> pl.DataFrame:
    """
    Enrich player data with statistics, rankings, and performance metrics.

    Args:
        player_df: DataFrame with player data (must have yahoo_player_id join key)
        ctx: League context with paths to scoring rules and roster settings
        current_week: Current week (optional, for incremental updates)
        current_year: Current year (optional)

    Returns:
        DataFrame with enriched player statistics

    Required columns in player_df:
        - yahoo_player_id (primary join key)
        - manager
        - year
        - week
        - position
        - Stat columns (passing_yds, rushing_tds, etc.)

    Columns added:
        - fantasy_points
        - league_wide_optimal_player (boolean flag - best players league-wide including free agents)
        - league_wide_optimal_position (QB, WR1, WR2, RB1, RB2, W/R/T, etc.)
        - optimal_player (boolean flag - best players from each manager's roster)
        - optimal_position (manager-specific optimal position)
        - optimal_points (manager-specific optimal total - sum of optimal_player==1)
        - lineup_efficiency (actual vs optimal %)
        - bench_points
        - player_personal_week_rank, player_personal_week_pct
        - player_personal_season_rank, player_personal_season_pct
        - position_week_rank, position_week_pct
        - position_season_rank, position_season_pct
        - position_alltime_rank, position_alltime_pct
        - manager_player_week_rank, manager_player_week_pct
        - manager_player_season_rank, manager_player_season_pct
        - manager_player_alltime_rank, manager_player_alltime_pct
        - season_ppg, season_games
        - alltime_ppg, alltime_games
        - rolling_3_avg, rolling_5_avg
        - weighted_ppg
        - ppg_trend
        - consistency_score
    """
    print(f"Starting player stats transformation...")
    print(f"Input: {len(player_df)} player records")

    df = player_df.clone()

    # Normalize player identifier: prefer yahoo_player_id, fall back to nfl_player_id
    df = unify_player_id(df)

    # Validate required columns (player_id is required now instead of yahoo_player_id)
    required_cols = ["player_id", "manager", "year", "week", "position"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # =========================================================
    # Step 0: Calculate is_rostered and is_started flags
    # =========================================================

    # Set manager = "Unrostered" for players without a manager (unrostered NFL players)
    # This allows league-wide optimal lineup to work for all years (1999-2025)
    if "manager" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("manager").is_null())
            .then(pl.lit("Unrostered"))
            .otherwise(pl.col("manager"))
            .alias("manager")
        )
        print(f"  Set manager='Unrostered' for unrostered players")

    # Create is_rostered flag if it doesn't exist
    # A player is rostered if they have a manager assigned (not "Unrostered")
    if "is_rostered" not in df.columns:
        if "manager" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("manager") != "Unrostered")
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int64)
                .alias("is_rostered")
            )
            print(f"  Added is_rostered column based on manager presence")
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("is_rostered"))
            print(f"  Warning: manager column not found, set is_rostered=0 for all rows")

    # Create is_started flag based on fantasy_position
    # is_started = 1 if:
    #   - Player is rostered (is_rostered == 1), AND
    #   - fantasy_position is not "BN" (bench) or "IR" (injured reserve)
    # This indicates the player was in an active lineup slot
    if "fantasy_position" in df.columns:
        df = df.with_columns(
            pl.when(
                (pl.col("is_rostered") == 1) &
                pl.col("fantasy_position").is_not_null() &
                ~pl.col("fantasy_position").is_in(["BN", "IR"])
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int64)
            .alias("is_started")
        )
        print(f"  Added is_started column based on is_rostered and fantasy_position")
    else:
        # If fantasy_position doesn't exist, set is_started to 0 for all rows
        df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("is_started"))
        print(f"  Warning: fantasy_position column not found, set is_started=0 for all rows")

    # Populate the "started" column based on fantasy_position
    # started = 1 if fantasy_position is a valid position (not blank, null, "0", "BN", "IR", or empty)
    # started = 0 otherwise
    # This is for compatibility with older data that uses "started" column name
    if "fantasy_position" in df.columns:
        # Non-positions to exclude: null, empty string, "0", "BN", "IR"
        non_positions = ["", "0", "BN", "IR"]
        df = df.with_columns(
            pl.when(
                pl.col("fantasy_position").is_not_null() &
                (pl.col("fantasy_position").str.strip_chars() != "") &
                ~pl.col("fantasy_position").is_in(non_positions)
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int64)
            .alias("started")
        )
        print(f"  Populated started column based on fantasy_position (1 if valid position, 0 otherwise)")
    else:
        # If fantasy_position doesn't exist, set started to 0 for all rows
        df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("started"))
        print(f"  Warning: fantasy_position column not found, set started=0 for all rows")

    # =========================================================
    # Step 1: Load Scoring Rules and Roster Settings
    # =========================================================
    print("\nStep 1: Loading scoring rules and roster settings...")

    # Try to load from league context directories
    # League settings are now stored at top-level: data_directory/league_settings
    try:
        scoring_dir = ctx.yahoo_data_directory / "scoring_rules"  # type: ignore[attr-defined]
        roster_dir = ctx.yahoo_data_directory / "roster_settings"  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback: use data_directory / "league_settings" (league-wide config)
        scoring_dir = ctx.data_directory / "league_settings"
        roster_dir = ctx.data_directory / "league_settings"

    scoring_rules_by_year = load_scoring_rules(scoring_dir)

    # Load roster settings from JSON files
    roster_by_year = load_roster_settings_from_json(roster_dir, league_id=ctx.league_id)

    print(f"  Loaded scoring rules for {len(scoring_rules_by_year)} years")
    print(f"  Loaded roster settings for {len(roster_by_year)} years")

    # =========================================================
    # Step 2: Calculate Fantasy Points
    # =========================================================
    print("\nStep 2: Calculating fantasy points from scoring rules...")

    # PRESERVE YAHOO POINTS BEFORE CALCULATION
    # Yahoo rostered players (especially DST) already have fantasy points in 'points' column
    # We should prefer these over calculated points
    has_yahoo_points = "points" in df.columns
    if has_yahoo_points:
        df = df.with_columns(pl.col("points").alias("yahoo_points_original"))
        print(f"  Preserved Yahoo points in 'yahoo_points_original' column")

    # Calculate fantasy points from NFL stats using scoring rules
    # Pass league start year from context so pre-league years use earliest available settings
    # PRE-LEAGUE YEARS: For years before league start (e.g., 1999-2013 for KMFFL),
    # this will correctly calculate points using earliest available scoring rules
    df = calculate_fantasy_points(
        df,
        scoring_rules_by_year,
        year_col="year",
        league_start_year=ctx.start_year
    )
    print(f"  Added fantasy_points column (calculated from NFL stats)")

    # PREFER YAHOO POINTS OVER CALCULATED POINTS (when Yahoo points exist and are not null)
    # For rostered players (especially DST), Yahoo already calculated points correctly
    # For pre-league years or non-rostered players, use calculated fantasy_points
    if has_yahoo_points:
        df = df.with_columns(
            pl.when(
                pl.col("yahoo_points_original").is_not_null() &
                (pl.col("yahoo_points_original") != 0)  # Also check for non-zero (Yahoo uses 0 for 0 points, not NULL)
            )
            .then(pl.col("yahoo_points_original"))  # Use Yahoo points (rostered players)
            .otherwise(pl.col("fantasy_points"))     # Use calculated points (pre-league, non-rostered, or when Yahoo is null)
            .alias("fantasy_points")
        )
        print(f"  Merged Yahoo points (preferred for rostered) with calculated points (pre-league/non-rostered)")

    # For pre-league years specifically, log the behavior
    if ctx.start_year:
        pre_league_count = df.filter(pl.col("year") < ctx.start_year).shape[0]
        if pre_league_count > 0:
            print(f"  Pre-league years (<{ctx.start_year}): {pre_league_count:,} players using calculated points")

    # Create 'points' column as alias of 'fantasy_points' for consistency
    df = df.with_columns(pl.col("fantasy_points").alias("points"))
    print(f"  Added points column (alias of fantasy_points)")

    # =========================================================
    # Step 3: Determine League-Wide Optimal Players
    # =========================================================
    print("\nStep 3: Determining league-wide optimal players...")

    df = compute_league_wide_optimal_players(
        df,
        roster_by_year=roster_by_year,
        year_col="year",
        week_col="week",
        position_col="position",
        points_col="points"  # Use 'points' column (canonical)
    )
    print(f"  Added league_wide_optimal_player and league_wide_optimal_position")

    # =========================================================
    # Step 3.5: Determine Manager-Specific Optimal Players
    # =========================================================
    print("\nStep 3.5: Determining manager-specific optimal players...")

    df = compute_manager_optimal_players(
        df,
        roster_by_year=roster_by_year,
        manager_col="manager",
        year_col="year",
        week_col="week",
        position_col="position",
        points_col="points",  # Use 'points' column (canonical)
        rostered_col="is_rostered"
    )
    print(f"  Added optimal_player and optimal_position (manager-specific)")

    # =========================================================
    # Step 4: Calculate Optimal Lineup Metrics
    # =========================================================
    print("\nStep 4: Calculating optimal lineup metrics (using manager-specific optimal)...")

    df = calculate_optimal_lineup_metrics(
        df,
        manager_col="manager",
        year_col="year",
        week_col="week",
        points_col="points",  # Use 'points' column (canonical)
        actual_points_col="team_points"
    )
    print(f"  Added optimal_points, lineup_efficiency columns")

    df = calculate_bench_points(
        df,
        manager_col="manager",
        year_col="year",
        week_col="week",
        points_col="points",  # Use 'points' column (canonical)
        rostered_col="is_rostered",
        started_col="is_started"
    )
    print(f"  Added bench_points column")

    # =========================================================
    # Step 5: Add Player Rankings
    # =========================================================
    print("\nStep 5: Adding player rankings...")

    # Position rankings (league-wide)
    df = add_league_wide_position_ranks(
        df,
        points_col="points",
        position_col="position",
        year_col="year",
        week_col="week"
    )
    print(f"  Added league-wide position rankings")

    # Player personal rankings (comparing player to their own history)
    df = add_player_personal_ranks(
        df,
        player_id_col="yahoo_player_id",
        points_col="points",
        year_col="year",
        week_col="week"
    )
    print(f"  Added player personal rankings")

    # All players all-time rankings (cross-position)
    df = add_all_players_alltime_ranks(
        df,
        points_col="points",
        year_col="year",
        week_col="week"
    )
    print(f"  Added all-players all-time rankings (cross-position)")

    # Manager-specific rankings
    #
    # The manager-based ranking system can fail when the underlying
    # ranking module returns a mask that does not match the shape of
    # the DataFrame (e.g., polars ShapeError).  To keep the pipeline
    # running, wrap the call in a try/except.  If an exception is
    # raised, log a warning and skip the manager-specific rankings.
    try:
        df = add_manager_player_ranks(
            df,
            manager_col="manager",
            points_col="points",
            year_col="year",
            week_col="week"
        )
        print(f"  Added manager-player rankings")
    except Exception as e:
        print(f"  [WARN] Manager-player rankings skipped due to error: {e}")

    # Add lineup position (QB1, WR1, WR2, BN1, BN2, etc.)
    # Based on fantasy_position (actual roster slot), not nfl_position
    df = add_lineup_position(
        df,
        manager_col="manager",
        year_col="year",
        week_col="week",
        position_col="fantasy_position",
        points_col="points"
    )
    print(f"  Added lineup_position")

    # Add optimal lineup position (based on optimal_player flag)
    df = add_optimal_lineup_position(
        df,
        manager_col="manager",
        year_col="year",
        week_col="week",
        position_col="position",
        points_col="points",
        is_optimal_col="optimal_player"
    )
    print(f"  Added optimal_lineup_position")

    # Manager-Player history rankings (how player performed with this manager)
    df = add_manager_player_history_ranks(
        df,
        manager_col="manager",
        player_id_col="player_id",
        points_col="points",
        year_col="year",
        week_col="week"
    )
    print(f"  Added manager-player history rankings")

    # Manager-Position history rankings (how position performed with this manager)
    df = add_manager_position_history_ranks(
        df,
        manager_col="manager",
        position_col="position",
        points_col="points",
        year_col="year",
        week_col="week"
    )
    print(f"  Added manager-position history rankings")

    # Player Personal history rankings (player vs their own NFL career)
    df = add_player_personal_history_ranks(
        df,
        player_id_col="NFL_player_id",
        points_col="points",
        year_col="year",
        week_col="week"
    )
    print(f"  Added player personal history rankings")

    # Position history rankings (all QBs vs all QBs, all RBs vs all RBs, etc.)
    df = add_position_history_ranks(
        df,
        position_col="position",
        points_col="points",
        year_col="year",
        week_col="week"
    )
    print(f"  Added position history rankings")

    # Manager all-player history rankings (cross-position comparison)
    # This enables comparing Josh Allen's 52-pt game to Alvin Kamara's 50-pt game for the same manager
    df = add_manager_all_player_history_ranks(
        df,
        manager_col="manager",
        points_col="points",
        year_col="year",
        week_col="week"
    )
    print(f"  Added manager all-player history rankings (cross-position)")

    # Add position helper columns (position_group and position_key)
    # position_group: simplified position grouping (SKILL, DEF, K)
    # position_key: normalized position for joins
    df = df.with_columns([
        pl.when(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        .then(pl.lit("SKILL"))
        .when(pl.col("position").is_in(["DEF", "DST", "D/ST"]))
        .then(pl.lit("DEF"))
        .when(pl.col("position") == "K")
        .then(pl.lit("K"))
        .otherwise(pl.lit("OTHER"))
        .alias("position_group"),

        pl.col("position").str.to_uppercase().str.strip_chars().alias("position_key")
    ])
    print(f"  Added position_group and position_key columns")

    # =========================================================
    # Step 6: Calculate PPG Metrics
    # =========================================================
    print("\nStep 6: Calculating PPG metrics...")

    # Calculate rolling point total (cumulative sum per player per season)
    df = calculate_rolling_point_total(
        df,
        player_id_col="NFL_player_id",
        points_col="fantasy_points",
        year_col="year",
        week_col="week"
    )
    print(f"  Added rolling_point_total")

    df = calculate_season_ppg(
        df,
        player_id_col="NFL_player_id",
        points_col="fantasy_points",
        year_col="year",
        min_games=1
    )
    print(f"  Added season_ppg, season_games")

    df = calculate_alltime_ppg(
        df,
        player_id_col="NFL_player_id",
        points_col="fantasy_points",
        min_games=1
    )
    print(f"  Added alltime_ppg, alltime_games")

    df = calculate_rolling_avg(
        df,
        player_id_col="NFL_player_id",
        points_col="fantasy_points",
        year_col="year",
        week_col="week",
        window_size=3,
        output_col="rolling_3_avg"
    )
    print(f"  Added rolling_3_avg")

    df = calculate_rolling_avg(
        df,
        player_id_col="NFL_player_id",
        points_col="fantasy_points",
        year_col="year",
        week_col="week",
        window_size=5,
        output_col="rolling_5_avg"
    )
    print(f"  Added rolling_5_avg")

    df = calculate_weighted_ppg(
        df,
        player_id_col="NFL_player_id",
        points_col="fantasy_points",
        year_col="year",
        week_col="week",
        decay_factor=0.9
    )
    print(f"  Added weighted_ppg")

    df = calculate_ppg_trend(
        df,
        player_id_col="NFL_player_id",
        year_col="year"
    )
    print(f"  Added ppg_trend")

    df = calculate_consistency_score(
        df,
        player_id_col="NFL_player_id",
        points_col="fantasy_points",
        year_col="year",
        min_games=3
    )
    print(f"  Added consistency_score")

    # =========================================================
    # Final Step: Validation
    # =========================================================
    print("\nValidating output...")

    # Verify player_id is present (canonical join key)
    if "player_id" not in df.columns:
        raise ValueError("player_id column missing - required for cross-file joins")

    print(f"\nPlayer stats transformation complete!")
    print(f"Output: {len(df)} player records with enriched statistics")

    return df


# =========================================================
# CLI Interface
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Calculate player statistics and rankings for multi-league setup"
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="Path to league_context.json"
    )
    parser.add_argument(
        "--current-week",
        type=int,
        help="Current week number (for incremental updates)"
    )
    parser.add_argument(
        "--current-year",
        type=int,
        help="Current year (for incremental updates)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Override input player data path (default: player_data/player.parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Override output path (default: player_data/player_stats.parquet)"
    )

    args = parser.parse_args()

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"Loaded league context: {ctx.league_name}")

    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = ctx.canonical_player_file

    if not input_path.exists():
        raise FileNotFoundError(f"Player data not found: {input_path}")

    # Load player data
    player_df = pl.read_parquet(input_path)
    print(f"Loaded {len(player_df)} player records from {input_path}")

    # Calculate player stats
    enriched_df = calculate_player_stats(
        player_df,
        ctx=ctx,
        current_week=args.current_week,
        current_year=args.current_year
    )

    # Determine output path (write back to canonical player file)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ctx.canonical_player_file

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.write_parquet(output_path)
    print(f"\nUpdated player data: {output_path}")

    # Also save CSV
    csv_path = output_path.with_suffix(".csv")
    enriched_df.write_csv(csv_path)
    print(f"Updated player data (CSV): {csv_path}")


if __name__ == "__main__":
    main()
