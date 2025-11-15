"""
PPG Calculator Module

Calculates points-per-game metrics and rolling averages.

Memory-optimized version that uses lazy evaluation when possible.

This module:
- Calculates all-time PPG
- Calculates rolling averages (last 3, last 5, etc.)
- Handles minimum game thresholds
- Optimized for memory efficiency using lazy evaluation
"""

from typing import Optional, Union
import polars as pl


def calculate_season_ppg(
    df: Union[pl.DataFrame, pl.LazyFrame],
    player_id_col: str = "yahoo_player_id",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    min_games: int = 1
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Calculate season PPG for each player-year.

    Memory-optimized version that uses lazy evaluation when possible.

    Args:
        df: DataFrame or LazyFrame with player stats
        player_id_col: Player identifier column
        points_col: Points column
        year_col: Year column
        min_games: Minimum games to qualify for PPG

    Returns:
        DataFrame or LazyFrame with season_ppg and season_games columns added
    """
    # Convert to lazy if not already
    is_lazy = isinstance(df, pl.LazyFrame)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Drop existing columns to prevent duplicates
    cols_to_drop = ["season_ppg", "season_games"]
    existing_cols = lf.collect_schema().names() if is_lazy else df.columns
    cols_to_drop = [c for c in cols_to_drop if c in existing_cols]
    if cols_to_drop:
        lf = lf.drop(cols_to_drop)

    # Aggregate by player-season and calculate PPG in one go
    season_agg = (
        lf.group_by([player_id_col, year_col])
        .agg([
            pl.col(points_col).sum().alias("_season_total"),
            pl.col(points_col).count().alias("season_games")
        ])
        .with_columns(
            pl.when(pl.col("season_games") >= min_games)
            .then((pl.col("_season_total") / pl.col("season_games")).round(2))
            .otherwise(None)
            .alias("season_ppg")
        )
        .select([player_id_col, year_col, "season_ppg", "season_games"])
    )

    # Join back to main df
    result = lf.join(season_agg, on=[player_id_col, year_col], how="left")

    return result if is_lazy else result.collect()


def calculate_alltime_ppg(
    df: Union[pl.DataFrame, pl.LazyFrame],
    player_id_col: str = "yahoo_player_id",
    points_col: str = "fantasy_points",
    min_games: int = 1
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Calculate all-time PPG for each player across all seasons.

    Memory-optimized version that uses lazy evaluation when possible.

    Args:
        df: DataFrame or LazyFrame with player stats
        player_id_col: Player identifier column
        points_col: Points column
        min_games: Minimum games to qualify for PPG

    Returns:
        DataFrame or LazyFrame with alltime_ppg and alltime_games columns added
    """
    # Convert to lazy if not already
    is_lazy = isinstance(df, pl.LazyFrame)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Drop existing columns to prevent duplicates
    cols_to_drop = ["alltime_ppg", "alltime_games"]
    existing_cols = lf.collect_schema().names() if is_lazy else df.columns
    cols_to_drop = [c for c in cols_to_drop if c in existing_cols]
    if cols_to_drop:
        lf = lf.drop(cols_to_drop)

    # Aggregate by player and calculate alltime PPG
    alltime_agg = (
        lf.group_by([player_id_col])
        .agg([
            pl.col(points_col).sum().alias("_alltime_total"),
            pl.col(points_col).count().alias("alltime_games")
        ])
        .with_columns(
            pl.when(pl.col("alltime_games") >= min_games)
            .then((pl.col("_alltime_total") / pl.col("alltime_games")).round(2))
            .otherwise(None)
            .alias("alltime_ppg")
        )
        .select([player_id_col, "alltime_ppg", "alltime_games"])
    )

    # Join back to main df
    result = lf.join(alltime_agg, on=[player_id_col], how="left")

    return result if is_lazy else result.collect()


def calculate_rolling_avg(
    df: Union[pl.DataFrame, pl.LazyFrame],
    player_id_col: str = "yahoo_player_id",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week",
    window_size: int = 3,
    output_col: Optional[str] = None
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Calculate rolling average over last N games.

    Memory-optimized version that uses lazy evaluation when possible.

    Args:
        df: DataFrame or LazyFrame with player stats
        player_id_col: Player identifier column
        points_col: Points column
        year_col: Year column
        week_col: Week column
        window_size: Number of games to average (e.g., 3 for last 3 games)
        output_col: Name for output column (default: "rolling_{window_size}_avg")

    Returns:
        DataFrame or LazyFrame with rolling average column added
    """
    if output_col is None:
        output_col = f"rolling_{window_size}_avg"

    # Convert to lazy if not already
    is_lazy = isinstance(df, pl.LazyFrame)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Chain sort and with_columns to avoid intermediate copy
    result = (
        lf.sort([player_id_col, year_col, week_col])
        .with_columns(
            pl.col(points_col)
            .rolling_mean(window_size=window_size, min_periods=1)
            .over([player_id_col])
            .round(2)
            .alias(output_col)
        )
    )

    return result if is_lazy else result.collect()


def calculate_weighted_ppg(
    df: Union[pl.DataFrame, pl.LazyFrame],
    player_id_col: str = "yahoo_player_id",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week",
    decay_factor: float = 0.9
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Calculate exponentially weighted PPG (recent games weighted higher).

    Memory-optimized version that uses lazy evaluation when possible.

    Args:
        df: DataFrame or LazyFrame with player stats
        player_id_col: Player identifier column
        points_col: Points column
        year_col: Year column
        week_col: Week column
        decay_factor: Weight decay (0.9 = recent games 10% more important)

    Returns:
        DataFrame or LazyFrame with weighted_ppg column added
    """
    # Convert to lazy if not already
    is_lazy = isinstance(df, pl.LazyFrame)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Chain sort and with_columns to avoid intermediate copy
    result = (
        lf.sort([player_id_col, year_col, week_col])
        .with_columns(
            pl.col(points_col)
            .ewm_mean(alpha=1 - decay_factor, min_periods=1)
            .over([player_id_col])
            .round(2)
            .alias("weighted_ppg")
        )
    )

    return result if is_lazy else result.collect()


def calculate_ppg_trend(
    df: Union[pl.DataFrame, pl.LazyFrame],
    player_id_col: str = "yahoo_player_id",
    year_col: str = "year"
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Calculate PPG trend (current season vs all-time).

    Positive trend = player improving
    Negative trend = player declining

    Memory-optimized version that uses lazy evaluation when possible.

    Args:
        df: DataFrame or LazyFrame with player stats (must have season_ppg and alltime_ppg)
        player_id_col: Player identifier column
        year_col: Year column

    Returns:
        DataFrame or LazyFrame with ppg_trend column added
    """
    # Convert to lazy if not already
    is_lazy = isinstance(df, pl.LazyFrame)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Check columns - need to collect schema for lazy frames
    columns = lf.collect_schema().names()

    if "season_ppg" not in columns or "alltime_ppg" not in columns:
        result = lf.with_columns(pl.lit(None).alias("ppg_trend"))
    else:
        result = lf.with_columns(
            (pl.col("season_ppg") - pl.col("alltime_ppg"))
            .round(2)
            .alias("ppg_trend")
        )

    return result if is_lazy else result.collect()


def calculate_consistency_score(
    df: Union[pl.DataFrame, pl.LazyFrame],
    player_id_col: str = "yahoo_player_id",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    min_games: int = 3
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Calculate consistency score (coefficient of variation).

    Lower score = more consistent
    Higher score = more boom/bust

    Memory-optimized version that uses lazy evaluation when possible.

    Args:
        df: DataFrame or LazyFrame with player stats
        player_id_col: Player identifier column
        points_col: Points column
        year_col: Year column
        min_games: Minimum games to calculate consistency

    Returns:
        DataFrame or LazyFrame with consistency_score column added
    """
    # Convert to lazy if not already
    is_lazy = isinstance(df, pl.LazyFrame)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Drop existing column to prevent duplicates
    existing_cols = lf.collect_schema().names() if is_lazy else df.columns
    if "consistency_score" in existing_cols:
        lf = lf.drop("consistency_score")

    # Calculate season mean, std, and consistency score in one go
    season_stats = (
        lf.group_by([player_id_col, year_col])
        .agg([
            pl.col(points_col).mean().alias("_mean"),
            pl.col(points_col).std().alias("_std"),
            pl.col(points_col).count().alias("_games")
        ])
        .with_columns(
            pl.when((pl.col("_games") >= min_games) & (pl.col("_mean") > 0))
            .then((pl.col("_std") / pl.col("_mean") * 100).round(2))
            .otherwise(None)
            .alias("consistency_score")
        )
        .select([player_id_col, year_col, "consistency_score"])
    )

    result = lf.join(season_stats, on=[player_id_col, year_col], how="left")

    return result if is_lazy else result.collect()


def calculate_rolling_point_total(
    df: Union[pl.DataFrame, pl.LazyFrame],
    player_id_col: str = "nfl_player_id",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week"
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Calculate rolling cumulative point total for each player within each season.

    This is a cumulative sum that resets each season. For example:
    - Week 1: 10 points -> rolling_point_total = 10
    - Week 2: 15 points -> rolling_point_total = 25
    - Week 3: 8 points -> rolling_point_total = 33

    Uses nfl_player_id to track the actual NFL player across their career.

    Memory-optimized version that uses lazy evaluation when possible.

    Args:
        df: DataFrame or LazyFrame with player stats
        player_id_col: Player identifier column (should be nfl_player_id)
        points_col: Points column to sum
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame or LazyFrame with rolling_point_total column added
    """
    # Convert to lazy if not already
    is_lazy = isinstance(df, pl.LazyFrame)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Check if the player_id_col exists
    columns = lf.collect_schema().names()
    if player_id_col not in columns:
        result = lf.with_columns(pl.lit(None).alias("rolling_point_total"))
    else:
        # Chain sort and with_columns to avoid intermediate copy
        result = (
            lf.sort([player_id_col, year_col, week_col])
            .with_columns(
                pl.col(points_col)
                .cum_sum()
                .over([player_id_col, year_col])
                .round(2)
                .alias("rolling_point_total")
            )
        )

    return result if is_lazy else result.collect()
