"""
Player Rankings Module

Handles all player ranking calculations with tiebreakers.

This module:
- Creates player personal rankings (all-time, season, weekly)
- Creates league-wide position rankings
- Creates manager-specific rankings
- Handles proper tiebreaking logic
"""

import polars as pl
from typing import List, Optional


def add_ranks_with_tiebreaker(
    df: pl.DataFrame,
    points_col: str,
    partition_cols: List[str],
    rank_name: str,
    pct_name: str,
    valid_mask: Optional[pl.Expr] = None
) -> pl.DataFrame:
    """
    Add rank and percentile columns with proper tiebreaker logic.

    Ranks descending by points (higher points = lower rank number).
    Uses ordinal ranking (1, 2, 3, 4...) to break ties.

    Args:
        df: DataFrame to rank
        points_col: Column to rank by
        partition_cols: Columns to partition by (e.g., ["year", "week"]).
                       Empty list means rank across entire dataframe.
        rank_name: Name for rank column
        pct_name: Name for percentile column
        valid_mask: Optional mask for valid rows (nulls otherwise)

    Returns:
        DataFrame with rank and percentile columns added
    """
    if valid_mask is None:
        valid_mask = pl.lit(True)

    # Handle empty partition_cols (rank entire dataframe, no grouping)
    if not partition_cols:
        # Rank without partitioning (entire dataframe)
        rank_expr = (
            pl.when(valid_mask)
            .then(
                pl.col(points_col)
                .rank(method="ordinal", descending=True)
            )
            .otherwise(None)
        )

        # Percentile without partitioning
        pct_expr = (
            pl.when(valid_mask)
            .then(
                (100.0 * pl.col(points_col).rank(method="average", descending=False) /
                 pl.col(points_col).count())
                .round(2)
            )
            .otherwise(None)
        )
    else:
        # Rank within partition (descending = higher points get lower rank)
        rank_expr = (
            pl.when(valid_mask)
            .then(
                pl.col(points_col)
                .rank(method="ordinal", descending=True)
                .over(partition_cols)
            )
            .otherwise(None)
        )

        # Percentile (0-100, higher is better)
        pct_expr = (
            pl.when(valid_mask)
            .then(
                (100.0 * pl.col(points_col).rank(method="average", descending=False).over(partition_cols) /
                 pl.col(points_col).count().over(partition_cols))
                .round(2)
            )
            .otherwise(None)
        )

    return df.with_columns([
        rank_expr.alias(rank_name),
        pct_expr.alias(pct_name)
    ])


def add_player_personal_ranks(
    df: pl.DataFrame,
    player_id_col: str = "yahoo_player_id",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week"
) -> pl.DataFrame:
    """
    Add player personal rankings (comparing player to their own history).

    Adds:
    - player_personal_week_rank: Rank within this player's all weekly performances
    - player_personal_week_pct: Percentile within player's weekly performances
    - player_personal_season_rank: Rank within this player's seasons
    - player_personal_season_pct: Percentile within player's seasons

    Args:
        df: DataFrame with player stats
        player_id_col: Player identifier column
        points_col: Points column to rank
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame with personal rank columns added
    """
    # Weekly personal ranks (within player's all weeks)
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[player_id_col],
        rank_name="player_personal_week_rank",
        pct_name="player_personal_week_pct",
        valid_mask=pl.col(points_col).is_not_null()
    )

    # Seasonal aggregation for personal season ranks
    season_agg = (
        df.group_by([player_id_col, year_col])
        .agg(pl.col(points_col).sum().alias("_season_points"))
    )

    season_agg = add_ranks_with_tiebreaker(
        season_agg,
        points_col="_season_points",
        partition_cols=[player_id_col],
        rank_name="player_personal_season_rank",
        pct_name="player_personal_season_pct",
        valid_mask=pl.col("_season_points").is_not_null()
    )

    # Join back to main df
    # Drop existing columns if present to avoid duplicates
    cols_to_drop = ["player_personal_season_rank", "player_personal_season_pct"]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    if existing_cols:
        df = df.drop(existing_cols)

    df = df.join(
        season_agg.select([
            player_id_col,
            year_col,
            "player_personal_season_rank",
            "player_personal_season_pct"
        ]),
        on=[player_id_col, year_col],
        how="left"
    )

    return df


def add_league_wide_position_ranks(
    df: pl.DataFrame,
    points_col: str = "fantasy_points",
    position_col: str = "position",
    year_col: str = "year",
    week_col: str = "week"
) -> pl.DataFrame:
    """
    Add league-wide position rankings.

    Adds:
    - position_week_rank: Rank among all players at this position this week
    - position_week_pct: Percentile among position this week
    - position_season_rank: Rank among all players at this position this season
    - position_season_pct: Percentile among position this season
    - position_alltime_rank: Rank among all players at this position all-time
    - position_alltime_pct: Percentile among position all-time

    Args:
        df: DataFrame with player stats
        points_col: Points column
        position_col: Position column
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame with position rank columns added
    """
    # Weekly position ranks
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[position_col, year_col, week_col],
        rank_name="position_week_rank",
        pct_name="position_week_pct",
        valid_mask=pl.col(points_col).is_not_null()
    )

    # Season position ranks
    season_agg = (
        df.group_by([position_col, year_col, "yahoo_player_id"])
        .agg(pl.col(points_col).sum().alias("_season_points"))
    )

    season_agg = add_ranks_with_tiebreaker(
        season_agg,
        points_col="_season_points",
        partition_cols=[position_col, year_col],
        rank_name="position_season_rank",
        pct_name="position_season_pct",
        valid_mask=pl.col("_season_points").is_not_null()
    )

    # Drop existing columns if they exist to avoid conflicts
    cols_to_drop = [c for c in ["position_season_rank", "position_season_pct"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    df = df.join(
        season_agg.select([
            "yahoo_player_id",
            position_col,
            year_col,
            "position_season_rank",
            "position_season_pct"
        ]),
        on=["yahoo_player_id", position_col, year_col],
        how="left"
    )

    # All-time position ranks
    alltime_agg = (
        df.group_by([position_col, "yahoo_player_id"])
        .agg(pl.col(points_col).sum().alias("_alltime_points"))
    )

    alltime_agg = add_ranks_with_tiebreaker(
        alltime_agg,
        points_col="_alltime_points",
        partition_cols=[position_col],
        rank_name="position_alltime_rank",
        pct_name="position_alltime_pct",
        valid_mask=pl.col("_alltime_points").is_not_null()
    )

    # Drop existing columns if they exist to avoid conflicts
    cols_to_drop = [c for c in ["position_alltime_rank", "position_alltime_pct"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    df = df.join(
        alltime_agg.select([
            "yahoo_player_id",
            position_col,
            "position_alltime_rank",
            "position_alltime_pct"
        ]),
        on=["yahoo_player_id", position_col],
        how="left"
    )

    return df


def add_all_players_alltime_ranks(
    df: pl.DataFrame,
    points_col: str = "points",
    year_col: str = "year",
    week_col: str = "week"
) -> pl.DataFrame:
    """
    Add all-players all-time rankings (cross-position).

    This ranks ALL individual player performances (every score ever), regardless of position.
    For example: "Tom Brady's 37.74-point game in week 5 ranks #9 all-time across all players"

    Percentile calculations exclude zero-point performances.

    Adds:
    - all_players_week_rank: Rank among ALL players this week (across positions)
    - all_players_week_pct: Percentile among all players this week (excludes zeros)
    - all_players_season_rank: Rank among ALL individual performances this season
    - all_players_season_pct: Percentile among all performances this season (excludes zeros)
    - all_players_alltime_rank: Rank among ALL individual performances all-time
    - all_players_alltime_pct: Percentile among all performances all-time (excludes zeros)

    Args:
        df: DataFrame with player stats
        points_col: Points column
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame with all-players rank columns added
    """
    # Weekly all-players ranks (no position filtering)
    # Percentile excludes zeros
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[year_col, week_col],
        rank_name="all_players_week_rank",
        pct_name="all_players_week_pct",
        valid_mask=pl.col(points_col).is_not_null() & (pl.col(points_col) > 0)
    )

    # Season all-players ranks - rank every individual weekly performance within the season
    # Not summed, but individual weekly scores
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[year_col],
        rank_name="all_players_season_rank",
        pct_name="all_players_season_pct",
        valid_mask=pl.col(points_col).is_not_null() & (pl.col(points_col) > 0)
    )

    # All-time all-players ranks - rank every individual weekly performance ever
    # Not summed, but individual weekly scores across all years
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[],  # No partitioning - truly all-time
        rank_name="all_players_alltime_rank",
        pct_name="all_players_alltime_pct",
        valid_mask=pl.col(points_col).is_not_null() & (pl.col(points_col) > 0)
    )

    return df


def add_manager_player_ranks(
    df: pl.DataFrame,
    manager_col: str = "manager",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week"
) -> pl.DataFrame:
    """
    Add manager-specific player rankings.

    Adds:
    - manager_player_week_rank: Rank among this manager's players this week
    - manager_player_week_pct: Percentile among manager's players this week
    - manager_player_season_rank: Rank among manager's players this season
    - manager_player_season_pct: Percentile among manager's players this season
    - manager_player_alltime_rank: Rank among manager's all-time players
    - manager_player_alltime_pct: Percentile among manager's all-time players

    Args:
        df: DataFrame with player stats
        manager_col: Manager column
        points_col: Points column
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame with manager-player rank columns added
    """
    # Only rank rostered players
    rostered_mask = pl.col("is_rostered") == 1 if "is_rostered" in df.columns else pl.lit(True)

    # Weekly manager-player ranks
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[manager_col, year_col, week_col],
        rank_name="manager_player_week_rank",
        pct_name="manager_player_week_pct",
        valid_mask=rostered_mask & pl.col(points_col).is_not_null()
    )

    # Season manager-player ranks
    season_agg = (
        df.filter(rostered_mask)
        .group_by([manager_col, year_col, "yahoo_player_id"])
        .agg(pl.col(points_col).sum().alias("_season_points"))
    )

    season_agg = add_ranks_with_tiebreaker(
        season_agg,
        points_col="_season_points",
        partition_cols=[manager_col, year_col],
        rank_name="manager_player_season_rank",
        pct_name="manager_player_season_pct",
        valid_mask=pl.col("_season_points").is_not_null()
    )

    # Drop existing columns if they exist to avoid conflicts
    cols_to_drop = [c for c in ["manager_player_season_rank", "manager_player_season_pct"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    df = df.join(
        season_agg.select([
            "yahoo_player_id",
            manager_col,
            year_col,
            "manager_player_season_rank",
            "manager_player_season_pct"
        ]),
        on=["yahoo_player_id", manager_col, year_col],
        how="left"
    )

    # All-time manager-player ranks
    alltime_agg = (
        df.filter(rostered_mask)
        .group_by([manager_col, "yahoo_player_id"])
        .agg(pl.col(points_col).sum().alias("_alltime_points"))
    )

    alltime_agg = add_ranks_with_tiebreaker(
        alltime_agg,
        points_col="_alltime_points",
        partition_cols=[manager_col],
        rank_name="manager_player_alltime_rank",
        pct_name="manager_player_alltime_pct",
        valid_mask=pl.col("_alltime_points").is_not_null()
    )

    # Drop existing columns if they exist to avoid conflicts
    cols_to_drop = [c for c in ["manager_player_alltime_rank", "manager_player_alltime_pct"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    df = df.join(
        alltime_agg.select([
            "yahoo_player_id",
            manager_col,
            "manager_player_alltime_rank",
            "manager_player_alltime_pct"
        ]),
        on=["yahoo_player_id", manager_col],
        how="left"
    )

    return df


def add_lineup_position(
    df: pl.DataFrame,
    manager_col: str = "manager",
    year_col: str = "year",
    week_col: str = "week",
    position_col: str = "position",
    points_col: str = "fantasy_points"
) -> pl.DataFrame:
    """
    Add lineup_position column (e.g., QB1, WR1, WR2, RB1, RB2, etc.).

    This ranks players within each manager's weekly roster by position and points,
    showing which QB/RB/WR/TE/etc. they were for that manager that week.

    Only populates for rostered players (manager is not null).

    Args:
        df: DataFrame with player stats
        manager_col: Manager column
        year_col: Year column
        week_col: Week column
        position_col: Position column (fantasy_position)
        points_col: Points column to rank by

    Returns:
        DataFrame with lineup_position column added
    """
    # Only rank rostered players (manager not null)
    rostered_mask = pl.col(manager_col).is_not_null()

    # Rank players within manager/week/position by points (descending)
    position_rank = (
        pl.when(rostered_mask)
        .then(
            pl.col(points_col)
            .rank(method="ordinal", descending=True)
            .over([manager_col, year_col, week_col, position_col])
        )
        .otherwise(None)
    )

    # Create lineup_position by concatenating position + rank
    # e.g., "QB" + 1 = "QB1", "WR" + 2 = "WR2"
    lineup_position_expr = (
        pl.when(rostered_mask)
        .then(
            pl.concat_str([
                pl.col(position_col),
                position_rank.cast(pl.Utf8)
            ])
        )
        .otherwise(None)
    )

    df = df.with_columns([
        lineup_position_expr.alias("lineup_position")
    ])

    return df


def add_manager_player_history_ranks(
    df: pl.DataFrame,
    manager_col: str = "manager",
    player_id_col: str = "yahoo_player_id",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week"
) -> pl.DataFrame:
    """
    Add manager-player history rankings.

    This ranks each player's performances across all times they've been
    on this specific manager's team.

    For example: "This was Lamar Jackson's 3rd best week when rostered by Daniel"

    Adds:
    - manager_player_all_time_history: Rank of this week among all weeks this player was on this manager's team
    - manager_player_all_time_history_percentile: Percentile ranking
    - manager_player_season_history: Rank among weeks in this season with this manager
    - manager_player_season_history_percentile: Percentile ranking

    Args:
        df: DataFrame with player stats
        manager_col: Manager column
        player_id_col: Player identifier column
        points_col: Points column to rank by
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame with manager-player history rank columns added
    """
    # Only rank rostered players (manager not null)
    rostered_mask = pl.col(manager_col).is_not_null()

    # All-time manager-player history ranks
    # Rank each player's performances across all time with this manager
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[manager_col, player_id_col],
        rank_name="manager_player_all_time_history",
        pct_name="manager_player_all_time_history_percentile",
        valid_mask=rostered_mask & pl.col(points_col).is_not_null()
    )

    # Season manager-player history ranks
    # Rank each player's performances within this season with this manager
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[manager_col, player_id_col, year_col],
        rank_name="manager_player_season_history",
        pct_name="manager_player_season_history_percentile",
        valid_mask=rostered_mask & pl.col(points_col).is_not_null()
    )

    return df


def add_manager_position_history_ranks(
    df: pl.DataFrame,
    manager_col: str = "manager",
    position_col: str = "position",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week"
) -> pl.DataFrame:
    """
    Add manager-position history rankings.

    This ranks how all players at a given position performed when
    rostered by this manager.

    For example: "Among all QBs Daniel has ever started, this was Lamar's 23rd best performance"

    Adds:
    - manager_position_all_time_history: Rank among all weeks with this manager-position combo
    - manager_position_all_time_history_percentile: Percentile ranking
    - manager_position_season_history: Rank within this season for this manager-position combo
    - manager_position_season_history_percentile: Percentile ranking

    Args:
        df: DataFrame with player stats
        manager_col: Manager column
        position_col: Position column
        points_col: Points column to rank by
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame with manager-position history rank columns added
    """
    # Only rank rostered players (manager not null)
    rostered_mask = pl.col(manager_col).is_not_null()

    # All-time manager-position history ranks
    # Rank all performances at this position with this manager across all time
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[manager_col, position_col],
        rank_name="manager_position_all_time_history",
        pct_name="manager_position_all_time_history_percentile",
        valid_mask=rostered_mask & pl.col(points_col).is_not_null()
    )

    # Season manager-position history ranks
    # Rank performances at this position with this manager within this season
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[manager_col, position_col, year_col],
        rank_name="manager_position_season_history",
        pct_name="manager_position_season_history_percentile",
        valid_mask=rostered_mask & pl.col(points_col).is_not_null()
    )

    return df


def add_optimal_lineup_position(
    df: pl.DataFrame,
    manager_col: str = "manager",
    year_col: str = "year",
    week_col: str = "week",
    position_col: str = "position",
    points_col: str = "fantasy_points",
    is_optimal_col: str = "is_optimal"
) -> pl.DataFrame:
    """
    Add optimal_lineup_position column (e.g., QB1, WR1, WR2, RB1, RB2, etc.).

    This is like lineup_position but based on the OPTIMAL lineup (highest scoring players),
    not the actual lineup the manager started.

    Only populates for rostered players where is_optimal flag is True.

    Args:
        df: DataFrame with player stats
        manager_col: Manager column
        year_col: Year column
        week_col: Week column
        position_col: Position column (fantasy_position)
        points_col: Points column to rank by
        is_optimal_col: Column indicating if player was optimal

    Returns:
        DataFrame with optimal_lineup_position column added
    """
    # Check if is_optimal column exists
    if is_optimal_col not in df.columns:
        # If no is_optimal flag, set all to None
        return df.with_columns([
            pl.lit(None).alias("optimal_lineup_position")
        ])

    # Only rank rostered players who would have been optimal starters
    rostered_mask = pl.col(manager_col).is_not_null()
    optimal_mask = rostered_mask & (pl.col(is_optimal_col) == 1)

    # Rank players within manager/week/position by points (descending)
    # Only among optimal players for this manager
    position_rank = (
        pl.when(optimal_mask)
        .then(
            pl.col(points_col)
            .rank(method="ordinal", descending=True)
            .over([manager_col, year_col, week_col, position_col])
        )
        .otherwise(None)
    )

    # Create optimal_lineup_position by concatenating position + rank
    # e.g., "QB" + 1 = "QB1", "WR" + 2 = "WR2"
    optimal_lineup_position_expr = (
        pl.when(optimal_mask)
        .then(
            pl.concat_str([
                pl.col(position_col),
                position_rank.cast(pl.Utf8)
            ])
        )
        .otherwise(None)
    )

    df = df.with_columns([
        optimal_lineup_position_expr.alias("optimal_lineup_position")
    ])

    return df


def add_player_personal_history_ranks(
    df: pl.DataFrame,
    player_id_col: str = "nfl_player_id",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week"
) -> pl.DataFrame:
    """
    Add player personal history rankings (player compared to their own career).

    This ranks each player's performances across their entire career,
    regardless of which fantasy manager rostered them.

    Uses nfl_player_id to track the actual NFL player's performance history.

    For example: "This was Lamar Jackson's 5th best game of his career"

    Adds:
    - player_personal_all_time_history: Rank among all games this player has ever played
    - player_personal_all_time_history_percentile: Percentile ranking
    - player_personal_season_history: Rank among games in this player's season
    - player_personal_season_history_percentile: Percentile ranking

    Args:
        df: DataFrame with player stats
        player_id_col: Player identifier column (should be nfl_player_id)
        points_col: Points column to rank by
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame with player personal history rank columns added
    """
    # Check if the player_id_col exists
    if player_id_col not in df.columns:
        # If nfl_player_id doesn't exist, return with null columns
        return df.with_columns([
            pl.lit(None).alias("player_personal_all_time_history"),
            pl.lit(None).alias("player_personal_all_time_history_percentile"),
            pl.lit(None).alias("player_personal_season_history"),
            pl.lit(None).alias("player_personal_season_history_percentile")
        ])

    # Only rank where player_id is not null and points are not null
    valid_mask = pl.col(player_id_col).is_not_null() & pl.col(points_col).is_not_null()

    # All-time personal history ranks
    # Rank each player's performances across their entire career
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[player_id_col],
        rank_name="player_personal_all_time_history",
        pct_name="player_personal_all_time_history_percentile",
        valid_mask=valid_mask
    )

    # Season personal history ranks
    # Rank each player's performances within each season
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[player_id_col, year_col],
        rank_name="player_personal_season_history",
        pct_name="player_personal_season_history_percentile",
        valid_mask=valid_mask
    )

    return df


def add_position_history_ranks(
    df: pl.DataFrame,
    position_col: str = "position",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week"
) -> pl.DataFrame:
    """
    Add position-based history rankings (all players at same position compared).

    This ranks all QBs against all QBs, all RBs against all RBs, etc.
    across all time and within seasons.

    For percentile calculations, excludes zero-point performances to avoid
    skewing percentiles with injury/bye weeks.

    For example: "Among all QB performances ever, this was the 234th best"

    Adds:
    - position_all_time_history: Rank among all performances at this position (all-time)
    - position_all_time_history_percentile: Percentile ranking (excludes zeros)
    - position_season_history: Rank among all performances at this position this season
    - position_season_history_percentile: Percentile ranking (excludes zeros)

    Args:
        df: DataFrame with player stats
        position_col: Position column
        points_col: Points column to rank by
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame with position history rank columns added
    """
    # Only rank where position and points are not null
    valid_mask = pl.col(position_col).is_not_null() & pl.col(points_col).is_not_null()

    # For percentile, exclude zeros (bye weeks, injuries, DNPs)
    # But still include zeros in the rank itself
    percentile_mask = valid_mask & (pl.col(points_col) > 0)

    # All-time position history ranks (all players at this position ever)
    # Rank includes zeros
    rank_expr = (
        pl.when(valid_mask)
        .then(
            pl.col(points_col)
            .rank(method="ordinal", descending=True)
            .over([position_col])
        )
        .otherwise(None)
    )

    # Percentile excludes zeros for better distribution
    pct_expr = (
        pl.when(percentile_mask)
        .then(
            (100.0 * pl.col(points_col).rank(method="average", descending=False)
             .over([position_col]) /
             (pl.col(points_col) > 0).sum().over([position_col]))
            .round(2)
        )
        .otherwise(None)
    )

    df = df.with_columns([
        rank_expr.alias("position_all_time_history"),
        pct_expr.alias("position_all_time_history_percentile")
    ])

    # Season position history ranks (all players at this position this season)
    season_rank_expr = (
        pl.when(valid_mask)
        .then(
            pl.col(points_col)
            .rank(method="ordinal", descending=True)
            .over([position_col, year_col])
        )
        .otherwise(None)
    )

    # Percentile excludes zeros
    season_pct_expr = (
        pl.when(percentile_mask)
        .then(
            (100.0 * pl.col(points_col).rank(method="average", descending=False)
             .over([position_col, year_col]) /
             ((pl.col(points_col) > 0) & (pl.col(year_col) == pl.col(year_col)))
             .sum().over([position_col, year_col]))
            .round(2)
        )
        .otherwise(None)
    )

    df = df.with_columns([
        season_rank_expr.alias("position_season_history"),
        season_pct_expr.alias("position_season_history_percentile")
    ])

    return df


def add_manager_all_player_history_ranks(
    df: pl.DataFrame,
    manager_col: str = "manager",
    points_col: str = "fantasy_points",
    year_col: str = "year",
    week_col: str = "week"
) -> pl.DataFrame:
    """
    Add manager all-player history rankings (cross-position).

    This ranks ALL player performances for a manager, regardless of position.
    Enables comparisons like: "Josh Allen's 52-point game vs Alvin Kamara's 50-point game for Daniel"

    For example: "This was the 3rd best performance by ANY player Daniel has ever rostered"

    Adds:
    - manager_all_player_all_time_history: Rank among ALL players this manager has ever rostered (all positions)
    - manager_all_player_all_time_history_percentile: Percentile ranking
    - manager_all_player_season_history: Rank among ALL players this manager rostered this season (all positions)
    - manager_all_player_season_history_percentile: Percentile ranking

    Args:
        df: DataFrame with player stats
        manager_col: Manager column
        points_col: Points column to rank by
        year_col: Year column
        week_col: Week column

    Returns:
        DataFrame with manager all-player history rank columns added
    """
    # Only rank rostered players (manager not null)
    rostered_mask = pl.col(manager_col).is_not_null()
    valid_mask = rostered_mask & pl.col(points_col).is_not_null()

    # All-time manager all-player history ranks
    # Rank ALL performances for this manager across all positions and all time
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[manager_col],
        rank_name="manager_all_player_all_time_history",
        pct_name="manager_all_player_all_time_history_percentile",
        valid_mask=valid_mask
    )

    # Season manager all-player history ranks
    # Rank ALL performances for this manager across all positions this season
    df = add_ranks_with_tiebreaker(
        df,
        points_col=points_col,
        partition_cols=[manager_col, year_col],
        rank_name="manager_all_player_season_history",
        pct_name="manager_all_player_season_history_percentile",
        valid_mask=valid_mask
    )

    return df
