"""
Draft Flags Calculator Module

Calculates dynamic draft flags (breakout, bust) and tiers (early/mid/late)
based on actual draft structure rather than hardcoded values.

All thresholds are calculated as percentiles of the actual data:
- Breakout: late round pick (bottom X% of rounds) with top finish
- Bust: early round pick (top X% of rounds) with bottom finish
- Draft tier: divides rounds into configurable segments

Key Features:
- Dynamically adapts to any number of draft rounds
- Works per year/league to handle varying draft sizes
- Configurable percentile thresholds
- No hardcoded round numbers
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union


# Default configuration
DEFAULT_LATE_ROUND_PERCENTILE = 0.60  # Bottom 40% of rounds = "late"
DEFAULT_EARLY_ROUND_PERCENTILE = 0.25  # Top 25% of rounds = "early"
DEFAULT_TOP_FINISH_PERCENTILE = 0.25  # Top 25% at position = "top finish"
DEFAULT_BOTTOM_FINISH_PERCENTILE = 0.50  # Bottom 50% at position = "bottom finish"
DEFAULT_TIER_PERCENTILES = [0.33, 0.67]  # Split into thirds
DEFAULT_INJURY_GAMES_MISSED_PCT = 0.25  # Missing 25%+ of season = injury


def get_round_thresholds(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None,
    round_column: str = 'round'
) -> pd.DataFrame:
    """
    Calculate round thresholds (min, max, count) for each group.

    Args:
        df: Draft DataFrame
        group_columns: Columns to group by (default: ['year'])
        round_column: Column containing round number

    Returns:
        DataFrame with round statistics per group
    """
    if group_columns is None:
        group_columns = ['year']

    if round_column not in df.columns:
        raise ValueError(f"Round column '{round_column}' not found")

    # Calculate round stats per group
    round_stats = df.groupby(group_columns, dropna=False).agg(
        min_round=(round_column, 'min'),
        max_round=(round_column, 'max'),
        total_rounds=(round_column, 'nunique'),
        total_picks=(round_column, 'count')
    ).reset_index()

    return round_stats


def calculate_round_percentile(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None,
    round_column: str = 'round'
) -> pd.DataFrame:
    """
    Calculate the percentile rank of each pick's round within its group.

    Lower percentile = earlier round, Higher percentile = later round

    Args:
        df: Draft DataFrame
        group_columns: Columns to group by (default: ['year'])
        round_column: Column containing round number

    Returns:
        DataFrame with round_percentile column added
    """
    df = df.copy()

    if group_columns is None:
        group_columns = ['year']

    if round_column not in df.columns:
        raise ValueError(f"Round column '{round_column}' not found")

    # Calculate round percentile within each group
    # Round 1 should have low percentile, last round should have high percentile
    df['round_percentile'] = df.groupby(group_columns, dropna=False)[round_column].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )

    return df


def calculate_breakout_flag(
    df: pd.DataFrame,
    late_round_percentile: float = DEFAULT_LATE_ROUND_PERCENTILE,
    top_finish_percentile: float = DEFAULT_TOP_FINISH_PERCENTILE,
    group_columns: Optional[List[str]] = None,
    round_column: str = 'round',
    finish_rank_column: str = 'season_position_rank',
    total_players_column: str = 'total_position_players'
) -> pd.DataFrame:
    """
    Calculate breakout flag: late round pick with top finish.

    A breakout is defined as:
    - Drafted in late rounds (round percentile >= late_round_percentile)
    - Finished in top X% at position

    Args:
        df: Draft DataFrame with ranking columns
        late_round_percentile: Percentile threshold for "late" rounds (default: 0.60 = bottom 40%)
        top_finish_percentile: Percentile threshold for "top" finish (default: 0.25 = top 25%)
        group_columns: Columns to group by for round percentile (default: ['year'])
        round_column: Column containing round number
        finish_rank_column: Column containing season finish rank at position
        total_players_column: Column containing total players at position

    Returns:
        DataFrame with is_breakout column added
    """
    df = df.copy()

    # Ensure round percentile exists
    if 'round_percentile' not in df.columns:
        df = calculate_round_percentile(df, group_columns, round_column)

    # Validate required columns
    if finish_rank_column not in df.columns:
        print(f"  [WARN] {finish_rank_column} not found, setting is_breakout to 0")
        df['is_breakout'] = 0
        return df

    if total_players_column not in df.columns:
        print(f"  [WARN] {total_players_column} not found, setting is_breakout to 0")
        df['is_breakout'] = 0
        return df

    # Calculate finish percentile (lower rank = better finish = lower percentile)
    finish_rank = pd.to_numeric(df[finish_rank_column], errors='coerce')
    total_players = pd.to_numeric(df[total_players_column], errors='coerce')
    finish_percentile = finish_rank / total_players

    # Breakout: late round (high round_percentile) + top finish (low finish_percentile)
    is_late_round = df['round_percentile'] >= late_round_percentile
    is_top_finish = finish_percentile <= top_finish_percentile

    # Handle NA values - fill with False before combining
    is_late_round = is_late_round.fillna(False)
    is_top_finish = is_top_finish.fillna(False)

    df['is_breakout'] = (is_late_round & is_top_finish).astype(int)

    return df


def calculate_bust_flag(
    df: pd.DataFrame,
    early_round_percentile: float = DEFAULT_EARLY_ROUND_PERCENTILE,
    bottom_finish_percentile: float = DEFAULT_BOTTOM_FINISH_PERCENTILE,
    group_columns: Optional[List[str]] = None,
    round_column: str = 'round',
    finish_rank_column: str = 'season_position_rank',
    total_players_column: str = 'total_position_players'
) -> pd.DataFrame:
    """
    Calculate bust flag: early round pick with bottom finish.

    A bust is defined as:
    - Drafted in early rounds (round percentile <= early_round_percentile)
    - Finished in bottom X% at position

    Args:
        df: Draft DataFrame with ranking columns
        early_round_percentile: Percentile threshold for "early" rounds (default: 0.25 = top 25%)
        bottom_finish_percentile: Percentile threshold for "bottom" finish (default: 0.50 = bottom 50%)
        group_columns: Columns to group by for round percentile (default: ['year'])
        round_column: Column containing round number
        finish_rank_column: Column containing season finish rank at position
        total_players_column: Column containing total players at position

    Returns:
        DataFrame with is_bust column added
    """
    df = df.copy()

    # Ensure round percentile exists
    if 'round_percentile' not in df.columns:
        df = calculate_round_percentile(df, group_columns, round_column)

    # Validate required columns
    if finish_rank_column not in df.columns:
        print(f"  [WARN] {finish_rank_column} not found, setting is_bust to 0")
        df['is_bust'] = 0
        return df

    if total_players_column not in df.columns:
        print(f"  [WARN] {total_players_column} not found, setting is_bust to 0")
        df['is_bust'] = 0
        return df

    # Calculate finish percentile (lower rank = better finish = lower percentile)
    finish_rank = pd.to_numeric(df[finish_rank_column], errors='coerce')
    total_players = pd.to_numeric(df[total_players_column], errors='coerce')
    finish_percentile = finish_rank / total_players

    # Bust: early round (low round_percentile) + bottom finish (high finish_percentile)
    is_early_round = df['round_percentile'] <= early_round_percentile
    is_bottom_finish = finish_percentile > bottom_finish_percentile

    # Handle NA values - fill with False before combining
    is_early_round = is_early_round.fillna(False)
    is_bottom_finish = is_bottom_finish.fillna(False)

    df['is_bust'] = (is_early_round & is_bottom_finish).astype(int)

    return df


def calculate_draft_tier(
    df: pd.DataFrame,
    tier_percentiles: Optional[List[float]] = None,
    tier_labels: Optional[List[str]] = None,
    group_columns: Optional[List[str]] = None,
    round_column: str = 'round'
) -> pd.DataFrame:
    """
    Calculate draft tier (Early/Mid/Late) based on round percentile.

    Args:
        df: Draft DataFrame
        tier_percentiles: List of percentile boundaries (default: [0.33, 0.67] for thirds)
        tier_labels: List of tier labels (default: ['Early', 'Mid', 'Late'])
        group_columns: Columns to group by for round percentile (default: ['year'])
        round_column: Column containing round number

    Returns:
        DataFrame with draft_tier column added
    """
    df = df.copy()

    # Use defaults if not provided
    if tier_percentiles is None:
        tier_percentiles = DEFAULT_TIER_PERCENTILES
    if tier_labels is None:
        tier_labels = ['Early', 'Mid', 'Late']

    # Validate configuration
    if len(tier_labels) != len(tier_percentiles) + 1:
        raise ValueError(
            f"Number of labels ({len(tier_labels)}) must be one more than "
            f"number of percentile boundaries ({len(tier_percentiles)})"
        )

    # Ensure round percentile exists
    if 'round_percentile' not in df.columns:
        df = calculate_round_percentile(df, group_columns, round_column)

    # Create bins from percentiles (add 0 and 1 as boundaries)
    bins = [0] + tier_percentiles + [1.0001]  # Small buffer to include 1.0

    # Assign tiers using pd.cut
    df['draft_tier'] = pd.cut(
        df['round_percentile'],
        bins=bins,
        labels=tier_labels,
        include_lowest=True
    )

    return df


def calculate_games_missed(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None,
    games_played_column: str = 'games_played',
    max_weeks_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate games missed for each player based on season max weeks.

    Args:
        df: Draft DataFrame
        group_columns: Columns to group by to determine max weeks (default: ['year'])
        games_played_column: Column containing games played count
        max_weeks_column: Column containing max weeks (if None, calculated from data)

    Returns:
        DataFrame with games_missed and season_max_weeks columns added
    """
    df = df.copy()

    if group_columns is None:
        group_columns = ['year']

    # Validate games_played column exists
    if games_played_column not in df.columns:
        print(f"  [WARN] {games_played_column} not found, cannot calculate games missed")
        df['games_missed'] = pd.NA
        df['season_max_weeks'] = pd.NA
        df['games_missed_pct'] = pd.NA
        return df

    # Calculate or use max weeks per season
    if max_weeks_column and max_weeks_column in df.columns:
        df['season_max_weeks'] = df[max_weeks_column]
    else:
        # Calculate max weeks from the data - max games_played per year
        # This represents the full season length
        df['season_max_weeks'] = df.groupby(group_columns, dropna=False)[games_played_column].transform('max')

    # Ensure numeric
    games_played = pd.to_numeric(df[games_played_column], errors='coerce').fillna(0)
    season_max = pd.to_numeric(df['season_max_weeks'], errors='coerce')

    # Calculate games missed
    df['games_missed'] = (season_max - games_played).clip(lower=0)

    # Calculate percentage of season missed
    df['games_missed_pct'] = np.where(
        season_max > 0,
        df['games_missed'] / season_max,
        0
    )

    return df


def calculate_injury_flag(
    df: pd.DataFrame,
    injury_threshold_pct: float = DEFAULT_INJURY_GAMES_MISSED_PCT,
    injury_threshold_games: Optional[int] = None,
    games_missed_column: str = 'games_missed',
    games_missed_pct_column: str = 'games_missed_pct'
) -> pd.DataFrame:
    """
    Calculate injury flag based on games missed.

    A player is flagged as injured if they missed significant games:
    - Either X% of the season (default: 25%)
    - Or a fixed number of games if specified

    Args:
        df: Draft DataFrame with games_missed calculated
        injury_threshold_pct: Percentage of season missed to flag as injury (default: 0.25)
        injury_threshold_games: Fixed games missed threshold (overrides pct if set)
        games_missed_column: Column containing games missed count
        games_missed_pct_column: Column containing games missed percentage

    Returns:
        DataFrame with is_injured column added
    """
    df = df.copy()

    # Check if games missed was calculated
    if games_missed_column not in df.columns:
        print(f"  [WARN] {games_missed_column} not found, setting is_injured to 0")
        df['is_injured'] = 0
        return df

    if injury_threshold_games is not None:
        # Use fixed games threshold
        games_missed = pd.to_numeric(df[games_missed_column], errors='coerce').fillna(0)
        df['is_injured'] = (games_missed >= injury_threshold_games).astype(int)
    else:
        # Use percentage threshold
        if games_missed_pct_column not in df.columns:
            print(f"  [WARN] {games_missed_pct_column} not found, setting is_injured to 0")
            df['is_injured'] = 0
            return df

        games_missed_pct = pd.to_numeric(df[games_missed_pct_column], errors='coerce').fillna(0)
        df['is_injured'] = (games_missed_pct >= injury_threshold_pct).astype(int)

    return df


def calculate_bust_type(
    df: pd.DataFrame,
    bust_column: str = 'is_bust',
    injured_column: str = 'is_injured'
) -> pd.DataFrame:
    """
    Split busts into injury busts vs performance busts.

    - is_injury_bust: Early pick that underperformed due to missing games
    - is_performance_bust: Early pick that played but still underperformed

    Args:
        df: DataFrame with is_bust and is_injured columns
        bust_column: Column containing bust flag
        injured_column: Column containing injury flag

    Returns:
        DataFrame with is_injury_bust and is_performance_bust columns added
    """
    df = df.copy()

    # Validate required columns
    if bust_column not in df.columns:
        df['is_injury_bust'] = 0
        df['is_performance_bust'] = 0
        return df

    is_bust = df[bust_column].fillna(0).astype(bool)

    if injured_column not in df.columns:
        # No injury data - all busts are "unknown" type, default to performance
        df['is_injury_bust'] = 0
        df['is_performance_bust'] = is_bust.astype(int)
        return df

    is_injured = df[injured_column].fillna(0).astype(bool)

    # Split busts by injury status
    df['is_injury_bust'] = (is_bust & is_injured).astype(int)
    df['is_performance_bust'] = (is_bust & ~is_injured).astype(int)

    return df


def calculate_draft_tier_with_rounds(
    df: pd.DataFrame,
    num_tiers: int = 3,
    tier_labels: Optional[List[str]] = None,
    group_columns: Optional[List[str]] = None,
    round_column: str = 'round'
) -> pd.DataFrame:
    """
    Calculate draft tier with dynamic round ranges in labels.

    This creates labels like "Early (1-5)", "Mid (6-10)", "Late (11-15)"
    based on actual round numbers in the data.

    Args:
        df: Draft DataFrame
        num_tiers: Number of tiers to create (default: 3)
        tier_labels: Base labels without round numbers (default: ['Early', 'Mid', 'Late'])
        group_columns: Columns to group by (default: ['year'])
        round_column: Column containing round number

    Returns:
        DataFrame with draft_tier column added (with dynamic round ranges)
    """
    df = df.copy()

    if tier_labels is None:
        if num_tiers == 3:
            tier_labels = ['Early', 'Mid', 'Late']
        else:
            tier_labels = [f'Tier {i+1}' for i in range(num_tiers)]

    if group_columns is None:
        group_columns = ['year']

    if round_column not in df.columns:
        raise ValueError(f"Round column '{round_column}' not found")

    # Calculate tier boundaries based on actual rounds
    def assign_tier_with_rounds(group_df):
        rounds = group_df[round_column].dropna()
        if rounds.empty:
            group_df['draft_tier'] = pd.NA
            return group_df

        min_round = int(rounds.min())
        max_round = int(rounds.max())
        total_rounds = max_round - min_round + 1

        # Calculate round boundaries for each tier
        rounds_per_tier = total_rounds / num_tiers
        boundaries = [min_round + int(rounds_per_tier * i) for i in range(num_tiers + 1)]
        boundaries[-1] = max_round + 1  # Ensure last boundary includes max round

        # Create labels with round ranges
        labels_with_rounds = []
        for i, label in enumerate(tier_labels):
            start = boundaries[i]
            end = boundaries[i + 1] - 1
            if start == end:
                labels_with_rounds.append(f'{label} ({start})')
            else:
                labels_with_rounds.append(f'{label} ({start}-{end})')

        # Assign tiers
        group_df['draft_tier'] = pd.cut(
            group_df[round_column],
            bins=[b - 0.5 for b in boundaries[:-1]] + [boundaries[-1] + 0.5],
            labels=labels_with_rounds,
            include_lowest=True
        )

        return group_df

    # Apply per group
    df = df.groupby(group_columns, dropna=False, group_keys=False).apply(assign_tier_with_rounds)

    return df


def calculate_all_draft_flags(
    df: pd.DataFrame,
    late_round_percentile: float = DEFAULT_LATE_ROUND_PERCENTILE,
    early_round_percentile: float = DEFAULT_EARLY_ROUND_PERCENTILE,
    top_finish_percentile: float = DEFAULT_TOP_FINISH_PERCENTILE,
    bottom_finish_percentile: float = DEFAULT_BOTTOM_FINISH_PERCENTILE,
    tier_percentiles: Optional[List[float]] = None,
    tier_labels: Optional[List[str]] = None,
    include_round_ranges: bool = True,
    group_columns: Optional[List[str]] = None,
    round_column: str = 'round',
    finish_rank_column: str = 'season_position_rank',
    total_players_column: str = 'total_position_players',
    games_played_column: str = 'games_played',
    injury_threshold_pct: float = DEFAULT_INJURY_GAMES_MISSED_PCT,
    injury_threshold_games: Optional[int] = None,
    pick_column: str = 'pick'
) -> pd.DataFrame:
    """
    Calculate all draft flags and tiers in one step.

    This is the main entry point for draft flag calculation.
    Only calculates flags for DRAFTED players (pick is not null).
    Undrafted players get NA/0 for all flag columns.

    Args:
        df: Draft DataFrame with ranking columns
        late_round_percentile: Percentile threshold for "late" rounds
        early_round_percentile: Percentile threshold for "early" rounds
        top_finish_percentile: Percentile threshold for "top" finish
        bottom_finish_percentile: Percentile threshold for "bottom" finish
        tier_percentiles: List of percentile boundaries for tiers
        tier_labels: List of tier labels
        include_round_ranges: If True, include round numbers in tier labels
        group_columns: Columns to group by (default: ['year'])
        round_column: Column containing round number
        finish_rank_column: Column containing season finish rank
        total_players_column: Column containing total players at position
        games_played_column: Column containing games played count
        injury_threshold_pct: Percentage of season missed to flag as injury (default: 0.25)
        injury_threshold_games: Fixed games missed threshold (overrides pct if set)
        pick_column: Column indicating drafted status (non-null = drafted)

    Returns:
        DataFrame with all draft flag columns added:
        - round_percentile: Percentile of round within draft
        - is_breakout: Late round pick with top finish
        - is_bust: Early round pick with bottom finish
        - draft_tier: Early/Mid/Late tier label
        - games_missed: Number of games missed
        - games_missed_pct: Percentage of season missed
        - is_injured: Flag for significant games missed
        - is_injury_bust: Bust due to injury
        - is_performance_bust: Bust due to poor performance (not injury)

    Example:
        # Basic usage with defaults
        df = calculate_all_draft_flags(df)

        # Custom thresholds
        df = calculate_all_draft_flags(
            df,
            late_round_percentile=0.70,  # Bottom 30% = late
            early_round_percentile=0.20,  # Top 20% = early
            top_finish_percentile=0.20,   # Top 20% = breakout candidate
            bottom_finish_percentile=0.60  # Bottom 40% = bust candidate
        )

        # Custom injury threshold (4+ games missed = injury)
        df = calculate_all_draft_flags(
            df,
            injury_threshold_games=4
        )

        # Custom tiers (4 tiers instead of 3)
        df = calculate_all_draft_flags(
            df,
            tier_percentiles=[0.25, 0.50, 0.75],
            tier_labels=['Elite', 'Early', 'Mid', 'Late']
        )
    """
    if group_columns is None:
        group_columns = ['year']

    # Initialize all flag columns with defaults for undrafted players
    df['round_percentile'] = pd.NA
    df['is_breakout'] = 0
    df['is_bust'] = 0
    df['draft_tier'] = pd.NA
    df['games_missed'] = pd.NA
    df['games_missed_pct'] = pd.NA
    df['is_injured'] = 0
    df['is_injury_bust'] = 0
    df['is_performance_bust'] = 0

    # Only process drafted players
    drafted_mask = df[pick_column].notna() if pick_column in df.columns else pd.Series(True, index=df.index)

    if not drafted_mask.any():
        return df

    # Validate round column exists
    if round_column not in df.columns:
        print(f"  [WARN] Round column '{round_column}' not found, skipping draft flags")
        return df

    # Work with only drafted players
    df_drafted = df[drafted_mask].copy()

    # Step 1: Calculate round percentile
    df_drafted = calculate_round_percentile(df_drafted, group_columns, round_column)

    # Step 2: Calculate breakout flag
    df_drafted = calculate_breakout_flag(
        df_drafted,
        late_round_percentile=late_round_percentile,
        top_finish_percentile=top_finish_percentile,
        group_columns=group_columns,
        round_column=round_column,
        finish_rank_column=finish_rank_column,
        total_players_column=total_players_column
    )

    # Step 3: Calculate bust flag
    df_drafted = calculate_bust_flag(
        df_drafted,
        early_round_percentile=early_round_percentile,
        bottom_finish_percentile=bottom_finish_percentile,
        group_columns=group_columns,
        round_column=round_column,
        finish_rank_column=finish_rank_column,
        total_players_column=total_players_column
    )

    # Step 4: Calculate draft tier
    if include_round_ranges:
        df_drafted = calculate_draft_tier_with_rounds(
            df_drafted,
            num_tiers=len(tier_labels) if tier_labels else 3,
            tier_labels=tier_labels,
            group_columns=group_columns,
            round_column=round_column
        )
    else:
        df_drafted = calculate_draft_tier(
            df_drafted,
            tier_percentiles=tier_percentiles,
            tier_labels=tier_labels,
            group_columns=group_columns,
            round_column=round_column
        )

    # Step 5: Calculate games missed
    df_drafted = calculate_games_missed(
        df_drafted,
        group_columns=group_columns,
        games_played_column=games_played_column
    )

    # Step 6: Calculate injury flag
    df_drafted = calculate_injury_flag(
        df_drafted,
        injury_threshold_pct=injury_threshold_pct,
        injury_threshold_games=injury_threshold_games
    )

    # Step 7: Split busts into injury vs performance busts
    df_drafted = calculate_bust_type(df_drafted)

    # Merge results back to original dataframe
    # Copy flag columns from df_drafted back to df for drafted players only
    flag_columns = [
        'round_percentile', 'is_breakout', 'is_bust', 'draft_tier',
        'games_missed', 'games_missed_pct', 'is_injured',
        'is_injury_bust', 'is_performance_bust'
    ]

    for col in flag_columns:
        if col in df_drafted.columns:
            df.loc[drafted_mask, col] = df_drafted[col].values

    return df


def get_flag_summary(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get summary of breakout and bust flags by group.

    Args:
        df: DataFrame with is_breakout and is_bust columns
        group_columns: Optional columns to group by

    Returns:
        DataFrame with flag counts and percentages, including bust type breakdown
    """
    required_cols = ['is_breakout', 'is_bust']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build aggregation dict
    agg_dict = {
        'total_picks': ('is_breakout', 'count'),
        'breakouts': ('is_breakout', 'sum'),
        'busts': ('is_bust', 'sum')
    }

    # Add injury/performance bust breakdown if available
    if 'is_injury_bust' in df.columns:
        agg_dict['injury_busts'] = ('is_injury_bust', 'sum')
    if 'is_performance_bust' in df.columns:
        agg_dict['performance_busts'] = ('is_performance_bust', 'sum')
    if 'is_injured' in df.columns:
        agg_dict['injured'] = ('is_injured', 'sum')

    if group_columns:
        summary = df.groupby(group_columns, dropna=False).agg(**agg_dict).reset_index()
    else:
        result = {'total_picks': len(df)}
        result['breakouts'] = df['is_breakout'].sum()
        result['busts'] = df['is_bust'].sum()
        if 'is_injury_bust' in df.columns:
            result['injury_busts'] = df['is_injury_bust'].sum()
        if 'is_performance_bust' in df.columns:
            result['performance_busts'] = df['is_performance_bust'].sum()
        if 'is_injured' in df.columns:
            result['injured'] = df['is_injured'].sum()
        summary = pd.DataFrame([result])

    # Calculate percentages
    summary['breakout_pct'] = (summary['breakouts'] / summary['total_picks'] * 100).round(1)
    summary['bust_pct'] = (summary['busts'] / summary['total_picks'] * 100).round(1)

    if 'injury_busts' in summary.columns:
        summary['injury_bust_pct'] = (summary['injury_busts'] / summary['total_picks'] * 100).round(1)
    if 'performance_busts' in summary.columns:
        summary['performance_bust_pct'] = (summary['performance_busts'] / summary['total_picks'] * 100).round(1)
    if 'injured' in summary.columns:
        summary['injured_pct'] = (summary['injured'] / summary['total_picks'] * 100).round(1)

    return summary


def get_tier_distribution(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get distribution of draft tiers.

    Args:
        df: DataFrame with draft_tier column
        group_columns: Optional columns to group by

    Returns:
        DataFrame with tier counts and percentages
    """
    if 'draft_tier' not in df.columns:
        raise ValueError("draft_tier column not found. Run calculate_all_draft_flags first.")

    if group_columns:
        tier_counts = df.groupby(group_columns + ['draft_tier'], dropna=False).size().reset_index(name='count')
        totals = df.groupby(group_columns, dropna=False).size().reset_index(name='total')
        tier_counts = tier_counts.merge(totals, on=group_columns)
        tier_counts['percentage'] = (tier_counts['count'] / tier_counts['total'] * 100).round(1)
    else:
        tier_counts = df['draft_tier'].value_counts().reset_index()
        tier_counts.columns = ['draft_tier', 'count']
        tier_counts['percentage'] = (tier_counts['count'] / len(df) * 100).round(1)

    return tier_counts
