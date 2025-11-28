"""
Manager Draft Grade Calculator Module

Calculates draft grades (A-F) and scores for each manager-year based on
aggregated performance of all their draft picks.

Key Concepts:
- pick_quality_score: How well each pick performed vs peers at similar draft capital
- manager_draft_score: Weighted average of pick quality scores
- manager_draft_grade: Letter grade (A-F) based on percentile among managers that year

Scoring Philosophy:
- Hitting on expensive/early picks matters MORE than late round picks
- Each pick is evaluated against peers at similar draft capital (cost bucket or round)
- Final score is weighted by draft capital invested

Draft Type Handling:
- Automatically detects auction vs snake PER YEAR from data
- For auction: uses cost_bucket for peer comparison, cost_weighted for weighting
- For snake: uses round for peer comparison, exponential decay for weighting
- Handles mixed datasets (some years auction, some snake)

Weighting Methods:
- 'cost_weighted': Weight by normalized cost (auction)
- 'round_weighted': Weight by round importance (snake)
- 'linear': Earlier picks weighted more (linear decay)
- 'exponential': Earlier picks weighted much more (exponential decay)
- 'auto': Automatically choose based on draft type per year
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Literal

from .draft_type_utils import (
    detect_draft_type_for_year,
    detect_draft_type_per_year,
    get_peer_group_column,
    get_weight_method_for_year
)


# Default grade configuration - uses test score style grading with +/-
# 97-100: A+, 93-97: A, 90-93: A-, 87-90: B+, 83-87: B, 80-83: B-
# 77-80: C+, 73-77: C, 70-73: C-, 67-70: D+, 63-67: D, 60-63: D-, 0-60: F
DEFAULT_MANAGER_PERCENTILE_BINS = [0, 60, 63, 67, 70, 73, 77, 80, 83, 87, 90, 93, 97, 100]
DEFAULT_MANAGER_GRADE_LABELS = ['F', 'D-', 'D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']


def get_base_grade(grade: str) -> str:
    """Get the base letter (A, B, C, D, F) from a grade like A+ or B-."""
    if not grade or grade == 'N/A' or pd.isna(grade):
        return ''
    return str(grade)[0]

# Weight decay for round-based weighting
DEFAULT_ROUND_WEIGHT_DECAY = 0.85  # Each subsequent round worth 85% of previous

# Asymmetric scoring multipliers
DEFAULT_EARLY_MISS_PENALTY = 1.5    # Early round misses hurt 50% more
DEFAULT_EARLY_HIT_BONUS = 1.0       # Early round hits - normal credit
DEFAULT_LATE_MISS_PENALTY = 0.5     # Late round misses - half penalty
DEFAULT_LATE_HIT_BONUS = 1.5        # Late round hits - 50% bonus (found a gem!)


def calculate_pick_quality_score(
    df: pd.DataFrame,
    spar_column: str = 'manager_spar',
    group_column: Optional[str] = None,
    year_column: str = 'year',
    position_column: str = 'position',
    per_year: bool = True,
    pick_column: str = 'pick'
) -> pd.DataFrame:
    """
    Calculate how well each pick performed relative to peers at similar draft capital.

    Only calculates scores for DRAFTED players (pick is not null).
    Undrafted players get pick_quality_score = NA.

    Each pick is compared to other picks in the same:
    - Year (same draft class)
    - Cost bucket or round tier (similar draft capital invested)
    - Optionally: position (apples to apples comparison)

    The result is a percentile score (0-100) where:
    - 100 = best performer in that cost tier
    - 50 = median performer
    - 0 = worst performer

    Automatically detects the appropriate peer group column per year:
    - Auction years: cost_bucket
    - Snake years: round

    Args:
        df: Draft DataFrame with player-level stats
        spar_column: Column containing SPAR values
        group_column: Column to group by for peer comparison (auto-detected if None)
        year_column: Column containing year
        position_column: Column containing position (optional grouping)
        per_year: If True, auto-detect peer group column per year
        pick_column: Column indicating drafted status (non-null = drafted)

    Returns:
        DataFrame with pick_quality_score column added
    """
    df = df.copy()

    # Initialize quality scores as NA (undrafted players stay NA)
    df['pick_quality_score'] = pd.NA

    # Only process drafted players
    drafted_mask = df[pick_column].notna() if pick_column in df.columns else pd.Series(True, index=df.index)

    if not drafted_mask.any():
        return df

    # Validate columns
    if spar_column not in df.columns:
        print(f"  [WARN] {spar_column} not found, setting pick_quality_score to 50 for drafted players")
        df.loc[drafted_mask, 'pick_quality_score'] = 50.0
        return df

    # Ensure SPAR is numeric
    df[spar_column] = pd.to_numeric(df[spar_column], errors='coerce').fillna(0)

    # Work with only drafted players for peer comparison
    df_drafted = df[drafted_mask].copy()

    # Handle per-year peer group detection
    if per_year and year_column in df_drafted.columns and group_column is None:
        # Process each year separately with appropriate peer group
        for year in df_drafted[year_column].dropna().unique():
            year_mask = df_drafted[year_column] == year
            year_group_col = get_peer_group_column(df_drafted, year=year, year_column=year_column)

            # Determine grouping columns for this year
            group_cols = [year_column]
            if year_group_col in df_drafted.columns:
                group_cols.append(year_group_col)

            # Calculate percentile within peer group for this year (drafted only)
            year_df = df_drafted[year_mask].copy()
            year_df['_temp_quality'] = year_df.groupby(group_cols, dropna=False)[spar_column].transform(
                lambda x: x.rank(pct=True, method='average') * 100 if len(x) > 1 else 50.0
            )
            df_drafted.loc[year_mask, 'pick_quality_score'] = year_df['_temp_quality'].fillna(50.0)
    else:
        # Single peer group column for all years
        group_cols = [year_column]

        # Determine peer group column
        if group_column is not None and group_column in df_drafted.columns:
            group_cols.append(group_column)
        elif group_column is None:
            # Auto-detect based on most common draft type
            detected_col = get_peer_group_column(df_drafted, year=None, year_column=year_column)
            if detected_col in df_drafted.columns:
                group_cols.append(detected_col)
                print(f"  [INFO] Using '{detected_col}' for peer grouping")
        elif 'round' in df_drafted.columns:
            # Fallback to round
            group_cols.append('round')
            print(f"  [INFO] Using 'round' for peer grouping ('{group_column}' not found)")

        # Calculate percentile within peer group (drafted only)
        df_drafted['pick_quality_score'] = df_drafted.groupby(group_cols, dropna=False)[spar_column].transform(
            lambda x: x.rank(pct=True, method='average') * 100 if len(x) > 1 else 50.0
        )

    # Handle NaN (ungrouped drafted picks default to 50)
    df_drafted['pick_quality_score'] = df_drafted['pick_quality_score'].fillna(50.0)

    # Copy results back to original dataframe
    df.loc[drafted_mask, 'pick_quality_score'] = df_drafted['pick_quality_score'].values

    return df


def apply_asymmetric_scoring(
    df: pd.DataFrame,
    quality_column: str = 'pick_quality_score',
    round_column: str = 'round',
    year_column: str = 'year',
    early_miss_penalty: float = DEFAULT_EARLY_MISS_PENALTY,
    early_hit_bonus: float = DEFAULT_EARLY_HIT_BONUS,
    late_miss_penalty: float = DEFAULT_LATE_MISS_PENALTY,
    late_hit_bonus: float = DEFAULT_LATE_HIT_BONUS
) -> pd.DataFrame:
    """
    Apply asymmetric scoring adjustments based on round position.

    Philosophy:
    - Early round misses are devastating (wasted premium capital) -> big penalty
    - Early round hits are expected -> normal credit
    - Late round misses are acceptable (low cost) -> small penalty
    - Late round hits are valuable (found a gem!) -> big bonus

    The adjustment scales linearly from early to late rounds:
    - Round 1: full early multipliers
    - Last round: full late multipliers
    - Middle rounds: interpolated

    Formula:
    - If hit (quality >= 50): adjusted = 50 + (quality - 50) * hit_multiplier
    - If miss (quality < 50): adjusted = 50 - (50 - quality) * miss_multiplier

    Args:
        df: Draft DataFrame with pick_quality_score
        quality_column: Column containing quality scores (0-100)
        round_column: Column containing round number
        year_column: Column containing year
        early_miss_penalty: Multiplier for early round misses (default: 1.5)
        early_hit_bonus: Multiplier for early round hits (default: 1.0)
        late_miss_penalty: Multiplier for late round misses (default: 0.5)
        late_hit_bonus: Multiplier for late round hits (default: 1.5)

    Returns:
        DataFrame with adjusted_quality_score column added

    Example impact:
        Round 1 miss (quality=20): 50 - 30*1.5 = 5 (severely penalized)
        Round 1 hit (quality=80): 50 + 30*1.0 = 80 (normal)
        Round 15 miss (quality=20): 50 - 30*0.5 = 35 (small penalty)
        Round 15 hit (quality=80): 50 + 30*1.5 = 95 (big bonus!)
    """
    df = df.copy()

    if quality_column not in df.columns:
        df['adjusted_quality_score'] = 50.0
        return df

    if round_column not in df.columns:
        # No round info - just use original score
        df['adjusted_quality_score'] = df[quality_column]
        return df

    # Calculate round position (0 = earliest, 1 = latest)
    round_num = pd.to_numeric(df[round_column], errors='coerce').fillna(1)
    min_round = df.groupby(year_column, dropna=False)[round_column].transform('min')
    max_round = df.groupby(year_column, dropna=False)[round_column].transform('max')

    min_round = pd.to_numeric(min_round, errors='coerce').fillna(1)
    max_round = pd.to_numeric(max_round, errors='coerce').fillna(1)

    # Round position: 0 = first round, 1 = last round
    round_range = (max_round - min_round).clip(lower=1)
    round_position = (round_num - min_round) / round_range

    # Interpolate multipliers based on round position
    # Early (position=0): early multipliers, Late (position=1): late multipliers
    hit_multiplier = early_hit_bonus + (late_hit_bonus - early_hit_bonus) * round_position
    miss_multiplier = early_miss_penalty + (late_miss_penalty - early_miss_penalty) * round_position

    # Get quality score
    quality = pd.to_numeric(df[quality_column], errors='coerce').fillna(50)

    # Calculate deviation from median (50)
    deviation = quality - 50

    # Apply asymmetric adjustment
    # Hits (quality >= 50): bonus based on how much above 50
    # Misses (quality < 50): penalty based on how much below 50
    is_hit = quality >= 50

    adjusted = pd.Series(50.0, index=df.index)
    adjusted[is_hit] = 50 + deviation[is_hit] * hit_multiplier[is_hit]
    adjusted[~is_hit] = 50 + deviation[~is_hit] * miss_multiplier[~is_hit]  # deviation is negative for misses

    # Clip to valid range (0-100)
    df['adjusted_quality_score'] = adjusted.clip(0, 100)

    # Also store the multipliers used for transparency
    df['hit_multiplier'] = hit_multiplier
    df['miss_multiplier'] = miss_multiplier

    return df


def calculate_pick_weight(
    df: pd.DataFrame,
    method: str = 'auto',
    round_column: str = 'round',
    cost_column: str = 'cost_norm',
    year_column: str = 'year',
    decay_rate: float = DEFAULT_ROUND_WEIGHT_DECAY,
    pick_column: str = 'pick'
) -> pd.DataFrame:
    """
    Calculate weight for each pick based on draft capital invested.

    Only calculates weights for DRAFTED players (pick is not null).
    Undrafted players get pick_weight = NA.

    Early/expensive picks should count more towards manager grade.

    Methods:
    - 'auto': Automatically choose based on draft type per year
              (auction = cost_weighted, snake = exponential)
    - 'exponential': weight = decay_rate ^ (round - 1)
      Round 1 = 1.0, Round 2 = 0.85, Round 3 = 0.72, etc.
    - 'linear': weight = (max_round - round + 1) / max_round
      Round 1 = 1.0, Round 8 = 0.5 (in 15-round draft), etc.
    - 'cost_weighted': weight = cost_norm / max(cost_norm)
      $50 pick = 1.0, $25 pick = 0.5, etc.
    - 'equal': weight = 1.0 for all picks

    Args:
        df: Draft DataFrame
        method: Weighting method to use
        round_column: Column containing round number
        cost_column: Column containing normalized cost
        year_column: Column containing year
        decay_rate: Decay rate for exponential method (default: 0.85)
        pick_column: Column indicating drafted status (non-null = drafted)

    Returns:
        DataFrame with pick_weight column added
    """
    df = df.copy()
    df['pick_weight'] = pd.NA  # Initialize as NA for undrafted

    # Only process drafted players
    drafted_mask = df[pick_column].notna() if pick_column in df.columns else pd.Series(True, index=df.index)

    if not drafted_mask.any():
        return df

    if method == 'auto':
        # Auto-detect weight method per year based on draft type (only for drafted players)
        for year in df[year_column].dropna().unique():
            year_drafted_mask = (df[year_column] == year) & drafted_mask
            if not year_drafted_mask.any():
                continue

            year_method = get_weight_method_for_year(df, year, year_column)

            if year_method == 'cost_weighted' and cost_column in df.columns:
                cost = pd.to_numeric(df.loc[year_drafted_mask, cost_column], errors='coerce').fillna(0)
                max_cost = cost.max() if cost.max() > 0 else 1
                df.loc[year_drafted_mask, 'pick_weight'] = (cost / max_cost).clip(lower=0.1)
            elif round_column in df.columns:
                round_num = pd.to_numeric(df.loc[year_drafted_mask, round_column], errors='coerce').fillna(1)
                df.loc[year_drafted_mask, 'pick_weight'] = decay_rate ** (round_num - 1)
            else:
                df.loc[year_drafted_mask, 'pick_weight'] = 1.0

        return df

    # For non-auto methods, work with drafted subset
    df_drafted = df[drafted_mask].copy()

    if method == 'exponential':
        if round_column not in df_drafted.columns:
            print(f"  [WARN] {round_column} not found, using equal weights")
            df.loc[drafted_mask, 'pick_weight'] = 1.0
            return df

        round_num = pd.to_numeric(df_drafted[round_column], errors='coerce').fillna(1)
        df.loc[drafted_mask, 'pick_weight'] = decay_rate ** (round_num - 1)

    elif method == 'linear':
        if round_column not in df_drafted.columns:
            df.loc[drafted_mask, 'pick_weight'] = 1.0
            return df

        round_num = pd.to_numeric(df_drafted[round_column], errors='coerce').fillna(1)

        # Calculate max round per year (from drafted only)
        max_round = df_drafted.groupby(year_column, dropna=False)[round_column].transform('max')
        max_round = pd.to_numeric(max_round, errors='coerce').fillna(1).clip(lower=1)

        df.loc[drafted_mask, 'pick_weight'] = ((max_round - round_num + 1) / max_round).values

    elif method == 'cost_weighted':
        if cost_column not in df_drafted.columns:
            print(f"  [WARN] {cost_column} not found, using equal weights")
            df.loc[drafted_mask, 'pick_weight'] = 1.0
            return df

        cost = pd.to_numeric(df_drafted[cost_column], errors='coerce').fillna(0)

        # Normalize to max cost per year (from drafted only)
        max_cost = df_drafted.groupby(year_column, dropna=False)[cost_column].transform('max')
        max_cost = pd.to_numeric(max_cost, errors='coerce').fillna(1).clip(lower=1)

        df.loc[drafted_mask, 'pick_weight'] = ((cost / max_cost).clip(lower=0.1)).values

    else:  # 'equal'
        df.loc[drafted_mask, 'pick_weight'] = 1.0

    return df


def calculate_weighted_manager_score(
    df: pd.DataFrame,
    manager_column: str = 'manager',
    year_column: str = 'year',
    quality_column: str = 'pick_quality_score',
    weight_column: str = 'pick_weight',
    pick_column: str = 'pick'
) -> pd.DataFrame:
    """
    Calculate weighted average of pick quality scores for each manager-year.

    Final score = sum(pick_quality_score * pick_weight) / sum(pick_weight)

    This rewards managers who:
    1. Hit on their expensive/early picks (high quality score on high weight picks)
    2. Consistently draft well across all rounds

    Args:
        df: Draft DataFrame with pick_quality_score and pick_weight
        manager_column: Column containing manager name
        year_column: Column containing year
        quality_column: Column containing pick quality scores
        weight_column: Column containing pick weights
        pick_column: Column indicating drafted (non-null = drafted)

    Returns:
        DataFrame with manager-year aggregated weighted scores
    """
    # Filter to drafted players only
    drafted_mask = df[pick_column].notna() & df[manager_column].notna()
    df_drafted = df[drafted_mask].copy()

    if df_drafted.empty:
        return pd.DataFrame()

    # Ensure numeric
    df_drafted[quality_column] = pd.to_numeric(df_drafted[quality_column], errors='coerce').fillna(50)
    df_drafted[weight_column] = pd.to_numeric(df_drafted[weight_column], errors='coerce').fillna(1)

    # Calculate weighted components
    df_drafted['_weighted_quality'] = df_drafted[quality_column] * df_drafted[weight_column]

    # Aggregate by manager-year
    manager_scores = df_drafted.groupby([year_column, manager_column], dropna=False).agg(
        total_weighted_quality=('_weighted_quality', 'sum'),
        total_weight=('pick_weight', 'sum'),
        total_picks=(pick_column, 'count'),
        avg_quality=(quality_column, 'mean'),
        avg_weight=(weight_column, 'mean')
    ).reset_index()

    # Calculate weighted average score
    manager_scores['manager_draft_score'] = (
        manager_scores['total_weighted_quality'] / manager_scores['total_weight'].clip(lower=0.01)
    )

    return manager_scores


def aggregate_manager_draft_stats(
    df: pd.DataFrame,
    manager_column: str = 'manager',
    year_column: str = 'year',
    spar_column: str = 'manager_spar',
    pick_column: str = 'pick',
    cost_column: str = 'cost',
    include_keepers: bool = True,
    keeper_column: str = 'is_keeper_status'
) -> pd.DataFrame:
    """
    Aggregate player-level draft stats to manager-year level.

    Args:
        df: Draft DataFrame with player-level stats
        manager_column: Column containing manager name/id
        year_column: Column containing year
        spar_column: Column containing SPAR values
        pick_column: Column indicating drafted (non-null = drafted)
        cost_column: Column containing draft cost
        include_keepers: Whether to include keepers in aggregation
        keeper_column: Column indicating keeper status

    Returns:
        DataFrame with one row per manager-year containing aggregated stats
    """
    df = df.copy()

    # Filter to drafted players only (must have a pick and a manager)
    drafted_mask = df[pick_column].notna() & df[manager_column].notna()

    # Optionally exclude keepers
    if not include_keepers and keeper_column in df.columns:
        keeper_mask = df[keeper_column].fillna(0).astype(bool)
        drafted_mask = drafted_mask & ~keeper_mask

    df_drafted = df[drafted_mask].copy()

    if df_drafted.empty:
        print("  [WARN] No drafted players found for aggregation")
        return pd.DataFrame()

    # Ensure numeric columns
    df_drafted[spar_column] = pd.to_numeric(df_drafted[spar_column], errors='coerce').fillna(0)
    df_drafted[cost_column] = pd.to_numeric(df_drafted[cost_column], errors='coerce').fillna(0)
    df_drafted[pick_column] = pd.to_numeric(df_drafted[pick_column], errors='coerce')

    # Get additional columns if available
    has_points = 'total_fantasy_points' in df_drafted.columns
    has_grade = 'draft_grade' in df_drafted.columns
    has_value_tier = 'value_tier' in df_drafted.columns
    has_breakout = 'is_breakout' in df_drafted.columns
    has_bust = 'is_bust' in df_drafted.columns
    has_injury_bust = 'is_injury_bust' in df_drafted.columns
    has_perf_bust = 'is_performance_bust' in df_drafted.columns
    has_roi = 'draft_roi' in df_drafted.columns

    # Build aggregation dictionary
    agg_dict = {
        # Count metrics
        'total_picks': (pick_column, 'count'),

        # SPAR metrics
        'total_spar': (spar_column, 'sum'),
        'avg_spar': (spar_column, 'mean'),
        'median_spar': (spar_column, 'median'),
        'min_spar': (spar_column, 'min'),
        'max_spar': (spar_column, 'max'),
        'std_spar': (spar_column, 'std'),

        # Draft capital
        'total_cost': (cost_column, 'sum'),
        'avg_cost': (cost_column, 'mean'),
        'earliest_pick': (pick_column, 'min'),
        'latest_pick': (pick_column, 'max'),
    }

    # Add points if available
    if has_points:
        df_drafted['total_fantasy_points'] = pd.to_numeric(
            df_drafted['total_fantasy_points'], errors='coerce'
        ).fillna(0)
        agg_dict['total_points'] = ('total_fantasy_points', 'sum')
        agg_dict['avg_points'] = ('total_fantasy_points', 'mean')

    # Add ROI if available
    if has_roi:
        df_drafted['draft_roi'] = pd.to_numeric(df_drafted['draft_roi'], errors='coerce').fillna(0)
        agg_dict['avg_roi'] = ('draft_roi', 'mean')
        agg_dict['total_roi'] = ('draft_roi', 'sum')

    # Add flag counts if available
    if has_breakout:
        df_drafted['is_breakout'] = pd.to_numeric(df_drafted['is_breakout'], errors='coerce').fillna(0)
        agg_dict['breakout_count'] = ('is_breakout', 'sum')
    if has_bust:
        df_drafted['is_bust'] = pd.to_numeric(df_drafted['is_bust'], errors='coerce').fillna(0)
        agg_dict['bust_count'] = ('is_bust', 'sum')
    if has_injury_bust:
        df_drafted['is_injury_bust'] = pd.to_numeric(df_drafted['is_injury_bust'], errors='coerce').fillna(0)
        agg_dict['injury_bust_count'] = ('is_injury_bust', 'sum')
    if has_perf_bust:
        df_drafted['is_performance_bust'] = pd.to_numeric(df_drafted['is_performance_bust'], errors='coerce').fillna(0)
        agg_dict['performance_bust_count'] = ('is_performance_bust', 'sum')

    # Aggregate by manager-year
    manager_stats = df_drafted.groupby([year_column, manager_column], dropna=False).agg(**agg_dict).reset_index()

    # Calculate derived metrics
    # SPAR per dollar (efficiency metric)
    if 'total_cost' in manager_stats.columns:
        cost_safe = manager_stats['total_cost'].clip(lower=1)
        manager_stats['spar_per_dollar'] = manager_stats['total_spar'] / cost_safe

    # SPAR per pick
    picks_safe = manager_stats['total_picks'].clip(lower=1)
    manager_stats['spar_per_pick'] = manager_stats['total_spar'] / picks_safe

    # Positive SPAR picks (how many picks contributed positive value)
    positive_spar_counts = df_drafted[df_drafted[spar_column] > 0].groupby(
        [year_column, manager_column], dropna=False
    ).size().reset_index(name='positive_spar_picks')
    manager_stats = manager_stats.merge(positive_spar_counts, on=[year_column, manager_column], how='left')
    manager_stats['positive_spar_picks'] = manager_stats['positive_spar_picks'].fillna(0).astype(int)

    # Hit rate (% of picks with positive SPAR)
    manager_stats['hit_rate'] = (manager_stats['positive_spar_picks'] / manager_stats['total_picks'] * 100).round(1)

    # Grade distribution counts if available
    if has_grade:
        # Filter to rows with valid draft_grade
        grade_df = df_drafted[df_drafted['draft_grade'].notna()]
        if not grade_df.empty:
            grade_counts = grade_df.groupby([year_column, manager_column, 'draft_grade'], dropna=False).size().unstack(fill_value=0)
            grade_counts.columns = [f'grade_{g}_count' for g in grade_counts.columns]
            grade_counts = grade_counts.reset_index()
            manager_stats = manager_stats.merge(grade_counts, on=[year_column, manager_column], how='left')

    # Value tier distribution if available
    if has_value_tier:
        # Filter to rows with valid value_tier
        tier_df = df_drafted[df_drafted['value_tier'].notna()]
        if not tier_df.empty:
            tier_counts = tier_df.groupby([year_column, manager_column, 'value_tier'], dropna=False).size().unstack(fill_value=0)
            tier_counts.columns = [f'tier_{str(t).replace(" ", "_")}_count' for t in tier_counts.columns]
            tier_counts = tier_counts.reset_index()
            manager_stats = manager_stats.merge(tier_counts, on=[year_column, manager_column], how='left')

    return manager_stats


def calculate_manager_draft_percentile(
    manager_stats: pd.DataFrame,
    score_column: str = 'total_spar',
    year_column: str = 'year',
    alltime: bool = True
) -> pd.DataFrame:
    """
    Calculate percentile rank of each manager.

    Args:
        manager_stats: DataFrame with manager-year aggregated stats
        score_column: Column to use for ranking (default: total_spar)
        year_column: Column containing year
        alltime: If True, calculate percentile across ALL years (for cross-year comparison)
                 If False, calculate percentile within each year only

    Returns:
        DataFrame with percentile columns added:
        - manager_draft_percentile_year: Percentile within that year
        - manager_draft_percentile_alltime: Percentile across all years (if alltime=True)
        - manager_draft_percentile: Same as alltime if alltime=True, else same as year
    """
    manager_stats = manager_stats.copy()

    if score_column not in manager_stats.columns:
        raise ValueError(f"Score column '{score_column}' not found")

    # Calculate percentile within each year
    manager_stats['manager_draft_percentile_year'] = manager_stats.groupby(year_column, dropna=False)[score_column].transform(
        lambda x: x.rank(pct=True, method='average') * 100
    )

    if alltime:
        # Calculate percentile across ALL manager-year combinations
        manager_stats['manager_draft_percentile_alltime'] = (
            manager_stats[score_column].rank(pct=True, method='average') * 100
        )
        # Use alltime as the primary percentile for grading
        manager_stats['manager_draft_percentile'] = manager_stats['manager_draft_percentile_alltime']
    else:
        # Use per-year as the primary percentile
        manager_stats['manager_draft_percentile'] = manager_stats['manager_draft_percentile_year']

    return manager_stats


def assign_manager_draft_grade(
    manager_stats: pd.DataFrame,
    percentile_column: str = 'manager_draft_percentile',
    percentile_bins: Optional[List[float]] = None,
    grade_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Assign letter grades based on manager draft percentile.

    Uses test score style grading: 97-100=A+, 93-97=A, 90-93=A-, etc.
    Players below 60th percentile get an F.

    Args:
        manager_stats: DataFrame with manager_draft_percentile column
        percentile_column: Column containing percentile values
        percentile_bins: List of bin edges (default: test score style with +/-)
        grade_labels: List of grade labels (default: F through A+)

    Returns:
        DataFrame with manager_draft_grade column added
    """
    manager_stats = manager_stats.copy()

    if percentile_bins is None:
        percentile_bins = DEFAULT_MANAGER_PERCENTILE_BINS
    if grade_labels is None:
        grade_labels = DEFAULT_MANAGER_GRADE_LABELS

    if len(grade_labels) != len(percentile_bins) - 1:
        raise ValueError(
            f"Number of grade labels ({len(grade_labels)}) must be one less than "
            f"number of bin edges ({len(percentile_bins)})"
        )

    if percentile_column not in manager_stats.columns:
        raise ValueError(f"Percentile column '{percentile_column}' not found")

    manager_stats['manager_draft_grade'] = pd.cut(
        manager_stats[percentile_column],
        bins=percentile_bins,
        labels=grade_labels,
        include_lowest=True
    )

    return manager_stats


def calculate_manager_draft_score(
    manager_stats: pd.DataFrame,
    method: str = 'total_spar',
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Calculate a normalized draft score for each manager.

    Scoring Methods:
    - 'total_spar': Raw total SPAR (rewards both volume and quality)
    - 'avg_spar': Average SPAR per pick (rewards efficiency only)
    - 'weighted': Custom weighted combination of metrics
    - 'composite': Balanced score combining total, avg, and hit rate

    Args:
        manager_stats: DataFrame with aggregated manager stats
        method: Scoring method to use
        weights: Custom weights for 'weighted' method

    Returns:
        DataFrame with manager_draft_score column added
    """
    manager_stats = manager_stats.copy()

    if method == 'total_spar':
        manager_stats['manager_draft_score'] = manager_stats['total_spar']

    elif method == 'avg_spar':
        manager_stats['manager_draft_score'] = manager_stats['avg_spar']

    elif method == 'composite':
        # Normalize each component to 0-100 scale within year
        def normalize_within_year(series, year_col):
            return series.groupby(manager_stats[year_col]).transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50
            )

        total_norm = normalize_within_year(manager_stats['total_spar'], 'year')
        avg_norm = normalize_within_year(manager_stats['avg_spar'], 'year')
        hit_norm = manager_stats['hit_rate']  # Already 0-100

        # Weighted average: 50% total, 30% avg, 20% hit rate
        manager_stats['manager_draft_score'] = (
            total_norm * 0.50 +
            avg_norm * 0.30 +
            hit_norm * 0.20
        )

    elif method == 'weighted' and weights:
        score = pd.Series(0, index=manager_stats.index, dtype=float)
        for col, weight in weights.items():
            if col in manager_stats.columns:
                score += manager_stats[col].fillna(0) * weight
        manager_stats['manager_draft_score'] = score

    else:
        # Default to total_spar
        manager_stats['manager_draft_score'] = manager_stats['total_spar']

    return manager_stats


def calculate_manager_draft_grades(
    df: pd.DataFrame,
    manager_column: str = 'manager',
    year_column: str = 'year',
    spar_column: str = 'manager_spar',
    pick_column: str = 'pick',
    peer_group_column: Optional[str] = None,
    weight_method: str = 'auto',
    weight_decay: float = DEFAULT_ROUND_WEIGHT_DECAY,
    use_asymmetric_scoring: bool = True,
    early_miss_penalty: float = DEFAULT_EARLY_MISS_PENALTY,
    early_hit_bonus: float = DEFAULT_EARLY_HIT_BONUS,
    late_miss_penalty: float = DEFAULT_LATE_MISS_PENALTY,
    late_hit_bonus: float = DEFAULT_LATE_HIT_BONUS,
    percentile_bins: Optional[List[float]] = None,
    grade_labels: Optional[List[str]] = None,
    include_keepers: bool = True
) -> pd.DataFrame:
    """
    Calculate manager draft grades using weighted pick quality scores with asymmetric adjustments.

    This is the main entry point for manager draft grade calculation.

    Automatically handles mixed datasets where some years are auction and some are snake:
    - Auction years: uses cost_bucket for peer comparison, cost_weighted for weighting
    - Snake years: uses round for peer comparison, exponential decay for weighting

    Scoring Philosophy:
    - Each pick is evaluated against peers at similar draft capital (cost bucket or round)
    - Early round MISSES are penalized heavily (wasted premium capital)
    - Late round HITS get bonus credit (found value where others didn't)
    - Final score = weighted average of adjusted quality scores

    Args:
        df: Draft DataFrame with player-level stats
        manager_column: Column containing manager name/id
        year_column: Column containing year
        spar_column: Column containing SPAR values
        pick_column: Column indicating drafted (non-null = drafted)
        peer_group_column: Column to group by for peer comparison (auto-detected if None)
        weight_method: How to weight picks ('auto', 'exponential', 'linear', 'cost_weighted', 'equal')
        weight_decay: Decay rate for exponential weighting (default: 0.85)
        use_asymmetric_scoring: Apply asymmetric hit/miss adjustments (default: True)
        early_miss_penalty: Multiplier for early round misses (default: 1.5)
        early_hit_bonus: Multiplier for early round hits (default: 1.0)
        late_miss_penalty: Multiplier for late round misses (default: 0.5)
        late_hit_bonus: Multiplier for late round hits (default: 1.5)
        percentile_bins: List of bin edges for grade assignment
        grade_labels: List of grade labels
        include_keepers: Whether to include keepers in aggregation

    Returns:
        Original DataFrame with columns added:
        - pick_quality_score: How well this pick performed vs peers (0-100)
        - adjusted_quality_score: Quality score after asymmetric adjustments
        - pick_weight: Weight of this pick for manager grade calculation
        - manager_draft_score: Weighted average of adjusted quality scores
        - manager_draft_percentile: Percentile rank among managers that year
        - manager_draft_grade: Letter grade (A-F)

    Example:
        # Basic usage (auto-detects draft type per year)
        df = calculate_manager_draft_grades(df)

        # Force specific peer group and weight method
        df = calculate_manager_draft_grades(df, peer_group_column='cost_bucket', weight_method='cost_weighted')

        # Disable asymmetric scoring (treat hits and misses equally)
        df = calculate_manager_draft_grades(df, use_asymmetric_scoring=False)

        # Harsher early round miss penalty
        df = calculate_manager_draft_grades(df, early_miss_penalty=2.0)

        # Bigger bonus for late round gems
        df = calculate_manager_draft_grades(df, late_hit_bonus=2.0)
    """
    df = df.copy()

    # Show detected draft types per year
    if year_column in df.columns:
        year_types = detect_draft_type_per_year(df, year_column)
        auction_years = [y for y, t in year_types.items() if t == 'auction']
        snake_years = [y for y, t in year_types.items() if t == 'snake']
        print(f"  [INFO] Draft types detected:")
        if auction_years:
            print(f"         Auction years ({len(auction_years)}): cost_bucket peers, cost-weighted")
        if snake_years:
            print(f"         Snake years ({len(snake_years)}): round peers, exponential decay")

    # Step 1: Calculate pick quality score (vs peer group)
    peer_desc = peer_group_column if peer_group_column else "auto-detected"
    print(f"  [QUALITY] Calculating pick quality scores (vs {peer_desc} peers)...")
    df = calculate_pick_quality_score(
        df,
        spar_column=spar_column,
        group_column=peer_group_column,
        year_column=year_column,
        per_year=(peer_group_column is None)  # Only per-year if auto-detecting
    )

    # Step 2: Apply asymmetric scoring adjustments
    if use_asymmetric_scoring:
        print("  [ASYMMETRIC] Applying asymmetric hit/miss adjustments...")
        print(f"      Early miss penalty: {early_miss_penalty}x | Early hit bonus: {early_hit_bonus}x")
        print(f"      Late miss penalty: {late_miss_penalty}x | Late hit bonus: {late_hit_bonus}x")
        df = apply_asymmetric_scoring(
            df,
            quality_column='pick_quality_score',
            year_column=year_column,
            early_miss_penalty=early_miss_penalty,
            early_hit_bonus=early_hit_bonus,
            late_miss_penalty=late_miss_penalty,
            late_hit_bonus=late_hit_bonus
        )
        quality_for_scoring = 'adjusted_quality_score'
    else:
        df['adjusted_quality_score'] = df['pick_quality_score']
        quality_for_scoring = 'pick_quality_score'

    # Step 3: Calculate pick weights
    print(f"  [WEIGHT] Calculating pick weights (method: {weight_method})...")
    df = calculate_pick_weight(
        df,
        method=weight_method,
        year_column=year_column,
        decay_rate=weight_decay
    )

    # Step 4: Calculate weighted manager scores using adjusted quality
    print("  [SCORE] Calculating weighted manager draft scores...")
    manager_scores = calculate_weighted_manager_score(
        df,
        manager_column=manager_column,
        year_column=year_column,
        quality_column=quality_for_scoring,
        pick_column=pick_column
    )

    if manager_scores.empty:
        print("  [WARN] No manager stats to calculate grades")
        df['manager_draft_score'] = pd.NA
        df['manager_draft_percentile'] = pd.NA
        df['manager_draft_grade'] = pd.NA
        return df

    print(f"  [OK] Calculated scores for {len(manager_scores):,} manager-year combinations")

    # Step 4: Also get traditional aggregate stats for reference
    agg_stats = aggregate_manager_draft_stats(
        df,
        manager_column=manager_column,
        year_column=year_column,
        spar_column=spar_column,
        pick_column=pick_column,
        include_keepers=include_keepers
    )

    # Merge weighted scores with aggregate stats
    manager_stats = manager_scores.merge(
        agg_stats[[year_column, manager_column, 'total_spar', 'avg_spar', 'hit_rate',
                   'total_picks', 'positive_spar_picks']],
        on=[year_column, manager_column],
        how='left'
    )

    # Step 5: Calculate percentile based on weighted score (alltime for cross-year comparison)
    print("  [PERCENTILE] Calculating all-time percentile for cross-year grading...")
    manager_stats = calculate_manager_draft_percentile(
        manager_stats,
        score_column='manager_draft_score',
        year_column=year_column,
        alltime=True  # Grade based on all-time percentile for cross-year comparison
    )

    # Step 6: Assign grades (based on all-time percentile)
    manager_stats = assign_manager_draft_grade(
        manager_stats,
        percentile_bins=percentile_bins,
        grade_labels=grade_labels
    )

    # Step 7: Select columns to merge back
    merge_cols = [year_column, manager_column]
    stats_cols = [
        'manager_draft_score', 'manager_draft_percentile', 'manager_draft_percentile_alltime',
        'manager_draft_percentile_year', 'manager_draft_grade',
        'total_spar', 'avg_spar', 'hit_rate', 'total_picks', 'positive_spar_picks',
        'avg_quality', 'avg_weight'
    ]

    # Rename for clarity when merged
    rename_dict = {
        'total_spar': 'manager_total_spar',
        'avg_spar': 'manager_avg_spar',
        'hit_rate': 'manager_hit_rate',
        'total_picks': 'manager_total_picks',
        'positive_spar_picks': 'manager_positive_picks',
        'avg_quality': 'manager_avg_pick_quality',
        'avg_weight': 'manager_avg_pick_weight'
    }

    available_stats = [c for c in stats_cols if c in manager_stats.columns]
    manager_subset = manager_stats[merge_cols + available_stats].copy()
    manager_subset = manager_subset.rename(columns=rename_dict)

    # Step 8: Merge back to original DataFrame
    # Remove any existing manager grade columns first
    cols_to_drop = [c for c in manager_subset.columns if c not in merge_cols and c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = df.merge(manager_subset, on=merge_cols, how='left')

    return df


def get_manager_draft_leaderboard(
    df: pd.DataFrame,
    year: Optional[int] = None,
    manager_column: str = 'manager',
    year_column: str = 'year'
) -> pd.DataFrame:
    """
    Get a leaderboard of manager draft grades.

    Args:
        df: DataFrame with manager draft grades calculated
        year: Optional year to filter to (None = all years)
        manager_column: Column containing manager name
        year_column: Column containing year

    Returns:
        DataFrame with one row per manager-year, sorted by grade/score
    """
    required_cols = ['manager_draft_grade', 'manager_draft_score', 'manager_draft_percentile']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Run calculate_manager_draft_grades first.")

    # Get unique manager-year combinations
    leaderboard_cols = [
        year_column, manager_column,
        'manager_draft_grade', 'manager_draft_score', 'manager_draft_percentile',
        'manager_total_spar', 'manager_avg_spar', 'manager_hit_rate',
        'manager_total_picks', 'manager_positive_picks'
    ]
    available_cols = [c for c in leaderboard_cols if c in df.columns]

    leaderboard = df[available_cols].drop_duplicates().copy()

    if year is not None:
        leaderboard = leaderboard[leaderboard[year_column] == year]

    # Sort by year (desc) then score (desc)
    leaderboard = leaderboard.sort_values(
        [year_column, 'manager_draft_score'],
        ascending=[False, False]
    ).reset_index(drop=True)

    return leaderboard


def get_manager_career_grades(
    df: pd.DataFrame,
    manager_column: str = 'manager'
) -> pd.DataFrame:
    """
    Get career-level draft grade summary for each manager.

    Args:
        df: DataFrame with manager draft grades calculated
        manager_column: Column containing manager name

    Returns:
        DataFrame with career stats per manager
    """
    required_cols = ['manager_draft_score', 'manager_total_spar']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Run calculate_manager_draft_grades first.")

    # Get unique manager-year stats first
    manager_years = df[[
        manager_column, 'year', 'manager_draft_score', 'manager_draft_grade',
        'manager_total_spar', 'manager_avg_spar', 'manager_hit_rate',
        'manager_total_picks', 'manager_positive_picks'
    ]].drop_duplicates()

    # Aggregate to career level
    career_stats = manager_years.groupby(manager_column, dropna=False).agg(
        seasons=('year', 'nunique'),
        career_total_spar=('manager_total_spar', 'sum'),
        career_avg_score=('manager_draft_score', 'mean'),
        career_avg_spar_per_pick=('manager_avg_spar', 'mean'),
        career_avg_hit_rate=('manager_hit_rate', 'mean'),
        total_career_picks=('manager_total_picks', 'sum'),
        total_positive_picks=('manager_positive_picks', 'sum'),
        best_season_score=('manager_draft_score', 'max'),
        worst_season_score=('manager_draft_score', 'min')
    ).reset_index()

    # Grade distribution
    grade_counts = manager_years.groupby([manager_column, 'manager_draft_grade'], dropna=False).size().unstack(fill_value=0)
    grade_counts.columns = [f'seasons_grade_{g}' for g in grade_counts.columns]
    grade_counts = grade_counts.reset_index()

    career_stats = career_stats.merge(grade_counts, on=manager_column, how='left')

    # Sort by career total SPAR
    career_stats = career_stats.sort_values('career_total_spar', ascending=False).reset_index(drop=True)

    return career_stats
