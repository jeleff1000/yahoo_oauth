"""
Bench Insurance Value Calculator Module

Calculates data-driven bench value metrics for draft optimization.
Replaces hardcoded discount factors with actual league data.

Core Philosophy:
- Bench value = P(starter fails) × E[backup SPAR when activated]
- Each position has different failure rates and activation values
- Manager behavior patterns affect optimal bench strategy

Key Metrics Calculated:
1. Starter Failure Rates (per position/tier):
   - bust_rate: % of starters with negative manager_SPAR
   - injury_rate: % of starters who missed significant weeks (< 10 weeks)
   - combined_failure_rate: Probability a starter needs replacement

2. Bench Activation Metrics (per position/tier):
   - activation_rate: % of backups who got starting time
   - activation_spar: Average SPAR when backup is actually started
   - activation_ppg: Average PPG when backup is started

3. Insurance Value (derived):
   - insurance_value = failure_rate × activation_rate × activation_spar
   - This replaces hardcoded position discounts (QB: 5%, RB: 40%, etc.)

4. Manager Behavior Patterns:
   - bench_utilization_style: aggressive/moderate/conservative
   - avg_bench_activation_rate: How often this manager uses bench
   - waiver_activity: Roster churn rate

Usage:
    from modules.bench_insurance_calculator import (
        calculate_bench_insurance_metrics,
        calculate_manager_bench_patterns,
        get_optimal_bench_composition
    )

    # Calculate league-wide metrics
    bench_metrics = calculate_bench_insurance_metrics(player_df, draft_df)

    # Get manager-specific patterns
    manager_patterns = calculate_manager_bench_patterns(player_df)

    # Get optimal bench for optimizer
    optimal_bench = get_optimal_bench_composition(
        roster_structure,
        bench_metrics,
        manager_style='average'
    )
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Literal
from pathlib import Path


# Default thresholds (can be overridden)
DEFAULT_MIN_WEEKS_FOR_STARTER = 10  # Starters expected to play at least this many weeks
DEFAULT_MIN_SAMPLE_SIZE = 5  # Minimum samples to calculate reliable metrics
DEFAULT_INJURY_WEEKS_THRESHOLD = 8  # Below this = injured/bust season


def calculate_starter_failure_rates(
    player_df: pd.DataFrame,
    draft_df: pd.DataFrame,
    spar_column: str = 'manager_spar',
    position_column: str = 'yahoo_position',
    year_column: str = 'year',
    min_weeks_for_starter: int = DEFAULT_MIN_WEEKS_FOR_STARTER,
    injury_weeks_threshold: int = DEFAULT_INJURY_WEEKS_THRESHOLD,
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE
) -> pd.DataFrame:
    """
    Calculate position-specific starter failure rates from historical data.

    A "failure" means the manager needs to rely on bench players, either due to:
    - Performance bust: Player has negative manager_SPAR
    - Availability bust: Player plays < injury_weeks_threshold weeks

    Args:
        player_df: Player DataFrame with weekly stats (for weeks_started)
        draft_df: Draft DataFrame with SPAR and starter designation
        spar_column: Column containing SPAR values
        position_column: Column containing position
        year_column: Column containing year
        min_weeks_for_starter: Expected minimum weeks for a healthy starter
        injury_weeks_threshold: Below this = availability failure
        min_sample_size: Minimum samples for reliable calculation

    Returns:
        DataFrame with columns:
        - position, position_tier
        - total_starters: Count of starters at this position/tier
        - bust_count: Count with negative SPAR
        - injury_bust_count: Count with < threshold weeks
        - bust_rate: bust_count / total_starters
        - injury_rate: injury_bust_count / total_starters
        - combined_failure_rate: P(bust OR injury)
    """
    # Use draft_df since it has starter designation and season-level SPAR
    df = draft_df.copy()

    # Ensure required columns exist
    required_cols = [spar_column, position_column, year_column]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing columns for starter failure calculation: {missing}")
        return pd.DataFrame()

    # Filter to starters only
    if 'drafted_as_starter' in df.columns:
        starters = df[df['drafted_as_starter'] == 1].copy()
    else:
        # Fallback: assume top N picks per position are starters
        print("  [INFO] No drafted_as_starter column, using all drafted players")
        starters = df[df['pick'].notna()].copy()

    if starters.empty:
        print("  [WARN] No starters found for failure rate calculation")
        return pd.DataFrame()

    # Ensure numeric types
    starters[spar_column] = pd.to_numeric(starters[spar_column], errors='coerce').fillna(0)

    # Get weeks_started from draft_df (should be derived in player_to_draft_v2.py)
    # DO NOT fallback to games_played - it inflates "starter usage" incorrectly
    if 'weeks_started' in starters.columns:
        starters['weeks_started'] = pd.to_numeric(starters['weeks_started'], errors='coerce').fillna(0)
    else:
        # No weeks_started column - log warning and use 0 (will mark as availability bust)
        print("  [WARN] No weeks_started column - derive it in player_to_draft_v2.py aggregation")
        starters['weeks_started'] = 0

    # Define failure conditions
    starters['is_performance_bust'] = (starters[spar_column] < 0).astype(int)
    starters['is_availability_bust'] = (starters['weeks_started'] < injury_weeks_threshold).astype(int)
    starters['is_any_failure'] = ((starters['is_performance_bust'] == 1) |
                                   (starters['is_availability_bust'] == 1)).astype(int)

    # Determine grouping columns
    group_cols = [position_column]
    if 'position_tier' in starters.columns:
        group_cols.append('position_tier')
    elif 'cost_bucket' in starters.columns:
        group_cols.append('cost_bucket')

    # Aggregate by position (and tier if available)
    failure_rates = starters.groupby(group_cols, dropna=False).agg(
        total_starters=('is_any_failure', 'count'),
        bust_count=('is_performance_bust', 'sum'),
        injury_bust_count=('is_availability_bust', 'sum'),
        any_failure_count=('is_any_failure', 'sum'),
        avg_spar=(spar_column, 'mean'),
        avg_weeks_started=('weeks_started', 'mean')
    ).reset_index()

    # Calculate rates
    failure_rates['bust_rate'] = (
        failure_rates['bust_count'] / failure_rates['total_starters'].clip(lower=1)
    ).round(3)

    failure_rates['injury_rate'] = (
        failure_rates['injury_bust_count'] / failure_rates['total_starters'].clip(lower=1)
    ).round(3)

    failure_rates['combined_failure_rate'] = (
        failure_rates['any_failure_count'] / failure_rates['total_starters'].clip(lower=1)
    ).round(3)

    # Filter to positions with enough samples
    failure_rates = failure_rates[failure_rates['total_starters'] >= min_sample_size].copy()

    print(f"  [OK] Calculated starter failure rates for {len(failure_rates)} position/tier combinations")

    return failure_rates


def calculate_bench_activation_metrics(
    player_df: pd.DataFrame,
    draft_df: pd.DataFrame,
    spar_column: str = 'manager_spar',
    position_column: str = 'yahoo_position',
    year_column: str = 'year',
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE
) -> pd.DataFrame:
    """
    Calculate metrics for when bench players are activated (actually started).

    This measures the actual value bench players provide when they see action,
    not their total season value (which is diluted by weeks they sit).

    Args:
        player_df: Player DataFrame with weekly stats
        draft_df: Draft DataFrame with starter designation
        spar_column: Column containing SPAR values
        position_column: Column containing position
        year_column: Column containing year
        min_sample_size: Minimum samples for reliable calculation

    Returns:
        DataFrame with columns:
        - position, position_tier
        - total_backups: Count of backup players drafted
        - activated_backups: Count who got at least 1 start
        - activation_rate: activated_backups / total_backups
        - avg_activation_spar: Average SPAR when backup is started
        - avg_activation_weeks: Average weeks started when activated
        - avg_activation_ppg: Average PPG during started weeks
    """
    df = draft_df.copy()

    # Filter to backups only
    if 'drafted_as_starter' in df.columns:
        backups = df[df['drafted_as_starter'] == 0].copy()
    elif 'drafted_as_backup' in df.columns:
        backups = df[df['drafted_as_backup'] == 1].copy()
    else:
        print("  [INFO] No backup designation, inferring from position rank")
        # Fallback: players ranked beyond starter slots are backups
        backups = df[df['pick'].notna()].copy()

    if backups.empty:
        print("  [WARN] No backups found for activation metrics")
        return pd.DataFrame()

    # Ensure numeric types
    backups[spar_column] = pd.to_numeric(backups[spar_column], errors='coerce').fillna(0)

    # Get weeks_started - DO NOT fallback to games_played
    if 'weeks_started' in backups.columns:
        backups['weeks_started'] = pd.to_numeric(backups['weeks_started'], errors='coerce').fillna(0)
    else:
        print("  [WARN] No weeks_started column for backups - derive it in player_to_draft_v2.py")
        backups['weeks_started'] = 0

    # Mark activated backups (got at least 1 start)
    backups['was_activated'] = (backups['weeks_started'] > 0).astype(int)

    # Determine grouping columns
    group_cols = [position_column]
    if 'position_tier' in backups.columns:
        group_cols.append('position_tier')
    elif 'cost_bucket' in backups.columns:
        group_cols.append('cost_bucket')

    # Aggregate by position (and tier if available)
    activation_metrics = backups.groupby(group_cols, dropna=False).agg(
        total_backups=('was_activated', 'count'),
        activated_backups=('was_activated', 'sum'),
        total_backup_spar=(spar_column, 'sum'),
        avg_backup_spar=(spar_column, 'mean'),
        total_weeks_started=('weeks_started', 'sum'),
        avg_weeks_started=('weeks_started', 'mean')
    ).reset_index()

    # Calculate activation rate
    activation_metrics['activation_rate'] = (
        activation_metrics['activated_backups'] / activation_metrics['total_backups'].clip(lower=1)
    ).round(3)

    # Calculate SPAR per activated backup (only those who played)
    activated_only = backups[backups['was_activated'] == 1].copy()

    if not activated_only.empty:
        # Calculate SPAR per start (per-game value, not season total)
        # A backup with +4 SPAR in 1 game is more valuable than +4 SPAR over 4 games
        activated_only['spar_per_start'] = (
            activated_only[spar_column] / activated_only['weeks_started'].clip(lower=1)
        )

        activated_spar = activated_only.groupby(group_cols, dropna=False).agg(
            avg_activation_spar=(spar_column, 'mean'),  # Season total (for reference)
            avg_activation_weeks=('weeks_started', 'mean'),
            avg_spar_per_start=('spar_per_start', 'mean')  # Per-start value (for insurance formula)
        ).reset_index()

        # Also get PPG if available
        if 'season_ppg' in activated_only.columns:
            activated_ppg = activated_only.groupby(group_cols, dropna=False).agg(
                avg_activation_ppg=('season_ppg', 'mean')
            ).reset_index()
            activated_spar = activated_spar.merge(activated_ppg, on=group_cols, how='left')

        activation_metrics = activation_metrics.merge(
            activated_spar, on=group_cols, how='left'
        )
    else:
        activation_metrics['avg_activation_spar'] = 0.0
        activation_metrics['avg_activation_weeks'] = 0.0
        activation_metrics['avg_spar_per_start'] = 0.0

    # Filter to positions with enough samples
    activation_metrics = activation_metrics[
        activation_metrics['total_backups'] >= min_sample_size
    ].copy()

    print(f"  [OK] Calculated bench activation metrics for {len(activation_metrics)} position/tier combinations")

    return activation_metrics


def calculate_insurance_value(
    failure_rates: pd.DataFrame,
    activation_metrics: pd.DataFrame,
    position_column: str = 'yahoo_position'
) -> pd.DataFrame:
    """
    Calculate the insurance value of bench players.

    Insurance Value = P(need backup) × P(backup activated) × E[SPAR per start]

    IMPORTANT: Uses per-start SPAR, not season SPAR.
    A backup with +4 SPAR in 1 game should be valued higher than +4 SPAR over 4 games.

    Args:
        failure_rates: DataFrame from calculate_starter_failure_rates
        activation_metrics: DataFrame from calculate_bench_activation_metrics
        position_column: Column containing position

    Returns:
        DataFrame with insurance_value per position/tier
    """
    if failure_rates.empty or activation_metrics.empty:
        print("  [WARN] Cannot calculate insurance value - missing input data")
        return pd.DataFrame()

    # Determine merge columns
    merge_cols = [position_column]
    if 'position_tier' in failure_rates.columns and 'position_tier' in activation_metrics.columns:
        merge_cols.append('position_tier')
    elif 'cost_bucket' in failure_rates.columns and 'cost_bucket' in activation_metrics.columns:
        merge_cols.append('cost_bucket')

    # Merge failure rates with activation metrics
    insurance_df = failure_rates.merge(
        activation_metrics,
        on=merge_cols,
        how='outer',
        suffixes=('_failure', '_activation')
    )

    # Calculate league averages from actual data (no hardcoded values)
    league_avg_failure_rate = failure_rates['combined_failure_rate'].mean() if 'combined_failure_rate' in failure_rates.columns else None
    league_avg_activation_rate = activation_metrics['activation_rate'].mean() if 'activation_rate' in activation_metrics.columns else None

    # Fill missing values with league averages (data-driven, not hardcoded)
    if league_avg_failure_rate is not None:
        insurance_df['combined_failure_rate'] = insurance_df['combined_failure_rate'].fillna(league_avg_failure_rate)
    else:
        insurance_df['combined_failure_rate'] = insurance_df['combined_failure_rate'].fillna(0)

    if league_avg_activation_rate is not None:
        insurance_df['activation_rate'] = insurance_df['activation_rate'].fillna(league_avg_activation_rate)
    else:
        insurance_df['activation_rate'] = insurance_df['activation_rate'].fillna(0)

    # Use per-start SPAR (not season total) for insurance formula
    # This properly values backups who provide high value when activated
    if 'avg_spar_per_start' in insurance_df.columns:
        insurance_df['avg_spar_per_start'] = insurance_df['avg_spar_per_start'].fillna(0)
        activation_value = insurance_df['avg_spar_per_start']
    else:
        # Fallback to season SPAR if per-start not available
        insurance_df['avg_activation_spar'] = insurance_df['avg_activation_spar'].fillna(0)
        activation_value = insurance_df['avg_activation_spar']

    # Calculate insurance value using per-start value
    # This is the expected SPAR contribution PER START from having this backup
    insurance_df['insurance_value'] = (
        insurance_df['combined_failure_rate'] *
        insurance_df['activation_rate'] *
        activation_value
    ).round(2)

    # Also calculate a "bench discount factor" for backward compatibility
    # This is what we'd use to discount starter SPAR for bench slots
    # Derived from: bench_value / starter_value ≈ insurance_value / avg_starter_spar_per_game
    if 'avg_spar' in insurance_df.columns and 'avg_weeks_started' in insurance_df.columns:
        avg_starter_spar_per_game = (
            insurance_df['avg_spar'] / insurance_df['avg_weeks_started'].clip(lower=1)
        ).fillna(1).clip(lower=0.1)
        insurance_df['bench_discount_factor'] = (
            insurance_df['insurance_value'] / avg_starter_spar_per_game
        ).clip(lower=0.01, upper=1.0).round(3)
    else:
        # Fallback discount based on activation rate and failure rate
        insurance_df['bench_discount_factor'] = (
            insurance_df['activation_rate'] *
            insurance_df['combined_failure_rate'].clip(lower=0.1)
        ).clip(lower=0.01, upper=0.5).round(3)

    print(f"  [OK] Calculated insurance values for {len(insurance_df)} position/tier combinations")

    return insurance_df


def calculate_manager_bench_patterns(
    player_df: pd.DataFrame,
    manager_column: str = 'manager',
    year_column: str = 'year',
    min_seasons: int = 1
) -> pd.DataFrame:
    """
    Calculate manager-specific bench utilization patterns.

    This helps personalize the optimizer - aggressive managers who
    actively manage their bench should value bench differently than
    conservative managers who set-and-forget.

    Args:
        player_df: Player DataFrame with weekly stats
        manager_column: Column containing manager name
        year_column: Column containing year
        min_seasons: Minimum seasons to calculate pattern

    Returns:
        DataFrame with manager-level metrics:
        - bench_utilization_rate: % of bench player-weeks that resulted in starts
        - avg_bench_weeks_per_player: Average weeks bench players were started
        - roster_churn: How many roster moves per season
        - style: 'aggressive' / 'moderate' / 'conservative'
    """
    df = player_df.copy()

    # Filter to rostered players only
    if manager_column not in df.columns:
        print(f"  [WARN] Manager column '{manager_column}' not found")
        return pd.DataFrame()

    rostered = df[df[manager_column] != 'Unrostered'].copy()

    if rostered.empty:
        return pd.DataFrame()

    # Identify bench weeks (BN or IR position)
    if 'fantasy_position' in rostered.columns:
        rostered['is_bench_week'] = rostered['fantasy_position'].isin(['BN', 'IR', 'IR+']).astype(int)
        rostered['is_started_week'] = (~rostered['fantasy_position'].isin(['BN', 'IR', 'IR+']) &
                                        rostered['fantasy_position'].notna()).astype(int)
    else:
        # Can't determine bench vs started without fantasy_position
        print("  [INFO] No fantasy_position column, using simplified metrics")
        rostered['is_bench_week'] = 0
        rostered['is_started_week'] = 1

    # Use franchise_id for career grouping if available (handles multi-team managers correctly)
    # Falls back to manager_column for backwards compatibility
    career_group_col = 'franchise_id' if 'franchise_id' in rostered.columns and rostered['franchise_id'].notna().any() else manager_column

    # Group by manager-year (still use manager_column for per-season stats)
    manager_stats = rostered.groupby([manager_column, year_column], dropna=False).agg(
        total_player_weeks=('is_started_week', 'count'),
        bench_weeks=('is_bench_week', 'sum'),
        started_weeks=('is_started_week', 'sum'),
        unique_players=('yahoo_player_id', 'nunique')
    ).reset_index()

    # Add franchise_id to manager_stats if available (for career grouping)
    if career_group_col == 'franchise_id':
        franchise_lookup = rostered[[manager_column, 'franchise_id']].drop_duplicates()
        manager_stats = manager_stats.merge(franchise_lookup, on=manager_column, how='left')

    # Calculate bench utilization
    manager_stats['bench_utilization_rate'] = (
        manager_stats['started_weeks'] /
        manager_stats['total_player_weeks'].clip(lower=1)
    ).round(3)

    # Aggregate to franchise/manager career level
    career_stats = manager_stats.groupby(career_group_col, dropna=False).agg(
        seasons=(year_column, 'nunique'),
        avg_bench_utilization=('bench_utilization_rate', 'mean'),
        avg_unique_players=('unique_players', 'mean'),
        total_player_weeks=('total_player_weeks', 'sum')
    ).reset_index()

    # Filter to managers with enough history
    career_stats = career_stats[career_stats['seasons'] >= min_seasons].copy()

    # Classify style based on utilization
    def classify_style(utilization):
        if utilization >= 0.7:
            return 'aggressive'
        elif utilization >= 0.5:
            return 'moderate'
        else:
            return 'conservative'

    career_stats['bench_style'] = career_stats['avg_bench_utilization'].apply(classify_style)

    print(f"  [OK] Calculated bench patterns for {len(career_stats)} managers")

    return career_stats


def calculate_bench_insurance_metrics(
    player_df: pd.DataFrame,
    draft_df: pd.DataFrame,
    position_column: str = 'yahoo_position',
    spar_column: str = 'manager_spar',
    year_column: str = 'year'
) -> Dict:
    """
    Main entry point: Calculate all bench insurance metrics.

    This is the function to call from the optimizer to get data-driven
    bench values instead of hardcoded discounts.

    Args:
        player_df: Player DataFrame with weekly stats
        draft_df: Draft DataFrame with SPAR and starter designation
        position_column: Column containing position
        spar_column: Column containing SPAR values
        year_column: Column containing year

    Returns:
        Dict containing:
        - failure_rates: DataFrame of starter failure rates by position/tier
        - activation_metrics: DataFrame of bench activation metrics
        - insurance_values: DataFrame with calculated insurance values
        - position_discounts: Dict mapping position to discount factor
        - summary: Summary statistics
    """
    print("\n[BENCH INSURANCE] Calculating data-driven bench metrics...")

    # Step 1: Calculate starter failure rates
    print("  Step 1: Calculating starter failure rates...")
    failure_rates = calculate_starter_failure_rates(
        player_df, draft_df,
        spar_column=spar_column,
        position_column=position_column,
        year_column=year_column
    )

    # Step 2: Calculate bench activation metrics
    print("  Step 2: Calculating bench activation metrics...")
    activation_metrics = calculate_bench_activation_metrics(
        player_df, draft_df,
        spar_column=spar_column,
        position_column=position_column,
        year_column=year_column
    )

    # Step 3: Calculate insurance values
    print("  Step 3: Calculating insurance values...")
    insurance_values = calculate_insurance_value(
        failure_rates, activation_metrics,
        position_column=position_column
    )

    # Step 4: Build position discount lookup (for optimizer) - fully data-driven
    print("  Step 4: Building position discount lookup...")
    position_discounts = {}

    if not insurance_values.empty:
        # Average discount by position (across all tiers)
        pos_discounts = insurance_values.groupby(position_column)['bench_discount_factor'].mean()
        position_discounts = pos_discounts.to_dict()

    # Calculate league average for positions with no data (no hardcoded values)
    league_avg_discount = None
    if position_discounts:
        league_avg_discount = sum(position_discounts.values()) / len(position_discounts)

    # Get all positions that exist in the data
    all_positions_in_data = set()
    if not failure_rates.empty:
        all_positions_in_data.update(failure_rates[position_column].unique())
    if not activation_metrics.empty:
        all_positions_in_data.update(activation_metrics[position_column].unique())

    # For positions with no discount calculated, use league average (not hardcoded)
    for pos in all_positions_in_data:
        if pos not in position_discounts:
            if league_avg_discount is not None:
                position_discounts[pos] = league_avg_discount
                print(f"    {pos}: {league_avg_discount:.2%} (league average - no position-specific data)")
            else:
                position_discounts[pos] = 0.01  # Minimum only when truly no data
                print(f"    {pos}: 1.00% (no data available)")
        else:
            print(f"    {pos}: {position_discounts[pos]:.2%} (data-driven)")

    # Build summary
    summary = {
        'positions_calculated': len(position_discounts),
        'failure_rate_samples': len(failure_rates) if not failure_rates.empty else 0,
        'activation_samples': len(activation_metrics) if not activation_metrics.empty else 0,
        'avg_failure_rate': failure_rates['combined_failure_rate'].mean() if not failure_rates.empty else None,
        'avg_activation_rate': activation_metrics['activation_rate'].mean() if not activation_metrics.empty else None
    }

    print(f"  [OK] Bench insurance calculation complete")

    return {
        'failure_rates': failure_rates,
        'activation_metrics': activation_metrics,
        'insurance_values': insurance_values,
        'position_discounts': position_discounts,
        'summary': summary
    }


def get_optimal_bench_composition(
    roster_structure: Dict[str, int],
    bench_metrics: Dict,
    total_bench_spots: int,
    manager_style: Literal['aggressive', 'moderate', 'conservative', 'average'] = 'average'
) -> Dict:
    """
    Calculate optimal bench composition based on insurance value analysis.

    Uses the principle: allocate more bench spots to positions with higher
    insurance value (higher failure rates AND higher activation value).

    Args:
        roster_structure: Dict mapping position to starter count
        bench_metrics: Output from calculate_bench_insurance_metrics
        total_bench_spots: Number of bench spots to fill
        manager_style: Manager's bench utilization style

    Returns:
        Dict with:
        - recommended_spots: Dict mapping position to recommended bench spots
        - position_weights: Normalized weights used for allocation
        - reasoning: Explanation of the allocation
    """
    insurance_values = bench_metrics.get('insurance_values', pd.DataFrame())
    position_discounts = bench_metrics.get('position_discounts', {})

    # Calculate base weights from insurance values
    bench_positions = ['QB', 'RB', 'WR', 'TE']  # Positions that can be benched
    weights = {}

    for pos in bench_positions:
        # Get insurance value for this position
        if not insurance_values.empty and 'yahoo_position' in insurance_values.columns:
            pos_data = insurance_values[insurance_values['yahoo_position'] == pos]
            if not pos_data.empty:
                insurance_val = pos_data['insurance_value'].mean()
            else:
                insurance_val = position_discounts.get(pos, 0.2)
        else:
            insurance_val = position_discounts.get(pos, 0.2)

        # Weight by starter count (more starters = need more depth)
        starter_count = roster_structure.get(pos, 1)

        # Combined weight
        weights[pos] = max(0.05, insurance_val * (1 + 0.2 * starter_count))

    # Adjust for manager style
    style_adjustments = {
        'aggressive': {'QB': 0.8, 'RB': 1.1, 'WR': 1.1, 'TE': 1.0},
        'moderate': {'QB': 1.0, 'RB': 1.0, 'WR': 1.0, 'TE': 1.0},
        'conservative': {'QB': 1.2, 'RB': 0.9, 'WR': 0.9, 'TE': 1.0},
        'average': {'QB': 1.0, 'RB': 1.0, 'WR': 1.0, 'TE': 1.0}
    }

    adjustments = style_adjustments.get(manager_style, style_adjustments['average'])
    for pos in weights:
        weights[pos] *= adjustments.get(pos, 1.0)

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {p: w / total_weight for p, w in weights.items()}

    # Allocate spots
    recommended_spots = {}
    remaining_spots = total_bench_spots

    for pos in sorted(weights.keys(), key=lambda p: weights[p], reverse=True):
        spots = max(1, round(total_bench_spots * weights[pos]))
        spots = min(spots, remaining_spots)
        recommended_spots[pos] = spots
        remaining_spots -= spots

        if remaining_spots <= 0:
            break

    # Distribute any remaining spots to highest weight positions
    while remaining_spots > 0:
        max_pos = max(weights.keys(), key=lambda p: weights[p])
        recommended_spots[max_pos] = recommended_spots.get(max_pos, 0) + 1
        remaining_spots -= 1

    # Build reasoning
    reasoning = []
    for pos in bench_positions:
        spots = recommended_spots.get(pos, 0)
        weight = weights.get(pos, 0)
        discount = position_discounts.get(pos, 0)
        reasoning.append(
            f"{pos}: {spots} spot(s) | "
            f"Insurance weight: {weight:.1%} | "
            f"Activation discount: {discount:.1%}"
        )

    return {
        'recommended_spots': recommended_spots,
        'position_weights': weights,
        'reasoning': reasoning,
        'manager_style': manager_style,
        'total_spots': total_bench_spots
    }


def get_position_bench_discount(
    position: str,
    tier: Optional[int],
    bench_metrics: Dict
) -> float:
    """
    Get the bench discount factor for a specific position/tier.

    This is the function to call from the optimizer when building
    the dual-role data (starter vs bench SPAR).

    All values are data-driven - no hardcoded fallbacks.

    Args:
        position: Position (QB, RB, WR, TE, etc.)
        tier: Optional tier/cost_bucket for tier-specific discount
        bench_metrics: Output from calculate_bench_insurance_metrics (required)

    Returns:
        Discount factor (0.0 to 1.0) to apply to starter SPAR for bench value
    """
    # Try tier-specific discount first
    insurance_values = bench_metrics.get('insurance_values', pd.DataFrame())

    if not insurance_values.empty and tier is not None:
        tier_col = 'position_tier' if 'position_tier' in insurance_values.columns else 'cost_bucket'
        if tier_col in insurance_values.columns:
            tier_match = insurance_values[
                (insurance_values['yahoo_position'] == position) &
                (insurance_values[tier_col] == tier)
            ]
            if not tier_match.empty:
                return tier_match['bench_discount_factor'].iloc[0]

    # Fall back to position-level discount (data-driven)
    position_discounts = bench_metrics.get('position_discounts', {})

    if position in position_discounts:
        return position_discounts[position]

    # No data for this position - use league average from the data
    if position_discounts:
        league_avg = sum(position_discounts.values()) / len(position_discounts)
        return league_avg

    # Truly no data - return minimum (1%)
    return 0.01


def calculate_bench_value_by_rank(
    draft_df: pd.DataFrame,
    position_column: str = 'yahoo_position',
    spar_column: str = 'manager_spar',
    rank_column: str = 'position_draft_label',
    min_sample_size: int = 5,
    exclude_keepers: bool = True
) -> pd.DataFrame:
    """
    Calculate median SPAR by position_draft_rank (QB1, QB2, RB1, RB2, etc.).

    This is the purest data-driven approach to bench value:
    - Uses actual historical SPAR for each draft slot
    - SPAR already incorporates replacement level (waiver wire value)
    - If SPAR <= 0, the slot is worth $0 (waiver pickup is just as good)
    - EXCLUDES KEEPERS by default (keepers inflate values artificially)

    Args:
        draft_df: Draft DataFrame with SPAR and position_draft_label
        position_column: Column containing position
        spar_column: Column containing SPAR values (manager_spar recommended)
        rank_column: Column containing position rank label (QB1, QB2, etc.)
        min_sample_size: Minimum samples for reliable calculation
        exclude_keepers: If True, exclude keeper picks (they inflate values artificially)

    Returns:
        DataFrame with columns:
        - position_draft_label: The rank label (QB1, QB2, RB1, etc.)
        - yahoo_position: Position (QB, RB, WR, TE)
        - position_rank: Numeric rank (1, 2, 3, etc.)
        - median_spar: Median SPAR for this slot
        - mean_spar: Mean SPAR for this slot
        - pct_positive: % of picks that beat replacement
        - sample_size: Number of historical picks
        - bench_value: Recommended bench value (median_spar clipped at 0)
    """
    print("\n[BENCH VALUE BY RANK] Calculating median SPAR by position draft rank...")

    df = draft_df.copy()

    # Filter to drafted players with position rank
    if rank_column not in df.columns:
        print(f"  [WARN] No {rank_column} column found - cannot calculate rank-based values")
        return pd.DataFrame()

    # Base filter: drafted players with position rank
    filter_mask = (df['cost'] > 0) & (df[rank_column].notna())

    # Exclude keepers - they artificially inflate slot values
    # (e.g., RB6 includes kept star RBs like CMC, Le'Veon Bell)
    if exclude_keepers and 'is_keeper_status' in df.columns:
        df['is_keeper_status'] = pd.to_numeric(df['is_keeper_status'], errors='coerce').fillna(0)
        keeper_count = (df['is_keeper_status'] == 1).sum()
        filter_mask = filter_mask & (df['is_keeper_status'] != 1)
        print(f"  [INFO] Excluding {keeper_count} keeper picks (they inflate values artificially)")

    drafted = df[filter_mask].copy()

    if drafted.empty:
        print("  [WARN] No drafted players with position rank found")
        return pd.DataFrame()

    # Ensure numeric SPAR
    drafted[spar_column] = pd.to_numeric(drafted[spar_column], errors='coerce').fillna(0)

    # Extract position and rank number from label (e.g., "QB2" -> "QB", 2)
    drafted['position_rank'] = drafted[rank_column].str.extract(r'(\d+)$').astype(float)

    # Aggregate by position_draft_label
    rank_stats = drafted.groupby([rank_column, position_column]).agg(
        sample_size=(spar_column, 'count'),
        median_spar=(spar_column, 'median'),
        mean_spar=(spar_column, 'mean'),
        std_spar=(spar_column, 'std'),
        pct_positive=(spar_column, lambda x: (x > 0).mean() * 100)
    ).reset_index()

    # Get position_rank
    rank_stats['position_rank'] = rank_stats[rank_column].str.extract(r'(\d+)$').astype(float)

    # Filter to ranks with enough samples
    rank_stats = rank_stats[rank_stats['sample_size'] >= min_sample_size].copy()

    # Calculate bench_value: median SPAR clipped at 0
    # If median SPAR <= 0, the slot is replacement level = not worth drafting
    rank_stats['bench_value'] = rank_stats['median_spar'].clip(lower=0)

    # Sort by position and rank
    rank_stats = rank_stats.sort_values([position_column, 'position_rank'])

    # Rename for clarity
    rank_stats = rank_stats.rename(columns={rank_column: 'position_draft_label'})

    print(f"  [OK] Calculated bench values for {len(rank_stats)} position/rank combinations")

    # Print summary
    print("\n  [BENCH VALUE BY RANK SUMMARY]:")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_data = rank_stats[rank_stats[position_column] == pos].sort_values('position_rank')
        if not pos_data.empty:
            print(f"    {pos}:")
            for _, row in pos_data.iterrows():
                label = row['position_draft_label']
                median = row['median_spar']
                bench_val = row['bench_value']
                pct = row['pct_positive']
                n = row['sample_size']
                verdict = "DRAFT" if bench_val > 5 else ("marginal" if bench_val > 0 else "SKIP")
                print(f"      {label}: median_spar={median:>6.1f}, bench_value={bench_val:>5.1f}, "
                      f"{pct:>4.0f}% beat waivers (n={n}) -> {verdict}")

    return rank_stats


def get_bench_value_for_rank(
    position_draft_label: str,
    bench_value_by_rank: pd.DataFrame
) -> float:
    """
    Get the bench value for a specific position draft rank.

    Args:
        position_draft_label: The rank label (e.g., "QB2", "RB3")
        bench_value_by_rank: DataFrame from calculate_bench_value_by_rank

    Returns:
        Bench value (median SPAR) for this slot, or 0 if not found
    """
    if bench_value_by_rank.empty:
        return 0.0

    match = bench_value_by_rank[
        bench_value_by_rank['position_draft_label'] == position_draft_label
    ]

    if not match.empty:
        return match['bench_value'].iloc[0]

    return 0.0
