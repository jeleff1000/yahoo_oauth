"""
Transaction SPAR Calculator Module

Calculates rest-of-season SPAR for waiver pickups, trades, and drops.

Dual metrics distinguish player talent from manager usage:
- player_spar_ros: All weeks the player played (talent/production)
- manager_spar_ros: Only weeks the manager started them (realized value)
- net_spar_ros: Aggregated gain/loss per transaction (grouped by transaction_id + manager)

Key Concepts:
- player_spar_ros = total_points_ros - (replacement_ppg_ros × weeks_ros)
- manager_spar_ros = total_points_started_ros - (replacement_ppg_ros × weeks_started_ros)
- Window-based replacement (Week W → 17, not full season)
- Waiver cost normalization (FAAB vs Priority)
- fa_roi = manager_spar_ros / waiver_cost_norm (ROI based on realized value)
"""

import pandas as pd
import numpy as np
from typing import Dict
import json
from pathlib import Path


def calculate_window_replacement(
    weekly_replacement_df: pd.DataFrame,
    year: int,
    position: str,
    start_week: int,
    end_week: int = 17
) -> float:
    """
    Calculate average replacement level for a specific window.

    Used for rest-of-season calculations: Week W → 17

    Args:
        weekly_replacement_df: Full weekly replacement DataFrame
        year: Season year
        position: Player position
        start_week: First week of window (transaction week + 1)
        end_week: Last week of window (default 17)

    Returns:
        Average replacement PPG for the window
    """
    window_data = weekly_replacement_df[
        (weekly_replacement_df['year'] == year) &
        (weekly_replacement_df['position'] == position) &
        (weekly_replacement_df['week'] >= start_week) &
        (weekly_replacement_df['week'] <= end_week)
    ]

    if window_data.empty:
        return 0.0

    return window_data['replacement_ppg'].mean()


def normalize_waiver_cost(
    transactions_df: pd.DataFrame,
    league_settings_path: Path
) -> pd.DataFrame:
    """
    Normalize waiver cost to make FAAB and Priority comparable.

    FAAB: waiver_cost_norm = FAAB spent
    Priority: waiver_cost_norm = (num_teams + 1) - priority
    Free: waiver_cost_norm = 0.1 (for ROI calculation)

    Args:
        transactions_df: Transactions DataFrame
        league_settings_path: Path to league settings JSON

    Returns:
        DataFrame with waiver_cost_norm column added
    """
    # Load league settings to get waiver type and team count
    with open(league_settings_path, 'r') as f:
        settings = json.load(f)

    num_teams = int(settings.get('metadata', {}).get('num_teams', 10))
    waiver_type = settings.get('settings', {}).get('waiver_type', 'FAAB')

    transactions_df = transactions_df.copy()

    # Determine waiver type (FAAB vs Priority)
    is_faab = waiver_type.upper() in ['FAAB', 'AUCTION']

    if is_faab:
        # FAAB: cost_norm = actual FAAB spent (or 0.1 for free agents)
        faab_bid = pd.to_numeric(transactions_df.get('faab_bid', 0), errors='coerce').fillna(0)
        transactions_df['waiver_cost_norm'] = faab_bid.clip(lower=0.1)
    else:
        # Priority: cost_norm = (N + 1) - priority
        # Top priority (1) = highest cost (N)
        # Bottom priority (N) = lowest cost (1)
        priority = pd.to_numeric(transactions_df.get('waiver_priority', num_teams), errors='coerce').fillna(num_teams)
        transactions_df['waiver_cost_norm'] = (num_teams + 1) - priority
        transactions_df['waiver_cost_norm'] = transactions_df['waiver_cost_norm'].clip(lower=0.1)

    return transactions_df


def calculate_transaction_spar(
    transactions_df: pd.DataFrame,
    weekly_replacement_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate rest-of-season SPAR for transactions with dual player/manager metrics.

    Args:
        transactions_df: Transactions DataFrame with ROS performance
        weekly_replacement_df: Weekly replacement levels

    Returns:
        DataFrame with transaction SPAR columns added:

        Baseline:
        - replacement_ppg_ros: Window-based replacement baseline

        Player SPAR (talent, all weeks):
        - player_spar_ros: ROS SPAR for all weeks played
        - player_ppg_ros: ROS PPG for all weeks
        - player_pgvor_ros: ROS per-game VOR (all weeks)

        Manager SPAR (usage, started weeks only):
        - manager_spar_ros: ROS SPAR for started weeks only
        - manager_ppg_ros: ROS PPG for started weeks
        - manager_pgvor_ros: ROS per-game VOR (started only)

        Legacy (for backward compatibility):
        - fa_spar_ros: Alias for manager_spar_ros
        - fa_ppg_ros: Alias for manager_ppg_ros
        - fa_pgvor_ros: Alias for manager_pgvor_ros
    """
    transactions_df = transactions_df.copy()

    # CRITICAL: Reset index to avoid duplicate label errors in arithmetic operations
    transactions_df = transactions_df.reset_index(drop=True)

    # Ensure numeric types for ALL weeks (player metrics)
    if 'total_points_rest_of_season' in transactions_df.columns:
        transactions_df['total_points_rest_of_season'] = pd.to_numeric(
            transactions_df['total_points_rest_of_season'], errors='coerce'
        ).fillna(0)

    if 'weeks_rest_of_season' in transactions_df.columns:
        transactions_df['weeks_rest_of_season'] = pd.to_numeric(
            transactions_df['weeks_rest_of_season'], errors='coerce'
        ).fillna(0)

    if 'ppg_rest_of_season' in transactions_df.columns:
        transactions_df['ppg_rest_of_season'] = pd.to_numeric(
            transactions_df['ppg_rest_of_season'], errors='coerce'
        ).fillna(0)

    # Ensure numeric types for STARTED weeks (manager metrics)
    if 'total_points_started_ros' in transactions_df.columns:
        transactions_df['total_points_started_ros'] = pd.to_numeric(
            transactions_df['total_points_started_ros'], errors='coerce'
        ).fillna(0)

    if 'weeks_started_ros' in transactions_df.columns:
        transactions_df['weeks_started_ros'] = pd.to_numeric(
            transactions_df['weeks_started_ros'], errors='coerce'
        ).fillna(0)

    if 'ppg_ros_managed' in transactions_df.columns:
        transactions_df['ppg_ros_managed'] = pd.to_numeric(
            transactions_df['ppg_ros_managed'], errors='coerce'
        )
    # Backward compatibility: ppg_started_ros was old name for ppg_ros_managed
    if 'ppg_started_ros' in transactions_df.columns:
        transactions_df['ppg_started_ros'] = pd.to_numeric(
            transactions_df['ppg_started_ros'], errors='coerce'
        ).fillna(0)

    # Calculate window-based replacement for each transaction
    replacement_ppg_list = []

    for _, row in transactions_df.iterrows():
        year = row.get('year')
        position = row.get('position')
        week = row.get('week')

        if pd.isna(year) or pd.isna(position) or pd.isna(week):
            replacement_ppg_list.append(0.0)
            continue

        # Calculate replacement for window: week+1 → 17
        replacement_ppg = calculate_window_replacement(
            weekly_replacement_df,
            year=int(year),
            position=position,
            start_week=int(week) + 1,
            end_week=17
        )

        replacement_ppg_list.append(replacement_ppg)

    transactions_df['replacement_ppg_ros'] = replacement_ppg_list

    # ===== PLAYER SPAR (all weeks the player played) =====
    if 'total_points_rest_of_season' in transactions_df.columns and 'weeks_rest_of_season' in transactions_df.columns:
        replacement_points_ros = transactions_df['replacement_ppg_ros'] * transactions_df['weeks_rest_of_season']
        transactions_df['player_spar_ros'] = transactions_df['total_points_rest_of_season'] - replacement_points_ros
    else:
        transactions_df['player_spar_ros'] = 0.0

    if 'ppg_rest_of_season' in transactions_df.columns:
        transactions_df['player_ppg_ros'] = transactions_df['ppg_rest_of_season'].values
    else:
        transactions_df['player_ppg_ros'] = 0.0

    transactions_df['player_pgvor_ros'] = transactions_df['player_ppg_ros'].values - transactions_df['replacement_ppg_ros'].values

    # ===== MANAGER SPAR (only weeks the manager started the player) =====
    if 'total_points_started_ros' in transactions_df.columns and 'weeks_started_ros' in transactions_df.columns:
        replacement_points_started_ros = transactions_df['replacement_ppg_ros'] * transactions_df['weeks_started_ros']
        transactions_df['manager_spar_ros'] = transactions_df['total_points_started_ros'] - replacement_points_started_ros
    else:
        # Fallback: if no started metrics, assume same as player metrics
        transactions_df['manager_spar_ros'] = transactions_df['player_spar_ros']

    # manager_ppg_ros comes from ppg_ros_managed (or old name ppg_started_ros)
    if 'ppg_ros_managed' in transactions_df.columns:
        transactions_df['manager_ppg_ros'] = transactions_df['ppg_ros_managed'].values
    elif 'ppg_started_ros' in transactions_df.columns:
        transactions_df['manager_ppg_ros'] = transactions_df['ppg_started_ros'].values
    else:
        transactions_df['manager_ppg_ros'] = transactions_df['player_ppg_ros']

    transactions_df['manager_pgvor_ros'] = transactions_df['manager_ppg_ros'].values - transactions_df['replacement_ppg_ros'].values

    # ===== BACKWARD COMPATIBILITY ALIASES =====
    # fa_* columns are aliases for manager_* (conservative default: show realized value)
    transactions_df['fa_spar_ros'] = transactions_df['manager_spar_ros']
    transactions_df['fa_ppg_ros'] = transactions_df['manager_ppg_ros']
    transactions_df['fa_pgvor_ros'] = transactions_df['manager_pgvor_ros']

    return transactions_df


def calculate_transaction_roi(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate transaction ROI (Return on Investment).

    ROI = fa_spar_ros / waiver_cost_norm

    Args:
        transactions_df: Transactions DataFrame with fa_spar_ros and waiver_cost_norm

    Returns:
        DataFrame with fa_roi column added
    """
    transactions_df = transactions_df.copy()

    if 'fa_spar_ros' in transactions_df.columns and 'waiver_cost_norm' in transactions_df.columns:
        # Ensure cost_norm > 0 to avoid division by zero
        cost_norm_safe = transactions_df['waiver_cost_norm'].clip(lower=0.1)
        transactions_df['fa_roi'] = transactions_df['fa_spar_ros'] / cost_norm_safe
    else:
        transactions_df['fa_roi'] = 0.0

    return transactions_df


def calculate_additional_transaction_metrics(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional SPAR-based metrics for transaction analysis.

    Adds:
        - spar_per_faab: manager_spar_ros / faab_bid
        - net_spar_ros: Sum of (adds - drops) grouped by transaction_id + manager
        - spar_efficiency: net_spar_ros / faab_bid
        - position_spar_percentile: Percentile rank within position/year
        - value_vs_avg_pickup: manager_spar_ros - position average
        - spar_per_faab_rank: Rank by SPAR/FAAB within position/year
        - net_spar_rank: Rank by net SPAR within position/year

    Args:
        transactions_df: Transactions DataFrame with manager_spar_ros and waiver_cost_norm

    Returns:
        DataFrame with additional SPAR metrics
    """
    transactions_df = transactions_df.copy()

    # SPAR per FAAB metrics (uses manager SPAR for ROI)
    if 'faab_bid' in transactions_df.columns:
        faab_safe = pd.to_numeric(transactions_df['faab_bid'], errors='coerce').fillna(1).clip(lower=1)
        transactions_df['spar_per_faab'] = transactions_df.get('manager_spar_ros', 0) / faab_safe
    else:
        transactions_df['spar_per_faab'] = 0.0

    # ===== NET SPAR (proper grouping by transaction_id + manager) =====
    # This is CRITICAL for multi-player transactions (add/drop combos, trades)
    # Group by transaction_id + manager to calculate net gain/loss
    if 'transaction_id' in transactions_df.columns and 'manager' in transactions_df.columns:
        # Calculate net SPAR for each transaction_id + manager group
        # Net = sum(manager_spar where type='add' or incoming trade) - sum(manager_spar where type='drop' or outgoing trade)

        def calculate_net_spar_for_group(group):
            """Calculate net SPAR for a transaction_id + manager group."""
            # Use MANAGED SPAR (only weeks on YOUR roster) for net calculation
            spar_col = 'manager_spar_ros_managed' if 'manager_spar_ros_managed' in group.columns else 'manager_spar_ros'

            # Adds/incoming trades: positive SPAR
            adds = group[group['type'].isin(['add', 'trade'])]  # All trades count as adds for now (refined later)
            add_spar = adds[spar_col].sum()

            # Drops: negative SPAR (we're losing this value)
            drops = group[group['type'] == 'drop']
            drop_spar = drops[spar_col].sum()

            # Net = gained - lost
            net_managed = add_spar - drop_spar

            # Also calculate opportunity cost (total ROS regardless of roster)
            spar_total_col = 'player_spar_ros_total' if 'player_spar_ros_total' in group.columns else 'player_spar_ros'
            add_spar_total = adds[spar_total_col].sum()
            drop_spar_total = drops[spar_total_col].sum()
            net_total = add_spar_total - drop_spar_total

            return pd.Series({
                'net_manager_spar_ros': net_managed,  # What you actually got
                'net_player_spar_ros': net_total       # Opportunity cost
            })

        # Group and calculate net SPAR
        net_spar_by_transaction = transactions_df.groupby(['transaction_id', 'manager']).apply(
            calculate_net_spar_for_group
        ).reset_index()

        # Merge back to original dataframe so every row gets the same net SPAR values
        transactions_df = transactions_df.merge(
            net_spar_by_transaction[['transaction_id', 'manager', 'net_manager_spar_ros', 'net_player_spar_ros']],
            on=['transaction_id', 'manager'],
            how='left',
            suffixes=('_old', '')
        )

        # Drop old columns if they exist
        for old_col in ['net_manager_spar_ros_old', 'net_player_spar_ros_old', 'net_spar_ros_old']:
            if old_col in transactions_df.columns:
                transactions_df = transactions_df.drop(columns=[old_col])

        # Fill NaN with individual values (for transactions without transaction_id)
        managed_col = 'manager_spar_ros_managed' if 'manager_spar_ros_managed' in transactions_df.columns else 'manager_spar_ros'
        total_col = 'player_spar_ros_total' if 'player_spar_ros_total' in transactions_df.columns else 'player_spar_ros'

        transactions_df['net_manager_spar_ros'] = transactions_df['net_manager_spar_ros'].fillna(
            transactions_df.get(managed_col, 0)
        )
        transactions_df['net_player_spar_ros'] = transactions_df['net_player_spar_ros'].fillna(
            transactions_df.get(total_col, 0)
        )

        # Legacy column for backward compatibility
        transactions_df['net_spar_ros'] = transactions_df['net_manager_spar_ros']
    else:
        # Fallback: if no transaction_id or manager, just use individual SPAR values
        managed_col = 'manager_spar_ros_managed' if 'manager_spar_ros_managed' in transactions_df.columns else 'manager_spar_ros'
        total_col = 'player_spar_ros_total' if 'player_spar_ros_total' in transactions_df.columns else 'player_spar_ros'

        transactions_df['net_manager_spar_ros'] = transactions_df.get(managed_col, 0)
        transactions_df['net_player_spar_ros'] = transactions_df.get(total_col, 0)
        transactions_df['net_spar_ros'] = transactions_df['net_manager_spar_ros']

    # SPAR efficiency (net MANAGED SPAR per FAAB - actual value received)
    if 'faab_bid' in transactions_df.columns:
        faab_safe = pd.to_numeric(transactions_df['faab_bid'], errors='coerce').fillna(1).clip(lower=1)
        transactions_df['spar_efficiency'] = transactions_df['net_manager_spar_ros'] / faab_safe
    else:
        transactions_df['spar_efficiency'] = 0.0

    # Position-adjusted metrics (using manager_spar_ros for realized value)
    pos_col = 'position' if 'position' in transactions_df.columns else 'yahoo_position'

    if pos_col in transactions_df.columns and 'year' in transactions_df.columns and 'manager_spar_ros' in transactions_df.columns:
        # Position percentile (0-100)
        transactions_df['position_spar_percentile'] = (
            transactions_df.groupby(['year', pos_col])['manager_spar_ros']
            .rank(pct=True) * 100
        )

        # Value vs average pickup at position
        avg_pickup_by_pos = transactions_df.groupby(['year', pos_col])['manager_spar_ros'].transform('mean')
        transactions_df['value_vs_avg_pickup'] = transactions_df['manager_spar_ros'] - avg_pickup_by_pos

        # Rankings within position/year
        transactions_df['spar_per_faab_rank'] = (
            transactions_df.groupby(['year', pos_col])['spar_per_faab']
            .rank(ascending=False, method='min')
        )

        transactions_df['net_spar_rank'] = (
            transactions_df.groupby(['year', pos_col])['net_manager_spar_ros']
            .rank(ascending=False, method='min')
        )
    else:
        transactions_df['position_spar_percentile'] = 0
        transactions_df['value_vs_avg_pickup'] = 0.0
        transactions_df['spar_per_faab_rank'] = 0
        transactions_df['net_spar_rank'] = 0

    return transactions_df


def add_engagement_metrics(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engagement-focused metrics for fun, intuitive UI displays.

    These pre-computed columns eliminate on-the-fly UI calculations and provide
    consistent, user-friendly labels across all views.

    Adds:
        - transaction_grade: A-F grade based on NET SPAR percentile within year/type
        - transaction_result: Human-readable result category ("Elite Win", "Small Loss", etc.)
        - faab_value_tier: FAAB efficiency category ("Steal", "Great Value", "Fair", "Overpay")
        - drop_regret_score: For drops, the SPAR the player produced after being dropped
        - drop_regret_tier: Category for drop regret ("No Regret" → "Disaster")
        - timing_category: When in season the transaction occurred
        - pickup_type: Source of the pickup (Waiver, Free Agent, Trade)

    Args:
        transactions_df: Transactions DataFrame with SPAR metrics already calculated

    Returns:
        DataFrame with engagement metric columns added
    """
    df = transactions_df.copy()

    # ===== 1. TRANSACTION GRADE (A-F based on NET SPAR percentile) =====
    # Grade within year and transaction type for fair comparison
    if 'net_manager_spar_ros' in df.columns and 'year' in df.columns and 'type' in df.columns:
        # Calculate percentile within year/type groups
        df['net_spar_percentile'] = df.groupby(['year', 'type'])['net_manager_spar_ros'].transform(
            lambda x: x.rank(pct=True) * 100
        )

        # Assign letter grades based on percentile
        def assign_grade(percentile):
            if pd.isna(percentile):
                return None
            elif percentile >= 80:
                return 'A'
            elif percentile >= 60:
                return 'B'
            elif percentile >= 40:
                return 'C'
            elif percentile >= 20:
                return 'D'
            else:
                return 'F'

        df['transaction_grade'] = df['net_spar_percentile'].apply(assign_grade)
    else:
        df['net_spar_percentile'] = np.nan
        df['transaction_grade'] = None

    # ===== 2. TRANSACTION RESULT (Human-readable category) =====
    # Use NET SPAR to determine result category
    def get_result_category(row):
        net_spar = row.get('net_manager_spar_ros', 0)
        trans_type = row.get('type', row.get('transaction_type', ''))

        if pd.isna(net_spar):
            return 'Unknown'

        # For individual adds/drops, use the individual SPAR value
        if trans_type == 'add':
            value = row.get('manager_spar_ros_managed', row.get('manager_spar_ros', net_spar))
            if pd.isna(value):
                value = net_spar
            if value > 100:
                return 'Elite Pickup'
            elif value > 50:
                return 'Great Pickup'
            elif value > 20:
                return 'Good Pickup'
            elif value > 0:
                return 'Decent Pickup'
            elif value > -10:
                return 'Neutral'
            else:
                return 'Poor Pickup'
        elif trans_type == 'drop':
            # For drops, use player_spar_ros_total (opportunity cost)
            value = row.get('player_spar_ros_total', row.get('player_spar_ros', 0))
            if pd.isna(value):
                value = 0
            if value > 100:
                return 'Major Regret'
            elif value > 50:
                return 'Big Regret'
            elif value > 20:
                return 'Some Regret'
            elif value > 0:
                return 'Minor Regret'
            else:
                return 'No Regret'
        elif trans_type == 'trade':
            # For trades, use net SPAR
            if net_spar > 100:
                return 'Elite Win'
            elif net_spar > 50:
                return 'Great Win'
            elif net_spar > 20:
                return 'Good Win'
            elif net_spar > 0:
                return 'Small Win'
            elif net_spar == 0:
                return 'Even'
            elif net_spar > -20:
                return 'Small Loss'
            elif net_spar > -50:
                return 'Bad Loss'
            else:
                return 'Major Loss'

        return 'Unknown'

    df['transaction_result'] = df.apply(get_result_category, axis=1)

    # ===== 3. FAAB VALUE TIER =====
    # Categorize FAAB efficiency for adds with FAAB > 0
    if 'spar_efficiency' in df.columns and 'faab_bid' in df.columns:
        def get_faab_tier(row):
            faab = row.get('faab_bid', 0)
            efficiency = row.get('spar_efficiency', 0)
            trans_type = row.get('type', row.get('transaction_type', ''))

            # Only applicable for adds with FAAB spent
            if trans_type != 'add' or pd.isna(faab) or faab <= 0:
                return None

            if pd.isna(efficiency):
                return 'Unknown'
            elif efficiency > 5:
                return 'Steal'
            elif efficiency > 3:
                return 'Great Value'
            elif efficiency > 1:
                return 'Good Value'
            elif efficiency > 0:
                return 'Fair'
            else:
                return 'Overpay'

        df['faab_value_tier'] = df.apply(get_faab_tier, axis=1)
    else:
        df['faab_value_tier'] = None

    # ===== 4. DROP REGRET SCORE & TIER =====
    # For drops only: how much SPAR did the player produce after you dropped them?
    if 'type' in df.columns:
        # Drop regret score = player's total ROS SPAR (what they did after you dropped them)
        df['drop_regret_score'] = np.where(
            df['type'] == 'drop',
            df.get('player_spar_ros_total', df.get('player_spar_ros', 0)),
            np.nan
        )

        # Categorize regret level
        def get_regret_tier(score):
            if pd.isna(score):
                return None
            elif score <= 0:
                return 'No Regret'
            elif score <= 10:
                return 'Minor Regret'
            elif score <= 30:
                return 'Some Regret'
            elif score <= 50:
                return 'Big Regret'
            elif score <= 100:
                return 'Major Regret'
            else:
                return 'Disaster'

        df['drop_regret_tier'] = df['drop_regret_score'].apply(get_regret_tier)
    else:
        df['drop_regret_score'] = np.nan
        df['drop_regret_tier'] = None

    # ===== 5. TIMING CATEGORY =====
    # Categorize when in the season the transaction occurred
    if 'week' in df.columns:
        def get_timing_category(week):
            if pd.isna(week):
                return 'Unknown'
            week = int(week)
            if week <= 4:
                return 'Early Season'
            elif week <= 10:
                return 'Mid Season'
            elif week <= 14:
                return 'Late Season'
            else:
                return 'Playoffs'

        df['timing_category'] = df['week'].apply(get_timing_category)
    else:
        df['timing_category'] = 'Unknown'

    # ===== 6. PICKUP TYPE =====
    # Categorize the source of the pickup
    if 'source_type' in df.columns:
        source_mapping = {
            'waivers': 'Waiver Claim',
            'freeagents': 'Free Agent',
            'free_agents': 'Free Agent',
            'team': 'Trade',
            'trade': 'Trade',
        }
        df['pickup_type'] = df['source_type'].str.lower().map(source_mapping).fillna('Other')
    elif 'type' in df.columns:
        # Fallback: use transaction type
        df['pickup_type'] = df['type'].map({
            'add': 'Pickup',
            'drop': 'Drop',
            'trade': 'Trade'
        }).fillna('Other')
    else:
        df['pickup_type'] = 'Unknown'

    # ===== 7. COMBINED RESULT EMOJI (for quick visual scanning) =====
    def get_result_emoji(row):
        result = row.get('transaction_result', '')
        if 'Elite' in str(result):
            return '🏆'
        elif 'Great' in str(result):
            return '✅'
        elif 'Good' in str(result):
            return '👍'
        elif 'Decent' in str(result) or 'Small Win' in str(result):
            return '➡️'
        elif 'Even' in str(result) or 'Neutral' in str(result) or 'No Regret' in str(result):
            return '➖'
        elif 'Minor' in str(result) or 'Small Loss' in str(result):
            return '📉'
        elif 'Some' in str(result) or 'Bad' in str(result):
            return '😬'
        elif 'Big' in str(result) or 'Major' in str(result) or 'Disaster' in str(result):
            return '❌'
        elif 'Poor' in str(result):
            return '👎'
        return ''

    df['result_emoji'] = df.apply(get_result_emoji, axis=1)

    print(f"  [OK] Added engagement metrics: transaction_grade, transaction_result, faab_value_tier, drop_regret_score/tier, timing_category, pickup_type, result_emoji")

    return df


def calculate_transaction_score(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weighted aggregate transaction score combining multiple factors.

    Uses ROW-LEVEL SPAR (not transaction-level net) so each row gets its own score:
    - Adds: Positive score based on manager_spar_ros_managed (what you got)
    - Drops: Negative score based on drop_regret_score (what you missed)
    - Trades: Score based on manager_spar_ros_managed (what you got from acquired player)

    Formula:
        For ADDS/TRADES:
            score = (row_spar × timing_mult × playoff_mult) + faab_bonus + hold_bonus
        For DROPS:
            score = -(drop_regret × timing_mult × playoff_mult × 0.5)

    Factors:
        - row_spar: ROW-LEVEL value (not transaction net)
        - timing_mult: Earlier in season = more valuable (1.5 at week 1 → 1.0 at final week)
        - playoff_mult: Last 3 weeks get 1.25x bonus
        - faab_bonus: SPAR per dollar spent (capped at 20, adds only)
        - hold_bonus: Commitment bonus based on weeks held (adds only)
        - drop_regret: For drops, the SPAR the player produced elsewhere (negative score)

    Args:
        transactions_df: DataFrame with SPAR metrics already calculated

    Returns:
        DataFrame with transaction_score column added
    """
    df = transactions_df.copy()

    # Get transaction type
    trans_type = df.get('type', df.get('transaction_type', ''))

    # ===== 1. Get max week per year from actual data =====
    max_week_by_year = df.groupby('year')['week'].transform('max')
    df['_max_week'] = max_week_by_year

    # ===== 2. Timing Multiplier: Week 1 = 1.5x, Final week = 1.0x =====
    week_range = (df['_max_week'] - 1).clip(lower=1)
    df['_timing_mult'] = 1 + (0.5 * (df['_max_week'] - df['week']) / week_range)

    # ===== 3. Playoff Multiplier: Last 3 weeks get 1.25x =====
    playoff_start = df['_max_week'] - 3
    df['_playoff_mult'] = np.where(df['week'] >= playoff_start, 1.25, 1.0)

    # ===== 4. ROW-LEVEL SPAR (not transaction-level net!) =====
    # For adds/trades: what you GOT from this specific player
    add_spar_col = 'manager_spar_ros_managed' if 'manager_spar_ros_managed' in df.columns else 'manager_spar_ros'
    if add_spar_col not in df.columns:
        add_spar_col = 'fa_spar_ros'

    row_spar = pd.to_numeric(df.get(add_spar_col, 0), errors='coerce').fillna(0)

    # For drops: use drop_regret_score (what you MISSED)
    drop_regret = pd.to_numeric(df.get('drop_regret_score', 0), errors='coerce').fillna(0)

    # ===== 5. FAAB Efficiency Bonus (adds only) =====
    faab_bid = pd.to_numeric(df.get('faab_bid', 0), errors='coerce').fillna(0)
    faab_efficiency = np.where(
        (faab_bid > 0) & (trans_type != 'drop'),
        row_spar / faab_bid.clip(lower=1),
        0
    )
    df['_faab_bonus'] = np.clip(faab_efficiency * 2, 0, 20)

    # ===== 6. Hold Duration Bonus (adds/trades only) =====
    weeks_col = 'weeks_ros_managed' if 'weeks_ros_managed' in df.columns else 'weeks_started_ros'
    if weeks_col not in df.columns:
        weeks_col = 'weeks_rest_of_season'

    weeks_held = pd.to_numeric(df.get(weeks_col, 0), errors='coerce').fillna(0)
    df['_hold_bonus'] = np.where(
        trans_type != 'drop',
        np.log(weeks_held + 1) * 3,
        0
    )

    # ===== 7. Calculate Final Score by Transaction Type =====
    # ADDS/TRADES: positive score based on what you got
    add_trade_score = (
        (row_spar * df['_timing_mult'] * df['_playoff_mult'])
        + df['_faab_bonus']
        + df['_hold_bonus']
    )

    # DROPS: negative score based on regret (what you missed)
    # Half-weighted since opportunity cost is speculative
    drop_score = -(drop_regret.clip(lower=0) * df['_timing_mult'] * df['_playoff_mult'] * 0.5)

    # Assign score based on type
    df['transaction_score'] = np.where(
        trans_type == 'drop',
        drop_score,
        add_trade_score
    )

    # Round for cleaner display
    df['transaction_score'] = df['transaction_score'].round(1)

    # ===== 8. Score Components (for debugging/transparency) =====
    df['score_row_spar'] = row_spar.round(1)
    df['score_drop_regret'] = drop_regret.round(1)
    df['score_faab_bonus'] = df['_faab_bonus'].round(1)
    df['score_hold_bonus'] = df['_hold_bonus'].round(1)
    df['score_timing_mult'] = df['_timing_mult'].round(3)
    df['score_playoff_mult'] = df['_playoff_mult']

    # Clean up temp columns
    df = df.drop(columns=['_max_week', '_timing_mult', '_playoff_mult', '_faab_bonus', '_hold_bonus'])

    # ===== 9. Recalculate transaction_grade based on transaction_score =====
    # Use ABSOLUTE thresholds instead of percentiles for more meaningful grades
    # SEPARATE grading for adds/trades vs drops to give meaningful differentiation:
    #   - For ADDS/TRADES: Score reflects value gained (higher = better)
    #   - For DROPS: Score reflects regret (0 = no regret = good, negative = bad)
    def assign_grade_from_score(row):
        score = row.get('transaction_score', np.nan)
        trans_type = str(row.get('type', row.get('transaction_type', ''))).lower()

        if pd.isna(score):
            return None

        if trans_type == 'drop':
            # For drops: grade based on regret (0 = no regret = good)
            # Drop scores are: 0 (no regret) to negative (regret)
            # B is "standard" for no regret, differentiation comes from bad drops
            if score >= 0:
                return 'B'  # Good: No regret - correctly identified end of value
            elif score >= -3:
                return 'C'  # Average: Minor regret - slight opportunity cost
            elif score >= -10:
                return 'D'  # Below average: Moderate regret
            else:
                return 'F'  # Poor: Disaster drop (dropped a stud)

        else:  # add or trade
            # For adds/trades: grade based on value gained
            # Thresholds based on score distribution (75th~2, 90th~15)
            if score >= 15:
                return 'A'  # Excellent: Top 10% - Elite impact pickup
            elif score >= 5:
                return 'B'  # Good: Solid positive value
            elif score >= 0:
                return 'C'  # Average: Neutral to slightly positive
            elif score >= -3:
                return 'D'  # Below average: Slight negative
            else:
                return 'F'  # Poor: Negative value transaction

    df['transaction_grade'] = df.apply(assign_grade_from_score, axis=1)

    # Update net_spar_percentile to reflect score-based ranking (for backward compatibility)
    df['net_spar_percentile'] = df['transaction_score'].rank(pct=True) * 100

    print(f"  [OK] Added transaction_score (row-level: adds=+SPAR, drops=-regret, timing/playoff multipliers)")
    print(f"  [OK] Recalculated transaction_grade based on transaction_score thresholds (A>=15, B>=5, C>=0, D>=-5, F<-5)")

    return df


def calculate_all_transaction_metrics(
    transactions_df: pd.DataFrame,
    weekly_replacement_df: pd.DataFrame,
    league_settings_path: Path
) -> pd.DataFrame:
    """
    Calculate all SPAR-based transaction metrics.

    Args:
        transactions_df: Transactions DataFrame with ROS performance
        weekly_replacement_df: Weekly replacement levels
        league_settings_path: Path to league settings JSON

    Returns:
        Transactions DataFrame with all SPAR metrics added:
        - replacement_ppg_ros: Window-based replacement baseline
        - fa_spar_ros: Rest-of-season SPAR
        - fa_ppg_ros: Rest-of-season PPG
        - fa_pgvor_ros: Rest-of-season per-game VOR
        - waiver_cost_norm: Normalized waiver cost
        - fa_roi: Return on investment
        - spar_per_faab: SPAR per FAAB dollar
        - net_spar_ros: Net SPAR gained/lost
        - spar_efficiency: SPAR per FAAB spent
        - position_spar_percentile: Percentile within position
        - value_vs_avg_pickup: Value above position average

        Engagement metrics (for UI):
        - transaction_grade: A-F grade based on NET SPAR percentile
        - transaction_result: Human-readable result ("Elite Pickup", "Big Regret", etc.)
        - faab_value_tier: FAAB efficiency tier ("Steal", "Great Value", "Fair", "Overpay")
        - drop_regret_score: For drops, SPAR player produced after being dropped
        - drop_regret_tier: Category for drop regret ("No Regret" → "Disaster")
        - timing_category: Season timing ("Early Season", "Mid Season", etc.)
        - pickup_type: Source type ("Waiver Claim", "Free Agent", "Trade")
        - result_emoji: Quick visual indicator emoji
    """
    # Step 1: Calculate SPAR (or use existing from player table aggregation)
    # Check if SPAR columns already exist from player_to_transactions_v2.py aggregation
    has_player_spar = 'player_spar_ros' in transactions_df.columns and 'manager_spar_ros' in transactions_df.columns
    has_replacement_ppg = 'replacement_ppg_ros' in transactions_df.columns

    if has_player_spar and has_replacement_ppg:
        print("  [SKIP] ROS SPAR columns already exist from player table aggregation - using existing values")

        # Create player_ppg_ros and manager_ppg_ros if they don't exist
        if 'player_ppg_ros' not in transactions_df.columns:
            if 'ppg_ros_total' in transactions_df.columns:
                transactions_df['player_ppg_ros'] = transactions_df['ppg_ros_total']
            elif 'ppg_rest_of_season' in transactions_df.columns:
                transactions_df['player_ppg_ros'] = transactions_df['ppg_rest_of_season']
            else:
                transactions_df['player_ppg_ros'] = 0.0

        if 'manager_ppg_ros' not in transactions_df.columns:
            if 'ppg_ros_managed' in transactions_df.columns:
                transactions_df['manager_ppg_ros'] = transactions_df['ppg_ros_managed']
            elif 'ppg_started_ros' in transactions_df.columns:
                transactions_df['manager_ppg_ros'] = transactions_df['ppg_started_ros']
            else:
                transactions_df['manager_ppg_ros'] = transactions_df['player_ppg_ros']

        # Create pgvor columns if they don't exist
        if 'player_pgvor_ros' not in transactions_df.columns:
            transactions_df['player_pgvor_ros'] = transactions_df['player_ppg_ros'] - transactions_df['replacement_ppg_ros']

        if 'manager_pgvor_ros' not in transactions_df.columns:
            transactions_df['manager_pgvor_ros'] = transactions_df['manager_ppg_ros'] - transactions_df['replacement_ppg_ros']

        # Ensure backward compatibility aliases exist
        if 'fa_spar_ros' not in transactions_df.columns:
            transactions_df['fa_spar_ros'] = transactions_df['manager_spar_ros']
        if 'fa_ppg_ros' not in transactions_df.columns:
            transactions_df['fa_ppg_ros'] = transactions_df['manager_ppg_ros']
        if 'fa_pgvor_ros' not in transactions_df.columns:
            transactions_df['fa_pgvor_ros'] = transactions_df['manager_pgvor_ros']
    else:
        print("  [CALC] Calculating ROS SPAR from transaction data (player table aggregation didn't provide SPAR)")
        transactions_df = calculate_transaction_spar(transactions_df, weekly_replacement_df)

    # Step 2: Normalize waiver cost
    transactions_df = normalize_waiver_cost(transactions_df, league_settings_path)

    # Step 3: Calculate ROI
    transactions_df = calculate_transaction_roi(transactions_df)

    # Step 4: Calculate additional SPAR metrics
    transactions_df = calculate_additional_transaction_metrics(transactions_df)

    # Step 5: Add engagement metrics (grades, tiers, regret scores)
    transactions_df = add_engagement_metrics(transactions_df)

    # Step 6: Calculate weighted aggregate transaction score
    transactions_df = calculate_transaction_score(transactions_df)

    return transactions_df
