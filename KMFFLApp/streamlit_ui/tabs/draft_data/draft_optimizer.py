#!/usr/bin/env python3
"""
Enhanced Draft Optimizer with better UX and preset templates

Improvements:
- Auto-detect roster config from team's lineup positions
- Display points prominently (not just SPAR)
- Strategy presets (Zero RB, Hero RB, etc.)
- Better FLEX handling
- Data-driven bench value calculation (replaces hardcoded discounts)
"""
import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD
from scipy import stats as scipy_stats
from md.data_access import run_query, T, detect_roster_structure
from typing import Dict, Optional


@st.cache_data(ttl=3600, show_spinner=False)
def detect_roster_config() -> dict:
    """
    Auto-detect roster configuration from the player table's lineup_position column.
    Looks at a single manager's roster from the most recent week to determine positions.

    Returns:
        Dict with position counts (qb, rb, wr, te, flex, def, k, bench) and budget
    """
    try:
        # Query ONE manager's lineup positions to get the roster structure
        # This ensures we count positions correctly (not multiplied by # of managers)
        sql = f"""
            SELECT DISTINCT lineup_position
            FROM {T['player']}
            WHERE year = (SELECT MAX(year) FROM {T['player']})
              AND week = 1
              AND lineup_position IS NOT NULL
              AND lineup_position != ''
              AND manager = (
                  SELECT manager FROM {T['player']}
                  WHERE year = (SELECT MAX(year) FROM {T['player']})
                    AND week = 1
                  LIMIT 1
              )
            ORDER BY lineup_position
        """
        df = run_query(sql)

        if df is None or df.empty:
            return None

        # Count positions - lineup_position is like "QB1", "RB1", "BN1", "BN2", etc.
        config = {"qb": 0, "rb": 0, "wr": 0, "te": 0, "flex": 0, "def": 0, "k": 0, "bench": 0, "budget": 200}

        for _, row in df.iterrows():
            pos = str(row['lineup_position']).upper().strip()
            if pos.startswith('BN') or pos.startswith('BENCH'):
                config['bench'] += 1
            elif pos.startswith('QB'):
                config['qb'] += 1
            elif pos.startswith('RB') and not pos.startswith('RB/'):
                config['rb'] += 1
            elif pos.startswith('WR') and not pos.startswith('WR/'):
                config['wr'] += 1
            elif pos.startswith('TE') and not pos.startswith('TE/'):
                config['te'] += 1
            elif pos in ('W/R/T', 'W/R/T1', 'W/R/T2', 'FLEX', 'FLEX1', 'FLEX2', 'RB/WR', 'WR/RB', 'RB/WR/TE', 'WR/RB/TE'):
                config['flex'] += 1
            elif pos.startswith('W/R/T') or 'FLEX' in pos:
                config['flex'] += 1
            elif pos.startswith('DEF') or pos == 'D' or pos.startswith('D/ST') or pos == 'DST':
                config['def'] += 1
            elif pos.startswith('K') and len(pos) <= 2:  # K or K1, not "KEEPER"
                config['k'] += 1
            elif pos.startswith('IR') or pos.startswith('IL'):
                pass  # Skip IR slots

        # Validate we got reasonable values
        total_starters = sum(config[k] for k in ['qb', 'rb', 'wr', 'te', 'flex', 'def', 'k'])
        if total_starters < 5:
            return None

        return config

    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_optimal_budget_allocation() -> dict:
    """
    Analyze historical drafts to find optimal starter/bench budget split.

    FULLY DATA-DRIVEN - no hardcoded roster assumptions.
    Uses detect_roster_structure() from data_access for roster detection.

    Returns:
        Dict with:
        - optimal_starter_pct: Best % to spend on starters (from regression analysis)
        - optimal_bench_pct: Best % to spend on bench (from regression analysis)
        - starter_position_allocation: Optimal spend % per starter position
        - bench_position_allocation: Optimal spend % per bench position
        - detected_starter_count: How many starters detected in this league
        - detected_bench_count: How many bench slots detected
        - position_counts: Detailed position breakdown
        - all derived from actual league history
    """
    try:
        # Use centralized roster structure detection from data_access
        roster_structure = detect_roster_structure()
        detected_starters = roster_structure['starter_count'] if roster_structure else None
        detected_bench = roster_structure['bench_count'] if roster_structure else None
        position_counts = roster_structure['position_counts'] if roster_structure else {}

        # Query ALL draft data with manager-level aggregation
        # Note: is_starter doesn't exist in draft table - we'll infer it from roster structure
        sql = f"""
            SELECT
                manager,
                year,
                yahoo_position,
                cost,
                manager_spar,
                manager_draft_score,
                manager_total_spar
            FROM {T['draft']}
            WHERE cost > 0
              AND manager IS NOT NULL
              AND yahoo_position IN ('QB', 'RB', 'WR', 'TE', 'DEF', 'K')
        """
        df = run_query(sql)

        if df is None or df.empty:
            return None

        # Convert to numeric
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df['manager_spar'] = pd.to_numeric(df['manager_spar'], errors='coerce').fillna(0)
        df['manager_total_spar'] = pd.to_numeric(df['manager_total_spar'], errors='coerce')

        # Determine starter vs bench - FULLY DATA-DRIVEN
        # Use detected starter count from league's lineup_position data
        if detected_starters:
            num_starters = detected_starters
        else:
            # Infer from data: find natural break in cost distribution per manager-year
            # Calculate median roster size and use cost clustering
            roster_sizes = df.groupby(['manager', 'year']).size()
            median_roster = int(roster_sizes.median())

            # Use cost distribution to find the natural break point
            # Starters typically cost significantly more than bench
            cost_by_rank = df.groupby(['manager', 'year']).apply(
                lambda g: g.nlargest(median_roster, 'cost')['cost'].values
            )
            # Find where the biggest cost drop occurs (likely starter/bench boundary)
            if len(cost_by_rank) > 0:
                avg_costs = np.zeros(median_roster)
                count = 0
                for costs in cost_by_rank:
                    if len(costs) >= median_roster:
                        avg_costs += costs[:median_roster]
                        count += 1
                if count > 0:
                    avg_costs /= count
                    # Find largest relative drop
                    drops = []
                    for i in range(1, len(avg_costs)):
                        if avg_costs[i] > 0:
                            drop = (avg_costs[i-1] - avg_costs[i]) / avg_costs[i]
                            drops.append((i, drop))
                    if drops:
                        # The position with biggest drop is likely the starter/bench boundary
                        best_break = max(drops, key=lambda x: x[1])
                        num_starters = best_break[0]
                    else:
                        num_starters = int(median_roster * 0.6)  # Fallback: 60% are starters
                else:
                    num_starters = int(median_roster * 0.6)
            else:
                num_starters = int(median_roster * 0.6)

        # Mark starters based on detected/inferred count
        def mark_starters(group, n_starters):
            group = group.sort_values('cost', ascending=False).copy()
            group['is_starter'] = 0
            actual_starters = min(n_starters, len(group))
            group.iloc[:actual_starters, group.columns.get_loc('is_starter')] = 1
            return group

        df = df.groupby(['manager', 'year'], group_keys=False).apply(
            lambda g: mark_starters(g, num_starters)
        )

        # Calculate per-manager-year totals
        manager_year = df.groupby(['manager', 'year']).agg({
            'cost': 'sum',
            'manager_spar': 'sum',
            'manager_total_spar': 'first'
        }).reset_index()
        manager_year.columns = ['manager', 'year', 'total_cost', 'total_spar', 'manager_total_spar']

        # Use manager_total_spar if available, otherwise use calculated total_spar
        manager_year['final_spar'] = manager_year['manager_total_spar'].fillna(manager_year['total_spar'])

        # Calculate starter vs bench spending per manager-year
        starter_spend = df[df['is_starter'] == 1].groupby(['manager', 'year'])['cost'].sum().reset_index()
        starter_spend.columns = ['manager', 'year', 'starter_cost']

        bench_spend = df[df['is_starter'] == 0].groupby(['manager', 'year'])['cost'].sum().reset_index()
        bench_spend.columns = ['manager', 'year', 'bench_cost']

        # Merge all together
        allocation_data = manager_year.merge(starter_spend, on=['manager', 'year'], how='left')
        allocation_data = allocation_data.merge(bench_spend, on=['manager', 'year'], how='left')
        allocation_data['starter_cost'] = allocation_data['starter_cost'].fillna(0)
        allocation_data['bench_cost'] = allocation_data['bench_cost'].fillna(0)

        # Calculate allocation percentages
        allocation_data['starter_pct'] = allocation_data['starter_cost'] / allocation_data['total_cost'] * 100
        allocation_data['bench_pct'] = allocation_data['bench_cost'] / allocation_data['total_cost'] * 100

        # === FIND OPTIMAL ALLOCATION USING BINNED ANALYSIS ===
        # Avoid linear regression which assumes monotonic relationship
        # Instead: bin bench_pct and find which bin has highest mean SPAR

        avg_starter_pct = allocation_data['starter_pct'].mean()
        avg_bench_pct = allocation_data['bench_pct'].mean()

        # Create bins for bench % (5% increments)
        allocation_data['bench_bin'] = pd.cut(
            allocation_data['bench_pct'],
            bins=[0, 5, 10, 15, 20, 25, 30, 100],
            labels=['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30%+']
        )

        # Calculate mean SPAR for each bin (weighted by sample size for stability)
        bin_stats = allocation_data.groupby('bench_bin', observed=True).agg({
            'final_spar': ['mean', 'std', 'count'],
            'bench_pct': 'mean'
        }).reset_index()
        bin_stats.columns = ['bin', 'mean_spar', 'std_spar', 'count', 'avg_bench_pct']

        # Find optimal bin: highest mean SPAR with at least 3 samples
        valid_bins = bin_stats[bin_stats['count'] >= 3]
        if len(valid_bins) > 0:
            best_bin = valid_bins.loc[valid_bins['mean_spar'].idxmax()]
            optimal_bench_pct = best_bin['avg_bench_pct']
        else:
            optimal_bench_pct = avg_bench_pct

        optimal_starter_pct = 100 - optimal_bench_pct

        # Store analysis info
        regression_info = {
            'method': 'binned_analysis',
            'best_bin': str(best_bin['bin']) if len(valid_bins) > 0 else 'N/A',
            'best_bin_spar': float(best_bin['mean_spar']) if len(valid_bins) > 0 else 0,
            'best_bin_count': int(best_bin['count']) if len(valid_bins) > 0 else 0,
            'bin_stats': bin_stats.to_dict('records') if len(bin_stats) > 0 else []
        }

        # === POSITION-LEVEL ALLOCATION (weighted by actual SPAR outcomes) ===
        # Weight by league-wide player SPAR, not manager behavior
        # This finds which positions actually produce the most value per dollar

        # Calculate SPAR per dollar for each position (league-wide)
        pos_efficiency = df.groupby('yahoo_position').agg({
            'manager_spar': 'sum',
            'cost': 'sum'
        }).reset_index()
        pos_efficiency['spar_per_dollar'] = pos_efficiency['manager_spar'] / pos_efficiency['cost']

        # For starters: weight allocation by position efficiency
        starter_df = df[df['is_starter'] == 1]
        starter_pos_stats = starter_df.groupby('yahoo_position').agg({
            'manager_spar': ['sum', 'mean', 'count'],
            'cost': ['sum', 'mean']
        }).reset_index()
        starter_pos_stats.columns = ['position', 'total_spar', 'avg_spar', 'count', 'total_cost', 'avg_cost']
        starter_pos_stats['spar_per_dollar'] = starter_pos_stats['total_spar'] / starter_pos_stats['total_cost']

        # Optimal starter allocation: proportional to SPAR per dollar (positive SPAR positions get more)
        starter_allocation = {}
        starter_correlations = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']:
            pos_row = starter_pos_stats[starter_pos_stats['position'] == pos]
            if len(pos_row) > 0:
                # Use actual average spend as baseline, adjusted by SPAR efficiency
                avg_spend = pos_row['avg_cost'].values[0]
                spar_eff = pos_row['spar_per_dollar'].values[0]
                starter_allocation[pos] = avg_spend
                starter_correlations[pos] = spar_eff
            else:
                starter_allocation[pos] = 0
                starter_correlations[pos] = 0

        # Normalize starter allocation to sum to 100
        total_starter_alloc = sum(starter_allocation.values())
        if total_starter_alloc > 0:
            starter_allocation = {p: v / total_starter_alloc * 100 for p, v in starter_allocation.items()}

        # For bench: same approach
        bench_df = df[df['is_starter'] == 0]
        bench_pos_stats = bench_df.groupby('yahoo_position').agg({
            'manager_spar': ['sum', 'mean', 'count'],
            'cost': ['sum', 'mean']
        }).reset_index()
        bench_pos_stats.columns = ['position', 'total_spar', 'avg_spar', 'count', 'total_cost', 'avg_cost']
        bench_pos_stats['spar_per_dollar'] = bench_pos_stats['total_spar'] / bench_pos_stats['total_cost']

        bench_allocation = {}
        bench_correlations = {}
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_row = bench_pos_stats[bench_pos_stats['position'] == pos]
            if len(pos_row) > 0:
                avg_spend = pos_row['avg_cost'].values[0]
                spar_eff = pos_row['spar_per_dollar'].values[0]
                bench_allocation[pos] = avg_spend
                bench_correlations[pos] = spar_eff
            else:
                bench_allocation[pos] = 0
                bench_correlations[pos] = 0

        # Normalize bench allocation to sum to 100
        total_bench_alloc = sum(bench_allocation.values())
        if total_bench_alloc > 0:
            bench_allocation = {p: v / total_bench_alloc * 100 for p, v in bench_allocation.items()}

        # Use detected_bench from roster_structure if available, otherwise calculate
        roster_sizes = df.groupby(['manager', 'year']).size()
        median_roster = int(roster_sizes.median())
        detected_bench_count = detected_bench if detected_bench else (median_roster - num_starters)

        # Calculate expected SPAR gain from optimal bin vs league average
        league_avg_spar = allocation_data['final_spar'].mean()
        best_bin_spar = regression_info.get('best_bin_spar', league_avg_spar)
        expected_spar_gain = best_bin_spar - league_avg_spar

        return {
            # Optimal allocations (from binned analysis)
            'optimal_starter_pct': optimal_starter_pct,
            'optimal_bench_pct': optimal_bench_pct,
            'avg_starter_pct': avg_starter_pct,
            'avg_bench_pct': avg_bench_pct,

            # Position-level allocations (weighted by SPAR efficiency)
            'starter_position_allocation': starter_allocation,
            'bench_position_allocation': bench_allocation,
            'starter_correlations': starter_correlations,  # Now SPAR per dollar
            'bench_correlations': bench_correlations,  # Now SPAR per dollar

            # Binned analysis results
            'regression': regression_info,
            'expected_spar_gain': expected_spar_gain,

            # Sample info
            'total_manager_years': len(allocation_data),
            'league_avg_spar': league_avg_spar,
            'league_std_spar': allocation_data['final_spar'].std(),

            # League structure (detected from data_access.detect_roster_structure())
            'detected_starter_count': num_starters,
            'detected_bench_count': detected_bench_count,
            'detected_roster_size': median_roster,
            'position_counts': position_counts  # Detailed breakdown from detect_roster_structure()
        }

    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_bench_history() -> dict:
    """
    Analyze league history to determine optimal bench composition.

    Uses Manager SPAR (realized value when STARTED) not Player SPAR (all weeks).
    This matters because bench players only provide value when they actually play.

    Key metrics calculated:
    - Utilization rate: weeks_started / weeks_rostered (how often bench players get used)
    - Hit rate: % of cheap picks with positive Manager SPAR
    - Marginal value: comparing $1-3 vs $4-6 vs $7-10 cost tiers
    - Total value by position: which positions contribute most bench value overall

    Returns:
        Dict with bench analysis including utilization, hit rates, cost tier analysis
    """
    try:
        # Query bench player performance from draft history
        sql = f"""
            SELECT
                yahoo_position,
                cost_bucket,
                cost,
                manager_spar,
                player_spar,
                season_ppg,
                weeks_started,
                weeks_rostered,
                games_played,
                year
            FROM {T['draft']}
            WHERE cost > 0
              AND cost <= 15
              AND manager_spar IS NOT NULL
              AND yahoo_position IN ('QB', 'RB', 'WR', 'TE', 'DEF', 'K')
        """
        df = run_query(sql)

        if df is None or df.empty:
            return None

        # Convert to numeric
        numeric_cols = ['cost', 'manager_spar', 'player_spar', 'season_ppg',
                       'cost_bucket', 'weeks_started', 'weeks_rostered', 'games_played']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create cost tier buckets for marginal value analysis
        df['cost_tier'] = pd.cut(
            df['cost'],
            bins=[0, 3, 6, 10, 15],
            labels=['$1-3', '$4-6', '$7-10', '$11-15'],
            include_lowest=True
        )

        # Calculate utilization rate (how often bench players actually get started)
        df['utilization_rate'] = np.where(
            df['weeks_rostered'] > 0,
            df['weeks_started'] / df['weeks_rostered'] * 100,
            0
        )

        # === POSITION ANALYSIS ===
        position_stats = df.groupby('yahoo_position').agg({
            'manager_spar': ['mean', 'median', 'sum', 'count'],
            'player_spar': 'mean',
            'season_ppg': 'mean',
            'utilization_rate': 'mean'
        }).round(2)
        position_stats.columns = ['avg_spar', 'median_spar', 'total_spar', 'count',
                                  'avg_player_spar', 'avg_ppg', 'avg_utilization']
        position_stats = position_stats.reset_index()

        # Calculate hit rate by position (% with positive Manager SPAR)
        hit_rates = df.groupby('yahoo_position').apply(
            lambda x: (x['manager_spar'] > 0).sum() / len(x) * 100 if len(x) > 0 else 0
        ).round(1)

        # === COST TIER ANALYSIS (Marginal Value) ===
        cost_tier_stats = df.groupby(['yahoo_position', 'cost_tier']).agg({
            'manager_spar': ['mean', 'sum', 'count'],
            'utilization_rate': 'mean'
        }).round(2)
        cost_tier_stats.columns = ['avg_spar', 'total_spar', 'count', 'utilization']
        cost_tier_stats = cost_tier_stats.reset_index()

        # Calculate marginal value (avg SPAR per dollar) by cost tier
        cost_tier_stats['avg_cost'] = cost_tier_stats['cost_tier'].map({
            '$1-3': 2, '$4-6': 5, '$7-10': 8.5, '$11-15': 13
        })
        cost_tier_stats['spar_per_dollar'] = (
            cost_tier_stats['avg_spar'] / cost_tier_stats['avg_cost']
        ).round(3)

        # === OPTIMAL PRICE POINT BY POSITION ===
        # Find the cost tier with best value for each position
        optimal_tiers = {}
        for pos in ['RB', 'WR', 'QB', 'TE']:
            pos_data = cost_tier_stats[cost_tier_stats['yahoo_position'] == pos]
            if not pos_data.empty:
                # Best tier = highest avg SPAR (not per dollar, because we want actual value)
                best_idx = pos_data['avg_spar'].idxmax()
                best_tier = pos_data.loc[best_idx]
                optimal_tiers[pos] = {
                    'tier': best_tier['cost_tier'],
                    'avg_spar': best_tier['avg_spar'],
                    'utilization': best_tier['utilization'],
                    'spar_per_dollar': best_tier['spar_per_dollar']
                }

        # === BUILD RANKINGS ===
        # Rank positions by total value contribution (accounts for sample size)
        pos_value = position_stats.set_index('yahoo_position')['avg_spar'].to_dict()
        pos_total_value = position_stats.set_index('yahoo_position')['total_spar'].to_dict()

        sorted_by_avg = sorted(
            [(p, v) for p, v in pos_value.items() if p in ['RB', 'WR', 'TE', 'QB']],
            key=lambda x: x[1],
            reverse=True
        )

        sorted_by_total = sorted(
            [(p, v) for p, v in pos_total_value.items() if p in ['RB', 'WR', 'TE', 'QB']],
            key=lambda x: x[1],
            reverse=True
        )

        # === RECOMMENDED ALLOCATION ===
        # Weight positions by hit rate AND total value contribution
        bench_positions = ['RB', 'WR', 'QB', 'TE']
        allocation_weights = {}

        for pos in bench_positions:
            hit_rate = hit_rates.get(pos, 50) / 100  # Convert to 0-1
            total_val = pos_total_value.get(pos, 0)
            # Weight = hit_rate^0.5 * log(total_val+1) to balance both factors
            weight = (hit_rate ** 0.5) * np.log1p(abs(total_val) + 1)
            allocation_weights[pos] = max(0.05, weight)  # Minimum 5%

        # Normalize to sum to 1
        total_weight = sum(allocation_weights.values())
        if total_weight > 0:
            allocation_weights = {p: w / total_weight for p, w in allocation_weights.items()}

        return {
            'position_stats': position_stats,
            'cost_tier_stats': cost_tier_stats,
            'hit_rates': hit_rates.to_dict(),
            'pos_value': pos_value,
            'pos_total_value': pos_total_value,
            'sorted_by_avg': sorted_by_avg,
            'sorted_by_total': sorted_by_total,
            'optimal_tiers': optimal_tiers,
            'allocation_weights': allocation_weights,
            'total_samples': len(df),
            'bench_price_insight': get_bench_price_insight(cost_tier_stats, optimal_tiers)
        }

    except Exception:
        return None


def get_bench_price_insight(cost_tier_stats: pd.DataFrame, optimal_tiers: dict) -> str:
    """Generate human-readable insight about optimal bench pricing."""
    insights = []

    for pos in ['RB', 'WR']:
        if pos in optimal_tiers:
            tier_info = optimal_tiers[pos]
            tier = tier_info['tier']
            if tier in ['$4-6', '$7-10']:
                insights.append(f"{pos}: ${tier} players outperform $1-3")
            elif tier == '$1-3':
                insights.append(f"{pos}: Dollar players are fine")

    if insights:
        return "; ".join(insights)
    return "Spend $4-7 per bench spot for best value"


@st.cache_data(ttl=3600, show_spinner=False)
def get_bench_value_by_rank() -> Dict[str, float]:
    """
    Get pre-computed bench values by position draft rank (QB2, RB3, etc.).

    This is the purest data-driven approach:
    - Uses actual historical median SPAR for each draft slot
    - SPAR already incorporates replacement level (waiver wire value)
    - If a slot has median SPAR <= 0, it's not worth drafting (waiver is just as good)

    Returns:
        Dict mapping position_draft_label to bench_value
        Example: {'QB2': 2.1, 'RB3': 14.0, 'TE2': 0.0, ...}
    """
    try:
        sql = f"""
            SELECT
                position_draft_label,
                yahoo_position,
                AVG(bench_value_by_rank) AS bench_value,
                AVG(slot_median_spar) AS median_spar,
                COUNT(*) AS sample_size
            FROM {T['draft']}
            WHERE position_draft_label IS NOT NULL
              AND bench_value_by_rank IS NOT NULL
              AND yahoo_position IN ('QB', 'RB', 'WR', 'TE', 'DEF', 'K')
            GROUP BY position_draft_label, yahoo_position
            ORDER BY yahoo_position, position_draft_label
        """
        df = run_query(sql)

        if df is None or df.empty:
            print("No bench_value_by_rank data found - falling back to discount method")
            return {}

        # Build lookup dict
        bench_values = {}
        for _, row in df.iterrows():
            label = row['position_draft_label']
            value = row['bench_value']
            if pd.notna(value):
                bench_values[label] = float(value)

        print(f"[BENCH VALUE BY RANK] Loaded {len(bench_values)} slot values from source data")
        return bench_values

    except Exception as e:
        print(f"Error loading bench values by rank: {e}")
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def calculate_bench_insurance_metrics() -> Dict:
    """
    Read pre-computed bench insurance metrics from source data.

    The draft pipeline (draft_value_metrics_v3.py) pre-computes:
    - bench_insurance_discount: Position-specific discount factor (0.0-1.0)
    - bench_spar: Pre-computed bench SPAR = max(0, manager_spar) * discount
    - position_failure_rate: Starter failure rates per position
    - position_activation_rate: Bench activation rates per position

    Returns:
        Dict with:
        - position_discounts: Dict mapping position to discount factor (0.0-1.0)
        - failure_rates: DataFrame with starter failure rates by position
        - activation_metrics: DataFrame with bench activation metrics
        - insurance_values: DataFrame with calculated insurance values
        - summary: Summary statistics
    """
    try:
        # Query pre-computed bench insurance data from source
        sql = f"""
            SELECT
                yahoo_position,
                AVG(bench_insurance_discount) AS bench_insurance_discount,
                AVG(position_failure_rate) AS position_failure_rate,
                AVG(position_activation_rate) AS position_activation_rate
            FROM {T['draft']}
            WHERE bench_insurance_discount IS NOT NULL
              AND yahoo_position IN ('QB', 'RB', 'WR', 'TE', 'DEF', 'K')
            GROUP BY yahoo_position
        """
        df = run_query(sql)

        if df is None or df.empty:
            # Fallback: calculate from data if pre-computed columns don't exist
            return _calculate_bench_insurance_metrics_from_data()

        # Convert to numeric
        for col in ['bench_insurance_discount', 'position_failure_rate', 'position_activation_rate']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Build position discounts from pre-computed values
        position_discounts = {}
        insurance_values = []

        for _, row in df.iterrows():
            pos = row['yahoo_position']
            discount = row.get('bench_insurance_discount', 0.01)
            failure_rate = row.get('position_failure_rate', 0)
            activation_rate = row.get('position_activation_rate', 0)

            if pd.isna(discount):
                discount = 0.01

            position_discounts[pos] = round(float(discount), 3)

            insurance_values.append({
                'yahoo_position': pos,
                'failure_rate': failure_rate if not pd.isna(failure_rate) else 0,
                'activation_rate': activation_rate if not pd.isna(activation_rate) else 0,
                'bench_discount_factor': discount,
                'data_source': 'pre-computed'
            })

        insurance_df = pd.DataFrame(insurance_values)

        # Build failure_rates DataFrame for compatibility
        failure_rates = insurance_df[['yahoo_position', 'failure_rate']].copy()
        failure_rates.columns = ['yahoo_position', 'failure_rate']

        # Build activation_metrics DataFrame for compatibility
        activation_metrics = insurance_df[['yahoo_position', 'activation_rate']].copy()
        activation_metrics.columns = ['yahoo_position', 'activation_rate']

        return {
            'position_discounts': position_discounts,
            'failure_rates': failure_rates,
            'activation_metrics': activation_metrics,
            'insurance_values': insurance_df,
            'summary': {
                'positions_calculated': len(position_discounts),
                'avg_failure_rate': failure_rates['failure_rate'].mean() if not failure_rates.empty else None,
                'avg_activation_rate': activation_metrics['activation_rate'].mean() if not activation_metrics.empty else None,
                'data_source': 'pre-computed'
            }
        }

    except Exception as e:
        print(f"Error reading pre-computed bench metrics: {e}")
        # Fallback to calculating from data
        return _calculate_bench_insurance_metrics_from_data()


def _calculate_bench_insurance_metrics_from_data() -> Dict:
    """
    Fallback: Calculate bench insurance metrics from data if pre-computed columns don't exist.
    This ensures backward compatibility with older data that doesn't have the new columns.
    """
    try:
        sql = f"""
            SELECT
                yahoo_position,
                drafted_as_starter,
                manager_spar,
                weeks_started,
                games_played,
                cost
            FROM {T['draft']}
            WHERE cost > 0
              AND manager IS NOT NULL
              AND yahoo_position IN ('QB', 'RB', 'WR', 'TE', 'DEF', 'K')
        """
        df = run_query(sql)

        if df is None or df.empty:
            return _get_default_bench_metrics()

        # Convert to numeric
        for col in ['manager_spar', 'weeks_started', 'games_played', 'cost', 'drafted_as_starter']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['manager_spar'] = df['manager_spar'].fillna(0)
        df['drafted_as_starter'] = df['drafted_as_starter'].fillna(1)
        df['weeks_started'] = df['weeks_started'].fillna(df.get('games_played', 0))

        # Calculate failure rates from starters
        starters = df[df['drafted_as_starter'] == 1].copy()
        if starters.empty:
            return _get_default_bench_metrics()

        starters['is_failure'] = (
            (starters['manager_spar'] < 0) | (starters['weeks_started'] < 8)
        ).astype(int)

        failure_rates = starters.groupby('yahoo_position').agg(
            total_starters=('is_failure', 'count'),
            failure_count=('is_failure', 'sum'),
            avg_spar=('manager_spar', 'mean')
        ).reset_index()

        failure_rates['failure_rate'] = (
            failure_rates['failure_count'] / failure_rates['total_starters'].clip(lower=1)
        ).round(3)

        # Calculate activation rates from backups
        backups = df[df['drafted_as_starter'] == 0].copy()
        if backups.empty:
            backups = df[df['cost'] <= 10].copy()

        if not backups.empty:
            backups['was_activated'] = (backups['weeks_started'] > 0).astype(int)
            activation_metrics = backups.groupby('yahoo_position').agg(
                total_backups=('was_activated', 'count'),
                activated_count=('was_activated', 'sum')
            ).reset_index()
            activation_metrics['activation_rate'] = (
                activation_metrics['activated_count'] / activation_metrics['total_backups'].clip(lower=1)
            ).round(3)
        else:
            activation_metrics = pd.DataFrame()

        # Calculate position discounts
        position_discounts = {}
        league_avg_failure = failure_rates['failure_rate'].mean() if not failure_rates.empty else 0.3
        league_avg_activation = activation_metrics['activation_rate'].mean() if not activation_metrics.empty else 0.3
        league_avg_starter_spar = failure_rates['avg_spar'].mean() if not failure_rates.empty else 30

        for pos in failure_rates['yahoo_position'].unique():
            pos_failure = failure_rates[failure_rates['yahoo_position'] == pos]
            failure_rate = pos_failure['failure_rate'].iloc[0] if not pos_failure.empty else league_avg_failure
            avg_spar = pos_failure['avg_spar'].iloc[0] if not pos_failure.empty else league_avg_starter_spar

            if not activation_metrics.empty:
                pos_act = activation_metrics[activation_metrics['yahoo_position'] == pos]
                activation_rate = pos_act['activation_rate'].iloc[0] if not pos_act.empty else league_avg_activation
            else:
                activation_rate = league_avg_activation

            if avg_spar > 0:
                discount = min(1.0, max(0.01, failure_rate * activation_rate * avg_spar / avg_spar))
            else:
                discount = failure_rate * activation_rate

            position_discounts[pos] = round(max(0.01, discount), 3)

        return {
            'position_discounts': position_discounts,
            'failure_rates': failure_rates,
            'activation_metrics': activation_metrics,
            'insurance_values': pd.DataFrame(),
            'summary': {
                'positions_calculated': len(position_discounts),
                'avg_failure_rate': league_avg_failure,
                'avg_activation_rate': league_avg_activation,
                'data_source': 'calculated_from_data'
            }
        }

    except Exception as e:
        print(f"Error calculating bench metrics from data: {e}")
        return _get_default_bench_metrics()


def _get_default_bench_metrics() -> Dict:
    """
    Return empty bench metrics when data is unavailable.

    Returns empty structure - the optimizer will handle missing data gracefully
    by using equal weighting for all positions until data is available.
    """
    return {
        'position_discounts': {},  # Empty - no hardcoded values
        'failure_rates': pd.DataFrame(),
        'activation_metrics': pd.DataFrame(),
        'insurance_values': pd.DataFrame(),
        'summary': {
            'positions_calculated': 0,
            'avg_failure_rate': None,
            'avg_activation_rate': None,
            'data_source': 'no_data'
        }
    }


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_optimal_roster_construction() -> dict:
    """
    Analyze historical roster construction to find optimal position allocation.

    Looks at FULL rosters (starters + bench) for each manager-year and correlates
    position spending allocation with total Manager SPAR to find winning strategies.

    Returns:
        Dict with:
        - top_allocations: Position spend % for top-performing manager-years
        - avg_allocations: Average position spend % across all manager-years
        - optimal_allocation: Recommended allocation based on top performers
        - position_correlations: How position spend correlates with total SPAR
    """
    try:
        # Query ALL draft data (not just cheap players)
        sql = f"""
            SELECT
                manager,
                year,
                yahoo_position,
                cost,
                manager_spar,
                manager_draft_score
            FROM {T['draft']}
            WHERE cost > 0
              AND manager IS NOT NULL
              AND yahoo_position IN ('QB', 'RB', 'WR', 'TE', 'DEF', 'K')
        """
        df = run_query(sql)

        if df is None or df.empty:
            return None

        # Convert to numeric
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df['manager_spar'] = pd.to_numeric(df['manager_spar'], errors='coerce').fillna(0)
        df['manager_draft_score'] = pd.to_numeric(df['manager_draft_score'], errors='coerce')

        # Calculate totals per manager-year
        manager_year_totals = df.groupby(['manager', 'year']).agg({
            'cost': 'sum',
            'manager_spar': 'sum',
            'manager_draft_score': 'first'
        }).reset_index()
        manager_year_totals.columns = ['manager', 'year', 'total_cost', 'total_spar', 'draft_score']

        # Calculate position spend per manager-year
        position_spend = df.groupby(['manager', 'year', 'yahoo_position'])['cost'].sum().reset_index()
        position_spend = position_spend.pivot(
            index=['manager', 'year'],
            columns='yahoo_position',
            values='cost'
        ).fillna(0).reset_index()

        # Merge with totals
        roster_data = position_spend.merge(manager_year_totals, on=['manager', 'year'])

        # Calculate allocation percentages
        for pos in ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']:
            if pos in roster_data.columns:
                roster_data[f'{pos}_pct'] = roster_data[pos] / roster_data['total_cost'] * 100
            else:
                roster_data[f'{pos}_pct'] = 0

        # Find top performers (top 25% by total SPAR)
        spar_75th = roster_data['total_spar'].quantile(0.75)
        top_performers = roster_data[roster_data['total_spar'] >= spar_75th]

        # Calculate average allocations for top performers
        top_allocations = {}
        avg_allocations = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']:
            col = f'{pos}_pct'
            if col in roster_data.columns:
                top_allocations[pos] = top_performers[col].mean()
                avg_allocations[pos] = roster_data[col].mean()

        # Calculate correlations between position spend % and total SPAR
        correlations = {}
        for pos in ['QB', 'RB', 'WR', 'TE']:
            col = f'{pos}_pct'
            if col in roster_data.columns and roster_data[col].std() > 0:
                corr = roster_data[col].corr(roster_data['total_spar'])
                correlations[pos] = corr if not pd.isna(corr) else 0

        # Calculate optimal allocation (weighted by top performer success)
        optimal_allocation = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']:
            # Weight top performer allocation more heavily
            top_val = top_allocations.get(pos, 0)
            avg_val = avg_allocations.get(pos, 0)
            optimal_allocation[pos] = (top_val * 0.7 + avg_val * 0.3)

        # Normalize to 100%
        total = sum(optimal_allocation.values())
        if total > 0:
            optimal_allocation = {p: v / total * 100 for p, v in optimal_allocation.items()}

        # Calculate position spend stats for display
        position_stats = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']:
            if pos in roster_data.columns:
                position_stats[pos] = {
                    'avg_spend': roster_data[pos].mean(),
                    'top_spend': top_performers[pos].mean() if not top_performers.empty else 0,
                    'min_spend': roster_data[pos].min(),
                    'max_spend': roster_data[pos].max()
                }

        return {
            'top_allocations': top_allocations,
            'avg_allocations': avg_allocations,
            'optimal_allocation': optimal_allocation,
            'correlations': correlations,
            'position_stats': position_stats,
            'top_performers_count': len(top_performers),
            'total_manager_years': len(roster_data),
            'avg_total_spar': roster_data['total_spar'].mean(),
            'top_avg_spar': top_performers['total_spar'].mean() if not top_performers.empty else 0
        }

    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def preprocess_optimizer_data(
        draft_history: pd.DataFrame,
        start_year: int,
        end_year: int,
        min_sample_size: int = 2
) -> pd.DataFrame:
    """
    Optimized preprocessing with better aggregation.

    Uses position-specific percentile-based tiers:
    - Groups by position + position_tier (from backend)
    - Tier count is fully dynamic per position-year (based on sample size)
    - Uses median PPG (robust to outliers like injury busts)
    - Excludes keepers (artificial prices skew tiers)
    """
    df = draft_history.copy()

    # Efficient conversions
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0)
    df["points"] = pd.to_numeric(df.get("points", 0), errors="coerce").fillna(0)
    df["season_ppg"] = pd.to_numeric(df.get("season_ppg", 0), errors="coerce").fillna(0)

    # Use position_tier if available (new format), fallback to cost_bucket
    # Tiers are fully dynamic - backend determines count per position-year
    if "position_tier" in df.columns:
        df["tier"] = pd.to_numeric(df["position_tier"], errors="coerce").fillna(0).astype(int)
        # Use backend-provided labels, fallback to generic "Tier X" if missing
        if "position_tier_label" in df.columns:
            df["tier_label"] = df["position_tier_label"].fillna("")
            # Fill any blanks with generic tier label
            df.loc[df["tier_label"] == "", "tier_label"] = df.loc[df["tier_label"] == "", "tier"].apply(lambda x: f"Tier {x}")
        else:
            df["tier_label"] = df["tier"].apply(lambda x: f"Tier {x}")
    elif "cost_bucket" in df.columns:
        df["tier"] = pd.to_numeric(df["cost_bucket"], errors="coerce").fillna(0).astype(int)
        # Dynamic fallback - just use "Tier X" format (no hardcoded tier count)
        df["tier_label"] = df["tier"].apply(lambda x: f"Tier {x}")
    else:
        df["tier"] = 1
        df["tier_label"] = "Tier 1"

    # Get position percentile if available
    if "position_percentile" in df.columns:
        df["position_percentile"] = pd.to_numeric(df["position_percentile"], errors="coerce").fillna(50)
    else:
        df["position_percentile"] = 50

    # Use manager_spar (actual value while rostered) with fallback to spar
    if "manager_spar" in df.columns:
        df["spar"] = pd.to_numeric(df["manager_spar"], errors="coerce").fillna(0)
    else:
        df["spar"] = pd.to_numeric(df.get("spar", 0), errors="coerce").fillna(0)

    # Get drafted_as_starter flag if available (from starter_designation_calculator)
    # This distinguishes picks that were QB1 (starter) vs QB2 (backup) within a manager's draft
    if "drafted_as_starter" in df.columns:
        df["drafted_as_starter"] = pd.to_numeric(df["drafted_as_starter"], errors="coerce").fillna(1)
    else:
        df["drafted_as_starter"] = 1  # Default to starter if column doesn't exist

    # Single comprehensive filter
    keeper_status = df.get("is_keeper_status", pd.Series([0] * len(df)))
    mask = (
            (df["year"] >= start_year) &
            (df["year"] <= end_year) &
            (df["cost"] > 0) &
            (keeper_status != 1) &
            (df["season_ppg"] > 0) &
            (df["tier"] > 0)
    )
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame()

    # Aggregate with sample size tracking - grouped by position + tier + drafted_as_starter
    # This separates SPAR for "Tier 3 QB drafted as starter" vs "Tier 3 QB drafted as backup"
    agg_dict = {
        "cost": ["mean", "std", "count"],
        "season_ppg": ["median", "mean", "std"],
        "points": "median",
        "spar": ["median", "mean", "std"],
        "tier_label": "first",  # Capture the label for display
        "position_percentile": ["mean", "min", "max"]  # Track percentile range
    }

    agg = df.groupby(
        ["yahoo_position", "tier", "drafted_as_starter"],
        dropna=False,
        as_index=False
    ).agg(agg_dict)

    # Flatten columns
    agg.columns = ['yahoo_position', 'position_tier', 'drafted_as_starter', 'avg_cost', 'cost_std', 'sample_size',
                   'median_ppg', 'mean_ppg', 'ppg_std', 'median_points',
                   'median_spar', 'mean_spar', 'spar_std', 'tier_label',
                   'avg_percentile', 'min_percentile', 'max_percentile']

    # Filter by sample size
    agg = agg[agg['sample_size'] >= min_sample_size].copy()

    # Round and clean
    agg['avg_cost'] = agg['avg_cost'].round(2)
    agg['median_ppg'] = agg['median_ppg'].round(2)
    agg['position_tier'] = agg['position_tier'].astype(int)
    agg['avg_percentile'] = agg['avg_percentile'].round(0)

    # Legacy column for backwards compatibility
    agg['cost_bucket'] = agg['position_tier']
    agg['bucket_label'] = agg['tier_label']

    # Sort for better display (Elite first = tier 1)
    agg = agg.sort_values(["yahoo_position", "position_tier", "drafted_as_starter"], ascending=[True, True, False])

    return agg


def get_optimizer_data(agg_data: pd.DataFrame, for_starters: bool = True) -> pd.DataFrame:
    """
    Filter aggregated data for optimizer use.

    Args:
        agg_data: Full aggregated data with drafted_as_starter column
        for_starters: If True, return data for starter picks (high SPAR)
                      If False, return data for backup picks (lower SPAR)

    Returns:
        Filtered DataFrame with one row per position-tier combination
    """
    if 'drafted_as_starter' not in agg_data.columns:
        # No starter designation data - return as-is
        return agg_data

    target_designation = 1 if for_starters else 0

    # Filter to the requested designation
    filtered = agg_data[agg_data['drafted_as_starter'] == target_designation].copy()

    # If we don't have enough data for backups, fall back to starters with discounted SPAR
    if not for_starters and filtered.empty:
        filtered = agg_data[agg_data['drafted_as_starter'] == 1].copy()
        # Apply a discount since backups typically provide less value
        filtered['median_spar'] = filtered['median_spar'] * 0.2  # 80% discount for bench
        filtered['mean_spar'] = filtered['mean_spar'] * 0.2

    return filtered


def prepare_dual_role_data(agg_data: pd.DataFrame, bench_metrics: Optional[Dict] = None) -> pd.DataFrame:
    """
    Prepare data with separate rows for starter vs bench usage.

    Uses rank-based bench values (median SPAR by position_draft_label) which is
    the purest data-driven approach:
    - QB2 bench value = historical median SPAR for QB2 picks (2.1 SPAR)
    - RB3 bench value = historical median SPAR for RB3 picks (14.0 SPAR)
    - TE2 bench value = 0 (historical median is negative, clipped to 0)

    Args:
        agg_data: Aggregated player data with SPAR metrics
        bench_metrics: Dict from calculate_bench_insurance_metrics() (legacy, kept for compatibility)

    Returns:
        DataFrame with separate starter and bench rows
    """
    # Get rank-based bench values (the purest data-driven approach)
    bench_value_by_rank = get_bench_value_by_rank()

    # Create starter rows
    starter_data = agg_data.copy()
    starter_data['slot_type'] = 'starter'

    # Create bench rows with rank-based SPAR values
    backup_data = agg_data.copy()
    backup_data['slot_type'] = 'bench'

    if bench_value_by_rank:
        # Use rank-based bench values (data-driven)
        backup_data['discount_source'] = 'rank-based'

        # For each position, set bench SPAR based on what backup slots are actually worth
        # Example: QB bench = QB2 value (2.1), RB bench = avg of RB3-RB6 (~30), etc.
        for pos in backup_data['yahoo_position'].unique():
            mask = backup_data['yahoo_position'] == pos

            # Find all backup slot values for this position (e.g., QB2, QB3 for QB)
            backup_slots = [k for k in bench_value_by_rank.keys()
                          if k.startswith(pos) and k != f'{pos}1']

            if backup_slots:
                # Use the average of backup slot values
                # This represents what a typical backup at this position is worth
                avg_bench_value = sum(bench_value_by_rank.get(s, 0) for s in backup_slots) / len(backup_slots)

                if 'median_spar' in backup_data.columns:
                    backup_data.loc[mask, 'median_spar'] = max(0, avg_bench_value)
                if 'mean_spar' in backup_data.columns:
                    backup_data.loc[mask, 'mean_spar'] = max(0, avg_bench_value)

                backup_data.loc[mask, 'bench_value_source'] = f"avg({','.join(backup_slots)})"
            else:
                # No backup data for this position - set to 0
                if 'median_spar' in backup_data.columns:
                    backup_data.loc[mask, 'median_spar'] = 0
                if 'mean_spar' in backup_data.columns:
                    backup_data.loc[mask, 'mean_spar'] = 0
                backup_data.loc[mask, 'bench_value_source'] = 'no_data'
    else:
        # Fallback to discount-based approach if no rank data available
        backup_data['discount_source'] = 'discount-fallback'

        # Get discount map from bench_metrics or use defaults
        if bench_metrics and bench_metrics.get('position_discounts'):
            discount_map = bench_metrics['position_discounts']
        else:
            # No data - use conservative 10% discount for all positions
            unique_positions = agg_data['yahoo_position'].unique()
            discount_map = {pos: 0.1 for pos in unique_positions}

        for pos in backup_data['yahoo_position'].unique():
            mask = backup_data['yahoo_position'] == pos
            discount = discount_map.get(pos, 0.1)

            if 'median_spar' in backup_data.columns:
                backup_data.loc[mask, 'median_spar'] = (
                    backup_data.loc[mask, 'median_spar'].clip(lower=0) * discount
                )
            if 'mean_spar' in backup_data.columns:
                backup_data.loc[mask, 'mean_spar'] = (
                    backup_data.loc[mask, 'mean_spar'].clip(lower=0) * discount
                )

        backup_data.loc[mask, 'bench_discount_applied'] = discount

    return pd.concat([starter_data, backup_data], ignore_index=True)


def get_roster_preset(preset_name: str, detected_config: dict = None) -> dict:
    """Return roster configuration presets"""
    presets = {
        "Standard": {
            "qb": 1, "rb": 2, "wr": 3, "te": 1,
            "flex": 1, "def": 1, "k": 1, "budget": 200
        },
        "2 QB": {
            "qb": 2, "rb": 2, "wr": 3, "te": 1,
            "flex": 1, "def": 1, "k": 1, "budget": 200
        },
        "PPR Flex Heavy": {
            "qb": 1, "rb": 2, "wr": 3, "te": 1,
            "flex": 2, "def": 1, "k": 1, "budget": 200
        },
        "Best Ball": {
            "qb": 2, "rb": 4, "wr": 4, "te": 2,
            "flex": 0, "def": 0, "k": 0, "budget": 200
        },
        "Custom": {
            "qb": 1, "rb": 2, "wr": 2, "te": 1,
            "flex": 1, "def": 1, "k": 1, "budget": 200
        }
    }

    # Use auto-detected config for "Your League"
    if preset_name == "Your League" and detected_config:
        return detected_config

    return presets.get(preset_name, presets["Standard"])


def get_strategy_preset(strategy_name: str, base_config: dict) -> dict:
    """
    Apply strategy modifiers to position spending caps.

    Strategy presets modify the spending caps for different draft strategies.
    They don't change roster requirements, just the max spend per position slot.

    Returns:
        Dict with position cap arrays, e.g., {"rb": [60, 15, 10], "wr": [40, 35, 30]}
    """
    strategies = {
        "Balanced": {
            # No specific caps - let optimizer decide
            "description": "No spending restrictions, optimizer decides allocation"
        },
        "Zero RB": {
            # Cheap RBs, expensive WRs
            "rb": [20, 15, 10, 5, 5],  # Max spend per RB slot
            "wr": [70, 55, 40, 30, 20],  # Higher WR caps
            "description": "Cheap RBs ($20 or less), invest heavily in WRs"
        },
        "Hero RB": {
            # One elite RB, cheap rest
            "rb": [65, 15, 10, 5, 5],  # RB1 can be elite, rest cheap
            "wr": [50, 40, 35, 25, 20],
            "description": "One elite RB ($60+), cheap backup RBs"
        },
        "Robust RB": {
            # Two solid RBs
            "rb": [55, 45, 15, 10, 5],  # Two strong RBs
            "wr": [45, 40, 30, 25, 20],
            "description": "Two solid RBs ($45-55 each)"
        },
        "Late-Round QB": {
            # Cheap QB
            "qb": [12, 8, 5],
            "description": "Cheap QB ($12 or less), invest in skill positions"
        },
        "Stars & Scrubs": {
            # Few expensive players, rest cheap
            "rb": [70, 10, 5, 3, 1],
            "wr": [65, 10, 5, 3, 1],
            "description": "2-3 elite players, fill rest with $5 or less"
        }
    }

    return strategies.get(strategy_name, strategies["Balanced"])


def get_bench_strategy(strategy_name: str, bench_spots: int, budget: int,
                       bench_analysis: dict = None, budget_allocation: dict = None) -> dict:
    """
    Get bench strategy configuration - now fully data-driven.

    Args:
        strategy_name: Name of the bench strategy
        bench_spots: Number of bench spots
        budget: Total draft budget
        bench_analysis: Historical bench analysis (position hit rates, cost tiers)
        budget_allocation: Optimal budget allocation data (starter/bench split)

    Returns:
        Dict with bench strategy details including:
        - budget: Total bench budget (derived from data)
        - starter_budget: Budget for starters
        - composition: Position allocation weights (from data)
        - suggested_spots: Recommended spots per position
        - insight: Data-driven recommendation text
    """
    strategy = {
        'spots': bench_spots,
        'composition': None,
        'description': '',
        'insight': ''
    }

    # === PURE DATA-DRIVEN (default) ===
    if strategy_name == "Data-Driven" and budget_allocation:
        # Use actual optimal percentages from top performers
        optimal_bench_pct = budget_allocation.get('optimal_bench_pct', 12)
        optimal_starter_pct = budget_allocation.get('optimal_starter_pct', 88)

        strategy['budget_pct'] = optimal_bench_pct
        strategy['starter_pct'] = optimal_starter_pct
        strategy['budget'] = max(bench_spots, int(budget * optimal_bench_pct / 100))
        strategy['starter_budget'] = budget - strategy['budget']

        # Use data-derived position allocation for bench
        bench_pos_alloc = budget_allocation.get('bench_position_allocation', {})
        if bench_pos_alloc:
            # Convert percentages to 0-1 weights
            total = sum(bench_pos_alloc.values())
            if total > 0:
                strategy['composition'] = {
                    pos: pct / total
                    for pos, pct in bench_pos_alloc.items()
                }

        # Build description from regression analysis
        regression = budget_allocation.get('regression', {})
        expected_gain = budget_allocation.get('expected_spar_gain', 0)

        if regression.get('is_significant'):
            direction = regression.get('direction', '')
            strategy['description'] = f"Statistical analysis: {direction}"
        else:
            strategy['description'] = f"No significant pattern found - using {optimal_bench_pct:.0f}% bench"

        # Build insights from regression
        insights = []
        bench_corr = budget_allocation.get('bench_spar_correlation', 0)
        if abs(bench_corr) > 0.1:
            if bench_corr > 0:
                insights.append(f"More bench spending → better results (r={bench_corr:+.2f})")
            else:
                insights.append(f"Less bench spending → better results (r={bench_corr:+.2f})")

        if abs(expected_gain) > 1:
            insights.append(f"Expected gain: {expected_gain:+.1f} SPAR vs league average")

        strategy['insight'] = " | ".join(insights) if insights else "Optimal allocation from regression analysis"

    # === MANUAL OVERRIDE STRATEGIES ===
    elif strategy_name == "Max Starters":
        # 90% starters, 10% bench - maximize starter investment
        bench_pct = 10
        strategy['budget_pct'] = bench_pct
        strategy['budget'] = max(bench_spots, int(budget * bench_pct / 100))
        strategy['starter_budget'] = budget - strategy['budget']
        strategy['description'] = f"90% starters (${strategy['starter_budget']}), 10% bench (${strategy['budget']})"
        strategy['composition'] = {"RB": 0.35, "WR": 0.35, "QB": 0.15, "TE": 0.15}
        strategy['insight'] = "Maximize starter quality with minimal bench investment"

    elif strategy_name == "Balanced":
        # 70% starters, 30% bench - balanced approach
        bench_pct = 30
        strategy['budget_pct'] = bench_pct
        strategy['budget'] = max(bench_spots, int(budget * bench_pct / 100))
        strategy['starter_budget'] = budget - strategy['budget']
        strategy['description'] = f"70% starters (${strategy['starter_budget']}), 30% bench (${strategy['budget']})"
        strategy['composition'] = {"RB": 0.35, "WR": 0.35, "QB": 0.15, "TE": 0.15}
        strategy['insight'] = "Balanced investment in starters and depth"

    elif strategy_name == "Custom":
        # Let user specify - default to 15%
        strategy['budget_pct'] = 15
        strategy['budget'] = max(bench_spots, int(budget * 0.15))
        strategy['starter_budget'] = budget - strategy['budget']
        strategy['description'] = "You choose the allocation"
        strategy['composition'] = {"RB": 0.35, "WR": 0.35, "QB": 0.15, "TE": 0.15}
        strategy['insight'] = "Adjust the sliders to set your own allocation"

    else:
        # Fallback - use data if available, otherwise reasonable defaults
        if budget_allocation:
            optimal_bench_pct = budget_allocation.get('optimal_bench_pct', 12)
        else:
            optimal_bench_pct = 12

        strategy['budget_pct'] = optimal_bench_pct
        strategy['budget'] = max(bench_spots, int(budget * optimal_bench_pct / 100))
        strategy['starter_budget'] = budget - strategy['budget']
        strategy['description'] = f"Suggested: {optimal_bench_pct:.0f}% on bench"
        strategy['composition'] = {"RB": 0.35, "WR": 0.35, "QB": 0.15, "TE": 0.15}

    # === FINAL BUDGET CALCULATION ===
    # Always use percentages of total budget, floor bench (extra goes to starters)
    bench_pct = strategy.get('budget_pct', 10)
    bench_budget_raw = budget * bench_pct / 100
    strategy['budget'] = max(bench_spots, int(bench_budget_raw))  # Floor bench
    strategy['starter_budget'] = budget - strategy['budget']  # Starters get remainder (ALWAYS sums to budget)
    strategy['avg_per_spot'] = round(strategy['budget'] / bench_spots, 1) if bench_spots > 0 else 0

    # Sanity check - must sum to total budget
    assert strategy['budget'] + strategy['starter_budget'] == budget, f"Budget mismatch: {strategy['budget']} + {strategy['starter_budget']} != {budget}"

    # Add optimal cost tiers from bench analysis if available
    if bench_analysis:
        strategy['optimal_tiers'] = bench_analysis.get('optimal_tiers', {})
        strategy['hit_rates'] = bench_analysis.get('hit_rates', {})

    # Calculate suggested spots per position from composition
    if strategy['composition']:
        strategy['suggested_spots'] = {
            pos: max(1, round(bench_spots * pct))
            for pos, pct in strategy['composition'].items()
        }
        # Adjust to match actual bench spots
        total_suggested = sum(strategy['suggested_spots'].values())
        if total_suggested != bench_spots:
            diff = bench_spots - total_suggested
            largest_pos = max(strategy['composition'], key=strategy['composition'].get)
            strategy['suggested_spots'][largest_pos] += diff
    else:
        # Default distribution
        strategy['suggested_spots'] = {'RB': 2, 'WR': 2, 'QB': 1, 'TE': 1}

    return strategy


@st.fragment
def display_draft_optimizer(draft_history: pd.DataFrame):
    """Enhanced optimizer with presets and better UX"""

    st.header("🔧 Draft Optimizer")
    st.caption("Build your optimal roster using historical performance data.")

    # Validate data
    required_cols = ["yahoo_position", "cost_bucket", "year", "cost", "season_ppg"]
    missing = [col for col in required_cols if col not in draft_history.columns]
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
        return

    draft_with_cost = draft_history[
        pd.to_numeric(draft_history["cost"], errors="coerce") > 0
        ].copy()

    if draft_with_cost.empty:
        st.error("❌ No draft data with cost > 0 found.")
        return

    # Year range
    min_year = int(pd.to_numeric(draft_with_cost["year"], errors="coerce").min())
    max_year = int(pd.to_numeric(draft_with_cost["year"], errors="coerce").max())

    # Try to auto-detect roster config (includes bench size)
    detected_config = detect_roster_config()

    # Analyze historical bench performance (position-level)
    bench_analysis = analyze_bench_history()

    # Analyze optimal budget allocation (starter/bench split) - pure data-driven
    budget_allocation = analyze_optimal_budget_allocation()

    # Analyze optimal roster construction (full roster, not just bench)
    roster_analysis = analyze_optimal_roster_construction()

    # Calculate data-driven bench insurance metrics (replaces hardcoded discounts)
    bench_insurance_metrics = calculate_bench_insurance_metrics()

    # === PRESET SELECTION ===
    st.subheader("🎨 Quick Start")

    # Build preset list - include "Your League" if we detected config
    preset_options = ["Standard", "2 QB", "PPR Flex Heavy", "Best Ball", "Custom"]
    if detected_config:
        preset_options.insert(0, "Your League")

    preset = st.selectbox(
        "Roster Template",
        preset_options,
        index=0,
        help="'Your League' uses auto-detected settings"
    )

    preset_config = get_roster_preset(preset, detected_config)

    # Show detected config info
    if preset == "Your League" and detected_config:
        bench_count = detected_config.get('bench', 0)
        if bench_count > 0:
            st.caption(f"✅ Detected: {bench_count} bench spots")

    st.markdown("---")

    # === STRATEGY SELECTION ===
    st.subheader("📊 Strategy")

    col1, col2 = st.columns(2)
    with col1:
        lineup_strategy = st.selectbox(
            "Lineup Strategy",
            ["Balanced", "Zero RB", "Hero RB", "Robust RB", "Late-Round QB", "Stars & Scrubs"],
            help="Guide your starter spending"
        )
    with col2:
        # Simplified bench strategy options
        bench_strategy_options = ["Data-Driven", "Max Starters", "Balanced"]

        bench_strategy_name = st.selectbox(
            "Bench Strategy",
            bench_strategy_options,
            index=0,
            help="Max Starters: 90/10 split | Balanced: 70/30 split | Data-Driven: uses historical slot values"
        )

    strategy_preset = get_strategy_preset(lineup_strategy, preset_config)

    # Show strategy descriptions
    if lineup_strategy != "Balanced" and "description" in strategy_preset:
        st.caption(f"_Lineup: {strategy_preset['description']}_")

    st.markdown("---")

    # === CORE SETTINGS ===
    st.subheader("⚙️ Settings")

    # Budget split based on bench strategy
    total_budget = preset_config['budget']

    # Bench strategy determines the split percentage
    if bench_strategy_name == "Max Starters":
        bench_pct = 10
    elif bench_strategy_name == "Balanced":
        bench_pct = 30
    else:  # Data-Driven
        bench_pct = budget_allocation.get('optimal_bench_pct', 20) if budget_allocation else 20

    # Calculate split: floor bench, give extra to starters
    default_bench = int(total_budget * bench_pct / 100)
    default_starter = total_budget - default_bench  # Extra dollar always goes to starters

    col1, col2 = st.columns(2)
    with col1:
        starter_budget = st.number_input(
            "Starters ($)", min_value=0, value=default_starter,
            help="Budget for starting lineup", label_visibility="visible"
        )
    with col2:
        bench_budget = st.number_input(
            "Bench ($)", min_value=0, value=default_bench,
            help="Budget for bench players", label_visibility="visible"
        )

    budget = starter_budget + bench_budget

    # Year range
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "From Year", min_value=min_year, max_value=max_year,
            value=max(min_year, max_year - 3), key="start_year_input"
        )
    with col2:
        end_year = st.number_input(
            "To Year", min_value=min_year, max_value=max_year,
            value=max_year, key="end_year_input"
        )

    # === BENCH CONFIGURATION ===
    # Detect bench size from multiple sources (prioritize most accurate)
    if detected_config and detected_config.get('bench', 0) > 0:
        detected_bench = detected_config.get('bench')
        bench_source = "player lineup data"
    elif budget_allocation and budget_allocation.get('detected_bench_count', 0) > 0:
        detected_bench = budget_allocation.get('detected_bench_count')
        bench_source = "draft history patterns"
    else:
        detected_bench = 6
        bench_source = None

    bench_spots = st.number_input(
        "Bench Spots", min_value=1, max_value=15, value=detected_bench,
        help=f"Detected from {bench_source}" if bench_source else "Number of bench roster spots"
    )

    # Get bench strategy configuration
    bench_config = get_bench_strategy(
        bench_strategy_name, bench_spots, budget,
        bench_analysis=bench_analysis,
        budget_allocation=budget_allocation
    )
    starter_budget = budget - bench_config['budget']

    # Store bench info for later
    st.session_state['bench_budget'] = bench_config['budget']
    st.session_state['bench_spots'] = bench_spots
    st.session_state['bench_config'] = bench_config
    st.session_state['bench_analysis'] = bench_analysis
    st.session_state['budget_allocation'] = budget_allocation
    st.session_state['roster_analysis'] = roster_analysis
    st.session_state['bench_insurance_metrics'] = bench_insurance_metrics

    st.markdown("---")

    # === ROSTER REQUIREMENTS ===
    if preset == "Custom":
        st.subheader("📋 Roster")
        col1, col2 = st.columns(2)
        with col1:
            num_qb = st.number_input("QB", 0, 3, preset_config['qb'], key="qb")
            num_rb = st.number_input("RB", 0, 5, preset_config['rb'], key="rb")
            num_wr = st.number_input("WR", 0, 5, preset_config['wr'], key="wr")
            num_te = st.number_input("TE", 0, 3, preset_config['te'], key="te")
        with col2:
            num_flex = st.number_input("FLEX", 0, 3, preset_config['flex'], key="flex")
            num_def = st.number_input("DEF", 0, 2, preset_config['def'], key="def")
            num_k = st.number_input("K", 0, 2, preset_config['k'], key="k")
    else:
        # Compact roster display
        roster_str = f"**Roster:** {preset_config['qb']}QB, {preset_config['rb']}RB, {preset_config['wr']}WR, {preset_config['te']}TE"
        if preset_config['flex'] > 0:
            roster_str += f", {preset_config['flex']}FLEX"
        if preset_config['def'] > 0:
            roster_str += f", {preset_config['def']}DEF"
        if preset_config['k'] > 0:
            roster_str += f", {preset_config['k']}K"
        st.caption(roster_str)

        num_qb = preset_config['qb']
        num_rb = preset_config['rb']
        num_wr = preset_config['wr']
        num_te = preset_config['te']
        num_flex = preset_config['flex']
        num_def = preset_config['def']
        num_k = preset_config['k']

    # === ADVANCED OPTIONS - Auto-populated by Lineup Strategy ===
    # Get caps from lineup strategy preset
    preset_caps = {
        'qb': strategy_preset.get('qb', []),
        'rb': strategy_preset.get('rb', []),
        'wr': strategy_preset.get('wr', []),
        'te': strategy_preset.get('te', []),
        'flex': strategy_preset.get('flex', []),
        'def': strategy_preset.get('def', []),
        'k': strategy_preset.get('k', []),
    }

    with st.expander("🎛️ Advanced: Position Spending Caps", expanded=False):
        st.caption(f"💡 Pre-filled from **{lineup_strategy}** strategy. Adjust as needed.")

        # Compact constraint inputs
        constraint_data = {}

        if num_qb > 0:
            st.markdown("**QB Limits**")
            qb_cols = st.columns(min(num_qb, 4))
            constraint_data['qb'] = []
            for i in range(num_qb):
                with qb_cols[i % 4]:
                    # Use preset if available, else previous value or 100
                    default = preset_caps['qb'][i] if i < len(preset_caps['qb']) else (constraint_data['qb'][i - 1] if i > 0 else 100)
                    constraint_data['qb'].append(
                        st.slider(f"QB{i + 1}", 0, 100, default, key=f"qb_cap_{i}")
                    )

        if num_rb > 0:
            st.markdown("**RB Limits**")
            rb_cols = st.columns(min(num_rb, 4))
            constraint_data['rb'] = []
            for i in range(num_rb):
                with rb_cols[i % 4]:
                    default = preset_caps['rb'][i] if i < len(preset_caps['rb']) else (constraint_data['rb'][i - 1] if i > 0 else 100)
                    constraint_data['rb'].append(
                        st.slider(f"RB{i + 1}", 0, 100, default, key=f"rb_cap_{i}")
                    )

        if num_wr > 0:
            st.markdown("**WR Limits**")
            wr_cols = st.columns(min(num_wr, 4))
            constraint_data['wr'] = []
            for i in range(num_wr):
                with wr_cols[i % 4]:
                    default = preset_caps['wr'][i] if i < len(preset_caps['wr']) else (constraint_data['wr'][i - 1] if i > 0 else 100)
                    constraint_data['wr'].append(
                        st.slider(f"WR{i + 1}", 0, 100, default, key=f"wr_cap_{i}")
                    )

        if num_te > 0:
            st.markdown("**TE Limits**")
            te_cols = st.columns(min(num_te, 4))
            constraint_data['te'] = []
            for i in range(num_te):
                with te_cols[i % 4]:
                    default = preset_caps['te'][i] if i < len(preset_caps['te']) else (constraint_data['te'][i - 1] if i > 0 else 100)
                    constraint_data['te'].append(
                        st.slider(f"TE{i + 1}", 0, 100, default, key=f"te_cap_{i}")
                    )

        if num_flex > 0:
            st.markdown("**FLEX Limits**")
            flex_cols = st.columns(min(num_flex, 4))
            constraint_data['flex'] = []
            for i in range(num_flex):
                with flex_cols[i % 4]:
                    default = preset_caps['flex'][i] if i < len(preset_caps['flex']) else 100
                    constraint_data['flex'].append(
                        st.slider(f"FLEX{i + 1}", 0, 100, default, key=f"flex_cap_{i}")
                    )

        # DEF/K
        col1, col2 = st.columns(2)
        with col1:
            if num_def > 0:
                st.markdown("**DEF Limits**")
                constraint_data['def'] = []
                for i in range(num_def):
                    default = preset_caps['def'][i] if i < len(preset_caps['def']) else 100
                    constraint_data['def'].append(
                        st.slider(f"DEF{i + 1}", 0, 100, default, key=f"def_cap_{i}")
                    )
        with col2:
            if num_k > 0:
                st.markdown("**K Limits**")
                constraint_data['k'] = []
                for i in range(num_k):
                    default = preset_caps['k'][i] if i < len(preset_caps['k']) else 100
                    constraint_data['k'].append(
                        st.slider(f"K{i + 1}", 0, 100, default, key=f"k_cap_{i}")
                    )

        st.markdown("---")
        max_bid = st.slider(
            "Global Max Bid",
            0, 100, 100,
            help="Maximum spend on any single player"
        )

    # === LIVE DRAFT TRACKER WITH RE-OPTIMIZATION ===
    st.markdown("---")

    # Initialize session state for draft tracker
    if 'draft_tracker' not in st.session_state:
        st.session_state.draft_tracker = {
            'filled_picks': [],  # List of {position, slot_type, cost, tier}
        }

    filled_picks = st.session_state.draft_tracker.get('filled_picks', [])

    # Calculate spent budget by category (starter vs bench)
    starter_spent = sum(p['cost'] for p in filled_picks if p.get('slot_type') != 'bench')
    bench_spent = sum(p['cost'] for p in filled_picks if p.get('slot_type') == 'bench')
    total_spent = starter_spent + bench_spent
    remaining_budget = budget - total_spent

    # Get budget split from strategy
    strategy_starter_budget = bench_config.get('starter_budget', int(budget * 0.85))
    strategy_bench_budget = bench_config.get('budget', int(budget * 0.15))

    # Calculate remaining budget per category
    remaining_starter_budget = max(0, strategy_starter_budget - starter_spent)
    remaining_bench_budget = max(0, strategy_bench_budget - bench_spent)

    # Count filled positions by type
    filled_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DEF': 0, 'K': 0, 'FLEX': 0, 'BENCH': 0}
    for pick in filled_picks:
        if pick.get('slot_type') == 'bench':
            filled_counts['BENCH'] += 1
        elif pick.get('is_flex'):
            filled_counts['FLEX'] += 1
        else:
            pos = pick['position']
            if pos in filled_counts:
                filled_counts[pos] += 1

    # Calculate remaining slots needed
    remaining_qb = max(0, num_qb - filled_counts['QB'])
    remaining_rb = max(0, num_rb - filled_counts['RB'])
    remaining_wr = max(0, num_wr - filled_counts['WR'])
    remaining_te = max(0, num_te - filled_counts['TE'])
    remaining_flex = max(0, num_flex - filled_counts['FLEX'])
    remaining_def = max(0, num_def - filled_counts['DEF'])
    remaining_k = max(0, num_k - filled_counts['K'])
    remaining_bench = max(0, bench_spots - filled_counts['BENCH'])

    total_remaining_slots = remaining_qb + remaining_rb + remaining_wr + remaining_te + remaining_flex + remaining_def + remaining_k + remaining_bench

    # Run optimization for REMAINING slots
    with st.spinner("🔄 Optimizing..."):
        try:
            # Preprocess data
            agg_data = preprocess_optimizer_data(
                draft_history,
                int(start_year),
                int(end_year),
                min_sample_size=2
            )

            if agg_data.empty:
                st.error("❌ No valid data after filtering. Try expanding your year range.")
                return

            # Apply global max bid if set, but don't cap by remaining budget
            # (the LP optimizer handles budget constraints)
            if 'max_bid' in locals() and max_bid < 200:
                agg_data = agg_data[agg_data["avg_cost"] <= float(max_bid)]

            if agg_data.empty and total_remaining_slots > 0:
                st.error("❌ No players available. Try adjusting constraints.")
                return

            # Run optimization for remaining slots WITH budget split constraints
            if total_remaining_slots > 0:
                # Apply position caps from constraint_data to filter agg_data
                filtered_agg_data = agg_data.copy()
                for pos_key, caps in constraint_data.items():
                    if not caps:
                        continue
                    pos_upper = pos_key.upper()
                    if pos_upper in ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']:
                        # Get max cap for this position (use highest cap as filter)
                        max_cap = max(caps) if caps else 100
                        if max_cap < 100:
                            filtered_agg_data = filtered_agg_data[
                                (filtered_agg_data['yahoo_position'] != pos_upper) |
                                (filtered_agg_data['avg_cost'] <= max_cap)
                            ]

                result = run_optimization(
                    filtered_agg_data,
                    remaining_budget,
                    remaining_qb, remaining_rb, remaining_wr, remaining_te, remaining_flex, remaining_def, remaining_k,
                    num_bench=remaining_bench,
                    bench_metrics=bench_insurance_metrics,
                    starter_budget=remaining_starter_budget,
                    bench_budget=remaining_bench_budget
                )
            else:
                result = None  # All slots filled

            # Display the live draft tracker
            display_draft_tracker(
                result,
                budget,
                remaining_budget,
                filled_picks,
                num_qb, num_rb, num_wr, num_te, num_flex, num_def, num_k, bench_spots,
                agg_data,
                strategy_starter_budget,
                strategy_bench_budget,
                remaining_starter_budget,
                remaining_bench_budget,
                constraint_data
            )

        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")
            with st.expander("Show error details"):
                st.exception(e)


def run_optimization(agg_data, budget, num_qb, num_rb, num_wr, num_te, num_flex, num_def, num_k,
                     num_bench=0, bench_metrics: Optional[Dict] = None,
                     starter_budget: Optional[float] = None, bench_budget: Optional[float] = None):
    """
    Run the linear programming optimization for FULL roster (starters + bench).

    Uses dual-role data where each position-tier has:
    - Starter SPAR (used when filling starter/flex slots)
    - Backup SPAR (used when filling bench slots, with data-driven discounts)

    This allows the optimizer to properly value picks based on how they'll be used.
    A backup QB has ~0 SPAR because it rarely plays, even if the same tier QB
    used as a starter produces good value.

    FLEX Constraint Logic:
    - Each position (RB, WR, TE) has a minimum requirement (dedicated slots)
    - FLEX slots can be filled by any of RB, WR, or TE
    - Total RB+WR+TE selected = dedicated slots + FLEX slots

    BENCH Constraint Logic:
    - Bench players can be QB, RB, WR, or TE (not DEF/K)
    - Bench uses backup SPAR values with data-driven discounts based on:
      * Starter failure rates (busts + injuries)
      * Bench activation rates (how often backups play)
      * Expected SPAR when activated

    BUDGET Constraints:
    - If starter_budget and bench_budget are provided, separate constraints are added
    - This forces the optimizer to respect the budget split, not just the total
    """

    # Prepare dual-role data with separate starter/bench SPAR values
    # Uses data-driven discounts if bench_metrics is provided
    dual_data = prepare_dual_role_data(agg_data, bench_metrics=bench_metrics)

    costs = dual_data["avg_cost"].values
    positions = dual_data["yahoo_position"].values
    slot_types = dual_data["slot_type"].values if "slot_type" in dual_data.columns else np.array(["starter"] * len(dual_data))

    # Get SPAR values (already differentiated by slot_type in dual_data)
    if "median_spar" in dual_data.columns:
        objective_values = dual_data["median_spar"].values
    elif "mean_spar" in dual_data.columns:
        objective_values = dual_data["mean_spar"].values
    else:
        objective_values = dual_data["median_ppg"].values

    # Create problem
    prob = LpProblem("Draft_Optimizer", LpMaximize)
    n = len(costs)
    x = [LpVariable(f"x{i}", cat="Binary") for i in range(n)]

    # Objective: maximize total Manager SPAR
    prob += lpSum(objective_values[i] * x[i] for i in range(n))

    # === CREATE MASKS FOR STARTER VS BENCH ROWS ===
    # (Must be defined before budget constraints that use them)
    starter_mask = (slot_types == "starter").astype(int)
    bench_mask = (slot_types == "bench").astype(int)

    # === BUDGET CONSTRAINTS ===
    # Total budget constraint (always applied as upper bound)
    prob += lpSum(costs[i] * x[i] for i in range(n)) <= float(budget)

    # Separate starter/bench budget constraints (if provided)
    if starter_budget is not None and bench_budget is not None:
        # Starter budget constraint: cost of starter slots <= starter_budget
        starter_costs = [(costs[i] * starter_mask[i]) for i in range(n)]
        prob += lpSum(starter_costs[i] * x[i] for i in range(n)) <= float(starter_budget)

        # Bench budget constraint: cost of bench slots <= bench_budget
        bench_costs = [(costs[i] * bench_mask[i]) for i in range(n)]
        prob += lpSum(bench_costs[i] * x[i] for i in range(n)) <= float(bench_budget)

    # === POSITION CONSTRAINTS ===
    total_starters = num_qb + num_rb + num_wr + num_te + num_flex + num_def + num_k
    total_roster = total_starters + num_bench

    # Total starter slots must be filled from starter rows
    prob += lpSum(starter_mask[i] * x[i] for i in range(n)) == total_starters

    # Total bench slots must be filled from bench rows
    prob += lpSum(bench_mask[i] * x[i] for i in range(n)) == num_bench

    # Non-flex positions: exact counts from starter rows
    for pos, count in [("QB", num_qb), ("DEF", num_def), ("K", num_k)]:
        if count > 0:
            pos_starter_mask = ((positions == pos) & (slot_types == "starter")).astype(int)
            prob += lpSum(pos_starter_mask[i] * x[i] for i in range(n)) == count

    # Flex-eligible positions: minimum starters from starter rows
    flex_positions = ["RB", "WR", "TE"]
    for pos, min_count in [("RB", num_rb), ("WR", num_wr), ("TE", num_te)]:
        if min_count > 0:
            pos_starter_mask = ((positions == pos) & (slot_types == "starter")).astype(int)
            prob += lpSum(pos_starter_mask[i] * x[i] for i in range(n)) >= min_count

    # Total flex-eligible starters (RB + WR + TE) = dedicated + FLEX slots
    flex_starter_mask = (np.isin(positions, flex_positions) & (slot_types == "starter")).astype(int)
    total_flex_starters = num_rb + num_wr + num_te + num_flex
    prob += lpSum(flex_starter_mask[i] * x[i] for i in range(n)) == total_flex_starters

    # Bench can only be QB, RB, WR, TE (not DEF/K)
    bench_eligible = ["QB", "RB", "WR", "TE"]
    for pos in ["DEF", "K"]:
        pos_bench_mask = ((positions == pos) & (slot_types == "bench")).astype(int)
        prob += lpSum(pos_bench_mask[i] * x[i] for i in range(n)) == 0

    # TE bench: at most 1
    te_bench_mask = ((positions == "TE") & (slot_types == "bench")).astype(int)
    prob += lpSum(te_bench_mask[i] * x[i] for i in range(n)) <= 1

    # Solve
    solver = PULP_CBC_CMD(msg=False, timeLimit=30)
    prob.solve(solver)

    if prob.status != 1:
        st.error("❌ No optimal solution found. Try relaxing constraints or increasing budget.")
        return None

    # Extract selected tiers from dual_data
    selected = [i for i in range(n) if value(x[i]) > 0.5]
    optimal_draft = dual_data.iloc[selected].copy()

    # === IDENTIFY STARTERS vs BENCH vs FLEX ===
    # Use slot_type from dual_data to identify bench picks
    optimal_draft["is_bench"] = optimal_draft["slot_type"] == "bench"
    optimal_draft["is_flex"] = False

    # Identify FLEX among starters
    starter_rows = optimal_draft[~optimal_draft["is_bench"]].copy()
    starter_rows = starter_rows.sort_values(["yahoo_position", "avg_cost"], ascending=[True, False])

    starter_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "DEF": 0, "K": 0}
    starter_limits = {"QB": num_qb, "RB": num_rb, "WR": num_wr, "TE": num_te, "DEF": num_def, "K": num_k}
    flex_remaining = num_flex

    for idx in starter_rows.index:
        pos = starter_rows.loc[idx, "yahoo_position"]

        if pos in ["DEF", "K"]:
            continue
        elif starter_counts.get(pos, 0) < starter_limits.get(pos, 0):
            starter_counts[pos] = starter_counts.get(pos, 0) + 1
        elif flex_remaining > 0 and pos in ["RB", "WR", "TE"]:
            optimal_draft.loc[idx, "is_flex"] = True
            flex_remaining -= 1

    # Re-sort for display: starters first, then FLEX, then bench
    def sort_key(row):
        if row["is_bench"]:
            return (2, row["yahoo_position"], -row["avg_cost"])
        elif row["is_flex"]:
            return (1, row["yahoo_position"], -row["avg_cost"])
        else:
            pos_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DEF": 4, "K": 5}
            return (0, pos_order.get(row["yahoo_position"], 99), -row["avg_cost"])

    optimal_draft["_sort"] = optimal_draft.apply(sort_key, axis=1)
    optimal_draft = optimal_draft.sort_values("_sort").drop(columns=["_sort"])

    return optimal_draft


def display_draft_tracker(optimal_draft, budget, remaining_budget, filled_picks,
                          num_qb, num_rb, num_wr, num_te, num_flex, num_def, num_k, num_bench,
                          agg_data,
                          starter_budget, bench_budget,
                          remaining_starter_budget, remaining_bench_budget,
                          constraint_data=None):
    """
    Display live draft tracker with re-optimization after each pick.

    Shows filled picks at top, then optimized plan for remaining slots.
    When you add a pick, it re-runs optimization with reduced budget and slots.
    Enforces position spending caps from constraint_data.
    """
    constraint_data = constraint_data or {}

    # Calculate spent by category
    starter_spent = sum(p['cost'] for p in filled_picks if p.get('slot_type') != 'bench')
    bench_spent = sum(p['cost'] for p in filled_picks if p.get('slot_type') == 'bench')
    total_spent = starter_spent + bench_spent

    total_slots = num_qb + num_rb + num_wr + num_te + num_flex + num_def + num_k + num_bench
    starter_slots = num_qb + num_rb + num_wr + num_te + num_flex + num_def + num_k
    slots_filled = len(filled_picks)
    slots_remaining = total_slots - slots_filled

    # === COMPACT STATUS BAR ===
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Remaining", f"${remaining_budget:.0f}", delta=f"of ${budget}")
    with col2:
        st.metric("Starters", f"${remaining_starter_budget:.0f}", delta=f"of ${starter_budget}")
    with col3:
        st.metric("Progress", f"{slots_filled}/{total_slots}", delta=f"Bench: ${remaining_bench_budget:.0f}")

    # === ADD PICK ROW ===
    st.markdown("---")

    # Count filled positions by type and track which slot number we're on
    filled_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DEF': 0, 'K': 0, 'FLEX': 0, 'BENCH': 0}
    for pick in filled_picks:
        if pick.get('slot_type') == 'bench':
            filled_counts['BENCH'] += 1
        elif pick.get('is_flex'):
            filled_counts['FLEX'] += 1
        else:
            pos = pick['position']
            if pos in filled_counts:
                filled_counts[pos] += 1

    # Determine available slot types based on what's still needed
    available_slots = []
    if filled_counts['QB'] < num_qb:
        available_slots.append('QB')
    if filled_counts['RB'] < num_rb:
        available_slots.append('RB')
    if filled_counts['WR'] < num_wr:
        available_slots.append('WR')
    if filled_counts['TE'] < num_te:
        available_slots.append('TE')
    if filled_counts['FLEX'] < num_flex:
        available_slots.extend(['FLEX (RB)', 'FLEX (WR)', 'FLEX (TE)'])
    if filled_counts['DEF'] < num_def:
        available_slots.append('DEF')
    if filled_counts['K'] < num_k:
        available_slots.append('K')
    if filled_counts['BENCH'] < num_bench:
        available_slots.extend(['BN (QB)', 'BN (RB)', 'BN (WR)', 'BN (TE)'])

    # Calculate max cost for selected slot
    slot_choice = available_slots[0] if available_slots else None
    max_cost = int(remaining_budget)
    position_cap = None
    is_bench_slot = False

    if slot_choice:
        if slot_choice.startswith('BN'):
            max_cost = min(max_cost, int(remaining_bench_budget))
            is_bench_slot = True
        else:
            max_cost = min(max_cost, int(remaining_starter_budget))
            if slot_choice.startswith('FLEX'):
                pos_key = 'flex'
                slot_num = filled_counts['FLEX']
            else:
                pos_key = slot_choice.lower()
                slot_num = filled_counts.get(slot_choice, 0)
            caps = constraint_data.get(pos_key, [])
            if slot_num < len(caps):
                position_cap = caps[slot_num]
                max_cost = min(max_cost, position_cap)

    # Compact add pick row
    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        if available_slots:
            slot_choice = st.selectbox("Slot", available_slots, key="add_slot", label_visibility="collapsed")
        else:
            st.info("✅ All slots filled!")

    with col2:
        if available_slots:
            cap_hint = f"max ${position_cap}" if position_cap and position_cap < 100 else f"max ${max_cost}"
            cost_input = st.number_input(cap_hint, min_value=1, max_value=max(1, max_cost), value=min(1, max_cost), key="add_cost", label_visibility="collapsed")

    with col3:
        if available_slots:
            add_disabled = cost_input > max_cost or max_cost < 1
            if st.button("➕ Add", type="primary", disabled=add_disabled, key="add_pick_btn", use_container_width=True):
                if slot_choice:
                    if slot_choice.startswith('FLEX'):
                        position = slot_choice.split('(')[1].replace(')', '')
                        is_flex = True
                        slot_type = 'starter'
                    elif slot_choice.startswith('BN'):
                        position = slot_choice.split('(')[1].replace(')', '')
                        is_flex = False
                        slot_type = 'bench'
                    else:
                        position = slot_choice
                        is_flex = False
                        slot_type = 'starter'

                    new_pick = {
                        'position': position,
                        'slot_type': slot_type,
                        'is_flex': is_flex,
                        'cost': cost_input,
                    }
                    st.session_state.draft_tracker['filled_picks'].append(new_pick)
                    st.rerun(scope="fragment")

    # === FILLED PICKS TABLE ===
    if filled_picks:
        st.markdown("---")
        st.subheader("✅ Filled Picks")

        filled_df = pd.DataFrame([
            {
                '#': i + 1,
                'Slot': f"FLEX ({p['position']})" if p.get('is_flex') else f"BN ({p['position']})" if p['slot_type'] == 'bench' else p['position'],
                'Cost': f"${p['cost']:.0f}",
            }
            for i, p in enumerate(filled_picks)
        ])

        col1, col2 = st.columns([4, 1])
        with col1:
            st.dataframe(filled_df, hide_index=True, use_container_width=True)
        with col2:
            if st.button("🗑️ Remove Last", key="remove_last"):
                if st.session_state.draft_tracker['filled_picks']:
                    st.session_state.draft_tracker['filled_picks'].pop()
                    st.rerun(scope="fragment")
            if st.button("🔄 Reset All", key="reset_all"):
                st.session_state.draft_tracker['filled_picks'] = []
                st.rerun(scope="fragment")

    # === OPTIMAL PLAN ===
    st.markdown("---")

    if optimal_draft is not None and len(optimal_draft) > 0:
        st.markdown(f"**Optimal Plan** — {slots_remaining} slots, ${remaining_budget:.0f} remaining")

        # Build display dataframe
        display_source = optimal_draft.reset_index(drop=True)
        rows = []
        for i, row in display_source.iterrows():
            is_bench = row.get("is_bench", False) if "is_bench" in display_source.columns else False
            is_flex = row.get("is_flex", False) if "is_flex" in display_source.columns else False
            pos = row["yahoo_position"]

            if is_bench:
                slot_label = f"BN ({pos})"
            elif is_flex:
                slot_label = f"FLEX ({pos})"
            else:
                slot_label = pos

            rows.append({
                'Slot': slot_label,
                'Target': f"${row['avg_cost']:.0f}",
                'PPG': round(row.get("median_ppg", 0), 1),
            })

        plan_df = pd.DataFrame(rows)
        st.dataframe(plan_df, hide_index=True, use_container_width=True)

        total_planned = optimal_draft["avg_cost"].sum()
        st.caption(f"Planned: ${total_planned:.0f} | Unallocated: ${remaining_budget - total_planned:.0f}")

    elif slots_remaining == 0:
        st.success("🎉 Draft Complete! All slots filled.")
        st.metric("Total Spent", f"${total_spent:.0f}")
    else:
        st.warning("No optimization result - check constraints.")


@st.fragment
def display_optimization_results(optimal_draft, budget):
    """Display results with enhanced visuals - points prominently displayed (legacy)"""
    import plotly.express as px
    import plotly.graph_objects as go

    st.success("✅ Optimization Complete!")

    # Add context - updated to focus on points
    st.info("""
    💡 **Reading Results:** Costs and PPG are historical medians for each price tier.
    Projected season points assume a 14-week regular season. Your actual results may vary!
    """)

    # === CALCULATE METRICS FROM ACTUAL OPTIMIZED ROSTER ===
    # Separate starters and bench
    is_bench_col = "is_bench" in optimal_draft.columns
    if is_bench_col:
        starters = optimal_draft[~optimal_draft["is_bench"]]
        bench = optimal_draft[optimal_draft["is_bench"]]
    else:
        starters = optimal_draft
        bench = pd.DataFrame()

    # Calculate costs and PPG
    starter_cost = float(starters["avg_cost"].sum())
    bench_cost = float(bench["avg_cost"].sum()) if len(bench) > 0 else 0
    total_cost = starter_cost + bench_cost

    starter_ppg = float(starters["median_ppg"].sum())
    bench_ppg = float(bench["median_ppg"].sum()) if len(bench) > 0 else 0
    total_ppg = starter_ppg  # Only starters contribute to weekly PPG

    num_starters = len(starters)
    num_bench = len(bench)
    total_players = num_starters + num_bench

    remaining = float(budget - total_cost)
    season_points = total_ppg * 14  # 14 week regular season

    # Big number display for projected points - using columns for mobile-friendly layout
    st.markdown("---")
    col_main, col_side = st.columns([2, 3])

    with col_main:
        st.metric("🏆 Season Points", f"{season_points:.0f}", help="Projected total (14 weeks)")
        st.metric("📊 Weekly PPG", f"{total_ppg:.1f}")

    with col_side:
        sub1, sub2, sub3 = st.columns(3)
        with sub1:
            st.metric("💰 Cost", f"${total_cost:.0f}", help=f"${starter_cost:.0f} starters + ${bench_cost:.0f} bench")
        with sub2:
            st.metric("💵 Left", f"${remaining:.0f}")
        with sub3:
            st.metric("👥 Roster", f"{total_players}", help=f"{num_starters} starters + {num_bench} bench")

    st.markdown("---")

    # === LINEUP TABLE ===
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📋 Your Optimal Lineup")

        # Reset index for clean iteration
        display_source = optimal_draft.reset_index(drop=True)

        # Use tier_label for display (position-specific labels like "Elite RB")
        tier_col = "tier_label" if "tier_label" in display_source.columns else "bucket_label"
        if tier_col not in display_source.columns:
            tier_col = "position_tier"

        # Include SPAR if available (what we're optimizing for)
        cols_to_show = ["yahoo_position", tier_col, "avg_cost", "median_ppg", "sample_size"]
        col_names = ["Position", "Tier", "Cost", "PPG", "Samples"]

        if "median_spar" in display_source.columns:
            cols_to_show.insert(4, "median_spar")
            col_names.insert(4, "SPAR")

        display_df = display_source[cols_to_show].copy()
        display_df.columns = col_names

        # Mark FLEX and BENCH players in Position column
        if "is_flex" in display_source.columns:
            display_df["Position"] = display_df["Position"].astype(str)
            for i in range(len(display_source)):
                is_flex = display_source.loc[i, "is_flex"] if "is_flex" in display_source.columns else False
                is_bench = display_source.loc[i, "is_bench"] if "is_bench" in display_source.columns else False
                pos = display_df.loc[i, "Position"]

                if is_bench:
                    # Count bench players by position to number them
                    bench_num = sum(1 for j in range(i) if display_source.loc[j, "is_bench"])
                    display_df.loc[i, "Position"] = f"BN{bench_num + 1} ({pos})"
                elif is_flex:
                    display_df.loc[i, "Position"] = f"{pos} (FLEX)"

        # Add season projection
        display_df["Season Pts"] = (display_df["PPG"] * 14).round(0).astype(int)

        # Format costs and round values
        display_df["Cost"] = display_df["Cost"].apply(lambda x: f"${x:.0f}")
        display_df["PPG"] = display_df["PPG"].round(1)
        if "SPAR" in display_df.columns:
            display_df["SPAR"] = display_df["SPAR"].round(1)

        # Add row numbers
        display_df.insert(0, "#", range(1, len(display_df) + 1))

        # Reorder columns - include SPAR if present
        base_cols = ["#", "Position", "Tier", "Cost", "PPG"]
        if "SPAR" in display_df.columns:
            base_cols.append("SPAR")
        base_cols.extend(["Season Pts", "Samples"])
        display_df = display_df[base_cols]

        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("📊 Position Summary")

        breakdown = optimal_draft.groupby("yahoo_position").agg({
            "avg_cost": ["sum", "mean"],
            "median_ppg": ["sum", "mean"],
            "yahoo_position": "count"
        }).round(2)

        breakdown.columns = ["Total $", "Avg $", "Total PPG", "Avg PPG", "Count"]

        # Store raw values for chart before formatting
        breakdown_raw = optimal_draft.groupby("yahoo_position").agg({
            "avg_cost": "sum",
            "median_ppg": "sum"
        }).round(2)

        # Format currency
        for col in ["Total $", "Avg $"]:
            breakdown[col] = breakdown[col].apply(lambda x: f"${x:.2f}")

        st.dataframe(breakdown, use_container_width=True)

        # Value metrics
        st.markdown("---")
        st.markdown("**💎 Value Metrics**")
        avg_cost_per_player = starter_cost / num_starters
        avg_ppg_per_player = total_ppg / num_starters
        value_ratio = total_ppg / total_cost if total_cost > 0 else 0

        st.metric("Avg Cost/Player", f"${avg_cost_per_player:.2f}")
        st.metric("Avg PPG/Player", f"{avg_ppg_per_player:.2f}")
        st.metric("PPG per $", f"{value_ratio:.3f}")

    st.markdown("---")

    # === BUDGET ALLOCATION VISUALIZATION ===
    st.subheader("💰 Budget Allocation")

    col1, col2 = st.columns(2)

    with col1:
        # Pie chart of spending by position (separate starters vs bench)
        is_bench_col = "is_bench" in optimal_draft.columns

        # Group by position AND starter/bench status
        if is_bench_col:
            pos_spending = optimal_draft.copy()
            pos_spending["Category"] = pos_spending.apply(
                lambda r: f"{r['yahoo_position']} (Bench)" if r["is_bench"] else r["yahoo_position"],
                axis=1
            )
            pos_spending = pos_spending.groupby("Category")["avg_cost"].sum().reset_index()
            pos_spending.columns = ["Position", "Spend"]
        else:
            pos_spending = optimal_draft.groupby("yahoo_position")["avg_cost"].sum().reset_index()
            pos_spending.columns = ["Position", "Spend"]

        # Add "Unspent" slice if there's remaining budget
        if remaining > 0:
            pos_spending = pd.concat([
                pos_spending,
                pd.DataFrame([{"Position": "Unspent", "Spend": remaining}])
            ], ignore_index=True)

        fig_pie = px.pie(
            pos_spending,
            values="Spend",
            names="Position",
            title="Spending by Position",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar chart comparing cost vs points contribution
        pos_data = optimal_draft.groupby("yahoo_position").agg({
            "avg_cost": "sum",
            "median_ppg": "sum"
        }).reset_index()
        pos_data.columns = ["Position", "Cost", "PPG"]

        # Normalize for comparison
        pos_data["Cost %"] = (pos_data["Cost"] / total_cost * 100).round(1)
        pos_data["PPG %"] = (pos_data["PPG"] / total_ppg * 100).round(1)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='Budget %',
            x=pos_data["Position"],
            y=pos_data["Cost %"],
            marker_color='#667eea'
        ))
        fig_bar.add_trace(go.Bar(
            name='PPG %',
            x=pos_data["Position"],
            y=pos_data["PPG %"],
            marker_color='#28a745'
        ))
        fig_bar.update_layout(
            title="Cost vs Points Contribution",
            barmode='group',
            height=350,
            yaxis_title="Percentage",
            xaxis_title="Position"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # === BENCH RECOMMENDATIONS ===
    remaining_for_bench = budget - total_cost
    if remaining_for_bench > 0:
        st.subheader("🪑 Bench Recommendations")

        # Get bench config from session state
        bench_config = st.session_state.get('bench_config', {})
        bench_spots = st.session_state.get('bench_spots', 6)
        bench_analysis = st.session_state.get('bench_analysis', None)

        avg_per_bench = remaining_for_bench / bench_spots if bench_spots > 0 else 0

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Bench Budget", f"${remaining_for_bench:.0f}")
            st.metric("Avg per Spot", f"${avg_per_bench:.1f}")
            st.metric("Bench Spots", f"{bench_spots}")

        with col2:
            # Show data-driven insights if available
            if bench_analysis and 'hit_rates' in bench_analysis:
                hit_rates = bench_analysis['hit_rates']
                best_pos = max(hit_rates, key=hit_rates.get)
                worst_pos = min(hit_rates, key=hit_rates.get)

                # Get optimal tier info if available
                optimal_tiers = bench_analysis.get('optimal_tiers', {})
                price_insight = bench_analysis.get('bench_price_insight', '')

                info_lines = [
                    "**📊 Your League's Bench History:**",
                    f"- Best cheap value: **{best_pos}** ({hit_rates.get(best_pos, 0):.0f}% hit rate)",
                    f"- Weakest cheap value: **{worst_pos}** ({hit_rates.get(worst_pos, 0):.0f}% hit rate)",
                ]

                if price_insight:
                    info_lines.append(f"- **Price insight:** {price_insight}")

                info_lines.append(f"- Based on {bench_analysis.get('total_samples', 0)} cheap picks ($15 or less)")

                st.info("\n".join(info_lines))
            else:
                st.warning("""
                **💡 General Bench Tips:**
                - Handcuff your elite RBs
                - Target high-upside rookies at $1-3
                - Get a backup QB if starting only one
                """)

        # Suggested bench composition - use data-driven or strategy-based
        st.markdown("**Suggested Bench Composition:**")

        suggested_spots = bench_config.get('suggested_spots', {'RB': 2, 'WR': 2, 'QB': 1, 'TE': 1})
        composition = bench_config.get('composition', {'RB': 0.35, 'WR': 0.35, 'QB': 0.15, 'TE': 0.15})
        optimal_tiers = bench_config.get('optimal_tiers', {})

        # Default strategies with notes on pricing
        default_strategies = {
            'RB': "Handcuff + high-upside backup",
            'WR': "Breakout candidate + depth",
            'QB': "Streamer or bye-week fill",
            'TE': "Cheap upside or streaming option"
        }

        # Build bench suggestion table with data-driven allocations
        bench_rows = []

        for pos in ['RB', 'WR', 'QB', 'TE']:
            spots = suggested_spots.get(pos, 1)
            pct = composition.get(pos, 0.25)
            pos_budget = remaining_for_bench * pct

            # Build strategy text with data-driven insights
            strategy_parts = [default_strategies[pos]]

            # Add hit rate if we have analysis
            if bench_analysis and 'hit_rates' in bench_analysis:
                hit_rate = bench_analysis['hit_rates'].get(pos, 0)
                strategy_parts.append(f"{hit_rate:.0f}% hit rate")

            # Add optimal price tier recommendation
            if pos in optimal_tiers:
                tier_info = optimal_tiers[pos]
                optimal_tier = tier_info.get('tier', '')
                if optimal_tier:
                    strategy_parts.append(f"Target {optimal_tier}")

            bench_rows.append({
                "Position": pos,
                "Spots": spots,
                "Budget": f"${pos_budget:.0f}",
                "$/Spot": f"${pos_budget / spots:.1f}" if spots > 0 else "-",
                "Strategy": " | ".join(strategy_parts)
            })

        bench_suggestion = pd.DataFrame(bench_rows)
        st.dataframe(bench_suggestion, hide_index=True, use_container_width=True)

        # Show optimal price tier detail if available
        if bench_analysis and 'optimal_tiers' in bench_analysis:
            with st.expander("📈 Optimal Bench Price Tiers (from historical data)"):
                tier_data = bench_analysis['optimal_tiers']
                tier_rows = []
                for pos in ['RB', 'WR', 'QB', 'TE']:
                    if pos in tier_data:
                        info = tier_data[pos]
                        tier_rows.append({
                            "Position": pos,
                            "Best Tier": info.get('tier', 'N/A'),
                            "Avg SPAR": f"{info.get('avg_spar', 0):.1f}",
                            "Utilization": f"{info.get('utilization', 0):.0f}%",
                            "SPAR/$": f"{info.get('spar_per_dollar', 0):.2f}"
                        })
                if tier_rows:
                    st.dataframe(pd.DataFrame(tier_rows), hide_index=True, use_container_width=True)
                    st.caption("**Best Tier**: Price range with highest avg Manager SPAR. **Utilization**: % of weeks the player was started. **SPAR/$**: Value efficiency.")

    st.markdown("---")

    # === DETAILED BREAKDOWN ===
    with st.expander("📈 Detailed Position Analysis"):
        for pos in optimal_draft["yahoo_position"].unique():
            pos_data = optimal_draft[optimal_draft["yahoo_position"] == pos]

            st.markdown(f"### {pos}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Players", len(pos_data))
            with col2:
                st.metric("Total Cost", f"${pos_data['avg_cost'].sum():.2f}")
            with col3:
                st.metric("Total PPG", f"{pos_data['median_ppg'].sum():.2f}")

            # Show individual players - use tier_label for position-specific labels
            tier_col = "tier_label" if "tier_label" in pos_data.columns else "bucket_label"
            if tier_col not in pos_data.columns:
                tier_col = "position_tier"

            pos_display = pos_data[[
                tier_col, "avg_cost", "median_ppg", "sample_size"
            ]].copy()
            pos_display.columns = ["Tier", "Cost", "PPG", "Samples"]
            pos_display["Cost"] = pos_display["Cost"].apply(lambda x: f"${x:.2f}")

            st.dataframe(pos_display, hide_index=True, use_container_width=True)
            st.markdown("---")

    # === EXPORT ===
    st.subheader("💾 Export Results")

    # Use tier_label for export (position-specific labels)
    tier_col = "tier_label" if "tier_label" in optimal_draft.columns else "bucket_label"
    if tier_col not in optimal_draft.columns:
        tier_col = "position_tier"

    export_df = optimal_draft[[
        "yahoo_position", tier_col, "avg_cost", "median_ppg", "sample_size"
    ]].copy()
    export_df.columns = ["Position", "Tier", "Avg_Cost", "Median_PPG", "Sample_Size"]

    csv = export_df.to_csv(index=False)
    st.download_button(
        "📥 Download as CSV",
        csv,
        "optimal_lineup.csv",
        "text/csv",
        use_container_width=True
    )