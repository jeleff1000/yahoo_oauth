"""
League Intelligence Module

Dynamically discovers league-specific drafting inefficiencies and optimal strategies
by analyzing historical draft data. No hardcoded values - everything is derived
from the league's own history.

Core Philosophy:
- Every league has unique inefficiencies based on manager behavior
- The optimal strategy is to exploit YOUR league's specific mispricing
- All recommendations are data-driven with confidence intervals

Key Features:
1. Position Market Efficiency Analysis - Find over/underpaid positions
2. Roster Construction Pattern Mining - What separates winners from losers
3. Cheap Pick Optimization - Optimal count and position targeting
4. Strategy Performance Backtesting - Which approaches work in THIS league
5. Manager Tendency Profiling - Opponent spending patterns
6. Confidence-Weighted Recommendations - Account for sample size

Usage:
    from league_intelligence import LeagueIntelligence

    intel = LeagueIntelligence(draft_df)
    insights = intel.analyze()
    recommendations = intel.get_optimal_strategy(budget=200)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
import warnings


@dataclass
class EfficiencyMetric:
    """Represents a position's market efficiency with confidence bounds."""
    position: str
    cost_share: float  # % of total budget spent
    value_share: float  # % of total SPAR generated
    efficiency_ratio: float  # value_share / cost_share
    sample_size: int
    confidence_interval: Tuple[float, float]
    verdict: str  # 'underpaid', 'fair', 'overpaid'
    exploit_priority: int  # 1 = highest priority to exploit


@dataclass
class StrategyPerformance:
    """Performance metrics for a draft strategy."""
    strategy_name: str
    avg_spar: float
    median_spar: float
    sample_size: int
    std_spar: float
    confidence_interval: Tuple[float, float]
    win_rate_vs_baseline: float  # % of times it beat league average


@dataclass
class LeagueInsights:
    """Container for all league intelligence outputs."""
    position_efficiency: Dict[str, EfficiencyMetric]
    optimal_allocation: Dict[str, float]
    cheap_pick_strategy: Dict[str, Any]
    te_strategy: Dict[str, Any]
    roster_patterns: Dict[str, Any]
    manager_tendencies: Dict[str, Dict]
    confidence_level: str  # 'high', 'medium', 'low' based on sample size
    recommendations: List[str]
    expected_edge_vs_average: float


class LeagueIntelligence:
    """
    Analyzes league draft history to discover exploitable inefficiencies.

    All analysis is dynamic - no hardcoded position values or strategies.
    """

    # Minimum samples for reliable analysis
    MIN_SEASONS_FOR_CONFIDENCE = 5
    MIN_PICKS_PER_POSITION = 20
    MIN_STRATEGY_SAMPLES = 10

    # Efficiency thresholds (relative, not absolute)
    UNDERPAID_THRESHOLD = 1.2  # 20% more value than cost
    OVERPAID_THRESHOLD = 0.8   # 20% less value than cost

    def __init__(
        self,
        draft_df: pd.DataFrame,
        spar_column: str = 'manager_spar',
        cost_column: str = 'cost',
        position_column: str = 'yahoo_position',
        manager_column: str = 'manager',
        year_column: str = 'year'
    ):
        """
        Initialize with draft data.

        Args:
            draft_df: Historical draft data with SPAR metrics
            spar_column: Column containing SPAR values
            cost_column: Column containing draft cost
            position_column: Column containing position
            manager_column: Column containing manager name
            year_column: Column containing year
        """
        self.df = draft_df.copy()
        self.spar_col = spar_column
        self.cost_col = cost_column
        self.pos_col = position_column
        self.manager_col = manager_column
        self.year_col = year_column

        # Ensure numeric types
        self.df[self.spar_col] = pd.to_numeric(self.df[self.spar_col], errors='coerce').fillna(0)
        self.df[self.cost_col] = pd.to_numeric(self.df[self.cost_col], errors='coerce').fillna(0)

        # Filter to valid draft picks (cost > 0)
        self.df = self.df[self.df[self.cost_col] > 0].copy()

        # Cache for computed values
        self._cache = {}

    @property
    def positions(self) -> List[str]:
        """Get unique positions in the data."""
        return self.df[self.pos_col].dropna().unique().tolist()

    @property
    def seasons(self) -> int:
        """Number of seasons of data."""
        return self.df[self.year_col].nunique()

    @property
    def confidence_level(self) -> str:
        """Overall confidence level based on data volume."""
        if self.seasons >= 10:
            return 'high'
        elif self.seasons >= 5:
            return 'medium'
        else:
            return 'low'

    def _bootstrap_confidence_interval(
        self,
        values: pd.Series,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a metric."""
        if len(values) < 3:
            return (values.mean(), values.mean())

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = values.sample(n=len(values), replace=True)
            bootstrap_means.append(sample.mean())

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

        return (lower, upper)

    # =========================================================================
    # POSITION MARKET EFFICIENCY ANALYSIS
    # =========================================================================

    def calculate_position_efficiency(self) -> Dict[str, EfficiencyMetric]:
        """
        Calculate market efficiency for each position.

        Efficiency = (% of total SPAR) / (% of total cost)
        - > 1.0 means position is underpaid (delivers more value than cost)
        - < 1.0 means position is overpaid (delivers less value than cost)

        Returns:
            Dict mapping position to EfficiencyMetric
        """
        if 'position_efficiency' in self._cache:
            return self._cache['position_efficiency']

        total_cost = self.df[self.cost_col].sum()
        total_spar = self.df[self.spar_col].sum()

        if total_cost == 0 or total_spar == 0:
            return {}

        results = {}

        for pos in self.positions:
            pos_df = self.df[self.df[self.pos_col] == pos]

            if len(pos_df) < self.MIN_PICKS_PER_POSITION:
                continue

            pos_cost = pos_df[self.cost_col].sum()
            pos_spar = pos_df[self.spar_col].sum()

            cost_share = pos_cost / total_cost
            value_share = pos_spar / total_spar if total_spar != 0 else 0

            # Handle edge case where cost_share is 0
            if cost_share == 0:
                efficiency = 1.0
            else:
                efficiency = value_share / cost_share

            # Bootstrap confidence interval on efficiency
            # Calculate per-pick efficiency and bootstrap
            pos_df = pos_df.copy()
            pos_df['pick_efficiency'] = pos_df[self.spar_col] / pos_df[self.cost_col].clip(lower=0.1)
            ci = self._bootstrap_confidence_interval(pos_df['pick_efficiency'])

            # Determine verdict
            if efficiency >= self.UNDERPAID_THRESHOLD:
                verdict = 'underpaid'
            elif efficiency <= self.OVERPAID_THRESHOLD:
                verdict = 'overpaid'
            else:
                verdict = 'fair'

            results[pos] = EfficiencyMetric(
                position=pos,
                cost_share=round(cost_share * 100, 2),
                value_share=round(value_share * 100, 2),
                efficiency_ratio=round(efficiency, 3),
                sample_size=len(pos_df),
                confidence_interval=(round(ci[0], 3), round(ci[1], 3)),
                verdict=verdict,
                exploit_priority=0  # Will be set after sorting
            )

        # Set exploit priority (1 = most underpaid)
        sorted_positions = sorted(
            results.values(),
            key=lambda x: x.efficiency_ratio,
            reverse=True
        )
        for i, metric in enumerate(sorted_positions):
            results[metric.position].exploit_priority = i + 1

        self._cache['position_efficiency'] = results
        return results

    def get_exploitable_positions(self) -> Tuple[List[str], List[str]]:
        """
        Get lists of underpaid (exploit) and overpaid (avoid) positions.

        Returns:
            Tuple of (underpaid_positions, overpaid_positions)
        """
        efficiency = self.calculate_position_efficiency()

        underpaid = [p for p, m in efficiency.items() if m.verdict == 'underpaid']
        overpaid = [p for p, m in efficiency.items() if m.verdict == 'overpaid']

        # Sort by efficiency ratio
        underpaid.sort(key=lambda p: efficiency[p].efficiency_ratio, reverse=True)
        overpaid.sort(key=lambda p: efficiency[p].efficiency_ratio)

        return underpaid, overpaid

    # =========================================================================
    # CHEAP PICK OPTIMIZATION
    # =========================================================================

    def analyze_cheap_pick_strategy(
        self,
        cheap_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze optimal cheap pick strategy (count and position targeting).

        Dynamically determines the cheap threshold based on cost distribution
        if not provided.

        Returns:
            Dict with optimal_count, position_priority, expected_spar, etc.
        """
        if 'cheap_pick_strategy' in self._cache and cheap_threshold is None:
            return self._cache['cheap_pick_strategy']

        # Dynamically determine cheap threshold if not provided
        # Use bottom 20th percentile of costs as "cheap"
        if cheap_threshold is None:
            cheap_threshold = self.df[self.cost_col].quantile(0.20)
            cheap_threshold = max(1, min(cheap_threshold, 5))  # Reasonable bounds

        cheap_picks = self.df[self.df[self.cost_col] <= cheap_threshold]

        if len(cheap_picks) < self.MIN_PICKS_PER_POSITION:
            return {
                'cheap_threshold': cheap_threshold,
                'confidence': 'insufficient_data',
                'optimal_count': None,
                'position_priority': [],
                'expected_spar_per_pick': None
            }

        # Analyze cheap pick value by position
        pos_value = cheap_picks.groupby(self.pos_col).agg({
            self.spar_col: ['mean', 'median', 'count', 'std'],
        }).round(2)
        pos_value.columns = ['avg_spar', 'median_spar', 'count', 'std_spar']
        pos_value = pos_value.reset_index()

        # Filter positions with enough samples
        pos_value = pos_value[pos_value['count'] >= 10]

        # Rank positions by average SPAR from cheap picks
        pos_value = pos_value.sort_values('avg_spar', ascending=False)
        position_priority = pos_value[self.pos_col].tolist()

        # Find optimal number of cheap picks
        # Group by manager-year and count cheap picks, correlate with total SPAR
        manager_year_stats = self.df.groupby([self.manager_col, self.year_col]).agg({
            'manager_total_spar': 'first' if 'manager_total_spar' in self.df.columns else (self.spar_col, 'sum')
        }).reset_index()

        if 'manager_total_spar' not in manager_year_stats.columns:
            manager_year_stats.columns = [self.manager_col, self.year_col, 'manager_total_spar']

        cheap_counts = cheap_picks.groupby([self.manager_col, self.year_col]).size().reset_index(name='num_cheap')
        manager_year_stats = manager_year_stats.merge(cheap_counts, on=[self.manager_col, self.year_col], how='left')
        manager_year_stats['num_cheap'] = manager_year_stats['num_cheap'].fillna(0)

        # Bin cheap pick counts and find optimal range
        optimal_bin = 'insufficient_data'
        bin_performance = pd.DataFrame()
        correlation = None

        if len(manager_year_stats) >= self.MIN_STRATEGY_SAMPLES:
            # Create dynamic bins based on data distribution
            max_cheap = int(manager_year_stats['num_cheap'].max())
            min_cheap = int(manager_year_stats['num_cheap'].min())

            # Ensure we have enough range for bins
            if max_cheap <= min_cheap + 1:
                # Not enough variance in cheap pick counts
                optimal_bin = f"{min_cheap}-{max_cheap}"
            else:
                # Create bins ensuring monotonically increasing edges
                if max_cheap <= 4:
                    bins = sorted(list(set([0, 2, 4, max_cheap + 1])))
                    labels = ['0-2', '3-4', '5+'][:len(bins)-1]
                else:
                    bins = sorted(list(set([0, 4, 6, 8, max_cheap + 1])))
                    labels = ['1-4', '5-6', '7-8', '9+'][:len(bins)-1]

                # Remove duplicate bins and ensure monotonic
                bins = sorted(list(set(bins)))
                if len(bins) < 2:
                    bins = [0, max_cheap + 1]
                    labels = ['all']
                elif len(bins) - 1 < len(labels):
                    labels = labels[:len(bins)-1]

                try:
                    manager_year_stats['cheap_bin'] = pd.cut(
                        manager_year_stats['num_cheap'],
                        bins=bins,
                        labels=labels,
                        include_lowest=True,
                        duplicates='drop'
                    )

                    bin_performance = manager_year_stats.groupby('cheap_bin', observed=True).agg({
                        'manager_total_spar': ['mean', 'count', 'std']
                    }).round(1)
                    bin_performance.columns = ['avg_spar', 'count', 'std']
                    bin_performance = bin_performance.reset_index()

                    # Find optimal bin (highest avg SPAR with sufficient samples)
                    valid_bins = bin_performance[bin_performance['count'] >= 5]
                    if not valid_bins.empty:
                        optimal_bin = valid_bins.loc[valid_bins['avg_spar'].idxmax(), 'cheap_bin']
                    elif not bin_performance.empty:
                        optimal_bin = bin_performance.loc[bin_performance['avg_spar'].idxmax(), 'cheap_bin']
                except Exception:
                    optimal_bin = f"{min_cheap}-{max_cheap}"

                # Calculate correlation
                correlation = manager_year_stats['num_cheap'].corr(manager_year_stats['manager_total_spar'])

        result = {
            'cheap_threshold': round(cheap_threshold, 2),
            'optimal_count_range': str(optimal_bin),
            'position_priority': position_priority,
            'position_details': pos_value.to_dict('records'),
            'expected_spar_per_pick': round(cheap_picks[self.spar_col].mean(), 2),
            'total_cheap_picks_analyzed': len(cheap_picks),
            'correlation_with_success': round(correlation, 3) if correlation else None,
            'bin_performance': bin_performance.to_dict('records') if not bin_performance.empty else [],
            'confidence': self.confidence_level
        }

        if cheap_threshold is None:
            self._cache['cheap_pick_strategy'] = result

        return result

    # =========================================================================
    # POSITION-SPECIFIC STRATEGY ANALYSIS (e.g., TE punt vs premium)
    # =========================================================================

    def analyze_position_strategy(
        self,
        position: str,
        tier_bins: Optional[List[float]] = None,
        tier_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze different spending strategies for a specific position.

        Dynamically creates tiers based on cost distribution if not provided.

        Args:
            position: Position to analyze (e.g., 'TE', 'QB')
            tier_bins: Optional custom bin edges
            tier_labels: Optional custom tier labels

        Returns:
            Dict with strategy performance by tier, recommendation, savings
        """
        pos_df = self.df[self.df[self.pos_col] == position].copy()

        if len(pos_df) < self.MIN_PICKS_PER_POSITION:
            return {
                'position': position,
                'confidence': 'insufficient_data',
                'recommended_strategy': None
            }

        # Create dynamic tiers based on cost distribution
        # Use meaningful breakpoints for fantasy football
        if tier_bins is None:
            max_cost = pos_df[self.cost_col].max()
            min_cost = pos_df[self.cost_col].min()

            # Create tiers using PERCENTILES (works for any budget size)
            # Budget = bottom 33%, Mid = middle 33%, Premium = top 33%
            p33 = pos_df[self.cost_col].quantile(0.33)
            p67 = pos_df[self.cost_col].quantile(0.67)

            # Check if there's enough cost variance for 3 tiers
            cost_range = max_cost - pos_df[self.cost_col].min()

            if cost_range < 3 or p33 == p67:
                # Not enough variance - use 2 tiers at median
                median_cost = pos_df[self.cost_col].median()
                tier_bins = [0, median_cost, max_cost + 1]
                tier_labels = ['budget', 'premium']
            else:
                # Use 3 tiers based on percentiles
                tier_bins = [0, p33, p67, max_cost + 1]
                tier_labels = ['budget', 'mid', 'premium']

            # Ensure unique bins (can happen with many identical costs)
            tier_bins = sorted(list(set(tier_bins)))
            if len(tier_bins) <= 2:
                tier_bins = [0, max_cost + 1]
                tier_labels = ['all']
            elif len(tier_bins) == 3:
                tier_labels = ['budget', 'premium']
            elif len(tier_bins) >= 4:
                # Keep only 4 bins max
                tier_bins = tier_bins[:4]
                tier_labels = ['budget', 'mid', 'premium']

        # Ensure labels match bins
        if len(tier_labels) != len(tier_bins) - 1:
            tier_labels = [f'tier_{i+1}' for i in range(len(tier_bins) - 1)]

        pos_df['tier'] = pd.cut(
            pos_df[self.cost_col],
            bins=tier_bins,
            labels=tier_labels,
            include_lowest=True,
            duplicates='drop'
        )

        # Analyze each tier
        tier_stats = pos_df.groupby('tier', observed=True).agg({
            self.spar_col: ['mean', 'median', 'count', 'std'],
            self.cost_col: 'mean'
        }).round(2)
        tier_stats.columns = ['avg_spar', 'median_spar', 'count', 'std_spar', 'avg_cost']
        tier_stats = tier_stats.reset_index()

        # Filter to tiers with enough samples
        valid_tiers = tier_stats[tier_stats['count'] >= 5]

        if valid_tiers.empty:
            return {
                'position': position,
                'confidence': 'insufficient_data',
                'recommended_strategy': None
            }

        # Calculate SPAR per dollar for each tier
        valid_tiers = valid_tiers.copy()
        valid_tiers['spar_per_dollar'] = valid_tiers['avg_spar'] / valid_tiers['avg_cost'].clip(lower=0.5)

        # Decision framework:
        # 1. If budget tier has SPAR/$ >= mid tier AND saves significant money, recommend budget
        # 2. Otherwise recommend tier with highest total value contribution potential

        budget_tier = valid_tiers[valid_tiers['tier'] == 'budget']
        mid_tier = valid_tiers[valid_tiers['tier'] == 'mid']
        premium_tier = valid_tiers[valid_tiers['tier'] == 'premium']

        # Calculate savings of budget vs league average
        league_avg_cost = pos_df[self.cost_col].mean()

        # Determine recommendation based on efficiency
        if not budget_tier.empty:
            budget_spar_per_dollar = budget_tier['spar_per_dollar'].iloc[0]
            budget_avg_cost = budget_tier['avg_cost'].iloc[0]
            budget_savings = league_avg_cost - budget_avg_cost

            # Check if budget tier is efficient enough
            # If budget SPAR > 0 and saves significant money, it's often the right call
            if budget_tier['avg_spar'].iloc[0] > 0 and budget_savings > 5:
                best_tier = 'budget'
                best_tier_spar = budget_tier['avg_spar'].iloc[0]
                best_tier_cost = budget_avg_cost
                rationale = 'efficiency'
            else:
                # Otherwise pick highest raw SPAR (they're paying for production)
                best_tier_idx = valid_tiers['avg_spar'].idxmax()
                best_tier = valid_tiers.loc[best_tier_idx, 'tier']
                best_tier_spar = valid_tiers.loc[best_tier_idx, 'avg_spar']
                best_tier_cost = valid_tiers.loc[best_tier_idx, 'avg_cost']
                rationale = 'production'
        else:
            best_tier_idx = valid_tiers['avg_spar'].idxmax()
            best_tier = valid_tiers.loc[best_tier_idx, 'tier']
            best_tier_spar = valid_tiers.loc[best_tier_idx, 'avg_spar']
            best_tier_cost = valid_tiers.loc[best_tier_idx, 'avg_cost']
            rationale = 'production'

        # Calculate savings vs league average
        savings = league_avg_cost - best_tier_cost

        # Compare to premium tier
        premium_tier = tier_labels[-1] if tier_labels else 'premium'
        premium_data = tier_stats[tier_stats['tier'] == premium_tier]
        premium_spar = premium_data['avg_spar'].iloc[0] if not premium_data.empty else None

        spar_advantage = best_tier_spar - premium_spar if premium_spar else None

        return {
            'position': position,
            'tier_bins': tier_bins,
            'tier_labels': tier_labels,
            'tier_performance': tier_stats.to_dict('records'),
            'recommended_strategy': str(best_tier),
            'recommendation_rationale': rationale,
            'recommended_tier_avg_cost': round(best_tier_cost, 2),
            'recommended_tier_avg_spar': round(best_tier_spar, 2),
            'league_avg_cost': round(league_avg_cost, 2),
            'budget_savings_vs_average': round(savings, 2),
            'spar_advantage_vs_premium': round(spar_advantage, 2) if spar_advantage else None,
            'efficiency_analysis': valid_tiers[['tier', 'avg_spar', 'avg_cost', 'spar_per_dollar']].to_dict('records'),
            'sample_size': len(pos_df),
            'confidence': self.confidence_level
        }

    # =========================================================================
    # ROSTER CONSTRUCTION PATTERN MINING
    # =========================================================================

    def analyze_winning_patterns(
        self,
        top_percentile: float = 0.8
    ) -> Dict[str, Any]:
        """
        Compare roster construction patterns of top vs bottom drafters.

        Args:
            top_percentile: Percentile threshold for "top" drafters (default: top 20%)

        Returns:
            Dict with allocation differences between winners and losers
        """
        # Get manager-year level SPAR
        if 'manager_total_spar' in self.df.columns:
            manager_spar = self.df.groupby([self.manager_col, self.year_col]).agg({
                'manager_total_spar': 'first'
            }).reset_index()
        else:
            manager_spar = self.df.groupby([self.manager_col, self.year_col]).agg({
                self.spar_col: 'sum'
            }).reset_index()
            manager_spar.columns = [self.manager_col, self.year_col, 'manager_total_spar']

        if len(manager_spar) < self.MIN_STRATEGY_SAMPLES:
            return {'confidence': 'insufficient_data'}

        # Split into top and bottom
        threshold = manager_spar['manager_total_spar'].quantile(top_percentile)
        bottom_threshold = manager_spar['manager_total_spar'].quantile(1 - top_percentile)

        top_seasons = manager_spar[manager_spar['manager_total_spar'] >= threshold]
        bottom_seasons = manager_spar[manager_spar['manager_total_spar'] <= bottom_threshold]

        # Get draft picks for each group
        top_picks = self.df.merge(
            top_seasons[[self.manager_col, self.year_col]],
            on=[self.manager_col, self.year_col]
        )
        bottom_picks = self.df.merge(
            bottom_seasons[[self.manager_col, self.year_col]],
            on=[self.manager_col, self.year_col]
        )

        if len(top_picks) < 50 or len(bottom_picks) < 50:
            return {'confidence': 'insufficient_data'}

        # Calculate allocation differences
        def calc_allocation(picks_df):
            pos_spend = picks_df.groupby(self.pos_col)[self.cost_col].sum()
            total_spend = pos_spend.sum()
            return (pos_spend / total_spend * 100).round(2)

        top_alloc = calc_allocation(top_picks)
        bottom_alloc = calc_allocation(bottom_picks)
        league_alloc = calc_allocation(self.df)

        # Calculate differences
        differences = {}
        for pos in self.positions:
            top_pct = top_alloc.get(pos, 0)
            bottom_pct = bottom_alloc.get(pos, 0)
            league_pct = league_alloc.get(pos, 0)

            differences[pos] = {
                'top_drafter_allocation': top_pct,
                'bottom_drafter_allocation': bottom_pct,
                'league_average': league_pct,
                'winner_vs_loser_diff': round(top_pct - bottom_pct, 2),
                'recommendation': 'increase' if top_pct > bottom_pct + 1 else (
                    'decrease' if top_pct < bottom_pct - 1 else 'neutral'
                )
            }

        # Sort by absolute difference
        sorted_diffs = sorted(
            differences.items(),
            key=lambda x: abs(x[1]['winner_vs_loser_diff']),
            reverse=True
        )

        return {
            'position_differences': dict(sorted_diffs),
            'top_seasons_analyzed': len(top_seasons),
            'bottom_seasons_analyzed': len(bottom_seasons),
            'top_avg_spar': round(top_seasons['manager_total_spar'].mean(), 1),
            'bottom_avg_spar': round(bottom_seasons['manager_total_spar'].mean(), 1),
            'key_insights': [
                f"Top drafters {'over' if d['winner_vs_loser_diff'] > 0 else 'under'}spend on {p} by {abs(d['winner_vs_loser_diff']):.1f}%"
                for p, d in sorted_diffs[:3]
                if abs(d['winner_vs_loser_diff']) > 2
            ],
            'confidence': self.confidence_level
        }

    # =========================================================================
    # MANAGER TENDENCY PROFILING
    # =========================================================================

    def profile_managers(self) -> Dict[str, Dict]:
        """
        Profile each manager's drafting tendencies.

        Returns:
            Dict mapping manager name to their profile
        """
        profiles = {}

        for manager in self.df[self.manager_col].unique():
            mgr_df = self.df[self.df[self.manager_col] == manager]

            if len(mgr_df) < 20:
                continue

            # Calculate position allocation
            pos_spend = mgr_df.groupby(self.pos_col)[self.cost_col].sum()
            total_spend = pos_spend.sum()
            allocation = (pos_spend / total_spend * 100).round(2).to_dict()

            # Calculate cheap pick tendency
            cheap_threshold = self.df[self.cost_col].quantile(0.20)
            cheap_pct = (mgr_df[self.cost_col] <= cheap_threshold).mean() * 100

            # Calculate average pick cost
            avg_cost = mgr_df[self.cost_col].mean()

            # Calculate draft success metrics
            avg_spar = mgr_df[self.spar_col].mean()

            # Bust/breakout rates
            bust_rate = mgr_df['is_bust'].mean() * 100 if 'is_bust' in mgr_df.columns else None
            breakout_rate = mgr_df['is_breakout'].mean() * 100 if 'is_breakout' in mgr_df.columns else None

            profiles[manager] = {
                'position_allocation': allocation,
                'cheap_pick_percentage': round(cheap_pct, 1),
                'avg_pick_cost': round(avg_cost, 2),
                'avg_spar_per_pick': round(avg_spar, 2),
                'bust_rate': round(bust_rate, 1) if bust_rate else None,
                'breakout_rate': round(breakout_rate, 1) if breakout_rate else None,
                'seasons': mgr_df[self.year_col].nunique(),
                'total_picks': len(mgr_df)
            }

        return profiles

    # =========================================================================
    # SLOT-LEVEL EFFICIENCY ANALYSIS
    # =========================================================================

    def analyze_slot_efficiency(self) -> Dict[str, Dict]:
        """
        Analyze efficiency by SLOT (QB1 vs QB2, RB1 vs RB2, etc.)

        This is critical for understanding that:
        - QB1 should get starter-level investment
        - QB2 is a bench slot, should be cheap
        - RB1/RB2 are both starters, different from RB4 (bench)

        Returns dict mapping slot (e.g., 'QB1', 'RB2') to efficiency metrics.
        """
        # Rank picks within each position for each manager-year
        def assign_position_rank(group):
            group = group.sort_values(self.cost_col, ascending=False)
            group = group.copy()
            group['position_rank'] = range(1, len(group) + 1)
            return group

        df_ranked = self.df.groupby(
            [self.manager_col, self.year_col, self.pos_col],
            group_keys=False
        ).apply(assign_position_rank, include_groups=False)

        # Merge back the position rank
        self.df['_temp_idx'] = range(len(self.df))
        df_ranked['_temp_idx'] = range(len(df_ranked))

        results = {}

        for pos in self.positions:
            pos_df = self.df[self.df[self.pos_col] == pos].copy()

            # Rank within position for each manager-year
            pos_df['position_rank'] = pos_df.groupby(
                [self.manager_col, self.year_col]
            )[self.cost_col].rank(method='first', ascending=False).astype(int)

            for rank in range(1, 6):  # Analyze up to 5 deep at each position
                slot_name = f"{pos}{rank}"
                slot_df = pos_df[pos_df['position_rank'] == rank]

                if len(slot_df) < 10:  # Lower threshold for slot analysis
                    continue

                avg_cost = slot_df[self.cost_col].mean()
                avg_spar = slot_df[self.spar_col].mean()
                spar_per_dollar = avg_spar / max(avg_cost, 0.5)

                # Determine if this is likely a starter or bench slot
                pos_median_cost = self.df[self.df[self.pos_col] == pos][self.cost_col].median()
                is_starter_slot = avg_cost >= pos_median_cost * 0.5

                results[slot_name] = {
                    'position': pos,
                    'rank': rank,
                    'avg_cost': round(avg_cost, 2),
                    'avg_spar': round(avg_spar, 2),
                    'spar_per_dollar': round(spar_per_dollar, 2),
                    'sample_size': len(slot_df),
                    'likely_starter': is_starter_slot,
                    'recommendation': self._get_slot_recommendation(pos, rank, avg_cost, avg_spar, spar_per_dollar)
                }

        return results

    def _get_slot_recommendation(self, pos: str, rank: int, avg_cost: float, avg_spar: float, spar_per_dollar: float) -> str:
        """Generate recommendation for a specific slot based on data, not assumptions."""

        # Calculate league-wide benchmarks for comparison
        league_avg_spar_per_dollar = self.df[self.spar_col].sum() / max(self.df[self.cost_col].sum(), 1)

        # How does this slot compare to league average efficiency?
        efficiency_vs_league = spar_per_dollar / max(league_avg_spar_per_dollar, 0.1)

        # Determine recommendation based on RELATIVE efficiency
        if efficiency_vs_league >= 2.0:
            return f"EXPLOIT ({efficiency_vs_league:.1f}x league avg efficiency)"
        elif efficiency_vs_league >= 1.3:
            return f"good value ({efficiency_vs_league:.1f}x avg)"
        elif efficiency_vs_league >= 0.8:
            return "fair value"
        elif efficiency_vs_league >= 0.5:
            return "below average value"
        else:
            return f"poor value ({efficiency_vs_league:.1f}x avg)"

    # =========================================================================
    # OPTIMAL ALLOCATION CALCULATOR
    # =========================================================================

    def calculate_optimal_allocation(
        self,
        budget: float = 200,
        roster_structure: Optional[Dict[str, int]] = None,
        strategy: str = 'beat_league'
    ) -> Dict[str, float]:
        """
        Calculate optimal budget allocation based on league analysis.

        Args:
            budget: Total draft budget
            roster_structure: Dict of position -> count (optional)
            strategy: 'beat_league', 'copy_winners', 'league_average', or 'mathematical_optimal'

        Returns:
            Dict mapping position to recommended spend
        """
        if strategy == 'league_average':
            # Simple: match league average allocation
            pos_spend = self.df.groupby(self.pos_col)[self.cost_col].sum()
            total = pos_spend.sum()
            return {pos: round(budget * (spend/total), 2) for pos, spend in pos_spend.items()}

        elif strategy == 'copy_winners':
            # Use top drafter allocation
            patterns = self.analyze_winning_patterns()
            if patterns.get('confidence') == 'insufficient_data':
                return self.calculate_optimal_allocation(budget, roster_structure, 'league_average')

            allocation = {}
            for pos, data in patterns['position_differences'].items():
                allocation[pos] = round(budget * data['top_drafter_allocation'] / 100, 2)
            return allocation

        else:  # 'beat_league' - exploit inefficiencies
            efficiency = self.calculate_position_efficiency()
            cheap_strategy = self.analyze_cheap_pick_strategy()
            patterns = self.analyze_winning_patterns()

            # Start with league average as baseline
            pos_spend = self.df.groupby(self.pos_col)[self.cost_col].sum()
            total = pos_spend.sum()
            baseline = {pos: spend/total for pos, spend in pos_spend.items()}

            # Adjust based on efficiency
            adjustment_factor = {}
            for pos, metric in efficiency.items():
                if metric.verdict == 'overpaid':
                    # Reduce allocation for overpaid positions
                    adjustment_factor[pos] = 0.7  # 30% reduction
                elif metric.verdict == 'underpaid':
                    # Increase allocation for underpaid positions (modest increase)
                    adjustment_factor[pos] = 1.1  # 10% increase
                else:
                    adjustment_factor[pos] = 1.0

            # Apply adjustments
            adjusted = {}
            for pos in baseline:
                factor = adjustment_factor.get(pos, 1.0)
                adjusted[pos] = baseline[pos] * factor

            # Normalize to sum to 1
            total_adjusted = sum(adjusted.values())
            normalized = {pos: val/total_adjusted for pos, val in adjusted.items()}

            # Convert to budget amounts
            allocation = {pos: round(budget * pct, 2) for pos, pct in normalized.items()}

            return allocation

    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================

    def analyze(self) -> LeagueInsights:
        """
        Run comprehensive league analysis and return all insights.

        Returns:
            LeagueInsights dataclass with all analysis results
        """
        # Run all analyses
        position_efficiency = self.calculate_position_efficiency()
        cheap_strategy = self.analyze_cheap_pick_strategy()
        patterns = self.analyze_winning_patterns()
        manager_profiles = self.profile_managers()

        # Analyze TE specifically (common pain point)
        te_strategy = self.analyze_position_strategy('TE') if 'TE' in self.positions else {}

        # Calculate optimal allocation - detect budget from data
        # Use the average total spend per manager-year as the budget baseline
        manager_year_spend = self.df.groupby([self.manager_col, self.year_col])[self.cost_col].sum()
        detected_budget = manager_year_spend.median()  # Use median to avoid outliers
        optimal_allocation = self.calculate_optimal_allocation(budget=detected_budget, strategy='beat_league')

        # Generate recommendations
        recommendations = self._generate_recommendations(
            position_efficiency, cheap_strategy, te_strategy, patterns
        )

        # Calculate expected edge
        expected_edge = self._estimate_expected_edge(patterns)

        return LeagueInsights(
            position_efficiency=position_efficiency,
            optimal_allocation=optimal_allocation,
            cheap_pick_strategy=cheap_strategy,
            te_strategy=te_strategy,
            roster_patterns=patterns,
            manager_tendencies=manager_profiles,
            confidence_level=self.confidence_level,
            recommendations=recommendations,
            expected_edge_vs_average=expected_edge
        )

    # =========================================================================
    # MATHEMATICAL OPTIMAL ALLOCATION (True Optimization)
    # =========================================================================

    def calculate_mathematical_optimal(
        self,
        budget: float = None,
        roster_slots: List[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate the mathematically optimal allocation using scipy optimization.

        This solves for maximum expected SPAR given:
        - Historical SPAR distributions at each cost tier for each slot
        - Budget constraint
        - Roster constraints

        Unlike 'winning patterns' which compares to other managers,
        this finds the TRUE optimal regardless of what your league does.

        Args:
            budget: Total draft budget (auto-detected if None)
            roster_slots: List of slots like ['QB1', 'RB1', 'RB2', ...] (auto-detected if None)

        Returns:
            Dict with optimal allocation and comparison to league average
        """
        try:
            from scipy.optimize import minimize
            from scipy.stats import linregress
        except ImportError:
            return {'error': 'scipy required for mathematical optimization'}

        # Auto-detect budget
        if budget is None:
            manager_year_spend = self.df.groupby([self.manager_col, self.year_col])[self.cost_col].sum()
            budget = manager_year_spend.median()

        # Build slot-level SPAR curves
        slot_curves = self._build_slot_curves()

        if not slot_curves:
            return {'error': 'Insufficient data to build slot curves'}

        # Auto-detect roster from data if not provided
        if roster_slots is None:
            # Use the slots we have data for, prioritizing starters
            starter_priority = ['QB1', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE1', 'DEF1', 'K1']
            bench_priority = ['QB2', 'RB3', 'RB4', 'RB5', 'WR4', 'WR5', 'TE2']

            roster_slots = [s for s in starter_priority + bench_priority if s in slot_curves]
        else:
            roster_slots = [s for s in roster_slots if s in slot_curves]

        if len(roster_slots) < 5:
            return {'error': f'Only {len(roster_slots)} slots have enough data'}

        # Objective function: maximize expected SPAR
        def expected_spar(cost, curve):
            return curve['slope'] * np.log(max(cost, 0.5) + 1) + curve['intercept']

        def neg_total_spar(allocations):
            return -sum(expected_spar(allocations[i], slot_curves[slot])
                       for i, slot in enumerate(roster_slots))

        # Constraints
        def budget_constraint(allocations):
            return budget - sum(allocations)

        # Bounds from historical data
        bounds = [(max(1, slot_curves[slot]['min_cost']),
                   slot_curves[slot]['max_cost'])
                  for slot in roster_slots]

        # Initial guess: league average, scaled to budget
        initial = [slot_curves[slot]['avg_cost'] for slot in roster_slots]
        scale = budget / max(sum(initial), 1)
        initial = [x * scale for x in initial]

        # Clip to bounds
        initial = [max(bounds[i][0], min(bounds[i][1], initial[i]))
                   for i in range(len(initial))]

        # Optimize
        result = minimize(
            neg_total_spar,
            initial,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': budget_constraint},
            options={'maxiter': 1000}
        )

        if not result.success:
            return {'error': f'Optimization failed: {result.message}'}

        # Build results
        slot_results = []
        total_league_spar = 0
        total_optimal_spar = 0

        for i, slot in enumerate(roster_slots):
            curve = slot_curves[slot]
            league_cost = curve['avg_cost']
            optimal_cost = result.x[i]

            league_spar = expected_spar(league_cost, curve)
            optimal_spar = expected_spar(optimal_cost, curve)

            total_league_spar += league_spar
            total_optimal_spar += optimal_spar

            slot_results.append({
                'slot': slot,
                'league_avg_cost': round(league_cost, 1),
                'optimal_cost': round(optimal_cost, 1),
                'cost_diff': round(optimal_cost - league_cost, 1),
                'league_spar': round(league_spar, 1),
                'optimal_spar': round(optimal_spar, 1),
                'spar_gain': round(optimal_spar - league_spar, 1),
                'sample_size': curve['n'],
                'confidence': 'high' if curve['n'] >= 50 else 'medium' if curve['n'] >= 20 else 'low'
            })

        # Sort by SPAR gain to highlight biggest opportunities
        slot_results.sort(key=lambda x: x['spar_gain'], reverse=True)

        # Generate insights
        spend_more = [s for s in slot_results if s['cost_diff'] > 2 and s['confidence'] != 'low']
        spend_less = [s for s in slot_results if s['cost_diff'] < -2 and s['confidence'] != 'low']

        insights = []
        if spend_more:
            top_more = spend_more[0]
            insights.append(f"SPEND MORE on {top_more['slot']}: +${top_more['cost_diff']:.0f} → +{top_more['spar_gain']:.1f} SPAR")
        if spend_less:
            top_less = spend_less[0]
            insights.append(f"SPEND LESS on {top_less['slot']}: ${top_less['cost_diff']:.0f} with only {top_less['spar_gain']:.1f} SPAR loss")

        improvement = total_optimal_spar - total_league_spar
        pct_improvement = 100 * improvement / max(total_league_spar, 1)
        insights.append(f"Total improvement: +{improvement:.0f} SPAR ({pct_improvement:.1f}% vs league average)")

        return {
            'success': True,
            'budget': budget,
            'slots_analyzed': len(roster_slots),
            'league_expected_spar': round(total_league_spar, 0),
            'optimal_expected_spar': round(total_optimal_spar, 0),
            'improvement_spar': round(improvement, 0),
            'improvement_pct': round(pct_improvement, 1),
            'slot_details': slot_results,
            'insights': insights,
            'spend_more': spend_more[:3],  # Top 3 opportunities
            'spend_less': spend_less[:3]
        }

    def _build_slot_curves(self) -> Dict[str, Dict]:
        """Build SPAR prediction curves for each slot (QB1, QB2, RB1, etc.)."""
        from scipy.stats import linregress

        curves = {}

        for pos in self.positions:
            pos_df = self.df[self.df[self.pos_col] == pos].copy()

            if len(pos_df) < 10:
                continue

            # Rank within each manager-year by cost (highest cost = slot 1)
            pos_df['slot_rank'] = pos_df.groupby(
                [self.manager_col, self.year_col]
            )[self.cost_col].rank(method='first', ascending=False)

            for rank in range(1, 6):
                slot_name = f'{pos}{rank}'
                slot_df = pos_df[pos_df['slot_rank'] == rank].copy()

                if len(slot_df) < 10:
                    continue

                # Build log curve: SPAR = slope * log(cost + 1) + intercept
                x = np.log(slot_df[self.cost_col].clip(lower=0.5) + 1).values
                y = slot_df[self.spar_col].values

                if np.std(x) < 0.1:  # No variance in cost
                    slope, intercept = 0, np.mean(y)
                else:
                    slope, intercept, _, _, _ = linregress(x, y)

                curves[slot_name] = {
                    'slope': slope,
                    'intercept': intercept,
                    'min_cost': max(1, slot_df[self.cost_col].min()),
                    'max_cost': slot_df[self.cost_col].max(),
                    'avg_cost': slot_df[self.cost_col].mean(),
                    'avg_spar': slot_df[self.spar_col].mean(),
                    'n': len(slot_df)
                }

        return curves

    def _generate_recommendations(
        self,
        efficiency: Dict[str, EfficiencyMetric],
        cheap_strategy: Dict,
        te_strategy: Dict,
        patterns: Dict
    ) -> List[str]:
        """Generate actionable recommendations from analysis."""
        recs = []

        # Efficiency-based recommendations
        underpaid, overpaid = self.get_exploitable_positions()

        if underpaid:
            top_exploit = underpaid[0]
            metric = efficiency[top_exploit]
            recs.append(
                f"EXPLOIT: {top_exploit} is underpaid ({metric.efficiency_ratio:.2f}x efficiency). "
                f"Cheap {top_exploit} picks deliver outsized value."
            )

        if overpaid:
            top_avoid = overpaid[0]
            metric = efficiency[top_avoid]
            recs.append(
                f"AVOID OVERPAYING: {top_avoid} has low efficiency ({metric.efficiency_ratio:.2f}x). "
                f"Consider budget/streaming approach."
            )

        # Cheap pick recommendations
        if cheap_strategy.get('optimal_count_range'):
            recs.append(
                f"TARGET {cheap_strategy['optimal_count_range']} cheap picks "
                f"(${cheap_strategy['cheap_threshold']:.0f} or less). "
                f"Position priority: {', '.join(cheap_strategy['position_priority'][:3])}"
            )

        # Position strategy recommendations
        if te_strategy.get('recommended_strategy'):
            strategy = te_strategy['recommended_strategy']
            savings = te_strategy.get('budget_savings_vs_average', 0)
            spar_adv = te_strategy.get('spar_advantage_vs_premium')

            if savings > 5 and strategy != 'premium':
                recs.append(
                    f"TE STRATEGY: '{strategy}' tier saves ${savings:.0f} vs league average"
                    + (f" with {spar_adv:.1f} more expected SPAR than premium" if spar_adv and spar_adv > 0 else "")
                )

        # Pattern-based recommendations
        if patterns.get('key_insights'):
            for insight in patterns['key_insights'][:2]:
                recs.append(f"PATTERN: {insight}")

        return recs

    def _estimate_expected_edge(self, patterns: Dict) -> float:
        """Estimate expected SPAR advantage of optimal strategy vs league average."""
        if patterns.get('confidence') == 'insufficient_data':
            return 0.0

        top_avg = patterns.get('top_avg_spar', 0)
        bottom_avg = patterns.get('bottom_avg_spar', 0)

        if top_avg and bottom_avg:
            # Estimate edge as moving from average (middle) toward top quartile
            # Conservative estimate: 25% of the top-bottom gap
            gap = top_avg - bottom_avg
            return round(gap * 0.25, 1)

        return 0.0


# =============================================================================
# CONVENIENCE FUNCTIONS FOR STREAMLIT INTEGRATION
# =============================================================================

def get_league_insights(draft_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to get league insights as a dictionary.

    Args:
        draft_df: Historical draft data
        **kwargs: Additional arguments for LeagueIntelligence

    Returns:
        Dictionary of all insights suitable for JSON serialization
    """
    intel = LeagueIntelligence(draft_df, **kwargs)
    insights = intel.analyze()

    # Convert to dictionary format
    return {
        'position_efficiency': {
            pos: {
                'cost_share': m.cost_share,
                'value_share': m.value_share,
                'efficiency': m.efficiency_ratio,
                'verdict': m.verdict,
                'priority': m.exploit_priority,
                'sample_size': m.sample_size
            }
            for pos, m in insights.position_efficiency.items()
        },
        'optimal_allocation': insights.optimal_allocation,
        'cheap_pick_strategy': insights.cheap_pick_strategy,
        'te_strategy': insights.te_strategy,
        'roster_patterns': insights.roster_patterns,
        'manager_tendencies': insights.manager_tendencies,
        'confidence': insights.confidence_level,
        'recommendations': insights.recommendations,
        'expected_edge': insights.expected_edge_vs_average
    }


def get_position_efficiency_summary(draft_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get position efficiency as a DataFrame for display.

    Returns:
        DataFrame with position efficiency metrics
    """
    intel = LeagueIntelligence(draft_df)
    efficiency = intel.calculate_position_efficiency()

    rows = []
    for pos, metric in efficiency.items():
        rows.append({
            'Position': pos,
            'Cost %': f"{metric.cost_share:.1f}%",
            'Value %': f"{metric.value_share:.1f}%",
            'Efficiency': f"{metric.efficiency_ratio:.2f}x",
            'Verdict': metric.verdict.upper(),
            'Priority': metric.exploit_priority,
            'Samples': metric.sample_size
        })

    df = pd.DataFrame(rows)
    return df.sort_values('Priority')
