"""
Draft Value Metrics V3 (Multi-League)

SPAR-based draft value analysis with draft capital efficiency metrics.

This transformation adds:
1. SPAR Metrics (performance-based value):
   - replacement_ppg: Position-specific replacement baseline
   - spar: Season Points Above Replacement (total_points - replacement × games)
   - pgvor: Per-Game Value Over Replacement (season_ppg - replacement_ppg)
   - cost_norm: Normalized draft cost (auction $ or snake pick-equivalent $)
   - draft_roi: Return on Investment (spar / cost_norm)

2. Draft Capital Efficiency (value vs expectations):
   - pick_savings: ADP - actual pick (positive = got player later than expected)
   - cost_savings: Avg cost - actual cost (positive = paid less than expected)
   - savings: Unified metric (cost_savings for auction, pick_savings for snake)

3. Keeper Economics:
   - kept_next_year: Whether this player was kept in following season
   - cost_bucket: Position-based value tier (1-3, 4-6, etc.)

Key Features:
- Uses true replacement levels (dynamically calculated from league settings)
- Comparable metrics across auction and snake drafts
- No hardcoded position lists or tiers
- Combines performance (SPAR) with draft efficiency (savings)

Usage:
    python draft_value_metrics_v3.py --context path/to/league_context.json
    python draft_value_metrics_v3.py --context path/to/league_context.json --dry-run
    python draft_value_metrics_v3.py --context path/to/league_context.json --backup
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# Add parent directories to path for imports
_script_file = Path(__file__).resolve()

# Auto-detect directory structure
if _script_file.parent.name == 'modules':
    _modules_dir = _script_file.parent
    _transformations_dir = _modules_dir.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'draft_enrichment':
    _transformations_dir = _script_file.parent.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'transformations':
    _transformations_dir = _script_file.parent
    _multi_league_dir = _transformations_dir.parent
else:
    _current = _script_file.parent
    while _current.name != 'multi_league' and _current.parent != _current:
        _current = _current.parent
    _multi_league_dir = _current

_scripts_dir = _multi_league_dir.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))

from core.league_context import LeagueContext
from multi_league.transformations.draft.modules.spar_calculator import (
    calculate_all_draft_metrics
)
from multi_league.transformations.draft.modules.draft_grade_calculator import (
    calculate_draft_grades
)
from multi_league.transformations.draft.modules.value_tier_calculator import (
    calculate_value_tiers
)
from multi_league.transformations.draft.modules.draft_flags_calculator import (
    calculate_all_draft_flags
)
from multi_league.transformations.draft.modules.manager_draft_grade_calculator import (
    calculate_manager_draft_grades,
    get_manager_draft_leaderboard
)
from multi_league.transformations.draft.modules.starter_designation_calculator import (
    calculate_starter_designation,
    get_spar_by_starter_designation
)
from multi_league.transformations.draft.modules.bench_insurance_calculator import (
    calculate_bench_insurance_metrics,
    get_position_bench_discount,
    calculate_bench_value_by_rank
)


def backup_file(file_path: Path) -> Path:
    """Create a timestamped backup of a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_name(f"{file_path.stem}_backup_{timestamp}{file_path.suffix}")

    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
        df.to_parquet(backup_path, index=False)
    else:
        import shutil
        shutil.copy2(file_path, backup_path)

    print(f"[OK] Backup created: {backup_path}")
    return backup_path


def add_keeper_economics(draft_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add keeper-related columns to draft data.

    Adds:
    - kept_next_year: Whether player was kept in following season

    Args:
        draft_df: Draft DataFrame

    Returns:
        Draft DataFrame with keeper columns added
    """
    draft_df = draft_df.copy()

    # Build lookup of players kept in each year (vectorized, not O(n²))
    # Find all keeper picks and shift year back by 1 to match source year
    if 'is_keeper_status' not in draft_df.columns:
        draft_df['kept_next_year'] = 0
        return draft_df

    keepers_next_year = (
        draft_df[draft_df['is_keeper_status'] == 1][['yahoo_player_id', 'year']]
        .copy()
        .dropna(subset=['yahoo_player_id', 'year'])
    )
    keepers_next_year['year'] = keepers_next_year['year'] - 1  # Shift to source year
    keepers_next_year = keepers_next_year.drop_duplicates()
    keepers_next_year['kept_next_year'] = 1

    # Merge to flag players who were kept the following year
    draft_df = draft_df.merge(
        keepers_next_year,
        on=['yahoo_player_id', 'year'],
        how='left'
    )
    draft_df['kept_next_year'] = draft_df['kept_next_year'].fillna(0).astype(int)

    return draft_df


def _find_natural_tiers(values: pd.Series, ascending: bool = False,
                        min_gap_absolute: float = 3.0,
                        min_gap_relative: float = 0.05,
                        max_tiers: int = 10) -> tuple:
    """
    Find natural tier boundaries using HYBRID approach:
    - TOP: Natural breaks at significant price gaps
    - BOTTOM: Guaranteed granularity for large price ranges

    This ensures:
    - $70 and $60 RBs are in different tiers (if gap exists)
    - $17 and $1 RBs are NOT lumped together (guaranteed splits)
    - $1-2 Kickers are in 1 tier (no meaningful differences)

    Algorithm:
    1. Find all significant gaps (large enough to matter)
    2. If price range is large, add percentile-based splits to ensure coverage
    3. Combine and return unique boundaries

    Args:
        values: Series of values (costs or rounds)
        ascending: If True, lower values are better (snake rounds)
                   If False, higher values are better (auction costs)
        min_gap_absolute: Minimum absolute gap to be significant (default $3)
        min_gap_relative: Minimum gap as fraction of range (default 5%)
        max_tiers: Maximum number of tiers to create (default 7)

    Returns:
        Tuple of (boundaries list, num_tiers)
    """
    data = np.array(sorted(values.dropna()))
    n = len(data)

    if n < 2:
        return [], 1

    # Calculate price range
    price_range = data[-1] - data[0]
    if price_range < 3:
        # All similar price (e.g., $1-2 kickers) - 1 tier
        return [], 1

    # Calculate all gaps between consecutive unique values
    unique_vals = np.unique(data)
    if len(unique_vals) < 2:
        return [], 1

    gaps = []
    for i in range(len(unique_vals) - 1):
        gap_size = unique_vals[i + 1] - unique_vals[i]
        boundary_val = (unique_vals[i] + unique_vals[i + 1]) / 2
        gaps.append({
            'size': gap_size,
            'boundary': boundary_val,
            'lower': unique_vals[i],
            'upper': unique_vals[i + 1]
        })

    if not gaps:
        return [], 1

    # Calculate gap statistics
    gap_sizes = [g['size'] for g in gaps]
    median_gap = np.median(gap_sizes)
    min_threshold = max(
        min_gap_absolute,
        price_range * min_gap_relative
    )

    # PART 1: Find ALL significant natural gaps
    # A gap is significant if it's large enough in absolute/relative terms
    # We use a tiered threshold: bigger gaps at top of distribution are more valuable
    significant_gaps = []
    for g in gaps:
        is_large_enough = g['size'] >= min_threshold

        # For gaps in the top half of the price range, be more generous
        # A $10 gap at $60-70 is more meaningful than $10 gap at $1-11
        is_top_half = g['boundary'] > (data[0] + price_range / 2)
        if is_top_half:
            # Top half: just needs to be large enough
            is_notable = True
        else:
            # Bottom half: needs to be notably larger than median
            is_notable = g['size'] > median_gap * 1.2 if median_gap > 0 else g['size'] > 0

        if is_large_enough and is_notable:
            significant_gaps.append(g)

    # PART 2: Determine minimum tiers based on price range
    # Large price ranges need minimum granularity even without big gaps
    if price_range >= 50:
        min_tiers = 5  # $50+ range needs at least 5 tiers
    elif price_range >= 30:
        min_tiers = 4  # $30-50 range needs at least 4 tiers
    elif price_range >= 15:
        min_tiers = 3  # $15-30 range needs at least 3 tiers
    elif price_range >= 5:
        min_tiers = 2  # $5-15 range needs at least 2 tiers
    else:
        min_tiers = 1  # Small range can be 1 tier

    # Determine actual tier count
    natural_tier_count = len(significant_gaps) + 1
    target_tiers = max(natural_tier_count, min_tiers)
    target_tiers = min(target_tiers, max_tiers)  # Don't exceed max

    # PART 3: Select boundaries ensuring coverage across the full range
    # Strategy: if we have more gaps than needed, select ones spread across the distribution
    if len(significant_gaps) <= target_tiers - 1:
        # Use all significant gaps
        boundaries = [g['boundary'] for g in significant_gaps]
    else:
        # Too many gaps - select ones that are spread out
        # Sort by boundary position (not size) and select evenly spaced ones
        significant_gaps.sort(key=lambda x: x['boundary'], reverse=True)

        # Take every Nth gap to spread across range
        step = len(significant_gaps) / (target_tiers - 1)
        selected_indices = [int(i * step) for i in range(target_tiers - 1)]
        boundaries = [significant_gaps[i]['boundary'] for i in selected_indices if i < len(significant_gaps)]

    # PART 4: Add percentile-based boundaries to ensure coverage
    # First, fill in if we need more boundaries to reach minimum
    needed_extra = (target_tiers - 1) - len(boundaries)
    if needed_extra > 0:
        percentiles = np.linspace(10, 90, needed_extra + 2)[1:-1]
        for p in percentiles:
            pct_val = np.percentile(data, p)
            if not any(abs(pct_val - b) < price_range * 0.05 for b in boundaries):
                boundaries.append(pct_val)

    # PART 5: Ensure no tier spans too large a range
    # Iterate until all tiers are within the span limit (forced splits override max_tiers cap)
    max_tier_span = max(8, price_range * 0.12)  # No tier should span > $8 or 12% of range

    # Allow more tiers if needed to achieve proper granularity
    hard_max_tiers = max_tiers + 4  # Allow up to 4 extra tiers for forced splits

    for _ in range(5):  # Max 5 iterations to prevent infinite loop
        # Sort and cap boundaries
        if ascending:
            boundaries = sorted(boundaries)
        else:
            boundaries = sorted(boundaries, reverse=True)

        # Cap at current limit
        current_max = min(len(boundaries), hard_max_tiers - 1)
        boundaries = boundaries[:current_max]

        # Build tier edges and check for wide tiers
        sorted_bounds = sorted(boundaries, reverse=True)
        tier_edges = [data[-1]] + sorted_bounds + [data[0]]

        needs_more_splits = False
        for i in range(len(tier_edges) - 1):
            tier_top = tier_edges[i]
            tier_bottom = tier_edges[i + 1]
            tier_span = tier_top - tier_bottom

            if tier_span > max_tier_span and len(boundaries) < hard_max_tiers - 1:
                # Add a split in the middle of this wide tier
                split_val = (tier_top + tier_bottom) / 2
                if not any(abs(split_val - b) < 2 for b in boundaries):
                    boundaries.append(split_val)
                    needs_more_splits = True

        if not needs_more_splits:
            break

    # Final sort
    if ascending:
        boundaries = sorted(boundaries)
    else:
        boundaries = sorted(boundaries, reverse=True)

    num_tiers = len(boundaries) + 1

    return boundaries, num_tiers


def assign_cost_buckets(draft_df: pd.DataFrame, default_budget: int = 200) -> pd.DataFrame:
    """
    Assign position-specific cost buckets using TRUE NATURAL BREAKS.

    The DATA determines tier count and boundaries - not arbitrary sample size rules.

    How it works:
    1. Find all price gaps between consecutive values
    2. Identify "significant" gaps (worth creating a tier boundary)
    3. Number of significant gaps + 1 = number of tiers
    4. No significant gaps = 1 tier (all players grouped together)

    A gap is "significant" if it's:
    - At least $3 (or 5% of price range, whichever is larger)
    - AND notably larger than typical gaps (> 1.5x median gap)

    Examples:
    - 10 Kickers at $1-2 each → no significant gaps → 1 tier
    - RBs at $70, $50, $30, $10, $5, $1 → multiple significant gaps → 5+ tiers
    - $60 and $70 RBs with gap → different tiers (never grouped together)

    Key features:
    - Each position-year gets its own natural tier structure
    - Keepers excluded from tier calculation (artificial prices)
    - Tier count varies by actual price distribution, not sample size

    Columns Added:
    - position_percentile: Exact percentile rank (0-100) within position-year
    - position_tier: Dynamic tier number (varies by position-year)
    - position_tier_label: Descriptive label ("Elite RB", "Starter WR", etc.)
    - position_tier_count: How many tiers exist for this position-year
    - cost_bucket: Legacy bucket for backwards compatibility
    - cost_bucket_label: Legacy label for backwards compatibility

    Args:
        draft_df: Draft DataFrame
        default_budget: Default auction budget (typically $200)

    Returns:
        DataFrame with position-specific bucket columns added
    """
    draft_df = draft_df.copy()

    # Initialize new columns
    draft_df['position_percentile'] = pd.NA
    draft_df['position_tier'] = pd.NA
    draft_df['position_tier_label'] = pd.NA
    draft_df['position_tier_count'] = pd.NA
    # Legacy columns for backwards compatibility
    draft_df['cost_bucket'] = pd.NA
    draft_df['cost_bucket_label'] = pd.NA

    # Ensure numeric columns
    draft_df['cost'] = pd.to_numeric(draft_df.get('cost', 0), errors='coerce')
    draft_df['round'] = pd.to_numeric(draft_df.get('round', 0), errors='coerce')
    draft_df['pick'] = pd.to_numeric(draft_df.get('pick', 0), errors='coerce')

    # Determine position column
    pos_col = 'position' if 'position' in draft_df.columns else 'yahoo_position'

    # Detect keeper column (check multiple variations)
    keeper_col = None
    for col in ['is_keeper_status', 'is_keeper', 'keeper', 'kept']:
        if col in draft_df.columns:
            keeper_col = col
            break

    # Dynamic tier label generation based on tier count
    def get_tier_labels(num_tiers: int) -> list:
        """Generate tier labels dynamically based on number of tiers found."""
        if num_tiers == 1:
            return ['All']
        elif num_tiers == 2:
            return ['High', 'Low']
        elif num_tiers == 3:
            return ['Top', 'Mid', 'Late']
        elif num_tiers == 4:
            return ['Elite', 'Starter', 'Depth', 'Late']
        elif num_tiers == 5:
            return ['Elite', 'Premium', 'Starter', 'Depth', 'Late']
        elif num_tiers == 6:
            return ['Elite', 'Premium', 'Starter', 'Depth+', 'Depth', 'Late']
        elif num_tiers == 7:
            return ['Elite', 'Premium', 'Starter+', 'Starter', 'Depth+', 'Depth', 'Late']
        elif num_tiers == 8:
            return ['Elite', 'Premium', 'Starter+', 'Starter', 'Depth+', 'Depth', 'Late+', 'Late']
        elif num_tiers == 9:
            return ['Elite', 'Premium+', 'Premium', 'Starter+', 'Starter', 'Depth+', 'Depth', 'Late+', 'Late']
        elif num_tiers == 10:
            return ['Elite', 'Premium+', 'Premium', 'Starter+', 'Starter', 'Depth+', 'Depth', 'Late+', 'Late', 'Flier']
        elif num_tiers == 11:
            return ['Elite', 'Premium+', 'Premium', 'Starter+', 'Starter', 'Mid', 'Depth+', 'Depth', 'Late+', 'Late', 'Flier']
        elif num_tiers >= 12:
            # For 12+ tiers, generate sufficient labels with meaningful names
            base = ['Elite', 'Premium+', 'Premium', 'Starter+', 'Starter', 'Starter-', 'Mid', 'Depth+', 'Depth', 'Late+', 'Late', 'Flier', 'Flier-', 'Deep', 'Deep-', 'Longshot']
            # Extend if needed
            while len(base) < num_tiers:
                base.append(f'Tier {len(base) + 1}')
            return base[:num_tiers]
        return [f'Tier {i+1}' for i in range(num_tiers)]

    # Process each year separately
    for year in draft_df['year'].dropna().unique():
        year_mask = draft_df['year'] == year
        year_df = draft_df[year_mask].copy()

        # Skip non-drafted players
        drafted_mask = year_df['pick'].notna()
        if not drafted_mask.any():
            continue

        drafted_df = year_df[drafted_mask].copy()

        # Detect draft type: auction if 25%+ have cost > 1
        cost_filled = drafted_df['cost'].fillna(0)
        has_cost = (cost_filled > 1).sum()
        is_auction = has_cost >= max(1, int(len(drafted_df) * 0.25))

        # Determine value column (cost for auction, round for snake)
        value_col = 'cost' if is_auction else 'round'

        # Process each position separately
        if pos_col not in drafted_df.columns:
            continue

        for pos in drafted_df[pos_col].dropna().unique():
            pos_mask = drafted_df[pos_col] == pos
            pos_df = drafted_df.loc[pos_mask].copy()
            pos_count = len(pos_df)

            if pos_count < 2:
                # Not enough data for meaningful tiers
                drafted_df.loc[pos_mask, 'position_percentile'] = 50.0
                drafted_df.loc[pos_mask, 'position_tier'] = 1
                drafted_df.loc[pos_mask, 'position_tier_label'] = f'{pos}'
                drafted_df.loc[pos_mask, 'position_tier_count'] = 1
                continue

            # EXCLUDE KEEPERS from tier calculation (they have artificial prices)
            if keeper_col and keeper_col in pos_df.columns:
                non_keeper_mask = ~(pos_df[keeper_col].fillna(False).astype(bool))
                calc_df = pos_df[non_keeper_mask]
            else:
                calc_df = pos_df

            # If all players are keepers or too few non-keepers, use all players
            if len(calc_df) < 3:
                calc_df = pos_df

            # Get values for tier calculation (non-keepers only)
            values = calc_df[value_col].dropna()
            if len(values) < 2:
                drafted_df.loc[pos_mask, 'position_percentile'] = 50.0
                drafted_df.loc[pos_mask, 'position_tier'] = 1
                drafted_df.loc[pos_mask, 'position_tier_label'] = f'{pos}'
                drafted_df.loc[pos_mask, 'position_tier_count'] = 1
                continue

            # Calculate percentile rank for ALL players (including keepers)
            if is_auction:
                # Higher cost = higher percentile
                pct_rank = pos_df[value_col].rank(pct=True, ascending=True, method='average')
            else:
                # Lower round = higher percentile (better)
                pct_rank = pos_df[value_col].rank(pct=True, ascending=False, method='average')

            # Store exact percentile (0-100 scale)
            drafted_df.loc[pos_mask, 'position_percentile'] = (pct_rank * 100).round(1)

            # Find NATURAL tier boundaries based on ACTUAL price gaps
            # The data determines both tier count AND boundaries
            # Higher max_tiers for larger price ranges to ensure granularity
            pos_price_range = values.max() - values.min()
            if pos_price_range >= 60:
                pos_max_tiers = 12  # Very large range - need many tiers
            elif pos_price_range >= 40:
                pos_max_tiers = 10
            elif pos_price_range >= 25:
                pos_max_tiers = 8
            elif pos_price_range >= 12:
                pos_max_tiers = 6
            else:
                pos_max_tiers = 4

            boundaries, num_tiers = _find_natural_tiers(
                values,
                ascending=not is_auction,  # ascending=True for snake (lower round = better)
                min_gap_absolute=3.0,  # At least $3 gap (or 3 round gap)
                min_gap_relative=0.05,  # Or 5% of price range
                max_tiers=pos_max_tiers
            )

            # Get dynamic labels based on tier count found
            tier_labels_base = get_tier_labels(num_tiers)

            # Assign tiers to each player based on their value
            tier_nums = []
            tier_labels_list = []
            player_values = pos_df[value_col].values

            for val in player_values:
                if pd.isna(val):
                    tier_nums.append(num_tiers)  # Default to lowest tier
                    tier_labels_list.append(f"{tier_labels_base[-1]} {pos}")
                    continue

                if num_tiers == 1:
                    # Only 1 tier - everyone goes in it
                    tier_nums.append(1)
                    tier_labels_list.append(f"{tier_labels_base[0]} {pos}")
                    continue

                tier_assigned = False
                if is_auction:
                    # Higher value = better tier (check from top down)
                    for i, boundary in enumerate(boundaries):
                        if val > boundary:
                            tier_nums.append(i + 1)
                            tier_labels_list.append(f"{tier_labels_base[i]} {pos}")
                            tier_assigned = True
                            break
                else:
                    # Lower value (round) = better tier
                    for i, boundary in enumerate(boundaries):
                        if val < boundary:
                            tier_nums.append(i + 1)
                            tier_labels_list.append(f"{tier_labels_base[i]} {pos}")
                            tier_assigned = True
                            break

                if not tier_assigned:
                    tier_nums.append(num_tiers)
                    tier_labels_list.append(f"{tier_labels_base[-1]} {pos}")

            # Renumber tiers to be contiguous (remove gaps from empty tiers)
            unique_tiers = sorted(set(tier_nums))
            tier_remap = {old: new + 1 for new, old in enumerate(unique_tiers)}
            tier_nums_contiguous = [tier_remap[t] for t in tier_nums]
            actual_tier_count = len(unique_tiers)

            # Re-assign labels based on contiguous tiers
            actual_labels = get_tier_labels(actual_tier_count)
            tier_labels_contiguous = [f"{actual_labels[tier_remap[t] - 1]} {pos}" for t in tier_nums]

            drafted_df.loc[pos_mask, 'position_tier'] = tier_nums_contiguous
            drafted_df.loc[pos_mask, 'position_tier_label'] = tier_labels_contiguous
            drafted_df.loc[pos_mask, 'position_tier_count'] = actual_tier_count

        # --- LEGACY BUCKETS FOR BACKWARDS COMPATIBILITY ---
        drafted_df['cost_bucket'] = drafted_df['position_tier']

        def simplify_label(label):
            if pd.isna(label):
                return label
            parts = str(label).split()
            return parts[0] if parts else label

        drafted_df['cost_bucket_label'] = drafted_df['position_tier_label'].apply(simplify_label)

        # Write back to main dataframe
        year_drafted_idx = draft_df.index[year_mask & (draft_df['pick'].notna())]
        for col in ['position_percentile', 'position_tier', 'position_tier_label',
                    'position_tier_count', 'cost_bucket', 'cost_bucket_label']:
            if col in drafted_df.columns:
                draft_df.loc[year_drafted_idx, col] = drafted_df[col].values

    return draft_df


def calculate_draft_value(draft_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate draft value vs expectations (pick_savings, cost_savings, savings).

    This function shows draft capital efficiency - how well did you draft relative
    to average draft position (ADP) and average auction values?

    Metrics:
    - pick_savings: avg_pick - actual pick (positive = drafted later than expected = good value)
    - cost_savings: avg_cost - actual cost (positive = paid less than expected = good value)
    - savings: unified metric (cost_savings for auction, pick_savings for snake)

    This is complementary to SPAR:
    - SPAR = performance-based value (what did the player produce?)
    - Savings = draft capital efficiency (how efficiently did you draft?)

    Args:
        draft_df: Draft DataFrame with draft data

    Returns:
        Draft DataFrame with value vs expectation metrics added
    """
    print("\n[Draft Value] Calculating value vs expectations (pick_savings, cost_savings)...")

    # Ensure required columns exist
    required_cols = ['year', 'pick', 'cost']
    missing = [c for c in required_cols if c not in draft_df.columns]
    if missing:
        print(f"  [WARNING] Missing required columns: {missing}. Skipping value calculation.")
        draft_df['pick_savings'] = pd.NA
        draft_df['cost_savings'] = pd.NA
        draft_df['savings'] = pd.NA
        return draft_df

    # Determine position column
    pos_col = 'position' if 'position' in draft_df.columns else 'yahoo_position'

    # Normalize numerics
    pick = pd.to_numeric(draft_df.get("pick"), errors="coerce")
    avg_pick = pd.to_numeric(draft_df.get("avg_pick"), errors="coerce")
    avg_round = pd.to_numeric(draft_df.get("avg_round"), errors="coerce")
    cost = pd.to_numeric(draft_df.get("cost"), errors="coerce")
    avg_cost = pd.to_numeric(draft_df.get("avg_cost"), errors="coerce")
    pre_avg_pk = pd.to_numeric(draft_df.get("preseason_avg_pick"), errors="coerce")
    pre_avg_cs = pd.to_numeric(draft_df.get("preseason_avg_cost"), errors="coerce")
    keeper_cost = pd.to_numeric(draft_df.get("is_keeper_cost"), errors="coerce")

    # --- PICK SAVINGS (snake draft value) ---
    pick_filled = avg_pick.fillna(pre_avg_pk)

    # Fallback: approximate from avg_round if team count available
    teams_per_year = draft_df.groupby("year")["team_key"].nunique().rename("teams")
    draft_temp = draft_df.merge(teams_per_year, on="year", how="left")
    teams = pd.to_numeric(draft_temp["teams"], errors="coerce")
    use_round_mask = pick_filled.isna() & avg_round.notna() & teams.notna() & (teams > 0)
    pick_filled.loc[use_round_mask] = ((avg_round[use_round_mask] - 1) * teams[use_round_mask]) + (teams[use_round_mask] / 2)
    pick_savings = (pick_filled - pick)

    # --- COST SAVINGS (auction draft value) ---
    cost_filled = avg_cost.fillna(pre_avg_cs)
    cost_savings = (cost_filled - cost)

    # Keeper override: for keepers, compare to keeper cost (not draft cost)
    kc_mask = keeper_cost.notna() & cost_filled.notna()
    cost_savings.loc[kc_mask] = cost_filled[kc_mask] - keeper_cost[kc_mask]

    # Store individual metrics
    draft_df["pick_savings"] = pick_savings
    draft_df["cost_savings"] = cost_savings

    # --- UNIFIED SAVINGS (based on draft_type) ---
    # Determine draft type from draft_type column or heuristic
    if 'draft_type' in draft_df.columns:
        # Use draft_type from fetcher (preferred method)
        is_auction_series = (draft_df['draft_type'].str.lower() == 'auction')
        draft_df["savings"] = np.where(is_auction_series, cost_savings, pick_savings)
    else:
        # Fallback heuristic: if 25%+ of picks have non-zero cost, treat as auction
        nonzero_cost = cost.fillna(0).gt(0).sum()
        is_auction = nonzero_cost >= max(1, int(len(draft_df) * 0.25))

        if is_auction:
            draft_df["savings"] = cost_savings
            print("  [INFO] Detected auction draft (using cost_savings for unified 'savings' metric)")
        else:
            draft_df["savings"] = pick_savings
            print("  [INFO] Detected snake draft (using pick_savings for unified 'savings' metric)")

    value_count = draft_df['savings'].notna().sum()
    print(f"  [OK] Calculated value metrics for {value_count:,} draft picks")

    return draft_df


def main(args):
    """Main entry point for draft value metrics calculation."""
    print("\n" + "="*80)
    print("DRAFT VALUE METRICS V3 - SPAR-BASED ANALYSIS")
    print("="*80 + "\n")

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"[League] {ctx.league_name} ({ctx.league_id})")
    print(f"[Dir] Data directory: {ctx.data_directory}")

    # Check for required files
    draft_file = ctx.canonical_draft_file
    if not draft_file.exists():
        raise FileNotFoundError(f"[ERROR] Draft file not found: {draft_file}")

    # Find replacement levels file
    transformations_dir = Path(ctx.data_directory) / 'transformations'
    replacement_file = transformations_dir / 'replacement_levels_season.parquet'

    if not replacement_file.exists():
        raise FileNotFoundError(
            f"[ERROR] Replacement levels file not found: {replacement_file}\n"
            f"   Make sure replacement_level_v2.py has run first!"
        )

    # Find league settings
    league_settings_dir = Path(ctx.data_directory) / 'league_settings'
    settings_files = sorted(league_settings_dir.glob('league_settings_*.json'))
    if not settings_files:
        raise FileNotFoundError(f"[ERROR] No league settings files found in {league_settings_dir}")

    league_settings_path = settings_files[-1]
    print(f"[Settings] Using league settings: {league_settings_path.name}")

    # Backup if requested
    if args.backup and draft_file.exists():
        backup_file(draft_file)

    # Dry run check
    if args.dry_run:
        print("\n[DRY RUN] - No changes will be made")
        print(f"   Would read from: {draft_file}")
        print(f"   Would read replacement levels from: {replacement_file}")
        print(f"   Would write to: {draft_file}")
        return

    # Load data
    print(f"\n[Loading] Draft data from {draft_file.name}...")
    draft_df = pd.read_parquet(draft_file)
    print(f"   Loaded {len(draft_df):,} draft picks")

    # CLEANUP: Remove any _x/_y suffix columns from previous merge issues
    # This makes the pipeline idempotent (can run multiple times safely)
    xy_cols = [c for c in draft_df.columns if c.endswith('_x') or c.endswith('_y')]
    if xy_cols:
        print(f"   [CLEANUP] Removing {len(xy_cols)} duplicate suffix columns: {xy_cols[:5]}{'...' if len(xy_cols) > 5 else ''}")
        # For each _x/_y pair, keep the non-suffixed version if it exists, otherwise use _x
        for col in xy_cols:
            base_col = col[:-2]  # Remove _x or _y suffix
            if base_col not in draft_df.columns:
                # Use the _x version as the canonical one (arbitrarily)
                if col.endswith('_x'):
                    draft_df[base_col] = draft_df[col]
        draft_df = draft_df.drop(columns=xy_cols)
        print(f"   [CLEANUP] Now have {len(draft_df.columns)} columns")

    print(f"\n[Loading] Replacement levels from {replacement_file.name}...")
    replacement_df = pd.read_parquet(replacement_file)
    print(f"   Loaded {len(replacement_df):,} replacement baselines")

    # Validate required columns
    required_cols = ['year', 'yahoo_player_id']
    missing_cols = [c for c in required_cols if c not in draft_df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Missing required columns: {missing_cols}")

    # Check that performance metrics exist (from player_to_draft)
    if 'total_fantasy_points' not in draft_df.columns:
        print("[WARNING] total_fantasy_points not found - SPAR will be 0")
        print("   Make sure player_to_draft_v2.py has run before this script!")

    # Calculate SPAR metrics
    print("\n[Calculating] SPAR-based value metrics...")
    draft_df = calculate_all_draft_metrics(
        draft_df, replacement_df, league_settings_path
    )
    print("   [OK] Added: replacement_ppg, spar, pgvor, cost_norm, draft_roi")
    print("   [OK] Added: spar_per_dollar, spar_per_pick, spar_per_round")
    print("   [INFO] Keeper SPAR metrics will be calculated in PASS 4 (after keeper_price exists)")

    # Add keeper economics
    print("\n[Keeper] Calculating keeper economics...")
    if 'is_keeper_status' in draft_df.columns:
        draft_df = add_keeper_economics(draft_df)
        print("   [OK] Added: kept_next_year")
    else:
        print("   [WARNING] Skipping keeper economics (is_keeper_status column missing)")

    # Add position-specific cost buckets (percentile-based within position-year)
    print("\n[Stats] Assigning position-specific cost buckets...")
    draft_df = assign_cost_buckets(draft_df)
    bucket_count = draft_df['position_tier'].notna().sum()
    print(f"   [OK] Added: position_percentile, position_tier, position_tier_label for {bucket_count:,} drafted players")
    print(f"   [OK] Added: cost_bucket, cost_bucket_label (legacy columns for compatibility)")

    # Calculate draft value vs expectations (complementary to SPAR)
    draft_df = calculate_draft_value(draft_df)

    # Calculate draft grades (A-F based on SPAR percentile among drafted players)
    print("\n[Grades] Calculating draft grades...")
    draft_df = calculate_draft_grades(
        draft_df,
        spar_column='manager_spar',
        drafted_only=True,
        draft_indicator_column='pick'
    )
    grade_count = draft_df['draft_grade'].notna().sum()
    print(f"   [OK] Added: spar_percentile, draft_grade for {grade_count:,} drafted players")

    # Calculate value tiers (Steal/Good/Fair/Reach/Bust based on draft rank vs finish rank)
    print("\n[Value Tiers] Calculating value tiers...")
    draft_df = calculate_value_tiers(draft_df)
    tier_count = draft_df['value_tier'].notna().sum()
    print(f"   [OK] Added: value_tier for {tier_count:,} drafted players")

    # Calculate draft flags (breakout/bust) and tiers (early/mid/late) - dynamic based on actual rounds
    print("\n[Draft Flags] Calculating dynamic breakout/bust flags and draft tiers...")
    draft_df = calculate_all_draft_flags(draft_df)
    breakout_count = draft_df['is_breakout'].sum() if 'is_breakout' in draft_df.columns else 0
    bust_count = draft_df['is_bust'].sum() if 'is_bust' in draft_df.columns else 0
    injury_bust_count = draft_df['is_injury_bust'].sum() if 'is_injury_bust' in draft_df.columns else 0
    perf_bust_count = draft_df['is_performance_bust'].sum() if 'is_performance_bust' in draft_df.columns else 0
    print(f"   [OK] Added: round_percentile, is_breakout ({breakout_count:,}), is_bust ({bust_count:,}), draft_tier")
    print(f"   [OK] Added: games_missed, is_injured, is_injury_bust ({injury_bust_count:,}), is_performance_bust ({perf_bust_count:,})")

    # Calculate manager-level draft grades (aggregates player grades to manager-year)
    print("\n[Manager Grades] Calculating manager draft grades...")
    draft_df = calculate_manager_draft_grades(draft_df)
    manager_years = draft_df[['year', 'manager', 'manager_draft_grade']].dropna().drop_duplicates()
    print(f"   [OK] Added: manager_draft_score, manager_draft_percentile, manager_draft_grade")
    print(f"   [OK] Graded {len(manager_years):,} manager-year combinations")

    # Show manager leaderboard for most recent year
    if 'manager_draft_grade' in draft_df.columns:
        recent_year = draft_df['year'].max()
        leaderboard = get_manager_draft_leaderboard(draft_df, year=recent_year)
        if not leaderboard.empty:
            print(f"\n[Leaderboard] {recent_year} Draft Grades:")
            display_lb_cols = ['manager', 'manager_draft_grade', 'manager_draft_score', 'manager_total_spar', 'manager_hit_rate']
            available_lb = [c for c in display_lb_cols if c in leaderboard.columns]
            print(leaderboard[available_lb].to_string(index=False))

    # Calculate starter/backup designation based on position rank within manager's draft
    print("\n[Starter Designation] Calculating drafted as starter vs backup...")
    draft_df = calculate_starter_designation(
        draft_df,
        league_settings_path=league_settings_path,
        group_columns=['year', 'manager'],
        position_column='yahoo_position'
    )
    starter_cols = ['position_draft_rank', 'position_draft_label', 'drafted_as_starter', 'drafted_as_backup']
    added_cols = [c for c in starter_cols if c in draft_df.columns]
    print(f"   [OK] Added: {', '.join(added_cols)}")

    # Show SPAR comparison between starters and backups
    if 'drafted_as_starter' in draft_df.columns and 'manager_spar' in draft_df.columns:
        try:
            spar_by_role = get_spar_by_starter_designation(
                draft_df,
                position_column='yahoo_position',
                spar_column='manager_spar'
            )
            print("\n[SPAR by Starter/Backup Designation]:")
            pivot_cols = ['yahoo_position', 'designation', 'count', 'median_spar']
            pivot_display = [c for c in pivot_cols if c in spar_by_role.columns]
            # Show key positions only
            key_positions = ['QB', 'RB', 'WR', 'TE']
            spar_summary = spar_by_role[spar_by_role['yahoo_position'].isin(key_positions)]
            if not spar_summary.empty:
                print(spar_summary[pivot_display].to_string(index=False))
        except Exception as e:
            print(f"   [WARN] Could not calculate SPAR by designation: {e}")

    # Calculate bench insurance metrics (data-driven bench value analysis)
    print("\n[Bench Insurance] Calculating data-driven bench value metrics...")
    try:
        # Load player data if available for activation metrics
        player_file = ctx.canonical_player_file
        if player_file.exists():
            player_df = pd.read_parquet(player_file)
            bench_metrics = calculate_bench_insurance_metrics(
                player_df=player_df,
                draft_df=draft_df,
                position_column='yahoo_position',
                spar_column='manager_spar',
                year_column='year'
            )
        else:
            # Use draft data only (limited but still useful)
            bench_metrics = calculate_bench_insurance_metrics(
                player_df=draft_df,  # Will use limited metrics
                draft_df=draft_df,
                position_column='yahoo_position',
                spar_column='manager_spar',
                year_column='year'
            )

        # Add bench discount factor to draft data for each position
        if bench_metrics and 'position_discounts' in bench_metrics:
            position_discounts = bench_metrics['position_discounts']
            draft_df['bench_insurance_discount'] = draft_df['yahoo_position'].map(position_discounts)
            print(f"   [OK] Added: bench_insurance_discount column")

            # Calculate pre-computed bench_spar: max(0, manager_spar) * discount
            # This is the expected value of a backup at this position/tier
            if 'manager_spar' in draft_df.columns:
                draft_df['bench_spar'] = (
                    draft_df['manager_spar'].clip(lower=0) *
                    draft_df['bench_insurance_discount'].fillna(0)
                )
                print(f"   [OK] Added: bench_spar column (pre-computed bench value)")

            # Add position-level failure rates and activation rates
            failure_rates = bench_metrics.get('failure_rates', pd.DataFrame())
            if not failure_rates.empty and 'yahoo_position' in failure_rates.columns:
                # Use combined_failure_rate (bust OR injury)
                rate_col = 'combined_failure_rate' if 'combined_failure_rate' in failure_rates.columns else 'failure_rate'
                if rate_col in failure_rates.columns:
                    failure_rate_map = dict(zip(
                        failure_rates['yahoo_position'],
                        failure_rates[rate_col]
                    ))
                    draft_df['position_failure_rate'] = draft_df['yahoo_position'].map(failure_rate_map)
                    print(f"   [OK] Added: position_failure_rate column")

            activation_metrics = bench_metrics.get('activation_metrics', pd.DataFrame())
            if not activation_metrics.empty and 'yahoo_position' in activation_metrics.columns:
                activation_rate_map = dict(zip(
                    activation_metrics['yahoo_position'],
                    activation_metrics['activation_rate']
                ))
                draft_df['position_activation_rate'] = draft_df['yahoo_position'].map(activation_rate_map)
                print(f"   [OK] Added: position_activation_rate column")

            # Show the calculated discounts
            print("\n[Bench Insurance Values by Position]:")
            for pos in ['QB', 'RB', 'WR', 'TE']:
                discount = position_discounts.get(pos, 0)
                print(f"   {pos}: {discount:.1%} bench value vs starter")

            # Show failure rates if available
            if not failure_rates.empty:
                print("\n[Starter Failure Rates]:")
                for _, row in failure_rates.iterrows():
                    pos = row.get('yahoo_position', 'N/A')
                    rate = row.get('combined_failure_rate', row.get('failure_rate', 0))
                    count = row.get('total_starters', 0)
                    if pos in ['QB', 'RB', 'WR', 'TE']:
                        print(f"   {pos}: {rate:.1%} failure rate ({count} starters)")
        else:
            print("   [WARN] Could not calculate bench insurance metrics - using defaults")
    except Exception as e:
        print(f"   [WARN] Bench insurance calculation failed: {e}")

    # Calculate rank-based bench values (the purest data-driven approach)
    # This uses actual historical median SPAR for each draft slot (QB1, QB2, RB1, etc.)
    print("\n[Bench Value by Rank] Calculating median SPAR by position draft rank...")
    try:
        bench_value_by_rank = calculate_bench_value_by_rank(
            draft_df,
            position_column='yahoo_position',
            spar_column='manager_spar',
            rank_column='position_draft_label',
            min_sample_size=5
        )

        if not bench_value_by_rank.empty:
            # Create a mapping from position_draft_label to bench_value
            rank_to_bench_value = dict(zip(
                bench_value_by_rank['position_draft_label'],
                bench_value_by_rank['bench_value']
            ))

            # Map bench_value to each row based on position_draft_label
            draft_df['bench_value_by_rank'] = draft_df['position_draft_label'].map(rank_to_bench_value)
            print(f"   [OK] Added: bench_value_by_rank column (data-driven bench value per slot)")

            # Also add the median_spar for reference
            rank_to_median = dict(zip(
                bench_value_by_rank['position_draft_label'],
                bench_value_by_rank['median_spar']
            ))
            draft_df['slot_median_spar'] = draft_df['position_draft_label'].map(rank_to_median)
            print(f"   [OK] Added: slot_median_spar column (historical median for this slot)")
        else:
            print("   [WARN] Could not calculate rank-based bench values")
    except Exception as e:
        print(f"   [WARN] Rank-based bench value calculation failed: {e}")

    # Display sample
    print("\n[Stats] Sample draft value metrics:")
    display_cols = [
        'year', 'player', 'yahoo_position', 'pick', 'cost',
        'position_draft_label', 'drafted_as_starter',
        'position_percentile', 'position_tier_label',
        'total_fantasy_points', 'season_ppg',
        'replacement_ppg', 'spar', 'pgvor',
        'cost_norm', 'draft_roi', 'savings',
        'spar_percentile', 'draft_grade', 'value_tier',
        'draft_tier', 'is_breakout', 'is_bust',
        'games_played', 'games_missed', 'is_injury_bust', 'is_performance_bust',
        'bench_insurance_discount', 'bench_spar',
        'position_failure_rate', 'position_activation_rate',
        'bench_value_by_rank', 'slot_median_spar'
    ]
    available_cols = [c for c in display_cols if c in draft_df.columns]
    print(draft_df[available_cols].head(10).to_string(index=False))

    # Show top ROI picks (SPAR-based)
    if 'draft_roi' in draft_df.columns and draft_df['draft_roi'].notna().any():
        print("\n[Top 10] SPAR ROI Draft Picks:")
        top_roi = draft_df.nlargest(10, 'draft_roi')[available_cols]
        print(top_roi.to_string(index=False))

    # Show best draft capital efficiency picks (savings-based)
    if 'savings' in draft_df.columns and draft_df['savings'].notna().any():
        print("\n[Top 10] Draft Capital Efficiency (Best Savings):")
        savings_cols = ['year', 'player', 'pick', 'cost', 'spar', 'savings']
        savings_display = [c for c in savings_cols if c in draft_df.columns]
        top_savings = draft_df.nlargest(10, 'savings')[savings_display]
        print(top_savings.to_string(index=False))

    # Save updated draft file (both parquet and CSV)
    print(f"\n[Saving] Writing updated draft data to {draft_file}...")
    draft_df.to_parquet(draft_file, index=False)
    print(f"   [OK] Saved {len(draft_df):,} rows to parquet")

    # Also save CSV version
    csv_file = draft_file.with_suffix('.csv')
    draft_df.to_csv(csv_file, index=False)
    print(f"   [OK] Saved {len(draft_df):,} rows to CSV: {csv_file.name}")

    print("\n" + "="*80)
    print("[SUCCESS] DRAFT VALUE METRICS CALCULATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate SPAR-based draft value metrics'
    )
    parser.add_argument(
        '--context',
        required=True,
        help='Path to league_context.json'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before modifying files'
    )

    args = parser.parse_args()
    main(args)
