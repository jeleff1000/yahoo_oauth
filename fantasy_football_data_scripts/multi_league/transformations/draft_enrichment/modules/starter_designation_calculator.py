"""
Starter Designation Calculator Module

Calculates whether a player was drafted as a starter or backup based on:
1. Position rank within the manager's draft (QB1, QB2, RB1, RB2, etc.)
2. Cross-reference with league roster structure (starter slots per position)

This enables more accurate SPAR analysis by distinguishing:
- Tier 3 QB drafted as STARTER (budget builds) → expect high utilization
- Tier 3 QB drafted as BACKUP → expect low utilization

Key Features:
- Automatically ranks players by position within each manager-year draft
- Uses draft order (pick) for snake drafts, cost for auction drafts
- Cross-references with roster_positions from league settings
- Handles FLEX-eligible positions (RB, WR, TE can fill FLEX slots)
- Adds: position_draft_rank, drafted_as_starter, drafted_as_backup

Usage:
    from modules.starter_designation_calculator import calculate_starter_designation
    df = calculate_starter_designation(df, league_settings_path)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union


# Default roster structure if settings not found
DEFAULT_ROSTER = {
    'QB': 1,
    'RB': 2,
    'WR': 3,
    'TE': 1,
    'K': 1,
    'DEF': 1,
    'FLEX': 1,  # W/R/T slot
    'BN': 6
}

# Positions eligible for FLEX slot
FLEX_ELIGIBLE_POSITIONS = ['RB', 'WR', 'TE']

# Position standardization mapping
POSITION_ALIASES = {
    'W/R/T': 'FLEX',
    'W/R': 'FLEX',
    'R/W/T': 'FLEX',
    'FLEX': 'FLEX',
    'Q/W/R/T': 'SUPERFLEX',
    'OP': 'SUPERFLEX',
    'SUPERFLEX': 'SUPERFLEX',
    'D': 'DEF',
    'DST': 'DEF',
    'D/ST': 'DEF',
    'PK': 'K'
}


def load_roster_structure(league_settings_path: Union[str, Path]) -> Dict[str, int]:
    """
    Load roster structure from league settings JSON file.

    Args:
        league_settings_path: Path to league_settings_YYYY.json file

    Returns:
        Dict mapping position -> count (e.g., {'QB': 1, 'RB': 2, 'WR': 3, ...})
    """
    try:
        with open(league_settings_path, 'r') as f:
            settings = json.load(f)

        roster = {}
        flex_count = 0
        superflex_count = 0

        for pos_config in settings.get('roster_positions', []):
            pos = pos_config.get('position', '').upper()
            count = int(pos_config.get('count', 0))

            # Skip bench/IR positions
            if pos in ['BN', 'IR', 'IL']:
                roster[pos] = count
                continue

            # Standardize position name
            std_pos = POSITION_ALIASES.get(pos, pos)

            if std_pos == 'FLEX':
                flex_count += count
            elif std_pos == 'SUPERFLEX':
                superflex_count += count
            else:
                roster[std_pos] = roster.get(std_pos, 0) + count

        # Store flex counts separately
        roster['FLEX'] = flex_count
        roster['SUPERFLEX'] = superflex_count

        return roster

    except Exception as e:
        print(f"  [WARN] Could not load roster structure: {e}")
        print(f"  [WARN] Using default roster structure")
        return DEFAULT_ROSTER.copy()


def get_starter_slots(roster: Dict[str, int], position: str) -> int:
    """
    Get total starter slots available for a position (including FLEX eligibility).

    Args:
        roster: Dict mapping position -> count
        position: Player position (QB, RB, WR, TE, K, DEF)

    Returns:
        Total starter slots this position can occupy
    """
    # Get dedicated slots for this position
    dedicated_slots = roster.get(position, 0)

    # Add FLEX slots if position is FLEX-eligible
    flex_slots = 0
    if position in FLEX_ELIGIBLE_POSITIONS:
        flex_slots = roster.get('FLEX', 0)

    # Add SUPERFLEX slots if position is QB or FLEX-eligible
    superflex_slots = 0
    if position == 'QB' or position in FLEX_ELIGIBLE_POSITIONS:
        superflex_slots = roster.get('SUPERFLEX', 0)

    return dedicated_slots + flex_slots + superflex_slots


def calculate_position_draft_rank(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None,
    position_column: str = 'yahoo_position',
    rank_by_column: str = 'pick',
    ascending: bool = True
) -> pd.DataFrame:
    """
    Calculate position rank within each manager's draft.

    Within each manager-year draft, ranks players by position based on
    draft order (pick) or cost. Creates QB1, QB2, RB1, RB2, etc.

    Args:
        df: Draft DataFrame
        group_columns: Columns to group by (default: ['year', 'manager'])
        position_column: Column containing player position
        rank_by_column: Column to rank by ('pick' for snake, 'cost' for auction)
        ascending: If True, lower values ranked first (pick order)
                   If False, higher values ranked first (cost - higher = earlier pick)

    Returns:
        DataFrame with position_draft_rank column added
    """
    df = df.copy()

    if group_columns is None:
        group_columns = ['year', 'manager']

    # Validate required columns
    missing = [c for c in group_columns + [position_column, rank_by_column] if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing columns for position rank: {missing}")
        df['position_draft_rank'] = pd.NA
        df['position_draft_label'] = pd.NA
        return df

    # Only process drafted players (pick is not null)
    if 'pick' in df.columns:
        drafted_mask = df['pick'].notna()
    else:
        drafted_mask = pd.Series(True, index=df.index)

    df['position_draft_rank'] = pd.NA
    df['position_draft_label'] = pd.NA

    if not drafted_mask.any():
        return df

    # Rank within manager-year-position groups
    group_cols_with_pos = group_columns + [position_column]

    # Ensure rank_by_column is numeric
    df[rank_by_column] = pd.to_numeric(df[rank_by_column], errors='coerce')

    # Calculate rank within position for each manager-year
    df.loc[drafted_mask, 'position_draft_rank'] = (
        df.loc[drafted_mask]
        .groupby(group_cols_with_pos, dropna=False)[rank_by_column]
        .rank(method='first', ascending=ascending)
    )

    # Create label (QB1, QB2, RB1, etc.)
    def make_label(row):
        pos = row[position_column]
        rank = row['position_draft_rank']
        if pd.isna(pos) or pd.isna(rank):
            return pd.NA
        return f"{pos}{int(rank)}"

    df.loc[drafted_mask, 'position_draft_label'] = (
        df.loc[drafted_mask].apply(make_label, axis=1)
    )

    return df


def calculate_starter_designation(
    df: pd.DataFrame,
    league_settings_path: Optional[Union[str, Path]] = None,
    roster_structure: Optional[Dict[str, int]] = None,
    group_columns: Optional[List[str]] = None,
    position_column: str = 'yahoo_position',
    rank_by_column: str = 'pick',
    use_cost_for_auction: bool = True
) -> pd.DataFrame:
    """
    Calculate starter/backup designation for each draft pick.

    This is the main entry point for starter designation calculation.

    TWO-PASS APPROACH for correct FLEX handling:
    1. First pass: Assign dedicated slots (QB, RB, WR, TE, K, DEF) by position rank
    2. Second pass: Pool FLEX-eligible players not yet starters, sort by draft capital,
       assign top N to FLEX slots

    Example with roster (QB:1, RB:2, WR:3, TE:1, FLEX:1, K:1, DEF:1):
    - Pass 1: QB1→starter, RB1-2→starter, WR1-3→starter, TE1→starter, K1→starter, DEF1→starter
    - Pass 2: Pool = [RB3, WR4, TE2, ...], sorted by cost/pick → top 1 fills FLEX

    This correctly handles FLEX competition (WR4 at $30 beats RB3 at $15).

    Args:
        df: Draft DataFrame
        league_settings_path: Path to league_settings_YYYY.json file
        roster_structure: Optional pre-loaded roster structure dict
        group_columns: Columns to group by (default: ['year', 'manager'])
        position_column: Column containing player position
        rank_by_column: Column to rank by (default: 'pick')
        use_cost_for_auction: If True and draft is auction, use cost (descending) for ranking

    Returns:
        DataFrame with these columns added:
        - position_draft_rank: Rank within position for this manager-year (1, 2, 3, ...)
        - position_draft_label: Label like "QB1", "RB2", "WR3"
        - starter_slots_available: How many dedicated starter slots for this position
        - drafted_as_starter: 1 if drafted to be a starter, 0 if backup
        - drafted_as_backup: 1 if drafted to be a backup, 0 if starter
        - is_flex_starter: 1 if filling a FLEX slot, 0 otherwise
    """
    df = df.copy()

    if group_columns is None:
        group_columns = ['year', 'manager']

    # Load roster structure
    if roster_structure is not None:
        roster = roster_structure
    elif league_settings_path is not None:
        roster = load_roster_structure(league_settings_path)
    else:
        print("  [WARN] No roster structure provided, using defaults")
        roster = DEFAULT_ROSTER.copy()

    print(f"  [INFO] Roster structure: {roster}")

    # Detect draft type for ranking method
    ascending = True  # Default: lower pick = earlier (snake)
    actual_rank_col = rank_by_column
    is_auction = False

    if use_cost_for_auction and 'cost' in df.columns:
        cost = pd.to_numeric(df['cost'], errors='coerce')
        has_cost = (cost.fillna(0) > 1).sum()
        is_auction = has_cost >= max(1, int(len(df) * 0.25))

        if is_auction:
            actual_rank_col = 'cost'
            ascending = False  # Higher cost = earlier pick in auction
            print("  [INFO] Detected auction draft - using cost for position ranking")
        else:
            print("  [INFO] Detected snake draft - using pick for position ranking")

    # Calculate position draft rank
    df = calculate_position_draft_rank(
        df,
        group_columns=group_columns,
        position_column=position_column,
        rank_by_column=actual_rank_col,
        ascending=ascending
    )

    # Get dedicated slots per position (NOT including FLEX)
    def get_dedicated_slots(pos):
        if pd.isna(pos):
            return 0
        return roster.get(str(pos).upper(), 0)

    df['starter_slots_available'] = df[position_column].apply(get_dedicated_slots)

    # Initialize columns
    df['drafted_as_starter'] = 0
    df['drafted_as_backup'] = 0
    df['is_flex_starter'] = 0

    # Get FLEX and SUPERFLEX slot counts
    flex_slots = roster.get('FLEX', 0)
    superflex_slots = roster.get('SUPERFLEX', 0)

    # Ensure rank column is numeric
    df[actual_rank_col] = pd.to_numeric(df[actual_rank_col], errors='coerce')

    # Process each manager-year group with two-pass approach
    def assign_starters_two_pass(group):
        group = group.copy()
        position_rank = pd.to_numeric(group['position_draft_rank'], errors='coerce')

        # === PASS 1: Assign dedicated position slots ===
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            dedicated = roster.get(pos, 0)
            pos_mask = (group[position_column] == pos) & (position_rank <= dedicated)
            group.loc[pos_mask, 'drafted_as_starter'] = 1

        # === PASS 2: Assign FLEX slots from pool of unassigned FLEX-eligible players ===
        if flex_slots > 0:
            # Pool: FLEX-eligible (RB/WR/TE) players not yet marked as starters
            flex_pool_mask = (
                group[position_column].isin(FLEX_ELIGIBLE_POSITIONS) &
                (group['drafted_as_starter'] == 0) &
                group[actual_rank_col].notna()
            )
            flex_pool = group[flex_pool_mask].copy()

            if len(flex_pool) > 0:
                # Sort by draft capital (cost desc for auction, pick asc for snake)
                if is_auction:
                    flex_pool = flex_pool.sort_values(actual_rank_col, ascending=False)
                else:
                    flex_pool = flex_pool.sort_values(actual_rank_col, ascending=True)

                # Assign top N to FLEX slots
                flex_starters = flex_pool.head(flex_slots).index
                group.loc[flex_starters, 'drafted_as_starter'] = 1
                group.loc[flex_starters, 'is_flex_starter'] = 1

        # === PASS 3: Assign SUPERFLEX slots (QB or FLEX-eligible) ===
        if superflex_slots > 0:
            # Pool: QB or FLEX-eligible not yet starters
            superflex_pool_mask = (
                (group[position_column].isin(['QB'] + FLEX_ELIGIBLE_POSITIONS)) &
                (group['drafted_as_starter'] == 0) &
                group[actual_rank_col].notna()
            )
            superflex_pool = group[superflex_pool_mask].copy()

            if len(superflex_pool) > 0:
                if is_auction:
                    superflex_pool = superflex_pool.sort_values(actual_rank_col, ascending=False)
                else:
                    superflex_pool = superflex_pool.sort_values(actual_rank_col, ascending=True)

                superflex_starters = superflex_pool.head(superflex_slots).index
                group.loc[superflex_starters, 'drafted_as_starter'] = 1
                group.loc[superflex_starters, 'is_flex_starter'] = 1  # Mark as flex-type starter

        # Mark remaining as backups
        group.loc[group['drafted_as_starter'] == 0, 'drafted_as_backup'] = 1

        return group

    # Apply two-pass logic per manager-year
    df = df.groupby(group_columns, group_keys=False).apply(assign_starters_two_pass)

    # Handle any rows with missing data
    df['drafted_as_starter'] = df['drafted_as_starter'].fillna(0).astype(int)
    df['drafted_as_backup'] = df['drafted_as_backup'].fillna(0).astype(int)
    df['is_flex_starter'] = df['is_flex_starter'].fillna(0).astype(int)

    # Summary stats
    starter_count = (df['drafted_as_starter'] == 1).sum()
    backup_count = (df['drafted_as_backup'] == 1).sum()
    flex_count = (df['is_flex_starter'] == 1).sum()
    print(f"  [OK] Designated {starter_count:,} starters ({flex_count:,} via FLEX), {backup_count:,} backups")

    return df


def get_starter_designation_summary(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None,
    position_column: str = 'yahoo_position'
) -> pd.DataFrame:
    """
    Get summary of starter/backup designations by position.

    Args:
        df: DataFrame with drafted_as_starter column
        group_columns: Optional columns to group by (in addition to position)
        position_column: Column containing player position

    Returns:
        DataFrame with designation counts and percentages by position
    """
    if 'drafted_as_starter' not in df.columns:
        raise ValueError("drafted_as_starter column not found. Run calculate_starter_designation first.")

    # Build aggregation columns
    agg_cols = [position_column]
    if group_columns:
        agg_cols = group_columns + agg_cols

    summary = df.groupby(agg_cols, dropna=False).agg(
        total_drafted=('drafted_as_starter', 'count'),
        drafted_as_starter=('drafted_as_starter', 'sum'),
        drafted_as_backup=('drafted_as_backup', 'sum')
    ).reset_index()

    summary['starter_pct'] = (summary['drafted_as_starter'] / summary['total_drafted'] * 100).round(1)
    summary['backup_pct'] = (summary['drafted_as_backup'] / summary['total_drafted'] * 100).round(1)

    return summary


def get_spar_by_starter_designation(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None,
    position_column: str = 'yahoo_position',
    spar_column: str = 'manager_spar'
) -> pd.DataFrame:
    """
    Get SPAR statistics split by starter/backup designation.

    This shows the difference in realized value between players
    drafted as starters vs those drafted as backups.

    Args:
        df: DataFrame with drafted_as_starter and SPAR columns
        group_columns: Optional columns to group by (in addition to position)
        position_column: Column containing player position
        spar_column: Column containing SPAR values

    Returns:
        DataFrame with SPAR stats by position and designation
    """
    required_cols = ['drafted_as_starter', spar_column]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build aggregation columns
    agg_cols = [position_column, 'drafted_as_starter']
    if group_columns:
        agg_cols = group_columns + agg_cols

    # Ensure spar is numeric
    df = df.copy()
    df[spar_column] = pd.to_numeric(df[spar_column], errors='coerce')

    summary = df.groupby(agg_cols, dropna=False).agg(
        count=(spar_column, 'count'),
        mean_spar=(spar_column, 'mean'),
        median_spar=(spar_column, 'median'),
        sum_spar=(spar_column, 'sum'),
        std_spar=(spar_column, 'std')
    ).reset_index()

    # Add label for clarity
    summary['designation'] = summary['drafted_as_starter'].map({
        1: 'Drafted as Starter',
        0: 'Drafted as Backup'
    })

    return summary.round(2)
