"""
Player to Transactions Import (Multi-League V2)

Enriches transaction data with player performance before/after the transaction.

This transformation adds performance context to each transaction:
- Position and team info at transaction time
- Points in week of transaction
- PPG before transaction (last 4 weeks)
- PPG after transaction (next 4 weeks)
- Total points rest of season after transaction
- Position rank at transaction (weekly rank that week)
- Position rank before transaction (season total rank before transaction)
- Position rank after transaction (rest of season rank after transaction)

Join Key: (yahoo_player_id, year, cumulative_week)
Critical: Transaction week must match player cumulative_week

Usage:
    python player_to_transactions_v2.py --context path/to/league_context.json
    python player_to_transactions_v2.py --context path/to/league_context.json --dry-run
    python player_to_transactions_v2.py --context path/to/league_context.json --backup
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import sys

import pandas as pd
import numpy as np



# Add parent directories to path for imports
_script_file = Path(__file__).resolve()

# Auto-detect if we're in modules/ subdirectory or transformations/ directory
if _script_file.parent.name == 'modules':
    # We're in multi_league/transformations/modules/
    _modules_dir = _script_file.parent
    _transformations_dir = _modules_dir.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'transformations':
    # We're in multi_league/transformations/
    _transformations_dir = _script_file.parent
    _multi_league_dir = _transformations_dir.parent
else:
    # Fallback: assume we're somewhere in the tree, navigate up to find multi_league
    _current = _script_file.parent
    while _current.name != 'multi_league' and _current.parent != _current:
        _current = _current.parent
    _multi_league_dir = _current

_scripts_dir = _multi_league_dir.parent  # fantasy_football_data_scripts directory
sys.path.insert(0, str(_scripts_dir))  # Allows: from multi_league.core.XXX
sys.path.insert(0, str(_multi_league_dir))  # Allows: from core.XXX

from core.league_context import LeagueContext


# =========================================================
# Helper Functions
# =========================================================

def backup_file(path: Path) -> Path:
    """Create timestamped backup of file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    dest.write_bytes(path.read_bytes())
    return dest


def get_player_performance_metrics(
    yahoo_player_id: str,
    trans_week: int,
    year: int,
    player_df: pd.DataFrame
) -> Dict:
    """
    Calculate player performance metrics relative to transaction week.

    Focuses on FUTURE performance (rest of season after transaction).

    Args:
        yahoo_player_id: Yahoo player ID
        trans_week: Transaction cumulative week
        year: Season year
        player_df: Full player DataFrame

    Returns:
        Dict of performance metrics
    """
    # Filter to this player and year
    player_data = player_df[
        (player_df['yahoo_player_id'] == yahoo_player_id) &
        (player_df['year'] == year)
    ].copy()

    if player_data.empty:
        return {}

    # Get position (most common) - handle empty mode result
    if 'position' in player_data.columns:
        position_mode = player_data['position'].mode()
        position = position_mode.iloc[0] if not position_mode.empty else None
    else:
        position = None

    # Data slices
    at_trans = player_data[player_data['cumulative_week'] == trans_week]
    before_4wks = player_data[
        (player_data['cumulative_week'] < trans_week) &
        (player_data['cumulative_week'] >= trans_week - 4)
    ]
    after_4wks = player_data[
        (player_data['cumulative_week'] > trans_week) &
        (player_data['cumulative_week'] <= trans_week + 4)
    ]
    rest_of_season = player_data[player_data['cumulative_week'] > trans_week]

    metrics = {}

    # Position and team context
    if not at_trans.empty:
        metrics['position'] = position
        metrics['nfl_team'] = at_trans['nfl_team'].iloc[0] if 'nfl_team' in at_trans.columns else None
        metrics['points_at_transaction'] = at_trans['fantasy_points'].iloc[0] if 'fantasy_points' in at_trans.columns else None

    # Before transaction (sunk cost - less important)
    if not before_4wks.empty and 'fantasy_points' in before_4wks.columns:
        metrics['ppg_before_transaction'] = round(before_4wks['fantasy_points'].mean(), 2)
        metrics['weeks_before'] = len(before_4wks)

    # After transaction (KEY METRICS - what you're getting)
    if not after_4wks.empty and 'fantasy_points' in after_4wks.columns:
        metrics['ppg_after_transaction'] = round(after_4wks['fantasy_points'].mean(), 2)
        metrics['total_points_after_4wks'] = round(after_4wks['fantasy_points'].sum(), 2)
        metrics['weeks_after'] = len(after_4wks)

    # Rest of season (total value acquired/lost)
    if not rest_of_season.empty and 'fantasy_points' in rest_of_season.columns:
        metrics['total_points_rest_of_season'] = round(rest_of_season['fantasy_points'].sum(), 2)
        metrics['ppg_rest_of_season'] = round(rest_of_season['fantasy_points'].mean(), 2)
        metrics['weeks_rest_of_season'] = len(rest_of_season)

    # Position rank at transaction (weekly rank based on that week's points)
    if position and not at_trans.empty:
        # Get all players at same position this week
        same_pos_week = player_df[
            (player_df['cumulative_week'] == trans_week) &
            (player_df['year'] == year) &
            (player_df['position'] == position) &
            (player_df['fantasy_points'].notna())
        ].copy()

        if not same_pos_week.empty:
            same_pos_week = same_pos_week.sort_values('fantasy_points', ascending=False).reset_index(drop=True)
            player_idx = same_pos_week[same_pos_week['yahoo_player_id'] == yahoo_player_id].index

            if len(player_idx) > 0:
                metrics['position_rank_at_transaction'] = int(player_idx[0] + 1)
                metrics['position_total_players'] = len(same_pos_week)

    # Position rank BEFORE transaction (based on season performance up to transaction)
    if position and not before_4wks.empty:
        # Get all players at same position for season before transaction
        season_before = player_df[
            (player_df['cumulative_week'] < trans_week) &
            (player_df['year'] == year) &
            (player_df['position'] == position)
        ].copy()

        if not season_before.empty:
            # Aggregate total points before transaction per player
            season_before_agg = season_before.groupby('yahoo_player_id')['fantasy_points'].sum().reset_index()
            season_before_agg = season_before_agg.sort_values('fantasy_points', ascending=False).reset_index(drop=True)

            player_idx = season_before_agg[season_before_agg['yahoo_player_id'] == yahoo_player_id].index
            if len(player_idx) > 0:
                metrics['position_rank_before_transaction'] = int(player_idx[0] + 1)

    # Position rank AFTER transaction (based on rest of season performance)
    if position and not rest_of_season.empty:
        # Get all players at same position for rest of season
        season_after = player_df[
            (player_df['cumulative_week'] > trans_week) &
            (player_df['year'] == year) &
            (player_df['position'] == position)
        ].copy()

        if not season_after.empty:
            # Aggregate total points after transaction per player
            season_after_agg = season_after.groupby('yahoo_player_id')['fantasy_points'].sum().reset_index()
            season_after_agg = season_after_agg.sort_values('fantasy_points', ascending=False).reset_index(drop=True)

            player_idx = season_after_agg[season_after_agg['yahoo_player_id'] == yahoo_player_id].index
            if len(player_idx) > 0:
                metrics['position_rank_after_transaction'] = int(player_idx[0] + 1)

    return metrics


def left_join_transactions_player(left: pd.DataFrame, right: pd.DataFrame, import_cols: list[str]) -> pd.DataFrame:
    """Left-join player-level performance (right) onto transactions (left).

    Strategy:
    1. Ensure both frames have a `player_week` key if possible (yahoo_player_id + '_' + cumulative_week).
    2. Prefer joining on (yahoo_player_id, cumulative_week) when that key exists in both
       frames and is cleanly 1:1 on the right (no nulls, no duplicates).
    3. If that isn't possible, fall back to joining on `player_week` when available.
    4. As a last resort, attempt a best-effort merge on available common keys.
    """
    L = left.copy()
    R = right.copy()

    # Filter import_cols to those present in R to avoid KeyErrors
    available_imports = [c for c in import_cols if c in R.columns]
    if not available_imports:
        return L

    # Build player_week if missing - use player_name + year + week format
    # Note: Only build if missing - preserve existing player_week if present
    for df in (L, R):
        if 'player_week' not in df.columns:
            # New format: player_name + year + week (e.g., "LamarJackson202104")
            if {'player_name', 'year', 'week'}.issubset(df.columns):
                df['player_week'] = df.apply(
                    lambda row: f"{str(row['player_name']).replace(' ', '')}{int(row['year'])}{int(row['week']):02d}"
                    if pd.notna(row['player_name']) and pd.notna(row['year']) and pd.notna(row['week']) else '', axis=1
                )
            # Fallback to old format if player_name not available
            elif {'yahoo_player_id', 'cumulative_week'}.issubset(df.columns):
                df['player_week'] = df['yahoo_player_id'].astype(str) + "_" + df['cumulative_week'].astype(str)

    # Normalize merge key data types to prevent type mismatches (float64 vs Int64, object vs int64, etc.)
    # Convert: numeric -> Int64 -> string for consistent merging
    def normalize_merge_keys(df, keys):
        """Normalize merge keys to string via Int64 to handle type mismatches"""
        for key in keys:
            if key in df.columns and key != 'league_id':  # Skip league_id (already string)
                df[key] = pd.to_numeric(df[key], errors='coerce').astype('Int64').astype(str)
        return df

    # Attempt 1: join on (league_id, yahoo_player_id, year, cumulative_week) - multi-league safe
    keys1 = ['league_id', 'yahoo_player_id', 'year', 'cumulative_week']
    if set(keys1).issubset(L.columns) and set(keys1).issubset(R.columns):
        # Normalize types BEFORE checking nulls/duplicates
        L_norm = normalize_merge_keys(L.copy(), keys1)
        R_norm = normalize_merge_keys(R.copy(), keys1)

        # check for nulls and uniqueness on right
        left_has_nulls = L_norm[keys1].isnull().any(axis=1).any()
        right_has_nulls = R_norm[keys1].isnull().any(axis=1).any()
        right_has_dupes = R_norm.duplicated(subset=keys1, keep=False).any()
        if not left_has_nulls and not right_has_nulls and not right_has_dupes:
            return L_norm.merge(R_norm[keys1 + available_imports], on=keys1, how='left', suffixes=('_old', ''))

    # Fallback: Try without league_id for backward compatibility
    keys1_fallback = ['yahoo_player_id', 'year', 'cumulative_week']
    if set(keys1_fallback).issubset(L.columns) and set(keys1_fallback).issubset(R.columns):
        # Normalize types BEFORE checking nulls/duplicates
        L_norm = normalize_merge_keys(L.copy(), keys1_fallback)
        R_norm = normalize_merge_keys(R.copy(), keys1_fallback)

        left_has_nulls = L_norm[keys1_fallback].isnull().any(axis=1).any()
        right_has_nulls = R_norm[keys1_fallback].isnull().any(axis=1).any()
        right_has_dupes = R_norm.duplicated(subset=keys1_fallback, keep=False).any()
        if not left_has_nulls and not right_has_nulls and not right_has_dupes:
            return L_norm.merge(R_norm[keys1_fallback + available_imports], on=keys1_fallback, how='left', suffixes=('_old', ''))

    # Attempt 2: join on player_week
    keys2 = ['player_week']
    if set(keys2).issubset(L.columns) and set(keys2).issubset(R.columns):
        right_has_dupes_py = R.duplicated(subset=keys2, keep=False).any()
        if not right_has_dupes_py:
            return L.merge(R[keys2 + available_imports], on=keys2, how='left', suffixes=('_old', ''))

    # Last-resort: best-effort merge on any common candidates
    common_candidates = [k for k in ['player_week', 'yahoo_player_id', 'cumulative_week'] if k in L.columns and k in R.columns]
    if common_candidates:
        on_cols = common_candidates
        return L.merge(R[on_cols + available_imports], on=on_cols, how='left', suffixes=('_old', ''))

    return L

# =========================================================
# Main Import Function
# =========================================================

def enrich_transactions_with_player_data(
    player_path: Path,
    transactions_path: Path,
    dry_run: bool = False,
    make_backup: bool = False,
) -> Dict[str, int]:
    """
    Enrich transaction data with player performance metrics.

    Args:
        player_path: Path to player.parquet
        transactions_path: Path to transactions.parquet
        dry_run: If True, don't write changes
        make_backup: If True, create backup before writing

    Returns:
        Dict with statistics about the enrichment
    """
    print(f"Loading player data from: {player_path}")
    player = pd.read_parquet(player_path)
    print(f"  Loaded {len(player):,} player records")

    # Filter out historical NFL data (yahoo_player_id="None") to prevent duplicate key issues
    # These rows are useful for display but shouldn't be used for transaction enrichment
    # since transactions only contain real Yahoo player IDs
    if 'yahoo_player_id' in player.columns:
        initial_count = len(player)
        player = player[
            (player['yahoo_player_id'].notna()) &
            (player['yahoo_player_id'] != 'None') &
            (player['yahoo_player_id'] != '')
        ].copy()
        filtered_count = initial_count - len(player)
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count:,} historical NFL rows (yahoo_player_id='None')")
            print(f"  Working with {len(player):,} player records for enrichment")

    print(f"\nLoading transaction data from: {transactions_path}")
    transactions = pd.read_parquet(transactions_path)
    print(f"  Loaded {len(transactions):,} transaction records")

    # CRITICAL: Normalize all merge key data types for consistent comparisons
    # This ensures yahoo_player_id, year, and cumulative_week have matching types
    # between player and transactions dataframes (prevents 0 matches due to type mismatches)
    print(f"\nNormalizing merge key data types...")
    for df_name, df in (("player", player), ("transactions", transactions)):
        # Normalize yahoo_player_id to Int64
        if 'yahoo_player_id' in df.columns:
            try:
                df['yahoo_player_id'] = pd.to_numeric(df['yahoo_player_id'], errors='coerce').astype('Int64')
            except Exception as e:
                print(f"  WARNING: Could not convert yahoo_player_id to Int64 in {df_name}: {e}")

        # Normalize year to Int64
        if 'year' in df.columns:
            try:
                df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            except Exception as e:
                print(f"  WARNING: Could not convert year to Int64 in {df_name}: {e}")

        # Normalize cumulative_week to Int64
        if 'cumulative_week' not in df.columns:
            # Only derive if year and week are available
            if 'year' in df.columns and 'week' in df.columns:
                try:
                    df['cumulative_week'] = (
                        pd.to_numeric(df['year'], errors='coerce').astype('Int64').fillna(-1) * 100 +
                        pd.to_numeric(df['week'], errors='coerce').astype('Int64').fillna(-1)
                    ).astype('Int64')
                except Exception:
                    # Fallback to simple string concatenation if numeric fails
                    df['cumulative_week'] = (
                        df['year'].astype(str).fillna('') + '_' + df['week'].astype(str).fillna('')
                    ).astype('string')
            else:
                df['cumulative_week'] = pd.NA
        else:
            # cumulative_week exists but may be wrong type - ensure it's numeric
            try:
                df['cumulative_week'] = pd.to_numeric(df['cumulative_week'], errors='coerce').astype('Int64')
            except Exception as e:
                print(f"  WARNING: Could not convert cumulative_week to Int64 in {df_name}: {e}")

    # Validate required columns
    required_player = ['yahoo_player_id', 'year', 'cumulative_week']
    required_trans = ['yahoo_player_id', 'year', 'cumulative_week']

    missing_player = [c for c in required_player if c not in player.columns]
    missing_trans = [c for c in required_trans if c not in transactions.columns]

    if missing_player:
        raise KeyError(f"player.parquet missing required columns: {missing_player}")
    if missing_trans:
        raise KeyError(f"transactions.parquet missing required columns: {missing_trans}")

    print(f"\nEnriching {len(transactions):,} transactions with player performance data...")

    # Calculate metrics for each transaction
    enrichment_data = []

    for idx, trans_row in transactions.iterrows():
        if idx % 100 == 0:
            print(f"  Processing transaction {idx+1:,}/{len(transactions):,}...", end='\r')

        player_id = trans_row['yahoo_player_id']
        week = trans_row['cumulative_week']
        year = trans_row['year']

        # Skip if missing required data
        if pd.isna(player_id) or pd.isna(week) or pd.isna(year):
            enrichment_data.append({})
            continue

        metrics = get_player_performance_metrics(
            yahoo_player_id=player_id,
            trans_week=int(week),
            year=int(year),
            player_df=player
        )

        enrichment_data.append(metrics)

    print(f"\n  Completed enrichment for {len(transactions):,} transactions")

    # Convert to DataFrame
    enrichment_df = pd.DataFrame(enrichment_data)

    # Add enrichment columns to transactions
    for col in enrichment_df.columns:
        transactions[col] = enrichment_df[col]

    # Calculate additional derived metrics
    print(f"\nCalculating derived metrics...")

    # Points per FAAB dollar (for adds with FAAB)
    if 'faab_bid' in transactions.columns and 'total_points_rest_of_season' in transactions.columns:
        transactions['points_per_faab_dollar'] = np.where(
            transactions['faab_bid'] > 0,
            (transactions['total_points_rest_of_season'] / transactions['faab_bid']).round(2),
            np.nan
        )
        print(f"  Added points_per_faab_dollar")

    # Transaction quality score (simple heuristic)
    if 'ppg_after_transaction' in transactions.columns and 'transaction_type' in transactions.columns:
        def score_transaction(row):
            ppg = row.get('ppg_after_transaction', 0)
            trans_type = row.get('transaction_type', '')

            if pd.isna(ppg):
                return None

            if trans_type == 'add':
                # Good adds: high PPG
                if ppg >= 15:
                    return 5
                elif ppg >= 10:
                    return 4
                elif ppg >= 7:
                    return 3
                elif ppg >= 5:
                    return 2
                else:
                    return 1
            elif trans_type == 'drop':
                # Good drops: low PPG
                if ppg <= 3:
                    return 3
                elif ppg <= 5:
                    return 2
                elif ppg <= 7:
                    return 1
                else:
                    return -1  # Bad drop
            else:
                return None

        transactions['transaction_quality_score'] = transactions.apply(score_transaction, axis=1)
        print(f"  Added transaction_quality_score")

    # Calculate statistics
    stats = {
        "total_transactions": len(transactions),
        "transactions_with_player_data": enrichment_df['position'].notna().sum() if 'position' in enrichment_df.columns else 0,
        "transactions_with_future_data": enrichment_df['ppg_rest_of_season'].notna().sum() if 'ppg_rest_of_season' in enrichment_df.columns else 0,
        "match_rate_pct": round(enrichment_df['position'].notna().sum() / len(transactions) * 100, 2) if len(transactions) > 0 and 'position' in enrichment_df.columns else 0,
    }

    # Save (unless dry-run)
    if dry_run:
        print("\n" + "="*60)
        print("[DRY RUN] No files were written.")
        print("="*60)

        print("\nSample enriched transactions:")
        sample_cols = ['player', 'transaction_type', 'position', 'ppg_after_transaction',
                       'total_points_rest_of_season', 'transaction_quality_score']
        sample_cols = [c for c in sample_cols if c in transactions.columns]
        print(transactions[sample_cols].head(10).to_string())

        return stats

    if make_backup and transactions_path.exists():
        bpath = backup_file(transactions_path)
        print(f"\n[Backup Created] {bpath}")

    # Write back to transactions.parquet
    transactions.to_parquet(transactions_path, index=False)
    print(f"\n[SAVED] Updated transaction data written to: {transactions_path}")

    return stats


# =========================================================
# CLI Interface
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enrich transaction data with player performance metrics for multi-league setup"
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Path to league_context.json (if not provided, uses hardcoded paths)"
    )
    parser.add_argument(
        "--player",
        type=str,
        help="Override player data path"
    )
    parser.add_argument(
        "--transactions",
        type=str,
        help="Override transactions data path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create timestamped backup before saving"
    )

    args = parser.parse_args()

    # Determine file paths
    if args.context:
        ctx = LeagueContext.load(args.context)
        print(f"Loaded league context: {ctx.league_name}")

        player_path = Path(args.player) if args.player else ctx.canonical_player_file
        transactions_path = Path(args.transactions) if args.transactions else ctx.canonical_transaction_file
    else:
        # Fallback to hardcoded relative paths (V1 compatibility)
        if not args.player or not args.transactions:
            THIS_FILE = Path(__file__).resolve()
            SCRIPT_DIR = THIS_FILE.parent.parent.parent  # Up to fantasy_football_data_scripts
            ROOT_DIR = SCRIPT_DIR.parent  # Up to fantasy_football_data_downloads
            DATA_DIR = ROOT_DIR / "fantasy_football_data"

            player_path = Path(args.player) if args.player else (DATA_DIR / "player.parquet")
            transactions_path = Path(args.transactions) if args.transactions else (DATA_DIR / "transactions.parquet")
        else:
            player_path = Path(args.player)
            transactions_path = Path(args.transactions)

    # Validate paths
    if not player_path.exists():
        raise FileNotFoundError(f"Player data not found: {player_path}")
    if not transactions_path.exists():
        raise FileNotFoundError(f"Transaction data not found: {transactions_path}")

    # Run enrichment
    print("\n" + "="*60)
    print("PLAYER TO TRANSACTIONS ENRICHMENT (V2)")
    print("="*60)

    stats = enrich_transactions_with_player_data(
        player_path=player_path,
        transactions_path=transactions_path,
        dry_run=args.dry_run,
        make_backup=args.backup
    )

    # Print final summary
    print("\n" + "="*60)
    print("ENRICHMENT SUMMARY")
    print("="*60)
    print(f"Total transactions:           {stats['total_transactions']:,}")
    print(f"With player data:             {stats['transactions_with_player_data']:,} ({stats['match_rate_pct']}%)")
    print(f"With future performance data: {stats['transactions_with_future_data']:,}")
    print("="*60)


if __name__ == "__main__":
    main()
