"""
Bye Week Filler Module

Fills in missing bye week rows in matchup data with appropriate values:
- Weekly stats: 0 (no game played)
- Cumulative stats: Carried forward from previous week
"""

import pandas as pd
import numpy as np
from typing import List


def fill_bye_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in missing bye week rows for teams.

    For bye weeks, creates rows with:
    - opponent = None (no opponent)
    - team_points = 0 (no game)
    - opponent_points = 0 (no game)
    - win/loss = 0 (no game)
    - Cumulative stats: Carried forward from previous week
    - Expected stats: Same as previous week (no change during bye)

    Args:
        df: Matchup DataFrame

    Returns:
        DataFrame with bye week rows added
    """
    print("\n[BYE WEEKS] Filling in missing bye week rows...")

    # DYNAMIC: Detect column categories from actual DataFrame columns
    # This makes the solution generic for any league size, week count, etc.
    all_cols = set(df.columns)

    # Weekly stat columns: Set to 0 for bye weeks (no game played)
    # Pattern: columns that represent single-week statistics
    weekly_stat_patterns = [
        'team_points', 'team_projected_points', 'opponent_points',
        'opponent_projected_points', 'margin', 'total_matchup_score',
        'close_margin', 'win', 'loss', 'proj_wins', 'proj_losses',
        'proj_score_error', 'abs_proj_score_error', 'above_proj_score',
        'below_proj_score', 'expected_spread', 'expected_odds'
    ]
    weekly_stat_cols = [col for col in weekly_stat_patterns if col in all_cols]

    # Cumulative stat columns: Carry forward from previous week (no change during bye)
    # Pattern: columns with "_to_date", "avg_", "vs_" that accumulate over season
    cumulative_stat_patterns = [
        'wins_to_date', 'losses_to_date', 'points_scored_to_date',
        'points_against_to_date', 'shuffle_avg_wins', 'shuffle_avg_seed',
        'shuffle_avg_playoffs', 'shuffle_avg_bye', 'wins_vs_shuffle_wins',
        'seed_vs_shuffle_seed', 'opp_shuffle_avg_wins', 'opp_shuffle_avg_seed',
        'opp_shuffle_avg_playoffs', 'opp_shuffle_avg_bye'
    ]
    cumulative_stat_cols = [col for col in cumulative_stat_patterns if col in all_cols]

    # DYNAMIC: Shuffle probability columns (carry forward)
    # Pattern: "shuffle_N_seed", "shuffle_N_win", "opp_shuffle_N_seed", "opp_shuffle_N_win"
    # Detect these dynamically to support any league size (10, 12, 14 teams, etc.)
    shuffle_prob_cols = [
        col for col in all_cols
        if (col.startswith('shuffle_') and (col.endswith('_seed') or col.endswith('_win'))) or
           (col.startswith('opp_shuffle_') and (col.endswith('_seed') or col.endswith('_win')))
    ]

    # Preserve columns: Team identifiers and playoff flags
    preserve_patterns = [
        'manager', 'team_name', 'year', 'league_id', 'is_playoffs',
        'is_consolation', 'championship', 'third_place'
    ]
    preserve_cols = [col for col in preserve_patterns if col in all_cols]

    print(f"  Detected {len(weekly_stat_cols)} weekly stat columns")
    print(f"  Detected {len(cumulative_stat_cols)} cumulative stat columns")
    print(f"  Detected {len(shuffle_prob_cols)} shuffle probability columns")

    # Track new rows
    bye_week_rows = []
    initial_count = len(df)

    # Process each season
    seasons = sorted(df['year'].unique())
    for year in seasons:
        df_year = df[df['year'] == year].copy()

        # Get all managers and weeks in this season
        all_managers = sorted(df_year['manager'].unique())
        all_weeks = sorted(df_year['week'].unique())

        if not all_managers or not all_weeks:
            continue

        print(f"  {year}: {len(all_managers)} managers, weeks {min(all_weeks)}-{max(all_weeks)}")

        # Check each manager for missing weeks
        for manager in all_managers:
            manager_data = df_year[df_year['manager'] == manager].sort_values('week')
            manager_weeks = set(manager_data['week'].tolist())
            missing_weeks = [w for w in all_weeks if w not in manager_weeks]

            if not missing_weeks:
                continue

            # Get reference row for this manager (most recent week)
            ref_row = manager_data.iloc[-1].to_dict() if len(manager_data) > 0 else None
            if not ref_row:
                continue

            print(f"    {manager}: Missing weeks {missing_weeks}")

            # Create bye week rows
            for bye_week in sorted(missing_weeks):
                # Start with a copy of the reference row
                bye_row = ref_row.copy()

                # Update week
                bye_row['week'] = bye_week

                # Set opponent to None
                bye_row['opponent'] = None

                # Set weekly stats to 0
                for col in weekly_stat_cols:
                    if col in bye_row:
                        if col in ['close_margin']:
                            bye_row[col] = 0  # Integer 0
                        else:
                            bye_row[col] = 0.0  # Float 0

                # For cumulative stats, carry forward from previous week (if available)
                prev_week_data = df_year[
                    (df_year['manager'] == manager) & (df_year['week'] < bye_week)
                ].sort_values('week', ascending=False)

                if len(prev_week_data) > 0:
                    prev_row = prev_week_data.iloc[0]

                    # Carry forward cumulative stats
                    for col in cumulative_stat_cols:
                        if col in prev_row and pd.notna(prev_row[col]):
                            bye_row[col] = prev_row[col]

                    # Carry forward shuffle probabilities
                    for col in shuffle_prob_cols:
                        if col in prev_row and pd.notna(prev_row[col]):
                            bye_row[col] = prev_row[col]

                # Set flags
                bye_row['is_bye_week'] = 1

                # Preserve team identifiers and playoff flags
                # (already copied from ref_row)

                bye_week_rows.append(bye_row)

    # Add bye week rows to dataframe
    if bye_week_rows:
        df_byes = pd.DataFrame(bye_week_rows)
        df = pd.concat([df, df_byes], ignore_index=True)

        # Sort by year, week, manager
        df = df.sort_values(['year', 'week', 'manager']).reset_index(drop=True)

        added_count = len(bye_week_rows)
        print(f"\n[BYE WEEKS] Added {added_count} bye week rows")
        print(f"[BYE WEEKS] Total rows: {initial_count} -> {len(df)} (+{added_count})")
    else:
        print("\n[BYE WEEKS] No bye weeks found")

    # Add is_bye_week column if it doesn't exist
    if 'is_bye_week' not in df.columns:
        df['is_bye_week'] = 0
    else:
        df['is_bye_week'] = df['is_bye_week'].fillna(0).astype(int)

    return df


def validate_bye_week_coverage(df: pd.DataFrame) -> None:
    """
    Validate that all teams have rows for all weeks (including bye weeks).

    Args:
        df: Matchup DataFrame with bye weeks filled
    """
    print("\n[BYE WEEKS] Validating coverage...")

    seasons = sorted(df['year'].unique())
    for year in seasons:
        df_year = df[df['year'] == year]

        all_managers = sorted(df_year['manager'].unique())
        all_weeks = sorted(df_year['week'].unique())

        # Check each manager
        missing_found = False
        for manager in all_managers:
            manager_weeks = set(df_year[df_year['manager'] == manager]['week'].tolist())
            missing_weeks = [w for w in all_weeks if w not in manager_weeks]

            if missing_weeks:
                print(f"  {year} - {manager}: Still missing weeks {missing_weeks}")
                missing_found = True

        if not missing_found:
            print(f"  {year}: OK - All {len(all_managers)} managers have rows for all {len(all_weeks)} weeks")


if __name__ == "__main__":
    # Test the module
    import sys
    from pathlib import Path

    # Add parent directory for imports
    SCRIPT_DIR = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(SCRIPT_DIR))

    from multi_league.core.league_context import LeagueContext

    # Load test data
    context_path = r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data\league_context.json"
    ctx = LeagueContext.load(context_path)

    matchup_file = ctx.data_directory / "matchup.parquet"
    df = pd.read_parquet(matchup_file)

    print(f"Loaded {len(df)} matchup rows")

    # Fill bye weeks
    df_filled = fill_bye_weeks(df)

    # Validate
    validate_bye_week_coverage(df_filled)

    # Show example
    print("\nExample bye week row:")
    bye_rows = df_filled[df_filled['is_bye_week'] == 1]
    if len(bye_rows) > 0:
        cols = ['manager', 'year', 'week', 'opponent', 'team_points', 'win',
                'shuffle_avg_wins', 'is_bye_week']
        print(bye_rows[cols].head(5).to_string(index=False))
