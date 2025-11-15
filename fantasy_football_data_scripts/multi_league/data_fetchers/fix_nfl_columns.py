#!/usr/bin/env python3
"""
Quick script to fix column names in existing nfl_stats_merged files.
"""
import sys
from pathlib import Path
import pandas as pd

# Get paths from arguments
if len(sys.argv) < 2:
    print("Usage: python fix_nfl_columns.py <path_to_nfl_stats_merged_file>")
    sys.exit(1)

input_file = Path(sys.argv[1])

if not input_file.exists():
    print(f"File not found: {input_file}")
    sys.exit(1)

print(f"Loading: {input_file}")
df = pd.read_parquet(input_file)
print(f"Original shape: {df.shape}")
print(f"Original columns (first 20): {list(df.columns[:20])}")

# Rename columns to match expected format
column_renames = {
    'season': 'year',
    'team': 'nfl_team',
    'opponent_team': 'opponent_nfl_team',
    'player_id': 'NFL_player_id',
}

# Apply renames intelligently - preserve data from whichever column has it
renames_to_apply = {}
for old, new in column_renames.items():
    if old in df.columns:
        if new in df.columns:
            # Both exist - check which has data
            old_has_data = df[old].notna().any()
            new_has_data = df[new].notna().any()

            if old_has_data and not new_has_data:
                # Old has data, new doesn't - drop new and rename old
                print(f"Dropping empty column '{new}', keeping data from '{old}'")
                df = df.drop(columns=[new])
                renames_to_apply[old] = new
            elif not old_has_data and new_has_data:
                # New has data, old doesn't - drop old
                print(f"Dropping empty column '{old}', keeping data in '{new}'")
                df = df.drop(columns=[old])
            elif old_has_data and new_has_data:
                # Both have data - prefer new, drop old
                print(f"Both '{old}' and '{new}' have data, keeping '{new}'")
                df = df.drop(columns=[old])
            else:
                # Neither has data - just drop old
                print(f"Both '{old}' and '{new}' are empty, dropping '{old}'")
                df = df.drop(columns=[old])
        else:
            # Only old exists - rename it
            renames_to_apply[old] = new

if renames_to_apply:
    df = df.rename(columns=renames_to_apply)
    print(f"\nRenamed columns:")
    for old, new in renames_to_apply.items():
        print(f"  {old} -> {new}")
else:
    print("\nNo columns to rename (already correct)")

# Ensure 'player' column exists and has data
if 'player' not in df.columns or not df['player'].notna().any():
    if 'player' in df.columns and not df['player'].notna().any():
        df = df.drop(columns=['player'])
        print("Dropped empty 'player' column")

    if 'player_display_name' in df.columns and df['player_display_name'].notna().any():
        df['player'] = df['player_display_name']
        print("Created 'player' from 'player_display_name'")
    elif 'player_name' in df.columns and df['player_name'].notna().any():
        df['player'] = df['player_name']
        print("Created 'player' from 'player_name'")
    else:
        print("WARNING: No player name column with data found!")

print(f"\nFinal columns (first 20): {list(df.columns[:20])}")
print(f"Final shape: {df.shape}")

# Save back to the same file
print(f"\nSaving to: {input_file}")
df.to_parquet(input_file, index=False)

print("Done!")
