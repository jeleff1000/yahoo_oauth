#!/usr/bin/env python3
"""
Simple script to combine NFL offense and defense stats with correct column names.
"""
import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) < 4:
    print("Usage: python simple_combine_nfl.py <offense_file> <defense_file> <output_file>")
    sys.exit(1)

offense_file = Path(sys.argv[1])
defense_file = Path(sys.argv[2])
output_file = Path(sys.argv[3])

# Load files
print(f"Loading offense: {offense_file}")
offense_df = pd.read_parquet(offense_file)
print(f"  Shape: {offense_df.shape}")

print(f"Loading defense: {defense_file}")
defense_df = pd.read_parquet(defense_file)
print(f"  Shape: {defense_df.shape}")

# Rename offense columns
print("\n[OFFENSE] Renaming columns...")
offense_renames = {
    'season': 'year',
    'team': 'nfl_team',
    'opponent_team': 'opponent_nfl_team',
    'player_id': 'NFL_player_id',
}
offense_df = offense_df.rename(columns={k: v for k, v in offense_renames.items() if k in offense_df.columns})

# Create player column from player_display_name or player_name
if 'player' not in offense_df.columns:
    if 'player_display_name' in offense_df.columns:
        offense_df['player'] = offense_df['player_display_name']
        print("  Created 'player' from 'player_display_name'")
    elif 'player_name' in offense_df.columns:
        offense_df['player'] = offense_df['player_name']
        print("  Created 'player' from 'player_name'")

# Rename defense columns
print("\n[DEFENSE] Renaming columns...")
defense_renames = {
    'season': 'year',
    'team': 'nfl_team',
    'opponent_team': 'opponent_nfl_team',
}
defense_df = defense_df.rename(columns={k: v for k, v in defense_renames.items() if k in defense_df.columns})

# Create player column for defense (use team name)
if 'player' not in defense_df.columns:
    if 'nfl_team' in defense_df.columns:
        defense_df['player'] = defense_df['nfl_team'] + " Defense"
        print("  Created 'player' from 'nfl_team'")

# Set position for defense
defense_df['position'] = 'DEF'
defense_df['nfl_position'] = 'DEF'

# Combine
print("\n[COMBINE] Merging offense and defense...")
combined_df = pd.concat([offense_df, defense_df], ignore_index=True, sort=False)
print(f"  Combined shape: {combined_df.shape}")

# Save
print(f"\n[SAVE] Writing to: {output_file}")
combined_df.to_parquet(output_file, index=False)

print("\nDone!")
print(f"Columns: {list(combined_df.columns[:30])}")
