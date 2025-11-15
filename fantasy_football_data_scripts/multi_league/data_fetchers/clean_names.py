#!/usr/bin/env python3
"""
Pre-clean player and team names before merge to improve matching.

This script normalizes names in Yahoo and NFL data files to handle:
- Punctuation (T.Y. → ty, C.J. → cj)
- Suffixes (Jr, Sr, III, etc.)
- Apostrophes (Le'Veon → leveon)
- Team name variations (Los Angeles → LA)
"""
import sys
from pathlib import Path
import pandas as pd

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def normalize_name(name):
    """Remove punctuation, suffixes, and normalize spacing."""
    if not name or pd.isna(name):
        return ""

    s = str(name).lower().strip()
    # Remove periods, hyphens, apostrophes
    s = s.replace(".", "").replace("-", "").replace("'", "")
    # Remove suffixes
    tokens = [t for t in s.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)

def normalize_team_name(team):
    """Normalize team names to handle variations."""
    if not team or pd.isna(team):
        return team

    s = str(team).upper().strip()

    team_mappings = {
        "LOS ANGELES RAMS": "LAR",
        "LA RAMS": "LAR",
        "LOS ANGELES CHARGERS": "LAC",
        "LA CHARGERS": "LAC",
        "LOS ANGELES": "LA",
        "NEW YORK JETS": "NYJ",
        "NY JETS": "NYJ",
        "NEW YORK GIANTS": "NYG",
        "NY GIANTS": "NYG",
        "NEW YORK": "NY",
    }

    return team_mappings.get(s, team)

def clean_file(input_file: Path, output_file: Path):
    """Clean player and team names in a parquet file."""
    print(f"Loading: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"  Rows: {len(df):,}")

    # Store original names
    if 'player' in df.columns:
        df['player_original'] = df['player'].copy()
        df['player'] = df['player'].apply(normalize_name)
        print(f"  Cleaned player names")

    # Normalize team names
    for col in ['nfl_team', 'team', 'opponent_nfl_team']:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team_name)
            print(f"  Cleaned {col}")

    # Save cleaned file
    df.to_parquet(output_file, index=False)
    print(f"Saved: {output_file}\n")

def main():
    year = 2014

    KMFFL_DIR = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data\KMFFL\player_data")

    # Clean Yahoo file
    yahoo_file = KMFFL_DIR / f"yahoo_player_stats_{year}_all_weeks.parquet"
    yahoo_cleaned = KMFFL_DIR / f"yahoo_player_stats_{year}_all_weeks_cleaned.parquet"
    if yahoo_file.exists():
        clean_file(yahoo_file, yahoo_cleaned)

    # Clean NFL file
    nfl_file = KMFFL_DIR / f"nfl_stats_merged_{year}_all_weeks.parquet"
    nfl_cleaned = KMFFL_DIR / f"nfl_stats_merged_{year}_all_weeks_cleaned.parquet"
    if nfl_file.exists():
        clean_file(nfl_file, nfl_cleaned)

    print("✓ Name cleaning complete!")
    print(f"Use the *_cleaned.parquet files for merging")

    return 0

if __name__ == "__main__":
    sys.exit(main())

