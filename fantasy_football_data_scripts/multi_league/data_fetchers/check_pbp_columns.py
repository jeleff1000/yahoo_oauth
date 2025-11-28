#!/usr/bin/env python3
"""
Quick diagnostic script to check what columns are available in play-by-play data across years.
This helps identify why 1999 might fail when later years succeed.
"""

import pandas as pd
import requests
from pathlib import Path

def check_pbp_columns(year: int):
    """Download and check columns for a specific year."""
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"

    print(f"\n{'='*70}")
    print(f"Checking {year} play-by-play data")
    print(f"{'='*70}")
    print(f"URL: {url}")

    try:
        # Download to temporary location
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Save to temp file
        temp_file = Path(f"temp_pbp_{year}.parquet")
        temp_file.write_bytes(response.content)

        # Read and check columns
        df = pd.read_parquet(temp_file)

        print(f"\n[SUCCESS] Downloaded {len(df):,} rows")
        print(f"Total columns: {len(df.columns)}")

        # Check for critical columns used by defense_stats.py
        critical_cols = {
            'drive_first_downs': 'Used for three_out calculation',
            'play_type_nfl': 'Used for three_out calculation',
            'down': 'Used for fourth_down_stop calculation',
            'play_type': 'Used for fourth_down_stop calculation',
            'first_down': 'Used for fourth_down_stop calculation',
            'defteam': 'Used for grouping defensive stats',
            'week': 'Used for weekly aggregation',
            'season': 'Used for yearly aggregation'
        }

        print(f"\nCritical columns check:")
        missing = []
        for col, purpose in critical_cols.items():
            if col in df.columns:
                print(f"  [OK] {col:20s} - {purpose}")
            else:
                print(f"  [MISSING] {col:20s} - {purpose}")
                missing.append(col)

        if missing:
            print(f"\n[WARNING] Missing {len(missing)} critical column(s): {', '.join(missing)}")
            print(f"This explains why defense_stats.py fails for {year}")
        else:
            print(f"\n[OK] All critical columns present for {year}")

        # Show sample of columns
        print(f"\nFirst 20 columns:")
        for i, col in enumerate(df.columns[:20], 1):
            print(f"  {i:2d}. {col}")

        # Clean up
        temp_file.unlink()

        return df.columns.tolist(), missing

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"\n[ERROR] Data not available for {year} (404 Not Found)")
        else:
            print(f"\n[ERROR] HTTP error: {e}")
        return None, None
    except Exception as e:
        print(f"\n[ERROR] Failed to process {year}: {e}")
        return None, None


def compare_years(years: list):
    """Compare columns across multiple years."""
    print(f"\n{'='*70}")
    print(f"COLUMN COMPARISON ACROSS YEARS")
    print(f"{'='*70}")

    year_columns = {}

    for year in years:
        cols, missing = check_pbp_columns(year)
        if cols:
            year_columns[year] = set(cols)

    if len(year_columns) < 2:
        print("\n[ERROR] Need at least 2 successful years to compare")
        return

    # Find columns that exist in all years
    all_cols = set.intersection(*year_columns.values())

    # Find columns unique to certain years
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    for year in sorted(year_columns.keys()):
        total = len(year_columns[year])
        common = len(year_columns[year] & all_cols)
        unique = len(year_columns[year] - all_cols)
        print(f"{year}: {total:3d} total columns ({common:3d} common, {unique:3d} unique)")

    print(f"\nCommon columns across all years: {len(all_cols)}")

    # Show what's missing in early years
    if 1999 in year_columns and 2020 in year_columns:
        missing_in_1999 = year_columns[2020] - year_columns[1999]
        if missing_in_1999:
            print(f"\nColumns in 2020 but NOT in 1999 ({len(missing_in_1999)}):")
            for col in sorted(list(missing_in_1999)[:20]):  # Show first 20
                print(f"  - {col}")
            if len(missing_in_1999) > 20:
                print(f"  ... and {len(missing_in_1999) - 20} more")


if __name__ == "__main__":
    # Check specific years
    years_to_check = [1999, 2000, 2010, 2020, 2024]

    compare_years(years_to_check)
