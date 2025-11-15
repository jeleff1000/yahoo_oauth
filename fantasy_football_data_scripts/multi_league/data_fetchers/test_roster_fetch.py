#!/usr/bin/env python3
"""
Test Roster Fetcher - Fetch just Week 1 of 2024 to verify it works
"""

import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

# Add to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fetch_roster_data import YahooRosterFetcher, log


def main():
    """Test fetching Week 1 of 2024 roster data."""

    # Configuration
    oauth_file = Path(r"C:\Users\joeye\OneDrive\Desktop\KMFFLApp\secrets.json")
    output_dir = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\data\kmffl\roster_data")
    league_id = "449.l.198278"  # 2024 KMFFL
    year = 2024

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Testing roster fetch for Week 1 of {year}")
    log(f"League ID: {league_id}")
    log("NOTE: If rate limits are hit, script will automatically wait 5 minutes and retry")

    try:
        # Initialize fetcher
        fetcher = YahooRosterFetcher(
            oauth_file=oauth_file,
            league_id=league_id,
            rate_limit=1.0,  # Very slow to avoid rate limits
            output_dir=output_dir
        )

        # Fetch just week 1
        df = fetcher.fetch_season_rosters(
            year=year,
            weeks=[1]  # Just week 1
        )

        if not df.empty:
            fetcher.save_to_parquet(df, year, filename=f"yahoo_roster_data_{year}_week1_test.parquet")

            log(f"\n{'='*70}")
            log(f"[OK] SUCCESS! Fetched {len(df)} roster records for Week 1")
            log(f"{'='*70}\n")

            # Show sample data
            log("Sample columns:")
            for col in df.columns:
                log(f"  - {col}")

            log(f"\nSample data (first 5 rows):")
            print(df[['manager_name', 'player_name', 'primary_position', 'fantasy_position', 'fantasy_points']].head())

            log(f"\nManagers in the league:")
            for manager in df['manager_name'].unique():
                count = len(df[df['manager_name'] == manager])
                log(f"  - {manager}: {count} players")
        else:
            log("[WARN] No data fetched", level="WARNING")

    except Exception as e:
        log(f"[FAIL] Error: {e}", level="ERROR")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
