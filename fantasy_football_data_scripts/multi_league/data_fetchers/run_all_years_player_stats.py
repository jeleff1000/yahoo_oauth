#!/usr/bin/env python3
"""
Run Yahoo Fantasy ROSTER Data Fetcher for all KMFFL seasons (2014-2024)

This script fetches ROSTER data showing which manager owned which player each week.
Output includes: manager_name, player_name, yahoo_position, primary_position, fantasy_position
Saves both Parquet and CSV files.
"""

import json
import sys
from pathlib import Path

# Add the data fetchers directory to path
data_fetchers_dir = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\data_fetchers")
sys.path.insert(0, str(data_fetchers_dir))

from fetch_roster_data import YahooRosterFetcher, log


def load_league_mapping(discovered_leagues_file: Path, league_name: str = "KMFFL"):
    """Load year-to-league_id mapping from discovered_leagues.json."""
    with open(discovered_leagues_file, 'r') as f:
        leagues = json.load(f)

    # Create mapping of year -> league_id for the specified league
    year_to_league_id = {}
    for league in leagues:
        if league['league_name'] == league_name:
            year_to_league_id[league['year']] = league['league_id']

    return year_to_league_id


def main():
    """Fetch all ROSTER data for all years (2014-2024)."""

    # Configuration
    oauth_file = Path(r"C:\Users\joeye\OneDrive\Desktop\KMFFLApp\secrets.json")
    discovered_leagues_file = Path(r"C:\Users\joeye\OneDrive\Desktop\KMFFLApp\discovered_leagues.json")
    output_dir = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\data\kmffl\roster_data")
    league_name = "KMFFL"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load year-to-league_id mapping
    log(f"Loading league mappings from {discovered_leagues_file}")
    year_to_league_id = load_league_mapping(discovered_leagues_file, league_name)

    if not year_to_league_id:
        log(f"No leagues found for '{league_name}'", level="ERROR")
        sys.exit(1)

    log(f"Found {len(year_to_league_id)} years for {league_name}:")
    for year, league_id in sorted(year_to_league_id.items()):
        log(f"  {year}: {league_id}")

    log("\nNOTE: Script will automatically wait 5 minutes if rate limits are hit")
    log("This will fetch ROSTER data (which manager owned which player each week)")
    log("This will take several hours to complete all years.\n")

    # Fetch data for each year
    years_to_fetch = sorted(year_to_league_id.keys())

    total_requests = 0

    for i, year in enumerate(years_to_fetch, 1):
        league_id = year_to_league_id[year]

        log(f"\n{'='*70}")
        log(f"Processing year {i}/{len(years_to_fetch)}: {year} (League ID: {league_id})")
        log(f"{'='*70}\n")

        try:
            # Initialize fetcher for this year
            fetcher = YahooRosterFetcher(
                oauth_file=oauth_file,
                league_id=league_id,
                rate_limit=1.0,  # Slower to avoid rate limits
                output_dir=output_dir
            )

            # Fetch all weeks for this year
            df = fetcher.fetch_season_rosters(
                year=year,
                weeks=None  # All weeks
            )

            if not df.empty:
                # Save both Parquet and CSV
                fetcher.save_to_parquet(df, year)

                # Also save CSV
                csv_filename = f"yahoo_roster_data_{year}.csv"
                csv_path = output_dir / csv_filename
                log(f"Saving CSV to {csv_path}")
                df.to_csv(csv_path, index=False)
                log(f"Successfully saved CSV to {csv_path}")

                log(f"[OK] Successfully fetched {len(df)} roster records for {year}")

                # Show sample info
                num_managers = df['manager_name'].nunique()
                avg_roster_size = len(df) / df['week'].nunique() / num_managers if 'week' in df.columns else 0
                log(f"  - {num_managers} managers")
                log(f"  - Average roster size: {avg_roster_size:.1f} players per team")
            else:
                log(f"[WARN] No data fetched for {year}", level="WARNING")

            total_requests += fetcher.request_count

        except Exception as e:
            log(f"[FAIL] Error processing year {year}: {e}", level="ERROR")
            import traceback
            traceback.print_exc()

            # Continue to next year instead of stopping
            log(f"Continuing to next year...", level="WARNING")
            continue

    log("\n" + "="*70)
    log(f"[OK] COMPLETED! Total API requests: {total_requests}")
    log(f"Output directory: {output_dir}")
    log("="*70)


if __name__ == "__main__":
    main()
