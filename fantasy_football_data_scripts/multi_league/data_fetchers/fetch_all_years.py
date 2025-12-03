#!/usr/bin/env python3
"""
Multi-Year Yahoo Fantasy Data Fetcher

This script fetches player stats across multiple years, automatically using
the correct league ID for each year from discovered_leagues.json.
"""

import json
import sys
from pathlib import Path

# Add the data fetchers directory to path
data_fetchers_dir = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\data_fetchers")
sys.path.insert(0, str(data_fetchers_dir))

from yahoo_fantasy_data import YahooPlayerStatsFetcher, log

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
    """Fetch all years using correct league IDs."""
    
    # Configuration
    oauth_file = Path(r"C:\Users\joeye\OneDrive\Desktop\analytics_app\secrets.json")
    discovered_leagues_file = Path(r"C:\Users\joeye\OneDrive\Desktop\analytics_app\discovered_leagues.json")
    output_dir = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\data\kmffl\player_data")
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
    
    # Initialize fetcher (league_id will be overridden per year)
    log("\nInitializing Yahoo Fantasy API fetcher")
    fetcher = YahooPlayerStatsFetcher(
        oauth_file=oauth_file,
        league_id=None,  # Will be provided per year
        rate_limit=2.0,  # Slower rate to avoid limits
        output_dir=output_dir
    )
    
    # Fetch data for each year
    years_to_fetch = sorted(year_to_league_id.keys())
    
    for i, year in enumerate(years_to_fetch, 1):
        league_id = year_to_league_id[year]
        
        log(f"\n{'='*70}")
        log(f"Processing year {i}/{len(years_to_fetch)}: {year} (League ID: {league_id})")
        log(f"{'='*70}\n")
        
        try:
            # Fetch all weeks for this year
            df = fetcher.fetch_season_stats(
                year=year,
                weeks=None,  # All weeks
                positions=None,  # All positions
                league_id_override=league_id  # Use year-specific league ID
            )
            
            if not df.empty:
                fetcher.save_to_parquet(df, year)
                log(f"✓ Successfully fetched {len(df)} records for {year}")
            else:
                log(f"⚠ No data fetched for {year}", level="WARNING")
        
        except Exception as e:
            log(f"✗ Error processing year {year}: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            
            # Continue to next year instead of stopping
            log(f"Continuing to next year...", level="WARNING")
            continue
    
    log("\n" + "="*70)
    log(f"✓ COMPLETED! Total API requests: {fetcher.request_count}")
    log(f"Output directory: {output_dir}")
    log("="*70)


if __name__ == "__main__":
    main()

