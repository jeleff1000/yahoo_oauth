#!/usr/bin/env python3
"""
Test script to diagnose 2014 data fetching issues.

This script will:
1. Verify the league exists for 2014
2. Try to fetch week 1 of 2014
3. Report detailed diagnostics
"""

import sys
import json
from pathlib import Path

# Add the fantasy football scripts to path
scripts_root = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts")
sys.path.insert(0, str(scripts_root))

from multi_league.core.league_context import LeagueContext
from imports_and_utils import OAuth2

# Import the yahoo data fetcher functions
sys.path.insert(0, str(scripts_root / "multi_league" / "data_fetchers"))
from yahoo_fantasy_data import (
    _get_league_key,
    _verify_league_exists,
    _get_actual_week_range,
    _get_league_settings,
    fetch_yahoo_data_for_week
)

def main():
    print("="*80)
    print("2014 DATA FETCH DIAGNOSTIC TEST")
    print("="*80)
    
    # Load context
    context_path = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data\league_context.json")
    print(f"\n1. Loading context from: {context_path}")
    ctx = LeagueContext.load(str(context_path))
    print(f"   League: {ctx.league_name}")
    print(f"   League ID: {ctx.league_id}")
    print(f"   Start Year: {ctx.start_year}")
    
    # Initialize OAuth
    print(f"\n2. Initializing OAuth...")
    if ctx.oauth_credentials:
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ctx.oauth_credentials, f)
            temp_path = f.name
        oauth = OAuth2(None, None, from_file=temp_path)
        os.unlink(temp_path)
        print(f"   OAuth initialized from embedded credentials")
    else:
        print(f"   ERROR: No OAuth credentials found")
        return
    
    # Check token validity
    if not oauth.token_is_valid():
        print(f"   Token expired, refreshing...")
        oauth.refresh_access_token()
        print(f"   Token refreshed successfully")
    else:
        print(f"   Token is valid")
    
    # Test 2014
    year = 2014
    print(f"\n3. Testing year {year}")
    
    try:
        # Get league key for 2014
        league_key = _get_league_key(ctx, year, oauth)
        print(f"   League Key: {league_key}")
        
        # Verify league exists
        print(f"\n4. Verifying league exists for {year}...")
        exists = _verify_league_exists(oauth, league_key)
        print(f"   League exists: {exists}")
        
        if not exists:
            print(f"\n   *** ISSUE FOUND: League does not exist for {year} ***")
            print(f"   This means the league was either:")
            print(f"   - Not created yet in {year}")
            print(f"   - Created under a different league ID")
            print(f"   - Not accessible via the current OAuth credentials")
            return
        
        # Get league settings
        print(f"\n5. Fetching league settings for {year}...")
        settings = _get_league_settings(oauth, league_key)
        if settings:
            print(f"   Settings found:")
            for key, value in settings.items():
                print(f"     {key}: {value}")
        else:
            print(f"   No settings available (may be historical year)")
        
        # Get actual week range
        print(f"\n6. Detecting actual week range for {year}...")
        start_week, end_week = _get_actual_week_range(oauth, league_key, year)
        print(f"   Week range: {start_week} - {end_week}")
        
        if start_week == 0 or end_week == 0:
            print(f"\n   *** ISSUE FOUND: No valid weeks detected for {year} ***")
            return
        
        # Try to fetch week 1
        print(f"\n7. Attempting to fetch week 1 data for {year}...")
        df = fetch_yahoo_data_for_week(ctx, oauth, year, 1)
        
        if df.empty:
            print(f"   ERROR: No data returned for {year} week 1")
        else:
            print(f"   SUCCESS: Fetched {len(df)} player records")
            print(f"\n   Sample data:")
            print(f"   Managers: {df['manager'].unique().tolist()}")
            print(f"   Sample players: {df['player'].head(5).tolist()}")
            
            # Check output path
            output_file = ctx.player_data_directory / f"yahoo_player_data_{year}_week_1.parquet"
            print(f"\n8. Expected output file: {output_file}")
            print(f"   File exists: {output_file.exists()}")
            
    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

