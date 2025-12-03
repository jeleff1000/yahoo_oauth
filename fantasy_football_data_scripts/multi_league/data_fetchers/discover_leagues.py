#!/usr/bin/env python3
"""
Discover your Yahoo Fantasy Football leagues and their IDs.

This script will:
1. Connect to Yahoo API using your OAuth credentials
2. List all your NFL fantasy leagues from 2014-2024
3. Show league IDs, names, and years
4. Help you create a proper league_context.json
"""

import json
import sys
from pathlib import Path

try:
    from yahoo_oauth import OAuth2
    from yahoo_fantasy_api import Game
except ImportError:
    print("ERROR: Required packages not installed")
    print("Run: pip install yahoo_oauth yahoo_fantasy_api")
    sys.exit(1)

def discover_leagues():
    """Discover all Yahoo Fantasy Football leagues."""

    # Load OAuth credentials
    secrets_file = Path(__file__).parent / "secrets.json"
    if not secrets_file.exists():
        print(f"ERROR: secrets.json not found at {secrets_file}!")
        sys.exit(1)

    print("Initializing Yahoo OAuth...")
    oauth = OAuth2(None, None, from_file=str(secrets_file))

    if not oauth.token_is_valid():
        print("Refreshing OAuth token...")
        oauth.refresh_access_token()

    print("✓ OAuth connected successfully!\n")

    # Discover leagues for each year
    print("=" * 70)
    print("DISCOVERING YOUR YAHOO FANTASY FOOTBALL LEAGUES")
    print("=" * 70)

    leagues_found = []

    for year in range(2014, 2025):
        print(f"\nChecking {year}...", end=" ")

        try:
            game = Game(oauth, 'nfl')
            league_ids = game.league_ids(year=year)

            if league_ids:
                print(f"✓ Found {len(league_ids)} league(s)")

                for league_id in league_ids:
                    try:
                        from yahoo_fantasy_api import League
                        league = League(oauth, league_id)
                        settings = league.settings()

                        league_info = {
                            'year': year,
                            'league_id': league_id,
                            'league_name': settings.get('name', 'Unknown'),
                            'num_teams': settings.get('num_teams', 'Unknown')
                        }
                        leagues_found.append(league_info)

                        print(f"  - {league_info['league_name']}: {league_id}")
                    except Exception as e:
                        print(f"  - {league_id} (couldn't fetch details)")
            else:
                print("(none)")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 70)
    print(f"TOTAL LEAGUES FOUND: {len(leagues_found)}")
    print("=" * 70)

    if not leagues_found:
        print("\nNo leagues found. Make sure:")
        print("1. Your OAuth credentials are correct")
        print("2. You have participated in Yahoo Fantasy Football")
        return

    # Group by league name to find the same league across years
    from collections import defaultdict
    by_name = defaultdict(list)
    for league in leagues_found:
        by_name[league['league_name']].append(league)

    print("\n\nLEAGUES GROUPED BY NAME:")
    print("-" * 70)

    for league_name, years_list in by_name.items():
        years_list.sort(key=lambda x: x['year'])
        print(f"\n📊 {league_name}")
        print(f"   Years: {years_list[0]['year']}-{years_list[-1]['year']} ({len(years_list)} seasons)")
        print(f"   Teams: {years_list[0]['num_teams']}")
        print(f"   Latest League ID: {years_list[-1]['league_id']}")

        # Suggest creating league_context.json for this league
        if len(years_list) >= 2:  # Multi-year league
            print(f"\n   💡 Suggested league_context.json:")
            context = {
                "league_id": years_list[-1]['league_id'],  # Use most recent
                "league_name": league_name,
                "oauth_file_path": str(secrets_file.absolute()).replace("\\", "\\\\"),
                "game_code": "nfl",
                "start_year": years_list[0]['year'],
                "end_year": years_list[-1]['year'],
                "num_teams": years_list[0]['num_teams'],
                "data_directory": f"C:\\\\Users\\\\joeye\\\\OneDrive\\\\Desktop\\\\fantasy_football_data_downloads\\\\data\\\\{league_name.replace(' ', '_').lower()}"
            }
            print(json.dumps(context, indent=4))

    # Save all discovered leagues to a file
    output_file = Path(__file__).parent.parent.parent.parent / "analytics_app" / "discovered_leagues.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(leagues_found, f, indent=4)

    print(f"\n\n✓ Full league details saved to: {output_file}")
    print("\nNext steps:")
    print("1. Copy the suggested league_context.json above")
    print("2. Save it as 'league_context.json'")
    print("3. Run: python run_yahoo_data_fetch.py --context league_context.json --year 2024")

if __name__ == "__main__":
    discover_leagues()
