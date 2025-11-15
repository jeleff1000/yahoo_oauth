#!/usr/bin/env python3
"""
Diagnostic script to see what the Yahoo API actually returns for roster data.
This will help us understand why fantasy points aren't being extracted.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from yahoo_oauth import OAuth2

# Configuration
oauth_file = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\data_fetchers\secrets.json")
league_id = "331.l.381581"  # 2014 KMFFL
week = 1
team_key = "331.l.381581.t.1"  # Jason's team

# Initialize OAuth
print("Initializing OAuth...")
oauth = OAuth2(None, None, from_file=str(oauth_file))

# Test different URL patterns to see what returns points
urls_to_test = [
    # Original URL
    (
        "Original (stats only)",
        f"https://fantasysports.yahooapis.com/fantasy/v2/"
        f"team/{team_key}/roster;week={week}/players/stats;type=week;week={week}"
    ),
    # With player_points explicitly requested
    (
        "With player_points",
        f"https://fantasysports.yahooapis.com/fantasy/v2/"
        f"team/{team_key}/roster;week={week}/players;out=stats,player_points"
    ),
    # Different stats syntax
    (
        "Stats with out parameter",
        f"https://fantasysports.yahooapis.com/fantasy/v2/"
        f"team/{team_key}/roster;week={week}/players;out=stats"
    ),
]

for url_name, url in urls_to_test:
    print(f"\n{'='*80}")
    print(f"Testing: {url_name}")
    print(f"URL: {url}")
    print('='*80)

    try:
        response = oauth.session.get(url, timeout=30)
        response.raise_for_status()

        text = response.text or ""

        # Remove namespace for easier reading
        text = pd.Series(text).str.replace(r' xmlns="[^"]+"', "", n=1, regex=True).iloc[0]

        # Parse XML
        root = ET.fromstring(text)

        # Find first player
        player = root.find(".//player")

        if player is not None:
            # Get player name
            name_elem = player.find(".//name/full")
            player_name = name_elem.text if name_elem is not None else "Unknown"

            print(f"\nPlayer: {player_name}")
            print("\nFull player XML structure:")
            print(ET.tostring(player, encoding='unicode')[:2000])  # First 2000 chars

            # Look for any points-related elements
            print("\n\nSearching for 'points' elements:")
            for elem in player.iter():
                if 'point' in elem.tag.lower():
                    print(f"  Found: {elem.tag} = {elem.text}")
                    if elem.text is None:
                        # Print children
                        for child in elem:
                            print(f"    Child: {child.tag} = {child.text}")

            # Look for stats
            print("\n\nStats section:")
            stats_elem = player.find(".//stats")
            if stats_elem is not None:
                print("  Found stats element")
                for stat in stats_elem.findall("stat")[:5]:  # First 5 stats
                    stat_id = stat.find("stat_id")
                    stat_value = stat.find("value")
                    if stat_id is not None and stat_value is not None:
                        print(f"    stat_id {stat_id.text} = {stat_value.text}")
            else:
                print("  No stats element found")

            # Look for player_stats wrapper
            print("\n\nPlayer stats wrapper:")
            player_stats = player.find(".//player_stats")
            if player_stats is not None:
                print("  Found player_stats element")
                print(ET.tostring(player_stats, encoding='unicode')[:1000])
            else:
                print("  No player_stats wrapper found")

        else:
            print("No player found in response")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n\nDiagnostic complete!")

