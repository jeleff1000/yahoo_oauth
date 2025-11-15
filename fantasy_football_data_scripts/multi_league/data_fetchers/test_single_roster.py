#!/usr/bin/env python3
"""Quick test to see what the API returns for one team's roster."""

from pathlib import Path
import xml.etree.ElementTree as ET
from yahoo_oauth import OAuth2
import pandas as pd

# Config
oauth_file = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\data_fetchers\secrets.json")
team_key = "331.l.381581.t.1"  # Jason's team 2014
week = 1

# Initialize OAuth
print("Initializing OAuth...")
oauth = OAuth2(None, None, from_file=str(oauth_file))

if not oauth.token_is_valid():
    print("Refreshing token...")
    oauth.refresh_access_token()

# Test URL
url = f"https://fantasysports.yahooapis.com/fantasy/v2/team/{team_key}/roster;week={week}/players/stats"

print(f"\nFetching: {url}\n")

response = oauth.session.get(url, timeout=30)
response.raise_for_status()

text = response.text or ""

# Remove namespace
text = pd.Series(text).str.replace(r' xmlns="[^"]+"', "", n=1, regex=True).iloc[0]

# Save raw XML for inspection
debug_file = Path("debug_raw_response.xml")
with open(debug_file, 'w', encoding='utf-8') as f:
    f.write(text)
print(f"Saved raw XML to: {debug_file}")

# Parse and check first player
root = ET.fromstring(text)

players = root.findall(".//player")
print(f"\nFound {len(players)} players\n")

if players:
    first_player = players[0]

    # Get player name
    name_elem = first_player.find(".//name/full")
    player_name = name_elem.text if name_elem is not None else "Unknown"

    print(f"First player: {player_name}")
    print("="*60)

    # Check for player_points at different locations
    print("\nLooking for player_points...")

    # Location 1: player_points/total
    pts1 = first_player.find("player_points/total")
    print(f"  player_points/total: {pts1.text if pts1 is not None else 'NOT FOUND'}")

    # Location 2: player_points (parent)
    pts_parent = first_player.find("player_points")
    if pts_parent is not None:
        print(f"  player_points element exists")
        for child in pts_parent:
            print(f"    child: {child.tag} = {child.text}")
    else:
        print(f"  player_points element: NOT FOUND")

    # Location 3: Check ALL elements with 'point' in the name
    print("\n  All elements with 'point' in name:")
    found_any = False
    for elem in first_player.iter():
        if 'point' in elem.tag.lower():
            found_any = True
            print(f"    {elem.tag} = {elem.text}")
            for child in elem:
                print(f"      -> {child.tag} = {child.text}")

    if not found_any:
        print("    (none found)")

    # Check player_stats
    print("\nLooking for player_stats...")
    player_stats = first_player.find("player_stats")
    if player_stats is not None:
        print("  player_stats element exists")
        stats_section = player_stats.find("stats")
        if stats_section is not None:
            print(f"  Found {len(list(stats_section.findall('stat')))} stats")
            # Show first 3 stats
            for i, stat in enumerate(list(stats_section.findall('stat'))[:3]):
                sid = stat.find("stat_id")
                sval = stat.find("value")
                print(f"    stat_id={sid.text if sid is not None else 'N/A'}, value={sval.text if sval is not None else 'N/A'}")
    else:
        print("  player_stats element: NOT FOUND")

    # Show the full XML structure for this player (first 2000 chars)
    print("\n" + "="*60)
    print("First 2000 chars of player XML:")
    print("="*60)
    player_xml = ET.tostring(first_player, encoding='unicode')
    print(player_xml[:2000])

print("\n\nDone! Check debug_raw_response.xml for full API response.")

