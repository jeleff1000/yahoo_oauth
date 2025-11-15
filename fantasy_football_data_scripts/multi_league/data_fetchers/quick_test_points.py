#!/usr/bin/env python3
"""Quick test to verify player_points is returned with ;out parameter"""

from pathlib import Path
import xml.etree.ElementTree as ET
from yahoo_oauth import OAuth2
import pandas as pd

# Config
oauth_file = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\data_fetchers\secrets.json")
team_key = "331.l.381581.t.1"  # Jason's team 2014
week = 1

print("Initializing OAuth...")
oauth = OAuth2(None, None, from_file=str(oauth_file))

if not oauth.token_is_valid():
    print("Refreshing token...")
    oauth.refresh_access_token()

# Test URL with ;out parameter
url = f"https://fantasysports.yahooapis.com/fantasy/v2/team/{team_key}/roster;week={week}/players;out=stats,player_points"

print(f"\nFetching: {url}\n")

response = oauth.session.get(url, timeout=30)
response.raise_for_status()

text = response.text or ""
text = pd.Series(text).str.replace(r' xmlns="[^"]+"', "", n=1, regex=True).iloc[0]

root = ET.fromstring(text)

players = root.findall(".//player")
print(f"Found {len(players)} players\n")

if players:
    first_player = players[0]

    # Get player name
    name_elem = first_player.find(".//name/full")
    player_name = name_elem.text if name_elem is not None else "Unknown"

    print(f"First player: {player_name}")
    print("="*60)

    # Check for player_points
    pts_node = first_player.find("player_points/total")
    if pts_node is not None and pts_node.text:
        print(f"SUCCESS! player_points/total = {pts_node.text}")
    else:
        print("FAILED: player_points/total NOT FOUND")

        # Check if player_points exists at all
        pp_elem = first_player.find("player_points")
        if pp_elem is not None:
            print(f"\nplayer_points element exists, children:")
            for child in pp_elem:
                print(f"  {child.tag} = {child.text}")
        else:
            print("\nplayer_points element does NOT exist")

    # Show all elements with 'point' in name
    print("\nAll elements containing 'point':")
    found = False
    for elem in first_player.iter():
        if 'point' in elem.tag.lower():
            found = True
            print(f"  {elem.tag} = {elem.text}")
            for child in elem:
                print(f"    -> {child.tag} = {child.text}")
    if not found:
        print("  (none found)")

    # Check stats
    print("\nPlayer stats (first 5):")
    stats_elem = first_player.find("player_stats/stats")
    if stats_elem is not None:
        for i, stat in enumerate(list(stats_elem.findall("stat"))[:5]):
            sid = stat.find("stat_id")
            sval = stat.find("value")
            print(f"  stat_id={sid.text if sid is not None else 'N/A'}, value={sval.text if sval is not None else 'N/A'}")
    else:
        print("  No stats found")

print("\nTest complete!")

