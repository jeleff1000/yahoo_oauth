#!/usr/bin/env python3
"""
Quick runner script for Yahoo Fantasy data fetcher.

This script makes it easy to run the yahoo_fantasy_data.py script from the KMFFLApp directory.
"""

import sys
from pathlib import Path

# Add the data fetchers directory to path
data_fetchers_dir = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\data_fetchers")
sys.path.insert(0, str(data_fetchers_dir))

# Import the script
import yahoo_fantasy_data

if __name__ == "__main__":
    # Run the main function
    yahoo_fantasy_data.main()

