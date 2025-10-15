#!/usr/bin/env python3
"""
Utility script to fetch and display all scoring rules for a given Yahoo Fantasy
football league season.  This script uses your existing OAuth credentials and
the same helper modules already used by the fantasy data downloader.  Given a
year (e.g. 2025) and optionally a league key (e.g. "461.l.90939"), it
authenticates to Yahoo, retrieves the league settings, and then prints the
scoring rules in a human‑readable JSON format.

For each stat category defined in the league settings, the script will look up
the corresponding score in the stat modifiers section.  Categories that use
buckets (such as points allowed for defenses) will have each bucket and its
point value listed individually.  Categories without a modifier (or disabled
categories) are omitted from the output.

Run this script from the repository root using:

    python3 fantasy_football_data_scripts/player_stats/pull_scoring_rules.py --year 2025 --league_key 461.l.90939

If no league_key is provided, the script will attempt to auto‑discover your
league for the given year (assuming your OAuth token has access to the league).

This script is intended for debugging and inspection only; it does not save
results to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

# Reuse OAuth and yfa helpers from the existing codebase.  These imports will
# succeed when run within the fantasy_football_data_downloads repository.  If
# they fail, we fall back to a simpler discovery method which will simply
# print an error message.
try:
    from imports_and_utils import OAuth2, yfa  # type: ignore
except Exception:
    OAuth2 = None  # type: ignore
    yfa = None  # type: ignore


def discover_league_key(oauth: OAuth2, year: int, league_key_arg: Optional[str]) -> Optional[str]:
    """Return the league_key for the given year.  If league_key_arg is provided,
    return it directly.  Otherwise, query the Yahoo Fantasy API for the user's
    leagues in that season and return the most recent league key.  Returns
    ``None`` if discovery fails or yfa is unavailable.
    """
    if league_key_arg:
        return league_key_arg.strip()
    if yfa is None or oauth is None:
        return None
    try:
        gm = yfa.Game(oauth, "nfl")
        keys = gm.league_ids(year=year)
        if keys:
            return keys[-1]
    except Exception:
        pass
    return None


def fetch_league_settings(oauth: OAuth2, league_key: str) -> ET.Element:
    """Fetch the league settings XML for a given league key and return the
    parsed XML root.  Raises an exception on any HTTP or parsing error.
    """
    import requests  # imported here to limit scope
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/settings"
    # Perform a single request; callers can wrap in retry logic if desired
    r = oauth.session.get(url, timeout=30)
    r.raise_for_status()
    text = (r.text or "").strip()
    if not text:
        raise RuntimeError(f"Empty response when fetching settings for {league_key}")
    # Strip the default XML namespace if present
    if text.startswith("<?xml"):
        # remove first occurrence of default xmlns attribute
        import re
        text = re.sub(r' xmlns="[^"]+"', "", text, count=1)
    return ET.fromstring(text)


def parse_scoring_rules(settings_root: ET.Element) -> List[Dict[str, object]]:
    """Parse the stat categories and modifiers from a league settings XML and
    return a list of dicts describing each scoring rule.  Each dict will
    include the stat's name, stat_id, any bucket information, and the point
    value assigned to that stat or bucket.  Disabled stats (i.e., lacking a
    modifier) are omitted.
    """
    # Build a mapping from stat_id to human‑readable display name
    id_to_name: Dict[str, str] = {}
    for stat in settings_root.findall("league/settings/stat_categories/stats/stat"):
        sid = (stat.findtext("stat_id") or "").strip()
        display_name = (stat.findtext("display_name") or "").strip()
        if sid and display_name:
            id_to_name[sid] = display_name

    # Build a mapping from stat_id to its point value (modifier)
    id_to_value: Dict[str, float] = {}
    for mod in settings_root.findall("league/settings/stat_modifiers/stats/stat"):
        sid = (mod.findtext("stat_id") or "").strip()
        val_text = (mod.findtext("value") or "").strip()
        try:
            id_to_value[sid] = float(val_text)
        except Exception:
            pass

    rules: List[Dict[str, object]] = []
    # Iterate over all stat categories to collect buckets and values
    for stat in settings_root.findall("league/settings/stat_categories/stats/stat"):
        sid = (stat.findtext("stat_id") or "").strip()
        if not sid:
            continue
        name = id_to_name.get(sid) or (stat.findtext("name") or "").strip() or sid

        # If the stat has buckets, record each bucket separately
        buckets = stat.findall("stat_buckets/stat_bucket")
        if buckets:
            for b in buckets:
                # Determine the point value (points or value field)
                pts_text = (b.findtext("points") or b.findtext("value") or "0").strip()
                try:
                    pts_val = float(pts_text)
                except Exception:
                    pts_val = None
                # Determine the range string
                start = (b.findtext("range/start") or "").strip()
                end = (b.findtext("range/end") or "").strip()
                maxv = (b.findtext("range/max") or "").strip()
                if start and end:
                    rng = f"{start}-{end}"
                elif start and maxv:
                    rng = f"{start}-{maxv}"
                else:
                    rng = start or maxv or ""
                rng = rng.replace(" ", "")  # normalize spacing
                rules.append({
                    "stat_id": sid,
                    "name": name,
                    "bucket_range": rng,
                    "points": pts_val,
                })
        else:
            # Non‑bucketed stat: only include if it has a modifier
            if sid in id_to_value:
                rules.append({
                    "stat_id": sid,
                    "name": name,
                    "points": id_to_value[sid],
                })
    return rules


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and display Yahoo Fantasy football scoring rules for a given season. "
            "If a year is not provided via --year, the script will prompt for it."
        )
    )
    parser.add_argument(
        "--year",
        type=int,
        required=False,
        help="Season year (e.g. 2025). If omitted, you will be prompted to enter it."
    )
    parser.add_argument(
        "--league_key",
        type=str,
        default=None,
        help=(
            "Optional league key (e.g. 461.l.90939).  If omitted, the script will attempt to "
            "auto‑discover your league for the given year."
        ),
    )
    args = parser.parse_args(argv)

    # Prompt for year if not provided
    year: Optional[int] = args.year
    if year is None:
        try:
            year_input = input("Enter the season year (e.g. 2025): ").strip()
            if year_input:
                year = int(year_input)
        except Exception:
            year = None
    if year is None:
        print("Error: A valid year must be specified.", file=sys.stderr)
        return 1

    # Ensure OAuth is available
    if OAuth2 is None:
        print("Error: OAuth2 module not found.  Please run this script within the fantasy_football_data_downloads repository.", file=sys.stderr)
        return 1

    # Load OAuth credentials
    repo_root = Path(__file__).resolve().parents[2]
    oauth_path = repo_root / "oauth" / "Oauth.json"
    if not oauth_path.exists():
        print(f"Error: OAuth credentials not found at {oauth_path}", file=sys.stderr)
        return 1
    oauth = OAuth2(None, None, from_file=str(oauth_path))
    if not oauth.token_is_valid():
        oauth.refresh_access_token()

    # Determine the league key
    league_key = discover_league_key(oauth, year, args.league_key)
    if not league_key:
        print(f"Error: Could not determine league key for year {year}.  Provide one using --league_key.", file=sys.stderr)
        return 1

    # Fetch settings
    try:
        settings_root = fetch_league_settings(oauth, league_key)
    except Exception as e:
        print(f"Error fetching league settings: {e}", file=sys.stderr)
        return 1

    # Parse rules
    rules = parse_scoring_rules(settings_root)
    # Print the rules as pretty JSON
    print(json.dumps(rules, indent=2, sort_keys=False))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))