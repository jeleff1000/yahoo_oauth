#!/usr/bin/env python3
"""
Yahoo League Settings Fetcher - Unified Module

Fetches ALL league settings from Yahoo API in a single call and saves to one comprehensive file.
This replaces the previous fragmented approach of multiple files for DST/offense/rules.

What this fetches in ONE API call:
- League metadata (name, size, draft type, etc.)
- Roster positions and requirements
- Scoring rules (offense, defense, kickers)
- Stat categories and modifiers
- Points Allowed buckets for DST
- Waiver rules and trade settings

Output: Single JSON file per year containing all settings
Location: {settings_dir}/league_settings_{year}_{league_key}.json
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

import pandas as pd

# Add parent directory to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_MULTI_LEAGUE_DIR = _SCRIPT_DIR.parent
_SCRIPTS_ROOT = _MULTI_LEAGUE_DIR.parent
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

try:
    from multi_league.core.league_context import LeagueContext
    LEAGUE_CONTEXT_AVAILABLE = True
except ImportError:
    LeagueContext = None
    LEAGUE_CONTEXT_AVAILABLE = False

try:
    import yahoo_fantasy_api as yfa
except ImportError:
    yfa = None

try:
    from imports_and_utils import OAuth2
except ImportError:
    try:
        from oauth_utils import create_oauth2 as OAuth2
    except ImportError:
        OAuth2 = None


def _fetch_url_xml(url: str, oauth: 'OAuth2', max_retries: int = 5, backoff: float = 0.5) -> ET.Element:
    """
    Fetch XML from Yahoo API with retries.

    Args:
        url: Yahoo API URL
        oauth: OAuth2 instance
        max_retries: Maximum retry attempts
        backoff: Initial backoff delay (doubled each retry)

    Returns:
        XML Element tree root

    Raises:
        RuntimeError: If fetch fails after retries
    """
    last_err = None
    for i in range(max_retries):
        try:
            r = oauth.session.get(url, timeout=30)
            r.raise_for_status()
            txt = (r.text or "")
            if "Request denied" in txt:
                raise RuntimeError("Request denied")
            # Strip default namespace for easier parsing
            txt = pd.Series(txt).str.replace(r' xmlns="[^"]+"', "", n=1, regex=True).iloc[0]
            return ET.fromstring(txt)
        except Exception as e:
            last_err = e
            if i == max_retries - 1:
                raise
            time.sleep(backoff * (2 ** i))
    raise last_err or RuntimeError("unknown fetch error")


def _discover_league_key(
    oauth: 'OAuth2',
    year: int,
    league_key_arg: Optional[str],
    league_name: Optional[str] = None,
    discovered_leagues_file: Optional[Path] = None
) -> Optional[str]:
    """
    Discover league key from OAuth if not provided.

    Args:
        oauth: OAuth2 instance
        year: Season year
        league_key_arg: Explicit league key (if provided)
        league_name: League name to match against (for multiple leagues)
        discovered_leagues_file: Path to discovered_leagues.json (for fast lookup)

    Returns:
        League key string or None
    """
    if league_key_arg:
        return league_key_arg.strip()

    # Try discovered_leagues.json first (fast path)
    if discovered_leagues_file and discovered_leagues_file.exists() and league_name:
        try:
            discovered = json.loads(discovered_leagues_file.read_text(encoding="utf-8"))
            for entry in discovered:
                if entry.get("year") == year and entry.get("league_name") == league_name:
                    key = entry.get("league_id")
                    if key:
                        print(f"[settings] Found league key for {year} in discovered_leagues.json: {key}")
                        return key.strip()
        except Exception as e:
            print(f"[settings] Warning: Could not read discovered_leagues.json: {e}")

    # Fall back to Yahoo API
    if yfa is None:
        return None

    try:
        gm = yfa.Game(oauth, "nfl")

        # If league_name provided, fetch all leagues and match by name
        if league_name:
            try:
                # Get all leagues for this year
                leagues = gm.to_league(str(year))
                if isinstance(leagues, list):
                    for league in leagues:
                        if hasattr(league, 'settings') and hasattr(league.settings, 'name'):
                            if league.settings.name == league_name:
                                key = league.league_id if hasattr(league, 'league_id') else None
                                if key:
                                    print(f"[settings] Matched league by name '{league_name}': {key}")
                                    return key
            except Exception:
                pass  # Fall through to league_ids method

        # Fallback: just get league IDs (may not be correct if multiple leagues)
        keys = gm.league_ids(year=year)
        if keys:
            if len(keys) > 1 and league_name:
                print(f"[settings] Warning: Multiple leagues found for {year}, but couldn't match by name. Using last one.")
            return keys[-1]
    except Exception:
        return None

    return None


def _parse_league_metadata(root: ET.Element) -> Dict[str, Any]:
    """
    Extract league metadata from settings XML.

    Args:
        root: XML Element tree root

    Returns:
        Dictionary with league metadata
    """
    league = root.find("league")
    if league is None:
        return {}

    # Helper to get text from settings or league node
    def get_val(path: str, default: str = "") -> str:
        # Try settings/path first, then league/path
        val = (league.findtext(f"settings/{path}") or league.findtext(path) or default).strip()
        return val

    # Extract playoff configuration
    playoff_start_week = get_val("playoff_start_week")
    num_playoff_teams = get_val("num_playoff_teams")
    num_playoff_consolation_teams = get_val("num_playoff_consolation_teams")
    has_multiweek_championship = get_val("has_multiweek_championship", "0")
    uses_playoff_reseeding = get_val("uses_playoff_reseeding", "0")

    # Read bye_teams from Yahoo API (don't calculate - Yahoo already provides this)
    bye_teams_raw = get_val("bye_teams")
    bye_teams = int(bye_teams_raw) if bye_teams_raw.isdigit() else 0

    # Only use fallback calculation if Yahoo doesn't provide bye_teams
    if bye_teams == 0:
        num_playoff_teams_int = int(num_playoff_teams) if num_playoff_teams.isdigit() else 0
        if num_playoff_teams_int == 6:
            bye_teams = 2  # Top 2 seeds get week 1 bye
        elif num_playoff_teams_int == 4:
            bye_teams = 0  # 4-team bracket, no byes (semifinals in week 1)
        elif num_playoff_teams_int == 8:
            bye_teams = 0  # 8-team bracket, no byes
        elif num_playoff_teams_int == 12:
            bye_teams = 4  # 12-team bracket, top 4 seeds get byes
        # If still 0, keep it as 0 (no byes)

    metadata = {
        "league_key": get_val("league_key"),
        "league_id": get_val("league_id"),
        "name": get_val("name"),
        "season": get_val("season"),
        "num_teams": get_val("num_teams"),
        "draft_type": get_val("draft_type"),
        "scoring_type": get_val("scoring_type"),
        "league_type": get_val("league_type"),
        "renew": get_val("renew"),
        "renewed": get_val("renewed"),
        "start_week": get_val("start_week"),
        "start_date": get_val("start_date"),
        "end_week": get_val("end_week"),
        "end_date": get_val("end_date"),
        "current_week": get_val("current_week"),
        # Playoff configuration settings
        "playoff_start_week": playoff_start_week,
        "num_playoff_teams": num_playoff_teams,
        "num_playoff_consolation_teams": num_playoff_consolation_teams,
        "has_multiweek_championship": has_multiweek_championship,
        "uses_playoff_reseeding": uses_playoff_reseeding,
        # Calculated fields for easier consumption
        "playoff_teams": num_playoff_teams_int if num_playoff_teams_int > 0 else None,
        "bye_teams": bye_teams,
    }

    return metadata


def _parse_roster_positions(root: ET.Element) -> List[Dict[str, Any]]:
    """
    Extract roster position requirements from settings XML.

    Args:
        root: XML Element tree root

    Returns:
        List of roster positions with counts
    """
    positions = []
    for pos in root.findall("league/settings/roster_positions/roster_position"):
        position_type = (pos.findtext("position") or "").strip()
        count = (pos.findtext("count") or "0").strip()
        positions.append({
            "position": position_type,
            "count": int(count) if count.isdigit() else 0
        })
    
    return positions


def _parse_stat_categories(root: ET.Element) -> Dict[str, Dict[str, Any]]:
    """
    Build a comprehensive map of all stat categories.

    Args:
        root: XML Element tree root

    Returns:
        Dictionary mapping stat_id to stat details
    """
    stat_map = {}
    
    for stat in root.findall("league/settings/stat_categories/stats/stat"):
        stat_id = (stat.findtext("stat_id") or "").strip()
        if not stat_id:
            continue
        
        stat_info = {
            "stat_id": stat_id,
            "enabled": (stat.findtext("enabled") or "1").strip() == "1",
            "name": (stat.findtext("name") or "").strip(),
            "display_name": (stat.findtext("display_name") or "").strip(),
            "sort_order": (stat.findtext("sort_order") or "").strip(),
            "position_type": (stat.findtext("position_type") or "").strip(),
            "stat_position_types": [],
            "is_only_display_stat": (stat.findtext("is_only_display_stat") or "0").strip() == "1",
        }
        
        # Get position types this stat applies to
        for pos_type in stat.findall("stat_position_types/stat_position_type"):
            position = (pos_type.findtext("position_type") or "").strip()
            if position:
                stat_info["stat_position_types"].append(position)
        
        # Get buckets if they exist (for Points Allowed, etc.)
        buckets = []
        for bucket in stat.findall("stat_buckets/stat_bucket"):
            start = (bucket.findtext("range/start") or "").strip()
            end = (bucket.findtext("range/end") or "").strip()
            maxv = (bucket.findtext("range/max") or "").strip()
            points = (bucket.findtext("points") or bucket.findtext("value") or "0").strip()
            
            if start and end:
                rng = f"{start}-{end}"
            elif start and maxv:
                rng = f"{start}-{maxv}"
            else:
                rng = start or maxv or ""
            
            try:
                points_val = float(points)
            except:
                points_val = 0.0
            
            buckets.append({
                "range": rng.replace(" ", ""),
                "points": points_val
            })
        
        if buckets:
            stat_info["buckets"] = buckets
        
        stat_map[stat_id] = stat_info
    
    return stat_map


def _parse_stat_modifiers(root: ET.Element) -> Dict[str, float]:
    """
    Extract point values for each stat from modifiers section.

    Args:
        root: XML Element tree root

    Returns:
        Dictionary mapping stat_id to point value
    """
    modifiers = {}
    
    for mod in root.findall("league/settings/stat_modifiers/stats/stat"):
        stat_id = (mod.findtext("stat_id") or "").strip()
        value = (mod.findtext("value") or "0").strip()
        
        if stat_id:
            try:
                modifiers[stat_id] = float(value)
            except:
                modifiers[stat_id] = 0.0
    
    return modifiers


def _build_scoring_rules(stat_map: Dict[str, Dict[str, Any]], modifiers: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Combine stat categories and modifiers into a unified scoring rules list.

    Args:
        stat_map: Map of stat details
        modifiers: Map of stat point values

    Returns:
        List of scoring rules with all details
    """
    rules = []
    
    for stat_id, stat_info in stat_map.items():
        # Skip disabled stats
        if not stat_info.get("enabled", True):
            continue
        
        # Skip display-only stats
        if stat_info.get("is_only_display_stat", False):
            continue
        
        base_rule = {
            "stat_id": stat_id,
            "name": stat_info.get("display_name") or stat_info.get("name") or stat_id,
            "position_types": stat_info.get("stat_position_types", []),
        }
        
        # Handle bucketed stats (like Points Allowed)
        if "buckets" in stat_info:
            for bucket in stat_info["buckets"]:
                rule = base_rule.copy()
                rule["bucket_range"] = bucket["range"]
                rule["points"] = bucket["points"]
                rules.append(rule)
        # Handle regular stats with modifiers
        elif stat_id in modifiers:
            rule = base_rule.copy()
            rule["points"] = modifiers[stat_id]
            rules.append(rule)
    
    return rules


def _extract_dst_scoring(scoring_rules: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract DST-specific scoring into a simple dictionary for backwards compatibility.

    Args:
        scoring_rules: Full list of scoring rules

    Returns:
        Dictionary of DST stat names to point values
    """
    dst_scoring = {}
    
    # DST stat names to extract
    dst_stats = {
        "Sack", "Interception", "Fumble Recovery", "Touchdown", "Safety",
        "Kickoff and Punt Return Touchdowns", "Blocked Punt or FG", "Block Kick"
    }
    
    # Points Allowed buckets
    pa_buckets = {}
    
    for rule in scoring_rules:
        name = rule.get("name", "")
        
        # Regular DST stats
        if name in dst_stats:
            dst_scoring[name] = rule.get("points", 0.0)
        
        # Points Allowed buckets
        if "points allowed" in name.lower() and "bucket_range" in rule:
            rng = rule["bucket_range"]
            points = rule.get("points", 0.0)
            
            # Map ranges to standard keys
            if rng in ("0", "0-0"):
                pa_buckets["PA_0"] = points
            elif rng == "1-6":
                pa_buckets["PA_1_6"] = points
            elif rng == "7-13":
                pa_buckets["PA_7_13"] = points
            elif rng == "14-20":
                pa_buckets["PA_14_20"] = points
            elif rng == "21-27":
                pa_buckets["PA_21_27"] = points
            elif rng == "28-34":
                pa_buckets["PA_28_34"] = points
            elif rng in ("35+", "35-", "35"):
                pa_buckets["PA_35_plus"] = points
    
    # Set defaults for missing PA buckets
    for key in ["PA_0", "PA_1_6", "PA_7_13", "PA_14_20", "PA_21_27", "PA_28_34", "PA_35_plus"]:
        pa_buckets.setdefault(key, 0.0)
    
    # Combine
    dst_scoring.update(pa_buckets)
    
    return dst_scoring


def fetch_league_settings(
    year: int,
    league_key: Optional[str] = None,
    oauth_file: Optional[Path] = None,
    settings_dir: Optional[Path] = None,
    context: Optional[str] = None,
    oauth: Optional['OAuth2'] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch ALL league settings from Yahoo API in a single call.

    This function makes ONE API request and parses ALL settings:
    - League metadata
    - Roster positions
    - All scoring rules (offense, defense, special teams)
    - Stat categories
    - Waiver/trade settings

    Args:
        year: Season year
        league_key: League key (e.g., "449.l.198278"), auto-discovered if None
        oauth_file: Path to OAuth credentials file
        settings_dir: Directory to save settings JSON
        context: Path to league_context.json (alternative to individual params)
        oauth: Pre-created OAuth2 session (avoids race conditions in parallel calls)

    Returns:
        Dictionary containing all league settings, or None if fetch fails

    Example output structure:
        {
            "fetched_at": "2025-01-01T12:00:00",
            "year": 2024,
            "league_key": "449.l.198278",
            "metadata": {...},
            "roster_positions": [...],
            "scoring_rules": [...],
            "dst_scoring": {...},  # Backwards compatibility
        }
    """
    # Load from context if provided
    ctx = None
    league_name = None
    discovered_leagues_file = None

    if context and LEAGUE_CONTEXT_AVAILABLE:
        try:
            ctx = LeagueContext.load(context)
            year = year or ctx.start_year
            league_name = ctx.league_name  # Get league name for matching

            # Don't use ctx.league_id - let auto-discovery find the correct year-specific key
            # league_key = league_key or ctx.league_id  # REMOVED

            if ctx.oauth_file_path:
                oauth_file = oauth_file or Path(ctx.oauth_file_path)
            settings_dir = settings_dir or Path(ctx.data_directory) / "league_settings"  # League-wide config, not player-specific

            # Look for discovered_leagues.json in the parent data directory
            data_parent = Path(ctx.data_directory).parent
            discovered_leagues_file = data_parent / "discovered_leagues.json"

        except Exception as e:
            print(f"[settings] Warning: Could not load context: {e}")

    # Use provided OAuth session or create one (from file or context)
    # IMPORTANT: When running in parallel, pass a shared oauth session to avoid race conditions
    if oauth is None:
        try:
            from yahoo_oauth import OAuth2
        except ImportError:
            print("[settings] Error: yahoo_oauth not installed. Install with: pip install yahoo_oauth")
            return None

        if oauth_file and oauth_file.exists():
            # Load from file
            try:
                oauth = OAuth2(None, None, from_file=str(oauth_file))
            except Exception as e:
                print(f"[settings] Error creating OAuth from file: {e}")
        elif ctx and ctx.oauth_credentials:
            # Create from context (inline credentials) - write to temp file
            try:
                import tempfile
                import json
                # Create temp JSON file with credentials
                temp_fd, temp_path = tempfile.mkstemp(suffix='.json', text=True)
                try:
                    with os.fdopen(temp_fd, 'w') as f:
                        json.dump(ctx.oauth_credentials, f, indent=2)
                    oauth = OAuth2(None, None, from_file=temp_path)
                    # Clean up temp file after OAuth is initialized
                    Path(temp_path).unlink(missing_ok=True)
                except Exception as e:
                    os.close(temp_fd)
                    Path(temp_path).unlink(missing_ok=True)
                    raise e
            except Exception as e:
                print(f"[settings] Error creating OAuth from context: {e}")

        if not oauth:
            print("[settings] Error: Could not create OAuth session")
            return None

    # Discover league key if not provided
    discovered_key = _discover_league_key(
        oauth,
        year,
        league_key,
        league_name=league_name,
        discovered_leagues_file=discovered_leagues_file
    )
    if not discovered_key:
        print(f"[settings] Error: Could not determine league key for {year}")
        return None

    league_key = discovered_key
    print(f"[settings] Fetching ALL settings for {league_key} (year {year})...")
    
    # SINGLE API CALL to get everything
    try:
        root = _fetch_url_xml(
            f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/settings",
            oauth
        )
    except Exception as e:
        print(f"[settings] Error fetching settings: {e}")
        return None
    
    # Parse all components
    print("[settings] Parsing league metadata...")
    metadata = _parse_league_metadata(root)
    
    print("[settings] Parsing roster positions...")
    roster_positions = _parse_roster_positions(root)
    
    print("[settings] Parsing stat categories...")
    stat_map = _parse_stat_categories(root)
    
    print("[settings] Parsing stat modifiers...")
    modifiers = _parse_stat_modifiers(root)
    
    print("[settings] Building unified scoring rules...")
    scoring_rules = _build_scoring_rules(stat_map, modifiers)
    
    print("[settings] Extracting DST scoring (backwards compatibility)...")
    dst_scoring = _extract_dst_scoring(scoring_rules)
    
    # Build comprehensive settings object
    settings = {
        "fetched_at": datetime.now().isoformat(),
        "year": year,
        "league_key": league_key,
        "metadata": metadata,
        "roster_positions": roster_positions,
        "scoring_rules": scoring_rules,
        "dst_scoring": dst_scoring,  # Backwards compatibility
        "stat_categories": stat_map,
        "stat_modifiers": modifiers,
    }
    
    # Save to single comprehensive file
    if settings_dir:
        settings_dir.mkdir(parents=True, exist_ok=True)
        safe_key = league_key.replace(".", "_")
        output_file = settings_dir / f"league_settings_{year}_{safe_key}.json"
        
        try:
            output_file.write_text(
                json.dumps(settings, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"[settings] [OK] Saved comprehensive settings -> {output_file.name}")
            print(f"[settings]   - {len(roster_positions)} roster positions")
            print(f"[settings]   - {len(scoring_rules)} scoring rules")
            print(f"[settings]   - {len(stat_map)} stat categories")
        except Exception as e:
            print(f"[settings] Warning: Could not save settings file: {e}")
    
    return settings


def load_league_settings(
    year: int,
    league_key: Optional[str] = None,
    settings_dir: Optional[Path] = None,
    context: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Load previously saved league settings from file.

    Args:
        year: Season year
        league_key: League key (used to find specific file)
        settings_dir: Directory containing settings files
        context: Path to league_context.json (alternative)

    Returns:
        Dictionary containing all league settings, or None if not found
    """
    # Load from context if provided
    if context and LEAGUE_CONTEXT_AVAILABLE:
        try:
            ctx = LeagueContext.load(context)
            league_key = league_key or ctx.league_id
            settings_dir = settings_dir or Path(ctx.data_directory) / "league_settings"  # League-wide config
        except Exception:
            pass
    
    if not settings_dir or not settings_dir.exists():
        return None
    
    # Find the settings file
    candidates: List[Path] = []
    
    if league_key:
        safe_key = league_key.replace(".", "_")
        specific_file = settings_dir / f"league_settings_{year}_{safe_key}.json"
        if specific_file.exists():
            candidates.append(specific_file)
    
    # Fallback: find any settings file for this year
    if not candidates:
        candidates = sorted(
            settings_dir.glob(f"league_settings_{year}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
    
    if not candidates:
        print(f"[settings] No saved settings found for year {year}")
        return None
    
    # Load the most recent file
    try:
        settings = json.loads(candidates[0].read_text(encoding="utf-8"))
        print(f"[settings] [OK] Loaded settings from {candidates[0].name}")
        return settings
    except Exception as e:
        print(f"[settings] Error loading settings: {e}")
        return None


# Backwards compatibility functions (for existing code that expects the old API)

def parse_scoring_rules(settings_root: ET.Element) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use fetch_league_settings() instead.
    
    Parse scoring rules from XML (old API for backwards compatibility).
    """
    print("[settings] WARNING: parse_scoring_rules() is deprecated. Use fetch_league_settings() instead.")
    
    stat_map = _parse_stat_categories(settings_root)
    modifiers = _parse_stat_modifiers(settings_root)
    return _build_scoring_rules(stat_map, modifiers)


def fetch_yahoo_dst_scoring(
    year: int,
    league_key_arg: Optional[str],
    oauth_file: Optional[Path] = None,
    settings_dir: Optional[Path] = None
) -> Optional[Dict[str, float]]:
    """
    DEPRECATED: Use fetch_league_settings() instead.
    
    Fetch only DST scoring (old API for backwards compatibility).
    """
    print("[settings] WARNING: fetch_yahoo_dst_scoring() is deprecated. Use fetch_league_settings() instead.")
    
    full_settings = fetch_league_settings(year, league_key_arg, oauth_file, settings_dir)
    if full_settings:
        return full_settings.get("dst_scoring")
    return None


if __name__ == "__main__":
    """
    CLI usage for testing:
    
    python yahoo_league_settings.py --year 2024 --league-key 449.l.198278 --oauth path/to/oauth.json --output ./settings
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Yahoo league settings")
    parser.add_argument("--year", type=int, required=True, help="Season year")
    parser.add_argument("--league-key", help="League key (e.g., 449.l.198278)")
    parser.add_argument("--oauth", type=Path, help="Path to OAuth JSON file")
    parser.add_argument("--output", type=Path, help="Output directory for settings")
    parser.add_argument("--context", help="Path to league_context.json")
    
    args = parser.parse_args()
    
    settings = fetch_league_settings(
        year=args.year,
        league_key=args.league_key,
        oauth_file=args.oauth,
        settings_dir=args.output,
        context=args.context
    )
    
    if settings:
        print("\n[SUCCESS] Successfully fetched league settings!")
        print(f"  League: {settings['metadata'].get('name', 'Unknown')}")
        print(f"  Teams: {settings['metadata'].get('num_teams', '?')}")
        print(f"  Scoring: {settings['metadata'].get('scoring_type', 'Unknown')}")
        print(f"  Rules: {len(settings['scoring_rules'])} scoring rules")
    else:
        print("\n[FAIL] Failed to fetch league settings")
        exit(1)
