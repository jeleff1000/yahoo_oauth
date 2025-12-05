#!/usr/bin/env python3
"""
Draft Data Fetcher V2 - Multi-League Edition

Fetches Yahoo Fantasy Football draft data for any league using LeagueContext.
Compatible with the multi-league infrastructure.

Key improvements over V1:
- Multi-league support via LeagueContext
- RunLogger integration for structured logging
- Backward compatible with old config.py system
- Cleaner API with context-based configuration
- Better error handling and retry logic

Usage:
    # With LeagueContext
    from multi_league.core.league_context import LeagueContext
    ctx = LeagueContext.load("leagues/kmffl/league_context.json")
    df = fetch_draft_data(ctx, year=2024)

    # All years
    df = fetch_all_draft_years(ctx)

    # CLI with context
    python draft_data_v2.py --context leagues/kmffl/league_context.json --year 2024

    # CLI all years
    python draft_data_v2.py --context leagues/kmffl/league_context.json --all-years
"""

import sys
import argparse
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import pandas as pd
import requests
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2

# Import shared name normalization
try:
    from .clean_names import normalize_manager_name
except ImportError:
    from clean_names import normalize_manager_name

# Add paths for imports
_script_file = Path(__file__).resolve()
_multi_league_dir = _script_file.parent.parent  # multi_league directory
sys.path.insert(0, str(_multi_league_dir / "core"))
sys.path.insert(0, str(_multi_league_dir / "utils"))

# Multi-league infrastructure
try:
    from league_context import LeagueContext
    LEAGUE_CONTEXT_AVAILABLE = True
except ImportError:
    LeagueContext = None
    LEAGUE_CONTEXT_AVAILABLE = False

try:
    from yahoo_league_settings import load_league_settings
    LOAD_SETTINGS_AVAILABLE = True
except ImportError:
    load_league_settings = None
    LOAD_SETTINGS_AVAILABLE = False

try:
    from run_metadata import RunLogger
    RUN_LOGGER_AVAILABLE = True
except ImportError:
    RunLogger = None
    RUN_LOGGER_AVAILABLE = False

# Default paths (for standalone mode)
THIS_FILE = Path(__file__).resolve()
SCRIPT_ROOT = THIS_FILE.parent.parent.parent  # Back to scripts root
DEFAULT_DATA_ROOT = SCRIPT_ROOT.parent / "fantasy_football_data" / "draft_data"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class DraftPick:
    """Represents a single draft pick."""
    year: int
    pick: int
    round: int
    team_key: str
    yahoo_player_id: str
    cost: Optional[float]
    player: Optional[str] = None
    yahoo_position: Optional[str] = None


# =============================================================================
# API Retry Logic
# =============================================================================

class APITimeoutError(Exception):
    """Recoverable API timeout."""
    pass


class RecoverableAPIError(APITimeoutError):
    """Transient API failures (rate-limit, 403/429, 'Request denied', empty XML, etc.)."""
    pass


_XML_NS_RE = re.compile(r' xmlns="[^"]+"')


def fetch_url(url: str, oauth, max_retries: int = 6, backoff: float = 0.5) -> ET.Element:
    """
    Fetch URL with retry logic and exponential backoff.

    Args:
        url: URL to fetch
        oauth: OAuth2 session
        max_retries: Maximum retry attempts
        backoff: Initial backoff time in seconds

    Returns:
        ET.Element: Parsed XML root element

    Raises:
        APITimeoutError: If timeout occurs after retries
        RecoverableAPIError: If recoverable error occurs after retries
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = oauth.session.get(url, timeout=30)
            try:
                r.raise_for_status()
            except requests.HTTPError as he:
                code = getattr(he.response, "status_code", None)
                if code in (429, 403, 502, 503, 504):
                    if attempt == max_retries - 1:
                        raise RecoverableAPIError(f"HTTP {code} on {url}") from he

                    # Special handling for 429 rate limit - use 10 minute cooldown
                    if code == 429:
                        print("[RATE LIMIT] Yahoo API rate limit hit; signaling wrapper to retry...")
                        raise RecoverableAPIError("rate_limited")

                    else:
                        # Normal exponential backoff for other transient errors
                        wait_time = backoff * (2 ** attempt)
                        print(f"[RETRY] HTTP {code} error. Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    continue
                raise

            text = (r.text or "").strip()
            if not text or "Request denied" in text:
                if attempt == max_retries - 1:
                    raise RecoverableAPIError(f"Empty or denied response from {url}")
                time.sleep(backoff * (2 ** attempt))
                continue

            xmlstring = _XML_NS_RE.sub("", text, count=1)
            return ET.fromstring(xmlstring)

        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise APITimeoutError(f"Timeout fetching {url}") from e
            time.sleep(backoff * (2 ** attempt))

        except (requests.RequestException, ET.ParseError, ValueError) as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise RecoverableAPIError(f"Transient error fetching {url}: {e}") from e
            time.sleep(backoff * (2 ** attempt))

    if isinstance(last_exc, requests.exceptions.Timeout):
        raise APITimeoutError(f"Timeout fetching {url}") from last_exc
    raise RecoverableAPIError(f"Unknown fetch_url failure for {url}: {last_exc}")


# --- REMOVED: league settings are now fetched in PHASE 0 of initial_import_v2.py ---
# draft_data_v2.py now READS from saved settings files instead of making API calls


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_draft_picks(oauth, league_id: str, year: int) -> List[DraftPick]:
    """
    Fetch draft picks for a league.

    Args:
        oauth: OAuth2 session
        league_id: Yahoo league key
        year: Draft year

    Returns:
        List of DraftPick objects
    """
    league = yfa.League(oauth, league_id)
    draft_results = league.draft_results()

    picks = []
    for result in draft_results:
        pick = DraftPick(
            year=year,
            pick=result.get('pick', 0),
            round=result.get('round', 0),
            team_key=result.get('team_key', ''),
            yahoo_player_id=result.get('player_id', ''),
            cost=float(result.get('cost', 0)) if result.get('cost') else None,
            player=result.get('player_name'),
            yahoo_position=result.get('position')
        )
        picks.append(pick)

    return picks


def fetch_team_and_player_mappings(
    oauth,
    league_id: str,
    timeout: int = 30,
    manager_name_overrides: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Fetch team and player mappings from Yahoo API.

    Args:
        oauth: OAuth2 session
        league_id: Yahoo league key
        timeout: Request timeout in seconds
        manager_name_overrides: Dict mapping team names/nicknames to real manager names

    Returns:
        Tuple of (team_key_to_manager, team_key_to_guid, team_key_to_team_name, player_id_to_name, player_id_to_team)
    """
    team_key_to_manager = {}
    team_key_to_guid = {}
    team_key_to_team_name = {}
    player_id_to_name = {}
    player_id_to_team = {}

    # Step 1: Fetch teams from /league/{league_id}/teams endpoint
    # This endpoint reliably returns both team name and manager info
    try:
        teams_url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_id}/teams"
        root = fetch_url(teams_url, oauth)

        for team_elem in root.findall(".//team"):
            team_key_elem = team_elem.find("team_key")
            if team_key_elem is None:
                continue
            team_key = team_key_elem.text

            # Get team name (used for franchise tracking and --hidden-- fallback)
            team_name_elem = team_elem.find("name")
            team_name = team_name_elem.text if team_name_elem is not None else None
            team_key_to_team_name[team_key] = team_name

            # Get raw nickname from manager element
            manager_elem = team_elem.find(".//manager/nickname")
            raw_nickname = manager_elem.text if manager_elem is not None else None

            # Get manager guid (persistent identifier across years)
            guid_elem = team_elem.find(".//manager/guid")
            manager_guid = guid_elem.text if guid_elem is not None else None
            team_key_to_guid[team_key] = manager_guid

            # Normalize manager name (handles --hidden-- with team_name fallback)
            manager_name = normalize_manager_name(
                nickname=raw_nickname,
                overrides=manager_name_overrides,
                team_name_fallback=team_name
            )
            team_key_to_manager[team_key] = manager_name

        print(f"[draft] Fetched {len(team_key_to_manager)} teams from /teams endpoint")
    except (APITimeoutError, RecoverableAPIError) as e:
        print(f"[draft] Warning: Could not fetch teams from /teams endpoint: {e}")

    # Step 2: Fetch player mappings from week 1 rosters (for players on rosters)
    for i in range(1, 20):  # Support up to 20 teams
        try:
            root = fetch_url(
                f"https://fantasysports.yahooapis.com/fantasy/v2/team/{league_id}.t.{i}/roster;week=1/players/stats",
                oauth
            )
        except (APITimeoutError, RecoverableAPIError):
            continue

        # If we didn't get team mappings from /teams endpoint, try to get them here
        if not team_key_to_manager:
            raw_nickname = (root.findtext("team/managers/manager/nickname") or "").strip()
            team_name = (root.findtext("team/name") or "").strip()
            team_key = (root.findtext("team/team_key") or "").strip()
            manager_guid = (root.findtext("team/managers/manager/guid") or "").strip()

            if team_key and team_key not in team_key_to_manager:
                manager_name = normalize_manager_name(
                    nickname=raw_nickname if raw_nickname else None,
                    overrides=manager_name_overrides,
                    team_name_fallback=team_name
                )
                team_key_to_manager[team_key] = manager_name
                team_key_to_guid[team_key] = manager_guid if manager_guid else None
                team_key_to_team_name[team_key] = team_name if team_name else None

        players = root.findall("team/roster/players/player")
        player_ids = [p.findtext("player_id") for p in players if p.find("player_id") is not None]
        names = [p.findtext("name/full") for p in players if p.find("name/full") is not None]
        team_abbrs = [p.findtext("editorial_team_abbr") for p in players if p.find("editorial_team_abbr") is not None]

        for pid, name, tabbr in zip(player_ids, names, team_abbrs):
            if pid and pid not in player_id_to_name:
                player_id_to_name[pid] = name
                player_id_to_team[pid] = tabbr

    # Step 3: Fetch all players (paginated) for comprehensive player mapping
    start = 0
    while True:
        try:
            root = fetch_url(
                f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_id}/players;start={start}",
                oauth
            )
        except (APITimeoutError, RecoverableAPIError):
            break

        players = root.findall("league/players/player")
        if not players:
            break

        for player in players:
            pid = player.findtext("player_id")
            if pid and pid not in player_id_to_name:
                player_id_to_name[pid] = player.findtext("name/full")
                player_id_to_team[pid] = player.findtext("editorial_team_abbr")

        start += len(players)

    return team_key_to_manager, team_key_to_guid, player_id_to_name, player_id_to_team


def fetch_draft_analysis(oauth, league_id: str, year: int, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch draft analysis data from Yahoo API.

    Args:
        oauth: OAuth2 session
        league_id: Yahoo league key
        year: Draft year

    Returns:
        DataFrame with draft analysis data
    """
    draft_analysis = []
    start = 0

    while True:
        try:
            root = fetch_url(
                f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_id}/players;start={start}/draft_analysis",
                oauth
            )
        except (APITimeoutError, RecoverableAPIError):
            break

        players = root.findall("league/players/player")
        if not players:
            break

        for player in players:
            player_data = {
                "year": year,
                "yahoo_player_id": player.findtext("player_id"),
                "player": player.findtext("name/full"),
                "yahoo_position": player.findtext("primary_position"),
                "avg_pick": player.findtext("draft_analysis/average_pick"),
                "avg_round": player.findtext("draft_analysis/average_round"),
                "avg_cost": player.findtext("draft_analysis/average_cost"),
                "percent_drafted": player.findtext("draft_analysis/percent_drafted"),
                "preseason_avg_pick": player.findtext("draft_analysis/preseason_average_pick"),
                "preseason_avg_round": player.findtext("draft_analysis/preseason_average_round"),
                "preseason_avg_cost": player.findtext("draft_analysis/preseason_average_cost"),
                "preseason_percent_drafted": player.findtext("draft_analysis/preseason_percent_drafted"),
                "is_keeper_status": (player.findtext("is_keeper/status") or ""),
                "is_keeper_cost": (player.findtext("is_keeper/cost") or "")
            }
            draft_analysis.append(player_data)

        start += len(players)

    return pd.DataFrame(draft_analysis)


# =============================================================================
# Data Processing
# =============================================================================



def merge_draft_data(
    picks: List[DraftPick],
    analysis_df: pd.DataFrame,
    team_key_to_manager: Dict[str, str],
    team_key_to_guid: Dict[str, str],
    player_id_to_team: Dict[str, str],
    player_id_to_name: Dict[str, str],
    manager_name_overrides: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Merge draft picks with analysis data and apply enrichments.

    IMPORTANT: This fetcher now outputs RAW DATA ONLY (no transformations).
    Value calculations (pick_savings, cost_savings, savings, cost_bucket) are
    handled by draft_enrichment_v2.py in the transformation layer.

    Args:
        picks: List of DraftPick objects
        analysis_df: DataFrame with draft analysis data
        team_key_to_manager: Dict mapping team keys to manager names
        team_key_to_guid: Dict mapping team keys to manager GUIDs
        player_id_to_team: Dict mapping player IDs to NFL teams
        player_id_to_name: Dict mapping player IDs to player names
        manager_name_overrides: Optional dict to override manager names

    Returns:
        Merged and enriched DataFrame with raw Yahoo API data only
    """
    # Convert picks to DataFrame
    picks_df = pd.DataFrame([p.__dict__ for p in picks])

    # Enrich picks with manager and team info
    picks_df['manager'] = picks_df['team_key'].map(team_key_to_manager).fillna("N/A")
    picks_df['manager_guid'] = picks_df['team_key'].map(team_key_to_guid)
    picks_df['nfl_team'] = picks_df['yahoo_player_id'].map(player_id_to_team).fillna("N/A")

    # Backfill missing player names
    picks_df['player'] = picks_df.apply(
        lambda row: player_id_to_name.get(str(row['yahoo_player_id']), row['player'])
        if row['player'] in ["", "N/A", None, pd.NA] else row['player'],
        axis=1
    )

    # Apply manager name overrides
    if manager_name_overrides:
        picks_df['manager'] = picks_df['manager'].apply(
            lambda x: manager_name_overrides.get(str(x or "").strip(), x)
        )

    # Normalize yahoo_player_id to string to avoid dtype mismatch during merge
    if 'yahoo_player_id' in picks_df.columns:
        picks_df['yahoo_player_id'] = picks_df['yahoo_player_id'].astype(str)
    if 'yahoo_player_id' in analysis_df.columns:
        analysis_df['yahoo_player_id'] = analysis_df['yahoo_player_id'].astype(str)

    # Merge with analysis data
    # Use analysis_df as base to get ALL draft-eligible players, not just drafted ones
    merged = pd.merge(
        analysis_df,
        picks_df,
        on=['yahoo_player_id', 'year'],
        how='left',
        suffixes=('_analysis', '')
    )

    # Fill missing columns from analysis
    for col in ['player', 'yahoo_position', 'avg_pick', 'avg_round', 'avg_cost', 'percent_drafted']:
        if f'{col}_analysis' in merged.columns:
            mask_empty = merged[col].isna() | (merged[col] == "")
            merged.loc[mask_empty, col] = merged.loc[mask_empty, f'{col}_analysis']
            merged = merged.drop(columns=[f'{col}_analysis'], errors='ignore')

    # Create composite keys
    merged['player_year'] = merged['player'].str.replace(" ", "", regex=False) + merged['year'].astype(str)
    merged['manager_year'] = merged['manager'].str.replace(" ", "", regex=False) + merged['year'].astype(str)

    # Add is_keeper_status and is_keeper_cost if missing
    if 'is_keeper_status' not in merged.columns:
        merged['is_keeper_status'] = ""
    if 'is_keeper_cost' not in merged.columns:
        merged['is_keeper_cost'] = ""

    # Add preseason columns if missing (for transformation layer)
    if 'preseason_avg_pick' not in merged.columns:
        merged['preseason_avg_pick'] = pd.NA
    if 'preseason_avg_round' not in merged.columns:
        merged['preseason_avg_round'] = pd.NA
    if 'preseason_avg_cost' not in merged.columns:
        merged['preseason_avg_cost'] = pd.NA
    if 'preseason_percent_drafted' not in merged.columns:
        merged['preseason_percent_drafted'] = pd.NA

    # Final column order (RAW DATA ONLY - no transformations)
    final_cols = [
        'year', 'pick', 'round', 'team_key', 'manager', 'manager_guid', 'yahoo_player_id', 'cost',
        'player', 'yahoo_position', 'avg_pick', 'avg_round', 'avg_cost', 'percent_drafted',
        'preseason_avg_pick', 'preseason_avg_round', 'preseason_avg_cost', 'preseason_percent_drafted',
        'is_keeper_status', 'is_keeper_cost',
        'player_year', 'manager_year', 'nfl_team', 'draft_type'
    ]

    # Only add columns that don't exist (don't overwrite existing ones)
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    # Ensure yahoo_player_id is consistently string for downstream joins
    out_df = merged[final_cols].copy()
    if 'yahoo_player_id' in out_df.columns:
        try:
            out_df['yahoo_player_id'] = out_df['yahoo_player_id'].astype('string')
        except Exception:
            out_df['yahoo_player_id'] = out_df['yahoo_player_id'].astype(str).astype('string')

    return out_df


# =============================================================================
# Main API
# =============================================================================

def fetch_draft_data(
    ctx: Optional[LeagueContext] = None,
    year: Optional[int] = None,
    oauth_file: Optional[Path] = None,
    league_key: Optional[str] = None,
    data_dir: Optional[Path] = None,
    logger: Optional[RunLogger] = None,
    timeout: int = 20
) -> pd.DataFrame:
    """
    Fetch draft data for a single year.

    Args:
        ctx: Optional LeagueContext for league-specific configuration
        year: Draft year to fetch
        oauth_file: Path to OAuth credentials (if no context)
        league_key: Yahoo league_key (if no context)
        data_dir: Custom data directory (overrides context)
        logger: Optional RunLogger instance

    Returns:
        DataFrame with draft data

    Raises:
        ValueError: If required parameters missing
        FileNotFoundError: If OAuth file not found
    """
    # Determine data directory
    if data_dir:
        draft_data_dir = Path(data_dir)
    elif ctx:
        draft_data_dir = ctx.draft_data_directory
    else:
        draft_data_dir = DEFAULT_DATA_ROOT

    if year is None:
        raise ValueError("year is required")

    # Create logger if context provided
    # Only instantiate RunLogger if it's available. Use a safe try/except so
    # a misconfigured RunLogger (None or non-callable) won't raise at runtime
    # and static analyzers are less likely to flag the call site.
    if logger is None and ctx and RUN_LOGGER_AVAILABLE:
        try:
            if RunLogger is None or not callable(RunLogger):
                raise RuntimeError("RunLogger not available or not callable")
            logger = RunLogger("draft_data", year=year, league_id=ctx.league_id)
            # If the logger implements context manager methods, enter it.
            if hasattr(logger, "__enter__"):
                logger.__enter__()
            close_logger = True
        except Exception:
            # Failed to initialize logger; continue without logging.
            logger = None
            close_logger = False
    else:
        close_logger = False

    try:
        # Initialize OAuth
        if logger:
            logger.start_step("initialize_oauth")

        if ctx:
            oauth = ctx.get_oauth_session()
            league_key = ctx.league_id
        elif oauth_file:
            oauth_path = Path(oauth_file)
            if not oauth_path.exists():
                raise FileNotFoundError(f"OAuth file not found: {oauth_path}")
            oauth = OAuth2(None, None, from_file=str(oauth_path))
            if not oauth.token_is_valid():
                oauth.refresh_access_token()
        else:
            raise ValueError("Either ctx or oauth_file is required")

        gm = yfa.Game(oauth, 'nfl')

        if logger:
            logger.complete_step()

        # Get league ID for year
        if logger:
            logger.start_step("get_league_id")

        # CRITICAL: Use specific league_id from context to avoid data mixing
        year_league_id = None

        # Try context first (safest - ensures league isolation)
        if ctx and hasattr(ctx, 'get_league_id_for_year'):
            year_league_id = ctx.get_league_id_for_year(year)
            if year_league_id:
                print(f"[draft] Using league_id from context for {year}: {year_league_id}")

        # Fallback to explicit league_key parameter
        if not year_league_id and league_key:
            year_league_id = league_key
            print(f"[draft] Using explicit league_key parameter: {year_league_id}")

        # Last resort: use API discovery (may mix leagues!)
        if not year_league_id:
            league_ids = gm.league_ids(year=year)
            if not league_ids:
                raise ValueError(f"No league found for year {year}")
            if len(league_ids) > 1:
                print(f"[draft] WARNING: Multiple leagues found for {year}: {league_ids}")
                print(f"[draft] WARNING: Using last one - this may cause data mixing!")
            year_league_id = league_ids[-1]

        print(f"[draft] Using league ID: {year_league_id}")

        # --- LOAD settings from saved file (fetched in PHASE 0 of initial_import_v2.py) ---
        if LOAD_SETTINGS_AVAILABLE and load_league_settings:
            # NEW location (as of 2025): {data_directory}/league_settings/
            # OLD location (backwards compatibility): {data_directory}/player_data/yahoo_league_settings/
            if ctx:
                # Try NEW location first
                settings_dir = Path(ctx.data_directory) / "league_settings"
                if not settings_dir.exists():
                    # Fallback to OLD location
                    settings_dir = Path(ctx.player_data_directory) / "yahoo_league_settings"
            elif data_dir:
                # Try NEW location first
                settings_dir = Path(data_dir) / "league_settings"
                if not settings_dir.exists():
                    # Fallback to OLD location
                    settings_dir = Path(data_dir) / "player_data" / "yahoo_league_settings"
            else:
                # Try NEW location first
                settings_dir = DEFAULT_DATA_ROOT / "league_settings"
                if not settings_dir.exists():
                    # Fallback to OLD location
                    settings_dir = DEFAULT_DATA_ROOT / "player_data" / "yahoo_league_settings"

            saved_settings = load_league_settings(year, league_key, settings_dir)
            if saved_settings:
                draft_type = (saved_settings.get("metadata", {}).get("draft_type") or "").lower()
                print(f"[draft] Loaded settings from file: draft_type={draft_type or 'unknown'}")
            else:
                draft_type = ""
                print(f"[draft] WARNING: No saved settings found for {year} - using fallback heuristics")
        else:
            draft_type = ""
            print(f"[draft] WARNING: Settings loader not available - using fallback heuristics")

        is_auction = (draft_type == "auction")
        print(f"[draft] is_auction={is_auction}")

        if logger:
            logger.complete_step(league_id=year_league_id)

        # --- Fallback heuristic if Yahoo draft_type missing/unknown ---
        if draft_type not in ("auction", "snake"):
            # We'll decide after we merge picks+analysis below. For now, keep False.
            pass

        # Fetch mappings
        if logger:
            logger.start_step("fetch_mappings")

        # Get manager_name_overrides from context (for --hidden-- fallback)
        manager_overrides = ctx.manager_name_overrides if ctx else {}

        team_key_to_manager, team_key_to_guid, player_id_to_name, player_id_to_team = \
            fetch_team_and_player_mappings(
                oauth,
                year_league_id,
                timeout=timeout,
                manager_name_overrides=manager_overrides
            )

        print(f"[draft] Fetched {len(team_key_to_manager)} teams, {len(player_id_to_name)} players")

        if logger:
            logger.complete_step(
                teams=len(team_key_to_manager),
                players=len(player_id_to_name)
            )

        # Fetch draft picks
        if logger:
            logger.start_step("fetch_draft_picks")

        picks = fetch_draft_picks(oauth, year_league_id, year)
        print(f"[draft] Fetched {len(picks)} draft picks")

        if logger:
            logger.complete_step(picks=len(picks))

        # Fetch draft analysis
        if logger:
            logger.start_step("fetch_draft_analysis")

        analysis_df = fetch_draft_analysis(oauth, year_league_id, year, timeout=timeout)
        print(f"[draft] Fetched draft analysis for {len(analysis_df)} players")

        if logger:
            logger.complete_step(players=len(analysis_df))

        # Merge data
        if logger:
            logger.start_step("merge_data")

        manager_overrides = ctx.manager_name_overrides if ctx else {}
        final_df = merge_draft_data(
            picks,
            analysis_df,
            team_key_to_manager,
            team_key_to_guid,
            player_id_to_team,
            player_id_to_name,
            manager_overrides,
        )

        # Store draft_type in DataFrame for downstream transformation layer
        # (draft_enrichment_v2.py needs this to calculate appropriate value metrics)
        final_df['draft_type'] = draft_type

        if logger:
            logger.complete_step(rows=len(final_df))

        # Ensure Yahoo player ID is a string.  Downstream merges depend
        # on matching yahoo_player_id across data sources; casting to a
        # consistent string dtype avoids object/int mismatches.
        if "yahoo_player_id" in final_df.columns:
            try:
                final_df["yahoo_player_id"] = final_df["yahoo_player_id"].astype("string")
            except Exception:
                final_df["yahoo_player_id"] = final_df["yahoo_player_id"].astype(str).astype("string")

        # Add league_id for multi-league isolation (use year-specific league_id)
        final_df["league_id"] = year_league_id

        # Save outputs
        if logger:
            logger.start_step("save_outputs")

        draft_data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = draft_data_dir / f"draft_data_{year}.csv"
        pq_path = draft_data_dir / f"draft_data_{year}.parquet"

        final_df.to_csv(csv_path, index=False, encoding="utf-8", errors="replace")
        try:
            final_df.to_parquet(pq_path, index=False)
        except Exception as e:
            print(f"[warn] Parquet write failed: {e}")

        print(f"\n[output] Saved CSV: {csv_path}")
        print(f"[output] Saved Parquet: {pq_path}")
        print(f"[output] Total rows: {len(final_df):,}")

        if logger:
            logger.complete_step(files_written=2, rows_written=len(final_df))

        return final_df

    finally:
        if close_logger and logger:
            logger.__exit__(None, None, None)


def _year_done(output_dir: Path, year: int) -> bool:
    return (output_dir / f"draft_data_{year}.parquet").exists()


def fetch_all_draft_years(
    ctx: Optional[LeagueContext] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    oauth_file: Optional[Path] = None,
    league_key: Optional[str] = None,
    data_dir: Optional[Path] = None,
    resume: bool = False,
    timeout: int = 20
) -> pd.DataFrame:
    """
    Fetch draft data for all years.

    Args:
        ctx: Optional LeagueContext for league-specific configuration
        start_year: First year to fetch (defaults to ctx.start_year or 2014)
        end_year: Last year to fetch (defaults to current year)
        oauth_file: Path to OAuth credentials (if no context)
        league_key: Yahoo league_key (if no context)
        data_dir: Custom data directory (overrides context)

    Returns:
        DataFrame with all years combined
    """
    from datetime import datetime

    if start_year is None:
        start_year = ctx.start_year if ctx else 2014

    if end_year is None:
        current_year = datetime.now().year
        end_year = ctx.end_year if ctx and ctx.end_year else current_year

    # Determine data directory
    if data_dir:
        draft_data_dir = Path(data_dir)
    elif ctx:
        draft_data_dir = ctx.draft_data_directory
    else:
        draft_data_dir = DEFAULT_DATA_ROOT

    print(f"[draft] Fetching years {start_year} to {end_year}")

    all_dfs = []
    for year in range(end_year, start_year - 1, -1):
        try:
            df = fetch_draft_data(
                ctx=ctx,
                year=year,
                oauth_file=oauth_file,
                league_key=league_key,
                data_dir=data_dir
            )
            all_dfs.append(df)
        except Exception as e:
            print(f"[warn] Failed to fetch year {year}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No draft data was successfully fetched")

    # Combine all years
    combined = pd.concat(all_dfs, ignore_index=True)

    # Save combined file
    csv_path = draft_data_dir / "draft_data_all_years.csv"
    pq_path = draft_data_dir / "draft_data_all_years.parquet"

    combined.to_csv(csv_path, index=False)
    try:
        combined.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"[warn] Parquet write failed: {e}")

    print(f"\n[output] Saved combined CSV: {csv_path}")
    print(f"[output] Saved combined Parquet: {pq_path}")
    print(f"[output] Total rows: {len(combined):,}")

    return combined


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Fetch Yahoo Fantasy Football draft data (multi-league compliant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With LeagueContext
  python draft_data_v2.py --context leagues/kmffl/league_context.json --year 2024

  # All years with context
  python draft_data_v2.py --context leagues/kmffl/league_context.json --all-years

  # Standalone (backward compatible)
  python draft_data_v2.py --year 2024 --oauth Oauth.json --league-key nfl.l.123456
        """
    )

    parser.add_argument('--context', type=Path, help='Path to league_context.json')
    parser.add_argument('--year', type=int, help='Single year to fetch (overrides --all-years if provided)')
    parser.add_argument('--all-years', action='store_true', help='Fetch all years')
    parser.add_argument('--resume', action='store_true', help='Skip years already written to output folder')
    parser.add_argument('--per-request-timeout', type=int, default=20, help='HTTP timeout seconds')
    parser.add_argument('--oauth', type=Path, help='Path to OAuth credentials (if no context)')
    parser.add_argument('--league-key', help='Yahoo league_key (if no context)')
    parser.add_argument('--data-dir', type=Path, help='Custom data directory')

    args = parser.parse_args()

    # Load context if provided
    ctx = None
    if args.context:
        if not LEAGUE_CONTEXT_AVAILABLE:
            print("Error: league_context module not available", file=sys.stderr)
            sys.exit(1)

        try:
            ctx = LeagueContext.load(args.context)
            print(f"Loaded context: {ctx.league_name}")
        except Exception as e:
            print(f"Error loading context: {e}", file=sys.stderr)
            sys.exit(1)

    # Validate arguments
    if not args.all_years and not args.year:
        print("Error: Must specify either --year or --all-years", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Run fetch
    try:
        if args.all_years:
            fetch_all_draft_years(
                ctx=ctx,
                oauth_file=args.oauth,
                league_key=args.league_key,
                data_dir=args.data_dir
            )
        else:
            fetch_draft_data(
                ctx=ctx,
                year=args.year,
                oauth_file=args.oauth,
                league_key=args.league_key,
                data_dir=args.data_dir
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
