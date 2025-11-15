#!/usr/bin/env python3
"""
Yahoo Fantasy Roster Data Fetcher - Multi-League Edition

Fetches weekly roster data showing which manager owned which player each week.
Compatible with multi-league infrastructure.

Output includes:
- manager_name: Team owner
- player_name: Player name
- yahoo_position: Yahoo's position designation
- primary_position: Primary position (QB, RB, WR, TE, K, DEF)
- fantasy_position: Roster slot (QB, RB1, RB2, FLEX, BN, etc.)
- year, week
- player stats for that week

Usage:
    # Using league context (RECOMMENDED)
    python yahoo_fantasy_data.py --context path/to/league_context.json

    # Fetch specific year
    python yahoo_fantasy_data.py --context path/to/league_context.json --year 2024

    # Fetch specific year and week
    python yahoo_fantasy_data.py --context path/to/league_context.json --year 2024 --week 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Add paths for imports (same approach as transactions_v2.py)
_script_file = Path(__file__).resolve()
_multi_league_dir = _script_file.parent.parent  # multi_league directory
sys.path.insert(0, str(_multi_league_dir / "core"))
sys.path.insert(0, str(_multi_league_dir / "utils"))

# Import from multi_league.core
try:
    from league_context import LeagueContext
    from league_discovery import LeagueDiscovery
    from script_runner import log
except ImportError as e:
    print(f"ERROR: Failed to import multi_league modules: {e}")
    print("Make sure you're running from the correct directory.")
    sys.exit(1)

# Try to import Yahoo OAuth
try:
    from yahoo_oauth import OAuth2
    YAHOO_OAUTH_AVAILABLE = True
except ImportError:
    OAuth2 = None
    YAHOO_OAUTH_AVAILABLE = False
    print("Warning: yahoo_oauth not available. Install with: pip install yahoo_oauth")


def retry_with_backoff(func, max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to call (should take no arguments)
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        backoff_factor: Multiplier for delay after each retry (default: 2.0)

    Returns:
        Result of func() if successful

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                # Check if it's a rate limit error (429) or server error (5xx)
                error_str = str(e).lower()
                is_retryable = (
                    'rate limit' in error_str or
                    '429' in error_str or
                    '5' in error_str[:3] if len(error_str) >= 3 else False or
                    'timeout' in error_str or
                    'connection' in error_str
                )

                if is_retryable:
                    log(f"  [RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                    log(f"  Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    # Non-retryable error, raise immediately
                    raise e
            else:
                # Max retries exceeded
                log(f"  [FAIL] Max retries ({max_retries}) exceeded")
                raise last_exception

    raise last_exception


class YahooRosterFetcher:
    """
    Fetch weekly roster data from Yahoo Fantasy API.

    Shows which manager had which player in which roster slot each week.
    """

    def __init__(
        self,
        oauth_file: Path,
        league_id: str,
        rate_limit: float = 2.0,
        max_retries: int = 5,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the roster fetcher.

        Args:
            oauth_file: Path to OAuth JSON file
            league_id: Yahoo league key (e.g., "331.l.381581")
            rate_limit: Max requests per second
            max_retries: Max retry attempts
            output_dir: Directory to save output files
        """
        self.league_id = league_id
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OAuth
        self.oauth = self._initialize_oauth(oauth_file)

        # Rate limiting
        self.last_request_time = 0.0
        self.request_count = 0

    def _initialize_oauth(self, oauth_file: Path) -> OAuth2:
        """Initialize OAuth session."""
        if not YAHOO_OAUTH_AVAILABLE:
            raise ImportError("yahoo_oauth is required. Install with: pip install yahoo_oauth")

        if not oauth_file.exists():
            raise FileNotFoundError(f"OAuth file not found: {oauth_file}")

        log(f"Initializing OAuth from file: {oauth_file}")
        oauth = OAuth2(None, None, from_file=str(oauth_file))

        # Validate token
        if not oauth.token_is_valid():
            log("Refreshing OAuth token")
            oauth.refresh_access_token()

        return oauth

    def _rate_limit_wait(self):
        """Wait if necessary to respect rate limit."""
        if self.rate_limit <= 0:
            return

        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.rate_limit

        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            time.sleep(wait_time)

    def _fetch_url_xml(self, url: str) -> ET.Element:
        """Fetch XML from Yahoo API with retries."""
        last_error = None
        backoff = 0.5

        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()

                # log(f"Fetching: {url[:100]}...")  # Removed DEBUG level logging
                response = self.oauth.session.get(url, timeout=30)
                response.raise_for_status()

                self.last_request_time = time.time()
                self.request_count += 1

                text = response.text or ""

                # Check for error messages
                if "Request denied" in text or "limit exceeded" in text.lower() or "permission" in text.lower():
                    raise RuntimeError("API request denied, rate limited, or permission error")

                # Remove XML namespace for easier parsing
                text = pd.Series(text).str.replace(r' xmlns="[^"]+"', "", n=1, regex=True).iloc[0]

                return ET.fromstring(text)

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Check if it's a rate limit or permission error
                if "rate limit" in error_msg or "permission" in error_msg or "denied" in error_msg or "limit exceeded" in error_msg:
                    log(f"Rate limit or permission error detected: {e}")
                    log(f"Waiting 5 minutes before retrying...")
                    time.sleep(300)  # Wait 5 minutes
                    log(f"Resuming after 5-minute wait")
                    continue

                log(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    sleep_time = backoff * (2 ** attempt)
                    log(f"Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)

        raise RuntimeError(f"Failed to fetch URL after {self.max_retries} attempts: {last_error}")

    def fetch_teams(self) -> Dict[str, str]:
        """
        Fetch all teams in the league.

        Returns:
            Dict mapping team_key to manager_name
        """
        url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{self.league_id}/teams"

        log(f"Fetching teams for league {self.league_id}")

        try:
            root = self._fetch_url_xml(url)

            teams = {}
            for team_elem in root.findall(".//team"):
                team_key_elem = team_elem.find("team_key")

                # Try to get manager name
                manager_elem = team_elem.find(".//manager/nickname")
                if manager_elem is None:
                    manager_elem = team_elem.find(".//manager/guid")

                # Fallback to team name
                name_elem = team_elem.find("name")

                if team_key_elem is not None:
                    team_key = team_key_elem.text

                    if manager_elem is not None:
                        manager_name = manager_elem.text
                    elif name_elem is not None:
                        manager_name = name_elem.text
                    else:
                        manager_name = f"Team {team_key}"

                    teams[team_key] = manager_name

            log(f"Found {len(teams)} teams")
            for team_key, manager_name in teams.items():
                log(f"  {team_key}: {manager_name}")

            return teams

        except Exception as e:
            log(f"Error fetching teams: {e}")
            return {}

    def fetch_league_weeks(self) -> Optional[int]:
        """
        Fetch the number of weeks in the fantasy season from league settings.

        Returns:
            Number of weeks in the fantasy season, or None if unable to determine
        """
        url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{self.league_id}/settings"

        log(f"Fetching league settings to determine number of weeks")

        try:
            root = self._fetch_url_xml(url)

            # Look for playoff_start_week to determine last week of fantasy season
            playoff_start = root.find(".//playoff_start_week")
            if playoff_start is not None:
                playoff_week = int(playoff_start.text)
                log(f"  Playoff start week: {playoff_week}")

                # Also check for number of playoff weeks
                num_playoff_teams = root.find(".//num_playoff_teams")

                # Most leagues have 2-3 weeks of playoffs
                # Conservative estimate: playoff_start + 2 weeks
                last_week = playoff_week + 2

                log(f"  Estimated last fantasy week: {last_week}")
                return last_week

            # Fallback: look for current_week or end_week
            current_week = root.find(".//current_week")
            if current_week is not None:
                weeks = int(current_week.text)
                log(f"  Using current_week from settings: {weeks}")
                return weeks

            # If we can't determine, return None
            log(f"  Could not determine number of weeks from league settings")
            return None

        except Exception as e:
            log(f"Error fetching league weeks: {e}")
            return None

    def fetch_current_week(self) -> Optional[int]:
        """
        Fetch the current week from Yahoo API.

        Returns:
            Current week number, or None if unable to determine
        """
        url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{self.league_id}/settings"

        try:
            root = self._fetch_url_xml(url)
            current_week = root.find(".//current_week")
            if current_week is not None:
                week = int(current_week.text)
                log(f"  Yahoo API current_week: {week}")
                return week
            return None
        except Exception as e:
            log(f"Error fetching current week: {e}")
            return None

    def fetch_roster_for_week(
        self,
        year: int,
        week: int,
        team_key: str,
        manager_name: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch roster for a specific team and week.

        Args:
            year: Season year
            week: Week number
            team_key: Yahoo team key
            manager_name: Manager name

        Returns:
            List of roster entries (one per player)
        """
        # Use the EXACT same URL format as the working old script
        url = (
            f"https://fantasysports.yahooapis.com/fantasy/v2/"
            f"team/{team_key}/roster;week={week}/players/stats"
        )

        roster_data = []

        try:
            root = self._fetch_url_xml(url)

            for player_elem in root.findall(".//player"):
                try:
                    player_info = {
                        'year': year,
                        'week': week,
                        'manager_name': manager_name,
                        'team_key': team_key,
                    }

                    # Player identification
                    player_key = player_elem.find("player_key")
                    if player_key is not None:
                        player_info['player_key'] = player_key.text

                    player_id = player_elem.find("player_id")
                    if player_id is not None:
                        player_info['player_id'] = player_id.text

                    # Player name
                    name_elem = player_elem.find("name")
                    if name_elem is not None:
                        full_name = name_elem.find("full")
                        if full_name is not None:
                            player_info['player_name'] = full_name.text

                    # NFL team
                    editorial_team = player_elem.find("editorial_team_abbr")
                    if editorial_team is not None:
                        # Normalize to uppercase for consistency with NFLverse data
                        player_info['nfl_team'] = editorial_team.text.upper() if editorial_team.text else None

                    # Yahoo position (display position)
                    display_position = player_elem.find("display_position")
                    if display_position is not None:
                        player_info['yahoo_position'] = display_position.text

                    # Primary position
                    primary_position = player_elem.find("primary_position")
                    if primary_position is not None:
                        player_info['primary_position'] = primary_position.text
                    else:
                        # Fallback to display_position
                        player_info['primary_position'] = player_info.get('yahoo_position')

                    # Eligible positions
                    eligible_positions = player_elem.find("eligible_positions")
                    if eligible_positions is not None:
                        positions_list = [p.text for p in eligible_positions.findall("position")]
                        player_info['eligible_positions'] = ",".join(positions_list)

                    # Selected position (roster slot)
                    selected_position = player_elem.find("selected_position")
                    if selected_position is not None:
                        position_elem = selected_position.find("position")
                        if position_elem is not None:
                            player_info['fantasy_position'] = position_elem.text

                    # Extract fantasy points - EXACT same logic as working script
                    pts_node = player_elem.find("player_points/total")
                    try:
                        # Use the exact same approach as the old working script
                        pts_text = pts_node.text if pts_node is not None else "0"
                        player_info['fantasy_points'] = round(float(pts_text or "0"), 2)
                    except (ValueError, TypeError):
                        player_info['fantasy_points'] = 0.0

                    roster_data.append(player_info)

                except Exception as e:
                    log(f"Error parsing player: {e}")
                    continue

        except Exception as e:
            log(f"Error fetching roster for {team_key} week {week}: {e}")

        return roster_data

    def fetch_all_rosters_for_week(
        self,
        year: int,
        week: int,
        teams: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Fetch all rosters for all teams for a specific week.
        Uses INDIVIDUAL calls per team to ensure we get player_points data.

        Args:
            year: Season year
            week: Week number
            teams: Dict mapping team_key to manager_name

        Returns:
            DataFrame with all roster data
        """
        log(f"Fetching all rosters for week {week}")

        # Use individual team requests since batch API doesn't return player_points reliably
        return self._fetch_all_rosters_individually(year, week, teams)

    def _fetch_all_rosters_individually(
        self,
        year: int,
        week: int,
        teams: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Fetch rosters individually for each team IN PARALLEL.

        Args:
            year: Season year
            week: Week number
            teams: Dict mapping team_key to manager_name

        Returns:
            DataFrame with all roster data
        """
        all_rosters = []

        def fetch_team_roster(team_key: str, manager_name: str):
            """Fetch roster for a single team (runs in parallel)"""
            try:
                roster_data = self.fetch_roster_for_week(year, week, team_key, manager_name)
                return (team_key, manager_name, True, roster_data)
            except Exception as e:
                return (team_key, manager_name, False, str(e))

        # Parallel execution with max 3 workers (respects rate limiting with built-in _rate_limit_wait)
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_team = {
                executor.submit(fetch_team_roster, team_key, manager_name): (team_key, manager_name)
                for team_key, manager_name in teams.items()
            }

            for future in as_completed(future_to_team):
                team_key, manager_name, success, result = future.result()

                if success:
                    all_rosters.extend(result)
                else:
                    log(f"Error fetching roster for {manager_name}: {result}")

        if not all_rosters:
            return pd.DataFrame()

        return pd.DataFrame(all_rosters)

    def fetch_season_rosters(
        self,
        year: int,
        weeks: Optional[List[int]] = None,
        end_week: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch roster data for an entire season using optimized batch API.

        Args:
            year: Season year
            weeks: List of weeks to fetch (auto-detect if None)
            end_week: Last week of fantasy season (from league settings)

        Returns:
            DataFrame with all roster data for the season
        """
        if weeks is None:
            if end_week is not None:
                # Use the end_week from league settings
                weeks = list(range(1, end_week + 1))
                log(f"Using league settings: weeks 1-{end_week}")
            else:
                # Fallback to defaults if we can't determine from league settings
                if year >= 2021:
                    weeks = list(range(1, 18))  # Weeks 1-17 (17 regular season weeks)
                else:
                    weeks = list(range(1, 17))  # Weeks 1-16 (16 regular season)
                log(f"Using default weeks for {year}: {weeks[0]}-{weeks[-1]}")

        log(f"Fetching season {year} roster data for weeks: {weeks}")

        # Fetch teams first
        teams = self.fetch_teams()

        if not teams:
            log("No teams found!")
            return pd.DataFrame()

        # Fetch all weeks using optimized batch API
        try:
            return self._fetch_all_weeks_batch(year, weeks, teams)
        except Exception as e:
            log(f"Batch fetch for all weeks failed: {e}")
            log("Falling back to week-by-week fetching...")
            return self._fetch_week_by_week(year, weeks, teams)

    def _fetch_all_weeks_batch(
        self,
        year: int,
        weeks: List[int],
        teams: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Fetch all weeks using optimized batch API calls.

        Yahoo's API limitation: Cannot fetch all teams+weeks+roster+stats in ONE call.
        Best approach: Fetch all teams for each week in one call (1 API call per week).

        This is significantly better than individual team calls (would be teams*weeks calls).
        """
        log(f"Using OPTIMIZED batch API: {len(weeks)} API calls (1 per week) instead of {len(teams)*len(weeks)} individual calls")

        all_rosters = []

        for i, week in enumerate(weeks, 1):
            log(f"  [{i}/{len(weeks)}] Fetching all {len(teams)} teams for week {week}...")

            try:
                # Fetch all teams' rosters for this week in ONE call
                week_df = self.fetch_all_rosters_for_week(year, week, teams)

                if not week_df.empty:
                    all_rosters.append(week_df)
                    log(f"    Fetched {len(week_df)} total player-week records for week {week}")

                # Small delay between weeks to be respectful to API
                if i < len(weeks):
                    time.sleep(0.5)

            except Exception as e:
                log(f"Error fetching week {week}: {e}")
                # Continue with next week instead of failing completely
                continue

        if not all_rosters:
            raise RuntimeError("No data returned from batch API")

        df = pd.concat(all_rosters, ignore_index=True)
        log(f"[OK] Successfully fetched {len(df)} total roster records across {len(weeks)} weeks using {len(weeks)} API calls")

        return df

    def _fetch_week_by_week(
        self,
        year: int,
        weeks: List[int],
        teams: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Fallback: Fetch rosters week by week using batch API for teams.
        """
        log(f"Using week-by-week batch API (1 call per week)")

        all_weeks_data = []

        for week in weeks:
            try:
                week_df = self.fetch_all_rosters_for_week(year, week, teams)

                if not week_df.empty:
                    all_weeks_data.append(week_df)

                # Delay between weeks
                time.sleep(1.0)

            except Exception as e:
                log(f"Error fetching week {week}: {e}")
                continue

        if not all_weeks_data:
            log(f"No roster data fetched for season {year}")
            return pd.DataFrame()

        df = pd.concat(all_weeks_data, ignore_index=True)
        log(f"Total roster records for {year}: {len(df)}")

        return df

    def save_to_parquet(self, df: pd.DataFrame, year: int, filename: Optional[str] = None):
        """Save DataFrame to Parquet and CSV files."""
        if df.empty:
            log("DataFrame is empty, skipping save")
            return

        if filename is None:
            # FIXED: Use naming pattern that yahoo_nfl_merge.py expects
            base_filename = f"yahoo_player_stats_{year}_all_weeks"
        else:
            base_filename = filename.replace('.parquet', '').replace('.csv', '')

        # Convert to proper types
        df = self._prepare_dataframe_for_export(df)

        # FIXED: Standardize column names to match merge script expectations
        column_mapping = {
            'player_name': 'player',
            'manager_name': 'manager',
            'player_id': 'yahoo_player_id',
            'fantasy_points': 'points',
            'nfl_team': 'nfl_team',
            'primary_position': 'nfl_position'
        }

        # Apply column renames (only if columns exist)
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Save to parquet
        parquet_path = self.output_dir / f"{base_filename}.parquet"
        log(f"Saving {len(df)} records to {parquet_path}")
        df.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy')
        log(f"Successfully saved Parquet to {parquet_path}")

        # Save to CSV
        csv_path = self.output_dir / f"{base_filename}.csv"
        log(f"Saving {len(df)} records to {csv_path}")
        df.to_csv(csv_path, index=False)
        log(f"Successfully saved CSV to {csv_path}")

    def _prepare_dataframe_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame with proper types for Parquet export."""
        df = df.copy()

        # Convert numeric columns
        numeric_cols = ['year', 'week', 'player_id', 'fantasy_points']
        stat_cols = [col for col in df.columns if col.startswith('stat_')]
        numeric_cols.extend(stat_cols)

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Reorder columns to show important fields first
        priority_cols = [
            'year', 'week', 'manager_name', 'player_name',
            'fantasy_points', 'yahoo_position', 'primary_position',
            'fantasy_position', 'nfl_team'
        ]

        # Get remaining columns
        other_cols = [col for col in df.columns if col not in priority_cols]

        # Reorder: priority columns first, then the rest
        final_cols = [col for col in priority_cols if col in df.columns] + other_cols
        df = df[final_cols]

        return df


# =============================================================================
# Multi-League Support Functions
# =============================================================================

def find_league_settings_files(league_dir: Path) -> Dict[int, Path]:
    """
    Find all league settings JSON files for a league.

    Args:
        league_dir: Path to league directory (e.g., .../fantasy_football_data/KMFFL)

    Returns:
        Dict mapping year to settings file path
    """
    # League settings are now at top level (league-wide config, not player-specific)
    settings_dir = league_dir / "league_settings"

    if not settings_dir.exists():
        log(f"League settings directory not found: {settings_dir}")
        return {}

    year_to_file = {}
    for settings_file in settings_dir.glob("league_settings_*.json"):
        try:
            # Extract year from filename: league_settings_2024_449_l_198278.json
            parts = settings_file.stem.split('_')
            if len(parts) >= 3 and parts[2].isdigit():
                year = int(parts[2])
                year_to_file[year] = settings_file
        except (ValueError, IndexError) as e:
            log(f"Could not parse year from {settings_file.name}: {e}")
            continue

    return year_to_file


def load_league_settings(settings_file: Path) -> Dict[str, Any]:
    """
    Load league settings from JSON file.

    Args:
        settings_file: Path to league settings JSON file

    Returns:
        Dict with league settings including year, league_key, end_week
    """
    try:
        with open(settings_file, 'r') as f:
            data = json.load(f)

        return {
            'year': data.get('year'),
            'league_key': data.get('league_key'),
            'end_week': data.get('metadata', {}).get('end_week'),
            'start_week': data.get('metadata', {}).get('start_week', 1),
            'num_teams': data.get('metadata', {}).get('num_teams'),
        }
    except Exception as e:
        log(f"Error loading league settings from {settings_file}: {e}")
        return {}


def load_discovered_leagues(league_dir: Path, league_name: str) -> Dict[int, str]:
    """
    Load year-specific league IDs from discovered_leagues.json.

    Args:
        league_dir: Path to the parent directory of the league (e.g., .../fantasy_football_data)
        league_name: Name of the league (e.g., KMFFL)

    Returns:
        Dict mapping year to league ID (e.g., 449.l.198278)
    """
    try:
        discovered_file = league_dir / "discovered_leagues.json"

        if not discovered_file.exists():
            log(f"discovered_leagues.json not found: {discovered_file}")
            return {}

        with open(discovered_file, 'r') as f:
            data = json.load(f)

        # Extract year and league ID mappings
        # The file is a flat array, not nested under 'leagues'
        year_to_league_id = {}
        for entry in data:
            if entry.get('league_name') == league_name:
                year = entry.get('year')
                league_id = entry.get('league_id')
                if year and league_id:
                    year_to_league_id[year] = league_id

        return year_to_league_id

    except Exception as e:
        log(f"Error loading discovered leagues from {league_dir}: {e}")
        return {}


def get_max_week_from_matchup_data(data_directory: Path, year: int) -> Optional[int]:
    """
    Get the maximum week from matchup data files.

    This allows player data to align with matchup data (only fetch weeks with actual matchups).

    Args:
        data_directory: League data directory (e.g., .../fantasy_football_data/KMFFL)
        year: Year to check

    Returns:
        Maximum week number found in matchup data, or None if no matchup data exists
    """
    try:
        matchup_dir = data_directory / "matchup_data"

        if not matchup_dir.exists():
            log(f"[matchup_max_week] Matchup directory not found: {matchup_dir}")
            return None

        # Try to find matchup file for this year
        # Prefer all-weeks file, fallback to individual week files
        all_weeks_file = matchup_dir / f"matchup_data_week_all_year_{year}.parquet"

        if all_weeks_file.exists():
            try:
                df = pd.read_parquet(all_weeks_file)
                if not df.empty and 'week' in df.columns:
                    max_week = int(df['week'].max())
                    log(f"[matchup_max_week] Found max week {max_week} from {all_weeks_file.name}")
                    return max_week
            except Exception as e:
                log(f"[matchup_max_week] Error reading {all_weeks_file.name}: {e}")

        # Fallback: check individual week files
        week_files = list(matchup_dir.glob(f"matchup_data_week_*_year_{year}.parquet"))
        if week_files:
            # Extract week numbers from filenames
            week_numbers = []
            for wf in week_files:
                try:
                    # Parse filename: matchup_data_week_05_year_2024.parquet
                    parts = wf.stem.split('_')
                    if len(parts) >= 5:
                        week_str = parts[3]  # "05"
                        if week_str != "all":
                            week_numbers.append(int(week_str))
                except (ValueError, IndexError):
                    continue

            if week_numbers:
                max_week = max(week_numbers)
                log(f"[matchup_max_week] Found max week {max_week} from {len(week_files)} individual week files")
                return max_week

        log(f"[matchup_max_week] No matchup data found for year {year}")
        return None

    except Exception as e:
        log(f"[matchup_max_week] Error getting max week from matchup data: {e}")
        return None


def main():
    """Fetch roster data for a league using LeagueContext."""

    parser = argparse.ArgumentParser(
        description="Fetch Yahoo Fantasy roster data using multi-league infrastructure",
        epilog="""
Examples:
    # Fetch all years using league context (RECOMMENDED)
    python yahoo_fantasy_data.py --context path/to/league_context.json

    # Fetch specific year
    python yahoo_fantasy_data.py --context path/to/league_context.json --year 2024

    # Fetch specific week
    python yahoo_fantasy_data.py --context path/to/league_context.json --year 2024 --week 5
        """
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="Path to league_context.json (required)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=0,
        help="Specific year to fetch (0 = all years, default: 0)"
    )
    parser.add_argument(
        "--week",
        type=int,
        default=0,
        help="Specific week to fetch (0 = all weeks, default: 0)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        help="Max API requests per second (default: from context or 2.0)"
    )

    args = parser.parse_args()

    log("=" * 70)
    log("Yahoo Fantasy Roster Data Fetcher")
    log("=" * 70)

    # Load league context using the proper LeagueContext class
    context_file = Path(args.context)

    if not context_file.exists():
        log(f"League context not found: {context_file}")
        sys.exit(1)

    try:
        ctx = LeagueContext.load(str(context_file))
        log(f"Loaded context for league: {ctx.league_name}")
    except Exception as e:
        log(f"Failed to load league context: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Set rate limit from args or context
    rate_limit = args.rate_limit if args.rate_limit else ctx.rate_limit_per_sec

    # Output directory from context
    output_dir = Path(ctx.data_directory) / "player_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup OAuth - use embedded credentials or file path
    oauth_file = None
    temp_oauth_file = None

    try:
        if ctx.oauth_credentials:
            # Create temporary OAuth file from embedded credentials
            import tempfile
            import os

            temp_fd, temp_path = tempfile.mkstemp(suffix='.json', text=True)
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(ctx.oauth_credentials, f, indent=2)
                oauth_file = Path(temp_path)
                temp_oauth_file = oauth_file
                log(f"Created temporary OAuth file from context credentials")
            except Exception as e:
                os.close(temp_fd)
                raise e

        elif ctx.oauth_file_path:
            oauth_file = Path(ctx.oauth_file_path)
            if not oauth_file.exists():
                log(f"OAuth file not found: {oauth_file}")
                sys.exit(1)
            log(f"Using OAuth file: {oauth_file}")
        else:
            log("No OAuth credentials found in context")
            sys.exit(1)

        # Use LeagueDiscovery to find league IDs for each year
        # First check cache (discovered_leagues.json) to avoid redundant API calls
        league_dir = Path(ctx.data_directory).parent if ctx.data_directory else Path.cwd()
        discovered_file = league_dir / "discovered_leagues.json"

        year_to_league_id = {}

        # Try to load from cache first
        if discovered_file.exists():
            log(f"\nLoading cached league IDs from {discovered_file.name}...")
            cached_leagues = load_discovered_leagues(league_dir, ctx.league_name)

            if cached_leagues:
                year_to_league_id = cached_leagues
                log(f"Loaded {len(year_to_league_id)} years from cache:")
                for year in sorted(year_to_league_id.keys()):
                    log(f"  {year}: {year_to_league_id[year]}")
            else:
                log(f"  No cached data found for '{ctx.league_name}'")

        # Determine current year and check if cache is complete
        current_year = datetime.now().year
        start_year = ctx.start_year or 2014
        end_year = ctx.end_year or current_year

        # Check if cache is missing years (either empty or outdated)
        cached_max_year = max(year_to_league_id.keys()) if year_to_league_id else 0
        needs_discovery = not year_to_league_id or cached_max_year < current_year

        if needs_discovery:
            if year_to_league_id:
                log(f"\nCache is outdated (max year: {cached_max_year}, current: {current_year})")
                log(f"Discovering missing years for {ctx.league_name}...")
                # Only discover missing years (start after the max cached year)
                years = list(range(cached_max_year + 1, end_year + 1))
            else:
                log(f"\nDiscovering league IDs for {ctx.league_name}...")
                # Discover all years from start
                years = list(range(start_year, end_year + 1))

            # Initialize discovery
            discovery = LeagueDiscovery(oauth_file=oauth_file, game_code=ctx.game_code)

            for year in years:
                log(f"  [{year}] Discovering leagues...")
                try:
                    leagues = discovery.discover_leagues(year=year)

                    # Find the league matching our league name
                    for league in leagues:
                        if ctx.league_name.lower() in league.get('league_name', '').lower():
                            league_id = league.get('league_id')
                            year_to_league_id[year] = league_id
                            log(f"    Found: {league_id} ('{league['league_name']}')")
                            break

                    if year not in year_to_league_id:
                        log(f"    No league found matching '{ctx.league_name}'")

                except Exception as e:
                    log(f"    Error discovering {year}: {e}")

            if not year_to_league_id:
                log(f"No leagues found for '{ctx.league_name}'")
                sys.exit(1)

            log(f"\nDiscovered {len(year_to_league_id)} total years for {ctx.league_name}:")
            for year in sorted(year_to_league_id.keys()):
                log(f"  {year}: {year_to_league_id[year]}")

            # Save updated cache (merge with existing data for this league)
            try:
                # Load existing cache to preserve other leagues' data
                existing_cache = []
                if discovered_file.exists():
                    try:
                        with open(discovered_file, 'r') as f:
                            existing_cache = json.load(f)
                    except Exception:
                        pass

                # Remove old entries for this league
                other_leagues_data = [
                    entry for entry in existing_cache
                    if entry.get('league_name') != ctx.league_name
                ]

                # Add current league's data
                current_league_data = [
                    {
                        'year': year,
                        'league_id': league_id,
                        'league_name': ctx.league_name
                    }
                    for year, league_id in year_to_league_id.items()
                ]

                # Merge
                merged_cache = other_leagues_data + current_league_data

                with open(discovered_file, 'w') as f:
                    json.dump(merged_cache, f, indent=2)
                log(f"\nUpdated league discovery cache in {discovered_file.name}")
            except Exception as e:
                log(f"[WARN] Could not save discovery cache: {e}")
        else:
            log(f"\n[CACHE HIT] Using cached league IDs ({len(year_to_league_id)} years)")

        # Determine which years to fetch
        if args.year > 0:
            # Specific year requested
            if args.year not in year_to_league_id:
                log(f"Year {args.year} not found in discovered leagues")
                sys.exit(1)
            years_to_fetch = [args.year]
        else:
            # Fetch all discovered years
            years_to_fetch = sorted(year_to_league_id.keys())
            log(f"\nFetching all {len(years_to_fetch)} years")

        # Load all league settings files upfront (for end_week lookup)
        league_dir = Path(ctx.data_directory).parent if ctx.data_directory else Path.cwd()
        year_to_settings_file = find_league_settings_files(league_dir)
        log(f"\nFound league settings for {len(year_to_settings_file)} years")

        # Fetch data for each year
        for i, year in enumerate(years_to_fetch, 1):
            # Ensure year is an integer (JSON may load as string)
            year = int(year)

            league_id = year_to_league_id[year] if year in year_to_league_id else year_to_league_id[str(year)]

            log(f"\n{'='*70}")
            log(f"Processing year {i}/{len(years_to_fetch)}: {year}")
            log(f"{'='*70}")
            log(f"League ID: {league_id}")

            # Load league settings for this year to get actual end_week
            settings = {}
            if year in year_to_settings_file:
                settings = load_league_settings(year_to_settings_file[year])
                log(f"Loaded settings from: {year_to_settings_file[year].name}")

            # Determine weeks based on league settings (NOT hardcoded by year)
            # Priority: 1) settings.end_week, 2) Yahoo API for current year, 3) fallback to NFL standard
            settings_end_week = settings.get('end_week')

            if settings_end_week:
                # Use end_week from league settings (most accurate)
                default_weeks = settings_end_week
                log(f"Using end_week from settings: {default_weeks} weeks")
            else:
                # Fallback to NFL standard if settings not available
                if year >= 2021:
                    default_weeks = 17  # 17 regular season weeks starting 2021
                else:
                    default_weeks = 16  # 16 regular season weeks before 2021
                log(f"[WARN] No settings found for {year}, using NFL standard: {default_weeks} weeks")

            # Initialize fetcher for this year
            fetcher = YahooRosterFetcher(
                oauth_file=oauth_file,
                league_id=league_id,
                rate_limit=rate_limit,
                output_dir=output_dir
            )

            # For current year, use max week from matchup data
            # This ensures player data aligns with matchup data (which already filters to completed weeks)
            current_year = datetime.now().year
            if year == current_year:
                log(f"[CURRENT YEAR] Checking matchup data for completed weeks...")

                try:
                    # Get max week from matchup files (which were already fetched by weekly_matchup_data_v2.py)
                    max_week_from_matchups = get_max_week_from_matchup_data(Path(ctx.data_directory), year)

                    if max_week_from_matchups:
                        default_weeks = max_week_from_matchups
                        log(f"[CURRENT YEAR] Using max week from matchup data: {default_weeks}")
                        log(f"[CURRENT YEAR] Only fetching weeks 1-{default_weeks}")
                    else:
                        log(f"[WARN] No matchup data found for {year}")
                        log(f"[WARN] Using league end_week ({default_weeks}) - may include incomplete weeks!")
                        # Continue with default_weeks as fallback

                except Exception as e:
                    log(f"[WARN] Could not get max week from matchup data: {e}")
                    log(f"[WARN] Using league end_week ({default_weeks}) - may include incomplete weeks!")
                    # Continue with default_weeks as fallback

            # Validate weeks
            if default_weeks <= 0:
                log(f"[ERROR] Invalid week count ({default_weeks}) for {year}, skipping")
                continue

            log(f"Fetching weeks for {year}: 1-{default_weeks}")

            try:

                # Determine weeks to fetch
                if args.week > 0:
                    weeks = [args.week]
                else:
                    weeks = list(range(1, default_weeks + 1))

                # Fetch roster data
                df = fetcher.fetch_season_rosters(
                    year=year,
                    weeks=weeks,
                    end_week=None
                )

                if not df.empty:
                    # Apply manager name overrides from context
                    if ctx.manager_name_overrides and 'manager_name' in df.columns:
                        log(f"Applying {len(ctx.manager_name_overrides)} manager name overrides")
                        df['manager_name'] = df['manager_name'].replace(ctx.manager_name_overrides)

                    # Validate league format (detect unsupported formats)
                    if 'fantasy_position' in df.columns:
                        unique_positions = df['fantasy_position'].dropna().unique()

                        # Standard positions
                        standard_positions = {'QB', 'RB', 'WR', 'TE', 'K', 'DEF', 'BN', 'IR', 'W/R/T', 'OP'}

                        # Detect special league formats
                        unsupported_positions = set(unique_positions) - standard_positions

                        if unsupported_positions:
                            # Check for specific league types
                            if any(pos in ['DL', 'LB', 'DB', 'DP'] for pos in unsupported_positions):
                                log(f"  [WARN] IDP (Individual Defensive Player) league detected!")
                                log(f"  Positions: {sorted(unsupported_positions)}")
                                log(f"  IDP leagues may not be fully supported by downstream transformations")
                            elif 'Q/W/R/T' in unsupported_positions or 'W/R/T/Q' in unsupported_positions:
                                log(f"  [INFO] Superflex league detected (QB in FLEX)")
                                log(f"  Position: {[p for p in unsupported_positions if 'Q' in p]}")
                            else:
                                log(f"  [WARN] Non-standard roster positions detected: {sorted(unsupported_positions)}")
                                log(f"  This league may have custom scoring or roster settings")
                                log(f"  Verify downstream transformations work correctly")

                    fetcher.save_to_parquet(df, year)

                    # Show summary of points
                    if 'fantasy_points' in df.columns:
                        total_points = df['fantasy_points'].sum()
                        avg_points = df['fantasy_points'].mean()
                        non_null_count = df['fantasy_points'].notna().sum()
                        log(f"Fantasy points stats for {year}:")
                        log(f"  Non-null values: {non_null_count}/{len(df)}")
                        log(f"  Total fantasy points: {total_points:.2f}")
                        log(f"  Average points per player-week: {avg_points:.2f}")

                    log(f"[OK] Successfully fetched {len(df)} roster records for {year}")
                else:
                    log(f"[WARN] No data fetched for {year}")

            except Exception as e:
                log(f"[FAIL] Error processing year {year}: {e}")
                import traceback
                traceback.print_exc()

                # Continue to next year instead of stopping
                log(f"Continuing to next year...")
                continue

        log(f"\n{'='*70}")
        log(f"[OK] COMPLETED!")
        log(f"Output directory: {output_dir}")
        log(f"Files saved in both CSV and Parquet formats")
        log(f"{'='*70}")

    finally:
        # Clean up temporary OAuth file if created
        if temp_oauth_file and temp_oauth_file.exists():
            try:
                import os
                os.unlink(temp_oauth_file)
                log("Cleaned up temporary OAuth file")
            except:
                pass


if __name__ == "__main__":
    main()
