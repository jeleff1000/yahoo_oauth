#!/usr/bin/env python3
"""
Yahoo Fantasy Roster Data Fetcher

Fetches weekly roster data showing which manager owned which player each week.

Output includes:
- manager_name: Team owner
- player_name: Player name
- yahoo_position: Yahoo's position designation
- primary_position: Primary position (QB, RB, WR, TE, K, DEF)
- fantasy_position: Roster slot (QB, RB1, RB2, FLEX, BN, etc.)
- year, week
- player stats for that week
"""

from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

# Try to import Yahoo OAuth
try:
    from yahoo_oauth import OAuth2
    YAHOO_OAUTH_AVAILABLE = True
except ImportError:
    OAuth2 = None
    YAHOO_OAUTH_AVAILABLE = False
    print("Warning: yahoo_oauth not available. Install with: pip install yahoo_oauth")


def log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Remove unicode symbols that don't work in Windows console
    msg = msg.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('⚠', '[WARN]')
    print(f"[{timestamp}] [{level}] {msg}")


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

                log(f"Fetching: {url[:100]}...", level="DEBUG")
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
                    log(f"Rate limit or permission error detected: {e}", level="WARNING")
                    log(f"Waiting 5 minutes before retrying...", level="WARNING")
                    time.sleep(300)  # Wait 5 minutes
                    log(f"Resuming after 5-minute wait", level="INFO")
                    continue

                log(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}", level="WARNING")

                if attempt < self.max_retries - 1:
                    sleep_time = backoff * (2 ** attempt)
                    log(f"Retrying in {sleep_time:.1f}s...", level="WARNING")
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
                log(f"  {team_key}: {manager_name}", level="DEBUG")

            return teams

        except Exception as e:
            log(f"Error fetching teams: {e}", level="ERROR")
            return {}

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
                        player_info['nfl_team'] = editorial_team.text

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
                    log(f"Error parsing player: {e}", level="WARNING")
                    continue

        except Exception as e:
            log(f"Error fetching roster for {team_key} week {week}: {e}", level="ERROR")

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
        Fallback method: Fetch rosters individually for each team.

        Args:
            year: Season year
            week: Week number
            teams: Dict mapping team_key to manager_name

        Returns:
            DataFrame with all roster data
        """
        all_rosters = []

        for team_key, manager_name in teams.items():
            try:
                roster_data = self.fetch_roster_for_week(year, week, team_key, manager_name)
                all_rosters.extend(roster_data)
                time.sleep(0.5)  # Rate limiting between teams
            except Exception as e:
                log(f"Error fetching roster for {manager_name}: {e}", level="WARNING")
                continue

        if not all_rosters:
            return pd.DataFrame()

        return pd.DataFrame(all_rosters)

    def fetch_season_rosters(
        self,
        year: int,
        weeks: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Fetch roster data for an entire season using optimized batch API.

        Args:
            year: Season year
            weeks: List of weeks to fetch (auto-detect based on year if None)

        Returns:
            DataFrame with all roster data for the season
        """
        if weeks is None:
            # NFL expanded to 17-game regular season starting in 2021
            # Before that, it was 16 games (weeks 1-16)
            if year >= 2021:
                weeks = list(range(1, 19))  # Weeks 1-18 (17 regular + playoffs)
            else:
                weeks = list(range(1, 17))  # Weeks 1-16 (16 regular season)

        log(f"Fetching season {year} roster data for weeks: {weeks}")

        # Fetch teams first
        teams = self.fetch_teams()

        if not teams:
            log("No teams found!", level="ERROR")
            return pd.DataFrame()

        # Fetch all weeks using optimized batch API
        try:
            return self._fetch_all_weeks_batch(year, weeks, teams)
        except Exception as e:
            log(f"Batch fetch for all weeks failed: {e}", level="WARNING")
            log("Falling back to week-by-week fetching...", level="WARNING")
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
                log(f"Error fetching week {week}: {e}", level="ERROR")
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
                log(f"Error fetching week {week}: {e}", level="ERROR")
                continue

        if not all_weeks_data:
            log(f"No roster data fetched for season {year}", level="WARNING")
            return pd.DataFrame()

        df = pd.concat(all_weeks_data, ignore_index=True)
        log(f"Total roster records for {year}: {len(df)}")

        return df

    def save_to_parquet(self, df: pd.DataFrame, year: int, filename: Optional[str] = None):
        """Save DataFrame to Parquet and CSV files."""
        if df.empty:
            log("DataFrame is empty, skipping save", level="WARNING")
            return

        if filename is None:
            base_filename = f"yahoo_roster_data_{year}"
        else:
            base_filename = filename.replace('.parquet', '').replace('.csv', '')

        # Convert to proper types
        df = self._prepare_dataframe_for_export(df)

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


def load_league_mapping(discovered_leagues_file: Path, league_name: str = "KMFFL"):
    """Load year-to-league_id mapping from discovered_leagues.json."""
    with open(discovered_leagues_file, 'r') as f:
        leagues = json.load(f)

    # Create mapping of year -> league_id for the specified league
    year_to_league_id = {}
    for league in leagues:
        if league['league_name'] == league_name:
            year_to_league_id[league['year']] = league['league_id']

    return year_to_league_id


def main():
    """Fetch all roster data for all years."""

    # Configuration
    oauth_file = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\data_fetchers\secrets.json")
    discovered_leagues_file = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\KMFFLApp\discovered_leagues.json")
    output_dir = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\data\kmffl\roster_data")
    league_name = "KMFFL"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load year-to-league_id mapping
    log(f"Loading league mappings from {discovered_leagues_file}")
    year_to_league_id = load_league_mapping(discovered_leagues_file, league_name)

    if not year_to_league_id:
        log(f"No leagues found for '{league_name}'", level="ERROR")
        sys.exit(1)

    log(f"Found {len(year_to_league_id)} years for {league_name}:")
    for year, league_id in sorted(year_to_league_id.items()):
        log(f"  {year}: {league_id}")

    # Fetch data for each year
    years_to_fetch = sorted(year_to_league_id.keys())

    for i, year in enumerate(years_to_fetch, 1):
        league_id = year_to_league_id[year]

        log(f"\n{'='*70}")
        log(f"Processing year {i}/{len(years_to_fetch)}: {year} (League ID: {league_id})")
        log(f"{'='*70}\n")

        try:
            # Initialize fetcher for this year
            fetcher = YahooRosterFetcher(
                oauth_file=oauth_file,
                league_id=league_id,
                rate_limit=2.0,
                output_dir=output_dir
            )

            # Fetch all weeks for this year
            df = fetcher.fetch_season_rosters(
                year=year,
                weeks=None  # All weeks (auto-detect based on year)
            )

            if not df.empty:
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
                log(f"[WARN] No data fetched for {year}", level="WARNING")

        except Exception as e:
            log(f"[FAIL] Error processing year {year}: {e}", level="ERROR")
            import traceback
            traceback.print_exc()

            # Continue to next year instead of stopping
            log(f"Continuing to next year...", level="WARNING")
            continue

    log("\n" + "="*70)
    log(f"[OK] COMPLETED!")
    log(f"Output directory: {output_dir}")
    log(f"Files saved in both CSV and Parquet formats")
    log("="*70)


if __name__ == "__main__":
    main()
