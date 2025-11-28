#!/usr/bin/env python3
"""
Fetch team defensive stats from NFLverse for fantasy football DST scoring.

This script downloads team-level defensive statistics from NFLverse and transforms
them into a format suitable for merging with Yahoo fantasy DST data.

Source: https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{year}.parquet
"""
import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import requests

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Try to import LeagueContext for multi-league support
try:
    from core.league_context import LeagueContext
    LEAGUE_CONTEXT_AVAILABLE = True
except ImportError:
    try:
        from multi_league.core.league_context import LeagueContext
        LEAGUE_CONTEXT_AVAILABLE = True
    except ImportError:
        LEAGUE_CONTEXT_AVAILABLE = False
        LeagueContext = None

# Default output directory
DEFAULT_OUTPUT_DIR = REPO_ROOT / "fantasy_football_data" / "player_data"

# Default cache directory
DEFAULT_CACHE_DIR = REPO_ROOT / "fantasy_football_data" / "cache" / "nflverse"

# Cache max age in hours (1 week)
CACHE_MAX_AGE_HOURS = 168


def retry_with_backoff(func, max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    """
    Retry a function with exponential backoff for transient failures.

    Args:
        func: Function to retry (should take no arguments)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry

    Returns:
        Result of successful function call

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
                error_str = str(e).lower()

                # Check if error is retryable
                is_retryable = (
                    'rate limit' in error_str or
                    '429' in error_str or
                    ('5' in error_str[:3] if len(error_str) >= 3 else False) or
                    'timeout' in error_str or
                    'connection' in error_str
                )

                if is_retryable:
                    print(f"  [RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f"  Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    delay *= backoff_factor
                    continue

            # Non-retryable error or final attempt, re-raise
            raise

    # Should never reach here, but just in case
    raise last_exception


def fetch_nflverse_team_stats(year: int, cache_dir: Path = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch team defensive stats from NFLverse for a given year.

    IMPORTANT: For the CURRENT year, cache expires more frequently (24 hours) to ensure
    we get updated data as games complete each week. For past years, cache lasts 7 days.

    Args:
        year: NFL season year (e.g., 2014)
        cache_dir: Directory to store cached downloads
        use_cache: Whether to use cached data if available

    Returns:
        DataFrame with team defensive stats
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"nflverse_team_stats_{year}.parquet"

    # Check cache first
    if use_cache and cache_file.exists():
        from datetime import datetime
        current_year = datetime.now().year

        # For current year: use shorter cache expiry (24 hours) to get fresh data as games complete
        # For past years: use longer cache expiry (168 hours = 7 days)
        max_cache_age = 24 if year == current_year else CACHE_MAX_AGE_HOURS

        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < max_cache_age:
            print(f"[DEF] Using cached team stats for {year} (age: {age_hours:.1f} hours, max: {max_cache_age}h)")
            return pd.read_parquet(cache_file)
        else:
            print(f"[DEF] Cache expired for {year} (age: {age_hours:.1f}h > max: {max_cache_age}h), re-downloading")

    # Download from NFLverse
    url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{year}.parquet"

    print(f"[DEF] Downloading {year} team stats from NFLverse...")
    print(f"[DEF] URL: {url}")

    import tempfile

    def download():
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"NFLverse team stats not available for {year} (404 Not Found). Data may not exist for this year.")
            else:
                raise

        # Use chunked download to avoid memory spike
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    tmp.write(chunk)
            tmp_path = tmp.name

        try:
            df = pd.read_parquet(tmp_path)

            # Validate downloaded data
            if df.empty:
                raise ValueError(f"Downloaded team stats for {year} is empty")
            if 'team' not in df.columns or 'opponent_team' not in df.columns:
                raise ValueError(f"Downloaded data missing required columns")

            # Save to cache
            df.to_parquet(cache_file)
            print(f"[DEF] Downloaded {len(df):,} rows of team stats for {year}")
            print(f"[DEF] Cached to: {cache_file}")

            return df
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    # Retry download with exponential backoff
    return retry_with_backoff(download, max_retries=3, initial_delay=1.0)


def fetch_nflverse_pbp_data(year: int, cache_dir: Path = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch play-by-play data from NFLverse for calculating three_out and fourth_down_stop.

    IMPORTANT: For the CURRENT year, cache expires more frequently (24 hours) to ensure
    we get updated data as games complete each week. For past years, cache lasts 7 days.

    Args:
        year: NFL season year (e.g., 2014)
        cache_dir: Directory to store cached downloads
        use_cache: Whether to use cached data if available

    Returns:
        DataFrame with play-by-play data
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"nflverse_pbp_{year}.parquet"

    # Check cache first
    if use_cache and cache_file.exists():
        from datetime import datetime
        current_year = datetime.now().year

        # For current year: use shorter cache expiry (24 hours) to get fresh data as games complete
        # For past years: use longer cache expiry (168 hours = 7 days)
        max_cache_age = 24 if year == current_year else CACHE_MAX_AGE_HOURS

        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < max_cache_age:
            print(f"[DEF] Using cached play-by-play data for {year} (age: {age_hours:.1f} hours, max: {max_cache_age}h)")
            return pd.read_parquet(cache_file)
        else:
            print(f"[DEF] Cache expired for {year} (age: {age_hours:.1f}h > max: {max_cache_age}h), re-downloading")

    # Download from NFLverse
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"

    print(f"[DEF] Downloading {year} play-by-play data from NFLverse...")
    print(f"[DEF] URL: {url}")

    import tempfile

    def download():
        response = requests.get(url, stream=True, timeout=120)  # Longer timeout for large file
        response.raise_for_status()

        # Use chunked download to avoid memory spike (files can be 40-200 MB)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
            total_bytes = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    tmp.write(chunk)
                    total_bytes += len(chunk)

                    # Print progress every 10 MB
                    if total_bytes % (10 * 1024 * 1024) < 8192:
                        print(f"[DEF] Downloaded {total_bytes / (1024 * 1024):.1f} MB...")

            tmp_path = tmp.name

        try:
            print(f"[DEF] Total downloaded: {total_bytes / (1024 * 1024):.1f} MB")
            print(f"[DEF] Reading parquet file...")
            df = pd.read_parquet(tmp_path)

            # Validate downloaded data
            if df.empty:
                raise ValueError(f"Downloaded play-by-play data for {year} is empty")
            if 'play_type' not in df.columns or 'defteam' not in df.columns:
                raise ValueError(f"Downloaded data missing required columns")

            # Save to cache
            print(f"[DEF] Caching play-by-play data...")
            df.to_parquet(cache_file)
            print(f"[DEF] Downloaded {len(df):,} rows of play-by-play data for {year}")
            print(f"[DEF] Cached to: {cache_file}")

            return df
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    # Retry download with exponential backoff
    return retry_with_backoff(download, max_retries=3, initial_delay=2.0)


def calculate_three_outs(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate three-and-out stats from play-by-play data.

    A three-and-out occurs when:
    - drive_first_downs == 0 AND play_type_nfl == 'PUNT'

    Args:
        pbp_df: Play-by-play DataFrame from NFLverse

    Returns:
        DataFrame with three_out counts by defteam, week, season
    """
    print(f"[DEF] Calculating three-and-outs from {len(pbp_df):,} plays...")

    # Filter to plays that meet three-and-out criteria
    three_out_plays = pbp_df[
        (pbp_df['drive_first_downs'] == 0) &
        (pbp_df['play_type_nfl'] == 'PUNT')
    ].copy()

    print(f"[DEF] Found {len(three_out_plays):,} three-and-out plays")

    # Group by defensive team, week, and season
    three_out_stats = three_out_plays.groupby(
        ['defteam', 'week', 'season'],
        dropna=False
    ).size().reset_index(name='three_out')

    print(f"[DEF] Calculated three-and-outs for {len(three_out_stats):,} team-week combinations")

    return three_out_stats


def calculate_fourth_down_stops(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fourth down stop stats from play-by-play data.

    A fourth down stop occurs when:
    - down == 4 AND play_type in ['pass', 'run'] AND first_down == 0

    Args:
        pbp_df: Play-by-play DataFrame from NFLverse

    Returns:
        DataFrame with fourth_down_stop counts by defteam, week, season
    """
    print(f"[DEF] Calculating fourth down stops from {len(pbp_df):,} plays...")

    # Filter to plays that meet fourth down stop criteria
    fourth_down_stop_plays = pbp_df[
        (pbp_df['down'] == 4) &
        (pbp_df['play_type'].isin(['pass', 'run'])) &
        (pbp_df['first_down'] == 0)
    ].copy()

    print(f"[DEF] Found {len(fourth_down_stop_plays):,} fourth down stop plays")

    # Group by defensive team, week, and season
    fourth_down_stop_stats = fourth_down_stop_plays.groupby(
        ['defteam', 'week', 'season'],
        dropna=False
    ).size().reset_index(name='fourth_down_stop')

    print(f"[DEF] Calculated fourth down stops for {len(fourth_down_stop_stats):,} team-week combinations")

    return fourth_down_stop_stats


def transform_to_defensive_stats(df: pd.DataFrame, three_out_stats: pd.DataFrame = None,
                                 fourth_down_stop_stats: pd.DataFrame = None) -> pd.DataFrame:
    """
    Transform team stats to defensive-oriented format for DST fantasy scoring.

    CRITICAL TRANSFORMATION:
    - Team A's defensive stats (sacks, INTs) = what Team A's defense did
    - Team A's points/yards ALLOWED = what Team B's offense did against Team A

    This requires a SELF-JOIN to match each team's defense with opponent's offense.

    Args:
        df: Raw team stats from NFLverse
        three_out_stats: Optional DataFrame with three_out counts
        fourth_down_stop_stats: Optional DataFrame with fourth_down_stop counts

    Returns:
        DataFrame transformed for defensive fantasy scoring
    """
    # Map team abbreviations to full names (for Yahoo compatibility)
    # Yahoo uses full team names including city + mascot for LA and NY teams
    # Standardized abbreviations: LAR (Rams), LAC (Chargers), LV (Raiders)
    # Historical team relocations map to CURRENT team names
    TEAM_NAMES = {
        'ARI': 'Arizona', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BUF': 'Buffalo',
        'CAR': 'Carolina', 'CHI': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland',
        'DAL': 'Dallas', 'DEN': 'Denver', 'DET': 'Detroit', 'GB': 'Green Bay',
        'HOU': 'Houston', 'IND': 'Indianapolis', 'JAX': 'Jacksonville', 'JAC': 'Jacksonville',
        'KC': 'Kansas City', 'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams',
        'LV': 'Las Vegas', 'MIA': 'Miami',
        'MIN': 'Minnesota', 'NE': 'New England', 'NO': 'New Orleans',
        'NYG': 'New York Giants', 'NYJ': 'New York Jets',
        'PHI': 'Philadelphia', 'PIT': 'Pittsburgh', 'SEA': 'Seattle',
        'SF': 'San Francisco', 'TB': 'Tampa Bay', 'TEN': 'Tennessee',
        'WAS': 'Washington', 'WSH': 'Washington',
        # Historical team relocations → Map to current names
        'STL': 'Los Angeles Rams',    # St. Louis Rams (1995-2015) → LAR (2016+)
        'SD': 'Los Angeles Chargers',  # San Diego Chargers (until 2016) → LAC (2017+)
        'OAK': 'Las Vegas'             # Oakland Raiders (until 2019) → LV (2020+)
    }

    # Map team abbreviations to logo URLs (to match offense headshot_url column)
    # Uses Wikipedia logos for consistency and availability
    # Historical teams map to CURRENT logo (maintains data consistency)
    TEAM_LOGO_MAP = {
        "ARI": "https://upload.wikimedia.org/wikipedia/en/thumb/7/72/Arizona_Cardinals_logo.svg/179px-Arizona_Cardinals_logo.svg.png",
        "ATL": "https://upload.wikimedia.org/wikipedia/en/thumb/c/c5/Atlanta_Falcons_logo.svg/192px-Atlanta_Falcons_logo.svg.png",
        "BAL": "https://upload.wikimedia.org/wikipedia/en/thumb/1/16/Baltimore_Ravens_logo.svg/193px-Baltimore_Ravens_logo.svg.png",
        "BUF": "https://upload.wikimedia.org/wikipedia/en/thumb/7/77/Buffalo_Bills_logo.svg/189px-Buffalo_Bills_logo.svg.png",
        "CAR": "https://upload.wikimedia.org/wikipedia/en/thumb/1/1c/Carolina_Panthers_logo.svg/100px-Carolina_Panthers_logo.svg.png",
        "CHI": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Chicago_Bears_logo.svg/100px-Chicago_Bears_logo.svg.png",
        "CIN": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Cincinnati_Bengals_logo.svg/100px-Cincinnati_Bengals_logo.svg.png",
        "CLE": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d9/Cleveland_Browns_logo.svg/100px-Cleveland_Browns_logo.svg.png",
        "DAL": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Dallas_Cowboys.svg/100px-Dallas_Cowboys.svg.png",
        "DEN": "https://upload.wikimedia.org/wikipedia/en/thumb/4/44/Denver_Broncos_logo.svg/100px-Denver_Broncos_logo.svg.png",
        "DET": "https://upload.wikimedia.org/wikipedia/en/thumb/7/71/Detroit_Lions_logo.svg/100px-Detroit_Lions_logo.svg.png",
        "GB":  "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Green_Bay_Packers_logo.svg/100px-Green_Bay_Packers_logo.svg.png",
        "HOU": "https://upload.wikimedia.org/wikipedia/en/thumb/2/28/Houston_Texans_logo.svg/100px-Houston_Texans_logo.svg.png",
        "IND": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Indianapolis_Colts_logo.svg/100px-Indianapolis_Colts_logo.svg.png",
        "JAX": "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/Jacksonville_Jaguars_logo.svg/100px-Jacksonville_Jaguars_logo.svg.png",
        "JAC": "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/Jacksonville_Jaguars_logo.svg/100px-Jacksonville_Jaguars_logo.svg.png",  # Alternate abbreviation
        "KC":  "https://upload.wikimedia.org/wikipedia/en/thumb/e/e1/Kansas_City_Chiefs_logo.svg/100px-Kansas_City_Chiefs_logo.svg.png",
        "LAC": "https://upload.wikimedia.org/wikipedia/en/thumb/7/72/NFL_Chargers_logo.svg/100px-NFL_Chargers_logo.svg.png",
        "LAR": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8a/Los_Angeles_Rams_logo.svg/100px-Los_Angeles_Rams_logo.svg.png",
        "MIA": "https://upload.wikimedia.org/wikipedia/en/thumb/3/37/Miami_Dolphins_logo.svg/100px-Miami_Dolphins_logo.svg.png",
        "MIN": "https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Minnesota_Vikings_logo.svg/98px-Minnesota_Vikings_logo.svg.png",
        "NE":  "https://upload.wikimedia.org/wikipedia/en/thumb/b/b9/New_England_Patriots_logo.svg/100px-New_England_Patriots_logo.svg.png",
        "NO":  "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/New_Orleans_Saints_logo.svg/98px-New_Orleans_Saints_logo.svg.png",
        "NYG": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/New_York_Giants_logo.svg/100px-New_York_Giants_logo.svg.png",
        "NYJ": "https://upload.wikimedia.org/wikipedia/en/thumb/6/6b/New_York_Jets_logo.svg/100px-New_York_Jets_logo.svg.png",
        "LV":  "https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Las_Vegas_Raiders_logo.svg/150px-Las_Vegas_Raiders_logo.svg.png",
        "PHI": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/Philadelphia_Eagles_logo.svg/100px-Philadelphia_Eagles_logo.svg.png",
        "PIT": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Pittsburgh_Steelers_logo.svg/100px-Pittsburgh_Steelers_logo.svg.png",
        "SF":  "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/San_Francisco_49ers_logo.svg/100px-San_Francisco_49ers_logo.svg.png",
        "SEA": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/Seattle_Seahawks_logo.svg/100px-Seattle_Seahawks_logo.svg.png",
        "TB":  "https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/Tampa_Bay_Buccaneers_logo.svg/100px-Tampa_Bay_Buccaneers_logo.svg.png",
        "TEN": "https://upload.wikimedia.org/wikipedia/en/thumb/c/c1/Tennessee_Titans_logo.svg/100px-Tennessee_Titans_logo.svg.png",
        "WAS": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Washington_football_team_wlogo.svg/1024px-Washington_football_team_wlogo.svg.png",
        "WSH": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Washington_football_team_wlogo.svg/1024px-Washington_football_team_wlogo.svg.png",  # Alternate abbreviation
        # Historical team relocations → Map to current logo
        'STL': 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8a/Los_Angeles_Rams_logo.svg/100px-Los_Angeles_Rams_logo.svg.png',    # St. Louis → LAR
        'SD': 'https://upload.wikimedia.org/wikipedia/en/thumb/7/72/NFL_Chargers_logo.svg/100px-NFL_Chargers_logo.svg.png',  # San Diego → LAC
        'OAK': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Las_Vegas_Raiders_logo.svg/150px-Las_Vegas_Raiders_logo.svg.png'   # Oakland → LV
    }

    print(f"[DEF] Transforming {len(df):,} rows to defensive format...")

    # Step 1: Create offense dataset (what each team's offense did)
    offense = df.copy()
    offense = offense.rename(columns={
        'team': 'offense_team',
        'opponent_team': 'defense_team',
        'season': 'year'
    })

    # Step 2: Create defense dataset (what each team's defense did)
    defense = df.copy()
    defense = defense.rename(columns={
        'team': 'defense_team',
        'opponent_team': 'offense_team',
        'season': 'year'
    })

    # Step 3: Self-join to match each team's defense with opponent's offense
    # Join on: year, week, defense_team=offense.offense_team, offense_team=defense.defense_team
    merged = defense.merge(
        offense[['year', 'week', 'season_type', 'offense_team', 'defense_team',
                 'passing_yards', 'rushing_yards', 'passing_tds', 'rushing_tds',
                 'receiving_tds', 'fg_made', 'pat_made', 'passing_2pt_conversions',
                 'rushing_2pt_conversions', 'receiving_2pt_conversions']],
        left_on=['year', 'week', 'season_type', 'defense_team', 'offense_team'],
        right_on=['year', 'week', 'season_type', 'defense_team', 'offense_team'],
        how='left',
        suffixes=('', '_allowed')
    )

    # Validate self-join merge
    missing_opponent_data = merged['passing_yards_allowed'].isna()
    if missing_opponent_data.any():
        print(f"[DEF] Warning: {missing_opponent_data.sum()} rows missing opponent offensive data")
        affected_teams = merged[missing_opponent_data]['defense_team'].unique()
        print(f"[DEF] Affected teams: {', '.join(affected_teams)}")

    # Step 4: Build defensive DataFrame with proper schema
    defensive_df = pd.DataFrame()

    # Normalize team abbreviations (NFLverse uses "LA" which is ambiguous)
    # Standardize to: LAR (Rams), LAC (Chargers), LV (Raiders)
    # Also handle historical team relocations for consistency
    team_abbreviation_fixes = {
        'LA': 'LAR',   # Disambiguate Los Angeles (Rams vs Chargers)
        'STL': 'LAR',  # St. Louis Rams → Los Angeles Rams (2016+)
        'SD': 'LAC',   # San Diego Chargers → Los Angeles Chargers (2017+)
        'OAK': 'LV',   # Oakland Raiders → Las Vegas Raiders (2020+)
    }

    normalized_defense_team = merged['defense_team'].replace(team_abbreviation_fixes)
    normalized_offense_team = merged['offense_team'].replace(team_abbreviation_fixes)

    # Basic identifiers
    defensive_df['nfl_team'] = normalized_defense_team
    defensive_df['opponent_nfl_team'] = normalized_offense_team
    defensive_df['year'] = merged['year']
    defensive_df['week'] = merged['week']
    defensive_df['season_type'] = merged['season_type']

    # Position
    defensive_df['nfl_position'] = 'DEF'
    defensive_df['fantasy_position'] = 'DEF'

    # Map team abbreviations to full names for Yahoo compatibility
    # Yahoo uses "Arizona", "Atlanta", etc. instead of "ARI Defense", "ATL Defense"
    defensive_df['player'] = normalized_defense_team.map(TEAM_NAMES).fillna(normalized_defense_team)

    # Map team abbreviations to logo URLs (to match offense headshot_url column)
    defensive_df['headshot_url'] = normalized_defense_team.map(TEAM_LOGO_MAP)

    # Check for unmapped teams (filter out None/NaN values)
    unmapped_names = normalized_defense_team[~normalized_defense_team.isin(TEAM_NAMES)].unique()
    unmapped_names = [str(x) for x in unmapped_names if pd.notna(x)]

    unmapped_logos = normalized_defense_team[~normalized_defense_team.isin(TEAM_LOGO_MAP)].unique()
    unmapped_logos = [str(x) for x in unmapped_logos if pd.notna(x)]

    if len(unmapped_names) > 0:
        print(f"[DEF] Warning: Unmapped team names: {', '.join(unmapped_names)}")
    if len(unmapped_logos) > 0:
        print(f"[DEF] Warning: Unmapped team logos: {', '.join(unmapped_logos)}")

    print(f"[DEF] Mapped team abbreviations to full names for {defensive_df['player'].notna().sum()} rows")
    print(f"[DEF] Mapped team logos (headshot_url) for {defensive_df['headshot_url'].notna().sum()} rows")

    # Defensive stats (from defense row - what this team's defense DID)
    defensive_df['def_sacks'] = merged['def_sacks']
    defensive_df['def_sack_yards'] = merged['def_sack_yards']
    defensive_df['def_qb_hits'] = merged['def_qb_hits']
    defensive_df['def_interceptions'] = merged['def_interceptions']
    defensive_df['def_interception_yards'] = merged['def_interception_yards']
    defensive_df['def_pass_defended'] = merged['def_pass_defended']
    defensive_df['def_tackles_solo'] = merged['def_tackles_solo']
    defensive_df['def_tackles_with_assist'] = merged['def_tackles_with_assist']
    defensive_df['def_tackle_assists'] = merged['def_tackle_assists']
    defensive_df['def_tackles_for_loss'] = merged['def_tackles_for_loss']
    defensive_df['def_tackles_for_loss_yards'] = merged['def_tackles_for_loss_yards']
    defensive_df['def_fumbles_forced'] = merged['def_fumbles_forced']
    defensive_df['def_tds'] = merged['def_tds']
    defensive_df['def_fumbles'] = merged['def_fumbles']
    defensive_df['def_safeties'] = merged['def_safeties']
    defensive_df['special_teams_tds'] = merged['special_teams_tds']
    defensive_df['fum_rec'] = merged['fumble_recovery_opp']
    defensive_df['fum_ret_td'] = merged['fumble_recovery_tds']

    # Points/Yards ALLOWED (from opponent's offense - what opponent DID against this defense)
    defensive_df['passing_yds_allowed'] = merged['passing_yards_allowed']
    defensive_df['rushing_yds_allowed'] = merged['rushing_yards_allowed']
    defensive_df['total_yds_allowed'] = (
        merged['passing_yards_allowed'].fillna(0) +
        merged['rushing_yards_allowed'].fillna(0)
    )

    # Calculate total points allowed
    defensive_df['passing_tds_allowed'] = merged['passing_tds_allowed']
    defensive_df['rushing_tds_allowed'] = merged['rushing_tds_allowed']
    defensive_df['receiving_tds_allowed'] = merged['receiving_tds_allowed']

    # Total touchdowns allowed
    total_tds_allowed = (
        merged['passing_tds_allowed'].fillna(0) +
        merged['rushing_tds_allowed'].fillna(0) +
        merged['receiving_tds_allowed'].fillna(0)
    )

    # 2-point conversions allowed
    two_pt_allowed = (
        merged['passing_2pt_conversions_allowed'].fillna(0) +
        merged['rushing_2pt_conversions_allowed'].fillna(0) +
        merged['receiving_2pt_conversions_allowed'].fillna(0)
    )

    # Field goals and PATs allowed
    fg_allowed = merged['fg_made_allowed'].fillna(0)
    pat_allowed = merged['pat_made_allowed'].fillna(0)

    # Total points allowed = TDs*6 + 2PT*2 + FG*3 + PAT*1
    defensive_df['pts_allow'] = (
        total_tds_allowed * 6 +
        two_pt_allowed * 2 +
        fg_allowed * 3 +
        pat_allowed * 1
    )
    defensive_df['dst_points_allowed'] = defensive_df['pts_allow']
    defensive_df['points_allowed'] = defensive_df['pts_allow']

    # Other stats
    defensive_df['misc_yards'] = merged['misc_yards']
    defensive_df['penalties'] = merged['penalties']
    defensive_df['penalty_yards'] = merged['penalty_yards']
    defensive_df['timeouts'] = merged['timeouts']

    print(f"[DEF] Transformation complete: {len(defensive_df):,} rows")
    print(f"[DEF] Sample points allowed: min={defensive_df['pts_allow'].min()}, max={defensive_df['pts_allow'].max()}, mean={defensive_df['pts_allow'].mean():.1f}")

    # Merge three-and-out stats if provided
    if three_out_stats is not None:
        print(f"[DEF] Merging three-and-out stats...")
        # Rename columns to match defensive_df
        three_out_stats = three_out_stats.rename(columns={
            'defteam': 'nfl_team',
            'season': 'year'
        })
        defensive_df = defensive_df.merge(
            three_out_stats[['nfl_team', 'week', 'year', 'three_out']],
            on=['nfl_team', 'week', 'year'],
            how='left'
        )
        defensive_df['three_out'] = defensive_df['three_out'].fillna(0).astype(int)
        print(f"[DEF] Added three_out column (mean={defensive_df['three_out'].mean():.1f})")

    # Merge fourth down stop stats if provided
    if fourth_down_stop_stats is not None:
        print(f"[DEF] Merging fourth down stop stats...")
        # Rename columns to match defensive_df
        fourth_down_stop_stats = fourth_down_stop_stats.rename(columns={
            'defteam': 'nfl_team',
            'season': 'year'
        })
        defensive_df = defensive_df.merge(
            fourth_down_stop_stats[['nfl_team', 'week', 'year', 'fourth_down_stop']],
            on=['nfl_team', 'week', 'year'],
            how='left'
        )
        defensive_df['fourth_down_stop'] = defensive_df['fourth_down_stop'].fillna(0).astype(int)
        print(f"[DEF] Added fourth_down_stop column (mean={defensive_df['fourth_down_stop'].mean():.1f})")

    return defensive_df


def get_max_week_from_matchup_data(data_directory: Path, year: int) -> int | None:
    """
    Get the maximum week from matchup data files.

    This allows NFLverse defense data to align with matchup data (only fetch weeks with actual matchups).

    Args:
        data_directory: League data directory (e.g., .../fantasy_football_data/KMFFL)
        year: Year to check

    Returns:
        Maximum week number found in matchup data, or None if no matchup data exists
    """
    try:
        matchup_dir = data_directory / "matchup_data"

        if not matchup_dir.exists():
            print(f"[matchup_max_week] Matchup directory not found: {matchup_dir}")
            return None

        # Try to find matchup file for this year
        # Prefer all-weeks file, fallback to individual week files
        all_weeks_file = matchup_dir / f"matchup_data_week_all_year_{year}.parquet"

        if all_weeks_file.exists():
            try:
                df = pd.read_parquet(all_weeks_file)
                if not df.empty and 'week' in df.columns:
                    max_week = int(df['week'].max())
                    print(f"[matchup_max_week] Found max week {max_week} from {all_weeks_file.name}")
                    return max_week
            except Exception as e:
                print(f"[matchup_max_week] Error reading {all_weeks_file.name}: {e}")

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
                print(f"[matchup_max_week] Found max week {max_week} from {len(week_files)} individual week files")
                return max_week

        print(f"[matchup_max_week] No matchup data found for year {year}")
        return None

    except Exception as e:
        print(f"[matchup_max_week] Error getting max week from matchup data: {e}")
        return None


def process_one_year(year: int, week: int = None, cache_dir: Path = None, use_cache: bool = True, data_directory: Path = None) -> pd.DataFrame:
    """
    Process defensive stats for a single year (used by combine_dst_to_nfl.py).

    Args:
        year: NFL season year (e.g., 2014)
        week: Optional week number (0 or None = all weeks)
        cache_dir: Directory to store cached downloads
        use_cache: Whether to use cached data if available
        data_directory: League data directory for matchup window context (optional)

    Returns:
        DataFrame with defensive stats including pts_allow, three_out, and fourth_down_stop
    """
    print(f"[DEF] Processing year {year}, week {week if week else 'all'}")

    # Fetch team stats from NFLverse (with caching)
    df = fetch_nflverse_team_stats(year, cache_dir=cache_dir, use_cache=use_cache)

    # Fetch play-by-play data for three-and-out and fourth down stop calculations (with caching)
    # For early years (1999-2000), play-by-play data may be incomplete or have missing columns
    three_out_stats = None
    fourth_down_stop_stats = None

    try:
        pbp_df = fetch_nflverse_pbp_data(year, cache_dir=cache_dir, use_cache=use_cache)

        # Verify required columns exist before calculating stats
        if 'drive_first_downs' in pbp_df.columns and 'play_type_nfl' in pbp_df.columns:
            three_out_stats = calculate_three_outs(pbp_df)
        else:
            print(f"[DEF] Warning: Missing columns for three_out calculation in {year} (drive_first_downs or play_type_nfl)")

        if 'down' in pbp_df.columns and 'play_type' in pbp_df.columns and 'first_down' in pbp_df.columns:
            fourth_down_stop_stats = calculate_fourth_down_stops(pbp_df)
        else:
            print(f"[DEF] Warning: Missing columns for fourth_down_stop calculation in {year} (down, play_type, or first_down)")

    except Exception as e:
        print(f"[DEF] Warning: Could not fetch or process play-by-play data for {year}: {e}")
        print(f"[DEF] Continuing without three_out and fourth_down_stop stats for {year}")

    # Transform to defensive format
    defensive_df = transform_to_defensive_stats(df, three_out_stats, fourth_down_stop_stats)

    # Calculate points allowed buckets
    defensive_df = calculate_points_allowed_buckets(defensive_df)

    # Week filtering logic:
    # - CURRENT YEAR (week=0/None): Limit to max week from matchup data to avoid incomplete weeks
    # - PAST YEARS (week=0/None): Pull ALL weeks (no matchup window limitation)
    # - ANY YEAR (specific week): Filter to that specific week only
    from datetime import datetime
    current_year = datetime.now().year

    # Filter by specific week if requested (applies to any year)
    if week and week > 0:
        defensive_df = defensive_df[defensive_df['week'] == week]
        print(f"[DEF] Filtered to week {week}: {len(defensive_df):,} rows")
    # For current year ONLY: limit to weeks with matchup data
    elif year == current_year and data_directory:
        max_week_from_matchups = get_max_week_from_matchup_data(data_directory, year)

        if max_week_from_matchups:
            print(f"[DEF] Current year {year}: filtering to max week from matchup data: {max_week_from_matchups}")
            defensive_df = defensive_df[defensive_df['week'] <= max_week_from_matchups]
            print(f"[DEF] Filtered to weeks 1-{max_week_from_matchups}: {len(defensive_df):,} rows")
        else:
            print(f"[DEF] WARNING: No matchup data found for current year {year}")
            print(f"[DEF] Using all available NFLverse data (may include incomplete weeks)")
    # For past years: use all available weeks (no filtering)
    else:
        if year < current_year:
            print(f"[DEF] Past year {year}: using all available weeks (no matchup window limitation)")

    return defensive_df


def calculate_points_allowed_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate points allowed buckets for DST fantasy scoring.

    Buckets (standard Yahoo/ESPN scoring):
    - 0 points allowed
    - 1-6 points allowed
    - 7-13 points allowed
    - 14-20 points allowed
    - 21-27 points allowed
    - 28-34 points allowed
    - 35+ points allowed

    Args:
        df: Defensive stats DataFrame (must have 'pts_allow' column)

    Returns:
        DataFrame with points allowed buckets added
    """
    if 'pts_allow' not in df.columns:
        print("[DEF] Warning: pts_allow column not found, cannot calculate buckets")
        return df

    result = df.copy()

    # Initialize all bucket columns to 0
    result['pts_allow_0'] = 0
    result['pts_allow_1_6'] = 0
    result['pts_allow_7_13'] = 0
    result['pts_allow_14_20'] = 0
    result['pts_allow_21_27'] = 0
    result['pts_allow_28_34'] = 0
    result['pts_allow_35_plus'] = 0

    # Set the appropriate bucket to 1 based on points allowed
    pts = result['pts_allow'].fillna(0)
    result.loc[pts == 0, 'pts_allow_0'] = 1
    result.loc[(pts >= 1) & (pts <= 6), 'pts_allow_1_6'] = 1
    result.loc[(pts >= 7) & (pts <= 13), 'pts_allow_7_13'] = 1
    result.loc[(pts >= 14) & (pts <= 20), 'pts_allow_14_20'] = 1
    result.loc[(pts >= 21) & (pts <= 27), 'pts_allow_21_27'] = 1
    result.loc[(pts >= 28) & (pts <= 34), 'pts_allow_28_34'] = 1
    result.loc[pts >= 35, 'pts_allow_35_plus'] = 1

    print(f"[DEF] Points allowed bucket distribution:")
    print(f"      0 pts: {result['pts_allow_0'].sum()}")
    print(f"      1-6: {result['pts_allow_1_6'].sum()}")
    print(f"      7-13: {result['pts_allow_7_13'].sum()}")
    print(f"      14-20: {result['pts_allow_14_20'].sum()}")
    print(f"      21-27: {result['pts_allow_21_27'].sum()}")
    print(f"      28-34: {result['pts_allow_28_34'].sum()}")
    print(f"      35+: {result['pts_allow_35_plus'].sum()}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch NFL team defensive stats from NFLverse")
    parser.add_argument("--year", type=int, required=True, help="Season year (e.g., 2014)")
    parser.add_argument("--week", type=int, default=0, help="Week number (0 = all weeks)")
    parser.add_argument("--context", type=str, default=None, help="Path to league_context.json")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download (skip cache)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom cache directory")
    args = parser.parse_args()

    use_cache = not args.no_cache
    cache_dir = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR

    # Load context if provided
    output_dir = DEFAULT_OUTPUT_DIR
    if args.context and LEAGUE_CONTEXT_AVAILABLE:
        try:
            ctx = LeagueContext.load(args.context)
            output_dir = Path(ctx.player_data_directory)
            print(f"[DEF] Using league: {ctx.league_name}")
            print(f"[DEF] Output: {output_dir}")
        except Exception as e:
            print(f"[DEF] Warning: Could not load context: {e}")
            print(f"[DEF] Falling back to default output: {output_dir}")
    else:
        print(f"[DEF] Using default output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch team stats from NFLverse (with caching)
    df = fetch_nflverse_team_stats(args.year, cache_dir=cache_dir, use_cache=use_cache)

    # Fetch play-by-play data for three-and-out and fourth down stop calculations (with caching)
    pbp_df = fetch_nflverse_pbp_data(args.year, cache_dir=cache_dir, use_cache=use_cache)
    three_out_stats = calculate_three_outs(pbp_df)
    fourth_down_stop_stats = calculate_fourth_down_stops(pbp_df)

    # Transform to defensive format
    defensive_df = transform_to_defensive_stats(df, three_out_stats, fourth_down_stop_stats)

    # Calculate points allowed buckets
    defensive_df = calculate_points_allowed_buckets(defensive_df)

    # Filter by week if specified
    if args.week > 0:
        defensive_df = defensive_df[defensive_df['week'] == args.week]
        print(f"[DEF] Filtered to week {args.week}: {len(defensive_df):,} rows")

    # Save output
    week_suffix = f"week_{args.week}" if args.week > 0 else "all_weeks"
    csv_path = output_dir / f"defense_stats_{args.year}_{week_suffix}.csv"
    parquet_path = output_dir / f"defense_stats_{args.year}_{week_suffix}.parquet"

    defensive_df.to_csv(csv_path, index=False)
    defensive_df.to_parquet(parquet_path, index=False)

    print(f"\n[DEF] Saved CSV: {csv_path}")
    print(f"[DEF] Saved Parquet: {parquet_path}")
    print(f"[DEF] Rows: {len(defensive_df):,}")
    print(f"[DEF] Columns: {len(defensive_df.columns)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
