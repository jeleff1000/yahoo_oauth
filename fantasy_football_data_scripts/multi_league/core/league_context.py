"""
League Context System - Multi-League Configuration

This module provides the LeagueContext dataclass that encapsulates all configuration
needed to process data for a specific fantasy football league.

Key features:
- League-specific paths (data directories, OAuth files)
- Processing configuration (workers, rate limits, caching)
- League metadata (team count, scoring rules, manager overrides)
- Serialization to/from JSON for persistence
- Automatic directory creation
- Validation of required fields

Usage:
    # Create new context
    ctx = LeagueContext(
        league_id="nfl.l.123456",
        league_name="KMFFL",
        oauth_file_path="path/to/Oauth.json",
        start_year=2014
    )

    # Save to file
    ctx.save("leagues/kmffl/league_context.json")

    # Load from file
    ctx = LeagueContext.load("leagues/kmffl/league_context.json")

    # Access league-specific directories
    player_data_path = ctx.player_data_directory
    matchup_data_path = ctx.matchup_data_directory
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


@dataclass
class LeagueContext:
    """
    Complete configuration context for processing a fantasy football league.

    This class centralizes all league-specific configuration including:
    - Identification (league_id, league_name)
    - Data locations (data_directory, oauth_file_path)
    - Processing settings (max_workers, rate_limit_per_sec, enable_caching)
    - League metadata (num_teams, playoff_teams, manager_name_overrides)
    - Time range (start_year, end_year)

    All paths are automatically resolved and created when the context is instantiated.
    """

    # === Required Fields ===
    league_id: str                          # Yahoo league_key (e.g., "nfl.l.123456")
    league_name: str                        # Human-readable name (e.g., "KMFFL")
    oauth_file_path: Optional[str] = None   # Path to Yahoo OAuth credentials (optional if oauth_credentials provided)
    oauth_credentials: Optional[Dict[str, Any]] = None  # Embedded OAuth credentials (access_token, consumer_key, consumer_secret, etc.)

    # === Optional League Metadata ===
    game_code: str = "nfl"                  # Game type (nfl, mlb, nba, nhl)
    start_year: int = 2014                  # First year of data to fetch
    end_year: Optional[int] = None          # Last year (None = current year)
    num_teams: Optional[int] = None         # Number of teams in league
    playoff_teams: Optional[int] = None     # Number of playoff teams
    regular_season_weeks: Optional[int] = None  # Weeks in regular season

    # === Data Storage ===
    data_directory: Optional[Path] = None   # Root directory for this league's data

    # === Processing Configuration ===
    max_workers: int = 3                    # Parallelism for data fetching
    enable_caching: bool = True             # Enable performance caching
    rate_limit_per_sec: float = 4.0         # API rate limit (requests/sec)

    # === Manager Overrides ===
    manager_name_overrides: Dict[str, str] = field(default_factory=dict)
    # Maps old manager names to new names (e.g., {"--hidden--": "Ilan"})

    # === League History (year -> league_id mapping) ===
    league_ids: Dict[str, str] = field(default_factory=dict)
    # Maps year (as string) to Yahoo league_key for that season
    # e.g., {"2016": "449.l.198278", "2017": "380.l.123456", ...}
    # This ensures we always fetch from the correct league for each year
    # Built by following the renew/renewed chain from Yahoo API

    # === Keeper Rules ===
    keeper_rules: Optional[Dict[str, Any]] = None
    # Keeper price calculation rules - see schema:
    # {
    #   "enabled": bool,
    #   "max_keepers": int,
    #   "budget": int,
    #   "formulas_by_keeper_year": {"1": {...}, "2+": {...}},
    #   "base_cost_rules": {"auction": {...}, "snake": {...}, ...},
    #   "min_price": int,
    #   "max_price": int or null,
    #   "round_to_integer": bool
    # }

    # === Metadata ===
    created_at: Optional[str] = None        # ISO timestamp of context creation
    updated_at: Optional[str] = None        # ISO timestamp of last update

    # === Validation Flags ===
    require_oauth: bool = True              # If False, skip OAuth validation (for read-only operations)

    def __post_init__(self):
        """
        Initialize derived fields and validate configuration.

        - Resolves data_directory to absolute path
        - Creates necessary subdirectories
        - Sets created_at timestamp if new
        - Validates required fields
        """
        # Set timestamps
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

        # Resolve data directory
        if self.data_directory is None:
            # Default: ~/fantasy_football_data/{league_id}/
            base_dir = Path.home() / "fantasy_football_data"
            self.data_directory = base_dir / self._sanitize_league_id()
        else:
            self.data_directory = Path(self.data_directory).resolve()

        # Create directory structure
        self._create_directories()

        # Validate
        self._validate()

    def _sanitize_league_id(self) -> str:
        """
        Convert league_id to filesystem-safe name.

        Replaces dots with underscores (e.g., "nfl.l.123456" -> "nfl_l_123456")
        """
        return self.league_id.replace(".", "_")

    def _create_directories(self):
        """
        Create all necessary subdirectories for league data.

        Directory structure:
            {data_directory}/
                player_data/
                    yahoo_league_settings/
                matchup_data/
                transaction_data/
                draft_data/
                schedule_data/
                logs/
                cache/
        """
        directories = [
            self.data_directory,
            self.data_directory / "league_settings",  # League-wide configuration
            self.player_data_directory,
            self.matchup_data_directory,
            self.transaction_data_directory,
            self.draft_data_directory,
            self.schedule_data_directory,
            self.logs_directory,
            self.cache_directory,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _validate(self):
        """
        Validate required fields and configuration.

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not self.league_id:
            raise ValueError("league_id is required")

        if not self.league_name:
            raise ValueError("league_name is required")

        # Validate OAuth configuration (need either file path or embedded credentials)
        # Skip validation if require_oauth is False (for read-only operations like playoff odds)
        if self.require_oauth and not self.oauth_file_path and not self.oauth_credentials:
            raise ValueError("Either oauth_file_path or oauth_credentials is required")

        # Validate OAuth file exists if file path provided
        if self.oauth_file_path:
            oauth_path = Path(self.oauth_file_path)
            if not oauth_path.exists():
                raise ValueError(f"OAuth file not found: {oauth_path}")

        # Validate embedded OAuth credentials if provided
        if self.oauth_credentials:
            required_keys = ['access_token', 'consumer_key', 'consumer_secret', 'refresh_token', 'token_type']
            missing = [k for k in required_keys if k not in self.oauth_credentials]
            if missing:
                raise ValueError(f"OAuth credentials missing required keys: {missing}")

        # Validate year range
        if self.start_year < 2000 or self.start_year > datetime.now().year:
            raise ValueError(f"Invalid start_year: {self.start_year}")

        if self.end_year is not None:
            if self.end_year < self.start_year:
                raise ValueError(f"end_year ({self.end_year}) cannot be before start_year ({self.start_year})")

        # Validate rate limit
        if self.rate_limit_per_sec <= 0:
            raise ValueError(f"rate_limit_per_sec must be positive: {self.rate_limit_per_sec}")

    # === Directory Properties ===

    @property
    def player_data_directory(self) -> Path:
        """Directory for player stats data (yahoo_player_stats_*.parquet)"""
        return self.data_directory / "player_data"

    @property
    def matchup_data_directory(self) -> Path:
        """Directory for matchup data (matchup_data_*.parquet)"""
        return self.data_directory / "matchup_data"

    @property
    def transaction_data_directory(self) -> Path:
        """Directory for transaction data (transactions_*.parquet)"""
        return self.data_directory / "transaction_data"

    @property
    def draft_data_directory(self) -> Path:
        """Directory for draft data (draft_results_*.parquet)"""
        return self.data_directory / "draft_data"

    @property
    def schedule_data_directory(self) -> Path:
        """Directory for schedule data (schedule.parquet)"""
        return self.data_directory / "schedule_data"

    @property
    def logs_directory(self) -> Path:
        """Directory for run logs (JSON files from RunLogger)"""
        return self.data_directory / "logs"

    @property
    def cache_directory(self) -> Path:
        """Directory for performance caching"""
        return self.data_directory / "cache"

    # === Canonical File Paths ===

    @property
    def canonical_player_file(self) -> Path:
        """Path to canonical player.parquet file"""
        return self.data_directory / "player.parquet"

    @property
    def canonical_matchup_file(self) -> Path:
        """Path to canonical matchup.parquet file"""
        return self.data_directory / "matchup.parquet"

    @property
    def canonical_transaction_file(self) -> Path:
        """Path to canonical transactions.parquet file (plural)"""
        return self.data_directory / "transactions.parquet"

    @property
    def canonical_draft_file(self) -> Path:
        """Path to canonical draft.parquet file"""
        return self.data_directory / "draft.parquet"

    # === Keeper Rules Helpers ===

    @property
    def keepers_enabled(self) -> bool:
        """Check if keeper rules are enabled for this league."""
        if self.keeper_rules is None:
            return False
        return self.keeper_rules.get('enabled', False)

    @property
    def max_keepers(self) -> int:
        """Get maximum number of keepers allowed (default 0 if not configured)."""
        if self.keeper_rules is None:
            return 0
        return self.keeper_rules.get('max_keepers', 0)

    @property
    def keeper_budget(self) -> int:
        """Get auction budget for keeper calculations (default 200)."""
        if self.keeper_rules is None:
            return 200
        return self.keeper_rules.get('budget', 200)

    def get_keeper_formula(self, keeper_year: int) -> Optional[Dict[str, Any]]:
        """
        Get keeper price formula for a specific keeper year.

        Args:
            keeper_year: How many times player has been kept (1 = first time)

        Returns:
            Formula dict with 'expression' and 'description', or None
        """
        if self.keeper_rules is None:
            return None

        formulas = self.keeper_rules.get('formulas_by_keeper_year', {})

        # Check for exact year match first
        if str(keeper_year) in formulas:
            return formulas[str(keeper_year)]

        # Check for "2+" style wildcards
        for key, formula in formulas.items():
            if '+' in key:
                base_year = int(key.replace('+', ''))
                if keeper_year >= base_year:
                    return formula

        return None

    def get_base_cost_rule(self, acquisition_type: str) -> Optional[Dict[str, Any]]:
        """
        Get base cost calculation rule for an acquisition type.

        Args:
            acquisition_type: 'auction', 'snake', 'faab_only', or 'free_agent'

        Returns:
            Rule dict with 'source', optional 'formula', 'multiplier', 'value'
        """
        if self.keeper_rules is None:
            return None

        base_rules = self.keeper_rules.get('base_cost_rules', {})
        return base_rules.get(acquisition_type)

    # === OAuth Helpers ===

    def get_oauth_session(self):
        """
        Get an OAuth2 session for Yahoo API calls.

        Returns an OAuth2 session that can be used with yahoo_fantasy_api.
        Works with either oauth_file_path or embedded oauth_credentials.

        Returns:
            OAuth2 session object

        Raises:
            ImportError: If yahoo_oauth is not installed
            ValueError: If neither oauth_file_path nor oauth_credentials is available
        """
        try:
            from yahoo_oauth import OAuth2
        except ImportError:
            raise ImportError("yahoo_oauth is required. Install with: pip install yahoo_oauth")

        # If oauth_file_path is provided, use it
        if self.oauth_file_path:
            oauth_path = Path(self.oauth_file_path)
            if not oauth_path.exists():
                raise ValueError(f"OAuth file not found: {oauth_path}")

            oauth = OAuth2(None, None, from_file=str(oauth_path))
            if not oauth.token_is_valid():
                oauth.refresh_access_token()
            return oauth

        # Otherwise use embedded credentials
        if self.oauth_credentials:
            # Create temporary OAuth session from credentials dict
            import tempfile
            import json

            # Write credentials to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(self.oauth_credentials, f)
                temp_path = f.name

            try:
                oauth = OAuth2(None, None, from_file=temp_path)
                if not oauth.token_is_valid():
                    oauth.refresh_access_token()
                return oauth
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(temp_path)
                except:
                    pass

        raise ValueError("Either oauth_file_path or oauth_credentials is required")

    # === League ID Helpers ===

    def get_league_id_for_year(self, year: int) -> Optional[str]:
        """
        Get the specific league_id for a given year.

        This ensures we always fetch from the correct league for each season,
        avoiding data mixing when a user is in multiple leagues.

        Args:
            year: The season year (e.g., 2016)

        Returns:
            The Yahoo league_key for that year, or None if not found

        Example:
            league_key = ctx.get_league_id_for_year(2016)
            # Returns "449.l.198278" for KMFFL 2016
        """
        year_str = str(year)
        if year_str in self.league_ids:
            return self.league_ids[year_str]

        # Fallback to current league_id only if year matches end_year or current year
        current_year = datetime.now().year
        end = self.end_year if self.end_year else current_year
        if year == end:
            return self.league_id

        return None

    def has_league_ids_mapping(self) -> bool:
        """
        Check if league_ids mapping is populated.

        Returns:
            True if league_ids dict has entries, False otherwise
        """
        return bool(self.league_ids)

    # === Serialization ===

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context to dictionary for serialization.

        Converts Path objects to strings for JSON compatibility.
        """
        data = asdict(self)

        # Convert Path to string
        if self.data_directory:
            data['data_directory'] = str(self.data_directory)

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LeagueContext':
        """
        Create LeagueContext from dictionary.

        Args:
            data: Dictionary with context fields

        Returns:
            LeagueContext instance
        """
        # Convert data_directory string to Path if present
        if 'data_directory' in data and data['data_directory']:
            data['data_directory'] = Path(data['data_directory'])

        return cls(**data)

    def save(self, path: Optional[Path] = None):
        """
        Save context to JSON file.

        Args:
            path: Path to save file. If None, saves to {data_directory}/league_context.json
        """
        if path is None:
            path = self.data_directory / "league_context.json"
        else:
            path = Path(path)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        self.updated_at = datetime.now().isoformat()

        # Write JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'LeagueContext':
        """
        Load context from JSON file.

        Args:
            path: Path to league_context.json file

        Returns:
            LeagueContext instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"League context file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def load_readonly(cls, path: Path) -> 'LeagueContext':
        """
        Load context from JSON file without OAuth validation.

        Use this for read-only operations that don't need Yahoo API access,
        such as playoff odds calculation, data analysis, or transformations
        that only read from existing parquet files.

        Args:
            path: Path to league_context.json file

        Returns:
            LeagueContext instance (with require_oauth=False)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"League context file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Skip OAuth validation for read-only operations
        data['require_oauth'] = False

        return cls.from_dict(data)

    # === Utility Methods ===

    def get_year_range(self) -> range:
        """
        Get range of years to process.

        Returns:
            range object from start_year to end_year (inclusive)
        """
        end = self.end_year if self.end_year else datetime.now().year
        return range(self.start_year, end + 1)

    def get_cache_path(self, cache_type: str) -> Path:
        """
        Get cache directory for specific cache type.

        Args:
            cache_type: Type of cache (e.g., 'yahoo_api', 'nfl_stats')

        Returns:
            Path to cache directory
        """
        cache_path = self.cache_directory / cache_type
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_log_path(self, script_name: str, year: Optional[int] = None,
                     week: Optional[int] = None) -> Path:
        """
        Generate log file path for a script run.

        Args:
            script_name: Name of script (e.g., 'yahoo_fantasy_data')
            year: Optional year parameter
            week: Optional week parameter

        Returns:
            Path to log file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename_parts = [script_name, timestamp]
        if year:
            filename_parts.append(f"y{year}")
        if week:
            filename_parts.append(f"w{week}")

        filename = "_".join(filename_parts) + ".json"
        return self.logs_directory / filename

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"LeagueContext("
            f"league_id='{self.league_id}', "
            f"league_name='{self.league_name}', "
            f"data_directory='{self.data_directory}')"
        )

    def summary(self) -> str:
        """
        Get human-readable summary of league context.

        Returns:
            Multi-line string with key configuration details
        """
        lines = [
            f"League: {self.league_name} ({self.league_id})",
            f"Data Directory: {self.data_directory}",
            f"Year Range: {self.start_year}-{self.end_year or 'current'}",
            f"OAuth: {self.oauth_file_path}",
            f"Teams: {self.num_teams or 'unknown'}",
            f"Rate Limit: {self.rate_limit_per_sec} req/sec",
            f"Max Workers: {self.max_workers}",
            f"Caching: {'enabled' if self.enable_caching else 'disabled'}",
        ]

        if self.manager_name_overrides:
            lines.append(f"Manager Overrides: {len(self.manager_name_overrides)}")

        if self.keepers_enabled:
            lines.append(f"Keepers: {self.max_keepers} max, ${self.keeper_budget} budget")
        else:
            lines.append("Keepers: disabled")

        lines.append(f"Created: {self.created_at}")
        lines.append(f"Updated: {self.updated_at}")

        return "\n".join(lines)


# === Helper Functions ===

def create_league_context(
    league_id: str,
    league_name: str,
    oauth_file_path: str,
    start_year: int = 2014,
    data_directory: Optional[Path] = None,
    **kwargs
) -> LeagueContext:
    """
    Factory function to create and save a new LeagueContext.

    Args:
        league_id: Yahoo league_key
        league_name: Human-readable league name
        oauth_file_path: Path to OAuth credentials
        start_year: First year to process
        data_directory: Custom data directory (optional)
        **kwargs: Additional context parameters

    Returns:
        LeagueContext instance (also saved to disk)

    Example:
        ctx = create_league_context(
            league_id="nfl.l.123456",
            league_name="KMFFL",
            oauth_file_path="oauth/Oauth.json",
            start_year=2014,
            num_teams=10,
            manager_name_overrides={"--hidden--": "Ilan"}
        )
    """
    ctx = LeagueContext(
        league_id=league_id,
        league_name=league_name,
        oauth_file_path=oauth_file_path,
        start_year=start_year,
        data_directory=data_directory,
        **kwargs
    )

    # Save to default location
    ctx.save()

    return ctx


def load_league_context(path: Path) -> LeagueContext:
    """
    Load league context from JSON file.

    Convenience wrapper around LeagueContext.load().

    Args:
        path: Path to league_context.json

    Returns:
        LeagueContext instance
    """
    return LeagueContext.load(path)


def discover_contexts(base_directory: Optional[Path] = None) -> Dict[str, LeagueContext]:
    """
    Discover all league contexts in a directory tree.

    Searches for all league_context.json files and loads them.

    Args:
        base_directory: Root directory to search (default: ~/fantasy_football_data)

    Returns:
        Dictionary mapping league_id to LeagueContext

    Example:
        contexts = discover_contexts()
        for league_id, ctx in contexts.items():
            print(f"Found league: {ctx.league_name}")
    """
    if base_directory is None:
        base_directory = Path.home() / "fantasy_football_data"
    else:
        base_directory = Path(base_directory)

    if not base_directory.exists():
        return {}

    contexts = {}

    # Search for all league_context.json files
    for context_file in base_directory.rglob("league_context.json"):
        try:
            ctx = LeagueContext.load(context_file)
            contexts[ctx.league_id] = ctx
        except Exception as e:
            print(f"Warning: Could not load {context_file}: {e}")

    return contexts


# === CLI Support ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="League Context Management")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create new league context')
    create_parser.add_argument('--league-id', required=True, help='Yahoo league_key')
    create_parser.add_argument('--league-name', required=True, help='League name')
    create_parser.add_argument('--oauth-file', required=True, help='Path to OAuth file')
    create_parser.add_argument('--start-year', type=int, default=2014, help='Start year')
    create_parser.add_argument('--num-teams', type=int, help='Number of teams')
    create_parser.add_argument('--data-dir', help='Custom data directory')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show league context')
    show_parser.add_argument('path', help='Path to league_context.json')

    # List command
    list_parser = subparsers.add_parser('list', help='List all discovered contexts')
    list_parser.add_argument('--base-dir', help='Base directory to search')

    args = parser.parse_args()

    if args.command == 'create':
        ctx = create_league_context(
            league_id=args.league_id,
            league_name=args.league_name,
            oauth_file_path=args.oauth_file,
            start_year=args.start_year,
            num_teams=args.num_teams,
            data_directory=Path(args.data_dir) if args.data_dir else None
        )
        print(f"Created league context: {ctx.data_directory / 'league_context.json'}")
        print("\n" + ctx.summary())

    elif args.command == 'show':
        ctx = load_league_context(Path(args.path))
        print(ctx.summary())

    elif args.command == 'list':
        base_dir = Path(args.base_dir) if args.base_dir else None
        contexts = discover_contexts(base_dir)

        if not contexts:
            print("No league contexts found")
        else:
            print(f"Found {len(contexts)} league(s):\n")
            for league_id, ctx in contexts.items():
                print(f"  {ctx.league_name} ({league_id})")
                print(f"    Location: {ctx.data_directory}")
                print(f"    Years: {ctx.start_year}-{ctx.end_year or 'current'}")
                print()

    else:
        parser.print_help()
