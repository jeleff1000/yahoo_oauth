"""
Franchise Registry - Track Manager Sub-Franchises Across Years

Solves two problems:
1. Different people with same display name (e.g., two "David"s with different GUIDs)
2. Same person with multiple teams (e.g., Jay Dog owns "King Ads" AND "GreyWolf")

Key concepts:
- franchise_id: Stable identifier for career stat grouping (never changes)
- franchise_name: Display name, updates to most recent team name, only disambiguates when needed

Usage:
    from multi_league.core.franchise_registry import FranchiseRegistry

    # Discover franchises from data
    registry = FranchiseRegistry.from_data(matchup_df, draft_df)
    registry.save(data_dir / "franchise_config.json")

    # Apply to dataframes
    df = registry.apply_franchise_columns(df)
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass
class TeamSeason:
    """A single team-year instance within a franchise."""
    year: int
    team_name: str
    team_key: str = ""
    team_slot: int = 0
    wins: int = 0
    losses: int = 0
    points_for: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TeamSeason':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Franchise:
    """
    A franchise = one continuous team lineage.

    One owner (manager_guid) can have multiple franchises (sub-franchises).
    The franchise_id stays stable; franchise_name updates to current team name.
    """
    franchise_id: str
    franchise_name: str  # Display name - updates to most recent
    owner_guid: str
    owner_name: str  # Yahoo display name (e.g., "David", "Jay Dog")
    team_history: List[TeamSeason] = field(default_factory=list)
    years_active: List[int] = field(default_factory=list)

    def add_season(self, season: TeamSeason):
        """Add a team-season to this franchise's history."""
        # Update existing season if same year
        for i, s in enumerate(self.team_history):
            if s.year == season.year:
                self.team_history[i] = season
                return

        self.team_history.append(season)
        if season.year not in self.years_active:
            self.years_active.append(season.year)
            self.years_active.sort()

    def get_current_team_name(self) -> str:
        """Get the most recent team name."""
        if not self.team_history:
            return self.owner_name
        most_recent = max(self.team_history, key=lambda s: s.year)
        return most_recent.team_name or self.owner_name

    def get_career_stats(self) -> dict:
        """Calculate career totals from team history."""
        return {
            'career_wins': sum(s.wins for s in self.team_history),
            'career_losses': sum(s.losses for s in self.team_history),
            'career_points': sum(s.points_for for s in self.team_history),
            'years_played': len(self.years_active)
        }

    def to_dict(self) -> dict:
        return {
            'franchise_id': self.franchise_id,
            'franchise_name': self.franchise_name,
            'owner_guid': self.owner_guid,
            'owner_name': self.owner_name,
            'years_active': self.years_active,
            'team_history': [s.to_dict() for s in self.team_history]
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Franchise':
        history = [TeamSeason.from_dict(s) for s in d.pop('team_history', [])]
        years = d.pop('years_active', [])
        franchise = cls(
            franchise_id=d.get('franchise_id', ''),
            franchise_name=d.get('franchise_name', ''),
            owner_guid=d.get('owner_guid', ''),
            owner_name=d.get('owner_name', '')
        )
        franchise.team_history = history
        franchise.years_active = years
        return franchise


class FranchiseRegistry:
    """
    Registry for tracking franchises across a league.

    Handles:
    - Different people with same display name (different GUIDs)
    - Same person with multiple teams (same GUID, different teams)
    - Team name changes across years (same franchise, updated name)
    """

    def __init__(self):
        self.franchises: Dict[str, Franchise] = {}  # franchise_id -> Franchise
        self._config_path: Optional[Path] = None

        # Lookup indices
        self._guid_team_to_franchise: Dict[Tuple[str, str], str] = {}  # (guid, team_name) -> franchise_id
        self._year_guid_team_to_franchise: Dict[Tuple[int, str, str], str] = {}  # (year, guid, team_name) -> franchise_id

    @classmethod
    def from_data(
        cls,
        matchup_df: pd.DataFrame,
        draft_df: pd.DataFrame = None,
        existing_config: Path = None
    ) -> 'FranchiseRegistry':
        """
        Create/update registry from matchup and draft data.

        Args:
            matchup_df: DataFrame with manager_guid, manager, team_name, year columns
            draft_df: Optional DataFrame with team_key for slot extraction
            existing_config: Optional path to existing franchise_config.json to preserve manual edits
        """
        registry = cls()

        # Load existing config if present (preserves manual linkages)
        if existing_config and existing_config.exists():
            try:
                registry = cls.load(existing_config)
                print(f"[Franchise] Loaded existing config with {len(registry.franchises)} franchises")
            except Exception as e:
                print(f"[Franchise] Could not load existing config: {e}")
                registry = cls()

        # Process data to discover/update franchises
        registry._discover_from_data(matchup_df, draft_df)

        return registry

    def _discover_from_data(self, matchup_df: pd.DataFrame, draft_df: pd.DataFrame = None):
        """Discover franchises from data."""
        if matchup_df is None or matchup_df.empty:
            print("[Franchise] No matchup data provided")
            return

        required_cols = ['manager_guid', 'manager', 'year']
        if not all(col in matchup_df.columns for col in required_cols):
            print(f"[Franchise] Missing required columns: {required_cols}")
            return

        # Build team_key lookup from draft data if available
        team_key_lookup = {}  # (year, manager_guid, team_name) -> team_key
        if draft_df is not None and 'team_key' in draft_df.columns:
            for _, row in draft_df[['year', 'manager_guid', 'team_key']].drop_duplicates().iterrows():
                if pd.notna(row['manager_guid']) and pd.notna(row['team_key']):
                    # We need to correlate team_key to team_name
                    # For now, store by (year, guid, slot)
                    slot = self._extract_team_slot(row['team_key'])
                    team_key_lookup[(int(row['year']), str(row['manager_guid']), slot)] = str(row['team_key'])

        # Step 1: Find all unique (manager_guid, team_name) combinations per year
        team_instances = []
        cols = ['manager_guid', 'manager', 'year']
        if 'team_name' in matchup_df.columns:
            cols.append('team_name')
        if 'team_key' in matchup_df.columns:
            cols.append('team_key')

        for _, row in matchup_df[cols].drop_duplicates().iterrows():
            guid = row.get('manager_guid')
            if not guid or pd.isna(guid) or str(guid) in ('--', '', 'None'):
                continue

            team_instances.append({
                'manager_guid': str(guid),
                'manager': str(row.get('manager', '')),
                'year': int(row.get('year')),
                'team_name': str(row.get('team_name', '')) if 'team_name' in row else '',
                'team_key': str(row.get('team_key', '')) if 'team_key' in row else ''
            })

        # Step 2: Determine which managers need disambiguation
        # - Same manager name, different GUIDs
        # - Same GUID, multiple team names (in same year)
        manager_to_guids = defaultdict(set)
        guid_year_to_teams = defaultdict(set)  # (guid, year) -> set of team_names

        for inst in team_instances:
            manager_to_guids[inst['manager']].add(inst['manager_guid'])
            guid_year_to_teams[(inst['manager_guid'], inst['year'])].add(inst['team_name'])

        # Managers needing disambiguation: multiple GUIDs OR any GUID has multiple teams in a year
        guids_needing_disambiguation = set()
        for manager, guids in manager_to_guids.items():
            if len(guids) > 1:
                # Different people with same name
                guids_needing_disambiguation.update(guids)

        for (guid, year), teams in guid_year_to_teams.items():
            if len(teams) > 1:
                # Same person with multiple teams
                guids_needing_disambiguation.add(guid)

        print(f"[Franchise] GUIDs needing disambiguation: {len(guids_needing_disambiguation)}")

        # Step 3: Group instances by (manager_guid, team_name) to form franchises
        # Each unique (guid, team_name_lineage) = one franchise
        guid_team_instances = defaultdict(list)
        for inst in team_instances:
            key = (inst['manager_guid'], inst['team_name'])
            guid_team_instances[key].append(inst)

        # Step 4: Create/update franchises
        # Track which franchises we've seen for this GUID to assign indices
        guid_to_franchise_count = defaultdict(int)

        for (guid, team_name), instances in guid_team_instances.items():
            # Check if this team already exists in registry
            existing_fid = self._guid_team_to_franchise.get((guid, team_name))

            if existing_fid and existing_fid in self.franchises:
                # Update existing franchise
                franchise = self.franchises[existing_fid]
            else:
                # Create new franchise
                guid_to_franchise_count[guid] += 1
                team_index = guid_to_franchise_count[guid]

                franchise_id = f"{guid[:8]}_{team_index}"
                owner_name = instances[0]['manager']

                # Determine franchise_name based on disambiguation needs
                if guid in guids_needing_disambiguation:
                    # Get most recent team name for display
                    most_recent = max(instances, key=lambda x: x['year'])
                    current_team = most_recent['team_name'] or owner_name
                    franchise_name = f"{owner_name} - {current_team}"
                else:
                    franchise_name = owner_name

                franchise = Franchise(
                    franchise_id=franchise_id,
                    franchise_name=franchise_name,
                    owner_guid=guid,
                    owner_name=owner_name
                )
                self.franchises[franchise_id] = franchise

            # Add seasons to franchise
            for inst in instances:
                team_key = inst.get('team_key', '')
                slot = self._extract_team_slot(team_key) if team_key else 0

                # Try to get season stats from matchup data
                year_mask = (
                    (matchup_df['manager_guid'] == guid) &
                    (matchup_df['year'] == inst['year'])
                )
                if 'team_name' in matchup_df.columns and inst['team_name']:
                    year_mask &= (matchup_df['team_name'] == inst['team_name'])

                year_data = matchup_df[year_mask]

                wins = int(year_data['win'].sum()) if 'win' in year_data.columns else 0
                losses = int(year_data['loss'].sum()) if 'loss' in year_data.columns else 0
                points = float(year_data['team_points'].sum()) if 'team_points' in year_data.columns else 0.0

                season = TeamSeason(
                    year=inst['year'],
                    team_name=inst['team_name'],
                    team_key=team_key,
                    team_slot=slot,
                    wins=wins,
                    losses=losses,
                    points_for=points
                )
                franchise.add_season(season)

            # Update franchise_name to most recent (in case team name changed)
            if guid in guids_needing_disambiguation:
                current_team = franchise.get_current_team_name()
                franchise.franchise_name = f"{franchise.owner_name} - {current_team}"

        # Rebuild indices
        self._build_indices()

        print(f"[Franchise] Total franchises: {len(self.franchises)}")

    @staticmethod
    def _extract_team_slot(team_key: str) -> int:
        """Extract team slot from team_key (e.g., 461.l.836761.t.3 -> 3)."""
        if not team_key:
            return 0
        match = re.search(r'\.t\.(\d+)$', str(team_key))
        return int(match.group(1)) if match else 0

    def _build_indices(self):
        """Build lookup indices for fast franchise resolution."""
        self._guid_team_to_franchise.clear()
        self._year_guid_team_to_franchise.clear()

        for franchise_id, franchise in self.franchises.items():
            for season in franchise.team_history:
                # (guid, team_name) -> franchise_id
                key = (franchise.owner_guid, season.team_name)
                self._guid_team_to_franchise[key] = franchise_id

                # (year, guid, team_name) -> franchise_id
                year_key = (season.year, franchise.owner_guid, season.team_name)
                self._year_guid_team_to_franchise[year_key] = franchise_id

    def get_franchise_id(
        self,
        manager_guid: str,
        team_name: str = None,
        year: int = None
    ) -> Optional[str]:
        """
        Look up franchise_id for a given manager/team/year.

        Args:
            manager_guid: Yahoo manager GUID
            team_name: Team name (optional but recommended)
            year: Year (optional, helps with year-specific lookup)
        """
        if not manager_guid or pd.isna(manager_guid):
            return None

        guid = str(manager_guid)
        team = str(team_name) if team_name and pd.notna(team_name) else ''

        # Try year-specific lookup first
        if year:
            key = (int(year), guid, team)
            if key in self._year_guid_team_to_franchise:
                return self._year_guid_team_to_franchise[key]

        # Fall back to (guid, team_name) lookup
        key = (guid, team)
        if key in self._guid_team_to_franchise:
            return self._guid_team_to_franchise[key]

        # Last resort: find any franchise with this GUID
        for fid, franchise in self.franchises.items():
            if franchise.owner_guid == guid:
                return fid

        return None

    def get_franchise_name(self, franchise_id: str) -> Optional[str]:
        """Get display name for a franchise."""
        if franchise_id and franchise_id in self.franchises:
            return self.franchises[franchise_id].franchise_name
        return None

    def apply_franchise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add franchise_id and franchise_name columns to a DataFrame.

        Args:
            df: DataFrame with manager_guid column (and optionally team_name, year)
        """
        if df.empty:
            return df

        if 'manager_guid' not in df.columns:
            print("[Franchise] Warning: manager_guid column not found")
            return df

        df = df.copy()

        def lookup_franchise_id(row):
            return self.get_franchise_id(
                manager_guid=row.get('manager_guid'),
                team_name=row.get('team_name'),
                year=row.get('year')
            )

        df['franchise_id'] = df.apply(lookup_franchise_id, axis=1)
        df['franchise_name'] = df['franchise_id'].map(
            lambda fid: self.get_franchise_name(fid)
        )

        return df

    def get_orphan_teams(self) -> List[dict]:
        """
        Find teams that couldn't be linked to existing franchises.

        Returns list of (year, manager_guid, team_name) that need manual linking.
        """
        # TODO: Implement orphan detection for cross-year linking
        return []

    def calculate_roster_continuity(
        self,
        player_df: pd.DataFrame,
        year1: int,
        team1_guid: str,
        team1_name: str,
        year2: int,
        team2_guid: str,
        team2_name: str
    ) -> float:
        """
        Calculate roster overlap between two team-years.

        Used to suggest linkages when team names change.
        Returns score 0.0 to 1.0 (1.0 = perfect overlap).
        """
        if player_df is None or player_df.empty:
            return 0.0

        # Get rosters for each year (week 1 or any week)
        team1_players = set()
        team2_players = set()

        if 'yahoo_player_id' in player_df.columns:
            mask1 = (
                (player_df['year'] == year1) &
                (player_df['manager_guid'] == team1_guid)
            )
            if 'team_name' in player_df.columns:
                mask1 &= (player_df['team_name'] == team1_name)
            team1_players = set(player_df[mask1]['yahoo_player_id'].dropna().unique())

            mask2 = (
                (player_df['year'] == year2) &
                (player_df['manager_guid'] == team2_guid)
            )
            if 'team_name' in player_df.columns:
                mask2 &= (player_df['team_name'] == team2_name)
            team2_players = set(player_df[mask2]['yahoo_player_id'].dropna().unique())

        if not team1_players or not team2_players:
            return 0.0

        overlap = len(team1_players & team2_players)
        total = len(team1_players | team2_players)

        return overlap / total if total > 0 else 0.0

    def get_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame of all franchises."""
        rows = []
        for franchise in self.franchises.values():
            stats = franchise.get_career_stats()
            rows.append({
                'franchise_id': franchise.franchise_id,
                'franchise_name': franchise.franchise_name,
                'owner_name': franchise.owner_name,
                'owner_guid': franchise.owner_guid[:12] + '...',
                'years_active': len(franchise.years_active),
                'first_year': min(franchise.years_active) if franchise.years_active else None,
                'last_year': max(franchise.years_active) if franchise.years_active else None,
                'career_wins': stats['career_wins'],
                'career_losses': stats['career_losses'],
                'career_points': round(stats['career_points'], 1),
                'team_names': [s.team_name for s in franchise.team_history]
            })
        return pd.DataFrame(rows)

    def save(self, path: Path):
        """Save franchise config to JSON."""
        path = Path(path)

        config = {
            'version': '1.0',
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_franchises': len(self.franchises),
            'franchises': [f.to_dict() for f in self.franchises.values()]
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str)

        self._config_path = path
        print(f"[Franchise] Saved config to {path}")

    @classmethod
    def load(cls, path: Path) -> 'FranchiseRegistry':
        """Load franchise config from JSON."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Franchise config not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        registry = cls()
        registry._config_path = path

        for f_dict in config.get('franchises', []):
            franchise = Franchise.from_dict(f_dict)
            registry.franchises[franchise.franchise_id] = franchise

        registry._build_indices()
        print(f"[Franchise] Loaded {len(registry.franchises)} franchises from {path.name}")
        return registry

    def merge_franchises(self, source_id: str, target_id: str):
        """
        Manually merge two franchises (for correcting auto-discovery mistakes).

        Args:
            source_id: Franchise to merge FROM (will be deleted)
            target_id: Franchise to merge INTO
        """
        if source_id not in self.franchises:
            raise ValueError(f"Source franchise not found: {source_id}")
        if target_id not in self.franchises:
            raise ValueError(f"Target franchise not found: {target_id}")

        source = self.franchises[source_id]
        target = self.franchises[target_id]

        # Move all seasons to target
        for season in source.team_history:
            target.add_season(season)

        # Remove source
        del self.franchises[source_id]

        # Rebuild indices
        self._build_indices()

        print(f"[Franchise] Merged {source.franchise_name} into {target.franchise_name}")

    def rename_franchise(self, franchise_id: str, new_name: str):
        """Manually override franchise display name."""
        if franchise_id not in self.franchises:
            raise ValueError(f"Franchise not found: {franchise_id}")

        self.franchises[franchise_id].franchise_name = new_name
        print(f"[Franchise] Renamed {franchise_id} to '{new_name}'")
