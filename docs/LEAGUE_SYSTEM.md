# League System Architecture

## League Concept

A **League** is the fundamental organizational unit in this codebase. Each league represents a fantasy football league on Yahoo with:
- A unique league_id (e.g., `nfl.l.123456`)
- A human-readable name (e.g., "KMFFL")
- Multiple seasons of data (2014-2024)
- Multiple teams (typically 8-14)
- Complete transaction history (trades, pickups, drops)

---

## League References Throughout Codebase

### 1. League Discovery (`league_discovery.py`)

**Purpose**: Find all leagues accessible via Yahoo OAuth

```python
from league_discovery import LeagueDiscovery

discovery = LeagueDiscovery(oauth_file=Path("Oauth.json"))
leagues = discovery.discover_leagues(year=2024)
# Returns:
# [
#   {
#     'league_id': 'nfl.l.123456',
#     'league_name': 'KMFFL',
#     'season': 2024,
#     'num_teams': 10,
#     'scoring_type': 'ppr',
#     ...
#   },
#   ...
# ]
```

**Key Methods**:
- `discover_leagues(year)` - Find all leagues for a year
- `_fetch_league_metadata(league_key)` - Get detailed settings
- `create_league_context()` - Create LeagueContext from league data
- `interactive_register_leagues()` - Interactive CLI for user selection

**Key Class**: `LeagueRegistry`
- Maintains JSON registry of configured leagues
- Allows discovery of registered contexts

---

### 2. League Context (`league_context.py`)

**Purpose**: Encapsulate all configuration for processing a league

```python
from league_context import LeagueContext

ctx = LeagueContext(
    league_id="nfl.l.123456",          # Yahoo league key
    league_name="KMFFL",                # Display name
    oauth_file_path="oauth/Oauth.json", # OAuth credentials
    start_year=2014,                    # First year to fetch
    num_teams=10,                       # Team count
    manager_name_overrides={            # Manager name mapping
        "--hidden--": "Ilan",
        "Manager2": "John"
    }
)

# Auto-creates directory structure:
# {data_directory}/
#   ├── league_context.json
#   ├── league_settings/
#   ├── player_data/
#   ├── matchup_data/
#   ├── transaction_data/
#   ├── draft_data/
#   ├── schedule_data/
#   ├── logs/
#   ├── cache/
#   └── [canonical parquet files]

# Save for future use
ctx.save()

# Load later
ctx = LeagueContext.load("fantasy_football_data/nfl_l_123456/league_context.json")
```

**Key Properties**:
- `league_id` - Yahoo league key
- `league_name` - Display name
- `data_directory` - Root directory for all league data
- `player_data_directory` - Player stats storage
- `matchup_data_directory` - Matchup results storage
- `canonical_player_file` - `player.parquet` path
- `canonical_matchup_file` - `matchup.parquet` path
- `canonical_draft_file` - `draft.parquet` path
- `canonical_transaction_file` - `transactions.parquet` path
- `get_year_range()` - Iterator over start_year to end_year
- `get_yahoo_years()` - Years to fetch from Yahoo (excludes ESPN CSV years)

**Key Methods**:
- `get_oauth_session()` - Create OAuth2 session for Yahoo API
- `to_dict() / from_dict()` - JSON serialization
- `save() / load()` - Persist configuration
- `summary()` - Human-readable configuration dump

---

### 3. Pipeline Orchestrator (`initial_import_v2.py`)

**Purpose**: Run complete import for a league

```bash
python initial_import_v2.py --context path/to/league_context.json
```

**What it does**:
1. Loads `LeagueContext` from JSON
2. PHASE 0: Fetch league settings for all years (parallel, 5 workers)
3. PHASE 1: Run 7 data fetchers (sequential, per-league)
4. PHASE 2: Merge Yahoo + NFL data (per-league)
5. PHASE 3: Run 13 transformations (per-league, dependency-ordered)

**League Isolation**:
- Reads league_id from context
- Passes to all child scripts via environment variables
- Each script reads context.league_id
- Output files tagged with league_id column

**Output** (all in context.data_directory):
- `player.parquet` - Player stats with league_id column
- `matchup.parquet` - Matchup results with league_id column
- `draft.parquet` - Draft data with league_id column
- `transactions.parquet` - Transactions with league_id column
- `schedule_data_all_years.parquet` - Schedule with league_id column
- `players_by_year.parquet` - Season aggregations with league_id column

---

### 4. Data Fetchers

All data fetchers **receive** league context and **produce** per-league data

```
Input: league_context.json (contains league_id, league_name, oauth_file)
Output: Per-league parquet files in {data_directory}/{type}_data/
```

**Example Flow**:
```python
# yahoo_fantasy_data.py runs for a single league
league = League(oauth, league_key)
roster = league.get_roster(week)  # Get roster for this league
# Save to {data_directory}/player_data/yahoo_player_stats_2024_week_1.parquet
```

**Key Fetchers**:
1. `weekly_matchup_data_v2.py` - Yahoo matchup results (by league)
2. `yahoo_fantasy_data.py` - Yahoo player stats (by league)
3. `nfl_offense_stats.py` - External NFL stats (shared across leagues)
4. `defense_stats.py` - External NFL DST stats (shared)
5. `draft_data_v2.py` - Draft results (by league)
6. `transactions_v2.py` - Transactions (by league)
7. `season_schedules.py` - Schedule (by league)

---

### 5. Transformations

All transformations **operate on single league** and **produce league-tagged output**

```python
# cumulative_stats_v2.py
league_id = ctx.league_id
df = pd.read_parquet(ctx.canonical_matchup_file)
# df already has league_id column from aggregation
# Add cumulative calculations
df['wins_to_date'] = df.groupby(['league_id', 'home_manager']).cumsum()
# Save with league_id intact
df.to_parquet(ctx.canonical_matchup_file)
```

**13 Transformation Scripts**:
1. **cumulative_stats_v2.py** - Win-loss, points-for/against tracking
2. **matchup_to_player_v2.py** - Attach matchup context to player stats
3. **player_stats_v2.py** - Player season stats calculations
4. **player_to_matchup_v2.py** - Aggregate player data into matchups
5. **player_to_transactions_v2.py** - Link players to transaction records
6. **transactions_to_player_v2.py** - Add FAAB data to player records
7. **draft_enrichment_v2.py** - Keeper economics calculations
8. **draft_to_player_v2.py** - Add draft info to player records
9. **player_to_draft_v2.py** - Add player stats to draft records
10. **expected_record_v2.py** - W-L if scheduled differently
11. **playoff_odds_import.py** - Monte Carlo playoff simulations
12. **aggregate_player_season_v2.py** - Create `players_by_year.parquet`
13. **keeper_economics_v2.py** - Keeper value analysis

---

### 6. MotherDuck Upload

Each league gets its own database:

```python
import duckdb

# Database naming
league_name = "KMFFL"
season = 2024
db_name = f"{league_name}_{season}".lower().replace(' ', '_')
# Result: "kmffl_2024"

# Connect and create database
con = duckdb.connect(f"md:{db_name}?motherduck_token={token}")

# Load all parquet files for this league
con.execute(f"CREATE OR REPLACE TABLE player AS SELECT * FROM read_parquet('{player_parquet}')")
con.execute(f"CREATE OR REPLACE TABLE matchup AS SELECT * FROM read_parquet('{matchup_parquet}')")
# ... etc for each parquet

# Create analysis views
con.execute("""
    CREATE OR REPLACE VIEW season_summary AS
    SELECT 
        season,
        manager_name,
        COUNT(*) as games_played,
        SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as wins,
        ...
    FROM matchup
    WHERE league_id = '{league_id}'
    GROUP BY season, manager_name
""")
```

**Result**:
- Database: `kmffl_2024`
- Tables: `player`, `matchup`, `draft`, `transactions`, `schedule_data`, `players_by_year`
- All tables have `league_id` column for isolation

---

### 7. Streamlit Integration

**League Selection Flow**:
```
1. User authenticates via OAuth
2. Streamlit calls Yahoo API: game.league_ids(year)
3. For each league, get metadata: league.settings()
4. Display leagues to user
5. User selects league
6. Streamlit creates LeagueContext
7. Streamlit saves OAuth token to oauth/Oauth_{league_key}.json
8. Streamlit calls: initial_import_v2.py --context league_context.json
9. Pipeline runs with league isolation
10. Results uploaded to MotherDuck database for that league
```

**Key Code** (main.py):
```python
# Extract leagues from Yahoo API response
leagues = extract_football_games(games_data)  # Yahoo API response
# Returns: [{'game_key': '...', 'season': 2024, 'name': 'NFL'}, ...]

# Get leagues for specific season
leagues_data = get_user_football_leagues(token, game_key)
# Returns: All leagues for that game/season

# Save league info for import
st.session_state.league_info = {
    "league_key": "nfl.l.123456",
    "name": "KMFFL",
    "season": 2024,
    "num_teams": 10
}

# Trigger import with league context
run_initial_import()  # Passes league_info to initial_import_v2.py
```

---

## Multi-League Support

The system handles multiple leagues simultaneously:

### Registry Management
```python
from league_discovery import LeagueRegistry

registry = LeagueRegistry()
# Maintains file: ~/fantasy_football_data/leagues.json
# {
#   "nfl.l.123456": {
#     "league_id": "nfl.l.123456",
#     "league_name": "KMFFL",
#     "context_file": "/path/to/league_context.json",
#     "data_directory": "/path/to/data",
#     "status": "active"
#   },
#   "nfl.l.789012": { ... },
#   ...
# }

registry.register_league(ctx)
registry.list_leagues(status="active")
registry.load_contexts()  # Load all LeagueContext objects
```

### Batch Processing
```python
from league_discovery import discover_contexts

all_contexts = discover_contexts(base_directory)
# Returns: {"nfl.l.123456": LeagueContext(...), ...}

for league_id, ctx in all_contexts.items():
    print(f"Processing: {ctx.league_name}")
    # Run import for each league
    run_script("initial_import_v2.py", ["--context", str(ctx.data_directory / "league_context.json")])
```

### Data Isolation
- Each league has separate `data_directory`
- Each parquet has `league_id` column
- MotherDuck database per league per season
- Queries automatically filter: `WHERE league_id = 'nfl.l.123456'`

---

## League Data in Practice

### Files Created
```
~/fantasy_football_data/
└── nfl_l_123456/                    # league_id with dots replaced
    ├── league_context.json          # Serialized LeagueContext
    ├── league_settings/             # Yahoo settings per year
    │   ├── 2024_settings.json
    │   ├── 2023_settings.json
    │   └── ... (2014-2024)
    ├── player_data/                 # Yahoo + NFL stats
    │   ├── yahoo_player_stats_2024_week_1.parquet
    │   ├── nfl_offense_stats_2024_week_1.parquet
    │   └── ... (all weeks, all years)
    ├── matchup_data/
    │   ├── matchup_data_2024_week_1.parquet
    │   └── ... (all weeks, all years)
    ├── draft_data/
    │   ├── draft_2024.parquet
    │   └── ... (one per year)
    ├── transaction_data/
    │   ├── transactions_2024.parquet
    │   └── ... (one per year)
    ├── schedule_data/
    │   └── schedule_data_all_years.parquet
    ├── logs/                        # JSON logs from runs
    ├── cache/                       # API cache for performance
    ├── player.parquet               # **MAIN OUTPUT**
    ├── matchup.parquet              # **MAIN OUTPUT**
    ├── draft.parquet                # **MAIN OUTPUT**
    ├── transactions.parquet         # **MAIN OUTPUT**
    ├── schedule_data_all_years.parquet
    └── players_by_year.parquet
```

### Canonical Parquets
All have `league_id` column for isolation:

```
player.parquet
├── league_id (e.g., 'nfl.l.123456')
├── season (2014-2024)
├── week (1-18)
├── player_id
├── player_name
├── position
├── yahoo_team
├── yahoo_points
├── nfl_points
└── ... other stats

matchup.parquet
├── league_id
├── season
├── week
├── home_team
├── away_team
├── home_manager
├── away_manager
├── home_points
├── away_points
├── won (1/0 for home team)
├── playoff_seed_to_date
└── ... calculated stats

draft.parquet
├── league_id
├── season
├── draft_order
├── team
├── player_name
├── player_id
├── pick_round
├── keeper_eligible
└── ... keeper tracking

transactions.parquet
├── league_id
├── season
├── timestamp
├── transaction_type (trade/pickup/drop)
├── player_name
├── player_id
├── from_team
├── to_team
├── faab_spent
└── ... transaction details
```

---

## Summary

**League** is the atomic unit of data organization:

1. **Discovery** - LeagueDiscovery finds leagues via Yahoo API
2. **Context** - LeagueContext encapsulates league configuration
3. **Fetching** - Data fetchers query Yahoo API for league data
4. **Processing** - Transformations enrich league-specific data
5. **Output** - Canonical parquets tagged with league_id
6. **Storage** - MotherDuck database per league per season
7. **Registry** - LeagueRegistry tracks configured leagues

All components respect league isolation:
- No cross-league data mixing
- Separate configuration per league
- Separate database per league
- Every output tagged with league_id column

