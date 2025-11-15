# Multi-League Infrastructure

This directory contains all the multi-league compliant modules for fantasy football data processing.

## Directory Structure

```
multi_league/
├── core/                   # Core multi-league infrastructure
│   ├── league_context.py   # LeagueContext dataclass (league-specific config)
│   └── league_discovery.py # Auto-discover and register leagues
│
├── data_fetchers/          # Data fetching modules
│   ├── yahoo_fantasy_data_v2.py  # Fetch Yahoo player data
│   └── nfl_offense_stats_v2.py   # Fetch NFL offensive stats
│
├── merge/                  # Data merging logic
│   ├── yahoo_settings.py        # Yahoo league settings fetcher
│   ├── points_calculator.py     # Fantasy points calculation
│   ├── player_matcher.py        # Yahoo vs NFL player matching
│   └── yahoo_nfl_merge_v2.py    # Main merge orchestrator
│
└── utils/                  # Utility modules
    ├── run_metadata.py           # Structured JSON logging
    ├── cluster_rate_limiter.py   # Multi-process rate limiting
    ├── data_validators.py        # Data quality checks
    ├── parquet_utils.py          # Partitioned parquet support
    └── merge_utils.py            # DuckDB/Polars merge utilities
```

## Quick Start

### 1. Discover Available Leagues

```bash
# Discover all leagues you have access to
python multi_league/core/league_discovery.py discover --oauth path/to/Oauth.json --year 2024

# Interactive registration
python multi_league/core/league_discovery.py register --oauth path/to/Oauth.json
```

### 2. Create League Context

```bash
# Create context for a specific league
python multi_league/core/league_discovery.py create \
  --league-id nfl.l.123456 \
  --oauth path/to/Oauth.json \
  --start-year 2020
```

### 3. Fetch Yahoo Data

```python
from pathlib import Path
from multi_league.core.league_context import LeagueContext
from multi_league.data_fetchers.yahoo_fantasy_data_v2 import yahoo_fantasy_data

# Load context
ctx = LeagueContext.load("path/to/league_context.json")

# Fetch data
df = yahoo_fantasy_data(ctx, year=2024, week=5)
```

### 4. Fetch NFL Data

```python
from multi_league.data_fetchers.nfl_offense_stats_v2 import nfl_offense_stats

# Fetch NFL stats (league-independent data)
df = nfl_offense_stats(ctx, year=2024, week=5)
```

### 5. Merge Yahoo + NFL Data

```python
from multi_league.merge.yahoo_nfl_merge_v2 import yahoo_nfl_merge

# Merge data
merged_df = yahoo_nfl_merge(ctx, year=2024, week=5)
```

## Key Concepts

### LeagueContext

The `LeagueContext` dataclass encapsulates all league-specific configuration:

```python
from multi_league.core.league_context import LeagueContext

ctx = LeagueContext(
    league_id="nfl.l.123456",
    league_name="KMFFL",
    oauth_file_path="path/to/Oauth.json",
    start_year=2014,
    num_teams=10,
    manager_name_overrides={"--hidden--": "Ilan"}
)

# Access league-specific paths
player_data_path = ctx.player_data_directory
matchup_data_path = ctx.matchup_data_directory
logs_path = ctx.logs_directory

# Save/load from JSON
ctx.save("league_context.json")
ctx2 = LeagueContext.load("league_context.json")
```

### Yahoo vs NFL Data Differences

The merge logic handles these key differences:

**Yahoo Data:**
- Contains ALL rostered players (including DNP/injured/bye)
- Players with 0 points are included
- Limited to rostered players only

**NFL Data:**
- Contains ONLY players who recorded stats
- No players with 0 stats (injured/bye players absent)
- Includes all players (rostered or not)

**Merge Strategy:**
1. Match Yahoo + NFL players (multi-layer matching)
2. Keep Yahoo-only rows (DNP/injured players)
3. Keep NFL-only rows (unrostered players with stats)
4. Compute points using league-specific scoring

### Multi-Layer Matching

The `player_matcher.py` module uses intelligent matching:

```
Layer 1: Exact match (name + position + year + week)
Layer 2: Last name match (last + position + year + week)
Layer 3: Name-only match (name + year + week) - only for players with points
```

Special logic:
- **Players WITH points (>0):** Looser matching (they played, should be in NFL data)
- **Players WITHOUT points (0):** Stricter matching (avoid false matches for DNP players)

## Module Details

### core/league_context.py

**Purpose:** Centralized league configuration

**Key Features:**
- League metadata (ID, name, teams)
- OAuth credentials path
- Data directory structure
- Processing settings (workers, rate limits, caching)
- Manager name overrides
- JSON serialization

**Example:**
```python
ctx = LeagueContext(
    league_id="nfl.l.123456",
    league_name="KMFFL",
    oauth_file_path="Oauth.json",
    start_year=2014,
    max_workers=3,
    rate_limit_per_sec=4.0
)
```

### core/league_discovery.py

**Purpose:** Auto-discover and register Yahoo leagues

**Key Features:**
- OAuth-based league discovery
- Fetch league metadata (teams, scoring, playoffs)
- Interactive registration CLI
- League registry management
- Batch context creation

**Example:**
```python
from multi_league.core.league_discovery import LeagueDiscovery

discovery = LeagueDiscovery(oauth_file=Path("Oauth.json"))
leagues = discovery.discover_leagues(year=2024)

for league in leagues:
    print(f"{league['league_name']}: {league['num_teams']} teams")
```

### data_fetchers/yahoo_fantasy_data_v2.py

**Purpose:** Fetch Yahoo player data for a league

**Key Changes from V1:**
- Accepts `LeagueContext` as first parameter
- League-specific output directories
- Cluster-aware rate limiting
- RunLogger integration
- Modularized functions

**Example:**
```python
from multi_league.data_fetchers.yahoo_fantasy_data_v2 import yahoo_fantasy_data

df = yahoo_fantasy_data(ctx, year=2024, week=5)
```

### data_fetchers/nfl_offense_stats_v2.py

**Purpose:** Fetch NFL offensive stats (league-independent)

**Key Features:**
- Fetches from nflverse data releases
- League-context aware for output paths
- Same data for all leagues (NFL stats are universal)
- RunLogger integration

**Example:**
```python
from multi_league.data_fetchers.nfl_offense_stats_v2 import nfl_offense_stats

df = nfl_offense_stats(ctx, year=2024, week=5)
```

### merge/yahoo_settings.py

**Purpose:** Fetch Yahoo league scoring settings

**Functions:**
- `fetch_yahoo_dst_scoring()` - Get DST scoring rules
- `load_saved_dst_scoring()` - Load cached DST scoring
- `load_saved_full_scoring()` - Load full offensive scoring rules

**Example:**
```python
from multi_league.merge.yahoo_settings import fetch_yahoo_dst_scoring

dst_scoring = fetch_yahoo_dst_scoring(
    year=2024,
    league_key="nfl.l.123456",
    oauth_file=Path("Oauth.json"),
    settings_dir=Path("settings")
)

# Returns:
# {
#     "Sack": 1.0,
#     "Interception": 2.0,
#     "Fumble Recovery": 2.0,
#     "Touchdown": 6.0,
#     "PA_0": 10.0,
#     ...
# }
```

### merge/points_calculator.py

**Purpose:** Compute fantasy points based on scoring rules

**Functions:**
- `compute_points_from_full_scoring()` - Use league-specific rules
- `compute_default_points()` - Fallback to half-PPR
- `compute_dst_points()` - DST/DEF scoring

**Example:**
```python
from multi_league.merge.points_calculator import compute_points_from_full_scoring

# With league scoring rules
points = row.apply(lambda r: compute_points_from_full_scoring(r, scoring_rules))
```

### merge/player_matcher.py

**Purpose:** Match Yahoo players to NFL players

**Key Functions:**
- `perform_multi_layer_matching()` - Multi-layer matching strategy
- `assemble_final_dataframe()` - Combine matched + unmatched data
- `with_keys()` - Add matching keys to DataFrames

**Matching Logic:**
```python
from multi_league.merge.player_matcher import perform_multi_layer_matching

layers, yahoo_unmatched, nfl_unmatched = perform_multi_layer_matching(
    yahoo_df,
    nfl_df,
    verbose=True
)

# Returns:
# - layers: List of matched DataFrames (one per matching layer)
# - yahoo_unmatched: Yahoo players not in NFL data (DNP/injured)
# - nfl_unmatched: NFL players not in Yahoo data (not rostered)
```

### merge/yahoo_nfl_merge_v2.py

**Purpose:** Main orchestrator for Yahoo + NFL merge

**Key Features:**
- Multi-league compliant
- Modular architecture (uses other merge modules)
- Handles Yahoo-only vs NFL-only players
- League-specific scoring
- RunLogger integration

**Example:**
```python
from multi_league.merge.yahoo_nfl_merge_v2 import yahoo_nfl_merge

merged_df = yahoo_nfl_merge(ctx, year=2024, week=5)

# Returns merged DataFrame with:
# - Matched players (Yahoo + NFL data)
# - Yahoo-only players (DNP/injured)
# - NFL-only players (not rostered)
```

### utils/run_metadata.py

**Purpose:** Structured JSON logging for debugging

**Example:**
```python
from multi_league.utils.run_metadata import RunLogger

with RunLogger("my_script", year=2024, week=5, league_id=ctx.league_id) as logger:
    logger.start_step("fetch_data")
    # ... do work ...
    logger.complete_step(rows_read=1000)

# Creates: data/logs/my_script_20251020_143052_y2024_w5.json
```

### utils/cluster_rate_limiter.py

**Purpose:** Multi-process rate limiting coordination

**Example:**
```python
from multi_league.utils.cluster_rate_limiter import ClusterRateLimiter

# Shared rate limit across multiple processes
limiter = ClusterRateLimiter(
    rate=4.0,
    shared_store="/tmp/rate_limit.json"
)

limiter.acquire()
# ... make API call ...
```

### utils/data_validators.py

**Purpose:** Automated data quality checks

**Example:**
```python
from multi_league.utils.data_validators import validate_all_data

results = validate_all_data(
    matchup_df=matchup_df,
    player_df=player_df
)

for table, errors in results.items():
    for error in errors:
        print(f"{error.severity}: {error.message}")
```

### utils/parquet_utils.py

**Purpose:** Partitioned parquet for 10-100x faster reads

**Example:**
```python
from multi_league.utils.parquet_utils import write_partitioned, read_partitioned

# Write partitioned by year
write_partitioned(df, "player.parquet", partition_cols=["year"])

# Read only 2024 data (skips all other years)
df_2024 = read_partitioned("player.parquet", filters=[("year", "==", 2024)])
```

### utils/merge_utils.py

**Purpose:** High-performance merges using DuckDB/Polars

**Example:**
```python
from multi_league.utils.merge_utils import duckdb_merge

# Merge two large parquet files without loading into memory
duckdb_merge(
    "yahoo_stats.parquet",
    "nfl_stats.parquet",
    on=["player_id", "year", "week"],
    output_path="merged.parquet"
)
```

## Backward Compatibility

All V2 scripts support standalone mode (without LeagueContext):

```python
# Without context (backward compatible)
from multi_league.data_fetchers.yahoo_fantasy_data_v2 import yahoo_fantasy_data

df = yahoo_fantasy_data(
    year=2024,
    week=5,
    oauth_file=Path("Oauth.json")
)
```

## CLI Usage

All modules support command-line usage:

```bash
# Discover leagues
python multi_league/core/league_discovery.py discover --oauth Oauth.json

# Fetch Yahoo data
python multi_league/data_fetchers/yahoo_fantasy_data_v2.py \
  --context leagues/kmffl/league_context.json \
  --year 2024 \
  --week 5

# Fetch NFL data
python multi_league/data_fetchers/nfl_offense_stats_v2.py \
  --year 2024 \
  --week 5

# Merge data
python multi_league/merge/yahoo_nfl_merge_v2.py \
  --context leagues/kmffl/league_context.json \
  --year 2024 \
  --week 5
```

## Testing

To test the multi-league infrastructure:

```bash
# 1. Discover leagues
python multi_league/core/league_discovery.py discover --oauth Oauth.json --year 2024

# 2. Register a test league
python multi_league/core/league_discovery.py create \
  --league-id nfl.l.123456 \
  --oauth Oauth.json \
  --start-year 2024

# 3. Fetch and merge data
python multi_league/merge/yahoo_nfl_merge_v2.py \
  --context ~/fantasy_football_data/nfl_l_123456/league_context.json \
  --year 2024 \
  --week 5
```

## Next Steps

See `MULTI_LEAGUE_SETUP_GUIDE.md` in the parent directory for the full implementation roadmap.
