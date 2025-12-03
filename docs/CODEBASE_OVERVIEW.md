# Yahoo Fantasy Football Codebase - Comprehensive Overview

## Executive Summary

This is a **multi-league fantasy football data pipeline** that:
1. Authenticates with Yahoo Fantasy Football API via OAuth
2. Fetches and normalizes league data (players, matchups, drafts, transactions, schedules)
3. Transforms and enriches data with advanced analytics
4. Uploads to MotherDuck for cloud-based SQL queries
5. Provides a Streamlit web UI for league selection and import

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  STREAMLIT WEB UI (main.py)                                 │
│  - OAuth authentication with Yahoo                          │
│  - League selection interface                               │
│  - Job monitoring & MotherDuck upload                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  PIPELINE ORCHESTRATOR (initial_import_v2.py)               │
│  - Manages 4-phase data import process                      │
│  - Runs fetchers, merges, transformations sequentially      │
│  - Uses LeagueContext for configuration                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    ┌─────────┐ ┌─────────┐ ┌──────────┐
    │ PHASE 0 │ │ PHASE 1 │ │ PHASE 2  │
    │Settings │ │ Fetchers│ │ Merges   │
    └─────────┘ └─────────┘ └──────────┘
         │         │         │
         └─────────┼─────────┘
                   ▼
            ┌──────────────┐
            │ PHASE 3      │
            │Transformations
            └──────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ OUTPUT PARQUET FILES         │
    │ - player.parquet             │
    │ - matchup.parquet            │
    │ - draft.parquet              │
    │ - transactions.parquet       │
    │ - schedule.parquet           │
    └──────────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ MOTHERDUCK UPLOAD            │
        │ (run_motherduck_upload.py)   │
        │ Creates database per league  │
        └──────────────────────────────┘
```

---

## Component Breakdown

### 1. Entry Points

**`main.py`** - Streamlit Web Application
- **Purpose**: User-facing dashboard for Yahoo authentication and league selection
- **Features**:
  - OAuth 2.0 login with Yahoo
  - League selection and season discovery
  - Triggers import pipeline
  - Monitors job status in MotherDuck
  - Displays upload results
- **Key Functions**:
  - `run_initial_import()`: Spawns subprocess for `initial_import_v2.py`
  - `upload_to_motherduck()`: Uploads parquet files to MotherDuck
  - `save_oauth_token()`: Saves OAuth credentials locally
- **Data Flow**: User browser → Yahoo OAuth → Streamlit Cloud → GitHub Actions

### 2. League Context System

**`league_context.py`** - Configuration Management
- **Purpose**: Encapsulates all configuration for a single league
- **Key Class**: `LeagueContext`
  - Stores: league_id, league_name, oauth_file_path, data_directory
  - Auto-creates directory structure
  - Provides year ranges and OAuth session management
- **Directory Structure Created**:
  ```
  {data_directory}/
    ├── league_context.json         # Configuration file (persisted)
    ├── league_settings/            # League metadata (all years)
    ├── player_data/               # Yahoo player stats
    ├── matchup_data/              # Weekly matchups
    ├── transaction_data/          # Trades and pickups
    ├── draft_data/                # Draft results
    ├── schedule_data/             # League schedules
    ├── logs/                      # Run logs (JSON)
    ├── cache/                     # Performance caching
    ├── player.parquet             # **CANONICAL OUTPUT**
    ├── matchup.parquet            # **CANONICAL OUTPUT**
    ├── transactions.parquet       # **CANONICAL OUTPUT**
    └── draft.parquet              # **CANONICAL OUTPUT**
  ```

**`league_discovery.py`** - League Registration
- **Purpose**: Discover and register Yahoo leagues
- **Key Class**: `LeagueDiscovery`
  - Discovers all leagues accessible via OAuth
  - Fetches metadata (teams, playoff config, scoring type)
  - Creates LeagueContext objects
- **Key Class**: `LeagueRegistry`
  - Maintains JSON registry of all configured leagues
  - Allows bulk operations across multiple leagues

### 3. Pipeline Orchestrator

**`initial_import_v2.py`** - Main Pipeline Script
- **Entry Point**: `python initial_import_v2.py --context league_context.json`
- **Phases**:
  
  **PHASE 0: League Settings Discovery**
  - Fetches league configuration from Yahoo API for all years
  - Saves to `league_settings/` directory
  - Parallelized (5 workers) for speed
  
  **PHASE 1: Data Fetchers** (if not skipped)
  - Runs 7 fetcher scripts sequentially:
    1. `weekly_matchup_data_v2.py` - Weekly matchups (all seasons)
    2. `yahoo_fantasy_data.py` - Yahoo player stats
    3. `nfl_offense_stats.py` - NFL offensive stats
    4. `defense_stats.py` - NFL defense stats
    5. `draft_data_v2.py` - Draft results (all years)
    6. `transactions_v2.py` - Trades/pickups (all years)
    7. `season_schedules.py` - League schedules
  
  **PHASE 2: Weekly Merges**
  - Merges Yahoo + NFL data per-week across all seasons
  - Normalizes multi-year files into weekly parquets
  - Outputs: `player.parquet` (canonical)
  
  **PHASE 3: Transformations** (13 scripts in dependency order)
  - Base calculations: cumulative stats, playoff flags
  - Cross-joins: matchup↔player, player↔transactions, draft↔player
  - Advanced analytics: expected records, playoff odds, keeper economics
  - Season aggregations: creates `players_by_year.parquet`

### 4. Data Fetchers

Located in `/fantasy_football_data_scripts/multi_league/data_fetchers/`

| Script | Output | Purpose |
|--------|--------|---------|
| `yahoo_fantasy_data.py` | `player_stats_*.parquet` | Yahoo player stats (all positions, all weeks) |
| `nfl_offense_stats.py` | `nfl_offense_*.parquet` | NFL offensive statistics from external sources |
| `defense_stats.py` | `nfl_dst_*.parquet` | Defense/Special Teams weekly scores |
| `draft_data_v2.py` | `draft_*.parquet` | Draft results with keeper info |
| `transactions_v2.py` | `transactions_*.parquet` | Trades, pickups, drops (all transactions) |
| `season_schedules.py` | `schedule_*.parquet` | League schedule (matchup pairings) |
| `weekly_matchup_data_v2.py` | `matchup_data_*.parquet` | Yahoo matchup results (scores, outcomes) |

### 5. Transformations

Located in `/fantasy_football_data_scripts/multi_league/transformations/`

**Base Transformations:**
- `cumulative_stats_v2.py` - Running win-loss records, points-for/against
- `aggregate_player_season_v2.py` - Season-level player summaries

**Enrichments:**
- `matchup_to_player_v2.py` - Attach matchup context to player stats
- `player_to_matchup_v2.py` - Aggregate player data into matchups
- `draft_to_player_v2.py` - Add draft info to player records
- `player_to_draft_v2.py` - Add player stats to draft records
- `draft_enrichment_v2.py` - Keeper economics calculations
- `player_to_transactions_v2.py` - Link players to transactions

**Advanced Analytics:**
- `expected_record_v2.py` - W-L if scheduled differently
- `playoff_odds_import.py` - Monte Carlo playoff simulations

### 6. Output Files (Canonical Parquets)

All parquet files are created in `{data_directory}/`

| File | Columns | Purpose |
|------|---------|---------|
| **player.parquet** | league_id, season, week, player_name, player_id, position, yahoo_team, yahoo_points, nfl_points, etc. | Complete player-week statistics with NFL data merged |
| **matchup.parquet** | league_id, season, week, home_team, away_team, home_manager, away_manager, home_points, away_points, won, etc. | Weekly matchup results with advanced stats |
| **draft.parquet** | league_id, season, draft_order, team, player_name, player_id, pick_round, keeper_eligible, etc. | Draft results with keeper tracking |
| **transactions.parquet** | league_id, season, timestamp, transaction_type, player_name, player_id, from_team, to_team, faab_spent, etc. | All trades, pickups, drops with FAAB tracking |
| **schedule_data_all_years.parquet** | league_id, season, week, home_team, away_team | League schedule structure |
| **players_by_year.parquet** | league_id, season, manager_name, games_played, total_points, avg_ppg, etc. | Season-level player aggregations |

---

## GitHub Workflows

### `.github/workflows/fantasy_import_worker.yml`
- **Trigger**: Manual `workflow_dispatch` or `repository_dispatch` from Streamlit
- **Job**: `process-import`
- **Steps**:
  1. Parse job data (league info, OAuth token)
  2. Save OAuth credentials to `oauth/Oauth.json`
  3. Run `initial_import_v2.py`
  4. Upload results to MotherDuck
  5. Update job status in MotherDuck
  6. Create GitHub issue on failure
- **Timeout**: 60 minutes
- **Artifacts**: Import logs, parquet files (7-day retention)

### `.github/workflows/import_to_md.yml`
- **Trigger**: Manual `workflow_dispatch`
- **Job**: Runs `scripts/run_worker.py`
- **Purpose**: Standalone worker for MotherDuck uploads

---

## League References Throughout Codebase

League is a first-class entity:

1. **Yahoo Fantasy API**
   - League discovery via `yahoo_fantasy_api.Game.league_ids()`
   - League settings via `League.settings()`
   - Team roster data fetched per-league
   - Transactions/draft queried per-league

2. **Data Structure**
   - Every parquet has `league_id` column for multi-league isolation
   - `LeagueContext` encapsulates league-specific config
   - Separate data directories per league

3. **Multi-League Support**
   - Pipeline can process multiple leagues in sequence
   - `LeagueRegistry` tracks multiple leagues
   - MotherDuck creates separate database per league: `{league_name}_{season}`

---

## Data Flow Example: A Complete Import

1. **User authenticates** in Streamlit
2. **User selects league** (e.g., "KMFFL", season 2024)
3. **Streamlit saves OAuth token** to `oauth/Oauth.json`
4. **Streamlit triggers import**:
   - Calls `initial_import_v2.py --context league_context.json`
5. **Phase 0**: Fetches all league settings for 2014-2024
6. **Phase 1**: Fetchers run in sequence:
   - `yahoo_fantasy_data.py` → `player_data/yahoo_player_stats_*.parquet`
   - `nfl_offense_stats.py` → `player_data/nfl_offense_stats_*.parquet`
   - etc.
7. **Phase 2**: Merge scripts run:
   - `yahoo_nfl_merge.py` → merges Yahoo + NFL per-week
   - Output: `player.parquet` (main player stats table)
8. **Phase 3**: 13 transformation scripts enrich data:
   - Add playoff flags, expected records, keeper tracking
   - Cross-join matchups, transactions, drafts
   - Create season summaries
9. **Outputs**: 6 canonical parquet files in `data_directory/`
10. **MotherDuck upload**:
    - Reads all `.parquet` files
    - Creates database `kmffl_2024`
    - Creates tables: `player`, `matchup`, `draft`, `transactions`, etc.
    - User can query: `SELECT * FROM kmffl_2024.public.matchup`

---

## MotherDuck Integration

**`run_motherduck_upload.py`** - Direct Upload Script
- Sets environment variables for league name and token
- Calls `fantasy_football_data/motherduck_upload.py`
- Creates database named: `{league_name}_{season}` (e.g., `kmffl_2024`)
- Uploads all parquet files as tables
- Creates views for analysis (e.g., `season_summary`)

**Database Naming**: Slugified league name + season (spaces/dashes removed)

**Tables Created**: One per parquet file (auto-detected)

---

## Configuration & Secrets

**Secrets** (from `.streamlit/secrets.toml` or environment):
- `YAHOO_CLIENT_ID` - OAuth app ID
- `YAHOO_CLIENT_SECRET` - OAuth app secret
- `MOTHERDUCK_TOKEN` - MotherDuck connection token

**OAuth File**: `oauth/Oauth.json`
- Created after user authenticates
- Contains: access_token, refresh_token, consumer_key, consumer_secret
- Per-league variant: `oauth/Oauth_{league_key}.json`

**League Context File**: `{data_directory}/league_context.json`
- Serialized `LeagueContext` dataclass
- Persisted for future runs
- Includes year range, team count, OAuth path

---

## Key Design Patterns

### Multi-League Isolation
- Each league has separate `data_directory`
- Every parquet includes `league_id` column
- Prevents data mixing across leagues

### Lazy Fetching
- League settings cached in JSON
- Data fetchers check for existing files
- Supports `--skip-fetchers` to reuse cached data

### Parallel Processing
- Phase 0: 5-worker thread pool for settings fetch
- Phase 1-3: Sequential (dependencies between scripts)

### Robust Output Validation
- `verify_unified_outputs()` checks canonical files exist
- League isolation validation (`validate_league_isolation()`)
- Column normalization and composite key generation

### Transformation Dependencies
- Ordered transformation list respects dependencies
- Example: `cumulative_stats_v2.py` must run before `expected_record_v2.py`

---

## Streamlit Architecture

**State Management**:
- `st.session_state` stores access_token, league_info, job_id
- Query params used for OAuth callback and import triggering

**Workflow**:
1. Landing page → "Connect Yahoo Account" button
2. OAuth callback → saves token, shows league selection
3. League selection → shows matchups, "Start Import" button
4. Import submission → calls `run_initial_import()` (subprocess)
5. Live log updates → streams stdout from subprocess
6. Completion → prompts MotherDuck upload

**MotherDuck Integration**:
- Stores job status in `ops.import_status` table
- Stores OAuth tokens in `secrets.yahoo_oauth_tokens` table
- Streamlit polls for job status via `get_job_status()`

---

## File Structure Summary

```
/home/user/yahoo_oauth/
├── main.py                              # Streamlit entry point
├── scripts/run_motherduck_upload.py             # Direct MD upload
├── fantasy_football_data/               # Output data directory
│   ├── {league_id}/                     # Per-league subdirectory
│   │   ├── league_context.json          # League config
│   │   ├── player.parquet               # **Output**
│   │   ├── matchup.parquet              # **Output**
│   │   ├── draft.parquet                # **Output**
│   │   ├── transactions.parquet         # **Output**
│   │   └── ...more subdirectories
│   └── motherduck_upload.py             # MD upload implementation
├── fantasy_football_data_scripts/
│   ├── initial_import_v2.py             # **MAIN ORCHESTRATOR**
│   └── multi_league/
│       ├── core/
│       │   ├── league_context.py        # League config class
│       │   ├── league_discovery.py      # League discovery
│       │   ├── yahoo_league_settings.py # Settings fetcher
│       │   ├── script_runner.py         # Script execution helpers
│       │   └── data_normalization.py    # Output validation
│       ├── data_fetchers/               # 7 fetcher scripts
│       │   ├── yahoo_fantasy_data.py
│       │   ├── nfl_offense_stats.py
│       │   ├── defense_stats.py
│       │   ├── draft_data_v2.py
│       │   ├── transactions_v2.py
│       │   ├── season_schedules.py
│       │   ├── weekly_matchup_data_v2.py
│       │   └── aggregators.py           # Output normalization
│       └── transformations/             # 13+ transformation scripts
│           ├── base/                    # Cumulative stats
│           ├── draft_enrichment/        # Draft analytics
│           ├── matchup_enrichment/      # Matchup analytics
│           ├── player_enrichment/       # Player analytics
│           └── transaction_enrichment/  # Transaction analytics
├── oauth/                               # OAuth credentials (local)
│   ├── Oauth.json                       # Global token
│   └── Oauth_{league_key}.json          # Per-league tokens
├── .github/workflows/
│   ├── fantasy_import_worker.yml        # Main pipeline workflow
│   └── import_to_md.yml                 # Standalone MD upload
├── .streamlit/
│   ├── secrets.toml                     # Local secrets
│   └── secrets.toml.template            # Template
└── scripts/
    └── run_worker.py                    # Job worker implementation
```

---

## Summary: How It All Connects

1. **Streamlit** (main.py) - User interface for league selection
2. **League Discovery** - Find available leagues via Yahoo OAuth
3. **League Context** - Encapsulate league-specific configuration
4. **initial_import_v2.py** - Orchestrate 4-phase import:
   - Phase 0: Fetch league settings
   - Phase 1: Run 7 data fetchers
   - Phase 2: Merge Yahoo + NFL data
   - Phase 3: Run 13 transformations
5. **Output** - 6 canonical parquet files (player, matchup, draft, transactions, schedule, players_by_year)
6. **MotherDuck** - Upload parquets to cloud SQL database
7. **GitHub Actions** - Run pipeline as workflow (optional)

All wrapped in a **multi-league** system where each league gets isolated data, configuration, and database.

