# Quick Reference Guide

## Main Entry Points

| File | Purpose | Command |
|------|---------|---------|
| `main.py` | Streamlit web UI | `streamlit run main.py` |
| `initial_import_v2.py` | Data pipeline orchestrator | `python initial_import_v2.py --context {path}` |
| `run_motherduck_upload.py` | Direct MotherDuck upload | `python run_motherduck_upload.py` |

---

## Key Absolute File Paths

```
/home/user/yahoo_oauth/

# Streamlit Entry Point
main.py                                        # Web UI (OAuth, league selection, job monitoring)

# Pipeline Scripts
fantasy_football_data_scripts/
  initial_import_v2.py                         # **MAIN ORCHESTRATOR** (4-phase import)
  
# Core Libraries
fantasy_football_data_scripts/multi_league/core/
  league_context.py                            # LeagueContext class (league config)
  league_discovery.py                          # LeagueDiscovery & LeagueRegistry (find/register leagues)
  yahoo_league_settings.py                     # Fetch league settings from Yahoo API
  script_runner.py                             # Execute child scripts safely
  data_normalization.py                        # Output validation & normalization

# Data Fetchers (7 scripts)
fantasy_football_data_scripts/multi_league/data_fetchers/
  weekly_matchup_data_v2.py                    # Weekly matchup results from Yahoo
  yahoo_fantasy_data.py                        # Player stats from Yahoo
  nfl_offense_stats.py                         # External NFL offense stats
  defense_stats.py                             # NFL DST stats
  draft_data_v2.py                             # Draft results & keeper info
  transactions_v2.py                           # Trades, pickups, drops
  season_schedules.py                          # League schedule structure
  aggregators.py                               # Output normalization utilities

# Transformations (13+ scripts)
fantasy_football_data_scripts/multi_league/transformations/
  base/
    cumulative_stats_v2.py                     # Win-loss, points-for/against tracking
    aggregate_player_season_v2.py              # Season-level player summaries
  draft_enrichment/
    draft_enrichment_v2.py                     # Keeper economics
    keeper_economics_v2.py                     # Keeper value analysis
    player_to_draft_v2.py                      # Add player stats to draft
  matchup_enrichment/
    player_to_matchup_v2.py                    # Aggregate player data into matchups
    expected_record_v2.py                      # W-L if scheduled differently
    playoff_odds_import.py                     # Monte Carlo simulations
  player_enrichment/
    matchup_to_player_v2.py                    # Attach matchup context to players
    draft_to_player_v2.py                      # Add draft info to players
  transaction_enrichment/
    player_to_transactions_v2.py               # Link players to transactions

# Output Data
fantasy_football_data/                         # Root data directory
  {league_id}/                                 # Per-league subdirectory
    league_context.json                        # League configuration (persisted)
    league_settings/                           # Yahoo settings per year
    player_data/                               # Player stats (per-week parquets)
    matchup_data/                              # Matchup results (per-week)
    draft_data/                                # Draft results (per-year)
    transaction_data/                          # Transactions (per-year)
    schedule_data/                             # Schedule structure
    logs/                                      # JSON run logs
    cache/                                     # API cache for performance
    player.parquet                             # **CANONICAL** Player stats
    matchup.parquet                            # **CANONICAL** Matchup results
    draft.parquet                              # **CANONICAL** Draft data
    transactions.parquet                       # **CANONICAL** Transactions
    schedule_data_all_years.parquet            # All-years schedule
    players_by_year.parquet                    # Season aggregations

# OAuth Credentials
oauth/
  Oauth.json                                   # Global OAuth token
  Oauth_{league_key}.json                      # Per-league OAuth token

# GitHub Workflows
.github/workflows/
  fantasy_import_worker.yml                    # Main pipeline workflow
  import_to_md.yml                             # Standalone MotherDuck upload

# Streamlit Configuration
.streamlit/
  secrets.toml                                 # Local secrets (not committed)
  secrets.toml.template                        # Template for secrets
```

---

## Key Classes & Methods

### LeagueContext
**File**: `fantasy_football_data_scripts/multi_league/core/league_context.py`

```python
from league_context import LeagueContext

# Create
ctx = LeagueContext(
    league_id="nfl.l.123456",
    league_name="KMFFL",
    oauth_file_path="oauth/Oauth.json",
    start_year=2014
)

# Properties
ctx.league_id                        # Yahoo league key
ctx.league_name                      # Display name
ctx.data_directory                   # Root data dir
ctx.player_data_directory            # Player stats dir
ctx.matchup_data_directory           # Matchup dir
ctx.canonical_player_file            # player.parquet path
ctx.canonical_matchup_file           # matchup.parquet path
ctx.canonical_draft_file             # draft.parquet path
ctx.canonical_transaction_file       # transactions.parquet path
ctx.get_year_range()                 # range(start_year, end_year+1)
ctx.get_yahoo_years()                # Years to fetch from Yahoo

# Methods
ctx.save()                           # Save to disk
ctx.load(path)                       # Load from disk
ctx.get_oauth_session()              # Create Yahoo OAuth session
ctx.summary()                        # Print summary
```

### LeagueDiscovery
**File**: `fantasy_football_data_scripts/multi_league/core/league_discovery.py`

```python
from league_discovery import LeagueDiscovery

discovery = LeagueDiscovery(oauth_file=Path("Oauth.json"))

# Methods
discovery.discover_leagues(year=2024)             # Find all leagues
discovery.create_league_context(league_id, ...)  # Create context
discovery.interactive_register_leagues()          # Interactive CLI
```

### LeagueRegistry
**File**: `fantasy_football_data_scripts/multi_league/core/league_discovery.py`

```python
from league_discovery import LeagueRegistry

registry = LeagueRegistry()

# Methods
registry.register_league(ctx)        # Add to registry
registry.list_leagues()              # List all leagues
registry.load_contexts()             # Load all contexts
registry.summary()                   # Print summary
```

---

## Common Commands

### Run Streamlit
```bash
streamlit run main.py
```

### Run Import for Specific League
```bash
python fantasy_football_data_scripts/initial_import_v2.py \
  --context fantasy_football_data/{league_id}/league_context.json
```

### Skip Data Fetchers (use cached data)
```bash
python fantasy_football_data_scripts/initial_import_v2.py \
  --context {path} \
  --skip-fetchers
```

### Dry Run (no file writes)
```bash
python fantasy_football_data_scripts/initial_import_v2.py \
  --context {path} \
  --dry-run
```

### List All Registered Leagues
```bash
python fantasy_football_data_scripts/multi_league/core/league_discovery.py list
```

### Discover Leagues Interactively
```bash
python fantasy_football_data_scripts/multi_league/core/league_discovery.py register \
  --oauth oauth/Oauth.json
```

---

## Configuration Files

### secrets.toml
```toml
YAHOO_CLIENT_ID = "..."
YAHOO_CLIENT_SECRET = "..."
MOTHERDUCK_TOKEN = "..."
REDIRECT_URI = "https://leaguehistory.streamlit.app"
```

### league_context.json
```json
{
  "league_id": "nfl.l.123456",
  "league_name": "KMFFL",
  "oauth_file_path": "oauth/Oauth.json",
  "data_directory": "/path/to/fantasy_football_data/nfl_l_123456",
  "start_year": 2014,
  "end_year": null,
  "num_teams": 10,
  "manager_name_overrides": {
    "--hidden--": "Ilan"
  },
  "created_at": "2024-11-15T...",
  "updated_at": "2024-11-15T..."
}
```

### OAuth token (Oauth.json)
```json
{
  "access_token": "...",
  "refresh_token": "...",
  "consumer_key": "...",
  "consumer_secret": "...",
  "token_type": "bearer",
  "expires_in": 3600,
  "guid": "..."
}
```

---

## Output Files Created

### Per-League Canonical Parquets
| File | Columns | Rows |
|------|---------|------|
| `player.parquet` | league_id, season, week, player_id, player_name, position, points, ... | ~1000s per league |
| `matchup.parquet` | league_id, season, week, home_team, away_team, home_points, away_points, ... | 200-500 per league |
| `draft.parquet` | league_id, season, draft_order, team, player_id, player_name, round, ... | 100-200 per league |
| `transactions.parquet` | league_id, season, timestamp, type, player_id, from_team, to_team, ... | Variable per league |
| `schedule_data_all_years.parquet` | league_id, season, week, home_team, away_team | ~1000-2000 per league |
| `players_by_year.parquet` | league_id, season, manager_name, games, points, ppg, ... | ~100-200 per league |

### Intermediate Files (per-year, per-type)
```
player_data/
  yahoo_player_stats_2024_week_1.parquet
  yahoo_player_stats_2024_week_2.parquet
  nfl_offense_stats_2024_week_1.parquet
  nfl_dst_2024_week_1.parquet
  ...

matchup_data/
  matchup_data_2024_week_1.parquet
  ...

draft_data/
  draft_2024.parquet
  ...

transaction_data/
  transactions_2024.parquet
  ...
```

---

## Environment Variables

Set automatically by Streamlit or pipeline:

| Variable | Set By | Used By | Example |
|----------|--------|---------|---------|
| `YAHOO_CLIENT_ID` | secrets.toml | main.py | OAuth app ID |
| `YAHOO_CLIENT_SECRET` | secrets.toml | main.py | OAuth app secret |
| `MOTHERDUCK_TOKEN` | secrets.toml | motherduck_upload.py | MD connection |
| `REDIRECT_URI` | secrets.toml | main.py | OAuth callback URL |
| `OAUTH_PATH` | main.py | initial_import_v2.py | Path to Oauth.json |
| `EXPORT_DATA_DIR` | main.py | Fetchers | Data directory |
| `LEAGUE_NAME` | main.py | run_motherduck_upload.py | League name |
| `LEAGUE_KEY` | main.py | Fetchers | Yahoo league_key |
| `LEAGUE_SEASON` | main.py | Fetchers | Season year |
| `AUTO_CONFIRM` | main.py | Fetchers | Skip prompts (1) |

---

## Database Naming in MotherDuck

```python
# Database names are created from league name + season
league_name = "KMFFL"
season = 2024
db_name = f"{league_name}_{season}".lower().replace(' ', '_').replace('-', '_')
# Result: "kmffl_2024"

# Tables are auto-created from parquet files:
# - player.parquet → public.player
# - matchup.parquet → public.matchup
# - draft.parquet → public.draft
# - transactions.parquet → public.transactions
# - schedule_data_all_years.parquet → public.schedule_data_all_years
# - players_by_year.parquet → public.players_by_year
```

---

## GitHub Actions

### Manual Trigger Fantasy Import Worker
```bash
gh workflow run fantasy_import_worker.yml \
  -f job_data='{"league_key":"nfl.l.123456",...}' \
  -f job_id='unique-id'
```

### View Workflow Runs
```bash
gh run list --workflow=fantasy_import_worker.yml
gh run view {run-id} --log
```

---

## Debugging Tips

### Check League Context Valid
```bash
python -c "
from fantasy_football_data_scripts.multi_league.core.league_context import LeagueContext
ctx = LeagueContext.load('fantasy_football_data/nfl_l_123456/league_context.json')
print(ctx.summary())
"
```

### List Files Created
```bash
find fantasy_football_data/{league_id} -name "*.parquet" | sort
```

### Check Parquet Columns
```bash
duckdb -c "SELECT * FROM read_parquet('fantasy_football_data/{league_id}/player.parquet') LIMIT 1"
```

### Run Single Transformation
```bash
python fantasy_football_data_scripts/multi_league/transformations/base/cumulative_stats_v2.py \
  --context {path-to-league-context.json}
```

### View Import Logs
```bash
find fantasy_football_data/{league_id}/logs -name "*.json" | sort -r | head -1 | xargs cat
```

---

## Directory Permissions

Ensure these are writable by the pipeline:
```bash
chmod -R 755 oauth/
chmod -R 755 fantasy_football_data/
chmod -R 755 fantasy_football_data_scripts/
```

---

## Next Steps

1. **Understand League System**: Read `LEAGUE_SYSTEM.md`
2. **Full Overview**: Read `CODEBASE_OVERVIEW.md`
3. **Run Import**: `streamlit run main.py`
4. **Query Results**: Use MotherDuck SQL UI or CLI

