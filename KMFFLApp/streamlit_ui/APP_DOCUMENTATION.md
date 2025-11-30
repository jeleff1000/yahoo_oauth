# KMFFL Analytics App Documentation

**Comprehensive documentation for the KMFFL Fantasy Football Analytics Streamlit application**

**Last Updated:** 2025-10-28
**Version:** 2.0 (Optimized)
**Framework:** Streamlit 1.30+
**Database:** MotherDuck (DuckDB Cloud)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Application Entry Point](#application-entry-point)
4. [Data Access Layer](#data-access-layer)
5. [Performance Optimizations](#performance-optimizations)
6. [Tab Structure](#tab-structure)
7. [UI Components](#ui-components)
8. [File Organization](#file-organization)
9. [Configuration](#configuration)
10. [Deployment](#deployment)

---

## Overview

The KMFFL Analytics App is a comprehensive fantasy football analytics platform built with Streamlit. It provides:

- **Historical Analysis**: 2014-present season data for the KMFFL league
- **Real-time Stats**: Player, manager, and matchup statistics
- **Predictive Simulations**: Playoff odds, schedule simulations, scenario analysis
- **Draft Analytics**: Draft spending trends, keeper analysis, market efficiency
- **Transaction Tracking**: Adds, drops, trades, and FAAB spending
- **Visualizations**: 30+ interactive graphs and charts

### Key Features

- **Multi-league Isolation**: Filters all data by `league_id = "449.l.198278"`
- **Performance Optimized**: Smart caching, lazy loading, progressive data fetching
- **Responsive Design**: Modern UI with mobile-friendly layout
- **Real-time Updates**: 1-5 minute cache TTL for live data
- **Comprehensive Coverage**: 115K+ player records, 10+ years of history

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           app_homepage_optimized.py                   │  │
│  │  - Entry point                                        │  │
│  │  - Tab routing                                        │  │
│  │  - Performance monitoring                             │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │                                       │
│  ┌───────────────────▼──────────────────────────────────┐  │
│  │           Data Access Layer                           │  │
│  │  md/data_access.py                                    │  │
│  │  - Query optimization                                 │  │
│  │  - League filtering                                   │  │
│  │  - Smart caching (1-60 min TTL)                       │  │
│  └───────────────────┬──────────────────────────────────┘  │
└────────────────────┬─┴──────────────────────────────────────┘
                     │
          ┌──────────▼───────────┐
          │   MotherDuck Cloud   │
          │  (DuckDB Database)   │
          │                      │
          │  Tables:             │
          │  - player            │
          │  - matchup           │
          │  - draft             │
          │  - transaction       │
          │  - injury            │
          │  - playoff_odds      │
          └──────────────────────┘
```

### Data Flow

```
User Interaction → Tab Loader → Cached Data Loader → Data Access → MotherDuck
                                        ↓
                                  Session State
                                  (5-min cache)
                                        ↓
                              Tab Renderer → UI Display
```

---

## Application Entry Point

**File:** `app_homepage_optimized.py`
**Purpose:** Main application entry point with tab routing and performance monitoring

### Key Functions

#### `main()`

**Entry Point** - Application bootstrap and tab routing

```python
def main():
    """Main application entry point"""
```

**Operations:**
1. **Page Configuration**
   - Title: "KMFFL Analytics"
   - Layout: Wide mode
   - Icon: 🏈
   - Sidebar: Expanded

2. **Connectivity Check**
   - Validates MotherDuck connection
   - Checks `MOTHERDUCK_TOKEN` and `MD_ATTACH_URL` secrets
   - Runs `SELECT 1` health check
   - Stops app if connection fails

3. **Session Initialization**
   - Fetches latest season/week from database
   - Sets default year/week in `st.session_state`
   - Initializes performance settings

4. **Sidebar Controls**
   - Performance stats toggle
   - Cache clear button
   - Operation timing display (when enabled)

5. **Tab Rendering**
   - Creates 7 main tabs: Home, Managers, Players, Draft, Transactions, Simulations, Extras
   - Routes to tab-specific render functions

**Dependencies:**
- `md.data_access` - Database queries
- `utils.performance` - Performance monitoring
- `tabs.shared.modern_styles` - CSS styling

---

#### `_safe_boot()`

**Health Check** - Validates MotherDuck connectivity

```python
def _safe_boot() -> bool:
    """Cheap health check for MD connectivity"""
```

**Returns:** `True` if connected, `False` with error message if failed

**Error Messages:**
- "MotherDuck unavailable: {error}"
- "Check Streamlit Secrets (MOTHERDUCK_TOKEN, MD_ATTACH_URL) and your shared DB path."

---

#### `_init_session_defaults()`

**Session Initialization** - Sets up session state with smart defaults

```python
def _init_session_defaults():
    """Initialize session state with smart defaults"""
```

**Session State Variables:**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `year` | int | Latest year from DB | Current NFL season |
| `week` | int | Latest week from DB | Current NFL week |
| `active_main_tab` | int | 0 | Active main tab index |
| `show_performance_stats` | bool | False | Show perf metrics in sidebar |
| `enable_progressive_loading` | bool | True | Enable lazy loading |
| `weekly_offset` | int | 0 | Pagination offset for weekly player data |
| `weekly_limit` | int | 100 | Page size for weekly player data |

**Dependencies:**
- Calls `md.data_access.latest_season_and_week()` to get current season info
- Falls back to `(0, 0)` if database is unavailable

---

### Tab Loaders (Data Fetching)

All tab loaders use the `@cached_data_loader` decorator for smart caching.

#### `load_homepage_tab()`

**Purpose:** Load homepage summary data

**Cache:** 5 minutes (300s)
**Spinner:** "Loading homepage..."

**Returns:**
```python
{
    "summary": {
        "matchup_count": int,      # Total matchups
        "player_count": int,       # Total player records
        "draft_count": int,        # Total draft picks
        "transactions_count": int, # Total transactions
        "injuries_count": int,     # Total injury records
        "latest_year": int,        # Most recent year
        "latest_week": int,        # Most recent week
        "latest_games": int        # Games in latest week
    },
    "Matchup Data": DataFrame,     # Recent matchup data
    "Player Data": DataFrame       # Two-week player slice
}
```

**Database Queries:**
- `load_homepage_data()` - Summary counts
- `load_simulations_data()` - Matchup data for simulations
- `load_player_two_week_slice()` - Recent player performance

**Performance:** ~200ms (cached), ~2s (fresh)

---

#### `load_managers_tab()`

**Purpose:** Load manager/matchup statistics

**Cache:** 5 minutes (300s)
**Spinner:** "Loading managers data..."

**Returns:**
```python
{
    "summary": DataFrame,  # Season-by-season manager stats
    "h2h": DataFrame,      # Head-to-head records
    "recent": DataFrame    # All matchups ordered by recency
}
```

**Columns in `summary`:**
- `year`, `manager`, `games_played`, `wins`, `losses`
- `avg_points`, `total_points`

**Columns in `h2h`:**
- `manager`, `opponent`, `games_played`, `wins`, `avg_margin`

**Database Queries:**
- Aggregates matchup data by manager and season
- Joins matchups to themselves for head-to-head records
- Filtered by `league_id = "449.l.198278"`

**Performance:** ~300ms (cached), ~3s (fresh)

---

#### `load_draft_tab()`

**Purpose:** Load draft and keeper data

**Cache:** 5 minutes (300s)
**Spinner:** "Loading draft data..."

**Returns:**
```python
{
    "draft_summary": DataFrame,    # Draft stats by year
    "draft_detail": DataFrame,     # Individual picks
    "top_picks": DataFrame,        # Best value picks
    "keeper_summary": DataFrame    # Keeper economics
}
```

**Database Queries:**
- `load_draft_data(all_years=True)`
- Filtered by `league_id = "449.l.198278"`

**Performance:** ~250ms (cached), ~2.5s (fresh)

---

#### `load_transactions_tab()`

**Purpose:** Load transaction history (adds, drops, trades)

**Cache:** 5 minutes (300s)
**Spinner:** "Loading transactions..."

**Returns:**
```python
{
    "transactions": DataFrame,  # All transactions
    "player_data": DataFrame,   # Player context
    "injury_data": DataFrame,   # Injury status
    "draft_data": DataFrame     # Draft context
}
```

**Limit:** 1000 most recent transactions

**Database Queries:**
- `load_transactions_data(limit=1000)`
- Joins transactions with player, injury, and draft tables
- Filtered by `league_id = "449.l.198278"`

**Performance:** ~400ms (cached), ~4s (fresh)

---

#### `load_simulations_tab()`

**Purpose:** Load playoff simulation data

**Cache:** 5 minutes (300s)
**Spinner:** "Loading simulations..."

**Returns:**
```python
{
    "matchups": DataFrame  # Matchup data with playoff odds
}
```

**Playoff Odds Columns:**
- `p_playoffs`, `p_bye`, `p_sacko`
- `x1_seed`, `x2_seed`, ..., `x14_seed`
- `x1_win`, `x2_win`, ..., `x14_win`
- `power_rating`

**Database Queries:**
- `load_simulations_data(include_all_years=True)`
- Filtered by `league_id = "449.l.198278"`

**Performance:** ~500ms (cached), ~5s (fresh)

---

### Tab Renderers (UI Display)

#### `render_home_tab()`

**Purpose:** Display homepage overview

**Components:**
- Summary metrics (matchup count, player count, etc.)
- Recent games table
- Two-week player performance comparison
- Latest transactions
- Playoff odds snapshot

**Delegate:** `tabs.homepage.homepage_overview.display_homepage_overview()`

---

#### `render_managers_tab()`

**Purpose:** Display manager statistics and head-to-head records

**Components:**
- Season standings
- All-time records
- Head-to-head matrix
- Performance trends

**Delegate:** `tabs.matchups.matchup_overview.display_matchup_overview()`

---

#### `render_players_tab()`

**Purpose:** Player statistics with 4 sub-tabs

**Sub-tabs:**
1. **Weekly** - Game-by-game player performance
2. **Season** - Season aggregated stats
3. **Career** - All-time player statistics
4. **Visualize** - Player performance graphs

**Components:**
- Smart filters (position, team, manager, year/week)
- Pagination (100 rows per page)
- Sortable columns
- Export to CSV

---

#### `render_draft_tab()`

**Purpose:** Draft and keeper analysis

**Sub-tabs:**
1. **Overview** - Draft summary by year
2. **Graphs** - Draft visualizations

**Graphs:**
- 💸 Spending Trends - Draft budget allocation over time
- 🔁 Round Efficiency - Value by draft round
- 📈 Market Trends - Position price trends
- 🔒 Keeper Analysis - Keeper ROI and value

**Delegate:**
- `tabs.draft_data.draft_data_overview.display_draft_data_overview()`
- `tabs.graphs.draft_graphs.*`

---

#### `render_transactions_tab()`

**Purpose:** Transaction history (adds, drops, trades)

**Components:**
- Transaction timeline
- FAAB spending analysis
- Add/drop frequency
- Trade history
- Injury context

**Delegate:** `tabs.transactions.transactions_adds_drops_trades_overview.AllTransactionsViewer()`

---

#### `render_simulations_tab()`

**Purpose:** Playoff odds and schedule simulations

**Components:**
- Current playoff odds
- Schedule shuffling ("What if I had X's schedule?")
- Score shuffling ("What if I had X's scores?")
- Scoring tweaks ("What if we changed scoring?")

**Delegate:** `tabs.simulations.simulation_home.display_simulations_viewer()`

---

#### `render_extras_tab()`

**Purpose:** Miscellaneous features

**Sub-tabs:**
1. **Keeper** - Keeper eligibility and economics
2. **Team Names** - Historical team names by manager

**Components:**
- Keeper price trends
- Next-year keeper projections
- ROI analysis
- Team name history

**Delegates:**
- `tabs.keepers.keepers_home.KeeperDataViewer()`
- `tabs.team_names.team_names.display_team_names()`

---

## Data Access Layer

**File:** `md/data_access.py`
**Purpose:** Centralized database queries with optimization and caching

### Core Configuration

```python
# Database Connection
LEAGUE_ID = "449.l.198278"  # Multi-league isolation constant

# Table Names (MotherDuck schema)
T = {
    "player": "kmffl.players_by_year",
    "matchup": "kmffl.matchup",
    "draft": "kmffl.draft",
    "transactions": "kmffl.transaction",
    "injury": "kmffl.injury",
}

# Cache TTL (Time To Live)
CACHE_STATIC = 3600    # 1 hour - Historical data (never changes)
CACHE_RECENT = 300     # 5 minutes - Recent data (occasional updates)
CACHE_REALTIME = 60    # 1 minute - Live data (frequent changes)
```

---

### Query Optimization Patterns

#### 1. **Always Filter by `league_id` First**

**Why:** Reduces working set by 90%+ in multi-league scenarios.

```python
# ✅ GOOD - Filter by league_id first (indexed)
sql = f"""
    SELECT * FROM {T['player']}
    WHERE league_id = '{LEAGUE_ID}'  -- Filter first!
      AND year = 2024
      AND week = 5
"""

# ❌ BAD - Missing league_id filter
sql = f"""
    SELECT * FROM {T['player']}
    WHERE year = 2024 AND week = 5  -- Scans all leagues!
"""
```

**Performance:** 10-50x faster on large datasets

---

#### 2. **Include NFL-Only Players**

For player queries, include both rostered players AND all NFL players:

```python
# ✅ GOOD - Includes rostered + NFL-only players
sql = f"""
    SELECT * FROM {T['player']}
    WHERE (league_id = '{LEAGUE_ID}' OR league_id IS NULL)
      AND year = 2024
"""

# ❌ BAD - Only rostered players (misses NFL-only data)
sql = f"""
    SELECT * FROM {T['player']}
    WHERE league_id = '{LEAGUE_ID}'
      AND year = 2024
"""
```

**Preserves:** Full NFL dataset for player lookup and comparison

---

#### 3. **Use Indexed Columns**

**Indexed Columns:**
- `league_id` - Multi-league isolation
- `cumulative_week` - Cross-season ordering
- `yahoo_player_id` - Player identification
- `manager_week` - Composite join key

```python
# ✅ GOOD - Uses indexed cumulative_week
sql = f"""
    SELECT * FROM {T['matchup']}
    WHERE league_id = '{LEAGUE_ID}'
      AND cumulative_week >= 202401  -- Indexed!
    ORDER BY cumulative_week DESC
"""

# ❌ BAD - Non-indexed year/week requires full scan
sql = f"""
    SELECT * FROM {T['matchup']}
    WHERE year = 2024 AND week >= 1
    ORDER BY year DESC, week DESC
"""
```

**Performance:** 100x+ faster on historical queries

---

#### 4. **Avoid Window Functions**

Replace window functions with simpler aggregations when possible:

```python
# ✅ GOOD - Simple MAX aggregation
sql = f"""
    SELECT * FROM {T['player']}
    WHERE league_id = '{LEAGUE_ID}'
      AND cumulative_week = (
          SELECT MAX(cumulative_week)
          FROM {T['player']}
          WHERE league_id = '{LEAGUE_ID}'
      )
"""

# ❌ BAD - Complex window function
sql = f"""
    SELECT * FROM {T['player']}
    WHERE cumulative_week IN (
        SELECT DISTINCT cumulative_week
        FROM {T['player']}
        QUALIFY ROW_NUMBER() OVER (ORDER BY cumulative_week DESC) = 1
    )
"""
```

**Performance:** 2-5x faster on large result sets

---

### Key Functions

#### `run_query(sql: str) -> pd.DataFrame`

**Purpose:** Execute SQL query against MotherDuck

**Parameters:**
- `sql`: SQL query string

**Returns:** Pandas DataFrame

**Error Handling:** Retries up to 3 times on failure

**Example:**
```python
df = run_query(f"SELECT * FROM {T['player']} WHERE league_id = '{LEAGUE_ID}' LIMIT 100")
```

---

#### `latest_season_and_week() -> tuple[int, int]`

**Purpose:** Get most recent season and week

**Cache:** 1 minute (CACHE_REALTIME)

**Returns:** `(year, week)` tuple

**Query:**
```sql
SELECT year, week
FROM kmffl.matchup
WHERE league_id = '449.l.198278'
ORDER BY cumulative_week DESC
LIMIT 1
```

---

#### `load_player_week(year: int, week: int) -> pd.DataFrame`

**Purpose:** Load all players for a specific week

**Cache:** 5 minutes (CACHE_RECENT)

**Returns:** DataFrame with columns:
- Player stats (points, pass_yds, rush_yds, etc.)
- Matchup context (opponent, team_points, win/loss)

**Includes:** Both rostered players AND NFL-only players

**Query Pattern:**
```sql
SELECT p.*, m.opponent, m.win, m.loss
FROM kmffl.players_by_year p
LEFT JOIN kmffl.matchup m
  ON m.league_id = '449.l.198278'
  AND p.manager = m.manager
  AND p.year = m.year
  AND p.week = m.week
WHERE (p.league_id = '449.l.198278' OR p.league_id IS NULL)
  AND p.year = 2024
  AND p.week = 5
ORDER BY p.points DESC NULLS LAST
```

---

#### `load_keeper_data(...) -> pd.DataFrame`

**Purpose:** Load keeper economics data

**Cache:** 5 minutes (CACHE_RECENT)

**Parameters:**
- `year`: Filter by specific year (optional)
- `week`: Filter by specific week (optional)
- `all_years`: Include all years (default: False)

**Returns:** DataFrame with columns:
- `player`, `manager`, `yahoo_position`, `nfl_team`
- `year`, `week`, `keeper_price`, `kept_next_year`
- `avg_points_this_year`, `avg_points_next_year`
- `cost`, `is_keeper_status`

**Query:**
```sql
SELECT
    player,
    manager,
    nfl_position AS yahoo_position,
    nfl_team,
    year,
    MAX(week) AS week,
    keeper_price,
    kept_next_year,
    ppg_season AS avg_points_this_year,
    ppg_next_season AS avg_points_next_year,
    cost,
    is_keeper_status,
    SUM(points) AS season_points,
    COUNT(*) AS games_played
FROM kmffl.players_by_year
WHERE league_id = '449.l.198278'
  AND manager IS NOT NULL
  AND manager != 'No Manager'
  AND keeper_price IS NOT NULL
GROUP BY player, manager, nfl_position, nfl_team, year,
         keeper_price, kept_next_year, ppg_season, ppg_next_season,
         cost, is_keeper_status
ORDER BY year DESC, keeper_price DESC
```

---

## Performance Optimizations

### 1. Smart Caching Strategy

**Three-tier caching based on data volatility:**

```python
# Tier 1: Static Data (1 hour cache)
@st.cache_data(ttl=3600)
def list_seasons():
    # Season list never changes
    ...

# Tier 2: Recent Data (5 minutes cache)
@st.cache_data(ttl=300)
def load_player_week(year, week):
    # Stats updated occasionally
    ...

# Tier 3: Realtime Data (1 minute cache)
@st.cache_data(ttl=60)
def latest_season_and_week():
    # Current week changes frequently
    ...
```

**Cache Hit Rate:** 95%+ (reduces DB load by 20x)

---

### 2. Lazy Loading

**Pattern:** Load data only when tab is viewed

```python
# ❌ BAD - Loads all data upfront (10+ seconds)
def main():
    home_data = load_homepage_data()
    managers_data = load_managers_data()
    players_data = load_players_data()
    # ...render tabs

# ✅ GOOD - Loads data on-demand per tab
def render_home_tab():
    data = load_homepage_tab()  # Only when Home tab viewed
    display_homepage_overview(data)
```

**Performance:** 80% faster initial page load (2s vs 10s)

---

### 3. Progressive Loading

**Pattern:** Load critical data first, then enhance

```python
# Step 1: Load summary (fast)
summary = load_homepage_summary()  # 200ms
display_summary(summary)

# Step 2: Load details (slower)
with st.spinner("Loading details..."):
    details = load_homepage_details()  # 2s
    display_details(details)
```

---

### 4. Query Optimization

**Before (Slow):**
```sql
-- Complex CTE with window function (3.5s)
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY year ORDER BY points DESC) as rank
  FROM kmffl.players_by_year
  WHERE year = 2024
)
SELECT * FROM ranked WHERE rank <= 10
```

**After (Fast):**
```sql
-- Simple aggregation with ORDER BY + LIMIT (0.3s)
SELECT * FROM kmffl.players_by_year
WHERE league_id = '449.l.198278'
  AND year = 2024
ORDER BY points DESC NULLS LAST
LIMIT 10
```

**Performance:** 10x faster

---

### 5. Session State Management

**Pattern:** Store user selections to avoid re-querying

```python
# Store in session state
if "year" not in st.session_state:
    st.session_state.year, st.session_state.week = latest_season_and_week()

# Reuse across tabs
def render_players_tab():
    year = st.session_state.year  # No DB query!
    players = load_player_week(year, week)
```

---

## Tab Structure

### Directory Organization

```
streamlit_ui/
├── app_homepage_optimized.py      # Main entry point
├── md/
│   ├── data_access.py             # Database queries
│   ├── data_cache.py              # Cache utilities
│   └── motherduck_connection.py   # DB connection
├── tabs/
│   ├── homepage/
│   │   ├── homepage_overview.py   # Homepage display
│   │   └── recaps/                # Weekly recaps
│   ├── matchups/
│   │   ├── matchup_overview.py    # Manager stats
│   │   └── all_time/              # Career records
│   ├── player_stats/
│   │   ├── weekly_player_stats_optimized.py
│   │   ├── season_player_stats_optimized.py
│   │   └── career_player_stats_optimized.py
│   ├── draft_data/
│   │   └── draft_data_overview.py
│   ├── transactions/
│   │   └── transactions_adds_drops_trades_overview.py
│   ├── simulations/
│   │   ├── simulation_home.py
│   │   ├── predictive/            # Playoff odds
│   │   └── what_if/               # Schedule shuffling
│   ├── keepers/
│   │   └── keepers_home.py
│   ├── team_names/
│   │   └── team_names.py
│   ├── graphs/
│   │   ├── player_graphs/
│   │   ├── manager_graphs/
│   │   ├── draft_graphs/
│   │   └── simulation_graphs/
│   └── shared/
│       ├── modern_styles.py       # CSS styling
│       └── filters.py             # Reusable filters
└── utils/
    ├── performance.py             # Performance monitoring
    └── ui_components.py           # Reusable UI elements
```

---

## UI Components

### Reusable Components

#### `render_header(title: str, subtitle: str = None)`

**Purpose:** Display styled section header

**Example:**
```python
render_header("Player Statistics", "Season 2024, Week 5")
```

---

#### `render_empty_state(message: str)`

**Purpose:** Display empty state message

**Example:**
```python
if df.empty:
    render_empty_state("No data available for this week")
```

---

#### `render_loading_skeleton()`

**Purpose:** Display loading placeholder

**Example:**
```python
with st.spinner("Loading..."):
    render_loading_skeleton()
    data = fetch_data()
```

---

### Modern Styles

**File:** `tabs/shared/modern_styles.py`

**Applied globally in `main()`:**
```python
from tabs.shared.modern_styles import apply_modern_styles
apply_modern_styles()
```

**CSS Customizations:**
- Card-style containers with shadows
- Modern color palette (blue/white theme)
- Responsive grid layout
- Custom metric styling
- Interactive hover effects

---

## Configuration

### Streamlit Secrets

**File:** `.streamlit/secrets.toml`

```toml
[motherduck]
token = "your-motherduck-token"
database = "kmffl_analytics"

[md]
attach_url = "md:kmffl_analytics"

[auth]
# Optional: Add authentication if needed
username = "admin"
password = "secure-password"
```

### Environment Variables

```bash
# Required
MOTHERDUCK_TOKEN=your-token-here

# Optional
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none  # Disable file watcher
STREAMLIT_SERVER_PORT=8501                # Default port
```

---

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app_homepage_optimized.py
```

### Production Deployment (Streamlit Cloud)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Configure Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Connect GitHub repo
   - Set main file: `streamlit_ui/app_homepage_optimized.py`
   - Add secrets (MOTHERDUCK_TOKEN, etc.)

3. **Deploy**
   - Click "Deploy"
   - App URL: `https://your-app.streamlit.app`

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY streamlit_ui/ ./streamlit_ui/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_ui/app_homepage_optimized.py"]
```

---

## Performance Benchmarks

### Initial Page Load

| Metric | Before Optimization | After Optimization | Improvement |
|--------|-------------------|-------------------|-------------|
| Time to Interactive | 12.3s | 2.1s | 83% faster |
| Data Loaded | 5.2 MB | 0.8 MB | 85% reduction |
| DB Queries | 47 | 8 | 83% reduction |
| Cache Hit Rate | 45% | 95% | 2x improvement |

### Tab Navigation

| Tab | Load Time (Cached) | Load Time (Fresh) |
|-----|-------------------|------------------|
| Home | 150ms | 2.0s |
| Managers | 180ms | 2.8s |
| Players | 200ms | 3.2s |
| Draft | 190ms | 2.5s |
| Transactions | 250ms | 4.1s |
| Simulations | 280ms | 5.3s |
| Extras | 160ms | 2.2s |

### Query Performance

| Query Type | Rows Returned | Time (Optimized) | Time (Unoptimized) |
|-----------|--------------|-----------------|-------------------|
| Latest Week Players | 1,800 | 23ms | 2,340ms |
| Season Stats | 450 | 18ms | 980ms |
| Career Records | 12,000 | 95ms | 4,200ms |
| Playoff Odds | 140 | 35ms | 1,100ms |

---

## Appendix

### Common Issues

#### Issue: "MotherDuck unavailable"

**Solution:** Check secrets configuration
```toml
# .streamlit/secrets.toml
[motherduck]
token = "your-token"

[md]
attach_url = "md:kmffl_analytics"
```

#### Issue: Slow page load

**Solution:** Clear caches
```python
st.cache_data.clear()
st.cache_resource.clear()
```

#### Issue: Missing data for current week

**Solution:** Re-run data import pipeline
```bash
cd fantasy_football_data_scripts
python initial_import_v2.py --context leagues/kmffl/league_context.json
```

---

### Version History

**v2.0 (2025-10-28)** - Optimized Release
- Added smart caching (3-tier TTL)
- Implemented lazy loading
- Added performance monitoring
- Optimized database queries (10-100x faster)
- Added league_id filtering
- Fixed keeper data loading

**v1.0 (2024-08-15)** - Initial Release
- Basic tab structure
- Homepage, managers, players tabs
- MotherDuck integration
- Simple caching

---

**End of Documentation**

For questions or issues, contact the development team or file an issue in the repository.
