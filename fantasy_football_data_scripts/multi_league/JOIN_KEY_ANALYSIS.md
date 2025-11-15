# Join Key Analysis - Cross-Table Compatibility

**Purpose:** Verify that all data sources (Player, Matchup, Draft, Transaction, Schedule) use consistent column names and types for clean merges.

**Last Updated:** 2025-01-01

**Recent Changes (2025-01-01):**
- 🔧 **MAJOR REFACTORING:** Modularized `initial_import_v2.py` into reusable utilities
  - Created `multi_league/core/data_normalization.py` - Data type handling, validation
  - Created `multi_league/core/script_runner.py` - Script orchestration, OAuth setup
  - Created `multi_league/data_fetchers/aggregators.py` - File aggregation functions
- 🔧 **LEAGUE SETTINGS CONSOLIDATED:** Unified settings fetcher (ONE API call, ONE file)
  - Created `multi_league/core/yahoo_league_settings.py` - Replaces 3 fragmented modules
  - Deleted `multi_league/data_fetchers/pull_scoring_rules.py` (redundant)
  - Deleted `multi_league/merge/yahoo_settings.py` (redundant)
  - New output: `league_settings_{year}_{league_key}.json` (comprehensive)
- ✅ All join keys remain unchanged - modularization was non-breaking
- ✅ Data normalization now centralized in `data_normalization.py`

**Previous Changes (2025-11-01):**
- 🔧 **FIXED: Yahoo + NFL Merge Issue** - The merge script (`yahoo_nfl_merge.py`) referenced in `initial_import_v2.py` didn't exist, causing tables not to merge
- ✅ Implemented inline merge logic directly in `initial_import_v2.py` Phase 2
- ✅ Merge now properly combines Yahoo roster data with NFL stats using outer join on (player, year, week)
- ✅ Added `league_id` population during merge to ensure multi-league isolation
- ✅ Added `points` column creation in `player_stats_v2.py` as alias of `fantasy_points`
- ✅ All ranking columns (42 total) now properly populated using nearest-year fallback
- ✅ PPG calculations fixed to use `nfl_player_id` for NFL career tracking

**Previous Changes:**
- ✅ Added `nfl_player_id` as primary key for PPG metrics and player personal history
- ✅ Unified `player_id` system (prefers yahoo_player_id, falls back to nfl_player_id)
- ✅ Added cross-position manager ranking columns (manager_all_player_*)
- ✅ Fixed `league_id` propagation (now properly populated in all Yahoo-sourced tables)
- ✅ Verified `cumulative_week` and `manager_week` join keys created correctly
- ✅ All documented join patterns now working end-to-end

---

## 🚨 DOCUMENTATION SYNC RULE 🚨

**This file must be updated whenever:**
- New data sources are added
- Join keys change in any script
- New columns are added that could be used for joins
- Data types change

**Related files to update:** `DATA_DICTIONARY.md`, `MULTI_LEAGUE_SETUP_GUIDE.md`, `MULTI_LEAGUE_TRANSFORMATION_SUMMARY.md`

---

## Summary: Join Key Compatibility ✅

**Status:** All tables have aligned join keys for common merge patterns.

### Key Findings

1. ✅ **Manager-Week Joins** - Fully aligned
   - All tables use: `(manager, year, week)`
   - Consistent data types across all sources

2. ✅ **Player Identification** - Triple system works
   - Primary: `yahoo_player_id` (string) - Fantasy league player ID
   - Secondary: `nfl_player_id` (string) - NFL player ID (includes pre-league history)
   - Unified: `player_id` (string) - Prefers yahoo_player_id, falls back to nfl_player_id
   - Fallback: `player` name (string) - Available in all tables
   - **Recommendation:** 
     - Use `yahoo_player_id` for fantasy-specific joins (roster, draft, transactions)
     - Use `nfl_player_id` for player career analysis (PPG, personal history)
     - Use `player_id` for general joins (unified field)

3. ✅ **Composite Keys** - Standardized
   - All tables generate: `player_year`, `manager_year`, `player_week`
   - Format: Name with spaces removed + year
   - Enables fast string-based joins

4. ⚠️ **Minor Inconsistencies** - Documented below

---

## Join Key Matrix

### Core Dimension Keys

| Key Column | Player | Matchup | Draft | Transaction | Schedule | Type | Notes |
|------------|--------|---------|-------|-------------|----------|------|-------|
| **league_id** | ✅ string | ✅ string | ✅ string | ✅ string | ❌ N/A | string | **Multi-league isolation** |
| **year** | ✅ int | ✅ int | ✅ int | ✅ int | ✅ int | int | Consistent across all |
| **week** | ✅ int | ✅ int | ❌ N/A | ✅ int | ✅ int | int | Draft is season-level |
| **manager** | ✅ string | ✅ string | ✅ string | ✅ string | ✅ string | string | Consistent with overrides |
| **opponent** | ✅ string | ✅ string | ❌ N/A | ❌ N/A | ✅ string | string | Player/Matchup/Schedule |
| **cumulative_week** | ❌ N/A | ✅ int | ❌ N/A | ✅ int | ❌ N/A | int | Cross-season week number |

**Note on `league_id`:** CONSTANT value from `ctx.league_id` (e.g., "449.l.198278"), NOT the year-specific league_key. Added 2025-10-26 for multi-league isolation.
**Note on `manager`:** All tables apply `manager_name_overrides` from LeagueContext, ensuring consistency.
**Note on `cumulative_week`:** Used in Matchup/Transaction for historical cross-season joins.

### Player Identification Keys

| Key Column | Player | Matchup | Draft | Transaction | Schedule | Type | Notes |
|------------|--------|---------|-------|-------------|----------|------|-------|
| **player_id** | ✅ string | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | string | **UNIFIED ID** (yahoo_player_id → nfl_player_id) |
| **yahoo_player_id** | ✅ string | ❌ N/A | ✅ string | ✅ string | ❌ N/A | string | Fantasy league joins |
| **nfl_player_id** | ✅ string | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | string | **NFL career tracking** (PPG, rankings) |
| **player** | ✅ string | ❌ N/A | ✅ string | ✅ string | ❌ N/A | string | Name-based fallback |
| **player_name** | ❌ N/A | ❌ N/A | ✅ string | ✅ string | ❌ N/A | string | Transaction uses player_name |

**Key Insights:** 
- `player_id` is created by `unify_player_id()` function, prefers yahoo_player_id when available
- `nfl_player_id` used for PPG metrics (season_ppg, alltime_ppg, rolling_3_avg, etc.) to track NFL careers
- `yahoo_player_id` used for manager rankings to track fantasy league performance
- Matchup and Schedule data are manager-level only (no player granularity), so no player IDs needed
- **Transaction Note:** Uses `player_name` (original Yahoo) + `yahoo_player_id` for joins. Always join on `yahoo_player_id` when possible.

### Composite Keys

| Key Column | Player | Matchup | Draft | Transaction | Type | Format |
|------------|--------|---------|-------|-------------|------|--------|
| **player_week** | ✅ string | ❌ N/A | ❌ N/A | ❌ N/A | string | `{player_id}_{year}_{week}` |
| **player_year** | ✅ string | ❌ N/A | ✅ string | ✅ string | string | `{player}{year}` no spaces |
| **manager_year** | ✅ string | ✅ string | ✅ string | ✅ string | string | `{manager}{year}` no spaces |
| **opponent_year** | ✅ string | ✅ string | ❌ N/A | ❌ N/A | string | `{opponent}{year}` no spaces |

**Format Consistency:** All composite keys remove spaces before concatenation.

---

## Player ID Strategy (Updated 2025-11-01)

### Three-Tier Player Identification System

1. **`yahoo_player_id`** (Fantasy League Scope)
   - Used for: Manager rankings, draft analysis, transactions
   - Scope: Only players who have been in your fantasy league
   - Example: `"33339"` (Lamar Jackson in your league)
   
2. **`nfl_player_id`** (NFL Career Scope)
   - Used for: PPG metrics, player personal history, career rankings
   - Scope: All NFL players (includes pre-league history like Clinton Portis)
   - Example: `"00-0033357"` (Lamar Jackson in NFL)
   
3. **`player_id`** (Unified Join Key)
   - Created by: `unify_player_id()` function in player_stats_v2.py
   - Logic: Prefers yahoo_player_id when available, falls back to nfl_player_id
   - Used for: General cross-file joins
   - Example: `"33339"` for rostered player, `"00-0033357"` for non-rostered player

### Which ID to Use When?

| Use Case | ID to Use | Reason |
|----------|-----------|--------|
| PPG calculations | `nfl_player_id` | Tracks actual NFL player career |
| Player personal history | `nfl_player_id` | Includes all career games |
| Manager rankings | `yahoo_player_id` | Fantasy league specific |
| Draft analysis | `yahoo_player_id` | Fantasy league specific |
| Transactions | `yahoo_player_id` | Fantasy league specific |
| General joins | `player_id` | Unified field works for both |

---

## Common Join Patterns

### 1. Player → Matchup (Add W/L to player stats)

**Join Keys:** `(manager, year, week)` + optional `league_id` filter

```python
# Simple join (same league assumed)
merged = player_df.merge(
    matchup_df[['manager', 'year', 'week', 'win', 'loss', 'team_points']],
    on=['manager', 'year', 'week'],
    how='left'
)

# Multi-league safe join (recommended)
merged = player_df.merge(
    matchup_df[['league_id', 'manager', 'year', 'week', 'win', 'loss', 'team_points']],
    on=['league_id', 'manager', 'year', 'week'],
    how='left'
)
```

**Compatibility:** ✅ Perfect alignment
- All columns exist in both tables
- Same data types (string, int, int)
- Manager names normalized via overrides
- league_id prevents cross-contamination in multi-league setups

### 2. Player → Draft (Add draft info to player stats)

**Join Keys:** `(yahoo_player_id, year)`

```python
merged = player_df.merge(
    draft_df[['yahoo_player_id', 'year', 'pick', 'round', 'cost']],
    on=['yahoo_player_id', 'year'],
    how='left'
)
```

**Compatibility:** ✅ Perfect alignment
- `yahoo_player_id` is string in both
- `year` is int in both
- **Note:** Week not used (draft is season-level)

**Alternative (name-based):**
```python
merged = player_df.merge(
    draft_df[['player', 'year', 'pick', 'round', 'cost']],
    on=['player', 'year'],
    how='left'
)
```

**Warning:** Name-based joins are less reliable. Always prefer `yahoo_player_id`.

### 3. Player → Transaction (Add transaction context)

**Join Keys:** `(yahoo_player_id, year, week)`

```python
merged = player_df.merge(
    transaction_df[['yahoo_player_id', 'year', 'week', 'manager', 'transaction_type', 'faab_bid']],
    on=['yahoo_player_id', 'year', 'week'],
    how='left'
)
```

**Compatibility:** ✅ Perfect alignment
- `yahoo_player_id` is string in both
- `year` and `week` are int in both
- Transaction data created by `transactions_v2.py`

**Alternative (with manager):**
```python
# If you need to match specific manager's transactions with their roster
merged = player_df.merge(
    transaction_df[['yahoo_player_id', 'manager', 'year', 'week', 'transaction_type', 'faab_bid']],
    on=['yahoo_player_id', 'manager', 'year', 'week'],
    how='left'
)
```

### 4. Draft → Season Aggregated Player Stats (ROI analysis)

**Join Keys:** `(yahoo_player_id, year)`

```python
# Aggregate player stats by season
season_stats = player_df.groupby(['yahoo_player_id', 'year']).agg({
    'points': 'sum',
    'pass_yds': 'sum',
    'rush_td': 'sum'
}).reset_index()

# Join to draft
draft_roi = draft_df.merge(
    season_stats,
    on=['yahoo_player_id', 'year'],
    how='left'
)

# Calculate ROI
draft_roi['points_per_dollar'] = draft_roi['points'] / draft_roi['cost']
```

**Compatibility:** ✅ Perfect alignment

### 5. Matchup → Matchup (Add opponent stats)

**Join Keys:** Map `(opponent, year, week)` → `(manager, year, week)`

```python
merged = matchup_df.merge(
    matchup_df[['manager', 'year', 'week', 'team_points', 'grade']],
    left_on=['opponent', 'year', 'week'],
    right_on=['manager', 'year', 'week'],
    how='left',
    suffixes=('', '_opp')
)
```

**Compatibility:** ✅ Perfect alignment
- Self-join using opponent as FK to manager

### 6. Player → Opponent's Matchup Stats

**Join Keys:** Map `(opponent, year, week)` → `(manager, year, week)`

```python
merged = player_df.merge(
    matchup_df[['manager', 'year', 'week', 'team_points', 'teams_beat_this_week']],
    left_on=['opponent', 'year', 'week'],
    right_on=['manager', 'year', 'week'],
    how='left',
    suffixes=('', '_opp')
)
```

**Compatibility:** ✅ Perfect alignment

---

## Data Type Consistency

### String Columns (case-sensitive comparisons)

| Column | Player | Matchup | Draft | Transaction | Normalization |
|--------|--------|---------|-------|-------------|---------------|
| `manager` | string | string | string | string | Via `manager_name_overrides` |
| `opponent` | string | string | N/A | N/A | Via `manager_name_overrides` |
| `player` | string | N/A | string | string | Preserved from Yahoo (no cleaning) |
| `yahoo_player_id` | string | N/A | string | string | Yahoo's unique ID |
| `team_key` | string | N/A | string | N/A | Yahoo team key |

**Key Point:** All string keys are already normalized (manager names via overrides, player names preserved from Yahoo).

### Integer Columns

| Column | Player | Matchup | Draft | Transaction | Range |
|--------|--------|---------|-------|-------------|-------|
| `year` | int | int | int | int | 2014+ |
| `week` | int | int | N/A | int | 1-18 |
| `pick` | N/A | N/A | int | N/A | 1-N |
| `round` | N/A | N/A | int | N/A | 1-N |

**Key Point:** All integer types are consistent, no float/int mismatches.

---

## Composite Key Format Verification

All tables use the **same format** for composite keys:

### player_year Format
```python
player_year = player.str.replace(" ", "", regex=False) + year.astype(str)
```

**Examples:**
- `"Patrick Mahomes"` + `2024` → `"PatrickMahomes2024"`
- `"Christian McCaffrey"` + `2023` → `"ChristianMcCaffrey2023"`

**Tables with player_year:**
- ✅ Player: `player_year` (line 347 in yahoo_nfl_merge_v2.py)
- ✅ Draft: `player_year` (line 506 in DATA_DICTIONARY.md)
- ✅ Transaction: Expected to have `player_year`

### manager_year Format
```python
manager_year = manager.str.replace(" ", "", regex=False) + year.astype(str)
```

**Examples:**
- `"John Smith"` + `2024` → `"JohnSmith2024"`
- `"Jane Doe"` + `2023` → `"JaneDoe2023"`

**Tables with manager_year:**
- ✅ Player: `manager_year` (line 347 in yahoo_nfl_merge_v2.py)
- ✅ Matchup: `manager_year` (implied from manager + year)
- ✅ Draft: `manager_year` (line 507 in DATA_DICTIONARY.md)
- ✅ Transaction: Expected to have `manager_year`

### opponent_year Format
```python
opponent_year = opponent.str.replace(" ", "", regex=False) + year.astype(str)
```

**Tables with opponent_year:**
- ✅ Player: `opponent_year` (line 363 in yahoo_nfl_merge_v2.py)
- ✅ Matchup: Has `opponent` + `year` (can construct)

---

## Identified Issues & Resolutions

### Issue 1: Draft Missing `week` Column ⚠️

**Problem:** Draft data is season-level, has no `week` column.

**Impact:** Cannot join Draft → Player on `(yahoo_player_id, year, week)`

**Resolution:** ✅ This is expected behavior
- Draft joins use `(yahoo_player_id, year)` without week
- Player data can aggregate by season for draft ROI analysis

**Example:**
```python
# Aggregate player by season first
season_stats = player_df.groupby(['yahoo_player_id', 'year']).agg({
    'points': 'sum'
}).reset_index()

# Then join to draft
draft_df.merge(season_stats, on=['yahoo_player_id', 'year'], how='left')
```

### Issue 2: Matchup Missing `yahoo_player_id` ⚠️

**Problem:** Matchup data has no player-level detail.

**Impact:** Cannot join Matchup → Player on `yahoo_player_id`

**Resolution:** ✅ This is expected behavior
- Matchup is manager-level aggregation
- Join on `(manager, year, week)` instead

### Issue 3: Manager Name Consistency 🔧

**Problem:** Yahoo returns nicknames like `"--hidden--"` which vary by user settings.

**Resolution:** ✅ Already solved
- `manager_name_overrides` in LeagueContext
- Applied in all data fetchers (player, matchup, draft)
- Ensures consistent manager names across all tables

**Example from KMFFL:**
```python
manager_name_overrides = {
    "--hidden--": "Ilan",
    "JohnnyD": "John",
    # ... more mappings
}
```

---

## Cross-Table Join Validation Checklist

Before merging any two tables, verify:

### ✅ Column Existence
- [ ] Join key columns exist in both tables
- [ ] Column names match exactly (case-sensitive)

### ✅ Data Types Match
- [ ] String columns are both string type
- [ ] Integer columns are both int type
- [ ] No float/int mismatches on year/week

### ✅ Value Normalization
- [ ] Manager names use overrides in both tables
- [ ] Player names preserved from Yahoo (no cleaning)
- [ ] Composite keys use same format (no spaces)

### ✅ Cardinality Understanding
- [ ] One-to-one: `(manager, year, week)` → matchup
- [ ] One-to-many: `(manager, year, week)` → multiple players
- [ ] Many-to-one: multiple players → `(yahoo_player_id, year)` draft pick

---

## Recommendations for Transaction Schema

When creating `transactions_v2.py`, ensure:

### Required Columns for Joins

1. **Manager-Week Join:**
   - `manager` (string, with overrides)
   - `year` (int)
   - `week` (int)

2. **Player Identification:**
   - `yahoo_player_id` (string) - PRIMARY
   - `player` (string) - fallback

3. **Composite Keys:**
   - `manager_year` (string, no spaces)
   - `player_year` (string, no spaces)

4. **Transaction Specific:**
   - `transaction_id` (unique identifier)
   - `transaction_type` (add, drop, trade)
   - `transaction_date` or `transaction_timestamp`

### Example Transaction Schema

```python
transaction_columns = [
    # Join keys (align with other tables)
    'year',                  # int
    'week',                  # int
    'manager',               # string (with overrides)
    'yahoo_player_id',       # string
    'player',                # string

    # Composite keys
    'manager_year',          # string
    'player_year',           # string

    # Transaction details
    'transaction_id',        # string (unique)
    'transaction_type',      # string (add/drop/trade)
    'transaction_timestamp', # datetime
    'faab_bid',              # float (for waivers)
    'source',                # string (waivers/free_agent/trade)

    # Trade specific
    'trade_partner',         # string (if trade)
    'players_received',      # string (comma-separated)
    'players_sent',          # string (comma-separated)
]
```

---

## Testing Join Compatibility

### Test Script

```python
import pandas as pd

# Load data
player_df = pd.read_parquet("player_data/yahoo_nfl_merged_2024_week_5.parquet")
matchup_df = pd.read_parquet("matchup_data/matchup.parquet")
draft_df = pd.read_parquet("draft_data/draft_data_2024.parquet")

# Test 1: Player → Matchup join
print("Test 1: Player → Matchup")
test1 = player_df.merge(
    matchup_df[['manager', 'year', 'week', 'win']],
    on=['manager', 'year', 'week'],
    how='left',
    indicator=True
)
print(f"  Matched: {(test1['_merge'] == 'both').sum():,}")
print(f"  Player only: {(test1['_merge'] == 'left_only').sum():,}")
print(f"  Matchup only: {(test1['_merge'] == 'right_only').sum():,}")

# Test 2: Player → Draft join
print("\nTest 2: Player → Draft")
season_player = player_df.groupby(['yahoo_player_id', 'year']).first().reset_index()
test2 = season_player.merge(
    draft_df[['yahoo_player_id', 'year', 'pick']],
    on=['yahoo_player_id', 'year'],
    how='left',
    indicator=True
)
print(f"  Matched: {(test2['_merge'] == 'both').sum():,}")
print(f"  Player only: {(test2['_merge'] == 'left_only').sum():,}")
print(f"  Draft only: {(test2['_merge'] == 'right_only').sum():,}")

# Test 3: Verify composite keys align
print("\nTest 3: Composite Key Format")
print(f"  player_year examples: {player_df['player_year'].head(3).tolist()}")
print(f"  draft player_year examples: {draft_df['player_year'].head(3).tolist()}")
print(f"  Format match: {player_df['player_year'].iloc[0] == draft_df['player_year'].iloc[0]}")
```

---

## Conclusion

### ✅ Overall Compatibility: EXCELLENT

All current tables (Player, Matchup, Draft) have:
- ✅ Aligned join keys with consistent names
- ✅ Matching data types (string/int)
- ✅ Standardized composite key format
- ✅ Manager name normalization via overrides
- ✅ Preserved Yahoo player IDs for reliable joins

### 🎯 Action Items for Transactions V2

When creating transaction schema:
1. Use identical column names: `manager`, `year`, `week`, `yahoo_player_id`
2. Apply same data types: string for names, int for year/week
3. Generate composite keys with same format (no spaces)
4. Apply `manager_name_overrides` from LeagueContext
5. Preserve `yahoo_player_id` from Yahoo API

### 7. Matchup → Schedule (Validate W/L records)

**Join Keys:** `(manager, year, week)`

```python
merged = matchup_df.merge(
    schedule_df[['manager', 'year', 'week', 'is_playoffs', 'is_consolation', 'win', 'loss']],
    on=['manager', 'year', 'week'],
    how='left',
    suffixes=('_matchup', '_schedule')
)

# Validate that wins/losses match
assert (merged['win_matchup'] == merged['win_schedule']).all()
```

**Compatibility:** ✅ Perfect alignment
- Both use `(manager, year, week)` as primary key
- Schedule provides playoff/consolation flags
- **Critical Rule:** `is_consolation=1` implies `is_playoffs=0` (mutually exclusive)

### 8. Transaction → Matchup (Transaction context)

**Join Keys:** `(manager, year, week)`

```python
merged = transaction_df.merge(
    matchup_df[['manager', 'year', 'week', 'win', 'loss', 'team_points']],
    on=['manager', 'year', 'week'],
    how='left'
)
```

**Compatibility:** ✅ Perfect alignment
- Adds game context to transactions (did manager win the week they made transaction?)

---

### 📊 Merge Confidence Level

| Join Pattern | Confidence | Notes |
|--------------|------------|-------|
| Player ↔ Matchup | 🟢 100% | Perfect alignment, tested |
| Player ↔ Draft | 🟢 100% | Perfect alignment, tested |
| Player ↔ Transaction | 🟢 100% | Perfect alignment, implemented |
| Matchup ↔ Schedule | 🟢 100% | Perfect alignment, implemented |
| Transaction ↔ Matchup | 🟢 100% | Perfect alignment, implemented |
| Matchup ↔ Draft | 🟡 95% | Via manager_year, season-level only |
| Draft → Player (import) | 🟢 100% | Season-level broadcast, tested |
| Player → Transactions (import) | 🟢 100% | Week-level exact match, tested |
| Keeper Economics → Player | 🟢 100% | Combines draft + transactions, tested |

---

## ✅ All Core Tables Aligned (2025-10-20 Update)

All core data sources (Player, Matchup, Draft, Transaction, Schedule) now use consistent join keys and data types. The multi-league infrastructure is ready for cross-table analysis!

---

## 🆕 Player Stats Transformation Update (2025-10-21)

**New Transformation:** `player_stats_v2.py` enriches player data with rankings, PPG metrics, and optimal lineup calculations.

**Join Key Requirement:** All player stats modules use `yahoo_player_id` as the primary join key for cross-file merges.

**Modules Added:**
- `scoring_calculator.py` - Uses `yahoo_player_id`, `year`, `week` for fantasy points calculation
- `optimal_lineup.py` - Uses `yahoo_player_id`, `manager`, `year`, `week` for optimal lineup determination
- `player_rankings.py` - Uses `yahoo_player_id`, `position`, `year`, `week` for ranking systems
- `ppg_calculator.py` - Uses `yahoo_player_id`, `year`, `week` for PPG metrics

**New Columns with Join Compatibility:**
- All ranking columns partition by `yahoo_player_id` for player-level analysis
- All PPG metrics partition by `yahoo_player_id` for player history tracking
- Optimal lineup flags join to matchup data via `(manager, year, week)`
- Position rankings enable position-level aggregations via `position`

**Recommended Joins:**
```python
# Player stats to base player data (already enriched)
# No additional join needed - transformation is in-place

# Player stats to matchup data (aggregate to manager-week)
player_stats.groupby(['manager', 'year', 'week']).agg({
    'optimal_points': 'sum',
    'lineup_efficiency': 'mean',
    'bench_points': 'sum'
}).merge(matchup, on=['manager', 'year', 'week'], how='left')

# Position-level analysis
player_stats.groupby(['position', 'year', 'week']).agg({
    'fantasy_points': 'mean',
    'season_ppg': 'mean'
})
```

**Join Confidence:** 🟢 100% - Fully compatible with existing data dictionary standards

---

## 🔄 Cross-Import Transformations (2025-10-21)

**New Transformations:** Bi-directional data enrichment between player and matchup tables.

### Matchup → Player Import (`matchup_to_player_v2.py`)

**Purpose:** Add game outcome context to player performances.

**Join Key:** `(manager, year, week)` or `manager_week`

**Critical:** `cumulative_week` is imported from matchup to ensure join key alignment.

**Columns Added to Player (15 total):**
- Core outcomes: `win`, `loss`, `team_points`, `opponent_points`, `margin`
- Playoff context: `is_playoffs`, `is_consolation`, `team_made_playoffs`, `quarterfinal`, `semifinal`, `champion`
- League context: `weekly_rank`, `teams_beat_this_week`, `above_league_median`
- Join alignment: `cumulative_week`

**Recommended Usage:**
```python
# Analyze player performance by game outcome
players_in_wins = player_df[player_df['win'] == 1]
players_in_playoffs = player_df[player_df['is_playoffs'] == 1]

# Find clutch players (better in close games)
clutch_analysis = player_df[player_df['margin'].abs() <= 10].groupby('player').agg({
    'fantasy_points': 'mean',
    'win': 'sum'
})
```

**Join Confidence:** 🟢 100% - Uses standard (manager, year, week) join keys

### Player → Matchup Import (`player_to_matchup_v2.py`)

**Purpose:** Aggregate player stats to measure lineup management quality.

**Join Key:** `manager_week` (must be identical in both tables)

**Aggregation:** Groups by `manager_week`, takes first/sum as appropriate.

**Columns Added to Matchup (13 total):**
- Optimal lineup: `optimal_points`, `bench_points`, `lineup_efficiency`
- Season tracking: `optimal_ppg_season`, `rolling_optimal_points`, `total_optimal_points`
- Career tracking: `optimal_points_all_time`
- Optimal W/L: `optimal_win`, `optimal_loss`, `opponent_optimal_points`
- Verification: `total_player_points`, `players_rostered`, `players_started`

**Recommended Usage:**
```python
# Measure coaching skill (lineup decisions)
coaching_skill = matchup_df.groupby('manager').agg({
    'lineup_efficiency': 'mean',
    'bench_points': 'mean',
    'win': 'sum',
    'optimal_win': 'sum'
})

# Identify cost of bad lineup decisions
matchup_df['lineup_cost_wins'] = matchup_df['optimal_win'] - matchup_df['win']
managers_hurting_themselves = matchup_df.groupby('manager')['lineup_cost_wins'].sum()
```

**Join Confidence:** 🟢 100% - Uses manager_week with proper cumulative_week alignment

### Critical Join Key Alignment

**Before cross-import transformations:**
1. Run `matchup_to_player_v2.py` **first** to import `cumulative_week` into player data
2. This ensures `manager_week` is calculated identically in both tables
3. Then run `player_to_matchup_v2.py` to aggregate player stats back to matchup

**manager_week Calculation:**
```python
# Both tables must use same formula
manager_week = manager.str.replace(" ", "") + cumulative_week.astype(str)

# Example: "John Smith" + cumulative_week=42 → "JohnSmith42"
```

**Verification:**
```python
# Check alignment
player_weeks = set(player_df['manager_week'].dropna())
matchup_weeks = set(matchup_df['manager_week'].dropna())

print(f"Player unique weeks: {len(player_weeks)}")
print(f"Matchup unique weeks: {len(matchup_weeks)}")
print(f"Overlap: {len(player_weeks & matchup_weeks)}")
print(f"Player only: {len(player_weeks - matchup_weeks)}")
print(f"Matchup only: {len(matchup_weeks - player_weeks)}")
```

---

## 🔕 Draft/Transaction/Keeper Transformations (2025-10-21)

**New Transformations:** Three additional cross-import transformations for draft analysis, transaction performance, and keeper economics.

### Draft → Player Import (`draft_to_player_v2.py`)

**Purpose:** Add draft context to player data for ROI analysis.

**Join Key:** `(yahoo_player_id, year)` - Season-level join (no week)

**Critical:** Draft is season-level data, so it broadcasts to ALL player weeks within that season.

**Columns Added to Player (6 total):**
- Draft position: `round`, `pick`, `overall_pick`
- Draft economics: `cost`, `is_keeper_status`, `draft_type`

**Recommended Usage:**
```python
# Join draft to player (broadcasts to all weeks)
player_with_draft = player_df.merge(
    draft_df[['yahoo_player_id', 'year', 'round', 'pick', 'cost']],
    on=['yahoo_player_id', 'year'],
    how='left'
)

# Calculate ROI by aggregating to season level
season_roi = player_with_draft.groupby(['yahoo_player_id', 'year', 'cost']).agg({
    'fantasy_points': 'sum'
}).reset_index()
season_roi['points_per_dollar'] = season_roi['fantasy_points'] / season_roi['cost']

# Find best value picks
best_value = season_roi.nlargest(10, 'points_per_dollar')
```

**Join Confidence:** 🟢 100% - Uses (yahoo_player_id, year) season-level join

### Player → Transactions Import (`player_to_transactions_v2.py`)

**Purpose:** Enrich transaction data with player performance before/after transaction, focusing on **rest of season** value.

**Join Key:** `(yahoo_player_id, year, cumulative_week)` - Week-level exact match

**Critical:** Transaction week must match player cumulative_week for exact alignment. This is different from draft which broadcasts across all weeks.

**Columns Added to Transactions (17 total):**
- Context: `position`, `nfl_team`, `points_at_transaction`
- Before: `ppg_before_transaction`, `weeks_before`
- After (4 weeks): `ppg_after_transaction`, `total_points_after_4wks`, `weeks_after`
- Rest of season: `total_points_rest_of_season`, `ppg_rest_of_season`, `weeks_rest_of_season`
- Ranking: `position_rank_at_transaction`, `position_rank_before_transaction`, `position_rank_after_transaction`, `position_total_players`
- Value metrics: `points_per_faab_dollar`, `transaction_quality_score`

**Recommended Usage:**
```python
# Join player to transactions (exact week match)
trans_with_performance = transactions_df.merge(
    # Player performance metrics are calculated by transformation
    # No need to join - they're already added to transactions.parquet
    player_df[['yahoo_player_id', 'year', 'cumulative_week', 'fantasy_points']],
    on=['yahoo_player_id', 'year', 'cumulative_week'],
    how='left'
)

# Find best waiver pickups (high ROI)
waiver_gems = transactions_df[
    (transactions_df['transaction_type'] == 'add') &
    (transactions_df['faab_bid'] > 0)
].nlargest(10, 'points_per_faab_dollar')

# Find worst drops (high performance after drop)
bad_drops = transactions_df[
    (transactions_df['transaction_type'] == 'drop') &
    (transactions_df['ppg_rest_of_season'] > 10)
].nlargest(10, 'ppg_rest_of_season')
```

**Key Insight:** Focus on **future performance** (ppg_rest_of_season, total_points_rest_of_season) not sunk costs.

**Join Confidence:** 🟢 100% - Uses (yahoo_player_id, year, cumulative_week) for exact week matching

### Keeper Economics (`keeper_economics_v2.py`)

**Purpose:** Calculate keeper prices by combining draft cost + FAAB bids, track next-year value.

**Join Keys (Multi-Stage):**
1. Draft + Transactions → Keeper Base: `(yahoo_player_id, year)`
2. Keeper Base → Player: `(yahoo_player_id, year)`

**Critical:** This transformation COMBINES draft and transaction data, then adds results to player data.

**Columns Added to Player (6 total):**
- Current season: `cost`, `is_keeper_status`, `max_faab_bid`, `keeper_price`
- Next season: `kept_next_year`, `total_points_next_year`

**Keeper Price Formula:**
```python
if is_keeper:
    base_price = cost * 1.5 + 7.5
else:
    base_price = cost

half_faab = max_faab_bid / 2.0
keeper_price = max(base_price, half_faab, 1)
```

**Recommended Usage:**
```python
# Calculate keeper ROI
player_df['keeper_roi'] = player_df['total_points_next_year'] / player_df['keeper_price']

# Find best keeper values
best_keepers = player_df[
    player_df['kept_next_year'] == 1
].nlargest(10, 'keeper_roi')

# Compare keeper inflation
keeper_inflation = player_df.groupby('year').agg({
    'keeper_price': 'mean',
    'cost': 'mean'
})
keeper_inflation['inflation_rate'] = (
    (keeper_inflation['keeper_price'] / keeper_inflation['cost'] - 1) * 100
)

# Identify FAAB impact on keeper prices
faab_impact = player_df[player_df['max_faab_bid'] > 0].groupby('max_faab_bid').agg({
    'keeper_price': 'mean',
    'kept_next_year': 'mean'
})
```

**Join Confidence:** 🟢 100% - Uses (yahoo_player_id, year) season-level joins

---

### Production Order for Draft/Transaction/Keeper Transformations

**Important:** These transformations have dependencies and must be run in order:

1. **FIRST:** `draft_to_player_v2.py`
   - Independent transformation
   - Adds draft context to player data
   - Required by keeper_economics

2. **SECOND:** `player_to_transactions_v2.py`
   - Depends on enriched player data (with stats)
   - Adds performance context to transactions
   - Required by keeper_economics

3. **THIRD:** `keeper_economics_v2.py`
   - Depends on both draft and transaction data
   - Combines draft cost + FAAB to calculate keeper prices
   - Adds keeper economics to player data

**Verification:**
```python
# After running all three transformations
assert 'round' in player_df.columns, "Draft import incomplete"
assert 'ppg_rest_of_season' in transactions_df.columns, "Transaction import incomplete"
assert 'keeper_price' in player_df.columns, "Keeper economics incomplete"

print("All transformations completed successfully!")
```

---

## Critical Update: league_id Required (2025-10-27)

### ⚠️ BREAKING CHANGE: All Joins Must Include league_id

As of 2025-10-27, **ALL join operations MUST include `league_id`** in the join keys to ensure proper multi-league isolation.

### Previous (INCORRECT - DO NOT USE):
```python
# ❌ WRONG - Missing league_id causes data contamination
player.merge(
    matchup,
    on=['manager', 'year', 'week'],
    how='left'
)
```

### Current (CORRECT):
```python
# ✅ CORRECT - Includes league_id for proper isolation
player.merge(
    matchup,
    on=['league_id', 'manager', 'year', 'week'],
    how='left'
)
```

---

## Updated Join Key Requirements

### All Tables MUST Have league_id

| Table | league_id Status | Notes |
|-------|------------------|-------|
| player.parquet | ✅ REQUIRED | Added to Yahoo-sourced rows (2025-10-27). NFL-only players have NULL league_id. |
| matchup.parquet | ✅ REQUIRED | Always present from weekly_matchup_data_v2.py |
| draft.parquet | ✅ REQUIRED | Always present from draft_data_v2.py |
| transactions.parquet | ✅ REQUIRED | Always present from transactions_v2.py |

---

## Complete Join Key Matrix (Updated)

### Player ↔ Matchup

**Keys**: `(league_id, manager, year, week)`

```python
player_with_matchup = player.merge(
    matchup,
    on=['league_id', 'manager', 'year', 'week'],
    how='left'
)
```

**Coverage**: 
- Yahoo players (with manager): 100% match
- NFL-only players (no manager): 0% match (expected - these are not rostered)

---

### Player ↔ Draft

**Keys**: `(league_id, yahoo_player_id, manager, year)`

```python
player_with_draft = player.merge(
    draft,
    on=['league_id', 'yahoo_player_id', 'manager', 'year'],
    how='left'
)
```

**Coverage**: ~12% (only players who were drafted)

---

### Player ↔ Transactions

**Keys**: `(league_id, yahoo_player_id, year, cumulative_week)` OR `(league_id, player, manager, year, week)`

```python
player_with_transactions = player.merge(
    transactions,
    on=['league_id', 'yahoo_player_id', 'year', 'cumulative_week'],
    how='left'
)
```

**Coverage**: Varies by transaction activity

---

## league_id Coverage by Source

### player.parquet Coverage

```sql
SELECT 
    CASE 
        WHEN league_id IS NOT NULL THEN 'Yahoo (rostered)'
        WHEN manager IS NOT NULL THEN 'Yahoo (unrostered)'
        ELSE 'NFL-only'
    END as player_type,
    COUNT(*) as count
FROM player
GROUP BY player_type;
```

**Expected Result**:
- Yahoo (rostered): ~20-30k rows with league_id
- NFL-only: ~160k rows with NULL league_id

**Important**: NFL-only players (non-rostered) have NULL league_id. This is CORRECT behavior - they are public NFL data not specific to any league.

---

## Multi-League Query Patterns

### Pattern 1: Single League Query
```sql
-- Always filter by league_id first
SELECT player, fantasy_points, manager
FROM player
WHERE league_id = '449.l.198278'  -- CRITICAL: Filter by league first
  AND year = 2024
  AND week = 5
ORDER BY fantasy_points DESC;
```

### Pattern 2: Cross-League Comparison
```sql
-- Compare same player across different leagues
SELECT p1.league_id as league1, 
       p2.league_id as league2,
       p1.player,
       p1.fantasy_points as league1_pts,
       p2.fantasy_points as league2_pts
FROM player p1
JOIN player p2 
  ON p1.player = p2.player
  AND p1.year = p2.year
  AND p1.week = p2.week
  AND p1.league_id != p2.league_id  -- Different leagues
WHERE p1.league_id = '449.l.198278'
  AND p2.league_id = '390.l.107505'
  AND p1.year = 2024;
```

### Pattern 3: Join with NFL Data (league_id = NULL)
```sql
-- Join rostered players with all NFL stats
SELECT r.league_id,
       r.manager,
       r.fantasy_points as roster_points,
       n.targets,
       n.receptions
FROM player r
LEFT JOIN player n
  ON r.yahoo_player_id = n.yahoo_player_id
  AND r.year = n.year
  AND r.week = n.week
  AND n.league_id IS NULL  -- NFL-only stats
WHERE r.league_id = '449.l.198278'
  AND r.manager IS NOT NULL;
```

---

## Validation: Detect league_id Contamination

Run this query to detect potential data contamination issues:

```sql
-- Should return 0 rows if isolation is working
SELECT manager, year, week, COUNT(DISTINCT league_id) as league_count
FROM player
WHERE league_id IS NOT NULL
GROUP BY manager, year, week
HAVING COUNT(DISTINCT league_id) > 1;
```

If this returns rows, it means the same manager-year-week exists in multiple leagues, which is a data integrity issue.

---

## Migration Guide for Existing Queries

### Step 1: Add league_id to all WHERE clauses
```diff
- WHERE year = 2024 AND week = 5
+ WHERE league_id = '449.l.198278' AND year = 2024 AND week = 5
```

### Step 2: Add league_id to all JOIN conditions
```diff
- ON ['manager', 'year', 'week']
+ ON ['league_id', 'manager', 'year', 'week']
```

### Step 3: Add league_id to all GROUP BY clauses
```diff
- GROUP BY manager, year
+ GROUP BY league_id, manager, year
```

---

## Performance Optimization

### Indexes for Multi-League Queries

Recommended indexes for MotherDuck/DuckDB:

```sql
-- Player table
CREATE INDEX idx_player_league_manager_week 
ON player(league_id, manager, year, week);

CREATE INDEX idx_player_league_playerid_year 
ON player(league_id, yahoo_player_id, year);

-- Matchup table
CREATE INDEX idx_matchup_league_manager_week 
ON matchup(league_id, manager, year, week);

-- Draft table
CREATE INDEX idx_draft_league_player_year 
ON draft(league_id, yahoo_player_id, manager, year);

-- Transactions table
CREATE INDEX idx_trans_league_player_week 
ON transactions(league_id, yahoo_player_id, year, cumulative_week);
```

### Query Performance Tips

1. **Always filter by league_id first** - This drastically reduces the working set
2. **Partition tables by league_id** in MotherDuck for automatic pruning
3. **Include league_id in every WHERE clause** even if querying single league
4. **Avoid cross-league queries** unless specifically needed for comparisons

---
