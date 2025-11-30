# Transactions Pipeline Documentation

> **Last Updated:** November 2024
> **Status:** Production
> **Data Source:** Yahoo Fantasy Football API

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Data Schema Reference](#data-schema-reference)
4. [UI Components](#ui-components)
5. [Feature Summaries (For Homepage/About)](#feature-summaries)
6. [Recommendations & Roadmap](#recommendations--roadmap)

---

## Executive Summary

The Transaction Analysis module provides comprehensive insights into all roster moves - waiver pickups, free agent additions, drops, and trades. It tracks every transaction with before/after performance metrics, SPAR-based value analysis, and FAAB efficiency calculations.

### Key Capabilities

- **Complete Transaction History**: Every add, drop, and trade from league inception
- **Dual SPAR Metrics**: Distinguish what you actually got (managed) vs opportunity cost (total)
- **NET SPAR Analysis**: True transaction value = adds - drops
- **FAAB ROI Tracking**: Measure FAAB spending efficiency
- **Trade Fairness Analysis**: Compare both sides of every trade

### Quick Stats

| Metric | Value |
|--------|-------|
| Total Transactions | 7,465+ |
| Data Columns | 78 |
| Enrichment Rate | ~76% of transactions have performance data |
| Transaction Types | add, drop, trade |

### What Makes This Special

Your transactions pipeline has a **dual SPAR architecture** that's quite sophisticated:

| Metric | Meaning | Use Case |
|--------|---------|----------|
| `manager_spar_ros_managed` | SPAR only for weeks on YOUR roster | "What did I actually get?" |
| `player_spar_ros_total` | SPAR for all ROS weeks regardless of roster | "What did I give up?" (opportunity cost) |
| `net_spar_ros` | Managed SPAR added - Total SPAR dropped | "Did I win this transaction?" |

---

## Pipeline Architecture

### High-Level Flow

```
Yahoo Fantasy API
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA FETCHING                      │
│                                                                │
│  transactions_v2.py                                            │
│  ├── Authenticates via OAuth2                                  │
│  ├── Fetches all transactions (adds, drops, trades)           │
│  ├── Maps timestamps to week windows                           │
│  ├── Extracts yahoo_player_id for reliable joins              │
│  ├── Creates composite keys (manager_week, player_week)       │
│  └── Outputs: transactions_year_{year}.parquet (per season)   │
│                                                                │
│  Key Features:                                                 │
│  • Automatic year caching for completed seasons               │
│  • Week mapping via matchup_windows for accuracy              │
│  • Retry logic with exponential backoff                       │
│  • Multi-player trade handling (no deduplication issues)      │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    PHASE 2: AGGREGATION                        │
│                                                                │
│  aggregators.py → normalize_transaction_parquet()              │
│  ├── Combines yearly files into transactions.parquet          │
│  ├── Normalizes data types (yahoo_player_id → Int64)          │
│  ├── Deduplicates by (transaction_id, yahoo_player_id)        │
│  └── Outputs: transactions.parquet (canonical file)           │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                 PHASE 3: ENRICHMENT (Multi-Pass)               │
│                                                                │
│  PASS 1: player_to_transactions_v2.py                          │
│  ├── Joins player.parquet weekly stats                        │
│  ├── Calculates ROS metrics DUAL style:                       │
│  │     • MANAGED: Only weeks on THIS manager's roster         │
│  │     • TOTAL: All weeks regardless of who rostered          │
│  ├── Adds position ranks (at, before, after transaction)      │
│  ├── Adds keeper status from draft.parquet                    │
│  └── Adds quality scores and derived metrics                  │
│                                                                │
│  PASS 2: transaction_value_metrics_v3.py                       │
│  ├── Loads weekly replacement_levels.parquet                  │
│  ├── Calculates window-based SPAR (week W+1 → 17)            │
│  ├── Adds dual SPAR metrics:                                  │
│  │     • player_spar_ros / manager_spar_ros                   │
│  │     • player_ppg_ros / manager_ppg_ros                     │
│  │     • player_pgvor_ros / manager_pgvor_ros                 │
│  ├── Normalizes waiver cost (FAAB vs Priority)                │
│  ├── Calculates fa_roi = spar / cost                          │
│  └── Adds NET SPAR grouped by transaction_id + manager        │
│                                                                │
│  Uses: transaction_spar_calculator.py (core SPAR logic)        │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                                │
│                                                                │
│  transactions.parquet                                          │
│  ├── 78 columns of transaction + performance data             │
│  ├── ~7,500 rows (all transactions)                           │
│  ├── Dual SPAR metrics (managed vs total)                     │
│  └── NET SPAR for transaction-level analysis                  │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI LAYER                          │
│                                                                │
│  Data Access: md/tab_data_access/transactions/                 │
│  ├── transaction_data.py - Optimized column selection         │
│  ├── summary_data.py - Aggregated summary stats               │
│  ├── combined.py - Entry point for data loading               │
│  └── Caching: 600s TTL with st.cache_data                     │
│                                                                │
│  UI Components: tabs/transactions/                             │
│  ├── transactions_adds_drops_trades_overview.py - Main hub    │
│  ├── Add/Drop Section:                                        │
│  │     ├── add_drop_overview.py - Sub-tab router              │
│  │     ├── weekly_add_drop.py - Weekly transaction view       │
│  │     ├── season_add_drop.py - Season aggregations           │
│  │     └── career_add_drop.py - Career stats                  │
│  ├── Trades Section:                                          │
│  │     ├── trade_overview.py - Sub-tab router                 │
│  │     ├── trade_by_trade_summary_data.py - Both sides view   │
│  │     ├── season_trade_data.py - Season trade stats          │
│  │     └── career_trade_data.py - Career trade stats          │
│  └── Combo Views:                                              │
│       ├── combo_transaction_overview.py                        │
│       ├── weekly_combo_transactions.py                         │
│       ├── season_combo_transactions.py                         │
│       └── career_combo_transactions.py                         │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Key Files Reference

| File | Location | Purpose |
|------|----------|---------|
| `transactions_v2.py` | `data_fetchers/` | Fetches raw data from Yahoo API |
| `aggregators.py` | `data_fetchers/` | Combines yearly files, deduplicates |
| `player_to_transactions_v2.py` | `transformations/transaction_enrichment/` | Joins player performance, dual ROS metrics |
| `transaction_value_metrics_v3.py` | `transformations/transaction_enrichment/` | SPAR calculation, ROI metrics |
| `transaction_spar_calculator.py` | `transformations/transaction_enrichment/modules/` | Core SPAR calculation logic |
| `fix_unknown_managers.py` | `transformations/transaction_enrichment/` | Repairs manager mapping issues |
| `transaction_data.py` | `streamlit_ui/md/tab_data_access/transactions/` | UI data loader |
| `transactions_adds_drops_trades_overview.py` | `streamlit_ui/tabs/transactions/` | Main UI component |

---

## Data Schema Reference

### Core Columns (78 Total)

#### Identifiers & Keys

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | String | Yahoo transaction identifier |
| `yahoo_player_id` | Int64 | Yahoo player identifier |
| `manager` | String | Manager name who made the transaction |
| `player_name` | String | Player full name |
| `league_id` | String | League identifier |

#### Time Dimensions

| Column | Type | Description |
|--------|------|-------------|
| `year` | Int64 | Season year |
| `week` | Int64 | Week number |
| `cumulative_week` | Int64 | YYYYWW format for sorting |
| `week_start` | String | Week start date |
| `week_end` | String | Week end date |
| `timestamp` | String | Unix timestamp of transaction |
| `human_readable_timestamp` | String | Formatted timestamp |
| `transaction_datetime` | DateTime | Parsed datetime |
| `transaction_sequence` | Int64 | Order within same-day transactions |

#### Transaction Details

| Column | Type | Description |
|--------|------|-------------|
| `transaction_type` | String | "add", "drop", or "trade" |
| `type` | String | Alias for transaction_type |
| `source_type` | String | Where player came from (waivers, freeagents, team) |
| `destination` | String | Where player went (team, waivers) |
| `status` | String | Transaction status (successful, etc.) |
| `faab_bid` | Int64 | FAAB dollars spent |

#### Player Context

| Column | Type | Description |
|--------|------|-------------|
| `position` | String | Player position (QB, RB, WR, TE, K, DEF) |
| `nfl_team` | String | NFL team at transaction time |
| `points_at_transaction` | Float64 | Points scored in transaction week |

#### Performance Before Transaction

| Column | Type | Description |
|--------|------|-------------|
| `ppg_before_transaction` | Float64 | PPG in 4 weeks before |
| `weeks_before` | Int64 | Weeks in "before" calculation |
| `position_rank_before_transaction` | Int64 | Season-to-date position rank |

#### Performance After Transaction (4-Week Window)

| Column | Type | Description |
|--------|------|-------------|
| `ppg_after_transaction` | Float64 | PPG in 4 weeks after |
| `total_points_after_4wks` | Float64 | Points in 4 weeks after |
| `weeks_after` | Int64 | Weeks in "after" calculation |

#### Rest of Season - MANAGED (on YOUR roster)

| Column | Type | Description |
|--------|------|-------------|
| `total_points_ros_managed` | Float64 | ROS points while on your roster |
| `ppg_ros_managed` | Float64 | ROS PPG while on your roster |
| `weeks_ros_managed` | Int64 | Weeks on your roster |
| `player_spar_ros_managed` | Float64 | ROS player SPAR (managed) |
| `manager_spar_ros_managed` | Float64 | ROS manager SPAR (started only) |
| `manager_spar_per_game_managed` | Float64 | Per-game SPAR (managed) |
| `replacement_ppg_ros_managed` | Float64 | Replacement baseline (managed window) |

#### Rest of Season - TOTAL (opportunity cost)

| Column | Type | Description |
|--------|------|-------------|
| `total_points_ros_total` | Float64 | ROS points regardless of roster |
| `ppg_ros_total` | Float64 | ROS PPG regardless of roster |
| `weeks_ros_total` | Int64 | Total ROS weeks |
| `player_spar_ros_total` | Float64 | ROS player SPAR (total) |
| `player_spar_per_game_total` | Float64 | Per-game SPAR (total) |
| `manager_spar_ros_total` | Float64 | ROS manager SPAR (total) |
| `replacement_ppg_ros_total` | Float64 | Replacement baseline (total window) |

#### NET Transaction Metrics (grouped by transaction_id + manager)

| Column | Type | Description |
|--------|------|-------------|
| `net_manager_spar_ros` | Float64 | Net SPAR actually captured (adds - drops) |
| `net_player_spar_ros` | Float64 | Net opportunity cost (total) |
| `net_spar_ros` | Float64 | Legacy alias for net_manager_spar_ros |
| `spar_efficiency` | Float64 | Net SPAR per FAAB dollar |
| `net_spar_rank` | Float64 | Rank by net managed SPAR |

#### Legacy SPAR Columns (backward compatibility)

| Column | Type | Description |
|--------|------|-------------|
| `replacement_ppg_ros` | Float64 | Legacy replacement PPG |
| `player_spar_ros` | Float64 | Legacy player SPAR |
| `manager_spar_ros` | Float64 | Legacy manager SPAR |
| `fa_spar_ros` | Float64 | Alias for manager_spar_ros |
| `fa_ppg_ros` | Float64 | Alias for manager_ppg_ros |
| `fa_pgvor_ros` | Float64 | Per-game VOR (manager) |
| `player_ppg_ros` | Float64 | Player PPG ROS |
| `manager_ppg_ros` | Float64 | Manager PPG ROS |
| `player_pgvor_ros` | Float64 | Per-game VOR (player) |
| `manager_pgvor_ros` | Float64 | Per-game VOR (manager) |

#### ROI & Value Metrics

| Column | Type | Description |
|--------|------|-------------|
| `waiver_cost_norm` | Float64 | Normalized cost (FAAB or priority-equivalent) |
| `fa_roi` | Float64 | SPAR / waiver_cost_norm |
| `spar_per_faab` | Float64 | SPAR per FAAB dollar |
| `points_per_faab_dollar` | Float64 | Points per FAAB dollar |
| `position_spar_percentile` | Float64 | Percentile within position/year |
| `value_vs_avg_pickup` | Float64 | SPAR vs position average pickup |
| `spar_per_faab_rank` | Float64 | SPAR/FAAB rank within position |

#### Rankings

| Column | Type | Description |
|--------|------|-------------|
| `position_rank_at_transaction` | Int64 | Weekly position rank |
| `position_total_players` | Int64 | Total players at position |
| `position_rank_after_transaction` | Int64 | ROS position rank |

#### Quality Metrics

| Column | Type | Description |
|--------|------|-------------|
| `transaction_quality_score` | Int64 | 1-5 quality rating |
| `kept_next_year` | Int64 | 1 = player kept next season |

#### Engagement Metrics (NEW - Pre-computed for UI)

| Column | Type | Description |
|--------|------|-------------|
| `transaction_grade` | String | A-F grade based on NET SPAR percentile within year/type |
| `transaction_result` | String | Human-readable result ("Elite Pickup", "Big Regret", etc.) |
| `faab_value_tier` | String | FAAB efficiency tier ("Steal", "Great Value", "Good Value", "Fair", "Overpay") |
| `drop_regret_score` | Float64 | For drops only: SPAR player produced after being dropped |
| `drop_regret_tier` | String | Category for drop regret ("No Regret", "Minor Regret", "Some Regret", "Big Regret", "Major Regret", "Disaster") |
| `timing_category` | String | Season timing ("Early Season", "Mid Season", "Late Season", "Playoffs") |
| `pickup_type` | String | Source type ("Waiver Claim", "Free Agent", "Trade", "Other") |
| `result_emoji` | String | Quick visual indicator emoji for UI display |
| `net_spar_percentile` | Float64 | Percentile rank within year/type for grade calculation |

#### Composite Keys

| Column | Type | Description |
|--------|------|-------------|
| `manager_week` | String | Manager + cumulative_week |
| `manager_year` | String | Manager + year |
| `player_week` | String | Player + year + week |
| `player_year` | String | Player + year |

---

## UI Components

### Current Implementation

The Transactions hub (`transactions_adds_drops_trades_overview.py`) contains **2 main tabs** with sub-tabs:

---

### Tab 1: Add/Drop

#### Sub-Tab: Weekly (weekly_add_drop.py)

**What It Shows:**
- Weekly transaction table with NET SPAR calculations
- Groups add/drop pairs by transaction_id
- Shows SPAR added, SPAR dropped, NET SPAR per transaction
- FAAB efficiency calculations

**Current Features:**
- Transaction-level grouping (combines add+drop into single row)
- Dual SPAR display (managed for adds, total for drops)
- Result indicators (Excellent/Great/Good/Positive/Neutral/Negative)
- Filters: Year, week, manager, type, position, result, NET SPAR
- Three sub-tabs: Transactions, Analytics, Leaderboards

**Analytics Tab Includes:**
- Transaction quality distribution histogram
- FAAB value return by spend bracket
- Position-level performance (adds vs drops)
- Weekly transaction trends
- Manager performance rankings (horizontal bar chart)
- Efficiency and average NET rankings

**Leaderboard Tab Includes:**
- Best pickups (most SPAR added)
- Best value (highest SPAR/$)
- Worst drops (most SPAR lost)
- Biggest FAAB spends
- Best/worst overall transactions by NET SPAR

**Suggested Additions:**
- [ ] Add "Pickup Grade" (A-F) based on SPAR percentile
- [ ] "Regret Meter" for drops - show what you would have gotten
- [ ] Time-of-week analysis (early vs late week transactions)
- [ ] Injury-related pickup success rate
- [ ] "Should have kept" alerts (dropped players who became top performers)

---

#### Sub-Tab: Season (season_add_drop.py)

**What It Shows:**
- Season-aggregated transaction stats by manager
- NET SPAR per season
- SPAR efficiency (per FAAB)
- Grades based on net performance

**Current Features:**
- Aggregates managed SPAR (adds) vs total SPAR (drops)
- Season grades: Elite/Great/Good/Average/Poor
- Manager career NET SPAR chart
- SPAR efficiency rankings
- Season trends over time

**Suggested Additions:**
- [ ] Pre-compute season aggregates in source table
- [ ] Add "Transaction Style" classification (volume vs precision)
- [ ] Correlation with final standings
- [ ] Position breakdown per season

---

#### Sub-Tab: Career (career_add_drop.py)

**What It Shows:**
- Career-long transaction statistics
- Total NET SPAR across all seasons

**Suggested Additions:**
- [ ] Career transaction efficiency rating
- [ ] Year-over-year improvement tracking
- [ ] "Best Move Ever" per manager
- [ ] Position tendency analysis

---

### Tab 2: Trades

#### Sub-Tab: Trade Summaries (trade_by_trade_summary_data.py)

**What It Shows:**
- Both sides of every trade
- Managed SPAR for acquired players
- Partner's SPAR from players you gave up
- NET SPAR calculation
- Result indicators

**Current Features:**
- Two-row view per trade (one per manager)
- Trade outcome distribution chart
- Manager win rate rankings
- Most lopsided trades list
- Most even trades list
- Most frequent trading partners
- Highest SPAR exchanges

**Suggested Additions:**
- [ ] Pre-compute trade side summaries in source table
- [ ] "Trade Grade" (A-F) based on NET SPAR
- [ ] Trade timing analysis (early season vs deadline)
- [ ] Position swaps analysis (WR-for-RB success rates)
- [ ] "Trade Tree" - track all moves stemming from a trade

---

#### Sub-Tab: Season Trades (season_trade_data.py)

**What It Shows:**
- Season-aggregated trade statistics
- Win/loss record per manager per season

**Suggested Additions:**
- [ ] Trade deadline analysis (pre vs post deadline success)
- [ ] Playoff push trades vs rebuilding trades

---

#### Sub-Tab: Career Trades (career_trade_data.py)

**What It Shows:**
- Career trade statistics
- All-time trading partners
- Win rates over career

**Suggested Additions:**
- [ ] "Trade Rival" identification (who you trade most with, who wins)
- [ ] Position expertise (which positions do you trade well?)

---

## Feature Summaries

### For Homepage

#### Transaction Analysis - Quick Summary

> **Track Every Roster Move**
>
> The Transaction Analysis module monitors all waiver pickups, free agent adds, drops, and trades throughout league history. Using SPAR-based metrics, you can see the true value of every transaction - what you gained, what you lost, and whether you came out ahead.
>
> **Key Features:**
> - Complete transaction history with before/after performance
> - Dual SPAR metrics showing actual vs potential value
> - FAAB efficiency tracking and ROI analysis
> - Trade fairness analysis showing both sides

---

### For About Page

#### Transaction Analysis - Detailed Description

> **What is Transaction Analysis?**
>
> The Transaction Analysis module provides comprehensive insights into all roster moves. Every add, drop, and trade is tracked and enriched with performance data to measure its true impact.
>
> **How It Works:**
>
> 1. **Data Collection**: Transaction data is fetched from Yahoo Fantasy API, including timestamps, FAAB bids, and player information.
>
> 2. **Performance Matching**: Each transaction is matched with player performance data before and after the move occurred.
>
> 3. **Dual SPAR Calculation**: We calculate two types of value:
>    - **Managed SPAR**: Value you actually captured (only weeks on YOUR roster)
>    - **Total SPAR**: Full opportunity cost (all remaining season weeks)
>
> 4. **NET SPAR**: For combined add/drop transactions, we calculate the net gain or loss.
>
> **Key Metrics Explained:**
>
> | Metric | What It Means |
> |--------|---------------|
> | **Managed SPAR** | SPAR from weeks the player was on your roster - your actual return |
> | **Total SPAR** | SPAR for all ROS weeks - shows opportunity cost for drops |
> | **NET SPAR** | Managed SPAR from adds minus Total SPAR from drops |
> | **SPAR Efficiency** | NET SPAR per FAAB dollar spent |
> | **Transaction Grade** | A-F grade based on NET SPAR percentile |
>
> **Why Dual SPAR Matters:**
>
> When you pick up a player, you want to know what YOU got (managed SPAR). When you drop a player, you want to know what you LOST (total SPAR - what they did the rest of the season, even on other rosters). This dual approach gives you the complete picture.
>
> **What You Can Learn:**
>
> - Which transactions delivered the best value
> - Your FAAB spending efficiency
> - Which drops hurt the most (opportunity cost)
> - Trade win/loss record with other managers
> - Position-specific transaction tendencies

---

### Section-by-Section Summaries (For Navigation/Tooltips)

#### Weekly Add/Drop
> "View all add/drop transactions with NET SPAR calculations. See what you gained from adds and what you lost from drops."

#### Season Add/Drop
> "Season-level transaction statistics aggregated by manager. Track your NET SPAR and efficiency across full seasons."

#### Career Add/Drop
> "Career-long transaction statistics. See your all-time transaction performance and tendencies."

#### Trade Summaries
> "View both sides of every trade with managed SPAR comparisons. See who won each deal."

#### Season Trades
> "Season-aggregated trade statistics and win rates by manager."

#### Career Trades
> "Career trading statistics including all-time partners and win rates."

---

## Recommendations & Roadmap

### CRITICAL ISSUES - STATUS

#### 1. **~~Missing Pre-Computed Columns~~** ✅ FIXED (Nov 2024)

The following engagement metrics are now pre-computed in `transaction_spar_calculator.py`:

| Column | Description | Status |
|--------|-------------|--------|
| `transaction_grade` | A-F grade based on NET SPAR percentile | ✅ Added |
| `transaction_result` | Human-readable result ("Elite Pickup", "Big Regret", etc.) | ✅ Added |
| `faab_value_tier` | FAAB efficiency tier ("Steal" → "Overpay") | ✅ Added |
| `drop_regret_score` | SPAR player produced after being dropped | ✅ Added |
| `drop_regret_tier` | Category for drop regret ("No Regret" → "Disaster") | ✅ Added |
| `timing_category` | Season timing ("Early Season", "Mid Season", etc.) | ✅ Added |
| `pickup_type` | Source type ("Waiver Claim", "Free Agent", "Trade") | ✅ Added |
| `result_emoji` | Quick visual indicator emoji | ✅ Added |

**To regenerate with new columns, run:**
```bash
python transaction_value_metrics_v3.py --context path/to/league_context.json
```

#### 2. **Schema Redundancy (78 columns)**
Several columns are duplicates or aliases:

| Redundant | Keep | Action |
|-----------|------|--------|
| `type` | `transaction_type` | Remove alias |
| `net_spar_ros` | `net_manager_spar_ros` | Keep for compatibility, deprecate |
| `fa_spar_ros` | `manager_spar_ros` | Keep for compatibility, deprecate |
| `total_points_rest_of_season` | `total_points_ros_total` | Keep for compatibility, deprecate |

#### 3. **UI Computation Overhead**
The weekly_add_drop.py file does significant processing on each load:
- Grouping by transaction_id
- Calculating net_value
- Creating result indicators
- Building combined add/drop rows

**Recommendation**: Pre-compute transaction-level summaries:

```python
# New table: transaction_summaries.parquet
# One row per transaction_id + manager with:
# - players_added, positions_added, spar_added
# - players_dropped, positions_dropped, spar_dropped
# - net_spar, transaction_grade, result_category
```

### Priority 1: Add Engagement Metrics to Source Table

Add these columns to `transaction_value_metrics_v3.py`:

```python
def add_engagement_metrics(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Add metrics designed for fun/engaging UI displays."""

    df = transactions_df.copy()

    # 1. TRANSACTION GRADE (A-F based on NET SPAR percentile)
    df['net_spar_percentile'] = df.groupby(['year', 'transaction_type'])['net_manager_spar_ros'].transform(
        lambda x: x.rank(pct=True) * 100
    )
    df['transaction_grade'] = pd.cut(
        df['net_spar_percentile'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['F', 'D', 'C', 'B', 'A']
    )

    # 2. VALUE TIER for FAAB efficiency
    df['faab_value_tier'] = pd.cut(
        df['spar_efficiency'].fillna(0),
        bins=[-float('inf'), 0, 1, 3, 5, float('inf')],
        labels=['Overpay', 'Fair', 'Good Value', 'Great Value', 'Steal']
    )

    # 3. DROP REGRET SCORE
    df['drop_regret'] = np.where(
        df['transaction_type'] == 'drop',
        df['player_spar_ros_total'].fillna(0),
        np.nan
    )
    df['drop_regret_tier'] = pd.cut(
        df['drop_regret'].fillna(0),
        bins=[0, 10, 30, 50, 100, float('inf')],
        labels=['No Regret', 'Minor Regret', 'Some Regret', 'Big Regret', 'Disaster']
    )

    # 4. PICKUP TYPE based on source
    df['pickup_type'] = df['source_type'].map({
        'waivers': 'Waiver Claim',
        'freeagents': 'Free Agent',
        'team': 'Trade'
    }).fillna('Other')

    # 5. TIMING CATEGORY
    df['timing_category'] = pd.cut(
        df['week'].fillna(1),
        bins=[0, 4, 10, 14, 18],
        labels=['Early Season', 'Mid Season', 'Late Season', 'Playoffs']
    )

    return df
```

### Priority 2: Create Transaction Summary Table

Pre-compute transaction-level summaries for faster UI:

```python
# New transformation: create_transaction_summaries.py

def create_transaction_summaries(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per transaction with grouped metrics."""

    # Group by transaction_id and manager
    summaries = transactions_df.groupby(['transaction_id', 'manager']).agg({
        'year': 'first',
        'week': 'first',
        'faab_bid': 'first',

        # Added players (type='add')
        'player_name': lambda x: ', '.join(
            transactions_df.loc[x.index[transactions_df.loc[x.index, 'transaction_type'] == 'add'], 'player_name'].dropna()
        ),
        'manager_spar_ros_managed': lambda x: x[
            transactions_df.loc[x.index, 'transaction_type'] == 'add'
        ].sum(),

        # Dropped players (type='drop')
        # ... similar aggregation

    }).reset_index()

    return summaries
```

### Priority 3: UI/UX Improvements

| Issue | File | Recommendation |
|-------|------|----------------|
| Heavy computation on load | `weekly_add_drop.py` | Use pre-computed summaries |
| No transaction grade display | Multiple files | Add grade column from source |
| Missing "regret" analysis | `weekly_add_drop.py` | Add drop regret visualization |
| No injury context | Transaction display | Join injury data for context |
| Trade tree missing | `trade_by_trade_summary_data.py` | Add linked transaction visualization |

### Priority 4: New Components to Add

1. **Transaction Report Card** - Summary of manager's transaction performance
2. **Drop Regret Analysis** - What did dropped players do after you cut them?
3. **FAAB Budget Tracker** - Visual of FAAB spending over season
4. **Trade Network Visualization** - Who trades with whom?
5. **Injury Response Analysis** - How well do managers handle injuries via transactions?

### Priority 5: Performance Optimizations

```python
# In transaction_data.py - add filter pushdown
query = f"""
    SELECT {cols_str}
    FROM {T['transactions']}
    WHERE manager IS NOT NULL
      AND transaction_type IN ('add', 'drop', 'trade')
    ORDER BY year DESC, week DESC
"""

# Consider creating a materialized view for transaction summaries
# This would dramatically speed up the weekly view
```

---

## Appendix: File Locations

### Data Pipeline Scripts
```
fantasy_football_data_scripts/multi_league/
├── data_fetchers/
│   ├── transactions_v2.py        # Yahoo API fetcher
│   └── aggregators.py            # File aggregation
├── transformations/
│   └── transaction_enrichment/
│       ├── player_to_transactions_v2.py  # Performance join
│       ├── transaction_value_metrics_v3.py # SPAR calculation
│       ├── fix_unknown_managers.py        # Manager repair
│       └── modules/
│           └── transaction_spar_calculator.py # Core SPAR logic
```

### UI Components
```
KMFFLApp/streamlit_ui/
├── md/tab_data_access/transactions/
│   ├── transaction_data.py       # Data loader
│   ├── summary_data.py           # Summary stats
│   └── combined.py               # Entry point
├── tabs/transactions/
│   ├── __init__.py
│   ├── transactions_adds_drops_trades_overview.py  # Main hub
│   ├── add_drop_overview.py      # Add/Drop router
│   ├── weekly_add_drop.py        # Weekly view
│   ├── season_add_drop.py        # Season stats
│   ├── career_add_drop.py        # Career stats
│   ├── trade_overview.py         # Trades router
│   ├── trade_by_trade_summary_data.py  # Trade details
│   ├── season_trade_data.py      # Season trades
│   ├── career_trade_data.py      # Career trades
│   ├── traded_player_data.py     # Player focus
│   ├── combo_transaction_overview.py   # Combined view
│   ├── weekly_combo_transactions.py
│   ├── season_combo_transactions.py
│   ├── career_combo_transactions.py
│   └── ENRICHMENT_COLUMNS_REFERENCE.md # Column docs
```

### Data Files
```
fantasy_football_data/KMFFL/
├── transactions.parquet          # Canonical transaction file
├── transactions.csv              # CSV backup
└── transaction_data/
    ├── transactions_year_2014.parquet   # Per-year files
    ├── transactions_year_2015.parquet
    └── ...
```

---

## Comparison: Transactions vs Draft Pipeline

| Aspect | Draft | Transactions |
|--------|-------|--------------|
| **Complexity** | Medium | High |
| **SPAR Metrics** | Dual (player/manager) | Dual + NET |
| **UI Calculations** | Minimal | Significant |
| **Pre-computed Metrics** | Good | Needs improvement |
| **Schema Size** | 67 columns | 78 columns |
| **Engagement Metrics** | Needs grades | Needs grades + regret |

---

*This documentation is auto-generated and should be updated when pipeline changes are made.*
