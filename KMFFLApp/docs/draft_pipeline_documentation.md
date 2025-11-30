# Draft Pipeline Documentation

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

The Draft Analysis module provides comprehensive insights into fantasy football draft performance across all league history. It tracks every draft pick, calculates value metrics like SPAR (Season Points Above Replacement), and helps managers understand their drafting tendencies and success rates.

### Key Capabilities

- **Historical Draft Data**: Complete draft history from league inception
- **Value Analysis**: SPAR-based metrics comparing draft cost to actual production
- **Keeper Economics**: Track keeper value and next-year projections
- **Manager Insights**: Identify drafting patterns, strengths, and weaknesses
- **Performance Rankings**: See how picks performed vs expectations

### Quick Stats

| Metric | Value |
|--------|-------|
| Total Draft Picks | 13,962+ |
| Data Columns | 90+ |
| Seasons Covered | All league history |
| Update Frequency | After each draft |

### New in v3 (November 2024)

- **Draft Grades (A-F)**: Letter grades based on SPAR percentile within position-year
- **Value Tiers**: Steal/Good Value/Fair/Overpay/Bust classification
- **Breakout/Bust Flags**: Identify late-round gems and early-round disappointments
- **Manager Draft Grades**: Manager-level performance with all-time percentile ranking
- **Keeper Economics**: Configurable keeper price rules via league context
- **Auto Draft Type Detection**: Handles mixed auction/snake datasets per year

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
│  draft_data_v2.py                                              │
│  ├── Authenticates via OAuth2                                  │
│  ├── Fetches draft picks (pick, round, cost, player)          │
│  ├── Fetches draft analysis (ADP, percent drafted)            │
│  ├── Fetches team/player mappings                              │
│  └── Outputs: draft_data_{year}.parquet (per season)          │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    PHASE 2: AGGREGATION                        │
│                                                                │
│  aggregators.py → normalize_draft_parquet()                    │
│  ├── Combines all yearly files                                 │
│  ├── Normalizes data types (IDs → string, years → Int64)      │
│  ├── Standardizes column names                                 │
│  └── Outputs: draft.parquet (canonical file)                  │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                 PHASE 3: ENRICHMENT (Multi-Pass)               │
│                                                                │
│  PASS 1: player_to_draft_v2.py                                 │
│  ├── Joins season performance stats from player.parquet       │
│  ├── Adds: total_fantasy_points, season_ppg, games_played     │
│  ├── Adds: position/overall rankings                           │
│  └── Adds: ROI metrics (points_per_dollar, points_per_pick)   │
│                                                                │
│  PASS 2: draft_value_metrics_v3.py + spar_calculator.py        │
│  ├── Calculates replacement levels by position                 │
│  ├── Computes SPAR (player_spar and manager_spar)             │
│  ├── Adds: replacement_ppg, pgvor, cost_norm, draft_roi       │
│  └── Adds: spar_per_dollar, pick_savings, cost_savings        │
│                                                                │
│  PASS 3: keeper_economics_v2.py                                │
│  ├── Calculates keeper prices using league formula            │
│  ├── Tracks kept_next_year flag                                │
│  └── Assigns cost_bucket tiers                                 │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                                │
│                                                                │
│  draft.parquet                                                 │
│  ├── 67 columns of draft + performance data                   │
│  ├── ~14,000 rows (all draft-eligible players)                │
│  └── Includes both drafted and undrafted players              │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI LAYER                          │
│                                                                │
│  Data Access: md/tab_data_access/draft/                        │
│  ├── draft_data.py - Optimized column selection (55 cols)     │
│  ├── combined.py - Entry point for data loading               │
│  └── Caching: 600s TTL with st.cache_data                     │
│                                                                │
│  UI Components: tabs/draft_data/                               │
│  ├── draft_data_overview.py - Main hub (8 tabs)               │
│  ├── draft_summary.py - Filterable draft table                │
│  ├── draft_scoring_outcomes.py - Performance charts           │
│  ├── draft_overviews.py - Position pricing analysis           │
│  ├── draft_optimizer.py - Lineup optimizer                    │
│  ├── draft_preferences.py - Manager tendencies                │
│  ├── career_draft_stats.py - Player draft history             │
│  └── graphs/ - Visualization components                        │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Key Files Reference

| File | Location | Purpose |
|------|----------|---------|
| `draft_data_v2.py` | `data_fetchers/` | Fetches raw data from Yahoo API |
| `aggregators.py` | `data_fetchers/` | Combines yearly files, normalizes types |
| `player_to_draft_v2.py` | `transformations/draft_enrichment/` | Joins player performance stats |
| `draft_value_metrics_v3.py` | `transformations/draft_enrichment/` | Calculates SPAR and value metrics |
| `spar_calculator.py` | `transformations/draft_enrichment/modules/` | Core SPAR calculation logic |
| `keeper_economics_v2.py` | `transformations/draft_enrichment/` | Keeper price calculations |
| `draft_data.py` | `streamlit_ui/md/tab_data_access/draft/` | UI data loader |
| `draft_data_overview.py` | `streamlit_ui/tabs/draft_data/` | Main UI component |

---

## Data Schema Reference

### Core Columns (67 Total)

#### Identifiers & Keys

| Column | Type | Description |
|--------|------|-------------|
| `year` | Int64 | Draft year |
| `pick` | Float64 | Overall pick number |
| `round` | Float64 | Draft round |
| `team_key` | String | Yahoo team identifier |
| `manager` | String | Manager name |
| `yahoo_player_id` | String | Yahoo player identifier |
| `league_id` | String | League identifier |
| `player_year` | String | Composite key (player + year) |
| `manager_year` | String | Composite key (manager + year) |
| `draft_type` | String | "auction" or "snake" |

#### Player Information

| Column | Type | Description |
|--------|------|-------------|
| `player` | String | Player full name |
| `yahoo_position` | String | Position from Yahoo (QB, RB, WR, TE, K, DEF) |
| `position` | String | Standardized position |
| `nfl_team` | String | NFL team abbreviation |

#### Draft Cost & ADP

| Column | Type | Description |
|--------|------|-------------|
| `cost` | Float64 | Auction cost (or null for snake) |
| `cost_norm` | Float64 | Normalized cost (comparable across draft types) |
| `cost_bucket` | Float64 | Position tier (1-3 = elite, 4-6 = good, etc.) |
| `avg_pick` | Float64 | Yahoo ADP (average draft position) |
| `avg_round` | Float64 | Yahoo average round |
| `avg_cost` | Float64 | Yahoo average auction cost |
| `percent_drafted` | Float64 | Percent of leagues where drafted |
| `preseason_avg_pick` | Float64 | Preseason ADP |
| `preseason_avg_round` | Float64 | Preseason average round |
| `preseason_avg_cost` | Float64 | Preseason average cost |
| `preseason_percent_drafted` | Float64 | Preseason draft percentage |

#### Keeper Information

| Column | Type | Description |
|--------|------|-------------|
| `is_keeper_status` | Int64 | 1 = keeper, 0 = drafted |
| `is_keeper_cost` | Int64 | Keeper cost if applicable |
| `kept_next_year` | Int64 | 1 = player was kept following season |

#### Season Performance (from player.parquet join)

| Column | Type | Description |
|--------|------|-------------|
| `total_fantasy_points` | Float64 | Total season points |
| `season_ppg` | Float64 | Points per game (all games) |
| `games_played` | Float64 | Total games with stats |
| `games_with_points` | Float64 | Games with >0 points |
| `best_game` | Float64 | Highest single-game score |
| `worst_game` | Float64 | Lowest single-game score |
| `season_std` | Float64 | Standard deviation (consistency) |
| `weeks_rostered` | Float64 | Weeks on a roster |
| `weeks_started` | Float64 | Weeks in starting lineup |

#### SPAR Metrics (Dual: Player vs Manager)

| Column | Type | Description |
|--------|------|-------------|
| `replacement_ppg` | Float64 | Position replacement-level PPG |
| `player_spar` | Float64 | SPAR for all games played (talent metric) |
| `manager_spar` | Float64 | SPAR for started games only (usage metric) |
| `spar` | Float64 | Legacy alias for manager_spar |
| `player_ppg` | Float64 | PPG for all games |
| `manager_ppg` | Float64 | PPG for started games |
| `player_pgvor` | Float64 | Per-game VOR (all games) |
| `manager_pgvor` | Float64 | Per-game VOR (started games) |
| `pgvor` | Float64 | Legacy alias for manager_pgvor |

#### ROI & Value Metrics

| Column | Type | Description |
|--------|------|-------------|
| `draft_roi` | Float64 | SPAR / cost_norm (return on investment) |
| `spar_per_dollar` | Float64 | SPAR / actual cost |
| `spar_per_dollar_norm` | Float64 | Same as draft_roi |
| `spar_per_pick` | Float64 | SPAR / pick number |
| `spar_per_round` | Float64 | SPAR / round number |
| `points_per_dollar` | Float64 | Total points / cost |
| `points_per_pick` | Float64 | Total points / pick |
| `value_over_replacement` | Float64 | Points above position average |
| `draft_position_delta` | Float64 | ADP - actual pick (positive = value) |

#### Rankings

| Column | Type | Description |
|--------|------|-------------|
| `season_overall_rank` | Int64 | Overall finish rank |
| `season_position_rank` | Int64 | Position finish rank |
| `total_position_players` | Float64 | Total players at position |
| `price_rank_within_position` | Int64 | Cost rank within position |
| `pick_rank_within_position` | Int64 | Pick rank within position |
| `spar_per_dollar_rank` | Float64 | SPAR/$ rank within position |
| `spar_per_pick_rank` | Float64 | SPAR/pick rank within position |
| `price_rank_vs_finish_rank` | Int64 | Cost rank - finish rank (+ = outperformed) |
| `pick_rank_vs_finish_rank` | Int64 | Pick rank - finish rank (+ = outperformed) |

#### Savings Metrics

| Column | Type | Description |
|--------|------|-------------|
| `pick_savings` | Float64 | ADP - actual pick (snake value) |
| `cost_savings` | Float64 | Avg cost - actual cost (auction value) |
| `savings` | Float64 | Unified metric (cost or pick based on draft type) |

---

## UI Components

### Current Implementation

The Draft Analysis hub (`draft_data_overview.py`) contains **9 tabs**:

---

### Tab 1: Summary (draft_summary.py)

**What It Shows:**
- Complete draft pick table with all historical data
- Quick metrics: Total picks, seasons, avg cost, avg PPG, keeper count
- Filterable by year, manager, position, player
- Sortable by recent, points, PPG, cost, or overall rank

**Current Features:**
- Multi-select filters for years, managers, positions
- Checkbox to include/exclude keepers
- Minimum cost filter
- Export to CSV

**Suggested Additions:**
- [ ] Add "Draft Grade" column (A-F based on SPAR percentile)
- [ ] Add "Value Tier" column (Steal/Good/Fair/Reach/Bust)
- [ ] Quick filter presets (e.g., "My Best Picks", "Biggest Busts")
- [ ] Comparison mode to see two managers side-by-side
- [ ] Mobile-responsive column hiding

---

### Tab 2: Performance (draft_scoring_outcomes.py)

**What It Shows:**
- Scatter plot: Draft position vs actual performance
- Box plot: Value distribution by position
- Bar chart: Top 20 value picks (outperformed draft position)
- Data table with cost rank, points rank, value differential

**Current Features:**
- Interactive Plotly charts with hover details
- Color coding by position
- Diagonal "perfect prediction" line on scatter
- Filters for year, position, manager, specific players

**Suggested Additions:**
- [ ] Add "Biggest Disappointments" bar chart (inverse of value picks)
- [ ] Year-over-year performance trend for selected players
- [ ] Position-specific breakdowns (e.g., RB1 vs RB2 performance)
- [ ] Manager performance comparison chart
- [ ] "Draft Day vs Season End" rank comparison table

---

### Tab 3: Value (inline in draft_data_overview.py)

**What It Shows:**
- Summary metrics: Avg SPAR/$, Best SPAR/$, Median cost
- Value by position table (Avg/Median/Max SPAR/$)
- Top 20 best value picks with Manager SPAR

**Current Features:**
- Uses Manager SPAR (actual value captured while rostered)
- Calculates draft_roi on-the-fly if not present
- Position grouping with aggregated stats

**Suggested Additions:**
- [ ] Value trends over time (are drafts getting more efficient?)
- [ ] "Hidden Gems" section (late round picks with high SPAR)
- [ ] Keeper value comparison (keeper SPAR/$ vs drafted SPAR/$)
- [ ] Position scarcity analysis (when to draft each position)
- [ ] Draft capital efficiency leaderboard by manager

---

### Tab 4: Optimizer (draft_optimizer.py)

**What It Shows:**
- Build optimal lineups given constraints
- Position requirements and budget limits

**Current Features:**
- Lineup builder with position slots
- Budget constraint handling

**Suggested Additions:**
- [ ] Historical "best possible draft" calculator
- [ ] "Redraft" mode - rebuild a past draft optimally
- [ ] Auction value calculator for upcoming drafts
- [ ] Trade value estimator based on draft capital
- [ ] Keeper recommendation engine

---

### Tab 5: Trends (draft_preferences.py)

**What It Shows:**
- Manager draft tendencies and preferences
- How spending patterns evolve over time

**Current Features:**
- Manager preference analysis
- Historical pattern detection

**Sub-visualizations (via graphs/):

#### Spending Trends (draft_spending_trends.py)
- Line chart: Spending by position over time
- Manager vs league average comparison
- Keeper vs drafted spending breakdown
- Summary statistics table with SPAR metrics

#### Market Trends (draft_market_trends.py)
- Position market value changes
- League-wide spending patterns

**Suggested Additions:**
- [ ] Manager "draft personality" profile (aggressive/conservative, position priorities)
- [ ] Draft strategy clustering (identify similar drafting styles)
- [ ] Prediction model: What will this manager draft next?
- [ ] Position run analysis (when positions get drafted in bunches)
- [ ] "If you like X, you might draft Y" recommendations

---

### Tab 6: Pricing (draft_overviews.py)

**What It Shows:**
- Average draft prices by position and rank
- Keeper prices vs drafted prices
- Cost tiers within positions

**Current Features:**
- Year range filter
- Position rank groupings (RB1, RB2, etc.)
- Separate tables for drafted vs keepers
- SPAR metrics integrated when available

**Suggested Additions:**
- [ ] Price inflation/deflation tracker over years
- [ ] Position premium calculator (how much more does RB1 cost vs RB2?)
- [ ] Salary cap planning tool
- [ ] Historical price comparison for specific players
- [ ] "Fair value" calculator based on historical data

---

### Tab 7: Career (career_draft_stats.py)

**What It Shows:**
- Long-term player draft history
- How players have been valued over time

**Current Features:**
- Player career timeline
- Multi-year draft data

**Suggested Additions:**
- [ ] Player "draft stock" chart over career
- [ ] Keeper retention analysis (who gets kept most?)
- [ ] Dynasty value tracker
- [ ] "Rising" and "Falling" players lists
- [ ] Age-adjusted value analysis

---

### Tab 8: Keeper Analysis (graphs/draft_keeper_analysis.py)

**What It Shows:**
- Keeper vs drafted player performance comparison
- Box plots: SPAR/$ and SPAR distribution
- Position breakdown of keeper performance
- Top/worst keeper value tables

**Current Features:**
- Year selection (all years or specific)
- Side-by-side comparison metrics
- Interactive Plotly box plots
- Position-level aggregation

**Suggested Additions:**
- [ ] Keeper ROI over time (are keepers getting better/worse?)
- [ ] "Should have been kept" analysis (best undrafted players from prior year)
- [ ] Keeper price vs production scatter plot
- [ ] Optimal keeper strategy recommendations
- [ ] Multi-year keeper tracking (follow a player through keeper years)

---

### Tab 9: Manager Grades (manager_draft_grades.py) - NEW

**What It Shows:**
- Manager-level draft performance grades (A-F)
- All-time percentile ranking for cross-year comparison
- Year-over-year grade trends
- Career draft summary

**Current Features:**
- **Leaderboard View**: Ranked by all-time percentile, shows grade, score, SPAR, hit rate
- **Year-over-Year View**: Grade heatmap across years, score trend line chart
- **Career Summary View**: Aggregate stats per manager (avg/best/worst score, career SPAR)
- Filter by year and manager
- Grade distribution charts
- Color-coded grades (A=green through F=red)

**Key Metrics Displayed:**
- `manager_draft_grade`: A-F letter grade based on all-time percentile
- `manager_draft_score`: Weighted pick quality score (37-82 range)
- `manager_draft_percentile_alltime`: Percentile across ALL drafts ever (for cross-year comparison)
- `manager_total_spar`: Sum of SPAR for all picks
- `manager_hit_rate`: Percentage of picks with positive SPAR

**Suggested Additions:**
- [ ] Draft style analysis (position preferences, price ranges)
- [ ] Head-to-head draft comparison between two managers
- [ ] "Draft personality" classification (aggressive/conservative)
- [ ] Best/worst draft deep dive

---

### Additional Graph Components (graphs/)

| File | Purpose |
|------|---------|
| `draft_round_efficiency.py` | ROI analysis by draft round |
| `draft_position_heatmap.py` | Visual heatmap of position drafting |
| `draft_market_trends.py` | Market-wide trend analysis |
| `draft_spending_trends.py` | Spending pattern visualization |
| `draft_keeper_analysis.py` | Keeper performance analysis |
| `draft_value_overview.py` | Value distribution overview |

---

## Feature Summaries

### For Homepage

#### Draft Analysis - Quick Summary

> **Understand Your Draft Performance**
>
> The Draft Analysis module tracks every pick in league history and calculates how well each selection performed. Using advanced metrics like SPAR (Season Points Above Replacement), you can see which picks were steals, which were busts, and how your drafting strategy compares to the rest of the league.
>
> **Key Features:**
> - Complete draft history with performance outcomes
> - Value analysis showing ROI on every pick
> - Manager tendencies and draft patterns
> - Keeper vs drafted player comparisons

---

### For About Page

#### Draft Analysis - Detailed Description

> **What is Draft Analysis?**
>
> The Draft Analysis module provides comprehensive insights into fantasy football drafts. Every pick from league history is tracked, enriched with season performance data, and analyzed using sophisticated value metrics.
>
> **How It Works:**
>
> 1. **Data Collection**: Draft data is fetched from Yahoo Fantasy API after each draft, including pick order, auction costs, and player information.
>
> 2. **Performance Matching**: Each draft pick is matched with that player's actual season performance - points scored, games played, consistency metrics.
>
> 3. **Value Calculation**: We calculate SPAR (Season Points Above Replacement) to measure true value. This compares each player's production to what a "replacement level" player at that position would have scored.
>
> 4. **Dual Metrics**: We track both "Player SPAR" (total production) and "Manager SPAR" (production while in your starting lineup) to distinguish between player talent and manager usage.
>
> **Key Metrics Explained:**
>
> | Metric | What It Means |
> |--------|---------------|
> | **SPAR** | Season Points Above Replacement - how many more points than a waiver-wire pickup |
> | **SPAR/$** | Value efficiency - SPAR divided by draft cost |
> | **Draft ROI** | Return on investment - normalized value metric |
> | **Value Tier** | Steal/Good/Fair/Reach/Bust classification |
> | **Draft Grade** | A-F letter grade based on SPAR percentile |
>
> **What You Can Learn:**
>
> - Which draft picks delivered the best value
> - Your drafting tendencies (positions you favor, price ranges you target)
> - How keeper selections compare to fresh draft picks
> - Historical pricing trends for roster planning
> - Which managers have the best draft track records

---

### Section-by-Section Summaries (For Navigation/Tooltips)

#### Summary Tab
> "View all draft picks with filters for year, manager, and position. See performance stats, SPAR values, and rankings at a glance."

#### Performance Tab
> "Visualize how draft position relates to actual performance. Find your biggest steals and disappointments with interactive charts."

#### Value Tab
> "Analyze draft efficiency using SPAR per dollar. See which positions offer the best value and who makes the smartest picks."

#### Optimizer Tab
> "Build optimal lineups within budget constraints. Plan your draft strategy with historical performance data."

#### Trends Tab
> "Discover drafting patterns over time. See how spending on positions evolves and compare manager strategies."

#### Pricing Tab
> "Reference guide for draft costs by position and rank. Know fair value before your next draft."

#### Career Tab
> "Track individual players across multiple drafts. See how player values change over their careers."

#### Keeper Analysis Tab
> "Compare keeper performance to fresh draft picks. Make informed decisions about who to keep."

---

## Recommendations & Roadmap

### Priority 1: Add Engagement Metrics to Source Table

Add these columns to `draft_value_metrics_v3.py`:

```python
# Draft Grade (A-F based on SPAR percentile within position/year)
df['draft_grade'] = pd.cut(
    df.groupby(['year', 'position'])['manager_spar'].transform(
        lambda x: x.rank(pct=True) * 100
    ),
    bins=[0, 20, 40, 60, 80, 100],
    labels=['F', 'D', 'C', 'B', 'A']
)

# Value Tier (Steal/Good/Fair/Reach/Bust)
df['value_tier'] = pd.cut(
    df['price_rank_vs_finish_rank'].fillna(0),
    bins=[-100, -5, -2, 2, 5, 100],
    labels=['Bust', 'Reach', 'Fair', 'Good', 'Steal']
)

# Player Type (consistency-based)
df['player_type'] = pd.cut(
    (df['season_std'] / df['season_ppg'].clip(lower=0.1)).fillna(0.5),
    bins=[0, 0.3, 0.5, 100],
    labels=['Steady', 'Normal', 'Boom/Bust']
)

# Breakout Flag
df['is_breakout'] = ((df['round'] >= 8) & (df['season_position_rank'] <= 10)).astype(int)

# Bust Flag
df['is_bust'] = ((df['round'] <= 3) & (df['season_position_rank'] > df['total_position_players'] / 2)).astype(int)
```

### Priority 2: Performance Optimizations

```python
# In draft_data.py - filter undrafted in SQL
query = f"""
    SELECT {cols_str}
    FROM {T['draft']}
    WHERE manager IS NOT NULL
      AND manager != ''
      AND cost > 0
    ORDER BY year DESC, round, pick
"""
```

### Priority 3: UI Consolidation

Reduce from 8 tabs to 5:

| New Tab | Contains |
|---------|----------|
| Overview | Summary + Quick Stats + Report Card |
| Performance | Scoring Outcomes + Charts |
| Value | Value Analysis + Pricing + Optimizer |
| Trends | Preferences + Spending + Market Trends |
| Keepers | Keeper Analysis + Career (keeper-focused) |

### Priority 4: New Components

1. **Draft Report Card** - Visual summary with grades
2. **Manager Comparison Tool** - Side-by-side analysis
3. **Position Radar Chart** - Spending patterns visualization
4. **Draft Stock Tracker** - Player value over time

### Priority 5: Mobile & Dark Mode

- Add CSS custom properties for theming
- Implement responsive column hiding
- Test on mobile devices

---

## Appendix: File Locations

### Data Pipeline Scripts
```
fantasy_football_data_scripts/multi_league/
├── data_fetchers/
│   ├── draft_data_v2.py          # Yahoo API fetcher
│   └── aggregators.py            # File aggregation
├── transformations/
│   └── draft_enrichment/
│       ├── player_to_draft_v2.py      # Performance join
│       ├── draft_value_metrics_v3.py  # SPAR calculation
│       ├── keeper_economics_v2.py     # Keeper pricing
│       └── modules/
│           └── spar_calculator.py     # Core SPAR logic
```

### UI Components
```
KMFFLApp/streamlit_ui/
├── md/tab_data_access/draft/
│   ├── draft_data.py             # Data loader
│   └── combined.py               # Entry point
├── tabs/draft_data/
│   ├── __init__.py
│   ├── draft_data_overview.py    # Main hub
│   ├── draft_summary.py          # Summary table
│   ├── draft_scoring_outcomes.py # Performance charts
│   ├── draft_overviews.py        # Pricing analysis
│   ├── draft_optimizer.py        # Lineup optimizer
│   ├── draft_preferences.py      # Manager tendencies
│   ├── career_draft_stats.py     # Player history
│   └── graphs/
│       ├── draft_value_overview.py
│       ├── draft_spending_trends.py
│       ├── draft_keeper_analysis.py
│       ├── draft_market_trends.py
│       ├── draft_round_efficiency.py
│       └── draft_position_heatmap.py
```

### Data Files
```
fantasy_football_data/KMFFL/
├── draft.parquet                 # Canonical draft file
├── draft.csv                     # CSV backup
└── draft_data/
    ├── draft_data_2014.parquet   # Per-year files
    ├── draft_data_2015.parquet
    └── ...
```

---

*This documentation is auto-generated and should be updated when pipeline changes are made.*
