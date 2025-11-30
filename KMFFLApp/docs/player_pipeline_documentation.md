# Player Pipeline Documentation

> **Last Updated:** November 2024
> **Data Sources:** player.parquet (280 cols, 182,650 rows), players_by_year.parquet (244 cols, 152,002 rows)
> **UI Locations:** `streamlit_ui/tabs/player_stats/`, `streamlit_ui/tabs/team_stats/`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Data Pipeline](#data-pipeline)
3. [Source Tables](#source-tables)
4. [UI Components - Player Stats](#ui-components---player-stats)
5. [UI Components - Team Stats](#ui-components---team-stats)
6. [Recommendations & Roadmap](#recommendations--roadmap)

---

## Executive Summary

The Player Pipeline is the **most data-rich section** of the fantasy football app, combining:
- Yahoo Fantasy API player data
- NFL play-by-play advanced stats (Next Gen Stats)
- Defense/Special Teams statistics
- SPAR (Season Points Above Replacement) metrics
- Roster/lineup context from matchups

### Key Capabilities

- **Weekly Player Stats**: Every player performance, every week
- **Season Aggregations**: Season totals by player and position
- **Career Stats**: All-time player performance
- **Team Composition**: Aggregated stats by manager and position group
- **Advanced Metrics**: EPA, CPOE, Target Share, WOPR, RACR, etc.
- **Dual SPAR Metrics**: player_spar (talent) vs manager_spar (usage)

### Data Volume

| Table | Columns | Rows | Purpose |
|-------|---------|------|---------|
| `player.parquet` | 280 | 182,650 | Weekly player performances |
| `players_by_year.parquet` | 244 | 152,002 | Season-aggregated player stats |

---

## Data Pipeline

### Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           DATA FETCHING                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   yahoo_fantasy_data.py          defense_stats.py       nfl_offense_stats.py
│   ┌─────────────────┐           ┌──────────────┐       ┌───────────────────┐
│   │ Yahoo API       │           │ DST Data     │       │ NFLverse Data     │
│   │ - Players       │           │ - Team DEF   │       │ - Play-by-play    │
│   │ - Rosters       │           │ - DST points │       │ - Advanced stats  │
│   │ - Lineups       │           │ - Sacks/INTs │       │ - EPA/CPOE        │
│   │ - Stats         │           │ - Yards allow│       │ - Target share    │
│   └────────┬────────┘           └──────┬───────┘       └─────────┬─────────┘
│            │                           │                         │
│            └───────────────────────────┴─────────────────────────┘
│                                        │
│                            combine_dst_to_nfl.py
│                            ┌───────────────────┐
│                            │ Merge DST + NFL   │
│                            │ offensive stats   │
│                            └─────────┬─────────┘
│                                      │
└──────────────────────────────────────┼─────────────────────────────────────┘
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         TRANSFORMATIONS                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   player_enrichment/                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │  player_stats_v2.py (923 lines) - Main enrichment                   │  │
│   │  ├── add_advanced_stats() - EPA, CPOE, target share, etc.          │  │
│   │  ├── add_replacement_level_columns() - Dual SPAR metrics           │  │
│   │  ├── add_position_ranks() - Season rankings by position            │  │
│   │  ├── add_optimal_lineup_flags() - Lineup optimization              │  │
│   │  └── add_consistency_metrics() - Boom/bust, CV, std dev            │  │
│   │                                                                     │  │
│   │  replacement_level_v2.py - SPAR calculation                         │  │
│   │  ├── calculate_replacement_level() - Position-specific baseline    │  │
│   │  ├── calculate_player_spar() - Total production above replacement  │  │
│   │  └── calculate_manager_spar() - Production while rostered          │  │
│   │                                                                     │  │
│   │  matchup_to_player_v2.py - Fantasy matchup context                  │  │
│   │  ├── Add opponent, team_points, opponent_points                    │  │
│   │  ├── Add win/loss, is_playoffs, championship flags                 │  │
│   │  └── Add lineup position (started vs bench)                        │  │
│   │                                                                     │  │
│   │  transactions_to_player_v2.py - Transaction history                 │  │
│   │  └── Add roster acquisition context                                │  │
│   │                                                                     │  │
│   │  draft_to_player_v2.py - Draft context                              │  │
│   │  └── Add draft round, pick, cost                                   │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                     │
└──────────────────────────────────────┼─────────────────────────────────────┘
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT TABLES                                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌─────────────────────┐          ┌─────────────────────────┐            │
│   │  player.parquet     │          │  players_by_year.parquet │            │
│   │  280 cols, 182K rows│          │  244 cols, 152K rows     │            │
│   │  Weekly granularity │   ──►    │  Season granularity      │            │
│   │  Most detailed      │          │  Pre-aggregated          │            │
│   └─────────────────────┘          └─────────────────────────┘            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Data Fetching Scripts

| Script | Purpose | Key Data |
|--------|---------|----------|
| `yahoo_fantasy_data.py` | Yahoo Fantasy API | Rosters, lineups, player stats |
| `defense_stats.py` | DST statistics | Team defense, points allowed |
| `nfl_offense_stats.py` | NFLverse data | Play-by-play, advanced stats |
| `combine_dst_to_nfl.py` | Merge NFL + DST | Combined player data |

### Transformation Scripts

| Script | Purpose | Key Transformations |
|--------|---------|---------------------|
| `player_stats_v2.py` | Main enrichment | Advanced stats, SPAR, rankings |
| `replacement_level_v2.py` | SPAR calculation | Player/Manager SPAR metrics |
| `matchup_to_player_v2.py` | Fantasy context | Win/loss, playoffs, lineups |
| `transactions_to_player_v2.py` | Roster context | Add/drop history |
| `draft_to_player_v2.py` | Draft context | Round, pick, cost |

---

## Source Tables

### player.parquet (280 columns, 182,650 rows)

The weekly player performance table with every stat tracked.

#### Column Categories

| Category | Count | Examples |
|----------|-------|----------|
| **Identifiers** | 15 | player, yahoo_player_id, NFL_player_id, nfl_team |
| **Time** | 5 | year, week, cumulative_week, season_type |
| **Positions** | 4 | nfl_position, fantasy_position, yahoo_position |
| **Fantasy Points** | 8 | points, season_ppg, position_season_rank |
| **SPAR Metrics** | 6 | spar, player_spar, manager_spar |
| **Passing** | 25 | pass_yds, pass_td, pass_int, passing_epa, cpoe |
| **Rushing** | 20 | rush_yds, rush_td, carries, rushing_epa |
| **Receiving** | 25 | rec, rec_yds, rec_td, targets, target_share, wopr |
| **Kicking** | 15 | fg_made, fg_att, fg_pct, pat_made |
| **Defense/DST** | 40 | def_sacks, def_int, pts_allow, def_td |
| **Matchup Context** | 30 | manager, opponent, team_points, win, is_playoffs |
| **Lineup** | 10 | fantasy_position, started, optimal_player |
| **Advanced** | 50+ | epa, cpoe, pacr, racr, target_share, air_yards_share |

#### Key SPAR Columns (Dual Metric System)

```python
# Player SPAR - Total production vs replacement level
# "How valuable was this player to ANYONE who had them?"
'player_spar'        # Total SPAR produced all season
'player_spar_weekly' # Weekly SPAR contribution

# Manager SPAR - Production while on your roster
# "How much value did YOU get from this player?"
'manager_spar'        # SPAR while rostered by you
'manager_spar_weekly' # Weekly SPAR while on your team

# The difference? If you traded a player mid-season:
# - player_spar = their full season value
# - manager_spar = only the weeks they were on YOUR team
```

#### Advanced Stats (NFL Play-by-Play)

```python
# Passing Efficiency
'passing_epa'         # Expected Points Added on pass plays
'passing_cpoe'        # Completion % Over Expected
'pacr'                # Passer Air Conversion Ratio

# Receiving Usage
'target_share'        # % of team targets
'air_yards_share'     # % of team air yards
'wopr'                # Weighted Opportunity Rating
'racr'                # Receiver Air Conversion Ratio
'receiving_epa'       # Expected Points Added on receptions

# Rushing Efficiency
'rushing_epa'         # Expected Points Added on rushes
```

### players_by_year.parquet (244 columns, 152,002 rows)

**Purpose**: Pre-aggregated season-level player stats for faster UI queries.

#### What It Provides

- **54% Column Reduction**: 244 vs 280 columns in weekly table
- **17% Row Reduction**: 152K vs 182K rows (aggregated by player/year)
- **Faster Queries**: Pre-grouped by player+year eliminates runtime aggregation
- **Same Stats**: All important stats preserved via SUM/AVG aggregations

#### When to Use Each Table

| Use Case | Table | Reason |
|----------|-------|--------|
| Weekly player browser | `player.parquet` | Need week granularity |
| Head-to-head comparisons | `player.parquet` | Need specific matchup data |
| Season leaderboards | `players_by_year.parquet` | Pre-aggregated, faster |
| Career stats | `players_by_year.parquet` | Aggregate by player only |
| Team composition (manager) | `player.parquet` | Need lineup positions |

**Verdict**: `players_by_year.parquet` is **valuable** for season/career views where you don't need week granularity. It provides meaningful performance improvement for season leaderboards and career stats pages.

---

## UI Components - Player Stats

### Location: `streamlit_ui/tabs/player_stats/`

### Architecture

```
player_stats/
├── weekly_player_stats_optimized.py    # Main weekly view (4 tabs)
├── season_player_stats_optimized.py    # Main season view
├── career_player_stats_optimized.py    # Main career view
├── weekly_player_subprocesses/
│   ├── weekly_player_basic_stats.py    # Basic stat columns
│   ├── weekly_player_advanced_stats.py # Advanced metrics
│   ├── weekly_player_matchup_stats.py  # Fantasy matchup context
│   └── head_to_head.py                 # H2H comparison tool
├── season_player_subprocesses/
│   ├── season_player_basic_stats.py
│   ├── season_player_advanced_stats.py
│   ├── season_player_matchup_stats.py
│   └── optimal_lineup_season.py        # Season optimal lineups
├── career_player_subprocesses/
│   ├── career_player_basic_stats.py
│   ├── career_player_advanced_stats.py
│   ├── career_player_matchup_stats.py
│   └── optimal_lineup_career.py
├── graphs/
│   ├── player_graphs/
│   │   ├── player_scoring_graph.py     # Scoring trends
│   │   ├── player_consistency.py       # Boom/bust analysis
│   │   ├── boom_bust_distribution.py   # Score distribution
│   │   ├── weekly_heatmap.py           # Week-by-week heatmap
│   │   ├── player_radar_comparison.py  # Radar chart compare
│   │   └── player_card.py              # Player profile card
│   ├── spar_graphs/
│   │   ├── cumulative_spar.py          # SPAR accumulation
│   │   ├── spar_per_week.py            # Weekly SPAR trend
│   │   ├── manager_capture_rate.py     # Manager vs Player SPAR
│   │   ├── spar_vs_ppg_efficiency.py   # SPAR efficiency
│   │   └── spar_consistency_scatter.py # SPAR consistency
│   └── league_graphs/
│       ├── position_group_scoring.py   # Position comparisons
│       ├── position_spar_boxplot.py    # SPAR by position
│       └── manager_spar_leaderboard.py # Manager SPAR rankings
└── base/
    ├── smart_filters.py                # Filter panel component
    ├── table_display.py                # Data table with load-more
    ├── pagination.py                   # Legacy pagination
    └── optimal_lineup_display.py       # Optimal lineup viz
```

### Weekly Player Stats (Main View)

**4 Sub-Tabs:**

1. **Basic Stats** - Core fantasy stats (points, passing, rushing, receiving)
2. **Advanced Stats** - EPA, CPOE, target share, WOPR, RACR
3. **Matchup Stats** - Fantasy matchup context (rostered players only)
4. **Head-to-Head** - Compare two managers' lineups

**Features:**
- Smart filter panel with position-aware columns
- Load-more pagination (default 5000 rows)
- Column sorting (affects database query, not just display)
- Export to CSV
- Position-specific column sets

### Season/Career Views

- Same structure as weekly but with aggregated data
- Uses `players_by_year.parquet` for faster queries
- Career view aggregates across all years

### Graphs (12 Visualizations)

| Graph | What It Shows |
|-------|---------------|
| Player Scoring | Points over time trend |
| Player Consistency | Boom/bust patterns |
| Boom/Bust Distribution | Score histogram |
| Weekly Heatmap | Week-by-week performance grid |
| Player Radar | Multi-stat comparison radar |
| Player Card | Profile summary card |
| Cumulative SPAR | SPAR accumulation over season |
| SPAR Per Week | Weekly SPAR trend line |
| Manager Capture Rate | Manager SPAR / Player SPAR ratio |
| SPAR vs PPG | Efficiency scatter plot |
| Position Group Scoring | Position comparison bars |
| Manager SPAR Leaderboard | Manager SPAR rankings |

---

## UI Components - Team Stats

### Location: `streamlit_ui/tabs/team_stats/`

### Architecture

```
team_stats/
├── team_stats_overview.py              # Main entry point
├── weekly_team_stats.py                # Weekly team view
├── season_team_stats.py                # Season team view
├── career_team_stats.py                # Career team view
├── team_stats_visualizations.py        # Team graphs
├── weekly_team_subprocesses/
│   ├── weekly_team_basic_stats.py
│   ├── weekly_team_basic_stats_by_manager.py
│   └── weekly_team_advanced_stats.py
├── season_team_subprocesses/
│   ├── season_team_basic_stats.py
│   ├── season_team_basic_stats_by_manager.py
│   └── season_team_advanced_stats.py
├── career_team_subprocesses/
│   ├── career_team_basic_stats.py
│   ├── career_team_basic_stats_by_manager.py
│   └── career_team_advanced_stats.py
└── shared/
    ├── constants.py
    ├── theme.py
    ├── modern_styles.py
    ├── filters.py
    ├── table_formatting.py
    └── column_config.py
```

### What Team Stats Shows

**Aggregates player data by manager and position group.**

Three Data Views:
1. **By Position** - Stats grouped by manager + fantasy_position (QB, RB, WR, etc.)
2. **By Manager** - Total stats for each manager
3. **By Lineup Position** - Stats by actual lineup slot (QB1, RB1, RB2, FLEX, etc.)

**Time Horizons:**
- Weekly - Week-by-week team composition
- Season - Full season aggregates
- Career - All-time team stats

### Key Metrics Displayed

```python
# Team composition shows:
- Total points by position
- SPAR contribution by position
- Position group efficiency
- Passing/Rushing/Receiving splits
- Optimal lineup utilization
```

### Data Access (Team Stats)

```python
# Loads from player.parquet with runtime aggregation:
# - Filters: manager NOT NULL, fantasy_position NOT IN ('BN', 'IR')
# - Groups by: manager, year, week, fantasy_position
# - Aggregates: SUM(points), SUM(spar), SUM(stats)...

# Example columns (from weekly_team_data.py):
WEEKLY_TEAM_COLUMNS = [
    "manager", "year", "week", "fantasy_position",
    "SUM(CAST(points AS DOUBLE)) as points",
    "SUM(CAST(player_spar AS DOUBLE)) as player_spar",
    "SUM(CAST(manager_spar AS DOUBLE)) as manager_spar",
    # ... 60+ aggregated columns
]
```

---

## Recommendations & Roadmap

### Priority 1: Add Engagement Metrics to player.parquet

```python
def add_player_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add fun/engaging metrics for player displays."""

    # 1. PERFORMANCE GRADE (A-F based on points vs position median)
    df['performance_grade'] = df.groupby(['year', 'week', 'nfl_position'])['points'].transform(
        lambda x: pd.cut(
            x.rank(pct=True) * 100,
            bins=[0, 20, 40, 60, 80, 100],
            labels=['F', 'D', 'C', 'B', 'A']
        )
    )

    # 2. BOOM FLAG (top 10% of position that week)
    df['is_boom'] = df.groupby(['year', 'week', 'nfl_position'])['points'].transform(
        lambda x: (x >= x.quantile(0.9)).astype(int)
    )

    # 3. BUST FLAG (bottom 20% when started)
    df['is_bust'] = (
        (df['started'] == 1) &
        (df.groupby(['year', 'week', 'nfl_position'])['points'].transform(
            lambda x: x <= x.quantile(0.2)
        ))
    ).astype(int)

    # 4. CONSISTENCY TIER
    season_cv = df.groupby(['player', 'year'])['points'].transform(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 1
    )
    df['consistency_tier'] = pd.cut(
        season_cv,
        bins=[0, 0.3, 0.5, 0.7, 100],
        labels=['Elite', 'Steady', 'Variable', 'Boom/Bust']
    )

    # 5. LEAGUE WINNER FLAG (top 3 at position + on championship team)
    df['is_league_winner'] = (
        (df['position_season_rank'] <= 3) &
        (df['champion'] == 1)
    ).astype(int)

    # 6. SPAR TIER (for rostered players)
    df['spar_tier'] = pd.cut(
        df['manager_spar'].fillna(0),
        bins=[-100, 0, 5, 10, 20, 100],
        labels=['Negative', 'Replacement', 'Solid', 'Good', 'Elite']
    )

    return df
```

**New Columns:** `performance_grade`, `is_boom`, `is_bust`, `consistency_tier`, `is_league_winner`, `spar_tier`

### Priority 2: UI Enhancements

| Enhancement | Impact | Effort |
|-------------|--------|--------|
| Add "Player Card" profile view | High engagement | Medium |
| Add "Start/Sit Analyzer" | Utility | High |
| Add "Boom/Bust Meter" visualization | Fun | Low |
| Add "SPAR Capture Rate" explanation | Education | Low |
| Add position group heat maps | Visual appeal | Medium |

### Priority 3: Performance Optimizations

```python
# Current: Season view loads from players_by_year (good!)
# Current: Weekly view loads 123 columns (optimized from 280)

# Further optimizations:
# 1. Create team_by_week.parquet for team stats
#    - Pre-aggregate by manager + week
#    - Eliminates runtime GROUP BY
#    - Est. 5-10x faster for team stats

# 2. Add indexed lookup tables
#    - player_lookup.parquet (id → name, position, team)
#    - manager_seasons.parquet (manager → years active)

# 3. Lazy-load graph tabs
#    - Only calculate graph data when tab is clicked
#    - Reduces initial page load
```

### Priority 4: Schema Cleanup

```python
# Redundant columns to deprecate in player.parquet:
# - 'spar' → Use 'player_spar' or 'manager_spar'
# - Position aliasing is inconsistent

# Recommendation: Add compatibility layer in data access
# Keep both during transition, then deprecate old names
```

---

## Appendix: players_by_year.parquet Utility Assessment

### Is It Useful?

**Yes - Highly Useful for Season/Career Views**

| Metric | player.parquet | players_by_year.parquet | Improvement |
|--------|----------------|------------------------|-------------|
| Columns | 280 | 244 | 13% reduction |
| Rows | 182,650 | 152,002 | 17% reduction |
| Season Query | Runtime GROUP BY | Pre-aggregated | 5-10x faster |
| Career Query | 2x GROUP BY | 1x GROUP BY | 2-5x faster |

### When to Use

| Scenario | Use | Reason |
|----------|-----|--------|
| Weekly player browser | player.parquet | Need week granularity |
| Season leaderboards | players_by_year | Pre-aggregated |
| Career stats | players_by_year | Faster aggregation |
| Player page | BOTH | Weekly for detail, year for summary |
| Team stats | player.parquet | Need lineup positions |
| Head-to-head | player.parquet | Need specific matchup |

### Recommendation

**Keep `players_by_year.parquet`** - It provides meaningful performance benefits for season/career views. The UI already uses it appropriately (see `season_player_data.py`).

---

## Appendix: File Locations

### Data Fetching
```
fantasy_football_data_scripts/multi_league/data_fetchers/
├── yahoo_fantasy_data.py       # Yahoo API
├── defense_stats.py            # DST data
├── nfl_offense_stats.py        # NFLverse data
└── combine_dst_to_nfl.py       # Merge DST + offense
```

### Transformations
```
fantasy_football_data_scripts/multi_league/transformations/player_enrichment/
├── player_stats_v2.py          # Main enrichment (923 lines)
├── replacement_level_v2.py     # SPAR calculation
├── matchup_to_player_v2.py     # Matchup context
├── transactions_to_player_v2.py # Transaction context
└── draft_to_player_v2.py       # Draft context
```

### UI Components
```
KMFFLApp/streamlit_ui/tabs/player_stats/    # Player stats UI
KMFFLApp/streamlit_ui/tabs/team_stats/      # Team stats UI
```

### Data Access
```
KMFFLApp/streamlit_ui/md/tab_data_access/players/
├── weekly_player_data.py       # Weekly loader (123 cols)
├── season_player_data.py       # Season loader (uses players_by_year)
└── career_player_data.py       # Career loader

KMFFLApp/streamlit_ui/md/tab_data_access/team_stats/
├── weekly_team_data.py         # Weekly aggregation
├── season_team_data.py         # Season aggregation
└── career_team_data.py         # Career aggregation
```

---

*This documentation covers the Player Pipeline for both Player Stats and Team Stats UI sections. See `RECOMMENDATIONS_DATA_PIPELINE.md` for consolidated improvement recommendations.*
