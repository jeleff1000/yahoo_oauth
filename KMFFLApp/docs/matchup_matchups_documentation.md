# Matchups UI Documentation

> **Last Updated:** November 2024
> **Data Source:** matchup.parquet (280 columns)
> **UI Location:** `streamlit_ui/tabs/matchups/`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [UI Components](#ui-components)
3. [Feature Summaries](#feature-summaries)
4. [Recommendations & Roadmap](#recommendations--roadmap)

---

## Executive Summary

The Matchups section provides detailed head-to-head analysis across weekly, seasonal, and career timeframes. It's the core analytics hub for understanding manager performance and matchup history.

### Key Capabilities

- **Weekly Analysis**: Current/recent week matchup details
- **Season Overview**: Season-long performance trends
- **Career Stats**: All-time head-to-head records
- **Visualizations**: 8 graph types for visual analysis
- **Optimal Lineup**: Lineup efficiency tracking

### Columns Used (60 of 280)

```python
MATCHUPS_COLUMNS = [
    # Core
    'year', 'week', 'manager', 'opponent', 'team_points', 'opponent_points',
    'team_projected_points', 'opponent_projected_points', 'margin',

    # Win/Loss
    'win', 'loss', 'tie',

    # League context
    'league_weekly_mean', 'league_weekly_median', 'weekly_rank',
    'teams_beat_this_week', 'above_league_median',

    # Manager stats
    'manager_season_mean', 'manager_season_median',
    'cumulative_wins', 'cumulative_losses', 'win_streak', 'loss_streak',

    # H2H columns (dynamic)
    'w_vs_*', 'l_vs_*',

    # Optimal lineup
    'optimal_points', 'bench_points', 'lineup_efficiency',
    'opponent_optimal_points',

    # Projections
    'expected_spread', 'expected_odds', 'win_probability',

    # Power ratings
    'felo_score', 'felo_tier', 'power_rating',

    # Playoff context
    'is_playoffs', 'is_consolation', 'playoff_round'
]
```

---

## UI Components

### Current Implementation

The Matchups hub (`matchup_overview.py`) contains **4 tabs**:

---

### Tab 1: Weekly

**Sub-components:**
- `weekly_matchup_overview.py` - Main weekly view
- `weekly_matchup_stats.py` - Detailed statistics
- `weekly_advanced_stats.py` - Advanced metrics
- `weekly_projected_stats.py` - Projection analysis
- `weekly_optimal_lineups.py` - Lineup efficiency
- `weekly_team_ratings.py` - Power ratings
- `weekly_head_to_head.py` - H2H comparison

**What It Shows:**
- Current week matchup details
- Score vs projection comparison
- Lineup efficiency (actual vs optimal)
- Head-to-head history with opponent
- League standings context

**Current Features:**
- Week/Year filters
- Manager selection
- Side-by-side matchup cards
- Projection analysis
- Historical H2H record

**Suggested Additions:**
- [ ] Add "Matchup Preview" with win probability
- [ ] Add "Key Players" section (who needs to perform)
- [ ] Add "History Against This Opponent" highlight card
- [ ] Add "Similar Past Matchups" comparison
- [ ] Add live score updates integration (future)

---

### Tab 2: Seasons

**Sub-components:**
- `season_matchup_overview.py` - Main season view
- `season_matchup_stats.py` - Season statistics
- `season_advanced_stats.py` - Advanced metrics
- `season_projected_stats.py` - Projection accuracy
- `season_optimal_lineups.py` - Season lineup efficiency
- `season_team_ratings.py` - Power rating trends
- `season_head_to_head.py` - Season H2H summary
- `season_graphs.py` - Season visualizations

**What It Shows:**
- Season record and standings
- PPG trends over the season
- H2H record for the season
- Playoff positioning
- Strength of schedule

**Current Features:**
- Year filter
- Manager filter
- Season summary stats
- Week-by-week breakdown
- Cumulative charts

**Suggested Additions:**
- [ ] Add "Season Grade" card (A-F based on performance)
- [ ] Add "Playoff Picture" visualization
- [ ] Add "Strength of Wins" analysis
- [ ] Add "Consistency Score" metric
- [ ] Add season comparison (this year vs last year)

---

### Tab 3: Career

**Sub-components:**
- `career_matchup_overview.py` - Main career view
- `career_matchup_stats.py` - Career statistics
- `career_advanced_stats.py` - Advanced metrics
- `career_projected_stats.py` - Projection accuracy
- `career_optimal_lineups.py` - Career lineup efficiency
- `career_team_ratings.py` - Historical power ratings
- `career_head_to_head_overview.py` - All-time H2H

**What It Shows:**
- All-time record
- Career PPG and trends
- H2H records against all opponents
- Championship history
- Career milestones

**Current Features:**
- Manager filter
- All-time statistics
- H2H matrix
- Historical trends

**Suggested Additions:**
- [ ] Add "Career Report Card" summary
- [ ] Add "Rivals" section (best/worst matchups)
- [ ] Add "Era Analysis" (performance by time period)
- [ ] Add "Milestone Tracker" (100 wins, 1000 points, etc.)
- [ ] Add "Career Graph" showing win % over time

---

### Tab 4: Visualize

**Sub-components (8 graphs):**

#### 1. Scoring Trends (`scoring_trends.py`)
- Weekly scoring over time
- Rolling averages
- Trend lines

#### 2. Win Percentage (`win_percentage_graph.py`)
- Win % by season
- Cumulative win %
- Win % vs specific opponents

#### 3. Power Rating (`power_rating.py`)
- Felo score trends
- Power ranking history
- Rating distribution

#### 4. Score Distribution (`scoring_distribution.py`)
- Histogram of weekly scores
- Score ranges
- Comparison to league average

#### 5. Margin of Victory (`margin_of_victory.py`)
- Win/loss margin distribution
- Close games vs blowouts
- Margin trends

#### 6. Lineup Efficiency (`optimal_lineup_efficiency.py`)
- Actual vs optimal points
- Efficiency trends
- Bench point analysis

#### 7. Playoff Performance (`playoff_vs_regular.py`)
- Regular season vs playoff PPG
- Clutch performance
- Playoff success rate

#### 8. Strength of Schedule (`strength_of_schedule.py`)
- Opponent difficulty
- Schedule luck
- SOS trends

**Suggested Additions:**
- [ ] Add "Head-to-Head Network" graph (who plays whom most)
- [ ] Add "Scoring Heatmap" by week/year
- [ ] Add "Dominance Chart" (teams beat this week trend)
- [ ] Add "Luck vs Skill" scatter plot
- [ ] Add "Best/Worst Week" highlight cards

---

## Feature Summaries

### For Homepage

#### Matchups - Quick Summary

> **Analyze Head-to-Head Performance**
>
> The Matchups section provides detailed analysis of every head-to-head matchup in league history. Track weekly performance, seasonal trends, and career statistics with comprehensive visualizations.
>
> **Key Features:**
> - Weekly matchup breakdowns with projections
> - Season-long performance tracking
> - Career head-to-head records
> - 8 visualization types for deep analysis

---

### For About Page

#### Matchups - Detailed Description

> **What is Matchup Analysis?**
>
> The Matchups section provides comprehensive head-to-head analysis across three time horizons: weekly, seasonal, and career. It helps managers understand their performance patterns and historical matchup records.
>
> **Time Horizons:**
>
> | View | What It Shows |
> |------|---------------|
> | **Weekly** | Current/recent week matchup details, projections, lineup efficiency |
> | **Season** | Season-long trends, cumulative stats, playoff positioning |
> | **Career** | All-time records, H2H history, career achievements |
>
> **Available Visualizations:**
> - Scoring Trends - How scores change over time
> - Win Percentage - Win rate analysis
> - Power Rating - Team strength tracking
> - Score Distribution - Performance consistency
> - Margin of Victory - Win/loss patterns
> - Lineup Efficiency - Optimal lineup tracking
> - Playoff Performance - Clutch analysis
> - Strength of Schedule - Opponent difficulty
>
> **What You Can Learn:**
> - How you perform against specific opponents
> - Your scoring trends and consistency
> - Where you leave points on the bench
> - How your schedule affects your record

---

### Section-by-Section Summaries

#### Weekly Tab
> "Detailed breakdown of weekly matchups including projections, lineup efficiency, and head-to-head history with your opponent."

#### Seasons Tab
> "Season-long performance analysis with trends, cumulative stats, and playoff positioning over the course of the year."

#### Career Tab
> "All-time statistics and head-to-head records against every opponent in league history."

#### Visualize Tab
> "Eight interactive charts for visual analysis of scoring trends, win rates, power ratings, and more."

---

## Recommendations & Roadmap

### Priority 1: Add Pre-Computed Metrics

```python
# Add to cumulative_stats_v2.py

# 1. GAME QUALITY GRADE
df['game_grade'] = pd.cut(
    df['team_points'],
    bins=[0, df['league_weekly_median'] * 0.7,
          df['league_weekly_median'] * 0.9,
          df['league_weekly_median'] * 1.1,
          df['league_weekly_median'] * 1.3,
          999],
    labels=['F', 'D', 'C', 'B', 'A']
)

# 2. CONSISTENCY SCORE (inverse of coefficient of variation)
df['consistency_score'] = df.groupby(['manager', 'year'])['team_points'].transform(
    lambda x: 100 - (x.std() / x.mean() * 100).clip(0, 100)
)

# 3. CLUTCH WIN FLAG
df['is_clutch_win'] = (
    (df['win'] == 1) &
    (df['expected_odds'] < 0.4)
).astype(int)

# 4. BENCH POINTS WASTED
df['bench_points_wasted'] = df['optimal_points'] - df['team_points']

# 5. LINEUP EFFICIENCY GRADE
df['efficiency_grade'] = pd.cut(
    df['lineup_efficiency'].fillna(0) * 100,
    bins=[0, 75, 85, 92, 97, 100],
    labels=['F', 'D', 'C', 'B', 'A']
)
```

### Priority 2: UI Enhancements

| Enhancement | Impact | Effort |
|-------------|--------|--------|
| Add Matchup Preview card | High engagement | Medium |
| Add "Rivalry" detection | Fun factor | Medium |
| Add consistency visualization | Insight | Low |
| Add milestone alerts | Engagement | Low |
| Add mobile-responsive tables | Accessibility | Medium |

### Priority 3: New Components

1. **Matchup Preview** - Pre-game analysis with win probability
2. **Rivalry Tracker** - Identify and highlight rivalries
3. **Season Report Card** - A-F grades for performance
4. **Milestone Tracker** - 100 wins, 1000 points, etc.
5. **Luck vs Skill Analysis** - Scatter plot of actual vs expected

### Priority 4: Performance Optimizations

```python
# Matchups uses ~60 columns - already optimized
# Current optimization in load_managers_matchup_data():
# - Loads only 22-25 columns (78% reduction)
# - Uses DuckDB for fast aggregations

# Further optimization possible:
# - Pre-compute H2H summaries
# - Cache season aggregates
# - Lazy-load graphs
```

---

## Appendix: File Locations

### UI Components
```
KMFFLApp/streamlit_ui/tabs/matchups/
├── __init__.py
├── matchup_overview.py              # Main hub
├── optimal_lineup_overview.py       # Optimal lineup view
├── weekly/
│   ├── weekly_matchup_overview.py
│   ├── weekly_matchup_stats.py
│   ├── weekly_advanced_stats.py
│   ├── weekly_projected_stats.py
│   ├── weekly_optimal_lineups.py
│   ├── weekly_team_ratings.py
│   ├── weekly_head_to_head.py
│   ├── weekly_matchup_graphs.py
│   └── H2H.py
├── season/
│   ├── season_matchup_overview.py
│   ├── season_matchup_stats.py
│   ├── season_advanced_stats.py
│   ├── season_projected_stats.py
│   ├── season_optimal_lineups.py
│   ├── season_team_ratings.py
│   ├── season_head_to_head.py
│   └── season_graphs.py
├── all_time/
│   ├── career_matchup_overview.py
│   ├── career_matchup_stats.py
│   ├── career_advanced_stats.py
│   ├── career_projected_stats.py
│   ├── career_optimal_lineups.py
│   ├── career_team_ratings.py
│   └── career_head_to_head_overview.py
├── graphs/
│   ├── __init__.py
│   ├── scoring_trends.py
│   ├── win_percentage_graph.py
│   ├── power_rating.py
│   ├── scoring_distribution.py
│   ├── margin_of_victory.py
│   ├── optimal_lineup_efficiency.py
│   ├── playoff_vs_regular.py
│   └── strength_of_schedule.py
└── shared/
    ├── __init__.py
    ├── theme.py
    ├── filters.py
    ├── config.py
    ├── fun_facts.py
    └── exceptions.py
```

### Data Access
```
KMFFLApp/streamlit_ui/md/tab_data_access/managers/
├── combined.py              # Entry point
├── matchup_data.py          # Optimized loader (22-25 cols)
└── summary_data.py          # Pre-aggregated summaries
```

---

*This documentation covers the Matchups UI. See `matchup_pipeline_documentation.md` for core pipeline details.*
