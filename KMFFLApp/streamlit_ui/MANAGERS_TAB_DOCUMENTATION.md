# Managers Tab - Detailed Documentation

**Complete technical documentation for the Managers/Matchups tab of the KMFFL Analytics App**

**Last Updated:** 2025-10-28
**Parent File:** `app_homepage_optimized.py`
**Main Component:** `tabs/matchups/matchup_overview.py`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Flow](#data-flow)
4. [Sub-Tabs Structure](#sub-tabs-structure)
5. [Components](#components)
6. [Database Queries](#database-queries)
7. [Calculations & Metrics](#calculations--metrics)
8. [UI Elements](#ui-elements)
9. [Performance Optimization](#performance-optimization)
10. [User Interactions](#user-interactions)

---

## Overview

The **Managers Tab** (also called "Matchups") provides comprehensive analysis of manager performance, head-to-head records, and team matchup history. It serves as the central hub for understanding manager success, rivalries, and competitive trends.

### Purpose

- **Performance Tracking**: Monitor wins, losses, scoring averages
- **Head-to-Head Analysis**: Compare matchup history between any two managers
- **Historical Trends**: Track performance across weeks, seasons, and careers
- **Visual Analytics**: 6 embedded graph types for trend analysis
- **Fun Facts**: Auto-generated insights from actual data

### Key Features

- **4 Sub-Tabs**: Weekly, Seasons, Career, Visualize
- **Smart Filters**: Filter by manager, opponent, year, game type
- **Advanced Stats**: Optimal lineups, power ratings, projections
- **Data-Driven Insights**: 15 automatically generated fun facts
- **Interactive Graphs**: Win%, scoring trends, position strength

---

## Architecture

### Component Hierarchy

```
app_homepage_optimized.py
└── render_managers_tab()
    └── load_managers_tab()              # Data loader
        ├── load_managers_data()         # DB queries
        │   ├── matchup_summary          # Season-by-season stats
        │   ├── h2h_summary              # Head-to-head records
        │   └── all_matchups             # Full matchup history
        └── display_matchup_overview()
            ├── Weekly Sub-tab           # Week-by-week analysis
            ├── Seasons Sub-tab          # Season aggregations
            ├── Career Sub-tab           # All-time records
            └── Visualize Sub-tab        # Manager graphs
```

### File Structure

```
tabs/matchups/
├── matchup_overview.py                      # Main orchestrator
├── weekly/
│   ├── weekly_matchup_overview.py          # Weekly tab main
│   ├── weekly_matchup_stats.py             # Weekly stats display
│   ├── weekly_advanced_stats.py            # Advanced metrics
│   ├── weekly_projected_stats.py           # Projections vs actuals
│   ├── weekly_optimal_lineups.py           # Optimal lineup analysis
│   ├── weekly_team_ratings.py              # Power ratings
│   ├── weekly_head_to_head.py              # Manager comparisons
│   └── weekly_matchup_graphs.py            # Weekly graphs
├── season/
│   ├── season_matchup_overview.py          # Season tab main
│   ├── season_matchup_stats.py             # Season aggregations
│   ├── season_advanced_stats.py            # Season metrics
│   ├── season_projected_stats.py           # Season projections
│   └── season_optimal_lineups.py           # Season optimal analysis
├── all_time/
│   ├── career_matchup_overview.py          # Career tab main
│   ├── career_matchup_stats.py             # Career totals
│   ├── career_advanced_stats.py            # Career metrics
│   ├── career_projected_stats.py           # Career projections
│   └── career_optimal_lineups.py           # Career optimal analysis
└── __init__.py                             # Module exports
```

---

## Data Flow

### Load Sequence

```
User Clicks "Managers" Tab
    ↓
render_managers_tab() triggered
    ↓
load_managers_tab() - Data fetching (cached 5 min)
    └── load_managers_data()
        ├── Query 1: Season summary (aggregated by year+manager)
        ├── Query 2: H2H records (manager vs all opponents)
        └── Query 3: All matchups (full history)
    ↓
Data packaged into df_dict
    {
        "summary": DataFrame (season stats),
        "h2h": DataFrame (head-to-head records),
        "recent": DataFrame (all matchups)
    }
    ↓
display_matchup_overview(df_dict)
    ↓
Renders 4 sub-tabs with filters and visualizations
```

### Data Dictionary: `df_dict`

**Structure returned by `load_managers_tab()`:**

```python
{
    "summary": pd.DataFrame,   # Season-by-season manager stats
    "h2h": pd.DataFrame,      # Head-to-head records between all managers
    "recent": pd.DataFrame    # Full matchup history (10,000+ rows)
}
```

### `summary` DataFrame Schema

**Purpose:** Aggregated stats by manager and season

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `year` | int | Season year | - |
| `manager` | string | Manager name | - |
| `games_played` | int | Total games in season | COUNT(*) |
| `wins` | int | Total wins | SUM(CASE WHEN team_points > opponent_points) |
| `losses` | int | Total losses | SUM(CASE WHEN team_points <= opponent_points) |
| `avg_points` | float | Average score per game | AVG(team_points) |
| `total_points` | float | Total points scored | SUM(team_points) |

**Row Count:** ~400 rows (10 years × ~14 managers × 3 seasons per manager average)

**Use Cases:**
- Season standings tables
- Win-loss records
- Scoring averages by year
- Performance trends over time

**Database Query:**

```sql
SELECT
    year, manager,
    COUNT(*) AS games_played,
    SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN team_points <= opponent_points THEN 1 ELSE 0 END) AS losses,
    AVG(team_points) AS avg_points,
    SUM(team_points) AS total_points
FROM kmffl.matchup
WHERE league_id = '449.l.198278'
GROUP BY year, manager
ORDER BY year DESC, wins DESC
```

---

### `h2h` DataFrame Schema

**Purpose:** Head-to-head records between every manager pair

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `manager` | string | First manager | - |
| `opponent` | string | Second manager (opponent) | - |
| `games_played` | int | Total games against this opponent | COUNT(*) |
| `wins` | int | Wins vs this opponent | SUM(CASE WHEN m1.team_points > m2.team_points) |
| `avg_margin` | float | Average point differential | AVG(m1.team_points - m2.team_points) |

**Row Count:** ~196 rows (14 managers × 14 opponents = 196 pairs)

**Use Cases:**
- Rivalry analysis
- Manager-specific matchup history
- "Who has my number?" insights
- Competitive advantage identification

**Database Query:**

```sql
SELECT
    m1.manager, m2.manager AS opponent,
    COUNT(*) AS games_played,
    SUM(CASE WHEN m1.team_points > m2.team_points THEN 1 ELSE 0 END) AS wins,
    AVG(m1.team_points - m2.team_points) AS avg_margin
FROM kmffl.matchup m1
JOIN kmffl.matchup m2
  ON m1.league_id = m2.league_id
  AND m1.year = m2.year
  AND m1.week = m2.week
  AND m1.opponent = m2.manager
WHERE m1.league_id = '449.l.198278'
GROUP BY m1.manager, m2.manager
ORDER BY m1.manager, wins DESC
```

---

### `recent` DataFrame Schema (All Matchups)

**Purpose:** Complete matchup history with all available columns

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `league_id` | string | League identifier (449.l.198278) | Yahoo |
| `manager` | string | Manager/team owner | Yahoo |
| `opponent` | string | Opponent manager | Yahoo |
| `year` | int | Season year | Yahoo |
| `week` | int | Week number (1-18) | Yahoo |
| `cumulative_week` | int | Cross-season week ID | Derived |
| `team_points` | float | Manager's score | Yahoo |
| `opponent_points` | float | Opponent's score | Yahoo |
| `win` | int | 1 if won, 0 if lost | Derived |
| `loss` | int | 1 if lost, 0 if won | Derived |
| `is_playoffs` | int | 1 if playoff game | Derived |
| `is_consolation` | int | 1 if consolation game | Derived |
| `margin` | float | Point differential | team_points - opponent_points |
| `optimal_points` | float | Best possible score with roster | Derived |
| `team_projected_points` | float | Pre-game projection | Yahoo |
| `opponent_projected_points` | float | Opponent's projection | Yahoo |
| `p_playoffs` | float | Playoff probability (0-1) | Simulation |
| `p_bye` | float | First-round bye probability | Simulation |
| `power_rating` | float | Team strength metric | Simulation |
| `teams_beat_this_week` | int | # teams scored more than | Derived |
| `total_matchup_score` | float | Combined score (team + opponent) | Derived |

**Row Count:** ~10,000 rows (10 years × 14 teams × 14 weeks × 2 sides)

**Use Cases:**
- Week-by-week analysis
- Filtering by year, manager, game type
- Advanced stats calculations
- Graph data sources

---

## Sub-Tabs Structure

### 1. Weekly Sub-Tab

**Purpose:** Week-by-week matchup analysis with advanced metrics

**File:** `weekly/weekly_matchup_overview.py`
**Class:** `WeeklyMatchupDataViewer`

**Components:**

1. **Data-Driven Fun Facts** (Rotating Display)
   - Auto-generated insights from actual data
   - 15 unique facts calculated on load
   - Examples:
     - "Highest single-week score: 158.7 pts by Manager A in Week 12, 2023"
     - "Manager B left 47.5 pts on bench but still won in Week 5, 2024"
     - "Biggest blowout: Manager C crushed Manager D by 72.3 pts"

2. **Filter Bar**
   - **Manager Selector**: Multi-select dropdown (all managers by default)
   - **Opponent Selector**: Multi-select dropdown (all opponents by default)
   - **Year Selector**: Multi-select dropdown (all years by default)
   - **Game Type Toggles**:
     - Regular Season
     - Playoffs
     - Consolation
   - **Reset Filters Button**: Clear all selections

3. **Sub-tabs Within Weekly**
   - **Matchup Stats**: Core stats table
   - **Advanced Stats**: Optimal lineups, efficiency metrics
   - **Projected Stats**: Projections vs actuals
   - **Optimal Lineups**: Best possible lineups by week
   - **Team Ratings**: Power ratings over time
   - **Head-to-Head**: Manager vs manager comparison

#### Weekly Matchup Stats

**Display:** Paginated table (25 rows per page)

**Columns:**

| Column | Description | Format |
|--------|-------------|--------|
| Year | Season | 2024 |
| Week | Week number | 5 |
| Manager | Team owner | "Manager A" |
| Opponent | Opposing manager | "Manager B" |
| Score | Manager's points | 128.5 |
| Opp Score | Opponent's points | 102.3 |
| Result | Win/Loss | W (green) / L (red) |
| Margin | Point differential | +26.2 |
| Optimal | Best possible score | 153.2 |
| Efficiency | Actual/Optimal % | 83.9% |

**Sorting:** Default by cumulative_week DESC (most recent first)

**Interactions:**
- Click column header to sort
- Use pagination controls
- Apply filters to narrow results

---

#### Weekly Advanced Stats

**Purpose:** Deep dive metrics beyond basic W-L

**Metrics Displayed:**

1. **Optimal Lineup Efficiency**
   ```
   Efficiency = (Actual Score / Optimal Score) × 100%
   ```
   - Measures lineup management quality
   - 100% = perfect lineup decisions
   - < 80% = significant points left on bench

2. **Bench Points**
   ```
   Bench Points = Optimal Score - Actual Score
   ```
   - Total points left on bench
   - Highlights missed opportunities
   - Average across season shown

3. **Power Rating**
   - Normalized team strength metric
   - Accounts for strength of schedule
   - Predicts future performance

4. **Teams Beat This Week**
   ```
   Teams Beat = Count(managers with lower score this week)
   ```
   - How many teams you outscored (regardless of opponent)
   - Shows "what if" schedule variations
   - Max: 13 teams (beat everyone)

5. **Luck Factor**
   ```
   Luck = (Actual Wins) - (Expected Wins from PPG rank)
   ```
   - Positive = won more than points suggest (lucky)
   - Negative = won less than points suggest (unlucky)
   - Zero = record matches scoring rank

**Display Format:**

```
┌─────────────────────────────────────────────┐
│ Advanced Metrics - Week 5, 2024            │
├─────────────────────────────────────────────┤
│ Manager A                                   │
│   Optimal Efficiency: 87.3%                │
│   Bench Points: 19.5                       │
│   Power Rating: 1,245                      │
│   Teams Beat: 11 of 13                     │
│   Luck Factor: +2 (lucky)                  │
└─────────────────────────────────────────────┘
```

---

#### Weekly Projected Stats

**Purpose:** Compare pre-game projections to actual results

**Columns:**

| Column | Description |
|--------|-------------|
| Manager | Team owner |
| Week | Week number |
| Projected | Pre-game projection |
| Actual | Actual score |
| Diff | Actual - Projected |
| Accuracy | How close to projection (%) |
| Beat Projection? | Yes/No indicator |

**Insights:**
- Which managers consistently outperform projections?
- Are projections getting better over time?
- Which positions have least accurate projections?

**Calculation:**

```python
# Projection differential
diff = actual_score - projected_score

# Accuracy percentage
accuracy = 100 - abs(diff / projected_score * 100)

# Beat projection indicator
beat = actual_score > projected_score
```

**Example:**

```
Manager A - Week 5, 2024
  Projected: 118.5
  Actual:    128.3
  Diff:      +9.8 (beat by 8.3%)
  Accuracy:  91.7%
```

---

#### Weekly Optimal Lineups

**Purpose:** Show best possible lineup for each manager each week

**Display:** Position-by-position breakdown

**Table Columns:**

| Position | Optimal Player | Actual Player | Optimal Pts | Actual Pts | Diff |
|----------|----------------|---------------|-------------|------------|------|
| QB | L. Jackson | J. Burrow | 32.8 | 18.2 | -14.6 |
| RB1 | D. Henry | D. Henry | 28.5 | 28.5 | 0.0 ✓ |
| RB2 | J. Gibbs | N. Harris | 24.3 | 9.4 | -14.9 |
| WR1 | C. Lamb | C. Lamb | 22.1 | 22.1 | 0.0 ✓ |
| WR2 | T. Hill | D.J. Moore | 27.5 | 3.2 | -24.3 |
| WR3 | A. St. Brown | G. Wilson | 19.8 | 11.7 | -8.1 |
| TE | T. Kelce | D. Goedert | 18.3 | 7.9 | -10.4 |
| FLEX | J. Cook | J. Cook | 21.0 | 21.0 | 0.0 ✓ |
| K | J. Tucker | J. Tucker | 14.0 | 14.0 | 0.0 ✓ |
| DEF | SF | SF | 16.0 | 16.0 | 0.0 ✓ |

**Summary:**
```
Actual Score:  128.5
Optimal Score: 153.2
Difference:    -24.7 (16.1% efficiency loss)
Correct Starts: 5 of 10 (50%)
```

**Insights:**
- Which positions had optimal starts? (✓ markers)
- Where were biggest misses? (large negative Diff)
- Was this an efficient week overall?

---

#### Weekly Team Ratings

**Purpose:** Track power ratings over time

**Display:** Line graph + table

**Power Rating Calculation:**

```python
# Weighted combination of:
# - Recent performance (40% weight)
# - Season average (30% weight)
# - Strength of schedule adjusted (20% weight)
# - Playoff odds (10% weight)

power_rating = (
    0.40 * recent_ppg +
    0.30 * season_ppg +
    0.20 * sos_adjusted_ppg +
    0.10 * p_playoffs * 100
)
```

**Rating Interpretation:**
- **> 1,300**: Championship contender
- **1,200-1,300**: Playoff lock
- **1,100-1,200**: Playoff bubble
- **1,000-1,100**: Middle of pack
- **< 1,000**: Rebuilding

**Graph Features:**
- X-axis: Week number
- Y-axis: Power rating
- Multiple lines (one per manager)
- Interactive hover tooltips
- Zoom/pan controls

---

#### Weekly Head-to-Head

**Purpose:** Direct comparison between two managers

**Selector:** Two dropdowns to choose managers

**Display:**

```
┌─────────────────────────────────────────────────────────────┐
│ Manager A vs Manager B - All-Time Record                    │
├─────────────────────────────────────────────────────────────┤
│ Games Played: 25                                            │
│ Manager A Wins: 14                                          │
│ Manager B Wins: 11                                          │
│ Win %: 56.0% (Manager A)                                    │
│ Avg Score (A): 112.3 PPG                                    │
│ Avg Score (B): 107.8 PPG                                    │
│ Avg Margin: +4.5 pts (Manager A)                            │
│ Biggest Win (A): 72.3 pts (Week 8, 2021)                   │
│ Closest Game: 0.8 pts (Week 3, 2022 - Manager B won)       │
└─────────────────────────────────────────────────────────────┘

Week-by-Week History:
Year  Week  Winner      Score    Loser       Score    Margin
2024   5    Manager A   128.5    Manager B   102.3    +26.2
2024   1    Manager B   118.7    Manager A   115.3    +3.4
2023  12    Manager A   132.1    Manager B   98.4     +33.7
...
```

**Insights:**
- Who has the upper hand historically?
- Recent trends (last 5 games)
- Blowouts vs close games
- Season-by-season breakdown

---

### 2. Seasons Sub-Tab

**Purpose:** Season-aggregated stats and trends

**File:** `season/season_matchup_overview.py`
**Class:** `SeasonMatchupOverviewViewer`

**Components:**

1. **Season Summary Table**
   - Win-Loss records by season
   - Points For/Against
   - Playoff appearances
   - Championships

2. **Season Comparison**
   - Select multiple seasons
   - Compare same manager across years
   - Identify improvement/decline

3. **Playoff Performance**
   - Regular season vs playoff stats
   - Consolation bracket records
   - Championship game history

#### Season Summary Table

**Columns:**

| Column | Description | Example |
|--------|-------------|---------|
| Year | Season | 2024 |
| Manager | Team owner | "Manager A" |
| W-L | Win-Loss record | 10-4 |
| Win % | Win percentage | .714 |
| PF | Points For | 1,482.3 |
| PA | Points Against | 1,301.7 |
| Diff | +/- differential | +180.6 |
| PPG | Points per game | 105.9 |
| Rank | Season finish | 2nd |
| Playoffs | Made playoffs? | ✓ |
| Result | Playoff outcome | Champion 🏆 |

**Sorting:** Default by year DESC, rank ASC

**Filters:**
- Year range selector
- Manager selector
- Playoff qualifiers only toggle

---

#### Season Advanced Metrics

**Metrics:**

1. **Consistency Score**
   ```
   Consistency = 100 - (StdDev(weekly_scores) / Mean(weekly_scores) * 100)
   ```
   - Higher = more consistent scoring
   - Low variance = predictable
   - High variance = boom/bust

2. **Strength of Schedule (SOS)**
   ```
   SOS = AVG(opponent_win_pct)
   ```
   - Average win% of opponents faced
   - > 0.500 = harder schedule
   - < 0.500 = easier schedule

3. **Expected Wins**
   ```
   Expected Wins = SUM(weekly_rank / total_teams)
   ```
   - Wins based on PPG rank
   - Compares to actual wins
   - Measures schedule luck

4. **Dominance Score**
   ```
   Dominance = (Total Teams Beat) / (Games × Opponents)
   ```
   - How often you beat league average
   - > 0.500 = above average
   - < 0.500 = below average

**Example Season Analysis:**

```
Manager A - 2024 Season
  Record: 10-4 (71.4%)
  PPG: 112.3 (2nd in league)
  Consistency: 87.2 (very consistent)
  SOS: 0.487 (easier schedule)
  Expected Wins: 8.7
  Luck Factor: +1.3 (slightly lucky)
  Dominance: 0.643 (beat 64% of teams weekly)
```

---

### 3. Career Sub-Tab

**Purpose:** All-time historical records and achievements

**File:** `all_time/career_matchup_overview.py`
**Class:** `CareerMatchupOverviewViewer`

**Components:**

1. **All-Time Leaderboards**
   - Total Wins
   - Win Percentage
   - Total Points
   - PPG Average
   - Championships

2. **Manager Career Stats**
   - Year-by-year progression
   - Career totals
   - Peak performance years
   - Playoff success rate

3. **Head-to-Head Matrix**
   - All manager pairs
   - Win-Loss records for each matchup
   - Heatmap visualization

#### All-Time Leaderboards

**Top Wins Table:**

| Rank | Manager | Games | Wins | Losses | Win % |
|------|---------|-------|------|--------|-------|
| 1 | Manager A | 140 | 95 | 45 | .679 |
| 2 | Manager B | 140 | 88 | 52 | .629 |
| 3 | Manager C | 126 | 82 | 44 | .651 |
| ... | ... | ... | ... | ... | ... |

**Top Scoring Table:**

| Rank | Manager | Games | Total Points | PPG | Titles |
|------|---------|-------|--------------|-----|--------|
| 1 | Manager A | 140 | 15,724.3 | 112.3 | 2 🏆🏆 |
| 2 | Manager B | 140 | 15,201.8 | 108.6 | 1 🏆 |
| 3 | Manager C | 126 | 13,887.5 | 110.2 | 3 🏆🏆🏆 |
| ... | ... | ... | ... | ... | ... |

---

#### Career Progression Graph

**X-axis:** Season (2014 - 2024)
**Y-axis:** Win Percentage

**Features:**
- Line per manager
- Trend lines
- Championship markers (🏆)
- Interactive tooltips
- Zoom controls

**Example Insights:**
- Manager A peaked in 2020 (.857 win%)
- Manager B declining since 2022
- Manager C consistently around .500

---

#### Head-to-Head Matrix

**Purpose:** Visualize all manager rivalries at once

**Display:** Heatmap table

```
                Manager A  Manager B  Manager C  Manager D
Manager A           -        14-11      8-7       12-8
Manager B         11-14        -        9-6       7-13
Manager C          7-8        6-9        -        10-5
Manager D          8-12      13-7       5-10        -
```

**Color Coding:**
- **Dark Green**: Dominant (> 60% wins)
- **Light Green**: Winning (50-60%)
- **Yellow**: Even (45-55%)
- **Orange**: Losing (40-50%)
- **Red**: Dominated (< 40%)

**Interactions:**
- Click cell to see detailed matchup history
- Hover for win percentage
- Sort by total wins

---

### 4. Visualize Sub-Tab

**Purpose:** 6 interactive graph types for visual analysis

**Components:**

1. **🏆 Total Wins**: Bar chart of career wins by manager
2. **📊 Win %**: Line graph of win percentage trends
3. **⚡ Power Rating**: Historical power rating evolution
4. **💪 Position Strength**: Heatmap of position performance
5. **📉 Score Distribution**: Box plots of scoring variance
6. **📈 Weekly Scoring**: Line graph of weekly PPG

#### Graph 1: Total Wins

**Type:** Horizontal bar chart

**Data:**
- X-axis: Total wins
- Y-axis: Manager names
- Color: Gradient by win count

**Sorting:** Descending by wins

**Interactions:**
- Hover: Show exact win count
- Click: Filter to that manager

**Example:**

```
Manager A   ████████████████████ 95 wins
Manager B   ██████████████████   88 wins
Manager C   █████████████████    82 wins
Manager D   ████████████████     78 wins
...
```

---

#### Graph 2: Win Percentage

**Type:** Multi-line chart

**Data:**
- X-axis: Season (2014-2024)
- Y-axis: Win % (0-100%)
- Lines: One per manager

**Features:**
- **Trend Lines**: Polynomial fit showing trajectory
- **League Average Line**: Horizontal at 50%
- **Playoff Threshold**: Dashed line at 60%
- **Markers**: Championship years (🏆)

**Insights:**
- Who's improving?
- Who's declining?
- Consistency across years

---

#### Graph 3: Power Rating

**Type:** Area chart with stacking

**Data:**
- X-axis: Week (all weeks across all seasons)
- Y-axis: Power rating (0-1500)
- Areas: Stacked by manager

**Color Zones:**
- **Red Zone (< 1000)**: Struggling
- **Yellow Zone (1000-1100)**: Below average
- **Green Zone (1100-1200)**: Playoff contender
- **Blue Zone (1200+)**: Elite

**Interactions:**
- Select date range
- Toggle managers on/off
- View specific week details

---

#### Graph 4: Position Strength

**Type:** Heatmap matrix

**Data:**
- Rows: Managers
- Columns: Positions (QB, RB, WR, TE, K, DEF)
- Color: PPG at that position

**Color Scale:**
- **Dark Blue**: Elite (> 90th percentile)
- **Light Blue**: Above Average (60-90th)
- **White**: Average (40-60th)
- **Light Red**: Below Average (10-40th)
- **Dark Red**: Weak (< 10th)

**Example:**

```
            QB     RB     WR     TE     K      DEF
Manager A   █████  ████   ███    ██     ████   █████
Manager B   ███    █████  ████   █████  ███    ██
Manager C   ████   ███    █████  ███    ████   ████
```

**Insights:**
- Who has strongest QB position?
- Which teams are WR-heavy?
- Position weaknesses to exploit

---

#### Graph 5: Score Distribution

**Type:** Box and whisker plot

**Data:**
- X-axis: Manager names
- Y-axis: Weekly scores
- Boxes: 25th-75th percentile
- Whiskers: Min-Max range
- Line: Median score

**Metrics Shown:**
- **Median**: Middle line
- **Q1-Q3**: Box boundaries
- **Min/Max**: Whisker ends
- **Outliers**: Dots beyond whiskers

**Interpretation:**
- **Tall box**: High variance (boom/bust)
- **Short box**: Low variance (consistent)
- **High median**: Strong scorer
- **Many outliers**: Volatile performance

**Example:**

```
Manager A:  ▁▁▁▁████████▁▁▁   (Consistent, high median)
Manager B:  ▁▁▁▁▁▁██████████  (Consistent, very high median)
Manager C:  ▁▁▁▁▁▁▁▁▁▁▁█████  (Boom/bust, wide range)
```

---

#### Graph 6: Weekly Scoring

**Type:** Multi-line time series

**Data:**
- X-axis: Week number (1-18)
- Y-axis: Weekly PPG
- Lines: One per manager
- Shaded area: League average ± 1 std dev

**Features:**
- **Smoothed Trend**: Rolling average (3 weeks)
- **Peak Markers**: Highlight highest scores
- **Low Markers**: Highlight lowest scores
- **Playoff Line**: Vertical line at week 15

**Insights:**
- Scoring trends across season
- Hot/cold streaks
- Who peaks in playoffs?
- Consistency vs volatility

---

## Components

### Helper Functions

#### `filter_data()`

**Purpose:** Apply user-selected filters to matchup data

**Parameters:**
- `df`: Matchup DataFrame
- `regular_season`: Include regular season games (bool)
- `playoffs`: Include playoff games (bool)
- `consolation`: Include consolation games (bool)
- `selected_managers`: List of managers to include
- `selected_opponents`: List of opponents to include
- `selected_years`: List of years to include

**Returns:** Filtered DataFrame

**Implementation:**

```python
@st.cache_data(show_spinner=False)
def filter_data(_, df, regular_season, playoffs, consolation,
                selected_managers, selected_opponents, selected_years):
    # Filter by managers and opponents
    filtered = df[
        df['manager'].isin(selected_managers) &
        df['opponent'].isin(selected_opponents)
    ]

    # Filter by game type
    if regular_season or playoffs or consolation:
        conditions = []
        if regular_season:
            conditions.append(
                (filtered['is_playoffs'] == 0) &
                (filtered['is_consolation'] == 0)
            )
        if playoffs:
            conditions.append(filtered['is_playoffs'] == 1)
        if consolation:
            conditions.append(filtered['is_consolation'] == 1)
        filtered = filtered[pd.concat(conditions, axis=1).any(axis=1)]

    # Filter by year
    if selected_years:
        filtered = filtered[filtered['year'].isin(selected_years)]

    return filtered
```

**Caching:** Uses `@st.cache_data` for instant re-filtering on UI interactions

---

#### `get_data_driven_fun_facts()`

**Purpose:** Generate 15 data-driven insights from matchup data

**Parameters:**
- `matchup_df`: Full matchup DataFrame

**Returns:** List of strings (fun facts)

**Facts Generated:**

1. **Highest Single-Week Score**
   ```python
   max_score_row = matchup_df.loc[matchup_df['team_points'].idxmax()]
   f"The highest single-week team score was {max_score_row['team_points']:.2f} points"
   ```

2. **Closest Margin**
   ```python
   matchup_df['abs_margin'] = matchup_df['margin'].abs()
   closest_row = matchup_df.loc[matchup_df['abs_margin'].idxmin()]
   f"{closest_row['manager']} won by just {abs(closest_row['margin']):.2f} points"
   ```

3. **Biggest Blowout**
   ```python
   biggest_win_row = matchup_df.loc[matchup_df['margin'].idxmax()]
   f"The biggest blowout was {biggest_win_row['margin']:.2f} points"
   ```

4. **Most Bench Points in a Win**
   ```python
   wins_df = matchup_df[matchup_df['win'] == True]
   wins_df['bench_pts'] = wins_df['optimal_points'] - wins_df['team_points']
   max_bench_row = wins_df.loc[wins_df['bench_pts'].idxmax()]
   ```

5. **Highest Scoring Loss**
   ```python
   losses_df = matchup_df[matchup_df['loss'] == True]
   highest_loss_row = losses_df.loc[losses_df['team_points'].idxmax()]
   ```

6. **Lowest Scoring Win**
   ```python
   wins_df = matchup_df[matchup_df['win'] == True]
   lowest_win_row = wins_df.loc[wins_df['team_points'].idxmin()]
   ```

7. **Biggest Upset** (projected underdog won)
8. **Perfect Optimal Lineup**
9. **Highest Combined Score**
10. **Most Dominant Week**
11. **Highest Average PPG** (all-time)
12. **Most Consistent Scorer** (lowest std dev)
13. **Longest Winning Streak**
14. **Most Playoff Appearances**
15. **Best Playoff Win %**

**Error Handling:** Try/catch around each fact calculation; skip if data unavailable

---

## Database Queries

### Query 1: Season Summary

**Function:** `load_managers_data()` - Part 1
**Cache:** 5 minutes (CACHE_RECENT)

```sql
SELECT
    year, manager,
    COUNT(*) AS games_played,
    SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN team_points <= opponent_points THEN 1 ELSE 0 END) AS losses,
    AVG(team_points) AS avg_points,
    SUM(team_points) AS total_points
FROM kmffl.matchup
WHERE league_id = '449.l.198278'
GROUP BY year, manager
ORDER BY year DESC, wins DESC
```

**Performance:** ~80ms (aggregation on indexed league_id)

**Optimization:**
- Indexed `league_id` for fast filtering
- GROUP BY on indexed columns
- Simple aggregations (SUM, AVG, COUNT)

**Row Count:** ~400 rows returned

---

### Query 2: Head-to-Head Records

**Function:** `load_managers_data()` - Part 2
**Cache:** 5 minutes (CACHE_RECENT)

```sql
SELECT
    m1.manager, m2.manager AS opponent,
    COUNT(*) AS games_played,
    SUM(CASE WHEN m1.team_points > m2.team_points THEN 1 ELSE 0 END) AS wins,
    AVG(m1.team_points - m2.team_points) AS avg_margin
FROM kmffl.matchup m1
JOIN kmffl.matchup m2
  ON m1.league_id = m2.league_id
  AND m1.year = m2.year
  AND m1.week = m2.week
  AND m1.opponent = m2.manager
WHERE m1.league_id = '449.l.198278'
GROUP BY m1.manager, m2.manager
ORDER BY m1.manager, wins DESC
```

**Performance:** ~200ms (self-join with indexed columns)

**Optimization:**
- JOIN on indexed `league_id`, `year`, `week`
- Filtered before JOIN (WHERE clause)
- Composite key usage

**Row Count:** ~196 rows (14 × 14 manager pairs)

---

### Query 3: All Matchups

**Function:** `load_managers_data()` - Part 3
**Cache:** 5 minutes (CACHE_RECENT)

```sql
SELECT *
FROM kmffl.matchup
WHERE league_id = '449.l.198278'
ORDER BY cumulative_week DESC
```

**Performance:** ~500ms (10,000+ rows)

**Optimization:**
- Indexed `league_id` for filtering
- Indexed `cumulative_week` for fast sorting
- All columns returned (no complex calculations)

**Row Count:** ~10,000 rows

---

## Calculations & Metrics

### Win Percentage

```python
win_pct = wins / (wins + losses)
```

**Format:** 0.714 (71.4%)

---

### Points Per Game (PPG)

```python
ppg = total_points / games_played
```

**Format:** 112.3 pts/game

---

### Margin of Victory

```python
margin = team_points - opponent_points
```

**Positive:** Won by X points
**Negative:** Lost by X points

---

### Optimal Efficiency

```python
efficiency = (actual_score / optimal_score) * 100
```

**Interpretation:**
- **100%**: Perfect lineup
- **90-99%**: Excellent decisions
- **80-89%**: Good decisions
- **70-79%**: Average decisions
- **< 70%**: Poor decisions

---

### Luck Factor

```python
# Expected wins based on points scored
expected_wins = 0
for week in season:
    teams_beat = count(teams with lower score this week)
    expected_wins += teams_beat / total_teams

luck = actual_wins - expected_wins
```

**Interpretation:**
- **Positive**: Won more than scoring suggests (lucky schedule)
- **Zero**: Record matches point production
- **Negative**: Won less than scoring suggests (unlucky schedule)

---

### Power Rating

```python
power_rating = (
    0.40 * recent_3wk_ppg +
    0.30 * season_ppg +
    0.20 * sos_adjusted_ppg +
    0.10 * p_playoffs * 100
)
```

**Ranges:**
- **> 1,300**: Elite
- **1,200-1,300**: Strong playoff team
- **1,100-1,200**: Playoff contender
- **1,000-1,100**: Middle of pack
- **< 1,000**: Struggling

---

### Strength of Schedule (SOS)

```python
sos = matchup_df.merge(
    standings[['manager', 'win_pct']],
    left_on='opponent',
    right_on='manager'
).groupby('manager')['win_pct'].mean()
```

**Interpretation:**
- **> 0.550**: Very hard schedule
- **0.500-0.550**: Above average
- **0.450-0.500**: Below average
- **< 0.450**: Easy schedule

---

## UI Elements

### Filter Bar

**Location:** Top of Weekly tab

**Components:**

1. **Manager Multi-Select**
   ```python
   selected_managers = st.multiselect(
       "Select Managers",
       options=all_managers,
       default=all_managers
   )
   ```

2. **Opponent Multi-Select**
   ```python
   selected_opponents = st.multiselect(
       "Select Opponents",
       options=all_managers,
       default=all_managers
   )
   ```

3. **Year Multi-Select**
   ```python
   selected_years = st.multiselect(
       "Select Years",
       options=all_years,
       default=[]  # Empty = all years
   )
   ```

4. **Game Type Toggles**
   ```python
   col1, col2, col3 = st.columns(3)
   with col1:
       regular = st.checkbox("Regular Season", value=True)
   with col2:
       playoffs = st.checkbox("Playoffs", value=True)
   with col3:
       consolation = st.checkbox("Consolation", value=True)
   ```

5. **Reset Button**
   ```python
   if st.button("Reset Filters"):
       st.session_state.clear()
       st.rerun()
   ```

---

### Fun Facts Carousel

**Location:** Top of Weekly tab (below filters)

**Display:**

```
┌─────────────────────────────────────────────────────────────┐
│ 💡 Did You Know?                                            │
├─────────────────────────────────────────────────────────────┤
│ The highest single-week team score was 158.7 points by     │
│ Manager A in Week 12, 2023                                  │
│                                                             │
│ [◀ Previous]  Fact 1 of 15  [Next ▶]                       │
└─────────────────────────────────────────────────────────────┘
```

**Interactions:**
- Click arrows to cycle through facts
- Auto-rotate every 10 seconds
- Pause on hover

---

### Metrics Display

**Purpose:** Quick stats at-a-glance

**Layout:** 3-4 columns with large numbers

```python
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Games", games_played)
with col2:
    st.metric("Win %", f"{win_pct:.1%}")
with col3:
    st.metric("PPG", f"{ppg:.1f}")
with col4:
    st.metric("Championships", titles, delta="+1" if recent_title else None)
```

---

### Data Tables

**Features:**
- Sortable columns
- Pagination (25 rows per page)
- Responsive layout
- Color-coded cells
- Export to CSV button

**Styling:**

```python
st.dataframe(
    df,
    use_container_width=True,
    height=600,
    hide_index=True
)
```

---

## Performance Optimization

### Caching Strategy

**Layer 1: Data Loader (5 min TTL)**
```python
@cached_data_loader(ttl=300, spinner_text="Loading managers data...")
def load_managers_tab():
    return load_managers_data()
```

**Layer 2: Filter Function (Instant)**
```python
@st.cache_data(show_spinner=False)
def filter_data(...):
    # Fast in-memory filtering
    return filtered_df
```

**Layer 3: Component State**
```python
# Store filter selections in session_state
if "selected_managers" not in st.session_state:
    st.session_state.selected_managers = all_managers
```

---

### Lazy Loading

**Pattern:** Load graphs only when sub-tab viewed

```python
with tabs[3]:  # Visualize tab
    # Graphs only loaded if user clicks this tab
    try:
        display_total_wins_graph(df_wrapper)
    except Exception as e:
        st.warning(f"Graph unavailable: {e}")
```

---

### Progressive Rendering

**Pattern:** Show UI first, load data in background

```python
# Show filters immediately
display_filters()

# Load data with spinner
with st.spinner("Loading matchup data..."):
    data = load_data()

# Render when ready
display_table(data)
```

---

## User Interactions

### Workflow 1: Check Career Record

```
1. User clicks "Managers" tab
   ↓
2. Career sub-tab loads
   ↓
3. User scans all-time leaderboard
   ↓
4. User finds their position
   ↓
5. User clicks name to see details
```

**Time to Insight:** ~2 seconds

---

### Workflow 2: Analyze Weekly Performance

```
1. User clicks "Managers" tab
   ↓
2. Weekly sub-tab active by default
   ↓
3. User selects Year: 2024, Week: 5
   ↓
4. User checks "Matchup Stats" table
   ↓
5. User sees win/loss with efficiency %
   ↓
6. User switches to "Optimal Lineups"
   ↓
7. User identifies lineup mistakes
```

**Time to Analysis:** ~3 seconds

---

### Workflow 3: Head-to-Head Rivalry

```
1. User clicks "Managers" tab
   ↓
2. Weekly sub-tab → "Head-to-Head"
   ↓
3. User selects Manager A and Manager B
   ↓
4. System shows all-time record
   ↓
5. User sees W-L, avg margin, biggest wins
   ↓
6. User reviews week-by-week history table
```

**Time to Rivalry Insight:** ~4 seconds

---

### Workflow 4: Visual Trend Analysis

```
1. User clicks "Managers" tab
   ↓
2. Visualize sub-tab
   ↓
3. User views "Win %" graph
   ↓
4. User identifies improving managers
   ↓
5. User switches to "Power Rating"
   ↓
6. User confirms trend with ratings
```

**Time to Trend Discovery:** ~5 seconds

---

## Appendix

### Quick Reference: Sub-Tab Files

| Sub-Tab | Main File | Purpose |
|---------|-----------|---------|
| Weekly | `weekly/weekly_matchup_overview.py` | Week-by-week analysis |
| Seasons | `season/season_matchup_overview.py` | Season aggregations |
| Career | `all_time/career_matchup_overview.py` | All-time records |
| Visualize | `matchup_overview.py` (inline) | Manager graphs |

---

### Common User Questions

**Q: Why is my win % lower than my PPG rank?**
**A:** You have negative "luck factor" - lost close games or faced top scorers.

**Q: How is power rating calculated?**
**A:** Weighted combination of recent performance (40%), season average (30%), SOS-adjusted (20%), and playoff odds (10%).

**Q: What does "teams beat this week" mean?**
**A:** How many teams you outscored that week, regardless of who you played.

**Q: Can I see my record against a specific opponent?**
**A:** Yes! Go to Weekly → Head-to-Head and select both managers.

**Q: What's the difference between Optimal and Actual score?**
**A:** Optimal = best possible with your roster. Actual = what you scored with your lineup.

---

**End of Managers Tab Documentation**

For questions or issues, refer to the main APP_DOCUMENTATION.md or contact the development team.
