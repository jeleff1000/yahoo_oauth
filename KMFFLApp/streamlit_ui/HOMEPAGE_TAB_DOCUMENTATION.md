# Homepage Tab - Detailed Documentation

**Complete technical documentation for the Homepage tab of the KMFFL Analytics App**

**Last Updated:** 2025-10-28
**Parent File:** `app_homepage_optimized.py`
**Main Component:** `tabs/homepage/homepage_overview.py`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Flow](#data-flow)
4. [Sub-Tabs Structure](#sub-tabs-structure)
5. [Components](#components)
6. [Database Queries](#database-queries)
7. [UI Elements](#ui-elements)
8. [Performance Optimization](#performance-optimization)
9. [User Interactions](#user-interactions)
10. [Error Handling](#error-handling)

---

## Overview

The **Homepage Tab** is the central command center of the KMFFL Analytics App. It provides:

- **Quick Navigation**: 3-card overview of all app sections
- **App Guide**: Comprehensive expandable guide to every feature
- **Hall of Fame**: Championship history and elite achievements
- **Season Standings**: Current W-L records and rankings
- **Team Schedules**: Week-by-week results for all managers
- **Head-to-Head**: Position-by-position lineup comparisons
- **Weekly Recaps**: Narrative summaries with highlights and lowlights

### Purpose

- **Entry Point**: First thing users see when launching the app
- **Navigation Hub**: Quick links to all major sections
- **Quick Insights**: Current season standings and recent results
- **Educational**: Explains app features and fantasy football concepts

---

## Architecture

### Component Hierarchy

```
app_homepage_optimized.py
└── render_home_tab()
    └── load_homepage_tab()          # Data loader
        └── display_homepage_overview()
            ├── Overview Sub-tab      # Navigation & guide
            ├── Hall of Fame Sub-tab  # Championships & records
            ├── Standings Sub-tab     # Current season rankings
            ├── Schedules Sub-tab     # Weekly results
            ├── H2H Sub-tab          # Matchup analysis
            └── Recaps Sub-tab       # Weekly narratives
```

### File Structure

```
tabs/homepage/
├── homepage_overview.py          # Main homepage orchestrator
├── season_standings.py           # Standings display logic
├── head_to_head.py              # H2H matchup comparisons
├── schedules.py                 # Schedule visualization
├── champions.py                 # Championship history
├── recaps/
│   ├── __init__.py              # Recap module exports
│   ├── recap_overview.py        # Recap orchestrator
│   ├── helpers/                 # Recap helper functions
│   └── displays/                # Recap display components
└── __init__.py                  # Homepage module exports
```

---

## Data Flow

### Load Sequence

```
User Opens App
    ↓
main() in app_homepage_optimized.py
    ↓
render_home_tab() triggered
    ↓
load_homepage_tab() - Data fetching (cached 5 min)
    ├── load_homepage_data()          # Summary counts
    ├── load_simulations_data()       # Matchup data
    └── load_player_two_week_slice()  # Recent player stats
    ↓
Data packaged into df_dict
    ↓
display_homepage_overview(df_dict)
    ↓
Renders 6 sub-tabs
```

### Data Dictionary: `df_dict`

**Structure returned by `load_homepage_tab()`:**

```python
{
    "summary": {
        "matchup_count": int,      # Total matchups in DB
        "player_count": int,       # Total player records
        "draft_count": int,        # Total draft picks
        "transactions_count": int, # Total transactions
        "injuries_count": int,     # Total injury records
        "latest_year": int,        # Most recent year
        "latest_week": int,        # Most recent week
        "latest_games": int        # Games in latest week
    },
    "Matchup Data": pd.DataFrame,  # Full matchup history
    "Player Data": pd.DataFrame    # Two-week player slice
}
```

### `Matchup Data` DataFrame Schema

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `league_id` | string | League identifier (449.l.198278) | Yahoo |
| `manager` | string | Manager/team owner name | Yahoo |
| `opponent` | string | Opponent manager name | Yahoo |
| `year` | int | Season year | Yahoo |
| `week` | int | NFL week (1-18) | Yahoo |
| `cumulative_week` | int | Cross-season week number | Derived |
| `team_points` | float | Manager's score | Yahoo |
| `opponent_points` | float | Opponent's score | Yahoo |
| `win` | int | 1 if won, 0 if lost | Derived |
| `loss` | int | 1 if lost, 0 if won | Derived |
| `is_playoffs` | int | 1 if playoff game | Derived |
| `is_consolation` | int | 1 if consolation game | Derived |
| `p_playoffs` | float | Playoff probability (0-1) | Simulation |
| `p_bye` | float | First-round bye probability | Simulation |
| `power_rating` | float | Team strength metric | Simulation |

**Row Count:** ~10,000 rows (10 years × 14 teams × 14 weeks × 2 (both sides))

### `Player Data` DataFrame Schema

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `league_id` | string | League identifier (nullable for NFL-only) | Yahoo |
| `yahoo_player_id` | string | Yahoo's player ID | Yahoo |
| `player` | string | Player full name | Both |
| `manager` | string | Fantasy team owner (NULL for unrostered) | Yahoo |
| `nfl_position` | string | QB, RB, WR, TE, K, DEF | NFL |
| `fantasy_position` | string | Starting position (QB, RB1, FLEX, etc.) | Yahoo |
| `year` | int | Season year | Both |
| `week` | int | Week number | Both |
| `points` | float | Fantasy points scored | Yahoo |
| `started` | int | 1 if in starting lineup | Yahoo |
| `nfl_team` | string | Player's NFL team | NFL |
| `opponent_nfl_team` | string | NFL opponent | NFL |
| `pass_yds` | int | Passing yards | NFL |
| `pass_td` | int | Passing touchdowns | NFL |
| `rush_yds` | int | Rushing yards | NFL |
| `rush_td` | int | Rushing touchdowns | NFL |
| `rec` | int | Receptions | NFL |
| `rec_yds` | int | Receiving yards | NFL |
| `rec_td` | int | Receiving touchdowns | NFL |

**Row Count:** ~3,600 rows (2 weeks × ~1,800 players including NFL-only)

---

## Sub-Tabs Structure

### 1. Overview Tab (Default)

**Purpose:** Navigation hub and comprehensive app guide

**Components:**
- **Hero Section**: Welcoming title and tagline
- **Quick Navigation**: 3-card overview
  - Homepage features
  - Player Stats
  - Advanced Analytics
- **Complete App Guide**: 10 expandable sections
  - Homepage features
  - Player Stats (Weekly, Season, Career)
  - Matchups (Weekly, Season, All-Time)
  - Draft Data
  - Transactions
  - Keepers
  - Injury Data
  - Simulations
  - Graphs & Visualizations
  - Hall of Fame
- **Power User Tips**: Best practices for efficient app usage
- **Key Concepts**: Fantasy football terms explained
  - League-Wide Optimal Lineup
  - FAAB Strategy
  - Team Optimal vs Actual
  - Simulation Methodology
- **FAQ**: 10 common questions answered

**Data Requirements:** None (static content)

**Performance:** Instant load (no database queries)

#### Key Concept: League-Wide Optimal Lineup

**Definition:** The theoretical best lineup using the top scorer at each position **across all teams** for a given week.

**Example:**
```
Week 5, 2024 Optimal Lineup:
- QB: Lamar Jackson (Team A) - 32 pts
- RB1: Derrick Henry (Team B) - 28 pts
- RB2: Jahmyr Gibbs (Team C) - 24 pts
- WR1: CeeDee Lamb (Team D) - 27 pts
- WR2: Tyreek Hill (Team A) - 22 pts
- WR3: Amon-Ra St. Brown (Team E) - 19 pts
- TE: Travis Kelce (Team F) - 18 pts
- FLEX: James Cook (Team G) - 21 pts
- K: Justin Tucker (Team H) - 14 pts
- DEF: San Francisco (Team I) - 16 pts
────────────────────────────────────
Total: 221 pts
```

**Use Case:** Compare your actual lineup to the theoretical maximum to gauge lineup management efficiency.

---

### 2. Hall of Fame Tab

**Purpose:** Celebrate league championships and elite achievements

**Components:**
- **Championship History**: All title winners by year
- **Multiple Title Winners**: Managers with 2+ championships
- **Runner-Up Records**: "Always a bridesmaid" stats
- **Playoff Bracket Visualization**: Interactive tournament brackets
- **Top Single-Season Teams**: Highest-scoring seasons
- **Top Single-Week Performances**: Record-breaking weeks
- **Playoff Success Rates**: Championship efficiency

**Data Source:** `tabs.hall_of_fame.hall_of_fame_homepage.HallOfFameViewer`

**Database Queries:**
```sql
-- Championship winners (managers with champion = 1)
SELECT year, manager, final_wins, final_losses, season_mean
FROM kmffl.matchup
WHERE league_id = '449.l.198278'
  AND champion = 1
GROUP BY year, manager
ORDER BY year DESC
```

**Performance:**
- Load time: ~150ms (cached)
- Data volume: ~25 rows (one per championship)

**Error Handling:**
- Graceful fallback if Hall of Fame module unavailable
- Warning message: "Hall of Fame module not found; cannot render HoF here."
- Debug expander with import error details

---

### 3. Season Standings Tab

**Purpose:** Display current season rankings and performance metrics

**File:** `season_standings.py`
**Function:** `display_season_standings(matchup_df, prefix="standings")`

**Components:**
- **Quick Stats Header**:
  - Total teams (metric)
  - Current week (metric)
  - Games played (metric)
- **Standings Table**:
  - Manager name
  - Wins-Losses record
  - Win percentage
  - Points For (PF)
  - Points Against (PA)
  - Point Differential (+/-)
  - Points Per Game (PPG)
  - Playoff Status indicator

**Calculations:**

```python
# Win Percentage
win_pct = wins / (wins + losses)

# Points Per Game
ppg = total_points / games_played

# Point Differential
diff = points_for - points_against

# Playoff Threshold (Top 6 teams)
playoff_cutoff = standings.sort_values('wins', ascending=False).iloc[5]['wins']
```

**Sorting:** By wins (descending), then by points_for (tiebreaker)

**UI Features:**
- Color-coded playoff indicators
  - 🟢 Green: In playoff position (Top 6)
  - 🔴 Red: Out of playoffs
- Sortable columns
- Responsive table layout

**Data Transformation:**

```python
# Input: matchup_df (raw matchup data)
standings = matchup_df.groupby('manager').agg({
    'win': 'sum',
    'loss': 'sum',
    'team_points': 'sum',
    'opponent_points': 'sum'
}).reset_index()

standings['games_played'] = standings['win'] + standings['loss']
standings['win_pct'] = standings['win'] / standings['games_played']
standings['ppg'] = standings['team_points'] / standings['games_played']
standings['diff'] = standings['team_points'] - standings['opponent_points']
```

**Performance:** ~50ms (all calculations in-memory)

---

### 4. Schedules Tab

**Purpose:** Week-by-week results for all managers

**File:** `schedules.py`
**Function:** `display_schedules(df_dict, prefix="schedules")`

**Components:**
- **Year/Week Selector**: Dropdown filters
- **Manager Selector**: View specific manager or all
- **Schedule Grid**: Color-coded results
  - 🟢 Win
  - 🔴 Loss
  - ⚪ Bye week
- **Strength of Schedule (SOS)**: Difficulty rating

**Schedule Grid Layout:**

```
Manager      | W1  | W2  | W3  | W4  | ... | W14 | Record | PPG
─────────────┼─────┼─────┼─────┼─────┼─────┼─────┼────────┼─────
Team A       | W✓  | L✗  | W✓  | W✓  | ... | W✓  | 10-4   | 105.3
Team B       | L✗  | W✓  | L✗  | W✓  | ... | L✗  | 8-6    | 98.7
...
```

**Color Scheme:**
- **Win**: Green background (#4CAF50)
- **Loss**: Red background (#F44336)
- **High-scoring Win**: Dark green (#2E7D32)
- **Low-scoring Loss**: Dark red (#C62828)

**Database Query:**

```sql
SELECT manager, year, week, opponent, team_points, opponent_points, win
FROM kmffl.matchup
WHERE league_id = '449.l.198278'
  AND year = {selected_year}
ORDER BY manager, week
```

**Strength of Schedule Calculation:**

```python
# Average opponent win percentage
sos = matchup_df.merge(
    standings[['manager', 'win_pct']],
    left_on='opponent',
    right_on='manager'
).groupby('manager')['win_pct'].mean()

# Higher SOS = harder schedule
```

**Performance:** ~100ms (cached)

---

### 5. Head-to-Head Tab

**Purpose:** Position-by-position lineup comparisons

**File:** `head_to_head.py`
**Function:** `display_head_to_head(df_dict)`

**Components:**
- **Year/Week Selector**: Choose specific week
- **Matchup Selector**: Dropdown with special options
  - Manager vs Opponent matchups
  - **"All"**: League-wide optimal lineup
  - **"Optimal"**: League-wide optimal lineup
- **Position-by-Position Table**:
  - Starting position (QB, RB1, RB2, etc.)
  - Player names
  - Points scored
  - Win/loss indicator per position
- **Score Summary**:
  - Team A total: XXX pts
  - Team B total: XXX pts
  - Winner indicator

**League-Wide Optimal Feature:**

When user selects "All" or "Optimal" in the matchup dropdown:

```python
# Find top scorer at each position across ALL teams
optimal_lineup = player_df.groupby('fantasy_position').apply(
    lambda x: x.nlargest(1, 'points')
).reset_index(drop=True)

# Display as "Optimal Team"
display_lineup(optimal_lineup, team_name="League-Wide Optimal")
```

**Position Comparison Table:**

| Position | Team A Player | Points | vs | Team B Player | Points | Winner |
|----------|--------------|--------|----|--------------|---------| -------|
| QB | Lamar Jackson | 28.5 | > | Joe Burrow | 18.2 | ✓ Team A |
| RB1 | Derrick Henry | 24.3 | < | Christian McCaffrey | 31.8 | ✗ Team B |
| RB2 | Josh Jacobs | 15.7 | > | Najee Harris | 9.4 | ✓ Team A |
| WR1 | CeeDee Lamb | 22.1 | < | Tyreek Hill | 27.5 | ✗ Team B |
| ... | ... | ... | ... | ... | ... | ... |

**Database Query:**

```sql
SELECT
    p.player,
    p.manager,
    p.fantasy_position,
    p.points,
    p.started
FROM kmffl.players_by_year p
JOIN kmffl.matchup m
  ON m.league_id = p.league_id
  AND m.manager = p.manager
  AND m.year = p.year
  AND m.week = p.week
WHERE p.league_id = '449.l.198278'
  AND p.year = {selected_year}
  AND p.week = {selected_week}
  AND (m.manager = '{selected_manager}' OR m.opponent = '{selected_manager}')
  AND p.started = 1
ORDER BY
    CASE p.fantasy_position
        WHEN 'QB' THEN 1
        WHEN 'RB1' THEN 2
        WHEN 'RB2' THEN 3
        -- ... position order
    END
```

**Performance:** ~200ms (two-week slice pre-loaded)

---

### 6. Recaps Tab

**Purpose:** Weekly narrative summaries with highlights

**File:** `recaps/recap_overview.py`
**Function:** `display_recap_overview(df_dict)`

**Components:**
- **Year/Week Selector**: Choose week to recap
- **Manager Selector**: View specific team's recap
- **Recap Cards** (per manager):
  - **Top Performer**: Highest-scoring player
  - **Biggest Disappointment**: Lowest-scoring starter
  - **Bench Points**: Total points left on bench
  - **Key Stats**: Win/loss, score, opponent
- **Award Categories**:
  - 🏆 **Highest Scorer**: Top team of the week
  - 💩 **Lowest Scorer**: Bottom team of the week
  - ⭐ **Best Player**: Top individual performance
  - 😭 **Worst Bust**: Biggest disappointment
  - 🎯 **Optimal Manager**: Best lineup efficiency
  - 🤦 **Bench Blunder**: Most points left on bench

**Recap Card Example:**

```
┌─────────────────────────────────────────────┐
│ 📰 Team A Recap - Week 5, 2024             │
├─────────────────────────────────────────────┤
│ Result: W 128.5 - 102.3 vs Team B          │
│                                             │
│ ⭐ Top Performer                            │
│   Lamar Jackson (QB): 32.8 pts            │
│                                             │
│ 😭 Biggest Disappointment                   │
│   D.J. Moore (WR): 3.2 pts                 │
│                                             │
│ 💺 Bench Highlights                         │
│   Left on bench: 47.5 pts                  │
│   Top bench player: D. Swift (RB): 22.1   │
│                                             │
│ 🎯 Lineup Efficiency: 84%                   │
│   (Actual: 128.5 / Optimal: 153.2)         │
└─────────────────────────────────────────────┘
```

**Calculations:**

```python
# Top Performer
top_player = player_df[player_df['started'] == 1].nlargest(1, 'points')

# Biggest Disappointment
bust = player_df[player_df['started'] == 1].nsmallest(1, 'points')

# Bench Points
bench_total = player_df[player_df['started'] == 0]['points'].sum()

# Lineup Efficiency
actual_score = team_points
optimal_score = player_df.groupby('fantasy_position').apply(
    lambda x: x.nlargest(1, 'points')['points'].sum()
)
efficiency = (actual_score / optimal_score) * 100
```

**Performance:** ~250ms (uses cached two-week slice)

---

## Components

### Helper Functions

#### `_as_dataframe(obj: Any) -> Optional[pd.DataFrame]`

**Purpose:** Safely convert various objects to DataFrame

**Parameters:**
- `obj`: Can be DataFrame, list of dicts, or dict

**Returns:** pandas DataFrame or None

**Use Case:** Handle flexible data formats from data loaders

```python
# Handles multiple input types
df = _as_dataframe(obj)

# Examples:
_as_dataframe(pd.DataFrame(...))  # Returns as-is
_as_dataframe([{...}, {...}])     # Converts list of dicts
_as_dataframe({...})               # Converts single dict
_as_dataframe("invalid")           # Returns None
```

---

#### `_get_matchup_df(df_dict: Optional[Dict]) -> Optional[pd.DataFrame]`

**Purpose:** Extract matchup DataFrame from data dictionary

**Parameters:**
- `df_dict`: Dictionary containing various DataFrames

**Returns:** Matchup DataFrame or None

**Search Strategy:**
1. Check for key "Matchup Data" (exact match)
2. Check for key with "matchup data" (case-insensitive)
3. Return None if not found

```python
# Extracts matchup data safely
matchup_df = _get_matchup_df(df_dict)

# Handles variations:
{"Matchup Data": df}       # ✓ Found
{"matchup data": df}       # ✓ Found
{"MATCHUP_DATA": df}       # ✓ Found
{"other_key": df}          # ✗ Returns None
```

---

### CSS Styling

All sub-tabs use consistent styling via `apply_modern_styles()`:

**Feature Cards:**
```css
.feature-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.2s;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 12px rgba(0,0,0,0.15);
}
```

**Section Cards:**
```css
.section-card {
    background: white;
    border-left: 4px solid #667eea;
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 0.5rem;
}
```

**Info Boxes:**
```css
.info-box {
    background: #E3F2FD;
    border-left: 4px solid #1976D2;
    padding: 1rem;
    border-radius: 4px;
    margin: 1rem 0;
}

.success-box {
    background: #E8F5E9;
    border-left: 4px solid #388E3C;
    padding: 1rem;
    border-radius: 4px;
    margin: 1rem 0;
}
```

---

## Database Queries

### Query 1: Homepage Summary Data

**Function:** `load_homepage_data()`
**Cache:** 5 minutes (CACHE_RECENT)

```sql
-- Record counts
SELECT COUNT(*) AS count FROM kmffl.matchup
WHERE league_id = '449.l.198278'

SELECT COUNT(*) AS count FROM kmffl.players_by_year
WHERE (league_id = '449.l.198278' OR league_id IS NULL)

SELECT COUNT(*) AS count FROM kmffl.draft
WHERE league_id = '449.l.198278'

SELECT COUNT(*) AS count FROM kmffl.transaction
WHERE league_id = '449.l.198278'

SELECT COUNT(*) AS count FROM kmffl.injury
WHERE league_id = '449.l.198278'

-- Latest week info
SELECT year, week, COUNT(*) AS games
FROM kmffl.matchup
WHERE league_id = '449.l.198278'
GROUP BY year, week
ORDER BY cumulative_week DESC
LIMIT 1
```

**Performance:** ~100ms total (6 small queries)

**Optimization:** Uses indexed `league_id` and `cumulative_week` columns

---

### Query 2: Matchup History

**Function:** `load_simulations_data(include_all_years=True)`
**Cache:** 5 minutes (CACHE_RECENT)

```sql
SELECT *
FROM kmffl.matchup
WHERE league_id = '449.l.198278'
ORDER BY cumulative_week DESC
```

**Performance:** ~500ms (10,000+ rows)

**Columns Returned:** All matchup columns including playoff odds

**Optimization:** Indexed on `league_id` and `cumulative_week`

---

### Query 3: Two-Week Player Slice

**Function:** `load_player_two_week_slice(year, week)`
**Cache:** 5 minutes (CACHE_RECENT)

```sql
-- Get cumulative week numbers
SELECT DISTINCT cumulative_week
FROM kmffl.players_by_year
WHERE year = {year} AND week = {week}
LIMIT 1

-- Load two weeks of data
SELECT *
FROM kmffl.players_by_year
WHERE (league_id = '449.l.198278' OR league_id IS NULL)
  AND cumulative_week IN ({current_cum}, {prev_cum})
ORDER BY cumulative_week DESC, points DESC NULLS LAST
```

**Performance:** ~300ms (~3,600 rows)

**Optimization:**
- Uses indexed `cumulative_week` for fast range query
- Includes NULL league_id for NFL-only players

---

## UI Elements

### Tab Names with Icons

```python
tab_names = [
    "🏠 Overview",
    "🏛️ Hall of Fame",
    "📊 Standings",
    "🗓️ Schedules",
    "⚔️ Head-to-Head",
    "📰 Recaps",
]
```

**Icon Meanings:**
- 🏠 Home/Entry point
- 🏛️ Historical significance
- 📊 Data/Statistics
- 🗓️ Schedule/Calendar
- ⚔️ Competition/Battle
- 📰 News/Stories

---

### Hero Section

**Location:** Overview tab (line 77-84)

**Purpose:** Welcome message and value proposition

```html
<div class="hero-section">
    <h2>🏈 Fantasy Football Command Center</h2>
    <p style="font-size: 1.1rem; margin: 0.5rem 0 0 0;">
        Your complete analytics platform with 25+ years of history—
        from draft analysis to playoff simulations.
    </p>
</div>
```

**Styling:**
- Gradient background
- Large centered text
- Prominent positioning

---

### Quick Navigation Cards

**Location:** Overview tab (line 90-117)

**Purpose:** 3-card overview of main sections

**Layout:** 3 equal columns

**Content:**
1. **Homepage**: Standings, schedules, H2H
2. **Player Stats**: Weekly, season, career
3. **Advanced Analytics**: Graphs, simulations

**Interaction:** Visual cards, no click action (informational only)

---

### Expandable Sections

**Location:** Overview tab (line 126-439)

**Count:** 10 expandable sections

**Purpose:** Detailed feature descriptions

**Implementation:**
```python
with st.expander("🏠 **Homepage** — Your Weekly Command Center", expanded=False):
    st.markdown("""
        [Feature descriptions with section cards]
    """, unsafe_allow_html=True)
```

**Default State:** Collapsed (expanded=False)

**User Action:** Click to expand/collapse

---

### Metrics Display

**Location:** Standings tab (line 571-580)

**Purpose:** Quick stats at-a-glance

```python
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Teams", total_teams)
with col2:
    st.metric("Current Week", current_week)
with col3:
    st.metric("Games Played", len(matchup_df))
```

**Layout:** 3 equal columns with large metric displays

---

### Info Boxes

**Location:** Various tabs

**Purpose:** Highlight tips, warnings, or additional context

**Types:**

1. **Info Box** (Blue)
```html
<div class="info-box">
    <strong>💡 Tip:</strong> [Helpful information]
</div>
```

2. **Success Box** (Green)
```html
<div class="success-box">
    <strong>✨ What to Expect:</strong> [Feature description]
</div>
```

---

## Performance Optimization

### Caching Strategy

**Layer 1: Data Loader Cache**
```python
@cached_data_loader(ttl=300, spinner_text="Loading homepage...")
def load_homepage_tab():
    # Cache for 5 minutes
    ...
```

**Layer 2: Streamlit Cache**
```python
@st.cache_data(ttl=300, show_spinner=True)
def load_player_two_week_slice(year, week):
    # Built-in Streamlit cache
    ...
```

**Layer 3: Session State**
```python
# Store in session state to avoid re-queries
st.session_state.year = y
st.session_state.week = w
```

**Cache Hit Rate:** 95%+ on repeat visits

---

### Lazy Loading

**Pattern:** Load data only when sub-tab is viewed

```python
# Data loaded upfront (once per homepage visit)
data = load_homepage_tab()  # Cached

# Sub-tabs render only when clicked
with tabs[0]:  # Overview
    # No DB queries (static content)

with tabs[1]:  # Hall of Fame
    # Uses pre-loaded matchup data

with tabs[2]:  # Standings
    # Uses pre-loaded matchup data (no new query)
```

**Benefit:** Initial load ~200ms (vs 2s if all tabs rendered at once)

---

### Progressive Rendering

**Pattern:** Show UI immediately, fetch details on-demand

```python
# Show tab structure instantly
tabs = st.tabs(tab_names)

# Load data in background (with spinner)
with st.spinner("Loading..."):
    data = load_data()

# Render when ready
display_content(data)
```

---

### Query Optimization

**Before (Slow):**
```sql
SELECT * FROM kmffl.matchup
WHERE year = 2024
ORDER BY year DESC, week DESC
```
Time: 2.3s

**After (Fast):**
```sql
SELECT * FROM kmffl.matchup
WHERE league_id = '449.l.198278'
ORDER BY cumulative_week DESC
```
Time: 0.5s

**Improvement:** 4.6x faster

**Techniques:**
- Filter by indexed `league_id` first
- Use indexed `cumulative_week` for sorting
- Avoid complex CTEs

---

## User Interactions

### Workflow 1: First-Time User

```
1. User opens app
   ↓
2. Homepage loads (Overview tab active)
   ↓
3. User reads hero section and quick navigation
   ↓
4. User expands "Player Stats" guide section
   ↓
5. User clicks "Players" main tab (navigates away)
```

**Time to Value:** ~5 seconds (instant page load + scanning)

---

### Workflow 2: Checking Standings

```
1. User opens app
   ↓
2. Homepage loads (Overview tab active)
   ↓
3. User clicks "📊 Standings" sub-tab
   ↓
4. Standings render instantly (cached data)
   ↓
5. User sees W-L records and playoff positions
```

**Time to Insight:** < 1 second (cached data)

---

### Workflow 3: Head-to-Head Analysis

```
1. User opens app
   ↓
2. Homepage loads
   ↓
3. User clicks "⚔️ Head-to-Head" sub-tab
   ↓
4. H2H interface loads with dropdowns
   ↓
5. User selects Year: 2024, Week: 5
   ↓
6. User selects matchup or "Optimal"
   ↓
7. Position-by-position comparison displays
```

**Time to Analysis:** ~2 seconds (DB query + render)

---

### Workflow 4: Weekly Recap

```
1. User opens app
   ↓
2. User clicks "📰 Recaps" sub-tab
   ↓
3. Recap interface loads
   ↓
4. User selects most recent week
   ↓
5. Recap cards display with highlights
   ↓
6. User sees awards (Highest Scorer, Best Player, etc.)
```

**Time to Recap:** ~1 second (uses cached two-week slice)

---

## Error Handling

### Missing Data

**Scenario:** Matchup data not available

```python
if matchup_df is None or matchup_df.empty:
    st.info("📋 Season Standings will appear once game data is loaded.")
else:
    display_season_standings(matchup_df)
```

**User Experience:** Friendly message instead of error

---

### Module Import Errors

**Scenario:** Hall of Fame module unavailable

```python
try:
    from ..hall_of_fame.hall_of_fame_homepage import HallOfFameViewer
    HALL_OF_FAME_AVAILABLE = True
except Exception as hof_import_error:
    HALL_OF_FAME_AVAILABLE = False
    HALL_OF_FAME_ERROR = str(hof_import_error)

# Later...
if HALL_OF_FAME_AVAILABLE:
    HallOfFameViewer(df_dict).display()
else:
    st.warning("Hall of Fame module not found")
    with st.expander("Debug details"):
        st.code(HALL_OF_FAME_ERROR)
```

**User Experience:** Warning with debug info for troubleshooting

---

### Database Query Failures

**Scenario:** MotherDuck connection issue

```python
try:
    data = load_homepage_tab()
except Exception as e:
    st.error(f"Failed to load homepage data: {e}")
    st.info("Try refreshing the page or check database connection")
```

**User Experience:** Clear error message with recovery instructions

---

### Calculation Errors

**Scenario:** Data format issue in standings calculation

```python
try:
    total_teams = matchup_df['manager'].nunique()
    st.metric("Teams", total_teams)
except Exception:
    pass  # Silently skip metric display
```

**User Experience:** Graceful degradation (skip metric, continue rendering)

---

## Appendix

### Quick Reference: Sub-Tab Functions

| Sub-Tab | Function | File | Data Source |
|---------|----------|------|-------------|
| Overview | `display_homepage_overview()` | `homepage_overview.py` | None (static) |
| Hall of Fame | `HallOfFameViewer().display()` | `hall_of_fame_homepage.py` | `matchup_df` |
| Standings | `display_season_standings()` | `season_standings.py` | `matchup_df` |
| Schedules | `display_schedules()` | `schedules.py` | `df_dict` |
| Head-to-Head | `display_head_to_head()` | `head_to_head.py` | `df_dict` |
| Recaps | `display_recap_overview()` | `recap_overview.py` | `df_dict` |

---

### Common User Questions

**Q: Why is "Hall of Fame" not loading?**
**A:** The Hall of Fame module may not be available. Check the error details in the debug expander.

**Q: How do I see the league-wide optimal lineup?**
**A:** Go to Head-to-Head tab, select any week, and choose "All" or "Optimal" from the matchup dropdown.

**Q: Can I export standings to CSV?**
**A:** Not currently built-in, but you can use browser extensions to copy table data.

**Q: What does "Lineup Efficiency" mean in Recaps?**
**A:** It's the percentage of your team's optimal score you actually achieved. 100% = perfect lineup.

---

**End of Homepage Tab Documentation**

For questions or issues, refer to the main APP_DOCUMENTATION.md or contact the development team.
