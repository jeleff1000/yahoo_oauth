# Simulations UI Documentation

> **Last Updated:** November 2024
> **Data Source:** matchup.parquet (ALL 280 columns required)
> **UI Location:** `streamlit_ui/tabs/simulations/`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [UI Components](#ui-components)
3. [Feature Summaries](#feature-summaries)
4. [Recommendations & Roadmap](#recommendations--roadmap)

---

## Executive Summary

The Simulations section provides advanced analytics using Monte Carlo simulations and predictive models. It answers "what-if" questions about schedules and provides real-time playoff projections.

### Key Capabilities

- **Playoff Predictions**: ML-based playoff odds, championship probability
- **Schedule Simulations**: 100,000 random schedule iterations
- **Strength of Schedule**: Opponent difficulty analysis
- **Score Sensitivity**: How small score changes affect outcomes

### Why ALL 280 Columns Are Needed

Unlike other sections that can use column subsets, Simulations requires:
- `shuffle_*` columns (35): Performance-based schedule simulations
- `opp_shuffle_*` columns (35): Opponent difficulty simulations
- `x*_seed`, `x*_win` columns (25): Predictive probabilities
- `p_*` columns (6): Playoff advancement odds
- All core columns for calculations

**Cannot be optimized further** - simulations depend on dynamically generated column sets.

---

## UI Components

### Current Implementation

The Simulations hub (`simulation_home.py`) contains **2 main tabs**:

---

### Tab 1: Predictive Analytics

**What It Shows:**
- ML-based predictions for season outcomes
- Real-time playoff odds
- Championship probability
- Week-by-week projection updates

**Sub-Tabs (5):**

#### 1. Playoff Dashboard (`playoff_simulation_enhanced.py`)
- Current playoff odds for all managers
- Championship probability
- Bye probability
- Advancement odds (Semis, Finals, Champion)
- Visual bracket projections

**Current Features:**
- Real-time odds calculation
- Visual probability bars
- Manager comparison
- Historical accuracy tracking

**Suggested Additions:**
- [ ] Add "Clinch Scenarios" - what needs to happen to clinch
- [ ] Add "Elimination Watch" - who's at risk
- [ ] Add "Magic Number" calculations
- [ ] Add playoff seeding tiebreaker explanations
- [ ] Add "Path to Championship" visualization

---

#### 2. Final Records (`predictive_record.py`)
- Predicted final win totals
- Win distribution histogram
- Confidence intervals
- Comparison to current record

**Suggested Additions:**
- [ ] Add "Best/Worst Case" scenarios
- [ ] Add remaining schedule difficulty
- [ ] Add win probability by remaining opponent

---

#### 3. Playoff Seeds (`predictive_seed.py`)
- Predicted final seeding
- Seed probability distribution
- Seed movement trends
- Seeding scenarios

**Suggested Additions:**
- [ ] Add "Seed Battles" - who's competing for same seed
- [ ] Add tiebreaker scenario analysis

---

#### 4. Weekly Odds (`playoff_odds_graph.py`)
- Week-by-week playoff odds chart
- Odds trajectory over season
- Key inflection points

**Suggested Additions:**
- [ ] Add "What Changed" annotations
- [ ] Add win/loss impact on odds

---

#### 5. Multi-Year Trends (`yearly_playoff_graph.py`)
- Historical playoff odds patterns
- Year-over-year comparisons
- Long-term trends

**Suggested Additions:**
- [ ] Add "Playoff Consistency" ranking
- [ ] Add historical accuracy analysis

---

### Tab 2: What-If Scenarios

**What It Shows:**
- Schedule-independent analysis
- "What if schedules were random?"
- "What if opponent difficulty was random?"
- Score sensitivity analysis

**Sub-Tabs (3):**

#### 1. Schedule Simulations

##### Win Distribution (`shuffled_win_total_viewer.py`)
- Distribution of wins across 100K simulated schedules
- Expected wins vs actual wins
- Schedule luck measurement

**Key Insight**: If your actual wins >> expected wins, you've been lucky with your schedule.

##### Head-to-Head (`vs_one_opponent_viewer.py`)
- "What if you played opponent X every week?"
- Simulated record vs each opponent
- Identifies "good matchups" and "bad matchups"

##### Expected Records (`expected_record_viewer.py`)
- Performance-based expected record
- `shuffle_avg_wins` analysis
- `wins_vs_shuffle_wins` (luck measure)

##### Expected Seeding (`expected_seed_viewer.py`)
- Performance-based expected seed
- `shuffle_avg_seed` analysis
- `seed_vs_shuffle_seed` (seed luck)

**Suggested Additions:**
- [ ] Add "Schedule Luck Leaderboard"
- [ ] Add "Hardest Schedule" analysis
- [ ] Add "If You Had X's Schedule" comparison

---

#### 2. Opponent Strength

##### Win Distribution (`opponent_shuffle_win_total.py`)
- Same simulation but shuffles OPPONENT points
- Measures opponent difficulty effect

**Key Insight**: This answers "How would your record change with easier/harder opponents?"

##### Your Scores vs All Schedules (`everyones_schedule_viewer.py`)
- Play your scores against every other schedule
- "If you had Jason's schedule, you'd be 10-4"
- Full schedule comparison matrix

##### Expected Records (`sos_expected_record_viewer.py`)
- `opp_shuffle_avg_wins` analysis
- Opponent-based expected record

##### Expected Seeding (`sos_expected_seed_viewer.py`)
- `opp_shuffle_avg_seed` analysis
- Opponent-based expected seed

**Suggested Additions:**
- [ ] Add "SOS Ranking" leaderboard
- [ ] Add "Opponent PPG Faced" comparison
- [ ] Add "Easy/Hard Schedule Weeks" breakdown

---

#### 3. Score Sensitivity (`tweak_scoring_viewer.py`)

**What It Shows:**
- How small score changes affect outcomes
- "If you scored 5 more points in Week 3..."
- Sensitivity analysis

**Suggested Additions:**
- [ ] Add "Close Game Impact" analysis
- [ ] Add "Points to Win" calculator
- [ ] Add "Margin Analysis" - how many close games

---

## Feature Summaries

### For Homepage

#### Simulations - Quick Summary

> **Advanced Analytics & Predictions**
>
> The Simulations section uses Monte Carlo methods and machine learning to provide playoff predictions and "what-if" scenario analysis. Run 100,000 schedule simulations to separate luck from skill.
>
> **Key Features:**
> - Real-time playoff and championship odds
> - Schedule luck analysis (actual vs expected record)
> - Strength of schedule comparisons
> - Score sensitivity analysis

---

### For About Page

#### Simulations - Detailed Description

> **What are Simulations?**
>
> The Simulations section provides advanced analytics using two main approaches:
>
> **1. Monte Carlo Simulations**
>
> We run 100,000 random schedule combinations to answer:
> - "What would my record be with a random schedule?"
> - "How much has schedule luck affected my season?"
> - "What's my expected seed based purely on performance?"
>
> **2. Predictive Analytics**
>
> Machine learning models provide:
> - Real-time playoff probability
> - Championship odds
> - Expected final record and seeding
> - Week-by-week projection updates
>
> **Key Metrics Explained:**
>
> | Metric | What It Means |
> |--------|---------------|
> | `shuffle_avg_wins` | Expected wins if schedules were random |
> | `wins_vs_shuffle_wins` | Actual - Expected (positive = lucky) |
> | `opp_shuffle_avg_wins` | Expected wins if opponent difficulty was random |
> | `p_playoffs` | Probability of making playoffs (%) |
> | `p_champ` | Probability of winning championship (%) |
> | `power_rating` | Composite team strength metric |
>
> **Simulation Types:**
>
> | Type | Question It Answers |
> |------|---------------------|
> | **Performance-Based** | "Given my scores, what should my record be?" |
> | **Opponent Difficulty** | "How hard has my schedule been?" |
> | **Head-to-Head** | "How would I do against any specific opponent?" |
> | **Score Sensitivity** | "How do small score changes affect outcomes?" |

---

### Section-by-Section Summaries

#### Predictive Analytics Tab
> "Machine learning-powered predictions for playoff odds, championship probability, and expected season outcomes."

#### What-If Scenarios Tab
> "Monte Carlo simulations answering 'what-if' questions about schedules, opponents, and scoring."

#### Schedule Simulations Sub-Tab
> "100,000 random schedule iterations to measure schedule luck and performance-based expectations."

#### Opponent Strength Sub-Tab
> "Analyze opponent difficulty and see how your record would change with different schedules."

#### Score Sensitivity Sub-Tab
> "Understand how small scoring changes affect wins, losses, and playoff positioning."

---

## Recommendations & Roadmap

### Priority 1: Add Summary Metrics to Source Table

```python
# Add to expected_record_v2.py

# 1. SCHEDULE LUCK TIER
df['schedule_luck_tier'] = pd.cut(
    df['wins_vs_shuffle_wins'].fillna(0),
    bins=[-10, -2, -0.5, 0.5, 2, 10],
    labels=['Very Unlucky', 'Unlucky', 'Normal', 'Lucky', 'Very Lucky']
)

# 2. SOS TIER (Strength of Schedule)
df['sos_tier'] = pd.cut(
    df['wins_vs_opp_shuffle_wins'].fillna(0),
    bins=[-10, -1.5, -0.5, 0.5, 1.5, 10],
    labels=['Very Easy', 'Easy', 'Average', 'Hard', 'Very Hard']
)

# 3. PLAYOFF ODDS TIER
df['playoff_odds_tier'] = pd.cut(
    df['p_playoffs'].fillna(0),
    bins=[0, 10, 30, 50, 70, 90, 100],
    labels=['Eliminated', 'Long Shot', 'Bubble', 'Likely', 'Very Likely', 'Locked']
)

# 4. CLINCH FLAG
df['has_clinched_playoffs'] = (df['p_playoffs'] >= 99.9).astype(int)
df['has_clinched_bye'] = (df['p_bye'] >= 99.9).astype(int)

# 5. ELIMINATION FLAG
df['is_eliminated'] = (df['p_playoffs'] <= 0.1).astype(int)
```

### Priority 2: UI Enhancements

| Enhancement | Impact | Effort |
|-------------|--------|--------|
| Add "Clinch Scenarios" | High engagement | High |
| Add "Magic Number" calculator | Understanding | Medium |
| Add schedule luck leaderboard | Fun | Low |
| Add "What Changed This Week" | Insight | Medium |
| Add elimination alerts | Drama | Low |

### Priority 3: New Components

1. **Clinch/Elimination Tracker** - What needs to happen to lock in
2. **Magic Number Calculator** - Games needed to clinch
3. **Scenario Builder** - "If X wins and Y loses..."
4. **Historical Accuracy Dashboard** - How accurate were predictions
5. **Playoff Bracket Simulator** - Interactive bracket projection

### Priority 4: Performance Considerations

```python
# Simulations REQUIRES all columns - cannot optimize column selection
# Current: Loads all 280 columns (SELECT *)

# Possible optimizations:
# 1. Lazy-load simulation tabs (only calculate when viewed)
# 2. Cache simulation results (don't recalculate every page load)
# 3. Pre-compute weekly summaries in source table
# 4. Background refresh for real-time odds

# Example caching improvement:
@st.cache_data(ttl=300)  # 5 minute cache
def load_simulation_data_cached():
    return load_simulation_matchup_data()
```

---

## Appendix: File Locations

### UI Components
```
KMFFLApp/streamlit_ui/tabs/simulations/
├── simulation_home.py                    # Main hub
├── predictive/
│   ├── __init__.py
│   ├── playoff_simulation_enhanced.py    # Dashboard
│   ├── playoff_odds.py                   # Odds snapshot
│   ├── playoff_odds_graph.py             # Weekly odds chart
│   ├── predictive_record.py              # Final record prediction
│   ├── predictive_seed.py                # Final seed prediction
│   ├── yearly_playoff_graph.py           # Multi-year trends
│   └── table_styles.py                   # Styling
├── what_if/
│   ├── __init__.py
│   ├── shuffle_schedules/
│   │   ├── __init__.py
│   │   ├── shuffled_win_total_viewer.py  # Win distribution
│   │   ├── vs_one_opponent_viewer.py     # H2H simulation
│   │   ├── expected_record_viewer.py     # Expected record
│   │   ├── expected_seed_viewer.py       # Expected seed
│   │   └── table_styles.py
│   ├── strength_of_schedule/
│   │   ├── __init__.py
│   │   ├── opponent_shuffle_win_total.py # Opp win distribution
│   │   ├── everyones_schedule_viewer.py  # Schedule comparison
│   │   ├── sos_expected_record_viewer.py # SOS record
│   │   ├── sos_expected_seed_viewer.py   # SOS seed
│   │   └── table_styles.py
│   └── tweak_scoring/
│       ├── shuffle_schedule.py
│       ├── shuffle_scores.py
│       └── tweak_scoring_viewer.py       # Score sensitivity
```

### Data Access
```
KMFFLApp/streamlit_ui/md/tab_data_access/simulations/
├── combined.py              # Entry point
└── matchup_data.py          # Full data loader (all 280 cols)
```

### Data Pipeline (Simulation Generation)
```
fantasy_football_data_scripts/multi_league/transformations/matchup_enrichment/
├── expected_record_v2.py              # Main simulation script
├── playoff_odds_import.py             # Predictive model import
└── modules/
    ├── schedule_simulation.py         # Core simulation logic
    └── bye_week_filler.py             # Bye week handling
```

---

## Technical Deep Dive: How Simulations Work

### Monte Carlo Schedule Simulation (100,000 iterations)

```python
# Pseudo-code for schedule simulation
for iteration in range(100_000):
    # Shuffle weekly scores across random matchups
    random_schedule = shuffle_scores(weekly_scores)

    # Calculate wins for each manager under this schedule
    for manager in managers:
        wins = count_wins(manager, random_schedule)
        win_histogram[manager][wins] += 1

# Result: Probability distribution of wins for each manager
# shuffle_0_win = P(0 wins), shuffle_1_win = P(1 win), etc.
```

### Dual Simulation Approach

**Performance-Based (`shuffle_*`)**:
- Keeps your scores fixed
- Shuffles who you play each week
- Answers: "Given my scoring, what should my record be?"

**Opponent Difficulty (`opp_shuffle_*`)**:
- Keeps your schedule fixed
- Shuffles opponent's scores
- Answers: "How hard has my schedule been?"

### Interpreting Results

| Metric | Positive Value Means | Negative Value Means |
|--------|---------------------|---------------------|
| `wins_vs_shuffle_wins` | Lucky schedule | Unlucky schedule |
| `wins_vs_opp_shuffle_wins` | Easy opponents | Hard opponents |
| `seed_vs_shuffle_seed` | Seeded better than expected | Seeded worse than expected |

---

*This documentation covers the Simulations UI. See `matchup_pipeline_documentation.md` for core pipeline details.*
