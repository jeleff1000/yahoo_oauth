# Player Visualization Analysis & Recommendations

**Date**: November 21, 2025
**Location**: `streamlit_ui/tabs/player_stats/graphs/`

---

## Current Player Visualizations

### 1. **Player Consistency Analysis** (`player_graphs/player_consistency.py`)

**Type**: Multi-tab analysis with bar charts, scatter plots, and data tables

**Features**:
- ✅ Horizontal bar chart showing top performers by PPG
- ✅ Scatter plot: PPG vs Games Started (volume analysis)
- ✅ Color-coded by position
- ✅ Size bubbles by total points
- ✅ Median reference lines
- ✅ Full data table with sorting
- ✅ Filters: Year range, position, roster status, started games only
- ✅ Minimum games threshold slider

**Strengths**:
- Excellent for identifying reliable players vs boom/bust
- Great visual separation of high-volume vs high-efficiency players
- Useful for finding "consistent performers"
- Good use of Plotly interactive features

**Potential Improvements**:
- Add coefficient of variation (CV) metric for true consistency
- Include week-to-week variance visualization
- Add floor/ceiling metrics (lowest/highest weekly scores)
- Show "bust rate" (% of games below certain threshold)

**Rating**: ⭐⭐⭐⭐ (4/5) - Very good, could add more consistency metrics

---

### 2. **Player Scoring Trends** (`player_graphs/player_scoring_graph.py`)

**Type**: Line charts with player search functionality

**Features**:
- ✅ Search multiple players by name
- ✅ Year-over-year performance comparison
- ✅ Cumulative average overlay (career trajectory)
- ✅ Single season vs multi-season views
- ✅ Position filters
- ✅ Roster filter (1999+ vs 2014+ data)

**Strengths**:
- Great for player comparisons
- Shows career progression clearly
- Cumulative average is insightful
- Flexible search interface

**Limitations**:
- Single season view only shows aggregates (no weekly breakdown yet)
- Comment in code notes: "Weekly breakdown requires weekly data access"
- Would benefit from actual weekly trends within a season

**Potential Improvements**:
- ✅ **HIGH PRIORITY**: Add weekly scoring line chart for single season
- Add moving average (3-game, 5-game rolling average)
- Show seasonal peaks and valleys
- Add injury markers or game context
- Compare to position average (z-score visualization)
- Add trendline/regression for career projection

**Rating**: ⭐⭐⭐⭐ (4/5) - Excellent concept, needs weekly granularity

---

### 3. **Position Group Scoring** (`player_graphs/position_group_scoring.py`)

**Type**: Multi-line chart comparing positions over time

**Features**:
- ✅ Position-level aggregation
- ✅ Average PPG by position over seasons
- ✅ Manager filter (see roster strength by position)
- ✅ Year-over-year trends table
- ✅ Detailed data pivot table
- ✅ Summary statistics

**Strengths**:
- Good for meta-analysis (position value trends)
- Useful for draft strategy
- Shows scoring inflation/deflation by position
- Manager filter adds strategic value

**Potential Improvements**:
- Add depth chart analysis (RB1 vs RB2 vs RB3 average)
- Show position scarcity metrics
- Add replacement value calculations
- Compare top-5, top-10, top-20 by position
- Show position volatility over time

**Rating**: ⭐⭐⭐⭐ (4/5) - Great strategic tool, could add depth analysis

---

## Visualization Gaps & Opportunities

### Missing Visualizations (High Priority)

#### 1. **Weekly Performance Heatmap** 🔥
**Description**: Calendar-style heatmap showing player performance week-by-week
- **X-axis**: Week (1-17)
- **Y-axis**: Players or seasons
- **Color**: Points scored (gradient)
- **Use case**: Quickly spot consistency patterns, identify hot/cold stretches

#### 2. **Player Usage & Efficiency Charts** 🔥
**Description**: Scatter plots showing volume vs efficiency
- **Examples**:
  - QB: Pass Attempts vs Yards/Attempt
  - RB: Carries vs Yards/Carry + target share
  - WR/TE: Targets vs Catch Rate + Yards/Reception
- **Use case**: Find efficient players, predict regression

#### 3. **Game Log Timeline** 🔥
**Description**: Detailed week-by-week view with context
- **Features**:
  - Points per week (line + bars)
  - Opponent defense ranking overlay
  - Home/away indicators
  - Injury designations
  - Weather conditions (if available)
- **Use case**: Deep dive into player performance factors

#### 4. **Player Comparison Spider/Radar Chart**
**Description**: Multi-dimensional comparison
- **Axes**: PPG, Consistency, Win Rate, Peak Performance, Floor, Ceiling
- **Use case**: Visual "player card" comparison

#### 5. **Boom/Bust Distribution** 🔥
**Description**: Histogram/violin plot showing score distribution
- **Show**: Frequency of point ranges (0-5, 5-10, 10-15, 15-20, 20+)
- **Compare**: Multiple players side-by-side
- **Use case**: Understand risk profile

#### 6. **Roster Construction Analyzer**
**Description**: Stacked area chart showing position allocation
- **Show**: Points contribution by position over time
- **Use case**: Understand roster balance and strength areas

#### 7. **Streak Tracker**
**Description**: Visual timeline of scoring streaks
- **Highlight**:
  - Games over X points (green)
  - Games under X points (red)
  - Win/loss streaks with player
- **Use case**: Identify momentum and reliability

#### 8. **Draft Pick Value Chart**
**Description**: Scatter plot of draft position vs season performance
- **X-axis**: Draft pick number
- **Y-axis**: Total points or PPG
- **Color**: Position
- **Use case**: Identify value picks and draft strategy

### Missing Visualizations (Medium Priority)

#### 9. **Age Curve Analysis**
**Description**: Performance by player age
- **Show**: Average PPG by age for each position
- **Use case**: Predict career arc, identify aging trends

#### 10. **Injury Impact Visualization**
**Description**: Before/after injury performance
- **Compare**: Pre-injury vs post-injury scoring
- **Use case**: Evaluate injury recovery

#### 11. **Strength of Schedule Impact**
**Description**: Player performance vs opponent strength
- **Show**: PPG vs opponent defensive ranking
- **Use case**: Identify schedule-driven variance

#### 12. **Position Rank Timeline**
**Description**: Track player's position rank over season
- **Y-axis**: Position rank (1=best, lower=worse)
- **X-axis**: Week
- **Use case**: See rising/falling trends

---

## Best Existing Visualizations (Ranked)

### 🥇 #1: Player Consistency - PPG vs Games Scatter
**Why it's great**:
- Shows both volume (games) and efficiency (PPG)
- Color by position adds another dimension
- Size by total points shows impact
- Median lines create quadrants for easy interpretation
- Actionable for roster decisions

**Best use case**: Finding reliable fantasy assets

---

### 🥈 #2: Player Scoring Trends - Multi-Year Comparison
**Why it's great**:
- Cumulative average shows career trajectory
- Player search is intuitive
- Good for comparing multiple players
- Shows progression/regression clearly

**Best use case**: Player evaluation and trade decisions

---

### 🥉 #3: Position Group Scoring - Position Trends
**Why it's great**:
- Strategic macro view
- Shows meta-game shifts
- Useful for draft prep
- Manager filter adds personalization

**Best use case**: Draft strategy and position value assessment

---

## Recommendations by Priority

### Immediate (Week 1-2)
1. ✅ **Move graphs to player_stats** (COMPLETED)
2. 🔥 **Add weekly heatmap visualization**
3. 🔥 **Create boom/bust distribution chart**
4. 🔥 **Build game log timeline view**

### Short Term (Week 3-4)
5. **Implement usage & efficiency charts** (QB/RB/WR/TE specific)
6. **Add player comparison radar charts**
7. **Create streak tracker visualization**
8. **Build draft value chart**

### Medium Term (Month 2)
9. **Develop roster construction analyzer**
10. **Add age curve analysis**
11. **Implement strength of schedule charts**
12. **Create position rank timeline**

---

## Technical Recommendations

### Visualization Library Choices

**Current**: Plotly (✅ Good choice!)
- Interactive by default
- Works well in Streamlit
- Good mobile support
- Rich chart types

**Consider adding**:
- `plotly.graph_objects` for custom layouts (already used)
- `plotly.subplots` for multi-chart dashboards
- `seaborn` for statistical visualizations (heatmaps, distributions)

### Performance Optimizations

1. **Caching**: Use `@st.cache_data` for data processing
2. **Fragments**: Keep using `@st.fragment` to isolate re-renders
3. **Lazy loading**: Only load charts when tabs are selected
4. **Data sampling**: For large datasets, offer summary views first

### UX Best Practices

1. ✅ **Multi-select filters** - Already using well
2. ✅ **Year range controls** - Good implementation
3. ✅ **Progressive disclosure** - Expanders hide complexity
4. **Add export options**: Let users download charts as PNG/SVG
5. **Add comparison mode**: Pin players to compare side-by-side
6. **Preset filters**: "Top 10 RBs", "Consistent WRs", etc.

---

## Data Access Patterns

**Current Architecture** (from code analysis):
```python
from md.data_access import (
    load_players_career_data,
    load_players_season_data,
    list_player_seasons
)
```

**Observations**:
- ✅ Good abstraction layer
- ✅ Supports year filtering
- ✅ Position filtering
- ✅ Rostered vs all players toggle

**Needs**:
- Weekly granularity functions (for weekly heatmaps)
- Opponent data for strength of schedule
- Stat categories beyond points (for usage charts)

---

## Integration with Existing Stats Tables

**Current Structure**:
```
player_stats/
├── weekly_player_stats_optimized.py
├── season_player_stats_optimized.py
├── career_player_stats_optimized.py
├── weekly_player_subprocesses/
├── season_player_subprocesses/
├── career_player_subprocesses/
└── graphs/  ← NEWLY MOVED HERE
    └── player_graphs/
        ├── player_consistency.py
        ├── player_scoring_graph.py
        └── position_group_scoring.py
```

**Recommendation**: Create a unified navigation
```python
# In player_stats main file, add visualization tab
tabs = st.tabs([
    "📊 Basic Stats",
    "🎯 Advanced Stats",
    "📈 Visualizations",  # ← Add this
    "🏆 Optimal Lineups"
])

with tabs[2]:
    viz_tabs = st.tabs([
        "Consistency",
        "Trends",
        "Position Groups",
        "Heatmaps",  # Future
        "Comparisons"  # Future
    ])
```

---

## Comparison to Team Stats

**Team Stats Structure** (for reference):
```
team_stats/
├── team_stats_overview.py
├── team_stats_visualizations.py  ← Centralized viz class
├── weekly_team_stats.py
├── season_team_stats.py
└── career_team_stats.py
```

**Key Difference**: Team stats has a `TeamStatsVisualizer` class

**Recommendation**: Consider creating `PlayerStatsVisualizer` class to:
- Centralize chart generation
- Share color schemes
- Reuse common patterns
- Make maintenance easier

---

## Color Scheme Recommendations

**Position Colors** (use consistently across all charts):
```python
POSITION_COLORS = {
    'QB': '#3B82F6',   # Blue (already using in badges)
    'RB': '#10B981',   # Green
    'WR': '#F59E0B',   # Orange
    'TE': '#EF4444',   # Red
    'K': '#8B5CF6',    # Purple
    'DEF': '#6B7280',  # Gray
    'FLEX': '#EC4899'  # Pink
}
```

**Performance Gradients**:
- **Good**: Green → Emerald (#10B981 → #059669)
- **Average**: Yellow → Amber (#F59E0B → #D97706)
- **Poor**: Red → Dark Red (#EF4444 → #DC2626)

---

## Conclusion

**Current State**:
- 3 solid visualizations with good interactivity
- Strong foundation using Plotly
- Good filtering and data access patterns

**Opportunity**:
- 12+ new visualization concepts identified
- Weekly granularity is the biggest gap
- Usage/efficiency metrics are missing
- Comparison tools need development

**Overall Assessment**: ⭐⭐⭐⭐ (4/5)
- Excellent start with room for significant expansion
- Moving to player_stats directory improves organization
- Ready for next phase of development

**Next Steps**:
1. Implement weekly heatmap (highest impact)
2. Add boom/bust distributions
3. Build game log timeline
4. Create usage efficiency charts

---

## Additional Resources

**Plotly Documentation**:
- Heatmaps: https://plotly.com/python/heatmaps/
- Subplots: https://plotly.com/python/subplots/
- Animations: https://plotly.com/python/animations/

**Inspiration**:
- FantasyPros player charts
- ESPN fantasy player card
- Yahoo fantasy trends
- NFL.com Next Gen Stats visualizations
