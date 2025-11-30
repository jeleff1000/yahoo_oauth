# App & UI/UX Recommendations

> **Scope:** Streamlit UI, user experience, visual design, mobile/dark mode, performance
> **Goal:** Create engaging, fast, professional UI that works everywhere

---

## Executive Summary

Your app has **excellent analytical depth** - the metrics and visualizations are sophisticated. The main opportunities are:

1. **Add "Report Card" summaries** - Quick visual grades at top of each section
2. **Consolidate tabs** - Reduce cognitive load (8 tabs → 5 in some areas)
3. **Add engagement features** - Badges, streaks, achievements
4. **Improve mobile experience** - Responsive column hiding
5. **Polish dark mode** - CSS custom properties

---

## Current State Assessment

### What's Working Well

| Area | Strength |
|------|----------|
| **Data Depth** | 280+ columns of analytics - best-in-class |
| **Dual SPAR** | Player vs Manager SPAR is sophisticated |
| **Simulations** | 100K Monte Carlo iterations is serious |
| **Caching** | 600s TTL caching is appropriate |
| **Column Selection** | 78% reduction for managers tab |
| **DuckDB Integration** | Fast in-memory queries |

### Areas for Improvement

| Area | Issue | Impact |
|------|-------|--------|
| **Quick Insights** | No summary cards at top | Users have to dig |
| **Grades/Tiers** | Data exists but not visualized as grades | Less engaging |
| **Tab Count** | Some sections have 8+ tabs | Overwhelming |
| **Mobile** | Tables too wide | Poor mobile UX |
| **Dark Mode** | Not consistently styled | Visual polish |

---

## Priority 1: Add Report Card Components

### Draft Report Card

```python
def display_draft_report_card(df: pd.DataFrame, manager: str = None):
    """Display quick visual summary at top of draft section."""

    if manager:
        data = df[df['manager'] == manager]
    else:
        data = df

    # Calculate metrics
    avg_grade = data['draft_grade'].mode().iloc[0] if 'draft_grade' in data else 'N/A'
    steals = (data['value_tier'] == 'Steal').sum() if 'value_tier' in data else 0
    busts = (data['value_tier'] == 'Bust').sum() if 'value_tier' in data else 0
    total_spar = data['manager_spar'].sum()
    avg_roi = data['draft_roi'].mean()

    # Display as metric cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="report-card-metric">
            <div class="metric-label">Draft Grade</div>
            <div class="metric-value grade-{avg_grade.lower()}">{avg_grade}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("💎 Steals", steals)

    with col3:
        st.metric("💀 Busts", busts)

    with col4:
        st.metric("Total SPAR", f"{total_spar:.0f}")

    with col5:
        st.metric("Avg ROI", f"{avg_roi:.2f}")
```

### Transaction Report Card

```python
def display_transaction_report_card(df: pd.DataFrame, manager: str = None):
    """Display quick visual summary at top of transactions section."""

    if manager:
        data = df[df['manager'] == manager]
    else:
        data = df

    # Calculate metrics
    net_spar = data['net_manager_spar_ros'].sum()
    total_adds = (data['transaction_type'] == 'add').sum()
    total_drops = (data['transaction_type'] == 'drop').sum()
    faab_spent = data['faab_bid'].sum()
    faab_efficiency = net_spar / max(faab_spent, 1)

    # Grade based on net SPAR
    if net_spar > 200: grade = 'A'
    elif net_spar > 100: grade = 'B'
    elif net_spar > 0: grade = 'C'
    elif net_spar > -50: grade = 'D'
    else: grade = 'F'

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="report-card-metric">
            <div class="metric-label">Transaction Grade</div>
            <div class="metric-value grade-{grade.lower()}">{grade}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("NET SPAR", f"{net_spar:+.0f}")

    with col3:
        st.metric("Adds / Drops", f"{total_adds} / {total_drops}")

    with col4:
        st.metric("FAAB Spent", f"${faab_spent:.0f}")

    with col5:
        st.metric("SPAR/FAAB", f"{faab_efficiency:.2f}")
```

### Season Report Card (Matchups)

```python
def display_season_report_card(df: pd.DataFrame, manager: str, year: int):
    """Display quick visual summary for a manager's season."""

    season = df[(df['manager'] == manager) & (df['year'] == year)]

    wins = season['win'].sum()
    losses = season['loss'].sum()
    ppg = season['team_points'].mean()
    schedule_luck = season['wins_vs_shuffle_wins'].iloc[-1] if len(season) > 0 else 0
    playoff_odds = season['p_playoffs'].iloc[-1] if 'p_playoffs' in season else 0
    efficiency = season['lineup_efficiency'].mean() * 100

    # Calculate grade
    win_pct = wins / max(wins + losses, 1)
    if win_pct >= 0.7: grade = 'A'
    elif win_pct >= 0.55: grade = 'B'
    elif win_pct >= 0.45: grade = 'C'
    elif win_pct >= 0.35: grade = 'D'
    else: grade = 'F'

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div class="report-card-metric">
            <div class="metric-label">Season Grade</div>
            <div class="metric-value grade-{grade.lower()}">{grade}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Record", f"{wins}-{losses}")

    with col3:
        st.metric("PPG", f"{ppg:.1f}")

    with col4:
        luck_label = "🍀" if schedule_luck > 0 else "😢"
        st.metric(f"Luck {luck_label}", f"{schedule_luck:+.1f}")

    with col5:
        st.metric("Playoff %", f"{playoff_odds:.0f}%")

    with col6:
        st.metric("Efficiency", f"{efficiency:.0f}%")
```

---

## Priority 2: Tab Consolidation

### Draft Section (8 → 5 tabs)

| Current | Proposed | Content |
|---------|----------|---------|
| Summary | **Overview** | Summary + Quick Stats + Report Card |
| Performance | **Performance** | Charts + Value Picks |
| Value | (merged) | Combined into Performance |
| Optimizer | **Tools** | Optimizer + Pricing Reference |
| Trends | **Trends** | Preferences + Spending Patterns |
| Pricing | (merged) | Combined into Tools |
| Career | **History** | Career Stats + Keeper Analysis |
| Keeper Analysis | (merged) | Combined into History |

### Transactions Section (Already Good)

Current structure is fine - 2 main tabs (Add/Drop, Trades) with sub-tabs.

### Matchups Section (4 tabs - Good)

Current structure is fine - Weekly, Seasons, Career, Visualize.

### Simulations Section (Good)

Current structure is fine - Predictive Analytics, What-If Scenarios.

---

## Priority 3: Engagement Features

### Achievement Badges

```python
ACHIEVEMENTS = {
    # Draft achievements
    'draft_master': {
        'name': 'Draft Master',
        'icon': '🎯',
        'condition': lambda df, mgr: (df[df['manager'] == mgr]['draft_grade'] == 'A').sum() >= 5,
        'description': '5+ A-grade draft picks'
    },
    'steal_hunter': {
        'name': 'Steal Hunter',
        'icon': '💎',
        'condition': lambda df, mgr: (df[df['manager'] == mgr]['value_tier'] == 'Steal').sum() >= 3,
        'description': '3+ draft steals in a season'
    },

    # Transaction achievements
    'waiver_wizard': {
        'name': 'Waiver Wizard',
        'icon': '🧙',
        'condition': lambda df, mgr: df[df['manager'] == mgr]['net_manager_spar_ros'].sum() > 150,
        'description': '150+ NET SPAR from transactions'
    },
    'faab_efficient': {
        'name': 'FAAB Efficient',
        'icon': '💰',
        'condition': lambda df, mgr: (df[df['manager'] == mgr]['spar_efficiency'] > 5).sum() >= 3,
        'description': '3+ transactions with SPAR/$ > 5'
    },

    # Matchup achievements
    'clutch_performer': {
        'name': 'Clutch Performer',
        'icon': '🎪',
        'condition': lambda df, mgr: (df[df['manager'] == mgr]['is_clutch_win'] == 1).sum() >= 5,
        'description': '5+ clutch wins (underdog or close game)'
    },
    'champion': {
        'name': 'Champion',
        'icon': '🏆',
        'condition': lambda df, mgr: (df[df['manager'] == mgr]['champion'] == 1).any(),
        'description': 'Won a championship'
    },
    'dynasty': {
        'name': 'Dynasty',
        'icon': '👑',
        'condition': lambda df, mgr: (df[df['manager'] == mgr]['champion'] == 1).sum() >= 3,
        'description': '3+ championships'
    }
}

def display_achievements(df: pd.DataFrame, manager: str):
    """Display earned achievement badges."""
    earned = []
    for key, achievement in ACHIEVEMENTS.items():
        if achievement['condition'](df, manager):
            earned.append(achievement)

    if earned:
        st.markdown("### 🏅 Achievements")
        cols = st.columns(min(4, len(earned)))
        for i, ach in enumerate(earned):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div class="achievement-badge">
                    <div class="badge-icon">{ach['icon']}</div>
                    <div class="badge-name">{ach['name']}</div>
                    <div class="badge-desc">{ach['description']}</div>
                </div>
                """, unsafe_allow_html=True)
```

### Streak Tracking

```python
def display_active_streaks(df: pd.DataFrame, manager: str):
    """Display current active streaks."""

    manager_data = df[df['manager'] == manager].sort_values(['year', 'week'])

    # Get most recent streaks
    latest = manager_data.iloc[-1] if len(manager_data) > 0 else None

    if latest is None:
        return

    streaks = []

    # Win streak
    if latest.get('win_streak', 0) >= 3:
        streaks.append({
            'icon': '🔥',
            'label': 'Win Streak',
            'value': f"{int(latest['win_streak'])} games"
        })

    # Loss streak (show as "cold streak")
    if latest.get('loss_streak', 0) >= 3:
        streaks.append({
            'icon': '🥶',
            'label': 'Cold Streak',
            'value': f"{int(latest['loss_streak'])} games"
        })

    # Scoring streak (above median)
    above_median_streak = (manager_data['above_league_median'] == 1).iloc[-5:].sum()
    if above_median_streak >= 4:
        streaks.append({
            'icon': '📈',
            'label': 'Hot Scoring',
            'value': f"{above_median_streak} weeks above median"
        })

    if streaks:
        st.markdown("#### Active Streaks")
        cols = st.columns(len(streaks))
        for i, streak in enumerate(streaks):
            with cols[i]:
                st.markdown(f"""
                <div class="streak-badge">
                    {streak['icon']} **{streak['label']}**: {streak['value']}
                </div>
                """, unsafe_allow_html=True)
```

---

## Priority 4: Mobile Responsiveness

### Responsive Column Hiding

```python
def get_responsive_columns(columns: list, is_mobile: bool) -> list:
    """Return appropriate columns based on screen size."""

    # Priority tiers for columns
    ESSENTIAL = ['year', 'manager', 'player', 'points', 'grade']
    IMPORTANT = ['position', 'cost', 'spar', 'opponent', 'margin']
    NICE_TO_HAVE = ['ppg', 'rank', 'efficiency', 'tier', 'type']

    if is_mobile:
        # Mobile: only essential + 1-2 important
        return [c for c in columns if any(e in c.lower() for e in ESSENTIAL + IMPORTANT[:2])]
    else:
        # Desktop: show all
        return columns

# Usage with Streamlit
def display_responsive_table(df: pd.DataFrame, key: str):
    """Display table with responsive columns."""

    # Detect mobile (approximate - Streamlit doesn't have true detection)
    # Could use query params or session state for user preference

    is_mobile = st.checkbox("Mobile view", value=False, key=f"{key}_mobile")

    display_cols = get_responsive_columns(df.columns.tolist(), is_mobile)

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True
    )
```

### Compact Card Layout for Mobile

```python
def display_matchup_card_mobile(row: pd.Series):
    """Display matchup as compact card for mobile."""

    st.markdown(f"""
    <div class="matchup-card-mobile">
        <div class="matchup-header">
            Week {row['week']} • {row['year']}
        </div>
        <div class="matchup-teams">
            <div class="team {'winner' if row['win'] else ''}">
                {row['manager']}: {row['team_points']:.1f}
            </div>
            <div class="team {'winner' if not row['win'] else ''}">
                {row['opponent']}: {row['opponent_points']:.1f}
            </div>
        </div>
        <div class="matchup-result">
            {'W' if row['win'] else 'L'} by {abs(row['margin']):.1f}
        </div>
    </div>
    """, unsafe_allow_html=True)
```

---

## Priority 5: Dark Mode & Theming

### CSS Custom Properties

```css
/* Add to your main CSS file */

:root {
    /* Light mode (default) */
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fa;
    --bg-card: #ffffff;
    --text-primary: #1a1a2e;
    --text-secondary: #6c757d;
    --text-muted: #adb5bd;
    --border-color: #dee2e6;
    --accent-primary: #667eea;
    --accent-secondary: #764ba2;
    --success: #28a745;
    --warning: #ffc107;
    --danger: #dc3545;
    --info: #17a2b8;

    /* Grade colors */
    --grade-a: #28a745;
    --grade-b: #5cb85c;
    --grade-c: #f0ad4e;
    --grade-d: #d9534f;
    --grade-f: #c9302c;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-card: #0f3460;
        --text-primary: #e8e8e8;
        --text-secondary: #b8b8b8;
        --text-muted: #888888;
        --border-color: #2d3748;
        --accent-primary: #a855f7;
        --accent-secondary: #ec4899;
    }
}

/* Component styles using variables */
.report-card-metric {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}

.metric-value {
    color: var(--text-primary);
    font-size: 2rem;
    font-weight: bold;
}

.grade-a { color: var(--grade-a); }
.grade-b { color: var(--grade-b); }
.grade-c { color: var(--grade-c); }
.grade-d { color: var(--grade-d); }
.grade-f { color: var(--grade-f); }

.achievement-badge {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    color: white;
}

.streak-badge {
    background: var(--bg-secondary);
    border-left: 4px solid var(--accent-primary);
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
}

/* Hero sections */
.hero-section {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}

/* Tables */
.dataframe {
    background: var(--bg-card);
    color: var(--text-primary);
}

.dataframe th {
    background: var(--bg-secondary);
    color: var(--text-primary);
}

.dataframe td {
    border-color: var(--border-color);
}
```

---

## Priority 6: New UI Components

### Clinch/Elimination Tracker (Simulations)

```python
def display_playoff_scenarios(df: pd.DataFrame, year: int, current_week: int):
    """Display clinch and elimination scenarios."""

    current = df[(df['year'] == year) & (df['week'] == current_week)]

    clinched = current[current['has_clinched_playoffs'] == 1]['manager'].tolist()
    eliminated = current[current['is_eliminated'] == 1]['manager'].tolist()
    bubble = current[
        (current['p_playoffs'] > 10) &
        (current['p_playoffs'] < 90)
    ]['manager'].tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ✅ Clinched")
        for mgr in clinched:
            st.markdown(f"- {mgr}")
        if not clinched:
            st.markdown("*No one yet*")

    with col2:
        st.markdown("### ❌ Eliminated")
        for mgr in eliminated:
            st.markdown(f"- {mgr}")
        if not eliminated:
            st.markdown("*No one yet*")

    with col3:
        st.markdown("### 🎯 Bubble Watch")
        for mgr in bubble:
            odds = current[current['manager'] == mgr]['p_playoffs'].iloc[0]
            st.markdown(f"- {mgr} ({odds:.0f}%)")
```

### Head-to-Head Rivalry Card

```python
def display_rivalry_card(df: pd.DataFrame, manager1: str, manager2: str):
    """Display head-to-head rivalry summary."""

    h2h = df[
        ((df['manager'] == manager1) & (df['opponent'] == manager2)) |
        ((df['manager'] == manager2) & (df['opponent'] == manager1))
    ]

    m1_wins = h2h[(h2h['manager'] == manager1) & (h2h['win'] == 1)].shape[0]
    m2_wins = h2h[(h2h['manager'] == manager2) & (h2h['win'] == 1)].shape[0]

    m1_ppg = h2h[h2h['manager'] == manager1]['team_points'].mean()
    m2_ppg = h2h[h2h['manager'] == manager2]['team_points'].mean()

    biggest_win = h2h[h2h['manager'] == manager1]['margin'].max()
    biggest_loss = h2h[h2h['manager'] == manager1]['margin'].min()

    st.markdown(f"""
    <div class="rivalry-card">
        <div class="rivalry-header">⚔️ {manager1} vs {manager2}</div>
        <div class="rivalry-record">
            <span class="manager1">{m1_wins}</span>
            <span class="separator">-</span>
            <span class="manager2">{m2_wins}</span>
        </div>
        <div class="rivalry-stats">
            <div>PPG: {m1_ppg:.1f} vs {m2_ppg:.1f}</div>
            <div>Biggest Win: +{biggest_win:.1f}</div>
            <div>Biggest Loss: {biggest_loss:.1f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

### Player Card Component

```python
def display_player_card(player_data: pd.Series, show_spar: bool = True):
    """Display an individual player summary card."""

    # Get player season stats
    name = player_data['player']
    position = player_data['nfl_position']
    team = player_data['nfl_team']
    points = player_data['points']
    rank = player_data.get('position_season_rank', 'N/A')
    headshot = player_data.get('headshot_url', '')

    # Calculate grade
    grade = player_data.get('performance_grade', 'C')
    spar = player_data.get('manager_spar', 0)
    consistency = player_data.get('consistency_tier', 'Normal')

    st.markdown(f"""
    <div class="player-card">
        <div class="player-header">
            <img src="{headshot}" class="player-headshot" onerror="this.style.display='none'"/>
            <div class="player-info">
                <h3>{name}</h3>
                <span class="position-badge">{position} • {team}</span>
            </div>
            <div class="player-grade grade-{grade.lower()}">{grade}</div>
        </div>
        <div class="player-stats">
            <div class="stat">
                <span class="stat-value">{points:.1f}</span>
                <span class="stat-label">Points</span>
            </div>
            <div class="stat">
                <span class="stat-value">#{rank}</span>
                <span class="stat-label">Position Rank</span>
            </div>
            <div class="stat">
                <span class="stat-value">{spar:+.1f}</span>
                <span class="stat-label">SPAR</span>
            </div>
            <div class="stat">
                <span class="stat-value">{consistency}</span>
                <span class="stat-label">Type</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

### Boom/Bust Meter

```python
def display_boom_bust_meter(player_df: pd.DataFrame, player_key: str, year: int):
    """Display visual boom/bust consistency meter for a player."""

    player_season = player_df[
        (player_df['player_key'] == player_key) &
        (player_df['year'] == year)
    ].sort_values('week')

    if player_season.empty:
        st.warning("No data for this player/season")
        return

    booms = player_season['is_boom'].sum()
    busts = player_season['is_bust'].sum()
    games = len(player_season)

    boom_pct = (booms / games) * 100 if games > 0 else 0
    bust_pct = (busts / games) * 100 if games > 0 else 0
    normal_pct = 100 - boom_pct - bust_pct

    st.markdown(f"""
    <div class="boom-bust-meter">
        <div class="meter-header">Boom/Bust Meter</div>
        <div class="meter-bar">
            <div class="meter-segment boom" style="width: {boom_pct}%"></div>
            <div class="meter-segment normal" style="width: {normal_pct}%"></div>
            <div class="meter-segment bust" style="width: {bust_pct}%"></div>
        </div>
        <div class="meter-labels">
            <span class="boom-label">🚀 {booms} Booms ({boom_pct:.0f}%)</span>
            <span class="bust-label">💀 {busts} Busts ({bust_pct:.0f}%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

### Team Composition Breakdown

```python
def display_team_composition(team_data: pd.DataFrame, manager: str, year: int = None):
    """Display visual breakdown of team composition by position."""

    if year:
        data = team_data[(team_data['manager'] == manager) & (team_data['year'] == year)]
    else:
        data = team_data[team_data['manager'] == manager]

    # Group by position
    by_position = data.groupby('fantasy_position').agg({
        'points': 'sum',
        'manager_spar': 'sum'
    }).reset_index()

    total_points = by_position['points'].sum()
    total_spar = by_position['manager_spar'].sum()

    # Position colors
    POS_COLORS = {
        'QB': '#ff6b6b', 'RB': '#4ecdc4', 'WR': '#45b7d1',
        'TE': '#f9ca24', 'K': '#a29bfe', 'DEF': '#fd79a8'
    }

    st.markdown("### Team Composition")

    for _, row in by_position.iterrows():
        pos = row['fantasy_position']
        pts = row['points']
        spar = row['manager_spar']
        pct = (pts / total_points * 100) if total_points > 0 else 0
        color = POS_COLORS.get(pos, '#95a5a6')

        st.markdown(f"""
        <div class="position-row">
            <div class="position-label" style="color: {color}">{pos}</div>
            <div class="position-bar-container">
                <div class="position-bar" style="width: {pct}%; background: {color}"></div>
            </div>
            <div class="position-stats">
                <span>{pts:.0f} pts</span>
                <span>({spar:+.0f} SPAR)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Summary
    st.markdown(f"""
    <div class="team-summary">
        <div class="summary-stat">
            <span class="label">Total Points</span>
            <span class="value">{total_points:.0f}</span>
        </div>
        <div class="summary-stat">
            <span class="label">Total SPAR</span>
            <span class="value">{total_spar:+.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

### SPAR Capture Rate Explainer

```python
def display_spar_capture_explainer(player_df: pd.DataFrame, manager: str, year: int):
    """Display visual explanation of SPAR capture rate."""

    data = player_df[(player_df['manager'] == manager) & (player_df['year'] == year)]

    total_player_spar = data['player_spar'].sum()
    total_manager_spar = data['manager_spar'].sum()
    capture_rate = (total_manager_spar / total_player_spar * 100) if total_player_spar > 0 else 0

    st.markdown(f"""
    <div class="spar-explainer">
        <h4>SPAR Capture Rate: {capture_rate:.0f}%</h4>
        <p class="explainer-text">
            <strong>Player SPAR ({total_player_spar:.0f})</strong>: Total value your players produced all season
            <br>
            <strong>Manager SPAR ({total_manager_spar:.0f})</strong>: Value produced while on YOUR roster
        </p>
        <div class="capture-visual">
            <div class="bar-container">
                <div class="bar player-spar" style="width: 100%">
                    <span>Player: {total_player_spar:.0f}</span>
                </div>
            </div>
            <div class="bar-container">
                <div class="bar manager-spar" style="width: {capture_rate}%">
                    <span>Manager: {total_manager_spar:.0f}</span>
                </div>
            </div>
        </div>
        <p class="insight">
            {f"🎯 Great capture rate! You maximized your players' value." if capture_rate > 90 else
             f"📈 Room to improve - traded away or benched {100-capture_rate:.0f}% of player value." if capture_rate < 70 else
             f"✅ Solid - you captured most of your players' production."}
        </p>
    </div>
    """, unsafe_allow_html=True)
```

---

## Implementation Checklist

### Phase 1: Report Cards (2-3 hours)
- [ ] Create `components/report_cards.py`
- [ ] Add Draft Report Card to draft overview
- [ ] Add Transaction Report Card to transactions overview
- [ ] Add Season Report Card to matchups overview

### Phase 2: Tab Consolidation (1-2 hours)
- [ ] Consolidate draft tabs (8 → 5)
- [ ] Update navigation
- [ ] Test all views still accessible

### Phase 3: Engagement Features (2-3 hours)
- [ ] Create `components/achievements.py`
- [ ] Add achievement badges to manager profiles
- [ ] Add streak tracking to weekly view

### Phase 4: Mobile/Responsive (2-3 hours)
- [ ] Create `utils/responsive.py`
- [ ] Add mobile toggle to key tables
- [ ] Test on mobile browser

### Phase 5: Dark Mode (1-2 hours)
- [ ] Create `static/theme.css`
- [ ] Apply CSS variables to all components
- [ ] Test in both modes

### Phase 6: New Components (3-4 hours)
- [ ] Add Clinch/Elimination tracker
- [ ] Add Rivalry cards
- [ ] Add This Day in History (Hall of Fame)

### Phase 7: Player Stats UI (2-3 hours)
- [ ] Add Player Card component
- [ ] Add Boom/Bust Meter visualization
- [ ] Add SPAR Capture Rate explainer
- [ ] Add player search with headshots

### Phase 8: Team Stats UI (2-3 hours)
- [ ] Add Team Composition breakdown visual
- [ ] Add Position contribution charts
- [ ] Add SPAR by position heatmap
- [ ] Add manager comparison tool

---

## Player Stats Section Enhancements

### Suggested New Features

| Feature | Impact | Effort |
|---------|--------|--------|
| Player Card with headshot | High engagement | Low |
| Boom/Bust meter | Visual appeal | Low |
| SPAR Capture Rate explainer | Education | Medium |
| Position group comparisons | Insight | Medium |
| "My Best Players" highlight | Engagement | Low |

### Tab Structure (Already Good)

Current structure is appropriate:
- Weekly (4 sub-tabs: Basic, Advanced, Matchup, H2H)
- Season
- Career
- Graphs (12 visualizations)

### Suggested Additions

1. **Player Profile Page** - Click a player to see dedicated profile with:
   - Career stats with this manager
   - Performance timeline
   - Boom/bust history
   - Trade/acquisition history

2. **"My Best Pickups" Section** - Show players with highest SPAR while on your roster

3. **Position Leaderboards** - Quick view of top performers by position

---

## Team Stats Section Enhancements

### Suggested New Features

| Feature | Impact | Effort |
|---------|--------|--------|
| Team Composition visual | High insight | Medium |
| Position contribution pie | Visual appeal | Low |
| SPAR by position breakdown | Education | Low |
| Manager comparison tool | Engagement | Medium |
| Lineup optimization suggestions | Utility | High |

### Suggested Additions

1. **Team Builder View** - Show optimal vs actual lineup side by side

2. **Position Weakness Analysis** - Highlight positions underperforming

3. **Historical Team Strength** - How team composition evolved over seasons

---

*This document consolidates all UI/UX recommendations. See `RECOMMENDATIONS_DATA_PIPELINE.md` for data pipeline recommendations.*
