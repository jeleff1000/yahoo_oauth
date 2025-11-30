# Homepage, Keepers & Extras UI Documentation

> **Last Updated:** November 2024
> **UI Locations:**
> - `streamlit_ui/tabs/homepage/`
> - `streamlit_ui/tabs/keepers/`
> - `streamlit_ui/tabs/team_names/`
> - `streamlit_ui/app_homepage.py`

---

## Table of Contents

1. [App Entry Point](#app-entry-point)
2. [Homepage Section](#homepage-section)
3. [Keepers Section](#keepers-section)
4. [Team Names Section](#team-names-section)
5. [Shared Styles](#shared-styles)
6. [Recommendations](#recommendations)

---

## App Entry Point

### File: `app_homepage.py`

The main Streamlit application entry point with optimized lazy loading.

### Architecture

```
app_homepage.py
├── Main Navigation (8 tabs as radio buttons)
│   ├── Home → render_home_tab()
│   ├── Managers → render_managers_tab()
│   ├── Team Stats → render_team_stats_tab()
│   ├── Players → render_players_tab() [4 subtabs]
│   ├── Draft → render_draft_tab()
│   ├── Transactions → render_transactions_tab()
│   ├── Simulations → render_simulations_tab()
│   └── Extras → render_extras_tab() [Keepers, Team Names]
│
├── Performance Features
│   ├── PerformanceMonitor - tracks operation timing
│   ├── cached_data_loader - 300s TTL caching
│   ├── @st.fragment - partial page updates
│   └── Lazy loading - only loads active tab data
│
└── Styling
    ├── modern_styles.py - shared CSS
    └── Inline CSS for radio-as-tabs
```

### Key Optimizations

| Optimization | Implementation |
|--------------|----------------|
| **Lazy Tab Loading** | Only renders active tab content |
| **Cached Data Loaders** | 300s TTL on all data loads |
| **Column Selection** | 78-94% column reduction per tab |
| **Session State** | Tracks active tab/subtab indices |
| **Fragments** | `@st.fragment` for partial updates |

### Navigation Flow

```
Main Tabs (Radio styled as tabs)
├── Home
├── Managers
├── Team Stats
├── Players
│   └── Subtabs: Weekly | Season | Career | Visualize
├── Draft
├── Transactions
├── Simulations
└── Extras
    └── Subtabs: Keeper | Team Names
```

---

## Homepage Section

### Location: `streamlit_ui/tabs/homepage/`

### File Structure

```
homepage/
├── __init__.py
├── homepage_overview.py          # Main entry point (6 tabs)
├── champions.py                   # Championship history
├── season_standings.py            # Current standings
├── schedules.py                   # Week-by-week results
├── head_to_head.py                # Lineup comparisons
│
├── recaps/                        # Weekly narrative recaps
│   ├── __init__.py
│   ├── recap_overview.py          # Recap entry point
│   ├── recap_builder.py           # Builds recap narratives
│   ├── recap_config.py            # Recap templates
│   ├── weekly_recap_config.py     # Weekly-specific config
│   ├── narrative_engine.py        # Generates narratives
│   ├── displays/
│   │   ├── weekly_recap.py        # Weekly recap display
│   │   ├── season_recap.py        # Season recap display
│   │   └── player_recap.py        # Player recap display
│   └── helpers/
│       ├── contextual_helpers.py  # Context generation
│       └── recap_dialogue.py      # Dialogue templates
│
└── hall_of_fame/                  # Hall of Fame section
    ├── __init__.py
    ├── hall_of_fame_homepage.py   # HoF main view
    ├── top_teams.py               # Best seasons
    ├── top_players.py             # Best player performances
    ├── top_players_viewer.py      # Alternative player view
    ├── top_weeks.py               # Best single weeks
    ├── legendary_games.py         # Memorable matchups
    ├── records.py                 # All-time records
    ├── playoff_brackets.py        # Bracket visualization
    ├── styles.py                  # HoF-specific styles
    └── top_players/
        ├── top_player_seasons.py
        └── top_player_weeks.py
```

### Homepage Tabs (6)

| Tab | Description | Key Features |
|-----|-------------|--------------|
| **Overview** | App tour & quick navigation | Feature cards, FAQ, power user tips |
| **Hall of Fame** | Champions & records | Dynasty badges, playoff brackets |
| **Standings** | Current season rankings | W-L, PF/PA, PPG, playoff positioning |
| **Schedules** | Week-by-week results | Manager schedules, SOS analysis |
| **Head-to-Head** | Lineup comparisons | Position-by-position, optimal lineups |
| **Recaps** | Weekly narratives | Auto-generated stories, awards |

### Current Homepage Overview Tab

The Overview tab currently contains:

1. **Hero Section** - Gradient banner with title
2. **Quick Navigation Cards** (3 columns)
3. **Complete App Guide** - 9 collapsible expanders explaining all sections
4. **Power User Tips** - 2-column layout with tips
5. **Key Concepts** - Explanations of metrics
6. **FAQ** - Expandable Q&A section

**Issues with Current Design:**
- Too text-heavy for a landing page
- All expanders are collapsed by default (low discoverability)
- No visual engagement (no charts, metrics, or dynamic content)
- Feature cards are static HTML, not clickable
- No personalized content (e.g., "your record this week")

---

## Keepers Section

### Location: `streamlit_ui/tabs/keepers/`

### File: `keepers_home.py`

A comprehensive keeper analysis tool.

### Features

**3 Main Tabs:**

1. **Keeper Explorer** - Interactive data browser
   - Multi-select filters (year, manager, position)
   - Max keeper price filter
   - Keepers-only toggle
   - Sortable dataframe with headshots

2. **Analytics** - Visual trends
   - Keepers by position (bar chart)
   - Average SPAR/$ by position
   - Keeper trends over time (dual-axis chart)

3. **Best Keepers** - Value rankings
   - Top 10 by SPAR
   - Top 10 by SPAR/$
   - Worst keeper values (busts)
   - Manager keeper success rates

### Data Columns Used

```python
KEEPER_COLUMNS = [
    'headshot_url', 'player', 'yahoo_position', 'manager', 'year',
    'keeper_price', 'cost', 'max_faab_bid_to_date',
    'avg_points_this_year', 'avg_points_next_year', 'is_keeper_status',
    'spar', 'kept_next_year', 'division_id'
]
```

### Summary Metrics Displayed

- Total Keeper Slots (managers × years × 2)
- Actually Kept count
- Average Keeper Cost
- Keep Rate percentage

---

## Team Names Section

### Location: `streamlit_ui/tabs/team_names/`

### File: `team_names.py`

Displays historical team names in a matrix view.

### Features

**2 View Modes:**

1. **All Managers** - Full pivot table (years × managers)
2. **By Division** - Separate tables per division

### Implementation

- Pivot table with year rows, manager columns
- Team names displayed as badges
- Sticky headers and first column
- Scrollable container (max-height: 72vh)
- HSL color generation from team name text

### Data Columns Required

```python
TEAM_NAMES_COLUMNS = ['manager', 'year', 'team_name', 'division_id']
```

---

## Shared Styles

### Location: `streamlit_ui/tabs/shared/modern_styles.py`

Provides consistent CSS styling across all pages.

### Style Categories

| Category | Classes | Purpose |
|----------|---------|---------|
| **Hero/Header** | `.hero-section`, `.tab-header` | Gradient banners |
| **Feature Cards** | `.feature-card` | Clickable cards with icons |
| **Info Boxes** | `.info-box`, `.success-box`, `.warning-box` | Colored callouts |
| **Metric Cards** | `.metric-card` | KPI display boxes |
| **Section Cards** | `.section-card` | Content containers |
| **Tab Styling** | Native tabs, radio-as-tabs | Navigation styling |

### Dark Mode Support

```css
@media (prefers-color-scheme: dark) {
    .feature-card {
        background: linear-gradient(145deg, #2b2d31 0%, #1e1f22 100%);
        border-left: 4px solid #7289da;
    }
    .section-card {
        background: #2b2d31;
        border: 2px solid #3a3c41;
    }
}
```

### Mobile Responsive Breakpoints

| Breakpoint | Target | Key Changes |
|------------|--------|-------------|
| `768px` | Tablets | Reduced padding, smaller fonts |
| `480px` | Phones | Compact layout, stacked columns |
| `landscape` | Landscape phones | Further reduced headers |
| `hover: none` | Touch devices | Larger touch targets (44px min) |

---

## Recommendations

### Priority 1: Homepage Revamp

The current homepage is too text-heavy. Recommendations:

1. **Add dynamic hero section** with current week info
2. **Add quick stats cards** (your record, playoff odds, etc.)
3. **Add recent activity feed** (latest transactions, matchup results)
4. **Make feature cards clickable** to navigate to sections
5. **Add a "This Week" highlight section**

### Priority 2: Keepers Enhancements

| Enhancement | Impact | Effort |
|-------------|--------|--------|
| Add keeper recommendation engine | High | High |
| Add "Who to keep?" calculator | High | Medium |
| Add keeper value projections | Medium | Medium |
| Add historical keeper success rates | Low | Low |

### Priority 3: Team Names Improvements

| Enhancement | Impact | Effort |
|-------------|--------|--------|
| Add team name word cloud | Fun | Low |
| Add name change history | Insight | Low |
| Add division rivalry indicators | Engagement | Medium |

### Priority 4: App-Wide Improvements

| Enhancement | Impact | Effort |
|-------------|--------|--------|
| Add global search | High utility | High |
| Add notification badges | Engagement | Medium |
| Add keyboard shortcuts | Power users | Low |
| Add print/export modes | Utility | Medium |

---

## Appendix: Data Flow

### Homepage Data Loading

```python
# app_homepage.py
@cached_data_loader(ttl=300, spinner_text="Loading homepage...")
def load_homepage_tab():
    """
    KEY OPTIMIZATIONS:
    1. Only loads 17 columns from matchup table (85% reduction)
    2. Combines 5 summary queries into 1
    3. Removes redundant data loads
    """
    from md.tab_data_access.homepage import load_optimized_homepage_data
    return load_optimized_homepage_data()
```

### Extras Tab Data Loading

```python
# Keepers
from md.tab_data_access.keepers import load_optimized_keepers_data
keepers_data = load_optimized_keepers_data()

# Team Names
from md.tab_data_access.team_names import load_optimized_team_names_data
team_names_data = load_optimized_team_names_data()
```

---

*This documentation covers the Homepage, Keepers, and Team Names UI sections. See other documentation files for Draft, Transactions, Players, and Matchups.*
