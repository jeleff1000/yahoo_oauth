# Tab-Specific Data Access Architecture

## 📁 Directory Structure

```
tab_data_access/
├── __init__.py              # Root package documentation
├── README.md                # This file
│
├── homepage/                # Homepage tab data access
│   ├── __init__.py         # Exports: load_optimized_homepage_data, etc.
│   ├── matchup_data.py     # Matchup data with column selection (17/276 cols)
│   ├── summary_stats.py    # Combined summary query (5 queries → 1)
│   └── combined.py         # Main entry point for homepage data
│
├── managers/                # Managers tab data access
│   └── __init__.py
│
├── keepers/                 # Keepers tab data access
│   ├── __init__.py
│   ├── keeper_data.py       # Keeper data: 17/272 cols + max week only (~99.7% reduction)
│   └── combined.py          # Main entry point for keepers data
│
├── team_names/              # Team Names tab data access
│   ├── __init__.py
│   ├── team_name_data.py    # Team name data: 5/276 cols + DISTINCT (~99.9% reduction)
│   └── combined.py          # Main entry point for team names data
│
├── players/                 # Players tab data access (TODO)
│   └── __init__.py
│
├── draft/                   # Draft tab data access (TODO)
│   └── __init__.py
│
├── transactions/            # Transactions tab data access (TODO)
│   └── __init__.py
│
├── simulations/             # Simulations tab data access (TODO)
│   └── __init__.py
│
└── hall_of_fame/            # Hall of Fame tab data access (TODO)
    └── __init__.py
```

## 🎯 Design Principles

### 1. **Tab-Specific Organization**
Each tab has its own directory containing focused data access modules. This:
- Keeps code modular and maintainable
- Makes it easy to find data logic for specific tabs
- Allows tabs to have multiple data access modules if needed

### 2. **Column-Specific Queries**
Always SELECT only the columns you need, not `SELECT *`:
```python
# ❌ BAD: Loads all 276 columns (5.8 MB)
query = "SELECT * FROM kmffl.matchup"

# ✅ GOOD: Loads only 17 needed columns (~1 MB)
HOMEPAGE_COLS = ["year", "week", "manager", "team_points", ...]
query = f"SELECT {', '.join(HOMEPAGE_COLS)} FROM kmffl.matchup"
```

### 3. **Combined Queries**
Reduce database round-trips by combining related queries:
```python
# ❌ BAD: 5 separate queries (5 round-trips)
matchup_count = run_query("SELECT COUNT(*) FROM matchup")
player_count = run_query("SELECT COUNT(*) FROM player")
...

# ✅ GOOD: 1 combined query (1 round-trip)
query = """
    WITH matchup_stats AS (SELECT COUNT(*) AS cnt FROM matchup),
         player_stats AS (SELECT COUNT(*) AS cnt FROM player),
         ...
    SELECT * FROM matchup_stats CROSS JOIN player_stats ...
"""
```

### 4. **Lazy Loading**
Load data when it's needed, not upfront:
```python
# ❌ BAD: Load all data at tab initialization
def load_tab():
    data1 = load_data_for_subtab1()  # Loaded even if user never visits
    data2 = load_data_for_subtab2()  # Loaded even if user never visits
    return {"data1": data1, "data2": data2}

# ✅ GOOD: Load data when user accesses it
def render_subtab1():
    data = load_data_for_subtab1()  # Only loaded when user clicks this tab
```

### 5. **Caching**
Always use `@st.cache_data` with appropriate TTL:
```python
@st.cache_data(show_spinner=True, ttl=600)  # 10 minutes
def load_my_data():
    return run_query("SELECT ...")
```

## 📖 Usage Examples

### Homepage Example (Complete)

**File: `homepage/matchup_data.py`**
```python
from md.data_access import run_query, T

HOMEPAGE_MATCHUP_COLUMNS = ["year", "week", "manager", ...]

@st.cache_data(ttl=600)
def load_homepage_matchup_data():
    cols_str = ", ".join(HOMEPAGE_MATCHUP_COLUMNS)
    query = f"SELECT {cols_str} FROM {T['matchup']}"
    df = run_query(query)
    return {"Matchup Data": df}
```

**File: `homepage/__init__.py`**
```python
from .matchup_data import load_homepage_matchup_data
from .summary_stats import load_homepage_summary_stats
from .combined import load_optimized_homepage_data

__all__ = [
    "load_homepage_matchup_data",
    "load_homepage_summary_stats",
    "load_optimized_homepage_data",
]
```

**Usage in app:**
```python
from md.tab_data_access.homepage import load_optimized_homepage_data

def load_homepage_tab():
    data = load_optimized_homepage_data()
    return data
```

### Future Tab Examples

#### Managers Tab (Planned)
```
managers/
├── __init__.py
├── manager_stats.py      # Season stats by manager
├── head_to_head.py       # H2H records between managers
└── matchup_history.py    # Historical matchup data
```

**Usage:**
```python
from md.tab_data_access.managers import (
    load_manager_stats,
    load_head_to_head_data,
)
```

#### Players Tab (Planned)
```
players/
├── __init__.py
├── weekly_players.py     # Weekly player data (with pagination)
├── season_players.py     # Season aggregates
└── career_players.py     # Career aggregates
```

#### Transactions Tab (Planned)
```
transactions/
├── __init__.py
├── trades.py             # Trade data
├── waivers.py            # Waiver/FAAB data
└── enrichment.py         # Player performance enrichment
```

## ✅ Migration Checklist

When optimizing a new tab, follow these steps:

1. **Analyze Current Usage**
   - [ ] Identify what data the tab actually needs
   - [ ] Find which columns are used from each table
   - [ ] Identify redundant data loads

2. **Create Directory Structure**
   ```bash
   mkdir md/tab_data_access/[tab_name]
   touch md/tab_data_access/[tab_name]/__init__.py
   ```

3. **Create Focused Modules**
   - [ ] Create one file per logical data domain (e.g., `matchup_data.py`)
   - [ ] Define column lists as constants at the top
   - [ ] Write focused query functions with `@st.cache_data`
   - [ ] Add docstrings explaining what columns are loaded and why

4. **Create Combined Loader**
   - [ ] Create `combined.py` as main entry point
   - [ ] Import and combine individual loaders
   - [ ] Handle errors gracefully

5. **Update __init__.py**
   - [ ] Export all public functions
   - [ ] Add docstring explaining what the module provides

6. **Update App Imports**
   - [ ] Replace old imports with new path
   - [ ] Test that everything still works

7. **Document**
   - [ ] Add comments explaining optimizations
   - [ ] Update this README with new tab structure
   - [ ] Create migration notes if needed

## 📊 Performance Guidelines

### Target Metrics

| Optimization | Target | Measurement |
|--------------|--------|-------------|
| Column Reduction | Load ≤20% of available columns | Compare loaded vs total columns |
| Query Combination | ≤3 queries per tab load | Count `run_query()` calls |
| Memory Usage | ≤5 MB per tab | Check DataFrame `.memory_usage()` |
| Load Time | ≤1 second per tab | Use `PerformanceMonitor` |

### Common Optimizations

1. **Column Selection**: Typically saves 70-90% of data transfer
2. **Query Combination**: Saves 3-5x on query execution time
3. **Lazy Loading**: Saves 50-80% of initial page load time
4. **Pagination**: For tables with >10K rows, load in chunks

## 🔧 Tools & Utilities

### Analyzing Column Usage

To find which columns a tab actually uses:
```bash
cd streamlit_ui/tabs/[tab_name]
grep -r "df\[" --include="*.py" | grep -o "\['[^']*'\]" | sort | uniq
```

### Measuring Memory Usage

```python
import pandas as pd

# Before optimization
df_full = pd.read_parquet("matchup.parquet")
print(f"Full: {df_full.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# After optimization
df_subset = df_full[NEEDED_COLUMNS]
print(f"Subset: {df_subset.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
```

### Testing Queries

```python
# Test a query outside Streamlit
from md.data_access import run_query
import time

start = time.time()
result = run_query("SELECT ...")
elapsed = time.time() - start

print(f"Rows: {len(result)}")
print(f"Columns: {len(result.columns)}")
print(f"Time: {elapsed:.2f}s")
print(f"Memory: {result.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
```

## 📚 Additional Resources

- **Homepage Optimization Summary**: See `HOMEPAGE_OPTIMIZATION_SUMMARY.md`
- **Data Access Guide**: See `md/DATA_ACCESS_OPTIMIZATION_GUIDE.md` (if exists)
- **Caching Best Practices**: [Streamlit Caching Docs](https://docs.streamlit.io/develop/concepts/architecture/caching)

## 🤝 Contributing

When adding new tab data access:
1. Follow the established patterns
2. Document your column choices
3. Add performance measurements to commit messages
4. Update this README with your tab's structure
