# Tab Data Access Migration Guide

## 🎯 Quick Start: Optimizing a New Tab

This guide shows you how to migrate a tab to use the optimized data access pattern.

## 📋 Step-by-Step Process

### Step 1: Analyze Current Data Usage

**Goal:** Identify what data the tab actually needs.

```bash
# Find all column accesses in the tab
cd streamlit_ui/tabs/[your_tab]
grep -r "\[\"" --include="*.py" | grep -v "#" | head -20

# Find DataFrame operations
grep -r "df\." --include="*.py" | head -20
```

**Questions to answer:**
- Which tables does this tab query?
- Which columns from each table are actually used?
- Are there any redundant data loads?
- Are there multiple queries that could be combined?

### Step 2: Check Source Data

**For matchup table:**
```python
import pandas as pd
df = pd.read_parquet("path/to/matchup.parquet")
print(f"Total columns: {len(df.columns)}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
print("\nColumns:", df.columns.tolist())
```

**Document findings:**
```
Tab: [Your Tab Name]
Source Table: matchup
Total Columns: 276
Columns Used: [list the 10-20 columns actually accessed]
Savings: ~XX% reduction
```

### Step 3: Create Directory and Files

```bash
# Create directory
mkdir streamlit_ui/md/tab_data_access/[your_tab]

# Create initial files
cd streamlit_ui/md/tab_data_access/[your_tab]
touch __init__.py
touch [data_source]_data.py  # e.g., matchup_data.py
touch combined.py
```

### Step 4: Define Column Constants

**File: `[your_tab]/matchup_data.py`** (example)

```python
#!/usr/bin/env python3
"""
[Brief description of what data this loads]

Optimization: [Explain the optimization, e.g., "Loads only 15 of 276 columns"]
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.data_access import run_query, T

# Define ONLY the columns your tab needs
YOUR_TAB_COLUMNS = [
    "year",
    "week",
    "manager",
    # ... add only columns you actually use
]


@st.cache_data(show_spinner=True, ttl=600)
def load_your_tab_matchup_data() -> Dict[str, Any]:
    """
    Load matchup data for [your tab].

    Columns loaded ([X] out of 276):
        - year, week: Time dimensions
        - manager: Team identifier
        - ... [document what each group of columns is for]

    Returns:
        Dict with "Matchup Data" key or "error" key on failure
    """
    try:
        cols_str = ", ".join(YOUR_TAB_COLUMNS)

        query = f"""
            SELECT {cols_str}
            FROM {T['matchup']}
            ORDER BY year DESC, week DESC
        """

        df = run_query(query)
        return {"Matchup Data": df}

    except Exception as e:
        st.error(f"Failed to load [your tab] matchup data: {e}")
        return {"error": str(e)}
```

### Step 5: Combine Related Queries

If you have multiple separate queries, combine them:

**Before (5 queries):**
```python
count1 = run_query("SELECT COUNT(*) FROM table1")
count2 = run_query("SELECT COUNT(*) FROM table2")
count3 = run_query("SELECT COUNT(*) FROM table3")
# ... etc
```

**After (1 query):**
```python
@st.cache_data(ttl=600)
def load_combined_stats():
    query = """
        WITH stats1 AS (SELECT COUNT(*) AS cnt1 FROM table1),
             stats2 AS (SELECT COUNT(*) AS cnt2 FROM table2),
             stats3 AS (SELECT COUNT(*) AS cnt3 FROM table3)
        SELECT *
        FROM stats1
        CROSS JOIN stats2
        CROSS JOIN stats3
    """
    result = run_query(query)
    row = result.iloc[0]
    return {
        "count1": int(row["cnt1"]),
        "count2": int(row["cnt2"]),
        "count3": int(row["cnt3"]),
    }
```

### Step 6: Create Combined Entry Point

**File: `[your_tab]/combined.py`**

```python
#!/usr/bin/env python3
"""
Combined data loader for [Your Tab].

Main entry point for loading all data needed by this tab.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from .[data_source]_data import load_your_tab_matchup_data
# Import other loaders as needed


@st.cache_data(show_spinner=True, ttl=600)
def load_optimized_[your_tab]_data() -> Dict[str, Any]:
    """
    Load all data for [your tab] in one optimized call.

    Returns:
        Dict with all data or "error" key on failure
    """
    try:
        # Load each data source
        matchup_data = load_your_tab_matchup_data()

        # Check for errors
        if "error" in matchup_data:
            return {"error": matchup_data["error"]}

        # Combine and return
        return {
            "Matchup Data": matchup_data["Matchup Data"],
            # Add other data sources as needed
        }

    except Exception as e:
        st.error(f"Failed to load [your tab] data: {e}")
        return {"error": str(e)}
```

### Step 7: Update `__init__.py`

**File: `[your_tab]/__init__.py`**

```python
"""
[Your Tab] tab data access.

Optimized loaders for [Your Tab] components.
"""
from .[data_source]_data import load_your_tab_matchup_data
from .combined import load_optimized_[your_tab]_data

__all__ = [
    "load_your_tab_matchup_data",
    "load_optimized_[your_tab]_data",
]
```

### Step 8: Update App Imports

**In `app_homepage_optimized.py` (or your main app file):**

**Before:**
```python
from md.data_access import load_[your_tab]_data

def load_[your_tab]_tab():
    return load_[your_tab]_data()
```

**After:**
```python
from md.tab_data_access.[your_tab] import load_optimized_[your_tab]_data

def load_[your_tab]_tab():
    with monitor.time_operation("load_[your_tab]"):
        data = load_optimized_[your_tab]_data()
        if "error" in data:
            st.error(f"Failed to load data: {data['error']}")
            return {}
        return data
```

### Step 9: Test & Measure

**Test functionality:**
```bash
streamlit run streamlit_ui/app_homepage_optimized.py
# Navigate to your tab and verify everything works
```

**Measure improvements:**
```python
# Add this to measure before/after
import time
import pandas as pd

# Before (if you have old data file)
start = time.time()
df_old = pd.read_parquet("old_data.parquet")  # All columns
time_old = time.time() - start
mem_old = df_old.memory_usage(deep=True).sum() / 1024 / 1024

# After
start = time.time()
data_new = load_optimized_[your_tab]_data()
time_new = time.time() - start
df_new = data_new.get("Matchup Data")
mem_new = df_new.memory_usage(deep=True).sum() / 1024 / 1024 if df_new is not None else 0

print(f"Columns: {len(df_old.columns)} → {len(df_new.columns)}")
print(f"Memory: {mem_old:.1f} MB → {mem_new:.1f} MB ({100*(1-mem_new/mem_old):.0f}% reduction)")
print(f"Time: {time_old:.2f}s → {time_new:.2f}s")
```

### Step 10: Document

Update these files:
1. **This migration guide** - Add your tab to completed list
2. **tab_data_access/README.md** - Add your tab structure
3. **Commit message** - Include performance metrics

## 📝 Template: Column Selection Checklist

Use this checklist when determining which columns to keep:

```python
# For matchup table (276 columns available)
COLUMN_SELECTION_CHECKLIST = {
    # Time dimensions (usually needed)
    "year": True,           # ✓ Needed for filtering by season
    "week": True,           # ✓ Needed for filtering by week
    "cumulative_week": ?,   # ? Only if doing time series

    # Team identifiers (usually needed)
    "manager": True,        # ✓ Always needed
    "manager_team": ?,      # ? Only if showing team names
    "opponent": ?,          # ? Only if showing opponent info

    # Scoring (depends on tab)
    "team_points": ?,       # ? Only if showing scores
    "opponent_points": ?,   # ? Only if showing opponent scores
    "margin": ?,            # ? Only if showing point differentials

    # Results (depends on tab)
    "win": ?,               # ? Only if showing W/L records
    "loss": ?,              # ? Only if showing W/L records

    # Playoffs (depends on tab)
    "is_playoffs": ?,       # ? Only if filtering by game type
    "playoff_round": ?,     # ? Only if showing playoff details
    "champion": ?,          # ? Only if showing championships

    # Advanced stats (usually not needed)
    "gavi_stat": False,     # ✗ Unless specifically displaying this
    "shuffle_*": False,     # ✗ Unless doing simulations
    # ... etc, default to False unless proven needed
}
```

## 🎯 Common Patterns

### Pattern 1: Simple Data Load

For tabs that just need basic matchup data:

```python
# Define columns
SIMPLE_COLUMNS = ["year", "week", "manager", "team_points", "win"]

# Single loader
@st.cache_data(ttl=600)
def load_simple_data():
    query = f"SELECT {', '.join(SIMPLE_COLUMNS)} FROM matchup"
    return {"data": run_query(query)}
```

### Pattern 2: Multiple Data Sources

For tabs that need data from multiple tables:

```python
# multiple_sources.py
@st.cache_data(ttl=600)
def load_matchup_subset():
    return run_query("SELECT [cols] FROM matchup")

@st.cache_data(ttl=600)
def load_player_subset():
    return run_query("SELECT [cols] FROM player")

# combined.py
def load_combined():
    return {
        "matchups": load_matchup_subset(),
        "players": load_player_subset(),
    }
```

### Pattern 3: Conditional Loading

For tabs with sub-tabs that need different data:

```python
# Load minimal data upfront
@st.cache_data(ttl=600)
def load_minimal_data():
    return run_query("SELECT year, week, manager FROM matchup")

# Load detailed data only when sub-tab is accessed
@st.cache_data(ttl=600)
def load_detailed_data(year: int, week: int):
    return run_query(f"SELECT * FROM matchup WHERE year={year} AND week={week}")

# In your tab renderer
def render_tab():
    minimal = load_minimal_data()
    selected_year = st.selectbox("Year", minimal["year"].unique())

    # Only load detailed when user selects
    detailed = load_detailed_data(selected_year, ...)
```

## ✅ Completed Migrations

- [x] **Homepage** - 17/276 columns (94% reduction, ~80% faster)
- [x] **Managers** - ~60/276 columns (78% reduction, ~70% faster)
- [x] **Keepers** - 16/272 cols + max week only (~99.7% data reduction!)
  - Column filtering: 94% reduction
  - Row filtering: ~95% reduction (max week per player/year)
  - Database-level filters: Excludes unrostered, DEF, K
- [ ] **Players** - TODO (already has pagination)
- [ ] **Draft** - TODO
- [ ] **Transactions** - TODO (has LIMIT, needs column selection)
- [ ] **Simulations** - TODO
- [ ] **Hall of Fame** - TODO

## 🆘 Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'md.tab_data_access'`

**Fix:** Make sure you have `__init__.py` files in:
- `md/tab_data_access/__init__.py`
- `md/tab_data_access/[your_tab]/__init__.py`

### Cache Errors

**Error:** `UnhashableType` or cache-related errors

**Fix:** Make sure your function arguments are hashable (strings, ints, tuples - not lists or dicts)

### Query Errors

**Error:** `Binder Error: Referenced column "X" not found in FROM clause`

**Fix:** Double-check your column names match the actual table schema. Use the parquet file to verify.

## 📚 Resources

- **Homepage Example**: See `streamlit_ui/md/tab_data_access/homepage/` for complete working example
- **README**: See `streamlit_ui/md/tab_data_access/README.md` for architecture overview
- **Original Summary**: See `streamlit_ui/HOMEPAGE_OPTIMIZATION_SUMMARY.md` for detailed metrics
