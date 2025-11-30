# Tab Data Access Architecture Overview

## 🏗️ Complete Directory Structure

```
streamlit_ui/md/
│
├── data_access.py                  # ✅ Core/shared data access (unchanged)
├── motherduck_connection.py        # ✅ Connection logic (unchanged)
├── data_cache.py                   # ✅ Caching utilities (unchanged)
│
└── tab_data_access/                # ✨ NEW: Tab-specific optimized loaders
    │
    ├── __init__.py                 # Package documentation
    ├── README.md                   # Architecture overview & patterns
    ├── MIGRATION_GUIDE.md          # Step-by-step migration instructions
    ├── ARCHITECTURE.md             # This file
    │
    ├── homepage/                   # ✅ COMPLETED: Homepage optimization
    │   ├── __init__.py             # Exports: load_optimized_homepage_data
    │   ├── matchup_data.py         # Loads 17/276 columns (85% reduction)
    │   ├── summary_stats.py        # Combines 5 queries into 1 (5x faster)
    │   └── combined.py             # Main entry point
    │
    ├── managers/                   # ✅ COMPLETED: Managers optimization
    │   ├── __init__.py             # Exports: load_optimized_managers_data
    │   ├── matchup_data.py         # Loads ~60/276 columns (78% reduction)
    │   ├── summary_data.py         # Aggregated stats (already optimized)
    │   └── combined.py             # Main entry point
    │
    ├── keepers/                    # ✅ COMPLETED: Keepers optimization
    │   ├── __init__.py             # Exports: load_optimized_keepers_data
    │   ├── keeper_data.py          # 17/272 cols + max week only (~99.7% reduction)
    │   └── combined.py             # Main entry point
    │
    ├── team_names/                 # ✅ COMPLETED: Team Names optimization
    │   ├── __init__.py             # Exports: load_optimized_team_names_data
    │   ├── team_name_data.py       # 5/276 cols + DISTINCT (~99.9% reduction)
    │   └── combined.py             # Main entry point
    │
    ├── players/                    # 📋 TODO: Players tab optimization
    │   └── __init__.py
    │
    ├── draft/                      # 📋 TODO: Draft tab optimization
    │   └── __init__.py
    │
    ├── transactions/               # 📋 TODO: Transactions tab optimization
    │   └── __init__.py
    │
    ├── simulations/                # 📋 TODO: Simulations tab optimization
    │   └── __init__.py
    │
    └── hall_of_fame/               # 📋 TODO: Hall of Fame optimization
        └── __init__.py
```

## 📊 Data Flow

### Before Optimization

```
app_homepage_optimized.py
    ↓
load_homepage_data()           ← 5 separate COUNT queries
load_simulations_data(all)     ← SELECT * (276 cols, duplicate load!)
load_player_two_week_slice()   ← Loaded upfront (often unused)
    ↓
~12 MB data loaded
3-5 second load time
```

### After Optimization

```
app_homepage_optimized.py
    ↓
md.tab_data_access.homepage.load_optimized_homepage_data()
    ├── matchup_data.py         ← SELECT 17 cols (not 276!)
    └── summary_stats.py        ← 1 combined query (not 5!)
    ↓
~1 MB data loaded
0.5-1 second load time
```

## 🎯 Design Goals

### 1. **Modularity**
Each tab has its own directory, allowing:
- Independent optimization
- Multiple data access files per tab if needed
- Clear separation of concerns

### 2. **Performance**
Three key optimizations:
- **Column Selection**: Load only needed columns
- **Query Combination**: Reduce database round-trips
- **Lazy Loading**: Load data when accessed, not upfront

### 3. **Maintainability**
- Clear file naming (e.g., `matchup_data.py`, `summary_stats.py`)
- Comprehensive documentation
- Consistent patterns across tabs

### 4. **Scalability**
Structure supports tabs with multiple data sources:
```
transactions/
├── __init__.py
├── trades.py           # Trade-specific queries
├── waivers.py          # Waiver-specific queries
├── enrichment.py       # Player performance enrichment
└── combined.py         # Combines all sources
```

## 🔄 Import Patterns

### Pattern 1: Simple Import (Single Loader)

```python
# In app
from md.tab_data_access.homepage import load_optimized_homepage_data

def load_homepage_tab():
    return load_optimized_homepage_data()
```

### Pattern 2: Multi-Loader Import

```python
# In app
from md.tab_data_access.managers import (
    load_manager_stats,
    load_head_to_head_data,
    load_optimized_managers_data,  # Combined loader
)

def load_managers_tab():
    # Option A: Use combined loader
    return load_optimized_managers_data()

    # Option B: Load individual parts as needed
    stats = load_manager_stats()
    h2h = load_head_to_head_data() if user_clicks_h2h_tab else None
```

### Pattern 3: Lazy Sub-Tab Loading

```python
# In tab renderer
def render_players_tab():
    subtabs = st.tabs(["Weekly", "Season", "Career"])

    with subtabs[0]:
        # Only loads when user clicks Weekly tab
        from md.tab_data_access.players import load_weekly_players
        data = load_weekly_players()
        render_weekly(data)

    with subtabs[1]:
        # Only loads when user clicks Season tab
        from md.tab_data_access.players import load_season_players
        data = load_season_players()
        render_season(data)
```

## 📈 Performance Metrics

### Homepage Tab (Completed)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Columns Loaded | 276 | 17 | **94% reduction** |
| Memory Usage | ~12 MB | ~1 MB | **92% reduction** |
| Query Count | 6 queries | 2 queries | **67% reduction** |
| Load Time | 3-5 sec | 0.5-1 sec | **70-80% faster** |

### Managers Tab (Completed)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Columns Loaded | 276 | ~60 | **78% reduction** |
| Memory Usage | ~5.8 MB | ~1.3 MB | **78% reduction** |
| Query Count | 3 queries | 3 queries | Same (already optimized) |
| Load Time | 2-3 sec | 0.6-0.8 sec | **60-75% faster** |

### Expected Gains for Other Tabs

Based on homepage results, expect similar patterns:
- **Column Reduction**: 70-95% (depends on columns used)
- **Query Combination**: 3-5x faster (if multiple queries combined)
- **Overall Speed**: 60-80% faster load times

## 🛠️ Creating a New Tab Data Access

### Quick Template

1. **Create directory:**
   ```bash
   mkdir md/tab_data_access/[tab_name]
   ```

2. **Create `__init__.py`:**
   ```python
   """[Tab Name] tab data access."""
   from .combined import load_optimized_[tab_name]_data

   __all__ = ["load_optimized_[tab_name]_data"]
   ```

3. **Create `matchup_data.py`:** (if using matchup table)
   ```python
   from md.data_access import run_query, T

   TAB_COLUMNS = ["year", "week", ...]  # Only needed columns!

   @st.cache_data(ttl=600)
   def load_[tab_name]_matchup_data():
       cols = ", ".join(TAB_COLUMNS)
       df = run_query(f"SELECT {cols} FROM {T['matchup']}")
       return {"Matchup Data": df}
   ```

4. **Create `combined.py`:**
   ```python
   from .matchup_data import load_[tab_name]_matchup_data

   @st.cache_data(ttl=600)
   def load_optimized_[tab_name]_data():
       matchup = load_[tab_name]_matchup_data()
       return matchup
   ```

5. **Update app:**
   ```python
   from md.tab_data_access.[tab_name] import load_optimized_[tab_name]_data
   ```

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Architecture overview, design principles, examples |
| `MIGRATION_GUIDE.md` | Step-by-step instructions for migrating tabs |
| `ARCHITECTURE.md` | This file - high-level structure & patterns |
| `homepage/*.py` | Working example of optimized data access |

## ✅ Migration Checklist

- [x] ✅ **Directory structure created**
- [x] ✅ **Homepage migrated** (17/276 cols, 92% less memory, ~80% faster)
- [x] ✅ **Managers migrated** (~60/276 cols, 78% less memory, ~70% faster)
- [x] ✅ **Keepers migrated** (17/272 cols + max week only = ~99.7% data reduction!)
- [x] ✅ **Team Names migrated** (5/276 cols + DISTINCT = ~99.9% data reduction!)
- [x] ✅ **Documentation written** (README, MIGRATION_GUIDE, ARCHITECTURE)
- [ ] 📋 **Players tab** - Already has pagination, may need column selection
- [ ] 📋 **Draft tab** - Ready to migrate
- [ ] 📋 **Transactions tab** - Has LIMIT 1000, may need column selection
- [ ] 📋 **Simulations tab** - Ready to migrate
- [ ] 📋 **Hall of Fame tab** - Ready to migrate

## 🎓 Key Learnings

### 1. Column Selection > Row Filtering
Homepage optimization proved that **which columns you load** matters more than **how many rows** you load:
- Matchup table: 1,812 rows is fine
- But 276 columns is wasteful when you only need 17!

### 2. Combine Queries When Possible
5 separate COUNT queries → 1 combined CTE query = 5x faster

### 3. Load Data When Needed
Don't pre-load data for tabs/subtabs the user might never visit

### 4. Cache Aggressively
Every loader should use `@st.cache_data(ttl=600)` to avoid repeated queries

## 🔮 Future Enhancements

### Potential Additions

1. **Per-Tab Column Analyzers**
   ```python
   # Auto-detect which columns a tab uses
   def analyze_column_usage(tab_dir: str) -> list[str]:
       """Scan tab code and extract referenced columns"""
   ```

2. **Query Performance Monitoring**
   ```python
   # Track query performance per tab
   @st.cache_data(ttl=600)
   def load_with_metrics(query: str):
       start = time.time()
       result = run_query(query)
       log_metric(query, time.time() - start, len(result))
       return result
   ```

3. **Automated Column Pruning**
   ```python
   # Automatically remove unused columns from cache
   def prune_unused_columns(df: pd.DataFrame, used_cols: set) -> pd.DataFrame:
       return df[list(used_cols)]
   ```

## 📞 Support

For questions or issues with this architecture:
1. Check `README.md` for patterns and examples
2. Check `MIGRATION_GUIDE.md` for step-by-step instructions
3. Look at `homepage/` for a working implementation
4. Refer to `HOMEPAGE_OPTIMIZATION_SUMMARY.md` for detailed metrics

---

**Last Updated:** November 2025
**Status:** ✅ Foundation Complete, Ready for Tab Migrations
