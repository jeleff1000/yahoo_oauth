# MD Data Access Refactoring Plan

## Status: FULLY COMPLETED & OPTIMIZED

Refactoring completed on 2025-12-03.
Full optimization (all imports updated) completed same day.

---

## Summary of Changes

### New Architecture

```
md/
├── __init__.py                    # Package exports (from core.py)
├── core.py                        # Core primitives (~250 lines)
│   ├── Database config (get_current_league_db, get_table_dict, T)
│   ├── Connection (get_motherduck_connection, run_query)
│   ├── SQL helpers (sql_quote, sql_in_list, sql_upper, etc.)
│   ├── Common queries (list_seasons, list_weeks, list_managers, etc.)
│   └── Utilities (detect_roster_structure)
│
├── data_access.py                 # MINIMAL: Re-exports from core.py (~50 lines)
│
├── data_cache.py                  # Unchanged (caching decorator)
├── motherduck_connection.py       # Unchanged (connection class)
│
└── tab_data_access/               # Tab-specific optimized loaders
    ├── homepage/
    ├── managers/
    ├── players/
    ├── draft/
    ├── transactions/
    ├── simulations/
    ├── keepers/
    ├── team_names/
    ├── team_stats/
    └── hall_of_fame/
```

### Import Pattern (Current Standard)

```python
# Core primitives
from md.core import run_query, T, sql_quote, list_seasons

# Tab-specific loaders
from md.tab_data_access.draft import load_draft_data
from md.tab_data_access.players import load_season_player_data
from md.tab_data_access.transactions import load_optimized_transactions_data
```

### Results

| Metric | Before | After |
|--------|--------|-------|
| data_access.py lines | 1,283 | 50 |
| core.py lines | N/A | 250 |
| Files using new imports | 0 | 44 |
| Total files verified | 274 | 274 |

---

## Files Changed

- `md/core.py` - NEW: Core primitives (~250 lines)
- `md/__init__.py` - Updated exports from core.py
- `md/data_access.py` - Slimmed to minimal re-exports (~50 lines)
- `md/tab_data_access/**/*.py` - Updated imports to use md.core
- 44 files in `tabs/` and root - Updated to use md.core or md.tab_data_access

---

## Validation

- All 274 analytics_app/ files compile successfully
- All tab_data_access/ modules use md.core imports
- No backward compatibility layer needed (all code updated)
