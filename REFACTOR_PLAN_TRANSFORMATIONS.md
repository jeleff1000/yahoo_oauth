# Multi-League Pipeline Refactoring Plan

## Architecture Overview

### The Three Layers
```
fantasy_football_data_scripts/
├── initial_import_v2.py          # Entry point: Full import
├── weekly_update_v2.py           # Entry point: Incremental update
│
└── multi_league/
    ├── core/                     # Layer 1: Shared Infrastructure
    │   ├── league_context.py     #   - LeagueContext class
    │   ├── script_runner.py      #   - run_script() utility
    │   ├── data_normalization.py #   - Normalization helpers
    │   └── yahoo_league_settings.py
    │
    ├── data_fetchers/            # Layer 2: Raw Data Acquisition
    │   ├── orchestrator.py       #   - NEW: Coordinates all fetchers
    │   ├── yahoo_fantasy_data.py #   - Yahoo roster/player data
    │   ├── nfl_offense_stats.py  #   - NFL offense stats
    │   ├── defense_stats.py      #   - NFL defense stats
    │   ├── weekly_matchup_data_v2.py
    │   ├── draft_data_v2.py
    │   ├── transactions_v2.py
    │   └── season_schedules.py
    │
    └── transformations/          # Layer 3: Data Enrichment
        ├── matchup/orchestrator.py
        ├── player/orchestrator.py
        ├── draft/orchestrator.py
        ├── transaction/orchestrator.py
        └── schedule/orchestrator.py
```

### Data Flow
```
┌─────────────────────────────────────────────────────────────────┐
│                    initial_import_v2.py                         │
│                    weekly_update_v2.py                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  data_fetchers/orchestrator.py                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Yahoo   │ │   NFL    │ │  Draft   │ │  Trans   │           │
│  │  Data    │ │  Stats   │ │  Data    │ │  Data    │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Raw Parquet Files                           │
│  player_raw.parquet  matchup_raw.parquet  draft.parquet  etc.   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              transformations/*/orchestrator.py                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Matchup  │ │  Player  │ │  Draft   │ │  Trans   │           │
│  │ Enrich   │ │  Enrich  │ │  Enrich  │ │  Enrich  │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Canonical Output Tables                       │
│  matchup.parquet   player.parquet   draft.parquet   etc.        │
└─────────────────────────────────────────────────────────────────┘
```

### Best Practice: Thin Entry Points

**Before (current):** Entry points have 300+ lines, hardcoded script lists
```python
# initial_import_v2.py (current - 500+ lines)
DATA_FETCHERS = [
    ("multi_league/data_fetchers/weekly_matchup_data_v2.py", ...),
    ("multi_league/data_fetchers/yahoo_fantasy_data.py", ...),
    # ... 7 more
]
TRANSFORMATIONS_PASS_1 = [...]  # 3 items
TRANSFORMATIONS_PASS_2 = [...]  # 3 items
TRANSFORMATIONS_PASS_3 = [...]  # 14 items

def main():
    for script, name, timeout in DATA_FETCHERS:
        run_script(script, ...)
    for script, name, timeout in TRANSFORMATIONS_PASS_1:
        run_script(script, ...)
    # ... etc
```

**After (proposed):** Entry points are thin coordinators (~100 lines)
```python
# initial_import_v2.py (proposed - ~100 lines)
from multi_league.data_fetchers.orchestrator import run_full_fetch
from multi_league.transformations.matchup.orchestrator import enrich_matchup
from multi_league.transformations.player.orchestrator import enrich_player
from multi_league.transformations.draft.orchestrator import enrich_draft
from multi_league.transformations.transaction.orchestrator import enrich_transaction
from multi_league.transformations.schedule.orchestrator import enrich_schedule
from multi_league.transformations.aggregation.aggregate_player_season_v2 import aggregate
from multi_league.transformations.finalize.normalize_canonical_types import normalize

def main():
    ctx = LeagueContext.load(context_path)

    # Phase 1: Fetch raw data from APIs
    run_full_fetch(ctx)

    # Phase 2: Enrich each table
    enrich_matchup(ctx)      # All matchup transformations
    enrich_schedule(ctx)     # Schedule transformations
    enrich_player(ctx)       # All player transformations
    enrich_draft(ctx)        # All draft transformations
    enrich_transaction(ctx)  # All transaction transformations

    # Phase 3: Final aggregations
    aggregate(ctx)
    normalize(ctx)
```

### Data Fetchers Orchestrator

**NEW: `data_fetchers/orchestrator.py`**
```python
#!/usr/bin/env python3
"""
Data Fetchers Orchestrator

Coordinates fetching raw data from Yahoo and NFL APIs.
"""
from pathlib import Path
from core.league_context import LeagueContext
from core.script_runner import run_script

SCRIPT_DIR = Path(__file__).parent

FETCHERS = [
    ("weekly_matchup_data_v2.py", "Matchup data"),
    ("yahoo_fantasy_data.py", "Yahoo player data"),
    ("nfl_offense_stats.py", "NFL offense stats"),
    ("defense_stats.py", "NFL defense stats"),
    ("draft_data_v2.py", "Draft data"),
    ("transactions_v2.py", "Transactions"),
    ("season_schedules.py", "Schedule data"),
]

MERGE_SCRIPTS = [
    ("yahoo_nfl_merge.py", "Merge Yahoo + NFL"),
    ("combine_dst_to_nfl.py", "Combine DST"),
]


def run_full_fetch(ctx: LeagueContext) -> None:
    """Run all data fetchers for a full import."""
    for script, description in FETCHERS:
        run_script(SCRIPT_DIR / script, ctx, description)

    for script, description in MERGE_SCRIPTS:
        run_script(SCRIPT_DIR / script, ctx, description)


def run_weekly_fetch(ctx: LeagueContext, week: int) -> None:
    """Run only the fetchers needed for a weekly update."""
    weekly_fetchers = [
        ("weekly_matchup_data_v2.py", "Matchup data"),
        ("yahoo_fantasy_data.py", "Yahoo player data"),
        ("transactions_v2.py", "Transactions"),
    ]
    for script, description in weekly_fetchers:
        run_script(SCRIPT_DIR / script, ctx, description, extra_args=["--week", str(week)])
```

### Initial Import vs Weekly Update

| Aspect | initial_import_v2.py | weekly_update_v2.py |
|--------|---------------------|---------------------|
| **Fetchers** | `run_full_fetch()` - all data, all years | `run_weekly_fetch()` - current week only |
| **Draft** | Yes | No (draft is done once) |
| **Transactions** | All years | Append new only |
| **Matchups** | All years | Append new week |
| **Transformations** | Full pipeline | Subset (skip draft enrichment) |

```python
# weekly_update_v2.py (proposed - ~80 lines)
from multi_league.data_fetchers.orchestrator import run_weekly_fetch
from multi_league.transformations.matchup.orchestrator import enrich_matchup
from multi_league.transformations.player.orchestrator import enrich_player
from multi_league.transformations.transaction.orchestrator import enrich_transaction

def main():
    ctx = LeagueContext.load(context_path)
    week = detect_current_week(ctx)

    # Phase 1: Fetch only new data
    run_weekly_fetch(ctx, week)

    # Phase 2: Re-run transformations (cumulative stats need updating)
    enrich_matchup(ctx)
    enrich_player(ctx)
    enrich_transaction(ctx)
    # Note: Skip draft enrichment (draft doesn't change weekly)
```

---

## Current State Analysis

### Problem Summary
The `fantasy_football_data_scripts/multi_league/transformations/` directory has organizational issues:

1. **Misnamed directory**: `base/` contains matchup table transformations, not "base" utilities
2. **Monolithic script**: `cumulative_stats_v2.py` (950+ lines) does too many things
3. **Duplicate modules**: Same files exist in both `base/modules/` and `matchup_enrichment/modules/`
4. **Unclear ownership**: Hard to know which folder handles which table

### Current Directory Structure
```
transformations/
├── base/
│   ├── cumulative_stats_v2.py          # 950+ lines - DOES TOO MUCH
│   ├── enrich_schedule_with_playoff_flags.py
│   ├── resolve_hidden_managers.py
│   └── modules/
│       ├── playoff_flags.py            # USED
│       ├── playoff_bracket.py          # USED (has subdirectory too)
│       ├── playoff_helpers.py          # USED
│       ├── playoff_scenarios.py        # USED
│       ├── cumulative_records.py       # USED
│       ├── head_to_head.py             # USED
│       ├── manager_ppg.py              # USED
│       ├── matchup_keys.py             # USED
│       ├── matchup_rankings.py         # USED
│       ├── season_rankings.py          # USED
│       ├── weekly_metrics.py           # USED
│       ├── comparative_schedule.py     # USED
│       ├── all_play_extended.py        # USED
│       └── playoff_bracket/            # Subdirectory with bracket logic
│
├── matchup_enrichment/
│   ├── expected_record_v2.py
│   ├── player_to_matchup_v2.py
│   ├── playoff_odds_import.py
│   └── modules/
│       ├── playoff_flags.py            # UNUSED - different from base
│       ├── playoff_helpers.py          # UNUSED - duplicate of base
│       ├── playoff_bracket.py          # UNUSED - different from base
│       ├── playoff_simulation.py       # UNUSED
│       ├── schedule_simulation.py      # USED by expected_record_v2.py
│       └── bye_week_filler.py          # USED by expected_record_v2.py
│
├── player_enrichment/
│   └── (various player table scripts)
│
├── draft_enrichment/
│   └── (various draft table scripts)
│
├── transaction_enrichment/
│   └── (various transaction table scripts)
│
├── aggregation/
│   └── aggregate_player_season_v2.py
│
├── finalize/
│   └── normalize_canonical_types.py
│
└── modules/
    └── type_utils.py                   # USED by multiple scripts
```

---

## Proposed New Structure

### Design Principles
1. **One folder per table**: Each canonical table (matchup, player, draft, transaction, schedule) gets its own folder
2. **Shared utilities in `common/`**: Cross-table utilities live in a shared location
3. **No duplicates**: Single source of truth for each module
4. **Smaller, focused scripts**: Break up monolithic scripts into logical pieces

### Target Directory Structure
```
transformations/
├── common/                             # Shared utilities (rename from modules/)
│   ├── __init__.py
│   ├── type_utils.py                   # Type normalization utilities
│   ├── playoff_helpers.py              # Playoff detection helpers
│   └── data_normalization.py           # If needed (or keep in core/)
│
├── matchup/                            # ALL matchup table transformations
│   ├── __init__.py
│   ├── cumulative_stats.py             # Main orchestrator (slimmed down)
│   ├── resolve_hidden_managers.py      # Move from base/
│   ├── player_to_matchup_v2.py         # Move from matchup_enrichment/
│   ├── expected_record_v2.py           # Move from matchup_enrichment/
│   ├── playoff_odds_import.py          # Move from matchup_enrichment/
│   └── modules/
│       ├── __init__.py
│       ├── matchup_keys.py             # Matchup key generation
│       ├── cumulative_records.py       # Win/loss records
│       ├── head_to_head.py             # H2H records
│       ├── manager_ppg.py              # Manager points per game
│       ├── weekly_metrics.py           # Weekly league-relative metrics
│       ├── season_rankings.py          # Season rankings
│       ├── matchup_rankings.py         # Manager matchup rankings
│       ├── comparative_schedule.py     # Schedule strength
│       ├── all_play_extended.py        # All-play metrics
│       ├── playoff_flags.py            # Playoff/consolation detection
│       ├── playoff_scenarios.py        # Clinch/elimination scenarios
│       ├── schedule_simulation.py      # Move from matchup_enrichment/modules/
│       ├── bye_week_filler.py          # Move from matchup_enrichment/modules/
│       └── playoff_bracket/            # Bracket simulation
│           ├── __init__.py
│           ├── utils.py
│           ├── championship_bracket.py
│           ├── consolation_bracket.py
│           └── placement_games.py
│
├── schedule/                           # Schedule table transformations
│   ├── __init__.py
│   └── enrich_schedule_with_playoff_flags.py  # Move from base/
│
├── player/                             # Rename from player_enrichment/
│   ├── __init__.py
│   ├── matchup_to_player_v2.py
│   ├── player_stats_v2.py
│   ├── replacement_level_v2.py
│   ├── draft_to_player_v2.py
│   ├── transactions_to_player_v2.py
│   └── modules/
│       ├── __init__.py
│       ├── optimal_lineup.py
│       ├── scoring_calculator.py
│       ├── ppg_calculator.py
│       ├── name_resolver.py
│       ├── player_rankings.py
│       ├── spar_calculator.py
│       └── replacement_calculator_dynamic.py
│
├── draft/                              # Rename from draft_enrichment/
│   ├── __init__.py
│   ├── player_to_draft_v2.py
│   ├── draft_value_metrics_v3.py
│   ├── keeper_economics_v2.py
│   └── modules/
│       └── (existing modules)
│
├── transaction/                        # Rename from transaction_enrichment/
│   ├── __init__.py
│   ├── fix_unknown_managers.py
│   ├── player_to_transactions_v2.py
│   ├── transaction_value_metrics_v3.py
│   └── modules/
│       └── transaction_spar_calculator.py
│
├── aggregation/
│   ├── __init__.py
│   └── aggregate_player_season_v2.py
│
└── finalize/
    ├── __init__.py
    └── normalize_canonical_types.py
```

---

---

## New: Table-Level Orchestrators

### Current Problem
`initial_import_v2.py` calls 20+ individual scripts directly. This makes it hard to:
- Understand what transformations apply to which table
- Run just matchup transformations for debugging
- Maintain dependency order

### Solution: Each Table Gets an Orchestrator

| Table | Orchestrator | Individual Scripts It Calls |
|-------|--------------|----------------------------|
| **matchup** | `matchup/orchestrator.py` | resolve_hidden_managers, cumulative_stats, expected_record, playoff_odds, player_to_matchup |
| **schedule** | `schedule/orchestrator.py` | enrich_schedule_with_playoff_flags |
| **player** | `player/orchestrator.py` | matchup_to_player, player_stats, replacement_level, draft_to_player, transactions_to_player |
| **draft** | `draft/orchestrator.py` | player_to_draft, draft_value_metrics, keeper_economics |
| **transaction** | `transaction/orchestrator.py` | fix_unknown_managers, player_to_transactions, transaction_value_metrics |

### Simplified initial_import_v2.py
```python
# Before: 20+ individual script calls with complex ordering
TRANSFORMATIONS_PASS_1 = [...]  # 3 scripts
TRANSFORMATIONS_PASS_2 = [...]  # 3 scripts
TRANSFORMATIONS_PASS_3 = [...]  # 14 scripts

# After: 5 orchestrator calls
TRANSFORMATIONS = [
    ("matchup/orchestrator.py", "Matchup Table Enrichment", 1800),
    ("schedule/orchestrator.py", "Schedule Table Enrichment", 120),
    ("player/orchestrator.py", "Player Table Enrichment", 1800),
    ("draft/orchestrator.py", "Draft Table Enrichment", 600),
    ("transaction/orchestrator.py", "Transaction Table Enrichment", 600),
    ("aggregation/aggregate_player_season_v2.py", "Aggregate Player Season", 600),
    ("finalize/normalize_canonical_types.py", "Normalize Types", 120),
]
```

### Example Orchestrator: `matchup/orchestrator.py`
```python
#!/usr/bin/env python3
"""
Matchup Table Orchestrator

Runs all transformations that enrich the matchup.parquet file.
Called by initial_import_v2.py or can be run standalone for debugging.

Usage:
    python orchestrator.py --context /path/to/league_context.json
"""
import argparse
from pathlib import Path

from core.league_context import LeagueContext
from .modules import (
    matchup_keys,
    cumulative_records,
    playoff_normalization,
    manager_ppg,
    weekly_metrics,
    all_play_extended,
    head_to_head,
    comparative_schedule,
    season_rankings,
    matchup_rankings,
    playoff_scenarios,
    inflation_rate,
)


def run_matchup_enrichment(ctx: LeagueContext) -> None:
    """Run all matchup table transformations."""

    # Load matchup data
    df = pd.read_parquet(ctx.canonical_matchup_file)

    # Step 1: Keys and records
    df = matchup_keys.add_matchup_keys(df)
    df = cumulative_records.calculate_cumulative_records(df)

    # Step 2: Playoff normalization
    df = playoff_normalization.normalize_playoff_data(df, ctx.data_directory)

    # Step 3: Manager metrics
    df = manager_ppg.calculate_manager_ppg(df)
    df = weekly_metrics.calculate_weekly_metrics(df)
    df = all_play_extended.calculate_opponent_all_play(df)

    # Step 4: Historical records
    df = head_to_head.calculate_head_to_head_records(df)
    df = comparative_schedule.calculate_comparative_schedule(df)

    # Step 5: Rankings
    df = season_rankings.calculate_season_rankings(df)
    df = matchup_rankings.calculate_all_matchup_rankings(df)
    df = season_rankings.calculate_alltime_rankings(df)

    # Step 6: Scenarios and inflation
    df = playoff_scenarios.add_playoff_scenario_columns(df)
    df = inflation_rate.calculate_inflation_rate(df)

    # Save enriched matchup
    df.to_parquet(ctx.canonical_matchup_file, index=False)
    print(f"[OK] Matchup enrichment complete: {len(df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=Path, required=True)
    args = parser.parse_args()

    ctx = LeagueContext.load(args.context)
    run_matchup_enrichment(ctx)
```

### Benefits
1. **Cleaner initial_import_v2.py** - Just 7 lines instead of 20+
2. **Standalone debugging** - Run `python matchup/orchestrator.py --context ...` to test just matchup
3. **Clear ownership** - Each table's transformations are self-contained
4. **Easier maintenance** - Add new matchup transforms in one place
5. **Better error messages** - "Matchup enrichment failed" vs "multi_league/transformations/base/cumulative_stats_v2.py failed"

---

## Refactoring Steps

### Phase 1: Delete Unused Files (Quick Win)
Delete these unused files from `matchup_enrichment/modules/`:
- `playoff_flags.py`
- `playoff_helpers.py`
- `playoff_bracket.py`
- `playoff_simulation.py`

**Estimated effort**: 5 minutes

### Phase 2: Create New Directory Structure
1. Create `transformations/common/`
2. Create `transformations/matchup/`
3. Create `transformations/schedule/`
4. Rename `player_enrichment/` to `player/`
5. Rename `draft_enrichment/` to `draft/`
6. Rename `transaction_enrichment/` to `transaction/`

**Estimated effort**: 10 minutes

### Phase 3: Move Files to New Locations

#### 3a: Move to `common/`
```
modules/type_utils.py → common/type_utils.py
base/modules/playoff_helpers.py → common/playoff_helpers.py (if truly shared)
```

#### 3b: Move to `matchup/`
```
base/cumulative_stats_v2.py → matchup/cumulative_stats.py
base/resolve_hidden_managers.py → matchup/resolve_hidden_managers.py
matchup_enrichment/player_to_matchup_v2.py → matchup/player_to_matchup_v2.py
matchup_enrichment/expected_record_v2.py → matchup/expected_record_v2.py
matchup_enrichment/playoff_odds_import.py → matchup/playoff_odds_import.py
base/modules/* → matchup/modules/*
matchup_enrichment/modules/schedule_simulation.py → matchup/modules/schedule_simulation.py
matchup_enrichment/modules/bye_week_filler.py → matchup/modules/bye_week_filler.py
```

#### 3c: Move to `schedule/`
```
base/enrich_schedule_with_playoff_flags.py → schedule/enrich_schedule_with_playoff_flags.py
```

**Estimated effort**: 30 minutes

### Phase 4: Update Import Paths
Every script that imports from moved modules needs updating:
- `initial_import_v2.py` - Update TRANSFORMATIONS_PASS_* paths
- `weekly_update_v2.py` - Update WEEKLY_TRANSFORMATIONS paths
- All transformation scripts - Update internal imports

**Estimated effort**: 1-2 hours

### Phase 5: Modularize cumulative_stats.py (Optional, Bigger Effort)

#### Current State
The computation logic is ALREADY modularized - `cumulative_stats_v2.py` calls these existing modules:
- `matchup_keys.py` - Step 1
- `cumulative_records.py` - Step 2
- `playoff_flags.py` - Step 3
- `manager_ppg.py` - Step 4
- `weekly_metrics.py` - Step 5
- `all_play_extended.py` - Step 6
- `head_to_head.py` - Step 7
- `comparative_schedule.py` - Step 8
- `season_rankings.py` - Steps 9, 10
- `matchup_rankings.py` - Step 9.5
- `playoff_scenarios.py` - Step 11

#### What's NOT Modularized (embedded in cumulative_stats_v2.py)
These chunks should be extracted to new modules:

| Lines | Function | New Module |
|-------|----------|------------|
| 109-460 | `apply_cumulative_fixes()` | `matchup/modules/playoff_normalization.py` |
| 744-772 | Inflation rate calculation | `matchup/modules/inflation_rate.py` |
| 809-838 | `save_enriched_matchup()` enforcement | `matchup/modules/matchup_validation.py` |

#### New Modules to Create

**1. `matchup/modules/playoff_normalization.py`**
```python
"""
Normalize playoff/consolation flags and simulate brackets.
Extracted from cumulative_stats_v2.py apply_cumulative_fixes()
"""
def normalize_playoff_data(df, data_directory=None):
    """
    - Ensure playoff flag columns exist
    - Calculate cumulative records (for final_playoff_seed)
    - Detect playoffs by seed
    - Mark playoff rounds
    - Simulate playoff brackets (champion, sacko, placement)
    - Enforce mutual exclusivity (is_playoffs vs is_consolation)
    """
    pass
```

**2. `matchup/modules/inflation_rate.py`**
```python
"""
Calculate year-over-year scoring inflation for cross-season comparisons.
"""
def calculate_inflation_rate(df):
    """
    - Calculate average team_points per year
    - Use earliest year as base (inflation_rate = 1.0)
    - Return df with inflation_rate column
    """
    pass
```

**3. `matchup/modules/matchup_validation.py`**
```python
"""
Final validation and enforcement before saving matchup data.
"""
def enforce_mutual_exclusivity(df):
    """
    - Playoff games: clear consolation labels
    - Consolation games: clear playoff labels
    - Set correct binary flags based on round labels
    """
    pass
```

#### Slimmed Down Orchestrator
After extraction, `matchup/cumulative_stats.py` becomes ~200 lines:

```python
def transform_cumulative_stats(df, ctx):
    """Orchestrate all matchup transformations."""

    # Step 1-3: Keys, records, playoff normalization
    df = matchup_keys.add_matchup_keys(df)
    df = cumulative_records.calculate_cumulative_records(df)
    df = playoff_normalization.normalize_playoff_data(df)  # NEW MODULE

    # Step 4-8: Weekly/seasonal metrics
    df = manager_ppg.calculate_manager_ppg(df)
    df = weekly_metrics.calculate_weekly_metrics(df)
    df = all_play_extended.calculate_opponent_all_play(df)
    df = head_to_head.calculate_head_to_head_records(df)
    df = comparative_schedule.calculate_comparative_schedule(df)

    # Step 9-11: Rankings and scenarios
    df = season_rankings.calculate_season_rankings(df)
    df = matchup_rankings.calculate_all_matchup_rankings(df)
    df = season_rankings.calculate_alltime_rankings(df)
    df = playoff_scenarios.add_playoff_scenario_columns(df)

    # Step 12: Inflation
    df = inflation_rate.calculate_inflation_rate(df)  # NEW MODULE

    return df
```

**Estimated effort**: 2-4 hours

### Phase 6: Update initial_import_v2.py Pipeline Paths
Update the transformation script paths:

```python
# Before
TRANSFORMATIONS_PASS_1 = [
    ("multi_league/transformations/base/resolve_hidden_managers.py", ...),
    ("multi_league/transformations/base/cumulative_stats_v2.py", ...),
    ("multi_league/transformations/base/enrich_schedule_with_playoff_flags.py", ...),
]

# After
TRANSFORMATIONS_PASS_1 = [
    ("multi_league/transformations/matchup/resolve_hidden_managers.py", ...),
    ("multi_league/transformations/matchup/cumulative_stats.py", ...),
    ("multi_league/transformations/schedule/enrich_schedule_with_playoff_flags.py", ...),
]
```

**Estimated effort**: 30 minutes

### Phase 7: Delete Empty Directories
After moving files:
- Delete `transformations/base/` (now empty)
- Delete `transformations/matchup_enrichment/` (now empty)
- Delete `transformations/modules/` (moved to common/)

**Estimated effort**: 5 minutes

### Phase 8: Test
1. Run `python -m py_compile` on all moved files
2. Run a test import with `initial_import_v2.py --dry-run`
3. Run actual import on test league

**Estimated effort**: 30 minutes - 1 hour

---

## Files to Delete (Immediate)

These can be deleted right now with no impact:

```
transformations/matchup_enrichment/modules/playoff_flags.py      # Unused
transformations/matchup_enrichment/modules/playoff_helpers.py    # Unused duplicate
transformations/matchup_enrichment/modules/playoff_bracket.py    # Unused
transformations/matchup_enrichment/modules/playoff_simulation.py # Unused
```

---

## Import Path Changes Reference

### For scripts importing from base/modules/
```python
# Before
from modules import playoff_flags
from modules.playoff_bracket import simulate_playoff_brackets

# After
from matchup.modules import playoff_flags
from matchup.modules.playoff_bracket import simulate_playoff_brackets
```

### For scripts importing type_utils
```python
# Before
from type_utils import safe_merge, ensure_canonical_types

# After
from common.type_utils import safe_merge, ensure_canonical_types
```

### For initial_import_v2.py script paths
```python
# Before
("multi_league/transformations/base/cumulative_stats_v2.py", "Cumulative Stats", 600)

# After
("multi_league/transformations/matchup/cumulative_stats.py", "Cumulative Stats", 600)
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking imports | Test each file compiles after moving |
| GitHub Actions fail | Test workflows after changes |
| Missed import updates | Use grep to find all import statements |
| Losing git history | Use `git mv` for moves (preserves history) |

---

## Execution Checklist

### Phase 1: Cleanup (Quick Wins)
- [x] Delete 4 unused files in matchup_enrichment/modules/ ✅ DONE

### Phase 2: Directory Restructure
- [ ] Create `transformations/common/`
- [ ] Create `transformations/matchup/` and `matchup/modules/`
- [ ] Create `transformations/schedule/`
- [ ] Rename `player_enrichment/` → `player/`
- [ ] Rename `draft_enrichment/` → `draft/`
- [ ] Rename `transaction_enrichment/` → `transaction/`

### Phase 3: Move Files
- [ ] Move `modules/type_utils.py` → `common/type_utils.py`
- [ ] Move `base/modules/*` → `matchup/modules/`
- [ ] Move `base/cumulative_stats_v2.py` → `matchup/cumulative_stats.py`
- [ ] Move `base/resolve_hidden_managers.py` → `matchup/`
- [ ] Move `base/enrich_schedule_with_playoff_flags.py` → `schedule/`
- [ ] Move `matchup_enrichment/*.py` → `matchup/`
- [ ] Move `matchup_enrichment/modules/schedule_simulation.py` → `matchup/modules/`
- [ ] Move `matchup_enrichment/modules/bye_week_filler.py` → `matchup/modules/`

### Phase 4: Create Orchestrators
- [ ] Create `data_fetchers/orchestrator.py` (run_full_fetch, run_weekly_fetch)
- [ ] Create `matchup/orchestrator.py`
- [ ] Create `player/orchestrator.py`
- [ ] Create `draft/orchestrator.py`
- [ ] Create `transaction/orchestrator.py`
- [ ] Create `schedule/orchestrator.py`

### Phase 5: Extract Modules from cumulative_stats.py
- [ ] Create `matchup/modules/playoff_normalization.py` (extract `apply_cumulative_fixes()`)
- [ ] Create `matchup/modules/inflation_rate.py`
- [ ] Create `matchup/modules/matchup_validation.py`
- [ ] Slim down `cumulative_stats.py` to ~200 lines

### Phase 6: Update Imports
- [ ] Update all scripts to use new import paths
- [ ] Update `initial_import_v2.py` to call orchestrators
- [ ] Update `weekly_update_v2.py` to call orchestrators

### Phase 7: Cleanup
- [ ] Delete empty `base/` directory
- [ ] Delete empty `matchup_enrichment/` directory
- [ ] Delete empty `modules/` directory (now `common/`)

### Phase 8: Test
- [ ] `python -m py_compile` on all files
- [ ] Run `initial_import_v2.py --dry-run`
- [ ] Run actual import on test league
- [ ] Commit with descriptive message

---

## Notes

- Use `git mv` instead of regular move to preserve history
- Run `python -m py_compile <file>` after each edit to catch syntax errors
- The modules in `base/modules/` are well-factored already - they just need to move
- Consider keeping `cumulative_stats.py` as the main orchestrator rather than splitting into multiple standalone scripts
