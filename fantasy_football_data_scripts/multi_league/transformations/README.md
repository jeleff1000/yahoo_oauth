# Multi-League Transformations

Data transformation pipelines that enrich raw data with analytics, rankings, and cumulative statistics.

## Overview

Transformations take raw data from `data_fetchers/` and add:
- Cumulative statistics (running totals, streaks)
- Rankings (season, all-time, percentiles)
- Derived metrics (efficiency, trends, comparisons)
- Join keys for cross-table analysis

## Directory Structure

```
transformations/
├── cumulative_stats_v2.py          # Main orchestrator for matchup transformations
├── modules/                         # Modular transformation functions
│   ├── __init__.py
│   ├── matchup_keys.py             # Add matchup_key for self-joins
│   ├── playoff_flags.py            # Normalize is_playoffs/is_consolation
│   ├── cumulative_records.py       # Running win/loss totals
│   ├── weekly_metrics.py           # League-relative performance
│   ├── head_to_head.py             # Manager vs manager records
│   └── season_rankings.py          # Season and all-time rankings
└── README.md                        # This file
```

## Key Concepts

### SET-AND-FORGET vs RECALCULATE Columns

**SET-AND-FORGET** (Calculated once after championship, never recalculated):
- `final_wins`, `final_losses` - Season totals (finalized after championship)
- `final_regular_wins`, `final_regular_losses` - Regular season totals
- `season_mean`, `season_median` - Season-level point aggregates
- `manager_season_ranking` - Final season rank
- `championship`, `sacko`, `quarterfinal`, `semifinal` - Playoff outcomes

**RECALCULATE WEEKLY** (Updated every week for cross-week/year comparisons):
- `cumulative_wins`, `cumulative_losses` - Running all-time totals
- `win_streak`, `loss_streak` - Current active streaks
- `teams_beat_this_week` - League-relative weekly performance
- `w_vs_{opponent}`, `l_vs_{opponent}` - Head-to-head records
- `manager_all_time_ranking` - Cross-season historical ranks
- All percentile/rank columns comparing across weeks or years

### matchup_key (NEW - Data Dictionary Addition)

Enables self-joins to get both sides of a matchup in one row:

```python
# matchup_key format: "Team1__vs__Team2__2024__10"
# - Teams sorted alphabetically for canonical key
# - Same key for both manager perspectives

# Self-join example
combined = df.merge(
    df[['matchup_key', 'manager', 'team_points']],
    on='matchup_key',
    suffixes=('', '_opp')
)
# Now you have both manager and opponent data in one row!
```

## Usage

### CLI

```bash
# Transform matchup data
python cumulative_stats_v2.py --context /path/to/league_context.json

# Mark championship as complete (finalizes SET-AND-FORGET columns)
python cumulative_stats_v2.py --context /path/to/league_context.json --championship-complete

# Specify current week for weekly updates
python cumulative_stats_v2.py --context /path/to/league_context.json --current-week 10 --current-year 2024
```

### Python API

```python
from multi_league.core.league_context import LeagueContext
from multi_league.transformations.cumulative_stats_v2 import transform_cumulative_stats

# Load context
ctx = LeagueContext.load("leagues/kmffl/league_context.json")

# Load matchup data
matchup_df = pd.read_parquet(ctx.matchup_data_directory / "matchup.parquet")

# Transform
enriched_df = transform_cumulative_stats(
    matchup_df,
    current_week=10,
    current_year=2024,
    championship_complete=False  # Set True after championship
)

# Save
enriched_df.to_parquet(ctx.matchup_data_directory / "matchup_enriched.parquet")
```

## Transformation Pipeline

The cumulative_stats_v2.py orchestrator runs 7 transformation steps in order:

1. **Add Matchup Keys** - Create matchup_key, matchup_id for joins
2. **Normalize Playoff Flags** - Enforce is_consolation=1 → is_playoffs=0
3. **Calculate Cumulative Records** - Running totals, streaks (RECALCULATE WEEKLY)
4. **Calculate Weekly Metrics** - League-relative performance (RECALCULATE WEEKLY)
5. **Calculate Head-to-Head** - Manager vs manager records (RECALCULATE WEEKLY)
6. **Calculate Season Rankings** - Final season stats (SET-AND-FORGET if complete)
7. **Calculate All-Time Rankings** - Cross-season ranks (RECALCULATE WEEKLY)

## Module Development Status

| Module | Status | Description |
|--------|--------|-------------|
| `matchup_keys.py` | ✅ Complete | Adds matchup_key for self-joins |
| `playoff_flags.py` | ✅ Complete | Normalizes playoff/consolation flags |
| `cumulative_records.py` | ✅ Complete | Running win/loss totals, streaks |
| `weekly_metrics.py` | ✅ Complete | League-relative performance metrics |
| `head_to_head.py` | ✅ Complete | Manager vs manager cumulative records |
| `season_rankings.py` | ✅ Complete | Season and all-time rankings |

## Data Dictionary Compliance

All transformations follow the data dictionary standards:

- **Primary keys preserved**: `(manager, year, week)`
- **New keys added**: `matchup_key`, `matchup_id`, `matchup_sort_key`
- **Proper data types**: Int64 for integers, float for decimals, string for text
- **Null handling**: Uses pd.NA for missing integers
- **Mutually exclusive flags**: is_consolation=1 → is_playoffs=0

## Next Steps

1. **Implement stub modules** - Fill in TODOs with actual logic from cumulative_stats.py
2. **Add player transformations** - Create player_stats_v2.py for player-level enrichment
3. **Add draft transformations** - ROI analysis, cost efficiency, keeper value
4. **Add transaction transformations** - FAAB efficiency, waiver wire success rate

## Related Documentation

- `DATA_DICTIONARY.md` - Column definitions and join keys
- `JOIN_KEY_ANALYSIS.md` - Join strategies and examples
- `MULTI_LEAGUE_SETUP_GUIDE.md` - Implementation roadmap
