# League-Wide Optimal Player Documentation

## Overview

The `league_wide_optimal_player` column is a **truly generic, multi-league aware** feature that determines which players should have been started in any given week based on **that league's specific configuration**.

## What Makes It Generic?

### 1. **League-Specific Scoring Rules**

The optimal player determination uses points calculated from **each league's unique scoring rules**:

```python
# Example: League A (Half-PPR)
Pass TD = 4 pts
Rush TD = 6 pts  
Reception = 0.5 pts

# Example: League B (Full-PPR, 6pt Pass TD)
Pass TD = 6 pts
Rush TD = 6 pts
Reception = 1.0 pts
```

**How it works:**
- `calculate_fantasy_points()` reads from `yahoo_full_scoring_{year}_{league_key}.json`
- Applies that league's specific scoring rules to all stat columns
- Creates `fantasy_points` column with league-specific values
- `compute_league_wide_optimal_players()` uses those league-specific points for ranking

### 2. **League-Specific Roster Settings**

The optimal player determination uses **each league's unique roster configuration**:

```python
# Example: League A (Standard)
1 QB, 3 WR, 2 RB, 1 TE, 1 W/R/T, 1 K, 1 DEF

# Example: League B (2QB/Superflex)
2 QB, 2 WR, 2 RB, 1 TE, 2 FLEX, 1 K, 1 DEF
```

**How it works:**
- `load_roster_settings_from_json()` reads from `yahoo_roster_{year}_{league_key}.json`
- Filters by `league_id` to get the correct league's settings
- Identifies flex positions (W/R/T, FLEX, OP, etc.) and their eligible players
- Uses those counts to determine how many players at each position are optimal

### 3. **Flex Position Handling**

The algorithm correctly handles **any flex position configuration**:

- **W/R/T** (WR/RB/TE flex) - Common in standard leagues
- **FLEX** (WR/RB/TE flex) - Alternative naming
- **W/R** (WR/RB flex) - Some leagues
- **W/T** (WR/TE flex) - Rare
- **R/T** (RB/TE flex) - Rare
- **Q/W/R/T** (Superflex) - 2QB leagues
- **OP** (Offensive Player) - Alternative superflex naming

**Algorithm:**
1. Fill dedicated position slots first (QB, WR, RB, TE, K, DEF)
2. Track which players were selected
3. For each flex position:
   - Get all eligible players (e.g., WR/RB/TE for W/R/T)
   - Exclude players already selected for dedicated slots
   - Select top N remaining players by points
   - Mark them as optimal for that flex slot

### 4. **Nearest-Year Fallback**

If a year doesn't have roster/scoring settings (e.g., pre-league years or future years):

```python
if year in roster_by_year:
    roster_settings = roster_by_year[year]
else:
    # Use nearest year's settings
    closest_year = min(available_roster_years, key=lambda y: abs(y - year))
    roster_settings = roster_by_year[closest_year]
```

**Examples:**
- 1999-2013 (pre-league): Uses 2014 settings
- 2026 (future): Uses 2025 settings
- 2017 missing: Uses 2016 or 2018 (whichever is closer)

### 5. **Works Across All Players**

Uses **`nfl_player_id`** as the identifier, which includes:
- ✅ Rostered players (in your league)
- ✅ Non-rostered players (free agents who had stats)
- ✅ Historical players (pre-league years 1999-2013)

This enables:
- "Who were the actual top 10 RBs this week?" (regardless of roster status)
- "Should I have picked up this free agent?" (compares to your starters)
- "How do historical players compare?" (e.g., 2005 LaDainian Tomlinson vs 2024 Christian McCaffrey)

## Complete Data Flow

```
1. League Context Loaded
   ↓
   ctx.league_id = "449.l.198278"

2. Scoring Rules Loaded
   ↓
   yahoo_full_scoring_2024_449_l_198278.json
   → Pass TD = 4 pts, Rush TD = 6 pts, Rec = 0.5 pts

3. Roster Settings Loaded
   ↓
   yahoo_roster_2024_449_l_198278.json
   → 1 QB, 3 WR, 2 RB, 1 TE, 1 W/R/T, 1 K, 1 DEF

4. Fantasy Points Calculated
   ↓
   Josh Allen: 320 pass yds (12.8) + 2 pass TD (8) + 40 rush yds (4) + 1 rush TD (6) = 30.8 pts

5. Optimal Players Determined
   ↓
   Week 5, 2024:
   - Top 1 QB by points → Josh Allen (30.8 pts) → league_wide_optimal_player = True
   - Top 3 WR by points → Jefferson (22.5), Chase (20.1), Lamb (19.8) → True
   - Top 2 RB by points → CMC (28.4), Barkley (24.2) → True
   - Top 1 TE by points → Kelce (15.6) → True
   - Top 1 W/R/T (remaining) → Gibbs (RB, 18.3 pts) → True
   - Top 1 K by points → Tucker (12 pts) → True
   - Top 1 DEF by points → SF (18 pts) → True
```

## Multi-League Example

### League A (KMFFL - Half-PPR, Standard Roster)
```python
# Settings
Scoring: Pass TD = 4, Rec = 0.5
Roster: 1 QB, 3 WR, 2 RB, 1 TE, 1 W/R/T

# Week 5 2024 Results
Josh Allen: 30.8 pts → Optimal = True, Position = "QB"
Tyreek Hill: 14.5 pts → Optimal = True, Position = "WR1"
CeeDee Lamb: 13.2 pts → Optimal = True, Position = "WR2"
Amon-Ra St. Brown: 12.8 pts → Optimal = True, Position = "WR3"
Christian McCaffrey: 28.4 pts → Optimal = True, Position = "RB1"
Saquon Barkley: 24.2 pts → Optimal = True, Position = "RB2"
Travis Kelce: 15.6 pts → Optimal = True, Position = "TE"
Jahmyr Gibbs: 18.3 pts → Optimal = True, Position = "W/R/T"
```

### League B (Different League - Full-PPR, 2QB)
```python
# Settings
Scoring: Pass TD = 6, Rec = 1.0
Roster: 2 QB, 2 WR, 2 RB, 1 TE, 2 FLEX

# Week 5 2024 Results  
Josh Allen: 34.8 pts → Optimal = True, Position = "QB1"
Lamar Jackson: 32.5 pts → Optimal = True, Position = "QB2"
Tyreek Hill: 20.5 pts → Optimal = True, Position = "WR1"
CeeDee Lamb: 19.2 pts → Optimal = True, Position = "WR2"
Christian McCaffrey: 32.4 pts → Optimal = True, Position = "RB1"
Saquon Barkley: 28.2 pts → Optimal = True, Position = "RB2"
Travis Kelce: 21.6 pts → Optimal = True, Position = "TE"
Amon-Ra St. Brown: 18.8 pts → Optimal = True, Position = "FLEX1"
Jahmyr Gibbs: 22.3 pts → Optimal = True, Position = "FLEX2"
```

**Same week, same players, different optimal lineups AND positions!** ✓

## Usage

### In player_stats_v2.py
```python
# Step 1: Load league-specific settings
roster_by_year = load_roster_settings_from_json(roster_dir, league_id=ctx.league_id)

# Step 2: Calculate league-specific fantasy points
df = calculate_fantasy_points(df, scoring_rules_by_year, year_col="year")

# Step 3: Determine optimal players using league-specific points + roster
df = compute_league_wide_optimal_players(
    df,
    roster_by_year=roster_by_year,
    points_col="fantasy_points"  # Uses league-specific points!
)
```

### Output Column
- **Column Name:** `league_wide_optimal_player`
- **Type:** Boolean (True/False)
- **Meaning:** Should this player have been started this week based on league roster + scoring rules?
- **Scope:** League-wide (not manager-specific)

- **Column Name:** `league_wide_optimal_position`
- **Type:** String (e.g., "QB", "WR1", "WR2", "RB1", "W/R/T", "FLEX1")
- **Meaning:** Which optimal lineup slot this player fills (null if not optimal)
- **Scope:** League-wide (not manager-specific)
- **Examples:**
  - Single-slot positions: "QB", "TE", "K", "DEF"
  - Multi-slot positions: "WR1", "WR2", "WR3", "RB1", "RB2"
  - Flex positions: "W/R/T", "FLEX1", "FLEX2", "OP"

## Key Features

✅ **Multi-League Aware:** Different scoring rules = different points = different optimal players

✅ **Flex Position Smart:** Correctly handles W/R/T, FLEX, Superflex, OP, etc.

✅ **Historical Compatible:** Works for 1999-2025 via nearest-year fallback

✅ **All Players Included:** Uses nfl_player_id (rostered + free agents + historical)

✅ **Position Column Unified:** Uses `position` (yahoo_position → nfl_position fallback)

✅ **Nearest-Year Fallback:** Missing year settings use closest available year

✅ **League Isolation:** Filters by league_id to prevent cross-contamination

## Related Columns

After `league_wide_optimal_player` is determined, additional metrics are calculated:

- **optimal_points** (manager-level): Sum of all optimal players' points for that manager-week
- **lineup_efficiency** (manager-level): (actual_points / optimal_points) × 100
- **bench_points** (manager-level): Points scored by rostered but not started players

These are calculated in `calculate_optimal_lineup_metrics()` and `calculate_bench_points()`.

## Files Involved

### Module Files
- `optimal_lineup.py` - Core logic for optimal player determination
- `player_stats_v2.py` - Orchestrates the calculation

### Data Files (League-Specific)
- `yahoo_roster_{year}_{league_key}.json` - Roster position counts
- `yahoo_full_scoring_{year}_{league_key}.json` - Scoring rules
- `league_context.json` - League configuration (league_id, paths)

### Output
- `player.parquet` - Contains `league_wide_optimal_player` column
- `player.csv` - CSV version for debugging

## Notes

- **League Isolation:** The `league_id` parameter ensures settings are loaded for the correct league
- **Position Priority:** Dedicated positions filled before flex positions
- **Tie Breaking:** Uses Polars `.sort()` with `descending=True` for consistent ordering
- **Performance:** Processes year-by-year, week-by-week for memory efficiency
- **Validation:** Prints roster settings loaded for each year for transparency

## Last Updated
2025-11-01
