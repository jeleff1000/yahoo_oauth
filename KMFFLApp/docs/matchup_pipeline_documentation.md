# Matchup Pipeline Documentation (Core)

> **Last Updated:** November 2024
> **Status:** Production - HIGHEST VALUE TABLE
> **Data Source:** Yahoo Fantasy Football API

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Data Schema Reference](#data-schema-reference)
4. [Column Categories](#column-categories)
5. [Recommendations & Roadmap](#recommendations--roadmap)

---

## Executive Summary

The Matchup table is the **foundation of your entire app** - feeding Hall of Fame, Matchups, and Simulations sections. Every weekly game result from league inception is tracked with 280 enriched columns covering performance, standings, playoff outcomes, schedule simulations, and predictive analytics.

### Key Capabilities

- **Complete Game History**: Every matchup from league inception
- **Cumulative Statistics**: Running totals, streaks, rankings
- **Playoff Bracket Tracking**: Champion, sacko, placement games
- **Schedule Simulations**: 100,000 Monte Carlo simulations per week
- **Head-to-Head Records**: Dynamic H2H tracking across all managers
- **Optimal Lineup Analysis**: Bench points, lineup efficiency

### Quick Stats

| Metric | Value |
|--------|-------|
| Total Rows | 1,924 |
| Total Columns | **280** |
| Simulation Iterations | 100,000 per week |
| Update Frequency | Weekly during season |

### What Makes This Special

1. **Dual Simulation System**:
   - `shuffle_*` columns: "What if schedules were random?" (measures team strength)
   - `opp_shuffle_*` columns: "What if opponent difficulty was random?" (measures schedule luck)

2. **Playoff Bracket Intelligence**:
   - Tracks championship AND consolation brackets
   - Properly handles bye weeks, placement games
   - Correctly identifies champion and sacko

3. **SET-AND-FORGET vs RECALCULATE**:
   - Some columns are finalized after championship (never change)
   - Some columns are recalculated weekly (cross-week comparisons)

---

## Pipeline Architecture

### High-Level Flow

```
Yahoo Fantasy API
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA FETCHING                      │
│                                                                │
│  weekly_matchup_data_v2.py                                     │
│  ├── Fetches scoreboard data per week                         │
│  ├── Gets: scores, projections, matchup_recap                 │
│  ├── Gets: standings, waiver priority, FAAB                   │
│  ├── Gets: felo_score (Yahoo's power ranking)                │
│  └── Outputs: matchup_data_{year}.parquet (per season)        │
│                                                                │
│  Key Features:                                                 │
│  • Week window detection (week_start, week_end)               │
│  • Team metadata (manager name, team name, division)          │
│  • Projection data (team_projected_points)                    │
│  • Yahoo analysis (matchup_recap_url)                         │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    PHASE 2: AGGREGATION                        │
│                                                                │
│  aggregators.py → normalize_matchup_parquet()                  │
│  ├── Combines yearly files into matchup.parquet               │
│  ├── Normalizes types (year → int32, week → int32)            │
│  └── Creates composite keys (year_week, manager_year)         │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│              PHASE 3: BASE ENRICHMENT (cumulative_stats_v2.py) │
│                                                                │
│  11-Step Transformation Pipeline:                              │
│                                                                │
│  [1/11] matchup_keys.py                                        │
│         → Add matchup_key, matchup_id, matchup_sort_key       │
│                                                                │
│  [2/11] cumulative_records.py                                  │
│         → cumulative_wins/losses, win_streak, loss_streak     │
│         → wins_to_date, points_scored_to_date                 │
│         → playoff_seed_to_date, final_playoff_seed            │
│                                                                │
│  [3/11] playoff_flags.py                                       │
│         → is_playoffs, is_consolation, postseason             │
│         → playoff_round, consolation_round                    │
│         → quarterfinal, semifinal, championship flags         │
│                                                                │
│  [3b/11] playoff_bracket.py                                    │
│          → Simulates playoff brackets                          │
│          → champion, sacko, placement_rank                     │
│          → season_result ("won championship", etc.)           │
│                                                                │
│  [4/11] manager_ppg.py                                         │
│         → manager_season_mean, manager_season_median          │
│         → personal_season_mean (alias)                        │
│                                                                │
│  [5/11] weekly_metrics.py                                      │
│         → league_weekly_mean, league_weekly_median            │
│         → above_league_median, weekly_rank                    │
│         → teams_beat_this_week                                │
│                                                                │
│  [6/11] all_play_extended.py                                   │
│         → opponent_teams_beat_this_week                       │
│                                                                │
│  [7/11] head_to_head.py                                        │
│         → w_vs_{manager}, l_vs_{manager} for ALL managers     │
│         → Dynamic columns based on league members             │
│                                                                │
│  [8/11] comparative_schedule.py                                │
│         → w_vs_{manager}_sched, l_vs_{manager}_sched          │
│         → "If you played X's schedule, what's your record?"   │
│                                                                │
│  [9/11] season_rankings.py                                     │
│         → manager_season_ranking, final_wins/losses           │
│                                                                │
│  [9.5/11] matchup_rankings.py                                  │
│           → manager_all_time_ranking, percentile              │
│                                                                │
│  [10/11] alltime_rankings.py                                   │
│          → manager_all_time_gp, wins, losses, ties            │
│          → manager_all_time_win_pct                           │
│                                                                │
│  [11/11] inflation_rate calculation                            │
│          → Year-over-year scoring normalization               │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│           PHASE 4: PLAYER ENRICHMENT (player_to_matchup_v2.py) │
│                                                                │
│  Joins player.parquet to add:                                  │
│  ├── optimal_points (best possible lineup)                    │
│  ├── bench_points (unused player points)                      │
│  ├── lineup_efficiency (actual / optimal)                     │
│  ├── players_rostered, players_started                        │
│  └── opponent_optimal_points                                  │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│         PHASE 5: SIMULATION ENRICHMENT (expected_record_v2.py) │
│                                                                │
│  Monte Carlo Schedule Simulations (100,000 iterations):        │
│                                                                │
│  PERFORMANCE-BASED (shuffle_* columns):                        │
│  ├── shuffle_{N}_win: Probability of N wins                   │
│  ├── shuffle_{N}_seed: Probability of seed N                  │
│  ├── shuffle_avg_wins: Expected wins                          │
│  ├── shuffle_avg_seed: Expected seed                          │
│  ├── shuffle_avg_playoffs: Playoff probability (%)            │
│  ├── shuffle_avg_bye: Bye probability (%)                     │
│  └── wins_vs_shuffle_wins: Actual - Expected (luck measure)   │
│                                                                │
│  OPPONENT DIFFICULTY (opp_shuffle_* columns):                  │
│  ├── opp_shuffle_{N}_win: Win prob with random opponents      │
│  ├── opp_shuffle_{N}_seed: Seed prob with random opponents    │
│  ├── opp_shuffle_avg_*: Expected values                       │
│  └── wins_vs_opp_shuffle_wins: Schedule luck measure          │
│                                                                │
│  Additional:                                                   │
│  ├── is_final_regular_week: Flag for end-of-season stats     │
│  ├── is_bye_week: Flag for bye weeks                          │
│  └── opp_pts_week_rank, opp_pts_week_pct                     │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│         PHASE 6: PREDICTIVE (playoff_odds_import.py)           │
│                                                                │
│  ML-based playoff predictions:                                 │
│  ├── avg_seed: Machine learning predicted seed                │
│  ├── p_playoffs: Playoff probability (%)                      │
│  ├── p_bye: Bye probability (%)                               │
│  ├── exp_final_wins: Expected final win total                 │
│  ├── p_semis, p_final, p_champ: Advancement probabilities    │
│  ├── x{N}_seed: Probability of each seed (x1_seed...x10_seed)│
│  ├── x{N}_win: Probability of N wins (x0_win...x14_win)      │
│  └── power_rating: Composite team strength                    │
│                                                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                                │
│                                                                │
│  matchup.parquet                                               │
│  ├── 280 columns of enriched matchup data                     │
│  ├── ~1,924 rows (all games ever played)                      │
│  └── Foundation for Hall of Fame, Matchups, Simulations       │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

---

## Data Schema Reference

### Column Categories (280 Total)

Given the massive size, columns are organized by category:

#### Core Game Data (25 columns)
```
week, year, manager, team_name, team_points, team_projected_points,
opponent, opponent_team, opponent_points, opponent_projected_points,
margin, total_matchup_score, close_margin, win, loss, tie,
matchup_recap_title, matchup_recap_url, url, image_url,
division_id, week_start, week_end, league_id
```

#### Cumulative Records (15 columns)
```
cumulative_wins, cumulative_losses, wins_to_date, losses_to_date,
points_scored_to_date, win_streak, loss_streak, winning_streak, losing_streak,
playoff_seed_to_date, postseason, final_playoff_seed, final_regular_wins,
final_regular_losses, final_wins, final_losses
```

#### Playoff/Consolation Flags (20 columns)
```
is_playoffs, is_consolation, playoff_week_index, playoff_round_num,
playoff_round, consolation_round, quarterfinal, semifinal, championship,
placement_game, placement_rank, consolation_semifinal, consolation_final,
champion, sacko, season_result, playoff_start_week, num_playoff_teams,
is_final_regular_week, is_bye_week
```

#### League Context (15 columns)
```
league_weekly_mean, league_weekly_median, above_league_median, below_league_median,
weekly_rank, teams_beat_this_week, opponent_teams_beat_this_week,
gpa, grade, waiver_priority, has_draft_grade, faab_balance,
number_of_moves, number_of_trades, auction_budget_spent, auction_budget_total
```

#### Manager Statistics (15 columns)
```
manager_season_mean, manager_season_median, personal_season_mean, personal_season_median,
manager_season_ranking, manager_all_time_ranking, manager_all_time_percentile,
manager_all_time_gp, manager_all_time_wins, manager_all_time_losses,
manager_all_time_ties, manager_all_time_win_pct, season_mean, season_median, inflation_rate
```

#### Projection/Spread (10 columns)
```
proj_wins, proj_losses, proj_score_error, abs_proj_score_error,
above_proj_score, below_proj_score, expected_spread, expected_odds,
win_vs_spread, lose_vs_spread, underdog_wins, favorite_losses, win_probability
```

#### Head-to-Head Records (~24 columns, dynamic)
```
w_vs_adin, l_vs_adin, w_vs_daniel, l_vs_daniel, w_vs_eleff, l_vs_eleff,
w_vs_ezra, l_vs_ezra, w_vs_gavi, l_vs_gavi, w_vs_ilan, l_vs_ilan,
w_vs_jason, l_vs_jason, w_vs_jesse, l_vs_jesse, w_vs_marc, l_vs_marc,
w_vs_rubinstein, l_vs_rubinstein, w_vs_tani, l_vs_tani, w_vs_yaacov, l_vs_yaacov
```

#### Comparative Schedule (~24 columns, dynamic)
```
w_vs_adin_sched, l_vs_adin_sched, ... (one pair per manager)
```

#### Optimal Lineup (10 columns)
```
optimal_points, bench_points, total_player_points, players_rostered, players_started,
lineup_efficiency, optimal_ppg_season, rolling_optimal_points, total_optimal_points,
optimal_points_all_time, opponent_optimal_points, optimal_win, optimal_loss
```

#### Schedule Simulation - Performance (~35 columns)
```
shuffle_1_seed through shuffle_10_seed (10 columns)
shuffle_0_win through shuffle_14_win (15 columns)
shuffle_avg_wins, shuffle_avg_seed, shuffle_avg_playoffs, shuffle_avg_bye
wins_vs_shuffle_wins, seed_vs_shuffle_seed
```

#### Schedule Simulation - Opponent (~35 columns)
```
opp_shuffle_1_seed through opp_shuffle_10_seed (10 columns)
opp_shuffle_0_win through opp_shuffle_14_win (15 columns)
opp_shuffle_avg_wins, opp_shuffle_avg_seed, opp_shuffle_avg_playoffs, opp_shuffle_avg_bye
wins_vs_opp_shuffle_wins, seed_vs_opp_shuffle_seed
opp_pts_week_rank, opp_pts_week_pct
```

#### Predictive Analytics (~30 columns)
```
avg_seed, p_playoffs, p_bye, exp_final_wins, exp_final_pf
p_semis, p_final, p_champ, power_rating
x1_seed through x10_seed (10 columns)
x0_win through x14_win (15 columns)
```

#### Keys & Identifiers (10 columns)
```
year_week, cumulative_week, opponent_week, manager_year_week, matchup_key,
matchup_id, matchup_sort_key, manager_year, opponent_year, manager_week
```

#### Metadata (5 columns)
```
felo_score, felo_tier, coverage_value, value, above_opponent_median, below_opponent_median
```

---

## Column Categories

### SET-AND-FORGET Columns (Calculated once after championship)

These columns are finalized at season end and never recalculated:

| Column | Description |
|--------|-------------|
| `final_wins` | Season total wins |
| `final_losses` | Season total losses |
| `final_regular_wins` | Regular season wins only |
| `final_regular_losses` | Regular season losses only |
| `season_mean` | Season average points |
| `season_median` | Season median points |
| `manager_season_ranking` | Final season rank |
| `championship` | 1 if championship game |
| `champion` | 1 if won championship |
| `sacko` | 1 if lost consolation final |
| `season_result` | "won championship", etc. |
| `final_playoff_seed` | End-of-season seed |

### RECALCULATE WEEKLY Columns

These columns are updated every week for cross-week/year comparisons:

| Column | Description |
|--------|-------------|
| `cumulative_wins` | Running all-time win total |
| `win_streak` | Current active win streak |
| `teams_beat_this_week` | League-relative weekly performance |
| `w_vs_{opponent}` | Head-to-head records (dynamic) |
| `manager_all_time_ranking` | Cross-season historical rank |
| `shuffle_*` | Schedule simulation results |
| `p_playoffs`, `p_champ` | Predictive probabilities |

---

## Recommendations & Roadmap

### Critical Performance Issue: 280 Columns

The table has grown to 280 columns, which creates challenges:

| Issue | Impact | Solution |
|-------|--------|----------|
| Memory usage | High RAM consumption | Column pruning in data access |
| Load time | Slow initial load | Already using optimized loaders |
| Query complexity | DuckDB handles well | Continue current approach |

**Current Mitigation (Already Implemented):**
- `md/tab_data_access/managers/matchup_data.py`: Loads only 22-25 columns (78% reduction)
- `md/tab_data_access/simulations/matchup_data.py`: Loads all columns (required for simulations)

### New Metrics to Pre-Compute

Add these to `cumulative_stats_v2.py`:

```python
# 1. GAME QUALITY GRADE (A-F)
df['game_grade'] = pd.cut(
    df['team_points'],
    bins=[0, 80, 100, 120, 140, 200],
    labels=['F', 'D', 'C', 'B', 'A']
)

# 2. CLUTCH FLAG (won as underdog or in close game)
df['is_clutch_win'] = (
    (df['win'] == 1) &
    ((df['expected_odds'] < 0.4) | (abs(df['margin']) < 5))
).astype(int)

# 3. BLOWOUT FLAG (won/lost by 30+)
df['is_blowout'] = (abs(df['margin']) >= 30).astype(int)

# 4. HEARTBREAKER FLAG (lost close game after leading projection)
df['is_heartbreaker'] = (
    (df['loss'] == 1) &
    (df['team_projected_points'] > df['opponent_projected_points']) &
    (abs(df['margin']) < 10)
).astype(int)

# 5. SCHEDULE LUCK TIER
df['schedule_luck_tier'] = pd.cut(
    df['wins_vs_shuffle_wins'],
    bins=[-10, -2, -0.5, 0.5, 2, 10],
    labels=['Very Unlucky', 'Unlucky', 'Normal', 'Lucky', 'Very Lucky']
)

# 6. DOMINANCE SCORE (weekly rank + margin)
df['weekly_dominance'] = (
    (df['teams_beat_this_week'] / 9) * 50 +  # 0-50 based on teams beat
    df['margin'].clip(-30, 30) + 30  # 0-60 based on margin
) / 110 * 100  # Normalize to 0-100
```

### UI/UX Improvements Summary

See separate documentation files for specific recommendations:
- `matchup_hall_of_fame_documentation.md` - Hall of Fame UI
- `matchup_matchups_documentation.md` - Matchups UI
- `matchup_simulations_documentation.md` - Simulations UI

### Performance Optimizations

```python
# Already implemented - column selection in data loaders
# Future: Consider creating materialized views for common queries

# Hall of Fame needs: ~40 columns
# Matchups needs: ~60 columns
# Simulations needs: ALL columns (required for shuffle_*)

# Possible optimization: Split into multiple tables
# matchup_core.parquet (~50 cols) - always needed
# matchup_simulations.parquet (~100 cols) - simulations only
# matchup_h2h.parquet (~50 cols) - head-to-head
```

---

## Appendix: File Locations

### Data Pipeline Scripts
```
fantasy_football_data_scripts/multi_league/
├── data_fetchers/
│   ├── weekly_matchup_data_v2.py     # Yahoo API fetcher
│   └── aggregators.py                 # File aggregation
├── transformations/
│   ├── base/
│   │   ├── cumulative_stats_v2.py    # Main transformation
│   │   ├── enrich_schedule_with_playoff_flags.py
│   │   └── modules/
│   │       ├── matchup_keys.py
│   │       ├── cumulative_records.py
│   │       ├── playoff_flags.py
│   │       ├── playoff_bracket.py    # Champion/Sacko detection
│   │       ├── weekly_metrics.py
│   │       ├── head_to_head.py
│   │       ├── comparative_schedule.py
│   │       ├── all_play_extended.py
│   │       ├── manager_ppg.py
│   │       ├── season_rankings.py
│   │       └── matchup_rankings.py
│   └── matchup_enrichment/
│       ├── player_to_matchup_v2.py   # Optimal lineup
│       ├── expected_record_v2.py     # Simulations
│       ├── playoff_odds_import.py    # Predictive
│       └── modules/
│           ├── schedule_simulation.py
│           └── bye_week_filler.py
```

### Data Files
```
fantasy_football_data/KMFFL/
├── matchup.parquet                    # Canonical matchup file (280 cols)
└── matchup_data/
    ├── matchup_data_2014.parquet      # Per-year files
    └── ...
```

---

*This is the core documentation. See separate files for Hall of Fame, Matchups, and Simulations UI documentation.*
