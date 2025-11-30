# Data Pipeline Recommendations

> **Scope:** Data fetching, transformations, enrichment, and source table optimizations
> **Goal:** Pre-compute metrics for faster UI, reduce redundancy, improve efficiency

---

## Executive Summary

Your data pipeline is **well-architected** with sophisticated features like dual SPAR metrics, 100K Monte Carlo simulations, and proper SET-AND-FORGET vs RECALCULATE column separation. The main opportunities are:

1. **Add engagement metrics** (grades, tiers) to source tables
2. **Reduce schema redundancy** (legacy aliases)
3. **Pre-compute aggregations** that UI currently calculates

---

## Priority 1: New Metrics to Add (High Impact)

### Draft Table (`draft_value_metrics_v3.py`)

```python
def add_draft_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add fun/engaging metrics for UI displays."""

    # 1. DRAFT GRADE (A-F based on SPAR percentile within position/year)
    df['spar_percentile'] = df.groupby(['year', 'position'])['manager_spar'].transform(
        lambda x: x.rank(pct=True) * 100
    )
    df['draft_grade'] = pd.cut(
        df['spar_percentile'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['F', 'D', 'C', 'B', 'A']
    )

    # 2. VALUE TIER (Steal/Good/Fair/Reach/Bust)
    df['value_tier'] = pd.cut(
        df['price_rank_vs_finish_rank'].fillna(0),
        bins=[-100, -5, -2, 2, 5, 100],
        labels=['Bust', 'Reach', 'Fair', 'Good Value', 'Steal']
    )

    # 3. PLAYER TYPE (consistency-based)
    df['consistency_cv'] = df['season_std'] / df['season_ppg'].clip(lower=0.1)
    df['player_type'] = pd.cut(
        df['consistency_cv'].fillna(0.5),
        bins=[0, 0.3, 0.5, 100],
        labels=['Steady Eddie', 'Normal', 'Boom/Bust']
    )

    # 4. BREAKOUT FLAG (late round, top 10 finish)
    df['is_breakout'] = (
        (df['round'] >= 8) &
        (df['season_position_rank'] <= 10)
    ).astype(int)

    # 5. BUST FLAG (early round, bottom half finish)
    df['is_bust'] = (
        (df['round'] <= 3) &
        (df['season_position_rank'] > df['total_position_players'] / 2)
    ).astype(int)

    # 6. DRAFT TIER (Early/Mid/Late)
    df['draft_tier'] = pd.cut(
        df['round'].fillna(10),
        bins=[0, 3, 7, 20],
        labels=['Early (1-3)', 'Mid (4-7)', 'Late (8+)']
    )

    return df
```

**New Columns:** `draft_grade`, `value_tier`, `player_type`, `is_breakout`, `is_bust`, `draft_tier`

---

### Transactions Table (`transaction_value_metrics_v3.py`)

```python
def add_transaction_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add fun/engaging metrics for transaction displays."""

    # 1. TRANSACTION GRADE (A-F based on NET SPAR percentile)
    df['net_spar_percentile'] = df.groupby(['year', 'transaction_type'])['net_manager_spar_ros'].transform(
        lambda x: x.rank(pct=True) * 100
    )
    df['transaction_grade'] = pd.cut(
        df['net_spar_percentile'].fillna(50),
        bins=[0, 20, 40, 60, 80, 100],
        labels=['F', 'D', 'C', 'B', 'A']
    )

    # 2. TRANSACTION RESULT CATEGORY
    def categorize_result(net_spar):
        if pd.isna(net_spar): return 'Unknown'
        if net_spar > 100: return 'Elite Win'
        if net_spar > 50: return 'Great Win'
        if net_spar > 20: return 'Good Win'
        if net_spar > 0: return 'Small Win'
        if net_spar == 0: return 'Even'
        if net_spar > -20: return 'Small Loss'
        if net_spar > -50: return 'Bad Loss'
        return 'Major Loss'

    df['transaction_result'] = df['net_manager_spar_ros'].apply(categorize_result)

    # 3. FAAB VALUE TIER
    df['faab_value_tier'] = pd.cut(
        df['spar_efficiency'].fillna(0),
        bins=[-float('inf'), 0, 1, 3, 5, float('inf')],
        labels=['Overpay', 'Fair', 'Good Value', 'Great Value', 'Steal']
    )

    # 4. DROP REGRET SCORE (for drops only)
    df['drop_regret'] = np.where(
        df['transaction_type'] == 'drop',
        df['player_spar_ros_total'].fillna(0),
        np.nan
    )
    df['drop_regret_tier'] = pd.cut(
        df['drop_regret'].fillna(-1),
        bins=[-1, 0, 10, 30, 50, 100, float('inf')],
        labels=['N/A', 'No Regret', 'Minor', 'Moderate', 'Major', 'Disaster']
    )

    # 5. TIMING CATEGORY
    df['timing_category'] = pd.cut(
        df['week'].fillna(1),
        bins=[0, 4, 10, 14, 18],
        labels=['Early Season', 'Mid Season', 'Late Season', 'Playoffs']
    )

    # 6. PICKUP TYPE
    df['pickup_type'] = df['source_type'].map({
        'waivers': 'Waiver Claim',
        'freeagents': 'Free Agent',
        'team': 'Trade'
    }).fillna('Other')

    return df
```

**New Columns:** `transaction_grade`, `transaction_result`, `faab_value_tier`, `drop_regret`, `drop_regret_tier`, `timing_category`, `pickup_type`

---

### Matchup Table (`cumulative_stats_v2.py`)

```python
def add_matchup_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add fun/engaging metrics for matchup displays."""

    # 1. GAME GRADE (A-F based on points vs league median)
    league_median = df.groupby(['year', 'week'])['team_points'].transform('median')
    df['game_grade'] = pd.cut(
        df['team_points'] / league_median.clip(lower=1),
        bins=[0, 0.7, 0.9, 1.1, 1.3, 10],
        labels=['F', 'D', 'C', 'B', 'A']
    )

    # 2. CLUTCH WIN FLAG
    df['is_clutch_win'] = (
        (df['win'] == 1) &
        ((df['expected_odds'] < 0.4) | (abs(df['margin']) < 5))
    ).astype(int)

    # 3. BLOWOUT FLAG (30+ point margin)
    df['is_blowout'] = (abs(df['margin']) >= 30).astype(int)

    # 4. HEARTBREAKER FLAG
    df['is_heartbreaker'] = (
        (df['loss'] == 1) &
        (df['team_projected_points'] > df['opponent_projected_points']) &
        (abs(df['margin']) < 10)
    ).astype(int)

    # 5. SCHEDULE LUCK TIER
    df['schedule_luck_tier'] = pd.cut(
        df['wins_vs_shuffle_wins'].fillna(0),
        bins=[-10, -2, -0.5, 0.5, 2, 10],
        labels=['Very Unlucky', 'Unlucky', 'Normal', 'Lucky', 'Very Lucky']
    )

    # 6. PLAYOFF ODDS TIER
    df['playoff_odds_tier'] = pd.cut(
        df['p_playoffs'].fillna(0),
        bins=[0, 10, 30, 50, 70, 90, 100],
        labels=['Eliminated', 'Long Shot', 'Bubble', 'Likely', 'Very Likely', 'Locked']
    )

    # 7. WEEKLY DOMINANCE SCORE (0-100)
    max_teams = df.groupby(['year', 'week'])['manager'].transform('count') - 1
    df['weekly_dominance'] = (
        (df['teams_beat_this_week'] / max_teams.clip(lower=1)) * 50 +
        (df['margin'].clip(-30, 30) + 30) / 60 * 50
    ).clip(0, 100)

    # 8. CONSISTENCY SCORE (season-level, inverse of CV)
    season_cv = df.groupby(['manager', 'year'])['team_points'].transform(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 0
    )
    df['consistency_score'] = (100 - season_cv * 100).clip(0, 100)

    # 9. CLINCH/ELIMINATION FLAGS
    df['has_clinched_playoffs'] = (df['p_playoffs'] >= 99.9).astype(int)
    df['is_eliminated'] = (df['p_playoffs'] <= 0.1).astype(int)

    # 10. LINEUP EFFICIENCY GRADE
    df['efficiency_grade'] = pd.cut(
        (df['lineup_efficiency'].fillna(0) * 100),
        bins=[0, 75, 85, 92, 97, 100],
        labels=['F', 'D', 'C', 'B', 'A']
    )

    return df
```

**New Columns:** `game_grade`, `is_clutch_win`, `is_blowout`, `is_heartbreaker`, `schedule_luck_tier`, `playoff_odds_tier`, `weekly_dominance`, `consistency_score`, `has_clinched_playoffs`, `is_eliminated`, `efficiency_grade`

---

### Player Table (`player_stats_v2.py`)

```python
def add_player_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add fun/engaging metrics for player displays."""

    # 1. PERFORMANCE GRADE (A-F based on points vs position median)
    df['performance_grade'] = df.groupby(['year', 'week', 'nfl_position'])['points'].transform(
        lambda x: pd.cut(
            x.rank(pct=True) * 100,
            bins=[0, 20, 40, 60, 80, 100],
            labels=['F', 'D', 'C', 'B', 'A']
        )
    )

    # 2. BOOM FLAG (top 10% of position that week)
    df['is_boom'] = df.groupby(['year', 'week', 'nfl_position'])['points'].transform(
        lambda x: (x >= x.quantile(0.9)).astype(int)
    )

    # 3. BUST FLAG (bottom 20% when started)
    df['is_bust'] = (
        (df['started'] == 1) &
        (df.groupby(['year', 'week', 'nfl_position'])['points'].transform(
            lambda x: x <= x.quantile(0.2)
        ))
    ).astype(int)

    # 4. CONSISTENCY TIER (season-level)
    season_cv = df.groupby(['player', 'year'])['points'].transform(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 1
    )
    df['consistency_tier'] = pd.cut(
        season_cv,
        bins=[0, 0.3, 0.5, 0.7, 100],
        labels=['Elite', 'Steady', 'Variable', 'Boom/Bust']
    )

    # 5. LEAGUE WINNER FLAG (top 3 at position + on championship team)
    df['is_league_winner'] = (
        (df['position_season_rank'] <= 3) &
        (df['champion'] == 1)
    ).astype(int)

    # 6. SPAR TIER (for rostered players)
    df['spar_tier'] = pd.cut(
        df['manager_spar'].fillna(0),
        bins=[-100, 0, 5, 10, 20, 100],
        labels=['Negative', 'Replacement', 'Solid', 'Good', 'Elite']
    )

    # 7. EFFICIENCY SCORE (points per target/carry)
    df['usage_efficiency'] = np.where(
        (df['targets'] + df['carries']) > 0,
        df['points'] / (df['targets'].fillna(0) + df['carries'].fillna(0)),
        np.nan
    )

    # 8. OPTIMAL LINEUP RATE (how often player was optimal choice)
    df['optimal_rate'] = df.groupby(['player', 'year'])['optimal_player'].transform('mean')

    return df
```

**New Columns:** `performance_grade`, `is_boom`, `is_bust`, `consistency_tier`, `is_league_winner`, `spar_tier`, `usage_efficiency`, `optimal_rate`

---

## Priority 2: Schema Cleanup (Reduce Redundancy)

### Draft Table (67 → ~60 columns)

| Redundant Column | Keep Instead | Action |
|------------------|--------------|--------|
| `spar` | `manager_spar` | Deprecate (keep for 1 version) |
| `pgvor` | `manager_pgvor` | Deprecate |
| `spar_per_dollar_norm` | `draft_roi` | Remove duplicate |
| `points_per_dollar` | `spar_per_dollar` | Keep SPAR version |

### Transactions Table (78 → ~65 columns)

| Redundant Column | Keep Instead | Action |
|------------------|--------------|--------|
| `type` | `transaction_type` | Remove alias |
| `net_spar_ros` | `net_manager_spar_ros` | Deprecate |
| `fa_spar_ros` | `manager_spar_ros` | Deprecate |
| `total_points_rest_of_season` | `total_points_ros_total` | Deprecate |

### Matchup Table (280 columns - Complex)

The matchup table is large but **mostly justified** due to:
- Dynamic H2H columns (w_vs_{manager} for each manager)
- Dual simulation columns (shuffle_* and opp_shuffle_*)
- Predictive columns (x*_seed, x*_win)

**Possible future optimization:**
```
matchup_core.parquet (~80 cols)      - Core game data, always needed
matchup_h2h.parquet (~50 cols)       - Head-to-head columns
matchup_simulations.parquet (~100 cols) - Simulation results
matchup_predictive.parquet (~50 cols)  - ML predictions
```

### Player Table (280 columns - Well Justified)

The player table is the **largest and most comprehensive**:
- 182,650 rows (every player, every week)
- 280 columns spanning identifiers, stats, advanced metrics, matchup context

| Redundant Column | Keep Instead | Action |
|------------------|--------------|--------|
| `spar` | `player_spar` or `manager_spar` | Deprecate legacy alias |
| Position aliases | Standardize on `nfl_position` | Document mapping |

**Keep `players_by_year.parquet`**: The pre-aggregated table provides meaningful performance benefits:
- 17% row reduction (152K vs 182K)
- Eliminates runtime GROUP BY for season/career views
- 5-10x faster for leaderboard queries

### Team Stats (No Dedicated Table - Consider Adding)

Currently team stats are computed at runtime from `player.parquet`. Consider:

```python
# Create team_by_week.parquet for faster team stats
team_weekly = player_df.groupby(['manager', 'year', 'week', 'fantasy_position']).agg({
    'points': 'sum',
    'player_spar': 'sum',
    'manager_spar': 'sum',
    # ... other aggregations
}).reset_index()
```

**Benefits:**
- Eliminates 60+ column aggregation on every page load
- Est. 5-10x faster for team stats views
- ~10K rows vs 182K rows (98% reduction)

---

## Priority 3: Pre-Compute Aggregations

### Transaction Summaries Table (NEW)

Create `transaction_summaries.parquet` with one row per transaction:

```python
def create_transaction_summaries(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute transaction-level summaries for faster UI."""

    # Group by transaction_id and manager
    adds = transactions_df[transactions_df['transaction_type'] == 'add'].groupby(
        ['transaction_id', 'manager']
    ).agg({
        'player_name': lambda x: ', '.join(x.dropna()),
        'position': lambda x: ', '.join(x.dropna()),
        'manager_spar_ros_managed': 'sum',
        'faab_bid': 'first'
    }).rename(columns={
        'player_name': 'players_added',
        'position': 'positions_added',
        'manager_spar_ros_managed': 'spar_added'
    })

    drops = transactions_df[transactions_df['transaction_type'] == 'drop'].groupby(
        ['transaction_id', 'manager']
    ).agg({
        'player_name': lambda x: ', '.join(x.dropna()),
        'position': lambda x: ', '.join(x.dropna()),
        'player_spar_ros_total': 'sum'
    }).rename(columns={
        'player_name': 'players_dropped',
        'position': 'positions_dropped',
        'player_spar_ros_total': 'spar_dropped'
    })

    # Merge and calculate NET
    summaries = adds.join(drops, how='outer').reset_index()
    summaries['net_spar'] = summaries['spar_added'].fillna(0) - summaries['spar_dropped'].fillna(0)
    summaries['transaction_grade'] = pd.cut(...)  # Add grade

    return summaries
```

**Benefits:** UI doesn't have to group by transaction_id on every load.

---

### Season Manager Summaries (Enhance Existing)

Add to season aggregation:

```python
# Add these to season-level summaries
manager_season_stats = df.groupby(['manager', 'year']).agg({
    'team_points': ['sum', 'mean', 'std', 'max', 'min'],
    'win': 'sum',
    'loss': 'sum',
    'is_clutch_win': 'sum',
    'is_blowout': 'sum',
    'lineup_efficiency': 'mean',
    'wins_vs_shuffle_wins': 'last',  # End of season value
    'p_playoffs': 'last',
    'champion': 'max'
}).reset_index()
```

---

## Priority 4: Query Optimizations

### Filter Undrafted in SQL (Draft)

```python
# Current (draft_data.py)
query = f"""
    SELECT {cols_str}
    FROM {T['draft']}
    ORDER BY year DESC, round, pick
"""

# Recommended - filter undrafted in DB
query = f"""
    SELECT {cols_str}
    FROM {T['draft']}
    WHERE manager IS NOT NULL
      AND manager != ''
      AND cost > 0
    ORDER BY year DESC, round, pick
"""
```

### Add Indexes for Common Queries

```python
# Consider adding these as separate lookup tables:

# Manager lookup (for fast filtering)
managers = df[['manager', 'manager_year']].drop_duplicates()

# Week lookup (for fast week filtering)
weeks = df[['year', 'week', 'cumulative_week', 'is_playoffs', 'is_consolation']].drop_duplicates()
```

---

## Priority 5: Data Validation

Add validation checks to transformation pipeline:

```python
def validate_draft_data(df: pd.DataFrame) -> bool:
    """Validate draft table integrity."""

    errors = []

    # Check for required columns
    required = ['year', 'pick', 'manager', 'player', 'cost', 'manager_spar']
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Check for orphan records
    orphan_count = df[df['manager'].isna()].shape[0]
    if orphan_count > 0:
        print(f"[WARN] {orphan_count} rows with null manager (undrafted players)")

    # Check SPAR consistency
    spar_issues = df[(df['manager_spar'].notna()) & (df['player_spar'].isna())]
    if len(spar_issues) > 0:
        errors.append(f"SPAR inconsistency: {len(spar_issues)} rows")

    # Check grade distribution
    if 'draft_grade' in df.columns:
        grade_dist = df['draft_grade'].value_counts(normalize=True)
        # Each grade should be ~20% (±10%)
        for grade in ['A', 'B', 'C', 'D', 'F']:
            pct = grade_dist.get(grade, 0)
            if pct < 0.10 or pct > 0.30:
                print(f"[WARN] Grade {grade} is {pct:.1%} (expected ~20%)")

    return len(errors) == 0
```

---

## Implementation Checklist

### Phase 1: Engagement Metrics (1-2 hours each)
- [ ] Add draft engagement metrics to `draft_value_metrics_v3.py`
- [ ] Add transaction engagement metrics to `transaction_value_metrics_v3.py`
- [ ] Add matchup engagement metrics to `cumulative_stats_v2.py`
- [ ] Add player engagement metrics to `player_stats_v2.py`
- [ ] Re-run pipeline to generate new columns

### Phase 2: Schema Cleanup (30 min each)
- [ ] Deprecate draft legacy columns (add to ignore list)
- [ ] Deprecate transaction legacy columns
- [ ] Document deprecated columns in schema

### Phase 3: Pre-computed Tables (2-3 hours)
- [ ] Create `transaction_summaries.parquet` generator
- [ ] Enhance season manager summaries
- [ ] Update UI to use new tables

### Phase 4: Query Optimizations (1 hour)
- [ ] Add filters to draft data loader
- [ ] Add filters to transaction data loader
- [ ] Verify performance improvement

### Phase 5: Validation (1 hour)
- [ ] Add validation functions
- [ ] Add to pipeline as post-processing check
- [ ] Log validation results

---

## Column Count Summary (After Changes)

| Table | Current | After Cleanup | After New Metrics |
|-------|---------|---------------|-------------------|
| Draft | 67 | ~60 | ~66 |
| Transactions | 78 | ~65 | ~73 |
| Matchup | 280 | 280 | ~291 |
| Player | 280 | ~278 | ~286 |
| Players by Year | 244 | 244 | 244 (pre-aggregated) |

---

## New Pre-Computed Tables (Recommended)

| Table | Purpose | Est. Rows | Benefits |
|-------|---------|-----------|----------|
| `transaction_summaries.parquet` | One row per transaction | ~2K | No runtime grouping |
| `team_by_week.parquet` | Team stats pre-aggregated | ~10K | 5-10x faster team stats |
| `player_lookup.parquet` | Player ID → name/position | ~5K | Fast filter dropdowns |

---

*This document consolidates all data pipeline recommendations. See `RECOMMENDATIONS_APP_UI.md` for UI/UX recommendations.*
