# SPAR Metrics Specification

## Architecture Overview

**Single Source of Truth**: SPAR calculated once at player-week level, then aggregated downstream.

### Data Flow
```
player.parquet (weekly SPAR)
    ↓
    ├─→ draft.parquet (season aggregation)
    └─→ transactions.parquet (ROS aggregation)
```

---

## Player Table (Weekly Level)

### Core SPAR Metrics
- **`player_spar`**: Weekly SPAR for all games played (talent metric)
  - Formula: `fantasy_points - replacement_ppg`
  - Measures player's total production

- **`manager_spar`**: Weekly SPAR for started games only (usage metric)
  - Formula: `(fantasy_points - replacement_ppg) if is_started==1 else 0`
  - Measures value actually captured by manager
  - **Excludes**: BN, IR, null, blank, "0", "nan", "NaN" positions

- **`replacement_ppg`**: Weekly position-specific replacement baseline
  - Calculated from actual roster composition (dynamic)
  - Example: QB16, RB52, WR60 for 10-team league

### Key Properties
- **Granularity**: One row per player-week
- **Invariant**: `player_spar >= manager_spar` (always true)
- **Join Key**: `player_id`, `year`, `week`

---

## Draft Table (Season Level)

### Aggregated SPAR Metrics
- **`player_spar`**: Season total SPAR (all games)
  - Sum of weekly `player_spar` from player table
  - Measures total player production for the season

- **`manager_spar`**: Season total SPAR (started games only)
  - Sum of weekly `manager_spar` from player table
  - Measures value manager actually captured

- **`replacement_ppg`**: Season average replacement baseline
  - Average of weekly `replacement_ppg` from player table

### Derived Metrics
- **`draft_roi`**: Return on investment
  - Formula: `manager_spar / cost_norm`
  - Uses managed SPAR (realized value) for ROI

- **`cost_norm`**: Normalized draft cost
  - Auction: Actual cost
  - Snake: Exponential decay from pick number

### Per-Game Metrics (TO BE ADDED)
- **`player_spar_per_game`**: `player_spar / games_played`
- **`manager_spar_per_game`**: `manager_spar / games_started`

### Key Properties
- **Granularity**: One row per player-season (drafted player)
- **Grouping**: `year`, `NFL_player_id`, `manager`
- **Source**: All metrics aggregated from player table (NO recalculation)

---

## Transaction Table (ROS Level)

### Individual Player ROS Metrics

#### Managed (On Your Roster Only)
- **`manager_spar_ros_managed`**: ROS SPAR while on YOUR roster
  - Only counts weeks where `(manager == this_manager) & (is_rostered == 1)`
  - Stops automatically at drops/trades (when manager changes)
  - **Use case**: "What value did this player give ME?"

- **`total_points_ros_managed`**: Same logic for fantasy points
- **`ppg_ros_managed`**: Average PPG while on your roster
- **`weeks_ros_managed`**: Number of weeks on your roster
- **`replacement_ppg_ros_managed`**: Average replacement baseline for managed window

#### Total (Regardless of Roster)
- **`player_spar_ros_total`**: Total ROS SPAR for all future games
  - Counts ALL future weeks after transaction
  - **Use case**: "What value did I miss by dropping/trading this player?"

- **`total_points_ros_total`**: Same logic for fantasy points
- **`ppg_ros_total`**: Average PPG for all ROS weeks
- **`weeks_ros_total`**: Number of ROS weeks
- **`replacement_ppg_ros_total`**: Average replacement baseline for total window

#### Per-Game Metrics (TO BE ADDED)
- **`manager_spar_per_game_managed`**: `manager_spar_ros_managed / weeks_ros_managed`
- **`player_spar_per_game_total`**: `player_spar_ros_total / weeks_ros_total`

### NET Transaction Metrics (Grouped by transaction_id + manager)

#### Current Implementation
- **`net_manager_spar_ros`**: Net SPAR actually captured
  - Formula: `sum(adds.manager_spar_ros_managed) - sum(drops.manager_spar_ros_managed)`
  - **Use case**: "Did this add/drop/trade help MY team?"

- **`net_player_spar_ros`**: Net opportunity cost
  - Formula: `sum(adds.player_spar_ros_total) - sum(drops.player_spar_ros_total)`
  - **Use case**: "What total value did I gain/lose?"

- **`spar_efficiency`**: SPAR per FAAB dollar
  - Formula: `net_manager_spar_ros / faab_bid`
  - **Use case**: "Which adds gave the best value per dollar?"

- **`net_spar_rank`**: Rank by net managed SPAR within position/year
  - **Use case**: "Was this one of my best transactions?"

#### Legacy Columns
- **`net_spar_ros`**: Alias for `net_manager_spar_ros` (backward compatibility)

### Key Properties
- **Granularity**: One row per player per transaction
- **Grouping**: NET metrics same for all rows in `transaction_id` + `manager`
- **Source**: ROS metrics aggregated from player table (NO recalculation)

---

## Trade-Specific NET vs Add/Drop NET

### Current Add/Drop Logic
```python
# For transaction_id + manager:
adds = transactions[type.isin(['add', 'trade'])]
drops = transactions[type == 'drop']
net = sum(adds.spar) - sum(drops.spar)
```

### What Trade-Specific Would Offer

#### Context Difference
- **Add/Drop**: Comparing against replacement level (waiver wire baseline)
- **Trade**: Comparing rostered players (both had value above replacement)

#### Potential Trade-Specific Metrics
1. **Bilateral Perspective**: Did both managers improve?
   - `net_spar_manager_a` vs `net_spar_manager_b`
   - Identifies win-win vs win-lose trades

2. **Trade Imbalance**:
   - `abs(net_spar_manager_a - net_spar_manager_b)`
   - Measures fairness of trade

3. **Trade Context**:
   - Both players were already rostered (not waiver wire)
   - Negotiation vs unilateral decision
   - Positional needs (trading strength for need)

#### Current Limitation
Current logic treats all trades as "adds" without distinguishing:
- **Incoming**: Players you received
- **Outgoing**: Players you gave up

For proper trade NET, would need to:
1. Separate incoming/outgoing players per manager
2. Calculate bilateral NET for both sides
3. Potentially weight by positional scarcity at time of trade

### Recommendation
**Start with current implementation** because:
1. Current NET already works for simple trades (1-for-1)
2. Multi-player trades are rare in most leagues
3. Can add trade-specific later if needed

**Add trade-specific if**:
- You have many multi-player trades
- You want to analyze trade fairness
- You want to identify trade winners/losers

---

## Missing Metrics to Add

### High Priority
1. **Per-game SPAR in draft table**
   - `player_spar_per_game = player_spar / games_played`
   - `manager_spar_per_game = manager_spar / games_started`

2. **Per-game SPAR in transaction table**
   - `manager_spar_per_game_managed = manager_spar_ros_managed / weeks_ros_managed`
   - `player_spar_per_game_total = player_spar_ros_total / weeks_ros_total`

### Medium Priority
3. **Consistency metrics**
   - `player_spar_std_dev`: Standard deviation of weekly player_spar
   - `manager_spar_std_dev`: Standard deviation of weekly manager_spar
   - Identifies boom/bust players

4. **Cumulative SPAR over time**
   - Rolling sum of SPAR through the season
   - Useful for tracking value accumulation

### Low Priority (Future Enhancements)
5. **Opponent-adjusted SPAR** (in matchup table)
   - SPAR against specific opponents
   - Context: strength of schedule

6. **Trade-specific NET metrics**
   - Bilateral analysis
   - Trade imbalance
   - Positional context

---

## Implementation Status

### ✅ Completed
- [x] Player table: Weekly `player_spar` and `manager_spar`
- [x] Draft table: Season aggregated SPAR from player table
- [x] Transaction table: Dual ROS metrics (managed vs total)
- [x] Transaction table: NET SPAR (managed and total)
- [x] `is_started` logic: Excludes BN, IR, null, blank, NaN
- [x] Single source of truth: No SPAR recalculation downstream
- [x] Pipeline: All 6 steps passing

### 🚧 To Be Added
- [ ] Draft table: Per-game SPAR metrics
- [ ] Transaction table: Per-game ROS SPAR metrics
- [ ] (Optional) Consistency metrics
- [ ] (Optional) Trade-specific NET metrics

---

## File Locations

### Core Scripts
- **Player SPAR**: `player_enrichment/player_stats_v2.py`
- **Draft aggregation**: `draft_enrichment/player_to_draft_v2.py`
- **Draft SPAR**: `draft_enrichment/modules/spar_calculator.py`
- **Transaction enrichment**: `transaction_enrichment/player_to_transactions_v2.py`
- **Transaction SPAR**: `transaction_enrichment/modules/transaction_spar_calculator.py`
- **Replacement levels**: `player_enrichment/replacement_level_v2.py`

### Data Files
- **Player**: `KMFFL/player.parquet`
- **Draft**: `KMFFL/draft.parquet`
- **Transactions**: `KMFFL/transactions.parquet`
- **Replacement levels**: `KMFFL/transformations/replacement_levels.parquet`

---

## Analysis Use Cases

### Draft Analysis
- "Which round gave the best SPAR per pick?"
- "Did I start my best picks?" (`manager_spar` vs `player_spar`)
- "Best ROI picks?" (`draft_roi`)

### Transaction Analysis
- "Best waiver wire adds by SPAR?" (`manager_spar_ros_managed`)
- "Worst drops?" (high `player_spar_ros_total` - missed opportunity)
- "Best FAAB value?" (`spar_efficiency`)
- "Did this trade help me?" (`net_manager_spar_ros`)

### Season Review
- "Where did I gain/lose value?" (compare draft vs transaction NET)
- "Did I manage my roster well?" (manager_spar vs player_spar gap)
- "Who were my most valuable players?" (total `player_spar`)
