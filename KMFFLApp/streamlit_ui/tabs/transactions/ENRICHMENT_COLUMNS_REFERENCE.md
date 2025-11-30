# Transaction Enrichment Columns Reference

This document describes all the enrichment columns available in transaction data after running `player_to_transactions_v2.py`.

## Enrichment Overview

Transaction data is enriched with player performance metrics to help analyze the quality and impact of adds, drops, and trades. **76.1%** of transactions have enrichment data.

## Available Enrichment Columns

### Player Context (at time of transaction)
- **`position`** (str): Player's position (QB, RB, WR, TE, K, DEF)
- **`nfl_team`** (str): Player's NFL team at transaction time
- **`points_at_transaction`** (float): Fantasy points scored in the week of transaction

### Performance Before Transaction
- **`ppg_before_transaction`** (float): Points per game average in the 4 weeks before transaction
- **`weeks_before`** (int): Number of weeks included in the "before" calculation
- **`position_rank_before_transaction`** (int): Season-to-date position rank before the transaction

### Performance After Transaction (KEY METRICS)
- **`ppg_after_transaction`** (float): Points per game average in the 4 weeks after transaction
- **`total_points_after_4wks`** (float): Total points in the 4 weeks after transaction
- **`weeks_after`** (int): Number of weeks included in the "after" calculation
- **`position_rank_after_transaction`** (int): Rest-of-season position rank after the transaction

### Rest of Season Performance
- **`total_points_rest_of_season`** (float): Total points for rest of season after transaction
- **`ppg_rest_of_season`** (float): Points per game for rest of season after transaction
- **`weeks_rest_of_season`** (int): Number of remaining weeks in season

### Position Rankings
- **`position_rank_at_transaction`** (int): Weekly position rank in the week of transaction
- **`position_total_players`** (int): Total players at position that week

### Value Metrics
- **`points_per_faab_dollar`** (float): Rest-of-season points divided by FAAB bid (for adds only)
- **`transaction_quality_score`** (int): Simple quality score (1-5 scale)
  - **For adds**: Based on ppg_after_transaction
    - 5 = 15+ PPG (Elite)
    - 4 = 10-15 PPG (Great)
    - 3 = 7-10 PPG (Good)
    - 2 = 5-7 PPG (Decent)
    - 1 = <5 PPG (Bust)
  - **For drops**: Based on ppg_after_transaction
    - 3 = ≤3 PPG (Good drop)
    - 2 = 3-5 PPG (OK drop)
    - 1 = 5-7 PPG (Questionable)
    - -1 = 7+ PPG (Bad drop)

## Column Aliases

For backward compatibility, the following aliases are created automatically:
- `transaction_score` → `transaction_quality_score`
- `nfl_team_at_transaction` → `nfl_team`
- `weeks_after_transaction` → `weeks_after`
- `weeks_before_transaction` → `weeks_before`
- `total_points_after` → `total_points_after_4wks`
- `avg_position_rank_after` → `position_rank_after_transaction`

## Usage Examples

### Example 1: Best Pickups by PPG
```python
# Filter to adds with enrichment data
adds = transaction_df[
    (transaction_df['transaction_type'] == 'add') &
    (transaction_df['ppg_rest_of_season'].notna())
]

# Sort by rest-of-season PPG
best_pickups = adds.nlargest(10, 'ppg_rest_of_season')[
    ['player_name', 'manager', 'week', 'year', 'faab_bid',
     'ppg_rest_of_season', 'total_points_rest_of_season']
]
```

### Example 2: Best FAAB Value
```python
# Filter to FAAB adds
faab_adds = transaction_df[
    (transaction_df['transaction_type'] == 'add') &
    (transaction_df['faab_bid'] > 0) &
    (transaction_df['points_per_faab_dollar'].notna())
]

# Sort by points per dollar
best_value = faab_adds.nlargest(10, 'points_per_faab_dollar')[
    ['player_name', 'manager', 'faab_bid',
     'points_per_faab_dollar', 'total_points_rest_of_season']
]
```

### Example 3: Worst Drops
```python
# Filter to drops with enrichment data
drops = transaction_df[
    (transaction_df['transaction_type'] == 'drop') &
    (transaction_df['ppg_rest_of_season'].notna())
]

# Sort by rest-of-season PPG (high = bad drop)
worst_drops = drops.nlargest(10, 'ppg_rest_of_season')[
    ['player_name', 'manager', 'week', 'year',
     'ppg_rest_of_season', 'total_points_rest_of_season']
]
```

### Example 4: Manager Transaction Quality
```python
# Calculate average transaction quality score by manager
manager_quality = transaction_df[
    transaction_df['transaction_quality_score'].notna()
].groupby('manager').agg({
    'transaction_quality_score': 'mean',
    'transaction_id': 'count',
    'ppg_rest_of_season': 'mean'
}).round(2)
```

### Example 5: Position-Specific Analysis
```python
# Analyze RB adds
rb_adds = transaction_df[
    (transaction_df['transaction_type'] == 'add') &
    (transaction_df['position'] == 'RB') &
    (transaction_df['ppg_rest_of_season'].notna())
]

# Show RB add stats
rb_stats = rb_adds.agg({
    'ppg_rest_of_season': 'mean',
    'total_points_rest_of_season': 'mean',
    'faab_bid': 'mean',
    'position_rank_after_transaction': 'mean'
}).round(2)
```

## Data Quality Notes

- **Coverage**: ~76% of transactions have enrichment data
- **Missing data**: Transactions are missing enrichment when:
  - Player wasn't rostered/tracked in player data
  - Transaction occurred in final weeks (no "rest of season" data)
  - Player ID couldn't be matched between transactions and player data
- **Best metrics for analysis**:
  - `ppg_rest_of_season` - Most comprehensive performance metric
  - `total_points_rest_of_season` - Total value added/lost
  - `points_per_faab_dollar` - Value efficiency for FAAB adds
  - `transaction_quality_score` - Simple overall quality indicator

## Updating Enrichment Data

Enrichment runs automatically when you execute:
```bash
python initial_import_v2.py --context league_context.json
```

To run only enrichments:
```bash
python initial_import_v2.py --context league_context.json --start-phase 3
```

Or run enrichments standalone:
```bash
cd fantasy_football_data_scripts/multi_league/transformations/transaction_enrichment
python player_to_transactions_v2.py --context path/to/league_context.json
```
