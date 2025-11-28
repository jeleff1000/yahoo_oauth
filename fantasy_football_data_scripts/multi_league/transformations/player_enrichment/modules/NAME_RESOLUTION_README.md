# Player Name Resolution

## Problem

After merging Yahoo and NFL data sources, player names can be inconsistent:

- **Yahoo data** (rostered players): Uses accurate names like "Tom Brady"
- **NFL data** (all players): May use variants like "Thomas Brady" or "T. Brady"
- **Historical data**: Players rostered 2014+ have their pre-2014 stats under different names

This makes it hard to track a player's full career history under one consistent name.

## Solution

We implement a **two-phase name restoration strategy**:

### Phase 1: Preserve Original Names (During Merge)
**File:** `yahoo_nfl_merge.py`

1. **Before cleaning** (line 511): Store `_original_player_name` for both Yahoo and NFL sources
2. **During merge** (line 721): For merged rows, prefer Yahoo's original name
3. **After merge** (line 1496): Restore original names from `_original_player_name`

### Phase 2: Canonicalize Across All Years (After Merge)
**File:** `yahoo_nfl_merge.py` (lines 1501-1537)

For each `NFL_player_id`:
1. Check if player was EVER rostered (has `yahoo_player_id` in any year)
2. If yes, use that Yahoo name for ALL rows with that `NFL_player_id`
3. This applies Yahoo names to historical unrostered rows too

**Example:**
```
Tom Brady NFL_player_id: 00-0019596
- 1999-2013: Unrostered NFL data → originally "Thomas Brady"
- 2014-2020: Rostered Yahoo data → "Tom Brady"

After canonicalization:
- 1999-2020: ALL rows use "Tom Brady" (Yahoo name applied to historical data)
```

## Module Functions

**File:** `modules/name_resolver.py`

### `canonicalize_names_by_id(df, add_debug_cols=False)`
Apply Yahoo names to ALL rows with the same NFL_player_id (including historical years).

```python
from name_resolver import canonicalize_names_by_id

# Canonicalize names (Yahoo preferred, NFL fallback)
df = canonicalize_names_by_id(df)

# With debugging columns
df = canonicalize_names_by_id(df, add_debug_cols=True)
# Adds: player_source (yahoo/nfl), player_variations (all name variants)
```

### `add_name_variations_column(df)`
Add a column showing all name variations for each player (for debugging).

```python
from name_resolver import add_name_variations_column

df = add_name_variations_column(df)
# Adds: player_name_variations = "Tom Brady|Thomas Brady|T. Brady"
```

### `get_name_history(df, nfl_player_id="12345")`
Get complete name history for a specific player across all years.

```python
from name_resolver import get_name_history

# Look up by NFL player ID
history = get_name_history(df, nfl_player_id="00-0019596")

# Look up by Yahoo player ID
history = get_name_history(df, yahoo_player_id="9876")
```

## How It Works

### Data Flow

```
BEFORE MERGE:
  Yahoo: "Tom Brady" → cleaned to "tom brady" for matching
  NFL:   "Thomas Brady" → cleaned to "thomas brady" for matching

MERGE PROCESS:
  1. Store originals: _original_player_name
  2. Match on cleaned keys: "brady" + year + week
  3. Merge succeeds → prefer Yahoo's _original_player_name

AFTER MERGE:
  Merged rows: "Tom Brady" (from Yahoo) ✅
  Yahoo-only: "Tom Brady" ✅
  NFL-only: "Thomas Brady" ❌ (still using NFL name)

CANONICALIZATION (NEW):
  1. Find NFL_player_id "00-0019596" appears with yahoo_player_id → Yahoo name = "Tom Brady"
  2. Apply "Tom Brady" to ALL rows with NFL_player_id "00-0019596"
  3. Result: All 1999-2020 rows use "Tom Brady" ✅
```

### Preference Order

1. **Yahoo name** (if player has yahoo_player_id in ANY year) ← PREFERRED
2. **NFL name** (if player never rostered) ← FALLBACK

## Benefits

✅ **Consistent naming**: Same player = same name across all years
✅ **Historical continuity**: Pre-league stats use same names as rostered years
✅ **Fewer merge conflicts**: Better name matching prevents duplicates
✅ **Better player tracking**: Easy to see full career under one name
✅ **Debugging support**: Track all name variations for conflict resolution

## Integration

### In yahoo_nfl_merge.py (Already Integrated)
Canonicalization happens automatically at line 1501 after the merge completes.

### In player_stats_v2.py (Optional)
Can add debugging columns during transformation:

```python
from name_resolver import add_name_variations_column

# Add name variations for debugging (optional)
df = add_name_variations_column(df)
```

### In player_enrichment transformations (Optional)
Can re-canonicalize if needed:

```python
from name_resolver import canonicalize_names_by_id

# Ensure names are canonical
df = canonicalize_names_by_id(df)
```

## Testing

To verify name resolution is working:

```python
import pandas as pd

# Load merged data
df = pd.read_parquet("yahoo_nfl_merged_2024_all_weeks.parquet")

# Check a specific player across years
player_id = "00-0019596"  # Tom Brady
player_history = df[df["NFL_player_id"] == player_id][["year", "player", "yahoo_player_id", "manager"]]

print("Tom Brady name history:")
print(player_history.groupby("year")["player"].first())
# All years should show "Tom Brady", not "Thomas Brady"
```

## Troubleshooting

**Problem:** Player has multiple names in different years

**Solution:**
1. Check if player has yahoo_player_id in ANY year
2. If yes, Yahoo name should be applied to all years
3. If not working, check NFL_player_id is consistent across years
4. Use `get_name_history()` to debug name variations

**Problem:** Want to see all name variations for a player

**Solution:**
```python
from name_resolver import add_name_variations_column, get_name_history

# Add variations column to entire dataset
df = add_name_variations_column(df)

# Or check specific player
history = get_name_history(df, nfl_player_id="00-0019596")
```
