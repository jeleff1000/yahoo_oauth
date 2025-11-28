# Replacement Level Strategy Guide

## The Two Questions You Asked

### Question 1: Should we count weeks started or weeks rostered?

**Current behavior** (`replacement_calculator.py`):
```python
# Counts any player who scored fantasy_points > 0
week_data = player_df[player_df['fantasy_points'] > 0]
```

❌ **Problem**: This counts bench players who scored points even if never started.

**V2 solution** (`replacement_calculator_v2.py`):
```python
# Option A: Count only started weeks (recommended)
count_started_only=True  # Filters to fantasy_position not in ['BN', 'IR']

# Option B: Count any week with points (current behavior)
count_started_only=False
```

### Question 2: Is replacement the first non-starter or first non-rostered?

**Current behavior** (`replacement_calculator.py`):
```python
n_pos = num_teams * starter_count  # e.g., 10 teams × 1 QB = 10
# Replacement = 11th QB (first bench QB)
```

**V2 solution** (`replacement_calculator_v2.py`):
```python
# Option A: First non-starter (current - traditional WAR)
strategy="starter"
n_pos = num_teams * starter_count  # 10 teams × 1 QB = 10

# Option B: First non-rostered (recommended - waiver wire based)
strategy="rostered"
n_pos = num_teams * roster_spots  # 10 teams × 1.8 QBs = 18
```

---

## Strategy Comparison

### Strategy 1: "First Non-Starter" (Current)

**Definition**: Replacement = teams × starters (e.g., 10th QB if 10 teams start 1 QB)

**Example League (10 teams, 1 QB starter, 2 QBs rostered per team):**
```
QB Rankings:
1. Josh Allen (25 PPG) - SPAR: +10 (above replacement)
2. Jalen Hurts (23 PPG) - SPAR: +8
...
10. Tua (15 PPG) - SPAR: +0 (replacement level)
11. Cousins (14 PPG) - SPAR: -1 (NEGATIVE! bench player)
...
18. Mariota (10 PPG) - SPAR: -5 (very negative, backup)
19. Trubisky (8 PPG) - SPAR: -7 (free agent)
```

**Use cases:**
- ✅ Comparing starter quality ("Is my QB1 worth starting over my QB2?")
- ✅ Traditional WAR analysis (baseball-style)
- ❌ Bench players have confusing **negative SPAR**
- ❌ Doesn't reflect waiver wire decisions

---

### Strategy 2: "First Non-Rostered" (Recommended)

**Definition**: Replacement = teams × roster_spots (e.g., 18th QB if 10 teams roster ~2 QBs)

**Example League (10 teams, 1 QB starter, 2 QBs rostered per team):**
```
QB Rankings:
1. Josh Allen (25 PPG) - SPAR: +15 (elite)
2. Jalen Hurts (23 PPG) - SPAR: +13
...
10. Tua (15 PPG) - SPAR: +5 (QB1 on a roster)
11. Cousins (14 PPG) - SPAR: +4 (QB2 on a roster)
...
18. Mariota (10 PPG) - SPAR: +0 (replacement level = best FA)
19. Trubisky (8 PPG) - SPAR: -2 (below replacement FA)
```

**Use cases:**
- ✅ "Should I roster this QB or leave on waivers?" ← **KEY DECISION**
- ✅ All rostered players have zero/positive value (intuitive)
- ✅ Transaction analysis (waiver pickups, drops, trades)
- ✅ Matches manager experience (bench value vs FA value)
- ❌ Need to estimate roster spots per position

---

## Recommended Settings for Fantasy Football

```python
from replacement_calculator_v2 import calculate_all_replacements

weekly, season = calculate_all_replacements(
    player_df,
    league_settings_path,
    strategy="rostered",        # Use waiver wire as baseline (19th QB, not 11th)
    count_started_only=True     # Only count weeks player was started
)
```

**Why these settings?**

1. **`strategy="rostered"`**:
   - Reflects actual scarcity (what's on waivers vs what's rostered)
   - Better for transaction decisions
   - All rostered players have positive/zero value

2. **`count_started_only=True`**:
   - Only counts weeks where `fantasy_position != BN/IR`
   - More accurate representation of player usage
   - Prevents bench performances from inflating replacement level

---

## Real-World Example

**Your league:**
- 10 teams
- QB roster: 1 starter, ~1.8 avg rostered (some teams 2, some 1)
- 18 QBs rostered league-wide

**Week 10 QB performances:**

| Rank | Player | PPG | Rostered? | Started? |
|------|--------|-----|-----------|----------|
| 1 | Josh Allen | 28 | Yes (Manager A) | Yes |
| 10 | Tua | 15 | Yes (Manager B) | Yes |
| 11 | Cousins | 14 | Yes (Manager C) | **No (BN)** |
| 18 | Mariota | 10 | Yes (Manager D) | **No (BN)** |
| 19 | Trubisky | 8 | **No (Free Agent)** | No |

**Strategy="starter", count_started_only=False** (Current):
- Replacement = 10th QB (Tua, 15 PPG)
- Cousins SPAR = negative (he's a bench player below "starter replacement")
- **Problem**: Your backup QB has negative value even though he's better than the best FA!

**Strategy="rostered", count_started_only=True** (Recommended):
- Replacement = 19th QB (Trubisky, 8 PPG)
- Cousins SPAR = positive (he's rosterable, above waiver wire)
- **Better**: Your backup QB has positive value because he's better than free agents

---

## How to Estimate Roster Spots

V2 uses these estimates (in `load_roster_structure`):

```python
position_roster_estimates = {
    'QB': 1.8,    # Most teams roster 2 QBs
    'RB': 5.0,    # 2-3 starters + 2-3 bench
    'WR': 5.5,    # 2-3 starters + 2-3 bench
    'TE': 1.5,    # 1 starter + maybe 1 bench
    'K': 1.2,     # 1 starter + occasional bench
    'DEF': 1.2    # 1 starter + occasional bench
}
```

**For more accuracy**, use the `analyze_roster_depth()` function:

```python
from replacement_calculator_v2 import analyze_roster_depth

# Check actual roster depth for a specific week
depth = analyze_roster_depth(player_df, year=2024, week=10)
print(depth)

# Output:
#   position  rostered_count  started_count
# 0       QB              18             10
# 1       RB              52             23
# 2       WR              56             26
```

This shows you the **actual** roster composition in your league, which you can use to refine estimates.

---

## Migration Path

### Option 1: Update Existing Calculator
Replace `replacement_calculator.py` with v2 and update all imports.

### Option 2: Side-by-Side Comparison
Keep both versions and compare results:

```python
# Old method
from replacement_calculator import calculate_all_replacements as calc_v1

# New method
from replacement_calculator_v2 import calculate_all_replacements as calc_v2

weekly_v1, season_v1 = calc_v1(player_df, settings_path)
weekly_v2, season_v2 = calc_v2(
    player_df, settings_path,
    strategy="rostered",
    count_started_only=True
)

# Compare replacement levels
comparison = season_v1.merge(
    season_v2,
    on=['year', 'position'],
    suffixes=('_v1_starter', '_v2_rostered')
)
print(comparison)
```

---

## Summary: Which Replacement Level Should You Use?

| Use Case | Strategy | Count Method | Example |
|----------|----------|--------------|---------|
| **Waiver/roster decisions** | `rostered` | `started_only` | Should I add Cousins? (vs best FA) |
| **Starter quality comparison** | `starter` | `started_only` | Is Tua better than my QB2? |
| **Traditional WAR analysis** | `starter` | `scored_points` | Baseball-style value analysis |
| **Transaction ROI** | `rostered` | `started_only` | Was this pickup worth the FAAB? |

**Recommendation for fantasy football:** `strategy="rostered"` + `count_started_only=True`

This gives you the most intuitive and actionable SPAR values for fantasy decisions.
