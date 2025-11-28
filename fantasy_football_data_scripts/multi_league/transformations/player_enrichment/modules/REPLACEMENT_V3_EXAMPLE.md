# Replacement Calculator V3 - Example Walkthrough

## Your Week 10, 2025 QB Data

**Total: 38 QBs played**
- **16 rostered** (some started, some benched)
- **22 unrostered** (all available as free agents)

## V3 Strategy: Roster-Agnostic Ranking

### Step 1: Filter to Started Players Only

With `count_started_only=True`, we **exclude** rostered players who were benched:

**Started (10 QBs):**
1. Jaxson Dart: 26.28 PPG (rostered, started)
2. Jared Goff: 24.8 PPG (rostered, started)
3. Josh Allen: 19.34 PPG (rostered, started)
4. Drake Maye: 18.1 PPG (rostered, started)
5. Lamar Jackson: 16.64 PPG (rostered, started)
6. Daniel Jones: 15.5 PPG (rostered, started)
7. Justin Herbert: 14.7 PPG (rostered, started)
8. Jalen Hurts: 12.02 PPG (rostered, started)
9. Bo Nix: 5.8 PPG (rostered, started)
10. Sam Darnold: 4.92 PPG (rostered, started)

**Benched (6 QBs) - EXCLUDED:**
- Matthew Stafford: 26.9 PPG (BN) ❌ Excluded
- Jordan Love: 7.84 PPG (BN) ❌ Excluded
- Dak Prescott: 0 PPG (BN, bye/injured) ❌ Excluded
- Patrick Mahomes: 0 PPG (BN, bye/injured) ❌ Excluded
- Joe Burrow: 0 PPG (BN, bye/injured) ❌ Excluded
- Jayden Daniels: 0 PPG (IR) ❌ Excluded

**Unrostered (22 QBs) - ALL INCLUDED:**
- Davis Mills: 27.68 PPG ✅ Included
- Caleb Williams: 25.1 PPG ✅ Included
- Mac Jones: 23.06 PPG ✅ Included
- Baker Mayfield: 22.92 PPG ✅ Included
- Dillon Gabriel: 20.08 PPG ✅ Included
- Jacoby Brissett: 19.42 PPG ✅ Included
- Marcus Mariota: 18.72 PPG ✅ Included
- Tyler Shough: 18.98 PPG ✅ Included
- Jj Mccarthy: 14.72 PPG ✅ Included
- Michael Penix: 10.98 PPG ✅ Included
- Tua Tagovailoa: 10.82 PPG ✅ Included
- Trevor Lawrence: 9.82 PPG ✅ Included
- Justin Fields: 6.96 PPG ✅ Included
- Geno Smith: 4.42 PPG ✅ Included
- Bryce Young: 3.56 PPG ✅ Included
- Aaron Rodgers: 6.44 PPG ✅ Included
- Russell Wilson: 3 PPG ✅ Included
- Mitchell Trubisky: 0.4 PPG ✅ Included
- Drew Lock: -0.4 PPG ✅ Included
- Kyle Allen: -0.1 PPG ✅ Included
- Kedon Slovis: 0 PPG ✅ Included
- Kenny Pickett: 0 PPG ✅ Included

### Step 2: Sort by Fantasy Points (ALL Players)

**Combined ranking (32 total QBs after filtering):**

| Rank | Player | PPG | Rostered? |
|------|--------|-----|-----------|
| 1 | Davis Mills | 27.68 | **No (FA)** |
| 2 | Jaxson Dart | 26.28 | Yes |
| 3 | Caleb Williams | 25.1 | **No (FA)** |
| 4 | Jared Goff | 24.8 | Yes |
| 5 | Mac Jones | 23.06 | **No (FA)** |
| 6 | Baker Mayfield | 22.92 | **No (FA)** |
| 7 | Dillon Gabriel | 20.08 | **No (FA)** |
| 8 | Josh Allen | 19.34 | Yes |
| 9 | Jacoby Brissett | 19.42 | **No (FA)** |
| 10 | Tyler Shough | 18.98 | **No (FA)** |
| 11 | Marcus Mariota | 18.72 | **No (FA)** |
| 12 | Drake Maye | 18.1 | Yes |
| 13 | Lamar Jackson | 16.64 | Yes |
| 14 | Daniel Jones | 15.5 | Yes |
| 15 | Justin Herbert | 14.7 | Yes |
| 16 | Jj Mccarthy | 14.72 | **No (FA)** |
| 17 | Jalen Hurts | 12.02 | Yes |
| **18** | **Michael Penix** | **10.98** | **No (FA)** ← **REPLACEMENT** |
| 19 | Tua Tagovailoa | 10.82 | **No (FA)** |
| ... | ... | ... | ... |

### Step 3: Calculate Replacement Level

**Roster capacity estimate:** 10 teams × 1.8 QBs/team = **18 roster spots**

**Replacement = 18th-ranked QB:**
- Michael Penix: 10.98 PPG (unrostered free agent)
- Replacement baseline: **10.98 PPG**

**Note:** In reality, only 16 QBs are rostered in your league (not 18), but replacement is based on **capacity**, not actual roster status. The 18th-best QB by performance is the baseline.

### Step 4: Calculate SPAR for Each Player

**Example: Josh Allen (ranked 8th)**
```
Season stats: 19.34 PPG × 10 weeks started = 193.4 total points
Replacement: 10.98 PPG × 10 weeks = 109.8 replacement points
SPAR = 193.4 - 109.8 = 83.6 points above replacement
```

**Example: Caleb Williams (ranked 3rd, unrostered FA)**
```
Season stats: 25.1 PPG × 1 week started = 25.1 total points
Replacement: 10.98 PPG × 1 week = 10.98 replacement points
SPAR = 25.1 - 10.98 = 14.12 points above replacement
```
**Caleb Williams has positive SPAR even though he's a free agent!** This shows he's rosterable.

**Example: Mitchell Trubisky (ranked 28th, unrostered FA)**
```
Season stats: 0.4 PPG × 1 week started = 0.4 total points
Replacement: 10.98 PPG × 1 week = 10.98 replacement points
SPAR = 0.4 - 10.98 = -10.58 points below replacement
```
**Trubisky has negative SPAR** - correctly shows he's not rosterable (below replacement).

---

## Why This Works Better

### Old V2 Approach: Filter by Roster Status
```python
# V2: Only rank rostered QBs
rostered_qbs = week_data[week_data['manager'] != 'Unrostered']
```

**Problem:** What if the 7 best QBs weren't rostered that week?
- Misses unrostered performers like Davis Mills (27.68 PPG)
- Replacement would be based on incomplete pool

### New V3 Approach: Rank All QBs
```python
# V3: Rank ALL QBs by performance (rostered or not)
all_qbs = week_data[week_data['position'] == 'QB']
```

**Better:** Captures true performance distribution
- Davis Mills (27.68 PPG) ranks #1 even though he's a FA
- Replacement is 18th-best QB **by performance**, not roster status
- More accurate representation of position scarcity

---

## Key Differences from V1 and V2

| Feature | V1 (Old) | V2 (Middle) | V3 (New) |
|---------|----------|-------------|----------|
| **Baseline** | Teams × starters (10) | Teams × roster_spots (18) | Teams × roster_spots (18) |
| **Pool** | fantasy_points > 0 | Rostered only | **ALL players** |
| **Filter** | Any points scored | Any points scored | **Started only** |
| **Example** | 11th QB (first bench) | 18th rostered QB | **18th-ranked QB** |

**V3 advantages:**
- ✅ Roster-agnostic (ranks all players by performance)
- ✅ Started-only filtering (counts actual usage weeks)
- ✅ Simpler logic (no roster status filtering)
- ✅ More accurate (captures true position scarcity)

---

## Calibration: Validating Roster Spots Estimates

Use `analyze_actual_roster_depth()` to check if your 1.8 QB estimate is accurate:

```python
from replacement_calculator_v3 import analyze_actual_roster_depth

# Analyze Week 10, 2025
depth = analyze_actual_roster_depth(player_df, year=2025, week=10)
print(depth)

# Output:
#   position  rostered_count  started_count  rostered_per_team  num_teams
# 0       QB              16             10               1.6         10
# 1       RB              52             23               5.2         10
# 2       WR              56             26               5.6         10
```

**Your actual data shows:**
- 1.6 QBs rostered per team (not 1.8)
- You could override: `roster_spots_override={'QB': 1.6}`

**But 1.8 is still reasonable** because:
- It's the **capacity** estimate (how many QBs are rosterable)
- Some teams roster 1, some roster 2 → average ~1.6-1.8
- Using 1.8 gives you the 18th-best QB as replacement (conservative)

---

## Final Recommendation

```python
from replacement_calculator_v3 import calculate_all_replacements

weekly, season = calculate_all_replacements(
    player_df,
    league_settings_path,
    count_started_only=True,  # Only count weeks actually started
    roster_spots_override=None  # Use defaults, or override specific positions
)
```

This gives you:
- **Roster-agnostic** replacement (ranks all players by performance)
- **Started-weeks-only** filtering (counts actual usage)
- **Capacity-based** cutoff (18th QB, not dependent on who's actually rostered)
