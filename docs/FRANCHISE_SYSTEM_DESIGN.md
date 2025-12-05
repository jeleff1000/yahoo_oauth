# Franchise System Design

## Problem Statement

Fantasy football leagues can have scenarios where career stats get incorrectly conflated:

### Scenario 1: Different People, Same Display Name
- **David** owns "FEAST MODE" (GUID: `4QBBEHQSA7PK...`)
- **David** (different person) owns "Tonya's RagTags" (GUID: `XQTCQ3DI4KNA...`)
- Both show as `manager = "David"` in the data
- Career stats incorrectly combine both people's records

### Scenario 2: Same Person, Multiple Teams
- **Jay Dog** owns "King Ads" AND "GreyWolf" (same GUID: `TPRCFNJMC3QT...`)
- Both teams show as `manager = "Jay Dog"`
- Career stats incorrectly combine both teams' records into one

### Scenario 3: Team Names Change Year-to-Year
- "King Ads" (2024) becomes "Queen Ads" (2025) becomes "Royal Ads" (2026)
- Need to track this as the SAME franchise across years
- Team slot numbers (`.t.X`) can also change between years

## Solution: Franchise Registry

### Core Concepts

| Term | Definition |
|------|------------|
| **franchise_id** | Stable unique identifier for a team lineage. Format: `{guid}_{team_index}` |
| **franchise_name** | Human-readable display name. Updates to most recent team name. |
| **owner_guid** | Yahoo's `manager_guid` - persistent per Yahoo account |
| **owner_name** | Display name from Yahoo (e.g., "David", "Jay Dog") |

### Identifier Stability

The franchise system is **GUID-anchored**. Understanding what's stable vs. what can change:

| Identifier | What It Is | Stability | Example |
|------------|------------|-----------|---------|
| `manager_guid` | Yahoo's internal user ID | **Rock solid** - never changes | `TPRCFNJMC3QT...` |
| `manager` (display name) | Yahoo nickname | Can change anytime | "Jason" → "Jay" |
| `team_name` | Team display name | Can change yearly | "King Ads" → "Queen Ads" |
| `team_key` | Yahoo team identifier | Changes yearly (includes game key) | `449.l.123.t.3` → `461.l.456.t.3` |
| `team_slot` (`.t.X`) | Position in league | Usually stable, can change | `.t.3` |

**Key insight:** If someone changes their Yahoo display name from "Jason" to "Jay":
- `manager_guid` stays the same (unchanged)
- `franchise_id` stays the same (GUID-based)
- `franchise_name` updates to use new name ("Jay" or "Jay - Team Name")
- Career stats remain intact

### Continuity Priority

When linking teams across years, the system checks in this order:

1. **`manager_guid` + `team_name`** (primary) - If same GUID and same team name, it's the same franchise
2. **`manager_guid` + `team_slot`** (secondary) - If team name changed but slot stayed same, likely same franchise
3. **Keeper/draft overlap** (heuristic) - If significant roster continuity, suggest linkage
4. **Manual configuration** (fallback) - User explicitly links teams in config

### Disambiguation Rules

```
franchise_name =
    IF (owner_name has multiple GUIDs) OR (owner_guid has multiple teams):
        "{owner_name} - {current_team_name}"   # e.g., "Jay Dog - King Ads"
    ELSE:
        "{owner_name}"                          # e.g., "Henry"
```

**Note:** `franchise_name` always uses the CURRENT display name and team name. If Jay changes his name to "J-Dawg" and team to "Queen Ads", the franchise_name becomes "J-Dawg - Queen Ads" while the `franchise_id` stays `TPRCFNJ_1`.

### Data Model

#### Franchise Registry (franchise_config.json)
```json
{
  "version": "1.0",
  "franchises": [
    {
      "franchise_id": "TPRCFNJ_1",
      "franchise_name": "Jay Dog - King Ads",
      "owner_guid": "TPRCFNJMC3QTZFZB...",
      "owner_name": "Jay Dog",
      "team_history": [
        {"year": 2024, "team_name": "King Ads", "team_key": "449.l.1068936.t.3"},
        {"year": 2025, "team_name": "King Ads", "team_key": "461.l.836761.t.3"},
        {"year": 2026, "team_name": "Queen Ads", "team_key": "473.l.xxxxx.t.3"}
      ]
    },
    {
      "franchise_id": "TPRCFNJ_2",
      "franchise_name": "Jay Dog - GreyWolf",
      "owner_guid": "TPRCFNJMC3QTZFZB...",
      "owner_name": "Jay Dog",
      "team_history": [
        {"year": 2024, "team_name": "GreyWolf", "team_key": "449.l.1068936.t.11"},
        {"year": 2025, "team_name": "GreyWolf", "team_key": "461.l.836761.t.9"}
      ]
    }
  ]
}
```

#### Columns Added to Data Tables
| Column | Type | Description |
|--------|------|-------------|
| `franchise_id` | string | Stable identifier for grouping career stats |
| `franchise_name` | string | Display name (updates to current team name) |

## Cross-Year Linkage Strategy

When team names change between years, we need to determine which Year N+1 team corresponds to which Year N team.

### Primary Method: Team Key Correlation
1. Draft data contains `team_key` (e.g., `461.l.836761.t.3`)
2. Extract team slot (`.t.3`) as a continuity hint
3. If same `(manager_guid, team_slot)` exists in consecutive years, likely same franchise

### Secondary Method: Keeper/Draft Heuristics
When team slots change (e.g., `.t.11` in 2024 → `.t.9` in 2025):

1. Look at keeper players: if "Queen Ads" (2026) kept players from "King Ads" (2025), they're the same franchise
2. Look at draft picks: if significant roster overlap between years, likely same franchise
3. Calculate a "continuity score" based on player overlap

### Fallback: Manual Configuration
When automatic methods fail:
1. System flags "orphan" teams that couldn't be linked
2. User manually updates `franchise_config.json` to specify linkages
3. Config is preserved across future imports

## Implementation Plan

### Phase 1: Core Module
**File:** `multi_league/core/franchise_registry.py`

```python
class TeamSeason:
    year: int
    team_name: str
    team_key: str
    team_slot: int
    wins: int
    losses: int
    points_for: float

class Franchise:
    franchise_id: str
    franchise_name: str
    owner_guid: str
    owner_name: str
    team_history: List[TeamSeason]

class FranchiseRegistry:
    @classmethod
    def from_data(cls, matchup_df, draft_df) -> FranchiseRegistry

    def get_franchise_id(self, manager_guid, team_name, year) -> str
    def apply_franchise_columns(self, df) -> pd.DataFrame
    def detect_orphan_teams(self) -> List[dict]
    def save(self, path: Path)
    def load(cls, path: Path) -> FranchiseRegistry
```

### Phase 2: Transformation Script
**File:** `multi_league/transformations/matchup/discover_franchises.py`

Pipeline position: AFTER `resolve_hidden_managers.py`, BEFORE `cumulative_stats.py`

```python
TRANSFORMATIONS_PASS_1 = [
    ("resolve_hidden_managers.py", ...),    # 1st
    ("discover_franchises.py", ...),        # 2nd - NEW
    ("cumulative_stats.py", ...),           # 3rd - MODIFIED
    ...
]
```

**Logic:**
1. Load existing `franchise_config.json` if present
2. Scan matchup + draft data for all `(manager_guid, team_name, year, team_key)` combinations
3. For each manager_guid:
   - If single team across all years → single franchise
   - If multiple teams → create sub-franchises, attempt cross-year linking
4. Apply disambiguation rules to generate `franchise_name`
5. Add `franchise_id` and `franchise_name` columns to matchup.parquet
6. Save updated `franchise_config.json`
7. Report any orphan teams needing manual linking

### Phase 3: Data Fetcher Update
**File:** `multi_league/data_fetchers/weekly_matchup_data_v2.py`

Add `team_key` to output columns so we can correlate with draft data.

### Phase 4: Cumulative Stats Update
**File:** `multi_league/transformations/matchup/cumulative_stats.py`

Change grouping from `manager` to `franchise_id`:

```python
# Before
.groupby('manager', dropna=False)

# After
.groupby('franchise_id', dropna=False)
```

### Phase 5: Keeper/Draft Heuristics
**File:** `multi_league/core/franchise_registry.py` (extend)

Add methods for cross-year linking:
```python
def calculate_roster_continuity(self, year1_team, year2_team) -> float:
    """
    Calculate overlap between two teams based on:
    - Keeper players
    - Draft picks that were on previous roster
    Returns score 0.0 to 1.0
    """

def suggest_linkages(self) -> List[dict]:
    """
    For orphan teams, suggest likely linkages based on:
    - Roster continuity scores
    - Team name similarity
    - Team slot patterns
    """
```

## UI Impact

### Standings Display

**Before (Problem):**
| Seed | Manager | Team | W | L |
|------|---------|------|---|---|
| 1 | David | FEAST MODE | 16 | 10 |
| 5 | David | Tonya's RagTags | 8 | 18 |

**After (Solution):**
| Seed | Manager | Team | W | L |
|------|---------|------|---|---|
| 1 | David - FEAST MODE | FEAST MODE | 16 | 10 |
| 5 | David - RagTags | Tonya's RagTags | 8 | 18 |

**No Disambiguation Needed:**
| Seed | Manager | Team | W | L |
|------|---------|------|---|---|
| 2 | Henry | Soccer player Henry | 13 | 9 |

### Career Stats Display

Career stats grouped by `franchise_id`, displayed with current `franchise_name`:

| Franchise | Years | Career W | Career L | Win % |
|-----------|-------|----------|----------|-------|
| Jay Dog - Queen Ads | 3 | 45 | 40 | 52.9% |
| Jay Dog - WhiteWolf | 3 | 38 | 47 | 44.7% |
| Henry | 2 | 26 | 18 | 59.1% |

## Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `multi_league/core/franchise_registry.py` | Core franchise logic |
| `multi_league/transformations/matchup/discover_franchises.py` | Pipeline transformation |
| `{data_dir}/franchise_config.json` | Per-league franchise configuration |

### Modified Files
| File | Changes |
|------|---------|
| `multi_league/data_fetchers/weekly_matchup_data_v2.py` | Add `team_key` to output |
| `multi_league/transformations/matchup/cumulative_stats.py` | Group by `franchise_id` |
| `multi_league/initial_import_v2.py` | Add `discover_franchises.py` to pipeline |

## Testing

### Test Case 1: David (Different People, Same Name)
- Input: Two GUIDs, both with `manager = "David"`
- Expected: Two separate franchises with disambiguated names
- Verify: Career stats calculated separately

### Test Case 2: Jay Dog (Same Person, Multiple Teams)
- Input: One GUID with two different team names
- Expected: Two separate sub-franchises
- Verify: Each team's career tracked independently

### Test Case 3: Name Change Across Years
- Input: "King Ads" (2024) → "Queen Ads" (2025), same team_slot
- Expected: Single franchise, `franchise_name` updates to "Queen Ads"
- Verify: Career stats span both years

### Test Case 4: Slot Change + Name Change
- Input: `.t.11` + "GreyWolf" (2024) → `.t.9` + "WhiteWolf" (2025)
- Expected: System flags for manual review OR uses keeper heuristics
- Verify: User can manually link in config

## Open Questions

1. **Orphan handling:** When auto-linking fails, should we:
   - Create a new franchise (potentially splitting career stats)?
   - Block import until user resolves?
   - Use best-guess and flag for review?

2. **Config migration:** When user manually edits `franchise_config.json`, how do we preserve their changes when new data arrives?

3. **UI for config editing:** Should we build a Streamlit UI for managing franchise linkages, or is JSON editing sufficient?
