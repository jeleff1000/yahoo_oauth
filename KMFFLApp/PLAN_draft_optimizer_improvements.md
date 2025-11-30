# Draft Optimizer Improvement Plan

## Overview
Enhance the draft optimizer with better UX, auto-configuration, strategy presets, and bench handling.

## Status: IMPLEMENTED ✅

The following improvements have been implemented in `draft_optimizer.py`:

---

## Phase 1: Bug Fixes & Quick Wins ✅

### 1.1 Fix FLEX Constraint Logic ✅
**Changes Made:**
- Clearer constraint logic with documented explanation
- Non-flex positions (QB, DEF, K) have exact counts
- Flex-eligible positions (RB, WR, TE) have minimum requirements with room for FLEX
- Total flex-eligible constraint ensures exactly `dedicated + FLEX` slots filled
- Results now show which players are in FLEX slots with `(FLEX)` designation

### 1.2 Change Display from SPAR to Points ✅
**Changes Made:**
- Big hero metric showing projected season points (PPG * 14 weeks)
- Lineup table now includes "Season Pts" column
- Summary shows Weekly PPG prominently
- Optimization objective changed to maximize PPG (what users want)
- Updated info banners to reference points instead of SPAR

---

## Phase 2: Auto-Configuration ✅

### 2.1 Auto-Detect Roster Config from Player Data ✅
**Implementation:**
- Added `detect_roster_config()` function
- Queries player table for `lineup_position` column from most recent season week 1
- Parses positions like "QB1", "RB1", "WR2", "W/R/T1" to count slots
- Automatically adds "Your League" preset option if detection succeeds
- Falls back to "Standard" preset if detection fails

---

## Phase 3: Budget Allocation Visualization ✅

### 3.1 Budget Breakdown Charts ✅
**Added:**
- Pie chart showing spending by position (includes bench budget slice)
- Bar chart comparing cost % vs points % contribution by position
- Both charts help users understand allocation efficiency

---

## Phase 4: Strategy Presets ✅

### 4.1 Strategy Templates Implemented ✅
**Available strategies:**
- **Balanced** - No spending restrictions
- **Zero RB** - Cheap RBs ($20 max), heavy WR investment
- **Hero RB** - One elite RB ($65), cheap rest
- **Robust RB** - Two solid RBs ($55, $45)
- **Late-Round QB** - QB capped at $12
- **Stars & Scrubs** - Few expensive, rest cheap

Each strategy shows a description when selected.

---

## Phase 5: Bench Handling ✅

### 5.1 Bench Depth Presets ✅
**Implemented options:**
- **Shallow (4-5 spots)** - 10% budget, $3-4 per slot
- **Standard (6-7 spots)** - 15% budget, $4-5 per slot
- **Deep (8+ spots)** - 20% budget, $4-5 per slot with upside room

### 5.2 Bench Recommendations Display ✅
**Results show:**
- Remaining bench budget and average per slot
- Strategy tips (handcuffs, rookies, QB backup, TE streaming)
- Suggested bench composition table with position/spots/budget/strategy

---

## Phase 6: Lock-In Feature 🔜 (Future Enhancement)

### 6.1 Position + Price Lock-In
**Not yet implemented** - planned for future:
- Checkbox per position to lock in spend
- Slider for locked amount
- Optimizer treats as fixed constraint

---

## Summary of Changes

| Feature | Status | File Location |
|---------|--------|---------------|
| FLEX constraint fix | ✅ | `run_optimization()` |
| Points display | ✅ | `display_optimization_results()` |
| Auto-detect roster | ✅ | `detect_roster_config()` |
| "Your League" preset | ✅ | `get_roster_preset()` |
| Budget visualization | ✅ | `display_optimization_results()` |
| Strategy presets | ✅ | `get_strategy_preset()` |
| Bench planning UI | ✅ | `display_draft_optimizer()` |
| Bench recommendations | ✅ | `display_optimization_results()` |

---

## Questions Resolved

1. **Roster detection:** Uses `lineup_position` column from player table (e.g., "QB1", "RB1", "W/R/T1")

2. **Bench optimization:** Using heuristics-based presets rather than full optimization

3. **Strategy comparison:** Showing one strategy at a time with descriptions

4. **Points vs SPAR:** Switched optimization objective to PPG - this is what users want to maximize
