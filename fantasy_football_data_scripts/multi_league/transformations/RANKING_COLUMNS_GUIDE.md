# Player Rankings Column Guide

## Complete Ranking System Overview

This document describes all ranking columns available in the player stats data.

---

## ✅ MANAGER-BASED RANKINGS (Fantasy League Specific)

### Cross-Position Rankings (NEW!)
**Compare any player to any other player for a manager**

| Column | Description | Example |
|--------|-------------|---------|
| `manager_all_player_all_time_history` | Rank among ALL players this manager has ever rostered (all positions) | "Josh Allen's 52-pt game was Daniel's #1 performance ever (beats Kamara's 50-pt game)" |
| `manager_all_player_all_time_history_percentile` | Percentile ranking | "This was in the 99th percentile of all performances for Daniel" |
| `manager_all_player_season_history` | Rank among ALL players this manager rostered this season (all positions) | "This was Daniel's #3 best player performance in 2024" |
| `manager_all_player_season_history_percentile` | Percentile ranking | "This was in the 95th percentile for Daniel's 2024 season" |

### Player-Specific Rankings
**How a specific player performed for a specific manager**

| Column | Description | Example |
|--------|-------------|---------|
| `manager_player_all_time_history` | Rank of this week among all weeks this player was on this manager's team | "This was Lamar Jackson's 3rd best week when rostered by Daniel" |
| `manager_player_all_time_history_percentile` | Percentile ranking | "This was in the 92nd percentile of Lamar's performances for Daniel" |
| `manager_player_season_history` | Rank among weeks in this season with this manager | "Lamar's #1 week for Daniel in 2024" |
| `manager_player_season_history_percentile` | Percentile ranking | "99th percentile for this player/manager combo this season" |

### Position-Specific Rankings
**How a position performed for a specific manager**

| Column | Description | Example |
|--------|-------------|---------|
| `manager_position_all_time_history` | Rank among all weeks with this manager-position combo | "Among all QBs Daniel has ever started, this was the 23rd best performance" |
| `manager_position_all_time_history_percentile` | Percentile ranking | "87th percentile among all QB performances for Daniel" |
| `manager_position_season_history` | Rank within this season for this manager-position combo | "Daniel's #2 QB performance in 2024" |
| `manager_position_season_history_percentile` | Percentile ranking | "95th percentile among Daniel's 2024 QB performances" |

---

## ✅ NFL-WIDE RANKINGS (All Players Ever)

### Player Personal Rankings
**Individual player's career performance (NFL player ID based)**

| Column | Description | Example |
|--------|-------------|---------|
| `player_personal_all_time_history` | Rank among all games this player has ever played (NFL career) | "This was Lamar Jackson's 5th best game of his NFL career" |
| `player_personal_all_time_history_percentile` | Percentile ranking | "98th percentile among all Lamar's games" |
| `player_personal_season_history` | Rank among games in this player's season | "Lamar's #2 best game in 2024" |
| `player_personal_season_history_percentile` | Percentile ranking | "96th percentile for Lamar's 2024 season" |

### Position Rankings (Game-Level)
**All players at a position compared (includes pre-league history)**

| Column | Description | Example |
|--------|-------------|---------|
| `position_all_time_history` | Rank among ALL performances at this position (all-time) | "Clinton Portis's 55-pt game was the 12th best RB game ever" |
| `position_all_time_history_percentile` | Percentile ranking (excludes zeros) | "99.8th percentile among all RB performances" |
| `position_season_history` | Rank among all performances at this position this season | "The 3rd best QB performance in 2024" |
| `position_season_history_percentile` | Percentile ranking (excludes zeros) | "99.2nd percentile among 2024 QB performances" |

---

## 📊 USE CASE MATRIX

| Question | Column to Use |
|----------|---------------|
| "What's the best game ANY player has given Daniel?" | `manager_all_player_all_time_history` |
| "How does Josh Allen's 52-pt game compare to Kamara's 50-pt game for Daniel?" | `manager_all_player_all_time_history` |
| "What's the best QB game in NFL history?" | `position_all_time_history` WHERE position='QB' |
| "How does this compare to Clinton Portis's legendary games?" | `position_all_time_history` (includes pre-league data) |
| "Is this Lamar's best game ever with Daniel?" | `manager_player_all_time_history` |
| "Is this Lamar's best game of his career?" | `player_personal_all_time_history` |
| "What's the best QB performance Daniel has ever gotten?" | `manager_position_all_time_history` WHERE position='QB' |
| "How does this week rank for Daniel this season?" | `manager_all_player_season_history` |

---

## 🎯 COMPLETE RANKING COVERAGE

### Manager-Based (Fantasy League)
- ✅ All players (cross-position) - all-time & season
- ✅ Specific player - all-time & season
- ✅ Specific position - all-time & season

### NFL-Wide (All History)
- ✅ Individual player career - all-time & season
- ✅ Position group - all-time & season

### What's NOT Included (Intentionally Removed)
- ❌ Weekly aggregated totals (not useful for game-by-game analysis)
- ❌ Season/career point totals (these rank players, not individual games)
- ❌ Rankings using wrong player IDs (yahoo_player_id instead of nfl_player_id)

---

## 🔑 KEY PLAYER ID NOTES

- **`yahoo_player_id`**: Fantasy-league specific ID (used for manager-based rankings)
- **`nfl_player_id`**: NFL player ID (used for player personal rankings, includes pre-league history)
- **Manager rankings**: Only include games where player was rostered (`manager IS NOT NULL`)
- **NFL rankings**: Include ALL games from NFL history (even before league existed)

---

## 📈 PERCENTILE NOTES

- Percentiles range from 0-100 (higher = better)
- Position rankings exclude 0-point performances from percentile calculation (bye weeks, injuries)
- Rank numbers still include zeros (so you can see "rank 1234" even if percentile is 85th)
- All other rankings include zeros in both rank and percentile

---

## 🚀 NEXT STEPS

When you run your data pipeline, you'll now have:
- **10 new columns** for manager all-player rankings (cross-position)
- **8 columns** for manager-player rankings
- **8 columns** for manager-position rankings
- **8 columns** for player personal rankings (NFL career)
- **8 columns** for position rankings (NFL-wide)

**Total: 42 ranking columns** covering every possible performance comparison!

