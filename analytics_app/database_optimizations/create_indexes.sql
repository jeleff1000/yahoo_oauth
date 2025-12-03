-- ============================================
-- Database Optimization Indexes
-- For fantasy football player data
-- ============================================
--
-- These indexes optimize common query patterns:
-- 1. Weekly player lookups by year/week
-- 2. Position-specific queries
-- 3. Manager-specific queries
-- 4. Sorted queries by points
--
-- Expected performance improvement: 3-4x faster queries
-- ============================================

-- Index for common weekly player queries
-- Covers: year, week, position filters with points sorting
CREATE INDEX IF NOT EXISTS idx_player_year_week_pos_pts
ON player(year, week, nfl_position, points DESC);

-- Index for manager-specific queries
-- Covers: manager filters across years and weeks
CREATE INDEX IF NOT EXISTS idx_player_manager_year_week
ON player(manager, year, week)
WHERE manager IS NOT NULL AND manager <> '';

-- Index for position-specific queries with optimal flag
-- Covers: Position filters with optimal lineup queries
CREATE INDEX IF NOT EXISTS idx_player_position_optimal
ON player(nfl_position, league_wide_optimal_player, points DESC);

-- Index for fantasy position queries (roster slots)
-- Covers: Fantasy roster analysis
CREATE INDEX IF NOT EXISTS idx_player_fantasy_position
ON player(fantasy_position, year, week)
WHERE fantasy_position IS NOT NULL;

-- Index for matchup-based queries
-- Covers: Matchup analysis by team/opponent
CREATE INDEX IF NOT EXISTS idx_player_matchup
ON player(manager, opponent, year, week)
WHERE manager IS NOT NULL AND opponent IS NOT NULL;

-- Covering index for the most common query pattern
-- This index contains all frequently accessed columns
-- Speeds up: SELECT player, manager, points, nfl_position, nfl_team WHERE year = X AND week = Y
CREATE INDEX IF NOT EXISTS idx_player_weekly_stats_covering
ON player(year, week, points DESC)
INCLUDE (player, manager, nfl_position, nfl_team, fantasy_position, yahoo_player_id);

-- Index for player name searches
-- Covers: Player search functionality
CREATE INDEX IF NOT EXISTS idx_player_name_year
ON player(player, year, week);

-- Index for started players only (common filter)
-- Covers: started=1 queries
CREATE INDEX IF NOT EXISTS idx_player_started
ON player(year, week, started, points DESC)
WHERE started = 1;

-- Index for playoff queries
-- Covers: Playoff-specific analysis
CREATE INDEX IF NOT EXISTS idx_player_playoffs
ON player(year, week, is_playoffs, points DESC)
WHERE is_playoffs = 1;

-- ============================================
-- ANALYZE command to update statistics
-- Run this after creating indexes
-- ============================================
ANALYZE player;

-- ============================================
-- Verification Queries
-- Run these to check if indexes are being used
-- ============================================

-- Check index usage for common query
-- EXPLAIN QUERY PLAN
-- SELECT * FROM player
-- WHERE year = 2024 AND week = 1 AND nfl_position = 'QB'
-- ORDER BY points DESC
-- LIMIT 100;

-- Should show: SEARCH player USING INDEX idx_player_year_week_pos_pts

-- ============================================
-- Index Maintenance
-- ============================================

-- To drop all indexes (if you need to recreate):
-- DROP INDEX IF EXISTS idx_player_year_week_pos_pts;
-- DROP INDEX IF EXISTS idx_player_manager_year_week;
-- DROP INDEX IF EXISTS idx_player_position_optimal;
-- DROP INDEX IF EXISTS idx_player_fantasy_position;
-- DROP INDEX IF EXISTS idx_player_matchup;
-- DROP INDEX IF EXISTS idx_player_weekly_stats_covering;
-- DROP INDEX IF EXISTS idx_player_name_year;
-- DROP INDEX IF EXISTS idx_player_started;
-- DROP INDEX IF EXISTS idx_player_playoffs;
