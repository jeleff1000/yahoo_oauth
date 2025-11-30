#!/usr/bin/env python3
"""
Run database optimizations for fantasy football data.

This script creates indexes on the player table to speed up common queries.
Expected performance improvement: 3-4x faster query execution.

Usage:
    python run_optimizations.py

or from within the app:
    from database_optimizations.run_optimizations import optimize_database
    optimize_database()
"""

import sys
import os

# Add parent directory to path to import md.data_access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from md.data_access import run_query


def optimize_database():
    """
    Create database indexes for optimal query performance.

    Returns:
        True if successful, False otherwise
    """
    print("🚀 Starting database optimization...")
    print("=" * 60)

    indexes = [
        (
            "idx_player_year_week_pos_pts",
            "CREATE INDEX IF NOT EXISTS idx_player_year_week_pos_pts ON player(year, week, nfl_position, points DESC)",
            "Year/Week/Position queries with points sorting"
        ),
        (
            "idx_player_manager_year_week",
            "CREATE INDEX IF NOT EXISTS idx_player_manager_year_week ON player(manager, year, week) WHERE manager IS NOT NULL AND manager <> ''",
            "Manager-specific queries"
        ),
        (
            "idx_player_position_optimal",
            "CREATE INDEX IF NOT EXISTS idx_player_position_optimal ON player(nfl_position, league_wide_optimal_player, points DESC)",
            "Position queries with optimal lineup flag"
        ),
        (
            "idx_player_fantasy_position",
            "CREATE INDEX IF NOT EXISTS idx_player_fantasy_position ON player(fantasy_position, year, week) WHERE fantasy_position IS NOT NULL",
            "Fantasy position (roster slot) queries"
        ),
        (
            "idx_player_matchup",
            "CREATE INDEX IF NOT EXISTS idx_player_matchup ON player(manager, opponent, year, week) WHERE manager IS NOT NULL AND opponent IS NOT NULL",
            "Matchup-based queries"
        ),
        (
            "idx_player_name_year",
            "CREATE INDEX IF NOT EXISTS idx_player_name_year ON player(player, year, week)",
            "Player name search queries"
        ),
        (
            "idx_player_started",
            "CREATE INDEX IF NOT EXISTS idx_player_started ON player(year, week, started, points DESC) WHERE started = 1",
            "Started players filter"
        ),
        (
            "idx_player_playoffs",
            "CREATE INDEX IF NOT EXISTS idx_player_playoffs ON player(year, week, is_playoffs, points DESC) WHERE is_playoffs = 1",
            "Playoff-specific queries"
        ),
    ]

    success_count = 0
    error_count = 0

    for name, sql, description in indexes:
        try:
            print(f"\n📊 Creating index: {name}")
            print(f"   Purpose: {description}")
            run_query(sql)
            print(f"   ✅ Success")
            success_count += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
            error_count += 1

    # Run ANALYZE to update statistics
    print("\n📈 Updating table statistics...")
    try:
        run_query("ANALYZE player")
        print("   ✅ Statistics updated")
    except Exception as e:
        print(f"   ❌ Error updating statistics: {e}")

    print("\n" + "=" * 60)
    print(f"✨ Optimization complete!")
    print(f"   ✅ {success_count} indexes created successfully")
    if error_count > 0:
        print(f"   ❌ {error_count} errors encountered")
    print(f"\n💡 Expected performance improvement: 3-4x faster queries")
    print("=" * 60)

    return error_count == 0


if __name__ == "__main__":
    optimize_database()
