#!/usr/bin/env python3
from __future__ import annotations

"""
Cumulative Stats V2 - Multi-League Matchup Transformations

Transforms raw matchup data into enriched analytics with cumulative statistics,
rankings, and cross-season comparisons.

Key Features:
- Multi-league support via LeagueContext
- Data dictionary compliant (adds matchup_key for joins)
- Clear SET-AND-FORGET vs RECALCULATE column logic
- Modular design for maintainability
- Backward compatible with legacy mode

Column Categories:
==================

SET-AND-FORGET COLUMNS (Calculated once after championship, never recalculated):
- final_wins, final_losses - Season totals (finalized after championship)
- final_regular_wins, final_regular_losses - Regular season totals
- season_mean, season_median - Season-level point aggregates
- manager_season_ranking - Final season rank
- championship, sacko, quarterfinal, semifinal - Playoff outcomes

RECALCULATE WEEKLY (Updated every week for cross-week/year comparisons):
- cumulative_wins, cumulative_losses - Running all-time totals
- win_streak, loss_streak - Current active streaks
- teams_beat_this_week - League-relative weekly performance
- w_vs_{opponent}, l_vs_{opponent} - Head-to-head records
- manager_all_time_ranking - Cross-season historical ranks
- All percentile/rank columns comparing across weeks or years

Data Dictionary Additions:
- matchup_key: Unique identifier for manager-opponent pair (for self-joins)
- matchup_id: Unique identifier for specific week matchup instance

Usage:
    # With LeagueContext
    from core.league_context import LeagueContext
    ctx = LeagueContext.load("leagues/kmffl/league_context.json")
    df = transform_cumulative_stats(ctx)

    # CLI
    python cumulative_stats_v2.py --context /path/to/league_context.json
"""

import sys
import argparse
import re
import numpy as np
from functools import wraps
from typing import Optional
import pandas as pd
from pathlib import Path

# Add parent directories to path for imports
_script_file = Path(__file__).resolve()

# Auto-detect if we're in modules/ subdirectory or transformations/ directory
if _script_file.parent.name == 'modules':
    # We're in multi_league/transformations/modules/
    _modules_dir = _script_file.parent
    _transformations_dir = _modules_dir.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'transformations':
    # We're in multi_league/transformations/
    _transformations_dir = _script_file.parent
    _multi_league_dir = _transformations_dir.parent
else:
    # Fallback: assume we're somewhere in the tree, navigate up to find multi_league
    _current = _script_file.parent
    while _current.name != 'multi_league' and _current.parent != _current:
        _current = _current.parent
    _multi_league_dir = _current
    _transformations_dir = _multi_league_dir / 'transformations'

_scripts_dir = _multi_league_dir.parent  # fantasy_football_data_scripts directory
sys.path.insert(0, str(_scripts_dir))  # Allows: from multi_league.core.XXX
sys.path.insert(0, str(_multi_league_dir))  # Allows: from core.XXX

from core.data_normalization import normalize_numeric_columns, ensure_league_id

def ensure_normalized(func):
    """Decorator to ensure data normalization"""
    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        # Normalize input
        df = normalize_numeric_columns(df)

        # Run transformation
        result = func(df, *args, **kwargs)

        # Normalize output
        result = normalize_numeric_columns(result)

        # Ensure league_id present
        if 'league_id' in df.columns:
            league_id = df['league_id'].iloc[0] if len(df) > 0 else None
            if league_id is not None and pd.notna(league_id):
                result = ensure_league_id(result, league_id)

        return result

    return wrapper

# Import the extracted playoff normalization module
# The apply_cumulative_fixes function has been moved to modules/playoff_normalization.py
try:
    from modules.playoff_normalization import normalize_playoff_data as apply_cumulative_fixes
except ImportError:
    from .modules.playoff_normalization import normalize_playoff_data as apply_cumulative_fixes

# Add paths for imports
_script_file = Path(__file__).resolve()
_multi_league_dir = _script_file.parent.parent.parent  # multi_league directory (go up from base -> transformations -> multi_league)
sys.path.insert(0, str(_multi_league_dir))  # Add multi_league parent to path
sys.path.insert(0, str(_multi_league_dir / "multi_league"))  # Add multi_league itself

# Multi-league infrastructure
try:
    from multi_league.core.league_context import LeagueContext
    LEAGUE_CONTEXT_AVAILABLE = True
except ImportError:
    try:
        from core.league_context import LeagueContext
        LEAGUE_CONTEXT_AVAILABLE = True
    except ImportError:
        LeagueContext = None
        LEAGUE_CONTEXT_AVAILABLE = False

# Add modules directory to path
import sys
from pathlib import Path
_modules_dir = Path(__file__).parent / "modules"
sys.path.insert(0, str(_modules_dir))

# Import transformation modules
import playoff_flags
import cumulative_records
import weekly_metrics
import season_rankings
import matchup_rankings
import matchup_keys
import head_to_head
import manager_ppg
import comparative_schedule
import all_play_extended
import playoff_scenarios


# =============================================================================
# Main Transformation Pipeline
# =============================================================================

@ensure_normalized
def transform_cumulative_stats(
    matchup_df: pd.DataFrame,
    current_week: Optional[int] = None,
    current_year: Optional[int] = None,
    championship_complete: bool = False,
    data_directory: Optional[str] = None
) -> pd.DataFrame:
    """
    Transform raw matchup data into enriched cumulative statistics.

    Args:
        matchup_df: Raw matchup data from weekly_matchup_data_v2.py
        current_week: Current week number (for weekly updates)
        current_year: Current year (for weekly updates)
        championship_complete: If True, calculate SET-AND-FORGET columns
        data_directory: Path to league data directory (for finding league settings)

    Returns:
        Enriched DataFrame with cumulative stats
    """
    df = matchup_df.copy()

    _safe_print("\n" + "="*80)
    _safe_print("CUMULATIVE STATS TRANSFORMATION PIPELINE")
    _safe_print("="*80)

    # Step 1: Add matchup keys (for joins)
    _safe_print("\n[1/11] Adding matchup keys...")
    df = matchup_keys.add_matchup_keys(df)

    # Step 2: Calculate cumulative records FIRST (needed for final_playoff_seed)
    _safe_print("[2/11] Calculating cumulative win/loss records...")
    df = cumulative_records.calculate_cumulative_records(df, data_directory=data_directory)

    # Step 3: Normalize playoff flags AFTER we have final_playoff_seed (CRITICAL: is_consolation=1 → is_playoffs=0)
    _safe_print("[3/11] Normalizing playoff/consolation flags...")

    # Import playoff_flags module first (before using it in except block)
    playoff_flags_module = None
    try:
        from multi_league.transformations.matchup.modules import playoff_flags as playoff_flags_module
    except ImportError:
        try:
            import playoff_flags as playoff_flags_module
        except ImportError:
            _safe_print("[WARN] playoff_flags module not found, skipping playoff normalization")

    # Prefer the consolidated fixes helper which also adds weekly H2H columns.
    try:
        df = apply_cumulative_fixes(df, data_directory=data_directory)
        if playoff_flags_module:
            try:
                # Note: enforce_postseason_eligibility and mark_playoff_rounds are already
                # called inside apply_cumulative_fixes, so we only need to call mark_champions_and_sackos here
                df = playoff_flags_module.mark_champions_and_sackos(df, settings_dir=data_directory)
            except Exception as e:
                _safe_print(f"[WARN] mark_champions_and_sackos failed: {e}")
    except Exception as e:
        try:
            print(f"[WARN] apply_cumulative_fixes failed: {e}. Falling back to playoff_flags.normalize_playoff_flags")
        except UnicodeEncodeError:
            print(f"[WARN] apply_cumulative_fixes failed: {str(e).encode('ascii', errors='replace').decode('ascii')}. Falling back to playoff_flags.normalize_playoff_flags")
        if playoff_flags_module:
            try:
                df = playoff_flags_module.normalize_playoff_flags(df)
                # Use seed-based detection to override Yahoo's incorrect API data
                # Settings are auto-loaded from league_settings JSON files
                df = playoff_flags_module.detect_playoffs_by_seed(df, settings_dir=data_directory)
                df = playoff_flags_module.mark_playoff_rounds(df, data_directory=data_directory)
                # Import playoff bracket module for champion/sacko detection
                try:
                    from modules import playoff_bracket as _playoff_bracket_fallback
                except ImportError:
                    import playoff_bracket as _playoff_bracket_fallback
                df = _playoff_bracket_fallback.simulate_playoff_brackets(df, data_directory=data_directory)
            except Exception as fallback_error:
                try:
                    print(f"[ERROR] Fallback also failed: {fallback_error}. Skipping playoff normalization.")
                except UnicodeEncodeError:
                    print(f"[ERROR] Fallback also failed: {str(fallback_error).encode('ascii', errors='replace').decode('ascii')}. Skipping playoff normalization.")

    # Step 3a: DEFENSIVE - Ensure all_time columns are calculated
    # This runs regardless of whether apply_cumulative_fixes succeeded or failed
    # Checks if columns exist and have valid (non-zero) data; recalculates if missing/invalid
    _safe_print("[3a/11] Ensuring manager all-time stats are calculated...")
    try:
        # Check if all_time columns need to be calculated
        all_time_cols_needed = ['manager_all_time_gp', 'manager_all_time_wins',
                                'manager_all_time_losses', 'manager_all_time_ties', 'manager_all_time_win_pct']
        needs_calculation = (
            any(col not in df.columns for col in all_time_cols_needed) or
            (df.get('manager_all_time_gp', pd.Series([0])).fillna(0).sum() == 0)
        )

        if needs_calculation and 'manager' in df.columns and 'win' in df.columns:
            _safe_print("  [INFO] Calculating manager all-time stats...")

            # Ensure win/loss/tie are numeric
            for c in ['win', 'loss', 'tie']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
                else:
                    df[c] = 0

            # Aggregate historical totals per manager
            all_time = (
                df.groupby('manager', dropna=False)
                  .agg(
                      manager_all_time_gp=('win', 'size'),
                      manager_all_time_wins=('win', 'sum'),
                      manager_all_time_losses=('loss', 'sum'),
                      manager_all_time_ties=('tie', 'sum')
                  )
                  .reset_index()
            )

            # Compute win percentage
            denom = (all_time['manager_all_time_wins'] + all_time['manager_all_time_losses']).replace(0, pd.NA)
            all_time['manager_all_time_win_pct'] = (all_time['manager_all_time_wins'] / denom).fillna(0.0)

            # Drop old columns if they exist
            df = df.drop(columns=[c for c in all_time_cols_needed if c in df.columns], errors='ignore')

            # Merge aggregates back onto main df
            df = df.merge(all_time, on='manager', how='left')

            _safe_print(f"  [OK] Calculated all-time stats for {len(all_time)} managers")
        else:
            _safe_print("  [OK] All-time stats already present and valid")
    except Exception as e:
        _safe_print(f"  [WARN] Failed to calculate all-time stats: {e}")

    # Step 3b: Add season_result column (AFTER playoff bracket simulation)
    _safe_print("[3b/11] Adding season_result column...")
    try:
        if playoff_flags_module:
            df = playoff_flags_module.add_season_result(df)
        else:
            _safe_print("  [WARN] playoff_flags module not loaded, initializing season_result to empty")
            if 'season_result' not in df.columns:
                df['season_result'] = ""
    except Exception as e:
        _safe_print(f"[WARN] Failed to add season_result: {e}. Skipping this step.")

    # DEFENSIVE: Ensure season_result column exists and has no NaN values
    if 'season_result' not in df.columns:
        df['season_result'] = ""
    else:
        # Replace NaN with empty string for consistency
        df['season_result'] = df['season_result'].fillna("")

    # Create aliases for streak columns (backwards compatibility)
    if 'win_streak' in df.columns:
        df['winning_streak'] = df['win_streak']
    if 'loss_streak' in df.columns:
        df['losing_streak'] = df['loss_streak']

    # Step 4: Calculate manager PPG metrics (RECALCULATE WEEKLY)
    _safe_print("[4/11] Calculating manager PPG (weekly_mean, weekly_median)...")
    try:
        df = manager_ppg.calculate_manager_ppg(df)
    except Exception as e:
        _safe_print(f"[WARN] Failed to calculate manager PPG: {e}. Skipping this step.")

    # Step 4b: Rename weekly mean/median and create aliases (MUST be after manager_ppg creates them!)
    # Drop old target columns if they exist (prevents conflicts during rename)
    for old_col in ['manager_season_mean', 'manager_season_median']:
        if old_col in df.columns and old_col != 'weekly_mean' and old_col != 'weekly_median':
            df = df.drop(columns=[old_col])

    df.rename(columns={
        'weekly_mean': 'manager_season_mean',
        'weekly_median': 'manager_season_median'
    }, inplace=True)

    # Create aliases for backwards compatibility
    if 'manager_season_mean' in df.columns:
        df['personal_season_mean'] = df['manager_season_mean'].values
    if 'manager_season_median' in df.columns:
        df['personal_season_median'] = df['manager_season_median'].values

    _safe_print("  [OK] Renamed weekly_mean → manager_season_mean, weekly_median → manager_season_median")

    # Step 5: Calculate weekly metrics (RECALCULATE WEEKLY)
    _safe_print("[5/11] Calculating weekly league-relative metrics...")
    df = weekly_metrics.calculate_weekly_metrics(df)

    # Step 6: Calculate opponent all-play metrics (RECALCULATE WEEKLY)
    _safe_print("[6/11] Calculating opponent all-play metrics...")
    try:
        df = all_play_extended.calculate_opponent_all_play(df)
    except Exception as e:
        _safe_print(f"[WARN] Failed to calculate opponent all-play: {e}. Skipping this step.")

    # Step 7: Calculate head-to-head records (RECALCULATE WEEKLY)
    _safe_print("[7/11] Calculating head-to-head records...")
    df = head_to_head.calculate_head_to_head_records(df)

    # Step 8: Calculate comparative schedule (RECALCULATE WEEKLY)
    _safe_print("[8/11] Calculating comparative schedule (w_vs_X_sched)...")
    try:
        df = comparative_schedule.calculate_comparative_schedule(df)
    except Exception as e:
        _safe_print(f"[WARN] Failed to calculate comparative schedule: {e}. Skipping this step.")

    # Step 9: Calculate season rankings (SET-AND-FORGET if championship complete)
    _safe_print("[9/11] Calculating season rankings...")
    df = season_rankings.calculate_season_rankings(
        df,
        championship_complete=championship_complete
    )

    # Step 9.5: Calculate manager-specific matchup rankings (RECALCULATE WEEKLY)
    _safe_print("[9.5/11] Calculating manager matchup rankings...")
    try:
        df = matchup_rankings.calculate_all_matchup_rankings(df)
    except Exception as e:
        _safe_print(f"[WARN] Failed to calculate matchup rankings: {e}. Skipping this step.")

    # Step 10: Calculate all-time rankings (RECALCULATE WEEKLY)
    _safe_print("[10/11] Calculating all-time rankings...")
    try:
        df = season_rankings.calculate_alltime_rankings(df)
    except Exception as e:
        _safe_print(f"[WARN] Failed to calculate all-time rankings: {e}. Skipping this step.")

    # ------------------------------------------------------------
    # Playoff Scenario Columns (magic numbers, clinch, weekly changes)
    # ------------------------------------------------------------
    _safe_print("[11/12] Adding playoff scenario columns...")
    try:
        df = playoff_scenarios.add_playoff_scenario_columns(df, data_directory=data_directory)
    except Exception as e:
        _safe_print(f"[WARN] Failed to add playoff scenario columns: {e}. Skipping this step.")

    # ------------------------------------------------------------
    # Inflation Rate (year-over-year scoring normalization)
    # ------------------------------------------------------------
    _safe_print("[12/12] Calculating inflation rate...")
    try:
        from modules.inflation_rate import calculate_inflation_rate
    except ImportError:
        from .modules.inflation_rate import calculate_inflation_rate
    df = calculate_inflation_rate(df, log_fn=_safe_print)

    # FINAL DEFENSIVE: Ensure string columns are never null (use empty string instead)
    # This prevents downstream issues with merges and comparisons
    string_cols_to_clean = ['playoff_round', 'consolation_round', 'season_result']
    for col in string_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    _safe_print("\n" + "="*80)
    _safe_print(f"[OK] Transformation complete: {len(df)} rows, {len(df.columns)} columns")
    _safe_print("="*80 + "\n")

    return df


def load_matchup_data(ctx: LeagueContext) -> pd.DataFrame:
    """Load matchup data from LeagueContext."""
    matchup_file = ctx.canonical_matchup_file

    if not matchup_file.exists():
        raise FileNotFoundError(f"Matchup data not found: {matchup_file}")

    print(f"Loading matchup data from: {matchup_file}")
    df = pd.read_parquet(matchup_file)
    print(f"Loaded {len(df)} matchup records")

    return df


def save_enriched_matchup(df: pd.DataFrame, ctx: LeagueContext) -> None:
    """Save enriched matchup data back to the canonical matchup file."""
    output_file = ctx.canonical_matchup_file

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ABSOLUTE FINAL ENFORCEMENT: Mutual exclusivity and correct binary flags
    # This is the LAST possible moment before data is saved
    print("[FINAL] Enforcing mutual exclusivity and correct binary flags...")

    for idx, row in df.iterrows():
        if row.get('is_playoffs') == 1:
            # Playoff game - clear consolation labels
            df.at[idx, 'consolation_round'] = ""
            df.at[idx, 'consolation_semifinal'] = 0
            df.at[idx, 'consolation_final'] = 0

            # Set ONLY the correct playoff binary flag based on playoff_round
            playoff_round = str(row.get('playoff_round', ''))
            df.at[idx, 'quarterfinal'] = 1 if playoff_round == 'quarterfinal' else 0
            df.at[idx, 'semifinal'] = 1 if playoff_round == 'semifinal' else 0
            df.at[idx, 'championship'] = 1 if playoff_round == 'championship' else 0

        elif row.get('is_consolation') == 1:
            # Consolation game - clear playoff labels
            df.at[idx, 'playoff_round'] = ""
            df.at[idx, 'quarterfinal'] = 0
            df.at[idx, 'semifinal'] = 0
            df.at[idx, 'championship'] = 0

            # Set ONLY the correct consolation binary flag based on consolation_round
            cons_round = str(row.get('consolation_round', ''))
            df.at[idx, 'consolation_semifinal'] = 1 if 'semifinal' in cons_round else 0
            df.at[idx, 'consolation_final'] = 1 if 'final' in cons_round else 0

    print("[FINAL] Enforcement complete - binary flags corrected")

    # Save
    df.to_parquet(output_file, index=False)
    _safe_print(f"[OK] Updated matchup data: {output_file}")

    # Also save CSV for convenience
    csv_file = output_file.with_suffix('.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    _safe_print(f"[OK] Updated matchup data (CSV): {csv_file}")


# =============================================================================
# CLI
# =============================================================================

def _safe_print(*args, **kwargs):
    # Proxy to playoff_flags.safe_print for consistent behavior
    if 'playoff_flags' in globals() and hasattr(playoff_flags, "safe_print"):
        playoff_flags.safe_print(*args, **kwargs)
    else:
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            print(*(str(a).encode("ascii", "replace").decode("ascii") for a in args), **kwargs)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transform matchup data with cumulative statistics (Multi-League V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform matchup data for a league
  python cumulative_stats_v2.py --context /path/to/league_context.json

  # Mark championship as complete (finalizes SET-AND-FORGET columns)
  python cumulative_stats_v2.py --context /path/to/league_context.json --championship-complete

  # Specify current week for weekly updates
  python cumulative_stats_v2.py --context /path/to/league_context.json --current-week 10 --current-year 2024
        """
    )

    parser.add_argument('--context', type=Path, required=True, help='Path to league_context.json')
    parser.add_argument('--championship-complete', action='store_true',
                        help='Mark championship as complete (finalizes season stats)')
    parser.add_argument('--current-week', type=int, help='Current week number')
    parser.add_argument('--current-year', type=int, help='Current year')

    args = parser.parse_args()

    # Validate context
    if not LEAGUE_CONTEXT_AVAILABLE:
        print("ERROR: LeagueContext not available. Ensure multi_league package is installed.")
        return 1

    if not args.context.exists():
        print(f"ERROR: Context file not found: {args.context}")
        return 1

    # Load context
    try:
        ctx = LeagueContext.load(args.context)
        print(f"Processing league: {ctx.league_name} ({ctx.league_id})")
    except Exception as e:
        print(f"ERROR: Failed to load league context: {e}")
        return 1

    # Load matchup data
    try:
        matchup_df = load_matchup_data(ctx)
    except Exception as e:
        print(f"ERROR: Failed to load matchup data: {e}")
        return 1

    # Transform
    try:
        enriched_df = transform_cumulative_stats(
            matchup_df,
            current_week=args.current_week,
            current_year=args.current_year,
            championship_complete=args.championship_complete,
            data_directory=str(ctx.data_directory)
        )
    except Exception as e:
        _safe_print(f"ERROR: Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Defensive: make sure optional all-time columns exist
    OPTIONAL_SAFE_DEFAULTS = {
        "manager_all_time_ranking": pd.NA,
        "manager_all_time_win_pct": pd.NA,
        "manager_all_time_gp": 0,
        "manager_all_time_wins": 0,
        "manager_all_time_losses": 0,
        "manager_all_time_ties": 0,
    }

    for col, default_val in OPTIONAL_SAFE_DEFAULTS.items():
        if col not in enriched_df.columns:
            enriched_df[col] = default_val

    # Save enriched matchup data back out
    try:
        save_enriched_matchup(enriched_df, ctx)
    except Exception as e:
        print(f"ERROR: Failed to save enriched data: {e}")
        return 1

    _safe_print("\n[OK] Cumulative stats transformation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
