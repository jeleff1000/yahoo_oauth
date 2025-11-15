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
            if league_id:
                result = ensure_league_id(result, league_id)

        return result

    return wrapper

@ensure_normalized
def apply_cumulative_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize playoff/consolation flags; ensure bracket columns always exist;
    compute seeds + rounds via normal path; fallback to row-count-based inference.
    """
    df = df.copy()

    # ---- flags guaranteed ----
    for c in ("is_playoffs", "is_consolation"):
        if c not in df.columns:
            df[c] = 0
    df["is_playoffs"] = pd.to_numeric(df["is_playoffs"], errors="coerce").fillna(0).astype(int)
    df["is_consolation"] = pd.to_numeric(df["is_consolation"], errors="coerce").fillna(0).astype(int)

    # ---- bracket columns always present ----
    int_cols = ["playoff_week_index","playoff_round_num","quarterfinal","semifinal","championship",
                "placement_game","placement_rank","consolation_semifinal","consolation_final",
                "champion","sacko"]
    str_cols = ["playoff_round","consolation_round"]
    for c in int_cols:
        if c not in df.columns: df[c] = 0
    for c in str_cols:
        if c not in df.columns: df[c] = ""

    # helpful placeholders
    for c in ("final_playoff_seed","playoff_start_week","num_playoff_teams"):
        if c not in df.columns: df[c] = np.nan
    if "postseason" not in df.columns:
        df["postseason"] = 0

    # ---- normal path (seed-based) ----
    try:
        # Import cumulative records module
        try:
            from modules import cumulative_records as _cumulative_records
            _calc_records = _cumulative_records.calculate_cumulative_records
        except ImportError:
            import cumulative_records as _cumulative_records
            _calc_records = _cumulative_records.calculate_cumulative_records

        # Import playoff flags module
        try:
            from modules import playoff_flags as _playoff_flags
        except ImportError:
            import playoff_flags as _playoff_flags

        # Calculate cumulative records (includes final_playoff_seed calculation)
        df = _calc_records(df)

        # Use seed-based detection to correctly identify playoffs vs consolation
        df = _playoff_flags.detect_playoffs_by_seed(df)

        # Mark playoff rounds (championship + consolation brackets)
        df = _playoff_flags.mark_playoff_rounds(df)

        # Mark champions and sackos
        df = _playoff_flags.mark_champions_and_sackos(df)

        # FINAL GUARDRAIL: Enforce mutual exclusivity
        df.loc[df["is_consolation"] == 1, "is_playoffs"] = 0
        df.loc[df["is_playoffs"] == 1, "is_consolation"] = 0

        print("[INFO] apply_cumulative_fixes: normal seed-based playoff detection succeeded")

    except Exception as e:
        print(f"[WARN] apply_cumulative_fixes: normal path failed -> {e}")

        # ---- fallback (row-count collapse) ----
        try:
            need = {"league_id","year","week"}
            if not need.issubset(df.columns):
                print("[WARN] fallback: missing league_id/year/week; cannot infer playoffs")
            else:
                g = (df.groupby(["league_id","year","week"]).size()
                       .reset_index(name="rows"))
                modal = (g.groupby(["league_id","year"])["rows"]
                           .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.max())
                           .rename("regular_rows").reset_index())
                g = g.merge(modal, on=["league_id","year"], how="left")
                g["is_playoff_week"] = (g["rows"] < g["regular_rows"]).astype(int)

                start = (g[g["is_playoff_week"]==1]
                         .groupby(["league_id","year"])["week"].min()
                         .rename("playoff_start_week").reset_index())

                df = df.merge(start, on=["league_id","year"], how="left")
                df["postseason"] = np.where(df["playoff_start_week"].notna() & (df["week"] >= df["playoff_start_week"]), 1, 0)
                df.loc[df["postseason"]==1, "is_playoffs"] = 1

                df["playoff_week_index"] = np.where(df["postseason"]==1,
                                                    (df["week"]-df["playoff_start_week"]+1).clip(lower=1),
                                                    0).astype(int)
                df["playoff_round_num"] = df["playoff_week_index"]

                def _round_name(i):
                    if i <= 0: return ""
                    return {1:"quarterfinal", 2:"semifinal", 3:"championship"}.get(int(i), f"playoff_round_{int(i)}")

                df["playoff_round"] = df["playoff_week_index"].map(_round_name).fillna("")
                df["quarterfinal"] = (df["playoff_round"]=="quarterfinal").astype(int)
                df["semifinal"]    = (df["playoff_round"]=="semifinal").astype(int)
                df["championship"] = (df["playoff_round"]=="championship").astype(int)

                # mark champion if a 'win' col exists and final week has <=2 rows
                df = df.merge(g[["league_id","year","week","rows"]], on=["league_id","year","week"], how="left")
                if "win" in df.columns:
                    df["champion"] = np.where(
                        (df["championship"]==1) & (df["rows"]<=2) & (df["win"]==1), 1, df["champion"]
                    ).astype(int)

                print("[INFO] apply_cumulative_fixes: used fallback playoff inference (row-count collapse).")

        except Exception as fallback_e:
            print(f"[WARN] fallback playoff inference failed -> {fallback_e}")

    # ---- Continue with rest of transformations ----
    # 1) Rename weekly mean/median and create aliases
    df.rename(columns={
        'weekly_mean': 'manager_season_mean',
        'weekly_median': 'manager_season_median'
    }, inplace=True)

    # Create aliases for backwards compatibility
    if 'manager_season_mean' in df.columns:
        df['personal_season_mean'] = df['manager_season_mean']
    if 'manager_season_median' in df.columns:
        df['personal_season_median'] = df['manager_season_median']

    # 2) Weekly one-vs-all columns based on same-week points
    pts_col = next((c for c in ['team_points','points_for','pf'] if c in df.columns), None)
    if pts_col is not None:
        try:
            # Prefer an explicit helper if available. Support both possible names
            # to be resilient to different head_to_head implementations.
            try:
                from head_to_head import calculate_weekly_one_vs_all as _hh_func
            except Exception:
                from head_to_head import calculate_head_to_head_records as _hh_func
            df = _hh_func(df)
        except Exception:
            # safe inline fallback if import/pathing fails
            managers = sorted(pd.unique(df.get("manager", pd.Series(dtype=object)).dropna()))
            tokens = {m: re.sub(r"[^a-zA-Z0-9]+", "_", str(m).strip().lower()).strip("_") or "na" for m in managers}
            for m in managers:
                df[f"w_vs_{tokens[m]}"] = 0
            for c in ['year','week']:
                if c not in df.columns:
                    df[c] = -1
            df[pts_col] = pd.to_numeric(df[pts_col], errors="coerce")
            for (y,w), g in df.groupby(['year','week']):
                pts_map = g.set_index('manager')[pts_col].to_dict() if 'manager' in g.columns else {}
                for idx, row in g.iterrows():
                    my_m = row.get('manager')
                    my_pts = row[pts_col]
                    for opp in managers:
                        col = f"w_vs_{tokens[opp]}"
                        if opp == my_m:
                            df.at[idx, col] = 0
                        else:
                            opp_pts = pts_map.get(opp, None)
                            df.at[idx, col] = int(pd.notna(my_pts) and pd.notna(opp_pts) and (my_pts > opp_pts))

    # 3) Season summaries + ranking
    # Preserve manager names in case any merge path drops them
    _manager_snapshot = df['manager'].copy()

    needed = [
        'final_wins','final_losses','final_regular_wins','final_regular_losses',
        'season_mean','season_median','manager_season_ranking',
        'champion','semifinal','quarterfinal','sacko'
    ]
    # Only compute if any of these are missing
    if any(c not in df.columns for c in needed):
        # Ensure win/loss/tie are numeric if present
        for c in ['win','loss','tie']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

        # Ensure manager_year exists
        if 'manager_year' not in df.columns:
            # Use separator to avoid collisions
            df['manager_year'] = df['manager'].astype(str) + '_' + df['year'].astype(str)

        # Non-consolation mask
        non_consol = (df['is_consolation'].fillna(0).astype(int) != 1)

        # Determine points column safely
        pts_col = next((c for c in ['team_points','points_for','pf'] if c in df.columns), None)
        if pts_col is not None:
            pts_series = pd.to_numeric(df[pts_col], errors='coerce').fillna(0.0)
        else:
            pts_series = pd.Series(0.0, index=df.index)

        # Regular-season totals (exclude consolation)
        reg = (
            df.assign(
                win_nc=(df['win'] * non_consol.astype(int)),
                loss_nc=(df['loss'] * non_consol.astype(int)),
                pts_nc=(pts_series * non_consol.astype(int))
            )
            .groupby(['year','manager_year','manager'], dropna=False)
            .agg(final_regular_wins=('win_nc','sum'),
                 final_regular_losses=('loss_nc','sum'),
                 season_mean=('pts_nc','mean'),
                 season_median=('pts_nc','median'))
            .reset_index()
        )

        # Full season totals (include playoffs)
        full = (
            df.groupby(['year','manager_year','manager'], dropna=False)
              .agg(final_wins=('win','sum'),
                   final_losses=('loss','sum'))
              .reset_index()
        )

        # Merge aggregates back onto df
        df = df.merge(reg, on=['year','manager_year','manager'], how='left')
        df = df.merge(full, on=['year','manager_year','manager'], how='left')

        # Rank each manager's weekly performance within their own season
        pts_col = next((c for c in ['team_points','points_for','pf'] if c in df.columns), None)
        if pts_col is not None:
            df['manager_season_ranking'] = (
                df.groupby(['manager','year'])[pts_col]
                  .rank(method='dense', ascending=False)
                  .astype(int)
            )
        else:
            df['manager_season_ranking'] = pd.NA

        # Ensure playoff-result flags exist
        for col in ['champion','semifinal','quarterfinal','sacko']:
            if col not in df.columns:
                df[col] = 0

    # 4) All-time aggregates (gp, w, l, t, win%)
    # Ensure win/loss/tie are numeric and present
    for c in ['win','loss','tie']:
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

    # Compute win percentage (wins / (wins + losses)), defensively handling zero denom
    denom = (all_time['manager_all_time_wins'] + all_time['manager_all_time_losses']).replace(0, pd.NA)
    all_time['manager_all_time_win_pct'] = (all_time['manager_all_time_wins'] / denom).fillna(0.0)

    # Merge aggregates back onto main df
    df = df.merge(all_time, on='manager', how='left')

    # Restore any lost manager names
    if df['manager'].isna().any():
        df['manager'] = df['manager'].fillna(_manager_snapshot)

    # 5) Per-week manager_all_time_ranking + percentile
    pts_col2 = next((c for c in ['team_points','points_for','pf'] if c in df.columns), None)
    if pts_col2 is not None:
        # Rank within each manager's entire history: 1 = best ever performance
        # Use method='min' so ties get the same best rank
        ranks = (df.groupby('manager')[pts_col2]
                   .rank(method='min', ascending=False)
                   .astype('Int64'))
        df['manager_all_time_ranking'] = ranks
        # Number of non-null historical records per manager
        counts = df.groupby('manager')[pts_col2].transform('count').astype(float)
        # Percentile: higher is better. If only one record, set percentile to 100.0
        pct = ((counts - ranks.astype(float)) / (counts - 1.0)).where(counts > 1, 1.0) * 100.0
        df['manager_all_time_percentile'] = pct.round(2)

    # --- Weekly, intra-season ranking: best weekly score = 1 per manager/year ---
    pts_col = next((c for c in ["team_points","points_for","pf"] if c in df.columns), None)
    if pts_col is not None:
        df[pts_col] = pd.to_numeric(df[pts_col], errors="coerce")
        df["manager_season_ranking"] = (
            df.groupby(["manager","year"])[pts_col]
              .rank(method="dense", ascending=False)
              .astype("Int64")
        )
    else:
        df["manager_season_ranking"] = pd.NA

    # 6) Quick assertions (recommended)
    # No row should have both flags set (mutual exclusivity)
    _bad = (df["is_playoffs"] == 1) & (df["is_consolation"] == 1)
    if _bad.any():
        raise AssertionError(f"Playoff/consolation flags conflict on {_bad.sum()} rows")

    # Champion rows must be playoff-eligible
    if "champion" in df.columns:
        _badc = (df["champion"] == 1) & (df["is_playoffs"] != 1)
        if _badc.any():
            raise AssertionError(f"Champion flagged outside playoffs on {_badc.sum()} rows")

    # Sacko rows must be consolation
    if "sacko" in df.columns:
        _bads = (df["sacko"] == 1) & (df["is_consolation"] != 1)
        if _bads.any():
            raise AssertionError(f"Sacko flagged outside consolation on {_bads.sum()} rows")

    # --- Champion & Sacko detection ---
    for col in ["champion","sacko"]:
        if col not in df.columns:
            df[col] = 0

    # Use normalized flags for logic
    df["win"]  = pd.to_numeric(df.get("win", 0), errors="coerce").fillna(0).astype(int)
    df["loss"] = pd.to_numeric(df.get("loss", 0), errors="coerce").fillna(0).astype(int)

    years = sorted(df["year"].dropna().unique().astype(int))

    for yr in years:
        ymask = df["year"] == yr

        # -------- Champion --------
        po = df[ymask & (df["is_playoffs"] == 1)]
        if not po.empty and "playoff_round" in df.columns:
            # Last playoff week with a 'championship' label
            last_po_wk = po.loc[po["playoff_round"].astype(str).str.lower().eq("championship"), "week"]
            if not last_po_wk.empty:
                wk = int(last_po_wk.min())
                champ_mask = ymask & (df["week"] == wk) & (df["is_playoffs"] == 1) & (df["win"] == 1)
                df.loc[champ_mask, "champion"] = 1

        # -------- Sacko (toilet loser) --------
        cons = df[ymask & (df["is_consolation"] == 1)]
        if not cons.empty:
            # Pick the *last* consolation week that has head-to-head rows
            last_cons_wk = int(cons["week"].max())
            # Mark Sacko = loser of the last consolation week (loss == 1)
            sacko_mask = ymask & (df["week"] == last_cons_wk) & (df["is_consolation"] == 1) & (df["loss"] == 1)
            df.loc[sacko_mask, "sacko"] = 1

    # ----------------------------------------------------------------------
    # SACKO: loses EVERY consolation game they play that postseason
    # ----------------------------------------------------------------------
    if "sacko" not in df.columns:
        df["sacko"] = 0

    # ensure ints - use Int64 dtype to allow NaN values during intermediate operations
    for c in ("is_consolation", "is_playoffs", "win", "loss", "year", "week"):
        if c not in df.columns:
            df[c] = 0
        # First ensure numeric, then fillna, then handle any remaining non-finite values
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

    # compute Sacko per (manager, year)
    for (mgr, yr), g in df.groupby(["manager", "year"]):
        cons = g[g["is_consolation"] == 1].sort_values("week")
        if cons.empty:
            continue

        # total consolation games for this manager this season
        games = len(cons)
        losses = int((cons["loss"] == 1).sum())
        wins   = int((cons["win"]  == 1).sum())
        # Sacko only if they lost *every* consolation game (no wins, no ties)
        if games > 0 and losses == games and wins == 0:
            last_wk = int(cons["week"].max())
            mask = (
                (df["manager"] == mgr) &
                (df["year"] == yr) &
                (df["week"] == last_wk) &
                (df["is_consolation"] == 1)
            )
            df.loc[mask, "sacko"] = 1

    return df

# Add paths for imports
_script_file = Path(__file__).resolve()
_multi_league_dir = _script_file.parent.parent  # multi_league directory
sys.path.insert(0, str(_multi_league_dir / "core"))

# Multi-league infrastructure
try:
    from league_context import LeagueContext
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
import matchup_keys
import head_to_head
import manager_ppg
import comparative_schedule
import all_play_extended


# =============================================================================
# Main Transformation Pipeline
# =============================================================================

@ensure_normalized
def transform_cumulative_stats(
    matchup_df: pd.DataFrame,
    current_week: Optional[int] = None,
    current_year: Optional[int] = None,
    championship_complete: bool = False
) -> pd.DataFrame:
    """
    Transform raw matchup data into enriched cumulative statistics.

    Args:
        matchup_df: Raw matchup data from weekly_matchup_data_v2.py
        current_week: Current week number (for weekly updates)
        current_year: Current year (for weekly updates)
        championship_complete: If True, calculate SET-AND-FORGET columns

    Returns:
        Enriched DataFrame with cumulative stats
    """
    df = matchup_df.copy()

    _safe_print("\n" + "="*80)
    _safe_print("CUMULATIVE STATS TRANSFORMATION PIPELINE")
    _safe_print("="*80)

    # Step 1: Add matchup keys (for joins)
    _safe_print("\n[1/7] Adding matchup keys...")
    df = matchup_keys.add_matchup_keys(df)

    # Step 2: Calculate cumulative records FIRST (needed for final_playoff_seed)
    _safe_print("[2/7] Calculating cumulative win/loss records...")
    df = cumulative_records.calculate_cumulative_records(df)

    # Step 3: Normalize playoff flags AFTER we have final_playoff_seed (CRITICAL: is_consolation=1 → is_playoffs=0)
    _safe_print("[3/7] Normalizing playoff/consolation flags...")

    # Import playoff_flags module first (before using it in except block)
    playoff_flags_module = None
    try:
        from multi_league.transformations.modules import playoff_flags as playoff_flags_module
    except ImportError:
        try:
            import playoff_flags as playoff_flags_module
        except ImportError:
            _safe_print("[WARN] playoff_flags module not found, skipping playoff normalization")

    # Prefer the consolidated fixes helper which also adds weekly H2H columns.
    try:
        df = apply_cumulative_fixes(df)
        if playoff_flags_module:
            try:
                # Note: enforce_postseason_eligibility and mark_playoff_rounds are already
                # called inside apply_cumulative_fixes, so we only need to call mark_champions_and_sackos here
                df = playoff_flags_module.mark_champions_and_sackos(df)
            except Exception as e:
                _safe_print(f"[WARN] mark_champions_and_sackos failed: {e}")
    except Exception as e:
        _safe_print(f"[WARN] apply_cumulative_fixes failed: {e}. Falling back to playoff_flags.normalize_playoff_flags")
        if playoff_flags_module:
            try:
                df = playoff_flags_module.normalize_playoff_flags(df)
                # Use seed-based detection to override Yahoo's incorrect API data
                # Settings are auto-loaded from league_settings JSON files
                df = playoff_flags_module.detect_playoffs_by_seed(df)
                df = playoff_flags_module.mark_playoff_rounds(df)
                df = playoff_flags_module.mark_champions_and_sackos(df)
            except Exception as fallback_error:
                _safe_print(f"[ERROR] Fallback also failed: {fallback_error}. Skipping playoff normalization.")

    # Create aliases for streak columns (backwards compatibility)
    if 'win_streak' in df.columns:
        df['winning_streak'] = df['win_streak']
    if 'loss_streak' in df.columns:
        df['losing_streak'] = df['loss_streak']

    # Step 4: Calculate manager PPG metrics (RECALCULATE WEEKLY)
    _safe_print("[4/10] Calculating manager PPG (weekly_mean, weekly_median)...")
    try:
        df = manager_ppg.calculate_manager_ppg(df)
    except Exception as e:
        _safe_print(f"[WARN] Failed to calculate manager PPG: {e}. Skipping this step.")

    # Step 5: Calculate weekly metrics (RECALCULATE WEEKLY)
    _safe_print("[5/10] Calculating weekly league-relative metrics...")
    df = weekly_metrics.calculate_weekly_metrics(df)

    # Step 6: Calculate opponent all-play metrics (RECALCULATE WEEKLY)
    _safe_print("[6/10] Calculating opponent all-play metrics...")
    try:
        df = all_play_extended.calculate_opponent_all_play(df)
    except Exception as e:
        _safe_print(f"[WARN] Failed to calculate opponent all-play: {e}. Skipping this step.")

    # Step 7: Calculate head-to-head records (RECALCULATE WEEKLY)
    _safe_print("[7/10] Calculating head-to-head records...")
    df = head_to_head.calculate_head_to_head_records(df)

    # Step 8: Calculate comparative schedule (RECALCULATE WEEKLY)
    _safe_print("[8/10] Calculating comparative schedule (w_vs_X_sched)...")
    try:
        df = comparative_schedule.calculate_comparative_schedule(df)
    except Exception as e:
        _safe_print(f"[WARN] Failed to calculate comparative schedule: {e}. Skipping this step.")

    # Step 9: Calculate season rankings (SET-AND-FORGET if championship complete)
    _safe_print("[9/10] Calculating season rankings...")
    df = season_rankings.calculate_season_rankings(
        df,
        championship_complete=championship_complete
    )

    # Step 10: Calculate all-time rankings (RECALCULATE WEEKLY)
    _safe_print("[10/10] Calculating all-time rankings...")
    try:
        df = season_rankings.calculate_alltime_rankings(df)
    except Exception as e:
        _safe_print(f"[WARN] Failed to calculate all-time rankings: {e}. Skipping this step.")

    # ------------------------------------------------------------
    # Inflation Rate (year-over-year scoring normalization)
    # ------------------------------------------------------------
    _safe_print("[11/11] Calculating inflation rate...")
    if "team_points" in df.columns and "year" in df.columns:
        # Calculate average team_points per year (exclude nulls/zeros)
        year_means = df[df["team_points"].notna() & (df["team_points"] > 0)].groupby("year")["team_points"].mean()

        if not year_means.empty and len(year_means) > 0:
            # Use earliest year WITH DATA as base (inflation_rate = 1.0)
            valid_years = year_means[year_means > 0]
            if len(valid_years) > 0:
                base_year = int(valid_years.index.min())
                base_mean = float(valid_years.loc[base_year])

                # Calculate inflation_rate for each year relative to base year
                infl_map = {int(y): float(m) / base_mean for y, m in valid_years.items()}
                df["inflation_rate"] = df["year"].map(infl_map).fillna(1.0).astype(float)
                _safe_print(f"[inflation] Base year: {base_year}, base avg: {base_mean:.2f} pts/game")
                _safe_print(f"[inflation] Calculated inflation rates for {len(infl_map)} years")

                # Log inflation rates for each year for debugging
                for y in sorted(infl_map.keys()):
                    _safe_print(f"[inflation]   {y}: {infl_map[y]:.3f} ({year_means.loc[y]:.2f} pts/game)")
            else:
                df["inflation_rate"] = 1.0
                _safe_print("[inflation] No valid year averages found, defaulting all inflation_rate to 1.0")
        else:
            df["inflation_rate"] = 1.0
            _safe_print("[inflation] No year data available, defaulting inflation_rate to 1.0")
    else:
        df["inflation_rate"] = 1.0
        _safe_print("[inflation] Missing required columns, defaulting inflation_rate to 1.0")

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
            championship_complete=args.championship_complete
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
