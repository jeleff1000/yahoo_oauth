#!/usr/bin/env python3
"""
Playoff Normalization Module

Normalizes playoff/consolation flags, ensures bracket columns exist,
computes seeds + rounds, and handles fallback inference.

Extracted from cumulative_stats.py for modularity.
"""
from __future__ import annotations

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Add modules directory to path for imports
_modules_dir = Path(__file__).parent
sys.path.insert(0, str(_modules_dir))


def normalize_playoff_data(df: pd.DataFrame, data_directory: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize playoff/consolation flags; ensure bracket columns always exist;
    compute seeds + rounds via normal path; fallback to row-count-based inference.

    This function:
    1. Ensures all playoff-related columns exist
    2. Uses seed-based detection to correctly identify playoffs vs consolation
    3. Marks playoff rounds and simulates brackets
    4. Calculates weekly head-to-head columns
    5. Computes season summaries and all-time aggregates
    6. Validates data consistency

    Args:
        df: DataFrame with matchup data
        data_directory: Path to league data directory (for finding league settings)

    Returns:
        DataFrame with normalized playoff data
    """
    df = df.copy()

    # ---- Ensure flags exist ----
    for c in ("is_playoffs", "is_consolation"):
        if c not in df.columns:
            df[c] = 0
    df["is_playoffs"] = pd.to_numeric(df["is_playoffs"], errors="coerce").fillna(0).astype(int)
    df["is_consolation"] = pd.to_numeric(df["is_consolation"], errors="coerce").fillna(0).astype(int)

    # ---- Bracket columns always present ----
    int_cols = ["playoff_week_index", "playoff_round_num", "quarterfinal", "semifinal", "championship",
                "placement_game", "placement_rank", "consolation_semifinal", "consolation_final",
                "champion", "sacko"]
    str_cols = ["playoff_round", "consolation_round"]
    for c in int_cols:
        if c not in df.columns:
            df[c] = 0
    for c in str_cols:
        if c not in df.columns:
            df[c] = ""

    # Helpful placeholders
    for c in ("final_playoff_seed", "playoff_start_week", "num_playoff_teams"):
        if c not in df.columns:
            df[c] = np.nan
    if "postseason" not in df.columns:
        df["postseason"] = 0

    # ---- Normal path (seed-based) ----
    try:
        # Import modules
        try:
            import cumulative_records as _cumulative_records
        except ImportError:
            from . import cumulative_records as _cumulative_records

        try:
            import playoff_flags as _playoff_flags
        except ImportError:
            from . import playoff_flags as _playoff_flags

        try:
            import playoff_bracket as _playoff_bracket
        except ImportError:
            from . import playoff_bracket as _playoff_bracket

        # Calculate cumulative records (includes final_playoff_seed calculation)
        df = _cumulative_records.calculate_cumulative_records(df, data_directory=data_directory)

        # Use seed-based detection to correctly identify playoffs vs consolation
        df = _playoff_flags.detect_playoffs_by_seed(df, settings_dir=data_directory)

        # Mark playoff rounds (championship + consolation brackets)
        df = _playoff_flags.mark_playoff_rounds(df, data_directory=data_directory)

        # Simulate playoff brackets to determine champion, sacko, and correct placement_rank
        df = _playoff_bracket.simulate_playoff_brackets(df, data_directory=data_directory)

        # FINAL GUARDRAIL: Enforce mutual exclusivity
        df.loc[df["is_consolation"] == 1, "is_playoffs"] = 0
        df.loc[df["is_playoffs"] == 1, "is_consolation"] = 0

        # Ensure round label columns exist and are clean
        if 'playoff_round' not in df.columns:
            df['playoff_round'] = ""
        if 'consolation_round' not in df.columns:
            df['consolation_round'] = ""

        df['playoff_round'] = df['playoff_round'].fillna("").astype(str)
        df['consolation_round'] = df['consolation_round'].fillna("").astype(str)

        # Clear conflicting labels
        playoff_mask = df['is_playoffs'] == 1
        consolation_mask = df['is_consolation'] == 1

        df.loc[playoff_mask, 'consolation_round'] = ""
        df.loc[playoff_mask, 'consolation_semifinal'] = 0
        df.loc[playoff_mask, 'consolation_final'] = 0

        df.loc[consolation_mask, 'playoff_round'] = ""
        df.loc[consolation_mask, 'quarterfinal'] = 0
        df.loc[consolation_mask, 'semifinal'] = 0
        df.loc[consolation_mask, 'championship'] = 0

        print("[INFO] normalize_playoff_data: normal seed-based playoff detection succeeded")

    except Exception as e:
        print(f"[WARN] normalize_playoff_data: normal path failed -> {e}")
        df = _fallback_playoff_inference(df)

    # Calculate weekly head-to-head columns
    df = _calculate_weekly_h2h(df)

    # Calculate season summaries
    df = _calculate_season_summaries(df)

    # Calculate all-time aggregates
    df = _calculate_all_time_aggregates(df)

    # Validate data consistency
    _validate_playoff_data(df)

    # Ensure champion/sacko columns exist
    for col in ["champion", "sacko"]:
        if col not in df.columns:
            df[col] = 0

    return df


def _fallback_playoff_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback playoff inference using row-count collapse."""
    try:
        need = {"league_id", "year", "week"}
        if not need.issubset(df.columns):
            print("[WARN] fallback: missing league_id/year/week; cannot infer playoffs")
            return df

        g = (df.groupby(["league_id", "year", "week"]).size()
             .reset_index(name="rows"))
        modal = (g.groupby(["league_id", "year"])["rows"]
                 .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.max())
                 .rename("regular_rows").reset_index())
        g = g.merge(modal, on=["league_id", "year"], how="left")
        g["is_playoff_week"] = (g["rows"] < g["regular_rows"]).astype(int)

        start = (g[g["is_playoff_week"] == 1]
                 .groupby(["league_id", "year"])["week"].min()
                 .rename("playoff_start_week").reset_index())

        df = df.merge(start, on=["league_id", "year"], how="left")
        df["postseason"] = np.where(
            df["playoff_start_week"].notna() & (df["week"] >= df["playoff_start_week"]), 1, 0)
        df.loc[df["postseason"] == 1, "is_playoffs"] = 1

        df["playoff_week_index"] = np.where(
            df["postseason"] == 1,
            (df["week"] - df["playoff_start_week"] + 1).clip(lower=1),
            0).astype(int)
        df["playoff_round_num"] = df["playoff_week_index"]

        def _round_name(i):
            if i <= 0:
                return ""
            return {1: "quarterfinal", 2: "semifinal", 3: "championship"}.get(int(i), f"playoff_round_{int(i)}")

        df["playoff_round"] = df["playoff_week_index"].map(_round_name).fillna("")
        df["quarterfinal"] = (df["playoff_round"] == "quarterfinal").astype(int)
        df["semifinal"] = (df["playoff_round"] == "semifinal").astype(int)
        df["championship"] = (df["playoff_round"] == "championship").astype(int)

        # Mark champion if a 'win' col exists and final week has <=2 rows
        df = df.merge(g[["league_id", "year", "week", "rows"]], on=["league_id", "year", "week"], how="left")
        if "win" in df.columns:
            df["champion"] = np.where(
                (df["championship"] == 1) & (df["rows"] <= 2) & (df["win"] == 1), 1, df["champion"]
            ).astype(int)

        print("[INFO] normalize_playoff_data: used fallback playoff inference (row-count collapse).")

    except Exception as fallback_e:
        print(f"[WARN] fallback playoff inference failed -> {fallback_e}")

    return df


def _calculate_weekly_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weekly head-to-head columns based on same-week points."""
    pts_col = next((c for c in ['team_points', 'points_for', 'pf'] if c in df.columns), None)
    if pts_col is None:
        return df

    try:
        try:
            from head_to_head import calculate_weekly_one_vs_all as _hh_func
        except ImportError:
            try:
                from head_to_head import calculate_head_to_head_records as _hh_func
            except ImportError:
                from . import head_to_head
                _hh_func = head_to_head.calculate_head_to_head_records
        df = _hh_func(df)
    except Exception:
        # Safe inline fallback
        managers = sorted(pd.unique(df.get("manager", pd.Series(dtype=object)).dropna()))
        tokens = {m: re.sub(r"[^a-zA-Z0-9]+", "_", str(m).strip().lower()).strip("_") or "na" for m in managers}
        for m in managers:
            df[f"w_vs_{tokens[m]}"] = 0
        for c in ['year', 'week']:
            if c not in df.columns:
                df[c] = -1
        df[pts_col] = pd.to_numeric(df[pts_col], errors="coerce")
        for (y, w), g in df.groupby(['year', 'week']):
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

    return df


def _calculate_season_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate season summary columns."""
    # Preserve manager names
    _manager_snapshot = df['manager'].copy() if 'manager' in df.columns else None

    needed = [
        'final_wins', 'final_losses', 'final_regular_wins', 'final_regular_losses',
        'season_mean', 'season_median', 'manager_season_ranking',
        'champion', 'semifinal', 'quarterfinal', 'sacko'
    ]

    # Only compute if any of these are missing
    if not any(c not in df.columns for c in needed):
        return df

    # Ensure win/loss/tie are numeric
    for c in ['win', 'loss', 'tie']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # Ensure manager_year exists
    if 'manager_year' not in df.columns and 'manager' in df.columns and 'year' in df.columns:
        df['manager_year'] = df['manager'].astype(str) + '_' + df['year'].astype(str)

    if 'manager_year' not in df.columns:
        return df

    # Non-consolation mask
    non_consol = (df['is_consolation'].fillna(0).astype(int) != 1)

    # Determine points column
    pts_col = next((c for c in ['team_points', 'points_for', 'pf'] if c in df.columns), None)
    pts_series = pd.to_numeric(df[pts_col], errors='coerce').fillna(0.0) if pts_col else pd.Series(0.0, index=df.index)

    # Regular-season totals (exclude consolation)
    reg = (
        df.assign(
            win_nc=(df['win'] * non_consol.astype(int)) if 'win' in df.columns else 0,
            loss_nc=(df['loss'] * non_consol.astype(int)) if 'loss' in df.columns else 0,
            pts_nc=(pts_series * non_consol.astype(int))
        )
        .groupby(['year', 'manager_year', 'manager'], dropna=False)
        .agg(final_regular_wins=('win_nc', 'sum'),
             final_regular_losses=('loss_nc', 'sum'),
             season_mean=('pts_nc', 'mean'),
             season_median=('pts_nc', 'median'))
        .reset_index()
    )

    # Full season totals
    if 'win' in df.columns and 'loss' in df.columns:
        full = (
            df.groupby(['year', 'manager_year', 'manager'], dropna=False)
            .agg(final_wins=('win', 'sum'),
                 final_losses=('loss', 'sum'))
            .reset_index()
        )
    else:
        full = df[['year', 'manager_year', 'manager']].drop_duplicates()
        full['final_wins'] = 0
        full['final_losses'] = 0

    # Drop old columns to prevent merge conflicts
    reg_cols = ['final_regular_wins', 'final_regular_losses', 'season_mean', 'season_median']
    full_cols = ['final_wins', 'final_losses']
    df = df.drop(columns=[c for c in reg_cols + full_cols if c in df.columns])

    # Merge aggregates back
    df = df.merge(reg, on=['year', 'manager_year', 'manager'], how='left')
    df = df.merge(full, on=['year', 'manager_year', 'manager'], how='left')

    # Rank manager's weekly performance within their season
    if pts_col is not None:
        df['manager_season_ranking'] = (
            df.groupby(['manager', 'year'])[pts_col]
            .rank(method='dense', ascending=False)
            .astype(int)
        )
    else:
        df['manager_season_ranking'] = pd.NA

    # Ensure playoff-result flags exist
    for col in ['champion', 'semifinal', 'quarterfinal', 'sacko']:
        if col not in df.columns:
            df[col] = 0

    # Restore manager names if needed
    if _manager_snapshot is not None and df['manager'].isna().any():
        df['manager'] = df['manager'].fillna(_manager_snapshot)

    return df


def _calculate_all_time_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all-time aggregates (gp, w, l, t, win%)."""
    if 'manager' not in df.columns:
        return df

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

    # Drop old columns
    all_time_cols = ['manager_all_time_gp', 'manager_all_time_wins', 'manager_all_time_losses',
                     'manager_all_time_ties', 'manager_all_time_win_pct']
    df = df.drop(columns=[c for c in all_time_cols if c in df.columns])

    # Merge aggregates back
    df = df.merge(all_time, on='manager', how='left')

    return df


def _validate_playoff_data(df: pd.DataFrame) -> None:
    """Validate playoff data consistency."""
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
