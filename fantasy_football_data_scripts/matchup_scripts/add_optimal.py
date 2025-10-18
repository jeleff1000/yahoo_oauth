#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import os
from md.md_utils import df_from_md_or_parquet

# --------------------------------------------------------------------------------------
# Paths (all relative to this file's location)
# --------------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent  # .../fantasy_football_data_scripts/matchup_scripts
REPO_ROOT = SCRIPT_DIR.parent.parent  # .../fantasy_football_data_downloads
DATA_DIR = REPO_ROOT / "fantasy_football_data"

DEFAULT_MATCHUP = DATA_DIR / "matchup.parquet"
DEFAULT_PLAYER = DATA_DIR / "player.parquet"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _to_str(s: pd.Series) -> pd.Series:
    """
    Normalize to consistent string keys for joins:
    - Cast to pandas 'string' dtype
    - Preserve NA as <NA>
    - Trim surrounding whitespace
    """
    return (
        s.astype("string")
         .fillna(pd.NA)
         .str.strip()
    )

def _safe_numeric(s: pd.Series) -> pd.Series:
    """
    Coerce to numeric; invalid parses become NaN (preserve missingness).
    """
    return pd.to_numeric(s, errors="coerce")

def load_parquet(path: Path, name: str) -> pd.DataFrame:
    """
    Load a parquet file or exit with a readable error.
    """
    # If MotherDuck is configured, prefer loading from MD (table name inferred from path stem)
    if os.getenv("MOTHERDUCK_TOKEN"):
        try:
            return df_from_md_or_parquet(path.stem, path)
        except Exception:
            pass

    if not path.exists():
        print(f"[ERROR] {name} parquet not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[ERROR] Failed reading {name} at {path}: {e}", file=sys.stderr)
        sys.exit(1)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Add optimal_points to matchup from player (first row per manager_week), then compute:\n"
            "- optimal_ppg_season (per manager_year)\n"
            "- rolling_optimal_points (by week within manager_year)\n"
            "- total_optimal_points (season total per manager_year)\n"
            "- optimal_points_all_time (per manager across all years)\n"
            "- optimal_win / optimal_loss by comparing to opponent_week's optimal_points"
        )
    )
    ap.add_argument(
        "--matchup", type=str, default=str(DEFAULT_MATCHUP),
        help="Path to matchup.parquet (default: relative to repo)",
    )
    ap.add_argument(
        "--player", type=str, default=str(DEFAULT_PLAYER),
        help="Path to player.parquet (default: relative to repo)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Compute and report but do not write the parquet",
    )
    args = ap.parse_args()

    matchup_path = Path(args.matchup).resolve()
    player_path = Path(args.player).resolve()

    # ---------------- Load ----------------
    m = load_parquet(matchup_path, "matchup")
    p = load_parquet(player_path, "player")

    # ---------------- Sanity checks ----------------
    # Required source columns per your spec
    needed_m_cols = ["manager_week", "manager_year", "manager", "week", "year", "opponent_week"]
    missing = [c for c in needed_m_cols if c not in m.columns]
    if missing:
        print(f"[ERROR] matchup missing required column(s): {missing}", file=sys.stderr)
        sys.exit(1)

    if "optimal_points" not in p.columns:
        print("[ERROR] player missing required column: optimal_points", file=sys.stderr)
        sys.exit(1)

    # ---------------- Normalize join keys ----------------
    # Use the existing manager_week column which has already been standardized
    # to strip spaces from manager names and use consistent cumulative_week values

    # Verify manager_week exists in both dataframes
    if "manager_week" not in m.columns:
        print(f"[ERROR] matchup missing required column: manager_week", file=sys.stderr)
        sys.exit(1)
    if "manager_week" not in p.columns:
        print(f"[ERROR] player missing required column: manager_week", file=sys.stderr)
        sys.exit(1)

    # Ensure manager_week is string type for joining
    m["manager_week"] = _to_str(m["manager_week"])
    p["manager_week"] = _to_str(p["manager_week"])

    # Also preserve opponent_week for later use
    if "opponent_week" in m.columns:
        m["opponent_week"] = _to_str(m["opponent_week"])

    # Diagnostics
    print("[DEBUG] First 20 manager_week keys in matchup:", m["manager_week"].dropna().unique()[:20])
    print("[DEBUG] First 20 manager_week keys in player:", p["manager_week"].dropna().unique()[:20])
    print("[DEBUG] Sample rows from matchup:")
    print(m[['manager', 'year', 'week', 'manager_week']].head(5))
    print("[DEBUG] Sample rows from player:")
    print(p[['manager', 'year', 'week', 'manager_week']].head(5) if all(c in p.columns for c in ['manager', 'year', 'week']) else p[['manager_week']].head(5))

    # ---------------- Take FIRST player row per manager_week ----------------
    # We only need one value of optimal_points per manager_week combo.
    # Take the first non-null occurrence.
    # If multiple exist (multiple players for same manager/week), this picks the first row.
    p_first = (
        p.loc[~p["manager_week"].isna(), ["manager_week", "optimal_points"]]
         .dropna(subset=["manager_week"])
         .copy()
    )
    p_first = p_first.groupby("manager_week", as_index=False).first()

    print(f"[DEBUG] Player data aggregated: {len(p_first)} unique manager_week combinations")

    # ---------------- Merge optimal_points into matchup ----------------
    # If matchup already had an optimal_points column, we preserve values and
    # prefer the newly merged value when present.
    before_nonnull = m.get("optimal_points", pd.Series([np.nan]*len(m))).notna().sum() if "optimal_points" in m.columns else 0
    m = m.merge(p_first, on="manager_week", how="left", suffixes=("_old", ""))

    # If we had an old optimal_points column, use new value when available, otherwise keep old
    if "optimal_points_old" in m.columns:
        # Prefer new value from player, fall back to old value from matchup if new is NA
        m["optimal_points"] = m["optimal_points"].fillna(m["optimal_points_old"])
        m.drop(columns=["optimal_points_old"], inplace=True)

    # Ensure numeric for downstream calcs
    m["optimal_points"] = _safe_numeric(m["optimal_points"])

    after_nonnull = m["optimal_points"].notna().sum()
    print(f"[INFO] optimal_points filled/available rows: {before_nonnull} -> {after_nonnull} of {len(m)}")

    if after_nonnull == 0:
        print("[WARN] No optimal_points data was merged! Check that player table has data for the same managers/years/weeks as matchup.", file=sys.stderr)

    # Drop the temporary join key
    m.drop(columns=["manager_week"], errors="ignore", inplace=True)

    # ---------------- Prepare order within season ----------------
    # rolling is defined "sum of all optimal_points in a year with week <= current week (and year matches)"
    # We'll calculate within manager_year; 'week' must be numeric for correct ordering.
    m["week"] = _safe_numeric(m["week"])
    if m["week"].isna().all():
        print("[WARN] All 'week' values are NaN; rolling will treat all as 'last'.", file=sys.stderr)

    # ---------------- optimal_ppg_season ----------------
    # PPG = (sum optimal_points for the season) / (count of weeks with non-null optimal_points)
    # This value is constant for all rows within the same manager_year.
    grp_season = m.groupby("manager_year", dropna=False)
    season_sum = grp_season["optimal_points"].transform("sum", engine="cython", numeric_only=True)
    season_cnt = grp_season["optimal_points"].transform(lambda s: s.notna().sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        m["optimal_ppg_season"] = season_sum / season_cnt.replace(0, np.nan)

    # ---------------- rolling_optimal_points ----------------
    # Cumulative sum by manager_year ordered by week (NaN weeks treated as very large so they come last).
    # Missing optimal_points count as 0 for rolling purposes.
    m["_week_sort"] = m["week"].fillna(10**9)
    m["_opt_pts_fill0"] = m["optimal_points"].fillna(0)

    m["rolling_optimal_points"] = (
        m.sort_values(["manager_year", "_week_sort"])
         .groupby("manager_year", dropna=False)["_opt_pts_fill0"]
         .cumsum()
         .values
    )

    # ---------------- total_optimal_points (season total, not rolling) ----------------
    m["total_optimal_points"] = season_sum.fillna(0)

    # ---------------- optimal_points_all_time (per manager across ALL years) ----------------
    # Note: using transform to write the per-manager total back to each row
    grp_all_time = m.groupby("manager", dropna=False)["optimal_points"].transform("sum", engine="cython", numeric_only=True)
    m["optimal_points_all_time"] = grp_all_time.fillna(0)

    # ---------------- Optimal Win / Loss ----------------
    # For each row, compare its optimal_points to the row whose manager_week == this row's opponent_week.
    # We self-join on opponent_week to retrieve that row's optimal_points as opponent_optimal_points.
    opp = (
        m[["manager_week", "optimal_points"]]
        .rename(columns={
            "manager_week": "opponent_week",
            "optimal_points": "opponent_optimal_points"
        })
    )

    # Left-join so we keep all matchup rows; opponent_optimal_points may be NaN if not found.
    m = m.merge(opp, on="opponent_week", how="left")

    # Ensure 'opponent_optimal_points' exists after merge (if no matches, pandas may not create it)
    if "opponent_optimal_points" not in m.columns:
        m["opponent_optimal_points"] = np.nan

    # Binary flags:
    # - optimal_win: 1 if own optimal_points > opponent_optimal_points, else 0
    # - optimal_loss: 1 if own optimal_points < opponent_optimal_points, else 0
    # Ties or any NaN comparison => both 0.
    m["optimal_win"] = np.where(m["optimal_points"].gt(m["opponent_optimal_points"]), 1, 0)
    m["optimal_loss"] = np.where(m["optimal_points"].lt(m["opponent_optimal_points"]), 1, 0)

    # ---------------- Cleanup temp cols ----------------
    m.drop(columns=["_week_sort", "_opt_pts_fill0"], errors="ignore", inplace=True)

    # After all merges, drop columns with suffixes or duplicate optimal columns
    # Make sure we keep the main optimal_points column but drop any _from_player, _old, _x, _y variants
    cols_to_drop = []
    for c in m.columns:
        if c == "optimal_points":
            continue  # Keep the main column
        if any(c.endswith(suffix) for suffix in ['_from_player', '_old', '_x', '_y']):
            cols_to_drop.append(c)
        elif c.startswith("optimal_points_") and c != "optimal_points_all_time":
            # Drop optimal_points_from_player or similar, but keep optimal_points_all_time
            cols_to_drop.append(c)

    if cols_to_drop:
        print(f"[DEBUG] Dropping cleanup columns: {cols_to_drop}")
        m.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # ---------------- Save ----------------
    if args.dry_run:
        print("[DRY RUN] Not writing parquet.")
        print("Columns now present:", ", ".join([
            c for c in [
                "optimal_points",
                "optimal_ppg_season",
                "rolling_optimal_points",
                "total_optimal_points",
                "optimal_points_all_time",
                "optimal_win",
                "optimal_loss",
                "opponent_optimal_points",  # helpful for debugging/QA
            ] if c in m.columns
        ]))
        # Show a few joined comparisons for quick sanity-check
        preview_cols = [
            "manager", "manager_year", "week", "manager_week", "opponent_week",
            "optimal_points", "opponent_optimal_points", "optimal_win", "optimal_loss"
        ]
        print(m[preview_cols].head(12))
        return

    try:
        m.to_parquet(matchup_path, index=False)
        print(f"[OK] Wrote updated matchup parquet: {matchup_path}")
    except Exception as e:
        print(f"[ERROR] Failed writing updated matchup parquet: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
