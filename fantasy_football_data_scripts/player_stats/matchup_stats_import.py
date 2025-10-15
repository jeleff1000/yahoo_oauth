#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import matchup results into player rows by manager_week.

- Joins player.parquet to matchup.parquet on `manager_week`
- Imports (overwrites when present) these columns from matchup -> player:
    ['win', 'team_points', 'opponent_points', 'is_playoffs']
- Uses relative paths based on this file's location:
    ../../fantasy_football_data/player.parquet
    ../../fantasy_football_data/matchup.parquet

Usage:
    python matchup_stats_import.py
    python matchup_stats_import.py --dry-run
    python matchup_stats_import.py --backup
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import re

import numpy as np
import pandas as pd


# -------------------------
# Paths (relative)
# -------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent                              # .../fantasy_football_data_scripts/player_stats
ROOT_DIR = SCRIPT_DIR.parent.parent                        # .../fantasy_football_data_downloads
DATA_DIR = ROOT_DIR / "fantasy_football_data"

PLAYER_PATH = DATA_DIR / "player.parquet"
MATCHUP_PATH = DATA_DIR / "matchup.parquet"


IMPORT_COLS: List[str] = [
    "win", "loss", "team_points", "opponent_points",
    "is_playoffs", "is_consolation", "sacko", "team_made_playoffs",
    "quarterfinal", "champion", "semifinal", "championship",
    "cumulative_week"  # Import cumulative_week from matchup to ensure alignment
]
JOIN_KEY = "manager_week"


def _ensure_join_key(df: pd.DataFrame, name: str) -> pd.Series:
    """Ensure join key column exists and is string-typed."""
    if JOIN_KEY not in df.columns:
        raise KeyError(
            f"'{JOIN_KEY}' not found in {name}. "
            f"Please add it or precompute it in that parquet."
        )
    return df[JOIN_KEY].astype("string")


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    dest.write_bytes(path.read_bytes())
    return dest


def import_matchup_columns(
    player_path: Path,
    matchup_path: Path,
    import_cols: List[str],
    dry_run: bool = False,
    make_backup: bool = False,
) -> Dict[str, int]:
    # Load data
    player = pd.read_parquet(player_path)
    matchup = pd.read_parquet(matchup_path)

    # Validate join key
    player[JOIN_KEY] = _ensure_join_key(player, "player")
    matchup[JOIN_KEY] = _ensure_join_key(matchup, "matchup")

    # Keep only needed columns from matchup
    keep_cols = [JOIN_KEY] + [c for c in import_cols if c in matchup.columns]
    missing = sorted(set(import_cols) - set(keep_cols))
    if missing:
        print(f"Warning: these columns not found in matchup and will be skipped: {missing}")

    m_small = matchup[keep_cols].copy()

    # Merge
    suffix = "_from_matchup"
    merged = player.merge(m_small, on=JOIN_KEY, how="left", suffixes=("", suffix))

    # For each import col, prefer matchup value when available; otherwise keep existing
    updated_counts: Dict[str, int] = {}
    for col in import_cols:
        src_col = f"{col}{suffix}"
        if src_col not in merged.columns:
            continue

        # Count how many rows will change
        # Define "change" as (src is notna) & (player col missing OR different)
        src = merged[src_col]
        if col in merged.columns:
            before = merged[col]
            will_update_mask = src.notna() & (before.isna() | (src != before))
        else:
            before = pd.Series([np.nan] * len(merged))
            will_update_mask = src.notna()

        updated_counts[col] = int(will_update_mask.sum())

        # **SPECIAL HANDLING for cumulative_week**: Force overwrite from matchup when available
        # This ensures manager_week alignment since both use the same cumulative_week
        if col == "cumulative_week":
            # Overwrite with matchup value when available, otherwise keep existing
            merged[col] = src.where(src.notna(), merged[col] if col in merged.columns else pd.Series([np.nan]*len(merged)))
        else:
            # Apply combine_first to preserve existing non-null when matchup is null
            merged[col] = src.combine_first(merged[col] if col in merged.columns else pd.Series([np.nan]*len(merged)))

        # Drop the helper column
        merged.drop(columns=[src_col], inplace=True)

    # **CRITICAL FIX**: Ensure is_playoffs=0 when is_consolation=1
    if "is_consolation" in merged.columns and "is_playoffs" in merged.columns:
        consolation_mask = merged["is_consolation"].fillna(0).astype(int) == 1
        if consolation_mask.any():
            merged.loc[consolation_mask, "is_playoffs"] = 0
            print(f"[Fix Applied] Set is_playoffs=0 for {consolation_mask.sum()} consolation game rows")

    # **CRITICAL: Recreate manager_week after importing cumulative_week**
    # This ensures manager_week is always based on the imported cumulative_week from matchup
    if "cumulative_week" in merged.columns and "manager" in merged.columns:
        print("[Fix] Recreating manager_week after cumulative_week import to ensure alignment...")
        # Strip spaces from manager names and concatenate with cumulative_week
        merged["manager_week"] = merged.apply(
            lambda r: (
                re.sub(r"\s+", "", str(r.get('manager', ''))) + str(int(r['cumulative_week']))
                if pd.notna(r.get("manager")) and pd.notna(r.get("cumulative_week")) else pd.NA
            ),
            axis=1,
        ).astype("string")
        print(f"[Fix Applied] Recreated manager_week for all rows with manager and cumulative_week")

    # Rows that had any match (i.e., at least one imported value present)
    matched_rows = int(m_small[JOIN_KEY].nunique())
    result = {"matched_manager_weeks": matched_rows, **updated_counts}

    # Save (unless dry-run)
    if dry_run:
        print("\n[Dry Run] No files were written.")
        return result

    if make_backup and player_path.exists():
        bpath = backup_file(player_path)
        print(f"Backup created: {bpath}")

    # Write back to the same parquet
    merged.to_parquet(player_path, index=False)
    print(f"Updated file written: {player_path}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Import matchup stats into player by manager_week.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would change without writing files.")
    ap.add_argument("--backup", action="store_true", help="Create a timestamped backup of player.parquet before saving.")
    args = ap.parse_args()

    print(f"Player file : {PLAYER_PATH}")
    print(f"Matchup file: {MATCHUP_PATH}")

    stats = import_matchup_columns(
        player_path=PLAYER_PATH,
        matchup_path=MATCHUP_PATH,
        import_cols=IMPORT_COLS,
        dry_run=args.dry_run,
        make_backup=args.backup,
    )

    print("\nSummary:")
    print(f"  matched manager_weeks: {stats.get('matched_manager_weeks', 0)}")
    for c in IMPORT_COLS:
        print(f"  rows updated for {c:>16}: {stats.get(c, 0)}")


if __name__ == "__main__":
    main()
