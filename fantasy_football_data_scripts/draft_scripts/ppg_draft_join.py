#!/usr/bin/env python3
"""
ppg_draft_join.py

Join season-level scoring onto draft data.

For each `player_year` present in draft.parquet, this script looks up the
corresponding `points` and `ppg_season` from players_by_year.parquet and merges
them into the draft dataframe.

Both source Parquet files are located via *relative* paths based on this file's
location:

  <repo_root>/
    fantasy_football_data_downloads/
      fantasy_football_data/                   # data_dir
        draft.parquet
        players_by_year.parquet
      fantasy_football_data_scripts/
        draft_scripts/
          ppg_draft_join.py    <-- this script

Usage:
  python ppg_draft_join.py
  python ppg_draft_join.py --dry-run
  python ppg_draft_join.py --output draft_with_ppg.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd


def script_paths() -> dict[str, Path]:
    """
    Resolve all paths relative to this script's location.
    """
    this_file = Path(__file__).resolve()
    # .../fantasy_football_data_scripts/draft_scripts
    scripts_dir = this_file.parent
    # .../fantasy_football_data_scripts
    scripts_root = scripts_dir.parent
    # .../fantasy_football_data_downloads
    downloads_root = scripts_root.parent

    data_dir = downloads_root / "fantasy_football_data"
    draft_parquet = data_dir / "draft.parquet"
    players_by_year_parquet = data_dir / "players_by_year.parquet"

    return {
        "data_dir": data_dir,
        "draft_parquet": draft_parquet,
        "players_by_year_parquet": players_by_year_parquet,
    }


def ensure_player_year(df: pd.DataFrame, who: str) -> pd.DataFrame:
    """
    Ensure a 'player_year' column exists. If missing, try to construct it from
    'player' and 'year' if available.
    """
    if "player_year" in df.columns:
        return df

    if {"player", "year"}.issubset(df.columns):
        out = df.copy()
        # Make sure year is string so concatenation is safe
        out["player_year"] = out["player"].astype(str) + "_" + out["year"].astype(str)
        return out

    raise KeyError(
        f"{who} is missing 'player_year' and also lacks ('player','year') to construct it."
    )


def coalesce_first_nonnull(series: pd.Series) -> Optional[float]:
    """
    Return the first non-null value in a Series, or None if all null.
    """
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else None


def load_and_prepare_players(players_path: Path) -> pd.DataFrame:
    """
    Load players_by_year.parquet and reduce to unique rows per player_year
    with the fields we need: points, ppg_season.
    Handles (rare) duplicates by taking the first non-null for each column.
    """
    ply = pd.read_parquet(players_path)
    ply = ensure_player_year(ply, who="players_by_year.parquet")

    needed_cols = ["player_year", "points", "ppg_season"]
    missing = [c for c in needed_cols if c not in ply.columns]
    if missing:
        raise KeyError(
            f"players_by_year.parquet missing required columns: {missing}"
        )

    # Deduplicate to one row per player_year (coalesce first non-null)
    grouped = (
        ply[needed_cols]
        .groupby("player_year", as_index=False)
        .agg({
            "points": coalesce_first_nonnull,
            "ppg_season": coalesce_first_nonnull,
        })
    )

    # Ensure numeric types where possible
    for c in ("points", "ppg_season"):
        grouped[c] = pd.to_numeric(grouped[c], errors="coerce")

    return grouped


def merge_points_onto_draft(
    draft_path: Path,
    players_grouped: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load draft.parquet, ensure player_year, then left-join points and ppg_season.
    """
    draft = pd.read_parquet(draft_path)
    draft = ensure_player_year(draft, who="draft.parquet")

    # Columns to bring over
    attach_cols = ["player_year", "points", "ppg_season"]

    merged = draft.merge(
        players_grouped[attach_cols],
        on="player_year",
        how="left",
        suffixes=("", "_from_players"),
    )

    # If draft already had 'points' or 'ppg_season', prefer the players_by_year values.
    # Otherwise, the left-joined columns already occupy 'points'/'ppg_season'.
    # If there were collisions, they'd be in *_from_players — handle that:
    for col in ("points", "ppg_season"):
        from_col = f"{col}_from_players"
        if from_col in merged.columns and col in merged.columns:
            merged[col] = merged[from_col].combine_first(merged[col])
            merged.drop(columns=[from_col], inplace=True)
        elif from_col in merged.columns and col not in merged.columns:
            merged.rename(columns={from_col: col}, inplace=True)

    return merged


def save_with_backup(df: pd.DataFrame, dst_parquet: Path) -> None:
    """
    Overwrite the target Parquet, but first make a timestamped backup next to it.
    """
    if dst_parquet.exists():
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = dst_parquet.with_suffix(f".backup_{timestamp}.parquet")
        dst_parquet.replace(backup)
        print(f"💾 Existing file backed up to: {backup.name}")

    df.to_parquet(dst_parquet, index=False)
    print(f"✅ Wrote updated file: {dst_parquet.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Join points + ppg_season onto draft.parquet by player_year.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write any files; just report changes.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output parquet path (relative to draft.parquet directory). If omitted and not dry-run, overwrites draft.parquet (with backup).",
    )
    args = ap.parse_args()

    paths = script_paths()

    draft_parquet = paths["draft_parquet"]
    players_by_year_parquet = paths["players_by_year_parquet"]

    print("📂 Data directory:", paths["data_dir"])
    print("📄 Draft file:    ", draft_parquet)
    print("📄 Players file:  ", players_by_year_parquet)

    players_grouped = load_and_prepare_players(players_by_year_parquet)
    out_df = merge_points_onto_draft(draft_parquet, players_grouped)

    # Report coverage
    total = len(out_df)
    matched = out_df["points"].notna().sum()
    print(f"🔗 Matched player_year rows: {matched} / {total} ({matched/total:.1%})")

    if args.dry_run:
        print("🧪 Dry run complete. No files written.")
        return

    # Determine destination
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = draft_parquet.parent / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)
        print(f"✅ Wrote output file: {out_path}")
    else:
        # In-place overwrite with backup
        save_with_backup(out_df, draft_parquet)


if __name__ == "__main__":
    main()
