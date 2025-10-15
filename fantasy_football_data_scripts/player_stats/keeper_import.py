#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Iterable, Dict

import numpy as np
import pandas as pd

# =========================
# Config - Relative Paths
# =========================
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = REPO_ROOT / "fantasy_football_data"

PLAYER_PATH = DATA_DIR / "player.parquet"
DRAFT_PATH = DATA_DIR / "draft.parquet"
TRANSACTION_PATH = DATA_DIR / "transactions.parquet"

OUTPUT_PLAYER_PARQUET = DATA_DIR / "player.parquet"
OUTPUT_PLAYER_CSV = DATA_DIR / "player.csv"
OUTPUT_KEEPER_CSV = DATA_DIR / "keeper_prices.csv"
UPDATE_COLS = [
    "cost",
    "is_keeper_status",
    "faab_bid",
    "keeper_price",
    "kept_next_year",
    "avg_cost_next_year",
    "total_points_next_year",
]

_SNAKE_FIXES = [
    (r"\s+", "_"),
    (r"-", "_"),
    (r"%", "pct"),
    (r"\+", "plus"),
    (r"__", "_"),
]


def to_snake(name: str) -> str:
    s = str(name).strip().lower()
    for pat, repl in _SNAKE_FIXES:
        s = re.sub(pat, repl, s)
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names in-place to avoid copy"""
    df.columns = [to_snake(c) for c in df.columns]
    return df


def dedupe_and_coalesce(df: pd.DataFrame) -> pd.DataFrame:
    """
    If duplicate column names exist, coalesce them left->right and keep a single column.
    """
    cols = pd.Index(df.columns)
    dup_names = cols[cols.duplicated()].unique().tolist()

    if not dup_names:
        return df

    for name in dup_names:
        block = df.loc[:, df.columns == name]
        combined = block.bfill(axis=1).iloc[:, 0].infer_objects(copy=False)
        to_drop = block.columns[1:]
        df = df.drop(columns=to_drop)
        df[name] = combined
    return df


def manager_year_is_noowner(val) -> bool:
    s = str(val)
    collapsed = "".join(s.split()).lower()
    return "noowner" in collapsed


def increment_year_identifier(value) -> Optional[str]:
    s = "" if value is None else str(value)
    m = re.match(r"^(.*?)(\d{4})$", s)
    if not m:
        return None
    prefix, year_str = m.groups()
    try:
        return f"{prefix}{int(year_str) + 1}"
    except ValueError:
        return None


def first_present(d: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        if cand in d.columns:
            return cand
    return None


def detect_and_create_cost_column(draft_df: pd.DataFrame) -> pd.DataFrame:
    preferred_order = [
        "cost", "draft_cost", "original_cost", "purchase_price",
        "average_cost", "avg_cost", "averagecost",
    ]
    found_src = first_present(draft_df, preferred_order)
    if "cost" not in draft_df.columns:
        draft_df["cost"] = pd.to_numeric(draft_df[found_src], errors="coerce") if found_src else 0
    return draft_df


def vectorized_keeper_price(cost: pd.Series, faab: pd.Series, is_keeper: pd.Series) -> pd.Series:
    cost = pd.to_numeric(cost, errors="coerce").fillna(0.0)
    faab = pd.to_numeric(faab, errors="coerce").fillna(0.0)
    is_keeper = pd.to_numeric(is_keeper, errors="coerce").fillna(0).astype(int)

    # Base calculation: 1.5 * cost + 7.5 for keepers, otherwise just cost
    base_price = 1.5 * cost + 7.5
    price = cost.where(is_keeper == 0, base_price)

    # Take the greater of: draft cost OR half of FAAB bid
    half_faab = faab / 2.0
    price = pd.concat([price, half_faab], axis=1).max(axis=1)

    # Minimum price is 1
    price = price.clip(lower=1.0)
    return pd.Series(np.floor(price.to_numpy() + 0.5).astype(int), index=price.index)


def _coalesce_serieslike(x):
    if isinstance(x, pd.DataFrame):
        return x.bfill(axis=1).iloc[:, 0]
    return x


def ensure_numeric(d: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """Convert columns to numeric types in-place where possible"""
    for col, dtype in schema.items():
        if col not in d.columns:
            continue
        col_data = _coalesce_serieslike(d[col])
        if dtype == "float64":
            d[col] = pd.to_numeric(col_data, errors="coerce").astype("float64").fillna(0.0)
        elif dtype == "Int64":
            tmp = pd.to_numeric(col_data, errors="coerce").fillna(0)
            d[col] = tmp.astype("Int64")
    return d


def rename_hidden_manager(df: pd.DataFrame) -> pd.DataFrame:
    if "manager" in df.columns:
        df.loc[df["manager"] == "--hidden--", "manager"] = "Ilan"
    return df


def main():
    print(f"Loading data from: {DATA_DIR}")

    # Load and process - avoid unnecessary copies
    print("Loading player data...")
    player_df = pd.read_parquet(PLAYER_PATH)
    normalize_columns(player_df)
    dedupe_and_coalesce(player_df)
    rename_hidden_manager(player_df)

    print("Loading draft data...")
    draft_df = pd.read_parquet(DRAFT_PATH)
    normalize_columns(draft_df)
    dedupe_and_coalesce(draft_df)
    rename_hidden_manager(draft_df)

    print("Loading transaction data...")
    trans_df = pd.read_parquet(TRANSACTION_PATH)
    normalize_columns(trans_df)
    dedupe_and_coalesce(trans_df)
    rename_hidden_manager(trans_df)

    print(f"Loaded {len(player_df)} player rows")
    print(f"Loaded {len(draft_df)} draft rows")
    print(f"Loaded {len(trans_df)} transaction rows")

    required_player_cols = {"manager_year", "week", "player", "player_year", "rolling_point_total"}
    required_draft_cols = {"player_year", "is_keeper_status"}
    required_trans_cols = {"player_year", "faab_bid"}

    for needed, df_name, df in [
        (required_player_cols, "player_data", player_df),
        (required_draft_cols, "draft_history", draft_df),
        (required_trans_cols, "all_transactions", trans_df),
    ]:
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {df_name}: {missing}")

    print("Normalizing player_year and manager_year (removing spaces)...")
    for _df in (player_df, draft_df, trans_df):
        _df["player_year"] = _df["player_year"].astype(str).str.replace(" ", "", regex=False)
        if "manager_year" in _df.columns:
            _df["manager_year"] = _df["manager_year"].astype(str).str.replace(" ", "", regex=False)

    detect_and_create_cost_column(draft_df)

    print("Computing keeper prices per player_year...")
    draft_subset = draft_df[["player_year", "cost", "is_keeper_status"]].copy()
    draft_subset["cost"] = pd.to_numeric(draft_subset["cost"], errors="coerce").fillna(0.0)
    draft_subset["is_keeper_status"] = pd.to_numeric(draft_subset["is_keeper_status"], errors="coerce").fillna(
        0).astype(int)

    trans_agg = (
        trans_df.dropna(subset=["player_year"])
        .groupby("player_year", as_index=False)["faab_bid"].max()
    )
    trans_agg["faab_bid"] = pd.to_numeric(trans_agg["faab_bid"], errors="coerce").fillna(0.0)

    keeper_base = (
        draft_subset.merge(trans_agg, on="player_year", how="left")
        .fillna({"faab_bid": 0.0})
    )

    keeper_base["keeper_price"] = vectorized_keeper_price(
        keeper_base["cost"], keeper_base["faab_bid"], keeper_base["is_keeper_status"]
    )
    keeper_base["player_year_next"] = keeper_base["player_year"].apply(increment_year_identifier)

    print("Looking up next-year info...")
    avg_cost_col = first_present(draft_df, ["average_cost", "avg_cost", "averagecost"])
    if avg_cost_col is None:
        draft_df["__avg_cost_internal__"] = 0.0
        avg_cost_col = "__avg_cost_internal__"

    draft_next = draft_df[["player_year", "is_keeper_status", avg_cost_col]].copy()
    draft_next.rename(
        columns={
            "player_year": "player_year_next",
            "is_keeper_status": "is_keeper_status_next",
            avg_cost_col: "avg_cost_next_year",
        },
        inplace=True,
    )

    keeper_base = keeper_base.merge(draft_next, on="player_year_next", how="left")
    keeper_base["kept_next_year"] = pd.to_numeric(keeper_base.get("is_keeper_status_next", 0), errors="coerce").fillna(
        0).astype(int)
    keeper_base["avg_cost_next_year"] = pd.to_numeric(keeper_base.get("avg_cost_next_year", 0.0),
                                                      errors="coerce").fillna(0.0)

    p_next_keys = keeper_base["player_year_next"].dropna().astype(str).unique().tolist()
    if p_next_keys:
        subset_next = player_df[player_df["player_year"].astype(str).isin(p_next_keys)].copy()
        if not subset_next.empty and "rolling_point_total" in subset_next.columns:
            # Get maximum rolling point total across all weeks for each player_year
            max_points = subset_next.groupby("player_year", as_index=False)["rolling_point_total"].max()
            points_map = dict(zip(max_points["player_year"], max_points["rolling_point_total"]))
        else:
            points_map = {}
    else:
        points_map = {}

    keeper_base["total_points_next_year"] = pd.to_numeric(
        keeper_base["player_year_next"].map(points_map), errors="coerce"
    ).fillna(0.0)

    keeper_cols = [
        "player_year",
        "cost",
        "is_keeper_status",
        "faab_bid",
        "keeper_price",
        "kept_next_year",
        "avg_cost_next_year",
        "total_points_next_year",
    ]
    per_player_year = keeper_base[keeper_cols].drop_duplicates("player_year").set_index("player_year")

    print(f"Computed keeper info for {len(per_player_year)} unique player_years")

    rep_cols = player_df[["player_year", "manager_year", "player"]].drop_duplicates(subset=["player_year"])
    csv_block = rep_cols.merge(per_player_year.reset_index(), on="player_year", how="left")

    print(f"\nSaving keeper prices CSV: {OUTPUT_KEEPER_CSV}")
    csv_out_cols = [
        "manager_year",
        "player",
        "cost",
        "is_keeper_status",
        "faab_bid",
        "keeper_price",
        "kept_next_year",
        "avg_cost_next_year",
        "total_points_next_year",
    ]
    csv_block[csv_out_cols].sort_values(["manager_year", "player"]).to_csv(OUTPUT_KEEPER_CSV, index=False)

    print("Merging keeper columns to every player/week row...")

    # CRITICAL FIX: Drop existing UPDATE_COLS from player_df BEFORE merge to avoid duplicates
    cols_to_drop = [c for c in UPDATE_COLS if c in player_df.columns]
    if cols_to_drop:
        print(f"  Dropping existing columns: {cols_to_drop}")
        player_df.drop(columns=cols_to_drop, inplace=True)

    # Now merge - no suffixes needed since we dropped the duplicates
    player_df = player_df.merge(
        per_player_year.reset_index(),
        on="player_year",
        how="left"
    )

    # Check for duplicate columns and remove them
    if player_df.columns.duplicated().any():
        print("  Removing duplicate columns after merge...")
        # Keep only the first occurrence of each column name
        player_df = player_df.loc[:, ~player_df.columns.duplicated(keep='first')]

    num_schema = {
        "cost": "float64",
        "is_keeper_status": "Int64",
        "faab_bid": "float64",
        "keeper_price": "Int64",
        "kept_next_year": "Int64",
        "avg_cost_next_year": "float64",
        "total_points_next_year": "float64",
    }
    ensure_numeric(player_df, num_schema)

    print(f"\nSaving updated player data:")
    print(f"  Parquet: {OUTPUT_PLAYER_PARQUET}")
    print(f"  CSV:     {OUTPUT_PLAYER_CSV}")

    player_df.to_parquet(OUTPUT_PLAYER_PARQUET, index=False)
    player_df.to_csv(OUTPUT_PLAYER_CSV, index=False)

    print("\nDone!")
    print(f"  Keeper prices: {OUTPUT_KEEPER_CSV}")
    print(f"  Updated player data rows: {len(player_df)}")


if __name__ == "__main__":
    main()