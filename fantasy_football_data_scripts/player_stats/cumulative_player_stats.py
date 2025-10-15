#!/usr/bin/env python3
from __future__ import annotations

import sys
import re
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import polars as pl

# --------------------------------------------------------------------------------------
# Paths
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = REPO_ROOT / "fantasy_football_data"

# Direct files (no longer in player_data subfolder)
PARQUET_PATH = DATA_DIR / "player.parquet"
CSV_PATH = DATA_DIR / "player.csv"

# Scoring directory (if still needed)
SCORING_DIR = DATA_DIR / "player_data" / "yahoo_league_settings"

# --------------------------------------------------------------------------------------
# Column names we will add (if missing)
RANK_COLS = [
    "player_personal_all_time_history",
    "player_personal_season_history",
    "player_all_time_history",
    "player_season_history",
    "position_all_time_history",
    "position_season_history",
    "manager_player_all_time_history",
    "manager_player_season_history",
    "manager_position_all_time_history",
    "manager_position_season_history",
]
PCT_COLS = [c + "_percentile" for c in RANK_COLS]


# --------------------------------------------------------------------------------------
# Scoring Rules Functions
def load_scoring_rules(scoring_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Load all scoring rules JSON files and organize by year."""
    rules_by_year: Dict[int, Dict[str, Any]] = {}

    if not scoring_dir.exists():
        print(f"[warning] Scoring directory not found: {scoring_dir}")
        return rules_by_year

    for json_file in scoring_dir.glob("yahoo_full_scoring_*.json"):
        try:
            match = re.search(r"yahoo_full_scoring_(\d{4})_", json_file.name)
            if not match:
                continue

            year = int(match.group(1))

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                rules_by_year[year] = data
                print(f"[scoring] Loaded rules for year {year}")

        except Exception as e:
            print(f"[warning] Failed to load {json_file.name}: {e}")

    return rules_by_year


def load_roster_settings(scoring_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Load all roster settings JSON files and organize by year."""
    roster_by_year: Dict[int, Dict[str, Any]] = {}

    if not scoring_dir.exists():
        print(f"[warning] Roster settings directory not found: {scoring_dir}")
        return roster_by_year

    for json_file in scoring_dir.glob("yahoo_roster_*.json"):
        try:
            match = re.search(r"yahoo_roster_(\d{4})_", json_file.name)
            if not match:
                continue

            year = int(match.group(1))

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                roster_by_year[year] = data
                print(f"[roster] Loaded roster settings for year {year}")

        except Exception as e:
            print(f"[warning] Failed to load {json_file.name}: {e}")

    return roster_by_year


def find_nearest_year_rules(target_year: int, rules_by_year: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the scoring rules for the nearest available year."""
    if not rules_by_year:
        return None
    if target_year in rules_by_year:
        return rules_by_year[target_year]

    available_years = sorted(rules_by_year.keys())
    closest_year = min(available_years, key=lambda y: abs(y - target_year))

    print(f"[scoring] Using {closest_year} rules for year {target_year}")
    return rules_by_year[closest_year]


def find_nearest_year_roster(target_year: int, roster_by_year: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the roster settings for the nearest available year."""
    if not roster_by_year:
        return None
    if target_year in roster_by_year:
        return roster_by_year[target_year]

    available_years = sorted(roster_by_year.keys())
    if not available_years:
        return None

    # For years before the earliest available, use the earliest
    if target_year < available_years[0]:
        closest_year = available_years[0]
    else:
        # For later years, find nearest
        closest_year = min(available_years, key=lambda y: abs(y - target_year))

    print(f"[roster] Using {closest_year} roster settings for year {target_year}")
    return roster_by_year[closest_year]


def compute_league_wide_optimal_players(
    df: pl.DataFrame,
    roster_by_year: Dict[int, Dict[str, Any]],
    year_col: str,
    week_col: str,
    position_col: str,
    points_col: str,
) -> pl.DataFrame:
    """
    Compute league_wide_optimal_player flag and league_wide_optimal_position based on roster settings from JSON files.

    For each year-week combination, selects the top N players at each position based on points,
    where N comes from the roster_positions in the JSON file for that year (or nearest year).

    This is league-wide optimal, not manager-specific.
    """
    if not roster_by_year:
        print("[warning] No roster settings available, skipping league_wide_optimal_player computation")
        return df.with_columns([
            pl.lit(None).cast(pl.Int64).alias("league_wide_optimal_player"),
            pl.lit(None).cast(pl.Utf8).alias("league_wide_optimal_position")
        ])

    years = [y for y in df.select(pl.col(year_col)).unique().to_series().to_list() if y is not None]
    if not years:
        return df.with_columns([
            pl.lit(0).cast(pl.Int64).alias("league_wide_optimal_player"),
            pl.lit(None).cast(pl.Utf8).alias("league_wide_optimal_position")
        ])

    print(f"[optimal] Computing league-wide optimal players for {len(years)} years...")

    dfs_by_year: list[pl.DataFrame] = []
    for y in years:
        year_df = df.filter(pl.col(year_col) == y)
        roster_data = find_nearest_year_roster(int(y), roster_by_year)

        if not roster_data:
            year_df = year_df.with_columns([
                pl.lit(0).cast(pl.Int64).alias("__optimal_flag"),
                pl.lit(None).cast(pl.Utf8).alias("__optimal_position")
            ])
            dfs_by_year.append(year_df)
            continue

        roster_positions = roster_data.get("roster_positions", []) or []
        if not roster_positions:
            year_df = year_df.with_columns([
                pl.lit(0).cast(pl.Int64).alias("__optimal_flag"),
                pl.lit(None).cast(pl.Utf8).alias("__optimal_position")
            ])
            dfs_by_year.append(year_df)
            continue

        # Parse roster requirements
        position_counts: Dict[str, int] = {}
        flex_positions: list[tuple[str, int]] = []

        for pos_info in roster_positions:
            pos_name = str(pos_info.get("position", "")).upper()
            count = int(pos_info.get("count", 0))

            # Skip bench/IR positions
            if pos_name in ["BN", "IR", "BENCH"]:
                continue

            # Flex?
            is_flex = ("/" in pos_name) or any(k in pos_name for k in ["FLEX", "UTIL", "SUPER"])

            if is_flex:
                flex_positions.append((pos_name, count))
            else:
                if pos_name in ["D/ST", "DST"]:
                    pos_name = "DEF"
                position_counts[pos_name] = count

        print(f"     Year {y}: Position requirements: {position_counts}")
        print(f"     Year {y}: Flex positions: {flex_positions}")

        # Prepare columns
        year_df = year_df.with_columns([
            pl.col(points_col).cast(pl.Float64, strict=False).fill_null(0.0).alias("__points"),
            pl.col(position_col).cast(pl.Utf8).str.to_uppercase().alias("__position"),
            pl.lit(None).cast(pl.Utf8).alias("__optimal_position")
        ])

        # Rank within position per week
        year_df = year_df.with_columns(
            pl.col("__points")
            .rank(method="ordinal", descending=True)
            .over([week_col, "__position"])
            .alias("__pos_rank")
        )

        # Mark regular starters
        optimal_conditions = []
        for pos, count in position_counts.items():
            condition = (pl.col("__position") == pos) & (pl.col("__pos_rank") <= count)
            optimal_conditions.append(condition)
            year_df = year_df.with_columns(
                pl.when(condition)
                .then(pl.lit(pos, dtype=pl.Utf8))
                .otherwise(pl.col("__optimal_position"))
                .alias("__optimal_position")
            )

        if optimal_conditions:
            year_df = year_df.with_columns(
                pl.when(pl.any_horizontal(*optimal_conditions)).then(1).otherwise(0).alias("__is_position_starter")
            )
        else:
            year_df = year_df.with_columns(pl.lit(0).alias("__is_position_starter"))

        # Flex slots
        for flex_name, flex_count in flex_positions:
            if "W/R/T" in flex_name:
                elig = ["WR", "RB", "TE"]
            elif "W/R" in flex_name and "T" not in flex_name:
                elig = ["WR", "RB"]
            elif "W/T" in flex_name:
                elig = ["WR", "TE"]
            elif "R/T" in flex_name:
                elig = ["RB", "TE"]
            elif "SUPER" in flex_name:
                elig = ["QB", "WR", "RB", "TE"]
            else:
                elig = ["WR", "RB", "TE"]

            print(f"     Year {y}: {flex_name} eligible for: {elig}")

            # Flex eligible = eligible position AND not already among top starters for that pos
            flex_conds = []
            for pos in elig:
                starters = position_counts.get(pos, 0)
                flex_conds.append((pl.col("__position") == pos) & (pl.col("__pos_rank") > starters))

            if flex_conds:
                flex_eligible_mask = pl.any_horizontal(*flex_conds)
                year_df = year_df.with_columns(flex_eligible_mask.alias("__flex_eligible"))
            else:
                year_df = year_df.with_columns(pl.lit(False).alias("__flex_eligible"))

            # Rank flex-eligible across week
            year_df = year_df.with_columns(
                pl.when(pl.col("__flex_eligible")).then(pl.col("__points")).otherwise(None).alias("__flex_points")
            ).with_columns(
                pl.col("__flex_points").rank(method="ordinal", descending=True).over([week_col]).alias("__flex_rank")
            )

            flex_condition = (pl.col("__flex_eligible") == True) & (pl.col("__flex_rank") <= flex_count)
            optimal_conditions.append(flex_condition)

            year_df = year_df.with_columns(
                pl.when(flex_condition)
                .then(pl.lit(flex_name, dtype=pl.Utf8))
                .otherwise(pl.col("__optimal_position"))
                .alias("__optimal_position")
            )

        # Final optimal flag
        if optimal_conditions:
            year_df = year_df.with_columns(
                pl.when(pl.any_horizontal(*optimal_conditions)).then(1).otherwise(0).alias("__optimal_flag")
            )
        else:
            year_df = year_df.with_columns(pl.lit(0).alias("__optimal_flag"))

        # Cleanup temps except the ones we need to carry
        temp_cols = [
            c for c in year_df.columns
            if c.startswith("__") and c not in ["__rowid", "__pos", "__season_year", "__optimal_flag", "__optimal_position"]
        ]
        year_df = year_df.drop(temp_cols)

        dfs_by_year.append(year_df)

    df = pl.concat(dfs_by_year) if dfs_by_year else df

    # Rename to final
    if "__optimal_flag" in df.columns:
        if "league_wide_optimal_player" not in df.columns:
            df = df.with_columns(pl.col("__optimal_flag").alias("league_wide_optimal_player"))
        else:
            df = df.with_columns(
                pl.when(pl.col("league_wide_optimal_player").is_null())
                .then(pl.col("__optimal_flag"))
                .otherwise(pl.col("league_wide_optimal_player"))
                .alias("league_wide_optimal_player")
            )
        df = df.drop("__optimal_flag")

    if "__optimal_position" in df.columns:
        if "league_wide_optimal_position" not in df.columns:
            df = df.with_columns(pl.col("__optimal_position").alias("league_wide_optimal_position"))
        else:
            df = df.with_columns(
                pl.when(pl.col("league_wide_optimal_position").is_null())
                .then(pl.col("__optimal_position"))
                .otherwise(pl.col("league_wide_optimal_position"))
                .alias("league_wide_optimal_position")
            )
        df = df.drop("__optimal_position")

    return df


def build_points_expression(rules: List[Dict[str, Any]], df_columns: List[str]) -> pl.Expr:
    """Build a Polars expression to compute fantasy points from scoring rules."""
    points_expr = pl.lit(0.0)

    def safe_col(col_name: str) -> pl.Expr:
        if col_name in df_columns:
            return pl.col(col_name).cast(pl.Float64, strict=False).fill_null(0.0)
        return pl.lit(0.0)

    def two_pt_total() -> pl.Expr:
        cols = ["2-pt", "rushing_2pt_conversions", "receiving_2pt_conversions", "passing_2pt_conversions"]
        total = pl.lit(0.0)
        for c in cols:
            total = total + safe_col(c)
        return total

    for rule in rules:
        pts = rule.get("points")
        if pts is None:
            continue
        try:
            pts_val = float(pts)
        except Exception:
            continue

        name = str(rule.get("name") or "").strip()
        if not name:
            continue

        key = name.lower().replace(" ", "").replace("-", "_").replace("+", "plus")

        # Exclude team points allowed bucketing here (handled separately when needed)
        if key in {"pointsallowed", "pointsallowedpts", "ptsallow"}:
            continue

        col_expr = None

        if key == "passyds":
            col_expr = safe_col("pass_yds")
        elif key in {"passtd", "passtdd"}:
            col_expr = safe_col("pass_td")
        elif key == "int":
            col_expr = safe_col("passing_interceptions")
        elif key == "rushyds":
            col_expr = safe_col("rush_yds")
        elif key == "rushtd":
            col_expr = safe_col("rush_td")
        elif key in {"rec", "receptions"}:
            col_expr = safe_col("rec")
        elif key in {"recyds", "receivingyds"}:
            col_expr = safe_col("rec_yds")
        elif key in {"rectd", "receivingtd"}:
            col_expr = safe_col("rec_td")
        elif key in {"retd", "returntd", "rettd"}:
            col_expr = safe_col("ret_td")
        elif key in {"2_pt", "two_pt", "2pt"}:
            col_expr = two_pt_total()
        elif key in {"fumlost", "fumbleslost"}:
            col_expr = safe_col("fum_lost")
        elif key in {"fumretd", "fumrettd"}:
            col_expr = safe_col("fum_ret_td")
        elif key in {"patmade", "pat"}:
            col_expr = safe_col("pat_made")
        elif key in {"patmiss", "patmissed"}:
            col_expr = safe_col("pat_missed") + safe_col("pat_miss")
        elif key in {"fgyds", "fgyards"}:
            col_expr = safe_col("fg_yds")
        elif key == "fg0_19":
            col_expr = safe_col("fg_made_0_19")
        elif key == "fg20_29":
            col_expr = safe_col("fg_made_20_29")
        elif key == "fg30_39":
            col_expr = safe_col("fg_made_30_39")
        elif key == "fg40_49":
            col_expr = safe_col("fg_made_40_49")
        elif key in {"fg50plus", "fg50_plus", "fg50"}:
            col_expr = safe_col("fg_made_50_59") + safe_col("fg_made_60") + safe_col("fg_made_60_")
        elif key == "fgm0_19":
            col_expr = safe_col("fg_missed_0_19")
        elif key == "fgm20_29":
            col_expr = safe_col("fg_missed_20_29")
        elif key == "fgm30_39":
            col_expr = safe_col("fg_missed_30_39")
        elif key == "fgm40_49":
            col_expr = safe_col("fg_missed_40_49")
        elif key in {"fgm50plus", "fgm50_plus", "fgm50"}:
            col_expr = safe_col("fg_missed_50_59") + safe_col("fg_missed_60") + safe_col("fg_missed_60_")
        elif key == "sack":
            col_expr = safe_col("def_sacks") + safe_col("def_sack")
        elif key == "interception":
            col_expr = safe_col("def_interceptions") + safe_col("int")
        elif key in {"fumblerecovery", "fumrec"}:
            col_expr = safe_col("fum_rec")
        elif key in {"touchdown", "td"}:
            col_expr = safe_col("def_tds") + safe_col("td")
        elif key in {"safety", "safe"}:
            col_expr = safe_col("def_safeties")
        elif key in {"kickoffandpuntreturntouchdowns", "kickandpuntrettd"}:
            col_expr = safe_col("special_teams_tds")
        elif key in {"blkick", "blkkick", "blockedkick"}:
            col_expr = safe_col("blk_kick")
        elif key in {"3andouts", "threeandouts", "3outs"}:
            col_expr = safe_col("3_and_outs")
        elif key in {"4dwnstops", "4thdownstops", "4dwstops"}:
            col_expr = safe_col("4_dwn_stops")
        elif key in {"tfl", "tacklesforloss"}:
            col_expr = safe_col("def_tackles_for_loss")
        elif key == "xpr":
            col_expr = safe_col("xpr")
        elif key.startswith("ptsallow"):
            suffix = key.replace("ptsallow", "")
            if suffix == "0":
                col_expr = safe_col("pts_allow_0")
            elif suffix == "1_6":
                col_expr = safe_col("pts_allow_1_6")
            elif suffix == "7_13":
                col_expr = safe_col("pts_allow_7_13")
            elif suffix == "14_20":
                col_expr = safe_col("pts_allow_14_20")
            elif suffix == "21_27":
                col_expr = safe_col("pts_allow_21_27")
            elif suffix == "28_34":
                col_expr = safe_col("pts_allow_28_34")
            elif suffix in {"35plus", "35_plus"}:
                col_expr = safe_col("pts_allow_35_plus")

        if col_expr is not None:
            points_expr = points_expr + (col_expr * pts_val)

    return points_expr


def compute_points_from_stats(
    df: pl.DataFrame,
    rules_by_year: Dict[int, Dict[str, Any]],
    year_col: str,
    *,
    points_col: Optional[str] = None,
    week_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compute fantasy points based on per-year scoring rules and (optionally)
    add a rolling_point_total by player_year ordered by (year, week).
    """
    if points_col is None:
        points_col = detect_points_col(df)

    if not rules_by_year:
        print("[warning] No scoring rules available, skipping point computation")
        return _maybe_add_rolling(df, points_col, year_col, week_col)

    years = [y for y in df.select(pl.col(year_col)).unique().to_series().to_list() if y is not None]
    if not years:
        return _maybe_add_rolling(df, points_col, year_col, week_col)

    dfs_by_year: list[pl.DataFrame] = []
    for y in years:
        year_df = df.filter(pl.col(year_col) == y)
        rules_data = find_nearest_year_rules(int(y), rules_by_year)
        if not rules_data:
            dfs_by_year.append(year_df)
            continue

        rules_list = rules_data.get("full_scoring", [])
        if not isinstance(rules_list, list):
            dfs_by_year.append(year_df)
            continue

        pts_expr = build_points_expression(rules_list, year_df.columns)
        dfs_by_year.append(year_df.with_columns(pts_expr.alias("__computed_points_temp")))

    df = pl.concat(dfs_by_year) if dfs_by_year else df

    # finalize computed_points column (fill only where missing)
    if "__computed_points_temp" in df.columns:
        if "computed_points" not in df.columns:
            df = df.with_columns(pl.col("__computed_points_temp").alias("computed_points"))
        else:
            df = df.with_columns(
                pl.when(pl.col("computed_points").is_null())
                .then(pl.col("__computed_points_temp"))
                .otherwise(pl.col("computed_points"))
                .alias("computed_points")
            )
        df = df.drop("__computed_points_temp")

    # substitute into points_col when original is 0 but computed is non-zero
    if "computed_points" in df.columns and points_col in df.columns:
        df = df.with_columns(
            pl.when(
                (pl.col(points_col).cast(pl.Float64, strict=False).fill_null(0.0) == 0.0)
                & (pl.col("computed_points").cast(pl.Float64, strict=False).fill_null(0.0) != 0.0)
            )
            .then(pl.col("computed_points"))
            .otherwise(pl.col(points_col))
            .alias(points_col)
        )

    return _maybe_add_rolling(df, points_col, year_col, week_col)


def _maybe_add_rolling(
    df: pl.DataFrame, points_col: str, year_col: Optional[str], week_col: Optional[str]
) -> pl.DataFrame:
    """
    If rolling_point_total is missing and we have player_year + year_col + week_col,
    compute cumulative season points by player_year ordered by (year, week).
    """
    if (
        "rolling_point_total" not in df.columns
        and "player_year" in df.columns
        and year_col is not None
        and week_col is not None
    ):
        df = df.with_columns([
            pl.col(year_col).cast(pl.Int64, strict=False).alias("__year_int"),
            pl.col(week_col).cast(pl.Int64, strict=False).alias("__week_int"),
            pl.col(points_col).cast(pl.Float64, strict=False).fill_null(0.0).alias("__points_clean")
        ])
        df = df.with_columns(
            pl.col("__points_clean")
            .cum_sum()
            .over("player_year", order_by=["__year_int", "__week_int"])
            .alias("rolling_point_total")
        )
        df = df.drop(["__year_int", "__week_int", "__points_clean"])

    return df


# --------------------------------------------------------------------------------------
# Utilities
def detect_column(df: pl.DataFrame, candidates: List[str], fuzzy: bool = False) -> Optional[str]:
    cols = df.columns
    for c in candidates:
        if c in cols:
            return c
    if fuzzy:
        norm = {re.sub(r"\W+", "", c.lower()): c for c in cols}
        for c in candidates:
            key = re.sub(r"\W+", "", c.lower())
            if key in norm:
                return norm[key]
    return None


def detect_points_col(df: pl.DataFrame) -> str:
    """
    Finds the points column (e.g., "points" or "team_points").
    Only use points not team_points.
    """
    if "points" in df.columns:
        return "points"

    candidates = [
        "Points", "weekly_points", "Week Points", "rolling_point_total",
        "player_points", "fantasy_points", "Fantasy Points"
    ]
    col = detect_column(df, candidates)
    if col:
        return col

    for c in df.columns:
        col_lower = c.lower()
        if "point" in col_lower and "team" not in col_lower:
            return c

    return df.columns[0] if df.columns else "points"


def detect_position_col(df: pl.DataFrame) -> Optional[str]:
    candidates = [
        "position", "Position", "pos", "Pos",
        "fantasy_position", "fantasy pos", "fantasyposition", "fantasy_pos",
        "lineup_slot", "lineup_position", "slot", "roster_slot", "Roster_Slot",
        "roster_position", "player_fantasy_position",
    ]
    col = detect_column(df, candidates, fuzzy=True)
    if col:
        return col
    for c in df.columns:
        lc = c.lower()
        if ("pos" in lc or "slot" in lc or "position" in lc) and "bench" not in lc:
            return c
    return None


def detect_nfl_position_col(df: pl.DataFrame) -> Optional[str]:
    return detect_column(df, ["nfl_position", "NFL_Position", "yahoo_position", "primary_position"])


def detect_yahoo_position_col(df: pl.DataFrame) -> Optional[str]:
    return detect_column(df, ["yahoo_position", "Yahoo_Position", "YahooPosition"])


def detect_player_col(df: pl.DataFrame) -> Optional[str]:
    return detect_column(df, ["player", "Player", "player_name", "Player Name", "name", "Name"])


def detect_manager_col(df: pl.DataFrame) -> Optional[str]:
    return detect_column(df, ["manager", "Manager", "owner", "Owner", "team_manager", "Team Manager"])


def detect_opponent_col(df: pl.DataFrame) -> Optional[str]:
    candidates = [
        "opponent", "Opponent", "opp", "Opp", "vs", "Vs", "versus", "Versus",
        "against", "Against", "opponent_manager", "Opponent_Manager",
    ]
    col = detect_column(df, candidates)
    if col:
        return col
    for c in df.columns:
        if "opponent" in c.lower() or c.lower() in {"opp", "vs"}:
            return c
    return None


def detect_year_col(df: pl.DataFrame) -> Optional[str]:
    return detect_column(df, ["year", "Year", "season", "Season", "season_year", "Season_Year"])


def detect_week_col(df: pl.DataFrame) -> Optional[str]:
    return detect_column(df, ["week", "Week", "game_week", "Game_Week"])


def detect_manager_year_col(df: pl.DataFrame) -> Optional[str]:
    col = detect_column(df, ["manager_year", "Manager_Year", "manageryear", "ManagerYear"])
    if col:
        return col
    for c in df.columns:
        if "manager" in c.lower() and "year" in c.lower():
            return c
    return None


def detect_headshot_col(df: pl.DataFrame) -> Optional[str]:
    return detect_column(df, ["headshot_url", "Headshot_URL", "headshot", "Headshot"])


def detect_rolling_point_total_col(df: pl.DataFrame) -> Optional[str]:
    return detect_column(df, ["rolling_point_total", "Rolling_Point_Total", "cumulative_points"])


def add_if_missing_or_fill_nulls(df: pl.DataFrame, name: str, expr: pl.Expr,
                                 dtype: Optional[pl.DataType] = None) -> pl.DataFrame:
    if dtype is not None:
        expr = expr.cast(dtype, strict=False)
    if name in df.columns:
        return df.with_columns(
            pl.when(pl.col(name).is_null()).then(expr).otherwise(pl.col(name)).alias(name)
        )
    else:
        return df.with_columns(expr.alias(name))


def ensure_cols(df: pl.DataFrame, cols: List[str]) -> Tuple[pl.DataFrame, List[str]]:
    resolved: List[str] = []
    new_cols: List[pl.Expr] = []
    for c in cols:
        if isinstance(c, str):
            if c in df.columns:
                resolved.append(c)
            else:
                placeholder = "__missing_" + re.sub(r"\W+", "_", c)
                if placeholder not in df.columns:
                    new_cols.append(pl.lit(None).alias(placeholder))
                resolved.append(placeholder)
        else:
            resolved.append(c)
    if new_cols:
        df = df.with_columns(new_cols)
    return df, resolved


# --------------------------------------------------------------------------------------
# Ranking helper with tiebreakers
def add_ranks_with_tiebreaker(
    df: pl.DataFrame,
    points_col: str,
    partition_cols: List[str],
    rank_name: str,
    pct_name: str,
    valid_mask: pl.Expr,
    rolling_col: Optional[str] = None,
    player_col: Optional[str] = None,
) -> pl.DataFrame:
    # Ensure at least one partition key
    if not partition_cols:
        if "_dummy" not in df.columns:
            df = df.with_columns(pl.lit(1).alias("_dummy"))
        partition_cols = ["_dummy"]

    points_expr = pl.col(points_col).cast(pl.Float64, strict=False)
    points_masked = pl.when(valid_mask).then(points_expr).otherwise(None)

    df = df.with_columns([points_masked.alias("__tb_points")])

    # rank within partitions
    df = df.with_columns(
        pl.when(valid_mask)
        .then(pl.col("__tb_points").rank(method="ordinal", descending=True).over(partition_cols))
        .otherwise(None)
        .alias("__sort_order")
    )

    # percentile (average rank)
    group_size = (pl.when(valid_mask).then(1).otherwise(0).sum().over(partition_cols))
    avg_rank = points_masked.rank(method="average", descending=True).over(partition_cols)
    pct_expr = pl.when(valid_mask & (group_size > 1)) \
        .then(((group_size - avg_rank) / (group_size - 1)) * 100.0) \
        .when(valid_mask) \
        .then(100.0) \
        .otherwise(None)

    df = add_if_missing_or_fill_nulls(df, rank_name, pl.col("__sort_order"), pl.Int64)
    df = add_if_missing_or_fill_nulls(df, pct_name, pct_expr, pl.Float64)

    temp_cols = ["__tb_points", "__sort_order"]
    df = df.drop([c for c in temp_cols if c in df.columns])

    return df


# --------------------------------------------------------------------------------------
# Main
def main() -> None:
    if not PARQUET_PATH.exists() and not CSV_PATH.exists():
        print(f"[error] No input found. Expected {PARQUET_PATH} or {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    print("[scoring] Loading scoring rules...")
    scoring_rules = load_scoring_rules(SCORING_DIR)

    print("[roster] Loading roster settings...")
    roster_settings = load_roster_settings(SCORING_DIR)

    print("[loading] Reading data...")
    if PARQUET_PATH.exists():
        df = pl.read_parquet(PARQUET_PATH)
    else:
        df = pl.read_csv(CSV_PATH)

    points_col = detect_points_col(df)
    fantasy_position_col = detect_position_col(df)
    nfl_position_col = detect_nfl_position_col(df)
    yahoo_position_col = detect_yahoo_position_col(df)
    player_col = detect_player_col(df)
    manager_col = detect_manager_col(df)
    opponent_col = detect_opponent_col(df)
    year_col = detect_year_col(df)
    week_col = detect_week_col(df)
    manageryear_col = detect_manager_year_col(df)
    headshot_col = detect_headshot_col(df)
    rolling_col = detect_rolling_point_total_col(df)

    print(f"[detected] points={points_col}, fantasy_position={fantasy_position_col}")
    print(f"           yahoo_position={yahoo_position_col}, nfl_position={nfl_position_col}")
    print(f"           player={player_col}, manager={manager_col}, year={year_col}, week={week_col}")
    print(f"           rolling_point_total={rolling_col}")

    if year_col and scoring_rules:
        print("[processing] Computing fantasy points from scoring rules...")
        df = compute_points_from_stats(
            df, scoring_rules, year_col,
            points_col=points_col,
            week_col=week_col,
        )
        num_years = len([y for y in df.select(pl.col(year_col)).unique().to_series().to_list() if y is not None])
        print(f"[scoring] Computed points for {num_years} unique years")

    # Substitute computed_points into points when points=0 but computed!=0
    print("[processing] Substituting computed_points into points when points is 0 but computed_points is not 0...")
    if "computed_points" in df.columns:
        df = df.with_columns(
            pl.when(
                (pl.col(points_col).cast(pl.Float64, strict=False).fill_null(0.0) == 0.0)
                & (pl.col("computed_points").cast(pl.Float64, strict=False).fill_null(0.0) != 0.0)
            )
            .then(pl.col("computed_points"))
            .otherwise(pl.col(points_col))
            .alias(points_col)
        )

    print("[processing] Creating unified position column...")
    if "position" not in df.columns:
        if yahoo_position_col and nfl_position_col:
            df = df.with_columns(
                pl.when(pl.col(yahoo_position_col).is_not_null())
                .then(pl.col(yahoo_position_col))
                .otherwise(pl.col(nfl_position_col))
                .alias("position")
            )
        elif yahoo_position_col:
            df = df.with_columns(pl.col(yahoo_position_col).alias("position"))
        elif nfl_position_col:
            df = df.with_columns(pl.col(nfl_position_col).alias("position"))
        else:
            df = df.with_columns(pl.lit(None).alias("position"))
    position_col = "position"

    if "__rowid" not in df.columns:
        df = df.with_row_index("__rowid")

    if "manager_year" not in df.columns:
        if manageryear_col:
            df = df.with_columns(pl.col(manageryear_col).alias("manager_year"))
        elif manager_col and year_col:
            df = df.with_columns(
                (pl.col(manager_col).cast(pl.Utf8) + pl.col(year_col).cast(pl.Int64).cast(pl.Utf8)).alias("manager_year")
            )

    # player_year helper
    if "player_year" not in df.columns and player_col and year_col:
        df = df.with_columns(
            pl.when(pl.col(player_col).is_not_null() & pl.col(year_col).is_not_null())
            .then(pl.col(player_col).cast(pl.Utf8) + pl.col(year_col).cast(pl.Int64).cast(pl.Utf8))
            .otherwise(None)
            .alias("player_year")
        )

    if "__pos" not in df.columns:
        df = df.with_columns(pl.col(position_col).alias("__pos"))

    if "__season_year" not in df.columns:
        if "manager_year" in df.columns:
            df = df.with_columns(
                pl.col("manager_year").cast(pl.Utf8).str.extract(r"(\d{4})", 1).cast(pl.Int64).alias("__season_year")
            )
        elif year_col:
            df = df.with_columns(pl.col(year_col).cast(pl.Int64).alias("__season_year"))
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("__season_year"))

    # cumulative week mapping (use actual week number for consistency with matchup data)
    if "cumulative_week" not in df.columns and year_col and week_col:
        print("[processing] Creating cumulative_week from actual week numbers...")
        # Use the actual week number to ensure alignment between player and matchup data
        # Week 1 in any year = cumulative_week 1, Week 17 in any year = cumulative_week 17
        df = df.with_columns(
            pl.col(week_col).cast(pl.Int64, strict=False).alias("cumulative_week")
        )

    # Force (re)create rolling_point_total by player_year
    print("[processing] Creating rolling_point_total (cumulative season points by player_year)...")

    if "rolling_point_total" in df.columns:
        df = df.drop("rolling_point_total")

    if "player_year" in df.columns and year_col and week_col:
        df = df.with_columns([
            pl.col(year_col).cast(pl.Int64, strict=False).alias("__year_int"),
            pl.col(week_col).cast(pl.Int64, strict=False).alias("__week_int"),
            pl.col(points_col).cast(pl.Float64, strict=False).fill_null(0.0).alias("__points_clean")
        ])
        df = df.with_columns(
            pl.col("__points_clean")
            .cum_sum()
            .over("player_year", order_by=["__year_int", "__week_int"])
            .alias("rolling_point_total")
        )
        df = df.drop(["__year_int", "__week_int", "__points_clean"])
    else:
        df = add_if_missing_or_fill_nulls(df, "rolling_point_total", pl.lit(0.0), pl.Float64)

    # ID helpers
    if "manager_week" not in df.columns and manager_col and "cumulative_week" in df.columns:
        df = df.with_columns(
            pl.when(pl.col(manager_col).is_not_null() & pl.col("cumulative_week").is_not_null())
            .then(pl.col(manager_col).cast(pl.Utf8).str.replace_all(r"\s+", "") + pl.col("cumulative_week").cast(pl.Utf8))
            .otherwise(None)
            .alias("manager_week")
        )

    if "player_week" not in df.columns and player_col and "cumulative_week" in df.columns:
        df = df.with_columns(
            pl.when(pl.col(player_col).is_not_null() & pl.col("cumulative_week").is_not_null())
            .then(pl.col(player_col).cast(pl.Utf8).str.replace_all(r"\s+", "") + pl.col("cumulative_week").cast(pl.Utf8))
            .otherwise(None)
            .alias("player_week")
        )

    if "opponent_week" not in df.columns and opponent_col and "cumulative_week" in df.columns:
        df = df.with_columns(
            pl.when(pl.col(opponent_col).is_not_null() & pl.col("cumulative_week").is_not_null())
            .then(pl.col(opponent_col).cast(pl.Utf8).str.replace_all(r"\s+", "") + pl.col("cumulative_week").cast(pl.Utf8))
            .otherwise(None)
            .alias("opponent_week")
        )

    points_ok = pl.col(points_col).cast(pl.Float64, strict=False).is_not_null()

    if fantasy_position_col:
        pos_upper = pl.col(fantasy_position_col).cast(pl.Utf8).str.to_uppercase()
        not_bench = ~pos_upper.is_in(["BN", "IR", "BENCH", "OUT"])
    else:
        not_bench = pl.lit(True)

    playoff_cols = [c for c in ["is_playoffs", "Is_Playoffs"] if c in df.columns]
    cons_cols = [c for c in ["is_consolation", "Is_Consolation"] if c in df.columns]

    not_playoff = pl.min_horizontal(*[pl.col(c).fill_null(0).cast(pl.Int64) == 0 for c in playoff_cols]) if playoff_cols else pl.lit(True)
    not_consolation = pl.min_horizontal(*[pl.col(c).fill_null(0).cast(pl.Int64) == 0 for c in cons_cols]) if cons_cols else pl.lit(True)

    valid_mask = points_ok & not_bench & not_playoff & not_consolation

    print("[processing] Computing rankings with tiebreakers...")

    nfl_active_mask = pl.lit(True)
    if "nfl_player_id" in df.columns:
        nfl_active_mask = pl.col("nfl_player_id").is_not_null() & (pl.col("nfl_player_id") != "")

    print("[processing] Computing league-wide player history rankings...")

    valid_positions = ["QB", "RB", "WR", "TE", "DEF", "K", "D/ST", "DST"]
    position_filter = pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(valid_positions)

    # All-time league-wide player rankings
    league_player_mask = points_ok & nfl_active_mask & position_filter
    df = add_ranks_with_tiebreaker(
        df, points_col, [],
        "player_all_time_history",
        "player_all_time_history_percentile",
        league_player_mask, rolling_col, player_col
    )

    # Season league-wide player rankings
    if "__season_year" in df.columns:
        df, parts = ensure_cols(df, ["__season_year"])
        df = add_ranks_with_tiebreaker(
            df, points_col, parts,
            "player_season_history",
            "player_season_history_percentile",
            league_player_mask, rolling_col, player_col
        )
    else:
        df = add_if_missing_or_fill_nulls(df, "player_season_history", pl.lit(None), pl.Int64)
        df = add_if_missing_or_fill_nulls(df, "player_season_history_percentile", pl.lit(None), pl.Float64)

    # Player personal (all-time)
    if player_col:
        df, parts = ensure_cols(df, [player_col])
        player_personal_mask = points_ok & nfl_active_mask & position_filter
        df = add_ranks_with_tiebreaker(
            df, points_col, parts,
            "player_personal_all_time_history",
            "player_personal_all_time_history_percentile",
            player_personal_mask, rolling_col, player_col
        )
    else:
        df = add_if_missing_or_fill_nulls(df, "player_personal_all_time_history", pl.lit(None), pl.Int64)
        df = add_if_missing_or_fill_nulls(df, "player_personal_all_time_history_percentile", pl.lit(None), pl.Float64)

    # Player personal (season)
    if "player_year" in df.columns:
        df, parts = ensure_cols(df, ["player_year"])
        player_personal_mask = points_ok & nfl_active_mask & position_filter
        df = add_ranks_with_tiebreaker(
            df, points_col, parts,
            "player_personal_season_history",
            "player_personal_season_history_percentile",
            player_personal_mask, rolling_col, player_col
        )
    else:
        df = add_if_missing_or_fill_nulls(df, "player_personal_season_history", pl.lit(None), pl.Int64)
        df = add_if_missing_or_fill_nulls(df, "player_personal_season_history_percentile", pl.lit(None), pl.Float64)

    # Position (all-time)
    df, parts = ensure_cols(df, ["__pos"])
    position_valid_mask = valid_mask & position_filter
    df = add_ranks_with_tiebreaker(
        df, points_col, parts,
        "position_all_time_history",
        "position_all_time_history_percentile",
        position_valid_mask, rolling_col, player_col
    )

    # Manager season rankings (by manager_year only)
    if "manager_year" in df.columns and manager_col:
        df, parts = ensure_cols(df, ["manager_year"])
        manager_valid_mask = valid_mask & (pl.col(manager_col).is_not_null() & (pl.col(manager_col) != ""))
        df = add_ranks_with_tiebreaker(
            df, points_col, parts,
            "manager_player_season_history",
            "manager_player_season_history_percentile",
            manager_valid_mask, rolling_col, player_col
        )
    else:
        df = add_if_missing_or_fill_nulls(df, "manager_player_season_history", pl.lit(None), pl.Int64)
        df = add_if_missing_or_fill_nulls(df, "manager_player_season_history_percentile", pl.lit(None), pl.Float64)

    # Manager position (season)
    if "manager_year" in df.columns and manager_col:
        df, parts = ensure_cols(df, ["manager_year", "__pos"])
        manager_pos_valid_mask = valid_mask & position_filter & (pl.col(manager_col).is_not_null() & (pl.col(manager_col) != ""))
        df = add_ranks_with_tiebreaker(
            df, points_col, parts,
            "manager_position_season_history",
            "manager_position_season_history_percentile",
            manager_pos_valid_mask, rolling_col, player_col
        )
    else:
        df = add_if_missing_or_fill_nulls(df, "manager_position_season_history", pl.lit(None), pl.Int64)
        df = add_if_missing_or_fill_nulls(df, "manager_position_season_history_percentile", pl.lit(None), pl.Float64)

    # Position (season)
    if "__season_year" in df.columns:
        df, parts = ensure_cols(df, ["__season_year", "__pos"])
        df = add_ranks_with_tiebreaker(
            df, points_col, parts,
            "position_season_history",
            "position_season_history_percentile",
            position_valid_mask, rolling_col, player_col
        )
    else:
        df = add_if_missing_or_fill_nulls(df, "position_season_history", pl.lit(None), pl.Int64)
        df = add_if_missing_or_fill_nulls(df, "position_season_history_percentile", pl.lit(None), pl.Float64)

    # Manager-player (all-time)
    if manager_col:
        df, parts = ensure_cols(df, [manager_col])
        manager_not_null_mask = valid_mask & (pl.col(manager_col).is_not_null() & (pl.col(manager_col) != ""))
        df = add_ranks_with_tiebreaker(
            df, points_col, parts,
            "manager_player_all_time_history",
            "manager_player_all_time_history_percentile",
            manager_not_null_mask, rolling_col, player_col
        )
    else:
        df = add_if_missing_or_fill_nulls(df, "manager_player_all_time_history", pl.lit(None), pl.Int64)
        df = add_if_missing_or_fill_nulls(df, "manager_player_all_time_history_percentile", pl.lit(None), pl.Float64)

    # Manager-position (all-time)
    if manager_col:
        df, parts = ensure_cols(df, [manager_col, "__pos"])
        manager_pos_not_null_mask = valid_mask & (pl.col(manager_col).is_not_null() & (pl.col(manager_col) != ""))
        df = add_ranks_with_tiebreaker(
            df, points_col, parts,
            "manager_position_all_time_history",
            "manager_position_all_time_history_percentile",
            manager_pos_not_null_mask, rolling_col, player_col
        )
    else:
        df = add_if_missing_or_fill_nulls(df, "manager_position_all_time_history", pl.lit(None), pl.Int64)
        df = add_if_missing_or_fill_nulls(df, "manager_position_all_time_history_percentile", pl.lit(None), pl.Float64)

    # --- Optimal lineup metrics by manager_week / manager_year (STRICT: manager-optimal only) ---
    print("[processing] Computing optimal lineup metrics (manager-optimal only)...")

    # Clean slate
    for col_name in ["optimal_points", "rolling_optimal_points", "optimal_ppg"]:
        if col_name in df.columns:
            df = df.drop(col_name)

    required = {"manager_week", "manager_year", points_col, manager_col, "optimal_player"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for manager-optimal metrics: {missing}")

    manager_not_null = pl.col(manager_col).is_not_null() & (pl.col(manager_col) != "")

    # Sum points from rows flagged manager-optimal per manager_week
    optimal_totals = (
        df.filter(manager_not_null & (pl.col("optimal_player") == 1))
        .group_by("manager_week")
        .agg(pl.col(points_col).cast(pl.Float64, strict=False).sum().alias("__optimal_total"))
    )

    # Join back so EVERY row in that manager_week gets the same team optimal total
    df = df.join(optimal_totals, on="manager_week", how="left")

    df = df.with_columns(
        pl.when(manager_not_null)
        .then(pl.col("__optimal_total").fill_null(0.0))
        .otherwise(None)
        .alias("optimal_points")
    ).drop("__optimal_total")

    # Order keys for rolling stats
    year_int = pl.col(year_col).cast(pl.Int64, strict=False) if year_col else pl.lit(None)
    week_int = pl.col(week_col).cast(pl.Int64, strict=False) if week_col else pl.lit(None)

    # Rolling team optimal points within (manager_year)
    df = df.with_columns(
        pl.when(manager_not_null)
        .then(pl.col("optimal_points").cast(pl.Float64, strict=False).cum_sum().over(["manager_year"], order_by=[year_int, week_int]))
        .otherwise(None)
        .alias("rolling_optimal_points")
    )

    # Optimal PPG (team-level weekly optimal avg) within (manager_year)
    df = df.with_columns(
        pl.when(manager_not_null)
        .then(pl.col("optimal_points").cast(pl.Float64, strict=False).mean().over(["manager_year"]))
        .otherwise(None)
        .alias("optimal_ppg")
    )

    # -------------------------------
    # Recompute PPG (season + all-time) on every run
    # -------------------------------
    print("[processing] Recomputing PPG (season + all-time)...")
    points_expr = pl.col(points_col).cast(pl.Float64, strict=False)

    # Ensure unified position exists (already built; keep reference)
    if "position" not in df.columns:
        pos_fallback = next(
            (c for c in [yahoo_position_col, nfl_position_col, fantasy_position_col] if c and c in df.columns),
            None
        )
        if pos_fallback:
            df = df.with_columns(pl.col(pos_fallback).alias("position"))
        else:
            df = df.with_columns(pl.lit(None).alias("position"))
    position_col = "position"

    # Season key (manager_year → 4-digit; else year)
    if "__season_year" not in df.columns:
        if "manager_year" in df.columns:
            df = df.with_columns(
                pl.col("manager_year").cast(pl.Utf8).str.extract(r"(\d{4})", 1).cast(pl.Int64).alias("__season_year")
            )
        elif year_col:
            df = df.with_columns(pl.col(year_col).cast(pl.Int64).alias("__season_year"))
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("__season_year"))

    # Robust player key
    player_candidates = [c for c in ["nfl_player_id", "yahoo_player_id", player_col] if c and c in df.columns]
    if player_candidates:
        df = df.with_columns(pl.coalesce([pl.col(c).cast(pl.Utf8) for c in player_candidates]).alias("__ppg_pid"))
    else:
        df = df.with_columns(pl.lit(None).alias("__ppg_pid"))

    # Activity mask with fallback to "has points"
    pos_u = pl.col(position_col).cast(pl.Utf8).str.to_uppercase()

    def int_ge1(name: str) -> pl.Expr:
        return (pl.col(name).cast(pl.Int64, strict=False) >= 1) if name in df.columns else pl.lit(False)

    def num_gt0(name: str) -> pl.Expr:
        return (pl.col(name).cast(pl.Float64, strict=False) > 0.0) if name in df.columns
