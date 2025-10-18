#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Iterable, Dict, List
import math
import pandas as pd
import numpy as np
import re
from md.md_utils import df_from_md_or_parquet

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = REPO_ROOT / "fantasy_football_data"  # <-- updated
PARQUET_PATH = DATA_DIR / "matchup.parquet"
CSV_PATH = DATA_DIR / "matchup.csv"

# Requested alias names to also populate
RANK_ALIAS_MAP = {
    "manager_season_ranking": ("manager_season_rank",),
    "league_season_ranking": ("league_season_rank",),
    "manager_all_time_ranking": ("manager_alltime_rank",),
    "manager_all_time_ranking_percentile": ("manager_alltime_percentile",),
    "league_all_time_ranking": ("league_alltime_rank",),
    "league_all_time_ranking_percentile": ("league_alltime_percentile",),
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _as_int(x) -> Optional[int]:
    try:
        if pd.isna(x) or x == "":
            return None
        return int(float(x))
    except Exception:
        return None

def _ensure_col(df: pd.DataFrame, col: str, dtype: str = "object") -> None:
    if col not in df.columns:
        df[col] = pd.Series([pd.NA] * len(df), index=df.index)
    try:
        df[col] = df[col].astype(dtype)
    except Exception:
        pass

def _to_int64(s: pd.Series) -> pd.Series:
    # robust: never try to int-cast strings like "Ezra2013"
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def _safe_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")

def _compute_rank_pct(
    df: pd.DataFrame,
    group_keys: Iterable[str] | None,
    value_col: str,
    tie_method: str = "min",
):
    if not isinstance(group_keys, (list, tuple)):
        group_keys = [group_keys] if group_keys else []

    mask = df[value_col].notna()
    if not mask.any():
        return (pd.Series(pd.NA, index=df.index, dtype="Int64"),
                pd.Series(float("nan"), index=df.index))

    valid = df.loc[mask].copy()
    if group_keys:
        g = valid.groupby(group_keys, observed=True, dropna=False)[value_col]
        ranks = g.rank(method=tie_method, ascending=False)
        counts = g.transform("count")
    else:
        ranks = valid[value_col].rank(method=tie_method, ascending=False)
        counts = pd.Series(len(valid), index=valid.index, dtype="int64")

    pct = (1 - (ranks - 1) / counts).astype(float) * 100.0

    rank_full = pd.Series(pd.NA, index=df.index, dtype="Int64")
    pct_full = pd.Series(float("nan"), index=df.index)
    rank_full.loc[valid.index] = ranks.astype("Int64")
    pct_full.loc[valid.index] = pct
    return rank_full, pct_full

# Strip zero-width chars
_ZW_RE = re.compile(r"[\u200B-\u200D\uFEFF]")

def _strip_zw(val):
    if pd.isna(val):
        return val
    return _ZW_RE.sub("", str(val))

def _mgr_token(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "na"

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
df = df_from_md_or_parquet("matchup", PARQUET_PATH)

# ------------------------------------------------------------
# Preserve original win/loss from source (don’t recalc)
# ------------------------------------------------------------
if 'win' in df.columns and 'loss' in df.columns:
    df['_original_win'] = _to_int64(df['win']).fillna(0).astype(int)
    df['_original_loss'] = _to_int64(df['loss']).fillna(0).astype(int)
else:
    df['_original_win'] = (_to_float(df['team_points']) > _to_float(df['opponent_points'])).astype(int)
    df['_original_loss'] = (_to_float(df['team_points']) < _to_float(df['opponent_points'])).astype(int)

# ------------------------------------------------------------
# Normalize postseason flags
# ------------------------------------------------------------
for col in ["is_playoffs", "is_consolation"]:
    if col not in df.columns:
        df[col] = 0

df["is_playoffs_raw"] = _to_int64(df["is_playoffs"]).fillna(0).astype(int)
df["is_consolation_raw"] = _to_int64(df["is_consolation"]).fillna(0).astype(int)

# **Hard rule**: consolation => not playoffs
df["is_consolation"] = pd.Series(df["is_consolation_raw"], index=df.index, dtype="Int64")
df["is_playoffs"] = pd.Series(
    np.where(df["is_consolation_raw"] == 1, 0, df["is_playoffs_raw"]),
    index=df.index, dtype="Int64"
)

df["postseason"] = ((df["is_playoffs"] == 1) | (df["is_consolation"] == 1)).astype("Int64")

# Restore original win/loss
df["win"] = pd.Series(df["_original_win"], index=df.index, dtype="Int64")
df["loss"] = pd.Series(df["_original_loss"], index=df.index, dtype="Int64")

# ------------------------------------------------------------
# Ensure required cols / types
# ------------------------------------------------------------
INT64_COLS = {
    "quarterfinal", "semifinal", "championship", "champion", "sacko",
    "final_wins", "final_losses", "final_regular_wins", "final_regular_losses",
    "final_playoff_seed", "playoff_seed_to_date",
    "team_made_playoffs", "team_got_bye",
    "manager_season_ranking", "league_season_ranking",
    "manager_all_time_ranking", "league_all_time_ranking",
    "manager_season_rank", "league_season_rank",
    "manager_alltime_rank", "league_alltime_rank",
    "cumulative_week", "winning_streak", "losing_streak",
    "year", "week", "is_playoffs", "is_consolation",
    "playoff_rounds_won", "playoff_rounds_played", "max_week",
}
FLOAT_COLS = {
    "team_points", "team_projected_points", "opponent_points",
    "inflation_rate", "real_score", "real_opponent_points", "real_margin", "real_total_matchup_score",
    "manager_all_time_ranking_percentile", "league_all_time_ranking_percentile",
    "manager_alltime_percentile", "league_alltime_percentile",
}
STRING_COLS = {
    "manager", "opponent", "team_name", "manager_year", "manager_week", "opponent_week",
    "opponent_year", "matchup_recap_title", "matchup_recap_url", "url", "image_url", "value",
}

for c in INT64_COLS: _ensure_col(df, c, "Int64")
for c in FLOAT_COLS: _ensure_col(df, c, "float64")
for c in STRING_COLS: _ensure_col(df, c, "string")

# harden core fields
df["year"] = _safe_int(df.get("year", pd.Series([], dtype="Int64")))
df["week"] = _safe_int(df.get("week", pd.Series([], dtype="Int64")))
df["manager"] = df["manager"].astype("string")
df["opponent"] = df["opponent"].astype("string")
df["team_name"] = df["team_name"].astype("string")

# ------------------------------------------------------------
# max_week flag
# ------------------------------------------------------------
max_week_by_year = df.groupby("year")["week"].max()
df["max_week"] = (df["week"] == df["year"].map(max_week_by_year)).astype("Int64")

# ------------------------------------------------------------
# cumulative_week (stable across years - based on actual year/week)
# ------------------------------------------------------------
# Formula: (year * 100) + week
# This ensures same week numbers across years align (e.g., 2013 week 1 = 201301, 2014 week 1 = 201401)
# But for manager_week matching, we want week-based alignment (week 1 = week 1 regardless of year)
# So we use just the week number as cumulative_week
df["cumulative_week"] = _safe_int(df["week"])

# ------------------------------------------------------------
# manager_week / opponent_week / manager_year / opponent_year
# ------------------------------------------------------------
df["manager_week"] = df.apply(
    lambda r: (re.sub(r"\s+", "", str(r.get('manager', ''))) + str(int(r['cumulative_week']))
               if pd.notna(r.get("manager")) and pd.notna(r.get("cumulative_week")) else pd.NA),
    axis=1,
).astype("string")
df["opponent_week"] = df.apply(
    lambda r: (re.sub(r"\s+", "", str(r.get('opponent', ''))) + str(int(r['cumulative_week']))
               if pd.notna(r.get("opponent")) and pd.notna(r.get("cumulative_week")) else pd.NA),
    axis=1,
).astype("string")

df["manager_year"] = (
    df["manager"].fillna("").astype(str).map(_strip_zw) + df["year"].astype("Int64").astype(str)
).astype("string")
df["opponent_year"] = (
    df["opponent"].fillna("").astype(str).map(_strip_zw) + df["year"].astype("Int64").astype(str)
).astype("string")

# ------------------------------------------------------------
# valid_season_game (true regular season only)
# ------------------------------------------------------------
df["valid_season_game"] = (
    (df["is_consolation"].fillna(0).astype(int) != 1) &
    (df["is_playoffs"].fillna(0).astype(int) != 1)
).astype(int)

# ------------------------------------------------------------
# Demote all games after first playoff loss to consolation
# ------------------------------------------------------------
po_loss_rows = df[
    (df["is_playoffs"].fillna(0).astype(int) == 1) &
    (df["loss"].fillna(0).astype(int) == 1)
][["manager_year", "week"]].dropna()

first_po_loss_week = (
    po_loss_rows.groupby("manager_year")["week"]
    .min()
    .astype("Int64")
)

loss_week_map = df["manager_year"].map(first_po_loss_week)
demote_mask = (
    loss_week_map.notna() &
    df["week"].notna() &
    (df["week"].astype("Int64") > loss_week_map.astype("Int64"))
)

df.loc[demote_mask, "is_consolation"] = pd.Series(1, index=df.index[demote_mask], dtype="Int64")
df.loc[demote_mask, "is_playoffs"] = pd.Series(0, index=df.index[demote_mask], dtype="Int64")
df["postseason"] = ((df["is_playoffs"] == 1) | (df["is_consolation"] == 1)).astype("Int64")

# ------------------------------------------------------------
# Head-to-head helper columns (optional)
# ------------------------------------------------------------
mgr_all = sorted(pd.unique(pd.concat([df["manager"], df["opponent"]]).dropna()))
mgr_tokens = {m: _mgr_token(m) for m in mgr_all}

for m in mgr_all:
    tok = mgr_tokens[m]
    opp_scores = (
        df.loc[df["manager"] == m]
          .set_index(["year", "week"])["team_points"]
          .groupby(level=[0, 1]).first()
    )
    aligned_opp_score = pd.Series(
        [opp_scores.get((y, w), np.nan) for y, w in zip(df["year"], df["week"])],
        index=df.index
    )
    has_opp_score = aligned_opp_score.notna()
    tp = _to_float(df["team_points"])

    df[f"w_vs_{tok}"] = (
        (df["manager"] != m) & has_opp_score & (tp > aligned_opp_score)
    ).astype("Int64")
    df[f"l_vs_{tok}"] = (
        (df["manager"] != m) & has_opp_score & (tp < aligned_opp_score)
    ).astype("Int64")

for m in mgr_all:
    tok = mgr_tokens[m]
    sched_opp = (
        df.loc[df["manager"] == m]
          .set_index(["year", "week"])["opponent_points"]
          .groupby(level=[0, 1]).first()
    )
    aligned = pd.Series(
        [sched_opp.get((y, w), np.nan) for y, w in zip(df["year"], df["week"])],
        index=df.index
    )
    has_sched = aligned.notna()
    tp = _to_float(df["team_points"])
    df[f"w_vs_{tok}_sched"] = ((tp > aligned) & has_sched).astype("Int64")
    df[f"l_vs_{tok}_sched"] = ((tp < aligned) & has_sched).astype("Int64")

# ------------------------------------------------------------
# playoff_seed_to_date (regular-season only, stable ranking)
# ------------------------------------------------------------
df = df.sort_values(["manager_year", "year", "week"]).reset_index(drop=True)

df["win_eff"] = (_to_int64(df["win"]).fillna(0).astype(int) * df["valid_season_game"]).astype(int)
df["loss_eff"] = (_to_int64(df["loss"]).fillna(0).astype(int) * df["valid_season_game"]).astype(int)
df["seed_wins_to_date"] = df.groupby(["year", "manager_year"])["win_eff"].cumsum()
df["seed_points_to_date"] = df.groupby(["year", "manager_year"])["team_points"].cumsum()

def _rank_seed_block(g: pd.DataFrame) -> pd.Series:
    # Rank by wins desc, then points desc
    wins = _to_float(g["seed_wins_to_date"]).fillna(-1).to_numpy()
    pts  = _to_float(g["seed_points_to_date"]).fillna(-1.0).to_numpy()
    order = np.lexsort((-pts, -wins))  # last key primary
    ranks = np.empty(len(g), dtype=int)
    ranks[order] = np.arange(1, len(g) + 1)
    return pd.Series(ranks, index=g.index, dtype="Int64")

df["playoff_seed_to_date"] = (
    df.groupby(["year", "week"], group_keys=False).apply(_rank_seed_block, include_groups=False)
).astype("Int64")

# ------------------------------------------------------------
# final_playoff_seed (rank ALL teams using cumulative stats up to last regular week)
# ------------------------------------------------------------
regular_mask = (df["is_playoffs"].fillna(0).astype(int) != 1)

# last regular-season week per year
last_regular_week = (
    df[regular_mask].groupby("year")["week"].max().dropna().astype(int)
)

# cumulative metrics (exclude consolation when accumulating)
non_consol = (df["is_consolation"].fillna(0).astype(int) != 1)
df["wins_to_date"] = (
    (_to_int64(df["win"]).fillna(0).astype(int) * non_consol.astype(int))
    .groupby([df["year"], df["manager_year"]]).cumsum()
).astype("Int64")
df["losses_to_date"] = (
    (_to_int64(df["loss"]).fillna(0).astype(int) * non_consol.astype(int))
    .groupby([df["year"], df["manager_year"]]).cumsum()
).astype("Int64")
df["points_scored_to_date"] = (
    (_to_float(df["team_points"]).fillna(0.0) * non_consol.astype(int))
    .groupby([df["year"], df["manager_year"]]).cumsum()
).astype(float)

seed_rows: List[pd.DataFrame] = []
for y, lrw in last_regular_week.items():
    reg_rows = df[
        (df["year"] == y) &
        (df["is_playoffs"].fillna(0).astype(int) != 1) &
        (df["is_consolation"].fillna(0).astype(int) != 1)
    ][["manager_year", "week", "wins_to_date", "points_scored_to_date"]].copy()

    if reg_rows.empty:
        continue

    # keep only rows at or before the last regular week
    reg_rows = reg_rows[pd.to_numeric(reg_rows["week"], errors="coerce").fillna(-1).astype(int) <= int(lrw)]

    # === robust: take the latest (max week) row per manager_year without idxmax ===
    snap = (
        reg_rows.sort_values(["manager_year", "week"])
                .groupby("manager_year", as_index=False, sort=False)
                .tail(1)[["manager_year", "wins_to_date", "points_scored_to_date"]]
                .copy()
    )

    # include any managers missing a regular-season row ≤ lrw (edge case) with zeros
    all_mys = (
        df.loc[(df["year"] == y), "manager_year"]
          .dropna()
          .astype(str)
          .unique()
          .tolist()
    )
    have = set(snap["manager_year"].astype(str))
    missing = [m for m in all_mys if m not in have]
    if missing:
        snap = pd.concat([
            snap,
            pd.DataFrame({
                "manager_year": missing,
                "wins_to_date": 0,
                "points_scored_to_date": 0.0,
            })
        ], ignore_index=True)

    # rank: wins desc, points desc, token asc
    snap["_mgr_token"] = snap["manager_year"].astype(str)
    snap = snap.sort_values(
        by=["wins_to_date", "points_scored_to_date", "_mgr_token"],
        ascending=[False, False, True],
        kind="mergesort"
    ).reset_index(drop=True)
    snap["final_playoff_seed"] = np.arange(1, len(snap) + 1, dtype=int)
    snap["year"] = int(y)
    seed_rows.append(snap[["year", "manager_year", "final_playoff_seed"]])

if seed_rows:
    seeds = pd.concat(seed_rows, ignore_index=True)
    df.drop(columns=["final_playoff_seed"], errors="ignore", inplace=True)
    df = df.merge(
        seeds,
        on=["year", "manager_year"],
        how="left",
        validate="m:1"
    )
    df["final_playoff_seed"] = df["final_playoff_seed"].astype("Int64")
else:
    if "final_playoff_seed" not in df.columns:
        df["final_playoff_seed"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

# ------------------------------------------------------------
# Round detection
# ------------------------------------------------------------
round_flags = pd.DataFrame(index=df.index, data={
    "quarterfinal": 0,
    "semifinal": 0,
    "championship": 0,
})

for y, ysub in df.groupby("year", dropna=True):
    if pd.isna(y):
        continue
    ply = ysub[(ysub["is_playoffs"].fillna(0).astype(int) == 1) &
               (ysub["is_consolation"].fillna(0).astype(int) != 1)]
    if ply.empty:
        continue

    wk_counts = (ply.groupby("week").size() / 2).astype(int)
    wk_counts = wk_counts[wk_counts > 0].sort_index()

    champ_weeks = wk_counts[wk_counts == 1]
    if champ_weeks.empty:
        continue

    champ_week = int(champ_weeks.index.max())
    champ_rows = ply[ply["week"] == champ_week]
    round_flags.loc[champ_rows.index, "championship"] = 1

    prev_playoff_weeks = wk_counts.index[wk_counts.index < champ_week]
    if len(prev_playoff_weeks) > 0:
        semi_week = int(prev_playoff_weeks.max())
        semi_rows = ply[ply["week"] == semi_week]
        round_flags.loc[semi_rows.index, "semifinal"] = 1

        prev2 = prev_playoff_weeks[prev_playoff_weeks < semi_week]
        if len(prev2) > 0:
            qf_week = int(prev2.max())
            qf_rows = ply[ply["week"] == qf_week]
            round_flags.loc[qf_rows.index, "quarterfinal"] = 1

df["quarterfinal"] = _to_int64(round_flags["quarterfinal"])
df["semifinal"] = _to_int64(round_flags["semifinal"])
df["championship"] = _to_int64(round_flags["championship"])

# ------------------------------------------------------------
# Year completion flags
# ------------------------------------------------------------
year_has_champ = (_to_int64(df.groupby("year")["championship"].max()).fillna(0).astype(int) == 1)
year_done_mask = df["year"].map(year_has_champ).fillna(False).astype(bool)

# Personal season mean/median up to championship week (exclude consolation)
champ_week_by_year = df.groupby("year").apply(
    lambda g: int(g.loc[g["championship"] == 1, "week"].max()) if (g["championship"] == 1).any() else np.inf,
    include_groups=False
)
df["__cap_week"] = df["year"].map(champ_week_by_year)

def _personal_stats(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("week").copy()
    cap = g["__cap_week"].iloc[0]
    valid_rows = (g["is_consolation"].fillna(0).astype(int) != 1) & (_to_int64(g["week"]).fillna(0).astype(int) <= cap)

    vals = _to_float(g.loc[valid_rows, "team_points"])
    mean_exp = vals.expanding().mean()
    med_exp  = vals.expanding().median()

    g["personal_season_mean"] = pd.Series(mean_exp.values, index=g.index[valid_rows], dtype="float64")
    g["personal_season_median"] = pd.Series(med_exp.values, index=g.index[valid_rows], dtype="float64")
    g["personal_season_mean"] = g["personal_season_mean"].ffill()
    g["personal_season_median"] = g["personal_season_median"].ffill()
    return g[["personal_season_mean", "personal_season_median"]]

df[["personal_season_mean", "personal_season_median"]] = (
    df.groupby(["year", "manager_year"], group_keys=False)
      .apply(_personal_stats, include_groups=False)
)

# Above/below opponent median for the year (exclude consolation when building the median)
non_consol_mask = (df["is_consolation"].fillna(0).astype(int) == 0)
opp_median_map = (
    df[non_consol_mask]
    .groupby(["year", "opponent"])["team_points"]
    .median()
)

def _opp_median_lookup(y, opp):
    try:
        return opp_median_map.loc[(y, opp)]
    except KeyError:
        return np.nan

op_med = pd.Series([_opp_median_lookup(y, o) for y, o in zip(df["year"], df["opponent"])], index=df.index)
tp_num = _to_float(df["team_points"])

df["above_opponent_median"] = (tp_num > op_med).astype("Int64")
df["below_opponent_median"] = (tp_num < op_med).astype("Int64")

# ------------------------------------------------------------
# Backfill league means/medians & projections if missing
# ------------------------------------------------------------
if "league_weekly_mean" not in df.columns or df["league_weekly_mean"].isna().all():
    lmeans = df.groupby(["year", "week"])["team_points"].mean(numeric_only=True)
    df["league_weekly_mean"] = [lmeans.get((y, w), np.nan) for y, w in zip(df["year"], df["week"])]

if "league_weekly_median" not in df.columns or df["league_weekly_median"].isna().all():
    lmeds = df.groupby(["year", "week"])["team_points"].median(numeric_only=True)
    df["league_weekly_median"] = [lmeds.get((y, w), np.nan) for y, w in zip(df["year"], df["week"])]

if "above_league_median" not in df.columns or df["above_league_median"].isna().all():
    df["above_league_median"] = (tp_num > df["league_weekly_median"]).astype("Int64")
if "below_league_median" not in df.columns or df["below_league_median"].isna().all():
    df["below_league_median"] = (tp_num < df["league_weekly_median"]).astype("Int64")

if "proj_wins" not in df.columns:
    df["proj_wins"] = (_to_float(df["team_projected_points"]) >
                       _to_float(df["opponent_projected_points"])).astype("Int64")
if "proj_losses" not in df.columns:
    df["proj_losses"] = (_to_float(df["team_projected_points"]) <
                         _to_float(df["opponent_projected_points"])).astype("Int64")

def _count_teams_beaten(s: pd.Series) -> pd.Series:
    s_num = _to_float(s)
    return s_num.groupby([df["year"], df["week"]]).transform(
        lambda g: (s_num.loc[g.index].values[:, None] > g.values).sum(axis=1)
    )

def _count_teams_beaten_by_opponent_points(opp_s: pd.Series) -> pd.Series:
    o = _to_float(opp_s)
    return o.groupby([df["year"], df["week"]]).transform(
        lambda g: (o.loc[g.index].values[:, None] > g.values).sum(axis=1)
    )

if "teams_beat_this_week" not in df.columns:
    df["teams_beat_this_week"] = _count_teams_beaten(df["team_points"]).astype("Int64")

if "opponent_teams_beat_this_week" not in df.columns:
    df["opponent_teams_beat_this_week"] = _count_teams_beaten_by_opponent_points(df["opponent_points"]).astype("Int64")

# ------------------------------------------------------------
# team_made_playoffs
# ------------------------------------------------------------
year_started_playoffs = (df.groupby("year")["is_playoffs"].max().fillna(0) == 1)
made_po_by_team = (df.groupby("manager_year")["is_playoffs"].max().fillna(0).astype(int))
df = df.merge(year_started_playoffs.rename("tmp_year_started_po"), left_on="year", right_index=True, how="left")
mask_started = df["tmp_year_started_po"] == True
df.loc[mask_started, "team_made_playoffs"] = df.loc[mask_started, "manager_year"].map(made_po_by_team).astype("Int64")
df.drop(columns=["tmp_year_started_po"], inplace=True)

# ------------------------------------------------------------
# champion (use original wins)
# ------------------------------------------------------------
champ_map: Dict[str, int] = {}
for y, ysub in df.groupby("year", dropna=True):
    champ_rows = ysub[_to_int64(ysub["championship"]).fillna(0).astype(int) == 1]
    if champ_rows.empty:
        continue
    winner_rows = champ_rows[_to_int64(champ_rows["win"]).fillna(0).astype(int) == 1]
    if winner_rows.empty:
        tp = _to_float(champ_rows["team_points"])
        op = _to_float(champ_rows["opponent_points"])
        winner_rows = champ_rows[(tp > op) & tp.notna() & op.notna()]
    if winner_rows.empty:
        continue
    my_val = winner_rows.iloc[0]["manager_year"]
    if pd.notna(my_val):
        wy = str(my_val)
        if wy and wy != "" and wy != "nan" and wy != "<NA>":
            champ_map[wy] = 1

df["champion"] = df["manager_year"].map(champ_map).fillna(0).astype("Int64")

# ------------------------------------------------------------
# final regular wins/losses (regular season only; original wins/losses)
# ------------------------------------------------------------
valid_games = df[df["valid_season_game"] == 1].copy()
valid_games["win_int"] = _to_int64(valid_games["win"]).fillna(0).astype(int)
valid_games["loss_int"] = _to_int64(valid_games["loss"]).fillna(0).astype(int)

agg_reg_wins = valid_games.groupby("manager_year")["win_int"].sum()
agg_reg_losses = valid_games.groupby("manager_year")["loss_int"].sum()

df.loc[year_done_mask, "final_regular_wins"] = df.loc[year_done_mask, "manager_year"].map(agg_reg_wins).astype("Int64")
df.loc[year_done_mask, "final_regular_losses"] = df.loc[year_done_mask, "manager_year"].map(agg_reg_losses).astype("Int64")

# ------------------------------------------------------------
# final total wins/losses (exclude consolation; original wins/losses)
# ------------------------------------------------------------
full = df[_to_int64(df["is_consolation"]).fillna(0).astype(int) != 1].copy()
full["win_int"] = _to_int64(full["win"]).fillna(0).astype(int)
full["loss_int"] = _to_int64(full["loss"]).fillna(0).astype(int)

full_wins = full.groupby("manager_year")["win_int"].sum()
full_losses = full.groupby("manager_year")["loss_int"].sum()

df.loc[year_done_mask, "final_wins"] = df.loc[year_done_mask, "manager_year"].map(full_wins).astype("Int64")
df.loc[year_done_mask, "final_losses"] = df.loc[year_done_mask, "manager_year"].map(full_losses).astype("Int64")

# ------------------------------------------------------------
# team_got_bye (first playoff game occurs after first playoff week)
# ------------------------------------------------------------
bye_flags = pd.Series(0, index=df.index, dtype="Int64")
for y, ysub in df.groupby("year", dropna=True):
    if pd.isna(y):
        continue
    ply = ysub[(_to_int64(ysub["is_playoffs"]).fillna(0).astype(int) == 1) &
               (_to_int64(ysub["is_consolation"]).fillna(0).astype(int) != 1)]
    if ply.empty:
        continue
    playoff_weeks = sorted(_to_int64(ply["week"]).dropna().unique().tolist())
    if not playoff_weeks:
        continue
    first_playoff_week = playoff_weeks[0]
    for my in ply["manager_year"].unique():
        if pd.isna(my):
            continue
        my_games = ply[ply["manager_year"] == my].sort_values("week")
        if my_games.empty:
            continue
        first_game_week = int(my_games.iloc[0]["week"])
        if first_game_week > first_playoff_week:
            mask = (df["manager_year"] == my) & (df["year"] == y)
            bye_flags.loc[mask] = 1
df["team_got_bye"] = bye_flags

# ------------------------------------------------------------
# Streaks (exclude consolation)
# ------------------------------------------------------------
def _streaks_across_cw(sub: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    sub = sub.sort_values(["cumulative_week", "year", "week"], na_position="last")
    ws, ls = [], []
    win_run, lose_run = 0, 0
    for _, r in sub.iterrows():
        is_consol = int((_to_int64(pd.Series([r.get("is_consolation", 0)])).iloc[0] or 0))
        if is_consol == 1:
            ws.append(win_run); ls.append(lose_run); continue
        w = int((_to_int64(pd.Series([r.get("win", 0)])).iloc[0] or 0))
        l = int((_to_int64(pd.Series([r.get("loss", 0)])).iloc[0] or 0))
        if w == 1:
            win_run += 1; lose_run = 0
        elif l == 1:
            lose_run += 1; win_run = 0
        else:
            win_run = 0; lose_run = 0
        ws.append(win_run); ls.append(lose_run)
    return pd.Series(ws, index=sub.index, dtype="Int64"), pd.Series(ls, index=sub.index, dtype="Int64")

all_ws = pd.Series(index=df.index, dtype="Int64")
all_ls = pd.Series(index=df.index, dtype="Int64")
for _, g in df.groupby("manager", sort=False):
    ws, ls = _streaks_across_cw(g)
    all_ws.loc[ws.index] = ws
    all_ls.loc[ls.index] = ls

df["winning_streak"] = all_ws.astype("Int64")
df["losing_streak"] = all_ls.astype("Int64")

# ------------------------------------------------------------
# Inflation + real_* (always recompute)
# ------------------------------------------------------------
year_means = df.groupby("year")["team_points"].mean(numeric_only=True)
if not year_means.empty:
    base_year = int(year_means.dropna().index.min())
    base_mean = float(year_means.loc[base_year]) if not math.isnan(year_means.loc[base_year]) else None
else:
    base_year, base_mean = None, None

infl_map = {int(y): 1.0 for y in df["year"].dropna().unique()} if (base_mean is None or base_mean == 0) \
    else {int(y): float(m) / base_mean for y, m in year_means.dropna().items()}

df["inflation_rate"] = df["year"].map(infl_map).astype(float)
df["real_score"] = (_to_float(df["team_points"]) / df["inflation_rate"]).astype(float)
df["real_opponent_points"] = (_to_float(df["opponent_points"]) / df["inflation_rate"]).astype(float)
df["real_margin"] = (df["real_score"] - df["real_opponent_points"]).astype(float)
df["real_total_matchup_score"] = (df["real_score"] + df["real_opponent_points"]).astype(float)

# ------------------------------------------------------------
# Rankings (exclude postseason & consolation)
# ------------------------------------------------------------
rank_work = df.copy()
rank_work["team_points"] = _to_float(rank_work["team_points"])
rank_work = rank_work[
    (rank_work["team_points"].notna()) &
    (_to_int64(rank_work["is_consolation"]).fillna(0).astype(int) == 0) &
    (_to_int64(rank_work["is_playoffs"]).fillna(0).astype(int) == 0)
]
r, _ = _compute_rank_pct(rank_work, ["manager", "year"], "team_points")
df["manager_season_ranking"] = r.reindex(df.index).astype("Int64")
r, _ = _compute_rank_pct(rank_work, ["year"], "team_points")
df["league_season_ranking"] = r.reindex(df.index).astype("Int64")
r, p = _compute_rank_pct(rank_work, ["manager"], "team_points")
df["manager_all_time_ranking"] = r.reindex(df.index).astype("Int64")
df["manager_all_time_ranking_percentile"] = p.reindex(df.index).astype(float)
r, p = _compute_rank_pct(rank_work, [], "team_points")
df["league_all_time_ranking"] = r.reindex(df.index).astype("Int64")
df["league_all_time_ranking_percentile"] = p.reindex(df.index).astype(float)

for src_col, aliases in RANK_ALIAS_MAP.items():
    if src_col in df.columns:
        for alias in aliases:
            df[alias] = df[src_col]

# ------------------------------------------------------------
# SACKO (exclude teams that made playoffs)
# ------------------------------------------------------------
cons = df[_to_int64(df["is_consolation"]).fillna(0).astype(int) == 1].copy()
cons["loss_calc"] = _to_int64(cons["loss"]).fillna(0).astype(int)
need_loss = cons["loss_calc"] == 0

if need_loss.any():
    tp = _to_float(cons.loc[need_loss, "team_points"])
    op = _to_float(cons.loc[need_loss, "opponent_points"])
    cons.loc[need_loss & tp.notna() & op.notna(), "loss_calc"] = (tp < op).astype(int)

cons_games = cons.groupby("manager_year").size()
cons_losses = cons.groupby("manager_year")["loss_calc"].sum()
losing_all = (cons_games > 0) & (cons_losses == cons_games)

has_playoff = _to_int64(df.groupby("manager_year")["is_playoffs"].max()).fillna(0).astype(int)
sacko_by_my = losing_all.reindex(has_playoff.index, fill_value=False).astype(bool) & (has_playoff == 0)
sacko_by_my = sacko_by_my.astype(int)

df.loc[year_done_mask, "sacko"] = df.loc[year_done_mask, "manager_year"].map(sacko_by_my).astype("Int64")

# ------------------------------------------------------------
# Playoff rounds (exclude consolation)
# ------------------------------------------------------------
ply = df[(_to_int64(df["is_playoffs"]).fillna(0).astype(int) == 1) &
         (_to_int64(df["is_consolation"]).fillna(0).astype(int) != 1)].copy()
ply["win_int"] = _to_int64(ply["win"]).fillna(0).astype(int)

rounds_won_by_my = ply.groupby("manager_year")["win_int"].sum()
rounds_played_by_my = ply.groupby("manager_year").size()

df["playoff_rounds_won"] = df["manager_year"].map(rounds_won_by_my).astype("Int64")
df["playoff_rounds_played"] = df["manager_year"].map(rounds_played_by_my).astype("Int64")

# ------------------------------------------------------------
# String scrub
# ------------------------------------------------------------
for col in [
    "manager", "opponent", "team_name", "manager_year", "opponent_year",
    "manager_week", "opponent_week", "matchup_recap_title",
    "matchup_recap_url", "url", "image_url", "value"
]:
    if col in df.columns:
        df[col] = df[col].astype("string").map(_strip_zw)

# ------------------------------------------------------------
# Column ordering (optional)
# ------------------------------------------------------------
BASE_ORDER: List[str] = [
    "week","year","manager","team_name","team_points","team_projected_points","opponent",
    "opponent_points","opponent_projected_points","margin","total_matchup_score","close_margin",
    "weekly_mean","weekly_median","league_weekly_mean","league_weekly_median",
    "above_league_median","below_league_median",
    "win","loss","proj_wins","proj_losses","teams_beat_this_week","opponent_teams_beat_this_week",
    "proj_score_error","abs_proj_score_error","above_proj_score","below_proj_score",
    "expected_spread","expected_odds","win_vs_spread","lose_vs_spread","underdog_wins","favorite_losses",
    "is_playoffs","is_consolation","gpa","grade","matchup_recap_title","matchup_recap_url","url","image_url",
    "division_id","week_start","week_end","waiver_priority","has_draft_grade","faab_balance","number_of_moves",
    "number_of_trades","auction_budget_spent","auction_budget_total","win_probability","coverage_value","value",
    "felo_score","felo_tier"
]

h2h_w_cols  = sorted([c for c in df.columns if re.fullmatch(r"w_vs_[a-z0-9_]+(?<!_sched)$", c)])
h2h_l_cols  = sorted([c for c in df.columns if re.fullmatch(r"l_vs_[a-z0-9_]+(?<!_sched)$", c)])
h2h_w_sched = sorted([c for c in df.columns if re.fullmatch(r"w_vs_[a-z0-9_]+_sched$", c)])
h2h_l_sched = sorted([c for c in df.columns if re.fullmatch(r"l_vs_[a-z0-9_]+_sched$", c)])

TAIL_ORDER: List[str] = [
    "is_playoffs_raw","is_consolation_raw","postseason","playoff_rounds_won","manager_season_rank",
    "final_regular_losses","sacko","league_alltime_rank","manager_alltime_rank","final_wins",
    "playoff_rounds_played","winning_streak","team_made_playoffs","league_all_time_ranking",
    "league_season_rank","quarterfinal","final_playoff_seed","final_losses","team_got_bye",
    "league_season_ranking","final_regular_wins","championship","manager_season_ranking",
    "manager_all_time_ranking","cumulative_week","semifinal","losing_streak","playoff_seed_to_date",
    "champion","manager_all_time_ranking_percentile","real_total_matchup_score","real_opponent_points",
    "real_score","league_all_time_ranking_percentile","manager_alltime_percentile","inflation_rate",
    "league_alltime_percentile","real_margin","opponent_week","manager_week","manager_year","opponent_year",
    "valid_season_game","wins_to_date","losses_to_date","points_scored_to_date",
    "personal_season_mean","personal_season_median","above_opponent_median","below_opponent_median"
]

preferred = BASE_ORDER + h2h_w_cols + h2h_l_cols + h2h_w_sched + h2h_l_sched + TAIL_ORDER
seen = set()
ordered = []
for c in preferred + list(df.columns):
    if c not in seen:
        seen.add(c)
        ordered.append(c)

for c in preferred:
    if c not in df.columns:
        df[c] = pd.NA

df = df.reindex(columns=ordered, fill_value=pd.NA)

# Write outputs
DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(PARQUET_PATH, index=False)
df.to_csv(CSV_PATH, index=False)
