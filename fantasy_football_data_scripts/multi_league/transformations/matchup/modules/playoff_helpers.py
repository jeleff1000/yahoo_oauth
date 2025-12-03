
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

"""
Playoff Helpers Module

Utility functions for playoff odds calculations.

This module contains helper functions for:
- Matchup canonicalization
- Wins/points aggregation
- Seeding/ranking
- Schedule parsing
- History snapshot generation
"""

from functools import wraps
import sys
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

_scripts_dir = _multi_league_dir.parent  # fantasy_football_data_scripts directory
sys.path.insert(0, str(_scripts_dir))  # Allows: from multi_league.core.XXX
sys.path.insert(0, str(_multi_league_dir))  # Allows: from core.XXX

from core.data_normalization import normalize_numeric_columns, ensure_league_id
import numpy as np
import pandas as pd
from typing import Tuple, Optional
@ensure_normalized
def add_match_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical matchup key (alphabetically sorted teams)."""
    df = df.copy()
    df["mA"] = df[["manager", "opponent"]].min(axis=1)
    df["mB"] = df[["manager", "opponent"]].max(axis=1)
    df["match_key"] = list(zip(df["year"], df["week"], df["mA"], df["mB"]))
    return df
@ensure_normalized
def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce to one row per matchup (canonical perspective)."""
    df = df.copy()
    df["mA"] = df[["manager", "opponent"]].min(axis=1)
    df["mB"] = df[["manager", "opponent"]].max(axis=1)
    df["match_key"] = list(zip(df["year"], df["week"], df["mA"], df["mB"]))
    return df[df["manager"] == df["mA"]]
def wins_points_to_date(played_raw: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate cumulative wins and points for each manager.
    Args:
        played_raw: DataFrame with matchup data (manager, opponent, team_points, year, week)
    Returns:
        wins: Series of cumulative wins per manager
        points: Series of cumulative points per manager
    """
    tmp = add_match_key(played_raw)
    def _win_val(s):
        """Determine win value (1.0 for win, 0.0 for loss, 0.5 for tie)."""
        if len(s) != 2:
            # Invalid matchup (duplicate or missing data) - return NaN for all records
            return pd.Series([np.nan] * len(s), index=s.index)
        a, b = s.iloc[0], s.iloc[1]
        if a > b:
            return pd.Series([1.0, 0.0], index=s.index)
        if b > a:
            return pd.Series([0.0, 1.0], index=s.index)
        return pd.Series([0.5, 0.5], index=s.index)
    tmp["win_val"] = tmp.groupby("match_key")["team_points"].transform(lambda s: _win_val(s).values)
    wins = tmp.groupby("manager")["win_val"].sum().astype(float)
    pts = tmp.groupby("manager")["team_points"].sum().astype(float)
    return wins, pts
def rank_and_seed(
    wins: pd.Series,
    points: pd.Series,
    playoff_slots: int,
    bye_slots: int,
    played_raw: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Rank managers and determine playoff seeding.
    Seeding is determined by:
    1. Total wins (descending)
    2. Total points (descending, tiebreaker)
    Args:
        wins: Series of wins per manager
        points: Series of points per manager
        playoff_slots: Number of playoff spots
        bye_slots: Number of first-round byes
        played_raw: Optional matchup data for calculating games played
    Returns:
        DataFrame with columns: seed, manager, W, L, PF, made_playoffs, bye
    """
    managers = sorted(set(wins.index) | set(points.index))
    w = wins.reindex(managers).fillna(0.0)
    pf = points.reindex(managers).fillna(0.0)
    # Calculate games played (for losses)
    if played_raw is not None and len(played_raw) > 0:
        mgr_weeks = pd.concat([
            played_raw[["manager", "year", "week"]].rename(columns={"manager": "name"}),
            played_raw[["opponent", "year", "week"]].rename(columns={"opponent": "name"})
        ])
        games_played = (mgr_weeks.drop_duplicates()
                       .groupby("name").size()
                       .reindex(managers).fillna(0).astype(int))
    else:
        games_played = pd.Series(0, index=managers)
    l = (games_played - w).clip(lower=0)
    # Create seeding table
    table = (pd.DataFrame({
        "manager": managers,
        "W": w.values,
        "L": l.values,
        "PF": pf.values
    })
             .sort_values(["W", "PF"], ascending=[False, False])
             .reset_index(drop=True))
    table["seed"] = np.arange(1, len(table) + 1)
    table["made_playoffs"] = table["seed"] <= playoff_slots
    table["bye"] = table["seed"] <= bye_slots
    return table[["seed", "manager", "W", "L", "PF", "made_playoffs", "bye"]]
@ensure_normalized
def enforce_playoff_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure once playoffs start, all subsequent weeks are marked as playoffs.
    This prevents data inconsistencies where later regular season weeks
    accidentally get marked as non-playoff.
    """
    df = df.copy()
    if "is_playoffs" not in df.columns:
        return df
    for yr in sorted(df["year"].dropna().unique().astype(int)):
        season_mask = df["year"] == yr
        po_weeks = df.loc[season_mask & (df["is_playoffs"] == 1), "week"].dropna()
        if po_weeks.empty:
            continue
        start_po = int(po_weeks.min())
        df.loc[season_mask & (df["week"] >= start_po), "is_playoffs"] = 1
    return df
def schedules_last_regular_week(df_sched: pd.DataFrame, season: int) -> Optional[int]:
    """Find the last regular season week for a given season."""
    # Handle empty schedule or missing required columns
    if df_sched.empty or "year" not in df_sched.columns:
        return None
    s = df_sched[(df_sched["year"] == season)]
    if s.empty:
        return None
    po = s.loc[s["is_playoffs"] == 1, "week"].dropna()
    if not po.empty:
        return int(po.min()) - 1
    return int(s["week"].max())
@ensure_normalized
def future_regular_from_schedule(
    df_sched: pd.DataFrame,
    season: int,
    current_week: int
) -> pd.DataFrame:
    """
    Extract future regular season games from schedule.
    Args:
        df_sched: Schedule DataFrame
        season: Current season
        current_week: Current week
    Returns:
        DataFrame with columns: year, week, manager, opponent
    """
    # Handle empty schedule or missing required columns
    if df_sched.empty or "year" not in df_sched.columns:
        return pd.DataFrame(columns=["year", "week", "manager", "opponent"])
    s = df_sched[(df_sched["year"] == season) & (df_sched["is_playoffs"] == 0)]
    if s.empty:
        return pd.DataFrame(columns=["year", "week", "manager", "opponent"])
    s = s[s["week"] > current_week].copy()
    if s.empty:
        return s[["year", "week", "manager", "opponent"]]
    # Canonicalize to avoid double-counting matchups
    s = canonicalize(s)
    return s[["year", "week", "manager", "opponent"]].drop_duplicates()
def history_snapshots(
    all_games: pd.DataFrame,
    playoff_slots: int
) -> pd.DataFrame:
    """
    Generate historical snapshots for kernel-based seed prediction.
    For each season and week, captures:
    - Current record (W, L)
    - Points percentile
    - Final playoff outcome (made_playoffs, final_seed)
    This data is used for empirical seed distribution estimation.
    Args:
        all_games: All historical matchup data
        playoff_slots: Number of playoff spots
    Returns:
        DataFrame with columns: year, week, manager, W, L, PF_pct, made_playoffs, final_seed
    """
    rows = []
    seasons = sorted(all_games["year"].dropna().unique().astype(int))
    for yr in seasons:
        reg = all_games[(all_games["year"] == yr) & (all_games["is_playoffs"] == 0)]
        if reg.empty:
            continue
        # Final standings for this season
        wins_f, pts_f = wins_points_to_date(reg)
        final_table = rank_and_seed(wins_f, pts_f, playoff_slots, 2, played_raw=reg)
        final_seed_map = dict(zip(final_table["manager"], final_table["seed"]))
        made = set(final_table.loc[final_table["made_playoffs"], "manager"])
        # Snapshot at each week
        weeks = sorted(reg["week"].dropna().unique().astype(int))
        for w in weeks:
            played = reg[reg["week"] <= w]
            wins_w, pts_w = wins_points_to_date(played)
            # Games played
            gp = (pd.concat([
                played[["manager", "year", "week"]].rename(columns={"manager": "name"}),
                played[["opponent", "year", "week"]].rename(columns={"opponent": "name"})
            ])
                  .drop_duplicates()
                  .groupby("name")["week"].nunique())
            mgrs = sorted(set(wins_w.index) | set(gp.index) | set(pts_w.index))
            if not mgrs:
                continue
            W = wins_w.reindex(mgrs).fillna(0.0)
            PF = pts_w.reindex(mgrs).fillna(0.0)
            GP = gp.reindex(mgrs).fillna(0).astype(int)
            L = (GP - W).clip(lower=0).astype(int)
            pf_pct = 100.0 * PF.rank(pct=True)
            for m in mgrs:
                rows.append({
                    "year": yr,
                    "week": int(w),
                    "manager": m,
                    "W": float(W.loc[m]),
                    "L": float(L.loc[m]),
                    "PF_pct": float(pf_pct.loc[m]),
                    "made_playoffs": 1.0 if m in made else 0.0,
                    "final_seed": final_seed_map.get(m, np.nan)
                })
    return pd.DataFrame(rows)
def gaussian_kernel(d2: np.ndarray) -> np.ndarray:
    """Gaussian kernel for distance weighting."""
    return np.exp(-0.5 * d2)
@ensure_normalized
def empirical_kernel_seed_dist(
    played_raw: pd.DataFrame,
    week: int,
    history_snapshots_df: pd.DataFrame,
    n_teams: int,
    h_W: float = 0.9,
    h_L: float = 0.9,
    h_PF: float = 15.0,
    h_week: float = 0.9,
    prior_strength: float = 3.0
) -> pd.DataFrame:
    """
    Empirical kernel-based seed distribution estimation.
    Uses historical snapshots with similar records to predict seed probabilities.
    Args:
        played_raw: Current season games played
        week: Current week
        history_snapshots_df: Historical snapshot data
        n_teams: Number of teams in league
        h_W: Bandwidth for wins
        h_L: Bandwidth for losses
        h_PF: Bandwidth for points percentile
        h_week: Bandwidth for week
        prior_strength: Dirichlet prior strength
    Returns:
        DataFrame with seed probabilities (managers x seeds)
    """
    if played_raw.empty or history_snapshots_df.empty:
        return pd.DataFrame()
    wins_w, pts_w = wins_points_to_date(played_raw)
    gp = (pd.concat([
        played_raw[["manager", "year", "week"]].rename(columns={"manager": "name"}),
        played_raw[["opponent", "year", "week"]].rename(columns={"opponent": "name"})
    ])
          .drop_duplicates()
          .groupby("name")["week"].nunique())
    mgrs = sorted(set(wins_w.index) | set(gp.index) | set(pts_w.index))
    if not mgrs:
        return pd.DataFrame()
    W = wins_w.reindex(mgrs).fillna(0.0)
    PF = pts_w.reindex(mgrs).fillna(0.0)
    GP = gp.reindex(mgrs).fillna(0).astype(int)
    L = (GP - W).clip(lower=0).astype(int)
    pf_pct = 100.0 * PF.rank(pct=True)
    H = history_snapshots_df.dropna(subset=["final_seed"]).copy()
    seeds = np.arange(1, n_teams + 1)
    cols = list(seeds)
    out = pd.DataFrame(0.0, index=mgrs, columns=cols)
    alpha0 = np.ones(n_teams) * (prior_strength / n_teams)
    H_seed = H["final_seed"].to_numpy()
    for m in mgrs:
        xW, xL, xPF, xwk = float(W.loc[m]), float(L.loc[m]), float(pf_pct.loc[m]), float(week)
        # Calculate squared distances
        dW = (H["W"].to_numpy() - xW) / max(1e-6, h_W)
        dL = (H["L"].to_numpy() - xL) / max(1e-6, h_L)
        dPF = (H["PF_pct"].to_numpy() - xPF) / max(1e-6, h_PF)
        dWK = (H["week"].to_numpy() - xwk) / max(1e-6, h_week)
        d2 = dW * dW + dL * dL + dPF * dPF + dWK * dWK
        # Kernel weights
        w = gaussian_kernel(d2)
        # Count weighted occurrences of each seed
        counts = np.zeros(n_teams)
        for k in range(1, n_teams + 1):
            counts[k - 1] = float(w[(H_seed == k)].sum())
        # Bayesian smoothing with Dirichlet prior
        probs = (counts + alpha0) / (counts.sum() + alpha0.sum() + 1e-12)
        out.loc[m, cols] = probs
    return out
@ensure_normalized
def normalize_seed_matrix_to_100(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize seed probability matrix so each seed sums to 100%."""
    if seed_df is None or seed_df.empty:
        return seed_df
    seed_df = seed_df.copy()
    for col in seed_df.columns:
        col_sum = float(seed_df[col].sum())
        seed_df[col] = seed_df[col] * (100.0 / col_sum) if col_sum > 0 else 0.0
    return seed_df
def p_playoffs_from_seeds(seed_df: pd.DataFrame, slots: int) -> pd.Series:
    """Calculate playoff probability from seed distribution."""
    if seed_df is None or seed_df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in seed_df.columns if isinstance(c, int) and 1 <= c <= slots]
    return seed_df[cols].sum(axis=1)
def p_bye_from_seeds(seed_df: pd.DataFrame, bye_slots: int) -> pd.Series:
    """Calculate bye probability from seed distribution."""
    if seed_df is None or seed_df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in seed_df.columns if isinstance(c, int) and 1 <= c <= bye_slots]
    return seed_df[cols].sum(axis=1)