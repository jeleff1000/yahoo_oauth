
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
Schedule Simulation Module

Core simulation engine for schedule-independent record calculations.

This module handles:
- Round-robin schedule generation
- Random schedule validation
- Expected record simulation (performance-based)
- Schedule strength simulation (opponent-based)
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
from typing import Dict, Tuple, List, Set
from collections import defaultdict
import random
def round_robin_weeks(managers: List[str]) -> Dict[int, List[Tuple[str, str]]]:
    """
    Generate round-robin schedule using rotation algorithm.
    Args:
        managers: List of manager names (must be even number)
    Returns:
        Dict mapping week number to list of matchups
    """
    teams = list(managers)
    n = len(teams)
    assert n % 2 == 0 and n >= 4, "Team count must be even and >= 4"
    left = teams[: n // 2]
    right = teams[n // 2:][::-1]
    weeks = {}
    for w in range(1, n):
        weeks[w] = [(a, b) for a, b in zip(left, right)]
        if n > 2:
            right.insert(0, left.pop(1))
            left.append(right.pop())
    return weeks
def validate_schedule(
    sched: Dict[int, List[Tuple[str, str]]],
    teams: List[str],
    no_repeats_weeks: int = 5,
    max_meetings: int = 2
) -> Tuple[bool, str]:
    """
    Validate schedule constraints.
    Checks:
    - No team plays twice in same week
    - No matchup repeats in first N weeks
    - No pair plays more than max_meetings times
    - Every team plays exactly n_weeks games
    Args:
        sched: Schedule dict
        teams: List of team names
        no_repeats_weeks: No repeat matchups in first N weeks
        max_meetings: Maximum times any pair can meet
    Returns:
        (is_valid, message)
    """
    pair_ct = defaultdict(int)
    earliest = min(sched.keys())
    latest = max(sched.keys())
    early_end = min(earliest + no_repeats_weeks - 1, latest)
    seen_early = set()
    # Check early weeks for repeats
    for w in range(earliest, early_end + 1):
        for a, b in sched[w]:
            p = tuple(sorted((a, b)))
            if p in seen_early:
                return False, f"Repeat within weeks {earliest}-{early_end}: {p} in week {w}"
            seen_early.add(p)
    # Check weekly constraints
    for w, games in sched.items():
        used = set()
        for a, b in games:
            if a in used or b in used:
                return False, f"Team plays twice in week {w}"
            used.add(a)
            used.add(b)
            p = tuple(sorted((a, b)))
            pair_ct[p] += 1
            if pair_ct[p] > max_meetings:
                return False, f"Pair > {max_meetings} meetings: {p}"
    # Check total games per team
    team_games = defaultdict(int)
    n_weeks = len(sched)
    for games in sched.values():
        for a, b in games:
            team_games[a] += 1
            team_games[b] += 1
    for t in teams:
        if team_games[t] != n_weeks:
            return False, f"{t} has {team_games[t]} games (need {n_weeks})"
    return True, "OK"
def schedule_nxN(
    managers: List[str],
    n_weeks: int,
    rng: random.Random,
    validate: bool = False
) -> Dict[int, List[Tuple[str, str]]]:
    """
    Generate random N-week schedule for N teams.
    Uses round-robin as base, shuffles teams and weeks for randomness.
    Args:
        managers: List of manager names
        n_weeks: Number of weeks to generate
        rng: Random number generator for reproducibility
        validate: Whether to validate constraints
    Returns:
        Dict mapping week to list of matchups
    """
    teams = list(managers)
    n = len(teams)
    assert n % 2 == 0 and n >= 4, "Team count must be even and ≥4"
    max_weeks = 2 * (n - 1)
    assert n_weeks <= max_weeks, f"n_weeks>{max_weeks} would force some pair >2 meetings"
    # Shuffle teams for randomness
    M = list(managers)
    rng.shuffle(M)
    # Generate round-robin
    rr = round_robin_weeks(M)
    base_weeks = list(rr.keys())
    last_rr = base_weeks[-1]
    # Take first n_weeks from round-robin
    sched = {w: rr[w][:] for w in range(1, min(last_rr, n_weeks) + 1)}
    # If need more weeks, randomly pick from round-robin
    extra = max(0, n_weeks - last_rr)
    if extra:
        pick = rng.sample(base_weeks, extra)
        rng.shuffle(pick)
        for off, base_w in enumerate(pick, start=last_rr + 1):
            sched[off] = rr[base_w][:]
    # Shuffle weeks 6+ for additional randomness
    if n_weeks > 5:
        tail_old = list(range(6, n_weeks + 1))
        rng.shuffle(tail_old)
        remapped = {}
        for w in range(1, 6):
            remapped[w] = sched[w]
        for new_w, old_w in zip(range(6, n_weeks + 1), tail_old):
            remapped[new_w] = sched[old_w]
        sched = remapped
    if validate:
        ok, msg = validate_schedule(sched, M)
        if not ok:
            raise RuntimeError(f"Schedule invalid: {msg}")
    return sched
def simulate_once_performance(
    points_by_mgr_week: Dict[Tuple[str, int], float],
    managers: List[str],
    n_weeks: int,
    rng: random.Random
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Simulate one random schedule using actual team_points.
    Compares each manager's team_points against opponent's team_points
    in randomly generated matchups.
    Args:
        points_by_mgr_week: Dict[(manager, week)] -> team_points
        managers: List of managers
        n_weeks: Number of weeks
        rng: Random generator
    Returns:
        (wins_dict, seeds_dict)
    """
    sched = schedule_nxN(managers, n_weeks, rng, validate=False)
    wins = defaultdict(int)
    cum_points = defaultdict(float)
    for w in range(1, n_weeks + 1):
        for a, b in sched[w]:
            pa = points_by_mgr_week.get((a, w))
            pb = points_by_mgr_week.get((b, w))
            if pa is None or pb is None:
                continue
            cum_points[a] += pa
            cum_points[b] += pb
            if pa > pb:
                wins[a] += 1
            elif pb > pa:
                wins[b] += 1
            else:
                # Coin flip for ties
                wins[a if rng.random() < 0.5 else b] += 1
    # Seed by wins (descending), then points (descending)
    ladder = [(m, wins[m], cum_points[m]) for m in managers]
    ladder.sort(key=lambda x: (-x[1], -x[2], x[0]))
    seeds = {m: i + 1 for i, (m, _, _) in enumerate(ladder)}
    return wins, seeds
def simulate_once_opponent_difficulty(
    opp_points_by_mgr_week: Dict[Tuple[str, int], float],
    managers: List[str],
    n_weeks: int,
    rng: random.Random
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Simulate one random schedule using opponent_points (schedule strength).
    Compares opponent difficulty: LOWER opponent_points = easier schedule = more "wins".
    This measures schedule luck rather than performance.
    Args:
        opp_points_by_mgr_week: Dict[(manager, week)] -> opponent_points
        managers: List of managers
        n_weeks: Number of weeks
        rng: Random generator
    Returns:
        (wins_dict, seeds_dict) where wins = # of easy weeks
    """
    sched = schedule_nxN(managers, n_weeks, rng, validate=False)
    wins = defaultdict(int)  # "wins" = easier weeks
    cum_opp = defaultdict(float)  # Lower is better
    for w in range(1, n_weeks + 1):
        for a, b in sched[w]:
            pa = opp_points_by_mgr_week.get((a, w))  # opponent_points manager A faced
            pb = opp_points_by_mgr_week.get((b, w))  # opponent_points manager B faced
            if pa is None or pb is None:
                continue
            cum_opp[a] += pa
            cum_opp[b] += pb
            # FLIPPED: Lower opponent_points = easier = win
            if pa < pb:
                wins[a] += 1
            elif pb < pa:
                wins[b] += 1
            else:
                wins[a if rng.random() < 0.5 else b] += 1
    # Seed by more easy weeks (wins descending), then lower cum_opp (ascending)
    ladder = [(m, wins[m], cum_opp[m]) for m in managers]
    ladder.sort(key=lambda x: (-x[1], x[2], x[0]))  # More wins better, lower opp better
    seeds = {m: i + 1 for i, (m, _, _) in enumerate(ladder)}
    return wins, seeds
def current_regular_week(df_season: pd.DataFrame) -> int:
    """
    Find the latest completed regular season week.
    A week is complete if all managers have played and all scores are recorded.
    Args:
        df_season: Season matchup data
    Returns:
        Latest complete week number (0 if none)
    """
    managers = sorted(df_season['manager'].unique())
    weeks = sorted(df_season['week'].unique())
    complete = 0
    for w in weeks:
        block = df_season[df_season['week'] == w]
        if len(block['manager'].unique()) == len(managers) and block['team_points'].notna().all():
            complete = w
        else:
            break
    return int(complete)
def run_simulations(
    points_dict: Dict[Tuple[str, int], float],
    managers: List[str],
    n_weeks: int,
    n_sims: int,
    rng: random.Random,
    mode: str = "performance"
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Run N simulations and collect win/seed distributions.
    Args:
        points_dict: Dict[(manager, week)] -> points value
        managers: List of managers
        n_weeks: Number of weeks
        n_sims: Number of simulations
        rng: Random generator
        mode: "performance" (team_points) or "opponent" (opponent_points)
    Returns:
        (win_histograms, seed_histograms)
    """
    win_hists = {mgr: [0] * (n_weeks + 1) for mgr in managers}
    seed_hists = {mgr: [0] * min(len(managers), 10) for mgr in managers}
    simulate_fn = (simulate_once_performance if mode == "performance"
                   else simulate_once_opponent_difficulty)
    for _ in range(n_sims):
        wins, seeds = simulate_fn(points_dict, managers, n_weeks, rng)
        for mgr in managers:
            wc = max(0, min(wins.get(mgr, 0), n_weeks))
            sd = seeds.get(mgr, 1)
            win_hists[mgr][wc] += 1
            if 1 <= sd <= len(seed_hists[mgr]):
                seed_hists[mgr][sd - 1] += 1
    return win_hists, seed_hists
@ensure_normalized
def calculate_summary_stats(
    df: pd.DataFrame,
    regular_mask: pd.Series,
    col_prefix: str
) -> pd.DataFrame:
    """
    Calculate summary statistics from probability distributions.
    Args:
        df: DataFrame with probability columns
        regular_mask: Boolean mask for regular season rows
        col_prefix: "shuffle" or "opp_shuffle"
    Returns:
        DataFrame with summary columns added
    """
    df = df.copy()
    # Expected wins
    win_prob_cols = [f"{col_prefix}_{w}_win" for w in range(0, 15)]
    win_prob_cols = [c for c in win_prob_cols if c in df.columns]
    if win_prob_cols:
        df_reg = df.loc[regular_mask, win_prob_cols]
        if not df_reg.empty:
            weights = np.arange(0, len(win_prob_cols), dtype=float)
            vals = np.round((df_reg.fillna(0.0).to_numpy(dtype=float) @ weights) / 100.0, 2)
            has_data = df_reg.notna().any(axis=1)
            df.loc[regular_mask, f"{col_prefix}_avg_wins"] = vals
            df.loc[regular_mask & ~has_data, f"{col_prefix}_avg_wins"] = np.nan
    # Expected seed
    seed_prob_cols = [f"{col_prefix}_{s}_seed" for s in range(1, 11)]
    seed_prob_cols = [c for c in seed_prob_cols if c in df.columns]
    if seed_prob_cols:
        df_reg = df.loc[regular_mask, seed_prob_cols]
        if not df_reg.empty:
            weights = np.arange(1, len(seed_prob_cols) + 1, dtype=float)
            vals = np.round((df_reg.fillna(0.0).to_numpy(dtype=float) @ weights) / 100.0, 2)
            has_data = df_reg.notna().any(axis=1)
            df.loc[regular_mask, f"{col_prefix}_avg_seed"] = vals
            df.loc[regular_mask & ~has_data, f"{col_prefix}_avg_seed"] = np.nan
    # Playoff odds (seeds 1-6)
    playoff_cols = [f"{col_prefix}_{s}_seed" for s in range(1, 7)]
    playoff_cols = [c for c in playoff_cols if c in df.columns]
    if playoff_cols:
        df_reg = df.loc[regular_mask, playoff_cols]
        if not df_reg.empty:
            vals = df_reg.fillna(0.0).sum(axis=1)
            has_data = df_reg.notna().any(axis=1)
            df.loc[regular_mask, f"{col_prefix}_avg_playoffs"] = vals
            df.loc[regular_mask & ~has_data, f"{col_prefix}_avg_playoffs"] = np.nan
    # Bye odds (seeds 1-2)
    bye_cols = [f"{col_prefix}_1_seed", f"{col_prefix}_2_seed"]
    bye_cols = [c for c in bye_cols if c in df.columns]
    if bye_cols:
        df_reg = df.loc[regular_mask, bye_cols]
        if not df_reg.empty:
            vals = df_reg.fillna(0.0).sum(axis=1)
            has_data = df_reg.notna().any(axis=1)
            df.loc[regular_mask, f"{col_prefix}_avg_bye"] = vals
            df.loc[regular_mask & ~has_data, f"{col_prefix}_avg_bye"] = np.nan
    return df
@ensure_normalized
def calculate_opponent_rank_percentile(df: pd.DataFrame, regular_mask: pd.Series) -> pd.DataFrame:
    """
    Calculate simple per-week opponent difficulty rank and percentile.
    Args:
        df: DataFrame with opponent_points
        regular_mask: Boolean mask for regular season rows
    Returns:
        DataFrame with opp_pts_week_rank and opp_pts_week_pct added
    """
    df = df.copy()
    if "opponent_points" in df.columns:
        idx = regular_mask & df["opponent_points"].notna()
        grouped = df.loc[idx].groupby(["year", "week"])["opponent_points"]
        # Rank: 1 = hardest opponent (highest opponent_points)
        df.loc[idx, "opp_pts_week_rank"] = grouped.rank(method="min", ascending=False).astype("Int64")
        # Percentile: 100 = hardest
        df.loc[idx, "opp_pts_week_pct"] = (100.0 * grouped.rank(pct=True, ascending=True)).round(2)
    return df
@ensure_normalized
def lock_postseason_to_final_week(
    df: pd.DataFrame,
    regular_mask: pd.Series,
    col_list: List[str]
) -> pd.DataFrame:
    """
    Copy final regular season values to playoff/consolation rows.
    Args:
        df: Full DataFrame
        regular_mask: Mask for regular season rows
        col_list: List of columns to copy
    Returns:
        DataFrame with postseason rows locked
    """
    df = df.copy()
    post_mask = (df['is_playoffs'] == 1) | (df['is_consolation'] == 1)
    if not post_mask.any():
        return df
    cols_to_copy = [c for c in col_list if c in df.columns]
    for year in sorted(df['year'].dropna().unique()):
        df_reg_year = df[(df['year'] == year) & regular_mask]
        if df_reg_year.empty:
            continue
        last_wk = current_regular_week(df_reg_year)
        if last_wk == 0:
            continue
        final_rows = df_reg_year[df_reg_year['week'] == last_wk]
        if final_rows.empty:
            continue
        idx_post_year = df.index[(df['year'] == year) & post_mask]
        for idx in idx_post_year:
            mgr = df.at[idx, 'manager']
            src = final_rows[final_rows['manager'] == mgr]
            if src.empty:
                continue
            src_row = src.iloc[0]
            for c in cols_to_copy:
                df.at[idx, c] = src_row[c]
    return df