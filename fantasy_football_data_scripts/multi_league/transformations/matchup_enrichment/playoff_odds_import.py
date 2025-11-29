import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time



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

from core.league_context import LeagueContext
from transformations.base.modules.playoff_helpers import (
    add_match_key,
    canonicalize,
    wins_points_to_date,
    rank_and_seed,
    history_snapshots,
    normalize_seed_matrix_to_100,
    p_playoffs_from_seeds,
    p_bye_from_seeds,
)
from transformations.base.modules.playoff_bracket import load_league_settings

# =========================================================
# Playoff Odds Engine – Multi-League Generic Version
# =========================================================

# -------------------------
# Default config (can be overridden by league settings)
# -------------------------
DEFAULT_PLAYOFF_SLOTS = 6
DEFAULT_BYE_SLOTS = 2
HALF_LIFE_WEEKS = 10
SHRINK_K = 3.0  # Reduced from 6.0 - less aggressive shrinkage toward league average
SIGMA_FLOOR_MIN = 18  # Increased from 10 - but will be overridden by league_sd * 0.75
N_SIMS = 10000
RNG_SEED = 42
DEFAULT_BRACKET_RESEED = False

# Global variables (set from league settings or defaults)
PLAYOFF_SLOTS = DEFAULT_PLAYOFF_SLOTS
BYE_SLOTS = DEFAULT_BYE_SLOTS
BRACKET_RESEED = DEFAULT_BRACKET_RESEED

# -------------------------
# Target columns to fill
# -------------------------
TARGET_COLS = [
    "avg_seed", "p_playoffs", "p_bye", "exp_final_wins", "exp_final_pf",
    "p_semis", "p_final", "p_champ",
    "x1_seed", "x2_seed", "x3_seed", "x4_seed", "x5_seed",
    "x6_seed", "x7_seed", "x8_seed", "x9_seed", "x10_seed",
    "power_rating",
    "x0_win", "x1_win", "x2_win", "x3_win", "x4_win", "x5_win", "x6_win", "x7_win",
    "x8_win", "x9_win", "x10_win", "x11_win", "x12_win", "x13_win", "x14_win",
]

# -------------------------
# Round column names
# -------------------------
ROUND_COLS = {
    "qf": "quarterfinal",
    "sf": "semifinal",
    "fn": "championship",
}


# -------------------------
# RNG helper
# -------------------------
def get_rng(season: int, week: int, base: int = RNG_SEED) -> np.random.Generator:
    mix = (int(base)
           ^ ((int(season) * 0x9E3779B1) & 0xFFFFFFFF)
           ^ ((int(week) * 0x85EBCA77) & 0xFFFFFFFF))
    return np.random.default_rng(mix & 0xFFFFFFFF)


# -------------------------
# Enforce playoff monotonicity
# -------------------------
def _sim_game(a, b, rng, mu_hat, sigma_hat, samples_by_team):
    sa = draw_score(a, rng, mu_hat, sigma_hat, samples_by_team)
    sb = draw_score(b, rng, mu_hat, sigma_hat, samples_by_team)
    if sa > sb: return a
    if sb > sa: return b
    return rng.choice([a, b])


def enforce_playoff_monotonicity(df):
    df = df.copy()
    if "is_playoffs" not in df.columns:
        return df
    for yr in sorted(df["year"].dropna().unique().astype(int)):
        season_mask = df["year"] == yr
        po_weeks = df.loc[season_mask & (df["is_playoffs"] == 1), "week"].dropna()
        if po_weeks.empty:
            continue
        # Ensure week is numeric before calling min()
        po_weeks = pd.to_numeric(po_weeks, errors='coerce').dropna()
        if po_weeks.empty:
            continue
        start_po = int(po_weeks.min())
        df.loc[season_mask & (df["week"] >= start_po), "is_playoffs"] = 1
    return df


# -------------------------
# Recency weights
# -------------------------
def recency_weights(df, season, week, half_life, boundary_penalty=0.05):
    hl = max(1, int(half_life))
    lam = np.log(2.0) / hl

    timeline = (df["year"] - season) * 100 + (df["week"] - week)
    weeks_ago = np.maximum(0, (-timeline).astype(float))
    base = np.exp(-lam * weeks_ago)

    prior = (df["year"] < season).astype(float)
    if week is None:
        fade = 1.0
    else:
        fade = min(max((week - 1) / 4.0, 0.0), 1.0)

    penalty = (1 - fade) * boundary_penalty + fade * 1.0
    return base * np.where(prior == 1.0, penalty, 1.0)


# -------------------------
# Team models
# -------------------------
def build_team_models(hist, season, week, half_life, shrink_k, sigma_floor,
                      boundary_penalty=0.05, prior_w_cap=2.0, season_stats=None):
    """
    Build team performance models with optional pre-calculated season statistics.

    Args:
        season_stats: Optional dict with keys 'mean', 'sd', 'n_games' for the current season.
                     If provided, uses full-season variance estimates (improves early-week accuracy).
    """
    h = hist.copy()
    h["w"] = recency_weights(h, season, week, half_life, boundary_penalty)

    prior_mask = h["year"] < season
    if prior_mask.any():
        w_prior = (h.loc[prior_mask]
                   .groupby("manager")["w"].transform(lambda s: s / max(1e-12, s.sum())))
        h.loc[prior_mask, "w"] = w_prior * prior_w_cap

    # Use pre-calculated season statistics if available (improves early-week estimates)
    # Otherwise fall back to calculating from available historical data
    if season_stats and 'mean' in season_stats and 'sd' in season_stats:
        league_mu = season_stats['mean']
        league_sd = season_stats['sd']
    else:
        league_mu = h["team_points"].mean()
        league_sd = h["team_points"].std(ddof=1)

    by_mgr = (
        h.groupby("manager")[["team_points", "w"]]
        .apply(lambda g: pd.Series({
            "n": g.shape[0],
            "w_sum": g["w"].sum(),
            "mu_raw": (g["team_points"] * g["w"]).sum() / (g["w"].sum() + 1e-12),
            "sd_raw": g["team_points"].std(ddof=1) if len(g) >= 2 else np.nan,
        }), include_groups=False)
    )

    k = float(shrink_k)
    weeks_played = max(0, int(week))
    # Gradual transition from league average to observed data
    # Week 1: k_eff = 3 * 2.2 = 6.6 → 13% weight on observed data
    # Week 2: k_eff = 3 * 1.8 = 5.4 → 27% weight on observed data
    # Week 3: k_eff = 3 * 1.4 = 4.2 → 42% weight on observed data
    # Week 4+: k_eff = 3 → 57%+ weight on observed data
    k_eff = k * max(1.0, (4 - min(weeks_played, 4)) * 0.4)

    w_eb = by_mgr["w_sum"] / (by_mgr["w_sum"] + k_eff)
    mu_hat = (w_eb * by_mgr["mu_raw"] + (1 - w_eb) * league_mu).to_dict()

    # Team-specific variance with shrinkage (respects boom/bust vs consistent teams)
    # Instead of hard floor, use empirical Bayes shrinkage toward league SD
    # This preserves differences: volatile teams stay volatile, consistent teams stay consistent
    sd_fill = by_mgr["sd_raw"].fillna(league_sd * 0.9)

    # Shrinkage parameter for variance (half of mean shrinkage for more stability)
    k_sigma = k_eff / 2.0

    # Calculate weight for each team's observed variance
    # More observations = more weight on team's actual volatility
    w_sigma = by_mgr["w_sum"] / (by_mgr["w_sum"] + k_sigma)

    # Blend team-specific SD with league SD (weighted by evidence)
    # Volatile teams will shrink toward league avg, but still remain more volatile
    # Consistent teams will shrink toward league avg, but still remain less volatile
    sigma_raw = (w_sigma * sd_fill + (1 - w_sigma) * league_sd)

    # Apply softer minimum floor (60% of league SD instead of 75%)
    # This allows consistent teams to stay below league average variance
    sigma_floor_soft = max(float(sigma_floor), league_sd * 0.6)
    sigma_hat = sigma_raw.clip(lower=sigma_floor_soft).to_dict()

    samples_by_team = {}
    for m, g in h.groupby("manager"):
        g_cur = g[g["year"] == season]
        if week <= 3 and len(g_cur) >= 2:
            samples_by_team[m] = g_cur[["team_points", "w"]].copy()
        else:
            samples_by_team[m] = g[["team_points", "w"]].copy()

    for m in h["manager"].unique():
        mu_hat.setdefault(m, league_mu)
        sigma_hat.setdefault(m, sigma_floor)

    return mu_hat, sigma_hat, samples_by_team, league_mu, sigma_floor


def ensure_params_for_future(mu_hat, sigma_hat, samples_by_team, df_future, league_mu, sigma_floor):
    if df_future is None or df_future.empty:
        return
    future_mgrs = set(df_future["manager"]).union(set(df_future["opponent"]))
    for m in future_mgrs:
        mu_hat.setdefault(m, league_mu)
        sigma_hat.setdefault(m, sigma_floor)
        samples_by_team.setdefault(m, pd.DataFrame({"team_points": [], "w": []}))


# -------------------------
# Power rating calculator
# -------------------------
def compute_power_ratings(mu_hat, samples_by_team, bootstrap_min=4):
    out = {}
    for m, samp in samples_by_team.items():
        use_boot = (
                isinstance(samp, pd.DataFrame)
                and "team_points" in samp.columns
                and "w" in samp.columns
                and len(samp) >= bootstrap_min
                and float(samp["w"].sum()) > 0
        )
        if use_boot:
            out[m] = float(np.average(samp["team_points"].to_numpy(),
                                      weights=samp["w"].to_numpy()))
        else:
            out[m] = float(mu_hat.get(m, np.mean(list(mu_hat.values()))))
    for m in mu_hat.keys():
        out.setdefault(m, float(mu_hat[m]))
    return pd.Series(out, name="Power_Rating")


# -------------------------
# Score draw
# -------------------------
def draw_score(manager, rng, mu_hat, sigma_hat, samples_by_team, use_bootstrap=True, bootstrap_min=4):
    samp = samples_by_team.get(manager)
    if use_bootstrap and isinstance(samp, pd.DataFrame) and len(samp) >= bootstrap_min and float(samp["w"].sum()) > 0:
        p = (samp["w"] / samp["w"].sum()).to_numpy()
        return float(rng.choice(samp["team_points"].to_numpy(), p=p))
    mu = mu_hat.get(manager, np.mean(list(mu_hat.values())))
    sd = sigma_hat.get(manager, np.mean(list(sigma_hat.values())))
    return float(max(0.0, rng.normal(mu, sd)))


# -------------------------
# Schedule helpers
# -------------------------
def schedules_last_regular_week(df_sched, season):
    s = df_sched[(df_sched["year"] == season)]
    if s.empty:
        return None
    po = s.loc[s["is_playoffs"] == 1, "week"].dropna()
    if not po.empty:
        # Ensure week is numeric before calling min()
        po = pd.to_numeric(po, errors='coerce').dropna()
        if po.empty:
            return None
        return int(po.min()) - 1

    weeks = pd.to_numeric(s["week"], errors='coerce').dropna()
    if weeks.empty:
        return None
    return int(weeks.max())


def future_regular_from_schedule(df_sched, season, current_week):
    # Get last regular season week to exclude consolation bracket games
    last_reg_week = schedules_last_regular_week(df_sched, season)

    s = df_sched[(df_sched["year"] == season) & (df_sched["is_playoffs"] == 0)]
    if s.empty:
        return pd.DataFrame(columns=["year", "week", "manager", "opponent"])

    # Only include games after current_week AND up to last_reg_week
    # This excludes consolation games that occur after playoffs start
    s = s[s["week"] > current_week].copy()
    if last_reg_week is not None:
        s = s[s["week"] <= last_reg_week].copy()

    if s.empty:
        return s[["year", "week", "manager", "opponent"]]
    s = canonicalize(s.rename(columns={"Opponent Week": "opponent_week", "OpponentYear": "opponent_year"}))
    return s[["year", "week", "manager", "opponent"]].drop_duplicates()


# -------------------------
# Vectorized regular and bracket sim
# -------------------------
def _vectorized_regular_and_bracket(reg_to_date, future_canon, mu_hat, sigma_hat, season, week, playoff_slots=None, bye_slots=None, bracket_reseed=None):
    import time
    t0 = time.time()
    rng = get_rng(season, week)

    # Use provided playoff configuration or fallback to globals
    _playoff_slots = playoff_slots if playoff_slots is not None else PLAYOFF_SLOTS
    _bye_slots = bye_slots if bye_slots is not None else BYE_SLOTS
    _bracket_reseed = bracket_reseed if bracket_reseed is not None else BRACKET_RESEED

    # Build manager list from BOTH reg_to_date AND future_canon to ensure all managers are indexed
    # This prevents IndexError when future schedule contains managers not yet in reg_to_date
    manager_set = set(reg_to_date["manager"].unique()) | set(reg_to_date["opponent"].unique())
    if future_canon is not None and not future_canon.empty:
        manager_set |= set(future_canon["manager"].dropna().unique())
        manager_set |= set(future_canon["opponent"].dropna().unique())
    # Remove any NaN values that may have slipped through
    manager_set = {m for m in manager_set if pd.notna(m) and m != ''}
    managers = sorted(manager_set)
    mgr2idx = {m: i for i, m in enumerate(managers)}
    idx2mgr = np.array(managers)
    n = len(managers)

    wins_to_date, pf_to_date = wins_points_to_date(reg_to_date)
    WTD = np.zeros(n, dtype=float)
    PFTD = np.zeros(n, dtype=float)
    for m, v in wins_to_date.items(): WTD[mgr2idx[m]] = v
    for m, v in pf_to_date.items():  PFTD[mgr2idx[m]] = v

    mu_arr = np.array([mu_hat.get(m, np.mean(list(mu_hat.values()))) for m in managers])
    sd_arr = np.array([sigma_hat.get(m, np.mean(list(sigma_hat.values()))) for m in managers])

    if future_canon is None or future_canon.empty:
        PF_all = np.tile(PFTD, (N_SIMS, 1))
        W_all = np.tile(WTD, (N_SIMS, 1))
    else:
        A = future_canon["manager"].map(mgr2idx).to_numpy()
        B = future_canon["opponent"].map(mgr2idx).to_numpy()

        # Safety check: Filter out any rows where manager or opponent couldn't be mapped
        # This handles edge cases where schedule has managers not in the matchup data
        valid_mask = ~(np.isnan(A.astype(float)) | np.isnan(B.astype(float)))
        if not valid_mask.all():
            unmapped_count = (~valid_mask).sum()
            print(f"[WARNING] Filtering out {unmapped_count} future games with unmapped managers")
            A = A[valid_mask].astype(int)
            B = B[valid_mask].astype(int)
        else:
            A = A.astype(int)
            B = B.astype(int)

        G = A.size

        if G == 0:
            # No valid future games - treat as if no future schedule
            PF_all = np.tile(PFTD, (N_SIMS, 1))
            W_all = np.tile(WTD, (N_SIMS, 1))
        else:
            SA = rng.normal(mu_arr[A][None, :], sd_arr[A][None, :], size=(N_SIMS, G))
            SB = rng.normal(mu_arr[B][None, :], sd_arr[B][None, :], size=(N_SIMS, G))

            PF_add = np.zeros((N_SIMS, n), dtype=float)
            rows = np.arange(N_SIMS)[:, None]
            np.add.at(PF_add, (rows, A[None, :]), SA)
            np.add.at(PF_add, (rows, B[None, :]), SB)
            PF_all = PF_add + PFTD

            a_wins = (SA > SB).astype(float)
            ties = (SA == SB)
            if ties.any():
                coin = rng.integers(0, 2, size=ties.shape)
                a_wins = np.where(ties, coin, a_wins)
            b_wins = 1.0 - a_wins

            W_add = np.zeros((N_SIMS, n), dtype=float)
            np.add.at(W_add, (rows, A[None, :]), a_wins)
            np.add.at(W_add, (rows, B[None, :]), b_wins)
            W_all = W_add + WTD

    order_w = np.argsort(-W_all, axis=1, kind="stable")
    W_sorted = np.take_along_axis(W_all, order_w, axis=1)
    PF_reord = np.take_along_axis(PF_all, order_w, axis=1)

    seeds_idx = np.empty_like(order_w)
    for s in range(N_SIMS):
        idx = np.lexsort(np.vstack([-PF_reord[s], -W_sorted[s]]))
        seeds_idx[s] = order_w[s, idx]

    inv_seed = np.full((N_SIMS, n), n, dtype=int)
    row_idx = np.arange(N_SIMS)[:, None]
    inv_seed[row_idx, seeds_idx] = np.arange(n)[None, :]
    seed_numbers = inv_seed + 1

    # DYNAMIC BRACKET STRUCTURE - Uses _playoff_slots and _bye_slots from parameters
    # Extract bye teams (top _bye_slots seeds)
    bye_teams = [seeds_idx[:, i] for i in range(_bye_slots)]

    # Extract first round teams (seeds _bye_slots+1 through _playoff_slots)
    first_round_teams = [seeds_idx[:, i] for i in range(_bye_slots, _playoff_slots)]

    # Determine first round matchups (high seed vs low seed)
    # For 6 teams with 2 byes: 3v6, 4v5
    # For 8 teams with 4 byes: 5v8, 6v7
    num_first_round_games = len(first_round_teams) // 2
    first_round_winners = []

    for game_idx in range(num_first_round_games):
        # Pair highest remaining seed with lowest remaining seed
        higher_seed = first_round_teams[game_idx]
        lower_seed = first_round_teams[-(game_idx+1)]

        # Simulate game
        ScoreA = rng.normal(mu_arr[higher_seed], sd_arr[higher_seed])
        ScoreB = rng.normal(mu_arr[lower_seed], sd_arr[lower_seed])
        ties = (ScoreA == ScoreB)
        coin = rng.integers(0, 2, size=ties.shape[0])
        winner = np.where(ScoreA > ScoreB, higher_seed,
                         np.where(ScoreB > ScoreA, lower_seed,
                                 np.where(coin == 0, higher_seed, lower_seed)))
        first_round_winners.append(winner)

    # Semifinals: bye teams face first round winners
    # Standard bracket (no reseeding): 1 vs winner of lower game, 2 vs winner of higher game
    # With reseeding: 1 vs lowest seed, 2 vs next lowest seed
    semi_teams = bye_teams + first_round_winners

    # CRITICAL FIX: Handle case where _bye_slots = 0 (e.g., 4-team playoff with no byes)
    # In this case, first_round_winners ARE the semifinalists
    semi_matchups = []

    if _bye_slots == 0:
        # Special case: No bye teams, so first_round_winners become semifinalists
        # Pair them for semifinals
        num_semi_games = len(first_round_winners) // 2

        if not _bracket_reseed:
            # Fixed bracket: pair high seed winner vs low seed winner
            # For 4-team: winner of 1v4 plays winner of 2v3
            for i in range(num_semi_games):
                team_a = first_round_winners[i]
                team_b = first_round_winners[-(i+1)]
                semi_matchups.append((team_a, team_b))
        else:
            # Reseeding: pair best remaining seed with worst remaining seed
            sorted_winners = []
            for sim_idx in range(N_SIMS):
                winners_seeds = [(first_round_winners[j][sim_idx], inv_seed[sim_idx, first_round_winners[j][sim_idx]])
                                for j in range(len(first_round_winners))]
                winners_seeds.sort(key=lambda x: x[1])
                sorted_winners.append([w[0] for w in winners_seeds])

            for i in range(num_semi_games):
                team_a = np.array([sorted_winners[s][i] for s in range(N_SIMS)])
                team_b = np.array([sorted_winners[s][-(i+1)] for s in range(N_SIMS)])
                semi_matchups.append((team_a, team_b))

    elif not _bracket_reseed:
        # Fixed bracket: each bye team faces predetermined opponent
        # For 6-team, 2-bye: seed1 faces winner of 4v5, seed2 faces winner of 3v6
        # Generically: bye team i faces first_round_winners[num_games - 1 - i]
        for i in range(_bye_slots):
            bye_team = bye_teams[i]
            opponent = first_round_winners[num_first_round_games - 1 - i] if i < num_first_round_games else first_round_winners[0]
            semi_matchups.append((bye_team, opponent))
    else:
        # Reseeding: pair best remaining seed with worst remaining seed
        # Sort all remaining teams by seed, then pair 1vworst, 2vsecond-worst
        all_remaining = bye_teams + first_round_winners
        # Sort by seed (lower seed number = better seed)
        sorted_teams = []
        for sim_idx in range(N_SIMS):
            teams_seeds = [(team[sim_idx], inv_seed[sim_idx, team[sim_idx]]) for team in all_remaining]
            teams_seeds.sort(key=lambda x: x[1])
            sorted_teams.append([t[0] for t in teams_seeds])

        # Pair best with worst
        for i in range(_bye_slots):
            team_a = np.array([sorted_teams[s][i] for s in range(N_SIMS)])
            team_b = np.array([sorted_teams[s][-(i+1)] for s in range(N_SIMS)])
            semi_matchups.append((team_a, team_b))

    # Simulate semifinals
    semi_winners = []
    for team_a, team_b in semi_matchups:
        ScoreA = rng.normal(mu_arr[team_a], sd_arr[team_a])
        ScoreB = rng.normal(mu_arr[team_b], sd_arr[team_b])
        ties = (ScoreA == ScoreB)
        coin = rng.integers(0, 2, size=ties.shape[0])
        winner = np.where(ScoreA > ScoreB, team_a,
                         np.where(ScoreB > ScoreA, team_b,
                                 np.where(coin == 0, team_a, team_b)))
        semi_winners.append(winner)

    # Championship game
    # Handle case where there's only 1 semifinal game (e.g., 4-team/0-bye playoff)
    if len(semi_winners) == 1:
        # Only 1 semifinal winner = champion (no separate championship game needed)
        # The semifinal winner IS the finalist AND champion
        s1_w = semi_winners[0]
        s2_w = semi_winners[0]  # Same as s1_w since there's only one finalist
        champ = semi_winners[0]
    elif len(semi_winners) >= 2:
        # Standard case: 2+ semifinal winners play for championship
        s1_w = semi_winners[0]
        s2_w = semi_winners[1]

        FA = rng.normal(mu_arr[s1_w], sd_arr[s1_w])
        FB = rng.normal(mu_arr[s2_w], sd_arr[s2_w])
        ties = (FA == FB)
        coin = rng.integers(0, 2, size=ties.shape[0])
        champ = np.where(FA > FB, s1_w,
                         np.where(FB > FA, s2_w,
                                  np.where(coin == 0, s1_w, s2_w)))
    else:
        # No semifinal winners - should not happen, but handle gracefully
        raise ValueError(f"No semifinal winners found! semi_winners={semi_winners}")

    def pct_counts(idx_arr, nteams):
        arr = np.asarray(idx_arr).reshape(-1).astype(np.int64, copy=False)
        if arr.size == 0:
            return np.zeros(nteams, dtype=float)
        c = np.bincount(arr, minlength=nteams).astype(float)
        return 100.0 * c / arr.size

    # Calculate probabilities - all teams in semifinals
    all_semi_teams = []
    for team_list in semi_teams:
        all_semi_teams.append(team_list)
    p_semis_sim = pct_counts(np.concatenate(all_semi_teams), n)
    p_final_sim = pct_counts(np.concatenate([s1_w, s2_w]), n)
    p_champ_sim = pct_counts(champ, n)

    exp_final_wins = W_all.mean(axis=0)
    exp_final_pf = PF_all.mean(axis=0)
    avg_seed = seed_numbers.mean(axis=0)

    W_int = np.rint(W_all).astype(int)
    W_int = np.clip(W_int, 0, 14)
    win_prob_mat = np.zeros((n, 15), dtype=float)
    for k in range(0, 15):
        mask = (W_int == k).astype(np.int32)
        counts_k = mask.sum(axis=0).astype(float)
        win_prob_mat[:, k] = 100.0 * counts_k / float(N_SIMS)

    seed_mat = np.zeros((n, n), dtype=float)
    for pos in range(n):
        teams_at_pos = seeds_idx[:, pos]
        seed_mat[:, pos] = np.bincount(teams_at_pos, minlength=n) / N_SIMS * 100.0
    seed_df = pd.DataFrame(seed_mat, index=idx2mgr, columns=list(range(1, n + 1)))

    s_index = pd.Index(idx2mgr, name="manager")
    series_pack = {
        "Avg_Seed": pd.Series(avg_seed, index=s_index),
        "Exp_Final_Wins": pd.Series(exp_final_wins, index=s_index),
        "Exp_Final_PF": pd.Series(exp_final_pf, index=s_index),
        "P_Semis_SIM": pd.Series(p_semis_sim, index=s_index),
        "P_Final": pd.Series(p_final_sim, index=s_index),
        "P_Champ": pd.Series(p_champ_sim, index=s_index),
    }

    win_cols = [f"x{k}_win" for k in range(15)]
    win_df = pd.DataFrame(win_prob_mat, index=idx2mgr, columns=win_cols)

    # Build bye and first round winner matrices dynamically
    bye_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    for bye_team in bye_teams:
        bye_mat[np.arange(N_SIMS), bye_team] = 1
    no_bye_mat = 1 - bye_mat

    qfwin_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    for winner in first_round_winners:
        qfwin_mat[np.arange(N_SIMS), winner] = 1

    sfwin_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    sfwin_mat[np.arange(N_SIMS), s1_w] = 1
    sfwin_mat[np.arange(N_SIMS), s2_w] = 1

    champ_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    champ_mat[np.arange(N_SIMS), champ] = 1

    no_bye_counts = no_bye_mat.sum(axis=0).astype(float)
    qfwin_and_no_bye = (qfwin_mat & no_bye_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        qf_win_given_no_bye = np.where(no_bye_counts > 0,
                                       100.0 * qfwin_and_no_bye / no_bye_counts,
                                       0.0)
    series_pack["P_QFWin_Given_NoBye_SIM"] = pd.Series(qf_win_given_no_bye, index=s_index)

    bye_counts = bye_mat.sum(axis=0).astype(float)
    sfwin_and_bye = (sfwin_mat & bye_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sf_win_given_bye = np.where(bye_counts > 0,
                                    100.0 * sfwin_and_bye / bye_counts,
                                    0.0)
    series_pack["P_SFWin_Given_Bye_SIM"] = pd.Series(sf_win_given_bye, index=s_index)

    qfwin_counts = qfwin_mat.sum(axis=0).astype(float)
    sfwin_and_qf = (sfwin_mat & qfwin_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sf_win_given_nonbye = np.where(qfwin_counts > 0,
                                       100.0 * sfwin_and_qf / qfwin_counts,
                                       0.0)
    series_pack["P_SFWin_Given_NonBye_SIM"] = pd.Series(sf_win_given_nonbye, index=s_index)

    final_counts = sfwin_mat.sum(axis=0).astype(float)
    champ_and_final = (champ_mat & sfwin_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        win_final_given_final = np.where(final_counts > 0,
                                         100.0 * champ_and_final / final_counts,
                                         0.0)
    series_pack["P_WinFinal_Given_Final_SIM"] = pd.Series(win_final_given_final, index=s_index)

    return series_pack, seed_df, win_df


# -------------------------
# History helpers
# -------------------------
def _gaussian_kernel(d2): return np.exp(-0.5 * d2)


def empirical_kernel_seed_dist(played_raw, week, history_snapshots_df, n_teams,
                               prior_strength=3.0):
    """
    Empirical kernel-based seed distribution with data-driven bandwidth selection.

    Uses Silverman's rule of thumb for bandwidth calculation:
    h = 1.06 * sigma * n^(-1/5)

    This makes the function generic and scalable to any league structure.
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

    # Calculate data-driven bandwidths using Silverman's rule of thumb
    # h = 1.06 * sigma * n^(-1/5)
    n_hist = len(H)

    if n_hist > 1:
        # Standard deviation of historical data for each dimension
        W_std = H["W"].std() if H["W"].std() > 0 else 1.0
        L_std = H["L"].std() if H["L"].std() > 0 else 1.0
        PF_std = H["PF_pct"].std() if H["PF_pct"].std() > 0 else 15.0
        week_std = H["week"].std() if H["week"].std() > 0 else 2.0

        # Silverman's rule: h = 1.06 * sigma * n^(-1/5)
        scaling_factor = 1.06 * (n_hist ** (-0.2))
        h_W = scaling_factor * W_std
        h_L = scaling_factor * L_std
        h_PF = scaling_factor * PF_std
        h_week = scaling_factor * week_std
    else:
        # Fallback for very small historical datasets
        h_W = 0.9
        h_L = 0.9
        h_PF = 15.0
        h_week = 0.9

    seeds = np.arange(1, n_teams + 1)
    cols = list(seeds)
    out = pd.DataFrame(0.0, index=mgrs, columns=cols)

    alpha0 = np.ones(n_teams) * (prior_strength / n_teams)
    H_seed = H["final_seed"].to_numpy()

    for m in mgrs:
        xW, xL, xPF, xwk = float(W.loc[m]), float(L.loc[m]), float(pf_pct.loc[m]), float(week)
        dW = (H["W"].to_numpy() - xW) / max(1e-6, h_W)
        dL = (H["L"].to_numpy() - xL) / max(1e-6, h_L)
        dPF = (H["PF_pct"].to_numpy() - xPF) / max(1e-6, h_PF)
        dWK = (H["week"].to_numpy() - xwk) / max(1e-6, h_week)
        d2 = dW * dW + dL * dL + dPF * dPF + dWK * dWK
        w = _gaussian_kernel(d2)

        counts = np.zeros(n_teams)
        for k in range(1, n_teams + 1):
            counts[k - 1] = float(w[(H_seed == k)].sum())
        probs = (counts + alpha0) / (counts.sum() + alpha0.sum() + 1e-12)
        out.loc[m, cols] = probs
    return out


# -------------------------
# Power rating normalization
# -------------------------
def normalize_power_rating(power_series, inflation_rate):
    """Normalize power rating by inflation rate."""
    if pd.isna(inflation_rate) or inflation_rate == 0:
        return power_series
    return power_series / float(inflation_rate)


# -------------------------
# Core calculators
# -------------------------
def calc_regular_week_outputs(df_season, df_sched, season, week, history_df, inflation_rate=None, season_stats=None, data_directory=None):
    df_to_date = df_season[df_season["week"] <= week].copy()
    reg_to_date = df_to_date[df_to_date["is_playoffs"] == 0].copy()
    played_raw = reg_to_date.copy()

    # Load year-specific playoff configuration
    from transformations.base.modules.playoff_bracket import load_league_settings
    settings = load_league_settings(season, data_directory=str(data_directory) if data_directory else None, df=df_to_date)

    num_playoff_teams = settings.get('num_playoff_teams', PLAYOFF_SLOTS)
    bye_teams = settings.get('bye_teams', BYE_SLOTS)
    uses_reseeding = settings.get('uses_playoff_reseeding', False)

    future_canon = future_regular_from_schedule(df_sched, season, week)
    simulate_future_reg = not future_canon.empty

    # Use higher variance floor for very early weeks (weeks 1-2) to account for higher uncertainty
    # This will be further adjusted by build_team_models based on actual league_sd
    sigma_floor_dynamic = SIGMA_FLOOR_MIN if week >= 3 else max(SIGMA_FLOOR_MIN, 20)
    mu_hat, sigma_hat, samples_by_team, league_mu, sigma_floor = build_team_models(
        df_to_date, season, week, HALF_LIFE_WEEKS, SHRINK_K, sigma_floor_dynamic,
        boundary_penalty=0.05, prior_w_cap=2.0, season_stats=season_stats
    )
    ensure_params_for_future(mu_hat, sigma_hat, samples_by_team, future_canon, league_mu, sigma_floor)

    power_s = compute_power_ratings(mu_hat, samples_by_team, bootstrap_min=4)

    # Normalize power ratings by inflation rate if provided
    if inflation_rate is not None and pd.notna(inflation_rate) and inflation_rate != 0:
        power_s = normalize_power_rating(power_s, inflation_rate)

    series_pack, seed_dist_sim, win_df = _vectorized_regular_and_bracket(
        reg_to_date, future_canon, mu_hat, sigma_hat, season, week,
        playoff_slots=num_playoff_teams, bye_slots=bye_teams, bracket_reseed=uses_reseeding
    )

    n_teams_all = seed_dist_sim.shape[1]
    if simulate_future_reg:
        # Check if there is historical data for years before the current season
        # If this is the first year in dataset, skip blending with historical data
        has_prior_history = not history_df.empty and len(history_df[history_df["year"] < season]) > 0

        # DISABLED historical blending - use 100% Monte Carlo simulations
        blended_seed_norm = normalize_seed_matrix_to_100(seed_dist_sim)
    else:
        blended_seed_norm = normalize_seed_matrix_to_100(seed_dist_sim)

    odds = pd.concat([
        series_pack["Exp_Final_Wins"].rename("Exp_Final_Wins"),
        series_pack["Exp_Final_PF"].rename("Exp_Final_PF"),
        series_pack["Avg_Seed"].rename("Avg_Seed"),
    ], axis=1)

    odds["P_Playoffs"] = p_playoffs_from_seeds(blended_seed_norm, num_playoff_teams)
    odds["P_Bye"] = p_bye_from_seeds(blended_seed_norm, bye_teams)

    qf_cnd = series_pack["P_QFWin_Given_NoBye_SIM"]
    odds["P_Semis"] = odds["P_Bye"] + (100.0 - odds["P_Bye"]) * (qf_cnd / 100.0)
    odds["P_Semis"] = np.minimum(100.0, np.maximum(odds["P_Semis"], odds["P_Bye"]))

    sf_bye = series_pack["P_SFWin_Given_Bye_SIM"]
    sf_nonbye = series_pack["P_SFWin_Given_NonBye_SIM"]
    semis_nonbye = np.maximum(0.0, odds["P_Semis"] - odds["P_Bye"])
    odds["P_Final"] = (odds["P_Bye"] * sf_bye + semis_nonbye * sf_nonbye) / 100.0

    win_final = series_pack["P_WinFinal_Given_Final_SIM"]
    odds["P_Champ"] = odds["P_Final"] * (win_final / 100.0)

    odds["P_Final"] = np.minimum(odds["P_Semis"], odds["P_Final"])
    odds["P_Champ"] = np.minimum(odds["P_Final"], odds["P_Champ"])
    odds[["P_Semis", "P_Final", "P_Champ"]] = odds[["P_Semis", "P_Final", "P_Champ"]].clip(lower=0.0, upper=100.0)

    if not power_s.empty:
        odds["Power_Rating"] = power_s.reindex(odds.index)

    return odds, blended_seed_norm, win_df



def get_actual_playoff_matchups(df_playoff, week, is_championship_bracket=True, playoff_qualifiers=None):
    """
    Extract actual playoff matchups from data for a given week.

    Uses is_playoffs flag to identify championship bracket games.
    Returns actual game results and teams still alive.

    CRITICAL FIX: Includes teams on bye weeks by using playoff_qualifiers.
    Teams on bye (e.g., seeds 1-2) haven't played yet, but should still be "alive"
    for championship odds calculations.

    Args:
        df_playoff: DataFrame with playoff game data
        week: Current week to analyze
        is_championship_bracket: If True, filter for is_playoffs=1, else is_consolation=1
        playoff_qualifiers: Set of all teams that qualified for playoffs (includes bye teams)

    Returns:
        dict with:
            - 'games': List of (manager, opponent, winner) tuples for completed games
            - 'alive': Set of teams still alive in bracket
            - 'completed_weeks': Set of weeks with completed games
    """
    flag_col = 'is_playoffs' if is_championship_bracket else 'is_consolation'
    bracket_df = df_playoff[df_playoff[flag_col] == 1].copy()

    if bracket_df.empty:
        return {'games': [], 'alive': set(), 'completed_weeks': set()}

    # CRITICAL FIX: Start with ALL playoff qualifiers (includes bye teams)
    # Teams on bye haven't played yet, but they're still alive for championship
    if playoff_qualifiers is not None:
        all_teams = playoff_qualifiers
    else:
        # Fallback: Get teams that have played (old behavior)
        all_teams = set(bracket_df['manager'].unique())
    
    # Track completed games by week
    games_by_week = {}
    for wk in sorted(bracket_df['week'].dropna().unique()):
        wk = int(wk)
        if wk > week:
            continue
            
        week_games = bracket_df[bracket_df['week'] == wk]
        games = []
        
        # Process each unique matchup (avoid duplicates)
        processed_pairs = set()
        for _, row in week_games.iterrows():
            mgr = row['manager']
            opp = row['opponent']
            pair = tuple(sorted([mgr, opp]))
            
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)
            
            # Determine winner
            mgr_row = week_games[week_games['manager'] == mgr].iloc[0]
            opp_row = week_games[week_games['manager'] == opp].iloc[0]
            
            if pd.notna(mgr_row.get('win')) and pd.notna(opp_row.get('win')):
                if mgr_row['win'] == 1:
                    winner = mgr
                elif opp_row['win'] == 1:
                    winner = opp
                else:
                    # Tie or no result - use points
                    mgr_pts = mgr_row.get('team_points', 0)
                    opp_pts = opp_row.get('team_points', 0)
                    winner = mgr if mgr_pts > opp_pts else opp
            else:
                winner = None
            
            games.append((mgr, opp, winner))
        
        games_by_week[wk] = games
    
    # Determine who's still alive (haven't lost yet)
    # CRITICAL FIX: For championship odds, exclude current week's games
    # We want to show who's alive BEFORE this week's games, not after
    losers = set()
    completed_weeks = set()

    for wk in sorted(games_by_week.keys()):
        if wk >= week:  # Changed from > to >= - exclude current week
            break
        completed_weeks.add(wk)
        for mgr, opp, winner in games_by_week[wk]:
            if winner:
                loser = opp if winner == mgr else mgr
                losers.add(loser)

    alive = all_teams - losers
    
    return {
        'games': games_by_week.get(week, []),
        'alive': alive,
        'completed_weeks': completed_weeks,
        'all_games': games_by_week
    }


def calc_playoff_week_outputs(df_season, df_sched, season, week, inflation_rate=None, season_stats=None, data_directory=None):
    """
    Calculate playoff odds using DYNAMIC, GENERIC, SCALABLE simulation.

    NO HARD-CODING: All playoff structure comes from actual data and league settings.
    GUARANTEED COMPLETION: All simulations run to completion (no skips).
    PROBABILITIES SUM TO 100%: Proper normalization ensures valid probabilities.

    Architecture:
    - Part 1: Setup & State Extraction
    - Part 2: Dynamic Simulation Logic
    - Part 3: Calculate & Normalize Probabilities
    """
    import time
    t0 = time.time()
    rng = get_rng(season, week)
    df_to_date = df_season[df_season["week"] <= week].copy()

    # ========================================================================
    # PART 1: SETUP & STATE EXTRACTION
    # ========================================================================

    # [1.1] Load year-specific playoff configuration from league settings
    # CRITICAL FIX: Load settings per year instead of using globals
    from transformations.base.modules.playoff_bracket import load_league_settings

    settings = load_league_settings(season, data_directory=str(data_directory) if data_directory else None, df=df_to_date)

    num_playoff_teams = settings.get('num_playoff_teams', PLAYOFF_SLOTS)
    bye_teams = settings.get('bye_teams', BYE_SLOTS)
    uses_reseeding = settings.get('uses_playoff_reseeding', False)

    # Determine playoff start week from actual data
    playoff_weeks = df_to_date[df_to_date["is_playoffs"] == 1]["week"].dropna().unique()
    if len(playoff_weeks) > 0:
        # Ensure week is numeric before calling min()
        playoff_weeks = pd.to_numeric(playoff_weeks, errors='coerce')
        playoff_weeks = playoff_weeks[~pd.isna(playoff_weeks)]
        if len(playoff_weeks) > 0:
            playoff_start_week = int(playoff_weeks.min())
        else:
            playoff_start_week = 15  # Default fallback
    else:
        playoff_start_week = 15  # Default fallback

    print(f"[Dynamic Settings] Year={season}, Playoff Teams={num_playoff_teams}, Byes={bye_teams}, Reseeding={uses_reseeding}")

    # [1.2] Calculate seeds from regular season performance
    # CRITICAL: Use only regular season games (is_playoffs=0) for seeding
    reg = df_to_date[df_to_date["is_playoffs"] == 0].copy()
    wins_to_date, pts_to_date = wins_points_to_date(reg)

    # Tiebreakers: 1st = record (wins), 2nd = total points
    seeds = rank_and_seed(wins_to_date, pts_to_date, num_playoff_teams, bye_teams, played_raw=reg)
    playoff_teams = seeds.loc[seeds["made_playoffs"], "manager"].tolist()

    print(f"  [INFO] Playoff teams from standings (W-L, then PF tiebreaker): {playoff_teams}")

    # [1.3] Get current bracket state from ACTUAL playoff data
    playoff_qualifiers_set = set(playoff_teams)
    bracket_info = get_actual_playoff_matchups(df_to_date, week, is_championship_bracket=True, playoff_qualifiers=playoff_qualifiers_set)
    teams_alive = bracket_info['alive']
    all_playoff_games = bracket_info['all_games']

    print(f"[Bracket State] Week {week}: {len(teams_alive)} teams alive - {sorted(teams_alive)}")
    print(f"[Timing] Bracket state extracted in {time.time() - t0:.2f}s")

    # [1.4] Build team power models
    last_reg_week = schedules_last_regular_week(df_sched, season)
    if last_reg_week is None:
        if not reg.empty:
            weeks = pd.to_numeric(reg["week"], errors='coerce').dropna()
            last_reg_week = int(weeks.max()) if len(weeks) > 0 else week
        else:
            last_reg_week = week

    mu_hat, sigma_hat, samples_by_team, league_mu, sigma_floor = build_team_models(
        df_to_date, season, last_reg_week, HALF_LIFE_WEEKS, SHRINK_K, SIGMA_FLOOR_MIN,
        boundary_penalty=0.05, prior_w_cap=2.0, season_stats=season_stats
    )

    power_s = compute_power_ratings(mu_hat, samples_by_team, bootstrap_min=4)

    print(f"[Timing] Power models built in {time.time() - t0:.2f}s")
    # Normalize power ratings by inflation rate if provided
    if inflation_rate is not None and pd.notna(inflation_rate) and inflation_rate != 0:
        power_s = normalize_power_rating(power_s, inflation_rate)

    # ========================================================================
    # PART 2: DYNAMIC SIMULATION LOGIC
    # ========================================================================

    def get_actual_winner(teamA, teamB, week_limit):
        """Get actual winner from playoff games, if the matchup has been played."""
        if teamA is None or teamB is None:
            return None

        playoff_df = df_to_date[df_to_date["is_playoffs"] == 1].copy()

        # Find games between these two teams
        matchup_games = playoff_df[
            (playoff_df["manager"].isin([teamA, teamB])) &
            (playoff_df["opponent"].isin([teamA, teamB])) &
            (playoff_df["week"] <= week_limit)
        ]

        if matchup_games.empty:
            return None

        # Get the most recent week they played
        weeks = pd.to_numeric(matchup_games["week"], errors='coerce').dropna()
        if weeks.empty:
            return None
        recent_week = weeks.max()
        recent_games = matchup_games[matchup_games["week"] == recent_week]

        # Calculate winner by points
        pts = recent_games.groupby("manager")["team_points"].mean()

        if len(pts) != 2 or teamA not in pts.index or teamB not in pts.index:
            return None

        if pd.isna(pts[teamA]) or pd.isna(pts[teamB]):
            return None

        if pts[teamA] > pts[teamB]:
            return teamA
        elif pts[teamB] > pts[teamA]:
            return teamB
        else:
            # Tie - use alphabetical
            return min(teamA, teamB)

    def simulate_round(alive_teams, current_week, seeds_map, use_reseeding):
        """
        Simulate one round of playoffs.
        Returns: (winners_list, matchups_played)
        """
        if len(alive_teams) <= 1:
            return alive_teams, []

        # Pair teams by seed
        alive_sorted = sorted(alive_teams, key=lambda m: seeds_map.get(m, 999))

        matchups = []
        num_games = len(alive_sorted) // 2

        if use_reseeding:
            # Reseed: best vs worst, 2nd best vs 2nd worst, etc.
            for i in range(num_games):
                higher_seed = alive_sorted[i]
                lower_seed = alive_sorted[-(i+1)]
                matchups.append((higher_seed, lower_seed))
        else:
            # Fixed bracket: pair sequentially
            for i in range(num_games):
                matchups.append((alive_sorted[i*2], alive_sorted[i*2 + 1]))

        # Simulate each matchup
        winners = []
        for teamA, teamB in matchups:
            # Check if actual result exists
            actual_winner = get_actual_winner(teamA, teamB, current_week)

            if actual_winner:
                winner = actual_winner
            else:
                # Simulate the game
                winner = _sim_game(teamA, teamB, rng, mu_hat, sigma_hat, samples_by_team)

            winners.append(winner)

        return winners, matchups

    # Track all simulation results
    all_semifinals = []  # Teams that make it to final 4
    all_finalists = []   # Teams that make it to final 2
    all_champions = []   # Teams that win it all
    sims_for_seed_dist = []

    # Determine if we're in playoffs yet
    in_playoffs = week >= playoff_start_week

    if not in_playoffs:
        # Regular season - no playoff simulation needed, just track seeds
        for _ in range(N_SIMS):
            sims_for_seed_dist.append(seeds)
        print(f"[Timing] Simulations completed in {time.time() - t_sim:.2f}s")
    else:
        # Playoff simulation
        seeds_map = dict(zip(seeds["manager"], seeds["seed"]))

        # Determine locked champion if championship has been played
        if len(teams_alive) == 1:
            champ_locked = list(teams_alive)[0]
        else:
            champ_locked = None

        print(f"[Timing] Starting {N_SIMS} simulations with {len(teams_alive) if teams_alive else len(playoff_teams)} teams...")
        t_sim = time.time()
        # Run simulations - START FROM CURRENT BRACKET STATE
        for sim_num in range(N_SIMS):
            # Start with teams that are actually still alive at current week
            # This is the KEY optimization - don't simulate impossible scenarios
            current_alive = teams_alive.copy() if teams_alive else set(playoff_teams)

            # Simulate until we have a champion
            round_num = 0
            max_rounds = 10  # Safety limit

            while len(current_alive) > 1 and round_num < max_rounds:
                # Record teams at different stages
                if len(current_alive) == 4:
                    all_semifinals.extend(current_alive)
                elif len(current_alive) == 2:
                    all_finalists.extend(current_alive)

                # Simulate this round
                winners, matchups = simulate_round(
                    current_alive,
                    week,  # Use current week for actual result lookup
                    seeds_map,
                    uses_reseeding
                )

                # Advance winners
                current_alive = set(winners)
                round_num += 1

            # Record champion
            if len(current_alive) == 1:
                champion = list(current_alive)[0]
            elif champ_locked:
                champion = champ_locked
            else:
                # Shouldn't happen, but handle gracefully
                champion = list(current_alive)[0] if current_alive else playoff_teams[0]

            all_champions.append(champion)
            sims_for_seed_dist.append(seeds)
        print(f"[Timing] Simulations completed in {time.time() - t_sim:.2f}s")

    # ========================================================================
    # PART 3: CALCULATE & NORMALIZE PROBABILITIES
    # ========================================================================

    if not sims_for_seed_dist:
        return pd.DataFrame(), pd.DataFrame(), seeds

    # Build output DataFrame
    tall = pd.concat(sims_for_seed_dist, ignore_index=True)

    odds = (tall.groupby("manager")
            .agg(Exp_Final_Wins=("W", "mean"),
                 Exp_Final_PF=("PF", "mean")))
    odds["Avg_Seed"] = tall.groupby("manager")["seed"].mean()

    idx_mgrs = odds.index.tolist()

    # Calculate probabilities
    if in_playoffs:
        # Playoff probabilities
        odds["P_Semis"] = [all_semifinals.count(m) / N_SIMS * 100 for m in idx_mgrs]
        odds["P_Final"] = [all_finalists.count(m) / N_SIMS * 100 for m in idx_mgrs]
        odds["P_Champ"] = [all_champions.count(m) / N_SIMS * 100 for m in idx_mgrs]

        # Override with actual results if locked
        if len(teams_alive) == 2:
            # Championship game - finalists are locked
            finals_set = set(teams_alive)
            non_finalists = [m for m in idx_mgrs if m not in finals_set]
            if non_finalists:
                odds.loc[non_finalists, ["P_Final", "P_Champ"]] = 0.0
            odds.loc[list(finals_set), "P_Final"] = 100.0

            # Check if championship game has been played (winner determined)
            # Look at current week's games in bracket_info
            if bracket_info and 'games' in bracket_info:
                for mgr, opp, winner in bracket_info['games']:
                    if winner and mgr in finals_set and opp in finals_set:
                        # Championship game has been played - set winner to 100%
                        odds["P_Champ"] = 0.0
                        if winner in odds.index:
                            odds.at[winner, "P_Champ"] = 100.0
                        print(f"[Championship Complete] {winner} won championship - P_Champ set to 100%")
                        break

        if len(teams_alive) == 1:
            # Champion is locked
            champ = list(teams_alive)[0]
            odds["P_Champ"] = 0.0
            if champ in odds.index:
                odds.at[champ, "P_Champ"] = 100.0
    else:
        # Regular season - no playoff probabilities yet
        odds["P_Semis"] = 0.0
        odds["P_Final"] = 0.0
        odds["P_Champ"] = 0.0

    # Playoff/Bye probabilities (always deterministic at this week)
    playoff_set = set(playoff_teams)
    odds["P_Playoffs"] = [100.0 if m in playoff_set else 0.0 for m in idx_mgrs]
    odds["P_Bye"] = [100.0 if (m in playoff_set and bool(seeds.loc[seeds['manager'] == m, 'bye'].iloc[0])) else 0.0
                     for m in idx_mgrs]

    # Add power ratings
    if "Power_Rating" not in odds.columns:
        odds["Power_Rating"] = power_s.reindex(odds.index)

    # Seed distribution
    team_count = seeds.shape[0]
    seed_dist = (tall.pivot_table(index="manager", columns="seed", values="W",
                                  aggfunc="size", fill_value=0)
                 .div(len(sims_for_seed_dist)) * 100.0)
    all_cols = list(range(1, team_count + 1))
    seed_dist = seed_dist.reindex(columns=all_cols, fill_value=0.0)
    seed_dist_norm = normalize_seed_matrix_to_100(seed_dist)

    # Diagnostic output
    print(f"[PO diag] season={season} week={week}")
    if in_playoffs and len(teams_alive) == 2:
        champ_probs = odds.loc[odds.index.isin(teams_alive), "P_Champ"]
        print(f"[Championship Odds] {dict(champ_probs)} - Sum: {champ_probs.sum():.2f}%")

    return odds, seed_dist_norm, seeds


# -------------------------
# League Settings Loading
# -------------------------
def load_playoff_config_from_settings(ctx: LeagueContext):
    """
    Load playoff configuration from league settings JSON files.

    Args:
        ctx: LeagueContext object

    Returns:
        Tuple of (playoff_slots, bye_slots, bracket_reseed)
    """
    global PLAYOFF_SLOTS, BYE_SLOTS, BRACKET_RESEED

    # Look for league settings files
    settings_dir = ctx.data_directory / "league_settings"

    if not settings_dir.exists():
        print(f"[WARN] League settings directory not found: {settings_dir}")
        print(f"[WARN] Using default playoff configuration: {DEFAULT_PLAYOFF_SLOTS} teams, {DEFAULT_BYE_SLOTS} byes")
        return DEFAULT_PLAYOFF_SLOTS, DEFAULT_BYE_SLOTS, DEFAULT_BRACKET_RESEED

    # Find the most recent league settings file
    settings_files = list(settings_dir.glob("league_settings_*.json"))

    if not settings_files:
        print(f"[WARN] No league settings files found in {settings_dir}")
        print(f"[WARN] Using default playoff configuration: {DEFAULT_PLAYOFF_SLOTS} teams, {DEFAULT_BYE_SLOTS} byes")
        return DEFAULT_PLAYOFF_SLOTS, DEFAULT_BYE_SLOTS, DEFAULT_BRACKET_RESEED

    # Sort by year (most recent first) and take the first one
    settings_files.sort(reverse=True)
    settings_file = settings_files[0]

    print(f"[INFO] Loading playoff configuration from: {settings_file}")

    try:
        with open(settings_file, 'r') as f:
            settings_data = json.load(f)

        # Extract playoff configuration
        metadata = settings_data.get("metadata", {})

        num_playoff_teams = metadata.get("num_playoff_teams") or metadata.get("playoff_teams")
        playoff_start_week = metadata.get("playoff_start_week")
        uses_playoff_reseeding = metadata.get("uses_playoff_reseeding", "0")
        bye_teams = metadata.get("bye_teams")

        # Parse playoff slots
        if num_playoff_teams:
            try:
                playoff_slots = int(num_playoff_teams)
            except (ValueError, TypeError):
                playoff_slots = DEFAULT_PLAYOFF_SLOTS
        else:
            playoff_slots = DEFAULT_PLAYOFF_SLOTS

        # Get bye slots from settings (explicitly stored now)
        if bye_teams is not None:
            try:
                bye_slots = int(bye_teams)
            except (ValueError, TypeError):
                # Fallback to heuristic if invalid
                bye_slots = max(2, playoff_slots // 3) if playoff_slots >= 6 else 0
        else:
            # Fallback to heuristic for old settings files
            bye_slots = max(2, playoff_slots // 3) if playoff_slots >= 6 else 0

        # Parse bracket reseeding
        bracket_reseed = uses_playoff_reseeding == "1"

        print(f"[INFO] Playoff Configuration:")
        print(f"  - Playoff Teams: {playoff_slots}")
        print(f"  - Bye Slots: {bye_slots}")
        print(f"  - Bracket Reseeding: {bracket_reseed}")
        print(f"  - Playoff Start Week: {playoff_start_week or 'N/A'}")

        # Set global variables
        PLAYOFF_SLOTS = playoff_slots
        BYE_SLOTS = bye_slots
        BRACKET_RESEED = bracket_reseed

        return playoff_slots, bye_slots, bracket_reseed

    except Exception as e:
        print(f"[ERROR] Failed to load playoff configuration from {settings_file}: {e}")
        print(f"[WARN] Using default playoff configuration: {DEFAULT_PLAYOFF_SLOTS} teams, {DEFAULT_BYE_SLOTS} byes")
        return DEFAULT_PLAYOFF_SLOTS, DEFAULT_BYE_SLOTS, DEFAULT_BRACKET_RESEED


# -------------------------
# Main processing function
# -------------------------
def process_parquet_files(ctx: LeagueContext):
    """Process playoff odds using LeagueContext for file paths."""
    print(f"Loading data from: {ctx.data_directory}")

    # Load parquet files from context directories
    matchup_path = ctx.canonical_matchup_file

    # Locate schedule data (check multiple locations in priority order)
    # Prefer canonical schedule.parquet (enriched with playoff flags by enrich_schedule_with_playoff_flags.py)
    schedule_candidates = [
        Path(ctx.data_directory) / "schedule.parquet",  # 1st: canonical name (enriched)
        Path(ctx.data_directory) / "schedule_data_all_years.parquet",  # 2nd: legacy name
        Path(ctx.schedule_data_directory) / "schedule_data_all_years.parquet",  # 3rd: subdirectory
    ]

    schedule_path = None
    for candidate in schedule_candidates:
        if candidate.exists():
            schedule_path = candidate
            break

    if not schedule_path:
        print(f"ERROR: Schedule file not found in any location:")
        for candidate in schedule_candidates:
            print(f"  - {candidate}")
        return 1

    print(f"Loading matchup data from: {matchup_path}")
    print(f"Loading schedule data from: {schedule_path}")

    if not matchup_path.exists():
        print(f"ERROR: Matchup file not found: {matchup_path}")
        return 1

    # Load data
    df_matches = pd.read_parquet(matchup_path)
    df_sched = pd.read_parquet(schedule_path)

    print(f"Loaded {len(df_matches)} matchup records (including bye weeks)")
    print(f"Loaded {len(df_sched)} schedule records")

    # Filter out bye weeks (opponent=None rows added by bye_week_filler.py)
    # Bye weeks are for visualization/completeness but should not be included in playoff odds calculations
    if 'is_bye_week' in df_matches.columns:
        bye_week_count = df_matches['is_bye_week'].sum()
        if bye_week_count > 0:
            print(f"Filtering out {int(bye_week_count)} bye week rows (opponent=None)")
            df_matches = df_matches[df_matches['is_bye_week'] != 1].copy()
            print(f"Remaining matchup records: {len(df_matches)}")

    # Also filter by opponent not null as a safety check
    if df_matches['opponent'].isna().sum() > 0:
        null_opponent_count = df_matches['opponent'].isna().sum()
        print(f"Filtering out {null_opponent_count} rows with null opponent")
        df_matches = df_matches[df_matches['opponent'].notna()].copy()
        print(f"Remaining matchup records: {len(df_matches)}")

    # Clean blanks
    for df in (df_matches, df_sched):
        df.replace("", np.nan, inplace=True)

    # Coerce numeric columns
    numeric_cols_matches = ["week", "year", "is_playoffs", "is_consolation", "inflation_rate"]
    for col in numeric_cols_matches:
        if col in df_matches.columns:
            df_matches[col] = pd.to_numeric(df_matches[col], errors="coerce")

    for col in ["week", "year", "is_playoffs", "is_consolation"]:
        if col in df_sched.columns:
            df_sched[col] = pd.to_numeric(df_sched[col], errors="coerce")

    # Ensure target columns exist
    for col in TARGET_COLS:
        if col not in df_matches.columns:
            df_matches[col] = np.nan

    # Generate historical snapshots for kernel-based seed prediction
    # Note: For the first season in dataset, hist_df will be empty - this is handled
    # in calc_regular_week_outputs() by checking for prior history before blending
    hist_df = history_snapshots(df_matches, PLAYOFF_SLOTS)

    seasons = sorted(df_matches["year"].dropna().unique().astype(int))
    df_all = df_matches.copy()

    # Pre-calculate season-level statistics for all seasons
    # This allows us to use full-season variance estimates even for early-week calculations
    # without introducing lookahead bias (we're using league-level variance, not individual outcomes)
    print("\nPre-calculating season-level statistics...")
    season_stats = {}
    for season in seasons:
        df_season_all = df_matches[df_matches["year"] == season].copy()
        if not df_season_all.empty:
            season_mean = df_season_all["team_points"].mean()
            season_sd = df_season_all["team_points"].std(ddof=1)
            season_stats[season] = {
                "mean": float(season_mean),
                "sd": float(season_sd),
                "n_games": len(df_season_all)
            }
            print(f"  {season}: Mean={season_mean:.1f}, SD={season_sd:.1f}, Games={len(df_season_all)}")

    for season in seasons:
        print(f"\nProcessing season {season}...")
        df_season = df_matches[df_matches["year"] == season].copy()
        if df_season.empty:
            continue

        # Get pre-calculated season statistics
        season_stat = season_stats.get(season, {})

        # Get inflation rate for this season (if column exists)
        if "inflation_rate" in df_season.columns:
            season_inflation = df_season["inflation_rate"].dropna()
            inflation_rate = season_inflation.iloc[0] if not season_inflation.empty else None
        else:
            # inflation_rate column doesn't exist (not created by cumulative_stats_v2.py)
            # This is optional - power ratings will not be normalized across seasons
            inflation_rate = None

        reg_weeks = sorted(df_season[df_season["is_playoffs"] == 0]["week"].dropna().unique().astype(int))
        po_weeks = sorted(df_season[(df_season["is_playoffs"] == 1) & (df_season["is_consolation"] == 0)][
                              "week"].dropna().unique().astype(int))

        # Process regular season weeks
        for w in reg_weeks:
            print(f"  Processing regular season week {w}...")
            mask_week = (df_all["year"] == season) & (df_all["week"] == w) & (df_all["is_consolation"] == 0)
            if not mask_week.any():
                continue

            odds_df, seed_df, win_df = calc_regular_week_outputs(
                df_season, df_sched, season, w, hist_df, inflation_rate, season_stat, data_directory=ctx.data_directory
            )

            for idx in df_all[mask_week].index:
                m = df_all.at[idx, "manager"]
                if m not in odds_df.index:
                    continue

                def set_val(col, val):
                    if col in df_all.columns and pd.notna(val):
                        df_all.at[idx, col] = round(float(val), 2) if isinstance(val,
                                                                                 (float, np.floating, int)) else val

                set_val("avg_seed", odds_df.at[m, "Avg_Seed"])
                set_val("exp_final_wins", odds_df.at[m, "Exp_Final_Wins"])
                set_val("exp_final_pf", odds_df.at[m, "Exp_Final_PF"])
                set_val("p_semis", odds_df.at[m, "P_Semis"])
                set_val("p_final", odds_df.at[m, "P_Final"])
                set_val("p_champ", odds_df.at[m, "P_Champ"])
                set_val("p_playoffs", odds_df.at[m, "P_Playoffs"])
                set_val("p_bye", odds_df.at[m, "P_Bye"])

                # Power rating is already normalized in the odds_df
                if "Power_Rating" in odds_df.columns:
                    set_val("power_rating", odds_df.at[m, "Power_Rating"])

                if not seed_df.empty and m in seed_df.index:
                    for k in range(1, 11):
                        if k in seed_df.columns:
                            set_val(f"x{k}_seed", seed_df.loc[m, k])

                if (win_df is not None) and (m in win_df.index):
                    for k in range(0, 15):
                        col = f"x{k}_win"
                        if col in win_df.columns:
                            set_val(col, win_df.at[m, col])

        # Process playoff weeks
        for w in po_weeks:
            print(f"  Processing playoff week {w}...")
            mask_week = (
                    (df_all["year"] == season)
                    & (df_all["week"] == w)
                    & (df_all["is_playoffs"] == 1)
                    & (df_all["is_consolation"] == 0)
            )
            if not mask_week.any():
                continue

            odds_df, seed_df, seeds = calc_playoff_week_outputs(
                df_season, df_sched, season, w, inflation_rate, season_stat, data_directory=ctx.data_directory
            )

            for idx in df_all[mask_week].index:
                m = df_all.at[idx, "manager"]
                if m not in odds_df.index:
                    continue

                def set_val(col, val):
                    if col in df_all.columns and pd.notna(val):
                        df_all.at[idx, col] = round(float(val), 2) if isinstance(val,
                                                                                 (float, np.floating, int)) else val

                set_val("avg_seed", odds_df.at[m, "Avg_Seed"])
                set_val("exp_final_wins", odds_df.at[m, "Exp_Final_Wins"])
                set_val("exp_final_pf", odds_df.at[m, "Exp_Final_PF"])
                set_val("p_semis", odds_df.at[m, "P_Semis"])
                set_val("p_final", odds_df.at[m, "P_Final"])
                set_val("p_champ", odds_df.at[m, "P_Champ"])
                set_val("p_playoffs", odds_df.at[m, "P_Playoffs"])
                set_val("p_bye", odds_df.at[m, "P_Bye"])

                # Power rating is already normalized in the odds_df
                if "Power_Rating" in odds_df.columns:
                    set_val("power_rating", odds_df.at[m, "Power_Rating"])

                if not seed_df.empty and m in seed_df.index:
                    for k in range(1, 11):
                        if k in seed_df.columns:
                            set_val(f"x{k}_seed", seed_df.loc[m, k])

    # Add playoff scenario columns (p_playoffs_change, p_champ_change, is_critical_matchup, etc.)
    # These require p_playoffs column to exist, which was just calculated above
    print("\nAdding playoff scenario columns (changes, critical matchups)...")
    try:
        from transformations.base.modules.playoff_scenarios import add_playoff_scenario_columns
        df_all = add_playoff_scenario_columns(df_all, data_directory=str(ctx.data_directory))
        print("  [OK] Playoff scenario columns added")
    except Exception as e:
        print(f"  [WARN] Could not add playoff scenario columns: {e}")

    # DEFENSIVE: Ensure string columns are never null (use empty string instead)
    string_cols_to_clean = ['playoff_round', 'consolation_round', 'season_result']
    for col in string_cols_to_clean:
        if col in df_all.columns:
            df_all[col] = df_all[col].fillna("").astype(str)

    # Save outputs using context paths
    output_matchup_parquet = ctx.canonical_matchup_file
    output_matchup_csv = output_matchup_parquet.with_suffix('.csv')

    print(f"\nSaving results...")
    print(f"  Parquet: {output_matchup_parquet}")
    print(f"  CSV: {output_matchup_csv}")

    df_all.to_parquet(output_matchup_parquet, index=False, engine='pyarrow')
    df_all.to_csv(output_matchup_csv, index=False)

    print("\nProcessing complete!")
    print(f"Updated {len(df_all)} records with playoff odds")


# -------------------------
# Main
# -------------------------
def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Calculate playoff odds and probabilities for fantasy football matchups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with league context (RECOMMENDED)
    python playoff_odds_import.py --context /path/to/league_context.json

Note:
    This script reads playoff configuration (num_playoff_teams, bye_slots, etc.)
    from the league settings JSON files. If settings are not found, it uses
    default values (6 playoff teams, 2 byes).

    The script takes >10 minutes to run due to Monte Carlo simulations.
        """
    )
    parser.add_argument(
        "--context",
        type=Path,
        required=True,
        help="Path to league_context.json (required)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Fantasy Football Playoff Odds Calculator")
    print("=" * 70)

    # Load league context
    if not args.context.exists():
        print(f"ERROR: League context file not found: {args.context}")
        sys.exit(1)

    try:
        ctx = LeagueContext.load(str(args.context))
        print(f"Loaded context for league: {ctx.league_name}")
    except Exception as e:
        print(f"ERROR: Failed to load league context: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load playoff configuration from league settings
    print("\n" + "=" * 70)
    print("LOADING PLAYOFF CONFIGURATION")
    print("=" * 70)

    try:
        playoff_slots, bye_slots, bracket_reseed = load_playoff_config_from_settings(ctx)
    except Exception as e:
        print(f"ERROR: Failed to load playoff configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Process playoff odds
    print("\n" + "=" * 70)
    print("CALCULATING PLAYOFF ODDS")
    print("=" * 70)

    try:
        process_parquet_files(ctx)
        print("\n" + "=" * 70)
        print("SUCCESS - Playoff odds calculation complete!")
        print("=" * 70)
    except Exception as e:
        print(f"\nERROR: Playoff odds calculation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()