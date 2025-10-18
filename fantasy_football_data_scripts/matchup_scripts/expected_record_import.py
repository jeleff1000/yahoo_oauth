#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import numpy as np
from collections import defaultdict
import random
from pathlib import Path
from md.md_utils import df_from_md_or_parquet

# =========================
# Config
# =========================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "fantasy_football_data"

MATCHUP_PATH = DATA_DIR / "matchup.parquet"
OUTPUT_MATCHUP_PARQUET = DATA_DIR / "matchup.parquet"
OUTPUT_MATCHUP_CSV = DATA_DIR / "matchup.csv"
N_SIMS = 100000
RNG_SEED = None

RNG = random.Random(RNG_SEED)


# =========================
# Round-robin generator
# =========================
def round_robin_weeks(managers):
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


# =========================
# Validator
# =========================
def validate_schedule(sched, teams, no_repeats_weeks=5, max_meetings=2):
    pair_ct = defaultdict(int)
    earliest = min(sched.keys())
    latest = max(sched.keys())
    early_end = min(earliest + no_repeats_weeks - 1, latest)
    seen_early = set()
    for w in range(earliest, early_end + 1):
        for a, b in sched[w]:
            p = tuple(sorted((a, b)))
            if p in seen_early:
                return False, f"Repeat within weeks {earliest}-{early_end}: {p} in week {w}"
            seen_early.add(p)

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


# =========================
# Schedule builder
# =========================
def schedule_nxN(managers, n_weeks, validate=False):
    teams = list(managers)
    n = len(teams)
    assert n % 2 == 0 and n >= 4, "Team count must be even and ≥4"
    max_weeks = 2 * (n - 1)
    assert n_weeks <= max_weeks, f"n_weeks>{max_weeks} would force some pair >2 meetings"

    M = list(managers)
    RNG.shuffle(M)

    rr = round_robin_weeks(M)
    base_weeks = list(rr.keys())
    last_rr = base_weeks[-1]

    sched = {w: rr[w][:] for w in range(1, min(last_rr, n_weeks) + 1)}

    extra = max(0, n_weeks - last_rr)
    if extra:
        pick = RNG.sample(base_weeks, extra)
        RNG.shuffle(pick)
        for off, base_w in enumerate(pick, start=last_rr + 1):
            sched[off] = rr[base_w][:]

    if n_weeks > 5:
        tail_old = list(range(6, n_weeks + 1))
        RNG.shuffle(tail_old)
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


# =========================
# One simulation
# =========================
def simulate_once(points_by_mgr_week, managers, n_weeks):
    sched = schedule_nxN(managers, n_weeks, validate=False)
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
                wins[a if RNG.random() < 0.5 else b] += 1

    ladder = [(m, wins[m], cum_points[m]) for m in managers]
    ladder.sort(key=lambda x: (-x[1], -x[2], x[0]))
    seeds = {m: i + 1 for i, (m, _, _) in enumerate(ladder)}
    return wins, seeds


# =========================
# Latest completed regular week
# =========================
def current_regular_week(df_season):
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


# =========================
# Main processing
# =========================
def process_expected_records():
    print(f"Loading data from: {MATCHUP_PATH}")

    # Load from MotherDuck if available, otherwise fall back to local parquet
    raw_all = df_from_md_or_parquet("matchup", MATCHUP_PATH)
    print(f"Loaded {len(raw_all)} records")

    # Regular season only for simulations
    regular_mask = (raw_all['is_playoffs'] == 0) & (raw_all['is_consolation'] == 0)
    raw = raw_all[regular_mask].copy()

    # Ensure output columns exist
    seed_cols = [f"shuffle_{i}_seed" for i in range(1, 11)]
    win_cols = [f"shuffle_{w}_win" for w in range(0, 15)]
    summary_cols = [
        "shuffle_avg_wins",
        "shuffle_avg_seed",
        "shuffle_avg_playoffs",
        "shuffle_avg_bye",
        "wins_vs_shuffle_wins",
        "seed_vs_shuffle_seed",
    ]

    for df in (raw, raw_all):
        for c in seed_cols + win_cols + summary_cols:
            if c not in df.columns:
                df[c] = np.nan

    # Clear all shuffle/summary fields for non-regular rows
    clear_cols = seed_cols + win_cols + summary_cols
    raw_all.loc[~regular_mask, clear_cols] = np.nan

    # Process by season
    for year, df_year in raw.groupby('year', sort=True):
        print(f"\nProcessing season {year}...")
        managers = tuple(sorted(df_year['manager'].unique()))
        n_managers = len(managers)

        if n_managers % 2 != 0:
            print(f"  Skipping - odd team count ({n_managers})")
            continue

        last_wk = current_regular_week(df_year)
        if last_wk == 0:
            print(f"  Skipping - no complete weeks")
            continue

        print(f"  {n_managers} managers, {last_wk} complete weeks")
        df_year_points = df_year[['manager', 'week', 'team_points']].copy()

        for wk in range(1, last_wk + 1):
            print(f"    Processing week {wk}...", end='', flush=True)
            n_weeks_cap = 2 * (n_managers - 1)
            n_weeks = min(wk, n_weeks_cap)

            df_to_week = df_year_points[df_year_points['week'] <= n_weeks]
            points = {(r['manager'], int(r['week'])): float(r['team_points'])
                      for _, r in df_to_week.iterrows()}

            win_hists = {mgr: [0] * (n_weeks + 1) for mgr in managers}
            seed_hists = {mgr: [0] * min(n_managers, 10) for mgr in managers}

            for _ in range(N_SIMS):
                wins, seeds = simulate_once(points, managers, n_weeks)
                for mgr in managers:
                    wc = max(0, min(wins.get(mgr, 0), n_weeks))
                    sd = seeds.get(mgr, 1)
                    win_hists[mgr][wc] += 1
                    if 1 <= sd <= len(seed_hists[mgr]):
                        seed_hists[mgr][sd - 1] += 1

            win_den = {m: max(1, sum(win_hists[m])) for m in managers}
            seed_den = {m: max(1, sum(seed_hists[m])) for m in managers}

            # Write snapshot
            rows = raw.index[(raw['year'] == year) & (raw['week'] == wk)]
            for idx in rows:
                mgr = raw.at[idx, 'manager']
                if mgr not in managers:
                    continue
                for wv in range(0, min(n_weeks, 14) + 1):
                    raw.at[idx, f"shuffle_{wv}_win"] = round(
                        100.0 * win_hists[mgr][wv] / win_den[mgr], 2
                    )
                for s in range(1, min(n_managers, 10) + 1):
                    raw.at[idx, f"shuffle_{s}_seed"] = round(
                        100.0 * seed_hists[mgr][s - 1] / seed_den[mgr], 2
                    )
            print(" done")

    # Merge results back
    raw_all.update(raw)

    # =========================
    # Compute summary columns (regular rows only)
    # =========================
    print("\nComputing summary statistics...")

    for c in summary_cols:
        raw_all[c] = np.nan

    # Expected wins
    win_prob_cols = [c for c in [f"shuffle_{w}_win" for w in range(0, 15)] if c in raw_all.columns]
    if win_prob_cols:
        df_reg = raw_all.loc[regular_mask, win_prob_cols]
        if not df_reg.empty:
            weights = np.arange(0, len(win_prob_cols), dtype=float)
            vals = np.round((df_reg.fillna(0.0).to_numpy(dtype=float) @ weights) / 100.0, 2)
            has_data = df_reg.notna().any(axis=1)
            raw_all.loc[regular_mask, "shuffle_avg_wins"] = vals
            raw_all.loc[regular_mask & ~has_data, "shuffle_avg_wins"] = np.nan

    # Expected seed
    seed_prob_cols = [c for c in [f"shuffle_{s}_seed" for s in range(1, 11)] if c in raw_all.columns]
    if seed_prob_cols:
        df_reg = raw_all.loc[regular_mask, seed_prob_cols]
        if not df_reg.empty:
            weights = np.arange(1, len(seed_prob_cols) + 1, dtype=float)
            vals = np.round((df_reg.fillna(0.0).to_numpy(dtype=float) @ weights) / 100.0, 2)
            has_data = df_reg.notna().any(axis=1)
            raw_all.loc[regular_mask, "shuffle_avg_seed"] = vals
            raw_all.loc[regular_mask & ~has_data, "shuffle_avg_seed"] = np.nan

    # Playoff odds (seeds 1-6)
    playoff_cols = [c for c in [f"shuffle_{s}_seed" for s in range(1, 7)] if c in raw_all.columns]
    if playoff_cols:
        df_reg = raw_all.loc[regular_mask, playoff_cols]
        if not df_reg.empty:
            vals = df_reg.fillna(0.0).sum(axis=1)
            has_data = df_reg.notna().any(axis=1)
            raw_all.loc[regular_mask, "shuffle_avg_playoffs"] = vals
            raw_all.loc[regular_mask & ~has_data, "shuffle_avg_playoffs"] = np.nan

    # Bye odds (seeds 1-2)
    bye_cols = [c for c in ["shuffle_1_seed", "shuffle_2_seed"] if c in raw_all.columns]
    if bye_cols:
        df_reg = raw_all.loc[regular_mask, bye_cols]
        if not df_reg.empty:
            vals = df_reg.fillna(0.0).sum(axis=1)
            has_data = df_reg.notna().any(axis=1)
            raw_all.loc[regular_mask, "shuffle_avg_bye"] = vals
            raw_all.loc[regular_mask & ~has_data, "shuffle_avg_bye"] = np.nan

    # =========================
    # Deltas vs actuals (manager-centric)
    # =========================
    # Use underscored source columns supplied in your dataset:
    #   wins_to_date, playoff_seed_to_date
    if "wins_to_date" in raw_all.columns and "shuffle_avg_wins" in raw_all.columns:
        mask = regular_mask & raw_all["wins_to_date"].notna() & raw_all["shuffle_avg_wins"].notna()
        raw_all.loc[mask, "wins_vs_shuffle_wins"] = (
            pd.to_numeric(raw_all.loc[mask, "wins_to_date"], errors="coerce") -
            pd.to_numeric(raw_all.loc[mask, "shuffle_avg_wins"], errors="coerce")
        ).round(2)

    if "playoff_seed_to_date" in raw_all.columns and "shuffle_avg_seed" in raw_all.columns:
        mask = regular_mask & raw_all["playoff_seed_to_date"].notna() & raw_all["shuffle_avg_seed"].notna()
        raw_all.loc[mask, "seed_vs_shuffle_seed"] = (
            pd.to_numeric(raw_all.loc[mask, "playoff_seed_to_date"], errors="coerce") -
            pd.to_numeric(raw_all.loc[mask, "shuffle_avg_seed"], errors="coerce")
        ).round(2)

    # =========================
    # Lock postseason rows to final regular-week snapshot
    # =========================
    print("Locking postseason values...")
    post_mask = (raw_all['is_playoffs'] == 1) | (raw_all['is_consolation'] == 1)
    if post_mask.any():
        cols_to_copy = [c for c in (seed_cols + win_cols + summary_cols) if c in raw_all.columns]
        for year in sorted(raw_all['year'].dropna().unique()):
            df_reg_year = raw_all[(raw_all['year'] == year) & regular_mask]
            if df_reg_year.empty:
                continue
            last_wk = current_regular_week(df_reg_year)
            if last_wk == 0:
                continue

            final_rows = df_reg_year[df_reg_year['week'] == last_wk]
            if final_rows.empty:
                continue

            idx_post_year = raw_all.index[(raw_all['year'] == year) & post_mask]
            for idx in idx_post_year:
                mgr = raw_all.at[idx, 'manager']
                src = final_rows[final_rows['manager'] == mgr]
                if src.empty:
                    continue
                src_row = src.iloc[0]
                for c in cols_to_copy:
                    raw_all.at[idx, c] = src_row[c]

    # Save outputs
    print(f"\nSaving results...")
    print(f"  Parquet: {OUTPUT_MATCHUP_PARQUET}")
    print(f"  CSV: {OUTPUT_MATCHUP_CSV}")

    raw_all.to_parquet(OUTPUT_MATCHUP_PARQUET, index=False)
    raw_all.to_csv(OUTPUT_MATCHUP_CSV, index=False)

    print("\nProcessing complete!")
    print(f"Updated {len(raw_all)} records with expected record simulations")


# =========================
# Main
# =========================
if __name__ == "__main__":
    process_expected_records()
