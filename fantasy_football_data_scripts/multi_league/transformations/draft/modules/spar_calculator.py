"""
SPAR (Season Points Above Replacement) Calculator Module for Draft Analysis

Provides functions for calculating value above replacement metrics for draft picks.

Dual metrics distinguish player talent from manager usage:
- player_spar: All games the player played (talent/production)
- manager_spar: Only games the manager started them (realized value)

Key Concepts:
- player_spar = total_fantasy_points - (replacement_ppg × games_played)
- manager_spar = total_fantasy_points_started - (replacement_ppg × games_started)
- PGVOR = season_ppg - replacement_ppg
- ROI = manager_spar / cost_norm (ROI based on realized value)

Draft Type Handling:
- Automatically detects auction vs snake per year from data
- For auction: cost_norm = actual cost
- For snake: cost_norm = synthetic cost based on pick position (exponential decay)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import json
from pathlib import Path

from .draft_type_utils import (
    detect_draft_type_for_year,
    detect_budget_for_year,
    get_normalized_cost,
    summarize_draft_types
)


def normalize_draft_cost(
    draft_df: pd.DataFrame,
    league_settings_path: Optional[Path] = None,
    year_column: str = 'year',
    cost_column: str = 'cost',
    pick_column: str = 'pick',
    default_budget: int = 200
) -> pd.DataFrame:
    """
    Normalize draft cost to make auction and snake comparable.

    Dynamically detects draft type PER YEAR from the data itself:
    - Auction years: cost_norm = actual cost
    - Snake years: cost_norm = synthetic cost (exponential decay from pick)

    This handles mixed datasets where some years are auction and some are snake.

    Args:
        draft_df: Draft DataFrame
        league_settings_path: Optional path to league settings JSON (for budget fallback)
        year_column: Column containing year
        cost_column: Column containing auction cost
        pick_column: Column containing pick number
        default_budget: Default auction budget if not detected

    Returns:
        DataFrame with cost_norm column added
    """
    draft_df = draft_df.copy()

    # Try to get default budget from league settings if provided
    budget = default_budget
    if league_settings_path and Path(league_settings_path).exists():
        try:
            with open(league_settings_path, 'r') as f:
                settings = json.load(f)
            budget = int(settings.get('settings', {}).get('auction_budget', default_budget))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Use the shared utility for per-year normalized cost
    draft_df['cost_norm'] = get_normalized_cost(
        draft_df,
        year_column=year_column,
        cost_column=cost_column,
        pick_column=pick_column,
        round_column='round',
        default_budget=budget
    )

    return draft_df


def calculate_draft_spar(
    draft_df: pd.DataFrame,
    replacement_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate SPAR (Season Points Above Replacement) for draft picks with dual player/manager metrics.

    Args:
        draft_df: Draft DataFrame with performance metrics
        replacement_df: Season replacement levels (year, position, replacement_ppg_season)

    Returns:
        DataFrame with SPAR columns added:

        Baseline:
        - replacement_ppg: Position replacement baseline

        Player SPAR (talent, all games):
        - player_spar: Season SPAR for all games played
        - player_ppg: Season PPG for all games
        - player_pgvor: Per-game VOR (all games)

        Manager SPAR (usage, started games only):
        - manager_spar: Season SPAR for started games only
        - manager_ppg: Season PPG for started games
        - manager_pgvor: Per-game VOR (started only)

        Legacy (backward compatibility):
        - spar: Alias for manager_spar
        - pgvor: Alias for manager_pgvor
    """
    draft_df = draft_df.copy()

    # Ensure numeric types for ALL games (player metrics)
    if 'total_fantasy_points' in draft_df.columns:
        draft_df['total_fantasy_points'] = pd.to_numeric(
            draft_df['total_fantasy_points'], errors='coerce'
        ).fillna(0)

    if 'games_played' in draft_df.columns:
        draft_df['games_played'] = pd.to_numeric(
            draft_df['games_played'], errors='coerce'
        ).fillna(0)

    if 'season_ppg' in draft_df.columns:
        draft_df['season_ppg'] = pd.to_numeric(
            draft_df['season_ppg'], errors='coerce'
        ).fillna(0)

    # Ensure numeric types for STARTED games (manager metrics)
    if 'total_fantasy_points_started' in draft_df.columns:
        draft_df['total_fantasy_points_started'] = pd.to_numeric(
            draft_df['total_fantasy_points_started'], errors='coerce'
        ).fillna(0)

    if 'games_started' in draft_df.columns:
        draft_df['games_started'] = pd.to_numeric(
            draft_df['games_started'], errors='coerce'
        ).fillna(0)

    if 'season_ppg_started' in draft_df.columns:
        draft_df['season_ppg_started'] = pd.to_numeric(
            draft_df['season_ppg_started'], errors='coerce'
        ).fillna(0)

    # Determine position column (prefer 'position', fallback to 'yahoo_position')
    pos_col = 'position' if 'position' in draft_df.columns else 'yahoo_position'

    # Join with replacement levels
    draft_with_rep = draft_df.merge(
        replacement_df[['year', 'position', 'replacement_ppg_season']],
        left_on=['year', pos_col],
        right_on=['year', 'position'],
        how='left',
        suffixes=('', '_rep')
    ).reset_index(drop=True)

    # Clean up duplicate position column if created
    if 'position_rep' in draft_with_rep.columns:
        draft_with_rep = draft_with_rep.drop(columns=['position_rep'])

    # DO NOT fill missing replacement values with 0 - leave as NaN
    # This makes missing data obvious (NaN SPAR) instead of silently wrong (SPAR = points)
    missing_replacement = draft_with_rep['replacement_ppg_season'].isna()
    if missing_replacement.any():
        missing_positions = draft_with_rep.loc[missing_replacement, pos_col].unique()
        print(f"  [WARN] Missing replacement levels for positions: {list(missing_positions)}")
        print(f"         SPAR will be NaN for {missing_replacement.sum()} rows")

    # Create baseline column (keeps NaN where missing)
    draft_with_rep['replacement_ppg'] = draft_with_rep['replacement_ppg_season']

    # Drop any duplicate columns
    draft_with_rep = draft_with_rep.loc[:, ~draft_with_rep.columns.duplicated()]

    # ===== PLAYER SPAR (all games the player played) =====
    if 'total_fantasy_points' in draft_with_rep.columns and 'games_played' in draft_with_rep.columns:
        replacement_points = draft_with_rep['replacement_ppg'].to_numpy() * draft_with_rep['games_played'].to_numpy()
        draft_with_rep['player_spar'] = draft_with_rep['total_fantasy_points'].to_numpy() - replacement_points
    else:
        draft_with_rep['player_spar'] = 0.0

    if 'season_ppg' in draft_with_rep.columns:
        draft_with_rep['player_ppg'] = draft_with_rep['season_ppg'].to_numpy()
        draft_with_rep['player_pgvor'] = draft_with_rep['season_ppg'].to_numpy() - draft_with_rep['replacement_ppg'].to_numpy()
    else:
        draft_with_rep['player_ppg'] = 0.0
        draft_with_rep['player_pgvor'] = 0.0

    # ===== MANAGER SPAR (only games the manager started the player) =====
    if 'total_fantasy_points_started' in draft_with_rep.columns and 'games_started' in draft_with_rep.columns:
        replacement_points_started = draft_with_rep['replacement_ppg'].to_numpy() * draft_with_rep['games_started'].to_numpy()
        draft_with_rep['manager_spar'] = draft_with_rep['total_fantasy_points_started'].to_numpy() - replacement_points_started
    else:
        # Fallback: if no started metrics, assume same as player metrics
        draft_with_rep['manager_spar'] = draft_with_rep['player_spar']

    if 'season_ppg_started' in draft_with_rep.columns:
        draft_with_rep['manager_ppg'] = draft_with_rep['season_ppg_started'].to_numpy()
        draft_with_rep['manager_pgvor'] = draft_with_rep['season_ppg_started'].to_numpy() - draft_with_rep['replacement_ppg'].to_numpy()
    else:
        draft_with_rep['manager_ppg'] = draft_with_rep['player_ppg']
        draft_with_rep['manager_pgvor'] = draft_with_rep['player_pgvor']

    # ===== PER-GAME SPAR METRICS =====
    # Player SPAR per game (all games)
    if 'player_spar' in draft_with_rep.columns and 'games_played' in draft_with_rep.columns:
        games_played_safe = draft_with_rep['games_played'].clip(lower=1)
        draft_with_rep['player_spar_per_game'] = draft_with_rep['player_spar'] / games_played_safe
    else:
        draft_with_rep['player_spar_per_game'] = 0.0

    # Manager SPAR per game (started games only)
    if 'manager_spar' in draft_with_rep.columns and 'games_started' in draft_with_rep.columns:
        games_started_safe = draft_with_rep['games_started'].clip(lower=1)
        draft_with_rep['manager_spar_per_game'] = draft_with_rep['manager_spar'] / games_started_safe
    else:
        draft_with_rep['manager_spar_per_game'] = 0.0

    # ===== BACKWARD COMPATIBILITY ALIASES =====
    # spar/pgvor are aliases for manager_* (conservative default: show realized value)
    draft_with_rep['spar'] = draft_with_rep['manager_spar']
    draft_with_rep['pgvor'] = draft_with_rep['manager_pgvor']

    return draft_with_rep


def calculate_draft_roi(draft_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate draft ROI (Return on Investment).

    ROI = SPAR / cost_norm

    Args:
        draft_df: Draft DataFrame with spar and cost_norm columns

    Returns:
        DataFrame with draft_roi column added
    """
    draft_df = draft_df.copy()

    if 'spar' in draft_df.columns and 'cost_norm' in draft_df.columns:
        # Ensure cost_norm > 0 to avoid division by zero
        cost_norm_safe = draft_df['cost_norm'].clip(lower=0.1)
        draft_df['draft_roi'] = draft_df['spar'] / cost_norm_safe
    else:
        draft_df['draft_roi'] = 0.0

    return draft_df


def calculate_additional_draft_metrics(draft_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional SPAR-based metrics for draft analysis.

    Adds:
        - spar_per_dollar: SPAR / cost (raw dollars)
        - spar_per_dollar_norm: SPAR / cost_norm (normalized)
        - spar_per_pick: SPAR / overall_pick_number
        - spar_per_round: SPAR / round
        - spar_per_dollar_rank: Rank by SPAR/$ within position/year
        - spar_per_pick_rank: Rank by SPAR/pick within position/year

    Args:
        draft_df: Draft DataFrame with spar, cost, and draft_roi already calculated

    Returns:
        DataFrame with additional SPAR metrics
    """
    draft_df = draft_df.copy()

    # SPAR per dollar metrics
    if 'cost' in draft_df.columns:
        cost_safe = pd.to_numeric(draft_df['cost'], errors='coerce').fillna(1).clip(lower=1)
        draft_df['spar_per_dollar'] = draft_df['spar'] / cost_safe
    else:
        draft_df['spar_per_dollar'] = 0.0

    # SPAR per normalized cost (alias for draft_roi for consistency)
    if 'draft_roi' in draft_df.columns:
        draft_df['spar_per_dollar_norm'] = draft_df['draft_roi']
    else:
        draft_df['spar_per_dollar_norm'] = 0.0

    # SPAR per draft position metrics
    if 'pick' in draft_df.columns:
        pick_safe = pd.to_numeric(draft_df['pick'], errors='coerce').fillna(1).clip(lower=1)
        draft_df['spar_per_pick'] = draft_df['spar'] / pick_safe
    else:
        draft_df['spar_per_pick'] = 0.0

    if 'round' in draft_df.columns:
        round_safe = pd.to_numeric(draft_df['round'], errors='coerce').fillna(1).clip(lower=1)
        draft_df['spar_per_round'] = draft_df['spar'] / round_safe
    else:
        draft_df['spar_per_round'] = 0.0

    # Rankings within position/year
    pos_col = 'position' if 'position' in draft_df.columns else 'yahoo_position'

    if pos_col in draft_df.columns and 'year' in draft_df.columns:
        draft_df['spar_per_dollar_rank'] = draft_df.groupby(['year', pos_col])['spar_per_dollar'].rank(
            ascending=False, method='min'
        )

        draft_df['spar_per_pick_rank'] = draft_df.groupby(['year', pos_col])['spar_per_pick'].rank(
            ascending=False, method='min'
        )
    else:
        draft_df['spar_per_dollar_rank'] = 0
        draft_df['spar_per_pick_rank'] = 0

    return draft_df


def calculate_keeper_metrics(draft_df: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED: Keeper-specific SPAR metrics have been removed.

    Keepers now use the same standard SPAR metrics as all drafted players:
    - spar: Season Points Above Replacement
    - draft_roi: SPAR / cost (same for keepers and non-keepers)

    To analyze keepers, filter by is_keeper_status = 1 and use standard metrics.

    This function is kept for backwards compatibility but does nothing.

    Args:
        draft_df: Draft DataFrame

    Returns:
        Unmodified DataFrame (no keeper-specific columns added)
    """
    # Function deprecated - keepers use same SPAR metrics as all drafted players
    return draft_df


def calculate_all_draft_metrics(
    draft_df: pd.DataFrame,
    replacement_df: pd.DataFrame,
    league_settings_path: Path
) -> pd.DataFrame:
    """
    Calculate all SPAR-based draft metrics.

    Args:
        draft_df: Draft DataFrame with performance metrics
        replacement_df: Replacement levels DataFrame
        league_settings_path: Path to league settings JSON

    Returns:
        Draft DataFrame with all SPAR metrics added:
        - replacement_ppg: Position replacement baseline
        - spar: Season points above replacement
        - pgvor: Per-game value over replacement
        - cost_norm: Normalized draft cost
        - draft_roi: Return on investment
        - spar_per_dollar: SPAR per raw dollar
        - spar_per_pick: SPAR per draft pick
        - spar_per_round: SPAR per round
        - keeper_spar_per_dollar: Keeper SPAR efficiency
        - keeper_surplus_spar: Keeper value gained
        - keeper_roi_spar: Keeper return on investment
    """
    # Step 1: Calculate SPAR (or use existing from player table join)
    # Check if SPAR columns already exist from player_to_draft_v2.py join
    has_player_spar = 'player_spar' in draft_df.columns and 'manager_spar' in draft_df.columns
    has_replacement_ppg = 'replacement_ppg' in draft_df.columns

    if has_player_spar and has_replacement_ppg:
        print("  [SKIP] SPAR columns already exist from player table join - using existing values")

        # Add player_ppg and manager_ppg columns (required by Streamlit queries)
        if 'season_ppg' in draft_df.columns:
            draft_df['player_ppg'] = draft_df['season_ppg']
            draft_df['player_pgvor'] = draft_df['season_ppg'] - draft_df['replacement_ppg']
        else:
            draft_df['player_ppg'] = 0.0
            draft_df['player_pgvor'] = 0.0

        if 'season_ppg_started' in draft_df.columns:
            draft_df['manager_ppg'] = draft_df['season_ppg_started']
            draft_df['manager_pgvor'] = draft_df['season_ppg_started'] - draft_df['replacement_ppg']
        else:
            draft_df['manager_ppg'] = draft_df['player_ppg']
            draft_df['manager_pgvor'] = draft_df['player_pgvor']

        # Ensure backward compatibility aliases exist
        if 'spar' not in draft_df.columns:
            draft_df['spar'] = draft_df['manager_spar']
        if 'pgvor' not in draft_df.columns and 'manager_pgvor' in draft_df.columns:
            draft_df['pgvor'] = draft_df['manager_pgvor']
        elif 'pgvor' not in draft_df.columns and 'season_ppg' in draft_df.columns:
            # Calculate pgvor from season_ppg if manager_pgvor doesn't exist
            draft_df['pgvor'] = draft_df['season_ppg'] - draft_df['replacement_ppg']
    else:
        print("  [CALC] Calculating SPAR from draft data (player table join didn't provide SPAR)")
        draft_df = calculate_draft_spar(draft_df, replacement_df)

    # Step 2: Normalize cost
    draft_df = normalize_draft_cost(draft_df, league_settings_path)

    # Step 3: Calculate ROI
    draft_df = calculate_draft_roi(draft_df)

    # Step 4: Calculate additional SPAR metrics
    draft_df = calculate_additional_draft_metrics(draft_df)

    # Step 5: Keeper-specific metrics now calculated in player table by player_keeper_spar_v3.py
    # and copied to draft table by keeper_spar_to_draft.py (PASS 3)
    # draft_df = calculate_keeper_metrics(draft_df)  # DEPRECATED: Moved to dedicated scripts

    return draft_df
