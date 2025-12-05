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
Cumulative Records Module

Calculates running win/loss totals and streaks.

RECALCULATE WEEKLY: All columns in this module must be recalculated every week.
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
import pandas as pd
import numpy as np
@ensure_normalized
def calculate_cumulative_records(df: pd.DataFrame, data_directory: str = None) -> pd.DataFrame:
    """
    Calculate cumulative win/loss records and streaks.
    RECALCULATE WEEKLY columns:
    - cumulative_wins: Running total of wins (all-time)
    - cumulative_losses: Running total of losses (all-time)
    - wins_to_date: Running total within season (excluding consolation)
    - losses_to_date: Running total within season (excluding consolation)
    - points_scored_to_date: Running total of points within season (excluding consolation)
    - win_streak: Current active win streak
    - loss_streak: Current active loss streak
    - playoff_seed_to_date: Current playoff seeding (ranked by wins then points)
    Args:
        df: DataFrame with manager, year, week, win, loss columns
        data_directory: Path to league data directory (for finding league settings)
    Returns:
        DataFrame with cumulative record columns added
    """
    df = df.copy()
    # --- FORCE_CONSOLATION_RULE: normalize consolation vs playoffs
    if 'is_playoffs' not in df.columns:
        df['is_playoffs'] = 0
    else:
        df['is_playoffs'] = pd.to_numeric(df['is_playoffs'], errors='coerce').fillna(0).astype(int)
    if 'is_consolation' not in df.columns:
        df['is_consolation'] = 0
    else:
        df['is_consolation'] = pd.to_numeric(df['is_consolation'], errors='coerce').fillna(0).astype(int)
    # Mutual exclusivity:
    #  - If playoffs == 1, force consolation = 0
    #  - If consolation == 1, force playoffs = 0
    df.loc[df['is_playoffs'] == 1, 'is_consolation'] = 0
    df.loc[df['is_consolation'] == 1, 'is_playoffs'] = 0
    # Ensure required columns exist
    for col in ['manager', 'year', 'week', 'win', 'loss']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Defensive: ensure manager is string (do not overwrite legit names)
    df['manager'] = df['manager'].astype(str)
    # Defensive: ensure team_points exists and is numeric for points_scored_to_date
    # PRESERVE NULLS - don't fill with 0.0 (null scores occur in special cases like KMFFL 2013 playoffs)
    if 'team_points' not in df.columns:
        df['team_points'] = 0.0
    else:
        df['team_points'] = pd.to_numeric(df['team_points'], errors='coerce').astype(float)
    # Convert to proper types
    df['win'] = pd.to_numeric(df['win'], errors='coerce').fillna(0).astype(int)
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce').fillna(0).astype(int)
    # Sort by manager, year, week
    df = df.sort_values(['manager', 'year', 'week']).reset_index(drop=True)
    # Create manager_year key if not exists
    if 'manager_year' not in df.columns:
        # use a separator to avoid accidental collisions (e.g., manager='12', year='34' -> '12_34')
        df['manager_year'] = df['manager'].astype(str) + '_' + df['year'].astype(str)
    # Exclude consolation games from season totals
    non_consol = (df['is_consolation'].fillna(0).astype(int) != 1)

    # Use franchise_id for grouping if available (handles multi-team managers correctly)
    # Falls back to manager name for backwards compatibility
    group_col = 'franchise_id' if 'franchise_id' in df.columns and df['franchise_id'].notna().any() else 'manager'

    # CUMULATIVE ALL-TIME TOTALS (includes everything)
    df['cumulative_wins'] = df.groupby(group_col)['win'].cumsum().astype("Int64")
    df['cumulative_losses'] = df.groupby(group_col)['loss'].cumsum().astype("Int64")
    # SEASON TOTALS (excludes consolation)
    df['wins_to_date'] = (
        (df['win'] * non_consol.astype(int))
        .groupby([df['year'], df['manager_year']]).cumsum()
    ).astype("Int64")
    df['losses_to_date'] = (
        (df['loss'] * non_consol.astype(int))
        .groupby([df['year'], df['manager_year']]).cumsum()
    ).astype("Int64")
    df['points_scored_to_date'] = (
        (df['team_points'].fillna(0.0) * non_consol.astype(int))
        .groupby([df['year'], df['manager_year']]).cumsum()
    ).astype(float)
    # WIN/LOSS STREAKS (current active streaks)
    df['win_streak'] = 0
    df['loss_streak'] = 0
    for entity in df[group_col].unique():
        mask = df[group_col] == entity
        entity_df = df[mask].copy()
        # Calculate streaks
        streaks_win = []
        streaks_loss = []
        current_win_streak = 0
        current_loss_streak = 0
        for _, row in entity_df.iterrows():
            if row['win'] == 1:
                current_win_streak += 1
                current_loss_streak = 0
            elif row['loss'] == 1:
                current_loss_streak += 1
                current_win_streak = 0
            else:
                # Tie
                current_win_streak = 0
                current_loss_streak = 0
            streaks_win.append(current_win_streak)
            streaks_loss.append(current_loss_streak)
        df.loc[mask, 'win_streak'] = streaks_win
        df.loc[mask, 'loss_streak'] = streaks_loss
    df['win_streak'] = df['win_streak'].astype("Int64")
    df['loss_streak'] = df['loss_streak'].astype("Int64")
    # PLAYOFF SEED TO DATE (weekly ranking based on current record)
    # Rank teams within each year-week based on wins_to_date (desc) then points_scored_to_date (desc)
    def _rank_seed_within_week(week_df: pd.DataFrame) -> pd.Series:
        """Rank teams by wins desc, then points desc"""
        wins = pd.to_numeric(week_df['wins_to_date'], errors='coerce').fillna(-1).to_numpy()
        pts = pd.to_numeric(week_df['points_scored_to_date'], errors='coerce').fillna(-1.0).to_numpy()
        # lexsort sorts by last key first, so we want (-wins) to be last (primary sort)
        order = np.lexsort((-pts, -wins))
        ranks = np.empty(len(week_df), dtype=int)
        ranks[order] = np.arange(1, len(week_df) + 1)
        return pd.Series(ranks, index=week_df.index, dtype="Int64")
    df['playoff_seed_to_date'] = (
        df.groupby(['year', 'week'], group_keys=False)
        .apply(_rank_seed_within_week, include_groups=False)
    ).astype("Int64")

    # Calculate final_playoff_seed (playoff seed at end of regular season)
    # This MUST be calculated BEFORE playoff detection runs
    # Drop any existing final_playoff_seed columns (from previous runs) to avoid merge conflicts
    cols_to_drop = [c for c in df.columns if c.startswith('final_playoff_seed')]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # CRITICAL FIX: Load playoff_start_week from league settings (not from is_playoffs flags)
    # This breaks the circular dependency: we need final_playoff_seed to SET is_playoffs,
    # but we were trying to DETECT playoffs to calculate final_playoff_seed!

    from core.data_normalization import find_league_settings_directory
    import json

    # Find league settings directory using provided data_directory (if available)
    # Otherwise fall back to search logic
    if data_directory:
        from pathlib import Path
        settings_path = find_league_settings_directory(data_directory=Path(data_directory), df=df)
    else:
        settings_path = find_league_settings_directory(df=df)

    playoff_start_by_year = {}

    if settings_path and settings_path.exists():
        print(f"  Loading playoff start weeks from: {settings_path}")
        # Load settings for each year
        for yr in df['year'].dropna().unique():
            yr_int = int(yr)
            settings_files = list(settings_path.glob(f"league_settings_{yr_int}_*.json"))
            if settings_files:
                try:
                    with open(settings_files[0], 'r') as f:
                        settings = json.load(f)
                        metadata = settings.get('metadata', settings)
                        playoff_start = int(metadata.get('playoff_start_week', -1))
                        if playoff_start > 0:
                            playoff_start_by_year[yr_int] = playoff_start
                            print(f"    Year {yr_int}: playoff_start_week = {playoff_start}")
                        else:
                            print(f"    [WARN] No playoff_start_week in settings for {yr_int}, will try to infer from data or fallback.")
                except Exception as e:
                    print(f"    [WARN] Failed to load settings for {yr_int}: {e}")
            else:
                print(f"    [WARN] No settings file found for {yr_int}, will try to infer from data or fallback.")
    else:
        print(f"  [WARN] League settings directory not found, will try to infer playoff_start_week from data or fallback.")

    # For any year not found, try to infer from data (row-count collapse)
    # Also track bye_teams to detect actual playoff start
    bye_teams_by_year = {}
    num_playoff_teams_by_year = {}

    if settings_path and settings_path.exists():
        for yr in df['year'].dropna().unique():
            yr_int = int(yr)
            settings_files = list(settings_path.glob(f"league_settings_{yr_int}_*.json"))
            if settings_files:
                try:
                    with open(settings_files[0], 'r') as f:
                        settings = json.load(f)
                        metadata = settings.get('metadata', settings)
                        bye_teams_by_year[yr_int] = int(metadata.get('bye_teams', 0))
                        num_playoff_teams_by_year[yr_int] = int(metadata.get('num_playoff_teams', metadata.get('playoff_teams', 6)))
                except Exception:
                    pass

    for yr in df['year'].dropna().unique():
        yr_int = int(yr)
        if yr_int not in playoff_start_by_year:
            league_ids = df[df['year'] == yr_int]['league_id'].unique()
            if len(league_ids) > 0:
                league_id = league_ids[0]
            else:
                league_id = None
            # Try to infer from data: look for drop in number of matchups per week
            g = (df[df['year'] == yr_int].groupby(['week']).size().reset_index(name='rows'))
            if not g.empty:
                modal = g['rows'].mode().iloc[0] if not g['rows'].mode().empty else g['rows'].max()
                playoff_weeks = g[g['rows'] < modal]['week']
                if not playoff_weeks.empty:
                    inferred_start = int(playoff_weeks.min())
                    playoff_start_by_year[yr_int] = inferred_start
                    print(f"    [INFO] Inferred playoff_start_week for {yr_int} from data: {inferred_start}")
                    continue
            # Fallback: use year-based default
            if yr_int <= 2020:
                playoff_start_by_year[yr_int] = 14
                print(f"    [WARN] Fallback: using default playoff_start_week=14 for {yr_int} (pre-2021)")
            else:
                playoff_start_by_year[yr_int] = 15
                print(f"    [WARN] Fallback: using default playoff_start_week=15 for {yr_int} (2021 or later)")

    # SETTINGS-DRIVEN: Trust playoff_start_week from settings
    # playoff_start_week means: the first week ANY playoff games occur
    #
    # Example (6-team playoffs, 2 byes):
    #   - playoff_start_week: 15 → Week 15 is first playoff week
    #   - Week 15: Seeds 3-6 play (non-bye teams)
    #   - Week 16: Seeds 1-2 play winners (bye teams enter in round 2)
    #
    # We calculate final_playoff_seed from the last regular season week,
    # which is the week BEFORE playoff_start_week.

    actual_playoff_start_by_year = {}
    for yr_int, playoff_start_config in playoff_start_by_year.items():
        bye_teams = bye_teams_by_year.get(yr_int, 0)

        # playoff_start_week from settings is ALWAYS the first week playoffs occur
        # Bye teams don't play in first playoff week, but it's still the playoff start
        actual_playoff_start = playoff_start_config

        if bye_teams > 0:
            print(f"    [SETTINGS] Year {yr_int}: playoff_start_week={playoff_start_config}, "
                  f"bye_teams={bye_teams} -> Playoffs start week {actual_playoff_start}, bye teams enter week {actual_playoff_start + 1}")
        else:
            print(f"    [SETTINGS] Year {yr_int}: playoff_start_week={playoff_start_config} (no byes)")

        actual_playoff_start_by_year[yr_int] = actual_playoff_start

    # Use actual playoff starts for final_playoff_seed calculation
    playoff_start_by_year = actual_playoff_start_by_year

    # Now calculate final_playoff_seed as the seed from last week before playoffs
    final_seeds_list = []
    for yr, playoff_start in playoff_start_by_year.items():
        yr_df = df[df['year'] == yr]

        # Find the last week before playoffs that has data for ALL managers
        # Sometimes not all managers play in the week immediately before playoffs
        weeks_before_playoffs = sorted([w for w in yr_df['week'].unique() if w < playoff_start], reverse=True)

        last_week_data = None
        last_reg_week = None
        total_managers = yr_df['manager'].nunique()

        for check_week in weeks_before_playoffs:
            week_data = yr_df[yr_df['week'] == check_week]
            if week_data['manager'].nunique() == total_managers:
                # This week has all managers - use it
                last_week_data = week_data
                last_reg_week = check_week
                break

        # Fallback: use the last available seed for each manager
        if last_week_data is None and weeks_before_playoffs:
            last_reg_week = weeks_before_playoffs[0]
            last_week_data = yr_df[yr_df['week'] == last_reg_week]

        if last_week_data is not None and not last_week_data.empty:
            # Get all unique managers from the entire year, not just last week
            for manager in yr_df['manager'].unique():
                # Try to get seed from last_week_data
                mgr_data = last_week_data[last_week_data['manager'] == manager]
                if not mgr_data.empty:
                    seed = mgr_data['playoff_seed_to_date'].iloc[0]
                else:
                    # Manager not in last week - get their most recent seed before playoffs
                    mgr_before_playoffs = yr_df[(yr_df['manager'] == manager) & (yr_df['week'] < playoff_start)]
                    if not mgr_before_playoffs.empty:
                        seed = mgr_before_playoffs.iloc[-1]['playoff_seed_to_date']
                    else:
                        seed = None

                if seed is not None and not pd.isna(seed):
                    final_seeds_list.append({
                        'manager': manager,
                        'year': yr,
                        'final_playoff_seed': seed
                    })

            print(f"    Calculated final_playoff_seed for {len([s for s in final_seeds_list if s['year'] == yr])} managers in {yr}")
        else:
            print(f"    [WARN] No data found for weeks before playoff start {playoff_start} in year {yr}")

    if final_seeds_list:
        final_seeds_df = pd.DataFrame(final_seeds_list)
        # Merge back to all weeks for this manager-year
        df = df.merge(
            final_seeds_df,
            on=['manager', 'year'],
            how='left'
        )
        print(f"  [OK] Added final_playoff_seed for {len(final_seeds_list)} manager-years")
    else:
        df['final_playoff_seed'] = pd.NA
        print(f"  [WARN] No final_playoff_seed data calculated!")

    print(f"  Cumulative wins: max={df['cumulative_wins'].max()}")
    print(f"  Longest win streak: {df['win_streak'].max()}")
    print(f"  Longest loss streak: {df['loss_streak'].max()}")

    return df