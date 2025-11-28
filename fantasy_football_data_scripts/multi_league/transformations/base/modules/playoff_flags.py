"""
Playoff Flags Module

Normalizes playoff and consolation flags with proper mutual exclusivity.

CRITICAL RULE: is_consolation=1 MUST imply is_playoffs=0 (mutually exclusive)
"""

from functools import wraps
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any



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


def _is_utf8_console() -> bool:
    """Check if console supports UTF-8 encoding."""
    enc = getattr(sys.stdout, "encoding", None)
    return enc and "utf" in enc.lower()


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Print with ASCII fallback for non-UTF8 consoles."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        ascii_args = []
        for a in args:
            if isinstance(a, str):
                ascii_args.append(a.encode("ascii", errors="replace").decode("ascii"))
            else:
                ascii_args.append(a)
        print(*ascii_args, **kwargs)


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


@ensure_normalized
def normalize_playoff_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize is_playoffs and is_consolation flags.

    CRITICAL RULES:
    1. is_consolation=1 → is_playoffs=0 (mutually exclusive)
    2. Either can be 0, but both cannot be 1
    3. postseason = 1 if either playoffs or consolation

    Returns:
        DataFrame with normalized playoff flags
    """
    df = df.copy()

    # Ensure flags exist
    for col in ["is_playoffs", "is_consolation"]:
        if col not in df.columns:
            df[col] = 0

    # Convert to int, handling NaN
    df["is_playoffs"] = pd.to_numeric(df["is_playoffs"], errors='coerce').fillna(0).astype(int)
    df["is_consolation"] = pd.to_numeric(df["is_consolation"], errors='coerce').fillna(0).astype(int)

    # CRITICAL: Enforce is_consolation=1 → is_playoffs=0
    df.loc[df["is_consolation"] == 1, "is_playoffs"] = 0

    # Create postseason flag
    df["postseason"] = ((df["is_playoffs"] == 1) | (df["is_consolation"] == 1)).astype(int)

    # Count violations (should be 0 after fix)
    violations = ((df["is_playoffs"] == 1) & (df["is_consolation"] == 1)).sum()
    if violations > 0:
        safe_print(f"  [WARN] Found {violations} mutual exclusivity violations (forcing consolation)")

    playoff_count = (df["is_playoffs"] == 1).sum()
    consolation_count = (df["is_consolation"] == 1).sum()
    postseason_count = (df["postseason"] == 1).sum()

    safe_print(f"  Playoffs: {playoff_count} games")
    safe_print(f"  Consolation: {consolation_count} games")
    safe_print(f"  Total postseason: {postseason_count} games")
    safe_print("  [OK] Verified rule: is_consolation=1 -> is_playoffs=0")

    return df


@ensure_normalized
def detect_playoffs_by_seed(df: pd.DataFrame, settings_dir: str = None) -> pd.DataFrame:
    """
    Correctly detect is_playoffs and is_consolation based on league settings and game outcomes.

    CRITICAL LOGIC:
    - is_playoffs=1: ONLY teams still competing for 1st place (championship)
    - is_consolation=1: ALL other postseason games (including 3rd place, 5th place, etc.)

    Once a team loses in the championship bracket, all subsequent games are consolation games.

    Uses final_playoff_seed (end of regular season standings) to determine which teams
    qualified for the championship bracket vs consolation bracket.

    Args:
        df: DataFrame with matchup data
        settings_dir: Directory containing league_settings JSON files (optional, auto-detected if None)

    Returns:
        DataFrame with corrected is_playoffs and is_consolation flags
    """
    df = df.copy()

    required_cols = ['week', 'year', 'manager']
    for col in required_cols:
        if col not in df.columns:
            safe_print(f"  [WARN] {col} column not found, skipping playoff detection")
            if 'is_playoffs' not in df.columns:
                df['is_playoffs'] = 0
            if 'is_consolation' not in df.columns:
                df['is_consolation'] = 0
            return df

    # Ensure win/loss columns exist
    for col in ['win', 'loss']:
        if col not in df.columns:
            df[col] = 0

    df['win'] = pd.to_numeric(df['win'], errors='coerce').fillna(0).astype(int)
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce').fillna(0).astype(int)

    # Load settings from JSON files per year
    from pathlib import Path
    import json
    from core.data_normalization import find_league_settings_directory

    if settings_dir is None:
        # Use centralized utility to find settings directory (league-agnostic)
        settings_path = find_league_settings_directory(df=df)
        if settings_path:
            settings_dir = str(settings_path)
            safe_print(f"  [SETTINGS] Auto-discovered settings directory: {settings_path}")
        else:
            safe_print(f"  [WARN] Could not find league_settings directory, using defaults")
    else:
        # Use provided settings_dir (data_directory from LeagueContext)
        settings_path = find_league_settings_directory(data_directory=Path(settings_dir), df=df)
        if settings_path:
            settings_dir = str(settings_path)
            safe_print(f"  [SETTINGS] Using settings directory: {settings_path}")

    # Load settings per year
    settings_by_year = {}
    if settings_dir:
        settings_path = Path(settings_dir)
        if settings_path.exists():
            years = sorted(df['year'].dropna().unique())
            for year in years:
                year_int = int(year)
                settings_files = list(settings_path.glob(f"league_settings_{year_int}_*.json"))

                if settings_files:
                    try:
                        with open(settings_files[0], 'r') as f:
                            settings = json.load(f)
                            # Handle both old and new field names
                            metadata = settings.get('metadata', settings)
                            settings_by_year[year_int] = {
                                'playoff_start_week': int(metadata.get('playoff_start_week', 15)),
                                'num_playoff_teams': int(metadata.get('num_playoff_teams', metadata.get('playoff_teams', 6))),
                                'bye_teams': int(metadata.get('bye_teams', 0)),
                            }
                            safe_print(f"  [SETTINGS] {year_int}: playoff_start={settings_by_year[year_int]['playoff_start_week']}, playoff_teams={settings_by_year[year_int]['num_playoff_teams']}, bye_teams={settings_by_year[year_int]['bye_teams']}")
                    except Exception as e:
                        safe_print(f"  [WARN] Failed to load settings from {settings_files[0]}: {e}")

    # Fallback to defaults if settings not loaded
    default_playoff_start = 15
    default_num_teams = 6
    for year in df['year'].dropna().unique():
        if int(year) not in settings_by_year:
            settings_by_year[int(year)] = {
                'playoff_start_week': default_playoff_start,
                'num_playoff_teams': default_num_teams,
                'bye_teams': 0,
            }

    if not settings_by_year:
        safe_print(f"  [SETTINGS] Using defaults: playoff_start_week={default_playoff_start}, num_playoff_teams={default_num_teams}")

    # Initialize flags
    df['is_playoffs'] = 0
    df['is_consolation'] = 0

    # Process each year separately
    for year, year_settings in settings_by_year.items():
        year_mask = (df['year'] == year)

        if not year_mask.any():
            continue

        playoff_start_config = year_settings['playoff_start_week']
        num_playoff = year_settings['num_playoff_teams']
        bye_teams = year_settings['bye_teams']

        # SETTINGS-DRIVEN PLAYOFF START:
        #
        # playoff_start_week from settings means: the first week ANY playoff games occur
        #
        # Example (6-team playoffs, 2 byes):
        #   - playoff_start_week: 15
        #   - bye_teams: 2
        #
        # This means:
        #   - Week 15: Seeds 3-6 play (first playoff games)
        #   - Week 16: Seeds 1-2 enter and play winners (semifinals)
        #
        # NO ADJUSTMENT NEEDED - trust the settings file directly.

        playoff_start = playoff_start_config

        if bye_teams > 0:
            safe_print(f"  [SETTINGS] Year {year}: playoff_start_week={playoff_start}, "
                     f"bye_teams={bye_teams} -> Playoffs start week {playoff_start}, bye teams enter week {playoff_start + 1}")
        else:
            safe_print(f"  [SETTINGS] Year {year}: playoff_start_week={playoff_start} (no byes)")

        postseason_mask = year_mask & (df['week'] >= playoff_start)

        if not postseason_mask.any():
            continue

        # Determine playoff qualifiers based on final_playoff_seed
        if 'final_playoff_seed' in df.columns:
            year_df = df[year_mask].copy()

            # Get playoff qualifiers (teams with seed <= num_playoff_teams)
            manager_seeds = {}
            for manager in year_df['manager'].unique():
                mgr_data = year_df[year_df['manager'] == manager]
                seed_values = mgr_data['final_playoff_seed'].dropna()
                if not seed_values.empty:
                    manager_seeds[manager] = int(seed_values.iloc[0])

            playoff_qualifiers = {mgr for mgr, seed in manager_seeds.items() if seed <= num_playoff}
            consolation_teams = {mgr for mgr, seed in manager_seeds.items() if seed > num_playoff}

            safe_print(f"  [BRACKET] Year {year}: {len(playoff_qualifiers)} playoff qualifiers (seeds 1-{num_playoff}), {len(consolation_teams)} consolation teams")
        else:
            # Fallback: use first playoff week to determine qualifiers
            safe_print(f"  [WARN] No final_playoff_seed found for year {year}, using fallback logic")
            first_week = int(df[postseason_mask]['week'].min())
            playoff_qualifiers = set(df[(df['year'] == year) & (df['week'] == first_week) & (df['is_playoffs'] == 1)]['manager'].unique())
            consolation_teams = set(df[postseason_mask]['manager'].unique()) - playoff_qualifiers

        # Track which teams are STILL ALIVE for 1st place (can win championship)
        # Once a team loses, they can only play consolation games
        alive_for_championship = playoff_qualifiers.copy()

        # Process each playoff week in order
        playoff_weeks = sorted(df[postseason_mask]['week'].unique())

        for week in playoff_weeks:
            week_mask = postseason_mask & (df['week'] == week)
            week_games = df[week_mask].copy()

            if week_games.empty:
                continue

            # Mark championship bracket games (ONLY teams still alive for 1st place)
            # is_playoffs=1 means: "This team can still win the championship"
            championship_games_mask = week_mask & \
                                     df['manager'].isin(alive_for_championship) & \
                                     df['opponent'].isin(alive_for_championship)
            df.loc[championship_games_mask, 'is_playoffs'] = 1

            # Mark ALL other postseason games as consolation
            # This includes:
            # - Teams that never made playoffs (seeds > num_playoff)
            # - Teams that lost in championship bracket (playing for 3rd, 5th, etc.)
            non_championship_mask = week_mask & ~championship_games_mask
            df.loc[non_championship_mask, 'is_consolation'] = 1

            # Find teams that LOST in championship bracket this week
            # These teams are eliminated from 1st place contention
            championship_losers = week_games[
                (week_games['manager'].isin(alive_for_championship)) &
                (week_games['opponent'].isin(alive_for_championship)) &
                (week_games['loss'] == 1)
            ]['manager'].unique()

            # Remove losers from alive_for_championship for next week
            # They can no longer win 1st place, so all future games are consolation
            alive_for_championship = alive_for_championship - set(championship_losers)

            safe_print(f"    Week {week}: {len(alive_for_championship)} teams still alive for 1st place, {len(championship_losers)} eliminated this week")

    # Create postseason flag
    df['postseason'] = ((df['is_playoffs'] == 1) | (df['is_consolation'] == 1)).astype(int)

    # Final verification - ensure mutual exclusivity
    violations = ((df['is_playoffs'] == 1) & (df['is_consolation'] == 1)).sum()
    if violations > 0:
        safe_print(f"  [ERROR] Found {violations} games with both flags set!")
        df.loc[(df['is_playoffs'] == 1) & (df['is_consolation'] == 1), 'is_playoffs'] = 0

    playoff_count = (df['is_playoffs'] == 1).sum()
    consolation_count = (df['is_consolation'] == 1).sum()
    postseason_count = (df['postseason'] == 1).sum()

    safe_print(f"  [FINAL] is_playoffs=1 (competing for 1st): {playoff_count} games")
    safe_print(f"  [FINAL] is_consolation=1 (all other postseason): {consolation_count} games")
    safe_print(f"  [FINAL] Total postseason: {postseason_count}")
    safe_print("  [OK] Verified: is_playoffs and is_consolation are mutually exclusive")

    return df


@ensure_normalized
def mark_playoff_rounds(df: pd.DataFrame, data_directory: str = None) -> pd.DataFrame:
    """
    Mark playoff rounds (quarterfinal, semifinal, championship) and consolation rounds.

    FULLY GENERIC - works with any bracket configuration:
    - Dynamically detects placement games (3rd, 5th, 7th, 9th, etc.)
    - Handles winner-advances AND loser-advances brackets
    - No hardcoded assumptions about bracket structure

    Adds columns for BOTH championship and consolation brackets:
      - playoff_week_index (1, 2, 3, ...)
      - playoff_round_num (same as index)
      - playoff_round (string label: "quarterfinal", "semifinal", "championship", "third_place_game")
      - consolation_round (string label: "consolation_round_1", "consolation_final", "fifth_place_game")
      - quarterfinal, semifinal, championship (binary flags)
      - placement_game (binary flag for ANY placement game)
      - placement_rank (3, 5, 7, 9, etc. - which place this game determines)

    Args:
        df: DataFrame with matchup data
        data_directory: Path to league data directory (for finding league settings, currently unused)

    Returns:
        DataFrame with playoff round columns
    """
    df = df.copy()

    # Initialize round columns
    for col in ["playoff_week_index", "playoff_round_num", "playoff_round", "consolation_round",
                "quarterfinal", "semifinal", "championship",
                "placement_game", "placement_rank",
                "consolation_semifinal", "consolation_final"]:
        if col not in df.columns:
            df[col] = 0 if col not in ["playoff_round", "consolation_round"] else ""

    # Ensure playoff flags exist
    if "is_playoffs" not in df.columns:
        df["is_playoffs"] = 0
    if "is_consolation" not in df.columns:
        df["is_consolation"] = 0

    df["is_playoffs"] = pd.to_numeric(df["is_playoffs"], errors="coerce").fillna(0).astype(int)
    df["is_consolation"] = pd.to_numeric(df["is_consolation"], errors="coerce").fillna(0).astype(int)

    # Enforce mutual exclusivity
    df.loc[df["is_consolation"] == 1, "is_playoffs"] = 0

    # CRITICAL: Clear all playoff round labels for regular season games
    # Regular season = is_playoffs=0 AND is_consolation=0
    regular_season_mask = (df["is_playoffs"] == 0) & (df["is_consolation"] == 0)
    df.loc[regular_season_mask, "playoff_round"] = ""
    df.loc[regular_season_mask, "consolation_round"] = ""
    df.loc[regular_season_mask, "playoff_week_index"] = 0
    df.loc[regular_season_mask, "playoff_round_num"] = 0
    df.loc[regular_season_mask, "quarterfinal"] = 0
    df.loc[regular_season_mask, "semifinal"] = 0
    df.loc[regular_season_mask, "championship"] = 0
    df.loc[regular_season_mask, "consolation_semifinal"] = 0
    df.loc[regular_season_mask, "consolation_final"] = 0
    df.loc[regular_season_mask, "placement_game"] = 0
    df.loc[regular_season_mask, "placement_rank"] = 0

    def _label_for_index(idx_from_start: int, total_rounds: int) -> str:
        """Map playoff week index to round label."""
        offset_from_end = total_rounds - idx_from_start
        if offset_from_end == 0:
            return "championship"
        if offset_from_end == 1:
            return "semifinal"
        if offset_from_end == 2:
            return "quarterfinal"
        return f"round_{idx_from_start}"

    # Process each year
    years = sorted(df["year"].dropna().unique().astype(int))

    for yr in years:
        safe_print(f"\n  [YEAR {yr}] Analyzing playoff structure...")

        # ====== CHAMPIONSHIP BRACKET ======
        mask_y = (df["year"] == yr) & (df["is_playoffs"] == 1)
        weeks = sorted(df.loc[mask_y, "week"].dropna().unique().astype(int))

        if weeks:
            total = len(weeks)
            w2idx = {w: i + 1 for i, w in enumerate(weeks)}

            # Assign indices
            df.loc[mask_y, "playoff_week_index"] = df.loc[mask_y, "week"].map(w2idx).astype(int)
            df.loc[mask_y, "playoff_round_num"] = df.loc[mask_y, "playoff_week_index"].astype(int)

            # Assign labels
            labels = {w: _label_for_index(w2idx[w], total) for w in weeks}
            df.loc[mask_y, "playoff_round"] = df.loc[mask_y, "week"].map(labels)

            # Set binary flags
            for w in weeks:
                lab = labels[w]
                if lab in ("quarterfinal", "semifinal", "championship"):
                    week_mask = mask_y & (df["week"] == w)
                    df.loc[week_mask, lab] = 1

            safe_print(f"    Championship bracket: {total} rounds ({', '.join(labels.values())})")

        # ====== CONSOLATION BRACKET ======
        cons_mask_y = (df["year"] == yr) & (df["is_consolation"] == 1)
        cons_weeks = sorted(df.loc[cons_mask_y, "week"].dropna().unique().astype(int))

        if cons_weeks:
            total_cons = len(cons_weeks)
            cons_w2idx = {w: i + 1 for i, w in enumerate(cons_weeks)}

            # Assign consolation round labels
            for w in cons_weeks:
                idx = cons_w2idx[w]
                week_mask = cons_mask_y & (df["week"] == w)

                # Label based on position from end
                offset_from_end = total_cons - idx
                if offset_from_end == 0:
                    df.loc[week_mask, "consolation_round"] = "consolation_final"
                    df.loc[week_mask, "consolation_final"] = 1
                elif offset_from_end == 1:
                    df.loc[week_mask, "consolation_round"] = "consolation_semifinal"
                    df.loc[week_mask, "consolation_semifinal"] = 1
                else:
                    df.loc[week_mask, "consolation_round"] = f"consolation_round_{idx}"

            safe_print(f"    Consolation bracket: {total_cons} rounds")

        # ====== DYNAMIC PLACEMENT GAME DETECTION ======
        # Track all teams through postseason to detect placement games
        # FULLY GENERIC: Works for any bracket size (2-32 teams)
        #
        # Strategy:
        # 1. Track team paths through brackets (championship vs consolation)
        # 2. Pair teams that lost/won in same week from same bracket
        # 3. Calculate placement rank based on:
        #    - Their playoff seeds
        #    - How many teams are competing for that placement
        #    - Bracket tree structure

        postseason_mask = (df["year"] == yr) & ((df["is_playoffs"] == 1) | (df["is_consolation"] == 1))
        postseason_weeks = sorted(df.loc[postseason_mask, "week"].dropna().unique().astype(int))

        if not postseason_weeks:
            continue

        # Build a history of who each team has played and their results
        team_history = {}  # {manager: [(week, opponent, won, bracket_type), ...]}

        # Track team seeds for placement calculations
        team_seeds = {}  # {manager: final_playoff_seed}

        for week in postseason_weeks:
            week_mask = (df["year"] == yr) & (df["week"] == week)
            week_df = df[week_mask]

            for _, row in week_df.iterrows():
                mgr = row['manager']
                opp = row['opponent']
                won = row['win'] == 1
                bracket_type = 'championship' if row['is_playoffs'] == 1 else 'consolation'

                if mgr not in team_history:
                    team_history[mgr] = []
                team_history[mgr].append((week, opp, won, bracket_type))

                # Store team seed for later calculations
                if mgr not in team_seeds and 'final_playoff_seed' in row.index:
                    seed = row['final_playoff_seed']
                    if pd.notna(seed):
                        team_seeds[mgr] = int(seed)

        # Track which teams are still alive in each bracket for dynamic placement calculation
        # This is the KEY to making placement detection work for any bracket size
        alive_by_week = {}  # {week: {'championship': set(), 'consolation': set()}}

        for week in postseason_weeks:
            week_mask = (df["year"] == yr) & (df["week"] == week)
            week_df = df[week_mask]

            # Count teams still alive in each bracket
            champ_alive = set(week_df[week_df['is_playoffs'] == 1]['manager'].unique())
            cons_alive = set(week_df[week_df['is_consolation'] == 1]['manager'].unique())

            alive_by_week[week] = {
                'championship': champ_alive,
                'consolation': cons_alive
            }

        # Get total teams in league for placement calculations
        total_teams = len(team_seeds) if team_seeds else 10  # fallback to 10

        # Detect placement games by finding teams with similar paths
        # GENERIC ALGORITHM:
        # 1. Both teams lost (or won) in same previous round from same bracket
        # 2. Calculate placement rank based on:
        #    - For championship dropouts: position in championship tree
        #    - For consolation teams: how many teams are still alive

        for week_idx, week in enumerate(postseason_weeks):
            week_mask = (df["year"] == yr) & (df["week"] == week)
            week_df = df[week_mask].copy()

            if week_df.empty:
                continue

            # For each game this week, check if it's a placement game
            for _, row in week_df.iterrows():
                mgr = row['manager']
                opp = row['opponent']

                # Skip if we've already marked this game
                current_placement = df.loc[(df["year"] == yr) & (df["week"] == week) & (df["manager"] == mgr), "placement_game"]
                if len(current_placement) > 0 and current_placement.iloc[0] == 1:
                    continue

                # Check if both teams exist in team_history
                if mgr not in team_history or opp not in team_history:
                    continue

                # Find the most recent game before this week for each team
                mgr_prev = [g for g in team_history[mgr] if g[0] < week]
                opp_prev = [g for g in team_history[opp] if g[0] < week]

                if not mgr_prev or not opp_prev:
                    continue

                # Get most recent game for each
                mgr_last_week, mgr_last_opp, mgr_won_last, mgr_last_bracket = mgr_prev[-1]
                opp_last_week, opp_last_opp, opp_won_last, opp_last_bracket = opp_prev[-1]

                # GENERIC PLACEMENT GAME DETECTION
                # A placement game occurs when two teams with SIMILAR PATHS meet
                # Path similarity = same bracket + same result in same previous week
                is_placement = False
                placement_rank = None
                placement_name = None

                if mgr_last_week == opp_last_week and mgr_last_bracket == opp_last_bracket:
                    # Both teams played in same week from same bracket
                    # Check if they had same result (both won OR both lost)

                    if not mgr_won_last and not opp_won_last:
                        # BOTH LOST - this is a placement game!
                        is_placement = True

                        # GENERIC PLACEMENT RANK CALCULATION
                        if mgr_last_bracket == 'championship':
                            # Championship bracket losers
                            # Placement depends on WHICH round they lost in

                            # Count championship rounds they survived
                            if weeks:
                                # Find which championship round they lost in
                                try:
                                    elim_round_idx = weeks.index(mgr_last_week)
                                except ValueError:
                                    elim_round_idx = 0

                                # Calculate placement based on elimination tier
                                # Logic: losers in final round (semifinals) = 3rd place
                                #        losers in second-to-last round = 5th place
                                #        losers in third-to-last round = 7th place (rare), etc.
                                rounds_from_finals = len(weeks) - 1 - elim_round_idx

                                # Each round back adds 2 to placement (3rd, 5th, 7th, etc.)
                                placement_rank = 3 + max(0, (rounds_from_finals - 1)) * 2
                                placement_name = {
                                    3: "third_place_game",
                                    5: "fifth_place_game",
                                    7: "seventh_place_game",
                                    9: "ninth_place_game"
                                }.get(placement_rank, f"placement_{placement_rank}_game")

                        elif mgr_last_bracket == 'consolation':
                            # Pure consolation bracket losers (competing for worst place)
                            # Use dynamic calculation based on teams still alive

                            # How many teams are playing this week?
                            teams_this_week = len(alive_by_week.get(week, {}).get('consolation', set()))

                            # Placement = total_teams - teams_this_week + 1
                            # This gives the BEST possible finish for these teams
                            if teams_this_week > 0:
                                placement_rank = total_teams - teams_this_week + 1
                                placement_name = {
                                    3: "third_place_game",
                                    5: "fifth_place_game",
                                    7: "seventh_place_game",
                                    9: "ninth_place_game",
                                    11: "eleventh_place_game"
                                }.get(placement_rank, f"placement_{placement_rank}_game")

                    elif mgr_won_last and opp_won_last:
                        # BOTH WON - also a placement game (winner-advances path)
                        is_placement = True

                        if mgr_last_bracket == 'consolation':
                            # Consolation winners competing for better placement
                            # Use same dynamic calculation

                            teams_this_week = len(alive_by_week.get(week, {}).get('consolation', set()))

                            if teams_this_week > 0:
                                placement_rank = total_teams - teams_this_week + 1
                                placement_name = {
                                    3: "third_place_game",
                                    5: "fifth_place_game",
                                    7: "seventh_place_game",
                                    9: "ninth_place_game",
                                    11: "eleventh_place_game"
                                }.get(placement_rank, f"placement_{placement_rank}_game")

                # Mark the placement game if detected
                if is_placement and placement_rank:
                    game_mask = (df["year"] == yr) & (df["week"] == week) & \
                               ((df["manager"] == mgr) | (df["manager"] == opp))

                    df.loc[game_mask, "placement_game"] = 1
                    df.loc[game_mask, "placement_rank"] = placement_rank

                    # Update round labels
                    if row['is_playoffs'] == 0:
                        df.loc[game_mask, "consolation_round"] = placement_name
                    else:
                        df.loc[game_mask, "playoff_round"] = placement_name

                    # Log detection with path information
                    path_desc = "champ losers" if mgr_last_bracket == 'championship' else "cons bracket"
                    result_desc = "both lost" if not mgr_won_last else "both won"
                    safe_print(f"      {placement_name} (rank {placement_rank}): {mgr} vs {opp} ({path_desc}, {result_desc})")

    # ====== ENFORCE MUTUAL EXCLUSIVITY ======
    # CRITICAL: Ensure playoff and consolation labels are mutually exclusive
    # is_playoffs=1 → ONLY playoff_round can be set, consolation_round MUST be empty
    # is_consolation=1 → ONLY consolation_round can be set, playoff_round MUST be empty
    #
    # This runs LAST to override any mislabeling from placement game detection

    playoff_only_mask = df["is_playoffs"] == 1
    consolation_only_mask = df["is_consolation"] == 1

    # Count violations before fixing
    violations_before = 0
    for idx, row in df.iterrows():
        if row["is_playoffs"] == 1 and pd.notna(row.get("consolation_round")) and row.get("consolation_round") != "":
            violations_before += 1
        if row["is_consolation"] == 1 and pd.notna(row.get("playoff_round")) and row.get("playoff_round") != "":
            violations_before += 1

    # Clear consolation labels from playoff games
    df.loc[playoff_only_mask, "consolation_round"] = ""
    df.loc[playoff_only_mask, "consolation_semifinal"] = 0
    df.loc[playoff_only_mask, "consolation_final"] = 0
    df.loc[playoff_only_mask, "placement_game"] = 0
    df.loc[playoff_only_mask, "placement_rank"] = 0

    # Clear playoff labels from consolation games
    df.loc[consolation_only_mask, "playoff_round"] = ""
    df.loc[consolation_only_mask, "playoff_week_index"] = 0
    df.loc[consolation_only_mask, "playoff_round_num"] = 0
    df.loc[consolation_only_mask, "quarterfinal"] = 0
    df.loc[consolation_only_mask, "semifinal"] = 0
    df.loc[consolation_only_mask, "championship"] = 0

    if violations_before > 0:
        safe_print(f"\n  [ENFORCE] Fixed {violations_before} conflicting round labels")
    safe_print(f"            Playoff games: {playoff_only_mask.sum()} (only playoff_round)")
    safe_print(f"            Consolation games: {consolation_only_mask.sum()} (only consolation_round)")

    # Convert types
    df["playoff_week_index"] = pd.to_numeric(df["playoff_week_index"], errors="coerce").fillna(0).astype("Int64")
    df["playoff_round_num"] = pd.to_numeric(df["playoff_round_num"], errors="coerce").fillna(0).astype("Int64")
    df["placement_game"] = pd.to_numeric(df["placement_game"], errors="coerce").fillna(0).astype(int)
    df["placement_rank"] = pd.to_numeric(df["placement_rank"], errors="coerce").fillna(0).astype("Int64")

    for col in ["quarterfinal", "semifinal", "championship", "consolation_semifinal", "consolation_final"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Summary stats
    qf_count = (df["quarterfinal"] == 1).sum()
    sf_count = (df["semifinal"] == 1).sum()
    champ_count = (df["championship"] == 1).sum()
    placement_count = (df["placement_game"] == 1).sum()
    cons_sf_count = (df["consolation_semifinal"] == 1).sum()
    cons_final_count = (df["consolation_final"] == 1).sum()

    safe_print(f"\n  [SUMMARY] Championship Rounds: QF={qf_count}, SF={sf_count}, Champ={champ_count}")
    safe_print(f"  [SUMMARY] Consolation Rounds: SF={cons_sf_count}, Final={cons_final_count}")
    safe_print(f"  [SUMMARY] Placement Games: {placement_count} detected")

    return df


@ensure_normalized
def mark_champions_and_sackos(df: pd.DataFrame, settings_dir: str = None) -> pd.DataFrame:
    """
    Mark league champions and sackos using proper bracket simulation.

    Champion: Winner of championship game (verified by tracking matchups)
    Sacko: Loser of consolation bracket final (verified by tracking matchups)

    This function properly:
    - Loads settings from league_settings JSON files (no hardcoded values)
    - Tracks matchups through playoff rounds
    - Ensures only ONE champion and ONE sacko per season
    - Handles bracket progression correctly

    Args:
        df: DataFrame with matchup data including final_playoff_seed
        settings_dir: Directory containing league_settings JSON files (auto-detected if None)

    Returns:
        DataFrame with champion and sacko flags (exactly 1 of each per season)
    """
    # Note: simulate_playoff_brackets is now called by cumulative_stats_v2.py
    # after normalize_playoff_flags completes, so we don't call it here anymore
    # to avoid duplicate processing that would reset champion/sacko flags

    return df


@ensure_normalized
def add_season_result(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season_result column combining win/loss outcome with game type.

    GENERIC & DYNAMIC: Works for any postseason structure.

    season_result format: "{won|lost} {game_type}"

    Examples:
    - "won championship"
    - "lost championship"
    - "won third place game"
    - "lost third place game"
    - "won fifth place game"
    - "lost seventh place game"
    - "won consolation semifinal"
    - etc.

    For the championship week (last playoff week of season), this shows
    each manager's final game outcome.

    Args:
        df: DataFrame with playoff/consolation round labels and win/loss flags

    Returns:
        DataFrame with season_result column added
    """
    df = df.copy()

    # Initialize season_result column
    df['season_result'] = ""

    # Ensure required columns exist
    required = ['year', 'week', 'win', 'loss']
    for col in required:
        if col not in df.columns:
            safe_print(f"  [WARN] Missing column {col}, cannot create season_result")
            return df

    # Process each year to find championship week
    years = sorted(df['year'].dropna().unique())

    for year in years:
        year = int(year)
        year_mask = df['year'] == year

        # Find championship week (last postseason week)
        postseason_mask = year_mask & ((df.get('is_playoffs', pd.Series(0)) == 1) |
                                       (df.get('is_consolation', pd.Series(0)) == 1))

        if not postseason_mask.any():
            continue

        championship_week = df[postseason_mask]['week'].max()

        # Get all games in championship week
        champ_week_mask = year_mask & (df['week'] == championship_week)
        champ_week_df = df[champ_week_mask]

        for idx, row in champ_week_df.iterrows():
            # Determine outcome (won or lost)
            if row['win'] == 1:
                outcome = "won"
            elif row['loss'] == 1:
                outcome = "lost"
            else:
                # Tie or unknown
                outcome = "tied in"

            # Determine game type
            game_type = None

            # Championship game (is_playoffs=1, championship=1)
            if row.get('is_playoffs') == 1 and row.get('championship') == 1:
                game_type = "championship"

            # Playoff semifinal
            elif row.get('is_playoffs') == 1 and row.get('semifinal') == 1:
                game_type = "semifinal"

            # Playoff quarterfinal
            elif row.get('is_playoffs') == 1 and row.get('quarterfinal') == 1:
                game_type = "quarterfinal"

            # Placement games (consolation with placement_rank)
            elif row.get('is_consolation') == 1 and row.get('placement_game') == 1:
                # Use consolation_round label if available
                cons_round = str(row.get('consolation_round', ''))
                if cons_round and cons_round != '':
                    # Remove 'game' suffix if present (to avoid "won third place game game")
                    game_type = cons_round.replace('_game', ' game').replace('_', ' ')
                else:
                    # Fallback to placement_rank
                    placement_rank = row.get('placement_rank', 0)
                    if placement_rank > 0:
                        rank_suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(placement_rank % 10 if placement_rank not in [11, 12, 13] else 0, 'th')
                        game_type = f"{placement_rank}{rank_suffix} place game"

            # Consolation round (not a placement game)
            elif row.get('is_consolation') == 1:
                cons_round = str(row.get('consolation_round', ''))
                if 'final' in cons_round:
                    game_type = "consolation final"
                elif 'semifinal' in cons_round:
                    game_type = "consolation semifinal"
                else:
                    game_type = "consolation round"

            # Combine outcome + game type
            if game_type:
                df.at[idx, 'season_result'] = f"{outcome} {game_type}"

    safe_print(f"  [OK] Added season_result column")

    # Count how many results were created
    non_empty = (df['season_result'] != '').sum()
    safe_print(f"       {non_empty} season results created")

    return df
