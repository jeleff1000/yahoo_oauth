"""
Replacement Level Calculator - Percentile-Based Hybrid Approach

Calculates WEEK-BY-WEEK replacement levels using a HYBRID approach:
- League weeks: Uses ACTUAL roster data (rostered players who played)
- Non-league weeks: Uses PERCENTILE-BASED cutoffs (adapts to player pool size)

Strategy:
1. League weeks (have Yahoo roster data):
   - Count players who were BOTH rostered AND played per position
     * Rostered = has yahoo_player_id (from Yahoo roster data)
     * Played = has NFL_player_id (from NFLverse - only players who played a game)
   - Calculate roster counts per position (e.g., 52 RBs rostered & played)
   - Excludes players rostered but didn't play (injured, bye, etc.)

2. Calculate PERCENTILE cutoffs from league weeks:
   - For each position, calculate: rostered_and_played / total_who_played
   - Example: 55 RBs rostered & played out of 100 who played = 55th percentile
   - Average this percentile across all league weeks

3. Non-league weeks (NFL playoffs, pre-league history):
   - Apply the PERCENTILE to the actual player pool that week
   - Example: If RB percentile is 55% and 60 RBs played in playoffs:
     → Replacement is 56th percentile = 33rd RB (60 × 0.55)
   - This adapts to varying player pool sizes (fewer teams play in playoffs)

4. Replacement baseline includes ALL performances (started, benched, unrostered)
   - This captures the true performance distribution at each position

This ensures:
- Accurate week-by-week replacement levels during league season
- Fair replacement levels for non-league weeks (adapts to smaller player pools)
- Replacement varies week-to-week (52nd RB one week, 55th another)
"""

import pandas as pd
from typing import Dict, Tuple
import json
from pathlib import Path


def calculate_replacement_percentiles(
    player_df: pd.DataFrame,
    num_teams: int
) -> Dict[str, float]:
    """
    Calculate replacement percentile cutoffs from league weeks.

    For each position, calculates what percentage of the total player pool
    was above replacement during league weeks. This percentile is then used
    for non-league weeks to adapt to varying player pool sizes.

    Example: If 55 out of 100 RBs were rostered & played during league weeks,
    the percentile is 0.55 (55th percentile). In NFL playoffs when only 60 RBs
    play, we'd use the 56th percentile = 33rd RB.

    Args:
        player_df: Player DataFrame with fantasy_points, yahoo_player_id, NFL_player_id
        num_teams: Number of teams in the league

    Returns:
        Dict mapping position to percentile cutoff (0.0 to 1.0)
        Example: {'QB': 0.55, 'RB': 0.52, 'WR': 0.58, ...}
    """
    # Filter to league weeks only (have Yahoo roster data)
    league_weeks = player_df[
        player_df['yahoo_player_id'].notna() &
        player_df['NFL_player_id'].notna()
    ].copy()

    if league_weeks.empty:
        print(f"[percentiles] Warning: No league weeks found (no yahoo_player_id + NFL_player_id).")
        print(f"[percentiles] Cannot calculate percentiles - will use league settings for all weeks.")
        return {}

    # Get unique year/week/position combinations from league weeks
    league_week_positions = league_weeks.groupby(['year', 'week', 'position']).size().reset_index(name='rostered_and_played')

    # For each year/week/position, also count total players who played (from full dataset)
    all_players = player_df[player_df['fantasy_points'] > 0].copy()
    total_players = all_players.groupby(['year', 'week', 'position']).size().reset_index(name='total_who_played')

    # Merge to get both counts for each league week
    comparison = league_week_positions.merge(
        total_players,
        on=['year', 'week', 'position'],
        how='left'
    )

    # Calculate percentile for each week
    comparison['percentile'] = comparison['rostered_and_played'] / comparison['total_who_played']

    # Average percentile across all league weeks for each position
    avg_percentiles = comparison.groupby('position')['percentile'].mean().to_dict()

    # Log results
    print(f"\n[percentiles] Calculated replacement percentiles from league weeks:")
    print(f"  (Ratio: rostered & played / total who played)")
    for position, pct in sorted(avg_percentiles.items()):
        print(f"  {position}: {pct:.2%} percentile cutoff")

    return avg_percentiles


def calculate_actual_roster_spots(
    player_df: pd.DataFrame,
    num_teams: int,
    manager_col: str = 'manager'
) -> Dict[str, float]:
    """
    Calculate actual roster spots per position from the data.

    This analyzes all weeks to find how many players at each position
    are rostered per team (on average).

    CRITICAL: Only counts players who were BOTH rostered AND actually played.
    - Rostered = has yahoo_player_id (from Yahoo roster data)
    - Actually played = has NFL_player_id (from NFLverse data)

    This excludes:
    - Players rostered but didn't play (injured, bye, etc.)
    - Players who played but weren't rostered

    Args:
        player_df: Player DataFrame (includes manager="Unrostered" for FAs)
        num_teams: Number of teams in the league
        manager_col: Column name for manager

    Returns:
        Dict mapping position to avg roster spots per team
        Example: {'QB': 1.6, 'RB': 5.2, 'WR': 5.8, ...}
    """
    # Filter to players who were BOTH rostered (have yahoo_player_id) AND played (have NFL_player_id)
    # This gives us the actual number of roster spots used by players who contributed points
    rostered_and_played = player_df[
        player_df['yahoo_player_id'].notna() &
        player_df['NFL_player_id'].notna()
    ].copy()

    if rostered_and_played.empty:
        print(f"[roster_spots] Warning: No players found with both yahoo_player_id and NFL_player_id.")
        print(f"[roster_spots] This is expected for non-league weeks (NFL playoffs, pre-league history).")
        print(f"[roster_spots] Will use percentile-based cutoffs for these weeks.")
        return {}  # Return empty dict - caller will use percentiles as fallback

    # Count rostered players who played per position per week
    roster_counts = rostered_and_played.groupby(['year', 'week', 'position']).size().reset_index(name='count')

    # Average across all weeks
    avg_roster_spots = roster_counts.groupby('position')['count'].mean() / num_teams

    # Convert to dict
    roster_spots = avg_roster_spots.to_dict()

    # Log results
    print(f"\n[roster_spots] Calculated from rostered players who actually played:")
    print(f"  (Filter: yahoo_player_id AND NFL_player_id both present)")
    for position, spots in sorted(roster_spots.items()):
        total_rostered = spots * num_teams
        print(f"  {position}: {spots:.1f} per team ({total_rostered:.0f} total rostered & played)")

    return roster_spots


def load_roster_structure(
    league_settings_path: Path,
    player_df: pd.DataFrame = None
) -> Dict:
    """
    Load roster structure from league_settings and actual data.

    Args:
        league_settings_path: Path to league_settings JSON
        player_df: Optional player DataFrame to calculate actual roster spots

    Returns:
        Dict with:
        - num_teams: int
        - starters: dict (from league settings)
        - roster_spots: dict (from actual data if provided)
    """
    with open(league_settings_path, 'r') as f:
        settings = json.load(f)

    num_teams = int(settings.get('metadata', {}).get('num_teams', 10))
    roster_positions = settings.get('roster_positions', [])

    # Calculate starters from settings (for reference)
    starters = {}
    flex_positions = []

    for pos_config in roster_positions:
        position = pos_config.get('position', '')
        count = int(pos_config.get('count', 0))

        if position in ['BN', 'IR', 'IR+']:
            continue

        if '/' in position:
            flex_positions.append((position, count))
        else:
            starters[position] = starters.get(position, 0) + count

    # Distribute flex
    for flex_name, flex_count in flex_positions:
        eligible = flex_name.split('/')
        flex_share = flex_count / len(eligible)
        for pos in eligible:
            starters[pos] = starters.get(pos, 0) + flex_share

    # Calculate actual roster spots from data (for league weeks)
    roster_spots = {}
    roster_percentiles = {}

    if player_df is not None:
        try:
            # Calculate roster spots (for league weeks with Yahoo data)
            roster_spots = calculate_actual_roster_spots(player_df, num_teams)

            # Calculate percentile cutoffs (for non-league weeks)
            roster_percentiles = calculate_replacement_percentiles(player_df, num_teams)

            # If no roster data found, that's OK - we'll use percentiles for non-league weeks
            if not roster_spots:
                print(f"[roster_spots] No rostered-and-played data found")
                print(f"[roster_spots] Will use percentile-based cutoffs for all weeks")
        except Exception as e:
            print(f"[roster_structure] Warning: Could not calculate from data: {e}")
            print(f"[roster_structure] Will use league settings (starters) as fallback")
            roster_spots = {}
            roster_percentiles = {}

    return {
        'num_teams': num_teams,
        'starters': starters,
        'roster_spots': roster_spots,
        'roster_percentiles': roster_percentiles
    }


def calculate_weekly_replacement(
    player_df: pd.DataFrame,
    year: int,
    week: int,
    roster_structure: Dict
) -> pd.DataFrame:
    """
    Calculate replacement level for a single week (week-by-week calculation).

    Includes ALL performances (started, benched, unrostered) to capture
    true performance distribution.

    PERCENTILE-BASED HYBRID APPROACH:
    - League weeks (has Yahoo roster data):
      Use actual rostered-and-played counts (varies week-to-week)
      Example: Week 5 has 52 RBs rostered & played → 52nd RB is replacement

    - Non-league weeks (no Yahoo data - NFL playoffs, pre-league history):
      Apply percentile cutoff to actual player pool that week
      Example: If RB percentile is 55% and 60 RBs played → 33rd RB (60 × 0.55)
      This adapts to varying player pool sizes

    Args:
        player_df: Full player DataFrame
        year: Season year
        week: Week number
        roster_structure: Dict with num_teams, roster_spots, roster_percentiles, and starters

    Returns:
        DataFrame with year, week, position, replacement_ppg, n_pos, roster_count_source
    """
    # Filter to this week - ALL players who scored points
    week_data = player_df[
        (player_df['year'] == year) &
        (player_df['week'] == week) &
        (player_df['fantasy_points'] > 0)
    ].copy()

    if week_data.empty:
        return pd.DataFrame(columns=[
            'year', 'week', 'position', 'replacement_ppg', 'n_pos', 'roster_count_source'
        ])

    num_teams = roster_structure['num_teams']
    roster_percentiles = roster_structure.get('roster_percentiles', {})

    # Check if this week has any Yahoo roster data (rostered players who played)
    has_yahoo_data = (
        week_data['yahoo_player_id'].notna() &
        week_data['NFL_player_id'].notna()
    ).any()

    results = []

    # HYBRID APPROACH:
    # - League weeks: Use actual rostered-and-played counts
    # - Non-league weeks: Use percentile-based cutoffs (adapts to player pool size)

    if has_yahoo_data:
        # ===== LEAGUE WEEK: Use actual rostered-and-played counts =====
        roster_counts_source = roster_structure['roster_spots']
        source_label = "rostered & played"

        for position, roster_count in roster_counts_source.items():
            # Filter to this position
            pos_data = week_data[week_data['position'] == position].copy()

            if pos_data.empty:
                continue

            # Sort by fantasy_points descending (ALL players)
            pos_data = pos_data.sort_values('fantasy_points', ascending=False)

            # Calculate N = teams × roster_spots
            n_pos = num_teams * roster_count
            n_pos_int = int(n_pos)

            # Get replacement level as average of ranks N and N+1
            if len(pos_data) > n_pos_int:
                rank_n = pos_data.iloc[n_pos_int]['fantasy_points']
                if len(pos_data) > n_pos_int + 1:
                    rank_n_plus_1 = pos_data.iloc[n_pos_int + 1]['fantasy_points']
                    replacement_ppg = (rank_n + rank_n_plus_1) / 2
                else:
                    replacement_ppg = rank_n
            elif len(pos_data) > 0:
                replacement_ppg = pos_data.iloc[-1]['fantasy_points']
            else:
                replacement_ppg = 0.0

            results.append({
                'year': year,
                'week': week,
                'position': position,
                'replacement_ppg': replacement_ppg,
                'n_pos': n_pos,
                'roster_count_source': source_label
            })

    else:
        # ===== NON-LEAGUE WEEK: Use percentile-based cutoffs =====
        source_label = "percentile cutoff"

        # If we have percentiles, use them; otherwise fall back to starters
        if roster_percentiles:
            for position in roster_percentiles.keys():
                # Filter to this position
                pos_data = week_data[week_data['position'] == position].copy()

                if pos_data.empty:
                    continue

                # Sort by fantasy_points descending (ALL players)
                pos_data = pos_data.sort_values('fantasy_points', ascending=False)

                # Calculate N based on percentile of actual player pool THIS WEEK
                total_players_this_week = len(pos_data)
                percentile = roster_percentiles[position]
                n_pos = total_players_this_week * percentile
                n_pos_int = int(n_pos)

                # Get replacement level as average of ranks N and N+1
                if len(pos_data) > n_pos_int:
                    rank_n = pos_data.iloc[n_pos_int]['fantasy_points']
                    if len(pos_data) > n_pos_int + 1:
                        rank_n_plus_1 = pos_data.iloc[n_pos_int + 1]['fantasy_points']
                        replacement_ppg = (rank_n + rank_n_plus_1) / 2
                    else:
                        replacement_ppg = rank_n
                elif len(pos_data) > 0:
                    replacement_ppg = pos_data.iloc[-1]['fantasy_points']
                else:
                    replacement_ppg = 0.0

                results.append({
                    'year': year,
                    'week': week,
                    'position': position,
                    'replacement_ppg': replacement_ppg,
                    'n_pos': n_pos,
                    'roster_count_source': source_label
                })
        else:
            # FALLBACK: Use league settings (starters) if no percentiles available
            source_label = "league settings (starters)"
            roster_counts_source = roster_structure['starters']

            for position, roster_count in roster_counts_source.items():
                # Filter to this position
                pos_data = week_data[week_data['position'] == position].copy()

                if pos_data.empty:
                    continue

                # Sort by fantasy_points descending (ALL players)
                pos_data = pos_data.sort_values('fantasy_points', ascending=False)

                # Calculate N = teams × roster_spots
                n_pos = num_teams * roster_count
                n_pos_int = int(n_pos)

                # Get replacement level as average of ranks N and N+1
                if len(pos_data) > n_pos_int:
                    rank_n = pos_data.iloc[n_pos_int]['fantasy_points']
                    if len(pos_data) > n_pos_int + 1:
                        rank_n_plus_1 = pos_data.iloc[n_pos_int + 1]['fantasy_points']
                        replacement_ppg = (rank_n + rank_n_plus_1) / 2
                    else:
                        replacement_ppg = rank_n
                elif len(pos_data) > 0:
                    replacement_ppg = pos_data.iloc[-1]['fantasy_points']
                else:
                    replacement_ppg = 0.0

                results.append({
                    'year': year,
                    'week': week,
                    'position': position,
                    'replacement_ppg': replacement_ppg,
                    'n_pos': n_pos,
                    'roster_count_source': source_label
                })

    return pd.DataFrame(results)


def calculate_season_replacement(weekly_replacement_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate season-average replacement levels from weekly data."""
    season_replacement = weekly_replacement_df.groupby(['year', 'position']).agg({
        'replacement_ppg': 'mean',
        'n_pos': 'mean'  # Average across weeks (can vary week-to-week)
    }).reset_index()

    season_replacement.rename(columns={'replacement_ppg': 'replacement_ppg_season'}, inplace=True)

    return season_replacement


def calculate_all_replacements(
    player_df: pd.DataFrame,
    league_settings_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate all replacement levels (weekly and season) from actual data.

    NO DEFAULTS - everything calculated from actual roster composition.

    Args:
        player_df: Player DataFrame with fantasy_points and manager columns
        league_settings_path: Path to league_settings JSON

    Returns:
        Tuple of (weekly_replacement_df, season_replacement_df)
    """
    # Load roster structure (requires player_df to calculate roster spots)
    roster_structure = load_roster_structure(league_settings_path, player_df)

    # Get all year/week combinations
    year_weeks = player_df[['year', 'week']].drop_duplicates()

    # Calculate weekly replacement for each year/week
    weekly_results = []
    for _, row in year_weeks.iterrows():
        year, week = int(row['year']), int(row['week'])
        weekly_rep = calculate_weekly_replacement(
            player_df, year, week, roster_structure
        )
        if not weekly_rep.empty:
            weekly_results.append(weekly_rep)

    if not weekly_results:
        raise ValueError("No weekly replacement levels calculated - check player data")

    weekly_replacement_df = pd.concat(weekly_results, ignore_index=True)

    # Calculate season averages
    season_replacement_df = calculate_season_replacement(weekly_replacement_df)

    # Log summary
    print(f"\n[replacement] Calculated replacement levels (hybrid approach):")
    print(f"  Performance pool: All players (started, benched, unrostered)")
    print(f"  League weeks: Rostered & played counts (yahoo_player_id AND NFL_player_id)")
    print(f"  Non-league weeks: Percentile-based cutoffs (adapts to player pool size)")

    # Show breakdown of which weeks used which method
    if not weekly_replacement_df.empty and 'roster_count_source' in weekly_replacement_df.columns:
        source_counts = weekly_replacement_df.groupby('roster_count_source').agg({
            'week': 'nunique',
            'year': lambda x: f"{x.min()}-{x.max()}"
        })
        print(f"\n  Week breakdown:")
        for source, row in source_counts.iterrows():
            week_count = row['week']
            year_range = row['year']
            print(f"    {source}: {week_count} unique weeks ({year_range})")

    # Show example
    if not season_replacement_df.empty:
        example_year = season_replacement_df['year'].max()
        example_season = season_replacement_df[season_replacement_df['year'] == example_year]

        # Get roster percentiles from roster_structure
        roster_percentiles = roster_structure.get('roster_percentiles', {})

        print(f"\n  {example_year} season averages (league weeks):")
        for _, row in example_season.iterrows():
            position = row['position']
            repl_ppg = row['replacement_ppg_season']
            n_pos = row['n_pos']
            num_teams = roster_structure['num_teams']
            per_team = n_pos / num_teams

            # Show percentile if available
            percentile_info = ""
            if position in roster_percentiles:
                pct = roster_percentiles[position]
                percentile_info = f" [{pct:.1%} percentile]"

            print(f"    {position}: {repl_ppg:.2f} PPG (cutoff: {n_pos:.0f} = {num_teams} teams × {per_team:.1f}{percentile_info})")

    return weekly_replacement_df, season_replacement_df
