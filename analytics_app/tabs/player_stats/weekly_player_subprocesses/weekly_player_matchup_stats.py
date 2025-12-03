import streamlit as st
import pandas as pd

class CombinedMatchupStatsViewer:
    def __init__(self, player_data):
        """
        Fantasy matchup viewer - shows how players performed in their fantasy matchups.
        """
        if player_data is not None and not player_data.empty:
            # CRITICAL: Remove duplicate columns from source data immediately
            if player_data.columns.duplicated().any():
                player_data = player_data.loc[:, ~player_data.columns.duplicated(keep='first')]
        self.player_data = player_data.copy() if player_data is not None else pd.DataFrame()

    def display(self, prefix):
        df = self.player_data.copy()

        # Filter out unrostered players - matchup stats only show rostered players
        if 'manager' in df.columns:
            df = df[df['manager'] != 'Unrostered'].copy()
            df = df[df['manager'].notna()].copy()
            df = df[df['manager'] != ''].copy()

        # Ensure correct dtypes
        numeric_cols = ['year', 'week', 'points', 'team_points', 'opponent_points']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'year' in df.columns:
            df['year'] = df['year'].astype('Int64')
        if 'week' in df.columns:
            df['week'] = df['week'].astype('Int64')

        # Create Round column from playoff_round and consolation_round
        # Priority: playoff_round > consolation_round
        df['Round'] = None
        if 'playoff_round' in df.columns:
            df['Round'] = df['playoff_round'].fillna('')
        if 'consolation_round' in df.columns:
            # If Round is empty/null, use consolation_round
            mask = (df['Round'].isna()) | (df['Round'] == '')
            df.loc[mask, 'Round'] = df.loc[mask, 'consolation_round'].fillna('')

        # Clean up Round column - replace empty strings with None
        df['Round'] = df['Round'].replace('', None)

        # Rename columns to be more user-friendly
        column_renames = {
            'player': 'Player',
            'nfl_team': 'Team',
            'week': 'Week',
            'year': 'Year',
            'manager': 'Manager',
            'opponent': 'Opponent',
            'points': 'Pts',
            'nfl_position': 'Pos',
            'fantasy_position': 'Slot',
            'is_started': 'Started',
            'optimal_player': 'Optimal',
            'league_wide_optimal_player': 'League Optimal',
            'win': 'Won',
            'loss': 'Lost',
            'team_points': 'My Team',
            'opponent_points': 'Opp Team',
            'is_playoffs': 'Playoffs',
            'is_consolation': 'Consolation',
            'champion': 'Champion',
            'sacko': 'Sacko',
        }

        df = df.rename(columns=column_renames)

        # Convert 'Started' to boolean if it exists (is_started = 1 means started)
        if 'Started' in df.columns:
            df['Started'] = df['Started'] == 1
        elif 'Slot' in df.columns:
            # Fallback: derive from Slot if Started column is missing
            df['Started'] = ~df['Slot'].isin(['BN', 'IR', '', None, pd.NA])
        else:
            df['Started'] = True

        # Convert 'Optimal' and 'League Optimal' to boolean if they exist
        if 'Optimal' in df.columns:
            df['Optimal'] = df['Optimal'] == 1
        if 'League Optimal' in df.columns:
            df['League Optimal'] = df['League Optimal'] == 1

        # Convert win/loss and playoff flags to boolean
        for col in ['Won', 'Lost', 'Playoffs', 'Consolation', 'Champion', 'Sacko']:
            if col in df.columns:
                df[col] = df[col] == 1

        # Calculate margin
        if 'My Team' in df.columns and 'Opp Team' in df.columns:
            df['Margin'] = (df['My Team'] - df['Opp Team']).round(1)

        # Rank players within RB and WR positions based on points
        if 'Slot' in df.columns:
            df['position_rank'] = df.groupby(
                ['Manager', 'Week', 'Year', 'Slot']
            )['Pts'].rank(ascending=False, method='first').astype('Int64')

            # Append the rank only for RB and WR positions
            df['Slot'] = df.apply(
                lambda row: f"{row['Slot']}{row['position_rank']}"
                if row['Slot'] in ['RB', 'WR'] else row['Slot'],
                axis=1
            )

            # Define the unique order for fantasy_positions
            unique_position_order = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'W/R/T', 'K', 'DEF', 'BN', 'IR']
            df['Slot'] = pd.Categorical(
                df['Slot'], categories=unique_position_order, ordered=True
            )

        # Column order: Player, Week, Year, then matchup info
        display_cols = ['Player', 'Week', 'Year', 'Pts', 'Pos', 'Team', 'Slot', 'Started', 'Optimal', 'League Optimal',
                       'Manager', 'Opponent', 'My Team', 'Opp Team', 'Margin', 'Won',
                       'Round', 'Playoffs', 'Consolation', 'Champion', 'Sacko']

        # Filter to only existing columns
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].copy()

        # Ensure proper data types for display
        if 'Year' in display_df.columns:
            display_df['Year'] = display_df['Year'].astype(int)
        if 'Week' in display_df.columns:
            display_df['Week'] = display_df['Week'].astype(int)
        if 'Pts' in display_df.columns:
            display_df['Pts'] = display_df['Pts'].round(2)
        if 'My Team' in display_df.columns:
            display_df['My Team'] = display_df['My Team'].round(2)
        if 'Opp Team' in display_df.columns:
            display_df['Opp Team'] = display_df['Opp Team'].round(2)

        # Sort by year (DESC), week (DESC), manager (ASC), slot order, points (DESC)
        # This groups by most recent seasons first, then by week, then by manager
        sort_cols = []
        sort_order = []

        if 'Year' in display_df.columns:
            sort_cols.append('Year')
            sort_order.append(False)  # DESC - most recent year first
        if 'Week' in display_df.columns:
            sort_cols.append('Week')
            sort_order.append(False)  # DESC - most recent week first
        if 'Manager' in display_df.columns:
            sort_cols.append('Manager')
            sort_order.append(True)  # ASC - alphabetical
        if 'Slot' in display_df.columns:
            sort_cols.append('Slot')
            sort_order.append(True)  # ASC - lineup order (QB, RB1, RB2, etc.)
        if 'Pts' in display_df.columns:
            sort_cols.append('Pts')
            sort_order.append(False)  # DESC - highest points first

        if sort_cols:
            display_df = display_df.sort_values(
                by=sort_cols,
                ascending=sort_order
            ).reset_index(drop=True)

        # CRITICAL: Remove any duplicate column names to prevent React error #185
        if display_df.columns.duplicated().any():
            display_df = display_df.loc[:, ~display_df.columns.duplicated(keep='first')]

        # Configure column display
        column_config = {}

        # Year - display without thousand separator
        if 'Year' in display_df.columns:
            column_config['Year'] = st.column_config.NumberColumn(
                'Year',
                format='%d',  # Integer format without separator
                help='Season year'
            )

        # Week - integer format
        if 'Week' in display_df.columns:
            column_config['Week'] = st.column_config.NumberColumn(
                'Week',
                format='%d',
                help='Week number'
            )

        # Points - decimal format
        if 'Pts' in display_df.columns:
            column_config['Pts'] = st.column_config.NumberColumn(
                'Pts',
                format='%.2f',
                help='Fantasy points'
            )

        # Team points - decimal format
        if 'My Team' in display_df.columns:
            column_config['My Team'] = st.column_config.NumberColumn(
                'My Team',
                format='%.2f',
                help='Team total points'
            )

        if 'Opp Team' in display_df.columns:
            column_config['Opp Team'] = st.column_config.NumberColumn(
                'Opp Team',
                format='%.2f',
                help='Opponent total points'
            )

        if 'Margin' in display_df.columns:
            column_config['Margin'] = st.column_config.NumberColumn(
                'Margin',
                format='%.1f',
                help='Point differential'
            )

        # Boolean columns as checkboxes
        boolean_cols = ['Started', 'Optimal', 'League Optimal', 'Won', 'Playoffs', 'Consolation', 'Champion', 'Sacko']
        for col in boolean_cols:
            if col in display_df.columns:
                column_config[col] = st.column_config.CheckboxColumn(
                    col,
                    help=f'{col}',
                    disabled=True
                )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config=column_config
        )
