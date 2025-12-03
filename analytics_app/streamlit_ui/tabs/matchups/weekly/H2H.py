import streamlit as st
import pandas as pd

class H2HViewer:
    def __init__(self, filtered_data, matchup_data):
        self.filtered_data = filtered_data
        self.matchup_data = matchup_data

    @st.fragment
    def display(self, prefix):
        # Ensure 'season' and 'year' columns have the same data type
        self.filtered_data['season'] = self.filtered_data['season'].astype(int)
        self.matchup_data['year'] = self.matchup_data['year'].astype(int)

        # Merge player data with matchup data
        merged_data = pd.merge(
            self.filtered_data,
            self.matchup_data,
            left_on=['owner', 'week', 'season'],
            right_on=['Manager', 'week', 'year'],
            how='inner'
        )

        # Add 'started' column
        merged_data['started'] = ~merged_data['fantasy position'].isin(['BN', 'IR'])

        # Convert 'optimal_player' to boolean
        merged_data['optimal_player'] = merged_data['optimal_player'] == 1

        if 'win' in merged_data.columns:
            merged_data['win'] = merged_data['win'] == 1
            merged_data['is_playoffs_check'] = merged_data['is_playoffs'] == 1

            # Rank players within RB and WR positions based on points
            merged_data['position_rank'] = merged_data.groupby(
                ['Manager', 'week', 'year', 'fantasy position']
            )['points'].rank(ascending=False, method='first').astype(int)

            # Append the rank only for RB and WR positions
            merged_data['fantasy position'] = merged_data.apply(
                lambda row: f"{row['fantasy position']}{row['position_rank']}"
                if row['fantasy position'] in ['RB', 'WR'] else row['fantasy position'],
                axis=1
            )

            # Define the unique order for fantasy positions
            unique_position_order = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'W/R/T', 'K', 'DEF', 'BN', 'IR']
            merged_data['fantasy position'] = pd.Categorical(
                merged_data['fantasy position'], categories=unique_position_order, ordered=True
            )

            # Create display DataFrame
            display_df = merged_data[
                ['player', 'points', 'Manager', 'week', 'year', 'fantasy position', 'opponent', 'team_points',
                 'opponent_points', 'win', 'is_playoffs_check', 'started', 'optimal_player']
            ]
            display_df['year'] = display_df['year'].astype(str)
            display_df = display_df.sort_values(by=['year', 'week', 'fantasy position']).reset_index(drop=True)

            # Display the DataFrame in Streamlit
            st.dataframe(display_df, hide_index=True)
        else:
            st.write("The required column 'win' is not available in the data.")

        # Display the entire unfiltered filtered_data DataFrame
        st.subheader("Full Filtered Data")
        st.dataframe(self.filtered_data, hide_index=True)

        # Display the entire unfiltered matchup_data DataFrame
        st.subheader("Full Matchup Data")
        st.dataframe(self.matchup_data, hide_index=True)