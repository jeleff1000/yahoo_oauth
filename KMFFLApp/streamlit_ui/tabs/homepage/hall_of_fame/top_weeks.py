import streamlit as st

class TopWeeksViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self):
        st.header("Top Weeks")

        # Default view: include regular season + playoffs (always). No checkboxes required.

        if self.df is not None:
            # Create columns for dropdowns with narrower widths
            col3, col4 = st.columns([1, 1])
            with col3:
                years = ["All Years"] + sorted(self.df['year'].unique().tolist())
                selected_year = st.selectbox("Select Year", years, index=0, key="top_weeks_year")
            with col4:
                managers = ["All Managers"] + sorted(self.df['manager'].unique().tolist())
                selected_manager = st.selectbox("Select Manager", managers, index=0, key="top_weeks_manager")

            # Filter data: include both regular season and playoffs, always exclude consolation games
            filtered_df = self.df[self.df['is_consolation'] == 0].copy()

            # Filter data based on selected year and Manager
            if selected_year != "All Years":
                filtered_df = filtered_df[filtered_df['year'] == selected_year]
            if selected_manager != "All Managers":
                filtered_df = filtered_df[filtered_df['manager'] == selected_manager]

            if not filtered_df.empty:
                # De-duplicate matchups so each game appears only once.
                # Create a normalized match key (unordered pair of manager and opponent lowered)
                if 'opponent' in filtered_df.columns:
                    dfv = filtered_df.copy()
                    dfv['match_key'] = dfv.apply(lambda r: '|'.join(sorted([str(r['manager']).lower(), str(r.get('opponent','')).lower()])), axis=1)
                    # Keep the row with the higher team_points for each year/week/match_key
                    dfv = dfv.sort_values(by=['year', 'week', 'match_key', 'team_points'], ascending=[True, True, True, False])
                    dfv = dfv.drop_duplicates(subset=['year', 'week', 'match_key'], keep='first')
                else:
                    dfv = filtered_df.copy()

                # Group by year, week, Manager and sum team_points
                grouped_df = dfv.groupby(['year', 'week', 'manager']).agg({'team_points': 'sum'}).reset_index()

                # Sort by team_points descending
                sorted_df = grouped_df.sort_values(by='team_points', ascending=False)

                # Display top 10
                top_10_df = sorted_df.head(10)

                # Format the display
                display_df = top_10_df[['year', 'week', 'manager', 'team_points']].copy()
                display_df.columns = ['Year', 'Week', 'Manager', 'Points']
                # Ensure Year is a string to avoid comma formatting in Streamlit tables
                display_df['Year'] = display_df['Year'].astype(str)

                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No data available for the selected filters.")
        else:
            st.info("No data available.")
