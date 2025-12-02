import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..shared.modern_styles import apply_modern_styles


class KeeperDataViewer:
    def __init__(self, keeper_data):
        self.keeper_data = keeper_data

    @st.fragment
    def display(self):
        apply_modern_styles()

        if self.keeper_data is None or self.keeper_data.empty:
            st.warning("No keeper data available.")
            return

        df = self.keeper_data.copy()
        df['year'] = df['year'].astype(str)
        df['manager'] = df['manager'].astype(str)

        # Summary metrics - FIXED calculations
        col1, col2, col3, col4 = st.columns(4)

        # Get unique year/manager combinations to calculate expected keeper slots
        year_manager_combos = df.groupby(['year', 'manager']).size().reset_index(name='roster_size')
        num_years = df['year'].nunique()
        num_managers_per_year = year_manager_combos.groupby('year')['manager'].nunique().mean()

        # Total keeper slots = 2 keepers per manager per year
        total_keeper_slots = int(num_years * num_managers_per_year * 2)

        # Count actually kept (unique player/year combinations where kept_next_year = True)
        total_kept = df[df['kept_next_year'] == True].groupby(['player', 'year']).size().shape[0]

        # Average keeper price for players who were actually kept
        avg_keeper_price = df[df['kept_next_year'] == True]['keeper_price'].mean()

        with col1:
            st.metric("Total Keeper Slots", total_keeper_slots,
                     help=f"~{int(num_managers_per_year)} managers × {num_years} years × 2 keepers")
        with col2:
            st.metric("Actually Kept", total_kept)
        with col3:
            st.metric("Avg Keeper Cost", f"${avg_keeper_price:.1f}" if pd.notna(avg_keeper_price) else "N/A",
                     help="Average cost of players who were actually kept")
        with col4:
            keep_rate = (total_kept / total_keeper_slots * 100) if total_keeper_slots > 0 else 0
            st.metric("Keep Rate", f"{keep_rate:.1f}%",
                     help="Percentage of available keeper slots that were used")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Keeper Explorer", "📈 Analytics", "💎 Best Keepers"])

        with tab1:
            self._display_keeper_explorer(df)

        with tab2:
            self._display_analytics(df)

        with tab3:
            self._display_best_keepers(df)

    def _get_color_scale(self, value, min_val, max_val, reverse=False):
        """Get color for a value on a scale from red to green"""
        if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            return '#808080'

        # Normalize to 0-1
        normalized = (value - min_val) / (max_val - min_val)
        if reverse:
            normalized = 1 - normalized

        # Color scale: red (0) -> yellow (0.5) -> green (1)
        if normalized < 0.5:
            # Red to yellow
            r, g, b = 255, int(normalized * 2 * 255), 0
        else:
            # Yellow to green
            r, g, b = int((1 - (normalized - 0.5) * 2) * 255), 255, 0

        return f'rgb({r},{g},{b})'

    @st.fragment
    def _display_keeper_explorer(self, df):
        """Interactive keeper data explorer with sortable AgGrid table"""
        st.subheader("Search & Filter Keepers")

        years = sorted(df['year'].unique().tolist(), reverse=True)
        managers = sorted(df['manager'].unique().tolist())
        # Handle None values in positions - filter them out, sort, then add back
        positions_raw = df['yahoo_position'].unique().tolist()
        positions = sorted([p for p in positions_raw if p is not None])
        if None in positions_raw:
            positions.append('(Unknown)')  # Add a label for None values

        # Cleaner filter layout
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_years = st.multiselect("Year(s)", years, default=[years[0]] if years else [], key="keeper_explorer_years")
            selected_managers = st.multiselect("Manager(s)", managers, default=[], key="keeper_explorer_managers")
        with col2:
            selected_positions = st.multiselect("Position(s)", positions, default=[], key="keeper_explorer_positions")
            show_only = st.radio("Show", ["All Players", "Keepers Only"], horizontal=True, key="keeper_explorer_show")
        with col3:
            # Max keeper price filter
            max_keeper_price = df['keeper_price'].max()
            min_keeper_price = df['keeper_price'].min()
            selected_max_price = st.number_input(
                "Max Keeper Price",
                min_value=float(min_keeper_price),
                max_value=float(max_keeper_price),
                value=float(max_keeper_price),
                step=1.0,
                format="%.0f",
                help="Filter by maximum keeper price",
                key="keeper_explorer_max_price"
            )

        # Apply filters
        filtered = df.copy()
        if selected_years:
            filtered = filtered[filtered['year'].isin(selected_years)]
        if selected_managers:
            filtered = filtered[filtered['manager'].isin(selected_managers)]
        if selected_positions:
            filtered = filtered[filtered['yahoo_position'].isin(selected_positions)]

        # Filter by max keeper price
        filtered = filtered[filtered['keeper_price'] <= selected_max_price]

        if show_only == "Keepers Only":
            filtered = filtered[filtered['is_keeper_status'] == True]

        # Display count
        st.caption(f"Showing {len(filtered)} player(s)")

        if filtered.empty:
            st.info("No players match your filters.")
            return

        # Prepare data for sortable dataframe (removed nfl_team per user request)
        # Include SPAR columns if available
        base_cols = [
            'headshot_url', 'player', 'yahoo_position', 'manager', 'year',
            'keeper_price', 'cost', 'max_faab_bid_to_date',
            'avg_points_this_year', 'avg_points_next_year', 'is_keeper_status'
        ]
        spar_cols = ['spar']
        display_cols = base_cols + [col for col in spar_cols if col in filtered.columns]
        display_df = filtered[display_cols].copy()

        # Calculate SPAR/$ on the fly for display
        if 'cost' in display_df.columns and 'spar' in display_df.columns:
            display_df['spar_per_dollar'] = (display_df['spar'] / display_df['cost']).fillna(0)

        # Format headshot URLs (handle NaN/None)
        display_df['headshot_url'] = display_df['headshot_url'].apply(
            lambda x: x if pd.notna(x) and str(x) != 'nan' else 'https://via.placeholder.com/32'
        )

        # Display using Streamlit's native dataframe with sortable columns
        st.dataframe(
            display_df,
            column_config={
                'headshot_url': st.column_config.ImageColumn(
                    '',
                    width='small',
                    help='Player headshot'
                ),
                'player': st.column_config.TextColumn(
                    'Player',
                    width='medium'
                ),
                'yahoo_position': st.column_config.TextColumn(
                    'Pos',
                    width='small'
                ),
                'manager': st.column_config.TextColumn(
                    'Manager',
                    width='medium'
                ),
                'year': st.column_config.NumberColumn(
                    'Year',
                    width='small',
                    format='%d'
                ),
                'keeper_price': st.column_config.NumberColumn(
                    'Keep $',
                    width='small',
                    format='$%.0f',
                    help='Keeper price'
                ),
                'cost': st.column_config.NumberColumn(
                    'Draft $',
                    width='small',
                    format='$%.0f',
                    help='Draft cost'
                ),
                'max_faab_bid_to_date': st.column_config.NumberColumn(
                    'Max FAAB',
                    width='small',
                    format='$%.0f',
                    help='Maximum FAAB bid to date'
                ),
                'avg_points_this_year': st.column_config.NumberColumn(
                    'PPG',
                    width='small',
                    format='%.1f',
                    help='Points per game this year'
                ),
                'avg_points_next_year': st.column_config.NumberColumn(
                    'Next PPG',
                    width='small',
                    format='%.1f',
                    help='Points per game next year'
                ),
                'is_keeper_status': st.column_config.CheckboxColumn(
                    'Keeper',
                    width='small',
                    help='Is this a keeper?'
                ),
                'spar': st.column_config.NumberColumn(
                    'SPAR',
                    width='small',
                    format='%.1f',
                    help='Season Points Above Replacement'
                ),
                'spar_per_dollar': st.column_config.NumberColumn(
                    'SPAR/$',
                    width='small',
                    format='%.2f',
                    help='SPAR per cost dollar'
                )
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )

    @st.fragment
    def _display_analytics(self, df):
        """Visual analytics for keeper trends"""
        st.subheader("Keeper Trends & Insights")

        # Add filters for manager and year
        years = sorted(df['year'].unique().tolist(), reverse=True)
        managers = sorted(df['manager'].unique().tolist())

        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Filter by Year", ["All Years"] + years, index=0, key="analytics_year")
        with col2:
            selected_manager = st.selectbox("Filter by Manager", ["All Managers"] + managers, index=0, key="analytics_manager")

        # Apply filters
        filtered_df = df.copy()
        if selected_year != "All Years":
            filtered_df = filtered_df[filtered_df['year'] == selected_year]
        if selected_manager != "All Managers":
            filtered_df = filtered_df[filtered_df['manager'] == selected_manager]

        # Filter to actual keepers
        keepers = filtered_df[filtered_df['is_keeper_status'] == True].copy()

        if keepers.empty:
            st.info("No keeper data to analyze with selected filters.")
            return

        # Keepers by position over time
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Keepers by Position**")
            pos_counts = keepers.groupby('yahoo_position').size().sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                x=pos_counts.values,
                y=pos_counts.index,
                orientation='h',
                marker_color='lightblue'
            ))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Average Keeper SPAR/$ by Position**")
            keepers_with_cost = keepers[keepers['cost'] > 0].copy()
            keepers_with_cost['spar_per_dollar'] = keepers_with_cost['spar'] / keepers_with_cost['cost']
            avg_spar_efficiency = keepers_with_cost.groupby('yahoo_position')['spar_per_dollar'].mean().sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                x=avg_spar_efficiency.values,
                y=avg_spar_efficiency.index,
                orientation='h',
                marker_color='lightgreen'
            ))
            fig.update_xaxes(title_text="SPAR per $")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Keeper trends over years (only if not filtering by a single year)
        if selected_year == "All Years" and len(keepers['year'].unique()) > 1:
            st.markdown("**Keeper Trends Over Time**")
            year_trends = keepers.groupby('year').agg({
                'player': 'count',
                'keeper_price': 'mean'
            }).reset_index()
            year_trends.columns = ['year', 'total_keepers', 'avg_price']

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=year_trends['year'], y=year_trends['total_keepers'], name="Total Keepers"),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=year_trends['year'], y=year_trends['avg_price'],
                          name="Avg Price", mode='lines+markers', line=dict(color='red', width=3)),
                secondary_y=True
            )
            fig.update_yaxes(title_text="Total Keepers", secondary_y=False)
            fig.update_yaxes(title_text="Avg Price ($)", secondary_y=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def _display_best_keepers(self, df):
        """Show the best keeper values with meaningful ROI metrics"""
        st.subheader("Top Keeper Values")

        # Add filters for manager and year
        years = sorted(df['year'].unique().tolist(), reverse=True)
        managers = sorted(df['manager'].unique().tolist())

        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Filter by Year", ["All Years"] + years, index=0, key="best_keepers_year")
        with col2:
            selected_manager = st.selectbox("Filter by Manager", ["All Managers"] + managers, index=0, key="best_keepers_manager")

        # Apply filters
        filtered_df = df.copy()
        if selected_year != "All Years":
            filtered_df = filtered_df[filtered_df['year'] == selected_year]
        if selected_manager != "All Managers":
            filtered_df = filtered_df[filtered_df['manager'] == selected_manager]

        # Calculate keeper value - only for players who ARE keepers this year
        keepers = filtered_df[filtered_df['is_keeper_status'] == True].copy()

        if keepers.empty:
            st.info("No keepers found with selected filters.")
            return

        # Filter to keepers with SPAR data
        keepers_with_data = keepers[keepers['spar'].notna()].copy()

        if keepers_with_data.empty:
            st.warning("No SPAR data available for keepers.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Best Keeper Value (SPAR)**")
            st.caption("Highest keeper SPAR values")

            top_spar_df = keepers_with_data[keepers_with_data['cost'] > 0].copy()
            top_spar = top_spar_df.nlargest(10, 'spar')[
                ['headshot_url', 'player', 'yahoo_position', 'manager', 'year', 'cost', 'spar']
            ].copy()

            top_spar['Cost'] = top_spar['cost'].round(1)
            top_spar['SPAR'] = top_spar['spar'].round(1)

            display_spar = top_spar[['headshot_url', 'player', 'yahoo_position', 'manager', 'year', 'Cost', 'SPAR']]

            st.dataframe(
                display_spar,
                hide_index=True,
                use_container_width=True,
                column_config={
                    'headshot_url': st.column_config.ImageColumn('', width='small'),
                    'player': 'Player',
                    'yahoo_position': 'Pos',
                    'manager': 'Manager',
                    'year': 'Year',
                    'Cost': st.column_config.NumberColumn('Cost', format='$%.1f'),
                    'SPAR': st.column_config.NumberColumn('SPAR', format='%.1f')
                }
            )

        with col2:
            st.markdown("**Best SPAR per Dollar**")
            st.caption("Most efficient keepers: SPAR ÷ Cost")

            efficiency_df = keepers_with_data[keepers_with_data['cost'] > 0].copy()
            efficiency_df['spar_per_dollar'] = efficiency_df['spar'] / efficiency_df['cost']
            ppg_value = efficiency_df.nlargest(10, 'spar_per_dollar')[
                ['headshot_url', 'player', 'yahoo_position', 'manager', 'year', 'cost', 'spar', 'spar_per_dollar']
            ].copy()

            ppg_value['Cost'] = ppg_value['cost'].round(1)
            ppg_value['SPAR'] = ppg_value['spar'].round(1)
            ppg_value['SPAR/$'] = ppg_value['spar_per_dollar'].round(2)

            display_ppg = ppg_value[['headshot_url', 'player', 'yahoo_position', 'manager', 'year', 'Cost', 'SPAR', 'SPAR/$']]

            st.dataframe(
                display_ppg,
                hide_index=True,
                use_container_width=True,
                column_config={
                    'headshot_url': st.column_config.ImageColumn('', width='small'),
                    'player': 'Player',
                    'yahoo_position': 'Pos',
                    'manager': 'Manager',
                    'year': 'Year',
                    'Keeper $': st.column_config.NumberColumn('Keeper $', format='$%.1f'),
                    'SPAR': st.column_config.NumberColumn('SPAR', format='%.1f'),
                    'SPAR/$': st.column_config.NumberColumn('SPAR/$', format='%.2f')
                }
            )

        # Additional analytics
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Worst Keeper Values**")
            st.caption("Lowest keeper SPAR values (busts)")

            worst_keepers = keepers_with_data.nsmallest(10, 'spar')[
                ['headshot_url', 'player', 'yahoo_position', 'manager', 'year', 'cost', 'spar']
            ].copy()

            worst_keepers['Cost'] = worst_keepers['cost'].round(1)
            worst_keepers['SPAR'] = worst_keepers['spar'].round(1)

            display_worst = worst_keepers[['headshot_url', 'player', 'yahoo_position', 'manager', 'year', 'Cost', 'SPAR']]

            st.dataframe(
                display_worst,
                hide_index=True,
                use_container_width=True,
                column_config={
                    'headshot_url': st.column_config.ImageColumn('', width='small'),
                    'player': 'Player',
                    'yahoo_position': 'Pos',
                    'manager': 'Manager',
                    'year': 'Year',
                    'Cost': st.column_config.NumberColumn('Cost', format='$%.1f'),
                    'SPAR': st.column_config.NumberColumn('SPAR', format='%.1f')
                }
            )

        with col4:
            st.markdown("**Manager Keeper Success**")
            st.caption("Average efficiency and production by manager")

            manager_keeper_df = keepers_with_data[keepers_with_data['cost'] > 0].copy()
            manager_keeper_df['spar_per_dollar'] = manager_keeper_df['spar'] / manager_keeper_df['cost']

            manager_stats = manager_keeper_df.groupby('manager').agg({
                'player': 'count',
                'spar_per_dollar': 'mean',
                'spar': 'mean',
                'cost': 'mean'
            }).round(2).reset_index()
            manager_stats.columns = ['Manager', 'Total Keepers', 'Avg SPAR/$', 'Avg SPAR', 'Avg Cost']
            manager_stats = manager_stats.sort_values('Avg SPAR/$', ascending=False)

            st.dataframe(manager_stats, hide_index=True, use_container_width=True)

        # Note about missing data (compact)
        missing_count = len(keepers) - len(keepers_with_data)
        if missing_count > 0:
            st.info(f"ℹ️ Note: {missing_count} keeper(s) excluded due to missing performance data")
