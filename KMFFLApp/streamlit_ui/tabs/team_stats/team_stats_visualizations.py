#!/usr/bin/env python3
"""Team Stats Visualizations - Charts and graphs for team performance analysis."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TeamStatsVisualizer:
    """Handles all visualizations for team stats."""

    def __init__(self, team_data_by_position: pd.DataFrame, team_data_by_manager: pd.DataFrame):
        self.team_data_by_position = team_data_by_position.copy() if team_data_by_position is not None else pd.DataFrame()
        self.team_data_by_manager = team_data_by_manager.copy() if team_data_by_manager is not None else pd.DataFrame()

    @st.fragment
    def display_weekly_visualizations(self):
        """Display weekly team stats visualizations."""
        st.markdown("### Weekly Team Stats Visualizations")

        if self.team_data_by_manager.empty:
            st.info("No data available for visualizations")
            return

        # Chart type selector
        chart_types = [
            "Points Over Time",
            "TD Breakdown by Manager",
            "Yards Breakdown by Manager",
            "Position Contribution"
        ]

        selected_chart = st.selectbox(
            "Select Visualization",
            chart_types,
            key="weekly_viz_selector"
        )

        if selected_chart == "Points Over Time":
            self.plot_points_over_time()
        elif selected_chart == "TD Breakdown by Manager":
            self.plot_td_breakdown()
        elif selected_chart == "Yards Breakdown by Manager":
            self.plot_yards_breakdown()
        elif selected_chart == "Position Contribution":
            self.plot_position_contribution()

    @st.fragment
    def plot_points_over_time(self):
        """Line chart showing points scored over time by manager."""
        df = self.team_data_by_manager.copy()

        # Ensure numeric types
        df['points'] = pd.to_numeric(df['points'], errors='coerce')
        df['week'] = pd.to_numeric(df['week'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Create week-year label
        df['week_year'] = df['year'].astype(str) + ' W' + df['week'].astype(str)
        df = df.sort_values(['year', 'week'])

        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            available_years = sorted(df['year'].dropna().unique().tolist(), reverse=True)
            selected_years = st.multiselect(
                "Select Years",
                available_years,
                default=available_years[:2] if len(available_years) >= 2 else available_years,
                key="points_over_time_years"
            )

        with col2:
            available_managers = sorted(df['manager'].dropna().unique().tolist())
            selected_managers = st.multiselect(
                "Select Managers",
                available_managers,
                default=available_managers,
                key="points_over_time_managers"
            )

        if not selected_years or not selected_managers:
            st.warning("Please select at least one year and one manager")
            return

        # Filter data
        filtered_df = df[df['year'].isin(selected_years) & df['manager'].isin(selected_managers)]

        if filtered_df.empty:
            st.info("No data available for selected filters")
            return

        # Create line chart
        fig = px.line(
            filtered_df,
            x='week_year',
            y='points',
            color='manager',
            title='Points Scored Over Time',
            labels={'points': 'Points', 'week_year': 'Week', 'manager': 'Manager'},
            markers=True,
            height=500
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def plot_td_breakdown(self):
        """Stacked bar chart showing TD breakdown by manager."""
        df = self.team_data_by_manager.copy()

        # Ensure numeric types
        numeric_cols = ['passing_tds', 'rushing_tds', 'receiving_tds', 'def_tds', 'week', 'year']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Filter by year
        available_years = sorted(df['year'].dropna().unique().tolist(), reverse=True)
        selected_year = st.selectbox(
            "Select Year",
            available_years,
            key="td_breakdown_year"
        )

        # Filter by week range
        year_data = df[df['year'] == selected_year].copy()
        available_weeks = sorted(year_data['week'].dropna().unique().tolist())

        if len(available_weeks) > 1:
            week_range = st.slider(
                "Select Week Range",
                min_value=int(min(available_weeks)),
                max_value=int(max(available_weeks)),
                value=(int(min(available_weeks)), int(max(available_weeks))),
                key="td_breakdown_weeks"
            )
            filtered_df = year_data[
                (year_data['week'] >= week_range[0]) &
                (year_data['week'] <= week_range[1])
            ].copy()
        else:
            filtered_df = year_data.copy()

        # Aggregate by manager
        agg_df = filtered_df.groupby('manager').agg({
            'passing_tds': 'sum',
            'rushing_tds': 'sum',
            'receiving_tds': 'sum',
            'def_tds': 'sum'
        }).reset_index()

        # Calculate total TDs
        agg_df['total_tds'] = (
            agg_df['passing_tds'] +
            agg_df['rushing_tds'] +
            agg_df['receiving_tds'] +
            agg_df['def_tds']
        )

        # Sort by total TDs
        agg_df = agg_df.sort_values('total_tds', ascending=True)

        # Create stacked horizontal bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Passing TD',
            y=agg_df['manager'],
            x=agg_df['passing_tds'],
            orientation='h',
            marker_color='#1f77b4'
        ))

        fig.add_trace(go.Bar(
            name='Rushing TD',
            y=agg_df['manager'],
            x=agg_df['rushing_tds'],
            orientation='h',
            marker_color='#ff7f0e'
        ))

        fig.add_trace(go.Bar(
            name='Receiving TD',
            y=agg_df['manager'],
            x=agg_df['receiving_tds'],
            orientation='h',
            marker_color='#2ca02c'
        ))

        fig.add_trace(go.Bar(
            name='Defensive TD',
            y=agg_df['manager'],
            x=agg_df['def_tds'],
            orientation='h',
            marker_color='#d62728'
        ))

        fig.update_layout(
            barmode='stack',
            title=f'Touchdown Breakdown by Manager ({selected_year})',
            xaxis_title='Total Touchdowns',
            yaxis_title='Manager',
            height=max(400, len(agg_df) * 40),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def plot_yards_breakdown(self):
        """Stacked bar chart showing yards breakdown by manager."""
        df = self.team_data_by_manager.copy()

        # Ensure numeric types
        numeric_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'week', 'year']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Filter by year
        available_years = sorted(df['year'].dropna().unique().tolist(), reverse=True)
        selected_year = st.selectbox(
            "Select Year",
            available_years,
            key="yards_breakdown_year"
        )

        # Filter by week range
        year_data = df[df['year'] == selected_year].copy()
        available_weeks = sorted(year_data['week'].dropna().unique().tolist())

        if len(available_weeks) > 1:
            week_range = st.slider(
                "Select Week Range",
                min_value=int(min(available_weeks)),
                max_value=int(max(available_weeks)),
                value=(int(min(available_weeks)), int(max(available_weeks))),
                key="yards_breakdown_weeks"
            )
            filtered_df = year_data[
                (year_data['week'] >= week_range[0]) &
                (year_data['week'] <= week_range[1])
            ].copy()
        else:
            filtered_df = year_data.copy()

        # Aggregate by manager
        agg_df = filtered_df.groupby('manager').agg({
            'passing_yards': 'sum',
            'rushing_yards': 'sum',
            'receiving_yards': 'sum'
        }).reset_index()

        # Calculate total yards
        agg_df['total_yards'] = (
            agg_df['passing_yards'] +
            agg_df['rushing_yards'] +
            agg_df['receiving_yards']
        )

        # Sort by total yards
        agg_df = agg_df.sort_values('total_yards', ascending=True)

        # Create stacked horizontal bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Passing Yards',
            y=agg_df['manager'],
            x=agg_df['passing_yards'],
            orientation='h',
            marker_color='#1f77b4'
        ))

        fig.add_trace(go.Bar(
            name='Rushing Yards',
            y=agg_df['manager'],
            x=agg_df['rushing_yards'],
            orientation='h',
            marker_color='#ff7f0e'
        ))

        fig.add_trace(go.Bar(
            name='Receiving Yards',
            y=agg_df['manager'],
            x=agg_df['receiving_yards'],
            orientation='h',
            marker_color='#2ca02c'
        ))

        fig.update_layout(
            barmode='stack',
            title=f'Yards Breakdown by Manager ({selected_year})',
            xaxis_title='Total Yards',
            yaxis_title='Manager',
            height=max(400, len(agg_df) * 40),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def plot_position_contribution(self):
        """Stacked area chart showing position contribution to points over time."""
        df = self.team_data_by_position.copy()

        if df.empty:
            st.info("Position breakdown data not available")
            return

        # Ensure numeric types
        df['points'] = pd.to_numeric(df['points'], errors='coerce')
        df['week'] = pd.to_numeric(df['week'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Filter by year
        available_years = sorted(df['year'].dropna().unique().tolist(), reverse=True)
        selected_year = st.selectbox(
            "Select Year",
            available_years,
            key="position_contribution_year"
        )

        # Filter by manager
        year_data = df[df['year'] == selected_year].copy()
        available_managers = sorted(year_data['manager'].dropna().unique().tolist())
        selected_manager = st.selectbox(
            "Select Manager",
            available_managers,
            key="position_contribution_manager"
        )

        # Filter data
        filtered_df = year_data[year_data['manager'] == selected_manager].copy()

        if filtered_df.empty:
            st.info("No data available for selected filters")
            return

        # Pivot data for stacked area chart
        pivot_df = filtered_df.pivot_table(
            index='week',
            columns='fantasy_position',
            values='points',
            aggfunc='sum',
            fill_value=0
        ).reset_index()

        # Create stacked area chart
        fig = go.Figure()

        # Get position columns (exclude week)
        position_cols = [col for col in pivot_df.columns if col != 'week']

        for position in position_cols:
            fig.add_trace(go.Scatter(
                x=pivot_df['week'],
                y=pivot_df[position],
                name=position,
                mode='lines',
                stackgroup='one',
                fillcolor=None
            ))

        fig.update_layout(
            title=f'Position Contribution to Points - {selected_manager} ({selected_year})',
            xaxis_title='Week',
            yaxis_title='Points',
            height=500,
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def display_season_visualizations(self):
        """Display season team stats visualizations."""
        st.markdown("### Season Team Stats Visualizations")

        if self.team_data_by_manager.empty:
            st.info("No data available for visualizations")
            return

        # Chart type selector
        chart_types = [
            "Season Points Comparison",
            "Position Performance Heatmap",
            "Manager Rankings Over Time"
        ]

        selected_chart = st.selectbox(
            "Select Visualization",
            chart_types,
            key="season_viz_selector"
        )

        if selected_chart == "Season Points Comparison":
            self.plot_season_points_comparison()
        elif selected_chart == "Position Performance Heatmap":
            self.plot_position_heatmap()
        elif selected_chart == "Manager Rankings Over Time":
            self.plot_manager_rankings()

    @st.fragment
    def plot_season_points_comparison(self):
        """Bar chart comparing season totals by manager."""
        df = self.team_data_by_manager.copy()

        # Ensure numeric types
        df['points'] = pd.to_numeric(df['points'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Year selector
        available_years = sorted(df['year'].dropna().unique().tolist(), reverse=True)
        selected_years = st.multiselect(
            "Select Years to Compare",
            available_years,
            default=available_years[:3] if len(available_years) >= 3 else available_years,
            key="season_comparison_years"
        )

        if not selected_years:
            st.warning("Please select at least one year")
            return

        # Filter and aggregate
        filtered_df = df[df['year'].isin(selected_years)].copy()

        # Create grouped bar chart
        fig = px.bar(
            filtered_df,
            x='manager',
            y='points',
            color='year',
            barmode='group',
            title='Season Points Comparison by Manager',
            labels={'points': 'Total Points', 'manager': 'Manager', 'year': 'Year'},
            height=500
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def plot_position_heatmap(self):
        """Heatmap showing position performance by manager."""
        df = self.team_data_by_position.copy()

        if df.empty:
            st.info("Position breakdown data not available")
            return

        # Ensure numeric types
        df['points'] = pd.to_numeric(df['points'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Year selector
        available_years = sorted(df['year'].dropna().unique().tolist(), reverse=True)
        selected_year = st.selectbox(
            "Select Year",
            available_years,
            key="heatmap_year"
        )

        # Filter data
        filtered_df = df[df['year'] == selected_year].copy()

        # Aggregate by manager and position
        pivot_df = filtered_df.pivot_table(
            index='manager',
            columns='fantasy_position',
            values='points',
            aggfunc='sum',
            fill_value=0
        )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Blues',
            text=pivot_df.values.round(1),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='Manager: %{y}<br>Position: %{x}<br>Points: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Position Performance Heatmap ({selected_year})',
            xaxis_title='Position',
            yaxis_title='Manager',
            height=max(400, len(pivot_df) * 50)
        )

        st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def plot_manager_rankings(self):
        """Line chart showing manager rankings over seasons."""
        df = self.team_data_by_manager.copy()

        # Ensure numeric types
        df['points'] = pd.to_numeric(df['points'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Calculate rankings per year
        df['rank'] = df.groupby('year')['points'].rank(ascending=False, method='min')

        # Create line chart
        fig = px.line(
            df,
            x='year',
            y='rank',
            color='manager',
            title='Manager Rankings Over Time (Lower is Better)',
            labels={'rank': 'Rank', 'year': 'Year', 'manager': 'Manager'},
            markers=True,
            height=500
        )

        fig.update_yaxes(autorange="reversed")  # Reverse so rank 1 is at top

        fig.update_layout(
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def display_career_visualizations(self):
        """Display career team stats visualizations."""
        st.markdown("### Career Team Stats Visualizations")

        if self.team_data_by_manager.empty:
            st.info("No data available for visualizations")
            return

        # Chart type selector
        chart_types = [
            "Career Totals Comparison",
            "Position Dominance by Manager",
            "All-Time Leaders"
        ]

        selected_chart = st.selectbox(
            "Select Visualization",
            chart_types,
            key="career_viz_selector"
        )

        if selected_chart == "Career Totals Comparison":
            self.plot_career_totals()
        elif selected_chart == "Position Dominance by Manager":
            self.plot_position_dominance()
        elif selected_chart == "All-Time Leaders":
            self.plot_alltime_leaders()

    @st.fragment
    def plot_career_totals(self):
        """Bar chart comparing career totals."""
        df = self.team_data_by_manager.copy()

        # Metric selector
        available_metrics = {
            'total_points': 'Total Points',
            'passing_yards': 'Passing Yards',
            'rushing_yards': 'Rushing Yards',
            'receiving_yards': 'Receiving Yards',
            'passing_tds': 'Passing TDs',
            'rushing_tds': 'Rushing TDs',
            'receiving_tds': 'Receiving TDs'
        }

        selected_metric = st.selectbox(
            "Select Metric",
            list(available_metrics.keys()),
            format_func=lambda x: available_metrics[x],
            key="career_totals_metric"
        )

        # Ensure numeric
        if selected_metric in df.columns:
            df[selected_metric] = pd.to_numeric(df[selected_metric], errors='coerce')

            # Sort by selected metric
            df_sorted = df.sort_values(selected_metric, ascending=True)

            # Create horizontal bar chart
            fig = px.bar(
                df_sorted,
                x=selected_metric,
                y='manager',
                orientation='h',
                title=f'Career {available_metrics[selected_metric]} by Manager',
                labels={selected_metric: available_metrics[selected_metric], 'manager': 'Manager'},
                height=max(400, len(df_sorted) * 50)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Metric '{selected_metric}' not available in data")

    @st.fragment
    def plot_position_dominance(self):
        """Stacked bar showing position contribution to career points."""
        df = self.team_data_by_position.copy()

        if df.empty:
            st.info("Position breakdown data not available")
            return

        # Ensure numeric types
        df['total_points'] = pd.to_numeric(df.get('total_points', df.get('points', 0)), errors='coerce')

        # Aggregate by manager and position
        agg_df = df.groupby(['manager', 'fantasy_position']).agg({
            'total_points': 'sum'
        }).reset_index()

        # Pivot for stacked bar
        pivot_df = agg_df.pivot(
            index='manager',
            columns='fantasy_position',
            values='total_points'
        ).fillna(0)

        # Calculate total for sorting
        pivot_df['total'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('total', ascending=True)
        pivot_df = pivot_df.drop('total', axis=1)

        # Create stacked horizontal bar chart
        fig = go.Figure()

        for position in pivot_df.columns:
            fig.add_trace(go.Bar(
                name=position,
                y=pivot_df.index,
                x=pivot_df[position],
                orientation='h'
            ))

        fig.update_layout(
            barmode='stack',
            title='Position Contribution to Career Points',
            xaxis_title='Total Points',
            yaxis_title='Manager',
            height=max(400, len(pivot_df) * 40),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def plot_alltime_leaders(self):
        """Bar chart showing all-time leaders in various categories."""
        df = self.team_data_by_manager.copy()

        # Category selector
        categories = {
            'total_points': 'Points',
            'passing_yards': 'Passing Yards',
            'rushing_yards': 'Rushing Yards',
            'receiving_yards': 'Receiving Yards',
            'passing_tds': 'Passing TDs',
            'rushing_tds': 'Rushing TDs',
            'receiving_tds': 'Receiving TDs',
            'receptions': 'Receptions',
            'fg_made': 'Field Goals Made'
        }

        col1, col2 = st.columns(2)

        with col1:
            selected_category = st.selectbox(
                "Select Category",
                list(categories.keys()),
                format_func=lambda x: categories[x],
                key="alltime_leaders_category"
            )

        with col2:
            top_n = st.slider("Show Top N", min_value=3, max_value=15, value=10, key="alltime_leaders_top_n")

        if selected_category in df.columns:
            df[selected_category] = pd.to_numeric(df[selected_category], errors='coerce')

            # Get top N
            top_df = df.nlargest(top_n, selected_category).sort_values(selected_category, ascending=True)

            # Create horizontal bar chart
            fig = px.bar(
                top_df,
                x=selected_category,
                y='manager',
                orientation='h',
                title=f'All-Time Leaders: {categories[selected_category]}',
                labels={selected_category: categories[selected_category], 'manager': 'Manager'},
                height=max(400, len(top_df) * 40),
                text=selected_category
            )

            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Category '{selected_category}' not available in data")
