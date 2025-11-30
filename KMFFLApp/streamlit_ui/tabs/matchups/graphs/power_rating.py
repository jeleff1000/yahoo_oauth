#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from md.data_access import run_query, T, list_seasons


@st.fragment
def display_power_rating_graph(df_dict=None, prefix="graphs_manager_power_rating"):
    """
    Display power rating trends over time using Plotly.
    Power rating = ELO-style team strength metric that evolves based on performance.
    """
    st.header("⚡ Power Rating Dashboard")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Team Strength Analysis:</strong> Power ratings measure overall team quality based on performance.
    Higher ratings indicate stronger teams. Track how team strength evolves over time.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading power rating data..."):
        query = f"""
            SELECT
                year,
                week,
                manager,
                power_rating,
                team_points,
                CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
            FROM {T['matchup']}
            WHERE manager IS NOT NULL
              AND power_rating IS NOT NULL
            ORDER BY year, week, manager
        """
        df = run_query(query)

        if df is None or df.empty:
            st.warning("No power rating data found.")
            return

    # Create cumulative week column for all-time view
    df = df.sort_values(['year', 'week'])
    year_week_map = df[['year', 'week']].drop_duplicates().sort_values(['year', 'week']).reset_index(drop=True)
    year_week_map['cumulative_week'] = range(1, len(year_week_map) + 1)
    df = df.merge(year_week_map, on=['year', 'week'], how='left')

    # Get available years
    available_years = sorted(df['year'].unique())
    managers_available = sorted(df['manager'].unique())

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        start_year = st.selectbox(
            "Start Year",
            options=available_years,
            index=0,
            key=f"{prefix}_start_year"
        )

    with col2:
        end_year = st.selectbox(
            "End Year",
            options=available_years,
            index=len(available_years) - 1,
            key=f"{prefix}_end_year"
        )

    with col3:
        selected_managers = st.multiselect(
            "Select Managers",
            options=managers_available,
            default=managers_available[:5] if len(managers_available) >= 5 else managers_available,
            key=f"{prefix}_managers"
        )

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    # Filter data
    df_filtered = df[
        (df['year'] >= start_year) &
        (df['year'] <= end_year) &
        (df['manager'].isin(selected_managers))
    ].copy()

    if df_filtered.empty:
        st.info("No data after applying filters.")
        return

    # Create power rating trend chart
    st.subheader("Power Rating Over Time")

    fig = go.Figure()

    for manager in selected_managers:
        manager_data = df_filtered[df_filtered['manager'] == manager].sort_values('cumulative_week')

        if manager_data.empty:
            continue

        fig.add_trace(go.Scatter(
            x=manager_data['cumulative_week'],
            y=manager_data['power_rating'],
            mode='lines+markers',
            name=manager,
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{manager}</b><br>"
                "Year: %{customdata[0]}<br>"
                "Week: %{customdata[1]}<br>"
                "Power Rating: %{y:.2f}<br>"
                "<extra></extra>"
            ),
            customdata=manager_data[['year', 'week']].values
        ))

    # Add year boundary markers
    year_boundaries = df_filtered.groupby('year')['cumulative_week'].min().reset_index()
    for _, row in year_boundaries.iterrows():
        fig.add_vline(
            x=row['cumulative_week'],
            line_dash="dash",
            line_color="gray",
            opacity=0.3,
            annotation_text=str(int(row['year'])),
            annotation_position="top"
        )

    fig.update_layout(
        xaxis_title="Cumulative Week",
        yaxis_title="Power Rating",
        hovermode="closest",
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_trend")

    # Power Rating Statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Current Power Ratings")

        # Get most recent power rating for each manager
        latest_ratings = df_filtered.sort_values('cumulative_week').groupby('manager').last().reset_index()
        latest_ratings = latest_ratings[['manager', 'power_rating', 'year', 'week']].sort_values('power_rating', ascending=False)

        display_latest = latest_ratings.copy()
        display_latest.columns = ['Manager', 'Power Rating', 'Year', 'Week']
        display_latest['Power Rating'] = display_latest['Power Rating'].round(2)

        st.dataframe(
            display_latest,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Power Rating": st.column_config.NumberColumn(format="%.2f")
            }
        )

    with col2:
        st.subheader("📈 Power Rating Range")

        # Calculate min, max, average for each manager
        rating_stats = df_filtered.groupby('manager')['power_rating'].agg(['min', 'max', 'mean']).round(2)
        rating_stats = rating_stats.sort_values('mean', ascending=False)
        rating_stats = rating_stats.reset_index()
        rating_stats.columns = ['Manager', 'Min', 'Max', 'Average']

        st.dataframe(
            rating_stats,
            hide_index=True,
            use_container_width=True
        )

    # Power Rating vs Wins correlation
    with st.expander("🔍 Power Rating vs Performance", expanded=False):
        st.caption("See how power rating correlates with winning")

        # Calculate average power rating and win percentage
        correlation_data = df_filtered.groupby('manager').agg({
            'power_rating': 'mean',
            'win': 'mean',
            'team_points': 'mean'
        }).reset_index()

        correlation_data['win_pct'] = (correlation_data['win'] * 100).round(1)
        correlation_data['avg_rating'] = correlation_data['power_rating'].round(2)
        correlation_data['avg_points'] = correlation_data['team_points'].round(2)

        # Scatter plot
        fig_scatter = go.Figure()

        fig_scatter.add_trace(go.Scatter(
            x=correlation_data['avg_rating'],
            y=correlation_data['win_pct'],
            mode='markers+text',
            text=correlation_data['manager'],
            textposition='top center',
            marker=dict(
                size=correlation_data['avg_points'] / 5,  # Size by avg points
                color=correlation_data['win_pct'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Win %"),
                line=dict(width=1, color='white')
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Avg Power Rating: %{x:.2f}<br>"
                "Win %: %{y:.1f}%<br>"
                "Avg Points: " + correlation_data['avg_points'].astype(str) + "<br>"
                "<extra></extra>"
            )
        ))

        fig_scatter.update_layout(
            xaxis_title="Average Power Rating",
            yaxis_title="Win Percentage",
            height=400,
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_scatter, use_container_width=True, key=f"{prefix}_correlation")

    # Power rating insights
    with st.expander("💡 Key Insights", expanded=False):
        if len(latest_ratings) >= 1:
            strongest = latest_ratings.iloc[0]
            weakest = latest_ratings.iloc[-1]

            # Calculate biggest improver
            if start_year != end_year:
                start_ratings = df_filtered[df_filtered['year'] == start_year].groupby('manager')['power_rating'].mean()
                end_ratings = df_filtered[df_filtered['year'] == end_year].groupby('manager')['power_rating'].mean()
                common_managers = set(start_ratings.index) & set(end_ratings.index)

                if common_managers:
                    improvements = {mgr: end_ratings[mgr] - start_ratings[mgr] for mgr in common_managers}
                    best_improver = max(improvements, key=improvements.get)
                    improvement = improvements[best_improver]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "💪 Strongest Team",
                            strongest['Manager'],
                            f"{strongest['Power Rating']:.2f}"
                        )

                    with col2:
                        st.metric(
                            "📈 Most Improved",
                            best_improver,
                            f"+{improvement:.2f}"
                        )

                    with col3:
                        st.metric(
                            "⚠️ Needs Work",
                            weakest['Manager'],
                            f"{weakest['Power Rating']:.2f}"
                        )
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "💪 Strongest Team",
                        strongest['Manager'],
                        f"{strongest['Power Rating']:.2f}"
                    )

                with col2:
                    st.metric(
                        "⚠️ Needs Work",
                        weakest['Manager'],
                        f"{weakest['Power Rating']:.2f}"
                    )
