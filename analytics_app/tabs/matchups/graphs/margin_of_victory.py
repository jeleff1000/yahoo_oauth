#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from md.core import T, list_seasons, run_query


@st.fragment
def display_margin_of_victory_graph(df_dict=None, prefix=""):
    """
    Analyze margin of victory/defeat - identify blowout winners, nail-biters, and unlucky losers.
    """
    st.header("📏 Margin of Victory Analysis")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Beyond W-L records:</strong> See how managers win and lose. Do you dominate opponents or squeak by?
    How many heartbreaking close losses have you suffered?
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Year selection
    available_years = list_seasons()
    if not available_years:
        st.error("No data available.")
        return

    year_options = ["All Seasons"] + available_years
    selected_year = st.selectbox(
        "Select Season",
        options=year_options,
        key=f"{prefix}_year"
    )

    # Load data
    with st.spinner("Loading margin data..."):
        if selected_year == "All Seasons":
            query = f"""
                SELECT
                    manager,
                    year,
                    week,
                    team_points,
                    opponent_points,
                    (team_points - opponent_points) as margin,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND team_points IS NOT NULL
                  AND opponent_points IS NOT NULL
                ORDER BY manager, year, week
            """
        else:
            query = f"""
                SELECT
                    manager,
                    week,
                    team_points,
                    opponent_points,
                    (team_points - opponent_points) as margin,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
                FROM {T['matchup']}
                WHERE year = {int(selected_year)}
                  AND manager IS NOT NULL
                  AND team_points IS NOT NULL
                  AND opponent_points IS NOT NULL
                ORDER BY manager, week
            """
        data = run_query(query)

        if data.empty:
            st.warning("No data found.")
            return

    # Manager selection
    managers = sorted(data["manager"].unique())
    selected_managers = st.multiselect(
        "Select Managers to Display",
        options=managers,
        default=managers[:3] if len(managers) >= 3 else managers,
        key=f"{prefix}_managers"
    )

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    filtered_data = data[data["manager"].isin(selected_managers)].copy()

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🎯 Close Games", "📈 Statistics"])

    with tab1:
        st.subheader("Victory Margin Distribution")
        st.caption("Positive = Win, Negative = Loss. See your pattern of dominance vs close games")

        # Histogram of margins by manager
        fig_hist = go.Figure()

        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]

            fig_hist.add_trace(go.Histogram(
                x=manager_data["margin"],
                name=manager,
                opacity=0.7,
                nbinsx=30,
                hovertemplate="<b>%{fullData.name}</b><br>Margin: %{x:.1f}<br>Count: %{y}<extra></extra>"
            ))

        # Add reference line at 0
        fig_hist.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text="Even",
            annotation_position="top"
        )

        fig_hist.update_layout(
            xaxis_title="Margin of Victory (Points)",
            yaxis_title="Number of Games",
            barmode='overlay',
            height=500,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig_hist, use_container_width=True, key=f"{prefix}_hist")

        # Box plot for comparison
        st.subheader("Margin Comparison (Box Plot)")

        fig_box = go.Figure()

        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]

            fig_box.add_trace(go.Box(
                y=manager_data["margin"],
                name=manager,
                boxmean='sd',
                hovertemplate="<b>%{fullData.name}</b><br>Margin: %{y:.1f}<extra></extra>"
            ))

        fig_box.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)

        fig_box.update_layout(
            yaxis_title="Margin of Victory (Points)",
            showlegend=False,
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig_box, use_container_width=True, key=f"{prefix}_box")

    with tab2:
        st.subheader("🎯 Close Game Analysis")
        st.caption("Games decided by 10 points or less - who's clutch and who's unlucky?")

        # Define close game threshold
        close_threshold = st.slider(
            "Define 'Close Game' (within X points)",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            key=f"{prefix}_threshold"
        )

        # Calculate close game stats
        close_stats = []
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]

            close_games = manager_data[abs(manager_data["margin"]) <= close_threshold]
            close_wins = close_games[close_games["win"] == 1]
            close_losses = close_games[close_games["win"] == 0]

            blowout_games = manager_data[abs(manager_data["margin"]) > close_threshold]
            blowout_wins = blowout_games[blowout_games["win"] == 1]
            blowout_losses = blowout_games[blowout_games["win"] == 0]

            close_stats.append({
                "Manager": manager,
                "Close Wins": len(close_wins),
                "Close Losses": len(close_losses),
                "Close Games": len(close_games),
                "Close Win %": (len(close_wins) / len(close_games) * 100) if len(close_games) > 0 else 0,
                "Blowout Wins": len(blowout_wins),
                "Blowout Losses": len(blowout_losses),
                "Total Games": len(manager_data)
            })

        close_df = pd.DataFrame(close_stats)

        # Stacked bar chart of close vs blowout
        fig_close = go.Figure()

        fig_close.add_trace(go.Bar(
            name='Close Wins',
            x=close_df['Manager'],
            y=close_df['Close Wins'],
            marker_color='lightgreen',
            text=close_df['Close Wins'],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>Close Wins: %{y}<extra></extra>"
        ))

        fig_close.add_trace(go.Bar(
            name='Close Losses',
            x=close_df['Manager'],
            y=close_df['Close Losses'],
            marker_color='lightcoral',
            text=close_df['Close Losses'],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>Close Losses: %{y}<extra></extra>"
        ))

        fig_close.add_trace(go.Bar(
            name='Blowout Wins',
            x=close_df['Manager'],
            y=close_df['Blowout Wins'],
            marker_color='darkgreen',
            text=close_df['Blowout Wins'],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>Blowout Wins: %{y}<extra></extra>"
        ))

        fig_close.add_trace(go.Bar(
            name='Blowout Losses',
            x=close_df['Manager'],
            y=close_df['Blowout Losses'],
            marker_color='darkred',
            text=close_df['Blowout Losses'],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>Blowout Losses: %{y}<extra></extra>"
        ))

        fig_close.update_layout(
            barmode='stack',
            xaxis_title="",
            yaxis_title="Number of Games",
            height=450,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig_close, use_container_width=True, key=f"{prefix}_close")

        # Close game win percentage
        st.subheader(f"Close Game Win % (≤{close_threshold} pts)")

        fig_close_pct = go.Figure()

        close_df_sorted = close_df.sort_values("Close Win %", ascending=True)

        fig_close_pct.add_trace(go.Bar(
            y=close_df_sorted['Manager'],
            x=close_df_sorted['Close Win %'],
            orientation='h',
            marker=dict(
                color=close_df_sorted['Close Win %'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Win %")
            ),
            text=close_df_sorted['Close Win %'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Close Win %: %{x:.1f}%<extra></extra>"
        ))

        fig_close_pct.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

        fig_close_pct.update_layout(
            xaxis_title="Win Percentage in Close Games",
            yaxis_title="",
            height=max(300, len(close_df) * 40),
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_close_pct, use_container_width=True, key=f"{prefix}_close_pct")

        # Display table
        with st.expander("📊 Detailed Close Game Stats", expanded=False):
            display_close = close_df.copy()
            display_close['Close Win %'] = display_close['Close Win %'].round(1)
            st.dataframe(display_close, hide_index=True, use_container_width=True)

    with tab3:
        st.subheader("📈 Margin Statistics")

        # Calculate stats by manager
        margin_stats = []
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]

            wins = manager_data[manager_data["win"] == 1]
            losses = manager_data[manager_data["win"] == 0]

            margin_stats.append({
                "Manager": manager,
                "Avg Margin (All)": manager_data["margin"].mean(),
                "Avg Win Margin": wins["margin"].mean() if len(wins) > 0 else 0,
                "Avg Loss Margin": losses["margin"].mean() if len(losses) > 0 else 0,
                "Biggest Win": manager_data["margin"].max(),
                "Worst Loss": manager_data["margin"].min(),
                "Std Dev": manager_data["margin"].std()
            })

        stats_df = pd.DataFrame(margin_stats).round(2)

        # Display stats table
        st.dataframe(
            stats_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Avg Margin (All)": st.column_config.NumberColumn(format="%.2f"),
                "Avg Win Margin": st.column_config.NumberColumn(format="%.2f"),
                "Avg Loss Margin": st.column_config.NumberColumn(format="%.2f"),
                "Biggest Win": st.column_config.NumberColumn(format="%.2f"),
                "Worst Loss": st.column_config.NumberColumn(format="%.2f"),
                "Std Dev": st.column_config.NumberColumn(format="%.2f")
            }
        )

        # Bar chart comparing average margins
        st.subheader("Average Win vs Loss Margin")

        fig_avg_margin = go.Figure()

        fig_avg_margin.add_trace(go.Bar(
            name='Avg Win Margin',
            x=stats_df['Manager'],
            y=stats_df['Avg Win Margin'],
            marker_color='green',
            text=stats_df['Avg Win Margin'].apply(lambda x: f"+{x:.1f}"),
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Avg Win Margin: +%{y:.1f}<extra></extra>"
        ))

        fig_avg_margin.add_trace(go.Bar(
            name='Avg Loss Margin',
            x=stats_df['Manager'],
            y=stats_df['Avg Loss Margin'],
            marker_color='red',
            text=stats_df['Avg Loss Margin'].apply(lambda x: f"{x:.1f}"),
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Avg Loss Margin: %{y:.1f}<extra></extra>"
        ))

        fig_avg_margin.update_layout(
            barmode='group',
            xaxis_title="",
            yaxis_title="Average Margin (Points)",
            height=400,
            template="plotly_white",
            showlegend=True
        )

        st.plotly_chart(fig_avg_margin, use_container_width=True, key=f"{prefix}_avg_margin")

    # Key insights
    with st.expander("💡 Key Insights", expanded=False):
        # Find the most dominant winner (highest avg win margin)
        most_dominant = stats_df.nlargest(1, "Avg Win Margin").iloc[0]

        # Find the unluckiest (most close losses)
        most_unlucky_idx = close_df["Close Losses"].idxmax()
        most_unlucky = close_df.loc[most_unlucky_idx]

        # Find clutch performer (highest close game win %)
        clutch_idx = close_df[close_df["Close Games"] >= 5]["Close Win %"].idxmax()
        if pd.notna(clutch_idx):
            most_clutch = close_df.loc[clutch_idx]
        else:
            most_clutch = None

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "💪 Most Dominant",
                most_dominant["Manager"],
                f"+{most_dominant['Avg Win Margin']:.1f} pts/win"
            )

        with col2:
            if most_clutch is not None:
                st.metric(
                    "🎯 Most Clutch",
                    most_clutch["Manager"],
                    f"{most_clutch['Close Win %']:.1f}% in close games"
                )
            else:
                st.info("Not enough close games for analysis")

        with col3:
            st.metric(
                "😢 Most Close Losses",
                most_unlucky["Manager"],
                f"{int(most_unlucky['Close Losses'])} games"
            )
