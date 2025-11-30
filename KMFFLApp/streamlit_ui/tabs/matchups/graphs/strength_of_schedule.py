#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.data_access import run_query, T, list_seasons


@st.fragment
def display_strength_of_schedule_graph(df_dict=None, prefix=""):
    """
    Analyze strength of schedule - who had the toughest/easiest road?
    Provides context for win-loss records.
    """
    st.header("💪 Strength of Schedule Analysis")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Context matters:</strong> Not all schedules are equal. See who faced the toughest opponents
    and how schedule difficulty impacted win-loss records.
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
    with st.spinner("Loading schedule data..."):
        if selected_year == "All Seasons":
            query = f"""
                SELECT
                    year,
                    week,
                    manager,
                    opponent,
                    team_points,
                    opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND opponent IS NOT NULL
                  AND team_points IS NOT NULL
                  AND opponent_points IS NOT NULL
                ORDER BY manager, year, week
            """
        else:
            query = f"""
                SELECT
                    week,
                    manager,
                    opponent,
                    team_points,
                    opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
                FROM {T['matchup']}
                WHERE year = {int(selected_year)}
                  AND manager IS NOT NULL
                  AND opponent IS NOT NULL
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
        default=managers[:5] if len(managers) >= 5 else managers,
        key=f"{prefix}_managers"
    )

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    filtered_data = data[data["manager"].isin(selected_managers)]

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Schedule Difficulty", "🔍 Opponent Analysis", "📈 SOS vs Record"])

    with tab1:
        st.subheader("Average Opponent Strength")
        st.caption("Higher = Tougher schedule (faced stronger opponents)")

        # Calculate SOS metrics
        sos_stats = []
        for manager in selected_managers:
            manager_games = filtered_data[filtered_data["manager"] == manager]

            # Average opponent points (primary SOS metric)
            avg_opp_points = manager_games["opponent_points"].mean()

            # Calculate opponent win percentage (how good were the teams you faced?)
            opponents_list = manager_games["opponent"].unique()
            opp_win_pcts = []

            for opp in opponents_list:
                opp_games = data[data["manager"] == opp]
                if len(opp_games) > 0:
                    opp_win_pct = opp_games["win"].mean()
                    opp_win_pcts.append(opp_win_pct)

            avg_opp_win_pct = sum(opp_win_pcts) / len(opp_win_pcts) if opp_win_pcts else 0

            # Count games against top teams
            # Define "top team" as scoring above league average
            league_avg = data["team_points"].mean()
            games_vs_above_avg = len(manager_games[manager_games["opponent_points"] > league_avg])

            sos_stats.append({
                "Manager": manager,
                "Avg Opp Points": avg_opp_points,
                "Avg Opp Win %": avg_opp_win_pct * 100,
                "Games vs Above Avg": games_vs_above_avg,
                "Total Games": len(manager_games),
                "Win Rate": manager_games["win"].mean() * 100,
                "Avg Points For": manager_games["team_points"].mean()
            })

        sos_df = pd.DataFrame(sos_stats).round(2)

        # Calculate league average for reference
        league_avg_opp_pts = sos_df["Avg Opp Points"].mean()

        # Calculate SOS as difference from league average
        sos_df["SOS (vs Avg)"] = (sos_df["Avg Opp Points"] - league_avg_opp_pts).round(2)

        # Bar chart of Average Opponent Points
        fig_sos = go.Figure()

        sos_sorted = sos_df.sort_values("Avg Opp Points", ascending=True)

        # Determine color based on above/below average
        colors = ['red' if x > league_avg_opp_pts else 'lightblue' for x in sos_sorted['Avg Opp Points']]

        fig_sos.add_trace(go.Bar(
            y=sos_sorted['Manager'],
            x=sos_sorted['Avg Opp Points'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='darkgray', width=1)
            ),
            text=sos_sorted['Avg Opp Points'].apply(lambda x: f"{x:.1f}"),
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Avg Opp Points: %{x:.1f}<br>" +
                         "vs League Avg: " + sos_sorted['SOS (vs Avg)'].apply(lambda x: f"{x:+.1f}").astype(str) + "<extra></extra>"
        ))

        # Add reference line at league average
        fig_sos.add_vline(
            x=league_avg_opp_pts,
            line_dash="dash",
            line_color="black",
            line_width=2,
            opacity=0.7,
            annotation_text=f"League Avg ({league_avg_opp_pts:.1f})",
            annotation_position="top"
        )

        fig_sos.update_layout(
            xaxis_title="Average Opponent Points (Red = Tougher Schedule)",
            yaxis_title="",
            height=max(300, len(sos_df) * 40),
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_sos, use_container_width=True, key=f"{prefix}_sos")

        # Add context metrics
        with st.expander("📊 Schedule Difficulty Context", expanded=False):
            col1, col2, col3 = st.columns(3)

            toughest = sos_df.nlargest(1, "Avg Opp Points").iloc[0]
            easiest = sos_df.nsmallest(1, "Avg Opp Points").iloc[0]
            spread = toughest["Avg Opp Points"] - easiest["Avg Opp Points"]

            with col1:
                st.metric(
                    "Toughest Schedule",
                    toughest["Manager"],
                    f"{toughest['Avg Opp Points']:.1f} PPG"
                )

            with col2:
                st.metric(
                    "League Average",
                    "",
                    f"{league_avg_opp_pts:.1f} PPG"
                )

            with col3:
                st.metric(
                    "Schedule Spread",
                    "",
                    f"{spread:.1f} points"
                )

            st.caption(f"""
            **Context:** The toughest schedule faced opponents averaging {spread:.1f} more points
            than the easiest schedule. Red bars = above average difficulty, Blue bars = below average.
            """)

        # SOS stats table
        with st.expander("📊 Detailed SOS Stats", expanded=False):
            display_sos = sos_df.sort_values("Avg Opp Points", ascending=False)
            st.dataframe(
                display_sos,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Avg Opp Points": st.column_config.NumberColumn(format="%.2f"),
                    "Avg Opp Win %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                    "Avg Points For": st.column_config.NumberColumn(format="%.2f"),
                    "SOS (vs Avg)": st.column_config.NumberColumn(format="%.2f")
                }
            )

    with tab2:
        st.subheader("🔍 Who Did You Face?")

        # Select a specific manager for detailed opponent analysis
        analysis_manager = st.selectbox(
            "Select Manager for Detailed Analysis",
            options=selected_managers,
            key=f"{prefix}_analysis_mgr"
        )

        manager_games = filtered_data[filtered_data["manager"] == analysis_manager]

        # Opponent breakdown
        opp_breakdown = manager_games.groupby("opponent").agg({
            "opponent_points": ["mean", "sum", "count"],
            "win": ["sum", "count"]
        }).round(2)

        opp_breakdown.columns = ["Avg Opp Pts", "Total Opp Pts", "Games Played", "Wins", "Games"]
        opp_breakdown["Win %"] = (opp_breakdown["Wins"] / opp_breakdown["Games"] * 100).round(1)
        opp_breakdown = opp_breakdown.reset_index()
        opp_breakdown = opp_breakdown.sort_values("Avg Opp Pts", ascending=False)

        # Bar chart of opponents faced
        fig_opp = go.Figure()

        fig_opp.add_trace(go.Bar(
            x=opp_breakdown['opponent'],
            y=opp_breakdown['Avg Opp Pts'],
            marker=dict(
                color=opp_breakdown['Avg Opp Pts'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Avg Pts")
            ),
            text=opp_breakdown['Avg Opp Pts'],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Avg Points: %{y:.1f}<br>Games: " +
                         opp_breakdown['Games Played'].astype(str) + "<extra></extra>"
        ))

        fig_opp.update_layout(
            xaxis_title="Opponent",
            yaxis_title="Average Opponent Points",
            height=450,
            template="plotly_white",
            showlegend=False
        )

        fig_opp.update_xaxes(tickangle=-45)

        st.plotly_chart(fig_opp, use_container_width=True, key=f"{prefix}_opponents")

        # Record vs each opponent
        st.subheader(f"{analysis_manager}'s Record vs Each Opponent")

        st.dataframe(
            opp_breakdown,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Win %": st.column_config.NumberColumn(format="%.1f%%")
            }
        )

    with tab3:
        st.subheader("📈 Schedule Strength vs Win Rate")
        st.caption("Does a tougher schedule correlate with fewer wins?")

        # Scatter plot: Avg Opp Points vs Win Rate
        fig_scatter = go.Figure()

        fig_scatter.add_trace(go.Scatter(
            x=sos_df['Avg Opp Points'],
            y=sos_df['Win Rate'],
            mode='markers+text',
            text=sos_df['Manager'],
            textposition='top center',
            marker=dict(
                size=sos_df['Total Games'] / 2,
                color=sos_df['Avg Points For'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Avg PPG"),
                line=dict(width=1, color='white')
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Avg Opp Points: %{x:.1f}<br>"
                "Win Rate: %{y:.1f}%<br>"
                "<extra></extra>"
            )
        ))

        # Add reference lines
        fig_scatter.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3)
        fig_scatter.add_vline(x=league_avg_opp_pts, line_dash="dash", line_color="gray", opacity=0.3)

        # Add quadrant labels
        fig_scatter.add_annotation(
            text="🏆 Elite<br>(Wins despite tough schedule)",
            xref="paper", yref="paper",
            x=0.85, y=0.95,
            showarrow=False,
            bgcolor="rgba(144,238,144,0.3)",
            borderpad=4
        )

        fig_scatter.add_annotation(
            text="💪 Dominance<br>(Wins with easy schedule)",
            xref="paper", yref="paper",
            x=0.15, y=0.95,
            showarrow=False,
            bgcolor="rgba(255,255,0,0.2)",
            borderpad=4
        )

        fig_scatter.add_annotation(
            text="⚠️ Struggling<br>(Tough schedule, few wins)",
            xref="paper", yref="paper",
            x=0.85, y=0.05,
            showarrow=False,
            bgcolor="rgba(255,0,0,0.2)",
            borderpad=4
        )

        fig_scatter.add_annotation(
            text="😬 Underperforming<br>(Easy schedule, still losing)",
            xref="paper", yref="paper",
            x=0.15, y=0.05,
            showarrow=False,
            bgcolor="rgba(255,165,0,0.3)",
            borderpad=4
        )

        fig_scatter.update_layout(
            xaxis_title="Average Opponent Points (Higher = Harder Schedule)",
            yaxis_title="Win Rate (%)",
            height=550,
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_scatter, use_container_width=True, key=f"{prefix}_scatter")

        # Calculate correlation between SOS and Win Rate
        st.subheader("📊 Schedule Impact Analysis")
        st.caption("Relationship between opponent strength and your success")

        # Simple correlation display
        correlation = sos_df[['Avg Opp Points', 'Win Rate']].corr().iloc[0, 1]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Schedule-Wins Correlation",
                f"{correlation:.3f}",
                "Negative = harder schedule hurts wins" if correlation < 0 else "Positive = harder schedule helps wins"
            )

        with col2:
            # Calculate how much harder/easier schedule affects record
            # For every 1 point increase in opponent strength, win rate changes by:
            if abs(correlation) > 0.1:
                st.metric(
                    "Schedule Impact",
                    "Moderate" if abs(correlation) > 0.3 else "Weak",
                    f"{abs(correlation)*100:.1f}% correlation strength"
                )
            else:
                st.metric(
                    "Schedule Impact",
                    "Minimal",
                    "Schedule doesn't significantly affect records"
                )

        # Show actual vs expected based on opponent strength
        st.subheader("🍀 Performance vs Schedule Difficulty")

        # Create a simple bar showing SOS vs Avg
        sos_comparison = sos_df.copy()
        sos_comparison = sos_comparison.sort_values("SOS (vs Avg)", ascending=True)

        fig_sos_comp = go.Figure()

        colors_comp = ['red' if x > 0 else 'lightblue' for x in sos_comparison['SOS (vs Avg)']]

        fig_sos_comp.add_trace(go.Bar(
            y=sos_comparison['Manager'],
            x=sos_comparison['SOS (vs Avg)'],
            orientation='h',
            marker_color=colors_comp,
            text=sos_comparison['SOS (vs Avg)'].apply(lambda x: f"{x:+.1f}"),
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>vs League Avg: %{x:+.1f} pts<extra></extra>"
        ))

        fig_sos_comp.add_vline(x=0, line_color="black", line_width=2)

        fig_sos_comp.update_layout(
            xaxis_title="Schedule Difficulty vs League Average (Points)",
            yaxis_title="",
            height=max(300, len(sos_df) * 40),
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_sos_comp, use_container_width=True, key=f"{prefix}_sos_comp")

        st.caption("""
        **Red bars** = Faced tougher opponents than average
        **Blue bars** = Faced easier opponents than average
        """)

    # Key insights
    with st.expander("💡 Key Insights", expanded=False):
        toughest_schedule = sos_df.nlargest(1, "Avg Opp Points").iloc[0]
        easiest_schedule = sos_df.nsmallest(1, "Avg Opp Points").iloc[0]

        # Find best win rate with tough schedule
        tough_winners = sos_df[sos_df["Avg Opp Points"] > league_avg_opp_pts].nlargest(1, "Win Rate")
        best_tough_winner = tough_winners.iloc[0] if len(tough_winners) > 0 else None

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "💪 Toughest Schedule",
                toughest_schedule["Manager"],
                f"{toughest_schedule['Avg Opp Points']:.1f} PPG faced"
            )

        with col2:
            if best_tough_winner is not None:
                st.metric(
                    "🏆 Best vs Tough Schedule",
                    best_tough_winner["Manager"],
                    f"{best_tough_winner['Win Rate']:.1f}% win rate"
                )
            else:
                st.metric(
                    "📊 League Average",
                    "",
                    f"{league_avg_opp_pts:.1f} PPG"
                )

        with col3:
            st.metric(
                "🍀 Easiest Schedule",
                easiest_schedule["Manager"],
                f"{easiest_schedule['Avg Opp Points']:.1f} PPG faced"
            )
