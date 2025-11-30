#!/usr/bin/env python3
"""
playoff_odds.py - Enhanced playoff odds visualization
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class PlayoffOddsViewer:
    def __init__(self, matchup_data_df):
        self.df = matchup_data_df.copy()

    @st.fragment
    def display(self):
        st.subheader("🏆 Playoff Odds & Championship Probability")

        st.info("Monte Carlo simulation of remaining games to project playoff and championship odds")

        # Filter out consolation games
        df = self.df[self.df['is_consolation'] == 0].copy()

        seasons = sorted(df["year"].unique())
        max_season = seasons[-1] if seasons else None

        if max_season is None:
            st.info("No season data available.")
            return

        # Selection mode
        mode = st.radio(
            "Selection Mode",
            ["Today's Date", "Specific Week"],
            horizontal=True,
            key="playoff_odds_mode",
            index=0
        )

        show_results = False

        if mode == "Today's Date":
            season = max_season
            all_weeks = sorted(df[df["year"] == season]["week"].unique())
            week = all_weeks[-1] if all_weeks else None
            st.caption(f"📅 Auto-selected Year {season}, Week {week}")
            show_results = True
        else:
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                season = st.selectbox(
                    "Select Season",
                    seasons,
                    index=len(seasons) - 1,
                    key="playoff_odds_season"
                )

            with col2:
                all_weeks = sorted(df[df["year"] == season]["week"].unique())
                week = st.selectbox(
                    "Select Week",
                    all_weeks,
                    index=len(all_weeks) - 1 if all_weeks else 0,
                    key="playoff_odds_week"
                )

            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                show_results = st.button("Go", key="go_specific")

        if week is None:
            st.info("No week data available.")
            return

        if not show_results:
            return

        # Get odds data
        odds = df[
            (df['year'] == season) &
            (df['is_playoffs'] == 0) &
            (df['week'] == week)
            ]

        odds_cols = [
            "avg_seed", "manager", "p_playoffs", "p_bye", "exp_final_wins",
            "exp_final_pf", "p_semis", "p_final", "p_champ"
        ]

        # Filter to only existing columns
        odds_cols = [c for c in odds_cols if c in odds.columns]
        odds_table = odds[odds_cols].sort_values("avg_seed", ascending=True).reset_index(drop=True)

        if len(odds_table) == 0:
            st.warning("No data available for this selection.")
            return

        st.markdown("---")
        st.markdown("### 📊 Simulation Results")

        # Summary metrics - responsive layout
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            avg_playoff_pct = odds_table['p_playoffs'].mean() if 'p_playoffs' in odds_table.columns else 0
            st.metric("Avg Playoff %", f"{avg_playoff_pct:.1f}%")

        with col2:
            if 'p_champ' in odds_table.columns:
                favorite = odds_table.loc[odds_table['p_champ'].idxmax(), 'manager']
                fav_odds = odds_table['p_champ'].max()
                st.metric("Championship Favorite", favorite, f"{fav_odds:.1f}%")

        with col3:
            if 'exp_final_wins' in odds_table.columns:
                avg_wins = odds_table['exp_final_wins'].mean()
                st.metric("Avg Projected Wins", f"{avg_wins:.1f}")

        with col4:
            if 'p_bye' in odds_table.columns:
                bye_count = (odds_table['p_bye'] > 50).sum()
                st.metric("Teams >50% Bye Odds", bye_count)

        st.markdown("---")

        # Data table with styling
        styled_table = odds_table.style.background_gradient(
            cmap='PuBuGn',
            subset=[c for c in odds_cols if c not in ['manager', 'avg_seed']]
        ).format({
            'p_playoffs': '{:.1f}%',
            'p_bye': '{:.1f}%',
            'p_semis': '{:.1f}%',
            'p_final': '{:.1f}%',
            'p_champ': '{:.1f}%',
            'exp_final_wins': '{:.2f}',
            'exp_final_pf': '{:.1f}',
            'avg_seed': '{:.2f}'
        })

        st.dataframe(styled_table, hide_index=True, use_container_width=True)

        # Visualizations
        st.markdown("---")
        show_viz = st.checkbox("📊 Show Visualizations", value=True, key="show_playoff_viz")

        if not show_viz:
            return

        # Probability breakdown chart
        st.markdown("### 📊 Probability Breakdown")

        fig_playoff = go.Figure()

        prob_cols = ['p_playoffs', 'p_bye', 'p_semis', 'p_final', 'p_champ']
        colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#2ecc71']
        names = ['Playoff', 'Bye Week', 'Semifinals', 'Finals', 'Champion']

        for col, color, name in zip(prob_cols, colors, names):
            if col in odds_table.columns:
                fig_playoff.add_trace(go.Bar(
                    name=name,
                    x=odds_table['manager'],
                    y=odds_table[col],
                    marker_color=color,
                    text=odds_table[col].round(1),
                    texttemplate='%{text}%',
                    textposition='auto',
                ))

        fig_playoff.update_layout(
            title="Playoff Probability Breakdown by Manager",
            xaxis_title="Manager",
            yaxis_title="Probability (%)",
            barmode='group',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_playoff, use_container_width=True)

        # Championship distribution and correlation charts
        if 'p_champ' in odds_table.columns:
            st.markdown("### 🏆 Championship Analysis")

            col1, col2 = st.columns([1, 1])

            with col1:
                fig_champ = go.Figure(data=[go.Pie(
                    labels=odds_table['manager'],
                    values=odds_table['p_champ'],
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Set3)
                )])
                fig_champ.update_layout(
                    title="Championship Probability Distribution",
                    height=400
                )
                st.plotly_chart(fig_champ, use_container_width=True)

            with col2:
                if 'exp_final_wins' in odds_table.columns:
                    fig_scatter = go.Figure(data=[go.Scatter(
                        x=odds_table['exp_final_wins'],
                        y=odds_table['p_playoffs'],
                        mode='markers+text',
                        text=odds_table['manager'],
                        textposition='top center',
                        marker=dict(
                            size=odds_table['p_champ'] * 2,
                            color=odds_table['p_champ'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Champ %")
                        ),
                        hovertemplate='<b>%{text}</b><br>Exp Wins: %{x:.1f}<br>Playoff %: %{y:.1f}%<extra></extra>'
                    )])
                    fig_scatter.update_layout(
                        title="Expected Wins vs Playoff Odds",
                        xaxis_title="Expected Final Wins",
                        yaxis_title="Playoff Probability (%)",
                        height=400,
                        font=dict(size=12),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)