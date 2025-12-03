#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.core import T, list_seasons, run_query


@st.fragment
def display_win_percentage_graph(df_dict, prefix=""):
    """
    Display win percentage trends over time - both yearly and cumulative all-time.
    """
    st.header("📊 Win Percentage Trends")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Track winning trends:</strong> See how win percentages evolve year-over-year
    and cumulatively across your entire career.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📅 Year-by-Year", "📈 Cumulative Career", "🎯 Win/Loss Records"])

    # ==================== YEAR-BY-YEAR TAB ====================
    with tab1:
        st.subheader("Season Win Percentage by Year")

        # Load data by year
        with st.spinner("Loading win percentage data..."):
            query = f"""
                SELECT
                    year,
                    manager,
                    SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) as wins,
                    COUNT(*) as games
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND team_points IS NOT NULL
                GROUP BY year, manager
                ORDER BY year, manager
            """
            data = run_query(query)

            if data.empty:
                st.warning("No data found.")
                return

        data["win_pct"] = (data["wins"] / data["games"] * 100).round(1)

        # Manager selection
        managers = sorted(data["manager"].unique())
        selected_managers = st.multiselect(
            "Select Managers to Display",
            options=managers,
            default=managers[:3] if len(managers) >= 3 else managers,
            key=f"{prefix}_yearly_managers"
        )

        if not selected_managers:
            st.info("Please select at least one manager.")
            return

        filtered_data = data[data["manager"].isin(selected_managers)]

        # Create line chart
        fig = go.Figure()

        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager].sort_values("year")

            fig.add_trace(go.Scatter(
                x=manager_data["year"],
                y=manager_data["win_pct"],
                mode='lines+markers',
                name=manager,
                line=dict(width=3),
                marker=dict(size=10),
                hovertemplate=f"<b>{manager}</b><br>Year: %{{x}}<br>Win %: %{{y:.1f}}%<extra></extra>"
            ))

        # Add 50% reference line
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5,
                      annotation_text=".500 (50%)", annotation_position="right")

        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Win Percentage",
            hovermode="x unified",
            height=500,
            template="plotly_white",
            showlegend=True,
            yaxis=dict(range=[0, 100])
        )

        fig.update_xaxes(tickmode='linear', dtick=1)

        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_winpct_line")

        # Year-over-year comparison table
        with st.expander("📊 Year-over-Year Records", expanded=False):
            pivot = filtered_data.pivot(index="year", columns="manager", values="win_pct")
            pivot = pivot.round(1)
            st.dataframe(pivot, use_container_width=True)

        # Trend analysis
        with st.expander("📈 Trend Analysis", expanded=False):
            trends = []
            for manager in selected_managers:
                manager_data = filtered_data[filtered_data["manager"] == manager].sort_values("year")
                if len(manager_data) >= 2:
                    latest = manager_data.iloc[-1]["win_pct"]
                    previous = manager_data.iloc[-2]["win_pct"]
                    change = latest - previous

                    # Calculate overall trend (slope)
                    years = manager_data["year"].values
                    win_pcts = manager_data["win_pct"].values
                    if len(years) > 1:
                        slope = (win_pcts[-1] - win_pcts[0]) / (years[-1] - years[0])
                        trend = "📈 Improving" if slope > 0 else "📉 Declining" if slope < 0 else "➡️ Stable"
                    else:
                        trend = "➡️ Stable"

                    trends.append({
                        "Manager": manager,
                        "Latest Win %": f"{latest:.1f}%",
                        "YoY Change": f"{change:+.1f}%",
                        "Trend": trend
                    })

            if trends:
                trends_df = pd.DataFrame(trends)
                st.dataframe(trends_df, hide_index=True, use_container_width=True)

    # ==================== CUMULATIVE CAREER TAB ====================
    with tab2:
        st.subheader("All-Time Win Percentage Progression")
        st.caption("Track how your career win percentage moves week-by-week")

        # Load game-by-game data
        with st.spinner("Loading cumulative win data..."):
            cumulative_query = f"""
                SELECT
                    year,
                    week,
                    manager,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win,
                    ROW_NUMBER() OVER (PARTITION BY manager ORDER BY year, week) as game_number
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND team_points IS NOT NULL
                ORDER BY manager, year, week
            """
            cumulative_data = run_query(cumulative_query)

            if cumulative_data.empty:
                st.warning("No game data found.")
                return

        # Calculate running totals
        cumulative_data["cumulative_wins"] = cumulative_data.groupby("manager")["win"].cumsum()
        cumulative_data["cumulative_win_pct"] = (
            cumulative_data["cumulative_wins"] / cumulative_data["game_number"] * 100
        ).round(2)

        # Manager selection for cumulative view
        managers_cum = sorted(cumulative_data["manager"].unique())
        selected_managers_cum = st.multiselect(
            "Select Managers to Display",
            options=managers_cum,
            default=managers_cum[:3] if len(managers_cum) >= 3 else managers_cum,
            key=f"{prefix}_cumulative_managers"
        )

        if not selected_managers_cum:
            st.info("Please select at least one manager.")
            return

        filtered_cumulative = cumulative_data[cumulative_data["manager"].isin(selected_managers_cum)]

        # Create cumulative line chart
        fig_cumulative = go.Figure()

        for manager in selected_managers_cum:
            manager_cum_data = filtered_cumulative[filtered_cumulative["manager"] == manager]

            fig_cumulative.add_trace(go.Scatter(
                x=manager_cum_data["game_number"],
                y=manager_cum_data["cumulative_win_pct"],
                mode='lines',
                name=manager,
                line=dict(width=2),
                hovertemplate=(
                    f"<b>{manager}</b><br>"
                    "Game: %{x}<br>"
                    "Win %: %{y:.2f}%<br>"
                    "<extra></extra>"
                )
            ))

        # Add 50% reference line
        fig_cumulative.add_hline(
            y=50, line_dash="dash", line_color="gray", opacity=0.5,
            annotation_text=".500 (50%)", annotation_position="right"
        )

        fig_cumulative.update_layout(
            xaxis_title="Career Game Number",
            yaxis_title="Cumulative Win Percentage",
            hovermode="x unified",
            height=500,
            template="plotly_white",
            showlegend=True,
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig_cumulative, use_container_width=True, key=f"{prefix}_cumulative_line")

        # Current standings
        st.subheader("📊 Current All-Time Records")

        current_records = []
        for manager in selected_managers_cum:
            manager_final = filtered_cumulative[filtered_cumulative["manager"] == manager].iloc[-1]
            total_games = int(manager_final["game_number"])
            total_wins = int(manager_final["cumulative_wins"])
            total_losses = total_games - total_wins
            final_pct = manager_final["cumulative_win_pct"]

            current_records.append({
                "Manager": manager,
                "Record": f"{total_wins}-{total_losses}",
                "Games": total_games,
                "Win %": f"{final_pct:.2f}%"
            })

        records_df = pd.DataFrame(current_records).sort_values("Win %", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
        st.dataframe(records_df, hide_index=True, use_container_width=True)

        # Insight: Show who's trending up/down in last 20 games
        with st.expander("📈 Recent Form (Last 20 Games)", expanded=False):
            recent_form = []
            for manager in selected_managers_cum:
                manager_games = filtered_cumulative[filtered_cumulative["manager"] == manager]
                if len(manager_games) >= 20:
                    last_20 = manager_games.tail(20)
                    recent_wins = last_20["win"].sum()
                    recent_pct = (recent_wins / 20 * 100).round(1)

                    # Compare to overall
                    overall_pct = manager_games.iloc[-1]["cumulative_win_pct"]
                    diff = recent_pct - overall_pct

                    recent_form.append({
                        "Manager": manager,
                        "Last 20 Win %": f"{recent_pct:.1f}%",
                        "Career Win %": f"{overall_pct:.2f}%",
                        "Difference": f"{diff:+.1f}%",
                        "Form": "🔥 Hot" if diff > 5 else "❄️ Cold" if diff < -5 else "➡️ Average"
                    })

            if recent_form:
                form_df = pd.DataFrame(recent_form)
                st.dataframe(form_df, hide_index=True, use_container_width=True)
            else:
                st.info("Need at least 20 games for recent form analysis.")

    # ==================== WIN/LOSS RECORDS TAB ====================
    with tab3:
        st.subheader("Win/Loss Records Comparison")

        # Year selection for Win/Loss view
        available_years_wl = list_seasons()
        if not available_years_wl:
            st.error("No data available.")
            return

        year_options_wl = ["All Seasons"] + available_years_wl
        selected_year_wl = st.selectbox(
            "Select Season",
            options=year_options_wl,
            key=f"{prefix}_wl_year"
        )

        # Load win/loss data
        with st.spinner("Loading win/loss data..."):
            if selected_year_wl == "All Seasons":
                wl_query = f"""
                    SELECT
                        manager,
                        SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN team_points <= opponent_points THEN 1 ELSE 0 END) as losses,
                        COUNT(*) as games
                    FROM {T['matchup']}
                    WHERE manager IS NOT NULL
                      AND team_points IS NOT NULL
                    GROUP BY manager
                    ORDER BY wins DESC
                """
            else:
                wl_query = f"""
                    SELECT
                        manager,
                        SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN team_points <= opponent_points THEN 1 ELSE 0 END) as losses,
                        COUNT(*) as games
                    FROM {T['matchup']}
                    WHERE year = {int(selected_year_wl)}
                      AND manager IS NOT NULL
                      AND team_points IS NOT NULL
                    GROUP BY manager
                    ORDER BY wins DESC
                """
            wl_data = run_query(wl_query)

            if wl_data.empty:
                st.warning("No data found.")
                return

        wl_data["win_pct"] = (wl_data["wins"] / wl_data["games"] * 100).round(1)

        # Stacked bar chart for W/L
        st.subheader("Total Wins and Losses")

        wl_sorted = wl_data.sort_values("wins", ascending=True)

        fig_wl_stack = go.Figure()

        fig_wl_stack.add_trace(go.Bar(
            y=wl_sorted["manager"],
            x=wl_sorted["wins"],
            name='Wins',
            orientation='h',
            marker=dict(color='green'),
            text=wl_sorted["wins"],
            textposition='inside',
            hovertemplate="<b>%{y}</b><br>Wins: %{x}<extra></extra>"
        ))

        fig_wl_stack.add_trace(go.Bar(
            y=wl_sorted["manager"],
            x=wl_sorted["losses"],
            name='Losses',
            orientation='h',
            marker=dict(color='red'),
            text=wl_sorted["losses"],
            textposition='inside',
            hovertemplate="<b>%{y}</b><br>Losses: %{x}<extra></extra>"
        ))

        fig_wl_stack.update_layout(
            barmode='stack',
            xaxis_title="Games",
            yaxis_title="",
            height=max(400, len(wl_data) * 35),
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_wl_stack, use_container_width=True, key=f"{prefix}_wl_stack")

        # Win percentage bar chart
        st.subheader("Win Percentage Comparison")

        wl_sorted_pct = wl_data.sort_values("win_pct", ascending=True)

        fig_wl_pct = go.Figure()

        fig_wl_pct.add_trace(go.Bar(
            y=wl_sorted_pct["manager"],
            x=wl_sorted_pct["win_pct"],
            orientation='h',
            marker=dict(
                color=wl_sorted_pct["win_pct"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Win %")
            ),
            text=wl_sorted_pct["win_pct"].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Win %: %{x:.1f}%<extra></extra>"
        ))

        fig_wl_pct.update_layout(
            xaxis_title="Win Percentage",
            yaxis_title="",
            height=max(400, len(wl_data) * 35),
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_wl_pct, use_container_width=True, key=f"{prefix}_wl_pct")

        # Detailed records table
        with st.expander("📊 Detailed Records", expanded=False):
            display_df = wl_data[["manager", "wins", "losses", "games", "win_pct"]].copy()
            display_df.columns = ["Manager", "Wins", "Losses", "Games", "Win %"]
            display_df = display_df.sort_values("Wins", ascending=False)

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Win %": st.column_config.NumberColumn(format="%.1f%%")
                }
            )
