import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


@st.fragment
def display_season_trade_data(transaction_df, player_df, draft_history_df):
    """Enhanced season trade summary using managed SPAR metrics."""

    if "trade_summary_df" not in st.session_state:
        st.warning("Please view Trade Summaries tab first to generate data.")
        return

    trade_df = st.session_state["trade_summary_df"].copy()

    # Season aggregation using SPAR metrics
    season_agg = (
        trade_df.groupby(["manager", "year"])
        .agg(
            {
                "transaction_id": "count",
                "net_spar": ["sum", "mean"],
                "spar_managed": "sum",
                "partner_spar": "sum",
                "trade_score": "mean",
                "rank_improvement": "mean",
            }
        )
        .reset_index()
    )

    season_agg.columns = [
        "manager",
        "year",
        "total_trades",
        "total_net_spar",
        "avg_net_per_trade",
        "total_spar_acquired",
        "total_spar_traded",
        "avg_trade_score",
        "avg_rank_improvement",
    ]

    season_agg["trade_efficiency"] = (
        season_agg["total_net_spar"] / season_agg["total_trades"]
    )
    season_agg["win_rate"] = (
        trade_df.groupby(["manager", "year"])["net_spar"]
        .apply(lambda x: (x > 0).sum() / len(x) * 100)
        .values
    )

    # Grades based on NET SPAR
    season_agg["Grade"] = season_agg["total_net_spar"].apply(
        lambda x: (
            "🥇 Elite"
            if pd.notna(x) and x > 200
            else (
                "🥈 Great"
                if pd.notna(x) and x > 100
                else (
                    "🥉 Good"
                    if pd.notna(x) and x > 50
                    else (
                        "📊 Average"
                        if pd.notna(x) and x > 0
                        else "📉 Poor" if pd.notna(x) else ""
                    )
                )
            )
        )
    )

    st.markdown("### 📊 Season Trade Summary")
    st.markdown("*Using Managed SPAR: actual value from players on your roster*")

    tab1, tab2, tab3 = st.tabs(["📋 Season Stats", "📈 Analytics", "🏆 Rankings"])

    with tab1:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Season Records", len(season_agg))
        with col2:
            st.metric("Avg Trades/Season", f"{season_agg['total_trades'].mean():.1f}")
        with col3:
            st.metric(
                "Avg NET SPAR/Season", f"{season_agg['total_net_spar'].mean():.1f}"
            )
        with col4:
            st.metric("Avg Win Rate", f"{season_agg['win_rate'].mean():.1f}%")
        with col5:
            st.metric(
                "Avg Efficiency",
                f"{season_agg['trade_efficiency'].mean():.1f} SPAR/trade",
            )

        st.divider()

        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            year_filter = st.selectbox(
                "Year",
                ["All"] + sorted(season_agg["year"].unique().tolist(), reverse=True),
                key="season_trade_year",
            )
        with col2:
            manager_search = st.text_input("Manager", key="season_trade_mgr")
        with col3:
            grade_filter = st.selectbox(
                "Grade",
                ["All", "Elite", "Great", "Good", "Average", "Poor"],
                key="season_trade_grade",
            )
        with col4:
            sort_by = st.selectbox(
                "Sort by",
                ["NET SPAR ↓", "Efficiency", "Trades", "Win Rate"],
                key="season_trade_sort",
            )

        filtered = season_agg.copy()
        if year_filter != "All":
            filtered = filtered[filtered["year"].astype(str) == str(year_filter)]
        if manager_search:
            filtered = filtered[
                filtered["manager"].str.contains(manager_search, case=False, na=False)
            ]
        if grade_filter != "All":
            filtered = filtered[filtered["Grade"].str.contains(grade_filter)]

        if sort_by == "NET SPAR ↓":
            filtered = filtered.sort_values("total_net_spar", ascending=False)
        elif sort_by == "Efficiency":
            filtered = filtered.sort_values("trade_efficiency", ascending=False)
        elif sort_by == "Trades":
            filtered = filtered.sort_values("total_trades", ascending=False)
        else:
            filtered = filtered.sort_values("win_rate", ascending=False)

        st.markdown(f"**Showing {len(filtered)} of {len(season_agg)} season records**")

        display_cols = [
            "manager",
            "year",
            "Grade",
            "total_trades",
            "total_net_spar",
            "avg_net_per_trade",
            "trade_efficiency",
            "win_rate",
            "total_spar_acquired",
            "total_spar_traded",
            "avg_trade_score",
            "avg_rank_improvement",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "year": st.column_config.TextColumn("Year", width="small"),
                "Grade": st.column_config.TextColumn("Grade", width="medium"),
                "total_trades": st.column_config.NumberColumn(
                    "Trades", format="%d", width="small"
                ),
                "total_net_spar": st.column_config.NumberColumn(
                    "NET SPAR", format="%.1f", help="Total NET SPAR for season"
                ),
                "avg_net_per_trade": st.column_config.NumberColumn(
                    "Avg NET", format="%.1f", help="Average NET SPAR per trade"
                ),
                "trade_efficiency": st.column_config.NumberColumn(
                    "Efficiency", format="%.2f", help="NET SPAR per trade"
                ),
                "win_rate": st.column_config.NumberColumn("Win %", format="%.1f"),
                "total_spar_acquired": st.column_config.NumberColumn(
                    "SPAR+", format="%.1f", help="Total SPAR acquired"
                ),
                "total_spar_traded": st.column_config.NumberColumn(
                    "SPAR-", format="%.1f", help="Total SPAR traded away"
                ),
                "avg_trade_score": st.column_config.NumberColumn(
                    "Avg Score", format="%.1f", width="small"
                ),
                "avg_rank_improvement": st.column_config.NumberColumn(
                    "Rank Δ", format="%.1f", width="small"
                ),
            },
        )

        csv = filtered[display_cols].to_csv(index=False)
        st.download_button(
            "📥 Export",
            csv,
            f"season_trades_{year_filter}.csv",
            "text/csv",
            key="season_trade_export",
        )

    with tab2:
        st.markdown("### 📈 Season Trade Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Trade Performance by Season")
            yearly = (
                season_agg.groupby("year")
                .agg(
                    {
                        "total_net_spar": "mean",
                        "total_trades": "mean",
                        "win_rate": "mean",
                    }
                )
                .reset_index()
            )
            yearly.columns = ["Year", "Avg NET SPAR", "Avg Trades", "Avg Win Rate"]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=yearly["Year"],
                    y=yearly["Avg NET SPAR"],
                    name="Avg NET SPAR",
                    marker_color=[
                        "#2ca02c" if x > 0 else "#d62728"
                        for x in yearly["Avg NET SPAR"]
                    ],
                    text=yearly["Avg NET SPAR"].round(1),
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Average NET SPAR by Season",
                xaxis_title="Season",
                yaxis_title="Avg NET SPAR",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Win Rate Distribution")
            fig = px.histogram(
                season_agg,
                x="win_rate",
                title="Distribution of Season Win Rates",
                labels={"win_rate": "Win Rate (%)"},
                nbins=20,
                color_discrete_sequence=["#1f77b4"],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Manager Trade Profiles")
        mgr_profile = (
            season_agg.groupby("manager")
            .agg(
                {
                    "total_trades": "sum",
                    "total_net_spar": "sum",
                    "trade_efficiency": "mean",
                    "win_rate": "mean",
                }
            )
            .reset_index()
            .sort_values("total_net_spar", ascending=False)
            .head(10)
        )

        fig = px.scatter(
            mgr_profile,
            x="total_trades",
            y="total_net_spar",
            size=[abs(x) + 10 for x in mgr_profile["trade_efficiency"]],
            color="win_rate",
            hover_data=["manager"],
            title="Top 10 Managers: Volume vs Results",
            color_continuous_scale="RdYlGn",
            labels={
                "total_trades": "Total Trades",
                "total_net_spar": "Career NET SPAR",
            },
            size_max=35,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 🏆 Season Trade Rankings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Best Trading Seasons (NET SPAR)")
            best = season_agg.nlargest(10, "total_net_spar")[
                ["manager", "year", "total_net_spar", "total_trades", "win_rate"]
            ].copy()
            best["Rank"] = range(1, len(best) + 1)
            st.dataframe(
                best[
                    [
                        "Rank",
                        "manager",
                        "year",
                        "total_net_spar",
                        "total_trades",
                        "win_rate",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("#", width="small"),
                    "manager": st.column_config.TextColumn("Manager"),
                    "year": st.column_config.TextColumn("Year", width="small"),
                    "total_net_spar": st.column_config.NumberColumn(
                        "NET SPAR", format="%.1f"
                    ),
                    "total_trades": st.column_config.NumberColumn(
                        "Trades", format="%d", width="small"
                    ),
                    "win_rate": st.column_config.NumberColumn(
                        "Win %", format="%.1f", width="small"
                    ),
                },
            )

        with col2:
            st.markdown("#### Highest Win Rate Seasons")
            win_rate_seasons = (
                season_agg[season_agg["total_trades"] >= 3]
                .nlargest(10, "win_rate")[
                    ["manager", "year", "win_rate", "total_net_spar", "total_trades"]
                ]
                .copy()
            )
            win_rate_seasons["Rank"] = range(1, len(win_rate_seasons) + 1)
            st.dataframe(
                win_rate_seasons[
                    [
                        "Rank",
                        "manager",
                        "year",
                        "win_rate",
                        "total_net_spar",
                        "total_trades",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("#", width="small"),
                    "manager": st.column_config.TextColumn("Manager"),
                    "year": st.column_config.TextColumn("Year", width="small"),
                    "win_rate": st.column_config.NumberColumn("Win %", format="%.1f"),
                    "total_net_spar": st.column_config.NumberColumn(
                        "NET SPAR", format="%.1f", width="small"
                    ),
                    "total_trades": st.column_config.NumberColumn(
                        "Trades", format="%d", width="small"
                    ),
                },
            )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Most Efficient Seasons")
            efficient = (
                season_agg[season_agg["total_trades"] >= 3]
                .nlargest(10, "trade_efficiency")[
                    [
                        "manager",
                        "year",
                        "trade_efficiency",
                        "total_net_spar",
                        "total_trades",
                    ]
                ]
                .copy()
            )
            efficient["Rank"] = range(1, len(efficient) + 1)
            st.dataframe(
                efficient[
                    ["Rank", "manager", "year", "trade_efficiency", "total_net_spar"]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("#", width="small"),
                    "manager": st.column_config.TextColumn("Manager"),
                    "year": st.column_config.TextColumn("Year", width="small"),
                    "trade_efficiency": st.column_config.NumberColumn(
                        "SPAR/Trade", format="%.2f", help="NET SPAR per trade"
                    ),
                    "total_net_spar": st.column_config.NumberColumn(
                        "NET SPAR", format="%.1f"
                    ),
                },
            )

        with col2:
            st.markdown("#### Most Trades in a Season")
            most_active = season_agg.nlargest(10, "total_trades")[
                ["manager", "year", "total_trades", "total_net_spar", "win_rate"]
            ].copy()
            most_active["Rank"] = range(1, len(most_active) + 1)
            st.dataframe(
                most_active[
                    [
                        "Rank",
                        "manager",
                        "year",
                        "total_trades",
                        "total_net_spar",
                        "win_rate",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("#", width="small"),
                    "manager": st.column_config.TextColumn("Manager"),
                    "year": st.column_config.TextColumn("Year", width="small"),
                    "total_trades": st.column_config.NumberColumn(
                        "Trades", format="%d", help="Number of trades in the season"
                    ),
                    "total_net_spar": st.column_config.NumberColumn(
                        "NET SPAR", format="%.1f", width="small"
                    ),
                    "win_rate": st.column_config.NumberColumn(
                        "Win %", format="%.1f", width="small"
                    ),
                },
            )
