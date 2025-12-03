import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


@st.fragment
def display_season_add_drop(
    transaction_df: pd.DataFrame,
    player_df: pd.DataFrame,
    return_df: bool = False,
    key_prefix: str = "season_adddrop",
):
    """Enhanced season add/drop with actual enriched columns."""
    t = transaction_df[transaction_df["transaction_type"].isin(["add", "drop"])].copy()

    if "year" in t.columns:
        t["year"] = t["year"].astype(str)

    # Aggregate by manager and year
    adds = t[t["transaction_type"] == "add"]
    drops = t[t["transaction_type"] == "drop"]

    # Aggregate adds (using managed SPAR - what you actually got)
    add_agg_cols = {
        "transaction_id": "count",
        "faab_bid": "sum",
        "manager_spar_ros_managed": "sum",
        "total_points_ros_managed": "sum",
        "ppg_ros_managed": "mean",
        "spar_efficiency": "mean",
        "points_per_faab_dollar": "mean",
        "transaction_quality_score": "mean",
        "position_rank_at_transaction": "mean",
        "position_rank_after_transaction": "mean",
    }
    # Add transaction_score if available
    if "transaction_score" in adds.columns:
        add_agg_cols["transaction_score"] = "sum"

    add_agg = adds.groupby(["manager", "year"]).agg(add_agg_cols).reset_index()

    col_names = [
        "manager",
        "year",
        "num_adds",
        "faab_spent",
        "spar_added_managed",
        "points_added_managed",
        "avg_ppg_managed",
        "avg_spar_efficiency",
        "avg_pts_per_faab",
        "avg_score",
        "avg_rank_at",
        "avg_rank_after",
    ]
    if "transaction_score" in adds.columns:
        col_names.append("add_score")
    add_agg.columns = col_names

    # Aggregate drops (using total SPAR - opportunity cost of what you lost)
    drop_agg_cols = {
        "transaction_id": "count",
        "player_spar_ros_total": "sum",
        "total_points_ros_total": "sum",
        "ppg_ros_total": "mean",
    }
    # Add transaction_score for drops (will be negative for regrets)
    if "transaction_score" in drops.columns:
        drop_agg_cols["transaction_score"] = "sum"

    drop_agg = drops.groupby(["manager", "year"]).agg(drop_agg_cols).reset_index()

    drop_col_names = [
        "manager",
        "year",
        "num_drops",
        "spar_dropped_total",
        "points_dropped_total",
        "avg_ppg_dropped",
    ]
    if "transaction_score" in drops.columns:
        drop_col_names.append("drop_score")
    drop_agg.columns = drop_col_names

    # Merge
    season_agg = add_agg.merge(drop_agg, on=["manager", "year"], how="outer").fillna(0)
    season_agg["total_transactions"] = season_agg["num_adds"] + season_agg["num_drops"]

    # NET SPAR: What you gained (managed) vs what you lost (total opportunity)
    season_agg["net_spar"] = (
        season_agg["spar_added_managed"] - season_agg["spar_dropped_total"]
    )
    season_agg["net_points"] = (
        season_agg["points_added_managed"] - season_agg["points_dropped_total"]
    )
    season_agg["rank_improvement"] = (
        season_agg["avg_rank_at"] - season_agg["avg_rank_after"]
    )

    # Total Score: Weighted aggregate (adds + drops, where drops are negative for regrets)
    if "add_score" in season_agg.columns and "drop_score" in season_agg.columns:
        season_agg["total_score"] = season_agg["add_score"] + season_agg["drop_score"]
    elif "add_score" in season_agg.columns:
        season_agg["total_score"] = season_agg["add_score"]
    else:
        season_agg["total_score"] = season_agg["net_spar"]  # Fallback

    if return_df:
        return season_agg

    # Grades (based on Total Score - our weighted metric)
    season_agg["Grade"] = season_agg["total_score"].apply(
        lambda x: (
            "🥇 Elite"
            if x > 200
            else (
                "🥈 Great"
                if x > 100
                else "🥉 Good" if x > 50 else "📊 Average" if x > 0 else "📉 Poor"
            )
        )
    )

    # Summary
    total_seasons = len(season_agg)
    avg_transactions = season_agg["total_transactions"].mean()
    avg_faab = season_agg["faab_spent"].mean()
    avg_net_spar = season_agg["net_spar"].mean()
    avg_total_score = season_agg["total_score"].mean()
    best = (
        season_agg.nlargest(1, "total_score")[
            ["manager", "year", "total_score", "net_spar"]
        ].iloc[0]
        if len(season_agg) > 0
        else None
    )

    # Compact CSS for season transactions
    st.markdown(
        """
    <style>
    .season-stat-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    .season-stat-card h4 {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.6;
        margin: 0 0 0.5rem 0;
    }
    .season-stat-row {
        display: flex;
        justify-content: space-around;
    }
    .season-stat-item { text-align: center; }
    .season-stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4ade80;
    }
    .season-stat-value.neutral { color: #94a3b8; }
    .season-stat-value.faab { color: #fbbf24; }
    .season-stat-label { font-size: 0.65rem; opacity: 0.7; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Season Add/Drop Summary")

    tab1, tab2, tab3 = st.tabs(["📋 Season Stats", "📈 Analytics", "🏆 Rankings"])

    with tab1:
        # Grouped stats in cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            <div class="season-stat-card">
                <h4>📊 Overview</h4>
                <div class="season-stat-row">
                    <div class="season-stat-item">
                        <div class="season-stat-value neutral">{}</div>
                        <div class="season-stat-label">Seasons</div>
                    </div>
                    <div class="season-stat-item">
                        <div class="season-stat-value neutral">{:.1f}</div>
                        <div class="season-stat-label">Avg Moves</div>
                    </div>
                </div>
            </div>
            """.format(
                    total_seasons, avg_transactions
                ),
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="season-stat-card">
                <h4>💰 FAAB</h4>
                <div class="season-stat-row">
                    <div class="season-stat-item">
                        <div class="season-stat-value faab">${:.0f}</div>
                        <div class="season-stat-label">Avg/Season</div>
                    </div>
                </div>
            </div>
            """.format(
                    avg_faab
                ),
                unsafe_allow_html=True,
            )

        with col3:
            net_class = "" if avg_net_spar >= 0 else "negative"
            st.markdown(
                """
            <div class="season-stat-card">
                <h4>📈 Performance</h4>
                <div class="season-stat-row">
                    <div class="season-stat-item">
                        <div class="season-stat-value">{:.0f}</div>
                        <div class="season-stat-label">Avg Score</div>
                    </div>
                    <div class="season-stat-item">
                        <div class="season-stat-value {}">{:+.1f}</div>
                        <div class="season-stat-label">Avg Net SPAR</div>
                    </div>
                </div>
            </div>
            """.format(
                    avg_total_score, net_class, avg_net_spar
                ),
                unsafe_allow_html=True,
            )

        if best is not None:
            st.success(
                f"🏆 **Best:** {best['manager']} ({best['year']}) - {best['total_score']:.0f} Score, {best['net_spar']:.1f} Net SPAR"
            )

        st.markdown(
            "<div style='margin: 0.5rem 0; border-top: 1px solid rgba(100,116,139,0.3);'></div>",
            unsafe_allow_html=True,
        )

        # Compact filters in one row
        col1, col2, col3, col4 = st.columns([1.5, 2, 1.5, 2])
        with col1:
            year_filter = st.selectbox(
                "Year",
                ["All"] + sorted(season_agg["year"].unique().tolist(), reverse=True),
                key=f"{key_prefix}_year",
                label_visibility="collapsed",
            )
        with col2:
            manager_search = st.text_input(
                "Manager",
                placeholder="Search manager...",
                key=f"{key_prefix}_mgr",
                label_visibility="collapsed",
            )
        with col3:
            grade_filter = st.selectbox(
                "Grade",
                ["All", "Elite", "Great", "Good", "Average", "Poor"],
                key=f"{key_prefix}_grade",
                label_visibility="collapsed",
            )
        with col4:
            sort_by = st.selectbox(
                "Sort",
                [
                    "Total Score",
                    "Net SPAR",
                    "Net Points",
                    "Transactions",
                    "FAAB",
                    "SPAR Efficiency",
                    "Recent",
                ],
                key=f"{key_prefix}_sort",
                label_visibility="collapsed",
            )

        filtered = season_agg.copy()
        if year_filter != "All":
            filtered = filtered[filtered["year"] == year_filter]
        if manager_search:
            filtered = filtered[
                filtered["manager"].str.contains(manager_search, case=False, na=False)
            ]
        if grade_filter != "All":
            filtered = filtered[
                filtered["Grade"].str.contains(grade_filter, case=False)
            ]

        if sort_by == "Total Score":
            filtered = filtered.sort_values("total_score", ascending=False)
        elif sort_by == "Net SPAR":
            filtered = filtered.sort_values("net_spar", ascending=False)
        elif sort_by == "Net Points":
            filtered = filtered.sort_values("net_points", ascending=False)
        elif sort_by == "Transactions":
            filtered = filtered.sort_values("total_transactions", ascending=False)
        elif sort_by == "FAAB":
            filtered = filtered.sort_values("faab_spent", ascending=False)
        elif sort_by == "SPAR Efficiency":
            filtered = filtered.sort_values("avg_spar_efficiency", ascending=False)
        else:
            filtered = filtered.sort_values("year", ascending=False)

        st.markdown(f"**Showing {len(filtered)} of {len(season_agg)} seasons**")

        display_cols = [
            "manager",
            "year",
            "Grade",
            "total_score",
            "total_transactions",
            "faab_spent",
            "net_spar",
            "spar_added_managed",
            "spar_dropped_total",
            "avg_spar_efficiency",
            "rank_improvement",
        ]
        # Only include columns that exist
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "total_score": st.column_config.NumberColumn(
                    "Score",
                    format="%+.0f",
                    help="Weighted score (timing, FAAB efficiency, regret)",
                ),
                "total_transactions": st.column_config.NumberColumn(
                    "Moves", format="%d"
                ),
                "faab_spent": st.column_config.NumberColumn("FAAB", format="$%.0f"),
                "net_spar": st.column_config.NumberColumn(
                    "Net SPAR",
                    format="%.1f",
                    help="Managed SPAR Added - Total SPAR Dropped",
                ),
                "spar_added_managed": st.column_config.NumberColumn(
                    "SPAR Added", format="%.1f", help="SPAR from adds while rostered"
                ),
                "spar_dropped_total": st.column_config.NumberColumn(
                    "SPAR Dropped", format="%.1f", help="Total SPAR opportunity cost"
                ),
                "avg_spar_efficiency": st.column_config.NumberColumn(
                    "SPAR/FAAB", format="%.2f", help="SPAR per FAAB dollar"
                ),
                "rank_improvement": st.column_config.NumberColumn(
                    "Rank Δ", format="%.1f"
                ),
            },
        )

        csv = filtered[display_cols].to_csv(index=False)
        st.download_button(
            "📥 Export",
            csv,
            f"season_add_drops_{year_filter}.csv",
            "text/csv",
            key=f"{key_prefix}_export",
        )

    with tab2:
        st.markdown("### 📈 Season Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Net SPAR by Manager")
            mgr_spar = (
                season_agg.groupby("manager")["net_spar"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig = px.bar(
                mgr_spar,
                x="manager",
                y="net_spar",
                title="Top 10 Managers by Career Net SPAR",
                color="net_spar",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### SPAR Efficiency")
            eff_data = season_agg[season_agg["faab_spent"] > 0].copy()
            top_eff = eff_data.nlargest(10, "avg_spar_efficiency")
            fig = px.bar(
                top_eff,
                x="manager",
                y="avg_spar_efficiency",
                hover_data=["year", "faab_spent", "net_spar"],
                title="Most Efficient FAAB Seasons (SPAR/$)",
                color="avg_spar_efficiency",
                color_continuous_scale="Greens",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Season Trends")
        yearly = (
            season_agg.groupby("year")
            .agg(
                {
                    "net_spar": "mean",
                    "net_points": "mean",
                    "total_transactions": "mean",
                    "faab_spent": "mean",
                }
            )
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=yearly["year"],
                y=yearly["net_spar"],
                mode="lines+markers",
                name="Avg Net SPAR",
                line=dict(color="green", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=yearly["year"],
                y=yearly["total_transactions"],
                mode="lines+markers",
                name="Avg Transactions",
                yaxis="y2",
                line=dict(color="blue", width=2),
            )
        )
        fig.update_layout(
            title="League Trends Over Time",
            yaxis=dict(title="Avg Net SPAR"),
            yaxis2=dict(title="Avg Transactions", overlaying="y", side="right"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 🏆 Season Rankings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Best Seasons")
            best_seasons = season_agg.nlargest(10, "net_points")[
                ["manager", "year", "net_points", "total_transactions", "faab_spent"]
            ]
            best_seasons["Rank"] = range(1, len(best_seasons) + 1)
            st.dataframe(
                best_seasons[
                    [
                        "Rank",
                        "manager",
                        "year",
                        "net_points",
                        "total_transactions",
                        "faab_spent",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "net_points": st.column_config.NumberColumn(
                        "Net Pts", format="%.1f"
                    ),
                    "total_transactions": st.column_config.NumberColumn(
                        "Moves", format="%d"
                    ),
                    "faab_spent": st.column_config.NumberColumn("FAAB", format="$%.0f"),
                },
            )

        with col2:
            st.markdown("#### Most Efficient Seasons (SPAR/FAAB)")
            efficient = season_agg[season_agg["avg_spar_efficiency"].notna()].nlargest(
                10, "avg_spar_efficiency"
            )[["manager", "year", "avg_spar_efficiency", "net_spar", "faab_spent"]]
            efficient["Rank"] = range(1, len(efficient) + 1)
            st.dataframe(
                efficient[
                    [
                        "Rank",
                        "manager",
                        "year",
                        "avg_spar_efficiency",
                        "net_spar",
                        "faab_spent",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "avg_spar_efficiency": st.column_config.NumberColumn(
                        "SPAR/FAAB", format="%.2f", help="SPAR per FAAB dollar"
                    ),
                    "net_spar": st.column_config.NumberColumn(
                        "Net SPAR", format="%.1f"
                    ),
                    "faab_spent": st.column_config.NumberColumn("FAAB", format="$%.0f"),
                },
            )
