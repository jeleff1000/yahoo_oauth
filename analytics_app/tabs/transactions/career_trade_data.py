import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


@st.fragment
def display_career_trade_data(transaction_df, player_df, draft_history_df):
    """Enhanced career trade summary using managed SPAR metrics."""

    if 'trade_summary_df' not in st.session_state:
        st.warning("Please view Trade Summaries tab first to generate data.")
        return

    trade_df = st.session_state['trade_summary_df'].copy()

    # Career aggregation using SPAR metrics
    career_agg = trade_df.groupby('manager').agg({
        'transaction_id': 'count',
        'net_spar': ['sum', 'mean'],
        'spar_managed': 'sum',
        'partner_spar': 'sum',
        'trade_score': 'mean',
        'rank_improvement': 'mean',
        'year': 'nunique'
    }).reset_index()

    career_agg.columns = [
        'manager', 'total_trades', 'career_net_spar', 'avg_net_per_trade',
        'total_spar_acquired', 'total_spar_traded',
        'avg_trade_score', 'avg_rank_improvement', 'seasons_active'
    ]

    career_agg['trades_per_season'] = career_agg['total_trades'] / career_agg['seasons_active']
    career_agg['net_spar_per_season'] = career_agg['career_net_spar'] / career_agg['seasons_active']
    career_agg['win_rate'] = trade_df.groupby('manager')['net_spar'].apply(
        lambda x: (x > 0).sum() / len(x) * 100
    ).values

    # Rankings based on NET SPAR
    career_agg['Rank'] = career_agg['career_net_spar'].rank(ascending=False, method='min').astype(int)
    career_agg['Grade'] = career_agg['career_net_spar'].apply(
        lambda x: "🥇 Elite" if x > 500
        else "🥈 Great" if x > 250
        else "🥉 Good" if x > 100
        else "📊 Average" if x > 0
        else "📉 Poor"
    )

    st.markdown("### 🏆 Career Trade Summary")
    st.markdown("*Using Managed SPAR: actual value from players on your roster*")

    tab1, tab2, tab3 = st.tabs(["📊 Career Stats", "📈 Analytics", "👥 Comparisons"])

    with tab1:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Managers", len(career_agg))
        with col2:
            st.metric("Total Trades", f"{career_agg['total_trades'].sum():,.0f}")
        with col3:
            st.metric("Avg Career NET SPAR", f"{career_agg['career_net_spar'].mean():.1f}")
        with col4:
            st.metric("Avg Win Rate", f"{career_agg['win_rate'].mean():.1f}%")
        with col5:
            st.metric("Avg Trades/Season", f"{career_agg['trades_per_season'].mean():.1f}")

        st.divider()

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            manager_search = st.text_input("Search Manager", key="career_trade_mgr")
        with col2:
            grade_filter = st.selectbox("Grade", ["All", "Elite", "Great", "Good", "Average", "Poor"],
                                        key="career_trade_grade")
        with col3:
            sort_by = st.selectbox("Sort by", ["Rank", "NET SPAR", "Win Rate", "Efficiency", "Trades"],
                                   key="career_trade_sort")

        filtered = career_agg.copy()
        if manager_search:
            filtered = filtered[filtered['manager'].str.contains(manager_search, case=False, na=False)]
        if grade_filter != "All":
            filtered = filtered[filtered['Grade'].str.contains(grade_filter)]

        if sort_by == "NET SPAR":
            filtered = filtered.sort_values('career_net_spar', ascending=False)
        elif sort_by == "Win Rate":
            filtered = filtered.sort_values('win_rate', ascending=False)
        elif sort_by == "Efficiency":
            filtered = filtered.sort_values('avg_net_per_trade', ascending=False)
        elif sort_by == "Trades":
            filtered = filtered.sort_values('total_trades', ascending=False)
        else:
            filtered = filtered.sort_values('Rank')

        st.markdown(f"**Showing {len(filtered)} of {len(career_agg)} managers**")

        display_cols = [
            'Rank', 'manager', 'Grade', 'seasons_active', 'total_trades',
            'career_net_spar', 'avg_net_per_trade', 'net_spar_per_season',
            'win_rate', 'trades_per_season',
            'total_spar_acquired', 'total_spar_traded',
            'avg_trade_score', 'avg_rank_improvement'
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                'Rank': st.column_config.NumberColumn("#", format="%d", width="small"),
                'manager': st.column_config.TextColumn("Manager", width="medium"),
                'Grade': st.column_config.TextColumn("Grade", width="medium"),
                'seasons_active': st.column_config.NumberColumn("Seasons", format="%d", width="small"),
                'total_trades': st.column_config.NumberColumn("Trades", format="%d", width="small"),
                'career_net_spar': st.column_config.NumberColumn("Career NET", format="%.1f", help="Total NET SPAR across all trades"),
                'avg_net_per_trade': st.column_config.NumberColumn("Avg NET", format="%.1f", help="Average NET SPAR per trade"),
                'net_spar_per_season': st.column_config.NumberColumn("NET/Season", format="%.1f", help="NET SPAR per season"),
                'win_rate': st.column_config.NumberColumn("Win %", format="%.1f"),
                'trades_per_season': st.column_config.NumberColumn("Trades/Szn", format="%.1f", width="small"),
                'total_spar_acquired': st.column_config.NumberColumn("SPAR+", format="%.1f", help="Total SPAR acquired"),
                'total_spar_traded': st.column_config.NumberColumn("SPAR-", format="%.1f", help="Total SPAR traded away"),
                'avg_trade_score': st.column_config.NumberColumn("Avg Score", format="%.1f", width="small"),
                'avg_rank_improvement': st.column_config.NumberColumn("Rank Δ", format="%.1f", width="small"),
            }
        )

        csv = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Export", csv, "career_trades.csv", "text/csv", key="career_trade_data_export")

    with tab2:
        st.markdown("### 📈 Career Trade Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Win Rate vs Volume")
            fig = px.scatter(career_agg, x='total_trades', y='career_net_spar',
                             size='seasons_active', color='win_rate', hover_data=['manager'],
                             title="Career NET SPAR vs Total Trades",
                             color_continuous_scale='RdYlGn',
                             labels={'total_trades': 'Total Trades', 'career_net_spar': 'Career NET SPAR'})
            fig.update_layout(coloraxis_colorbar=dict(title="Win %"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Trading Efficiency")
            if 'avg_trade_score' in career_agg.columns:
                fig = px.scatter(career_agg, x='avg_trade_score', y='avg_net_per_trade',
                                 size='total_trades', color='avg_net_per_trade', hover_data=['manager'],
                                 title="Trade Score vs Avg NET SPAR",
                                 color_continuous_scale='Viridis',
                                 labels={'avg_trade_score': 'Avg Trade Score', 'avg_net_per_trade': 'Avg NET per Trade'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Trade score not available for analytics.")

        st.markdown("#### 🏆 Top Managers by Career NET SPAR")
        top10 = career_agg.nlargest(10, 'career_net_spar').sort_values('career_net_spar', ascending=True)

        fig = go.Figure()
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in top10['career_net_spar']]

        fig.add_trace(go.Bar(
            y=top10['manager'],
            x=top10['career_net_spar'],
            orientation='h',
            marker=dict(color=colors, line=dict(width=1, color='white')),
            text=top10['career_net_spar'].round(1),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Career NET SPAR: %{x:.1f}<br>Total Trades: %{customdata[0]}<br>Win Rate: %{customdata[1]:.1f}%<extra></extra>',
            customdata=top10[['total_trades', 'win_rate']].values
        ))

        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)

        fig.update_layout(
            title="Top 10 Managers by Career NET SPAR",
            xaxis_title="Career NET SPAR",
            yaxis_title="Manager",
            showlegend=False,
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # Manager comparison tables
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🎯 Most Efficient (Avg NET SPAR)")
            efficient = career_agg[career_agg['total_trades'] >= 5].nlargest(10, 'avg_net_per_trade')[
                ['manager', 'avg_net_per_trade', 'career_net_spar', 'total_trades', 'win_rate']
            ].copy()
            efficient['Rank'] = range(1, len(efficient) + 1)

            st.dataframe(
                efficient[['Rank', 'manager', 'avg_net_per_trade', 'total_trades', 'win_rate']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Rank': st.column_config.NumberColumn("#", width="small"),
                    'manager': st.column_config.TextColumn("Manager"),
                    'avg_net_per_trade': st.column_config.NumberColumn("Avg NET", format="%.1f", help="Average NET SPAR per trade"),
                    'total_trades': st.column_config.NumberColumn("Trades", format="%d", width="small"),
                    'win_rate': st.column_config.NumberColumn("Win %", format="%.1f", width="small"),
                }
            )

        with col2:
            st.markdown("##### 💪 Best Win Rates")
            best_wr = career_agg[career_agg['total_trades'] >= 5].nlargest(10, 'win_rate')[
                ['manager', 'win_rate', 'career_net_spar', 'total_trades']
            ].copy()
            best_wr['Rank'] = range(1, len(best_wr) + 1)

            st.dataframe(
                best_wr[['Rank', 'manager', 'win_rate', 'career_net_spar', 'total_trades']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Rank': st.column_config.NumberColumn("#", width="small"),
                    'manager': st.column_config.TextColumn("Manager"),
                    'win_rate': st.column_config.NumberColumn("Win %", format="%.1f"),
                    'career_net_spar': st.column_config.NumberColumn("NET SPAR", format="%.1f", width="small"),
                    'total_trades': st.column_config.NumberColumn("Trades", format="%d", width="small"),
                }
            )

    with tab3:
        st.markdown("### 👥 Manager Comparisons")

        managers = st.multiselect("Select managers to compare",
                                  options=career_agg['manager'].tolist(),
                                  default=career_agg.nlargest(3, 'career_net_spar')['manager'].tolist(),
                                  key="career_trade_compare")

        if managers:
            compare_df = career_agg[career_agg['manager'].isin(managers)]

            # Radar chart
            fig = go.Figure()
            for manager in managers:
                mgr_data = compare_df[compare_df['manager'] == manager].iloc[0]
                values = [
                    mgr_data['career_net_spar'] / career_agg['career_net_spar'].max() * 100 if career_agg['career_net_spar'].max() > 0 else 0,
                    mgr_data['avg_net_per_trade'] / career_agg['avg_net_per_trade'].max() * 100 if pd.notna(mgr_data['avg_net_per_trade']) and career_agg['avg_net_per_trade'].max() > 0 else 0,
                    mgr_data['win_rate'] if pd.notna(mgr_data['win_rate']) else 0,
                    mgr_data['total_trades'] / career_agg['total_trades'].max() * 100,
                    mgr_data['avg_trade_score'] / career_agg['avg_trade_score'].max() * 100 if 'avg_trade_score' in career_agg.columns and pd.notna(mgr_data.get('avg_trade_score')) and career_agg['avg_trade_score'].max() > 0 else 0
                ]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=['Career NET', 'Avg NET/Trade', 'Win %', 'Volume', 'Avg Score'],
                    fill='toself',
                    name=manager
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Manager Comparison (Normalized)",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Detailed Comparison")
            st.dataframe(compare_df[display_cols], hide_index=True, use_container_width=True)

    return
