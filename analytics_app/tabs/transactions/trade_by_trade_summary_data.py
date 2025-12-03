from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

__all__ = ["display_trade_by_trade_summary_data"]


@st.fragment
def display_trade_by_trade_summary_data(
        transaction_df: pd.DataFrame,
        player_df: pd.DataFrame,
        draft_history_df: pd.DataFrame,
) -> None:
    """Enhanced trade-by-trade showing BOTH SIDES of each trade using managed SPAR metrics."""

    t = transaction_df[transaction_df['transaction_type'] == 'trade'].copy()

    if len(t) == 0:
        st.warning("No trade transactions found.")
        return

    if 'year' in t.columns:
        t['year'] = t['year'].astype(str)

    # Use manager SPAR (what you actually started) as the primary metric
    spar_col = 'manager_spar_ros_managed' if 'manager_spar_ros_managed' in t.columns else 'manager_spar_ros'
    pts_col = 'total_points_ros_managed' if 'total_points_ros_managed' in t.columns else 'total_points_rest_of_season'
    ppg_col = 'ppg_ros_managed' if 'ppg_ros_managed' in t.columns else 'ppg_after_transaction'
    weeks_col = 'weeks_ros_managed' if 'weeks_ros_managed' in t.columns else 'weeks_rest_of_season'

    # Group by transaction_id and manager to get each side
    trade_sides = t.groupby(['transaction_id', 'manager']).agg({
        'player_name': lambda x: ', '.join(x.dropna().astype(str)),
        'position': lambda x: ', '.join(x.dropna().astype(str).unique()),
        'year': 'first',
        'week': 'first',
        spar_col: 'sum',
        pts_col: 'sum',
        ppg_col: 'mean',
        weeks_col: 'mean',
        'points_at_transaction': 'sum',
        'position_rank_at_transaction': 'mean',
        'position_rank_after_transaction': 'mean',
        'transaction_quality_score': 'mean',
    }).reset_index()

    trade_sides.columns = [
        'transaction_id', 'manager', 'players', 'positions', 'year', 'week',
        'spar_managed', 'pts_managed', 'ppg_managed', 'weeks_managed',
        'points_at_trade', 'avg_rank_at', 'avg_rank_after',
        'trade_score'
    ]

    # Get trade partners and their SPAR
    partners_map = t.groupby('transaction_id')['manager'].apply(lambda x: list(x.unique())).to_dict()

    def get_partner(row):
        partners = partners_map.get(row['transaction_id'], [])
        other = [p for p in partners if p != row['manager']]
        return other[0] if len(other) > 0 else 'Unknown'

    trade_sides['partner'] = trade_sides.apply(get_partner, axis=1)

    spar_map = trade_sides.set_index(['transaction_id', 'manager'])['spar_managed'].to_dict()

    def get_partner_spar(row):
        partner_key = (row['transaction_id'], row['partner'])
        return spar_map.get(partner_key, 0)

    trade_sides['partner_spar'] = trade_sides.apply(get_partner_spar, axis=1)
    trade_sides['net_spar'] = trade_sides['spar_managed'] - trade_sides['partner_spar']
    trade_sides['rank_improvement'] = trade_sides['avg_rank_at'] - trade_sides['avg_rank_after']

    # Cache for other tabs (keep the per-manager view)
    st.session_state['trade_summary_df'] = trade_sides.copy()

    # NOW CREATE 2 ROWS PER TRADE VIEW: Show what each manager gave up and got
    # Get partner's players for each manager
    players_map = trade_sides.set_index(['transaction_id', 'manager'])['players'].to_dict()

    def get_partner_players(row):
        partner_key = (row['transaction_id'], row['partner'])
        return players_map.get(partner_key, '')

    trade_sides['players_gave_up'] = trade_sides.apply(get_partner_players, axis=1)
    trade_sides['players_acquired'] = trade_sides['players']

    # Result indicator based on NET SPAR
    def get_result(net_spar):
        if net_spar > 100:
            return "🏆 Elite Win"
        elif net_spar > 50:
            return "✅ Great Win"
        elif net_spar > 20:
            return "👍 Good Win"
        elif net_spar > 0:
            return "➡️ Small Win"
        elif net_spar == 0:
            return "➖ Even"
        elif net_spar > -20:
            return "📉 Small Loss"
        elif net_spar > -50:
            return "😬 Bad Loss"
        else:
            return "❌ Major Loss"

    trade_sides['result'] = trade_sides['net_spar'].apply(get_result)

    both_sides_df = trade_sides.copy()

    st.markdown("### 📊 Trade-by-Trade Summary")
    st.markdown("*Using Managed SPAR: actual value from players while on your roster*")

    tab1, tab2, tab3 = st.tabs(["📋 All Trades", "📈 Analytics", "🏆 Leaderboards"])

    with tab1:
        # Summary metrics - unique trades count
        unique_trades_count = len(both_sides_df['transaction_id'].unique())
        total_rows = len(both_sides_df)
        avg_net_spar = both_sides_df['net_spar'].mean()
        wins = len(both_sides_df[both_sides_df['net_spar'] > 0])
        losses = len(both_sides_df[both_sides_df['net_spar'] < 0])

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Unique Trades", unique_trades_count)
        with col2:
            st.metric("Trade Sides", total_rows, help="2 rows per trade")
        with col3:
            st.metric("Avg NET SPAR", f"{avg_net_spar:.1f}")
        with col4:
            st.metric("Wins", wins, help="Positive NET SPAR")
        with col5:
            st.metric("Losses", losses, help="Negative NET SPAR")

        best_trade = both_sides_df.nlargest(1, 'net_spar') if len(both_sides_df) > 0 else None
        if best_trade is not None and len(best_trade) > 0:
            best = best_trade.iloc[0]
            st.success(f"🌟 **Best Trade:** {best['manager']} (+{best['net_spar']:.1f} NET SPAR) - Gave up {best['players_gave_up']}, Got {best['players_acquired']}")

        st.divider()

        # Filters
        with st.expander("🔍 Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                year_filter = st.selectbox("Year", ["All"] + sorted(both_sides_df['year'].unique().tolist(), reverse=True),
                                           key="trade_year")
            with col2:
                manager_filter = st.text_input("Manager", key="trade_mgr")
            with col3:
                partner_filter = st.text_input("Partner", key="trade_partner")

            col1, col2 = st.columns(2)
            with col1:
                result_filter = st.selectbox("Result", ["All", "Wins (NET>0)", "Losses (NET<0)", "Elite Wins (NET>100)", "Major Losses (NET<-50)"],
                                              key="trade_result")
            with col2:
                sort_by = st.selectbox("Sort by", ["NET SPAR ↓", "Recent", "SPAR Acquired ↓"], key="trade_sort")

        filtered = both_sides_df.copy()
        if year_filter != "All":
            filtered = filtered[filtered['year'] == year_filter]
        if manager_filter:
            filtered = filtered[filtered['manager'].str.contains(manager_filter, case=False, na=False)]
        if partner_filter:
            filtered = filtered[filtered['partner'].str.contains(partner_filter, case=False, na=False)]
        if result_filter == "Wins (NET>0)":
            filtered = filtered[filtered['net_spar'] > 0]
        elif result_filter == "Losses (NET<0)":
            filtered = filtered[filtered['net_spar'] < 0]
        elif result_filter == "Elite Wins (NET>100)":
            filtered = filtered[filtered['net_spar'] > 100]
        elif result_filter == "Major Losses (NET<-50)":
            filtered = filtered[filtered['net_spar'] < -50]

        if sort_by == "NET SPAR ↓":
            filtered = filtered.sort_values('net_spar', ascending=False)
        elif sort_by == "SPAR Acquired ↓":
            filtered = filtered.sort_values('spar_managed', ascending=False)
        else:
            filtered = filtered.sort_values(['year', 'week'], ascending=[False, False])

        st.markdown(f"**Showing {len(filtered)} of {len(both_sides_df)} trade sides**")

        display_cols = [
            'year', 'week', 'manager', 'partner',
            'players_acquired', 'spar_managed',
            'players_gave_up', 'partner_spar',
            'net_spar', 'result'
        ]

        st.dataframe(
            filtered[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                'year': st.column_config.TextColumn("Year", width="small"),
                'week': st.column_config.NumberColumn("Wk", format="%d", width="small"),
                'manager': st.column_config.TextColumn("Manager", width="medium"),
                'partner': st.column_config.TextColumn("Partner", width="medium"),
                'players_acquired': st.column_config.TextColumn("Acquired"),
                'spar_managed': st.column_config.NumberColumn("SPAR Got", format="%.1f", help="SPAR from players you acquired", width="small"),
                'players_gave_up': st.column_config.TextColumn("Gave Up"),
                'partner_spar': st.column_config.NumberColumn("SPAR Lost", format="%.1f", help="SPAR from players you gave up", width="small"),
                'net_spar': st.column_config.NumberColumn("NET SPAR", format="%.1f", help="SPAR Got - SPAR Lost"),
                'result': st.column_config.TextColumn("Result", width="medium"),
            }
        )

        csv = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Export", csv, f"trades_{year_filter}.csv", "text/csv", key="trade_by_trade_export")

    with tab2:
        st.markdown("### 📈 Trade Analytics")

        # Use the per-manager trade_sides for analytics
        mgr_stats = trade_sides.groupby('manager').agg({
            'transaction_id': 'count',
            'net_spar': ['sum', 'mean'],
            'spar_managed': 'sum',
            'partner_spar': 'sum',
            'trade_score': 'mean',
            'rank_improvement': 'mean',
        }).reset_index()

        mgr_stats.columns = ['manager', 'total_trades', 'career_net_spar', 'avg_net_spar',
                             'total_spar_acquired', 'total_spar_traded', 'avg_score', 'avg_rank_delta']

        mgr_stats['win_rate'] = trade_sides.groupby('manager')['net_spar'].apply(
            lambda x: (x > 0).sum() / len(x) * 100
        ).values

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Trade Outcome Distribution")
            # Get the maximum NET SPAR from each trade (both sides)
            trade_outcomes = both_sides_df.groupby('transaction_id')['net_spar'].apply(
                lambda x: x.abs().max()
            ).reset_index()
            trade_outcomes.columns = ['transaction_id', 'max_net']

            trade_outcomes['outcome_bin'] = pd.cut(
                trade_outcomes['max_net'],
                bins=[0, 10, 20, 50, 100, float('inf')],
                labels=['Even\n(0-10)', 'Slight\n(10-20)', 'Clear\n(20-50)', 'Strong\n(50-100)', 'Lopsided\n(>100)']
            )

            dist_data = trade_outcomes['outcome_bin'].value_counts().sort_index().reset_index()
            dist_data.columns = ['Category', 'Count']

            colors = ['#98df8a', '#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

            fig = px.bar(dist_data, x='Category', y='Count',
                         title="Trade Balance Distribution",
                         color='Category',
                         color_discrete_sequence=colors)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 💪 Manager Win Rates")
            win_rate_data = mgr_stats.nlargest(10, 'win_rate')
            fig = go.Figure(go.Bar(
                x=win_rate_data['win_rate'],
                y=win_rate_data['manager'],
                orientation='h',
                marker_color=win_rate_data['win_rate'],
                marker_colorscale='RdYlGn',
                marker_cmin=0,
                marker_cmax=100,
                text=win_rate_data['win_rate'].round(1),
                texttemplate='%{text}%',
                textposition='outside'
            ))
            fig.update_layout(
                title="Top 10 Managers by Win Rate",
                xaxis_title="Win Rate (%)",
                yaxis_title="Manager",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Manager Rankings
        st.markdown("#### 🏆 Manager Trade Rankings")
        mgr_ranked = mgr_stats.sort_values('career_net_spar', ascending=True)

        fig = go.Figure()

        colors = ['#2ca02c' if x > 0 else '#d62728' for x in mgr_ranked['career_net_spar']]

        fig.add_trace(go.Bar(
            y=mgr_ranked['manager'],
            x=mgr_ranked['career_net_spar'],
            orientation='h',
            marker=dict(color=colors, line=dict(width=1, color='white')),
            text=mgr_ranked['career_net_spar'].round(1),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Career NET SPAR: %{x:.1f}<br>Total Trades: %{customdata[0]}<br>Avg NET: %{customdata[1]:.1f}<br>Win Rate: %{customdata[2]:.1f}%<extra></extra>',
            customdata=mgr_ranked[['total_trades', 'avg_net_spar', 'win_rate']].values
        ))

        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)

        fig.update_layout(
            title="Manager Rankings by Career NET SPAR",
            xaxis_title="Career NET SPAR",
            yaxis_title="Manager",
            showlegend=False,
            height=max(400, len(mgr_ranked) * 35)
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 🏆 Trade Leaderboards")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🌟 Most Lopsided Trades")
            # Group by transaction_id to get both sides and compute difference
            lopsided_data = []
            for tid in both_sides_df['transaction_id'].unique():
                trade_rows = both_sides_df[both_sides_df['transaction_id'] == tid]
                if len(trade_rows) == 2:
                    row1, row2 = trade_rows.iloc[0], trade_rows.iloc[1]
                    net_diff = abs(row1['net_spar'] - row2['net_spar'])
                    winner = row1 if row1['net_spar'] > row2['net_spar'] else row2
                    loser = row2 if row1['net_spar'] > row2['net_spar'] else row1
                    lopsided_data.append({
                        'transaction_id': tid,
                        'year': winner['year'],
                        'week': winner['week'],
                        'winner_name': winner['manager'],
                        'winner_got': winner['players_acquired'],
                        'loser_name': loser['manager'],
                        'loser_got': loser['players_acquired'],
                        'net_diff': net_diff
                    })

            if len(lopsided_data) > 0:
                lopsided_df = pd.DataFrame(lopsided_data)
                top_lopsided = lopsided_df.nlargest(10, 'net_diff').copy()
                top_lopsided['Rank'] = range(1, len(top_lopsided) + 1)

                st.dataframe(
                    top_lopsided[['Rank', 'year', 'winner_name', 'winner_got', 'loser_name', 'loser_got', 'net_diff']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        'Rank': st.column_config.NumberColumn("#", width="small"),
                        'year': st.column_config.TextColumn("Year", width="small"),
                        'winner_name': st.column_config.TextColumn("Winner"),
                        'winner_got': st.column_config.TextColumn("Got", help="Players winner acquired"),
                        'loser_name': st.column_config.TextColumn("Loser"),
                        'loser_got': st.column_config.TextColumn("Got", help="Players loser acquired"),
                        'net_diff': st.column_config.NumberColumn("Diff", format="%.1f", help="NET SPAR difference"),
                    }
                )
            else:
                st.info("No lopsided trade data available.")

        with col2:
            st.markdown("#### ⚖️ Most Even Trades")
            if len(lopsided_data) > 0:
                even = lopsided_df.nsmallest(10, 'net_diff')[
                    ['year', 'winner_name', 'winner_got', 'loser_name', 'loser_got', 'net_diff']
                ].copy()
                even['Rank'] = range(1, len(even) + 1)

                st.dataframe(
                    even[['Rank', 'year', 'winner_name', 'winner_got', 'loser_name', 'loser_got', 'net_diff']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        'Rank': st.column_config.NumberColumn("#", width="small"),
                        'year': st.column_config.TextColumn("Year", width="small"),
                        'winner_name': st.column_config.TextColumn("Manager A"),
                        'winner_got': st.column_config.TextColumn("Got"),
                        'loser_name': st.column_config.TextColumn("Manager B"),
                        'loser_got': st.column_config.TextColumn("Got"),
                        'net_diff': st.column_config.NumberColumn("Diff", format="%.1f"),
                    }
                )
            else:
                st.info("No even trade data available.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Most Efficient Traders")
            efficient = mgr_stats[mgr_stats['total_trades'] >= 3].nlargest(10, 'avg_net_spar')[
                ['manager', 'avg_net_spar', 'total_trades', 'win_rate']
            ].copy()
            efficient['Rank'] = range(1, len(efficient) + 1)

            st.dataframe(
                efficient[['Rank', 'manager', 'avg_net_spar', 'total_trades', 'win_rate']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Rank': st.column_config.NumberColumn("#", width="small"),
                    'manager': st.column_config.TextColumn("Manager"),
                    'avg_net_spar': st.column_config.NumberColumn("Avg NET", format="%.1f", help="Average NET SPAR per trade"),
                    'total_trades': st.column_config.NumberColumn("Trades", format="%d", width="small"),
                    'win_rate': st.column_config.NumberColumn("Win %", format="%.1f", width="small"),
                }
            )

        with col2:
            st.markdown("#### 🏅 Highest SPAR Exchanges")
            # Calculate total SPAR managed for each trade
            high_spar_data = []
            for tid in both_sides_df['transaction_id'].unique():
                trade_rows = both_sides_df[both_sides_df['transaction_id'] == tid]
                if len(trade_rows) == 2:
                    row1, row2 = trade_rows.iloc[0], trade_rows.iloc[1]
                    total_spar = row1['spar_managed'] + row2['spar_managed']
                    high_spar_data.append({
                        'transaction_id': tid,
                        'year': row1['year'],
                        'week': row1['week'],
                        'manager_a': row1['manager'],
                        'players_a': row1['players_acquired'],
                        'manager_b': row2['manager'],
                        'players_b': row2['players_acquired'],
                        'total_spar': total_spar
                    })

            if len(high_spar_data) > 0:
                high_spar_df = pd.DataFrame(high_spar_data)
                valuable = high_spar_df.nlargest(10, 'total_spar').copy()
                valuable['Rank'] = range(1, len(valuable) + 1)

                st.dataframe(
                    valuable[['Rank', 'year', 'manager_a', 'players_a', 'manager_b', 'players_b', 'total_spar']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        'Rank': st.column_config.NumberColumn("#", width="small"),
                        'year': st.column_config.TextColumn("Year", width="small"),
                        'manager_a': st.column_config.TextColumn("Manager A"),
                        'players_a': st.column_config.TextColumn("Got"),
                        'manager_b': st.column_config.TextColumn("Manager B"),
                        'players_b': st.column_config.TextColumn("Got"),
                        'total_spar': st.column_config.NumberColumn("Total SPAR", format="%.1f", help="Combined SPAR from both sides"),
                    }
                )
            else:
                st.info("No high SPAR exchange data available.")

        # Most Frequent Trading Partners
        st.markdown("#### 🤝 Most Frequent Trading Partners")
        # Build trading partner pairs
        partner_pairs = []
        for tid in both_sides_df['transaction_id'].unique():
            trade_rows = both_sides_df[both_sides_df['transaction_id'] == tid]
            if len(trade_rows) == 2:
                row1, row2 = trade_rows.iloc[0], trade_rows.iloc[1]
                # Sort manager names to ensure consistent pairing
                managers = sorted([row1['manager'], row2['manager']])
                partner_pairs.append({
                    'transaction_id': tid,
                    'pair': f"{managers[0]} ↔ {managers[1]}",
                    'manager_a': managers[0],
                    'manager_b': managers[1],
                    'net_spar_a': row1['net_spar'] if row1['manager'] == managers[0] else row2['net_spar'],
                    'net_spar_b': row2['net_spar'] if row2['manager'] == managers[1] else row1['net_spar'],
                })

        if len(partner_pairs) > 0:
            partner_df = pd.DataFrame(partner_pairs)

            # Aggregate by pair
            partner_summary = partner_df.groupby('pair').agg({
                'transaction_id': 'count',
                'net_spar_a': 'sum',
                'net_spar_b': 'sum',
            }).reset_index()
            partner_summary.columns = ['Trading Partners', 'Trades', 'Total NET (A)', 'Total NET (B)']

            # Calculate who "won" the partnership overall
            partner_summary['NET Diff'] = (partner_summary['Total NET (A)'] - partner_summary['Total NET (B)']).abs()
            partner_summary['Balance'] = partner_summary['NET Diff'].apply(
                lambda x: "Even" if x < 20 else "Balanced" if x < 50 else "Lopsided"
            )

            # Sort by trade count
            top_partners = partner_summary.nlargest(10, 'Trades').copy()
            top_partners['Rank'] = range(1, len(top_partners) + 1)

            st.dataframe(
                top_partners[['Rank', 'Trading Partners', 'Trades', 'Total NET (A)', 'Total NET (B)', 'NET Diff', 'Balance']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Rank': st.column_config.NumberColumn("#", width="small"),
                    'Trading Partners': st.column_config.TextColumn("Partners", width="large"),
                    'Trades': st.column_config.NumberColumn("# Trades", format="%d", width="small"),
                    'Total NET (A)': st.column_config.NumberColumn("NET (A)", format="%.1f", help="Total NET SPAR for first manager", width="small"),
                    'Total NET (B)': st.column_config.NumberColumn("NET (B)", format="%.1f", help="Total NET SPAR for second manager", width="small"),
                    'NET Diff': st.column_config.NumberColumn("Diff", format="%.1f", help="Difference between managers", width="small"),
                    'Balance': st.column_config.TextColumn("Balance", width="small"),
                }
            )
        else:
            st.info("No trading partner data available.")
