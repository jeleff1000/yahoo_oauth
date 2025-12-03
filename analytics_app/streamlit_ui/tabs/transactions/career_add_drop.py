import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


@st.fragment
def display_career_add_drop(transaction_df, player_df):
    """Enhanced career add/drop with actual enriched columns."""
    from .season_add_drop import display_season_add_drop

    season_df = display_season_add_drop(transaction_df, player_df, return_df=True, key_prefix="career_internal")

    # Career aggregation
    agg_dict = {
        'total_transactions': 'sum',
        'faab_spent': 'sum',
        'spar_added_managed': 'sum',
        'spar_dropped_total': 'sum',
        'net_spar': 'sum',
        'net_points': 'sum',
        'avg_spar_efficiency': 'mean',
        'avg_score': 'mean',
        'rank_improvement': 'mean',
        'year': 'count'  # seasons played
    }
    # Add total_score if available
    if 'total_score' in season_df.columns:
        agg_dict['total_score'] = 'sum'

    career_agg = season_df.groupby('manager').agg(agg_dict).reset_index()

    col_names = [
        'manager', 'total_moves', 'total_faab', 'total_spar_added',
        'total_spar_dropped', 'career_net_spar', 'career_net_pts', 'avg_spar_efficiency', 'avg_score',
        'avg_rank_improvement', 'seasons_played'
    ]
    if 'total_score' in season_df.columns:
        col_names.append('career_total_score')
    career_agg.columns = col_names

    # Fallback if total_score doesn't exist
    if 'career_total_score' not in career_agg.columns:
        career_agg['career_total_score'] = career_agg['career_net_spar']

    career_agg['spar_per_season'] = career_agg['career_net_spar'] / career_agg['seasons_played']
    career_agg['score_per_season'] = career_agg['career_total_score'] / career_agg['seasons_played']
    career_agg['pts_per_season'] = career_agg['career_net_pts'] / career_agg['seasons_played']
    career_agg['moves_per_season'] = career_agg['total_moves'] / career_agg['seasons_played']
    career_agg['efficiency'] = career_agg['career_net_spar'] / career_agg['total_moves']

    # Rankings (based on Total Score - our weighted metric)
    career_agg['Rank'] = career_agg['career_total_score'].rank(ascending=False, method='min').astype(int)
    career_agg['Grade'] = career_agg['career_total_score'].apply(
        lambda x: "🥇 Elite" if x > 1000
        else "🥈 Great" if x > 500
        else "🥉 Good" if x > 250
        else "📊 Average" if x > 0
        else "📉 Poor"
    )

    # Compact CSS for career transactions
    st.markdown("""
    <style>
    .career-stat-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    .career-stat-card h4 {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.6;
        margin: 0 0 0.5rem 0;
    }
    .career-stat-row { display: flex; justify-content: space-around; }
    .career-stat-item { text-align: center; }
    .career-stat-value { font-size: 1.4rem; font-weight: 700; color: #4ade80; }
    .career-stat-value.neutral { color: #94a3b8; }
    .career-stat-value.faab { color: #fbbf24; }
    .career-stat-label { font-size: 0.65rem; opacity: 0.7; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### 🏆 Career Add/Drop Summary")

    tab1, tab2, tab3 = st.tabs(["📊 Career Stats", "📈 Analytics", "👥 Comparisons"])

    with tab1:
        # Grouped stats in cards
        col1, col2, col3 = st.columns(3)

        total_moves = career_agg['total_moves'].sum()
        total_faab = career_agg['total_faab'].sum()
        avg_score = career_agg['career_total_score'].mean()
        avg_net_spar = career_agg['career_net_spar'].mean()
        avg_efficiency = career_agg['efficiency'].mean()

        with col1:
            st.markdown("""
            <div class="career-stat-card">
                <h4>📊 Volume</h4>
                <div class="career-stat-row">
                    <div class="career-stat-item">
                        <div class="career-stat-value neutral">{}</div>
                        <div class="career-stat-label">Managers</div>
                    </div>
                    <div class="career-stat-item">
                        <div class="career-stat-value neutral">{:,}</div>
                        <div class="career-stat-label">Total Moves</div>
                    </div>
                </div>
            </div>
            """.format(len(career_agg), int(total_moves)), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="career-stat-card">
                <h4>💰 FAAB</h4>
                <div class="career-stat-row">
                    <div class="career-stat-item">
                        <div class="career-stat-value faab">${:,.0f}</div>
                        <div class="career-stat-label">Total Spent</div>
                    </div>
                </div>
            </div>
            """.format(total_faab), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="career-stat-card">
                <h4>📈 Performance</h4>
                <div class="career-stat-row">
                    <div class="career-stat-item">
                        <div class="career-stat-value">{:.0f}</div>
                        <div class="career-stat-label">Avg Score</div>
                    </div>
                    <div class="career-stat-item">
                        <div class="career-stat-value">{:+.1f}</div>
                        <div class="career-stat-label">Avg Net SPAR</div>
                    </div>
                    <div class="career-stat-item">
                        <div class="career-stat-value">{:.2f}</div>
                        <div class="career-stat-label">SPAR/Move</div>
                    </div>
                </div>
            </div>
            """.format(avg_score, avg_net_spar, avg_efficiency), unsafe_allow_html=True)

        st.markdown("<div style='margin: 0.5rem 0; border-top: 1px solid rgba(100,116,139,0.3);'></div>", unsafe_allow_html=True)

        # Compact filters in one row
        col1, col2, col3 = st.columns([2, 1.5, 2])
        with col1:
            manager_search = st.text_input("Manager", placeholder="Search manager...", key="career_mgr_search", label_visibility="collapsed")
        with col2:
            grade_filter = st.selectbox("Grade", ["All", "Elite", "Great", "Good", "Average", "Poor"],
                                        key="career_grade", label_visibility="collapsed")
        with col3:
            sort_by = st.selectbox("Sort", ["Rank", "Total Score", "Net SPAR", "Efficiency", "Total Moves", "SPAR/FAAB"],
                                   key="career_sort", label_visibility="collapsed")

        filtered = career_agg.copy()
        if manager_search:
            filtered = filtered[filtered['manager'].str.contains(manager_search, case=False, na=False)]
        if grade_filter != "All":
            filtered = filtered[filtered['Grade'].str.contains(grade_filter)]

        if sort_by == "Total Score":
            filtered = filtered.sort_values('career_total_score', ascending=False)
        elif sort_by == "Net SPAR":
            filtered = filtered.sort_values('career_net_spar', ascending=False)
        elif sort_by == "Efficiency":
            filtered = filtered.sort_values('efficiency', ascending=False)
        elif sort_by == "Total Moves":
            filtered = filtered.sort_values('total_moves', ascending=False)
        elif sort_by == "SPAR/FAAB":
            filtered = filtered.sort_values('avg_spar_efficiency', ascending=False)
        else:
            filtered = filtered.sort_values('Rank')

        st.markdown(f"**Showing {len(filtered)} of {len(career_agg)} managers**")

        display_cols = [
            'Rank', 'manager', 'Grade', 'career_total_score', 'score_per_season',
            'seasons_played', 'total_moves', 'career_net_spar', 'efficiency',
            'total_faab', 'avg_spar_efficiency'
        ]
        # Only include columns that exist
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                'Rank': st.column_config.NumberColumn("Rank", format="%d"),
                'career_total_score': st.column_config.NumberColumn("Career Score", format="%+.0f", help="Weighted score"),
                'score_per_season': st.column_config.NumberColumn("Score/Season", format="%+.1f"),
                'seasons_played': st.column_config.NumberColumn("Seasons", format="%d"),
                'total_moves': st.column_config.NumberColumn("Total Moves", format="%d"),
                'career_net_spar': st.column_config.NumberColumn("Career Net SPAR", format="%.1f"),
                'efficiency': st.column_config.NumberColumn("SPAR/Move", format="%.2f"),
                'total_faab': st.column_config.NumberColumn("Total FAAB", format="$%.0f"),
                'avg_spar_efficiency': st.column_config.NumberColumn("SPAR/FAAB", format="%.2f", help="Average SPAR per FAAB dollar"),
            }
        )

        csv = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Export", csv, "career_add_drops.csv", "text/csv", key="career_export")

    with tab2:
        st.markdown("### 📈 Career Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Efficiency vs Volume")
            fig = px.scatter(career_agg, x='total_moves', y='career_net_spar',
                             size='seasons_played', color='efficiency',
                             hover_data=['manager'],
                             title="Career Net SPAR vs Total Moves",
                             color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### FAAB Spending vs Results")
            fig = px.scatter(career_agg, x='total_faab', y='career_net_spar',
                             size='total_moves', color='avg_spar_efficiency',
                             hover_data=['manager'],
                             title="FAAB Efficiency (SPAR/$)",
                             color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Manager Profiles")
        fig = px.bar(career_agg.sort_values('career_net_spar', ascending=False).head(10),
                     x='manager', y=['total_spar_added', 'total_spar_dropped'],
                     title="Top 10 Managers: SPAR Added vs Dropped",
                     barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 👥 Manager Comparisons")

        managers = st.multiselect("Select managers to compare",
                                  options=career_agg['manager'].tolist(),
                                  default=career_agg.nlargest(3, 'career_net_spar')['manager'].tolist(),
                                  key="career_compare")

        if managers:
            compare_df = career_agg[career_agg['manager'].isin(managers)]

            # Radar chart
            fig = go.Figure()
            for manager in managers:
                mgr_data = compare_df[compare_df['manager'] == manager].iloc[0]
                values = [
                    mgr_data['career_net_spar'] / career_agg['career_net_spar'].max() * 100,
                    mgr_data['efficiency'] / career_agg['efficiency'].max() * 100,
                    mgr_data['avg_spar_efficiency'] / career_agg['avg_spar_efficiency'].max() * 100 if pd.notna(mgr_data['avg_spar_efficiency']) else 0,
                    mgr_data['total_moves'] / career_agg['total_moves'].max() * 100,
                    mgr_data['avg_score'] / career_agg['avg_score'].max() * 100 if pd.notna(
                        mgr_data['avg_score']) else 0
                ]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=['Career Net SPAR', 'SPAR/Move', 'SPAR/FAAB', 'Volume', 'Avg Score'],
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

# ============================================================================
# USAGE NOTES:
# ============================================================================
# These corrected versions properly use the actual enriched transaction columns:
# - total_points_rest_of_season (the actual ROS points for each player)
# - faab_bid (the actual FAAB amount spent)
# - roi is calculated as total_points_rest_of_season / faab_bid
# - transaction_score (from your enrichment script)
# - position_rank_at_transaction and avg_position_rank_after
#
# Key corrections:
# 1. Separates adds and drops before aggregating
# 2. Uses actual column names from your enriched data
# 3. Calculates net_points as points_added - points_dropped
# 4. Handles missing data properly with fillna(0) and notna() checks
# 5. All aggregations work with the actual structure of your data

# This corrected version properly uses the actual enriched transaction columns:
# - total_points_rest_of_season (not points_gained_rest_of_season)
# - faab_bid (not faab)
# - Calculates ROI directly from total_points_rest_of_season / faab_bid
# - Uses position_rank_at_transaction and avg_position_rank_after correctly
# - Handles adds vs drops properly for net points calculation
#
# For the season and career views, you'll need similar corrections to aggregate
# these actual columns properly. The key is:
# 1. Use total_points_rest_of_season for all ROS calculations
# 2. Use faab_bid (not faab)
# 3. Calculate metrics from the actual enriched columns
# 4. Don't assume pre-aggregated columns that don't exist