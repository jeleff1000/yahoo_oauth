from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import chart theming utilities
try:
    from shared.chart_themes import (
        apply_chart_theme,
        get_chart_colors,
        create_grade_bar_chart,
        create_regret_bar_chart,
        create_faab_tier_chart,
        create_horizontal_bar_chart,
        get_grade_colors_list,
        get_regret_colors_list,
        get_faab_tier_colors_list,
    )
    HAS_CHART_THEMES = True
except ImportError:
    HAS_CHART_THEMES = False

__all__ = ["display_weekly_add_drop"]


@st.fragment
def display_weekly_add_drop(
        transaction_df: pd.DataFrame,
        player_df: pd.DataFrame,
        keys: dict | None = None,
        include_search_bars: bool = True,
) -> None:
    """
    Enhanced weekly add/drop using pre-computed engagement metrics.

    Now uses pre-computed columns from source data:
    - transaction_result, result_emoji (instead of calculating get_result())
    - transaction_grade (A-F grades)
    - faab_value_tier (Steal/Great Value/etc.)
    - drop_regret_score, drop_regret_tier
    - timing_category, pickup_type
    """
    t = transaction_df.copy()

    # Filter to adds/drops only
    t = t[t['transaction_type'].isin(['add', 'drop'])].copy()

    if len(t) == 0:
        st.warning("No add/drop transactions found.")
        return

    # Convert year to string for display
    if 'year' in t.columns:
        t['year'] = t['year'].astype(str)

    # Ensure position column exists
    if 'position' not in t.columns and 'player_position' in t.columns:
        t['position'] = t['player_position']
    elif 'position' not in t.columns:
        t['position'] = pd.NA

    # Use pre-computed columns - with fallbacks for backward compatibility
    # Result display: prefer pre-computed result_emoji + transaction_result
    if 'result_emoji' not in t.columns:
        t['result_emoji'] = ''
    if 'transaction_result' not in t.columns:
        t['transaction_result'] = 'Unknown'

    # Combine emoji and result for display
    t['Result'] = t['result_emoji'].fillna('') + ' ' + t['transaction_result'].fillna('')
    t['Result'] = t['Result'].str.strip()

    # Use pre-computed spar_efficiency for ROI
    if 'spar_efficiency' in t.columns:
        t['roi'] = pd.to_numeric(t['spar_efficiency'], errors='coerce')
    else:
        t['roi'] = pd.NA

    # Determine value columns (prefer SPAR, fallback to PPG)
    add_value_col = 'manager_spar_ros_managed' if 'manager_spar_ros_managed' in t.columns else 'ppg_after_transaction'
    drop_value_col = 'player_spar_ros_total' if 'player_spar_ros_total' in t.columns else 'ppg_after_transaction'

    # Split into adds and drops
    adds = t[t['transaction_type'] == 'add'].copy()
    drops = t[t['transaction_type'] == 'drop'].copy()

    # Basic metrics
    total_moves = len(t)
    total_adds = len(adds)
    total_drops = len(drops)
    total_faab = adds['faab_bid'].sum() if 'faab_bid' in adds.columns else 0
    avg_faab = adds[adds['faab_bid'] > 0]['faab_bid'].mean() if 'faab_bid' in adds.columns and len(adds[adds['faab_bid'] > 0]) > 0 else 0

    # Calculate totals using appropriate value columns
    total_value_added = adds[add_value_col].sum() if add_value_col in adds.columns else 0
    total_value_dropped = drops[drop_value_col].sum() if drop_value_col in drops.columns else 0
    net_value = total_value_added - total_value_dropped

    # Best pickup and worst drop
    best_pickup = None
    if len(adds) > 0 and add_value_col in adds.columns and adds[add_value_col].notna().any():
        best = adds.nlargest(1, add_value_col)
        if len(best) > 0:
            best_pickup = best.iloc[0]

    worst_drop = None
    if len(drops) > 0 and drop_value_col in drops.columns and drops[drop_value_col].notna().any():
        worst = drops.nlargest(1, drop_value_col)
        if len(worst) > 0:
            worst_drop = worst.iloc[0]

    # Compact CSS for transactions page
    st.markdown("""
    <style>
    /* Transaction stat cards */
    .txn-stats-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .txn-stat-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px;
        padding: 1rem;
        flex: 1;
    }
    .txn-stat-card h4 {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.6;
        margin: 0 0 0.75rem 0;
        font-weight: 500;
    }
    .txn-stat-row {
        display: flex;
        justify-content: space-around;
        gap: 0.5rem;
    }
    .txn-stat-item {
        text-align: center;
    }
    .txn-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4ade80;
    }
    .txn-stat-value.neutral { color: #94a3b8; }
    .txn-stat-value.negative { color: #f87171; }
    .txn-stat-value.faab { color: #fbbf24; }
    .txn-stat-label {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }
    /* Grade badges */
    .grade-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .grade-A { background: #22c55e; color: white; }
    .grade-B { background: #3b82f6; color: white; }
    .grade-C { background: #6b7280; color: white; }
    .grade-D { background: #f97316; color: white; }
    .grade-F { background: #ef4444; color: white; }
    /* Compact filter styling */
    .txn-filters [data-testid="stExpander"] {
        margin-bottom: 0.5rem !important;
    }
    .txn-filters .stSelectbox, .txn-filters .stTextInput {
        margin-bottom: 0.25rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 📄 Weekly Add/Drop Transactions")

    tab1, tab2, tab3 = st.tabs(["📋 Transactions", "📊 Analytics", "🏆 Leaderboards"])

    with tab1:
        # Grouped stats in cards
        value_label = "SPAR" if 'manager_spar_ros_managed' in t.columns else "PPG"

        # Create 3 card columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="txn-stat-card">
                <h4>📊 Volume</h4>
                <div class="txn-stat-row">
                    <div class="txn-stat-item">
                        <div class="txn-stat-value neutral">{:,}</div>
                        <div class="txn-stat-label">Total</div>
                    </div>
                    <div class="txn-stat-item">
                        <div class="txn-stat-value">{:,}</div>
                        <div class="txn-stat-label">➕ Adds</div>
                    </div>
                    <div class="txn-stat-item">
                        <div class="txn-stat-value negative">{:,}</div>
                        <div class="txn-stat-label">➖ Drops</div>
                    </div>
                </div>
            </div>
            """.format(total_moves, total_adds, total_drops), unsafe_allow_html=True)

        with col2:
            avg_faab_display = f"${avg_faab:.0f}" if pd.notna(avg_faab) else "$0"
            st.markdown("""
            <div class="txn-stat-card">
                <h4>💰 FAAB</h4>
                <div class="txn-stat-row">
                    <div class="txn-stat-item">
                        <div class="txn-stat-value faab">${:,.0f}</div>
                        <div class="txn-stat-label">Total Spent</div>
                    </div>
                    <div class="txn-stat-item">
                        <div class="txn-stat-value faab">{}</div>
                        <div class="txn-stat-label">Avg Bid</div>
                    </div>
                </div>
            </div>
            """.format(total_faab, avg_faab_display), unsafe_allow_html=True)

        with col3:
            net_class = "" if net_value >= 0 else "negative"
            st.markdown("""
            <div class="txn-stat-card">
                <h4>📈 Performance</h4>
                <div class="txn-stat-row">
                    <div class="txn-stat-item">
                        <div class="txn-stat-value">{:+.1f}</div>
                        <div class="txn-stat-label">{} Gained</div>
                    </div>
                    <div class="txn-stat-item">
                        <div class="txn-stat-value negative">{:.1f}</div>
                        <div class="txn-stat-label">{} Lost</div>
                    </div>
                    <div class="txn-stat-item">
                        <div class="txn-stat-value {}">{:+.1f}</div>
                        <div class="txn-stat-label">Net {}</div>
                    </div>
                </div>
            </div>
            """.format(total_value_added, value_label, total_value_dropped, value_label, net_class, net_value, value_label), unsafe_allow_html=True)

        # Best/Worst highlights
        if best_pickup is not None or worst_drop is not None:
            col1, col2 = st.columns(2)
            with col1:
                if best_pickup is not None:
                    best_val = best_pickup[add_value_col] if pd.notna(best_pickup[add_value_col]) else 0
                    faab_val = best_pickup.get('faab_bid', 0)
                    faab_val = faab_val if pd.notna(faab_val) else 0
                    st.success(f"🌟 **Best:** {best_pickup['player_name']} (+{best_val:.1f} {value_label}, ${faab_val:.0f}) - {best_pickup['manager']}")
            with col2:
                if worst_drop is not None:
                    worst_val = worst_drop[drop_value_col] if pd.notna(worst_drop[drop_value_col]) else 0
                    st.error(f"💔 **Worst:** {worst_drop['player_name']} ({worst_val:.1f} {value_label} lost) - {worst_drop['manager']}")

        st.markdown("<div style='margin: 0.75rem 0; border-top: 1px solid rgba(100,116,139,0.3);'></div>", unsafe_allow_html=True)

        # Compact two-row filters
        st.markdown('<div class="txn-filters">', unsafe_allow_html=True)
        with st.expander("🔍 Filters", expanded=False):
            # Row 1: Time + Manager + Type + Grade + Sort
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1.5, 1, 1, 1.5])
            with col1:
                year_filter = st.selectbox("Year", ["All"] + sorted(t['year'].unique().tolist(), reverse=True),
                                           key="weekly_year", label_visibility="collapsed")
            with col2:
                week_filter = st.selectbox("Week", ["All"] + sorted(t['week'].dropna().unique().tolist()),
                                           key="weekly_week", label_visibility="collapsed")
            with col3:
                manager_filter = st.selectbox("Manager", ["All"] + sorted(t['manager'].dropna().unique().tolist()),
                                              key="weekly_manager", label_visibility="collapsed")
            with col4:
                trans_type_filter = st.selectbox("Type", ["All", "➕ Adds", "➖ Drops"], key="weekly_type", label_visibility="collapsed")
            with col5:
                grade_options = ["All"]
                if 'transaction_grade' in t.columns:
                    grade_options += sorted([g for g in t['transaction_grade'].dropna().unique().tolist() if g])
                grade_filter = st.selectbox("Grade", grade_options, key="weekly_grade", label_visibility="collapsed")
            with col6:
                sort_by = st.selectbox("Sort", ["Recent First", "Score ↓", "SPAR ↓", "Grade ↓", "FAAB ↓", "ROI ↓"],
                                       key="weekly_sort", label_visibility="collapsed")

            # Row 2: Player search (full width) + Position + Result
            col1, col2, col3 = st.columns([3, 1, 1.5])
            with col1:
                player_search = st.text_input("Player", placeholder="Search player name...", key="weekly_player", label_visibility="collapsed")
            with col2:
                position_filter = st.selectbox("Position", ["All", "QB", "RB", "WR", "TE", "K", "DEF"],
                                               key="weekly_position", label_visibility="collapsed")
            with col3:
                result_options = ["All"]
                if 'transaction_result' in t.columns:
                    result_options += sorted(t['transaction_result'].dropna().unique().tolist())
                result_filter = st.selectbox("Result", result_options, key="weekly_result", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Apply filters
        filtered = t.copy()
        if year_filter != "All":
            filtered = filtered[filtered['year'] == year_filter]
        if week_filter != "All":
            filtered = filtered[filtered['week'] == week_filter]
        if manager_filter != "All":
            filtered = filtered[filtered['manager'] == manager_filter]
        if trans_type_filter == "➕ Adds":
            filtered = filtered[filtered['transaction_type'] == 'add']
        elif trans_type_filter == "➖ Drops":
            filtered = filtered[filtered['transaction_type'] == 'drop']
        if player_search:
            filtered = filtered[filtered['player_name'].str.contains(player_search, case=False, na=False)]
        if position_filter != "All":
            filtered = filtered[filtered['position'] == position_filter]
        if result_filter != "All" and 'transaction_result' in filtered.columns:
            filtered = filtered[filtered['transaction_result'] == result_filter]
        if grade_filter != "All" and 'transaction_grade' in filtered.columns:
            filtered = filtered[filtered['transaction_grade'] == grade_filter]

        # Group by transaction_id for combined add/drop view
        if 'transaction_id' in filtered.columns and len(filtered) > 0:
            # Separate adds and drops
            adds_filtered = filtered[filtered['transaction_type'] == 'add'].copy()
            drops_filtered = filtered[filtered['transaction_type'] == 'drop'].copy()

            # Group adds by transaction_id
            agg_dict = {
                'year': 'first',
                'week': 'first',
                'manager': 'first',
                'player_name': lambda x: ', '.join(x.astype(str)),
                'position': lambda x: ', '.join(x.dropna().astype(str).unique()),
                add_value_col: 'sum',
                'faab_bid': 'first',
                'transaction_grade': 'first',
                'transaction_result': 'first',
                'result_emoji': 'first',
                'faab_value_tier': 'first',
            }
            # Add transaction_score if available
            if 'transaction_score' in adds_filtered.columns:
                agg_dict['transaction_score'] = 'sum'
            add_grouped = adds_filtered.groupby('transaction_id').agg(agg_dict).reset_index()

            # Group drops by transaction_id
            drop_cols = {
                'year': 'first',
                'week': 'first',
                'manager': 'first',
                'player_name': lambda x: ', '.join(x.astype(str)),
                'position': lambda x: ', '.join(x.dropna().astype(str).unique()),
                drop_value_col: 'sum',
                'drop_regret_tier': 'first',
            }
            # Add transaction_score for drops (will be negative for regrets)
            if 'transaction_score' in drops_filtered.columns:
                drop_cols['transaction_score'] = 'sum'
            drop_grouped = drops_filtered.groupby('transaction_id').agg(drop_cols).reset_index()

            # Merge adds and drops
            combined = add_grouped.merge(
                drop_grouped,
                on='transaction_id',
                how='outer',
                suffixes=('_added', '_dropped')
            )

            # Clean up columns - safely get column values with coalescing
            def safe_coalesce(df, col_primary, col_fallback=None, default=''):
                """Get column value, preferring primary, falling back to fallback."""
                if col_primary in df.columns:
                    result = df[col_primary].copy()
                    if col_fallback and col_fallback in df.columns:
                        # Fill NaN from primary with values from fallback
                        result = result.fillna(df[col_fallback])
                    return result.fillna(default)
                elif col_fallback and col_fallback in df.columns:
                    return df[col_fallback].fillna(default)
                return default

            # Coalesce year/week/manager from add side or drop side
            combined['year'] = safe_coalesce(combined, 'year_added', 'year_dropped', '')
            combined['week'] = safe_coalesce(combined, 'week_added', 'week_dropped', '')
            combined['manager'] = safe_coalesce(combined, 'manager_added', 'manager_dropped', '')

            combined['player_added'] = safe_coalesce(combined, 'player_name_added', 'player_name', '(None)')
            combined['pos_added'] = safe_coalesce(combined, 'position_added', 'position', '')
            combined['player_dropped'] = safe_coalesce(combined, 'player_name_dropped', default='(None)')
            combined['pos_dropped'] = safe_coalesce(combined, 'position_dropped', default='')

            # Numeric columns with safe handling
            if add_value_col in combined.columns:
                combined['spar_added'] = pd.to_numeric(combined[add_value_col], errors='coerce').fillna(0)
            else:
                combined['spar_added'] = 0

            # Check for dropped SPAR column (may have suffix from merge)
            drop_spar_col = f'{drop_value_col}_dropped' if f'{drop_value_col}_dropped' in combined.columns else drop_value_col
            if drop_spar_col in combined.columns:
                combined['spar_dropped'] = pd.to_numeric(combined[drop_spar_col], errors='coerce').fillna(0)
            else:
                combined['spar_dropped'] = 0

            combined['net_spar'] = combined['spar_added'] - combined['spar_dropped']
            combined['faab'] = pd.to_numeric(combined['faab_bid'], errors='coerce').fillna(0) if 'faab_bid' in combined.columns else 0

            # Total transaction score (adds + drops combined)
            # After merge, columns may have _added/_dropped suffixes
            add_score = 0
            for col in ['transaction_score', 'transaction_score_added', 'transaction_score_x']:
                if col in combined.columns:
                    add_score = pd.to_numeric(combined[col], errors='coerce').fillna(0)
                    break

            drop_score = 0
            for col in ['transaction_score_dropped', 'transaction_score_y']:
                if col in combined.columns:
                    drop_score = pd.to_numeric(combined[col], errors='coerce').fillna(0)
                    break

            combined['total_score'] = add_score + drop_score

            # Use pre-computed grade and tier
            combined['grade'] = combined['transaction_grade'].fillna('') if 'transaction_grade' in combined.columns else ''
            combined['value_tier'] = combined['faab_value_tier'].fillna('') if 'faab_value_tier' in combined.columns else ''
            combined['regret'] = combined['drop_regret_tier'].fillna('') if 'drop_regret_tier' in combined.columns else ''

            # Result indicator using pre-computed values
            def get_display_result(row):
                emoji = row.get('result_emoji', '')
                result = row.get('transaction_result', '')
                # Handle NaN values
                if pd.isna(emoji):
                    emoji = ''
                if pd.isna(result):
                    result = ''
                if emoji and result:
                    return f"{emoji} {result}"
                # Fallback based on net_spar
                net = row.get('net_spar', 0)
                if pd.isna(net):
                    net = 0
                if net > 100:
                    return "🏆 Excellent"
                elif net > 50:
                    return "✅ Great"
                elif net > 20:
                    return "👍 Good"
                elif net > 0:
                    return "➡️ Positive"
                elif net == 0:
                    return "➖ Neutral"
                else:
                    return "📉 Negative"

            combined['Result'] = combined.apply(get_display_result, axis=1)

            # Calculate efficiency
            combined['efficiency'] = combined.apply(
                lambda row: row['net_spar'] / row['faab'] if row['faab'] > 0 else None,
                axis=1
            )

            # Apply sorting
            if sort_by == "Score ↓":
                combined = combined.sort_values('total_score', ascending=False)
            elif sort_by == "SPAR ↓":
                combined = combined.sort_values('net_spar', ascending=False)
            elif sort_by == "Grade ↓":
                grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4, '': 5}
                combined['grade_sort'] = combined['grade'].map(grade_order).fillna(5)
                combined = combined.sort_values('grade_sort')
            elif sort_by == "FAAB ↓":
                combined = combined.sort_values('faab', ascending=False)
            elif sort_by == "ROI ↓":
                combined = combined.sort_values('efficiency', ascending=False, na_position='last')
            else:
                combined = combined.sort_values(['year', 'week'], ascending=[False, False])

            # Reordered columns for better cognitive flow: When → Who → What (Added) → What (Dropped) → Impact
            display_cols = ['week', 'year', 'manager', 'grade',
                           'player_added', 'pos_added', 'spar_added',
                           'player_dropped', 'pos_dropped', 'spar_dropped',
                           'net_spar', 'faab', 'efficiency', 'Result']
            display_cols = [c for c in display_cols if c in combined.columns]
            display_data = combined[display_cols].copy()

            # Apply color formatting using background gradient for SPAR columns
            def style_spar_column(val):
                """Return color for SPAR values."""
                if pd.isna(val) or val == 0:
                    return ''
                if val > 0:
                    intensity = min(abs(val) / 100, 1)  # Normalize to 0-1
                    return f'background-color: rgba(34, 197, 94, {intensity * 0.4})'  # Green
                else:
                    intensity = min(abs(val) / 100, 1)
                    return f'background-color: rgba(239, 68, 68, {intensity * 0.4})'  # Red

            st.markdown(f"**{len(display_data):,} transactions** (from {len(filtered):,} adds/drops)")

            # Create styled dataframe
            styled_df = display_data.style.applymap(
                style_spar_column,
                subset=[c for c in ['spar_added', 'spar_dropped', 'net_spar'] if c in display_data.columns]
            )

            st.dataframe(
                display_data,
                hide_index=True,
                use_container_width=True,
                height=500,
                column_config={
                    'week': st.column_config.NumberColumn("Wk", format="%d", width="small", help="Week of transaction"),
                    'year': st.column_config.TextColumn("Yr", width="small"),
                    'manager': st.column_config.TextColumn("Manager", width="medium"),
                    'grade': st.column_config.TextColumn("📊", width="small", help="Transaction grade (A-F)"),
                    'player_added': st.column_config.TextColumn("➕ Added", width="medium"),
                    'pos_added': st.column_config.TextColumn("Pos", width="small"),
                    'spar_added': st.column_config.NumberColumn("SPAR+", format="%.1f", help="SPAR gained from pickup"),
                    'player_dropped': st.column_config.TextColumn("➖ Dropped", width="medium"),
                    'pos_dropped': st.column_config.TextColumn("Pos", width="small"),
                    'spar_dropped': st.column_config.NumberColumn("SPAR-", format="%.1f", help="SPAR lost from drop"),
                    'net_spar': st.column_config.NumberColumn("NET", format="%+.1f", help="Net SPAR = SPAR+ minus SPAR-"),
                    'faab': st.column_config.NumberColumn("💰", format="$%.0f", help="FAAB spent"),
                    'efficiency': st.column_config.NumberColumn("$/PT", format="%.2f", help="SPAR per FAAB dollar"),
                    'Result': st.column_config.TextColumn("Outcome", width="medium", help="Transaction result classification"),
                }
            )

            csv = display_data.to_csv(index=False)
            st.download_button("📥 Export", csv, f"weekly_add_drops_{year_filter}.csv", "text/csv", key="weekly_export")

        else:
            # Fallback: simple display without grouping
            display_cols = ['week', 'year', 'manager', 'transaction_type', 'player_name', 'position',
                           'Result', 'transaction_grade', add_value_col, 'faab_bid', 'roi']
            display_cols = [c for c in display_cols if c in filtered.columns]

            st.dataframe(filtered[display_cols], hide_index=True, use_container_width=True)

    with tab2:
        st.markdown("### 📊 Transaction Analytics")
        _render_analytics_tab(t, adds, drops, add_value_col, drop_value_col)

    with tab3:
        st.markdown("### 🏆 Transaction Leaderboards")
        _render_leaderboards_tab(adds, drops, add_value_col, drop_value_col)


def _render_analytics_tab(t: pd.DataFrame, adds: pd.DataFrame, drops: pd.DataFrame,
                          add_value_col: str, drop_value_col: str) -> None:
    """Render the analytics tab using pre-computed metrics with themed charts."""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Transaction Grade Distribution")
        if 'transaction_grade' in t.columns and t['transaction_grade'].notna().any():
            grade_counts = t['transaction_grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F']).fillna(0)

            # Use themed chart if available
            if HAS_CHART_THEMES:
                fig = create_grade_bar_chart(grade_counts.to_dict(), "Transactions by Grade")
            else:
                colors = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728']
                fig = go.Figure(go.Bar(
                    x=grade_counts.index,
                    y=grade_counts.values,
                    marker_color=colors,
                    text=grade_counts.values,
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Transactions by Grade",
                    xaxis_title="Grade",
                    yaxis_title="Count",
                    showlegend=False,
                    height=350
                )
            st.plotly_chart(fig, use_container_width=True)

            # Calculate GPA
            grade_points = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
            total_points = sum(grade_points.get(g, 0) * c for g, c in grade_counts.items())
            total_count = grade_counts.sum()
            gpa = total_points / total_count if total_count > 0 else 0
            st.metric("Transaction GPA", f"{gpa:.2f}")
        else:
            st.info("Grade data not available - regenerate source data with new engagement metrics")

    with col2:
        st.markdown("#### 💰 FAAB Value Tier Distribution")
        if 'faab_value_tier' in adds.columns and adds['faab_value_tier'].notna().any():
            tier_counts = adds['faab_value_tier'].value_counts()
            tier_order = ['Steal', 'Great Value', 'Good Value', 'Fair', 'Overpay']
            tier_counts = tier_counts.reindex([tc for tc in tier_order if tc in tier_counts.index])

            # Use themed chart if available
            if HAS_CHART_THEMES:
                fig = create_faab_tier_chart(tier_counts.to_dict(), "FAAB Value Distribution (Adds Only)")
            else:
                colors = ['#0d47a1', '#1f77b4', '#2ca02c', '#ffbb78', '#d62728'][:len(tier_counts)]
                fig = go.Figure(go.Bar(
                    x=tier_counts.index,
                    y=tier_counts.values,
                    marker_color=colors,
                    text=tier_counts.values,
                    textposition='outside'
                ))
                fig.update_layout(
                    title="FAAB Value Distribution (Adds Only)",
                    xaxis_title="Value Tier",
                    yaxis_title="Count",
                    showlegend=False,
                    height=350
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("FAAB value tier data not available")

    # Drop Regret Analysis
    st.markdown("#### 😬 Drop Regret Analysis")
    col1, col2 = st.columns(2)

    with col1:
        if 'drop_regret_tier' in drops.columns and drops['drop_regret_tier'].notna().any():
            regret_counts = drops['drop_regret_tier'].value_counts()
            tier_order = ['No Regret', 'Minor Regret', 'Some Regret', 'Big Regret', 'Major Regret', 'Disaster']
            regret_counts = regret_counts.reindex([tc for tc in tier_order if tc in regret_counts.index])

            # Use themed chart if available
            if HAS_CHART_THEMES:
                fig = create_regret_bar_chart(regret_counts.to_dict(), "Drop Regret Distribution")
            else:
                colors = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728', '#8b0000'][:len(regret_counts)]
                fig = go.Figure(go.Bar(
                    x=regret_counts.index,
                    y=regret_counts.values,
                    marker_color=colors,
                    text=regret_counts.values,
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Drop Regret Distribution",
                    xaxis_title="Regret Level",
                    yaxis_title="Count",
                    showlegend=False,
                    height=350,
                    xaxis_tickangle=-45
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Drop regret data not available")

    with col2:
        if 'drop_regret_score' in drops.columns and drops['drop_regret_score'].notna().any():
            # Top regret drops
            st.markdown("##### 💔 Biggest Regret Drops")
            top_regrets = drops.nlargest(5, 'drop_regret_score')[
                ['player_name', 'manager', 'year', 'week', 'drop_regret_score', 'drop_regret_tier']
            ].copy()
            if len(top_regrets) > 0:
                st.dataframe(
                    top_regrets,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        'player_name': st.column_config.TextColumn("Player"),
                        'manager': st.column_config.TextColumn("Manager"),
                        'year': st.column_config.TextColumn("Year"),
                        'week': st.column_config.NumberColumn("Wk"),
                        'drop_regret_score': st.column_config.NumberColumn("SPAR Lost", format="%.1f"),
                        'drop_regret_tier': st.column_config.TextColumn("Regret"),
                    }
                )
        else:
            st.info("Drop regret scores not available")

    # Timing Analysis
    st.markdown("#### 📅 Transaction Timing")
    if 'timing_category' in t.columns and t['timing_category'].notna().any():
        timing_data = t.groupby('timing_category').agg({
            'transaction_type': 'count',
            'net_manager_spar_ros': 'mean'
        }).reset_index()
        timing_data.columns = ['Timing', 'Count', 'Avg NET SPAR']

        # Reorder
        timing_order = ['Early Season', 'Mid Season', 'Late Season', 'Playoffs']
        timing_data['sort_order'] = timing_data['Timing'].map({tc: i for i, tc in enumerate(timing_order)})
        timing_data = timing_data.sort_values('sort_order')

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(timing_data, x='Timing', y='Count', title="Transaction Volume by Timing")
            if HAS_CHART_THEMES:
                fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(timing_data, x='Timing', y='Avg NET SPAR', title="Avg NET SPAR by Timing",
                        color='Avg NET SPAR', color_continuous_scale='RdYlGn')
            if HAS_CHART_THEMES:
                fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)


def _render_leaderboards_tab(adds: pd.DataFrame, drops: pd.DataFrame,
                             add_value_col: str, drop_value_col: str) -> None:
    """Render the leaderboards tab."""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌟 Best Pickups (Highest SPAR)")
        if len(adds) > 0 and add_value_col in adds.columns and adds[add_value_col].notna().any():
            best = adds.nlargest(10, add_value_col)[
                ['year', 'week', 'player_name', 'position', 'manager',
                 'faab_bid', add_value_col, 'transaction_grade', 'faab_value_tier']
            ].copy()
            best['Rank'] = range(1, len(best) + 1)

            display_cols = ['Rank', 'year', 'week', 'player_name', 'position', 'manager',
                           add_value_col, 'faab_bid', 'transaction_grade']
            display_cols = [c for c in display_cols if c in best.columns]

            st.dataframe(
                best[display_cols],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Rank': st.column_config.NumberColumn("#", width="small"),
                    'year': st.column_config.TextColumn("Year", width="small"),
                    'week': st.column_config.NumberColumn("Wk", width="small"),
                    'player_name': st.column_config.TextColumn("Player"),
                    'position': st.column_config.TextColumn("Pos", width="small"),
                    'manager': st.column_config.TextColumn("Manager"),
                    add_value_col: st.column_config.NumberColumn("SPAR", format="%.1f"),
                    'faab_bid': st.column_config.NumberColumn("FAAB", format="$%.0f", width="small"),
                    'transaction_grade': st.column_config.TextColumn("Grade", width="small"),
                }
            )
        else:
            st.info("No add data available")

    with col2:
        st.markdown("#### 💎 Best Value (Steals)")
        if len(adds) > 0 and 'faab_value_tier' in adds.columns:
            steals = adds[adds['faab_value_tier'] == 'Steal'].copy()
            if len(steals) > 0:
                steals = steals.nlargest(10, add_value_col)[
                    ['year', 'week', 'player_name', 'position', 'manager',
                     'faab_bid', add_value_col, 'spar_efficiency']
                ].copy()
                steals['Rank'] = range(1, len(steals) + 1)

                st.dataframe(
                    steals[['Rank', 'year', 'week', 'player_name', 'manager', add_value_col, 'faab_bid', 'spar_efficiency']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        'Rank': st.column_config.NumberColumn("#", width="small"),
                        'year': st.column_config.TextColumn("Year", width="small"),
                        'week': st.column_config.NumberColumn("Wk", width="small"),
                        'player_name': st.column_config.TextColumn("Player"),
                        'manager': st.column_config.TextColumn("Manager"),
                        add_value_col: st.column_config.NumberColumn("SPAR", format="%.1f"),
                        'faab_bid': st.column_config.NumberColumn("FAAB", format="$%.0f"),
                        'spar_efficiency': st.column_config.NumberColumn("SPAR/$", format="%.2f"),
                    }
                )
            else:
                st.info("No steals found")
        else:
            st.info("Value tier data not available")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💔 Worst Drops (Most SPAR Lost)")
        if len(drops) > 0 and drop_value_col in drops.columns and drops[drop_value_col].notna().any():
            worst = drops.nlargest(10, drop_value_col)[
                ['year', 'week', 'player_name', 'position', 'manager',
                 drop_value_col, 'drop_regret_tier']
            ].copy()
            worst['Rank'] = range(1, len(worst) + 1)

            st.dataframe(
                worst[['Rank', 'year', 'week', 'player_name', 'position', 'manager', drop_value_col, 'drop_regret_tier']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Rank': st.column_config.NumberColumn("#", width="small"),
                    'year': st.column_config.TextColumn("Year", width="small"),
                    'week': st.column_config.NumberColumn("Wk", width="small"),
                    'player_name': st.column_config.TextColumn("Player"),
                    'position': st.column_config.TextColumn("Pos", width="small"),
                    'manager': st.column_config.TextColumn("Manager"),
                    drop_value_col: st.column_config.NumberColumn("SPAR Lost", format="%.1f"),
                    'drop_regret_tier': st.column_config.TextColumn("Regret"),
                }
            )
        else:
            st.info("No drop data available")

    with col2:
        st.markdown("#### 💰 Biggest FAAB Spends")
        if len(adds) > 0 and 'faab_bid' in adds.columns:
            big_spends = adds.nlargest(10, 'faab_bid')[
                ['year', 'week', 'player_name', 'position', 'manager',
                 'faab_bid', add_value_col, 'faab_value_tier']
            ].copy()
            big_spends['Rank'] = range(1, len(big_spends) + 1)

            st.dataframe(
                big_spends[['Rank', 'year', 'week', 'player_name', 'manager', 'faab_bid', add_value_col, 'faab_value_tier']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Rank': st.column_config.NumberColumn("#", width="small"),
                    'year': st.column_config.TextColumn("Year", width="small"),
                    'week': st.column_config.NumberColumn("Wk", width="small"),
                    'player_name': st.column_config.TextColumn("Player"),
                    'manager': st.column_config.TextColumn("Manager"),
                    'faab_bid': st.column_config.NumberColumn("FAAB", format="$%.0f"),
                    add_value_col: st.column_config.NumberColumn("SPAR", format="%.1f"),
                    'faab_value_tier': st.column_config.TextColumn("Value"),
                }
            )
        else:
            st.info("No FAAB data available")

    # A-Grade Transactions
    st.markdown("#### 🏆 A-Grade Transactions")
    if 'transaction_grade' in adds.columns:
        a_grades = adds[adds['transaction_grade'] == 'A'].copy()
        if len(a_grades) > 0:
            a_grades = a_grades.nlargest(10, add_value_col)[
                ['year', 'week', 'player_name', 'position', 'manager',
                 'faab_bid', add_value_col, 'transaction_result']
            ].copy()
            a_grades['Rank'] = range(1, len(a_grades) + 1)

            st.dataframe(
                a_grades[['Rank', 'year', 'week', 'player_name', 'manager', add_value_col, 'faab_bid', 'transaction_result']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Rank': st.column_config.NumberColumn("#", width="small"),
                    'year': st.column_config.TextColumn("Year", width="small"),
                    'week': st.column_config.NumberColumn("Wk", width="small"),
                    'player_name': st.column_config.TextColumn("Player"),
                    'manager': st.column_config.TextColumn("Manager"),
                    add_value_col: st.column_config.NumberColumn("SPAR", format="%.1f"),
                    'faab_bid': st.column_config.NumberColumn("FAAB", format="$%.0f"),
                    'transaction_result': st.column_config.TextColumn("Result"),
                }
            )
        else:
            st.info("No A-grade transactions found")
