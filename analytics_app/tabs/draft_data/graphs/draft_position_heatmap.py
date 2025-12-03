#!/usr/bin/env python3
"""
draft_spending_trends.py - Fixed version of cost over time analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.core import T, run_query


@st.fragment
def display_draft_spending_trends(prefix=""):
    """
    Display draft spending trends over time - FIXED VERSION
    Shows how managers spend their draft budget across positions and years
    """
    st.subheader("💰 Draft Spending Trends")

    # Load draft data
    with st.spinner("Loading draft data..."):
        draft_data = run_query(f"""
            SELECT
                year, manager, player, yahoo_position,
                round, pick, cost,
                COALESCE(TRY_CAST(is_keeper_status AS INTEGER), 0) as is_keeper,
                COALESCE(spar, 0) as spar
            FROM {T['draft']}
            WHERE cost > 0
            ORDER BY year DESC, pick
        """)

    if draft_data.empty:
        st.warning("No draft data available.")
        return

    # Clean manager names
    draft_data['manager'] = draft_data['manager'].fillna('Unknown').str.strip()
    draft_data = draft_data[draft_data['manager'] != '']

    # Get unique managers and positions
    managers = sorted(draft_data['manager'].unique())
    positions = sorted(draft_data['yahoo_position'].unique())

    # UI Controls
    col1, col2 = st.columns(2)

    with col1:
        selected_manager = st.selectbox(
            "Manager",
            options=['League Average'] + managers,
            index=0,
            key=f"{prefix}_manager"
        )

    with col2:
        selected_position = st.selectbox(
            "Position",
            options=['All Positions'] + positions,
            index=0,
            key=f"{prefix}_position"
        )

    # Advanced options
    with st.expander("Show/Hide Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            show_drafted = st.checkbox("Show Drafted", value=True, key=f"{prefix}_drafted")
        with col2:
            show_keepers = st.checkbox("Show Keepers", value=True, key=f"{prefix}_keepers")

        if selected_position == 'All Positions':
            st.markdown("**Select Positions to Include:**")
            cols = st.columns(len(positions))
            include_positions = []
            for i, pos in enumerate(positions):
                if cols[i].checkbox(pos, value=True, key=f"{prefix}_pos_{pos}"):
                    include_positions.append(pos)
        else:
            include_positions = None

    # Filter by keeper status
    filtered_data = draft_data.copy()

    if not (show_drafted and show_keepers):
        if show_drafted:
            filtered_data = filtered_data[filtered_data['is_keeper'] == 0]
        elif show_keepers:
            filtered_data = filtered_data[filtered_data['is_keeper'] == 1]
        else:
            st.info("Please select at least one option (Drafted or Keepers)")
            return

    # Create the visualization
    fig = go.Figure()

    if selected_position == 'All Positions':
        # Show spending by position over time
        if include_positions:
            if selected_manager == 'League Average':
                # League average: sum all spending by position/year, then divide by number of managers
                for pos in include_positions:
                    pos_data = filtered_data[filtered_data['yahoo_position'] == pos]
                    yearly_totals = pos_data.groupby('year').agg({
                        'cost': 'sum',
                        'manager': 'nunique'
                    }).reset_index()
                    yearly_totals['avg_cost'] = yearly_totals['cost'] / yearly_totals['manager']

                    fig.add_trace(go.Scatter(
                        x=yearly_totals['year'],
                        y=yearly_totals['avg_cost'],
                        mode='lines+markers',
                        name=pos,
                        line=dict(width=3),
                        marker=dict(size=10),
                        hovertemplate=f'<b>{pos}</b><br>' +
                                      'Year: %{x}<br>' +
                                      'Avg Spending: $%{y:.2f}<extra></extra>'
                    ))
            else:
                # Specific manager: show their spending by position
                mgr_data = filtered_data[filtered_data['manager'] == selected_manager]
                for pos in include_positions:
                    pos_data = mgr_data[mgr_data['yahoo_position'] == pos]
                    yearly_totals = pos_data.groupby('year')['cost'].sum().reset_index()

                    fig.add_trace(go.Scatter(
                        x=yearly_totals['year'],
                        y=yearly_totals['cost'],
                        mode='lines+markers',
                        name=pos,
                        line=dict(width=3),
                        marker=dict(size=10),
                        hovertemplate=f'<b>{pos}</b><br>' +
                                      'Year: %{x}<br>' +
                                      'Total Spending: $%{y:.2f}<extra></extra>'
                    ))

    else:
        # Show spending on specific position over time
        pos_data = filtered_data[filtered_data['yahoo_position'] == selected_position]

        if selected_manager == 'League Average':
            # Average spending per manager on this position
            yearly_avg = pos_data.groupby('year').agg({
                'cost': 'sum',
                'manager': 'nunique'
            }).reset_index()
            yearly_avg['avg_cost'] = yearly_avg['cost'] / yearly_avg['manager']

            fig.add_trace(go.Scatter(
                x=yearly_avg['year'],
                y=yearly_avg['avg_cost'],
                mode='lines+markers',
                name='League Average',
                line=dict(width=3, color='blue'),
                marker=dict(size=10),
                hovertemplate='Year: %{x}<br>' +
                              'Avg Spending: $%{y:.2f}<extra></extra>'
            ))
        else:
            # Specific manager's spending on this position
            mgr_pos_data = pos_data[pos_data['manager'] == selected_manager]
            yearly_totals = mgr_pos_data.groupby('year')['cost'].sum().reset_index()

            fig.add_trace(go.Scatter(
                x=yearly_totals['year'],
                y=yearly_totals['cost'],
                mode='lines+markers',
                name=selected_manager,
                line=dict(width=3, color='green'),
                marker=dict(size=10),
                hovertemplate='Year: %{x}<br>' +
                              'Total Spending: $%{y:.2f}<extra></extra>'
            ))

    # Update layout
    title = f"Draft Spending: {selected_manager} - {selected_position}"
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Cost ($)",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.15
        )
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_heatmap_chart")

    # Summary statistics table
    st.markdown("### 📊 Summary Statistics")

    if selected_position == 'All Positions' and include_positions:
        summary_data = []
        for pos in include_positions:
            pos_subset = filtered_data[filtered_data['yahoo_position'] == pos]
            if selected_manager != 'League Average':
                pos_subset = pos_subset[pos_subset['manager'] == selected_manager]

            summary_row = {
                'Position': pos,
                'Total Spent': pos_subset['cost'].sum(),
                'Avg per Year': pos_subset.groupby('year')['cost'].sum().mean(),
                'Total Players': len(pos_subset),
                'Avg Cost per Player': pos_subset['cost'].mean()
            }

            # Add SPAR metrics if available
            if 'spar' in pos_subset.columns:
                summary_row['Total SPAR'] = pos_subset['spar'].sum()
                summary_row['Avg SPAR per Player'] = pos_subset['spar'].mean()
                total_cost = pos_subset['cost'].sum()
                if total_cost > 0:
                    summary_row['SPAR/$'] = pos_subset['spar'].sum() / total_cost
                else:
                    summary_row['SPAR/$'] = 0

            summary_data.append(summary_row)

        summary_df = pd.DataFrame(summary_data).round(2)

        # Configure column formatting
        column_config = {}
        if 'Total SPAR' in summary_df.columns:
            column_config['Total SPAR'] = st.column_config.NumberColumn('Total SPAR', format='%.1f')
            column_config['Avg SPAR per Player'] = st.column_config.NumberColumn('Avg SPAR/Player', format='%.1f')
            column_config['SPAR/$'] = st.column_config.NumberColumn('SPAR/$', format='%.2f')

        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
            column_config=column_config if column_config else None
        )

    else:
        # Single position summary
        pos_subset = filtered_data[filtered_data['yahoo_position'] == selected_position]
        if selected_manager != 'League Average':
            pos_subset = pos_subset[pos_subset['manager'] == selected_manager]

        # Calculate metrics
        total_spent = pos_subset['cost'].sum()
        avg_per_year = pos_subset.groupby('year')['cost'].sum().mean()
        total_players = len(pos_subset)
        avg_cost = pos_subset['cost'].mean()

        # Check for SPAR metrics
        has_spar = 'spar' in pos_subset.columns and pos_subset['spar'].sum() != 0

        if has_spar:
            # Show 7 metrics in two rows
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Spent", f"${total_spent:.2f}")
            with col2:
                st.metric("Avg per Year", f"${avg_per_year:.2f}")
            with col3:
                st.metric("Total Players", f"{total_players}")
            with col4:
                st.metric("Avg Cost/Player", f"${avg_cost:.2f}")

            col5, col6, col7 = st.columns(3)
            total_spar = pos_subset['spar'].sum()
            avg_spar = pos_subset['spar'].mean()
            spar_per_dollar = total_spar / total_spent if total_spent > 0 else 0

            with col5:
                st.metric("Total SPAR", f"{total_spar:.1f}")
            with col6:
                st.metric("Avg SPAR/Player", f"{avg_spar:.1f}")
            with col7:
                st.metric("SPAR/$", f"{spar_per_dollar:.2f}")
        else:
            # Show 4 metrics in single row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Spent", f"${total_spent:.2f}")
            with col2:
                st.metric("Avg per Year", f"${avg_per_year:.2f}")
            with col3:
                st.metric("Total Players", f"{total_players}")
            with col4:
                st.metric("Avg Cost/Player", f"${avg_cost:.2f}")
