#!/usr/bin/env python3
"""
Manager Draft Grades Tab

Shows manager-level draft performance with grades, scores, and year-over-year comparisons.
Uses all-time percentile for cross-year comparison.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Grade color mapping (uses base letter for +/- grades)
GRADE_COLORS = {
    'A': '#28a745',  # Green
    'B': '#6c9a1f',  # Yellow-green
    'C': '#ffc107',  # Yellow
    'D': '#fd7e14',  # Orange
    'F': '#dc3545',  # Red
}


def get_base_grade(grade) -> str:
    """Get the base letter (A, B, C, D, F) from a grade like A+ or B-."""
    if not grade or grade == 'N/A' or pd.isna(grade):
        return ''
    return str(grade)[0]


@st.fragment
def display_manager_draft_grades(draft_data: pd.DataFrame) -> None:
    """Display manager-level draft grades and performance analysis."""

    st.markdown("### 🎯 Manager Draft Report Card")
    st.markdown("*Draft performance grades based on all-time percentile ranking*")

    df = draft_data.copy()

    # Check if we have manager draft grade columns
    required_cols = ['manager', 'year', 'manager_draft_grade', 'manager_draft_score']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}. Run the draft_value_metrics_v3 pipeline to generate these metrics.")
        return

    # Get unique manager-year combinations (one row per manager-year)
    manager_cols = [
        'manager', 'year', 'manager_draft_score', 'manager_draft_grade',
        'manager_draft_percentile', 'manager_draft_percentile_year',
        'manager_draft_percentile_alltime', 'manager_total_spar',
        'manager_avg_spar', 'manager_hit_rate', 'manager_picks_count'
    ]
    available_cols = [c for c in manager_cols if c in df.columns]

    # Get one row per manager-year
    manager_df = df[df['manager'].notna() & (df['manager'] != '')].groupby(['manager', 'year']).first().reset_index()
    manager_df = manager_df[available_cols].dropna(subset=['manager_draft_score'])

    if manager_df.empty:
        st.warning("No manager draft grades available. Run the draft enrichment pipeline.")
        return

    # Ensure numeric
    numeric_cols = ['manager_draft_score', 'manager_draft_percentile', 'manager_draft_percentile_year',
                    'manager_draft_percentile_alltime', 'manager_total_spar', 'manager_avg_spar',
                    'manager_hit_rate', 'manager_picks_count']
    for col in numeric_cols:
        if col in manager_df.columns:
            manager_df[col] = pd.to_numeric(manager_df[col], errors='coerce')

    # Filters
    with st.expander("🔍 Filters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            years = sorted(manager_df['year'].dropna().unique().tolist(), reverse=True)
            # Default to all years, not just the last 3
            selected_years = st.multiselect("Year", years, default=years,
                                           key="mgr_grade_years")

        with col2:
            managers = sorted(manager_df['manager'].unique().tolist())
            selected_managers = st.multiselect("Manager", managers, default=[],
                                              key="mgr_grade_managers")

    # Apply filters
    filtered = manager_df.copy()
    if selected_years:
        filtered = filtered[filtered['year'].isin(selected_years)]
    if selected_managers:
        filtered = filtered[filtered['manager'].isin(selected_managers)]

    if filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_score = filtered['manager_draft_score'].mean()
        st.metric("Avg Draft Score", f"{avg_score:.2f}")
    with col2:
        avg_spar = filtered['manager_total_spar'].mean() if 'manager_total_spar' in filtered.columns else 0
        st.metric("Avg Total SPAR", f"{avg_spar:.2f}")
    with col3:
        # manager_hit_rate is already stored as percentage (e.g., 68.8 = 68.8%)
        avg_hit_rate = filtered['manager_hit_rate'].mean() if 'manager_hit_rate' in filtered.columns else 0
        st.metric("Avg Hit Rate", f"{avg_hit_rate:.0f}%")
    with col4:
        total_drafts = len(filtered)
        st.metric("Drafts Analyzed", f"{total_drafts}")

    st.divider()

    # View selector
    view = st.radio("View", ["Leaderboard", "Year-over-Year", "Career Summary"], horizontal=True,
                   key="mgr_grade_view")

    if view == "Leaderboard":
        display_leaderboard(filtered)
    elif view == "Year-over-Year":
        display_year_over_year(filtered)
    else:
        display_career_summary(manager_df)  # Use unfiltered for career summary


def display_leaderboard(df: pd.DataFrame) -> None:
    """Display draft grade leaderboard."""

    st.markdown("#### 🏆 Draft Grade Leaderboard")
    st.markdown("*Ranked by Total SPAR*")

    # Sort by total SPAR descending
    leaderboard = df.sort_values('manager_total_spar', ascending=False).copy()

    # Format for display
    display_cols = ['year', 'manager', 'manager_draft_grade', 'manager_draft_score',
                    'manager_draft_percentile_alltime', 'manager_total_spar', 'manager_hit_rate']
    display_cols = [c for c in display_cols if c in leaderboard.columns]

    # Rename for display
    rename_map = {
        'year': 'Year',
        'manager': 'Manager',
        'manager_draft_grade': 'Grade',
        'manager_draft_score': 'Score',
        'manager_draft_percentile_alltime': 'All-Time %ile',
        'manager_total_spar': 'Total SPAR',
        'manager_hit_rate': 'Hit Rate'
    }

    display_df = leaderboard[display_cols].copy()
    display_df = display_df.rename(columns=rename_map)

    # Format columns
    if 'Score' in display_df.columns:
        display_df['Score'] = display_df['Score'].round(2)
    if 'All-Time %ile' in display_df.columns:
        display_df['All-Time %ile'] = display_df['All-Time %ile'].round(1).astype(str) + '%'
    if 'Total SPAR' in display_df.columns:
        display_df['Total SPAR'] = display_df['Total SPAR'].round(2)
    if 'Hit Rate' in display_df.columns:
        # manager_hit_rate is already stored as percentage (e.g., 68.8 = 68.8%)
        display_df['Hit Rate'] = display_df['Hit Rate'].round(0).astype(str) + '%'

    # Add rank column
    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))

    # Display with grade colors (use base grade for +/- grades)
    def style_grade(val):
        base = get_base_grade(val)
        if base in GRADE_COLORS:
            return f'background-color: {GRADE_COLORS[base]}; color: white; font-weight: bold;'
        return ''

    styled = display_df.style.applymap(style_grade, subset=['Grade'] if 'Grade' in display_df.columns else [])
    st.dataframe(styled, hide_index=True, use_container_width=True)

    # Grade distribution chart
    if 'Grade' in display_df.columns:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Grade Distribution")
            # Group by base grade for chart (A+/A/A- all count as A)
            leaderboard['_base_grade'] = leaderboard['manager_draft_grade'].apply(get_base_grade)
            grade_counts = leaderboard['_base_grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F']).fillna(0)

            fig = px.bar(
                x=grade_counts.index,
                y=grade_counts.values,
                color=grade_counts.index,
                color_discrete_map=GRADE_COLORS,
                labels={'x': 'Grade', 'y': 'Count'}
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Score Distribution")
            fig = px.histogram(leaderboard, x='manager_draft_score', nbins=15,
                             color_discrete_sequence=['#667eea'])
            fig.update_layout(xaxis_title='Draft Score', yaxis_title='Count', height=300)
            st.plotly_chart(fig, use_container_width=True)


def display_year_over_year(df: pd.DataFrame) -> None:
    """Display year-over-year draft performance."""

    st.markdown("#### 📈 Year-over-Year Performance")

    # Pivot table: managers as rows, years as columns
    pivot_data = df.pivot_table(
        index='manager',
        columns='year',
        values='manager_draft_grade',
        aggfunc='first'
    )

    # Sort managers by most recent year performance (handle +/- grades)
    recent_year = pivot_data.columns.max()
    # Sort by base grade first, then +/- modifier (A+ < A < A-)
    grade_order = {
        'A+': 0, 'A': 1, 'A-': 2,
        'B+': 3, 'B': 4, 'B-': 5,
        'C+': 6, 'C': 7, 'C-': 8,
        'D+': 9, 'D': 10, 'D-': 11,
        'F': 12
    }
    if recent_year in pivot_data.columns:
        pivot_data['_sort'] = pivot_data[recent_year].map(grade_order).fillna(13)
        pivot_data = pivot_data.sort_values('_sort')
        pivot_data = pivot_data.drop(columns=['_sort'])

    # Style the pivot table (use base grade for color)
    def color_grades(val):
        if pd.isna(val):
            return ''
        base = get_base_grade(val)
        if base in GRADE_COLORS:
            return f'background-color: {GRADE_COLORS[base]}; color: white; font-weight: bold; text-align: center;'
        return ''

    styled = pivot_data.style.applymap(color_grades)
    st.dataframe(styled, use_container_width=True)

    # Line chart of scores over time
    st.markdown("---")
    st.markdown("#### Draft Score Trends")

    if 'manager_draft_score' in df.columns:
        fig = px.line(
            df.sort_values('year'),
            x='year',
            y='manager_draft_score',
            color='manager',
            markers=True,
            labels={'year': 'Year', 'manager_draft_score': 'Draft Score', 'manager': 'Manager'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def display_career_summary(df: pd.DataFrame) -> None:
    """Display career draft summary for each manager."""

    st.markdown("#### 🎖️ Career Draft Summary")
    st.markdown("*All-time performance across all drafts*")

    # Aggregate by manager
    career = df.groupby('manager').agg({
        'manager_draft_score': ['mean', 'min', 'max', 'count'],
        'manager_draft_grade': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
        'manager_total_spar': 'sum',
        'manager_hit_rate': 'mean'
    }).round(2)

    career.columns = ['Avg Score', 'Worst Score', 'Best Score', 'Drafts', 'Typical Grade', 'Career SPAR', 'Avg Hit Rate']
    career = career.sort_values('Avg Score', ascending=False)

    # Add career rank
    career.insert(0, 'Career Rank', range(1, len(career) + 1))

    # Format - manager_hit_rate is already stored as percentage (e.g., 68.8 = 68.8%)
    career['Avg Hit Rate'] = career['Avg Hit Rate'].round(0).astype(str) + '%'
    career['Career SPAR'] = career['Career SPAR'].round(2)

    # Style grades (use base grade for +/- grades)
    def color_grades(val):
        base = get_base_grade(val)
        if base in GRADE_COLORS:
            return f'background-color: {GRADE_COLORS[base]}; color: white; font-weight: bold;'
        return ''

    styled = career.style.applymap(color_grades, subset=['Typical Grade'])
    st.dataframe(styled, use_container_width=True)

    # Career comparison chart
    st.markdown("---")
    st.markdown("#### Career Performance Comparison")

    fig = go.Figure()

    # Add bar for average score
    fig.add_trace(go.Bar(
        name='Avg Score',
        x=career.index,
        y=career['Avg Score'],
        marker_color='#667eea'
    ))

    # Add scatter for best/worst
    fig.add_trace(go.Scatter(
        name='Best Draft',
        x=career.index,
        y=career['Best Score'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='#28a745')
    ))

    fig.add_trace(go.Scatter(
        name='Worst Draft',
        x=career.index,
        y=career['Worst Score'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=12, color='#dc3545')
    ))

    fig.update_layout(
        xaxis_title='Manager',
        yaxis_title='Draft Score',
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)
