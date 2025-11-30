#!/usr/bin/env python3
"""
Integrated Draft Data Overview - drop-in replacement
This file is defensive: works when given either the output of load_draft_data()
(or a dict containing a 'Draft History' DataFrame) or a DataFrame directly.
It inlines value analysis to avoid import-time failures and provides helpful
error messages so the Streamlit UI doesn't go blank when data is unexpected.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Optional


def setup_page_config() -> None:
    """Small page styling helper retained from the previous file."""
    st.markdown(
        """
        <style>
        .info-box { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:1rem; border-radius:8px; color:white }
        .info-box h3{ margin:0 }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def validate_and_prep_data(draft_data: pd.DataFrame) -> pd.DataFrame:
    """Validate basic shape and coerce common columns.

    Raises ValueError when required columns are missing.
    Returns a cleaned copy (never the original object) to avoid side effects.
    """
    if draft_data is None:
        raise ValueError("draft_data is None")
    if not hasattr(draft_data, 'columns'):
        raise ValueError("draft_data is not a DataFrame-like object")

    required = ['player', 'yahoo_position', 'cost', 'year', 'manager']
    missing = [c for c in required if c not in draft_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = draft_data.copy()

    # Ensure optional analytics columns exist
    for col in ['points', 'season_ppg', 'is_keeper_status', 'is_keeper_cost', 'cost_bucket', 'pick', 'round']:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce numerics safely
    num_cols = ['cost', 'points', 'season_ppg', 'year', 'cost_bucket', 'pick', 'round', 'is_keeper_status', 'is_keeper_cost']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Clean string columns
    for c in ['manager', 'player', 'yahoo_position']:
        if c in df.columns:
            try:
                df[c] = df[c].astype(str).str.strip()
            except Exception:
                # fallback: leave as-is
                pass

    # Derive is_keeper_cost when possible
    if 'is_keeper_cost' not in draft_data.columns or df['is_keeper_cost'].isna().all():
        if 'is_keeper_status' in df.columns and 'cost' in df.columns:
            try:
                df['is_keeper_cost'] = df['cost'].where(df['is_keeper_status'] == 1, 0).fillna(0)
            except Exception:
                df['is_keeper_cost'] = 0

    # Remove rows with manager literally 'nan' (string) or missing manager
    if 'manager' in df.columns:
        try:
            mgr_mask = df['manager'].astype(str).str.lower() != 'nan'
            mgr_mask = mgr_mask & df['manager'].notna()
            df = df[mgr_mask]
        except Exception:
            pass

    # As a final defensive step, if df ended up empty return an informative empty DataFrame
    return df.reset_index(drop=True)


@st.fragment
def display_value_analysis(df: pd.DataFrame) -> None:
    """Inline value analysis using Manager SPAR (actual value captured while rostered)."""
    st.header("💰 Draft Value Analysis")

    st.info("Using Manager SPAR: actual value captured while players were on your roster. Manager SPAR / Cost = draft ROI. Higher is better.")

    if df is None or df.empty:
        st.warning("No data available for value analysis")
        return

    # Ensure numeric
    df = df.copy()
    df['cost'] = pd.to_numeric(df.get('cost'), errors='coerce')
    df['season_ppg'] = pd.to_numeric(df.get('season_ppg'), errors='coerce')
    df['points'] = pd.to_numeric(df.get('points'), errors='coerce')

    # Use manager_spar (actual value while rostered) with fallback to spar
    if 'manager_spar' in df.columns:
        df['manager_spar'] = pd.to_numeric(df['manager_spar'], errors='coerce')
        spar_col = 'manager_spar'
    elif 'spar' in df.columns:
        df['spar'] = pd.to_numeric(df['spar'], errors='coerce')
        spar_col = 'spar'
    else:
        st.warning('⚠️ No SPAR data available for value analysis')
        return

    # Check if we have SPAR data
    if df[spar_col].isna().all() or (df[spar_col] == 0).all():
        st.warning('⚠️ No SPAR data available for value analysis')
        return

    # Calculate draft_roi (SPAR/$) on the fly if not present
    if 'draft_roi' not in df.columns or df['draft_roi'].isna().all():
        df['draft_roi'] = (df[spar_col] / df['cost'].replace(0, pd.NA)).fillna(0)
    else:
        df['draft_roi'] = pd.to_numeric(df.get('draft_roi'), errors='coerce')

    # Filter to valid data (cost > 0 and SPAR exists)
    value_df = df[(df['cost'] > 0) & (df[spar_col].notna())].copy()
    if value_df.empty:
        st.warning('⚠️ No rows with valid cost and SPAR for value analysis')
        return

    value_df['value'] = value_df['draft_roi']
    value_df['points_per_dollar'] = value_df[spar_col] / value_df['cost'].clip(lower=0.1)
    value_df['spar_display'] = value_df[spar_col]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Avg SPAR/$', f"{value_df['value'].mean():.3f}")
    with col2:
        st.metric('Best SPAR/$', f"{value_df['value'].max():.3f}")
    with col3:
        st.metric('Median SPAR/$', f"{value_df['points_per_dollar'].median():.2f}")
    with col4:
        st.metric('Median Cost', f"${value_df['cost'].median():.0f}")

    st.markdown('---')

    # Value by position
    try:
        pos = value_df.groupby('yahoo_position').agg({
            'value': ['mean', 'median', 'max'],
            'cost': 'mean',
            'season_ppg': 'mean',
            'player': 'count'
        }).round(3)
        pos.columns = ['Avg SPAR/$', 'Median SPAR/$', 'Max SPAR/$', 'Avg Cost', 'Avg PPG', 'Count']
        pos = pos.sort_values('Avg SPAR/$', ascending=False)
        st.subheader('📊 SPAR/$ Efficiency by Position')
        st.dataframe(pos, use_container_width=True)
    except Exception as e:
        st.warning(f'Could not compute value by position: {e}')

    st.markdown('---')

    # Top value picks
    try:
        cols = ['year', 'player', 'yahoo_position', 'manager', 'cost', 'season_ppg', 'spar_display', 'value']
        available_cols = [c for c in cols if c in value_df.columns]
        top = value_df.nlargest(20, 'value')[available_cols].copy()
        if 'cost' in top.columns:
            top['cost'] = top['cost'].apply(lambda x: f"${x:.0f}" if pd.notna(x) else x)
        if 'spar_display' in top.columns:
            top['spar_display'] = top['spar_display'].round(1)
        top['value'] = top['value'].round(3)
        if 'season_ppg' in top.columns:
            top['season_ppg'] = top['season_ppg'].round(2)
        col_names = ['Year', 'Player', 'Pos', 'Manager', 'Cost', 'PPG', 'Manager SPAR', 'SPAR/$']
        top.columns = col_names[:len(top.columns)]

        st.subheader('🏆 Best Value Picks (Top 20 by Manager SPAR/$)')
        st.dataframe(top, hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f'Could not render top value picks: {e}')


@st.fragment
def display_draft_data_overview(df_dict: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    """Main entry point used by the Streamlit app.

    Accepts either: (a) a dict like {'Draft History': DataFrame, ...} or
    (b) a DataFrame directly. If given a dict, it looks for the 'Draft History' key.
    """
    setup_page_config()

    st.markdown('<div class="info-box"><h3>📊 Draft Analysis Hub</h3><p style="margin:0.25rem 0 0 0;opacity:0.95">Comprehensive insights into your fantasy draft history</p></div>', unsafe_allow_html=True)

    # Resolve input
    draft_data = None
    if isinstance(df_dict, dict):
        draft_data = df_dict.get('Draft History')
        if draft_data is None:
            st.error("'Draft History' key missing from provided dict")
            st.write('Available keys:', list(df_dict.keys()))
            # show any non-empty DataFrames to help debug
            for k, v in df_dict.items():
                if hasattr(v, 'columns') and len(v) > 0:
                    st.write(f'**{k}**: {len(v):,} rows')
                    st.dataframe(v.head(5))
            return
    elif hasattr(df_dict, 'columns'):
        draft_data = df_dict
    else:
        st.error(f'Unsupported input type: {type(df_dict)}')
        return

    # Validate / prep data
    try:
        df = validate_and_prep_data(draft_data)
    except Exception as e:
        st.error(f'Error validating draft data: {e}')
        # show sample for debugging
        try:
            st.write('Incoming sample:')
            st.dataframe((draft_data.head(20) if hasattr(draft_data, 'head') else draft_data))
        except Exception:
            pass
        return

    if df.empty:
        st.warning('Validated draft DataFrame is empty — showing raw sample')
        try:
            st.dataframe(draft_data.head(50))
        except Exception:
            pass
        return

    # Data quality metrics
    try:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Draft Picks', f"{len(df):,}")
        with col2:
            if 'year' in df.columns and df['year'].notna().any():
                st.metric('Year Range', f"{int(df['year'].min())}-{int(df['year'].max())}")
        with col3:
            if 'manager' in df.columns:
                mgrs = df['manager'].dropna().astype(str)
                mgrs = mgrs[mgrs.str.lower() != 'nan']
                st.metric('Active Managers', mgrs.nunique())
        with col4:
            if 'cost' in df.columns:
                costs = pd.to_numeric(df['cost'], errors='coerce')
                avg = costs[costs > 0].mean()
                if pd.notna(avg):
                    st.metric('Avg Draft Cost', f"${avg:.1f}")
    except Exception as e:
        st.warning(f'Could not compute quality metrics: {e}')

    st.markdown('---')

    # Filters
    with st.expander('🎯 Filter All Views', expanded=False):
        try:
            years = sorted(df['year'].dropna().unique().tolist())
            if years:
                year_range = st.select_slider('Year Range', options=years, value=(years[0], years[-1]))
            else:
                year_range = (int(df['year'].min()) if 'year' in df.columns and df['year'].notna().any() else None, int(df['year'].max()) if 'year' in df.columns and df['year'].notna().any() else None)

            positions = sorted(df['yahoo_position'].dropna().unique().tolist()) if 'yahoo_position' in df.columns else []
            selected_positions = st.multiselect('Positions', options=positions, default=positions)

            managers = sorted(df['manager'].dropna().astype(str).unique().tolist()) if 'manager' in df.columns else []
            selected_managers = st.multiselect('Managers', options=managers, default=managers)

            filters = {
                'year_range': year_range,
                'positions': selected_positions,
                'managers': selected_managers,
            }
            # Quick preview count
            try:
                mask = (
                    (df['year'] >= filters['year_range'][0]) & (df['year'] <= filters['year_range'][1]) &
                    (df['yahoo_position'].isin(filters['positions'])) & (df['manager'].isin(filters['managers']))
                )
                st.info(f"Filtering {mask.sum():,} picks from {len(df):,} total")
            except Exception:
                pass
        except Exception as e:
            st.warning(f'Could not show filters: {e}')
            filters = {'year_range': (df['year'].min(), df['year'].max()), 'positions': df['yahoo_position'].unique().tolist() if 'yahoo_position' in df.columns else [], 'managers': df['manager'].unique().tolist() if 'manager' in df.columns else []}

    st.markdown('---')

    # Flattened tab structure with integrated visualizations
    tabs = st.tabs(["📋 Summary", "🎯 Performance", "💰 Value", "🔧 Optimizer", "📈 Trends", "💵 Pricing", "🏆 Career", "🔒 Keeper Analysis", "🎖️ Manager Grades", "📜 Report Card"])

    # Try to import submodules but fail gracefully
    sub_import_errors = []
    display_draft_summary = None
    display_scoring_outcomes = None
    display_draft_optimizer = None
    display_draft_preferences = None
    display_draft_overview = None
    display_career_draft = None

    try:
        from .draft_summary import display_draft_summary as _s
        display_draft_summary = _s
    except Exception as e:
        sub_import_errors.append(f'draft_summary: {e}')

    try:
        from .draft_scoring_outcomes import display_scoring_outcomes as _p
        display_scoring_outcomes = _p
    except Exception as e:
        sub_import_errors.append(f'draft_scoring_outcomes: {e}')

    try:
        from .draft_optimizer import display_draft_optimizer as _o
        display_draft_optimizer = _o
    except Exception as e:
        sub_import_errors.append(f'draft_optimizer: {e}')

    try:
        from .draft_preferences import display_draft_preferences as _pref
        display_draft_preferences = _pref
    except Exception as e:
        sub_import_errors.append(f'draft_preferences: {e}')

    try:
        from .draft_overviews import display_draft_overview as _ov
        display_draft_overview = _ov
    except Exception as e:
        sub_import_errors.append(f'draft_overviews: {e}')

    try:
        from .career_draft_stats import display_career_draft as _c
        display_career_draft = _c
    except Exception as e:
        sub_import_errors.append(f'career_draft_stats: {e}')

    display_manager_draft_grades = None
    try:
        from .manager_draft_grades import display_manager_draft_grades as _mgr
        display_manager_draft_grades = _mgr
    except Exception as e:
        sub_import_errors.append(f'manager_draft_grades: {e}')

    display_draft_report_card = None
    try:
        from .draft_report_card import display_draft_report_card as _rc
        display_draft_report_card = _rc
    except Exception as e:
        sub_import_errors.append(f'draft_report_card: {e}')

    if sub_import_errors:
        st.info('Some draft feature modules could not be imported:')
        for msg in sub_import_errors:
            st.write('-', msg)

    # Tab 0: Summary
    with tabs[0]:
        st.markdown('*Quick overview of draft picks and costs*')
        if display_draft_summary:
            try:
                display_draft_summary(df)
            except Exception as e:
                st.error(f'Summary rendering error: {e}')
                st.dataframe(df.head(50))
        else:
            st.warning('Summary module unavailable — showing raw data preview')
            st.dataframe(df)

    # Tab 1: Performance
    with tabs[1]:
        st.markdown('*Player scoring outcomes and rankings*')
        if display_scoring_outcomes:
            try:
                display_scoring_outcomes(df)
            except Exception as e:
                st.error(f'Performance rendering error: {e}')
        else:
            st.info('Performance module unavailable')

        # Add Round Efficiency visualization
        st.markdown('---')
        st.subheader('📈 Draft Round Efficiency')
        try:
            from .graphs.draft_round_efficiency import display_draft_round_efficiency
            display_draft_round_efficiency(prefix="performance_round_eff")
        except Exception as e:
            st.warning(f'Round efficiency graph unavailable: {e}')

    # Tab 2: Value (inline)
    with tabs[2]:
        st.markdown('*Draft value and ROI metrics*')
        display_value_analysis(df)

    # Tab 3: Optimizer
    with tabs[3]:
        st.markdown('*Build optimal lineups with constraints*')
        if display_draft_optimizer:
            try:
                display_draft_optimizer(df)
            except Exception as e:
                st.error(f'Optimizer rendering error: {e}')
        else:
            st.info('Optimizer module unavailable')

    # Tab 4: Trends
    with tabs[4]:
        st.markdown('*Historical draft patterns and preferences*')
        if display_draft_preferences:
            try:
                display_draft_preferences(df)
            except Exception as e:
                st.error(f'Trends rendering error: {e}')
        else:
            st.info('Preferences module unavailable')

        # Add Spending & Market Trends visualizations
        st.markdown('---')
        st.subheader('📊 Spending & Market Analysis')
        try:
            from .graphs.draft_spending_trends import display_draft_spending_trends
            from .graphs.draft_market_trends import display_draft_market_trends

            trend_tabs = st.tabs(["💰 Spending Trends", "🔥 Market Trends"])
            with trend_tabs[0]:
                display_draft_spending_trends(prefix="trends_spending")
            with trend_tabs[1]:
                display_draft_market_trends(prefix="trends_market")
        except Exception as e:
            st.warning(f'Trend graphs unavailable: {e}')

    # Tab 5: Pricing
    with tabs[5]:
        st.markdown('*Average costs by position and tier*')
        if display_draft_overview:
            try:
                display_draft_overview(df)
            except Exception as e:
                st.error(f'Pricing rendering error: {e}')
        else:
            st.info('Pricing module unavailable')

    # Tab 6: Career
    with tabs[6]:
        st.markdown('*Long-term player draft history*')
        if display_career_draft:
            try:
                display_career_draft(df)
            except Exception as e:
                st.error(f'Career rendering error: {e}')
        else:
            st.info('Career module unavailable')

    # Tab 7: Keeper Analysis
    with tabs[7]:
        st.markdown('*Keeper vs drafted player performance and value*')
        try:
            from .graphs.draft_keeper_analysis import display_draft_keeper_analysis
            display_draft_keeper_analysis(prefix="keeper_analysis")
        except Exception as e:
            st.warning(f'Keeper analysis unavailable: {e}')

    # Tab 8: Manager Grades
    with tabs[8]:
        st.markdown('*Manager-level draft grades and all-time performance*')
        if display_manager_draft_grades:
            try:
                display_manager_draft_grades(df)
            except Exception as e:
                st.error(f'Manager grades rendering error: {e}')
        else:
            st.info('Manager grades module unavailable')

    # Tab 9: Report Card
    with tabs[9]:
        st.markdown('*Official draft transcript styled as a report card*')
        if display_draft_report_card:
            try:
                display_draft_report_card(df)
            except Exception as e:
                st.error(f'Report card rendering error: {e}')
        else:
            st.info('Report card module unavailable')
