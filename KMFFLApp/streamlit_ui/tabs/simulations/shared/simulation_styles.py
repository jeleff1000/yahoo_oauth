"""
Shared styling and components for Simulation tabs.

Provides:
- Unified navigation styling matching Weekly/Season/Career sections
- Sticky subnavigation
- Summary tiles
- Compact week selectors
- Manager filter dropdown
- Mobile-optimized layouts
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any


def apply_simulation_styles():
    """
    Apply comprehensive simulation-specific CSS styles.

    Key features:
    - Unified purple accent navigation
    - Sticky subnavigation bar
    - Reduced header/spacer heights (~40% reduction)
    - Compact controls
    - Mobile-responsive layouts
    """
    st.markdown("""
    <style>
    /* ===========================================
       SIMULATION UNIFIED NAVIGATION
       Purple accent tabs matching Weekly/Season/Career
       =========================================== */

    /* Main navigation buttons container */
    .sim-nav-container {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border, #E5E7EB);
    }

    /* Navigation pill buttons */
    .sim-nav-btn {
        padding: 0.4rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid var(--border, #E5E7EB);
        background: var(--bg-secondary, #F8F9FA);
        color: var(--text-secondary, #6B7280);
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .sim-nav-btn:hover {
        background: var(--accent-subtle, rgba(102, 126, 234, 0.1));
        border-color: var(--accent, #667eea);
    }
    .sim-nav-btn.active {
        background: var(--accent, #667eea);
        color: white;
        border-color: var(--accent, #667eea);
    }

    /* ===========================================
       STICKY SUBNAVIGATION
       =========================================== */
    .sim-sticky-subnav {
        position: sticky;
        top: 0;
        z-index: 100;
        background: var(--bg-primary, #FFFFFF);
        padding: 0.5rem 0;
        margin: 0 -1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        border-bottom: 1px solid var(--border, #E5E7EB);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    @media (prefers-color-scheme: dark) {
        .sim-sticky-subnav {
            background: rgba(14, 17, 23, 0.95);
        }
    }

    /* ===========================================
       COMPACT WEEK SELECTOR
       Reduced height, inline layout
       =========================================== */
    .sim-week-selector {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .sim-week-btn {
        padding: 0.3rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 500;
        border-radius: 4px;
        border: 1px solid var(--border, #E5E7EB);
        background: var(--bg-secondary, #F8F9FA);
        color: var(--text-secondary, #6B7280);
        cursor: pointer;
        transition: all 0.1s ease;
        min-height: 32px;
    }
    .sim-week-btn:hover {
        border-color: var(--accent, #667eea);
        color: var(--accent, #667eea);
    }
    .sim-week-btn.active {
        background: var(--accent, #667eea);
        color: white;
        border-color: var(--accent, #667eea);
    }

    /* Compact selectbox wrapper */
    .sim-compact-select {
        min-width: 100px;
        max-width: 140px;
    }
    .sim-compact-select .stSelectbox label {
        display: none !important;
    }
    .sim-compact-select .stSelectbox > div {
        min-height: 32px !important;
    }

    /* ===========================================
       SUMMARY TILES
       Compact metric tiles at top of sections
       =========================================== */
    .sim-summary-tiles {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .sim-summary-tile {
        background: linear-gradient(135deg, var(--gradient-start, rgba(102, 126, 234, 0.08)) 0%, var(--gradient-end, rgba(118, 75, 162, 0.05)) 100%);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .sim-summary-tile:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }

    .sim-tile-icon {
        font-size: 1.25rem;
        margin-bottom: 0.25rem;
    }
    .sim-tile-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-muted, #9CA3AF);
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.25rem;
    }
    .sim-tile-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary, #1F2937);
    }
    .sim-tile-sublabel {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--accent, #667eea);
        margin-top: 0.125rem;
    }

    /* ===========================================
       SECTION HEADERS - Reduced height
       =========================================== */
    .sim-section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.5rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid var(--accent, #667eea);
    }
    .sim-section-header h3 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary, #1F2937);
        margin: 0 !important;
    }
    .sim-section-icon {
        font-size: 1rem;
    }

    /* ===========================================
       COMPACT METRIC ROW
       Aligned stats in a single row
       =========================================== */
    .sim-metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 0.75rem;
        padding: 0.5rem 0.75rem;
        background: var(--bg-secondary, #F8F9FA);
        border-radius: 8px;
        border: 1px solid var(--border, #E5E7EB);
    }

    .sim-metric-item {
        display: flex;
        align-items: center;
        gap: 0.375rem;
    }
    .sim-metric-icon {
        font-size: 1rem;
    }
    .sim-metric-label {
        font-size: 0.75rem;
        color: var(--text-muted, #9CA3AF);
    }
    .sim-metric-value {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary, #1F2937);
    }

    /* ===========================================
       GROUPED CARDS
       For Critical Moments section
       =========================================== */
    .sim-group-card {
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .sim-group-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary, #1F2937);
    }

    /* ===========================================
       MANAGER FILTER DROPDOWN
       =========================================== */
    .sim-filter-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
        margin-bottom: 0.5rem;
        padding: 0.375rem 0.5rem;
        background: var(--bg-secondary, #F8F9FA);
        border-radius: 6px;
        border: 1px solid var(--border, #E5E7EB);
    }
    .sim-filter-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary, #6B7280);
    }

    /* ===========================================
       COMPACT GAME CARDS (Playoff Machine)
       =========================================== */
    .sim-matchup-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.375rem 0.5rem;
        background: var(--bg-primary, #FFFFFF);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: 6px;
        margin-bottom: 0.375rem;
    }

    .sim-matchup-team {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.1s ease;
    }
    .sim-matchup-team.unpicked {
        background: #f1c40f;
        color: #000;
        border: 1px solid #d4ac0d;
    }
    .sim-matchup-team.winner {
        background: #27ae60;
        color: #fff;
        border: 1px solid #1e8449;
    }
    .sim-matchup-team.loser {
        background: #e74c3c;
        color: #fff;
        border: 1px solid #c0392b;
    }
    .sim-matchup-vs {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--text-muted, #9CA3AF);
        padding: 0 0.25rem;
    }

    /* ===========================================
       SIMULATION SUMMARY PANEL (Collapsible)
       =========================================== */
    .sim-summary-panel {
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .sim-summary-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: pointer;
    }
    .sim-summary-panel-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary, #1F2937);
    }
    .sim-summary-panel-content {
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid var(--border, #E5E7EB);
    }
    .sim-summary-stat {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0;
        font-size: 0.8rem;
    }
    .sim-summary-stat-label {
        color: var(--text-secondary, #6B7280);
    }
    .sim-summary-stat-value {
        font-weight: 600;
        color: var(--text-primary, #1F2937);
    }

    /* ===========================================
       IMPROVED HEATMAP STYLING
       =========================================== */
    .sim-heatmap-container {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border, #E5E7EB);
    }

    /* Row striping for better legibility */
    .sim-heatmap-row:nth-child(even) {
        background: rgba(0, 0, 0, 0.02);
    }
    @media (prefers-color-scheme: dark) {
        .sim-heatmap-row:nth-child(even) {
            background: rgba(255, 255, 255, 0.02);
        }
    }

    /* Rounded cell corners */
    .sim-heatmap-cell {
        border-radius: 3px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.25rem 0.375rem;
        font-size: 0.8rem;
        text-align: center;
    }

    /* ===========================================
       ODDS COMPARISON CARDS
       =========================================== */
    .sim-odds-card {
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .sim-odds-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.375rem;
        border-bottom: 1px solid var(--border, #E5E7EB);
    }
    .sim-odds-icon {
        font-size: 1rem;
    }
    .sim-odds-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary, #1F2937);
    }

    /* ===========================================
       MOBILE RESPONSIVE OVERRIDES
       =========================================== */
    @media (max-width: 768px) {
        /* Stack summary tiles 2x2 on tablet */
        .sim-summary-tiles {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
        }

        .sim-summary-tile {
            padding: 0.5rem;
        }
        .sim-tile-value {
            font-size: 1rem;
        }

        /* Compact metric row stacks vertically */
        .sim-metric-row {
            flex-direction: column;
            gap: 0.5rem;
        }

        /* Matchup cards more compact */
        .sim-matchup-card {
            padding: 0.25rem 0.375rem;
        }
        .sim-matchup-team {
            font-size: 0.75rem;
            padding: 0.2rem 0.375rem;
        }
    }

    @media (max-width: 600px) {
        /* Single column tiles on mobile */
        .sim-summary-tiles {
            grid-template-columns: 1fr 1fr;
        }

        /* Even more compact */
        .sim-tile-icon { font-size: 1rem; }
        .sim-tile-label { font-size: 0.65rem; }
        .sim-tile-value { font-size: 0.9rem; }

        /* Horizontal scroll for wide content */
        .sim-heatmap-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        /* Sticky nav smaller padding */
        .sim-sticky-subnav {
            padding: 0.375rem 0.5rem;
            margin: 0 -0.5rem;
        }
    }

    @media (max-width: 480px) {
        .sim-summary-tile {
            padding: 0.375rem;
        }
        .sim-tile-icon { font-size: 0.9rem; }
        .sim-tile-value { font-size: 0.85rem; }
        .sim-tile-sublabel { font-size: 0.65rem; }

        .sim-section-header h3 {
            font-size: 0.9rem;
        }
    }

    /* ===========================================
       REDUCE STREAMLIT DEFAULT SPACING
       =========================================== */
    /* Tighter spacing for simulation sections */
    .sim-container .stTabs {
        margin-bottom: 0.375rem !important;
    }

    .sim-container [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important;
    }

    .sim-container h1, .sim-container h2, .sim-container h3 {
        margin-top: 0.375rem !important;
        margin-bottom: 0.25rem !important;
    }

    .sim-container hr {
        margin: 0.375rem 0 !important;
    }

    /* Compact buttons */
    .sim-container .stButton > button {
        padding: 0.3rem 0.75rem !important;
        font-size: 0.85rem !important;
        min-height: 36px !important;
    }

    /* Compact captions */
    .sim-container .stCaption {
        margin-bottom: 0.25rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_summary_tiles(tiles: List[Dict[str, Any]]) -> None:
    """
    Render summary tiles at the top of a section.

    Args:
        tiles: List of dicts with keys: icon, label, value, sublabel (optional)

    Example:
        render_summary_tiles([
            {"icon": "🏆", "label": "Championship Favorite", "value": "12.5%", "sublabel": "Jason"},
            {"icon": "🎯", "label": "Highest Playoff Odds", "value": "94.2%", "sublabel": "Mike"},
            {"icon": "📊", "label": "Most Likely #1 Seed", "value": "38.1%", "sublabel": "Daniel"},
        ])
    """
    tiles_html = ""
    for tile in tiles:
        sublabel_html = f'<div class="sim-tile-sublabel">{tile.get("sublabel", "")}</div>' if tile.get("sublabel") else ""
        tiles_html += f"""
        <div class="sim-summary-tile">
            <div class="sim-tile-icon">{tile['icon']}</div>
            <div class="sim-tile-label">{tile['label']}</div>
            <div class="sim-tile-value">{tile['value']}</div>
            {sublabel_html}
        </div>
        """

    st.markdown(f'<div class="sim-summary-tiles">{tiles_html}</div>', unsafe_allow_html=True)


def render_section_header(title: str, icon: str = "") -> None:
    """Render a compact section header with icon."""
    icon_html = f'<span class="sim-section-icon">{icon}</span>' if icon else ""
    st.markdown(f'''
    <div class="sim-section-header">
        {icon_html}
        <h3>{title}</h3>
    </div>
    ''', unsafe_allow_html=True)


def render_metric_row(metrics: List[Dict[str, Any]]) -> None:
    """
    Render a row of compact metrics.

    Args:
        metrics: List of dicts with keys: icon, label, value
    """
    items_html = ""
    for m in metrics:
        items_html += f'''
        <div class="sim-metric-item">
            <span class="sim-metric-icon">{m.get('icon', '')}</span>
            <span class="sim-metric-label">{m['label']}:</span>
            <span class="sim-metric-value">{m['value']}</span>
        </div>
        '''
    st.markdown(f'<div class="sim-metric-row">{items_html}</div>', unsafe_allow_html=True)


def render_group_card(title: str, icon: str = "") -> None:
    """
    Start a grouped card section. Use st.markdown to close with </div>.

    Example:
        render_group_card("Biggest Weekly Gains", "")
        st.dataframe(gains_df)
        st.markdown('</div>', unsafe_allow_html=True)
    """
    icon_html = f'{icon} ' if icon else ""
    st.markdown(f'''
    <div class="sim-group-card">
        <div class="sim-group-header">{icon_html}{title}</div>
    ''', unsafe_allow_html=True)


def render_odds_card(title: str, icon: str, subtitle: str = "") -> None:
    """
    Start an odds comparison card. Use with st.container for content.
    """
    subtitle_html = f'<span style="font-size:0.75rem;color:var(--text-muted);">{subtitle}</span>' if subtitle else ""
    st.markdown(f'''
    <div class="sim-odds-card">
        <div class="sim-odds-header">
            <span class="sim-odds-icon">{icon}</span>
            <span class="sim-odds-title">{title}</span>
            {subtitle_html}
        </div>
    ''', unsafe_allow_html=True)


def close_card() -> None:
    """Close an open card div."""
    st.markdown('</div>', unsafe_allow_html=True)


def render_summary_panel(
    title: str,
    stats: List[Dict[str, str]],
    expanded: bool = True
) -> None:
    """
    Render a collapsible summary panel.

    Args:
        title: Panel title
        stats: List of dicts with 'label' and 'value' keys
        expanded: Whether panel starts expanded
    """
    with st.expander(title, expanded=expanded):
        for stat in stats:
            st.markdown(f'''
            <div class="sim-summary-stat">
                <span class="sim-summary-stat-label">{stat['label']}</span>
                <span class="sim-summary-stat-value">{stat['value']}</span>
            </div>
            ''', unsafe_allow_html=True)


def render_manager_filter(
    managers: List[str],
    key: str = "sim_manager_filter",
    label: str = "Filter by Manager"
) -> Optional[str]:
    """
    Render a manager filter dropdown.

    Returns:
        Selected manager name or None if "All Managers" selected
    """
    options = ["All Managers"] + sorted(managers)
    selected = st.selectbox(
        label,
        options,
        key=key,
        label_visibility="collapsed"
    )
    return None if selected == "All Managers" else selected


def compact_week_selector(
    base_df: pd.DataFrame,
    prefix: str,
    show_go_button: bool = False
) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Compact week selector with Today's Date / Specific Week toggle.

    Returns:
        (year, week, auto_display) - auto_display is True for Today's Date mode
    """
    mode_key = f"{prefix}_week_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = 0

    modes = ["Today's Date", "Specific Week"]

    # Inline buttons
    cols = st.columns([1, 1, 2] if not show_go_button else [1, 1, 1, 1])

    for idx, (col, name) in enumerate(zip(cols[:2], modes)):
        with col:
            is_active = (st.session_state[mode_key] == idx)
            if st.button(
                name,
                key=f"{prefix}_mode_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if not is_active:
                    st.session_state[mode_key] = idx
                    st.rerun()

    mode = modes[st.session_state[mode_key]]

    if mode == "Today's Date":
        year = int(base_df['year'].max())
        week = int(base_df[base_df['year'] == year]['week'].max())
        st.caption(f"Year {year}, Week {week}")
        return year, week, True
    else:
        years = sorted(base_df['year'].astype(int).unique(), reverse=True)

        with cols[2] if len(cols) > 2 else st.columns([1, 1])[0]:
            c1, c2 = st.columns(2)
            with c1:
                year_choice = st.selectbox(
                    "Year",
                    years,
                    key=f"{prefix}_year",
                    label_visibility="collapsed"
                )

            year = int(year_choice)
            weeks = sorted(base_df[base_df['year'] == year]['week'].astype(int).unique())

            with c2:
                week_choice = st.selectbox(
                    "Week",
                    weeks,
                    index=len(weeks) - 1 if weeks else 0,
                    key=f"{prefix}_week",
                    label_visibility="collapsed"
                )

            week = int(week_choice)

        return year, week, False


def start_simulation_container() -> None:
    """Start the simulation container with reduced spacing."""
    st.markdown('<div class="sim-container">', unsafe_allow_html=True)


def end_simulation_container() -> None:
    """End the simulation container."""
    st.markdown('</div>', unsafe_allow_html=True)
