"""
Unified header components for simulations tab.

Provides consolidated navigation, context display, and KPI rendering.
"""
from __future__ import annotations

import streamlit as st
from datetime import datetime
from typing import List, Dict, Optional, Callable


def render_segmented_control(
    options: List[str],
    selected_index: int,
    key_prefix: str,
    on_change: Optional[Callable[[int], None]] = None,
) -> int:
    """
    Render a modern segmented control (toggle between options).

    Args:
        options: List of option labels
        selected_index: Currently selected index
        key_prefix: Unique key prefix for session state
        on_change: Optional callback when selection changes

    Returns:
        Currently selected index
    """
    # Build the segmented control HTML
    segments_html = ""
    for idx, option in enumerate(options):
        active_class = "active" if idx == selected_index else ""
        segments_html += f'<span class="sim-segment {active_class}" data-idx="{idx}">{option}</span>'

    st.markdown(
        f'<div class="sim-segmented-control">{segments_html}</div>',
        unsafe_allow_html=True,
    )

    # Use columns with buttons for actual interaction (CSS makes them look unified)
    cols = st.columns(len(options))
    for idx, (col, option) in enumerate(zip(cols, options)):
        with col:
            is_active = idx == selected_index
            if st.button(
                option,
                key=f"{key_prefix}_seg_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    if on_change:
                        on_change(idx)
                    return idx

    return selected_index


def render_context_card(
    season: int,
    week: int,
    num_simulations: int = 10000,
    help_text: str = "",
    show_advanced: bool = False,
) -> None:
    """
    Render a small bordered context card pinned under tabs.

    Args:
        season: Selected season year
        week: Selected week number
        num_simulations: Number of simulations run
        help_text: Optional help text to display
        show_advanced: Whether to show advanced options expander
    """
    context_items = f"""
    <div class="sim-context-card">
        <div class="sim-context-item">
            <span class="sim-context-label">Season:</span>
            <span class="sim-context-value">{season}</span>
        </div>
        <div class="sim-context-item">
            <span class="sim-context-label">Week:</span>
            <span class="sim-context-value">{week}</span>
        </div>
        <div class="sim-context-item">
            <span class="sim-context-label">Simulations:</span>
            <span class="sim-context-value">{num_simulations:,}</span>
        </div>
        {f'<div class="sim-context-help">{help_text}</div>' if help_text else ''}
    </div>
    """
    st.markdown(context_items, unsafe_allow_html=True)

    if show_advanced:
        with st.expander("Advanced Options", expanded=False):
            st.caption("Advanced simulation settings will appear here.")


def render_kpi_hero_card(
    title: str,
    kpis: List[Dict],
    manager_dropdown_options: Optional[List[str]] = None,
    manager_dropdown_key: Optional[str] = None,
    selected_manager: Optional[str] = None,
) -> Optional[str]:
    """
    Render a KPI hero card with title, optional manager dropdown, and KPI grid.

    Args:
        title: Card title (e.g., "Championship Path - 2024 Week 12")
        kpis: List of KPI dicts with keys: label, value, owner (optional), delta (optional)
        manager_dropdown_options: Optional list of managers for dropdown filter
        manager_dropdown_key: Session state key for dropdown
        selected_manager: Currently selected manager (if any)

    Returns:
        Selected manager name if dropdown is shown, else None

    Example KPI dict:
        {"label": "Highest Playoff Odds", "value": "100%", "owner": "Yaacov", "delta": "+5.2%"}
    """
    selected = None

    with st.container(border=True):
        # Header row with title and optional dropdown
        header_cols = st.columns([3, 1]) if manager_dropdown_options else [st.container()]

        if manager_dropdown_options:
            with header_cols[0]:
                st.markdown(f"**{title}**")
            with header_cols[1]:
                all_option = ["All Managers"] + list(manager_dropdown_options)
                default_idx = 0
                if selected_manager and selected_manager in all_option:
                    default_idx = all_option.index(selected_manager)
                selected = st.selectbox(
                    "Manager",
                    all_option,
                    index=default_idx,
                    key=manager_dropdown_key,
                    label_visibility="collapsed",
                )
                if selected == "All Managers":
                    selected = None
        else:
            with header_cols[0]:
                st.markdown(f"**{title}**")

        # KPI Grid
        kpi_html = '<div class="sim-kpi-grid">'
        for kpi in kpis:
            owner_html = ""
            if kpi.get("owner"):
                owner_html = f'<div class="sim-kpi-owner">{kpi["owner"]}</div>'

            delta_html = ""
            if kpi.get("delta"):
                delta_val = kpi["delta"]
                delta_class = "positive" if not str(delta_val).startswith("-") else "negative"
                delta_html = f'<div class="sim-kpi-delta {delta_class}">{delta_val}</div>'

            kpi_html += f"""
            <div class="sim-kpi-item">
                <div class="sim-kpi-value">{kpi["value"]}</div>
                <div class="sim-kpi-label">{kpi["label"]}</div>
                {owner_html}
                {delta_html}
            </div>
            """
        kpi_html += "</div>"

        st.markdown(kpi_html, unsafe_allow_html=True)

    return selected


def render_summary_strip(message: str, icon: str = "") -> None:
    """
    Render a summary strip with gradient background.

    Args:
        message: The summary message to display
        icon: Optional emoji icon
    """
    icon_html = f'<span style="margin-right: 0.5rem;">{icon}</span>' if icon else ""
    st.markdown(
        f'<div class="sim-summary-strip">{icon_html}{message}</div>',
        unsafe_allow_html=True,
    )


def render_delta_pill(value: str, is_positive: bool = True) -> str:
    """
    Return HTML for a colored delta pill.

    Args:
        value: The delta value to display (e.g., "+36.9%")
        is_positive: Whether this is a positive (green) or negative (red) delta

    Returns:
        HTML string for the delta pill
    """
    pill_class = "positive" if is_positive else "negative"
    return f'<span class="sim-delta-pill {pill_class}">{value}</span>'


def render_compact_filter_card(
    presets: List[str],
    selected_preset: str,
    seasons: List[int],
    selected_seasons: List[int],
    week_range: tuple,
    managers: List[str],
    selected_managers: List[str],
    key_prefix: str,
) -> Dict:
    """
    Render a compact 3-row filter card.

    Args:
        presets: List of preset options
        selected_preset: Currently selected preset
        seasons: Available seasons
        selected_seasons: Currently selected seasons
        week_range: Tuple of (start_week, end_week)
        managers: Available managers
        selected_managers: Currently selected managers
        key_prefix: Unique key prefix

    Returns:
        Dict with filter values: preset, seasons, week_start, week_end, managers
    """
    result = {}

    with st.container(border=True):
        # Row 1: Quick presets
        preset_cols = st.columns(len(presets))
        for idx, (col, preset) in enumerate(zip(preset_cols, presets)):
            with col:
                is_active = preset == selected_preset
                if st.button(
                    preset,
                    key=f"{key_prefix}_preset_{idx}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    result["preset"] = preset

        if "preset" not in result:
            result["preset"] = selected_preset

        # Row 2: Date range
        range_cols = st.columns([2, 1, 1])
        with range_cols[0]:
            result["seasons"] = st.multiselect(
                "Seasons",
                seasons,
                default=selected_seasons,
                key=f"{key_prefix}_seasons",
            )
        with range_cols[1]:
            weeks = list(range(1, 18))
            result["week_start"] = st.selectbox(
                "Start Week",
                weeks,
                index=week_range[0] - 1 if week_range[0] else 0,
                key=f"{key_prefix}_week_start",
            )
        with range_cols[2]:
            result["week_end"] = st.selectbox(
                "End Week",
                weeks,
                index=week_range[1] - 1 if week_range[1] else len(weeks) - 1,
                key=f"{key_prefix}_week_end",
            )

        # Row 3: Managers
        result["managers"] = st.multiselect(
            "Managers",
            managers,
            default=selected_managers,
            key=f"{key_prefix}_managers",
        )

    return result


def render_mini_kpis(kpis: List[Dict]) -> None:
    """
    Render a compact row of mini KPIs for side panels.

    Args:
        kpis: List of KPI dicts with keys: label, value, color (optional)
    """
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            color = kpi.get("color", "var(--text-primary)")
            st.markdown(
                f"""
                <div style="text-align: center; padding: 0.25rem;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: {color};">{kpi["value"]}</div>
                    <div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase;">{kpi["label"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def get_today_display() -> str:
    """Get formatted today's date for display."""
    return datetime.now().strftime("%b %d")
