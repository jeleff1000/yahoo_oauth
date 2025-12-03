"""
Reusable UI components for consistent design across the app
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any


def render_metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    icon: str = "📊",
    color: str = "#667eea",
    help_text: Optional[str] = None,
):
    """
    Render a modern metric card with gradient background
    """
    delta_html = (
        f"""
        <div style='font-size: 0.85rem; color: #059669; margin-top: 0.3rem;'>
            {delta}
        </div>
    """
        if delta
        else ""
    )

    help_html = (
        f"""
        <div style='font-size: 0.75rem; color: #6B7280; margin-top: 0.5rem;'>
            {help_text}
        </div>
    """
        if help_text
        else ""
    )

    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                    border-left: 4px solid {color};
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                <span style='font-size: 1.2rem;'>{icon}</span>
                <span style='font-size: 0.85rem; color: #6B7280; font-weight: 500;'>{title}</span>
            </div>
            <div style='font-size: 1.8rem; font-weight: bold; color: #1a1a1a;'>
                {value}
            </div>
            {delta_html}
            {help_html}
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_header(
    title: str, subtitle: Optional[str] = None, gradient: tuple = ("#667eea", "#764ba2")
):
    """
    Render a gradient header section
    """
    subtitle_html = (
        f"""
        <p style='margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.95); font-size: 1rem;'>
            {subtitle}
        </p>
    """
        if subtitle
        else ""
    )

    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, {gradient[0]} 0%, {gradient[1]} 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);'>
            <h2 style='margin: 0; color: white; font-size: 1.8rem;'>
                {title}
            </h2>
            {subtitle_html}
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_info_card(content: str, card_type: str = "info", icon: Optional[str] = None):
    """
    Render an info/warning/error card
    """
    colors = {
        "info": {"bg": "#EFF6FF", "border": "#3B82F6", "icon": "ℹ️"},
        "warning": {"bg": "#FFF7ED", "border": "#F97316", "icon": "⚠️"},
        "error": {"bg": "#FEF2F2", "border": "#EF4444", "icon": "❌"},
        "success": {"bg": "#F0FDF4", "border": "#10B981", "icon": "✅"},
    }

    config = colors.get(card_type, colors["info"])
    icon = icon or config["icon"]

    st.markdown(
        f"""
        <div style='background: {config["bg"]};
                    border-left: 4px solid {config["border"]};
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 1rem 0;'>
            <div style='display: flex; gap: 0.8rem; align-items: start;'>
                <span style='font-size: 1.2rem;'>{icon}</span>
                <div style='flex: 1; color: #374151;'>
                    {content}
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_stat_badge(label: str, value: str, color: str = "#667eea"):
    """
    Render a small stat badge
    """
    st.markdown(
        f"""
        <div style='display: inline-block;
                    background: {color};
                    color: white;
                    padding: 0.4rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    margin: 0.2rem;'>
            {label}: {value}
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_progress_bar(value: float, label: str = "", max_value: float = 100.0):
    """
    Render a custom progress bar
    """
    percentage = min(100, (value / max_value) * 100)

    # Color based on percentage
    if percentage >= 75:
        color = "#10B981"
    elif percentage >= 50:
        color = "#F59E0B"
    else:
        color = "#EF4444"

    st.markdown(
        f"""
        <div style='margin: 1rem 0;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.3rem;'>
                <span style='font-size: 0.85rem; color: #6B7280;'>{label}</span>
                <span style='font-size: 0.85rem; font-weight: 600;'>{percentage:.1f}%</span>
            </div>
            <div style='background: #E5E7EB; border-radius: 10px; height: 8px; overflow: hidden;'>
                <div style='background: {color}; height: 100%; width: {percentage}%;
                            transition: width 0.3s ease;'>
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_comparison_card(team1: Dict[str, Any], team2: Dict[str, Any], winner: str):
    """
    Render a matchup comparison card
    """
    team1_border = (
        "3px solid #10B981" if winner == team1["name"] else "1px solid #E5E7EB"
    )
    team2_border = (
        "3px solid #10B981" if winner == team2["name"] else "1px solid #E5E7EB"
    )

    st.markdown(
        f"""
        <div style='display: grid; grid-template-columns: 1fr auto 1fr; gap: 1rem;
                    background: white; border-radius: 12px; padding: 1.5rem;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>

            <!-- Team 1 -->
            <div style='border: {team1_border}; border-radius: 8px; padding: 1rem; text-align: center;'>
                <div style='font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;'>
                    {team1["name"]}
                </div>
                <div style='font-size: 2rem; font-weight: bold; color: #1a1a1a;'>
                    {team1["score"]}
                </div>
                <div style='font-size: 0.85rem; color: #6B7280; margin-top: 0.3rem;'>
                    {team1.get("record", "")}
                </div>
            </div>

            <!-- VS -->
            <div style='display: flex; align-items: center; justify-content: center;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: #6B7280;'>VS</div>
            </div>

            <!-- Team 2 -->
            <div style='border: {team2_border}; border-radius: 8px; padding: 1rem; text-align: center;'>
                <div style='font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;'>
                    {team2["name"]}
                </div>
                <div style='font-size: 2rem; font-weight: bold; color: #1a1a1a;'>
                    {team2["score"]}
                </div>
                <div style='font-size: 0.85rem; color: #6B7280; margin-top: 0.3rem;'>
                    {team2.get("record", "")}
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_data_table(
    df: pd.DataFrame,
    title: Optional[str] = None,
    highlight_column: Optional[str] = None,
    highlight_condition: str = "max",
):
    """
    Render a styled data table with optional highlighting
    """
    if title:
        st.markdown(f"#### {title}")

    # Apply styling if highlight requested
    if highlight_column and highlight_column in df.columns:
        if highlight_condition == "max":
            mask = df[highlight_column] == df[highlight_column].max()
        elif highlight_condition == "min":
            mask = df[highlight_column] == df[highlight_column].min()
        else:
            mask = df[highlight_column] > 0

        styled_df = df.style.apply(
            lambda x: [
                "background-color: #D1FAE5" if mask[i] else "" for i in range(len(x))
            ],
            axis=0,
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_empty_state(
    message: str = "No data available",
    icon: str = "📭",
    suggestion: Optional[str] = None,
):
    """
    Render an empty state placeholder
    """
    suggestion_html = (
        f"""
        <div style='margin-top: 1rem; font-size: 0.9rem; color: #6B7280;'>
            💡 {suggestion}
        </div>
    """
        if suggestion
        else ""
    )

    st.markdown(
        f"""
        <div style='text-align: center; padding: 3rem 1rem;
                    background: #F9FAFB; border-radius: 12px;
                    border: 2px dashed #E5E7EB;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>
                {icon}
            </div>
            <div style='font-size: 1.2rem; color: #6B7280; font-weight: 500;'>
                {message}
            </div>
            {suggestion_html}
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_loading_skeleton(num_rows: int = 5):
    """
    Render a loading skeleton for better UX
    """
    for _ in range(num_rows):
        st.markdown(
            """
            <div style='background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                        background-size: 200% 100%;
                        animation: loading 1.5s infinite;
                        height: 60px;
                        border-radius: 8px;
                        margin-bottom: 0.5rem;'>
            </div>
            <style>
                @keyframes loading {
                    0% { background-position: 200% 0; }
                    100% { background-position: -200% 0; }
                }
            </style>
        """,
            unsafe_allow_html=True,
        )
