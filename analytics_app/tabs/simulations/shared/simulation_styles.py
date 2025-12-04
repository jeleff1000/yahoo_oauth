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
    st.markdown(
        """
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
       UNIFIED HEADER BAR (NEW)
       =========================================== */
    .sim-unified-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 1rem;
        background: var(--bg-secondary, #F8F9FA);
        border-bottom: 1px solid var(--border, #E5E7EB);
        border-radius: 8px 8px 0 0;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .sim-header-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary, #1F2937);
        white-space: nowrap;
    }

    .sim-header-controls {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .sim-header-meta {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: var(--text-secondary, #6B7280);
    }

    .sim-header-meta-item {
        padding: 0.25rem 0.5rem;
        background: var(--bg-tertiary, #F3F4F6);
        border-radius: 4px;
        border: 1px solid var(--border, #E5E7EB);
    }

    /* ===========================================
       SEGMENTED CONTROL (NEW)
       =========================================== */
    .sim-segmented-control {
        display: inline-flex;
        border: 1px solid var(--border, #E5E7EB);
        border-radius: 6px;
        overflow: hidden;
        background: var(--bg-primary, #FFFFFF);
    }

    .sim-segment {
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        border: none;
        background: transparent;
        color: var(--text-secondary, #6B7280);
        cursor: pointer;
        transition: all 0.15s ease;
        border-right: 1px solid var(--border, #E5E7EB);
    }

    .sim-segment:last-child {
        border-right: none;
    }

    .sim-segment:hover:not(.active) {
        background: var(--accent-subtle, rgba(102, 126, 234, 0.1));
    }

    .sim-segment.active {
        background: var(--accent, #667eea);
        color: white;
    }

    /* ===========================================
       COMPACT UNIFIED HEADER
       Single-line header with reduced padding
       =========================================== */
    .sim-compact-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--border, #E5E7EB);
    }

    .sim-compact-header .stButton > button {
        padding: 0.25rem 0.75rem !important;
        font-size: 0.8rem !important;
        min-height: unset !important;
    }

    .sim-compact-header .stSelectbox > div {
        min-height: unset !important;
    }

    .sim-compact-header .stSelectbox [data-baseweb="select"] {
        min-height: 32px !important;
    }

    /* Reduce overall tab and button padding */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.4rem 0.75rem !important;
        font-size: 0.85rem !important;
    }

    /* ===========================================
       CONTEXT CARD (NEW)
       =========================================== */
    .sim-context-card {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        flex-wrap: wrap;
        gap: 0.75rem;
        padding: 0.5rem 0.75rem;
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: 6px;
        margin-bottom: 0.75rem;
        font-size: 0.8rem;
    }

    .sim-context-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .sim-context-label {
        color: var(--text-muted, #9CA3AF);
    }

    .sim-context-value {
        font-weight: 600;
        color: var(--text-primary, #1F2937);
    }

    .sim-context-help {
        font-size: 0.75rem;
        color: var(--text-muted, #9CA3AF);
        font-style: italic;
    }

    /* ===========================================
       KPI HERO CARD (NEW)
       =========================================== */
    .sim-kpi-hero {
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .sim-kpi-hero-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border, #E5E7EB);
    }

    .sim-kpi-hero-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary, #1F2937);
    }

    .sim-kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
    }

    .sim-kpi-item {
        text-align: center;
        padding: 0.6rem 0.4rem;
        background: linear-gradient(135deg, var(--gradient-start, rgba(102, 126, 234, 0.08)) 0%, var(--gradient-end, rgba(118, 75, 162, 0.05)) 100%);
        border-radius: 6px;
        border: 1px solid var(--border, #E5E7EB);
    }

    .sim-kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary, #1F2937);
        line-height: 1.1;
        margin-bottom: 0.1rem;
    }

    .sim-kpi-label {
        font-size: 0.6rem;
        color: var(--text-muted, #9CA3AF);
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .sim-kpi-owner {
        display: inline-block;
        margin-top: 0.2rem;
        padding: 0.1rem 0.4rem;
        background: var(--accent, #667eea);
        color: white;
        border-radius: 10px;
        font-size: 0.65rem;
        font-weight: 500;
    }

    .sim-kpi-delta {
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    .sim-kpi-delta.positive {
        color: var(--success, #10B981);
    }

    .sim-kpi-delta.negative {
        color: var(--error, #EF4444);
    }

    /* ===========================================
       DELTA PILLS (NEW)
       =========================================== */
    .sim-delta-pill {
        display: inline-block;
        padding: 0.125rem 0.375rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .sim-delta-pill.positive {
        background: var(--success-bg, rgba(16, 185, 129, 0.1));
        color: var(--success, #10B981);
    }

    .sim-delta-pill.negative {
        background: var(--error-bg, rgba(239, 68, 68, 0.1));
        color: var(--error, #EF4444);
    }

    /* ===========================================
       EQUAL HEIGHT ROW
       =========================================== */
    .sim-equal-height-row {
        display: flex;
        align-items: stretch;
    }

    .sim-equal-height-row > div {
        display: flex;
        flex-direction: column;
    }

    .sim-equal-height-row [data-testid="stVerticalBlock"] > div:has(> [data-testid="stContainer"]) {
        flex: 1;
    }

    /* Text link button style */
    .sim-text-link {
        background: none !important;
        border: none !important;
        color: var(--accent, #667eea) !important;
        font-size: 0.75rem !important;
        padding: 0 !important;
        text-decoration: none;
        cursor: pointer;
    }

    .sim-text-link:hover {
        text-decoration: underline;
    }

    /* ===========================================
       CENTERED CONSTRAINED CONTENT
       =========================================== */
    .sim-centered-content {
        max-width: 1100px;
        margin: 0 auto;
    }

    /* ===========================================
       SUMMARY STRIP (NEW)
       =========================================== */
    .sim-summary-strip {
        display: flex;
        align-items: center;
        padding: 0.5rem 0.75rem;
        background: linear-gradient(135deg, var(--gradient-start, rgba(102, 126, 234, 0.08)) 0%, var(--gradient-end, rgba(118, 75, 162, 0.05)) 100%);
        border-radius: 6px;
        border: 1px solid var(--border, #E5E7EB);
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-primary, #1F2937);
    }

    /* ===========================================
       ENHANCED MATCHUP PICKER (NEW)
       =========================================== */
    .sim-matchup-picker {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0.75rem;
        background: var(--bg-primary, #FFFFFF);
        border: 2px solid var(--border, #E5E7EB);
        border-radius: 6px;
        margin-bottom: 0.5rem;
        transition: border-color 0.15s ease;
    }

    .sim-matchup-picker:hover {
        border-color: var(--accent, #667eea);
    }

    .sim-matchup-picker-team {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.15s ease;
        min-width: 100px;
        text-align: center;
        position: relative;
    }

    .sim-matchup-picker-team:hover {
        background: var(--accent-subtle, rgba(102, 126, 234, 0.1));
    }

    .sim-matchup-picker-team.selected {
        background: var(--success, #10B981);
        color: white;
    }

    .sim-matchup-picker-team.loser {
        background: var(--error-bg, rgba(239, 68, 68, 0.1));
        opacity: 0.7;
    }

    .sim-team-name {
        font-weight: 600;
        font-size: 0.85rem;
    }

    .sim-team-record {
        font-size: 0.7rem;
        color: var(--text-muted, #9CA3AF);
    }

    .sim-team-projected {
        font-size: 0.7rem;
        color: var(--accent, #667eea);
        font-weight: 500;
    }

    .sim-matchup-picker-team.selected .sim-team-record,
    .sim-matchup-picker-team.selected .sim-team-projected {
        color: rgba(255, 255, 255, 0.85);
    }

    .sim-matchup-picker-vs {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--text-muted, #9CA3AF);
        padding: 0 0.5rem;
    }

    /* Selection check indicator */
    .sim-matchup-picker-team.selected::after {
        content: "";
        position: absolute;
        top: -6px;
        right: -6px;
        width: 18px;
        height: 18px;
        background: var(--success, #10B981);
        border: 2px solid white;
        border-radius: 50%;
        font-size: 0.6rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Ghost button style */
    .sim-btn-ghost {
        background: transparent !important;
        border: 1px dashed var(--border, #E5E7EB) !important;
        color: var(--text-secondary, #6B7280) !important;
    }

    .sim-btn-ghost:hover {
        border-color: var(--accent, #667eea) !important;
        color: var(--accent, #667eea) !important;
    }

    /* ===========================================
       ADDITIONAL MOBILE RESPONSIVE (NEW)
       =========================================== */
    @media (max-width: 768px) {
        .sim-unified-header {
            flex-direction: column;
            align-items: stretch;
            padding: 0.375rem 0.5rem;
        }

        .sim-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
        }

        .sim-segmented-control {
            width: 100%;
        }

        .sim-segment {
            flex: 1;
            text-align: center;
            padding: 0.3rem 0.5rem;
            font-size: 0.8rem;
        }

        .sim-kpi-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        /* Horizontal scrolling tabs */
        .stTabs [data-baseweb="tab-list"] {
            display: flex !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            padding-bottom: 0.25rem;
        }

        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }

        .stTabs [data-baseweb="tab"] {
            flex-shrink: 0 !important;
            white-space: nowrap !important;
        }
    }

    @media (max-width: 600px) {
        .sim-header-meta {
            flex-wrap: wrap;
        }

        .sim-context-card {
            flex-direction: column;
            align-items: flex-start;
        }

        .sim-matchup-picker {
            flex-direction: column;
            gap: 0.25rem;
        }

        .sim-matchup-picker-vs {
            padding: 0.25rem 0;
        }
    }

    @media (max-width: 480px) {
        .sim-kpi-grid {
            grid-template-columns: 1fr;
        }

        .sim-kpi-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            text-align: left;
            padding: 0.5rem 0.75rem;
        }

        .sim-kpi-value {
            font-size: 1.25rem;
        }

        /* Mobile filters in expander */
        .sim-mobile-filters {
            display: block;
        }

        .sim-desktop-filters {
            display: none;
        }
    }

    @media (min-width: 481px) {
        .sim-mobile-filters {
            display: none;
        }

        .sim-desktop-filters {
            display: block;
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
    """,
        unsafe_allow_html=True,
    )


def render_summary_tiles(tiles: List[Dict[str, Any]]) -> None:
    """
    Render summary tiles at the top of a section using Streamlit columns.

    Args:
        tiles: List of dicts with keys: icon, label, value, sublabel (optional)
    """
    cols = st.columns(len(tiles))
    for col, tile in zip(cols, tiles):
        with col:
            st.metric(
                label=tile["label"],
                value=tile["value"],
                delta=tile.get("sublabel", None),
            )


def render_section_header(title: str, icon: str = "") -> None:
    """Render a compact section header with icon using native Streamlit."""
    display_title = f"{icon} {title}" if icon else title
    st.subheader(display_title)


def render_metric_row(metrics: List[Dict[str, Any]]) -> None:
    """
    Render a row of compact metrics using Streamlit columns.

    Args:
        metrics: List of dicts with keys: icon, label, value
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            icon = m.get("icon", "")
            st.caption(f"{icon} **{m['label']}:** {m['value']}")


def render_group_card(title: str, icon: str = "") -> None:
    """
    Render a group card header. Uses native Streamlit markdown.

    Example:
        render_group_card("Biggest Weekly Gains", "")
        st.dataframe(gains_df)
    """
    display_title = f"{icon} {title}" if icon else title
    st.markdown(f"**{display_title}**")


def render_odds_card(title: str, icon: str, subtitle: str = "") -> None:
    """
    Render an odds card header. Uses native Streamlit markdown.
    """
    display_title = f"{icon} {title}" if icon else title
    if subtitle:
        st.markdown(f"**{display_title}** *{subtitle}*")
    else:
        st.markdown(f"**{display_title}**")


def close_card() -> None:
    """No-op - kept for backwards compatibility."""
    pass


def render_summary_panel(
    title: str, stats: List[Dict[str, str]], expanded: bool = True
) -> None:
    """
    Render a collapsible summary panel using native Streamlit expander.

    Args:
        title: Panel title
        stats: List of dicts with 'label' and 'value' keys
        expanded: Whether panel starts expanded
    """
    with st.expander(title, expanded=expanded):
        cols = st.columns(len(stats))
        for col, stat in zip(cols, stats):
            with col:
                st.caption(stat["label"])
                st.markdown(f"**{stat['value']}**")


def render_manager_filter(
    managers: List[str],
    key: str = "sim_manager_filter",
    label: str = "Filter by Manager",
) -> Optional[str]:
    """
    Render a manager filter dropdown.

    Returns:
        Selected manager name or None if "All Managers" selected
    """
    options = ["All Managers"] + sorted(managers)
    selected = st.selectbox(label, options, key=key, label_visibility="collapsed")
    return None if selected == "All Managers" else selected


def compact_week_selector(
    base_df: pd.DataFrame, prefix: str, show_go_button: bool = False
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
            is_active = st.session_state[mode_key] == idx
            if st.button(
                name,
                key=f"{prefix}_mode_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    st.session_state[mode_key] = idx
                    st.rerun()

    mode = modes[st.session_state[mode_key]]

    if mode == "Today's Date":
        year = int(base_df["year"].max())
        week = int(base_df[base_df["year"] == year]["week"].max())
        st.caption(f"Year {year}, Week {week}")
        return year, week, True
    else:
        years = sorted(base_df["year"].astype(int).unique(), reverse=True)

        with cols[2] if len(cols) > 2 else st.columns([1, 1])[0]:
            c1, c2 = st.columns(2)
            with c1:
                year_choice = st.selectbox(
                    "Year", years, key=f"{prefix}_year", label_visibility="collapsed"
                )

            year = int(year_choice)
            weeks = sorted(
                base_df[base_df["year"] == year]["week"].astype(int).unique()
            )

            with c2:
                week_choice = st.selectbox(
                    "Week",
                    weeks,
                    index=len(weeks) - 1 if weeks else 0,
                    key=f"{prefix}_week",
                    label_visibility="collapsed",
                )

            week = int(week_choice)

        return year, week, False


def start_simulation_container() -> None:
    """Start the simulation container with reduced spacing."""
    st.markdown('<div class="sim-container">', unsafe_allow_html=True)


def end_simulation_container() -> None:
    """End the simulation container."""
    st.markdown("</div>", unsafe_allow_html=True)
