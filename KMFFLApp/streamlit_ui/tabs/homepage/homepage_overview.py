#!/usr/bin/env python3
"""
KMFFL Homepage Overview - Clean Landing Page

A clean, decluttered landing page that:
- Shows dynamic current season info
- Provides quick navigation to key sections
- Uses the unified theme system
- Works well in dark mode and mobile
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st
import sys
from pathlib import Path

# Ensure streamlit_ui directory is in path for imports
_streamlit_ui_dir = Path(__file__).parent.parent.parent.resolve()
if str(_streamlit_ui_dir) not in sys.path:
    sys.path.insert(0, str(_streamlit_ui_dir))

# Theme and styles
from shared.themes import inject_theme_css
from ..shared.modern_styles import apply_modern_styles

# Data helpers
from md.data_access import load_player_two_week_slice
from shared.dataframe_utils import as_dataframe, get_matchup_df

# Homepage sections
from .season_standings import display_season_standings
from .head_to_head import display_head_to_head
from .schedules import display_schedules
from .recaps import display_recap_overview

# Hall of Fame
try:
    from .hall_of_fame.hall_of_fame_homepage import HallOfFameViewer
    HALL_OF_FAME_AVAILABLE = True
    HALL_OF_FAME_ERROR = None
except Exception as hof_import_error:
    HALL_OF_FAME_AVAILABLE = False
    HALL_OF_FAME_ERROR = str(hof_import_error)
    HallOfFameViewer = None


def _apply_homepage_styles():
    """Apply minimal homepage-specific styles."""
    st.markdown("""
    <style>
    /* ========================================
       HOMEPAGE HERO - Subtle gradient
       ======================================== */
    .homepage-hero {
        background: linear-gradient(135deg,
            var(--gradient-start, rgba(102, 126, 234, 0.1)) 0%,
            var(--gradient-end, rgba(118, 75, 162, 0.06)) 100%);
        padding: var(--space-xl, 2rem);
        border-radius: var(--radius-lg, 12px);
        margin-bottom: var(--space-lg, 1.5rem);
        border: 1px solid var(--border, #E5E7EB);
    }
    .homepage-hero h1 {
        color: var(--text-primary, #1F2937) !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .homepage-hero .subtitle {
        color: var(--text-secondary, #6B7280);
        font-size: 1rem;
        margin: 0;
    }
    .homepage-hero .season-badge {
        display: inline-block;
        background: var(--accent, #667eea);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: var(--radius-full, 20px);
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: var(--space-md, 1rem);
    }

    /* ========================================
       QUICK STATS ROW - Static, no hover
       ======================================== */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: var(--space-md, 1rem);
        margin-bottom: var(--space-lg, 1.5rem);
    }
    .stat-card {
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        text-align: center;
        /* NO shadow, NO hover - static display */
    }
    .stat-card .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--accent, #667eea);
        margin-bottom: 0.25rem;
    }
    .stat-card .stat-label {
        font-size: 0.8rem;
        color: var(--text-muted, #9CA3AF);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ========================================
       NAVIGATION TILES - Static info cards
       ======================================== */
    .nav-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: var(--space-md, 1rem);
        margin-bottom: var(--space-lg, 1.5rem);
    }
    .nav-tile {
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        /* Static - no shadow, no hover */
    }
    .nav-tile-icon {
        font-size: 2rem;
        margin-bottom: var(--space-sm, 0.5rem);
        display: block;
    }
    .nav-tile-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary, #1F2937);
        margin-bottom: var(--space-xs, 0.25rem);
    }
    .nav-tile-desc {
        font-size: 0.875rem;
        color: var(--text-secondary, #6B7280);
        line-height: 1.5;
    }
    .nav-tile-tag {
        display: inline-block;
        background: var(--accent-subtle, rgba(102, 126, 234, 0.1));
        color: var(--accent, #667eea);
        padding: 0.2rem 0.6rem;
        border-radius: var(--radius-sm, 4px);
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: var(--space-sm, 0.5rem);
    }

    /* ========================================
       SECTION HEADERS
       ======================================== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: var(--space-lg, 1.5rem) 0 var(--space-md, 1rem) 0;
        padding-bottom: var(--space-sm, 0.5rem);
        border-bottom: 2px solid var(--accent, #667eea);
    }
    .section-header h3 {
        color: var(--text-primary, #1F2937) !important;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
    }

    /* ========================================
       INFO CARDS - Static display
       ======================================== */
    .info-card {
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin-bottom: var(--space-md, 1rem);
        /* NO shadow, NO hover - static display */
    }
    .info-card h4 {
        color: var(--accent, #667eea);
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    .info-card p, .info-card li {
        color: var(--text-secondary, #6B7280);
        line-height: 1.6;
        font-size: 0.9rem;
    }

    /* ========================================
       MOBILE RESPONSIVE
       ======================================== */
    @media (max-width: 768px) {
        .homepage-hero {
            padding: var(--space-md, 1rem);
        }
        .homepage-hero h1 {
            font-size: 1.5rem;
        }
        .homepage-hero .subtitle {
            font-size: 0.9rem;
        }
        .stats-row {
            grid-template-columns: repeat(2, 1fr);
            gap: var(--space-sm, 0.5rem);
        }
        .stat-card {
            padding: var(--space-sm, 0.5rem);
        }
        .stat-card .stat-value {
            font-size: 1.25rem;
        }
        .stat-card .stat-label {
            font-size: 0.7rem;
        }
        .nav-grid {
            grid-template-columns: 1fr;
        }
        .nav-tile {
            padding: var(--space-sm, 0.5rem);
        }
        .nav-tile-icon {
            font-size: 1.5rem;
        }
        .nav-tile-title {
            font-size: 1rem;
        }
    }

    @media (max-width: 480px) {
        .homepage-hero {
            padding: var(--space-sm, 0.5rem);
        }
        .homepage-hero h1 {
            font-size: 1.3rem;
        }
        .stat-card .stat-value {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def _render_hero(summary: Dict[str, Any]) -> None:
    """Render the dynamic hero section."""
    latest_year = summary.get("latest_year", 2024)
    latest_week = summary.get("latest_week", 1)

    st.markdown(f"""
    <div class="homepage-hero">
        <h1>Fantasy Football Command Center</h1>
        <p class="subtitle">25+ years of data, analytics, and insights at your fingertips</p>
        <span class="season-badge">Season {latest_year} - Week {latest_week}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_quick_stats(summary: Dict[str, Any]) -> None:
    """Render the quick stats row."""
    matchups = summary.get("matchup_count", 0)
    players = summary.get("player_count", 0)
    drafts = summary.get("draft_count", 0)
    transactions = summary.get("transactions_count", 0)

    def fmt(n):
        if n >= 1000:
            return f"{n/1000:.1f}K"
        return str(n)

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">{fmt(matchups)}</div>
            <div class="stat-label">Matchups</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{fmt(players)}</div>
            <div class="stat-label">Player Records</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{fmt(drafts)}</div>
            <div class="stat-label">Draft Picks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{fmt(transactions)}</div>
            <div class="stat-label">Transactions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_navigation_tiles() -> None:
    """Render the main navigation tiles."""
    st.markdown('<div class="section-header"><h3>Explore the App</h3></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="nav-grid">
        <div class="nav-tile">
            <span class="nav-tile-icon">📊</span>
            <div class="nav-tile-title">Matchups & Standings</div>
            <div class="nav-tile-desc">Current standings, weekly matchups, head-to-head comparisons</div>
            <span class="nav-tile-tag">Use Managers Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">👤</span>
            <div class="nav-tile-title">Player Stats</div>
            <div class="nav-tile-desc">Weekly, season, and career player analytics</div>
            <span class="nav-tile-tag">Use Players Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">🎯</span>
            <div class="nav-tile-title">Draft Analysis</div>
            <div class="nav-tile-desc">Draft boards, SPAR values, ROI analysis</div>
            <span class="nav-tile-tag">Use Draft Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">💼</span>
            <div class="nav-tile-title">Transactions</div>
            <div class="nav-tile-desc">Trades, adds, drops, FAAB spending</div>
            <span class="nav-tile-tag">Use Transactions Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">🔮</span>
            <div class="nav-tile-title">Simulations</div>
            <div class="nav-tile-desc">Playoff odds and what-if scenarios</div>
            <span class="nav-tile-tag">Use Simulations Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">🏆</span>
            <div class="nav-tile-title">Hall of Fame</div>
            <div class="nav-tile-desc">Champions, records, legendary games</div>
            <span class="nav-tile-tag">See Below</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_key_concepts() -> None:
    """Render a compact key concepts section."""
    st.markdown('<div class="section-header"><h3>Key Concepts</h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>SPAR (Season Points Above Replacement)</h4>
            <p>Measures player value vs a replacement-level player at the same position.
            Higher SPAR = more valuable.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Optimal Lineup</h4>
            <p>The perfect lineup using hindsight. Compare actual vs optimal to measure
            lineup efficiency.</p>
        </div>
        """, unsafe_allow_html=True)


def _render_quick_tips() -> None:
    """Render quick tips for new users."""
    st.markdown('<div class="section-header"><h3>Quick Tips</h3></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Finding Players</h4>
            <p>Go to <strong>Players</strong> tab and use the filters to search.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Playoff Odds</h4>
            <p>Check <strong>Simulations</strong> tab for live playoff probabilities.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>Draft Value</h4>
            <p>Use <strong>Draft</strong> tab to see which picks delivered value.</p>
        </div>
        """, unsafe_allow_html=True)


@st.fragment
def display_homepage_overview(df_dict: Optional[Dict[str, Any]] = None) -> None:
    """Homepage with subtab controlled via hamburger menu."""
    # Apply theme and styles
    inject_theme_css()
    apply_modern_styles()
    _apply_homepage_styles()

    df_dict = df_dict or {}
    summary = df_dict.get("summary", {})
    matchup_df = get_matchup_df(df_dict)

    # Get subtab from session state (controlled by hamburger menu)
    subtab_idx = st.session_state.get("subtab_Home", 0)
    section_names = ["Overview", "Hall of Fame", "Standings", "Schedules", "Head-to-Head", "Recaps"]
    section_name = section_names[subtab_idx] if subtab_idx < len(section_names) else "Overview"

    # Render only the active section (lazy loading)
    if section_name == "Overview":
        _render_hero(summary)
        _render_quick_stats(summary)
        _render_navigation_tiles()
        _render_key_concepts()
        _render_quick_tips()

    elif section_name == "Hall of Fame":
        if HALL_OF_FAME_AVAILABLE:
            try:
                HallOfFameViewer(df_dict).display()
            except Exception as e:
                st.error(f"Failed to render Hall of Fame: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("Hall of Fame module not found.")
            with st.expander("Debug: Import error"):
                st.code(HALL_OF_FAME_ERROR)

    elif section_name == "Standings":
        st.markdown('<div class="section-header"><h3>Season Standings</h3></div>', unsafe_allow_html=True)

        if matchup_df is None or matchup_df.empty:
            st.info("Season Standings will appear once game data is loaded.")
        else:
            try:
                total_teams = matchup_df['manager'].nunique()
                current_week = int(matchup_df['week'].max())
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Teams", total_teams)
                with col2:
                    st.metric("Current Week", current_week)
                with col3:
                    st.metric("Games Played", len(matchup_df))
            except Exception:
                pass

            display_season_standings(matchup_df, prefix="standings")

    elif section_name == "Schedules":
        st.markdown('<div class="section-header"><h3>Team Schedules</h3></div>', unsafe_allow_html=True)
        display_schedules(df_dict, prefix="schedules")

    elif section_name == "Head-to-Head":
        st.markdown('<div class="section-header"><h3>Head-to-Head Matchups</h3></div>', unsafe_allow_html=True)
        st.info("Select 'All' or 'Optimal' in the matchup dropdown to see the league-wide optimal lineup!")

        if matchup_df is not None and not matchup_df.empty:
            try:
                y = int(matchup_df["year"].max())
                w = int(matchup_df[matchup_df["year"] == y]["week"].max())
                player_two_week = load_player_two_week_slice(y, w)
                df_dict_combined = dict(df_dict or {})
                df_dict_combined["Player Data"] = player_two_week
                display_head_to_head(df_dict_combined)
            except Exception as e:
                st.warning(f"Failed to load two-week player slice: {e}")
                display_head_to_head(df_dict)
        else:
            display_head_to_head(df_dict)

    elif section_name == "Recaps":
        st.markdown('<div class="section-header"><h3>Weekly Team Recaps</h3></div>', unsafe_allow_html=True)
        st.success("Get narrative recaps for each team including top performers and award-worthy moments.")

        from md.tab_data_access.homepage import load_recaps_matchup_data
        recaps_data = load_recaps_matchup_data()
        display_recap_overview(recaps_data)
