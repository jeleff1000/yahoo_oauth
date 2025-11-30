#!/usr/bin/env python3
"""
KMFFL Homepage Overview - Revamped Landing Page

A visually engaging landing page that:
- Shows dynamic current season info
- Provides quick navigation to key sections
- Works well in dark mode and mobile
- Minimizes text, maximizes visual engagement
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st
from ..shared.modern_styles import apply_modern_styles

# Data helpers
from md.data_access import load_player_two_week_slice

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
    """Apply homepage-specific styles optimized for dark mode and mobile."""
    st.markdown("""
    <style>
    /* ========================================
       HOMEPAGE HERO - Dark Mode Optimized
       ======================================== */
    .homepage-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .homepage-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(102,126,234,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .homepage-hero h1 {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .homepage-hero .subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin: 0;
    }
    .homepage-hero .season-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* ========================================
       QUICK STATS ROW (Static info cards)
       ======================================== */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #252538 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.25rem;
    }
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ========================================
       NAVIGATION TILES (Non-clickable info cards)
       ======================================== */
    .nav-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    .nav-tile {
        background: linear-gradient(145deg, #1e1e2f 0%, #252538 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        cursor: default;
        text-decoration: none;
        display: block;
    }
    .nav-tile-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    .nav-tile-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .nav-tile-desc {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
        line-height: 1.5;
    }
    .nav-tile-tag {
        display: inline-block;
        background: rgba(102,126,234,0.2);
        color: #667eea;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.75rem;
    }

    /* ========================================
       HAMBURGER MENU DROPDOWN STYLES
       ======================================== */
    .section-dropdown {
        margin-bottom: 1.5rem;
    }
    .section-dropdown .stSelectbox > div > div {
        background: linear-gradient(145deg, #1e1e2f 0%, #252538 100%);
        border: 2px solid rgba(102,126,234,0.4);
        border-radius: 8px;
    }
    .section-dropdown .stSelectbox > div > div:hover {
        border-color: rgba(102,126,234,0.6);
    }

    /* ========================================
       SECTION HEADERS
       ======================================== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(102,126,234,0.3);
    }
    .section-header h3 {
        color: #ffffff !important;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
    }

    /* ========================================
       INFO CARDS (Light Mode Compatible)
       ======================================== */
    .info-card {
        background: linear-gradient(145deg, #252538 0%, #1e1e2f 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .info-card h4 {
        color: #667eea;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    .info-card p, .info-card li {
        color: rgba(255,255,255,0.8);
        line-height: 1.6;
    }

    /* ========================================
       LIGHT MODE OVERRIDES
       ======================================== */
    @media (prefers-color-scheme: light) {
        .homepage-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
        }
        .stat-card, .nav-tile, .info-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
            border: 1px solid #e0e0e0;
        }
        .stat-value {
            color: #5c6bc0;
        }
        .stat-label {
            color: #666;
        }
        .nav-tile-title {
            color: #333;
        }
        .nav-tile-desc {
            color: #666;
        }
        .info-card p, .info-card li {
            color: #555;
        }
    }

    /* ========================================
       MOBILE RESPONSIVE
       ======================================== */
    @media (max-width: 768px) {
        .homepage-hero {
            padding: 1.5rem 1.25rem;
            border-radius: 12px;
        }
        .homepage-hero h1 {
            font-size: 1.75rem;
        }
        .homepage-hero .subtitle {
            font-size: 0.95rem;
        }
        .stats-row {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
        .stat-card {
            padding: 1rem;
        }
        .stat-value {
            font-size: 1.5rem;
        }
        .stat-label {
            font-size: 0.75rem;
        }
        .nav-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        .nav-tile {
            padding: 1.25rem;
        }
        .nav-tile-icon {
            font-size: 2rem;
        }
        .nav-tile-title {
            font-size: 1.1rem;
        }
    }

    @media (max-width: 480px) {
        .homepage-hero {
            padding: 1.25rem 1rem;
        }
        .homepage-hero h1 {
            font-size: 1.5rem;
        }
        .stats-row {
            gap: 0.5rem;
        }
        .stat-card {
            padding: 0.75rem;
        }
        .stat-value {
            font-size: 1.25rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def _as_dataframe(obj: Any) -> Optional[pd.DataFrame]:
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
    except Exception:
        return None
    return None


def _get_matchup_df(df_dict: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Extract a matchup DataFrame from df_dict if provided."""
    if not isinstance(df_dict, dict):
        return None
    if "Matchup Data" in df_dict:
        return _as_dataframe(df_dict["Matchup Data"])
    for k, v in df_dict.items():
        if str(k).strip().lower() == "matchup data":
            return _as_dataframe(v)
    return None


def _render_hero(summary: Dict[str, Any]) -> None:
    """Render the dynamic hero section."""
    latest_year = summary.get("latest_year", 2024)
    latest_week = summary.get("latest_week", 1)

    st.markdown(f"""
    <div class="homepage-hero">
        <h1>Fantasy Football Command Center</h1>
        <p class="subtitle">25+ years of data, analytics, and insights at your fingertips</p>
        <span class="season-badge">Season {latest_year} &bull; Week {latest_week}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_quick_stats(summary: Dict[str, Any]) -> None:
    """Render the quick stats row."""
    matchups = summary.get("matchup_count", 0)
    players = summary.get("player_count", 0)
    drafts = summary.get("draft_count", 0)
    transactions = summary.get("transactions_count", 0)

    # Format large numbers
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
            <div class="nav-tile-desc">Current standings, weekly matchups, head-to-head comparisons, and optimal lineup analysis</div>
            <span class="nav-tile-tag">Use Managers Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">👤</span>
            <div class="nav-tile-title">Player Stats</div>
            <div class="nav-tile-desc">Weekly, season, and career player analytics with 12+ visualization types</div>
            <span class="nav-tile-tag">Use Players Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">🎯</span>
            <div class="nav-tile-title">Draft Analysis</div>
            <div class="nav-tile-desc">Draft boards, SPAR values, ROI analysis, keeper decisions, and draft trends</div>
            <span class="nav-tile-tag">Use Draft Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">💼</span>
            <div class="nav-tile-title">Transactions</div>
            <div class="nav-tile-desc">Trades, adds, drops, FAAB spending, and waiver wire success rates</div>
            <span class="nav-tile-tag">Use Transactions Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">🔮</span>
            <div class="nav-tile-title">Simulations</div>
            <div class="nav-tile-desc">Playoff odds, schedule simulations, what-if scenarios, and predictions</div>
            <span class="nav-tile-tag">Use Simulations Tab</span>
        </div>
        <div class="nav-tile">
            <span class="nav-tile-icon">🏆</span>
            <div class="nav-tile-title">Hall of Fame</div>
            <div class="nav-tile-desc">Champions, records, legendary games, and dynasty tracking</div>
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
            <ul style="margin: 0.5rem 0 0 1rem; padding: 0;">
                <li><strong>Player SPAR:</strong> Total production all season</li>
                <li><strong>Manager SPAR:</strong> Production while on YOUR roster</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Optimal Lineup</h4>
            <p>The perfect lineup using hindsight. Compare actual vs optimal to measure
            lineup efficiency.</p>
            <ul style="margin: 0.5rem 0 0 1rem; padding: 0;">
                <li><strong>Team Optimal:</strong> Best from YOUR roster</li>
                <li><strong>League Optimal:</strong> Best across ALL teams</li>
            </ul>
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
            <p>Go to <strong>Players</strong> tab and use the filters to search by name, position, or manager.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Playoff Odds</h4>
            <p>Check <strong>Simulations</strong> tab for live playoff probabilities and championship odds.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>Draft Value</h4>
            <p>Use <strong>Draft</strong> tab to see which picks delivered value and which were busts.</p>
        </div>
        """, unsafe_allow_html=True)


@st.fragment
def display_homepage_overview(df_dict: Optional[Dict[str, Any]] = None) -> None:
    """Homepage with hamburger menu navigation - revamped for visual appeal."""
    apply_modern_styles()
    _apply_homepage_styles()

    df_dict = df_dict or {}
    summary = df_dict.get("summary", {})
    matchup_df = _get_matchup_df(df_dict)

    # Section names for dropdown menu
    section_names = [
        "Overview",
        "Hall of Fame",
        "Standings",
        "Schedules",
        "Head-to-Head",
        "Recaps",
    ]

    # Initialize session state for selected section
    if "homepage_section" not in st.session_state:
        st.session_state["homepage_section"] = "Overview"

    # Hamburger-style dropdown menu with visible label
    selected_section = st.selectbox(
        "📂 Section",
        section_names,
        index=section_names.index(st.session_state.get("homepage_section", "Overview")),
        key="homepage_section_selector"
    )
    st.session_state["homepage_section"] = selected_section

    # ========================================
    # SECTION: OVERVIEW (Revamped Landing Page)
    # ========================================
    if selected_section == "Overview":
        # Hero section with current season info
        _render_hero(summary)

        # Quick stats row
        _render_quick_stats(summary)

        # Navigation tiles
        _render_navigation_tiles()

        # Key concepts (compact)
        _render_key_concepts()

        # Quick tips
        _render_quick_tips()

    # ========================================
    # SECTION: HALL OF FAME
    # ========================================
    elif selected_section == "Hall of Fame":
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

    # ========================================
    # SECTION: STANDINGS
    # ========================================
    elif selected_section == "Standings":
        st.markdown("""
        <div class="section-header">
            <h3>Season Standings</h3>
        </div>
        """, unsafe_allow_html=True)

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

    # ========================================
    # SECTION: SCHEDULES
    # ========================================
    elif selected_section == "Schedules":
        st.markdown("""
        <div class="section-header">
            <h3>Team Schedules</h3>
        </div>
        """, unsafe_allow_html=True)
        display_schedules(df_dict, prefix="schedules")

    # ========================================
    # SECTION: HEAD-TO-HEAD
    # ========================================
    elif selected_section == "Head-to-Head":
        st.markdown("""
        <div class="section-header">
            <h3>Head-to-Head Matchups</h3>
        </div>
        """, unsafe_allow_html=True)

        st.info("**Tip:** Select 'All' or 'Optimal' in the matchup dropdown to see the league-wide optimal lineup!")

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

    # ========================================
    # SECTION: RECAPS
    # ========================================
    elif selected_section == "Recaps":
        st.markdown("""
        <div class="section-header">
            <h3>Weekly Team Recaps</h3>
        </div>
        """, unsafe_allow_html=True)

        st.success("Get narrative recaps for each team including top performers, biggest disappointments, and award-worthy moments.")

        from md.tab_data_access.homepage import load_recaps_matchup_data
        recaps_data = load_recaps_matchup_data()
        display_recap_overview(recaps_data)
