"""
Streamlit column configurations for team stats tables.

Provides intelligent column configs with:
- Proper formatting (decimals, prefixes, suffixes)
- Help text tooltips
- Appropriate widths
- Number formatting
- Progress bars for percentages
"""

import streamlit as st
from typing import Dict, Any


# ============================================================================
# COLUMN CONFIGURATION BUILDERS
# ============================================================================

def get_base_column_config() -> Dict[str, Any]:
    """
    Get base column configurations that apply to all tables.

    Returns:
        Dictionary of column configurations
    """
    return {
        # Identity columns
        'Manager': st.column_config.TextColumn(
            'Manager',
            width='medium',
            help='Team manager name'
        ),
        'Year': st.column_config.NumberColumn(
            'Year',
            format='%d',
            width='small',
            help='Season year'
        ),
        'Week': st.column_config.NumberColumn(
            'Week',
            format='%d',
            width='small',
            help='Week number'
        ),
        'Position': st.column_config.TextColumn(
            'Pos',
            width='small',
            help='Fantasy position'
        ),

        # Core metrics
        'Points': st.column_config.NumberColumn(
            'Points',
            format='%.1f',
            width='small',
            help='Total fantasy points scored'
        ),
        'SPAR': st.column_config.NumberColumn(
            'SPAR',
            format='%.1f',
            width='small',
            help='Season Points Above Replacement - value generated above replacement level'
        ),
        'Player SPAR': st.column_config.NumberColumn(
            'Player SPAR',
            format='%.1f',
            width='small',
            help='Player-specific SPAR contribution'
        ),
        'Manager SPAR': st.column_config.NumberColumn(
            'Manager SPAR',
            format='%.1f',
            width='small',
            help='Manager decision-making SPAR contribution'
        ),
    }


def get_passing_column_config() -> Dict[str, Any]:
    """
    Get column configurations for passing stats.

    Returns:
        Dictionary of column configurations
    """
    return {
        'Pass Yds': st.column_config.NumberColumn(
            'Pass Yds',
            format='%.0f',
            width='small',
            help='Total passing yards'
        ),
        'Pass TD': st.column_config.NumberColumn(
            'Pass TD',
            format='%d',
            width='small',
            help='Passing touchdowns'
        ),
        'INT': st.column_config.NumberColumn(
            'INT',
            format='%d',
            width='small',
            help='Interceptions thrown'
        ),
        'Comp': st.column_config.NumberColumn(
            'Comp',
            format='%d',
            width='small',
            help='Completions'
        ),
        'Pass Att': st.column_config.NumberColumn(
            'Att',
            format='%d',
            width='small',
            help='Pass attempts'
        ),
        'Comp%': st.column_config.NumberColumn(
            'Comp%',
            format='%.1f%%',
            width='small',
            help='Completion percentage'
        ),
        'YPA': st.column_config.NumberColumn(
            'YPA',
            format='%.1f',
            width='small',
            help='Yards per attempt'
        ),
        'TD%': st.column_config.NumberColumn(
            'TD%',
            format='%.1f%%',
            width='small',
            help='Touchdown percentage (TDs per attempt)'
        ),
        'INT%': st.column_config.NumberColumn(
            'INT%',
            format='%.1f%%',
            width='small',
            help='Interception percentage (INTs per attempt)'
        ),
        'Air Yds': st.column_config.NumberColumn(
            'Air Yds',
            format='%.0f',
            width='small',
            help='Air yards (yards ball traveled in air)'
        ),
        'Pass YAC': st.column_config.NumberColumn(
            'YAC',
            format='%.0f',
            width='small',
            help='Yards after catch on completions'
        ),
        'Pass EPA': st.column_config.NumberColumn(
            'EPA',
            format='%.1f',
            width='small',
            help='Expected Points Added from passing'
        ),
        'CPOE': st.column_config.NumberColumn(
            'CPOE',
            format='%.1f',
            width='small',
            help='Completion Percentage Over Expected'
        ),
        'PACR': st.column_config.NumberColumn(
            'PACR',
            format='%.2f',
            width='small',
            help='Passing Air Conversion Ratio'
        ),
        'Pass 1st': st.column_config.NumberColumn(
            '1st Downs',
            format='%d',
            width='small',
            help='First downs via passing'
        ),
        'Pass 2PT': st.column_config.NumberColumn(
            '2PT',
            format='%d',
            width='small',
            help='Two-point conversions (passing)'
        ),
    }


def get_rushing_column_config() -> Dict[str, Any]:
    """
    Get column configurations for rushing stats.

    Returns:
        Dictionary of column configurations
    """
    return {
        'Rush Yds': st.column_config.NumberColumn(
            'Rush Yds',
            format='%.0f',
            width='small',
            help='Total rushing yards'
        ),
        'Rush TD': st.column_config.NumberColumn(
            'Rush TD',
            format='%d',
            width='small',
            help='Rushing touchdowns'
        ),
        'Rush Att': st.column_config.NumberColumn(
            'Carries',
            format='%d',
            width='small',
            help='Rushing attempts (carries)'
        ),
        'YPC': st.column_config.NumberColumn(
            'YPC',
            format='%.1f',
            width='small',
            help='Yards per carry'
        ),
        'Rush Fum': st.column_config.NumberColumn(
            'Fum',
            format='%d',
            width='small',
            help='Rushing fumbles'
        ),
        'Rush Fum Lost': st.column_config.NumberColumn(
            'Fum Lost',
            format='%d',
            width='small',
            help='Rushing fumbles lost'
        ),
        'Rush 1st': st.column_config.NumberColumn(
            '1st Downs',
            format='%d',
            width='small',
            help='First downs via rushing'
        ),
        'Rush EPA': st.column_config.NumberColumn(
            'EPA',
            format='%.1f',
            width='small',
            help='Expected Points Added from rushing'
        ),
        'Rush 2PT': st.column_config.NumberColumn(
            '2PT',
            format='%d',
            width='small',
            help='Two-point conversions (rushing)'
        ),
    }


def get_receiving_column_config() -> Dict[str, Any]:
    """
    Get column configurations for receiving stats.

    Returns:
        Dictionary of column configurations
    """
    return {
        'Rec': st.column_config.NumberColumn(
            'Rec',
            format='%d',
            width='small',
            help='Receptions'
        ),
        'Rec Yds': st.column_config.NumberColumn(
            'Rec Yds',
            format='%.0f',
            width='small',
            help='Receiving yards'
        ),
        'Rec TD': st.column_config.NumberColumn(
            'Rec TD',
            format='%d',
            width='small',
            help='Receiving touchdowns'
        ),
        'Targets': st.column_config.NumberColumn(
            'Tgt',
            format='%d',
            width='small',
            help='Times targeted'
        ),
        'Catch%': st.column_config.NumberColumn(
            'Catch%',
            format='%.1f%%',
            width='small',
            help='Catch rate (receptions / targets)'
        ),
        'YPR': st.column_config.NumberColumn(
            'YPR',
            format='%.1f',
            width='small',
            help='Yards per reception'
        ),
        'YPRT': st.column_config.NumberColumn(
            'YPT',
            format='%.1f',
            width='small',
            help='Yards per target'
        ),
        'Rec Air Yds': st.column_config.NumberColumn(
            'Air Yds',
            format='%.0f',
            width='small',
            help='Air yards on receptions'
        ),
        'Rec YAC': st.column_config.NumberColumn(
            'YAC',
            format='%.0f',
            width='small',
            help='Yards after catch'
        ),
        'Rec EPA': st.column_config.NumberColumn(
            'EPA',
            format='%.1f',
            width='small',
            help='Expected Points Added from receiving'
        ),
        'Target Share': st.column_config.NumberColumn(
            'Tgt%',
            format='%.1f%%',
            width='small',
            help='Target share percentage'
        ),
        'WOPR': st.column_config.NumberColumn(
            'WOPR',
            format='%.2f',
            width='small',
            help='Weighted Opportunity Rating'
        ),
        'RACR': st.column_config.NumberColumn(
            'RACR',
            format='%.2f',
            width='small',
            help='Receiver Air Conversion Ratio'
        ),
        'Air Yds Share': st.column_config.NumberColumn(
            'Air%',
            format='%.1f%%',
            width='small',
            help='Air yards share percentage'
        ),
        'Rec 1st': st.column_config.NumberColumn(
            '1st Downs',
            format='%d',
            width='small',
            help='First downs via receiving'
        ),
        'Rec Fum': st.column_config.NumberColumn(
            'Fum',
            format='%d',
            width='small',
            help='Receiving fumbles'
        ),
        'Rec Fum Lost': st.column_config.NumberColumn(
            'Fum Lost',
            format='%d',
            width='small',
            help='Receiving fumbles lost'
        ),
        'Rec 2PT': st.column_config.NumberColumn(
            '2PT',
            format='%d',
            width='small',
            help='Two-point conversions (receiving)'
        ),
    }


def get_kicking_column_config() -> Dict[str, Any]:
    """
    Get column configurations for kicking stats.

    Returns:
        Dictionary of column configurations
    """
    return {
        'FGM': st.column_config.NumberColumn(
            'FGM',
            format='%d',
            width='small',
            help='Field goals made'
        ),
        'FGA': st.column_config.NumberColumn(
            'FGA',
            format='%d',
            width='small',
            help='Field goal attempts'
        ),
        'FG%': st.column_config.NumberColumn(
            'FG%',
            format='%.1f%%',
            width='small',
            help='Field goal percentage'
        ),
        'FG Long': st.column_config.NumberColumn(
            'Long',
            format='%d',
            width='small',
            help='Longest field goal made'
        ),
        'FG 0-19': st.column_config.NumberColumn(
            '0-19',
            format='%d',
            width='small',
            help='FG made from 0-19 yards'
        ),
        'FG 20-29': st.column_config.NumberColumn(
            '20-29',
            format='%d',
            width='small',
            help='FG made from 20-29 yards'
        ),
        'FG 30-39': st.column_config.NumberColumn(
            '30-39',
            format='%d',
            width='small',
            help='FG made from 30-39 yards'
        ),
        'FG 40-49': st.column_config.NumberColumn(
            '40-49',
            format='%d',
            width='small',
            help='FG made from 40-49 yards'
        ),
        'FG 50+': st.column_config.NumberColumn(
            '50+',
            format='%d',
            width='small',
            help='FG made from 50+ yards'
        ),
        'FG 40+': st.column_config.NumberColumn(
            '40+',
            format='%d',
            width='small',
            help='FG made from 40+ yards (combined)'
        ),
        'FG Miss': st.column_config.NumberColumn(
            'Miss',
            format='%d',
            width='small',
            help='Field goals missed'
        ),
        'PAT Made': st.column_config.NumberColumn(
            'PAT',
            format='%d',
            width='small',
            help='Extra points made'
        ),
        'PAT Att': st.column_config.NumberColumn(
            'PAT Att',
            format='%d',
            width='small',
            help='Extra point attempts'
        ),
        'PAT Missed': st.column_config.NumberColumn(
            'PAT Miss',
            format='%d',
            width='small',
            help='Extra points missed'
        ),
    }


def get_defense_column_config() -> Dict[str, Any]:
    """
    Get column configurations for defensive stats.

    Returns:
        Dictionary of column configurations
    """
    return {
        'Sacks': st.column_config.NumberColumn(
            'Sacks',
            format='%.1f',
            width='small',
            help='Total sacks'
        ),
        'Sack Yds': st.column_config.NumberColumn(
            'Sack Yds',
            format='%.0f',
            width='small',
            help='Yards lost on sacks'
        ),
        'QB Hits': st.column_config.NumberColumn(
            'QB Hits',
            format='%d',
            width='small',
            help='Quarterback hits'
        ),
        'Def INT': st.column_config.NumberColumn(
            'INT',
            format='%d',
            width='small',
            help='Interceptions'
        ),
        'INT Yds': st.column_config.NumberColumn(
            'INT Yds',
            format='%.0f',
            width='small',
            help='Interception return yards'
        ),
        'Pass Def': st.column_config.NumberColumn(
            'PD',
            format='%d',
            width='small',
            help='Passes defended'
        ),
        'Tackles': st.column_config.NumberColumn(
            'Tkl',
            format='%d',
            width='small',
            help='Total tackles (solo + assists)'
        ),
        'Solo Tkl': st.column_config.NumberColumn(
            'Solo',
            format='%d',
            width='small',
            help='Solo tackles'
        ),
        'Tkl Ast': st.column_config.NumberColumn(
            'Ast',
            format='%d',
            width='small',
            help='Tackle assists'
        ),
        'TFL': st.column_config.NumberColumn(
            'TFL',
            format='%d',
            width='small',
            help='Tackles for loss'
        ),
        'TFL Yds': st.column_config.NumberColumn(
            'TFL Yds',
            format='%.0f',
            width='small',
            help='Yards lost on tackles for loss'
        ),
        'Fum Rec': st.column_config.NumberColumn(
            'FR',
            format='%d',
            width='small',
            help='Fumble recoveries'
        ),
        'FF': st.column_config.NumberColumn(
            'FF',
            format='%d',
            width='small',
            help='Forced fumbles'
        ),
        'Def TD': st.column_config.NumberColumn(
            'TD',
            format='%d',
            width='small',
            help='Defensive touchdowns'
        ),
        'Safeties': st.column_config.NumberColumn(
            'Sfty',
            format='%d',
            width='small',
            help='Safeties'
        ),
        'Pts Allow': st.column_config.NumberColumn(
            'PA',
            format='%.0f',
            width='small',
            help='Points allowed'
        ),
        'Pass Yds Allow': st.column_config.NumberColumn(
            'Pass YA',
            format='%.0f',
            width='small',
            help='Passing yards allowed'
        ),
        'Rush Yds Allow': st.column_config.NumberColumn(
            'Rush YA',
            format='%.0f',
            width='small',
            help='Rushing yards allowed'
        ),
        'Total Yds Allow': st.column_config.NumberColumn(
            'Total YA',
            format='%.0f',
            width='small',
            help='Total yards allowed'
        ),
        '3 and Out': st.column_config.NumberColumn(
            '3&Out',
            format='%d',
            width='small',
            help='Three and outs forced'
        ),
    }


def get_comparison_column_config() -> Dict[str, Any]:
    """
    Get column configurations for comparison metrics.

    Returns:
        Dictionary of column configurations
    """
    return {
        'vs Avg': st.column_config.NumberColumn(
            'vs Avg',
            format='%+.1f',
            width='small',
            help='Points above/below league average'
        ),
        'Points Rank': st.column_config.NumberColumn(
            'Rank',
            format='#%d',
            width='small',
            help='Ranking by points scored'
        ),
    }


# ============================================================================
# POSITION-SPECIFIC CONFIGS
# ============================================================================

def get_qb_column_config() -> Dict[str, Any]:
    """Get complete column config for QB stats."""
    config = get_base_column_config()
    config.update(get_passing_column_config())
    config.update(get_rushing_column_config())
    config.update(get_comparison_column_config())
    return config


def get_rb_column_config() -> Dict[str, Any]:
    """Get complete column config for RB stats."""
    config = get_base_column_config()
    config.update(get_rushing_column_config())
    config.update(get_receiving_column_config())
    config.update(get_comparison_column_config())
    return config


def get_wr_column_config() -> Dict[str, Any]:
    """Get complete column config for WR stats."""
    config = get_base_column_config()
    config.update(get_receiving_column_config())
    config.update(get_rushing_column_config())  # Some WRs rush
    config.update(get_comparison_column_config())
    return config


def get_te_column_config() -> Dict[str, Any]:
    """Get complete column config for TE stats."""
    config = get_base_column_config()
    config.update(get_receiving_column_config())
    config.update(get_comparison_column_config())
    return config


def get_k_column_config() -> Dict[str, Any]:
    """Get complete column config for K stats."""
    config = get_base_column_config()
    config.update(get_kicking_column_config())
    config.update(get_comparison_column_config())
    return config


def get_def_column_config() -> Dict[str, Any]:
    """Get complete column config for DEF stats."""
    config = get_base_column_config()
    config.update(get_defense_column_config())
    config.update(get_comparison_column_config())
    return config


def get_all_positions_column_config() -> Dict[str, Any]:
    """Get column config for All positions view."""
    config = get_base_column_config()
    config.update(get_passing_column_config())
    config.update(get_rushing_column_config())
    config.update(get_receiving_column_config())
    config.update(get_comparison_column_config())
    return config


# ============================================================================
# MASTER FUNCTION
# ============================================================================

def get_column_config_for_position(position: str = "All") -> Dict[str, Any]:
    """
    Get appropriate column configuration based on position.

    Args:
        position: Position type (QB, RB, WR, TE, K, DEF, All)

    Returns:
        Dictionary of column configurations for st.dataframe
    """
    position_configs = {
        'QB': get_qb_column_config,
        'RB': get_rb_column_config,
        'WR': get_wr_column_config,
        'TE': get_te_column_config,
        'K': get_k_column_config,
        'DEF': get_def_column_config,
        'All': get_all_positions_column_config,
    }

    config_func = position_configs.get(position, get_all_positions_column_config)
    return config_func()
