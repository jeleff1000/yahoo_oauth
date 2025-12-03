#!/usr/bin/env python3
"""
Transaction Report Card Tab

A visually styled report card showing a manager's transaction performance.
Designed to look like a printed school transcript with hand-written grade styling.
Modeled after draft_report_card.py.
"""
from __future__ import annotations
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import random
from md.core import T, run_query


def get_league_name() -> str:
    """Extract league name from table configuration."""
    try:
        table_ref = T.get('transactions', 'league.transactions')
        schema = table_ref.split('.')[0] if '.' in table_ref else 'League'
        # Strip 'l_' prefix if it was added for digit-starting names (e.g., 'l_5townsfootball')
        if schema.startswith("l_") and len(schema) > 2 and schema[2].isdigit():
            schema = schema[2:]
        return schema.upper()
    except Exception:
        return 'Fantasy League'


def calculate_gpa(grade_counts: dict) -> float:
    """Calculate GPA from grade distribution."""
    grade_points = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    total_points = sum(grade_points.get(g, 0) * count for g, count in grade_counts.items())
    total_count = sum(grade_counts.values())
    return total_points / total_count if total_count > 0 else 0


def get_base_grade(grade: str) -> str:
    """Get the base letter (A, B, C, D, F) from a grade."""
    if not grade or grade == 'N/A' or pd.isna(grade):
        return ''
    return str(grade)[0].upper()


def gpa_to_letter_grade(gpa: float) -> str:
    """Convert GPA (0-4) to letter grade with +/-."""
    if gpa >= 3.7:
        return 'A' if gpa >= 3.85 else 'A-'
    elif gpa >= 3.3:
        return 'B+' if gpa >= 3.5 else 'B'
    elif gpa >= 3.0:
        return 'B-'
    elif gpa >= 2.7:
        return 'C+'
    elif gpa >= 2.3:
        return 'C'
    elif gpa >= 2.0:
        return 'C-'
    elif gpa >= 1.7:
        return 'D+'
    elif gpa >= 1.3:
        return 'D'
    elif gpa >= 1.0:
        return 'D-'
    else:
        return 'F'


def build_transaction_report_card_html(
    league_name: str,
    manager: str,
    year: str,
    overall_grade: str,
    gpa: float,
    transactions_data: list,
    total_transactions: int,
    total_adds: int,
    total_drops: int,
    total_trades: int,
    total_net_spar: float,
    total_score: float,
    total_faab_spent: float,
    win_rate: float,
    best_pickup: dict,
    best_trade: dict,
    worst_drop: dict,
    grade_distribution: dict,
) -> str:
    """Build the complete HTML for the transaction report card."""

    # Build table rows - separate sections for Top 5 and Bottom 5
    def build_row(txn, show_year=False):
        base_grade = get_base_grade(txn.get('grade', ''))
        grade_class = f'grade-{base_grade}' if base_grade in ['A', 'B', 'C', 'D', 'F'] else ''
        result_emoji = txn.get('result_emoji', '')
        type_icons = {'add': '+', 'drop': '-', 'trade': '↔'}
        type_icon = type_icons.get(txn.get('type', '').lower(), '?')
        type_class = f"type-{txn.get('type', 'unknown').lower()}"
        faab_display = f"${txn.get('faab', 0):.0f}" if txn.get('faab', 0) > 0 else '-'

        score_val = txn.get('score', 0)
        if score_val >= 50:
            score_class = 'score-excellent'
        elif score_val >= 20:
            score_class = 'score-good'
        elif score_val >= 0:
            score_class = 'score-neutral'
        elif score_val >= -20:
            score_class = 'score-poor'
        else:
            score_class = 'score-terrible'

        # Headshot image
        headshot = txn.get('headshot_url', '')
        if headshot:
            player_cell = f'<img src="{headshot}" class="player-headshot" onerror="this.style.display=\'none\'"> {txn.get("player", "Unknown")}'
        else:
            player_cell = txn.get('player', 'Unknown')

        # Year column for career view
        year_cell = f"<td>{txn.get('year', '-')}</td>" if show_year else ""

        return f'''
        <tr>
            {year_cell}
            <td>{txn.get('week', '-')}</td>
            <td class="{type_class}">{type_icon}</td>
            <td class="player-cell">{player_cell}</td>
            <td>{txn.get('position', '-')}</td>
            <td>{faab_display}</td>
            <td>{txn.get('spar', 0):.1f}</td>
            <td class="{score_class}">{score_val:+.0f}</td>
            <td class="grade-cell {grade_class}">{txn.get('grade', '-')}</td>
            <td>{result_emoji}</td>
        </tr>
        '''

    # Determine if we should show year column (career view)
    show_year = year == "Career"

    # Split into 4 sections by type
    best_adds = [t for t in transactions_data if t.get('section') == 'best_adds']
    worst_drops = [t for t in transactions_data if t.get('section') == 'worst_drops']
    best_trades = [t for t in transactions_data if t.get('section') == 'best_trades']
    worst_trades = [t for t in transactions_data if t.get('section') == 'worst_trades']

    year_header = "<th>Year</th>" if show_year else ""
    colspan = '10' if show_year else '9'

    table_rows = ""

    # Best Adds section
    if best_adds:
        table_rows += f'''
        <tr class="section-header best-section">
            <td colspan="{colspan}">🏆 Best Adds ({len(best_adds)})</td>
        </tr>
        '''
        for txn in best_adds:
            table_rows += build_row(txn, show_year)

    # Worst Drops section
    if worst_drops:
        table_rows += f'''
        <tr class="section-header worst-section">
            <td colspan="{colspan}">😬 Worst Drops ({len(worst_drops)})</td>
        </tr>
        '''
        for txn in worst_drops:
            table_rows += build_row(txn, show_year)

    # Best Trades section
    if best_trades:
        table_rows += f'''
        <tr class="section-header trade-section">
            <td colspan="{colspan}">🤝 Best Trades ({len(best_trades)})</td>
        </tr>
        '''
        for txn in best_trades:
            table_rows += build_row(txn, show_year)

    # Worst Trades section
    if worst_trades:
        table_rows += f'''
        <tr class="section-header trade-worst-section">
            <td colspan="{colspan}">💔 Worst Trades ({len(worst_trades)})</td>
        </tr>
        '''
        for txn in worst_trades:
            table_rows += build_row(txn, show_year)

    # Grade messages
    base_overall = get_base_grade(overall_grade)
    grade_messages = {
        'A': ['Waiver Wire Wizard!', 'Elite GM!', 'Transaction Master!', 'Outstanding moves!'],
        'B': ['Solid pickups!', 'Good eye for talent!', 'Nice work!', 'Well managed!'],
        'C': ['Average performance', 'Room to improve', 'Keep studying!', 'Not bad, not great'],
        'D': ['Needs work...', 'Study the waiver wire!', 'Disappointing', 'Try harder'],
        'F': ['Disaster!', 'What happened?!', 'See me after class!', 'Complete failure']
    }
    messages = grade_messages.get(base_overall, [''])
    random.seed(hash(f"{manager}{year}"))
    grade_message = random.choice(messages)

    # Build grade distribution bars
    grade_bars = ""
    max_count = max(grade_distribution.values()) if grade_distribution else 1
    for grade in ['A', 'B', 'C', 'D', 'F']:
        count = grade_distribution.get(grade, 0)
        width_pct = (count / max_count * 100) if max_count > 0 else 0
        grade_bars += f'''
        <div class="grade-bar-row">
            <span class="grade-bar-label grade-{grade}">{grade}</span>
            <div class="grade-bar-container">
                <div class="grade-bar grade-bar-{grade}" style="width: {width_pct}%"></div>
            </div>
            <span class="grade-bar-count">{count}</span>
        </div>
        '''

    # Best/Worst highlights
    best_html = ""
    if best_pickup:
        best_html = f'''
        <div class="highlight-box highlight-best">
            <div class="highlight-label">Best Pickup</div>
            <div class="highlight-player">{best_pickup.get('player', 'N/A')}</div>
            <div class="highlight-stats">+{best_pickup.get('spar', 0):.1f} SPAR | ${best_pickup.get('faab', 0):.0f}</div>
        </div>
        '''

    trade_html = ""
    if best_trade:
        trade_html = f'''
        <div class="highlight-box highlight-trade">
            <div class="highlight-label">Best Trade</div>
            <div class="highlight-player">{best_trade.get('player', 'N/A')}</div>
            <div class="highlight-stats">+{best_trade.get('spar', 0):.1f} SPAR</div>
        </div>
        '''

    worst_html = ""
    if worst_drop:
        worst_html = f'''
        <div class="highlight-box highlight-worst">
            <div class="highlight-label">Worst Drop</div>
            <div class="highlight-player">{worst_drop.get('player', 'N/A')}</div>
            <div class="highlight-stats">-{worst_drop.get('spar', 0):.1f} SPAR Lost</div>
        </div>
        '''

    html = f'''
<!DOCTYPE html>
<html>
<head>
<style>
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: 'Times New Roman', Times, serif;
    background: transparent;
    padding: 20px;
}}

.report-card-container {{
    max-width: 900px;
    margin: 0 auto;
}}

.report-card {{
    background: linear-gradient(to bottom, #fffef5 0%, #f5f5dc 100%);
    border: 3px solid #8B4513;
    border-radius: 5px;
    padding: 30px 40px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1), inset 0 0 50px rgba(139, 69, 19, 0.05);
    position: relative;
    min-height: 800px;
}}

.report-card::before {{
    content: '';
    position: absolute;
    top: 15px;
    left: 15px;
    right: 15px;
    bottom: 15px;
    border: 1px solid #d4c4a8;
    pointer-events: none;
}}

.school-header {{
    text-align: center;
    border-bottom: 2px solid #8B4513;
    padding-bottom: 15px;
    margin-bottom: 20px;
}}

.school-name {{
    font-size: 26px;
    font-weight: bold;
    color: #2c1810;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 5px;
}}

.school-subtitle {{
    font-size: 14px;
    color: #5c4033;
    font-style: italic;
}}

.student-info {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
    padding: 10px 0;
    border-bottom: 1px dashed #8B4513;
    flex-wrap: wrap;
    gap: 10px;
}}

.student-info-item {{
    font-size: 15px;
}}

.student-info-label {{
    color: #5c4033;
    font-weight: bold;
}}

.student-info-value {{
    color: #2c1810;
    margin-left: 8px;
}}

.final-grade-section {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    margin: 15px 0;
    padding: 10px 0;
    flex-wrap: wrap;
}}

.final-grade-circle {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 80px;
    height: 80px;
    border: 3px solid #cc0000;
    border-radius: 50%;
    background: transparent;
    transform: rotate(-8deg);
    flex-shrink: 0;
}}

.final-grade-letter {{
    font-family: 'Segoe Script', 'Bradley Hand', 'Comic Sans MS', cursive;
    font-size: 36px;
    color: #cc0000;
    font-weight: normal;
    line-height: 1;
}}

.grade-message {{
    font-family: 'Segoe Script', 'Bradley Hand', 'Comic Sans MS', cursive;
    font-size: 22px;
    color: #cc0000;
    transform: rotate(-10deg);
}}

.gpa-display {{
    text-align: center;
}}

.gpa-label {{
    font-size: 12px;
    color: #5c4033;
}}

.gpa-value {{
    font-family: 'Segoe Script', 'Bradley Hand', 'Comic Sans MS', cursive;
    font-size: 28px;
    color: #cc0000;
}}

.highlights-row {{
    display: flex;
    gap: 20px;
    margin: 20px 0;
    justify-content: center;
    flex-wrap: wrap;
}}

.highlight-box {{
    padding: 12px 20px;
    border-radius: 8px;
    text-align: center;
    min-width: 180px;
}}

.highlight-best {{
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border: 2px solid #4caf50;
}}

.highlight-trade {{
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border: 2px solid #2196f3;
}}

.highlight-worst {{
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border: 2px solid #f44336;
}}

.highlight-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #5c4033;
    margin-bottom: 5px;
}}

.highlight-player {{
    font-size: 16px;
    font-weight: bold;
    color: #2c1810;
}}

.highlight-stats {{
    font-size: 13px;
    color: #5c4033;
    margin-top: 3px;
}}

.grades-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 13px;
}}

.grades-table th {{
    background-color: #d4c4a8;
    color: #2c1810;
    padding: 10px 6px;
    text-align: left;
    border: 1px solid #8B4513;
    font-weight: bold;
    font-size: 12px;
}}

.section-header td {{
    background: linear-gradient(135deg, #f5f5dc 0%, #e8e4d4 100%);
    font-weight: bold;
    font-size: 14px;
    padding: 12px 10px;
    border: 1px solid #8B4513;
    color: #2c1810;
}}

.section-header.best-section td {{
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    color: #2e7d32;
}}

.section-header.worst-section td {{
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    color: #c62828;
}}

.section-header.trade-section td {{
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    color: #1565c0;
}}

.section-header.trade-worst-section td {{
    background: linear-gradient(135deg, #fce4ec 0%, #f8bbd9 100%);
    color: #ad1457;
}}

.player-headshot {{
    width: 28px;
    height: 28px;
    border-radius: 50%;
    vertical-align: middle;
    margin-right: 8px;
    border: 1px solid #ccc;
    object-fit: cover;
}}

.player-cell {{
    display: flex;
    align-items: center;
}}

.grades-table td {{
    padding: 8px 6px;
    border: 1px solid #c4b49a;
    background-color: rgba(255, 255, 255, 0.5);
}}

.grades-table tr:nth-child(even) td {{
    background-color: rgba(212, 196, 168, 0.2);
}}

.grade-cell {{
    font-weight: bold;
    font-size: 16px;
    text-align: center;
}}

.grade-A {{ color: #28a745; }}
.grade-B {{ color: #6c9a1f; }}
.grade-C {{ color: #b8860b; }}
.grade-D {{ color: #fd7e14; }}
.grade-F {{ color: #dc3545; }}

.score-excellent {{ color: #28a745; font-weight: bold; }}
.score-good {{ color: #6c9a1f; font-weight: bold; }}
.score-neutral {{ color: #5c4033; }}
.score-poor {{ color: #fd7e14; font-weight: bold; }}
.score-terrible {{ color: #dc3545; font-weight: bold; }}

.type-add {{ color: #28a745; font-weight: bold; }}
.type-drop {{ color: #dc3545; font-weight: bold; }}
.type-trade {{ color: #007bff; font-weight: bold; }}

.grade-distribution {{
    margin: 20px 0;
    padding: 15px;
    background: rgba(255,255,255,0.3);
    border-radius: 8px;
}}

.grade-distribution-title {{
    font-size: 14px;
    font-weight: bold;
    color: #2c1810;
    margin-bottom: 10px;
    text-align: center;
}}

.grade-bar-row {{
    display: flex;
    align-items: center;
    margin: 5px 0;
    gap: 10px;
}}

.grade-bar-label {{
    width: 25px;
    font-weight: bold;
    font-size: 14px;
}}

.grade-bar-container {{
    flex: 1;
    height: 18px;
    background: #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
}}

.grade-bar {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}}

.grade-bar-A {{ background: #28a745; }}
.grade-bar-B {{ background: #6c9a1f; }}
.grade-bar-C {{ background: #b8860b; }}
.grade-bar-D {{ background: #fd7e14; }}
.grade-bar-F {{ background: #dc3545; }}

.grade-bar-count {{
    width: 30px;
    font-size: 13px;
    color: #5c4033;
    text-align: right;
}}

.summary-section {{
    margin-top: 25px;
    padding-top: 15px;
    border-top: 2px solid #8B4513;
}}

.summary-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    text-align: center;
}}

.summary-item {{
    padding: 10px;
}}

.summary-label {{
    font-size: 11px;
    color: #5c4033;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.summary-value {{
    font-size: 22px;
    font-weight: bold;
    color: #2c1810;
    margin-top: 5px;
}}

.summary-value.positive {{ color: #28a745; }}
.summary-value.negative {{ color: #dc3545; }}

.signature-section {{
    margin-top: 25px;
    display: flex;
    justify-content: space-between;
    padding-top: 15px;
}}

.signature-line {{
    width: 180px;
    border-top: 1px solid #2c1810;
    padding-top: 5px;
    font-size: 11px;
    color: #5c4033;
    text-align: center;
}}

.stamp {{
    position: absolute;
    bottom: 35px;
    right: 35px;
    width: 75px;
    height: 75px;
    border: 3px solid #cc0000;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transform: rotate(15deg);
    opacity: 0.7;
}}

.stamp-text {{
    font-size: 9px;
    color: #cc0000;
    font-weight: bold;
    text-transform: uppercase;
}}

.stamp-year {{
    font-size: 16px;
    color: #cc0000;
    font-weight: bold;
}}

.watermark {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-30deg);
    font-size: 80px;
    color: rgba(139, 69, 19, 0.03);
    font-weight: bold;
    pointer-events: none;
    white-space: nowrap;
}}
</style>
</head>
<body>
<div class="report-card-container">
<div class="report-card">

<div class="watermark">{league_name}</div>

<div class="school-header">
    <div class="school-name">{league_name} Transaction Academy</div>
    <div class="school-subtitle">Official Transcript of Waiver Wire Performance</div>
</div>

<div class="student-info">
    <div class="student-info-item">
        <span class="student-info-label">Manager:</span>
        <span class="student-info-value">{manager}</span>
    </div>
    <div class="student-info-item">
        <span class="student-info-label">Season:</span>
        <span class="student-info-value">{year}</span>
    </div>
    <div class="student-info-item">
        <span class="student-info-label">Transactions:</span>
        <span class="student-info-value">{total_transactions}</span>
    </div>
    <div class="student-info-item">
        <span class="student-info-label">Win Rate:</span>
        <span class="student-info-value">{win_rate:.0f}%</span>
    </div>
</div>

<div class="final-grade-section">
    <div class="final-grade-circle">
        <span class="final-grade-letter">{overall_grade}</span>
    </div>
    <div class="grade-message">{grade_message}</div>
    <div class="gpa-display">
        <div class="gpa-label">Transaction GPA</div>
        <div class="gpa-value">{gpa:.2f}</div>
    </div>
</div>

<div class="highlights-row">
    {best_html}
    {trade_html}
    {worst_html}
</div>

<div class="grade-distribution">
    <div class="grade-distribution-title">Grade Distribution</div>
    {grade_bars}
</div>

<table class="grades-table">
<thead>
<tr>
    {year_header}
    <th>Wk</th>
    <th>Type</th>
    <th>Player</th>
    <th>Pos</th>
    <th>FAAB</th>
    <th>SPAR</th>
    <th>Score</th>
    <th>Grade</th>
    <th></th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<div class="summary-section">
    <div class="summary-grid">
        <div class="summary-item">
            <div class="summary-label">Adds</div>
            <div class="summary-value">{total_adds}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Drops</div>
            <div class="summary-value">{total_drops}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">FAAB Spent</div>
            <div class="summary-value">${total_faab_spent:.0f}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Total Score</div>
            <div class="summary-value {'positive' if total_score >= 0 else 'negative'}">{'+' if total_score >= 0 else ''}{total_score:.0f}</div>
        </div>
    </div>
</div>

<div class="signature-section">
    <div class="signature-line">Commissioner Signature</div>
    <div class="signature-line">Date Issued</div>
</div>

<div class="stamp">
    <div class="stamp-text">Official</div>
    <div class="stamp-year">{year}</div>
    <div class="stamp-text">{league_name}</div>
</div>

</div>
</div>
</body>
</html>
'''
    return html


@st.fragment
def display_transaction_report_card(transaction_df: pd.DataFrame, player_df: pd.DataFrame = None, career_view: bool = False) -> None:
    """Display a styled report card for a manager's transaction performance.

    Args:
        transaction_df: Transaction data with grades
        player_df: Optional player data for headshot_url lookup
        career_view: If True, defaults to Career view and uses different widget keys
    """

    st.markdown("### Transaction Report Card")
    st.markdown("*Official transcript of waiver wire performance*")

    df = transaction_df.copy()

    # Join with player data to get headshot_url if not already in transaction data
    # (headshot_url is now added in pipeline, but fallback to player join for older data)
    if 'headshot_url' not in df.columns and player_df is not None and 'yahoo_player_id' in df.columns:
        # Get unique player headshots
        headshot_lookup = player_df[['yahoo_player_id', 'headshot_url']].drop_duplicates('yahoo_player_id')
        df = df.merge(headshot_lookup, on='yahoo_player_id', how='left')

    if df.empty:
        st.warning("No transaction data available.")
        return

    # Check for required columns
    required_cols = ['manager', 'year', 'transaction_type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")
        return

    # Get managers and years
    managers = sorted(df['manager'].dropna().unique().tolist())
    years = sorted(df['year'].dropna().unique().tolist(), reverse=True)

    if not managers or not years:
        st.warning("No transaction data available.")
        return

    # Get league name
    league_name = get_league_name()

    # Widget keys differ based on view to avoid conflicts
    key_suffix = "_career" if career_view else "_season"

    # Dropdowns
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        selected_manager = st.selectbox("Select Manager", managers, key=f"txn_report_card_manager{key_suffix}")
    with col2:
        year_options = ["Career"] + [str(y) for y in years]
        # Default to Career for career_view, otherwise most recent season
        default_idx = 0 if career_view else 1
        selected_year = st.selectbox("Select Year", year_options, index=default_idx, key=f"txn_report_card_year{key_suffix}")

    # Filter data
    manager_df = df[df['manager'] == selected_manager].copy()

    if selected_year != "Career":
        manager_df = manager_df[manager_df['year'] == int(selected_year)]

    if manager_df.empty:
        st.warning(f"No transaction data for {selected_manager} in {selected_year}.")
        return

    # Calculate stats
    total_transactions = len(manager_df)
    total_adds = len(manager_df[manager_df['transaction_type'] == 'add'])
    total_drops = len(manager_df[manager_df['transaction_type'] == 'drop'])
    total_trades = len(manager_df[manager_df['transaction_type'] == 'trade'])

    # SPAR calculations
    spar_col = 'manager_spar_ros_managed' if 'manager_spar_ros_managed' in manager_df.columns else 'net_manager_spar_ros'
    if spar_col not in manager_df.columns:
        spar_col = 'fa_spar_ros'

    total_net_spar = manager_df['net_manager_spar_ros'].sum() if 'net_manager_spar_ros' in manager_df.columns else 0

    # Total transaction score (weighted metric)
    total_score = manager_df['transaction_score'].sum() if 'transaction_score' in manager_df.columns else total_net_spar

    # FAAB spent
    total_faab_spent = manager_df['faab_bid'].sum() if 'faab_bid' in manager_df.columns else 0

    # Win rate (positive NET SPAR)
    if 'net_manager_spar_ros' in manager_df.columns:
        wins = (manager_df['net_manager_spar_ros'] > 0).sum()
        win_rate = (wins / total_transactions * 100) if total_transactions > 0 else 0
    else:
        win_rate = 50  # Default

    # Grade distribution
    grade_distribution = {}
    if 'transaction_grade' in manager_df.columns:
        grade_counts = manager_df['transaction_grade'].value_counts()
        for grade in ['A', 'B', 'C', 'D', 'F']:
            grade_distribution[grade] = int(grade_counts.get(grade, 0))

    # Calculate GPA and overall grade
    gpa = calculate_gpa(grade_distribution)
    overall_grade = gpa_to_letter_grade(gpa)

    # Best pickup (adds only)
    best_pickup = None
    adds_df = manager_df[manager_df['transaction_type'] == 'add']
    if len(adds_df) > 0 and spar_col in adds_df.columns:
        best_idx = adds_df[spar_col].idxmax() if adds_df[spar_col].notna().any() else None
        if best_idx is not None:
            best_row = adds_df.loc[best_idx]
            best_pickup = {
                'player': best_row.get('player_name', 'Unknown'),
                'spar': best_row.get(spar_col, 0) or 0,
                'faab': best_row.get('faab_bid', 0) or 0,
            }

    # Best trade (trades only)
    best_trade = None
    trades_df = manager_df[manager_df['transaction_type'] == 'trade']
    if len(trades_df) > 0 and spar_col in trades_df.columns:
        best_trade_idx = trades_df[spar_col].idxmax() if trades_df[spar_col].notna().any() else None
        if best_trade_idx is not None:
            best_trade_row = trades_df.loc[best_trade_idx]
            best_trade = {
                'player': best_trade_row.get('player_name', 'Unknown'),
                'spar': best_trade_row.get(spar_col, 0) or 0,
            }

    # Worst drop (drops only - highest player_spar_ros_total = most regret)
    worst_drop = None
    drops_df = manager_df[manager_df['transaction_type'] == 'drop']
    drop_spar_col = 'player_spar_ros_total' if 'player_spar_ros_total' in drops_df.columns else 'drop_regret_score'
    if len(drops_df) > 0 and drop_spar_col in drops_df.columns:
        worst_idx = drops_df[drop_spar_col].idxmax() if drops_df[drop_spar_col].notna().any() else None
        if worst_idx is not None:
            worst_row = drops_df.loc[worst_idx]
            worst_drop = {
                'player': worst_row.get('player_name', 'Unknown'),
                'spar': worst_row.get(drop_spar_col, 0) or 0,
            }

    # Build transactions data for table - 4 sections by type
    transactions_data = []
    sort_col = 'transaction_score' if 'transaction_score' in manager_df.columns else spar_col
    if sort_col not in manager_df.columns:
        sort_col = 'week'

    def build_transaction_entry(row, section):
        spar_val = row.get(spar_col, 0)
        if pd.isna(spar_val):
            spar_val = 0
        score_val = row.get('transaction_score', spar_val)
        if pd.isna(score_val):
            score_val = 0

        # Get headshot URL if available
        headshot = row.get('headshot_url', '')
        if pd.isna(headshot):
            headshot = ''

        return {
            'week': int(row['week']) if pd.notna(row.get('week')) else '-',
            'year': int(row['year']) if pd.notna(row.get('year')) else '-',
            'type': row.get('transaction_type', row.get('type', '')),
            'player': row.get('player_name', 'Unknown'),
            'position': row.get('position', '-'),
            'faab': row.get('faab_bid', 0) or 0,
            'spar': spar_val,
            'score': score_val,
            'grade': row.get('transaction_grade', '-'),
            'result_emoji': row.get('result_emoji', ''),
            'transaction_result': row.get('transaction_result', ''),
            'headshot_url': headshot,
            'section': section,
        }

    # Split by transaction type
    adds_df = manager_df[manager_df['transaction_type'] == 'add'].copy()
    drops_df = manager_df[manager_df['transaction_type'] == 'drop'].copy()
    trades_df = manager_df[manager_df['transaction_type'] == 'trade'].copy()

    # For drops, use drop_regret_score or player_spar_ros_total (higher = worse drop)
    drop_sort_col = 'drop_regret_score' if 'drop_regret_score' in drops_df.columns else 'player_spar_ros_total'
    if drop_sort_col not in drops_df.columns:
        drop_sort_col = sort_col

    # Best 5 adds (highest score)
    if len(adds_df) > 0:
        best_adds = adds_df.sort_values(sort_col, ascending=False).head(5)
        for _, row in best_adds.iterrows():
            transactions_data.append(build_transaction_entry(row, 'best_adds'))

    # Worst 5 drops (highest regret score = most SPAR lost)
    if len(drops_df) > 0:
        worst_drops = drops_df.sort_values(drop_sort_col, ascending=False).head(5)
        for _, row in worst_drops.iterrows():
            transactions_data.append(build_transaction_entry(row, 'worst_drops'))

    # Best 2 trades (highest score)
    if len(trades_df) > 0:
        best_trades = trades_df.sort_values(sort_col, ascending=False).head(2)
        for _, row in best_trades.iterrows():
            transactions_data.append(build_transaction_entry(row, 'best_trades'))

    # Worst 2 trades (lowest score)
    if len(trades_df) > 1:  # Need at least 2 trades for worst section
        worst_trades = trades_df.sort_values(sort_col, ascending=True).head(2)
        # Avoid duplicates if only 2-3 trades total
        best_trade_keys = [(t['player'], t['week'], t['year']) for t in transactions_data if t['section'] == 'best_trades']
        for _, row in worst_trades.iterrows():
            entry = build_transaction_entry(row, 'worst_trades')
            if (entry['player'], entry['week'], entry['year']) not in best_trade_keys:
                transactions_data.append(entry)

    # Build and render HTML
    html = build_transaction_report_card_html(
        league_name=league_name,
        manager=selected_manager,
        year=str(selected_year),
        overall_grade=overall_grade,
        gpa=gpa,
        transactions_data=transactions_data,
        total_transactions=total_transactions,
        total_adds=total_adds,
        total_drops=total_drops,
        total_trades=total_trades,
        total_net_spar=total_net_spar,
        total_score=total_score,
        total_faab_spent=total_faab_spent,
        win_rate=win_rate,
        best_pickup=best_pickup,
        best_trade=best_trade,
        worst_drop=worst_drop,
        grade_distribution=grade_distribution,
    )

    # Calculate height based on content
    base_height = 850
    row_height = 35
    section_header_height = 40
    num_sections = sum([
        1 if any(t['section'] == 'best_adds' for t in transactions_data) else 0,
        1 if any(t['section'] == 'worst_drops' for t in transactions_data) else 0,
        1 if any(t['section'] == 'best_trades' for t in transactions_data) else 0,
        1 if any(t['section'] == 'worst_trades' for t in transactions_data) else 0,
    ])
    num_rows = len(transactions_data)  # Max ~14 (5+5+2+2)
    calculated_height = base_height + (num_rows * row_height) + (num_sections * section_header_height)

    # Render using components.html
    components.html(html, height=calculated_height, scrolling=True)
