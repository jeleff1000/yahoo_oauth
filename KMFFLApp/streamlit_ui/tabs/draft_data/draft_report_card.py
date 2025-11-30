#!/usr/bin/env python3
"""
Draft Report Card Tab

A visually styled report card showing a manager's draft performance for a specific year.
Designed to look like a printed school transcript with hand-written grade styling.
"""
from __future__ import annotations
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from md.data_access import run_query, T


def get_league_name() -> str:
    """Extract league name from table configuration."""
    try:
        # T['draft'] is something like 'kmffl.draft' - extract the schema name
        table_ref = T.get('draft', 'league.draft')
        schema = table_ref.split('.')[0] if '.' in table_ref else 'League'
        # Strip 'l_' prefix if it was added for digit-starting names (e.g., 'l_5townsfootball')
        if schema.startswith("l_") and len(schema) > 2 and schema[2].isdigit():
            schema = schema[2:]
        return schema.upper()
    except Exception:
        return 'Fantasy League'


@st.cache_data(ttl=600)
def get_player_headshots() -> dict:
    """Get player headshots from the player table."""
    try:
        query = f"""
            SELECT DISTINCT player, headshot_url
            FROM {T['player']}
            WHERE headshot_url IS NOT NULL
        """
        df = run_query(query)
        return dict(zip(df['player'], df['headshot_url']))
    except Exception:
        return {}


def percentile_to_letter_grade(percentile: float) -> str:
    """Convert percentile (0-100) to letter grade with +/- like a test score.

    Grading scale:
        97-100: A+    87-90: B+    77-80: C+    67-70: D+    0-60: F
        93-97:  A     83-87: B     73-77: C     63-67: D
        90-93:  A-    80-83: B-    70-73: C-    60-63: D-
    """
    if percentile is None:
        return 'N/A'

    if percentile >= 97:
        return 'A+'
    elif percentile >= 93:
        return 'A'
    elif percentile >= 90:
        return 'A-'
    elif percentile >= 87:
        return 'B+'
    elif percentile >= 83:
        return 'B'
    elif percentile >= 80:
        return 'B-'
    elif percentile >= 77:
        return 'C+'
    elif percentile >= 73:
        return 'C'
    elif percentile >= 70:
        return 'C-'
    elif percentile >= 67:
        return 'D+'
    elif percentile >= 63:
        return 'D'
    elif percentile >= 60:
        return 'D-'
    else:
        return 'F'


def get_base_grade(grade: str) -> str:
    """Get the base letter (A, B, C, D, F) from a grade like A+ or B-."""
    if not grade or grade == 'N/A':
        return ''
    return grade[0]


def build_report_card_html(
    league_name: str,
    manager: str,
    year: int,
    is_auction: bool,
    manager_grade: str,
    manager_percentile: float,
    picks_data: list,
    total_picks: int,
    total_spar: float,
    hit_rate: float
) -> str:
    """Build the complete HTML for the report card."""

    # Build table rows
    table_rows = ""
    for pick in picks_data:
        # Add def-logo class for DEF position
        headshot_class = "player-headshot def-logo" if pick['position'] == 'DEF' else "player-headshot"

        headshot_html = ""
        if pick['headshot_url']:
            headshot_html = f'<img src="{pick["headshot_url"]}" class="{headshot_class}" onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\'"><div class="player-headshot-placeholder" style="display:none">{pick["initials"]}</div>'
        else:
            headshot_html = f'<div class="player-headshot-placeholder">{pick["initials"]}</div>'

        # Add keeper mark if applicable
        keeper_mark = '<span class="keeper-mark">K</span>' if pick.get('is_keeper') else ''

        cost_cell = f'<td>${pick["cost"]}</td>' if is_auction else ''
        # Use base grade for color class (A+, A, A- all use grade-A)
        pick_base_grade = get_base_grade(pick["grade"]) if pick["grade"] else ''
        grade_class = f'grade-{pick_base_grade}' if pick_base_grade in ['A', 'B', 'C', 'D', 'F'] else ''

        table_rows += f'''
        <tr>
            <td>{pick["round"]}</td>
            {cost_cell}
            <td><div class="player-cell">{headshot_html}<span>{pick["player"]}</span>{keeper_mark}</div></td>
            <td>{pick["position"]}</td>
            <td>{pick["points"]}</td>
            <td>{pick["spar"]}</td>
            <td class="grade-cell {grade_class}">{pick["grade"]}</td>
        </tr>
        '''

    # Table header
    cost_header = '<th>Cost</th>' if is_auction else ''

    # Percentile text - just the score number
    percentile_text = f"{manager_percentile:.1f}" if manager_percentile else ""

    # Use the pipeline-calculated grade directly (already has +/-)
    letter_grade = manager_grade if manager_grade and manager_grade != 'N/A' else 'N/A'
    base_grade = get_base_grade(letter_grade)

    # Grade messages - multiple options per grade (use base letter)
    import random
    grade_messages = {
        'A': [
            'Great Work! :)',
            'Excellent drafting!',
            'Gold star for you!',
            'Top of the class!',
            'Outstanding!',
            'Nailed it!'
        ],
        'B': [
            'Pretty Good!',
            'Nice job!',
            'Solid effort!',
            'Well done!',
            'Keep it up!',
            'Good picks!'
        ],
        'C': [
            'See you in Study Hall!',
            'Room for improvement',
            'Average work',
            'Could be better...',
            'Needs more effort',
            'Just okay'
        ],
        'D': [
            'See me after class',
            'We need to talk...',
            'Disappointing',
            'Try harder next time',
            'Not your best work',
            'Concerning...'
        ],
        'F': [
            'Terrible! :(',
            'What happened?!',
            'Yikes...',
            'Did you even try?',
            'Complete disaster',
            'See me ASAP!'
        ]
    }
    messages = grade_messages.get(base_grade, [''])
    # Use manager + year as seed for consistent message per report card
    random.seed(hash(f"{manager}{year}"))
    grade_message = random.choice(messages)

    # Format summary values
    total_spar_str = f"{total_spar:.2f}" if total_spar else "N/A"
    hit_rate_str = f"{hit_rate:.0f}%" if hit_rate else "N/A"

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
    max-width: 850px;
    margin: 0 auto;
}}

.report-card {{
    background: linear-gradient(to bottom, #fffef5 0%, #f5f5dc 100%);
    border: 3px solid #8B4513;
    border-radius: 5px;
    padding: 30px 40px;
    box-shadow:
        0 4px 6px rgba(0,0,0,0.1),
        inset 0 0 50px rgba(139, 69, 19, 0.05);
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
    font-size: 28px;
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
    margin-bottom: 25px;
    padding: 10px 0;
    border-bottom: 1px dashed #8B4513;
}}

.student-info-item {{
    font-size: 16px;
}}

.student-info-label {{
    color: #5c4033;
    font-weight: bold;
}}

.student-info-value {{
    color: #2c1810;
    margin-left: 10px;
}}

.final-grade-section {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
    margin: 10px 0;
    padding: 5px 0;
}}

.final-grade-circle {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 70px;
    height: 70px;
    border: 3px solid #cc0000;
    border-radius: 50%;
    background: transparent;
    transform: rotate(-8deg);
    flex-shrink: 0;
}}

.final-grade-letter {{
    font-family: 'Segoe Script', 'Bradley Hand', 'Comic Sans MS', cursive;
    font-size: 32px;
    color: #cc0000;
    font-weight: normal;
    line-height: 1;
}}

.final-grade-info {{
    display: flex;
    align-items: baseline;
    gap: 8px;
}}

.final-grade-label {{
    font-size: 13px;
    color: #5c4033;
    font-style: italic;
}}

.percentile-text {{
    font-family: 'Segoe Script', 'Bradley Hand', 'Comic Sans MS', cursive;
    font-size: 28px;
    color: #cc0000;
}}

.grade-message {{
    font-family: 'Segoe Script', 'Bradley Hand', 'Comic Sans MS', cursive;
    font-size: 24px;
    color: #cc0000;
    transform: rotate(-12deg);
}}

.grades-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 14px;
}}

.grades-table th {{
    background-color: #d4c4a8;
    color: #2c1810;
    padding: 12px 8px;
    text-align: left;
    border: 1px solid #8B4513;
    font-weight: bold;
}}

.grades-table td {{
    padding: 10px 8px;
    border: 1px solid #c4b49a;
    background-color: rgba(255, 255, 255, 0.5);
}}

.grades-table tr:nth-child(even) td {{
    background-color: rgba(212, 196, 168, 0.2);
}}

.player-cell {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.player-headshot {{
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    object-position: center top;
    border: 2px solid #8B4513;
}}

.player-headshot.def-logo {{
    width: 36px;
    height: 36px;
    object-fit: contain;
    object-position: center center;
    background: #fff;
    padding: 4px;
}}

.keeper-mark {{
    font-family: 'Segoe Script', 'Bradley Hand', 'Comic Sans MS', cursive;
    color: #cc0000;
    font-size: 14px;
    font-weight: bold;
    margin-left: 3px;
}}

.player-headshot-placeholder {{
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #d4c4a8;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: #5c4033;
    border: 2px solid #8B4513;
    font-weight: bold;
}}

.grade-cell {{
    font-weight: bold;
    font-size: 18px;
    text-align: center;
}}

.grade-A {{ color: #28a745; }}
.grade-B {{ color: #6c9a1f; }}
.grade-C {{ color: #b8860b; }}
.grade-D {{ color: #fd7e14; }}
.grade-F {{ color: #dc3545; }}

.summary-section {{
    margin-top: 25px;
    padding-top: 15px;
    border-top: 2px solid #8B4513;
}}

.summary-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    text-align: center;
}}

.summary-item {{
    padding: 10px;
}}

.summary-label {{
    font-size: 12px;
    color: #5c4033;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.summary-value {{
    font-size: 24px;
    font-weight: bold;
    color: #2c1810;
    margin-top: 5px;
}}

.signature-section {{
    margin-top: 30px;
    display: flex;
    justify-content: space-between;
    padding-top: 20px;
}}

.signature-line {{
    width: 200px;
    border-top: 1px solid #2c1810;
    padding-top: 5px;
    font-size: 12px;
    color: #5c4033;
    text-align: center;
}}

.stamp {{
    position: absolute;
    bottom: 40px;
    right: 40px;
    width: 80px;
    height: 80px;
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
    font-size: 10px;
    color: #cc0000;
    font-weight: bold;
    text-transform: uppercase;
}}

.stamp-year {{
    font-size: 18px;
    color: #cc0000;
    font-weight: bold;
}}

.watermark {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-30deg);
    font-size: 100px;
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
    <div class="school-name">{league_name} Draft Academy</div>
    <div class="school-subtitle">Official Transcript of Draft Performance</div>
</div>

<div class="student-info">
    <div class="student-info-item">
        <span class="student-info-label">Student:</span>
        <span class="student-info-value">{manager}</span>
    </div>
    <div class="student-info-item">
        <span class="student-info-label">Academic Year:</span>
        <span class="student-info-value">{year}</span>
    </div>
    <div class="student-info-item">
        <span class="student-info-label">Draft Type:</span>
        <span class="student-info-value">{"Auction" if is_auction else "Snake"}</span>
    </div>
</div>

<div class="final-grade-section">
    <div class="final-grade-circle">
        <span class="final-grade-letter">{letter_grade}</span>
    </div>
    <div class="grade-message">{grade_message}</div>
    <div class="final-grade-info">
        <div class="final-grade-label">Overall Draft Grade</div>
        <div class="percentile-text">{percentile_text}</div>
    </div>
</div>

<table class="grades-table">
<thead>
<tr>
    <th>Rd</th>
    {cost_header}
    <th>Player</th>
    <th>Pos</th>
    <th>Points</th>
    <th>SPAR</th>
    <th>Grade</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<div class="summary-section">
    <div class="summary-grid">
        <div class="summary-item">
            <div class="summary-label">Total Picks</div>
            <div class="summary-value">{total_picks}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Total SPAR</div>
            <div class="summary-value">{total_spar_str}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Hit Rate</div>
            <div class="summary-value">{hit_rate_str}</div>
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
def display_draft_report_card(draft_data: pd.DataFrame) -> None:
    """Display a styled report card for a manager's draft performance."""

    st.markdown("### 📜 Draft Report Card")
    st.markdown("*Official transcript of draft performance*")

    df = draft_data.copy()

    # Check required columns
    required_cols = ['manager', 'year', 'draft_grade', 'manager_draft_grade']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}. Run the draft enrichment pipeline.")
        return

    # Filter to drafted players only (has a pick)
    if 'pick' in df.columns:
        df = df[df['pick'].notna()]

    # Get managers and years
    managers = sorted(df['manager'].dropna().unique().tolist())
    years = sorted(df['year'].dropna().unique().tolist(), reverse=True)

    if not managers or not years:
        st.warning("No draft data available.")
        return

    # Get league name dynamically
    league_name = get_league_name()

    # Dropdowns
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        selected_manager = st.selectbox("Select Manager", managers, key="report_card_manager")
    with col2:
        selected_year = st.selectbox("Select Year", years, key="report_card_year")

    # Filter data
    manager_year_df = df[
        (df['manager'] == selected_manager) &
        (df['year'] == selected_year)
    ].copy()

    if manager_year_df.empty:
        st.warning(f"No draft data for {selected_manager} in {selected_year}.")
        return

    # Get headshots from player table
    headshots = get_player_headshots()

    # Sort by round, then pick
    sort_cols = []
    if 'round' in manager_year_df.columns:
        sort_cols.append('round')
    if 'pick' in manager_year_df.columns:
        sort_cols.append('pick')
    if sort_cols:
        manager_year_df = manager_year_df.sort_values(sort_cols)

    # Get manager-level stats (same for all rows)
    manager_grade = manager_year_df['manager_draft_grade'].iloc[0] if 'manager_draft_grade' in manager_year_df.columns else 'N/A'
    manager_percentile = manager_year_df['manager_draft_percentile_alltime'].iloc[0] if 'manager_draft_percentile_alltime' in manager_year_df.columns else None
    total_spar = manager_year_df['manager_total_spar'].iloc[0] if 'manager_total_spar' in manager_year_df.columns else None
    hit_rate = manager_year_df['manager_hit_rate'].iloc[0] if 'manager_hit_rate' in manager_year_df.columns else None

    # Convert to float if not None
    if manager_percentile is not None:
        manager_percentile = float(manager_percentile) if pd.notna(manager_percentile) else None
    if total_spar is not None:
        total_spar = float(total_spar) if pd.notna(total_spar) else None
    if hit_rate is not None:
        hit_rate = float(hit_rate) if pd.notna(hit_rate) else None

    # Determine if auction or snake draft
    is_auction = manager_year_df['cost'].notna().any() if 'cost' in manager_year_df.columns else False

    # Build picks data
    picks_data = []
    for _, row in manager_year_df.iterrows():
        player_name = row.get('player', 'Unknown')
        position = row.get('yahoo_position', row.get('position', ''))
        round_num = int(row['round']) if pd.notna(row.get('round')) else '-'
        cost = int(row['cost']) if is_auction and pd.notna(row.get('cost')) else 0
        points = f"{row['points']:.1f}" if pd.notna(row.get('points')) else '-'

        # Get SPAR - prefer manager_spar, fall back to spar
        spar_val = row.get('manager_spar') if pd.notna(row.get('manager_spar')) else row.get('spar')
        spar = f"{spar_val:.1f}" if pd.notna(spar_val) else '-'

        grade = row.get('draft_grade', '-')
        if pd.isna(grade):
            grade = '-'

        # Check if keeper
        is_keeper = row.get('is_keeper_status', 0) == 1 or row.get('is_keeper_status') == True

        # Get headshot
        headshot_url = headshots.get(player_name, '')
        initials = ''.join([n[0] for n in str(player_name).split()[:2]]).upper()

        picks_data.append({
            'round': round_num,
            'cost': cost,
            'player': player_name,
            'position': position,
            'points': points,
            'spar': spar,
            'grade': grade,
            'headshot_url': headshot_url,
            'initials': initials,
            'is_keeper': is_keeper
        })

    # Build and render HTML
    html = build_report_card_html(
        league_name=league_name,
        manager=selected_manager,
        year=int(selected_year),
        is_auction=is_auction,
        manager_grade=str(manager_grade) if pd.notna(manager_grade) else 'N/A',
        manager_percentile=manager_percentile,
        picks_data=picks_data,
        total_picks=len(picks_data),
        total_spar=total_spar,
        hit_rate=hit_rate
    )

    # Calculate height based on number of picks
    base_height = 480
    row_height = 48
    calculated_height = base_height + (len(picks_data) * row_height)

    # Render using components.html for proper HTML/CSS rendering
    components.html(html, height=calculated_height, scrolling=True)
