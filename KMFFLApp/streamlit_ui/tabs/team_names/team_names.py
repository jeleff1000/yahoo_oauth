import streamlit as st
from ..shared.modern_styles import apply_modern_styles


def _hsl_from_text(s: str) -> str:
    """Create a deterministic H value from text and return an HSL color string."""
    # Simple deterministic hash -> hue 0-359
    h = sum(ord(c) for c in (s or "")) % 360
    # Use pleasant saturation/lightness
    return f"hsl({h}, 70%, 70%)"


@st.fragment
def display_team_names(team_names_data):
    """Team Names view with matrix and division grouping.

    - Expects team_names_data (DataFrame) with columns: manager, year, team_name, division_id
    - Renders a compact HTML table: rows = year, columns = manager, values = team_name
    - Option to group by division
    """
    apply_modern_styles()

    # Expecting columns: manager, year, team_name, division_id
    if team_names_data is None or not {'manager', 'year', 'team_name'}.issubset(team_names_data.columns):
        st.error("Team Names Data not found or missing required columns (manager, year, team_name).")
        return

    # View selector
    view_mode = st.radio("View", ["All Managers", "By Division"], horizontal=True, label_visibility="collapsed")

    data = team_names_data.copy()
    data = data[['manager', 'team_name', 'year', 'division_id']].dropna(subset=['manager', 'team_name'])
    data['year'] = data['year'].astype(str)
    data['manager'] = data['manager'].astype(str)
    data['team_name'] = data['team_name'].astype(str)
    data['division_id'] = data['division_id'].fillna('Unknown').astype(str)

    if view_mode == "All Managers":
        _render_all_managers_view(data)
    else:
        _render_division_view(data)


def _render_all_managers_view(data):
    """Render matrix view with all managers."""
    # Build pivot: rows = year, cols = manager
    pivot = data.pivot_table(
        index='year',
        columns='manager',
        values='team_name',
        aggfunc=lambda vals: '<br>'.join(sorted(set([str(v) for v in vals if v and str(v).strip()])))
    ).fillna('')

    # Ensure consistent column order (sorted alphabetically)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Add dark-mode compatible CSS
    st.markdown("""
    <style>
    .team-matrix { border-collapse: collapse; width: 100%; table-layout: fixed; font-family: Inter, system-ui, Arial; background: #1e293b; }
    .team-matrix th, .team-matrix td { border: 1px solid #334155; padding: 6px 8px; text-align: left; vertical-align: top; word-break: break-word; color: white; }
    .team-matrix th { position: sticky; top: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; z-index: 3; }
    .team-matrix .year-col { position: sticky; left: 0; background: #334155; font-weight:700; z-index: 4; color: white; }
    .team-matrix .manager-col { min-width: 140px; }
    .team-matrix .badge { display:inline-block; padding:4px 8px; border-radius:8px; background: rgba(102,126,234,0.3); margin:2px 0; color: white; }
    .team-matrix-wrapper { overflow: auto; max-height: 72vh; border-radius:6px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
    .team-matrix tbody tr:nth-child(even) td { background: #334155; }
    .team-matrix tbody tr:nth-child(odd) td { background: #1e293b; }
    .team-matrix tbody tr:nth-child(even) .year-col { background: #475569; }
    </style>
    """, unsafe_allow_html=True)

    html = ['<div class="team-matrix-wrapper"><table class="team-matrix">']
    # header
    html.append('<thead><tr>')
    html.append('<th class="year-col">Year</th>')
    for mgr in pivot.columns:
        html.append(f'<th class="manager-col">{mgr}</th>')
    html.append('</tr></thead>')

    # body
    html.append('<tbody>')
    for year in pivot.index:
        html.append('<tr>')
        html.append(f'<td class="year-col">{year}</td>')
        for mgr in pivot.columns:
            cell = pivot.at[year, mgr] if mgr in pivot.columns else ''
            if not cell:
                html.append('<td></td>')
            else:
                parts = [p.strip() for p in str(cell).split('<br>') if p.strip()]
                badges = ''.join([f'<div class="badge">{p}</div>' for p in parts])
                html.append(f'<td>{badges}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')

    st.html(''.join(html))


def _render_division_view(data):
    """Render separate tables for each division."""
    # Get unique divisions
    divisions = sorted(data['division_id'].unique())

    # Add dark-mode compatible CSS
    st.markdown("""
    <style>
    .division-section { margin-bottom: 2rem; }
    .division-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 6px;
        color: white;
    }
    .team-matrix { border-collapse: collapse; width: 100%; table-layout: fixed; font-family: Inter, system-ui, Arial; background: #1e293b; }
    .team-matrix th, .team-matrix td { border: 1px solid #334155; padding: 6px 8px; text-align: left; vertical-align: top; word-break: break-word; color: white; }
    .team-matrix th { position: sticky; top: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; z-index: 3; }
    .team-matrix .year-col { position: sticky; left: 0; background: #334155; font-weight:700; z-index: 4; color: white; }
    .team-matrix .manager-col { min-width: 140px; }
    .team-matrix .badge { display:inline-block; padding:4px 8px; border-radius:8px; background: rgba(102,126,234,0.3); margin:2px 0; color: white; }
    .team-matrix-wrapper { overflow: auto; max-height: 50vh; border-radius:6px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
    .team-matrix tbody tr:nth-child(even) td { background: #334155; }
    .team-matrix tbody tr:nth-child(odd) td { background: #1e293b; }
    .team-matrix tbody tr:nth-child(even) .year-col { background: #475569; }
    </style>
    """, unsafe_allow_html=True)

    for division in divisions:
        division_data = data[data['division_id'] == division]

        # Build pivot for this division
        pivot = division_data.pivot_table(
            index='year',
            columns='manager',
            values='team_name',
            aggfunc=lambda vals: '<br>'.join(sorted(set([str(v) for v in vals if v and str(v).strip()])))
        ).fillna('')

        # Ensure consistent column order
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        # Render division header and table
        html = ['<div class="division-section">']
        html.append(f'<div class="division-header">Division {division}</div>')
        html.append('<div class="team-matrix-wrapper"><table class="team-matrix">')

        # header
        html.append('<thead><tr>')
        html.append('<th class="year-col">Year</th>')
        for mgr in pivot.columns:
            html.append(f'<th class="manager-col">{mgr}</th>')
        html.append('</tr></thead>')

        # body
        html.append('<tbody>')
        for year in pivot.index:
            html.append('<tr>')
            html.append(f'<td class="year-col">{year}</td>')
            for mgr in pivot.columns:
                cell = pivot.at[year, mgr] if mgr in pivot.columns else ''
                if not cell:
                    html.append('<td></td>')
                else:
                    parts = [p.strip() for p in str(cell).split('<br>') if p.strip()]
                    badges = ''.join([f'<div class="badge">{p}</div>' for p in parts])
                    html.append(f'<td>{badges}</td>')
            html.append('</tr>')
        html.append('</tbody></table></div></div>')

        st.html(''.join(html))

