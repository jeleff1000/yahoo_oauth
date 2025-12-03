import streamlit as st
import pandas as pd
import numpy as np

@st.fragment
def display_career_draft(draft_data):
    st.header("Career Draft Stats")

    # Only rename columns that need it
    draft_data = draft_data.rename(columns={
        'Team Manager': 'manager',
        'Year': 'year',
        'Name Full': 'player',
        'Primary Position': 'yahoo_position',
        'Cost': 'cost',
        'Is Keeper Status': 'is_keeper_status',
        'Pick': 'pick'
    })

    draft_data['manager'] = draft_data['manager'].astype(str)
    draft_data['year'] = draft_data['year'].astype(str)
    draft_data = draft_data[draft_data['manager'] != "nan"]
    draft_data['player'] = draft_data['player'].str.lower()

    col1, col2 = st.columns(2)
    with col1:
        team_managers = sorted(draft_data['manager'].unique().tolist())
        selected_team_managers = st.multiselect("Select Manager", options=team_managers, default=[], key='manager')
    with col2:
        yahoo_positions = sorted([pos for pos in draft_data['yahoo_position'].unique().tolist() if pos is not None])
        selected_yahoo_positions = st.multiselect("Select Position", options=yahoo_positions, default=[], key='yahoo_position')

    names_full = sorted([name for name in draft_data['player'].unique().tolist() if name is not None])
    selected_names_full = st.multiselect("Search Player Name", options=names_full, default=[], key='player')

    if selected_team_managers:
        draft_data = draft_data[draft_data['manager'].isin(selected_team_managers)]
    if selected_yahoo_positions:
        draft_data = draft_data[draft_data['yahoo_position'].isin(selected_yahoo_positions)]
    if selected_names_full:
        draft_data = draft_data[draft_data['player'].isin(selected_names_full)]

    # Ensure numeric columns exist and coerce types
    for col in ['cost', 'is_keeper_status', 'is_keeper_cost', 'pick']:
        if col in draft_data.columns:
            draft_data[col] = pd.to_numeric(draft_data[col], errors='coerce').fillna(0)

    # Use manager_spar (actual value while rostered) with fallback to spar
    if 'manager_spar' in draft_data.columns:
        draft_data['spar'] = pd.to_numeric(draft_data['manager_spar'], errors='coerce').fillna(0)
    elif 'spar' in draft_data.columns:
        draft_data['spar'] = pd.to_numeric(draft_data['spar'], errors='coerce').fillna(0)

    # times_drafted: derive if missing
    if 'times_drafted' not in draft_data.columns:
        # Prefer explicit 'pick' column if available; otherwise infer from cost > 0
        if 'pick' in draft_data.columns:
            draft_data['times_drafted'] = (pd.to_numeric(draft_data['pick'], errors='coerce').fillna(0) >= 1).astype(int)
        else:
            draft_data['times_drafted'] = (pd.to_numeric(draft_data.get('cost', 0), errors='coerce').fillna(0) > 0).astype(int)

    # is_keeper_cost: derive if missing (cost when keeper else 0)
    if 'is_keeper_cost' not in draft_data.columns:
        if 'is_keeper_status' in draft_data.columns:
            draft_data['is_keeper_cost'] = draft_data['cost'].where(draft_data['is_keeper_status'] == 1, 0).fillna(0)
        else:
            draft_data['is_keeper_cost'] = 0

    # Build aggregation dict
    agg_dict = {
        'cost': 'sum',
        'is_keeper_status': 'sum',
        'is_keeper_cost': 'sum',
        'times_drafted': 'sum'
    }

    # Add SPAR metrics if available
    if 'spar' in draft_data.columns:
        agg_dict['spar'] = 'sum'

    # Aggregation - safe because we've ensured columns exist
    try:
        aggregated_data = draft_data.groupby(['player', 'yahoo_position']).agg(agg_dict).reset_index()
    except KeyError as e:
        # Surface helpful debug information in-app rather than throwing an unhelpful exception
        missing = str(e)
        st.error(f"Aggregation failed - missing column(s): {missing}")
        st.write("Available columns:", draft_data.columns.tolist())
        st.write("Sample rows:")
        st.dataframe(draft_data)
        return

    # Calculate SPAR efficiency
    if 'spar' in aggregated_data.columns:
        aggregated_data['spar_per_dollar'] = (
            aggregated_data['spar'] / aggregated_data['cost']
        ).replace([np.inf, -np.inf], 0).fillna(0).round(2)

    aggregated_data['player'] = aggregated_data['player'].str.title()

    columns_to_display = [
        'player', 'yahoo_position', 'cost', 'is_keeper_status', 'is_keeper_cost', 'times_drafted'
    ]

    # Add SPAR metrics if available
    if 'spar' in aggregated_data.columns:
        columns_to_display.append('spar')
    if 'spar_per_dollar' in aggregated_data.columns:
        columns_to_display.append('spar_per_dollar')

    columns_present = [col for col in columns_to_display if col in aggregated_data.columns]

    # Configure column formatting
    column_config = {}
    if 'spar' in aggregated_data.columns:
        column_config['spar'] = st.column_config.NumberColumn('Total Manager SPAR', format='%.1f', help='Total Manager SPAR (actual value while rostered) across all drafts')
    if 'spar_per_dollar' in aggregated_data.columns:
        column_config['spar_per_dollar'] = st.column_config.NumberColumn('SPAR/$', format='%.2f', help='Total Manager SPAR per total dollar spent')

    st.dataframe(
        aggregated_data[columns_present],
        hide_index=True,
        column_config=column_config if column_config else None
    )