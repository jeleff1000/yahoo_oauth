#!/usr/bin/env python3
"""Script to update all md.data_access imports to use md.core or md.tab_data_access."""
import os
import re

# Define what goes where
CORE_IMPORTS = {
    'run_query', 'T', 'get_current_league_db', 'get_table_dict',
    'get_motherduck_connection', '_execute_query',
    'sql_quote', 'sql_in_list', 'sql_upper', 'sql_upper_in_list', 'sql_manager_norm',
    'SORT_MARKER', 'latest_season_and_week',
    'list_seasons', 'list_weeks', 'list_managers',
    'list_player_seasons', 'list_player_weeks', 'list_player_positions',
    'list_optimal_seasons', 'list_optimal_weeks',
    'detect_roster_structure', 'STARTER_POSITIONS', 'BENCH_POSITIONS',
}

TAB_LOADERS = {
    'load_homepage_data': 'homepage',
    'load_optimized_homepage_data': 'homepage',
    'load_player_two_week_slice': 'homepage',
    'load_managers_data': 'managers',
    'load_optimized_managers_data': 'managers',
    'load_players_weekly_data': 'players',
    'load_filtered_weekly_data': 'players',
    'load_weekly_player_data': 'players',
    'load_filtered_weekly_player_data': 'players',
    'load_players_season_data': 'players',
    'load_season_player_data': 'players',
    'load_players_career_data': 'players',
    'load_career_player_data': 'players',
    'load_player_week': 'players',
    'load_h2h_week_data': 'players',
    'load_optimal_week': 'players',
    'load_h2h_optimal_week_data': 'players',
    'load_draft_data': 'draft',
    'load_optimized_draft_data': 'draft',
    'load_draft_optimizer_data': 'draft',
    'load_transactions_data': 'transactions',
    'load_optimized_transactions_data': 'transactions',
    'load_simulations_data': 'simulations',
    'load_optimized_simulations_data': 'simulations',
    'load_graphs_data': 'data_access',
    'load_keepers_data': 'keepers',
    'load_optimized_keepers_data': 'keepers',
    'load_team_names_data': 'team_names',
    'load_optimized_team_names_data': 'team_names',
}

def parse_imports(content):
    imports = set()
    pattern = r'from md\.data_access import \(([^)]+)\)|from md\.data_access import ([^\n]+)'

    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        items = match.group(1) if match.group(1) else match.group(2)
        for item in re.split(r'[,\n]', items):
            item = item.strip()
            if item and not item.startswith('#'):
                name = item.split()[0] if ' as ' not in item else item.split(' as ')[0].strip()
                imports.add(name)
    return imports

def generate_new_imports(imports):
    core = []
    tabs = {}

    for imp in imports:
        if imp in CORE_IMPORTS:
            core.append(imp)
        elif imp in TAB_LOADERS:
            tab = TAB_LOADERS[imp]
            if tab not in tabs:
                tabs[tab] = []
            tabs[tab].append(imp)
        else:
            core.append(imp)

    lines = []
    if core:
        lines.append('from md.core import ' + ', '.join(sorted(core)))

    for tab, imps in sorted(tabs.items()):
        if tab == 'data_access':
            lines.append('from md.data_access import ' + ', '.join(sorted(imps)))
        else:
            lines.append('from md.tab_data_access.' + tab + ' import ' + ', '.join(sorted(imps)))

    return lines

def main():
    base_path = r'C:\Users\joeye\OneDrive\Desktop\yahoo_oauth\analytics_app'
    updated = []

    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for f in files:
            if not f.endswith('.py'):
                continue

            path = os.path.join(root, f)
            rel_path = os.path.relpath(path, base_path)

            # Skip md/ internal files (already updated)
            if rel_path.startswith('md\\') or rel_path.startswith('md/'):
                continue

            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()

            if 'from md.data_access import' not in content:
                continue

            imports = parse_imports(content)
            if not imports:
                continue

            new_imports = generate_new_imports(imports)
            new_import_str = '\n'.join(new_imports)

            # Replace multi-line imports
            new_content = re.sub(
                r'from md\.data_access import \([^)]+\)',
                new_import_str,
                content,
                flags=re.MULTILINE | re.DOTALL
            )
            # Replace single-line imports
            new_content = re.sub(
                r'from md\.data_access import [^\n]+',
                new_import_str,
                new_content
            )

            if new_content != content:
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                updated.append(rel_path)
                print('Updated: ' + rel_path)

    print('')
    print('Total: ' + str(len(updated)) + ' files updated')

if __name__ == '__main__':
    main()
