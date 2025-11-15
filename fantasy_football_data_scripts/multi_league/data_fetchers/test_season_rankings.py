import importlib.util
from pathlib import Path
import pandas as pd

module_path = Path(r"c:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\transformations\modules\season_rankings.py")
spec = importlib.util.spec_from_file_location("season_rankings_mod", str(module_path))
season_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(season_mod)

calculate = season_mod.calculate_season_rankings

# Build sample data
rows = [
    {'year':2024, 'manager':'Alice', 'win':1, 'loss':0, 'team_points':100, 'is_playoffs':0, 'is_consolation':0},
    {'year':2024, 'manager':'Alice', 'win':1, 'loss':0, 'team_points':90,  'is_playoffs':0, 'is_consolation':0},
    {'year':2024, 'manager':'Bob',   'win':0, 'loss':1, 'team_points':80,  'is_playoffs':0, 'is_consolation':0},
    {'year':2024, 'manager':'Bob',   'win':0, 'loss':1, 'team_points':70,  'is_playoffs':0, 'is_consolation':0},
]

df = pd.DataFrame(rows)

print('--- Championship NOT complete (provisional) ---')
out_prov = calculate(df, championship_complete=False)
print(out_prov[['year','manager','final_wins','final_regular_wins','season_mean','manager_season_ranking','champion']].drop_duplicates().to_string(index=False))

print('\n--- Championship complete (finalized) ---')
out_final = calculate(df, championship_complete=True)
print(out_final[['year','manager','final_wins','final_regular_wins','season_mean','manager_season_ranking','champion']].drop_duplicates().to_string(index=False))

# Verify playoff flags exist
print('\nPlayoff columns present (prov):', all(col in out_prov.columns for col in ['champion','semifinal','quarterfinal','sacko']))
print('Playoff columns present (final):', all(col in out_final.columns for col in ['champion','semifinal','quarterfinal','sacko']))

