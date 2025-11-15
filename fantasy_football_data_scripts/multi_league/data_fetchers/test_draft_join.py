import importlib.util
from pathlib import Path
import pandas as pd

module_path = Path(r"c:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\transformations\draft_to_player_v2.py")
spec = importlib.util.spec_from_file_location("draft_mod", str(module_path))
draft_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draft_mod)

left_join = draft_mod.left_join_draft_player

print('Running left_join_draft_player tests')

# Test 1: clean 1:1 join on yahoo_player_id + year
L1 = pd.DataFrame([
    {'yahoo_player_id':'1','year':2024,'player_name':'P1'},
    {'yahoo_player_id':'2','year':2024,'player_name':'P2'}
])
R1 = pd.DataFrame([
    {'yahoo_player_id':'1','year':2024,'round':1},
    {'yahoo_player_id':'2','year':2024,'round':2}
])
print('\nTest 1: clean 1:1')
print(left_join(L1, R1, ['round']))

# Test 2: right has duplicate keys1 but contains distinct player_year already (simulate messy source)
L2 = pd.DataFrame([
    {'yahoo_player_id':'1','year':2024,'player_name':'P1'},
    {'yahoo_player_id':'2','year':2024,'player_name':'P2'}
])
R2 = pd.DataFrame([
    {'yahoo_player_id':'1','year':2024,'player_year':'1_alt_2024','round':10},
    {'yahoo_player_id':'1','year':2024,'player_year':'1_alt2_2024','round':11},
    {'yahoo_player_id':'2','year':2024,'player_year':'2_2024','round':2},
])
print('\nTest 2: right has duplicate (yahoo_player_id,year) but contains distinct player_year column')
print(left_join(L2, R2, ['round']))

# Test 3: left has nulls in yahoo_player_id forcing fallback to player_year
L3 = pd.DataFrame([
    {'yahoo_player_id':None,'year':2024,'player_year':'x_2024','player_name':'X'},
    {'yahoo_player_id':'3','year':2024,'player_year':'3_2024','player_name':'P3'}
])
R3 = pd.DataFrame([
    {'player_year':'x_2024','round':7},
    {'player_year':'3_2024','round':3}
])
print('\nTest 3: left has null yahoo_player_id -> fallback to player_year')
print(left_join(L3, R3, ['round']))

