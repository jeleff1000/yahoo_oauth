import importlib.util
from pathlib import Path
import pandas as pd

module_path = Path(r"c:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\multi_league\transformations\player_to_transactions_v2.py")
spec = importlib.util.spec_from_file_location("ptt_mod", str(module_path))
ptt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptt_mod)

left_join = ptt_mod.left_join_transactions_player

print('Running left_join_transactions_player tests')

# Test 1: clean 1:1 join on yahoo_player_id + cumulative_week
L1 = pd.DataFrame([
    {'yahoo_player_id':'1','year':2024,'cumulative_week':202401,'tx_id':'T1'},
    {'yahoo_player_id':'2','year':2024,'cumulative_week':202401,'tx_id':'T2'}
])
R1 = pd.DataFrame([
    {'yahoo_player_id':'1','cumulative_week':202401,'fantasy_points':10},
    {'yahoo_player_id':'2','cumulative_week':202401,'fantasy_points':8}
])
print('\nTest 1: clean 1:1')
print(left_join(L1, R1, ['fantasy_points']))

# Test 2: right has duplicate keys1 but contains distinct player_week already
L2 = pd.DataFrame([
    {'yahoo_player_id':'1','year':2024,'cumulative_week':202401,'tx_id':'T1'},
    {'yahoo_player_id':'2','year':2024,'cumulative_week':202402,'tx_id':'T2'}
])
R2 = pd.DataFrame([
    {'yahoo_player_id':'1','cumulative_week':202401,'player_week':'1_alt_202401','fantasy_points':10},
    {'yahoo_player_id':'1','cumulative_week':202401,'player_week':'1_alt2_202401','fantasy_points':11},
    {'yahoo_player_id':'2','cumulative_week':202402,'player_week':'2_202402','fantasy_points':9},
])
print('\nTest 2: right has duplicate (yahoo_player_id,cumulative_week) -> should fallback to player_week where possible')
print(left_join(L2, R2, ['fantasy_points']))

# Test 3: left has nulls in yahoo_player_id forcing fallback to player_week
L3 = pd.DataFrame([
    {'yahoo_player_id':None,'year':2024,'cumulative_week':202403,'player_week':'x_202403','tx_id':'T3'},
    {'yahoo_player_id':'3','year':2024,'cumulative_week':202403,'player_week':'3_202403','tx_id':'T4'}
])
R3 = pd.DataFrame([
    {'player_week':'x_202403','fantasy_points':7},
    {'player_week':'3_202403','fantasy_points':6}
])
print('\nTest 3: left has null yahoo_player_id -> fallback to player_week')
print(left_join(L3, R3, ['fantasy_points']))

