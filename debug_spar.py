import pandas as pd

player_file = r'C:\Users\joeye\Downloads\league-import-76e1b4f048a2dfb5\fantasy_football_data\player.parquet'
draft_file = r'C:\Users\joeye\Downloads\league-import-76e1b4f048a2dfb5\fantasy_football_data\draft.parquet'

player_df = pd.read_parquet(player_file)
draft_df = pd.read_parquet(draft_file)

print('=== FULL SIMULATION OF player_to_draft_v2.py ===')

# Convert league_id to string
if str(player_df['league_id'].dtype) != 'string':
    player_df['league_id'] = player_df['league_id'].astype('string')

draft_league_id = str(draft_df['league_id'].iloc[0])
print(f'Draft league_id: {draft_league_id}')

# Filter - keep NA league_ids
player_df = player_df[
    (player_df['league_id'] == draft_league_id) |
    (player_df['league_id'].isna()) |
    (player_df['league_id'] == '<NA>')
].copy()

# Fill NAs
mask_na = player_df['league_id'].isna()
mask_str_na = player_df['league_id'] == '<NA>'
player_df.loc[mask_na | mask_str_na, 'league_id'] = draft_league_id

print(f'Player df after filtering: {len(player_df):,} rows')

# Group keys
group_keys = ['yahoo_player_id', 'year', 'league_id']

# Build aggregation
points_col = 'fantasy_points' if 'fantasy_points' in player_df.columns else 'points'

agg_dict = {}
if points_col in player_df.columns:
    agg_dict['total_fantasy_points'] = (points_col, lambda x: x.fillna(0).sum())
    agg_dict['games_played'] = ('week', 'nunique')
    agg_dict['season_ppg'] = (points_col, lambda x: x.mean() if len(x) > 0 else 0.0)

# SPAR columns
if 'player_spar' in player_df.columns:
    agg_dict['player_spar'] = ('player_spar', lambda x: x.fillna(0).sum())
if 'manager_spar' in player_df.columns:
    agg_dict['manager_spar'] = ('manager_spar', lambda x: x.fillna(0).sum())
if 'replacement_ppg' in player_df.columns:
    agg_dict['replacement_ppg'] = ('replacement_ppg', lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else 0.0)

print(f'Aggregation dict keys: {list(agg_dict.keys())}')

metrics_df = player_df.groupby(group_keys, as_index=False).agg(**agg_dict)
print(f'Aggregated to {len(metrics_df):,} rows')
print(f'Metrics with SPAR > 0: {(metrics_df["player_spar"] > 0).sum() if "player_spar" in metrics_df.columns else 0}')

# Normalize types for merge
for key in group_keys:
    if key == 'league_id':
        continue
    if key in draft_df.columns:
        draft_df[key] = pd.to_numeric(draft_df[key], errors='coerce').astype('Int64')
    if key in metrics_df.columns:
        metrics_df[key] = pd.to_numeric(metrics_df[key], errors='coerce').astype('Int64')

# Check join key alignment
print()
print(f'metrics_df league_id values: {metrics_df["league_id"].unique().tolist()}')
print(f'draft_df league_id values: {draft_df["league_id"].unique().tolist()}')

# Do the join
enriched = draft_df.merge(metrics_df, on=group_keys, how='left', suffixes=('', '_m'))
matched = enriched['player_spar'].notna().sum() if 'player_spar' in enriched.columns else 0
print(f'Final match: {matched}/{len(enriched)} draft rows have SPAR')

# Check why it's still 0 even for matching league_id
draft_first_league = draft_df[draft_df['league_id'] == draft_league_id]
print(f'\nDraft rows with first league_id: {len(draft_first_league)}')

# Check specific match
sample = draft_first_league.iloc[0]
pid, yr, lid = sample['yahoo_player_id'], sample['year'], sample['league_id']
print(f'Sample: yahoo_player_id={pid}, year={yr}, league_id={lid}')

match = metrics_df[(metrics_df['yahoo_player_id'] == pid) & (metrics_df['year'] == yr) & (metrics_df['league_id'] == lid)]
print(f'Matching rows in metrics_df: {len(match)}')
if len(match) > 0:
    print(f'  player_spar: {match["player_spar"].iloc[0]}')
