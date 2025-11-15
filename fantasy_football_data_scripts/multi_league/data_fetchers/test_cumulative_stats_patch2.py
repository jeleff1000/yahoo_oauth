import pandas as pd
from multi_league.transformations.cumulative_stats_v2 import apply_cumulative_fixes

# Prepare test data to exercise mutual exclusivity and manager preservation
data = [
    {'manager': 'Alice', 'year': 2024, 'week': 1, 'win': 1, 'loss': 0, 'team_points': 100, 'is_playoffs': 1, 'is_consolation': 1},
    {'manager': 'Bob', 'year': 2024, 'week': 2, 'win': 0, 'loss': 1, 'team_points': 80, 'is_playoffs': 0, 'is_consolation': 1},
    {'manager': None, 'year': 2024, 'week': 3, 'win': 1, 'loss': 0, 'team_points': 90, 'is_playoffs': 0, 'is_consolation': 0},
]

df = pd.DataFrame(data)
print("=== INPUT ===")
print(df)

out = apply_cumulative_fixes(df)

print("\n=== OUTPUT (selected cols) ===")
print(out[['manager','is_playoffs','is_consolation','final_regular_wins','final_wins']])

# Compare manager values pre/post for non-null original managers
orig_mgrs = df['manager'].fillna('<<NA>>').astype(str).tolist()
post_mgrs = out['manager'].fillna('<<NA>>').astype(str).tolist()
print('\nManager values (orig -> post):')
for o,p in zip(orig_mgrs, post_mgrs):
    print(f"  {o} -> {p}")

