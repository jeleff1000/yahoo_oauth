import pandas as pd
import os

NEW_HEADERS = [
    "year", "pick", "round", "team_key", "manager", "yahoo_player_id", "cost", "player",
    "yahoo_position", "avg_pick", "avg_round", "avg_cost", "percent_drafted",
    "is_keeper_status", "is_keeper_cost", "savings", "player_year", "manager_year", "nfl_team",
    "cost_bucket"
]

managerNameToStatName = {
    "--hidden--": "Ilan"
}

def _norm(s):
    return str(s or "").strip()

def assign_cost_buckets(df):
    df['cost_bucket'] = pd.NA
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    mask = (
        df['cost'].notna()
        & (df['cost'] > 0)
        & df['yahoo_position'].notna()
        & (df['yahoo_position'] != "")
    )
    if not mask.any():
        return df
    df_to_bucket = df[mask].copy()
    df_to_bucket = df_to_bucket.sort_values(['yahoo_position', 'year', 'cost'], ascending=[True, True, True])
    df_to_bucket['cost_bucket'] = (df_to_bucket.groupby(['yahoo_position', 'year'], dropna=False).cumcount() // 3) + 1
    df.loc[mask, 'cost_bucket'] = df_to_bucket['cost_bucket']
    return df

def merge_csvs(draft_data_file, draft_analysis_file, merged_file):
    df1 = pd.read_csv(draft_data_file, dtype=str)
    df2 = pd.read_csv(draft_analysis_file, dtype=str)

    # keep rows that actually have a yahoo_player_id
    df1 = df1[df1['yahoo_player_id'].notna() & (df1['yahoo_player_id'] != "")]
    df2 = df2[df2['yahoo_player_id'].notna() & (df2['yahoo_player_id'] != "")]

    # ensure all expected columns exist
    for col in NEW_HEADERS:
        if col not in df1.columns:
            df1[col] = pd.NA
        if col not in df2.columns:
            df2[col] = pd.NA

    # merge on yahoo_player_id + year
    merged = pd.merge(df1, df2, how='outer', on=['yahoo_player_id', 'year'], suffixes=('', '_y'))

    # prefer left; fill from right only when left empty
    for col in NEW_HEADERS:
        if col in ['yahoo_player_id', 'year', 'cost_bucket']:
            continue
        cy = f"{col}_y"
        if cy in merged.columns:
            left = merged[col]
            right = merged[cy]
            mask_empty = (
                left.isna()
                | (left == "")
                | (left.astype(str).str.strip().str.lower().isin(["nan", "none"]))
            )
            merged.loc[mask_empty, col] = right
            merged.drop(columns=[cy], inplace=True, errors='ignore')

    # normalize manager names
    merged['manager'] = merged['manager'].apply(lambda x: managerNameToStatName.get(_norm(x), x))

    # composite keys (space-free)
    if 'player' in merged.columns and 'year' in merged.columns:
        merged['player_year'] = merged['player'].str.replace(" ", "", regex=False) + merged['year'].astype(str)
    if 'manager' in merged.columns and 'year' in merged.columns:
        merged['manager_year'] = merged['manager'].str.replace(" ", "", regex=False) + merged['year'].astype(str)

    print("Calculating cost buckets...")
    merged = assign_cost_buckets(merged)
    print(f"Cost buckets calculated. Non-null buckets: {merged['cost_bucket'].notna().sum()}")

    # final typing/order
    for col in NEW_HEADERS:
        if col == 'cost_bucket':
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')
        else:
            merged[col] = merged[col].astype("string").fillna("")

    merged = merged[NEW_HEADERS]
    merged.to_csv(merged_file, index=False)
    print(f"CSV saved: {merged_file}")

    try:
        parquet_file = merged_file.replace('.csv', '.parquet')
        merged.to_parquet(parquet_file, index=False)
        print(f"Parquet saved: {parquet_file}")
    except Exception as e:
        print(f"Parquet save failed (install pyarrow/fastparquet?): {e}")