import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# -----------------------------
# Subprocess runner
# -----------------------------
def run_script(script_path: str, input_data: str) -> str:
    """
    Runs a Python script that reads from stdin and prints a line like:
    'Excel file with formatted table has been created at <PATH>'
    Returns the parsed <PATH>.
    """
    print(f"Running script: {script_path}")
    process = subprocess.Popen(
        ['python', script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=input_data)

    if process.returncode != 0:
        raise RuntimeError(f"Script failed with error:\n{stderr}")

    # Prefer explicit message
    for line in stdout.splitlines():
        if line.startswith('Excel file with formatted table has been created at'):
            file_path = line.split(' at ', 1)[-1].strip()
            print(f"Generated file: {file_path}")
            return file_path

    # Fallback: any .xlsx-looking path in stdout
    m = re.search(r'([A-Za-z]:\\[^\r\n]+?\.xlsx)', stdout)
    if m:
        guessed = m.group(1)
        print(f"(Guessed) generated file: {guessed}")
        return guessed

    raise RuntimeError("No valid .xlsx path found in producer script output. "
                       "Make sure it prints the Excel path line.")

# -----------------------------
# I/O helpers
# -----------------------------
def save_df_pair(df: pd.DataFrame, base_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save DataFrame to CSV and Parquet next to base_path (without suffix).
    Returns (csv_path, parquet_path). Parquet may require pyarrow/fastparquet.
    """
    csv_path = base_path.with_suffix('.csv')
    parquet_path = base_path.with_suffix('.parquet')

    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet: {parquet_path}")
    except Exception as e:
        print(f"Parquet save failed (install pyarrow or fastparquet?): {e}")
        parquet_path = None

    return csv_path, parquet_path

def read_all_sheets(xlsx_path: str) -> Dict[str, pd.DataFrame]:
    with pd.ExcelFile(xlsx_path) as xls:
        sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}
    for k, v in sheets.items():
        v.columns = [str(c).strip() for c in v.columns]
    return sheets

# -----------------------------
# Cost-bucket logic (ported)
# -----------------------------
def _detect_player_cols(df: pd.DataFrame) -> bool:
    needed = {'playeryear', 'player', 'points', 'week', 'season'}
    return needed.issubset({c.lower() for c in df.columns})

def _detect_draft_cols(df: pd.DataFrame) -> bool:
    needed = {'name full', 'primary position', 'year', 'cost'}
    have = {c.lower() for c in df.columns}
    return needed.issubset(have)

def _get_player_df(sheets: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    for _, df in sheets.items():
        if _detect_player_cols(df):
            cols = {c.lower(): c for c in df.columns}
            return df.rename(columns={
                cols['playeryear']: 'playeryear',
                cols['player']: 'player',
                cols['points']: 'points',
                cols['week']: 'week',
                cols['season']: 'season',
            })
    return None

def _get_draft_df(sheets: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    for _, df in sheets.items():
        if _detect_draft_cols(df):
            cols = {c.lower(): c for c in df.columns}
            out = df.rename(columns={
                cols['name full']: 'Name Full',
                cols['primary position']: 'Primary Position',
                cols['year']: 'Year',
                cols['cost']: 'Cost',
            }).copy()
            if 'is keeper status' in cols:
                out.rename(columns={cols['is keeper status']: 'Is Keeper Status'}, inplace=True)
            else:
                out['Is Keeper Status'] = 0
            return out
    return None

def preprocess_data(draft_history: pd.DataFrame,
                    player_data: pd.DataFrame,
                    start_year: int,
                    end_year: int) -> pd.DataFrame:
    draft_history = draft_history.copy()
    player_data = player_data.copy()

    draft_history['Year'] = pd.to_numeric(draft_history['Year'], errors='coerce').astype('Int64')
    draft_history['Cost'] = pd.to_numeric(draft_history['Cost'], errors='coerce')
    draft_history['Is Keeper Status'] = pd.to_numeric(draft_history['Is Keeper Status'],
                                                      errors='coerce').fillna(0).astype(int)
    player_data['season'] = pd.to_numeric(player_data['season'], errors='coerce').astype('Int64')

    draft_filtered = draft_history[
        draft_history['Year'].between(int(start_year), int(end_year), inclusive='both')
        & (draft_history['Cost'] > 0)
        & (draft_history['Is Keeper Status'] != 1)
    ].copy()

    player_filtered = player_data[
        player_data['season'].between(int(start_year), int(end_year), inclusive='both')
    ].copy()

    draft_filtered['Name Full'] = draft_filtered['Name Full'].astype(str).str.strip()
    player_filtered['player'] = player_filtered['player'].astype(str).str.strip()

    merged = draft_filtered.merge(
        player_filtered[['playeryear', 'player', 'points', 'week', 'season']],
        left_on=['Name Full', 'Year'],
        right_on=['player', 'season'],
        how='left',
        suffixes=('', '_p')
    )

    merged = merged[
        ((merged['Year'] < 2021) & (merged['week'] <= 16))
        | ((merged['Year'] >= 2021) & (merged['week'] <= 17))
    ]
    merged = merged[merged['points'] > 0]

    agg = (merged.groupby(['Name Full', 'Year', 'Primary Position'], dropna=False)
           .agg(Cost=('Cost', 'max'),
                points=('points', 'sum'),
                week=('week', 'nunique'))
           .reset_index())

    agg['PPG'] = agg['points'] / agg['week']
    agg = agg.dropna(subset=['PPG'])
    agg = agg[agg['PPG'] != float('inf')]
    agg = agg[agg['Cost'] > 0]
    return agg

def assign_cost_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(['Primary Position', 'Year', 'Cost', 'Name Full'], kind='mergesort').copy()
    out['Cost Bucket'] = out.groupby(['Primary Position', 'Year']).cumcount().floordiv(3).add(1)
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    # >>> UPDATED PATH HERE <<<
    script_paths = [
        r'C:\Users\joeye\OneDrive\Desktop\kmffl\Adin\Scripts\Sheet 2.0\Scripts 2.0\Draft Data\Draft Data.py'
    ]

    year = input("Select the year to get data for: ").strip()
    week = input("Select the week to get data for: ").strip()
    input_data = f"{year}\n{week}\n"

    generated_paths = []
    for script_path in script_paths:
        gen_path = run_script(script_path, input_data)
        generated_paths.append(gen_path)

    for gen in generated_paths:
        gen_path = Path(gen)
        if not gen_path.exists():
            raise FileNotFoundError(f"Producer script reported Excel path that does not exist: {gen_path}")

        out_base = gen_path.with_suffix('')  # strip .xlsx
        print(f"Reading sheets from: {gen_path}")
        sheets = read_all_sheets(str(gen_path))

        # 1) Save each original sheet (deduped) as CSV + Parquet
        for sheet_name, df in sheets.items():
            df = df.drop_duplicates()
            safe_sheet = re.sub(r'[^A-Za-z0-9_]+', '_', sheet_name.strip()).strip('_')
            base = out_base.parent / f"{out_base.name}__{safe_sheet}"
            save_df_pair(df, base)

        # 2) Build and save cost-bucket aggregate if we can find the right inputs
        draft_df = _get_draft_df(sheets)
        player_df = _get_player_df(sheets)

        if draft_df is not None and player_df is not None:
            try:
                start_year = end_year = int(year) if year.isdigit() else int(pd.Timestamp.today().year)
            except Exception:
                start_year = end_year = int(pd.Timestamp.today().year)

            agg = preprocess_data(draft_df, player_df, start_year, end_year)
            agg_b = assign_cost_buckets(agg)

            base = out_base.parent / f"{out_base.name}__draft_cost_buckets"
            save_df_pair(agg_b, base)
        else:
            missing = []
            if draft_df is None:
                missing.append("draft_history sheet")
            if player_df is None:
                missing.append("player_data sheet")
            print(f"Skipping cost-bucket output; missing: {', '.join(missing)}")

    print("All exports complete (CSV + Parquet).")

if __name__ == "__main__":
    main()
