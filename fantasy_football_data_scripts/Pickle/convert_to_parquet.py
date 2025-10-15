
import re
from pathlib import Path
import pandas as pd

PLACEHOLDER_NULLS = {"-", "", "NA", "N/A", "None", "null", "Null", "NULL", "nan", "NaN"}
FORCE_INT_COLS = {"player_id"}
FORCE_FLOAT_COLS = {"Average Pick"}

def _safe_filename(name: str) -> str:
    # Remove Windows-invalid chars and trim
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip()

def _clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize and type-coerce per column
    for col in df.columns:
        s = df[col]

        # Decode any bytes
        if s.dtype == "object":
            s = s.map(
                lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else x
            )

        # Treat common placeholders as missing
        if s.dtype == "object":
            s = s.replace({v: pd.NA for v in PLACEHOLDER_NULLS})

        # Forced numeric columns
        if col in FORCE_INT_COLS:
            df[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
            continue
        if col in FORCE_FLOAT_COLS:
            df[col] = pd.to_numeric(s, errors="coerce").astype("Float64")
            continue

        # Heuristic: if all non-null values are numeric, choose Int64/Float64
        if s.dtype == "object":
            num = pd.to_numeric(s, errors="coerce")
            if s.notna().any() and num[s.notna()].notna().all():
                is_int = (num.dropna() % 1 == 0).all()
                df[col] = num.astype("Int64") if is_int else num.astype("Float64")
            else:
                # Keep as plain Python strings with object dtype (avoid pandas StringArray)
                s = s.map(lambda x: x if (pd.isna(x) or isinstance(x, str)) else str(x))
                df[col] = s.astype(object)
        else:
            df[col] = s

    return df

def convert_excel_to_parquet(excel_file_path: str, output_dir: str, prefix: str = "Sheet 2.0") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Explicit mapping for tabs the app loads
    required_name_map = {
        "Injury Data": f"{prefix}_Injury Data.parquet",
        "Schedules": f"{prefix}_Schedules.parquet",
        "All Transactions": f"{prefix}_All Transactions.parquet",
        "Draft History": f"{prefix}_Draft History.parquet",
        "Player Data": f"{prefix}_Player Data.parquet",
        "Matchup Data": f"{prefix}_Matchup Data.parquet",
    }

    try:
        sheets = pd.read_excel(excel_file_path, sheet_name=None)
    except Exception as e:
        print(f"Failed to read Excel: {e}")
        return

    written = set()
    for sheet_name, df in sheets.items():
        try:
            if df.shape[1] == 0:
                print(f"Skip sheet '{sheet_name}': no columns.")
                continue

            df = _clean_for_parquet(df)

            # Determine output filename
            if sheet_name in required_name_map:
                filename = required_name_map[sheet_name]
            else:
                safe = _safe_filename(sheet_name)
                filename = f"{prefix}_{safe}.parquet" if safe else f"{prefix}_Sheet.parquet"

            # Handle accidental duplicates
            cand = out_dir / filename
            if cand.name in written:
                stem, suffix, i = cand.stem, cand.suffix, 2
                while (out_dir / f"{stem}_{i}{suffix}").name in written:
                    i += 1
                cand = out_dir / f"{stem}_{i}{suffix}"

            df.to_parquet(cand, engine="pyarrow", index=False)
            written.add(cand.name)
            print(f"Wrote sheet '{sheet_name}' -> {cand}")
        except Exception as e:
            print(f"Failed on sheet '{sheet_name}': {e}")

    print("Done.")

if __name__ == "__main__":
    excel_file_path = r"C:\Users\joeye\OneDrive\Desktop\kmffl\Adin\Scripts\Sheet 2.0\Sheet 2.0\Sheet 2.0.xlsx"
    output_dir = r"C:\Users\joeye\OneDrive\Desktop\KMFFLApp\streamlit_ui"
    convert_excel_to_parquet(excel_file_path, output_dir, prefix="Sheet 2.0")