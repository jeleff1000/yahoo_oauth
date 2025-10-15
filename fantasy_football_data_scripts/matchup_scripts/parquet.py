# python
import pandas as pd
import os

def file_to_parquet():
    file_path = input("Enter the path to your file: ").strip()
    if file_path.startswith('"') and file_path.endswith('"'):
        file_path = file_path[1:-1]
    if not os.path.isfile(file_path):
        print("File not found:", file_path)
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file type:", ext)
        return

    parquet_path = os.path.splitext(file_path)[0] + ".parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        print("Saved Parquet:", parquet_path)
    except Exception as e:
        print("Error saving Parquet:", e)

if __name__ == "__main__":
    file_to_parquet()