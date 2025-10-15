import pandas as pd
import pickle
import gzip

def convert_excel_to_pickle_gz(excel_file_path, pickle_file_path):
    try:
        # Read the Excel file
        df_dict = pd.read_excel(excel_file_path, sheet_name=None)

        # Save the dataframe dictionary to a compressed pickle file
        with gzip.open(pickle_file_path, 'wb') as f:
            pickle.dump(df_dict, f)

        print(f"Successfully converted {excel_file_path} to {pickle_file_path}")
    except Exception as e:
        print(f"Failed to convert file: {e}")

if __name__ == "__main__":
    excel_file_path = r'C:\Users\joeye\OneDrive\Desktop\kmffl\Adin\Scripts\Sheet 2.0\Sheet 2.0\Sheet 2.0.xlsx'
    pickle_file_path = r'C:\Users\joeye\OneDrive\Desktop\kmffl\Adin\Scripts\Sheet 2.0\Sheet 2.0\Sheet 2.0.pkl.gz'

    convert_excel_to_pickle_gz(excel_file_path, pickle_file_path)