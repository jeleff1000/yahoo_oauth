import os
matches = []
root = r"C:\Users\joeye\OneDrive\Desktop\yahoo_oauth"
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if f.endswith('.parquet'):
            matches.append(os.path.join(dirpath, f))
print('FOUND', len(matches), 'parquet files')
for p in matches:
    print(p)

