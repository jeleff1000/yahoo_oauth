import re
import os
log_path = 'schedule_run_log.txt'
if not os.path.exists(log_path):
    print('Log not found:', log_path)
    raise SystemExit(1)
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()
paths = re.findall(r'([A-Za-z]:\\[^\n\r]+?\.parquet)', text)
if not paths:
    print('No parquet paths found in log')
else:
    for p in paths:
        exists = os.path.exists(p)
        print(p, 'EXISTS' if exists else 'MISSING')

