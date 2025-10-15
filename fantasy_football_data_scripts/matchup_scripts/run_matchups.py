import subprocess
import os

# Change working directory to the script's directory
os.chdir(os.path.dirname(__file__))

scripts = [
    'cumulative_stats.py',
    'opponent_expected_record.py',
    'expected_record_import.py',
    'playoff_odds_import.py'
]

for script in scripts:
    result = subprocess.run(['python', script])
    if result.returncode != 0:
        print(f"Error running {script}")
        break