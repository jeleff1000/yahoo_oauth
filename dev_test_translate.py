from pathlib import Path
from main import _translate_remote_path

tests = [
    Path('/mount/src/yahoo_oauth/fantasy_football_data/schedule.parquet'),
    Path('/mount/src/yahoo_oauth/fantasy_football_data/matchup.parquet'),
    Path('/usr/src/app/fantasy_football_data/player.parquet'),
]

for p in tests:
    mapped = _translate_remote_path(p)
    print('REMOTE:', p)
    print('MAPPED :', mapped)
    print('MAPPED EXISTS:', mapped.exists())
    print('---')

