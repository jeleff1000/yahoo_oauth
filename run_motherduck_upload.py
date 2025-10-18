import os
import runpy

# Set required environment variables (user-provided token; not echoed)
os.environ['MOTHERDUCK_TOKEN'] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImpvZXllbGVmZkBnbWFpbC5jb20iLCJzZXNzaW9uIjoiam9leWVsZWZmLmdtYWlsLmNvbSIsInBhdCI6IlJtRTlJMjZ0dVI5VUlzLXBQVDJMdGh4SmpXMlV2cktEX0V3NXF5S19TTkUiLCJ1c2VySWQiOiJkMzAzMjVjNC0zYjE5LTRmMWYtYTAxZi1iNGVhMTA1ZWI3OGEiLCJpc3MiOiJtZF9wYXQiLCJyZWFkT25seSI6ZmFsc2UsInRva2VuVHlwZSI6InJlYWRfd3JpdGUiLCJpYXQiOjE3NjA2MjkwNDN9.dZZ17fPNSLETORsJmaWaCVx2Zwh6zEFGexbx_1cw-aE"
# Provide a league name (used to name the MotherDuck database)
os.environ['LEAGUE_NAME'] = "You Are A Pirate"

# Run the upload script
runpy.run_path("fantasy_football_data/motherduck_upload.py", run_name="__main__")

