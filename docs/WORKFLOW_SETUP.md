# On-Demand League Import Workflow Setup

This document explains how the on-demand fantasy football league import system works and how to set it up.

## Overview

The workflow allows users to create their own fantasy football analytics website through a simple Streamlit interface:

```
User visits Streamlit site
    ↓
User provides Yahoo OAuth credentials and selects their league
    ↓
Streamlit triggers GitHub Actions workflow (avoids overrunning Streamlit resources)
    ↓
GitHub Actions runs initial_import_v2.py for that specific league (60-120 min)
    ↓
All parquet files (player.parquet, matchup.parquet, etc.) are created
    ↓
Files are uploaded to MotherDuck with unique database name (e.g., "users_league_2024")
    ↓
User gets their own custom analytics site based on KMFFLApp template
```

## Files Created by initial_import_v2.py

The import process creates 6 canonical parquet files in `fantasy_football_data/`:

1. **player.parquet** - Weekly player stats merged with NFL data
2. **matchup.parquet** - Matchup results with advanced analytics
3. **draft.parquet** - Draft results with keeper tracking
4. **transactions.parquet** - Trades, add/drops, FAAB
5. **schedule_data_all_years.parquet** - Full schedule structure
6. **players_by_year.parquet** - Season-level aggregations

All files are automatically uploaded to MotherDuck with the database name format: `{league_name}_{season}`

## GitHub Workflow

**File**: `.github/workflows/league_import_worker.yml`

### Trigger Methods

#### From Streamlit (Recommended)

```python
from streamlit_helpers.trigger_import_workflow import trigger_import_workflow

result = trigger_import_workflow(
    league_data={
        "league_id": "449.l.198278",
        "league_name": "My League",
        "season": 2024,
        "start_year": 2020,
        "oauth_token": {
            "access_token": "...",
            "refresh_token": "...",
            "token_type": "bearer",
            "expires_in": 3600
        },
        "num_teams": 10,
        "playoff_teams": 6,
        "regular_season_weeks": 14
    },
    github_token=os.getenv("GITHUB_TOKEN")
)

if result['success']:
    st.success(f"Import started! Job ID: {result['user_id']}")
    st.info(f"Estimated time: {result['estimated_time']}")
    st.markdown(f"[Track progress]({result['workflow_run_url']})")
```

#### Manual Trigger (Testing)

Via GitHub UI:
1. Go to: https://github.com/jeleff1000/yahoo_oauth/actions/workflows/league_import_worker.yml
2. Click "Run workflow"
3. Provide inputs:
   - `league_data`: JSON with league configuration
   - `user_id`: Unique identifier for this job

Via GitHub CLI:
```bash
gh workflow run league_import_worker.yml \
  -f league_data='{"league_id":"449.l.198278","league_name":"Test",...}' \
  -f user_id="test123"
```

## GitHub Secrets Required

Add these secrets to your GitHub repository:

### 1. MOTHERDUCK_TOKEN
Your MotherDuck authentication token

**Get it from**: https://app.motherduck.com/ → Settings → Access Tokens

```bash
gh secret set MOTHERDUCK_TOKEN
# Paste your token when prompted
```

### 2. YAHOO_OAUTH_JSON (if using static credentials)
If you want to use the same OAuth credentials for all imports

```bash
gh secret set YAHOO_OAUTH_JSON
# Paste your OAuth JSON when prompted
```

**Note**: For user-specific imports, OAuth credentials are passed via `league_data.oauth_token` instead.

### 3. GITHUB_TOKEN (for Streamlit)
Personal Access Token for Streamlit to trigger workflows

**Create it**:
1. Go to: https://github.com/settings/tokens/new
2. Select scopes: `repo`, `workflow`
3. Generate token
4. Save it to your Streamlit secrets or environment

## Workflow Steps

The workflow performs these steps:

1. **Parse League Data** (30 sec)
   - Extracts league info from input JSON
   - Generates unique user ID
   - Creates OAuth credentials file
   - Creates sanitized MotherDuck database name

2. **Create League Context** (10 sec)
   - Builds `league_context.json` for initial_import_v2.py
   - Configures data directories and settings

3. **Run Initial Import** (60-120 min)
   - Executes `fantasy_football_data_scripts/initial_import_v2.py`
   - Fetches Yahoo, NFL, draft, transaction data
   - Merges and transforms all data
   - Creates 6 canonical parquet files

4. **Verify Parquet Files** (10 sec)
   - Confirms all expected files were created
   - Counts rows in each file

5. **Upload to MotherDuck** (5-10 min)
   - Uses `fantasy_football_data/motherduck_upload.py`
   - Creates database: `{league_name}_{season}`
   - Uploads all parquet files as tables
   - Creates analysis views

6. **Create Deployment Manifest** (5 sec)
   - Generates JSON with database info
   - Provides details needed to create user's site

7. **Upload Artifacts** (1-2 min)
   - Saves logs, parquet files, manifests
   - Retained for 30 days

## Streamlit Integration

### Example Streamlit Page

```python
import streamlit as st
from streamlit_helpers.trigger_import_workflow import trigger_import_workflow
import os

st.title("Create Your Fantasy Football Analytics Site")

# Step 1: User provides Yahoo OAuth
st.header("1. Connect to Yahoo")
# ... Yahoo OAuth flow ...

# Step 2: User selects league
st.header("2. Select Your League")
selected_league = st.selectbox("Choose your league", leagues)

# Step 3: Configure import
st.header("3. Configure Import")
start_year = st.number_input("Start Year", min_value=2000, value=2020)
end_year = st.number_input("End Year", min_value=2000, value=2024)

# Step 4: Trigger import
if st.button("Create My Analytics Site"):
    with st.spinner("Starting import..."):
        result = trigger_import_workflow(
            league_data={
                "league_id": selected_league['league_id'],
                "league_name": selected_league['name'],
                "season": end_year,
                "start_year": start_year,
                "oauth_token": st.session_state.oauth_token,
                # ... other config ...
            },
            github_token=os.getenv("GITHUB_TOKEN")
        )

        if result['success']:
            st.success("✅ Import Started!")
            st.session_state.user_id = result['user_id']
            st.info(f"""
            Your data is being processed. This takes 1-2 hours.

            **Job ID**: {result['user_id']}
            **Estimated Time**: {result['estimated_time']}

            [Track Progress]({result['workflow_run_url']})
            """)

            # Show progress tracker
            st_autorefresh(interval=30000)  # Refresh every 30 sec
        else:
            st.error(f"Failed to start import: {result.get('error')}")
```

## Monitoring Progress

### Check Workflow Status

```python
from streamlit_helpers.trigger_import_workflow import check_import_status

status = check_import_status(
    user_id=st.session_state.user_id,
    github_token=os.getenv("GITHUB_TOKEN")
)

if status['success']:
    st.write(f"Status: {status['status']}")
    st.write(f"Conclusion: {status['conclusion']}")
```

### View Artifacts

After the workflow completes, download artifacts:

```bash
gh run list --workflow=league_import_worker.yml
gh run view <run-id>
gh run download <run-id>
```

Artifacts include:
- `import.log` - Full import logs
- `motherduck.log` - MotherDuck upload logs
- `deployment_manifest.json` - Deployment configuration
- `*.parquet` files - All generated data files

## MotherDuck Database Structure

After import, the MotherDuck database contains:

### Tables (from parquet files)
- `player` - Player stats by week
- `matchup` - Weekly matchups
- `draft` - Draft results
- `transactions` - All transactions
- `schedule_data_all_years` - Schedule structure
- `players_by_year` - Season aggregations

### Views (auto-created)
- `season_summary` - Manager records by season

### Accessing the Data

```python
import duckdb

# Connect to user's database
con = duckdb.connect(f"md:my_league_2024?motherduck_token={token}")

# Query the data
df = con.execute("SELECT * FROM matchup WHERE season = 2024").df()
```

## Next Steps After Import

Once the workflow completes:

1. **Database is ready** in MotherDuck
   - Name: `{league_name}_{season}`
   - All tables populated

2. **Create user's analytics site**
   - Clone KMFFLApp template
   - Configure to use user's MotherDuck database
   - Deploy to Streamlit Cloud or other hosting

3. **Notify user**
   - Send email with site URL
   - Provide database credentials
   - Share usage instructions

## Troubleshooting

### Workflow fails immediately
- Check GitHub secrets are set correctly
- Verify `league_data` JSON is valid
- Check OAuth credentials are not expired

### Import fails during data collection
- Yahoo API rate limits (workflow will retry)
- Invalid league ID
- OAuth token expired

### MotherDuck upload fails
- Check MOTHERDUCK_TOKEN is valid
- Database name may contain invalid characters
- Network issues (workflow will retry)

### No parquet files created
- Check import logs in artifacts
- Verify league has data for specified seasons
- May be missing required fields in league context

## Cost Considerations

### GitHub Actions
- Free tier: 2,000 minutes/month for private repos
- Each import: ~90-120 minutes
- ~15-20 imports/month on free tier

### MotherDuck
- Free tier: 10GB storage
- Each league database: ~10-100MB depending on history
- Monitor usage in MotherDuck dashboard

## Security Notes

1. **OAuth Credentials**: Passed securely via workflow inputs, never committed to repo
2. **Secrets**: Use GitHub Secrets for sensitive tokens
3. **User Isolation**: Each user gets separate MotherDuck database
4. **Artifact Retention**: Set to 30 days, contains sensitive data
5. **Rate Limiting**: Built into initial_import_v2.py (4 req/sec)

## Support

If you encounter issues:

1. Check workflow run logs in GitHub Actions
2. Download artifacts for detailed logs
3. Review MotherDuck console for upload issues
4. Check this repo's Issues tab for known problems
