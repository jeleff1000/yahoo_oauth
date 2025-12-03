# Quick Setup Checklist

Get your on-demand fantasy football import workflow running in 5 minutes.

## Prerequisites

- [ ] GitHub repository with this code
- [ ] MotherDuck account (free tier works)
- [ ] GitHub account with Actions enabled

## 1. Add GitHub Secrets (2 minutes)

Go to: `https://github.com/YOUR_USERNAME/yahoo_oauth/settings/secrets/actions`

Add these secrets:

```bash
# Required for MotherDuck uploads
MOTHERDUCK_TOKEN=your_motherduck_token_here

# Optional: For triggering workflow from Streamlit
GITHUB_WORKFLOW_TOKEN=your_github_pat_here
```

**Get MotherDuck Token**:
1. Visit https://app.motherduck.com/
2. Click Settings → Access Tokens
3. Create new token
4. Copy and paste into GitHub secret

**Get GitHub Token** (for Streamlit integration):
1. Visit https://github.com/settings/tokens/new
2. Name: "Workflow Trigger"
3. Select scopes: `repo`, `workflow`
4. Generate and copy token
5. Save to GitHub Secrets as `GITHUB_WORKFLOW_TOKEN`

## 2. Test the Workflow (1 minute)

### Option A: Manual Test via GitHub UI

1. Go to: `https://github.com/YOUR_USERNAME/yahoo_oauth/actions/workflows/league_import_worker.yml`
2. Click "Run workflow"
3. Fill in test data:
   ```json
   league_data: {
     "league_id": "449.l.198278",
     "league_name": "Test League",
     "season": 2024,
     "start_year": 2024,
     "oauth_token": {
       "access_token": "YOUR_YAHOO_ACCESS_TOKEN",
       "refresh_token": "YOUR_YAHOO_REFRESH_TOKEN",
       "token_type": "bearer"
     }
   }

   user_id: test_run_001
   ```
4. Click "Run workflow"
5. Watch it run (takes 60-120 min)

### Option B: Test via GitHub CLI

```bash
# Create test data file
cat > test_league.json <<EOF
{
  "league_id": "449.l.198278",
  "league_name": "Test League",
  "season": 2024,
  "start_year": 2024,
  "oauth_token": {
    "access_token": "YOUR_ACCESS_TOKEN",
    "refresh_token": "YOUR_REFRESH_TOKEN",
    "token_type": "bearer"
  }
}
EOF

# Trigger workflow
gh workflow run league_import_worker.yml \
  -f league_data="$(cat test_league.json)" \
  -f user_id="test_$(date +%s)"

# Watch progress
gh run watch
```

## 3. Integrate with Streamlit (2 minutes)

Add to your Streamlit app:

```python
# streamlit_app.py
import streamlit as st
import os
from streamlit_helpers.trigger_import_workflow import trigger_import_workflow

st.title("Create Your Fantasy Football Site")

# ... collect user's OAuth and league selection ...

if st.button("Start Import"):
    result = trigger_import_workflow(
        league_data={
            "league_id": selected_league_id,
            "league_name": league_name,
            "season": 2024,
            "oauth_token": oauth_credentials,
            # ... other settings ...
        },
        github_token=os.getenv("GITHUB_WORKFLOW_TOKEN")
    )

    if result['success']:
        st.success(f"Import started! Job ID: {result['user_id']}")
        st.info(f"Check progress: {result['workflow_run_url']}")
```

Add to Streamlit secrets (`.streamlit/secrets.toml`):

```toml
GITHUB_WORKFLOW_TOKEN = "ghp_your_token_here"
```

## 4. Monitor Results

### Check Workflow Status

```bash
# List recent runs
gh run list --workflow=league_import_worker.yml

# View specific run
gh run view RUN_ID

# Download artifacts
gh run download RUN_ID
```

### Check MotherDuck

1. Visit https://app.motherduck.com/
2. Look for database: `{league_name}_{season}`
3. Query tables:
   ```sql
   SELECT COUNT(*) FROM player;
   SELECT COUNT(*) FROM matchup;
   SELECT * FROM season_summary;
   ```

## 5. Verify Success

After workflow completes (~90 min):

- [ ] Workflow shows green checkmark
- [ ] Artifacts uploaded (see Actions tab)
- [ ] MotherDuck database created
- [ ] Tables populated with data
- [ ] `deployment_manifest.json` created

## What Gets Created

### Files in GitHub Actions Artifacts
```
league-import-{user_id}/
├── import.log                    # Full import logs
├── motherduck.log                # Upload logs
├── deployment_manifest.json      # Deployment config
├── league_context.json           # League configuration
├── job_status/{user_id}.json    # Final status
└── fantasy_football_data/
    ├── player.parquet
    ├── matchup.parquet
    ├── draft.parquet
    ├── transactions.parquet
    ├── schedule_data_all_years.parquet
    └── players_by_year.parquet
```

### MotherDuck Database
```
Database: {league_name}_{season}
Tables:
  - player (100K+ rows)
  - matchup (1K+ rows)
  - draft (100+ rows)
  - transactions (500+ rows)
  - schedule_data_all_years
  - players_by_year
Views:
  - season_summary
```

## Troubleshooting

### "MOTHERDUCK_TOKEN not set"
- Add the secret in GitHub repo settings
- Get token from https://app.motherduck.com/

### "Failed to parse league data"
- Verify JSON is valid
- Check all required fields present
- Ensure OAuth token is not expired

### "No parquet files created"
- Check import.log in artifacts
- Verify league_id is correct
- Ensure Yahoo OAuth is valid

### Workflow times out
- Default timeout is 120 minutes
- Large leagues (10+ years) may need more time
- Consider reducing `start_year`

## Next Steps

Once workflow succeeds:

1. **Access your data** in MotherDuck
2. **Clone analytics_app** template
3. **Configure** to use your database
4. **Deploy** your custom analytics site

See [WORKFLOW_SETUP.md](./WORKFLOW_SETUP.md) for detailed documentation.

## Cost Summary

### Free Tier Limits
- **GitHub Actions**: 2,000 min/month (private repos)
  - ~15-20 imports/month
- **MotherDuck**: 10GB storage
  - ~100-200 league databases

### Per Import
- **Time**: 60-120 minutes
- **Storage**: 10-100 MB per league
- **Cost**: $0 (within free tiers)

## Support

Issues? Check:
1. Workflow logs in GitHub Actions
2. Artifacts for detailed logs
3. [GitHub Issues](https://github.com/YOUR_USERNAME/yahoo_oauth/issues)
4. [Full Documentation](./WORKFLOW_SETUP.md)
