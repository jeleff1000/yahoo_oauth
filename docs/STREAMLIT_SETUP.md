# Streamlit Cloud Setup Guide

This guide explains how to configure your Streamlit app to trigger GitHub Actions workflows for data imports.

## Why GitHub Actions?

**Streamlit Cloud limitations**:
- ⏰ Times out after 10-15 minutes
- 💾 Limited memory and CPU
- ❌ Can't handle 60-120 minute imports

**GitHub Actions solution**:
- ✅ Runs up to 6 hours
- ✅ Full CPU and memory
- ✅ Runs in background
- ✅ Auto-uploads to MotherDuck

## Setup Steps (5 minutes)

### 1. Create GitHub Personal Access Token

1. Go to: https://github.com/settings/tokens/new
2. **Token name**: `Streamlit Workflow Trigger`
3. **Expiration**: 90 days (or No expiration)
4. **Select scopes**:
   - ✅ `repo` (Full control of private repositories)
   - ✅ `workflow` (Update GitHub Action workflows)
5. Click **"Generate token"**
6. **Copy the token** (starts with `ghp_...`)

### 2. Add Token to Streamlit Secrets

#### Option A: Via Streamlit Cloud Dashboard (Recommended)

1. Go to: https://share.streamlit.io/
2. Click on your `yahoo_oauth` app
3. Click **"Settings"** (⚙️ icon)
4. Click **"Secrets"** in the left sidebar
5. Add this content:

```toml
GITHUB_WORKFLOW_TOKEN = "ghp_YOUR_TOKEN_HERE"
```

6. Click **"Save"**
7. Your app will redeploy automatically (~30 seconds)

#### Option B: Via Local `.streamlit/secrets.toml` (For Local Testing)

Create file: `.streamlit/secrets.toml`

```toml
GITHUB_WORKFLOW_TOKEN = "ghp_YOUR_TOKEN_HERE"
```

**Important**: Add to `.gitignore`:
```
.streamlit/secrets.toml
```

### 3. Add MotherDuck Token (Optional but Recommended)

If you want automatic MotherDuck uploads:

1. Get your token from: https://app.motherduck.com/ → Settings → Access Tokens
2. Add to Streamlit secrets:

```toml
GITHUB_WORKFLOW_TOKEN = "ghp_YOUR_TOKEN_HERE"
MOTHERDUCK_TOKEN = "your_motherduck_token_here"
```

### 4. Configure GitHub Secrets

The workflow needs the MotherDuck token to upload data. Add it to GitHub:

1. Go to: https://github.com/jeleff1000/yahoo_oauth/settings/secrets/actions
2. Click **"New repository secret"**
3. **Name**: `MOTHERDUCK_TOKEN`
4. **Value**: Your MotherDuck token
5. Click **"Add secret"**

## How It Works

### Before (Broken)
```
User clicks "Start Import"
  → Streamlit tries to run import
  → Times out after 15 minutes ❌
```

### After (Working)
```
User clicks "Start Import"
  → Streamlit triggers GitHub Actions
  → GitHub runs full import (60-120 min)
  → Uploads to MotherDuck
  → User checks back later ✅
```

## Using the App

Once configured:

1. **User connects** their Yahoo account
2. **User selects** their league
3. **User clicks** "Start Import via GitHub Actions"
4. **Streamlit shows**:
   - Job ID
   - Estimated time (60-120 min)
   - Link to track progress
5. **User checks back** in 1-2 hours
6. **Data is ready** in MotherDuck!

## Tracking Progress

### View Workflow Run

Click the link shown after clicking "Start Import", or:

1. Go to: https://github.com/jeleff1000/yahoo_oauth/actions
2. Click on the running workflow
3. Watch the live logs

### Check MotherDuck

After ~90 minutes:

1. Go to: https://app.motherduck.com/
2. Look for database: `{league_name}_{season}`
3. Query your data!

## Troubleshooting

### "GitHub workflow token not configured"

**Solution**: Add `GITHUB_WORKFLOW_TOKEN` to Streamlit secrets (see Step 2 above)

### "Failed to start import"

**Check**:
1. GitHub token has `repo` and `workflow` scopes
2. Token hasn't expired
3. Repository name is correct in the trigger script

### Workflow starts but fails

**Check**:
1. GitHub Actions logs: https://github.com/jeleff1000/yahoo_oauth/actions
2. Verify `MOTHERDUCK_TOKEN` is set in GitHub secrets
3. Check Yahoo OAuth credentials are valid

### Import times out on Streamlit

This is expected! That's why we use GitHub Actions. The timeout message means you need to:
1. Add `GITHUB_WORKFLOW_TOKEN` to secrets
2. Use the GitHub Actions workflow instead

## Cost

### Free Tier Limits
- **Streamlit Cloud**: Unlimited apps
- **GitHub Actions**: 2,000 minutes/month (private repos)
  - Each import: ~90-120 minutes
  - ~15-20 imports/month free
- **MotherDuck**: 10GB storage free
  - Each league: ~10-100MB

### If You Exceed Free Tier

**GitHub Actions**:
- Public repo = unlimited free
- Or pay $0.008/minute for overages

**MotherDuck**:
- $0/GB/month for first 10GB
- Then usage-based pricing

## Security Notes

1. **GitHub Token**: Has access to trigger workflows
   - Keep it secret
   - Rotate every 90 days
   - Use minimal scopes (`repo`, `workflow`)

2. **OAuth Credentials**: Sent to GitHub Actions
   - Stored temporarily during workflow
   - Deleted after completion
   - Never committed to repo

3. **MotherDuck Token**: Stored in GitHub Secrets
   - Encrypted at rest
   - Only accessible to workflows
   - Can be rotated anytime

## Next Steps

After setup:

1. ✅ Test the import with one league
2. ✅ Verify data appears in MotherDuck
3. ✅ Share the app URL with users
4. ✅ Monitor GitHub Actions usage

## Support

Having issues? Check:

1. Streamlit app logs (click "Manage app" → "Logs")
2. GitHub Actions logs (Actions tab)
3. MotherDuck console (app.motherduck.com)
4. [Full documentation](./docs/WORKFLOW_SETUP.md)
