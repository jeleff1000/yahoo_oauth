#!/usr/bin/env python3
"""
Fantasy Football Import Manager - Streamlit Interface

This lightweight Streamlit app handles:
1. Yahoo OAuth authentication
2. League selection
3. Job submission to GitHub Actions
4. Status tracking and monitoring
5. MotherDuck database connections

The heavy data processing is offloaded to GitHub Actions workers.
"""

import os
import streamlit as st
import requests
import json
import base64
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import urllib.parse

# =========================
# Configuration
# =========================
# Yahoo OAuth Config
CLIENT_ID = st.secrets.get("YAHOO_CLIENT_ID")
CLIENT_SECRET = st.secrets.get("YAHOO_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("REDIRECT_URI", "https://leaguehistory.streamlit.app")

# GitHub Config
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")  # Personal access token with workflow dispatch permissions
GITHUB_OWNER = st.secrets.get("GITHUB_OWNER", "your-username")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "fantasy-football-etl")
GITHUB_WORKFLOW = "fantasy_import_worker.yml"

# MotherDuck Config
MOTHERDUCK_TOKEN = st.secrets.get("MOTHERDUCK_TOKEN", "")

# Yahoo OAuth URLs
AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
YAHOO_API_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"

# Job storage (in production, use a database)
JOBS_FILE = Path("jobs.json")


# =========================
# CSS Styling
# =========================
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .hero {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }

    .hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .status-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .status-running { color: #3182ce; }
    .status-complete { color: #48bb78; }
    .status-failed { color: #f56565; }
    .status-queued { color: #ed8936; }

    .job-id {
        font-family: 'Courier New', monospace;
        background: #f7fafc;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
    }

    .info-box {
        background: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }

    .success-box {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }

    .warning-box {
        background: #fffaf0;
        border-left: 4px solid #ed8936;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================
# Job Management
# =========================
def load_jobs() -> Dict:
    """Load job history from file"""
    if JOBS_FILE.exists():
        with open(JOBS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_jobs(jobs: Dict):
    """Save job history to file"""
    with open(JOBS_FILE, 'w') as f:
        json.dump(jobs, f, indent=2, default=str)


def create_job(league_info: Dict, oauth_token: Dict) -> Dict:
    """Create a new import job"""
    job_id = str(uuid.uuid4())[:8]

    job = {
        "job_id": job_id,
        "league_key": league_info.get("league_key"),
        "league_name": league_info.get("name"),
        "season": league_info.get("season"),
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database_name": generate_database_name(league_info)
    }

    # Store job locally
    jobs = load_jobs()
    jobs[job_id] = job
    save_jobs(jobs)

    return job


def generate_database_name(league_info: Dict) -> str:
    """Generate a clean database name from league info"""
    name = f"{league_info.get('name', 'league')}_{league_info.get('season', '2024')}"
    # Clean up special characters
    name = name.lower().replace(' ', '_').replace('-', '_')
    return ''.join(c if c.isalnum() or c == '_' else '' for c in name)


# =========================
# GitHub Actions Integration
# =========================
def trigger_github_workflow(job: Dict, oauth_token: Dict) -> bool:
    """Trigger GitHub Actions workflow for import job"""

    if not GITHUB_TOKEN:
        st.error("GitHub token not configured. Please add GITHUB_TOKEN to secrets.")
        return False

    # Prepare job data
    job_data = {
        "job_id": job["job_id"],
        "league_key": job["league_key"],
        "league_name": job["league_name"],
        "season": job["season"],
        "database_name": job["database_name"],
        "oauth_token": oauth_token
    }

    # GitHub API endpoint
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW}/dispatches"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "ref": "main",  # or your default branch
        "inputs": {
            "job_id": job["job_id"],
            "job_data": json.dumps(job_data)
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 204:
            st.success(f"✅ Job {job['job_id']} submitted successfully!")
            return True
        else:
            st.error(f"Failed to trigger workflow: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        st.error(f"Error triggering workflow: {e}")
        return False


def check_github_job_status(job_id: str) -> Optional[Dict]:
    """Check job status from GitHub artifacts"""

    if not GITHUB_TOKEN:
        return None

    # Get recent workflow runs
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }

    try:
        response = requests.get(url, headers=headers, params={"per_page": 10})
        if response.status_code != 200:
            return None

        runs = response.json().get("workflow_runs", [])

        # Find run for this job
        for run in runs:
            # Check if this run is for our job (you might need to adjust this logic)
            if job_id in str(run.get("name", "")):
                status = run.get("status")
                conclusion = run.get("conclusion")

                if status == "completed":
                    return {
                        "status": "complete" if conclusion == "success" else "failed",
                        "completed_at": run.get("updated_at")
                    }
                elif status == "in_progress":
                    return {"status": "running"}

    except Exception:
        pass

    return None


# =========================
# Yahoo OAuth Functions
# =========================
def get_auth_url() -> str:
    """Generate Yahoo OAuth authorization URL"""
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "language": "en-us"
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_tokens(code: str) -> Dict:
    """Exchange authorization code for access tokens"""
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI
    }

    response = requests.post(TOKEN_URL, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Token exchange failed: {response.status_code}")


def refresh_token(refresh_token: str) -> Dict:
    """Refresh an expired access token"""
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }

    response = requests.post(TOKEN_URL, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Token refresh failed: {response.status_code}")


def get_user_leagues(token_data: Dict) -> List[Dict]:
    """Fetch user's fantasy leagues from Yahoo"""
    access_token = token_data.get("access_token")

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Get user's games
    url = f"{YAHOO_API_BASE}/users;use_login=1/games;game_keys=nfl/leagues"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch leagues: {response.status_code}")

    # Parse XML response (simplified - in production use proper XML parser)
    leagues = []
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)

    # Extract league information (simplified)
    for league in root.findall(".//{http://fantasysports.yahooapis.com/fantasy/v2/base.rng}league"):
        league_key = league.find("{http://fantasysports.yahooapis.com/fantasy/v2/base.rng}league_key")
        name = league.find("{http://fantasysports.yahooapis.com/fantasy/v2/base.rng}name")
        season = league.find("{http://fantasysports.yahooapis.com/fantasy/v2/base.rng}season")

        if league_key is not None and name is not None:
            leagues.append({
                "league_key": league_key.text,
                "name": name.text,
                "season": season.text if season is not None else "2024"
            })

    return leagues


# =========================
# UI Components
# =========================
def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🏈 Fantasy Football Import Manager</h1>
        <p>Import your league data to MotherDuck for analysis</p>
    </div>
    """, unsafe_allow_html=True)


def render_job_status(job: Dict):
    """Render a job status card"""
    status_icon = {
        "queued": "⏳",
        "running": "🔄",
        "complete": "✅",
        "failed": "❌"
    }.get(job["status"], "❓")

    status_class = f"status-{job['status']}"

    st.markdown(f"""
    <div class="status-card">
        <h4>{status_icon} {job['league_name']} ({job['season']})</h4>
        <p>Job ID: <span class="job-id">{job['job_id']}</span></p>
        <p>Status: <span class="{status_class}">{job['status'].title()}</span></p>
        <p>Database: <code>{job.get('database_name', 'N/A')}</code></p>
        <p>Created: {job['created_at']}</p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# Main Application
# =========================
def main():
    st.set_page_config(
        page_title="Fantasy Football Import Manager",
        page_icon="🏈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_custom_css()
    render_hero()

    # Check configuration
    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("⚠️ Yahoo OAuth credentials not configured. Add to Streamlit secrets.")
        return

    if not GITHUB_TOKEN:
        st.warning("⚠️ GitHub token not configured. Add GITHUB_TOKEN to secrets for automated imports.")

    # Initialize session state
    if "token_data" not in st.session_state:
        st.session_state.token_data = None
    if "leagues" not in st.session_state:
        st.session_state.leagues = []

    # Sidebar for navigation
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.radio("Select Page", ["🏠 Home", "📥 Import", "📈 Status", "🗄️ Databases"])

    # Handle OAuth callback
    query_params = st.query_params
    if "code" in query_params:
        with st.spinner("Authenticating with Yahoo..."):
            try:
                token_data = exchange_code_for_tokens(query_params["code"])
                st.session_state.token_data = token_data
                st.query_params.clear()
                st.success("✅ Successfully authenticated with Yahoo!")
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")

    # Page routing
    if page == "🏠 Home":
        render_home_page()
    elif page == "📥 Import":
        render_import_page()
    elif page == "📈 Status":
        render_status_page()
    elif page == "🗄️ Databases":
        render_databases_page()


def render_home_page():
    """Render the home/welcome page"""
    st.header("Welcome to Fantasy Football Import Manager")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔐 Secure Authentication
        Connect your Yahoo Fantasy account securely with OAuth 2.0
        """)

    with col2:
        st.markdown("""
        ### ⚡ Fast Processing
        Imports run in GitHub Actions - no timeout issues!
        """)

    with col3:
        st.markdown("""
        ### 📊 MotherDuck Integration
        Query your data instantly with DuckDB in the cloud
        """)

    st.markdown("""
    <div class="info-box">
    <h4>🚀 How It Works</h4>
    <ol>
        <li>Authenticate with your Yahoo account</li>
        <li>Select the league you want to import</li>
        <li>Submit the import job to GitHub Actions</li>
        <li>Monitor the job status</li>
        <li>Query your data in MotherDuck!</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    # Show recent jobs
    st.header("📋 Recent Import Jobs")
    jobs = load_jobs()
    if jobs:
        recent_jobs = sorted(jobs.values(), key=lambda x: x["created_at"], reverse=True)[:5]
        for job in recent_jobs:
            render_job_status(job)
    else:
        st.info("No import jobs yet. Go to the Import page to get started!")


def render_import_page():
    """Render the import/league selection page"""
    st.header("📥 Import League Data")

    # Check authentication
    if not st.session_state.token_data:
        st.markdown("""
        <div class="warning-box">
        <h4>🔐 Authentication Required</h4>
        <p>You need to authenticate with Yahoo to import your league data.</p>
        </div>
        """, unsafe_allow_html=True)

        auth_url = get_auth_url()
        st.markdown(f"""
        <a href="{auth_url}" target="_self">
            <button style="background:#5f40d4; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer; font-size:16px;">
                🔐 Connect with Yahoo
            </button>
        </a>
        """, unsafe_allow_html=True)
        return

    # Fetch leagues
    if st.button("🔄 Refresh Leagues"):
        with st.spinner("Fetching your leagues..."):
            try:
                leagues = get_user_leagues(st.session_state.token_data)
                st.session_state.leagues = leagues
                st.success(f"Found {len(leagues)} league(s)")
            except Exception as e:
                st.error(f"Failed to fetch leagues: {e}")
                # Try refreshing token
                if "refresh_token" in st.session_state.token_data:
                    try:
                        new_token = refresh_token(st.session_state.token_data["refresh_token"])
                        st.session_state.token_data = new_token
                        leagues = get_user_leagues(new_token)
                        st.session_state.leagues = leagues
                        st.success(f"Found {len(leagues)} league(s) after token refresh")
                    except Exception as e2:
                        st.error(f"Token refresh failed: {e2}")
                        st.session_state.token_data = None

    # League selection
    if st.session_state.leagues:
        st.subheader("Select a League to Import")

        for league in st.session_state.leagues:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 2])

            with col1:
                st.write(f"**{league['name']}**")
            with col2:
                st.write(f"Season: {league['season']}")
            with col3:
                st.write(f"Key: {league['league_key']}")
            with col4:
                if st.button(f"Import", key=f"import_{league['league_key']}"):
                    # Create import job
                    job = create_job(league, st.session_state.token_data)

                    # Trigger GitHub workflow
                    if trigger_github_workflow(job, st.session_state.token_data):
                        st.balloons()
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>✅ Import Job Created!</h4>
                        <p>Job ID: <span class="job-id">{job['job_id']}</span></p>
                        <p>Your import has been queued and will be processed by GitHub Actions.</p>
                        <p>Check the Status page to monitor progress.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Update job status to failed
                        jobs = load_jobs()
                        jobs[job["job_id"]]["status"] = "failed"
                        save_jobs(jobs)
    else:
        st.info("Click 'Refresh Leagues' to load your Yahoo Fantasy leagues.")


def render_status_page():
    """Render the job status page"""
    st.header("📈 Import Job Status")

    jobs = load_jobs()
    if not jobs:
        st.info("No import jobs found. Go to the Import page to create one!")
        return

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (every 10 seconds)", value=False)

    if auto_refresh:
        time.sleep(10)
        st.rerun()

    # Refresh button
    if st.button("🔄 Refresh Status"):
        # Check GitHub for updates
        for job_id, job in jobs.items():
            if job["status"] in ["queued", "running"]:
                github_status = check_github_job_status(job_id)
                if github_status:
                    job.update(github_status)
        save_jobs(jobs)
        st.rerun()

    # Group jobs by status
    queued = [j for j in jobs.values() if j["status"] == "queued"]
    running = [j for j in jobs.values() if j["status"] == "running"]
    complete = [j for j in jobs.values() if j["status"] == "complete"]
    failed = [j for j in jobs.values() if j["status"] == "failed"]

    # Display jobs by status
    tab1, tab2, tab3, tab4 = st.tabs([
        f"⏳ Queued ({len(queued)})",
        f"🔄 Running ({len(running)})",
        f"✅ Complete ({len(complete)})",
        f"❌ Failed ({len(failed)})"
    ])

    with tab1:
        if queued:
            for job in sorted(queued, key=lambda x: x["created_at"], reverse=True):
                render_job_status(job)
        else:
            st.info("No queued jobs")

    with tab2:
        if running:
            for job in sorted(running, key=lambda x: x["created_at"], reverse=True):
                render_job_status(job)
                st.progress(0.5)  # Placeholder progress
        else:
            st.info("No running jobs")

    with tab3:
        if complete:
            for job in sorted(complete, key=lambda x: x["created_at"], reverse=True):
                render_job_status(job)
                # Show connection info
                if MOTHERDUCK_TOKEN and job.get("database_name"):
                    st.code(f"""
-- Connect to your data in MotherDuck:
ATTACH 'md:{job['database_name']}';
USE {job['database_name']};
SELECT * FROM matchup LIMIT 10;
                    """, language="sql")
        else:
            st.info("No completed jobs")

    with tab4:
        if failed:
            for job in sorted(failed, key=lambda x: x["created_at"], reverse=True):
                render_job_status(job)
                if st.button(f"Retry", key=f"retry_{job['job_id']}"):
                    # Reset status and re-trigger
                    job["status"] = "queued"
                    save_jobs(jobs)
                    if trigger_github_workflow(job, st.session_state.token_data):
                        st.success("Job resubmitted!")
                        st.rerun()
        else:
            st.info("No failed jobs")


def render_databases_page():
    """Render the databases/MotherDuck connection page"""
    st.header("🗄️ MotherDuck Databases")

    if not MOTHERDUCK_TOKEN:
        st.warning("MotherDuck token not configured. Add MOTHERDUCK_TOKEN to secrets.")
        st.markdown("""
        ### How to get a MotherDuck token:
        1. Go to [MotherDuck](https://motherduck.com)
        2. Sign up or log in
        3. Go to Settings > API Tokens
        4. Create a new token
        5. Add it to your Streamlit secrets
        """)
        return

    # Show completed imports with database info
    jobs = load_jobs()
    complete_jobs = [j for j in jobs.values() if j["status"] == "complete"]

    if not complete_jobs:
        st.info("No completed imports yet. Complete an import to see your databases.")
        return

    st.subheader("📊 Available Databases")

    for job in sorted(complete_jobs, key=lambda x: x["created_at"], reverse=True):
        with st.expander(f"{job['league_name']} ({job['season']})"):
            st.write(f"**Database:** `{job.get('database_name', 'N/A')}`")
            st.write(f"**Imported:** {job.get('completed_at', job['created_at'])}")

            # Sample queries
            st.markdown("### Sample Queries")

            st.code(f"""
-- Season summary
SELECT 
    manager_name,
    COUNT(*) as games,
    SUM(won) as wins,
    ROUND(AVG(team_points), 2) as avg_points
FROM {job['database_name']}.matchup
WHERE season = {job['season']}
GROUP BY manager_name
ORDER BY wins DESC;
            """, language="sql")

            st.code(f"""
-- Top scoring weeks
SELECT 
    week,
    manager_name,
    team_points,
    opponent_name,
    opponent_points
FROM {job['database_name']}.matchup
WHERE season = {job['season']}
ORDER BY team_points DESC
LIMIT 10;
            """, language="sql")

            # Connection string
            st.markdown("### Connection Info")
            st.code(f"md:{job['database_name']}?motherduck_token=YOUR_TOKEN", language="text")


if __name__ == "__main__":
    main()