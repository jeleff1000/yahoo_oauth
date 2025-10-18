#!/usr/bin/env python3
from __future__ import annotations

import os
import urllib.parse
import base64
import json
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import Optional

try:
    import streamlit as st
except ImportError as e:
    raise ImportError("Missing dependency: streamlit. Install with `pip install streamlit`") from e

try:
    import requests
except ImportError as e:
    raise ImportError("Missing dependency: requests. Install with `pip install requests`") from e

try:
    import duckdb
except ImportError as e:
    raise ImportError("Missing dependency: duckdb. Install with `pip install duckdb`") from e

# =========================
# Config / Secrets
# =========================
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID") or st.secrets.get("YAHOO_CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET") or st.secrets.get("YAHOO_CLIENT_SECRET", None)
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN") or st.secrets.get("MOTHERDUCK_TOKEN", "")
REDIRECT_URI = os.environ.get("REDIRECT_URI", "https://leaguehistory.streamlit.app")

AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

ROOT_DIR = Path(__file__).parent
OAUTH_DIR = ROOT_DIR / "oauth"

# =========================
# Custom CSS
# =========================
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
    
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero p {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-bottom: 2rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1a202c;
    }
    
    .feature-desc {
        color: #718096;
        font-size: 0.9rem;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-queued {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-running {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .status-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-failed {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Job card */
    .job-card {
        background: linear-gradient(135deg, #f6f8fc 0%, #ffffff 100%);
        border-left: 4px solid #667eea;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .job-id {
        font-family: 'Courier New', monospace;
        background: #f1f5f9;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        color: #718096;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Timeline */
    .timeline-item {
        position: relative;
        padding-left: 2rem;
        padding-bottom: 1.5rem;
        border-left: 2px solid #e2e8f0;
    }
    
    .timeline-item:last-child {
        border-left: 2px solid transparent;
    }
    
    .timeline-dot {
        position: absolute;
        left: -0.5rem;
        width: 1rem;
        height: 1rem;
        border-radius: 50%;
        background: #667eea;
        border: 2px solid white;
        box-shadow: 0 0 0 3px #e2e8f0;
    }
    
    .timeline-content {
        margin-top: -0.25rem;
    }
    
    /* Button enhancements */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# =========================
# OAuth Helpers (same as before)
# =========================
def get_auth_header() -> str:
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"

def build_authorize_url(state: str | None = None) -> str:
    params = {"client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI, "response_type": "code"}
    if state:
        params["state"] = state
    return AUTH_URL + "?" + urllib.parse.urlencode(params)

def exchange_code_for_tokens(code: str) -> dict:
    headers = {"Authorization": get_auth_header(), "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", "redirect_uri": REDIRECT_URI, "code": code}
    resp = requests.post(TOKEN_URL, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()

def yahoo_api_call(access_token: str, endpoint: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/{endpoint}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def get_user_games(access_token: str):
    return yahoo_api_call(access_token, "users;use_login=1/games?format=json")

def get_user_football_leagues(access_token: str, game_key: str):
    return yahoo_api_call(access_token, f"users;use_login=1/games;game_keys={game_key}/leagues?format=json")

def extract_football_games(games_data):
    football_games = []
    try:
        games = games_data.get("fantasy_content", {}).get("users", {}).get("0", {}).get("user", [])[1].get("games", {})
        for key in games:
            if key == "count":
                continue
            game = games[key].get("game")
            if isinstance(game, list):
                game = game[0]
            if game and game.get("code") == "nfl":
                football_games.append({
                    "game_key": game.get("game_key"),
                    "season": game.get("season"),
                    "name": game.get("name"),
                })
    except Exception:
        pass
    return football_games

def save_oauth_token(token_data: dict, league_info: dict | None = None) -> Path:
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    oauth_file = OAUTH_DIR / "Oauth.json"
    oauth_data = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "consumer_key": CLIENT_ID,
        "consumer_secret": CLIENT_SECRET,
        "token_type": token_data.get("token_type", "bearer"),
        "expires_in": token_data.get("expires_in", 3600),
        "token_time": datetime.now(timezone.utc).timestamp(),
        "guid": token_data.get("xoauth_yahoo_guid"),
        "timestamp": datetime.now().isoformat(),
    }
    if league_info:
        oauth_data["league_info"] = league_info
    with open(oauth_file, "w", encoding="utf-8") as f:
        json.dump(oauth_data, f, indent=2)
    return oauth_file

def save_token_to_motherduck(token_data: dict, league_info: Optional[dict] = None) -> Optional[str]:
    try:
        os.environ.setdefault("MOTHERDUCK_TOKEN", MOTHERDUCK_TOKEN)
        con = duckdb.connect("md:")
        con.execute("""
            CREATE TABLE IF NOT EXISTS secrets.yahoo_oauth_tokens (
                id TEXT, league_key TEXT, token_json TEXT, updated_at TIMESTAMP
            )
        """)
        row_id = str(uuid.uuid4())
        league_key = league_info.get("league_key") if league_info else None
        token_for_storage = {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "consumer_key": CLIENT_ID,
            "consumer_secret": CLIENT_SECRET,
            "token_type": token_data.get("token_type", "bearer"),
            "expires_in": token_data.get("expires_in", 3600),
            "token_time": datetime.now(timezone.utc).timestamp(),
            "guid": token_data.get("xoauth_yahoo_guid"),
            "league_info": league_info,
        }
        token_json = json.dumps(token_for_storage)
        con.execute("INSERT INTO secrets.yahoo_oauth_tokens VALUES (?,?,?,?)",
                   [row_id, league_key, token_json, datetime.now()])
        con.close()
        return row_id
    except Exception:
        return None

def create_import_job_in_motherduck(league_info: dict) -> Optional[str]:
    try:
        os.environ.setdefault("MOTHERDUCK_TOKEN", MOTHERDUCK_TOKEN)
        con = duckdb.connect("md:")
        con.execute("""
            CREATE TABLE IF NOT EXISTS ops.import_status (
                job_id TEXT, league_key TEXT, league_name TEXT, season TEXT,
                status TEXT, created_at TIMESTAMP, updated_at TIMESTAMP
            )
        """)
        job_id = str(uuid.uuid4())
        now = datetime.now()
        con.execute("INSERT INTO ops.import_status VALUES (?,?,?,?,?,?,?)",
                   [job_id, league_info.get("league_key"), league_info.get("name"),
                    str(league_info.get("season", "")), "queued", now, now])
        con.close()
        return job_id
    except Exception:
        return None

def get_job_status(job_id: str) -> dict:
    try:
        os.environ.setdefault("MOTHERDUCK_TOKEN", MOTHERDUCK_TOKEN)
        con = duckdb.connect("md:")
        result = con.execute(
            "SELECT status, updated_at FROM ops.import_status WHERE job_id = ?",
            [job_id]
        ).fetchone()
        con.close()
        if result:
            return {"status": result[0], "updated_at": result[1]}
        return {"status": "not_found"}
    except Exception:
        return {"status": "error"}

# =========================
# UI Components
# =========================
def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🏈 Fantasy Football Analytics</h1>
        <p>Transform your Yahoo Fantasy Football data into powerful insights</p>
    </div>
    """, unsafe_allow_html=True)

def render_feature_card(icon: str, title: str, description: str):
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def render_status_badge(status: str) -> str:
    status_map = {
        "queued": ("⏳", "status-queued", "Queued"),
        "running": ("🔄", "status-running", "Running"),
        "success": ("✅", "status-success", "Complete"),
        "failed": ("❌", "status-failed", "Failed"),
    }
    icon, css_class, label = status_map.get(status, ("", "status-queued", status))
    return f'<span class="status-badge {css_class}">{icon} {label}</span>'

def render_job_card(job_id: str, league_name: str, status: str):
    st.markdown(f"""
    <div class="job-card">
        <h3>🎯 {league_name}</h3>
        <p><strong>Job ID:</strong> <span class="job-id">{job_id}</span></p>
        <p><strong>Status:</strong> {render_status_badge(status)}</p>
    </div>
    """, unsafe_allow_html=True)

def render_timeline():
    st.markdown("""
    <div style="margin: 2rem 0;">
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 1: Connect</strong>
                <p style="color: #718096; font-size: 0.9rem;">Authenticate with your Yahoo account</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 2: Select</strong>
                <p style="color: #718096; font-size: 0.9rem;">Choose your league and season</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 3: Import</strong>
                <p style="color: #718096; font-size: 0.9rem;">Queue your data for processing</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 4: Analyze</strong>
                <p style="color: #718096; font-size: 0.9rem;">Query your data from anywhere</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Main App
# =========================
def main():
    st.set_page_config(
        page_title="Fantasy Football Analytics",
        page_icon="🏈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    load_custom_css()

    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("⚠️ Service configuration error. Please contact support.")
        return

    qp = st.query_params

    # Handle OAuth callback
    if "code" in qp:
        with st.spinner("🔐 Connecting to Yahoo..."):
            try:
                token_data = exchange_code_for_tokens(qp["code"])
                st.session_state.token_data = token_data
                st.session_state.access_token = token_data.get("access_token")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        return

    # Main application
    if "access_token" in st.session_state:
        render_hero()

        # Load games
        if "games_data" not in st.session_state:
            with st.spinner("Loading your leagues..."):
                try:
                    st.session_state.games_data = get_user_games(st.session_state.access_token)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

        football_games = extract_football_games(st.session_state.games_data)

        if not football_games:
            st.warning("No fantasy football leagues found.")
            return

        # League selection
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📋 Select Your League")

            season_options = {f"{g['season']} NFL Season": g['game_key'] for g in football_games}
            selected_season = st.selectbox("Season:", list(season_options.keys()), label_visibility="collapsed")

            if selected_season:
                game_key = season_options[selected_season]

                if "current_game_key" not in st.session_state or st.session_state.current_game_key != game_key:
                    with st.spinner("Loading leagues..."):
                        try:
                            leagues_data = get_user_football_leagues(st.session_state.access_token, game_key)
                            st.session_state.current_leagues = leagues_data
                            st.session_state.current_game_key = game_key
                        except Exception as e:
                            st.error(f"Error: {e}")

                if "current_leagues" in st.session_state:
                    try:
                        leagues = (
                            st.session_state.current_leagues.get("fantasy_content", {})
                            .get("users", {}).get("0", {}).get("user", [])[1]
                            .get("games", {}).get("0", {}).get("game", [])[1]
                            .get("leagues", {})
                        )
                        league_list = []
                        for key in leagues:
                            if key == "count":
                                continue
                            league = leagues[key].get("league", [])[0]
                            league_list.append({
                                "league_key": league.get("league_key"),
                                "name": league.get("name"),
                                "num_teams": league.get("num_teams"),
                                "season": league.get("season"),
                            })

                        if league_list:
                            league_names = [f"{l['name']} ({l['num_teams']} teams)" for l in league_list]
                            selected_name = st.radio("", league_names, label_visibility="collapsed")
                            selected_league = league_list[league_names.index(selected_name)]

                            # Stats display
                            st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
                            cols = st.columns(3)
                            with cols[0]:
                                st.markdown(f"""
                                <div class="stat-card">
                                    <div class="stat-value">{selected_league['season']}</div>
                                    <div class="stat-label">Season</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with cols[1]:
                                st.markdown(f"""
                                <div class="stat-card">
                                    <div class="stat-value">{selected_league['num_teams']}</div>
                                    <div class="stat-label">Teams</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with cols[2]:
                                st.markdown(f"""
                                <div class="stat-card">
                                    <div class="stat-value">📊</div>
                                    <div class="stat-label">Ready</div>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)

                            # Import button
                            if st.button("🚀 Start Import", type="primary", use_container_width=True):
                                with st.spinner("Queuing import job..."):
                                    save_oauth_token(st.session_state.token_data, selected_league)

                                    if MOTHERDUCK_TOKEN:
                                        save_token_to_motherduck(st.session_state.token_data, selected_league)
                                        job_id = create_import_job_in_motherduck(selected_league)

                                        if job_id:
                                            st.session_state.job_id = job_id
                                            st.session_state.job_league_name = selected_league['name']
                                            st.success("✅ Import job queued!")
                                            st.balloons()
                                            st.rerun()

                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            st.markdown("### 💡 What You'll Get")
            render_feature_card("📅", "Schedules", "All-time matchups and records")
            render_feature_card("👥", "Players", "Complete stat history")
            render_feature_card("💰", "Transactions", "Trades and pickups")
            render_feature_card("🏆", "Playoffs", "Championship data")

        # Job status section
        if "job_id" in st.session_state:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 📊 Your Import Job")

            status_info = get_job_status(st.session_state.job_id)
            render_job_card(
                st.session_state.job_id,
                st.session_state.get("job_league_name", "League"),
                status_info.get("status", "queued")
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Status", use_container_width=True):
                    st.rerun()
            with col2:
                github_repo = os.environ.get("GITHUB_REPOSITORY", "your-username/your-repo")
                st.link_button(
                    "🔗 View on GitHub",
                    f"https://github.com/{github_repo}/actions",
                    use_container_width=True
                )

            if status_info.get("status") == "success":
                st.success("🎉 Your data is ready in MotherDuck!")
                league_info = st.session_state.get("league_info", {})
                db_name = f"{league_info.get('name', 'league')}_{league_info.get('season', '')}".lower().replace(' ', '_')
                st.code(f"SELECT * FROM {db_name}.public.matchup LIMIT 10;", language="sql")

    else:
        # Landing page
        render_hero()

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("### 🎯 Transform Your League Data")
            st.markdown("""
            Connect your Yahoo Fantasy Football account and unlock powerful analytics:
            """)

            features_col1, features_col2 = st.columns(2)
            with features_col1:
                render_feature_card("📈", "Win Probability", "Track your playoff chances")
                render_feature_card("🎯", "Optimal Lineups", "See your best possible scores")
            with features_col2:
                render_feature_card("📊", "Advanced Stats", "Deep dive into performance")
                render_feature_card("🔮", "Predictions", "Expected vs actual records")

            st.markdown("<br>", unsafe_allow_html=True)
            auth_url = build_authorize_url()
            st.link_button("🔐 Connect Yahoo Account", auth_url, type="primary", use_container_width=True)

            st.caption("🔒 Your data is secure. We only access league statistics, never personal information.")

        with col2:
            st.markdown("### 🚀 How It Works")
            render_timeline()

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Made with ❤️ for fantasy football managers | Powered by MotherDuck & GitHub Actions")

if __name__ == "__main__":
    main()
