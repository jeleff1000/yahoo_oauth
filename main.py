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
import subprocess
import sys
import re

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
    """
    Save OAuth token. Behavior:
    - Always write a global token-only file at oauth/Oauth.json (no league_info) for yahoo-oauth compatibility.
    - If `league_info` is provided, also write a per-league file named oauth/Oauth_<league_key>.json that includes league_info.
    Returns the Path to the file written (per-league file when league_info provided, otherwise global file).
    """
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    oauth_file = OAUTH_DIR / "Oauth.json"

    # Token data (keeps global file free of league metadata)
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

    # Write the global token-only file. For Streamlit Cloud behavior we want
    # the global oauth/Oauth.json to reflect the token for the league being
    # imported so library code reading the default path picks it up. We'll
    # write atomically to avoid partial files.
    def _atomic_write(path: Path, data: dict):
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            tmp.replace(path)
        except Exception:
            # best-effort fallback
            tmp.rename(path)

    try:
        # Always ensure a global token exists (overwrite when saving per-league)
        _atomic_write(oauth_file, oauth_data)
    except Exception:
        # If writing global file fails, continue — per-league file (below) may still be written
        pass

    # If league_info provided, write a per-league file so selecting a league doesn't overwrite the global token file
    if league_info:
        league_key = league_info.get("league_key") or league_info.get("league_id") or "unknown"
        # sanitize league_key for filename
        safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", str(league_key))
        per_file = OAUTH_DIR / f"Oauth_{safe_key}.json"
        per_data = oauth_data.copy()
        per_data["league_info"] = league_info
        try:
            # Write per-league file atomically as well
            tmp = per_file.with_name(f".{per_file.name}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(per_data, f, indent=2)
            try:
                tmp.replace(per_file)
            except Exception:
                tmp.rename(per_file)
            return per_file
        except Exception:
            # fallback to returning global file when per-league write fails
            return oauth_file

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


# Paths used by the import runner
DATA_DIR = ROOT_DIR / "fantasy_football_data"
SCRIPTS_DIR = ROOT_DIR / "fantasy_football_data_scripts"
INITIAL_IMPORT_SCRIPT = SCRIPTS_DIR / "initial_import_v2.py"


# =========================
# Simplified File Collection
# =========================
def collect_parquet_files() -> list[Path]:
    """
    Collect parquet files in priority order:
    1. Canonical files at DATA_DIR root (schedule.parquet, matchup.parquet, etc.)
    2. Any other parquet files in DATA_DIR subdirectories
    """
    files = []
    seen = set()

    # Priority 1: Canonical files at root
    canonical_names = ["schedule.parquet", "matchup.parquet", "transactions.parquet",
                       "player.parquet", "players_by_year.parquet"]

    for name in canonical_names:
        p = DATA_DIR / name
        if p.exists() and p.is_file():
            files.append(p)
            seen.add(p.resolve())

    # Priority 2: Subdirectories (schedule_data, matchup_data, etc.)
    if DATA_DIR.exists():
        for subdir in ["schedule_data", "matchup_data", "transaction_data", "player_data"]:
            sub_path = DATA_DIR / subdir
            if sub_path.exists() and sub_path.is_dir():
                for p in sub_path.glob("*.parquet"):
                    resolved = p.resolve()
                    if resolved not in seen:
                        files.append(p)
                        seen.add(resolved)

    # Priority 3: Any other parquet files in DATA_DIR (non-recursive, to avoid noise)
    if DATA_DIR.exists():
        for p in DATA_DIR.glob("*.parquet"):
            resolved = p.resolve()
            if resolved not in seen:
                files.append(p)
                seen.add(resolved)

    return files


# =========================
# MotherDuck Upload
# =========================
def _slug(s: str, lead_prefix: str) -> str:
    """Create a valid database/table name from a string"""
    x = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip().lower()).strip("_")
    if not x:
        x = "db"
    if re.match(r"^\d", x):
        x = f"{lead_prefix}_{x}"
    return x[:63]


def upload_to_motherduck(files: list[Path], db_name: str, token: str) -> list[tuple[str, int]]:
    """Upload parquet files directly to MotherDuck"""
    if not files:
        return []

    if token:
        os.environ["MOTHERDUCK_TOKEN"] = token

    db = _slug(db_name, "l")

    con = duckdb.connect("md:")
    con.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    con.execute(f"USE {db}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS public")

    # Table name mapping (handle common aliases)
    aliases = {
        "players_by_year": "player",
        "yahoo_player_stats_multi_year_all_weeks": "player",
        "matchups": "matchup",
        "schedules": "schedule",
        "transaction": "transactions",
    }

    results = []
    for pf in files:
        stem = pf.stem.lower()
        stem = aliases.get(stem, stem)
        tbl = _slug(stem, "t")

        try:
            st.info(f"📤 Uploading {pf.name} → {db}.public.{tbl}...")
            con.execute(f"CREATE OR REPLACE TABLE public.{tbl} AS SELECT * FROM read_parquet(?)", [str(pf)])
            cnt = con.execute(f"SELECT COUNT(*) FROM public.{tbl}").fetchone()[0]
            results.append((tbl, int(cnt)))
            st.success(f"✅ {tbl}: {cnt:,} rows")
        except Exception as e:
            st.error(f"❌ Failed to upload {pf.name}: {e}")

    con.close()
    return results


# =========================
# Season Discovery
# =========================
def seasons_for_league_name(access_token: str, all_games: list[dict], target_league_name: str) -> list[str]:
    """Find all seasons where this league exists"""
    seasons = set()
    for g in all_games:
        game_key = g.get("game_key")
        season = str(g.get("season", "")).strip()
        if not game_key or not season:
            continue
        try:
            leagues_data = get_user_football_leagues(access_token, game_key)
            leagues = (
                leagues_data.get("fantasy_content", {})
                .get("users", {}).get("0", {}).get("user", [])[1]
                .get("games", {}).get("0", {}).get("game", [])[1]
                .get("leagues", {})
            )
            for key in leagues:
                if key == "count":
                    continue
                league = leagues[key].get("league", [])[0]
                name = league.get("name")
                if name == target_league_name:
                    seasons.add(season)
                    break
        except Exception:
            pass
    return sorted(seasons)


# =========================
# Import Runner
# =========================
def run_initial_import() -> bool:
    """Run the initial data import script"""
    if not INITIAL_IMPORT_SCRIPT.exists():
        st.error(f"❌ Initial import script not found at: {INITIAL_IMPORT_SCRIPT}")
        return False

    try:
        st.info("🚀 Starting initial data import... This may take several minutes.")

        log_placeholder = st.empty()
        status_placeholder = st.empty()

        IMPORT_LOG_DIR = DATA_DIR / "import_logs"
        IMPORT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        import_log_path = IMPORT_LOG_DIR / f"initial_import_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if MOTHERDUCK_TOKEN:
            env["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN
        env["AUTO_CONFIRM"] = "1"
        env["EXPORT_DATA_DIR"] = str(DATA_DIR.resolve())

        # Prefer a per-league oauth file if we have league_info in the session; fall back to global Oauth.json
        oauth_file = OAUTH_DIR / "Oauth.json"
        try:
            if "league_info" in st.session_state and st.session_state.league_info:
                league_info = st.session_state.league_info
                league_key = league_info.get("league_key") or league_info.get("league_id") or "unknown"
                safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", str(league_key))
                per_file = OAUTH_DIR / f"Oauth_{safe_key}.json"
                if per_file.exists():
                    env["OAUTH_PATH"] = str(per_file.resolve())
                elif oauth_file.exists():
                    env["OAUTH_PATH"] = str(oauth_file.resolve())
            else:
                if oauth_file.exists():
                    env["OAUTH_PATH"] = str(oauth_file.resolve())
        except Exception:
            # If anything goes wrong, don't block the import; initial_import.py may still attempt other auth flows
            if oauth_file.exists():
                env["OAUTH_PATH"] = str(oauth_file.resolve())

        # Surface which oauth file we'll use (helps debug which token the import picks up)
        try:
            used_oauth = env.get("OAUTH_PATH")
            if used_oauth:
                status_placeholder.info(f"Using OAuth file: {used_oauth}")
            else:
                status_placeholder.info(
                    "No OAuth file set; initial_import may use other auth flows or environment variables.")
        except Exception:
            # status_placeholder may not be available in some failure branches; ignore
            pass

        if "league_info" in st.session_state:
            league_info = st.session_state.league_info
            env["LEAGUE_NAME"] = league_info.get("name", "Unknown League")
            env["LEAGUE_KEY"] = league_info.get("league_key", "unknown")
            env["LEAGUE_SEASON"] = str(league_info.get("season", ""))
            env["LEAGUE_NUM_TEAMS"] = str(league_info.get("num_teams", ""))

        cmd = [sys.executable, str(INITIAL_IMPORT_SCRIPT)]

        with st.spinner("Importing league data..."):
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(ROOT_DIR)
            )

            output_lines = []
            with open(import_log_path, 'a', encoding='utf-8') as lf:
                for line in process.stdout:
                    stripped = line.rstrip('\n')
                    output_lines.append(stripped)
                    lf.write(stripped + "\n")
                    lf.flush()
                    status_placeholder.info(stripped)
                    # Show more lines in the log window (50 instead of 10)
                    log_placeholder.code('\n'.join(output_lines[-50:]))

            process.wait()

            if process.returncode == 0:
                status_placeholder.success("✅ Import finished successfully.")
                st.success("✅ Data import completed successfully!")

                # Show full log in expander for debugging
                with st.expander("📋 View Full Import Log"):
                    st.code('\n'.join(output_lines))

                return True
            else:
                status_placeholder.error(f"❌ Import failed (exit code {process.returncode}).")
                st.error(f"❌ Import failed with exit code {process.returncode}")
                st.code('\n'.join(output_lines[-100:]))  # Show more error context
                return False

    except Exception as e:
        st.error(f"❌ Error running import: {e}")
        return False


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

                            # Prominent CTA card: the import button is the main focus; other details are secondary
                            # We'll make the whole card clickable by wrapping it in a link that sets query params.
                            # Clicking the card will reload the app with the parameters and the import will run server-side.
                            link_params = {
                                "start_import": "1",
                                "league_key": selected_league.get("league_key", ""),
                                "league_name": selected_league.get("name", ""),
                                "league_season": str(selected_league.get("season", "")),
                            }
                            link_url = "?" + urllib.parse.urlencode(link_params)

                            # Use an anchor link to reliably set query params in Streamlit Cloud
                            safe_link = link_url.replace('"', '%22')
                            card_html = f'''
                            <a href="{safe_link}" style="text-decoration:none; display:block;">
                                <div class="cta-card" style="background: linear-gradient(135deg,#667eea,#7f5af0); padding:1.5rem; border-radius:0.75rem; color:white; text-align:center; cursor:pointer;">
                                    <h2 style="margin:0;">🚀 Start Import for {selected_league['name']}</h2>
                                    <p style="margin:0.25rem 0 0.75rem; opacity:0.95;">Season {selected_league['season']} — {selected_league['num_teams']} teams</p>
                                    <p style="margin:0.2rem 0 0; font-size:0.9rem; opacity:0.95;">Click anywhere on this card to start the import.</p>
                                </div>
                            </a>
                            '''

                            st.markdown(card_html, unsafe_allow_html=True)

                            # Debug & server-start fallback: visible expander to diagnose why the card click may not trigger
                            try:
                                league_key_for_debug = selected_league.get("league_key") or "unknown"
                                safe_key_dbg = re.sub(r"[^a-zA-Z0-9_-]", "_", str(league_key_for_debug))
                            except Exception:
                                safe_key_dbg = "unknown"

                            with st.expander("Debug & Server Start", expanded=True):
                                st.write("Session keys:", list(st.session_state.keys()))
                                has_token = "token_data" in st.session_state
                                st.write("Session has token_data:", has_token)
                                if has_token:
                                    at = st.session_state.token_data.get("access_token")
                                    st.write("Access token present:", bool(at))

                                # List oauth files present on disk
                                try:
                                    files = [p.name for p in OAUTH_DIR.glob("Oauth*.json")]
                                except Exception:
                                    files = []
                                st.write("Oauth files:", files)

                                st.write(
                                    "If clicking the card does nothing, use the button below to start the import server-side.")
                                if st.button("🚀 Start Import (Server)", key=f"start_import_btn_{safe_key_dbg}",
                                             use_container_width=True):
                                    with st.spinner("Starting import (server)..."):
                                        perform_import_flow(selected_league)

                            # Explicit Start Import button: fallback to ensure import can be triggered reliably
                            if st.button("🔄 Start Import", use_container_width=True):
                                with st.spinner("Starting import..."):
                                    perform_import_flow(selected_league)

                            # Note: the card above is a clickable div that starts the import via query params.
                            # Secondary details moved into an expander (non-primary action)
                            with st.expander("Details & Stats", expanded=False):
                                st.markdown(f"**Season:** {selected_league['season']}  \
                                **Teams:** {selected_league['num_teams']}")
                                st.markdown("\n")
                                st.markdown("### More options")
                                st.markdown(
                                    "You can review league metadata, download tokens, or run the import immediately.")

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
                db_name = f"{league_info.get('name', 'league')}_{league_info.get('season', '')}".lower().replace(' ',
                                                                                                                 '_')
                st.code(f"SELECT * FROM {db_name}.public.matchup LIMIT 10;", language="sql")

    else:
        # Landing page: show hero and make the Connect CTA the primary focus (large centered block)
        render_hero()

        auth_url = build_authorize_url()
        # Full-width centered CTA with max-width so it looks prominent on desktop and mobile
        connect_html_center = f'''
        <div style="display:flex; justify-content:center; margin: 1.25rem 0;">
            <a href="{auth_url}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; width:100%; max-width:980px;">
                <div style="background: linear-gradient(90deg,#ff6b4a,#ff8a5a); color:white; padding:1.25rem 1.5rem; border-radius:0.75rem; text-align:center; font-weight:700; font-size:1.15rem; box-shadow:0 10px 30px rgba(0,0,0,0.08);">
                    🔐 Connect Yahoo Account
                </div>
            </a>
        </div>
        '''
        st.markdown(connect_html_center, unsafe_allow_html=True)

        # Put feature cards below in a compact grid so the CTA remains the main focus
        st.markdown("<div style='max-width:980px; margin:0 auto;'>", unsafe_allow_html=True)
        features_col1, features_col2 = st.columns(2)
        with features_col1:
            render_feature_card("📈", "Win Probability", "Track your playoff chances")
            render_feature_card("🎯", "Optimal Lineups", "See your best possible scores")
        with features_col2:
            render_feature_card("📊", "Advanced Stats", "Deep dive into performance")
            render_feature_card("🔮", "Predictions", "Expected vs actual records")
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("🔒 Your data is secure. We only access league statistics, never personal information.")

    # If the page was loaded with start_import query params, run the import flow
    if "start_import" in qp and qp.get("start_import"):
        try:
            # Extract params (Streamlit returns lists for qp values)
            lkey = qp.get("league_key", [None])[0]
            lname = qp.get("league_name", [None])[0]
            lseason = qp.get("league_season", [None])[0]

            if lkey and lname:
                st.info(f"📥 Starting import for {lname} ({lseason})")
                # Restore league info into session_state for downstream use
                st.session_state.league_info = {"league_key": lkey, "name": lname, "season": lseason}

                # Save OAuth token locally (ensure oauth file exists for the import script)
                if "token_data" in st.session_state:
                    try:
                        saved_path = save_oauth_token(st.session_state.token_data, st.session_state.league_info)
                        st.info(f"Saved OAuth token to: {saved_path}")
                    except Exception:
                        st.warning(
                            "Failed to write per-league OAuth file; import will fall back to global Oauth.json if present.")

                # Run the import (calls the initial_import.py script)
                ok = run_initial_import()

                if ok:
                    st.success("🎉 Import finished — collecting files and uploading (if configured)...")

                    files = collect_parquet_files()
                    if not files:
                        st.warning("⚠️ No parquet files found after import. Check the import logs.")
                    else:
                        st.success(f"✅ Found {len(files)} parquet file(s)")
                        # Upload to MotherDuck if token available
                        if MOTHERDUCK_TOKEN:
                            league_name = st.session_state.league_info.get("name", "league")
                            all_games = extract_football_games(st.session_state.get("games_data", {}))
                            season_list = seasons_for_league_name(st.session_state.access_token, all_games, league_name)
                            selected_season = str(st.session_state.league_info.get("season", "")).strip()
                            if selected_season and selected_season not in season_list:
                                season_list.append(selected_season)

                            dbs = [f"{league_name}_{season}" for season in sorted(set(s for s in season_list if s))]
                            if not dbs:
                                dbs = [league_name]

                            overall_summary = []
                            for db_name in dbs:
                                st.write(f"**Database:** `{db_name}`")
                                uploaded = upload_to_motherduck(files, db_name, MOTHERDUCK_TOKEN)
                                if uploaded:
                                    overall_summary.append((db_name, uploaded))

                            if overall_summary:
                                st.success("✅ Upload complete!")
                                with st.expander("📊 Upload Summary"):
                                    for db_name, items in overall_summary:
                                        st.write(f"**{db_name}**")
                                        for tbl, cnt in items:
                                            st.write(f"- `public.{tbl}` → {cnt:,} rows")

                # Clear the query params and rerun to reset UI state
                st.query_params.clear()
                st.button("Continue")
                st.rerun()
        except Exception as e:
            st.error(f"Error starting import: {e}")
            st.query_params.clear()
            st.rerun()

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Made with ❤️ for fantasy football managers | Powered by MotherDuck & GitHub Actions")


def perform_import_flow(league_info: dict):
    """Run the same import flow used for query-param driven imports.
    This lets us trigger the import server-side (via a Streamlit button) as
    a reliable fallback to the clickable card link.
    """
    try:
        st.session_state.league_info = league_info

        # Save OAuth token locally (ensure oauth file exists for the import script)
        if "token_data" in st.session_state:
            try:
                saved_path = save_oauth_token(st.session_state.token_data, st.session_state.league_info)
                st.info(f"Saved OAuth token to: {saved_path}")
            except Exception:
                st.warning(
                    "Failed to write per-league OAuth file; import will fall back to global Oauth.json if present.")

        ok = run_initial_import()

        if ok:
            st.success("🎉 Import finished — collecting files and uploading (if configured)...")

            files = collect_parquet_files()
            if not files:
                st.warning("⚠️ No parquet files found after import. Check the import logs.")
            else:
                st.success(f"✅ Found {len(files)} parquet file(s)")
                # Upload to MotherDuck if token available
                if MOTHERDUCK_TOKEN:
                    league_name = st.session_state.league_info.get("name", "league")
                    all_games = extract_football_games(st.session_state.get("games_data", {}))
                    season_list = seasons_for_league_name(st.session_state.access_token, all_games, league_name)
                    selected_season = str(st.session_state.league_info.get("season", "")).strip()
                    if selected_season and selected_season not in season_list:
                        season_list.append(selected_season)

                    dbs = [f"{league_name}_{season}" for season in sorted(set(s for s in season_list if s))]
                    if not dbs:
                        dbs = [league_name]

                    overall_summary = []
                    for db_name in dbs:
                        st.write(f"**Database:** `{db_name}`")
                        uploaded = upload_to_motherduck(files, db_name, MOTHERDUCK_TOKEN)
                        if uploaded:
                            overall_summary.append((db_name, uploaded))

                    if overall_summary:
                        st.success("✅ Upload complete!")
                        with st.expander("📊 Upload Summary"):
                            for db_name, items in overall_summary:
                                st.write(f"**{db_name}**")
                                for tbl, cnt in items:
                                    st.write(f"- `public.{tbl}` → {cnt:,} rows")

        # Reset UI state (clear query params) and rerun so UI refreshes
        st.query_params.clear()
        st.button("Continue")
        st.rerun()

    except Exception as e:
        st.error(f"Error starting import: {e}")
        st.query_params.clear()
        st.rerun()


if __name__ == "__main__":
    main()
