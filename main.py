#!/usr/bin/env python3
from __future__ import annotations

# =========================
# Standard libs
# =========================
import os
import urllib.parse
import base64
import json
import subprocess
import sys
import zipfile
import io
import re
from pathlib import Path
from datetime import datetime, timedelta

# =========================
# Third-party deps
# =========================
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

# MotherDuck token loaded from environment or Streamlit secrets (no interactive prompt)
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN") or st.secrets.get("MOTHERDUCK_TOKEN", "")

# For deployment - NO TRAILING SLASH
REDIRECT_URI = os.environ.get("REDIRECT_URI", "https://leaguehistory.streamlit.app")

# OAuth 2.0 endpoints
AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

# Paths
ROOT_DIR = Path(__file__).parent
OAUTH_DIR = ROOT_DIR / "oauth"
DATA_DIR = ROOT_DIR / "fantasy_football_data"
SCRIPTS_DIR = ROOT_DIR / "fantasy_football_data_scripts"
INITIAL_IMPORT_SCRIPT = SCRIPTS_DIR / "initial_import.py"

# =========================
# Yahoo OAuth Helpers
# =========================
def get_auth_header() -> str:
    """Create Basic Auth header as per Yahoo's requirements"""
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"

def build_authorize_url(state: str | None = None) -> str:
    """Build the Yahoo OAuth authorization URL"""
    params = {"client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI, "response_type": "code"}
    if state:
        params["state"] = state
    return AUTH_URL + "?" + urllib.parse.urlencode(params)

def exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for access token"""
    headers = {"Authorization": get_auth_header(), "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", "redirect_uri": REDIRECT_URI, "code": code}
    resp = requests.post(TOKEN_URL, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()

def yahoo_api_call(access_token: str, endpoint: str):
    """Make a call to Yahoo Fantasy API"""
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/{endpoint}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def get_user_games(access_token: str):
    """Get all games the user has participated in"""
    return yahoo_api_call(access_token, "users;use_login=1/games?format=json")

def get_user_football_leagues(access_token: str, game_key: str):
    """Get user's leagues for a specific football game"""
    return yahoo_api_call(access_token, f"users;use_login=1/games;game_keys={game_key}/leagues?format=json")

def extract_football_games(games_data):
    """Extract football games from the games data"""
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
                    "is_game_over": game.get("is_game_over"),
                })
    except Exception as e:
        st.error(f"Error parsing games: {e}")
    return football_games

def save_oauth_token(token_data: dict, league_info: dict | None = None) -> Path:
    """Save OAuth token in the format expected by downstream scripts"""
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    oauth_file = OAUTH_DIR / "Oauth.json"

    oauth_data = {
        "token_data": {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "consumer_key": CLIENT_ID,
            "consumer_secret": CLIENT_SECRET,
            "token_type": token_data.get("token_type", "bearer"),
            "expires_in": token_data.get("expires_in", 3600),
            "token_time": datetime.utcnow().timestamp(),
            "guid": token_data.get("xoauth_yahoo_guid")
        },
        "timestamp": datetime.now().isoformat()
    }
    if league_info:
        oauth_data["league_info"] = league_info

    with open(oauth_file, "w", encoding="utf-8") as f:
        json.dump(oauth_data, f, indent=2)
    return oauth_file

# =========================
# Slug + Collector + Uploader
# =========================
def _slug(s: str, lead_prefix: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip().lower()).strip("_")
    if not x:
        x = "db"
    if re.match(r"^\d", x):
        x = f"{lead_prefix}_{x}"
    return x[:63]

def collect_parquet_candidates(repo_root: Path, data_dir: Path) -> list[Path]:
    """
    Recursively collect parquet files to upload and to show for download.
    Priority:
      1) canonical files at data_dir root (schedule.parquet, matchup.parquet, transactions.parquet, player.parquet)
      2) any parquet under fantasy_football_data subfolders
      3) repo-wide fallback if still none
    """
    wanted_stems = {"schedule", "matchup", "transactions", "player"}
    seen = set()
    files: list[Path] = []

    # (1) canonical root
    for stem in ["schedule", "matchup", "transactions", "player"]:
        p = data_dir / f"{stem}.parquet"
        if p.exists() and p.is_file():
            files.append(p); seen.add(p.resolve())

    # (2) subfolders
    if data_dir.exists():
        for p in data_dir.rglob("*.parquet"):
            if "import_logs" in p.parts:
                continue
            rp = p.resolve()
            if rp not in seen:
                files.append(p); seen.add(rp)

    # (3) repo-wide fallback
    if not files:
        for p in repo_root.rglob("*.parquet"):
            if "import_logs" in p.parts:
                continue
            rp = p.resolve()
            if rp not in seen:
                files.append(p); seen.add(rp)

    # rank: canonical first, then subfolder, then repo fallback; prefer wanted stems
    def rank(p: Path) -> tuple[int, int, str]:
        # first key: exact canonical = 0, data_dir subfolder = 1, elsewhere = 2
        if p.parent == data_dir:
            a = 0
        elif data_dir in p.parents:
            a = 1
        else:
            a = 2
        # second key: wanted stem gets higher priority
        b = 0 if p.stem.lower() in wanted_stems else 1
        return (a, b, str(p))

    files.sort(key=rank)
    return files

def upload_files_to_motherduck(
    files: list[Path],
    db_name: str,
    schema: str = "public",
    token: str | None = None,
    status_cb=None
) -> list[tuple[str, int]]:
    """
    Upload explicit parquet files to MotherDuck.
    - Connect to md:, CREATE DATABASE IF NOT EXISTS <db>, USE <db>, ensure schema
    - CREATE OR REPLACE TABLE <schema>.<table> AS SELECT * FROM read_parquet(?)
    """
    if token:
        os.environ["MOTHERDUCK_TOKEN"] = token

    db = _slug(db_name, "l")
    sch = _slug(schema, "s")

    con = duckdb.connect("md:")
    con.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    con.execute(f"USE {db}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {sch}")

    # Normalize common stems to nicer table names
    aliases = {
        "players_by_year": "player",
        "yahoo_player_stats_multi_year_all_weeks": "player",
        "matchups": "matchup",
        "schedules": "schedule",
        "transaction": "transactions",
    }

    results: list[tuple[str, int]] = []
    for pf in files:
        stem = pf.stem.lower()
        stem = aliases.get(stem, stem)
        tbl = _slug(stem, "t")
        if status_cb:
            status_cb(f"Uploading {pf.name} → {db}.{sch}.{tbl} ...")
        con.execute(f"CREATE OR REPLACE TABLE {sch}.{tbl} AS SELECT * FROM read_parquet(?)", [str(pf)])
        cnt = con.execute(f"SELECT COUNT(*) FROM {sch}.{tbl}").fetchone()[0]
        results.append((tbl, int(cnt)))
        if status_cb:
            status_cb(f"✓ {tbl}: {cnt} rows")

    return results

def seasons_for_league_name(access_token: str, all_games: list[dict], target_league_name: str) -> list[str]:
    """
    Return all season strings where a league with the given name exists for the user.
    """
    seasons: set[str] = set()
    for g in all_games:
        game_key = g.get("game_key")
        season   = str(g.get("season", "")).strip()
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
# Import runner (subprocess)
# =========================
def run_initial_import() -> bool:
    """Run the initial_import.py script to fetch all league data"""
    if not INITIAL_IMPORT_SCRIPT.exists():
        st.error(f"❌ Initial import script not found at: {INITIAL_IMPORT_SCRIPT}")
        return False

    try:
        st.info("🚀 Starting initial data import... This may take several minutes.")

        # Create placeholders for progress
        log_placeholder = st.empty()
        status_placeholder = st.empty()

        # Create a timestamped log file for the import subprocess
        IMPORT_LOG_DIR = DATA_DIR / "import_logs"
        IMPORT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        import_log_path = IMPORT_LOG_DIR / f"initial_import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

        # Run the script
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        # Ensure MotherDuck token is present for subprocess if configured
        if MOTHERDUCK_TOKEN:
            env["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN

        # Ensure non-interactive auto-confirm so initial_import doesn't prompt
        env["AUTO_CONFIRM"] = "1"

        # Pass league info as environment variables for downstream scripts (if they read them)
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
                bufsize=1
            )

            # Stream output to UI and write to the import log file
            output_lines = []
            try:
                with open(import_log_path, 'a', encoding='utf-8') as lf:
                    for line in process.stdout:
                        stripped = line.rstrip('\n')
                        output_lines.append(stripped)

                        try:
                            lf.write(stripped + "\n"); lf.flush()
                        except Exception:
                            pass

                        try:
                            status_placeholder.info(stripped)
                        except Exception:
                            status_placeholder.text(stripped)

                        log_placeholder.code('\n'.join(output_lines[-10:]))
            except Exception:
                try:
                    remaining, _ = process.communicate(timeout=1)
                    if remaining:
                        output_lines.extend(remaining.splitlines())
                        with open(import_log_path, 'a', encoding='utf-8') as lf:
                            lf.write(remaining)
                except Exception:
                    pass

            process.wait()

            if process.returncode == 0:
                status_placeholder.success("Import finished successfully.")
                st.success("✅ Data import completed successfully!")
                st.write(f"Import log: {import_log_path}")
                return True
            else:
                status_placeholder.error(f"Import failed (exit code {process.returncode}).")
                st.error(f"❌ Import failed with exit code {process.returncode}")
                st.code('\n'.join(output_lines))
                st.write(f"Import log: {import_log_path}")
                return False

    except Exception as e:
        st.error(f"❌ Error running import: {e}")
        return False

# =========================
# Streamlit UI
# =========================
def main():
    st.title("🏈 Yahoo Fantasy Football League History")

    # Check credentials
    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("❌ Credentials not configured!")
        st.info("Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET in environment or Streamlit secrets.")
        return

    # Show MD token presence
    if MOTHERDUCK_TOKEN:
        st.success("✅ MotherDuck token loaded.")
    else:
        st.warning("⚠️ MotherDuck token not configured. Data will be saved locally only.")

    # Handle OAuth errors
    qp = st.query_params
    if "error" in qp:
        st.error(f"❌ OAuth Error: {qp.get('error')}")
        if "error_description" in qp:
            st.error(f"Description: {qp.get('error_description')}")
        if st.button("Clear Error & Retry"):
            st.query_params.clear()
            st.rerun()
        return

    # OAuth callback
    if "code" in qp:
        code = qp["code"]
        with st.spinner("Connecting to Yahoo..."):
            try:
                token_data = exchange_code_for_tokens(code)
                st.session_state.token_data = {
                    "access_token": token_data.get("access_token"),
                    "refresh_token": token_data.get("refresh_token"),
                    "token_type": token_data.get("token_type"),
                    "expires_in": token_data.get("expires_in"),
                    "xoauth_yahoo_guid": token_data.get("xoauth_yahoo_guid")
                }
                st.session_state.access_token = token_data.get("access_token")
                st.session_state.token_expiry = datetime.utcnow() + timedelta(seconds=token_data.get("expires_in", 3600))
                st.success("✅ Successfully connected!")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
        return

    # Main flow (token present)
    if "access_token" in st.session_state:
        access_token = st.session_state.access_token
        st.success("🔐 Connected to Yahoo Fantasy!")

        # Load seasons
        if "games_data" not in st.session_state:
            with st.spinner("Loading your fantasy seasons..."):
                try:
                    games_data = get_user_games(access_token)
                    st.session_state.games_data = games_data
                except Exception as e:
                    st.error(f"Error: {e}")
                    if st.button("Start Over"):
                        st.session_state.clear()
                        st.rerun()
                    return

        football_games = extract_football_games(st.session_state.games_data)
        if not football_games:
            st.warning("No football leagues found for your account.")
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()
            return

        st.subheader("📋 Select Your League")

        season_options = {f"{game['season']} NFL Season": game['game_key'] for game in football_games}
        selected_season = st.selectbox("1. Choose a season:", options=list(season_options.keys()))

        if selected_season:
            game_key = season_options[selected_season]

            # Load leagues for season
            if "current_game_key" not in st.session_state or st.session_state.current_game_key != game_key:
                with st.spinner("Loading leagues..."):
                    try:
                        leagues_data = get_user_football_leagues(access_token, game_key)
                        st.session_state.current_leagues = leagues_data
                        st.session_state.current_game_key = game_key
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Show leagues
            if "current_leagues" in st.session_state:
                leagues_data = st.session_state.current_leagues
                try:
                    leagues = (
                        leagues_data.get("fantasy_content", {})
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
                        st.write("2. Choose your league:")
                        league_names = [f"{l['name']} ({l['num_teams']} teams)" for l in league_list]
                        selected_league_name = st.radio("", league_names, key="league_radio")
                        selected_league = league_list[league_names.index(selected_league_name)]

                        st.divider()
                        st.write("3. Review league details:")
                        col1, col2, col3 = st.columns(3)
                        with col1: st.metric("League", selected_league['name'])
                        with col2: st.metric("Season", selected_league['season'])
                        with col3: st.metric("Teams", selected_league['num_teams'])

                        st.info(f"📊 Data to import: All historical data for `{selected_league['name']}` "
                                f"(league key: {selected_league['league_key']})")

                        st.divider()
                        st.write("4. Import your league data:")
                        st.info("This fetches all historical data and saves it locally (and uploads to MotherDuck if configured).")

                        with st.expander("🦆 MotherDuck Configuration (Read-only)"):
                            if MOTHERDUCK_TOKEN:
                                st.success("✅ MotherDuck token is loaded.")
                                sanitized_db = selected_league['name'].lower().replace(' ', '_')
                                st.info(f"Database(s) will be created as: `{sanitized_db}_<season>` (for every season this league exists)")
                            else:
                                st.warning("No MotherDuck token; upload will be skipped.")

                        if st.button("📥 Import League Data Now", type="primary"):
                            # Persist league info
                            st.session_state.league_info = selected_league

                            with st.spinner("Saving OAuth credentials..."):
                                oauth_file = save_oauth_token(st.session_state.token_data, selected_league)
                                st.success(f"✅ OAuth credentials saved to: {oauth_file}")

                            if MOTHERDUCK_TOKEN:
                                os.environ["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN

                            # Run the import
                            if run_initial_import():
                                st.success("🎉 All done! Your league data has been imported.")

                                # --------- COLLECT FILES (for upload + downloads) ----------
                                repo_root = ROOT_DIR
                                files = collect_parquet_candidates(repo_root, DATA_DIR)

                                # --- Upload to MotherDuck for EVERY season this league exists ---
                                if MOTHERDUCK_TOKEN:
                                    st.info("🦆 Uploading data to MotherDuck (all seasons for this league)...")

                                    league_info  = st.session_state.get("league_info", {})
                                    league_name  = league_info.get("name", "league")
                                    access_token = st.session_state.get("access_token")
                                    all_games    = extract_football_games(st.session_state.get("games_data", {}))

                                    # discover all seasons where this league name appears for the user
                                    season_list = seasons_for_league_name(access_token, all_games, league_name)

                                    # fallback: at least use the selected season
                                    selected_season = str(league_info.get("season", "")).strip()
                                    if selected_season and selected_season not in season_list:
                                        season_list.append(selected_season)

                                    # de-dupe and upload to each <league_name>_<season> database
                                    dbs = []
                                    for season in sorted({s for s in season_list if s}):
                                        dbs.append(f"{league_name}_{season}")
                                    if not dbs:
                                        dbs = [league_name]

                                    if not files:
                                        st.error("❌ No parquet files found to upload. Check producer outputs or paths.")
                                    else:
                                        st.caption(f"Found {len(files)} parquet file(s) to upload.")
                                        overall_summary = []
                                        for db_name in dbs:
                                            progress = st.empty()
                                            def _status(msg: str):
                                                try: progress.info(msg)
                                                except Exception: progress.text(msg)
                                            st.write(f"**Uploading to DB:** `{db_name}`")
                                            try:
                                                uploaded = upload_files_to_motherduck(
                                                    files,
                                                    db_name=db_name,
                                                    schema="public",
                                                    token=MOTHERDUCK_TOKEN,
                                                    status_cb=_status
                                                )
                                                if uploaded:
                                                    st.success(f"✅ `{db_name}`: uploaded {len(uploaded)} tables.")
                                                    overall_summary.append((db_name, uploaded))
                                                else:
                                                    st.warning(f"⚠️ `{db_name}`: nothing to upload.")
                                            except Exception as e:
                                                st.error(f"❌ `{db_name}` upload failed: {e}")

                                        if overall_summary:
                                            with st.expander("View upload summaries"):
                                                for db_name, items in overall_summary:
                                                    st.write(f"**{db_name}**")
                                                    for t, n in items:
                                                        st.write(f"- `public.{t}` → {n} rows")
                                        else:
                                            st.warning("No successful uploads recorded.")
                                else:
                                    st.warning("⚠️ MotherDuck token not configured—skipping cloud upload.")

                                # -------------------- DOWNLOADS --------------------
                                st.write("### 📁 Files Saved / Ready to Download")
                                st.write(f"**OAuth Token:** `{OAUTH_DIR / 'Oauth.json'}`")
                                st.write(f"**Data Root:** `{DATA_DIR}/`")

                                parquet_files = files  # use collected set
                                if parquet_files:
                                    st.write("#### Data Files Discovered:")
                                    for pf in parquet_files:
                                        try:
                                            size = pf.stat().st_size / 1024
                                            st.write(f"- `{pf.relative_to(ROOT_DIR)}` ({size:.1f} KB)")
                                        except Exception:
                                            st.write(f"- `{pf}`")

                                st.divider()
                                st.write("#### 💾 Download Your Data")
                                st.info("⚠️ On Streamlit Cloud, these files are temporary. Download them now!")

                                oauth_file_path = OAUTH_DIR / "Oauth.json"
                                if oauth_file_path.exists():
                                    with open(oauth_file_path, "r", encoding="utf-8") as f:
                                        oauth_json = f.read()
                                    st.download_button(
                                        "📥 Download OAuth Token (Oauth.json)",
                                        oauth_json,
                                        file_name="Oauth.json",
                                        mime="application/json"
                                    )

                                if parquet_files:
                                    st.write("**Download Individual Parquet Files:**")
                                    for pf in parquet_files:
                                        try:
                                            with open(pf, "rb") as f:
                                                st.download_button(
                                                    f"📥 {pf.relative_to(ROOT_DIR)}",
                                                    f.read(),
                                                    file_name=pf.name,
                                                    mime="application/octet-stream"
                                                )
                                        except Exception:
                                            pass

                                    with st.spinner("Creating ZIP archive of your data..."):
                                        zip_buffer = io.BytesIO()
                                        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                                            if oauth_file_path.exists():
                                                zip_file.write(oauth_file_path, arcname="Oauth.json")
                                            for pf in parquet_files:
                                                try:
                                                    arc = pf.relative_to(ROOT_DIR)
                                                except Exception:
                                                    arc = pf.name
                                                zip_file.write(pf, arcname=str(arc))
                                        zip_buffer.seek(0)
                                        st.download_button(
                                            "📥 Download All (OAuth + Parquets) as ZIP",
                                            zip_buffer,
                                            file_name="fantasy_football_data.zip",
                                            mime="application/zip"
                                        )
                                    st.success("✅ All files ready for download above!")
                                else:
                                    st.warning("No parquet files discovered to download.")

                    else:
                        st.info("No leagues found for this season.")
                except Exception as e:
                    st.error(f"Error parsing leagues: {e}")

        st.divider()
        if st.button("Start Over"):
            st.session_state.clear()
            st.rerun()

    else:
        # Landing state — connect button
        st.write("### Import Your Fantasy Football League Data")
        st.write("- All-time schedules and matchups")
        st.write("- Player statistics")
        st.write("- Transaction history")
        st.write("- Draft data")
        st.write("- Playoff information")
        st.divider()
        st.write("**How it works:**")
        st.write("1) Connect your Yahoo account → 2) Select your league → 3) Import runs → 4) Parquets saved (and uploaded)")
        st.warning("We only access your league data to build local files. Your Yahoo credentials are stored in `oauth/`.")
        auth_url = build_authorize_url()
        st.link_button("🔐 Connect Yahoo Account", auth_url, type="primary")


if __name__ == "__main__":
    main()
