#!/usr/bin/env python3
from __future__ import annotations

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
from datetime import datetime, timedelta, timezone

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
DATA_DIR = ROOT_DIR / "fantasy_football_data"
SCRIPTS_DIR = ROOT_DIR / "fantasy_football_data_scripts"
INITIAL_IMPORT_SCRIPT = SCRIPTS_DIR / "initial_import.py"


# =========================
# Yahoo OAuth Helpers
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
                    "is_game_over": game.get("is_game_over"),
                })
    except Exception as e:
        st.error(f"Error parsing games: {e}")
    return football_games


def save_oauth_token(token_data: dict, league_info: dict | None = None) -> Path:
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
            "token_time": datetime.now(timezone.utc).timestamp(),
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

        oauth_file = OAUTH_DIR / "Oauth.json"
        if oauth_file.exists():
            env["OAUTH_PATH"] = str(oauth_file.resolve())

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
# Streamlit UI
# =========================
def main():
    st.title("🏈 Yahoo Fantasy Football League History")

    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("❌ Credentials not configured!")
        st.info("Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET in environment or Streamlit secrets.")
        return

    if MOTHERDUCK_TOKEN:
        st.success("✅ MotherDuck token loaded.")
    else:
        st.warning("⚠️ MotherDuck token not configured. Data will be saved locally only.")

    qp = st.query_params

    # Handle OAuth errors
    if "error" in qp:
        st.error(f"❌ OAuth Error: {qp.get('error')}")
        if "error_description" in qp:
            st.error(f"Description: {qp.get('error_description')}")
        if st.button("Clear Error & Retry"):
            st.query_params.clear()
            st.rerun()
        return

    # Handle OAuth callback
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
                st.session_state.token_expiry = datetime.now(timezone.utc) + timedelta(
                    seconds=token_data.get("expires_in", 3600))
                st.success("✅ Successfully connected!")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
        return

    # Main application flow
    if "access_token" in st.session_state:
        access_token = st.session_state.access_token
        st.success("🔐 Connected to Yahoo Fantasy!")

        # Load games data
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

        # Season selection
        season_options = {f"{game['season']} NFL Season": game['game_key'] for game in football_games}
        selected_season = st.selectbox("1. Choose a season:", options=list(season_options.keys()))

        if selected_season:
            game_key = season_options[selected_season]

            # Load leagues for selected season
            if "current_game_key" not in st.session_state or st.session_state.current_game_key != game_key:
                with st.spinner("Loading leagues..."):
                    try:
                        leagues_data = get_user_football_leagues(access_token, game_key)
                        st.session_state.current_leagues = leagues_data
                        st.session_state.current_game_key = game_key
                    except Exception as e:
                        st.error(f"Error: {e}")

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
                        selected_league_name = st.radio("league_selection", league_names, key="league_radio",
                                                        label_visibility="collapsed")
                        selected_league = league_list[league_names.index(selected_league_name)]

                        st.divider()
                        st.write("3. Review league details:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("League", selected_league['name'])
                        with col2:
                            st.metric("Season", selected_league['season'])
                        with col3:
                            st.metric("Teams", selected_league['num_teams'])

                        st.info(f"📊 Data to import: All historical data for `{selected_league['name']}` "
                                f"(league key: {selected_league['league_key']})")

                        st.divider()
                        st.write("4. Import your league data:")
                        st.info(
                            "This fetches all historical data and saves it locally (and uploads to MotherDuck if configured).")

                        if st.button("📥 Import League Data Now", type="primary"):
                            st.session_state.league_info = selected_league

                            # Save OAuth
                            with st.spinner("Saving OAuth credentials..."):
                                oauth_file = save_oauth_token(st.session_state.token_data, selected_league)
                                st.success(f"✅ OAuth credentials saved to: {oauth_file}")

                            # Run import
                            if run_initial_import():
                                st.success("🎉 All done! Your league data has been imported.")

                                # Provide download of import log for debugging
                                import_log_dir = DATA_DIR / "import_logs"
                                if import_log_dir.exists():
                                    log_files = sorted(import_log_dir.glob("initial_import_*.log"),
                                                       key=lambda x: x.stat().st_mtime, reverse=True)
                                    if log_files:
                                        latest_log = log_files[0]
                                        with open(latest_log, 'r', encoding='utf-8') as f:
                                            log_content = f.read()
                                        st.download_button(
                                            "📄 Download Full Import Log",
                                            log_content,
                                            file_name=latest_log.name,
                                            mime="text/plain"
                                        )

                                # Collect files
                                files = collect_parquet_files()

                                if not files:
                                    st.warning("⚠️ No parquet files found. Check if the import completed successfully.")
                                    st.write(f"**Expected location:** {DATA_DIR}")
                                    if DATA_DIR.exists():
                                        st.write("**Contents:**")
                                        for item in DATA_DIR.iterdir():
                                            st.write(f"- {item.name}")
                                else:
                                    st.success(f"✅ Found {len(files)} parquet file(s)")

                                    # Show files
                                    with st.expander("📁 Files discovered"):
                                        for pf in files:
                                            try:
                                                size = pf.stat().st_size / 1024
                                                st.write(f"- `{pf.name}` ({size:.1f} KB)")
                                            except Exception:
                                                st.write(f"- `{pf.name}`")

                                    # Upload to MotherDuck
                                    if MOTHERDUCK_TOKEN:
                                        st.divider()
                                        st.write("### 🦆 Uploading to MotherDuck")

                                        league_info = st.session_state.get("league_info", {})
                                        league_name = league_info.get("name", "league")
                                        all_games = extract_football_games(st.session_state.get("games_data", {}))

                                        # Get all seasons for this league
                                        season_list = seasons_for_league_name(access_token, all_games, league_name)
                                        selected_season = str(league_info.get("season", "")).strip()
                                        if selected_season and selected_season not in season_list:
                                            season_list.append(selected_season)

                                        # Create DB per season
                                        dbs = [f"{league_name}_{season}" for season in
                                               sorted(set(s for s in season_list if s))]
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

                                    # Downloads
                                    st.divider()
                                    st.write("### 💾 Download Your Data")
                                    st.info("⚠️ Files may be temporary on cloud hosts. Download if needed.")

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

                                    # Individual files
                                    st.write("**Individual Parquet Files:**")
                                    for pf in files:
                                        try:
                                            with open(pf, "rb") as f:
                                                st.download_button(
                                                    f"📥 {pf.name}",
                                                    f.read(),
                                                    file_name=pf.name,
                                                    mime="application/octet-stream",
                                                    key=f"download_{pf.name}"
                                                )
                                        except Exception:
                                            pass

                                    # ZIP download
                                    with st.spinner("Creating ZIP archive..."):
                                        zip_buffer = io.BytesIO()
                                        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                                            if oauth_file_path.exists():
                                                zip_file.write(oauth_file_path, arcname="Oauth.json")
                                            for pf in files:
                                                zip_file.write(pf, arcname=pf.name)
                                        zip_buffer.seek(0)
                                        st.download_button(
                                            "📥 Download All (OAuth + Parquets) as ZIP",
                                            zip_buffer,
                                            file_name="fantasy_football_data.zip",
                                            mime="application/zip"
                                        )

                    else:
                        st.info("No leagues found for this season.")
                except Exception as e:
                    st.error(f"Error parsing leagues: {e}")

        st.divider()
        if st.button("Start Over"):
            st.session_state.clear()
            st.rerun()

    else:
        # Landing page
        st.write("### Import Your Fantasy Football League Data")
        st.write("- All-time schedules and matchups")
        st.write("- Player statistics")
        st.write("- Transaction history")
        st.write("- Draft data")
        st.write("- Playoff information")
        st.divider()
        st.write("**How it works:**")
        st.write(
            "1) Connect your Yahoo account → 2) Select your league → 3) Import runs → 4) Data uploaded to MotherDuck")
        st.warning("We only access your league data. Your Yahoo credentials are stored locally in `oauth/`.")
        auth_url = build_authorize_url()
        st.link_button("🔐 Connect Yahoo Account", auth_url, type="primary")


if __name__ == "__main__":
    main()