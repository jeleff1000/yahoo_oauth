# python
import os
import urllib.parse
import base64
import json
import subprocess
import sys
import zipfile
import io
from pathlib import Path
from datetime import datetime, timedelta

# Provide clear messages if required dependencies are missing
try:
    import streamlit as st
except ImportError as e:
    raise ImportError("Missing dependency: streamlit. Install with `pip install streamlit`") from e

try:
    import requests
except ImportError as e:
    raise ImportError("Missing dependency: requests. Install with `pip install requests`") from e

# Load your client credentials
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID") or st.secrets.get("YAHOO_CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET") or st.secrets.get("YAHOO_CLIENT_SECRET", None)

# MotherDuck token loaded from environment or Streamlit secrets (no interactive prompt)
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN") or st.secrets.get("MOTHERDUCK_TOKEN", "")

# For deployment - NO TRAILING SLASH
REDIRECT_URI = os.environ.get("REDIRECT_URI", "https://leaguehistory.streamlit.app/")

# OAuth 2.0 endpoints
AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

# Paths
ROOT_DIR = Path(__file__).parent
OAUTH_DIR = ROOT_DIR / "oauth"
DATA_DIR = ROOT_DIR / "fantasy_football_data"
SCRIPTS_DIR = ROOT_DIR / "fantasy_football_data_scripts"
INITIAL_IMPORT_SCRIPT = SCRIPTS_DIR / "initial_import.py"


def get_auth_header():
    """Create Basic Auth header as per Yahoo's requirements"""
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


def build_authorize_url(state: str = None) -> str:
    """Build the Yahoo OAuth authorization URL"""
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
    }
    if state:
        params["state"] = state
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for access token"""
    headers = {
        "Authorization": get_auth_header(),
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code": code,
    }

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


def save_oauth_token(token_data: dict, league_info: dict = None):
    """Save OAuth token in the format expected by the scripts"""
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    oauth_file = OAUTH_DIR / "Oauth.json"

    # Format 2 (nested structure) - compatible with oauth_utils.py
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

    with open(oauth_file, 'w') as f:
        json.dump(oauth_data, f, indent=2)

    return oauth_file


def run_initial_import():
    """Run the initial_import.py script to fetch all league data"""
    if not INITIAL_IMPORT_SCRIPT.exists():
        st.error(f"❌ Initial import script not found at: {INITIAL_IMPORT_SCRIPT}")
        return False

    try:
        st.info("🚀 Starting initial data import... This may take several minutes.")

        # Create placeholders for progress
        log_placeholder = st.empty()

        # Run the script
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        # Ensure MotherDuck token is present for subprocess if configured
        if MOTHERDUCK_TOKEN:
            env["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN

        # Pass league info as environment variables for MotherDuck upload
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

            # Stream output to UI
            output_lines = []
            for line in process.stdout:
                output_lines.append(line.strip())
                # Show last 10 lines of output
                log_placeholder.code('\n'.join(output_lines[-10:]))

            process.wait()

            if process.returncode == 0:
                st.success("✅ Data import completed successfully!")
                st.balloons()
                return True
            else:
                st.error(f"❌ Import failed with exit code {process.returncode}")
                st.code('\n'.join(output_lines))
                return False

    except Exception as e:
        st.error(f"❌ Error running import: {e}")
        return False


def main():
    st.title("🏈 Yahoo Fantasy Football League History")

    # Check if credentials are loaded
    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("❌ Credentials not configured!")
        st.info("Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET environment variables or in Streamlit secrets.")
        return

    # Show whether MotherDuck token is configured (no interactive entry)
    if MOTHERDUCK_TOKEN:
        st.success("✅ MotherDuck token loaded from environment or `st.secrets`.")
    else:
        st.warning("⚠️ MotherDuck token not configured. Data will be saved locally only. To enable automatic upload, add `MOTHERDUCK_TOKEN` to `st.secrets` or environment variables.")

    # Check for errors in URL
    qp = st.query_params
    if "error" in qp:
        st.error(f"❌ OAuth Error: {qp.get('error')}")
        if "error_description" in qp:
            st.error(f"Description: {qp.get('error_description')}")
        if st.button("Clear Error & Retry"):
            st.query_params.clear()
            st.rerun()
        return

    # Check if Yahoo redirected back with authorization code
    if "code" in qp:
        code = qp["code"]

        with st.spinner("Connecting to Yahoo..."):
            try:
                token_data = exchange_code_for_tokens(code)

                # Store token data in session state
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

                # Clear the code from URL
                st.query_params.clear()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")

        return

    # Check if we have a stored access token
    if "access_token" in st.session_state:
        access_token = st.session_state.access_token

        st.success("🔐 Connected to Yahoo Fantasy!")

        # Fetch user's games if not already loaded
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

        games_data = st.session_state.games_data
        football_games = extract_football_games(games_data)

        if not football_games:
            st.warning("No football leagues found for your account.")
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()
            return

        st.subheader("📋 Select Your League")

        # Display football seasons
        season_options = {f"{game['season']} NFL Season": game['game_key']
                         for game in football_games}

        selected_season = st.selectbox(
            "1. Choose a season:",
            options=list(season_options.keys())
        )

        if selected_season:
            game_key = season_options[selected_season]

            # Auto-load leagues for selected season
            if "current_game_key" not in st.session_state or st.session_state.current_game_key != game_key:
                with st.spinner("Loading leagues..."):
                    try:
                        leagues_data = get_user_football_leagues(access_token, game_key)
                        st.session_state.current_leagues = leagues_data
                        st.session_state.current_game_key = game_key
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Display leagues if loaded
            if "current_leagues" in st.session_state:
                leagues_data = st.session_state.current_leagues

                try:
                    leagues = leagues_data.get("fantasy_content", {}).get("users", {}).get("0", {}).get("user", [])[1].get("games", {}).get("0", {}).get("game", [])[1].get("leagues", {})

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

                        league_names = [f"{league['name']} ({league['num_teams']} teams)" for league in league_list]
                        selected_league_name = st.radio("", league_names, key="league_radio")

                        selected_league = league_list[league_names.index(selected_league_name)]

                        st.divider()

                        # Show league details
                        st.write("3. Review league details:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("League", selected_league['name'])
                        with col2:
                            st.metric("Season", selected_league['season'])
                        with col3:
                            st.metric("Teams", selected_league['num_teams'])

                        st.info(f"📊 `Data to import:` All historical data for `{selected_league['name']}` (league key: {selected_league['league_key']})")

                        st.divider()

                        st.write("4. Import your league data:")
                        st.info("This will fetch all historical data from your league and save it locally (and upload to MotherDuck if configured).")

                        # Show MotherDuck configuration state (read-only)
                        with st.expander("🦆 MotherDuck Configuration (Read-only)"):
                            if MOTHERDUCK_TOKEN:
                                st.success("✅ MotherDuck token is loaded from environment or `st.secrets`.")
                                sanitized_db = selected_league['name'].lower().replace(' ', '_')
                                st.info(f"Database will be created as: `{sanitized_db}`")
                            else:
                                st.warning("MotherDuck token not configured. Upload will be skipped; files will be saved locally.")

                        if st.button("📥 Import League Data Now", type="primary"):
                            # Store league info in session state for environment variables
                            st.session_state.league_info = selected_league

                            with st.spinner("Saving OAuth credentials..."):
                                # Save OAuth token with league info
                                oauth_file = save_oauth_token(
                                    st.session_state.token_data,
                                    selected_league
                                )
                                st.success(f"✅ OAuth credentials saved to: {oauth_file}")

                            # Ensure subprocess sees MOTHERDUCK_TOKEN if configured
                            if MOTHERDUCK_TOKEN:
                                os.environ["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN

                            # Run initial import
                            if run_initial_import():
                                st.success("🎉 All done! Your league data has been imported.")

                                # Show file locations
                                st.write("### 📁 Files Saved:")
                                st.write(f"**OAuth Token:** `{OAUTH_DIR / 'Oauth.json'}`")
                                st.write(f"**League Data:** `{DATA_DIR}/`")

                                # Show what was created
                                if DATA_DIR.exists():
                                    st.write("#### Data Files Created:")
                                    parquet_files = list(DATA_DIR.glob("*.parquet"))
                                    if parquet_files:
                                        for pf in sorted(parquet_files):
                                            size = pf.stat().st_size / 1024  # KB
                                            st.write(f"- `{pf.name}` ({size:.1f} KB)")

                                    st.divider()

                                    # Download options
                                    st.write("#### 💾 Download Your Data:")
                                    st.info("⚠️ `Important:` On Streamlit Cloud, these files are temporary. Download them now to save locally!")

                                    # Download OAuth token
                                    oauth_file_path = OAUTH_DIR / "Oauth.json"
                                    if oauth_file_path.exists():
                                        with open(oauth_file_path, 'r') as f:
                                            oauth_json = f.read()
                                        st.download_button(
                                            "📥 Download OAuth Token (Oauth.json)",
                                            oauth_json,
                                            file_name="Oauth.json",
                                            mime="application/json",
                                            help="Save this file to your oauth/ folder to use with your scripts"
                                        )

                                    # Download individual parquet files
                                    if parquet_files:
                                        st.write("**Download Data Files:**")
                                        for pf in sorted(parquet_files):
                                            with open(pf, 'rb') as f:
                                                st.download_button(
                                                    f"📥 {pf.name}",
                                                    f.read(),
                                                    file_name=pf.name,
                                                    mime="application/octet-stream",
                                                    help=f"Download {pf.name} to your fantasy_football_data/ folder"
                                                )

                                    # Download all files as ZIP
                                    with st.spinner("Creating ZIP archive of your data..."):
                                        zip_buffer = io.BytesIO()
                                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                            # Add OAuth file
                                            if oauth_file_path.exists():
                                                zip_file.write(oauth_file_path, arcname="Oauth.json")

                                            # Add all parquet files
                                            for pf in sorted(parquet_files):
                                                zip_file.write(pf, arcname=pf.name)

                                        zip_buffer.seek(0)

                                        st.download_button(
                                            "📥 Download All Files as ZIP",
                                            zip_buffer,
                                            file_name="fantasy_football_data.zip",
                                            mime="application/zip",
                                            help="Download all data files as a single ZIP archive"
                                        )

                                    st.success("✅ All files ready for download above!")

                    else:
                        st.info("No leagues found for this season.")

                except Exception as e:
                    st.error(f"Error parsing leagues: {e}")

        st.divider()

        if st.button("Start Over"):
            st.session_state.clear()
            st.rerun()

    else:
        # Show the authorization button
        st.write("### Import Your Fantasy Football League Data")

        st.write("📊 This tool will fetch and save your complete league history including:")
        st.write("- All-time schedules and matchups")
        st.write("- Player statistics")
        st.write("- Transaction history")
        st.write("- Draft data")
        st.write("- Playoff information")

        st.divider()

        st.write("**How it works:**")
        st.write("1. Connect your Yahoo account")
        st.write("2. Select your league")
        st.write("3. Data is automatically imported and saved locally")
        st.write("4. Use the data with your analysis scripts")

        st.warning("⚠️ We only access your league data to build local files. Your Yahoo credentials are stored securely in the `oauth/` folder.")

        auth_url = build_authorize_url()

        st.link_button("🔐 Connect Yahoo Account", auth_url, type="primary")


if __name__ == "__main__":
    main()
