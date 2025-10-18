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
from datetime import datetime, timedelta, timezone

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
        Recursively collect parquet files to upload and to include in the ZIP.
        Priority:
          1) canonical files at data_dir root
          2) any parquet under fantasy_football_data subfolders
          3) repo-wide fallback if still none

        This function also honors an environment override EXPORT_DATA_DIR and includes
        extra fallbacks (resolved paths and os.walk) to be robust across different
        mounts/container paths (e.g. Streamlit cloud `/mount/src/...`).
        """
        # Also allow a runtime override path (for Streamlit Cloud or different mount points)
        export_dir = None
        try:
            ed = os.environ.get("EXPORT_DATA_DIR")
            if ed:
                export_dir = Path(ed)
        except Exception:
            export_dir = None

        wanted_stems = {"schedule", "matchup", "transactions", "player"}
        seen = set()
        files: list[Path] = []

        # Helper to add file if valid and not seen
        def _maybe_add(p: Path):
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            if rp in seen:
                return
            try:
                if p.exists() and p.is_file():
                    files.append(p)
                    seen.add(rp)
            except Exception:
                # if we can't check exists/is_file, still try to add by path
                files.append(p)
                seen.add(rp)

        # (1) canonical root (data_dir)
        for stem in ["schedule", "matchup", "transactions", "player"]:
            p = data_dir / f"{stem}.parquet"
            _maybe_add(p)

        # (1b) canonical at export_dir if provided
        if export_dir:
            for stem in ["schedule", "matchup", "transactions", "player"]:
                p = export_dir / f"{stem}.parquet"
                _maybe_add(p)

        # (2) subfolders under data_dir
        try:
            if data_dir.exists():
                for p in data_dir.rglob("*.parquet"):
                    if "import_logs" in p.parts:
                        continue
                    _maybe_add(p)
                # also pick up files with extra suffixes like .parquet.gz or .parquet.snappy
                for p in data_dir.rglob("*.parquet.*"):
                    if "import_logs" in p.parts:
                        continue
                    _maybe_add(p)
        except Exception:
            pass

        # (2b) subfolders under export_dir
        try:
            if export_dir and export_dir.exists():
                for p in export_dir.rglob("*.parquet"):
                    if "import_logs" in p.parts:
                        continue
                    _maybe_add(p)
                for p in export_dir.rglob("*.parquet.*"):
                    if "import_logs" in p.parts:
                        continue
                    _maybe_add(p)
        except Exception:
            pass

        # Additional common locations to check (Streamlit Cloud and local dev variations)
        candidate_dirs = []
        try:
            candidate_dirs.append(repo_root / "schedule_data")
            candidate_dirs.append(repo_root / "fantasy_football_data")
            candidate_dirs.append(repo_root / "fantasy_football_data" / "schedule_data")
            candidate_dirs.append(Path.cwd())
            candidate_dirs.append(Path.home())
            if export_dir:
                candidate_dirs.append(export_dir)
        except Exception:
            pass

        # Add a few container/mount fallbacks (only on POSIX containers) that Streamlit Cloud
        # commonly exposes. We limit deeper scans below to avoid long filesystem walks.
        extra_roots = []
        try:
            if os.name != "nt":
                for p in ("/mount", "/mount/src", "/mnt", "/home/appuser", "/home/adminuser", "/tmp", "/root", "/usr/src/app", "/workspace", "/srv", "/app"):
                    extra_roots.append(Path(p))
        except Exception:
            extra_roots = []

        # (2c) search these additional candidate dirs
        try:
            for d in candidate_dirs:
                if not d:
                    continue
                try:
                    if d.exists():
                        for p in d.rglob("*.parquet"):
                            if "import_logs" in p.parts:
                                continue
                            _maybe_add(p)
                        for p in d.rglob("*.parquet.*"):
                            if "import_logs" in p.parts:
                                continue
                            _maybe_add(p)
                except Exception:
                    continue
        except Exception:
            pass

        # (3) repo-wide fallback
        if not files:
            try:
                for p in repo_root.rglob("*.parquet"):
                    if "import_logs" in p.parts:
                        continue
                    _maybe_add(p)
                for p in repo_root.rglob("*.parquet.*"):
                    if "import_logs" in p.parts:
                        continue
                    _maybe_add(p)
            except Exception:
                pass

        # Final robust fallback: os.walk (case-insensitive extension match) in repo_root
        # This helps when the earlier pathlib searches miss files due to mount/symlink differences.
        if not files:
            try:
                for root, dirs, fnames in os.walk(str(repo_root)):
                    # skip import_logs folders quickly
                    if "import_logs" in Path(root).parts:
                        continue
                    for fn in fnames:
                        lower = fn.lower()
                        if lower.endswith(".parquet") or \
                           ".parquet." in lower or \
                           lower.endswith(".parq"):
                            p = Path(root) / fn
                            _maybe_add(p)
            except Exception:
                pass

        # If still nothing, perform a limited-depth scan under extra common roots
        if not files and extra_roots:
            MAX_FOUND = 300
            MAX_DEPTH = 5
            try:
                found_count = 0
                for root_base in extra_roots:
                    if found_count >= MAX_FOUND:
                        break
                    if not root_base.exists():
                        continue
                    for root, dirs, fnames in os.walk(str(root_base)):
                        # limit depth to avoid full filesystem traversal
                        try:
                            rel_depth = len(Path(root).relative_to(root_base).parts)
                        except Exception:
                            rel_depth = 0
                        if rel_depth > MAX_DEPTH:
                            # prune by clearing dirs so walk won't go deeper
                            dirs.clear()
                            continue
                        if "import_logs" in Path(root).parts:
                            continue
                        for fn in fnames:
                            lower = fn.lower()
                            if lower.endswith(".parquet") or ".parquet." in lower or lower.endswith(".parq"):
                                p = Path(root) / fn
                                _maybe_add(p)
                                found_count += 1
                                if found_count >= MAX_FOUND:
                                    break
                        if found_count >= MAX_FOUND:
                            break
            except Exception:
                pass

        def rank(p: Path) -> tuple[int, int, str]:
            try:
                if p.parent == data_dir:
                    a = 0
                elif data_dir in p.parents:
                    a = 1
                elif export_dir and (p.parent == export_dir or export_dir in p.parents):
                    a = 0
                else:
                    a = 2
            except Exception:
                a = 2
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
    if token:
        os.environ["MOTHERDUCK_TOKEN"] = token

    db = _slug(db_name, "l")
    sch = _slug(schema, "s")

    con = duckdb.connect("md:")
    con.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    con.execute(f"USE {db}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {sch}")

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
    if not INITIAL_IMPORT_SCRIPT.exists():
        st.error(f"❌ Initial import script not found at: {INITIAL_IMPORT_SCRIPT}")
        return False

    try:
        st.info("🚀 Starting initial data import... This may take several minutes.")

        # UI placeholders
        log_placeholder = st.empty()
        status_placeholder = st.empty()

        # Log file
        IMPORT_LOG_DIR = DATA_DIR / "import_logs"
        IMPORT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        import_log_path = IMPORT_LOG_DIR / f"initial_import_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if MOTHERDUCK_TOKEN:
            env["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN
        env["AUTO_CONFIRM"] = "1"
        # Tell the child import process where to write data. This can be overridden by
        # setting EXPORT_DATA_DIR in the environment (useful for Streamlit Cloud).
        try:
            env["EXPORT_DATA_DIR"] = str(DATA_DIR.resolve())
        except Exception:
            env["EXPORT_DATA_DIR"] = str(DATA_DIR)

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

            # Persist last stdout into session so the parent can parse paths even if
            # the import log file isn't readable directly from this environment.
            try:
                st.session_state["last_import_stdout"] = "\n".join(output_lines)
            except Exception:
                pass

            if process.returncode == 0:
                status_placeholder.success("Import finished successfully.")
                st.success("✅ Data import completed successfully!")
                st.write(f"Import log: {import_log_path}")
                try:
                    st.session_state["last_import_log"] = str(import_log_path)
                except Exception:
                    pass
                return True
            else:
                status_placeholder.error(f"Import failed (exit code {process.returncode}).")
                st.error(f"❌ Import failed with exit code {process.returncode}")
                st.code('\n'.join(output_lines))
                st.write(f"Import log: {import_log_path}")
                try:
                    st.session_state["last_import_log"] = str(import_log_path)
                except Exception:
                    pass
                return False

    except Exception as e:
        st.error(f"❌ Error running import: {e}")
        return False

# =========================
# Debug Helpers
# =========================
def _debug_list_parquet_locations(repo_root: Path, data_dir: Path, export_dir: Path | None = None, max_items: int = 200) -> list[tuple[Path, int]]:
    """Return a list of (path, size_bytes) for parquet-like files found under
    repo_root, data_dir, and export_dir. Case-insensitive and matches .parquet, .parq,
    and .parquet.* suffixes. Limits the returned list to max_items to avoid huge dumps.
    """
    found = []
    def _scan(base: Path):
        try:
            for root, dirs, files in os.walk(str(base)):
                if 'import_logs' in Path(root).parts:
                    continue
                for fn in files:
                    lower = fn.lower()
                    if lower.endswith('.parquet') or '.parquet.' in lower or lower.endswith('.parq'):
                        p = Path(root) / fn
                        try:
                            size = p.stat().st_size
                        except Exception:
                            size = -1
                        found.append((p, size))
                        if len(found) >= max_items:
                            return
        except Exception:
            return

    # scan the most likely places first
    for d in (data_dir, export_dir, repo_root):
        if not d:
            continue
        try:
            if d.exists():
                _scan(d)
        except Exception:
            continue

    # final fallback to repo_root if nothing yet
    if not found:
        try:
            _scan(repo_root)
        except Exception:
            pass

    return found

def _extract_parquet_paths_from_log(log_path: Path) -> list[Path]:
    """Scan a textual import log for absolute or repo-relative parquet paths and return existing Path objects."""
    paths: list[Path] = []
    try:
        import re
        txt = log_path.read_text(encoding='utf-8', errors='ignore')
        # crude regex: match strings that look like paths ending with .parquet or .parquet.*
        regex = r"([\w\-\./\\]+\.parquet(?:\.[a-zA-Z0-9]+)?)"
        for m in re.findall(regex, txt):
            try:
                p = Path(m)
                # try resolving relative to repo root as well
                if not p.is_absolute():
                    cand = ROOT_DIR / m
                    if cand.exists():
                        paths.append(cand)
                        continue
                if p.exists():
                    paths.append(p)
            except Exception:
                continue
    except Exception:
        pass
    return paths

def _extract_parquet_paths_from_text(text: str) -> list[Path]:
    """Extract candidate parquet paths from an arbitrary text blob; resolve relative to repo root if needed."""
    paths: list[Path] = []
    try:
        regex = r"([\w\-\./\\]+\.parquet(?:\.[a-zA-Z0-9]+)?)"
        for m in re.findall(regex, text or ""):
            try:
                p = Path(m)
                if not p.is_absolute():
                    cand = ROOT_DIR / m
                    if cand.exists():
                        paths.append(cand)
                        continue
                if p.exists():
                    paths.append(p)
            except Exception:
                continue
    except Exception:
        pass
    return paths

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
    if "error" in qp:
        st.error(f"❌ OAuth Error: {qp.get('error')}")
        if "error_description" in qp:
            st.error(f"Description: {qp.get('error_description')}")
        if st.button("Clear Error & Retry"):
            st.query_params.clear()
            st.rerun()
        return

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
                st.session_state.token_expiry = datetime.now(timezone.utc) + timedelta(seconds=token_data.get("expires_in", 3600))
                st.success("✅ Successfully connected!")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
        return

    if "access_token" in st.session_state:
        access_token = st.session_state.access_token
        st.success("🔐 Connected to Yahoo Fantasy!")

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
                        # Provide a non-empty label (hidden) to satisfy Streamlit accessibility warnings.
                        selected_league_name = st.radio("league_selection", league_names, key="league_radio", label_visibility="collapsed")
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
                            st.session_state.league_info = selected_league

                            with st.spinner("Saving OAuth credentials..."):
                                oauth_file = save_oauth_token(st.session_state.token_data, selected_league)
                                st.success(f"✅ OAuth credentials saved to: {oauth_file}")

                            if MOTHERDUCK_TOKEN:
                                os.environ["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN

                            # Ensure parent process exposes EXPORT_DATA_DIR so downstream
                            # parquet discovery can look in the same path the child used.
                            try:
                                os.environ["EXPORT_DATA_DIR"] = str(DATA_DIR.resolve())
                            except Exception:
                                os.environ["EXPORT_DATA_DIR"] = str(DATA_DIR)

                            if run_initial_import():
                                st.success("🎉 All done! Your league data has been imported.")

                                # --------- COLLECT FILES (for upload + downloads) ----------
                                repo_root = ROOT_DIR
                                files = collect_parquet_candidates(repo_root, DATA_DIR)

                                # --- EXPLICIT DEBUG SCAN (os.walk-based) ---
                                try:
                                    ed = os.environ.get('EXPORT_DATA_DIR')
                                    export_path = Path(ed) if ed else None
                                    debug_found = _debug_list_parquet_locations(repo_root, DATA_DIR, export_path)
                                    st.write("**Debug: explicit os.walk scan results**")
                                    if debug_found:
                                        st.write(f"Found {len(debug_found)} parquet-like file(s) via explicit scan:")
                                        for p, size in debug_found:
                                            try:
                                                rel = p.relative_to(ROOT_DIR)
                                            except Exception:
                                                rel = p
                                            try:
                                                st.write(f"- `{rel}` ({size/1024:.1f} KB)")
                                            except Exception:
                                                st.write(f"- `{rel}`")
                                        if not files:
                                            st.warning("collect_parquet_candidates() returned no files but explicit scan found files — there may be a path/mount mismatch. Check file permissions and where the import process wrote data.")
                                    else:
                                        st.write("No parquet-like files found by explicit scan.")
                                except Exception as e:
                                    st.write(f"Debug scan failed: {e}")

                                # If no files discovered, also try to parse the import log for paths
                                try:
                                    if not files and st.session_state.get("last_import_log"):
                                        logpath = Path(st.session_state.get("last_import_log"))
                                        if logpath.exists():
                                            st.write("(Attempting to recover parquet paths from import log)")
                                            log_paths = _extract_parquet_paths_from_log(logpath)
                                            if log_paths:
                                                st.write(f"Found {len(log_paths)} path(s) in import log:")
                                                for lp in log_paths:
                                                    try:
                                                        st.write(f"- `{lp}`")
                                                        if lp.exists():
                                                            files.append(lp)
                                                    except Exception:
                                                        pass
                                            else:
                                                st.write("No parquet paths found in import log.")
                                except Exception:
                                    pass

                                # Also try scanning stdout captured from the import process
                                try:
                                    if not files and st.session_state.get("last_import_stdout"):
                                        st.write("(Attempting to recover parquet paths from import stdout)")
                                        txt = st.session_state.get("last_import_stdout", "")
                                        txt_paths = _extract_parquet_paths_from_text(txt)
                                        if txt_paths:
                                            st.write(f"Found {len(txt_paths)} path(s) in import stdout:")
                                            for tp in txt_paths:
                                                try:
                                                    st.write(f"- `{tp}`")
                                                    if tp.exists():
                                                        files.append(tp)
                                                except Exception:
                                                    pass
                                        else:
                                            st.write("No parquet paths found in import stdout.")
                                except Exception:
                                    pass

                                # DEBUG: show paths and files found to help diagnose missing uploads
                                try:
                                    st.write("**Debug: paths used for parquet discovery (resolved)**")
                                    try:
                                        st.write(f"repo_root: `{repo_root.resolve()}`")
                                    except Exception:
                                        st.write(f"repo_root: `{repo_root}`")
                                    try:
                                        st.write(f"data_dir: `{DATA_DIR.resolve()}`")
                                    except Exception:
                                        st.write(f"data_dir: `{DATA_DIR}`")
                                    st.write(f"EXPORT_DATA_DIR env: `{os.environ.get('EXPORT_DATA_DIR')}`")
                                    if files:
                                        st.write(f"Found {len(files)} parquet file(s):")
                                        for pf in files:
                                            try:
                                                rel = pf.relative_to(ROOT_DIR)
                                            except Exception:
                                                rel = pf
                                            try:
                                                size_kb = pf.stat().st_size / 1024
                                                st.write(f"- `{rel}` ({size_kb:.1f} KB)")
                                            except Exception:
                                                st.write(f"- `{rel}`")
                                    else:
                                        st.warning("(Debug) No parquet files were discovered by collect_parquet_candidates().")

                                    # Small helper UI to dump raw import stdout/log so the user can copy exact lines
                                    try:
                                        if st.button("Show raw import output (stdout + log)"):
                                            st.write("--- BEGIN: import stdout (last run) ---")
                                            stdout_txt = st.session_state.get("last_import_stdout", "(none)")
                                            # cap to reasonable size
                                            if len(stdout_txt) > 20000:
                                                st.code(stdout_txt[:20000] + "\n... (truncated) ...")
                                            else:
                                                st.code(stdout_txt or "(none)")

                                            st.write("--- BEGIN: import log file (if readable) ---")
                                            logpath_str = st.session_state.get("last_import_log")
                                            if logpath_str:
                                                logpath = Path(logpath_str)
                                                try:
                                                    if logpath.exists():
                                                        txt = logpath.read_text(encoding='utf-8', errors='ignore')
                                                        if len(txt) > 20000:
                                                            st.code(txt[:20000] + "\n... (truncated) ...")
                                                        else:
                                                            st.code(txt)
                                                    else:
                                                        st.write(f"Import log file not found at: {logpath}")
                                                except Exception as e:
                                                    st.write(f"Could not read import log: {e}")
                                            else:
                                                st.write("No import log path recorded in session.")
                                    except Exception:
                                        pass

                                except Exception:
                                    pass

                                # --- Upload to MotherDuck for EVERY season this league exists ---
                                if MOTHERDUCK_TOKEN:
                                    st.info("🦆 Uploading data to MotherDuck (all seasons for this league)...")

                                    league_info  = st.session_state.get("league_info", {})
                                    league_name  = league_info.get("name", "league")
                                    access_token = st.session_state.get("access_token")
                                    all_games    = extract_football_games(st.session_state.get("games_data", {}))

                                    season_list = seasons_for_league_name(access_token, all_games, league_name)
                                    selected_season = str(league_info.get("season", "")).strip()
                                    if selected_season and selected_season not in season_list:
                                        season_list.append(selected_season)

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
                                st.info("⚠️ On your host, these files may be temporary. Download them now!")

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
                                                label = pf.relative_to(ROOT_DIR) if pf.is_relative_to(ROOT_DIR) else pf.name
                                                st.download_button(
                                                    f"📥 {label}",
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
