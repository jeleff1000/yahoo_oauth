from streamlit.connections import ExperimentalBaseConnection
import streamlit as st
import duckdb
import os


def get_selected_league_db() -> str:
    """Get the currently selected league database from session state or environment."""
    if "selected_league_db" in st.session_state:
        return st.session_state.selected_league_db
    return os.environ.get("SELECTED_LEAGUE_DB", "kmffl")


class MotherDuckConnection(ExperimentalBaseConnection[duckdb.DuckDBPyConnection]):
    def _connect(self, **kwargs) -> duckdb.DuckDBPyConnection:
        # Get token from secrets
        token = (
            kwargs.pop("motherduck_token", None)
            or self._secrets.get("motherduck_token")
            or st.secrets.get("connections", {}).get("motherduck", {}).get("motherduck_token")
            or st.secrets.get("motherduck_token")
            or os.getenv("MOTHERDUCK_TOKEN")
        )

        if not token:
            raise ValueError("MotherDuck token not found. Check your secrets configuration.")

        # Get the selected league database - this is now the primary source of truth
        selected_db = get_selected_league_db()

        # Get share path from secrets (used as fallback for backwards compatibility)
        share_path = (
            kwargs.pop("share_path", None)
            or self._secrets.get("share_path")
            or st.secrets.get("connections", {}).get("motherduck", {}).get("share_path")
            or st.secrets.get("share_path")
            or os.getenv("MD_ATTACH_URL")
        )

        # Determine alias - use selected_db if available, otherwise fall back to config
        alias = selected_db if selected_db else (
            kwargs.pop("alias", None)
            or self._secrets.get("alias")
            or st.secrets.get("connections", {}).get("motherduck", {}).get("alias")
            or "kmffl"
        )

        # Create connection once and reuse
        try:
            con = duckdb.connect()
            con.execute("INSTALL motherduck")
            con.execute("LOAD motherduck")
            con.execute(f"SET motherduck_token = '{token}'")

            # Check if already attached
            attached_dbs = [row[1] for row in con.execute("PRAGMA database_list").fetchall()]

            # For the selected database, connect directly to MotherDuck
            # This allows access to any database without needing a share_path
            if alias not in attached_dbs:
                # Try direct connection first (works for user's own databases)
                try:
                    con.execute(f"ATTACH 'md:{alias}' AS {alias}")
                except Exception:
                    # Fall back to share_path if direct connection fails
                    if share_path and isinstance(share_path, str) and share_path.startswith("md:"):
                        con.execute(f"ATTACH '{share_path}' AS {alias} (READ_ONLY)")
                    else:
                        raise ValueError(f"Could not connect to database '{alias}'. No share_path configured.")

            # Quick connection test - try to verify the database has expected tables
            try:
                con.execute(f"SELECT 1 FROM {alias}.matchup LIMIT 1")
            except Exception:
                # If matchup table doesn't exist, still return connection
                # The database might be valid but have different table names
                pass

            return con

        except Exception as e:
            raise ConnectionError(f"Failed to connect to MotherDuck: {e}")

    def query(self, sql: str, ttl: int | None = 600, **kwargs):
        """Execute query with retry logic for catalog errors."""
        import time

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Execute directly without nested cache
                result = self._instance.execute(sql).df()
                return result

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Check if it's a catalog error
                is_catalog_error = "catalog" in error_msg or "remote catalog has changed" in error_msg

                if attempt < max_retries - 1:
                    # Clear streamlit caches
                    st.cache_data.clear()
                    st.cache_resource.clear()

                    if is_catalog_error:
                        # For catalog errors, force complete reconnection
                        if hasattr(self, '_instance') and self._instance:
                            try:
                                self._instance.close()
                            except:
                                pass

                        # Force recreation of connection
                        try:
                            self._instance = self._connect()
                        except Exception as conn_error:
                            # If reconnection fails, try once more after a pause
                            time.sleep(1)
                            try:
                                self._instance = self._connect()
                            except:
                                # If still failing, continue to next attempt
                                pass

                        time.sleep(0.5)
                    else:
                        # For non-catalog errors, just wait and retry
                        time.sleep(0.3)

                    continue

        # If we exhausted all retries, raise the last error
        if last_error:
            raise last_error
        else:
            raise Exception(f"Query failed after {max_retries} attempts")

    def reset(self):
        """Reset the connection"""
        if hasattr(self, '_instance') and self._instance:
            try:
                self._instance.close()
            except:
                pass
        super().reset()