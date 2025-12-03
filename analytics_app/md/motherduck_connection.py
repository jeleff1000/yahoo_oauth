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

    def _is_connection_valid(self) -> bool:
        """Check if the current connection is still valid."""
        try:
            if self._instance is None:
                return False
            # Try a simple query to test connection
            self._instance.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def _force_reconnect(self):
        """Force a reconnection by resetting the connection."""
        try:
            # Try to close existing connection
            if self._instance is not None:
                try:
                    self._instance.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Use parent's reset method to clear the cached connection
        try:
            super().reset()
        except Exception:
            pass

    def query(self, sql: str, ttl: int | None = 600, **kwargs):
        """Execute query with retry logic for connection and catalog errors."""
        import time

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Check if connection is valid, reconnect if not
                if not self._is_connection_valid():
                    self._force_reconnect()

                # Execute query - _instance will be recreated by parent if needed
                result = self._instance.execute(sql).df()
                return result

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Check error types
                is_connection_error = "connection" in error_msg and "closed" in error_msg
                is_catalog_error = "catalog" in error_msg or "remote catalog has changed" in error_msg
                is_setter_error = "no setter" in error_msg

                if attempt < max_retries - 1:
                    # For connection, catalog, or setter errors, force reconnection
                    if is_connection_error or is_catalog_error or is_setter_error:
                        self._force_reconnect()

                        # Clear streamlit caches
                        try:
                            st.cache_data.clear()
                            st.cache_resource.clear()
                        except Exception:
                            pass

                        # Wait before retry
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        # For other errors, just wait and retry
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