import os
import duckdb
from pathlib import Path
from typing import Optional

import pandas as pd


def df_from_md_or_parquet(table: str, parquet_path: Path, **kwargs) -> pd.DataFrame:
    """Return a DataFrame from MotherDuck (via duckdb "md:") if MOTHERDUCK_TOKEN is set,
    otherwise fall back to reading a local Parquet file.

    Supports optional kwargs forwarded to pandas.read_parquet. Special-case kwarg:
      - columns: list of column names to select

    The function is tolerant about table naming: it accepts any of the following
    - 'matchup' (table name only) -> will try to find a matching table in any DB/schema
    - 'public.matchup' (schema.table)
    - 'db_name.public.matchup' (db.schema.table)
    """
    # If no MD token, fall back directly
    if not os.getenv("MOTHERDUCK_TOKEN"):
        return pd.read_parquet(parquet_path, **kwargs)

    # Extract columns kwarg and remove from kwargs for duckdb path
    cols = kwargs.pop('columns', None)

    con = None
    try:
        con = duckdb.connect("md:")
        # 1) Try direct select (user provided fully-qualified name)
        try:
            if cols:
                cols_sql = ", ".join([f'"{c}"' for c in cols])
                return con.execute(f"SELECT {cols_sql} FROM {table}").df()
            return con.execute(f"SELECT * FROM {table}").df()
        except Exception:
            pass

        # 2) If table contains dots, try USE db and select schema.table
        parts = table.split('.')
        if len(parts) == 3:
            db, schema, tbl = parts
            try:
                con.execute(f"USE {db}")
                if cols:
                    cols_sql = ", ".join([f'"{c}"' for c in cols])
                    return con.execute(f"SELECT {cols_sql} FROM {schema}.{tbl}").df()
                return con.execute(f"SELECT * FROM {schema}.{tbl}").df()
            except Exception:
                pass

        # 3) If only table or schema.table provided, try to locate it in information_schema
        search_name = parts[-1]
        try:
            rows = con.execute(
                "SELECT table_catalog, table_schema, table_name FROM information_schema.tables WHERE lower(table_name)=lower(?)",
                [search_name]
            ).fetchall()
            if rows:
                # prefer a public schema if available
                for catalog, schema, name in rows:
                    candidate = None
                    try:
                        if catalog:
                            con.execute(f"USE {catalog}")
                        candidate = f"{schema}.{name}" if schema else name
                        if cols:
                            cols_sql = ", ".join([f'"{c}"' for c in cols])
                            return con.execute(f"SELECT {cols_sql} FROM {candidate}").df()
                        return con.execute(f"SELECT * FROM {candidate}").df()
                    except Exception:
                        continue
        except Exception:
            pass

    except Exception:
        # any MD connection error -> fall back to parquet
        pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

    # Final fallback: local parquet
    return pd.read_parquet(parquet_path, **kwargs)
