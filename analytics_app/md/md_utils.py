# Fallback shim for md.md_utils
# Try to import an existing implementation from the streamlit_ui package first
try:
    from streamlit_ui.md.md_utils import df_from_md_or_parquet  # type: ignore
    __all__ = ["df_from_md_or_parquet"]
except Exception:
    from typing import Any
    import pandas as pd
    import os
    import glob

    def _read_md_table(path: str) -> pd.DataFrame:
        """
        Parse the first pipe-style Markdown table found in the file and return a DataFrame.
        This supports simple GitHub-style tables like:

        | col1 | col2 |
        | ---- | ---- |
        | a    | 1    |

        Returns empty DataFrame if no table is found.
        """
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip('\n') for ln in f]

        # Find a header line followed immediately by a separator line with dashes
        for i in range(len(lines) - 1):
            header = lines[i].strip()
            sep = lines[i + 1].strip()
            if '|' in header and set(sep.replace('|', '').strip()) <= set('- :') and any(c == '-' for c in sep):
                # collect subsequent table rows
                rows = []
                j = i + 2
                while j < len(lines) and '|' in lines[j]:
                    rows.append(lines[j])
                    j += 1

                # Split header and rows by | and strip empties at ends
                def split_row(r: str):
                    parts = [p.strip() for p in r.strip().split('|')]
                    # remove leading/trailing empty strings from leading/trailing pipes
                    if parts and parts[0] == '':
                        parts = parts[1:]
                    if parts and parts[-1] == '':
                        parts = parts[:-1]
                    return parts

                headers = split_row(header)
                data = [split_row(r) for r in rows]

                # Normalize row lengths
                maxlen = max((len(r) for r in data), default=0)
                if len(headers) < maxlen:
                    headers = headers + [f"col_{k}" for k in range(len(headers), maxlen)]
                normalized = [row + [''] * (len(headers) - len(row)) for row in data]

                df = pd.DataFrame(normalized, columns=headers)
                # Try to infer dtypes
                try:
                    df = df.infer_objects()
                    for c in df.columns:
                        # convert numeric-like columns
                        df[c] = pd.to_numeric(df[c], errors='ignore')
                except Exception:
                    pass
                return df

        # No table found
        return pd.DataFrame()

    def df_from_md_or_parquet(*args, **kwargs: Any) -> pd.DataFrame:
        """
        Load a DataFrame from a parquet file, CSV, or the first Markdown pipe table found.

        Backwards-compatible calling styles supported:
        - df_from_md_or_parquet(path_or_dir)
        - df_from_md_or_parquet(name, path_or_dir)  # name is ignored by this shim but accepted for callers

        Behavior:
        - If the path_or_dir is a directory, searches for parquet -> csv -> *.md and uses the first match.
        - If the path ends with .parquet or .pq -> uses pd.read_parquet
        - If the path ends with .csv -> pd.read_csv
        - If the path ends with .md or .markdown -> attempts to parse the first pipe table found.

        Pass-through kwargs are forwarded to the pandas reader (for csv/parquet).
        Raises FileNotFoundError when no suitable file is found and ValueError when parsing fails.
        """
        # Support both df_from_md_or_parquet(path) and df_from_md_or_parquet(name, path)
        resource_name = None
        if len(args) == 1:
            path_or_dir = args[0]
            # Remove any resource_name passed in kwargs so it doesn't reach pandas
            resource_name = kwargs.pop("resource_name", None)
        elif len(args) >= 2:
            # First arg may be a logical name (e.g. 'matchup'), second arg is the path
            resource_name = args[0]
            path_or_dir = args[1]
            # Remove any resource_name in kwargs as well to avoid passing it through
            kwargs.pop("resource_name", None)
        else:
            raise TypeError("df_from_md_or_parquet() missing required positional argument: path_or_dir")

        # Expand user and absolve path
        p = os.path.expanduser(path_or_dir)

        # If directory, look for common data files
        if os.path.isdir(p):
            # prefer parquet
            candidates = []
            candidates.extend(sorted(glob.glob(os.path.join(p, "*.parquet"))))
            candidates.extend(sorted(glob.glob(os.path.join(p, "*.parq"))))
            candidates.extend(sorted(glob.glob(os.path.join(p, "*.csv"))))
            candidates.extend(sorted(glob.glob(os.path.join(p, "*.md"))))
            if not candidates:
                raise FileNotFoundError(f"No parquet/csv/md files found in directory: {p}")
            p = candidates[0]

        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

        lower = p.lower()
        try:
            if lower.endswith(".parquet") or lower.endswith(".parq"):
                return pd.read_parquet(p, **kwargs)
            if lower.endswith(".csv"):
                return pd.read_csv(p, **kwargs)
            if lower.endswith(".md") or lower.endswith(".markdown"):
                df = _read_md_table(p)
                if df.empty:
                    raise ValueError(f"No Markdown table found in file: {p}")
                return df

            # As a last resort try parquet then csv
            try:
                return pd.read_parquet(p, **kwargs)
            except Exception:
                return pd.read_csv(p, **kwargs)
        except Exception as e:
            # Re-wrap for clarity
            raise ValueError(f"Failed to load data from {p}: {e}") from e

    __all__ = ["df_from_md_or_parquet"]
