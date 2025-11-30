import pandas as pd

NUMERIC_INT_COLS = {"week", "year"}
NUMERIC_FLOAT_COLS = {"points", "team_points", "opponent_points", "projected_points"}


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Arrow-friendly dtypes (no mixed types / stray strings)."""
    out = df.copy()

    # Drop obvious non-numeric rows like 'Total' that sneak into numeric columns
    if "week" in out.columns:
        out = out[out["week"].astype(str).str.fullmatch(r"\d+")]  # keep only digits

    # Coerce common numerics
    for c in out.columns:
        if c in NUMERIC_INT_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
        elif c in NUMERIC_FLOAT_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Strings should be real string dtype (not 'object')
    for c in out.select_dtypes("object").columns:
        out[c] = out[c].astype("string")

    return out
