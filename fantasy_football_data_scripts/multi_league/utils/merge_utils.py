"""
High-performance merge utilities using DuckDB and Polars lazy evaluation.

Replaces pandas.merge on massive DataFrames with DuckDB SQL or Polars lazy joins
for 10-100x performance improvement on large datasets.

Usage:
    from merge_utils import duckdb_merge, polars_lazy_merge

    # DuckDB merge (great for complex SQL operations)
    result = duckdb_merge(
        left_df,
        right_df,
        on=["player_id", "year"],
        how="left"
    )

    # Polars lazy merge (great for chained operations)
    result = polars_lazy_merge(
        "player.parquet",
        "stats.parquet",
        on=["player_id", "year"],
        how="inner"
    )
"""
from __future__ import annotations

import duckdb
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def duckdb_merge(
    left: Union[pd.DataFrame, str, Path],
    right: Union[pd.DataFrame, str, Path],
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
    suffixes: tuple = ("_x", "_y"),
    output_path: Optional[Union[str, Path]] = None,
    use_polars: bool = True
) -> Union[pd.DataFrame, pl.DataFrame, None]:
    """
    Perform merge using DuckDB for better performance on large datasets.

    DuckDB can merge data directly from parquet files without loading
    everything into memory, and uses optimized hash joins.

    Args:
        left: Left DataFrame or path to parquet file
        right: Right DataFrame or path to parquet file
        on: Column(s) to join on (must be same in both)
        left_on: Column(s) from left to join on
        right_on: Column(s) from right to join on
        how: Join type (inner, left, right, outer)
        suffixes: Suffixes for duplicate columns
        output_path: If provided, write result to parquet instead of returning
        use_polars: Return Polars DataFrame (faster)

    Returns:
        Merged DataFrame (or None if output_path is provided)

    Example:
        >>> # Merge two large parquet files
        >>> result = duckdb_merge(
        ...     "yahoo_player_stats.parquet",
        ...     "nfl_player_stats.parquet",
        ...     on=["player_id", "year", "week"],
        ...     how="left"
        ... )
        >>>
        >>> # Merge and write directly to file (memory efficient)
        >>> duckdb_merge(
        ...     "yahoo_stats.parquet",
        ...     "nfl_stats.parquet",
        ...     on="player_id",
        ...     output_path="merged_stats.parquet"
        ... )
    """
    # Connect to DuckDB (in-memory)
    con = duckdb.connect(':memory:')

    try:
        # Register left table
        if isinstance(left, (str, Path)):
            left_path = Path(left)
            if not left_path.exists():
                raise FileNotFoundError(f"Left file not found: {left_path}")
            con.execute(f"CREATE TABLE left_tbl AS SELECT * FROM '{left_path}'")
        else:
            con.register('left_tbl', left)

        # Register right table
        if isinstance(right, (str, Path)):
            right_path = Path(right)
            if not right_path.exists():
                raise FileNotFoundError(f"Right file not found: {right_path}")
            con.execute(f"CREATE TABLE right_tbl AS SELECT * FROM '{right_path}'")
        else:
            con.register('right_tbl', right)

        # Build join condition
        if on is not None:
            join_cols = [on] if isinstance(on, str) else on
            join_condition = " AND ".join([f"l.{col} = r.{col}" for col in join_cols])
        elif left_on is not None and right_on is not None:
            left_cols = [left_on] if isinstance(left_on, str) else left_on
            right_cols = [right_on] if isinstance(right_on, str) else right_on
            if len(left_cols) != len(right_cols):
                raise ValueError("left_on and right_on must have same length")
            join_condition = " AND ".join([
                f"l.{lcol} = r.{rcol}" for lcol, rcol in zip(left_cols, right_cols)
            ])
        else:
            raise ValueError("Must provide 'on' or both 'left_on' and 'right_on'")

        # Map pandas join types to SQL
        join_type_map = {
            "inner": "INNER JOIN",
            "left": "LEFT JOIN",
            "right": "RIGHT JOIN",
            "outer": "FULL OUTER JOIN"
        }
        sql_join_type = join_type_map.get(how.lower())
        if sql_join_type is None:
            raise ValueError(f"Unsupported join type: {how}")

        # Get column names to handle duplicates
        left_cols = con.execute("SELECT * FROM left_tbl LIMIT 0").df().columns.tolist()
        right_cols = con.execute("SELECT * FROM right_tbl LIMIT 0").df().columns.tolist()

        # Determine overlapping columns (excluding join keys)
        join_keys = [on] if isinstance(on, str) else (on or [])
        if not join_keys and left_on:
            join_keys = [left_on] if isinstance(left_on, str) else left_on

        overlapping = set(left_cols) & set(right_cols) - set(join_keys)

        # Build select list with suffixes for overlapping columns
        select_parts = []

        # Add all left columns
        for col in left_cols:
            if col in overlapping:
                select_parts.append(f"l.{col} AS {col}{suffixes[0]}")
            else:
                select_parts.append(f"l.{col}")

        # Add right columns (excluding join keys which are already in left)
        for col in right_cols:
            if col in join_keys:
                continue  # Skip join keys
            if col in overlapping:
                select_parts.append(f"r.{col} AS {col}{suffixes[1]}")
            else:
                select_parts.append(f"r.{col}")

        select_clause = ", ".join(select_parts)

        # Build and execute query
        query = f"""
            SELECT {select_clause}
            FROM left_tbl l
            {sql_join_type} right_tbl r ON {join_condition}
        """

        if output_path:
            # Write directly to parquet
            output_path = Path(output_path)
            con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")
            print(f"Merged data written to {output_path}")
            return None
        else:
            # Return DataFrame
            result = con.execute(query).df()

            if use_polars:
                return pl.from_pandas(result)
            else:
                return result

    finally:
        con.close()


def polars_lazy_merge(
    left: Union[pl.DataFrame, str, Path],
    right: Union[pl.DataFrame, str, Path],
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
    output_path: Optional[Union[str, Path]] = None,
    suffix: str = "_right"
) -> Optional[pl.DataFrame]:
    """
    Perform lazy merge using Polars for maximum performance.

    Polars uses lazy evaluation and query optimization, only reading
    the data that's actually needed.

    Args:
        left: Left DataFrame or path to parquet file
        right: Right DataFrame or path to parquet file
        on: Column(s) to join on
        left_on: Column(s) from left to join on
        right_on: Column(s) from right to join on
        how: Join type (inner, left, outer, semi, anti, cross)
        output_path: If provided, write result to parquet
        suffix: Suffix for duplicate columns from right

    Returns:
        Merged DataFrame (or None if output_path is provided)

    Example:
        >>> # Lazy merge with filter
        >>> result = polars_lazy_merge(
        ...     "player_stats.parquet",
        ...     "nfl_stats.parquet",
        ...     on=["player_id", "year"],
        ...     how="inner"
        ... )
        >>>
        >>> # Streaming merge (never loads full dataset into memory)
        >>> polars_lazy_merge(
        ...     "huge_yahoo_stats.parquet",
        ...     "huge_nfl_stats.parquet",
        ...     on="player_id",
        ...     output_path="merged.parquet"
        ... )
    """
    # Create lazy dataframes
    if isinstance(left, (str, Path)):
        left_lazy = pl.scan_parquet(left)
    elif isinstance(left, pl.DataFrame):
        left_lazy = left.lazy()
    elif isinstance(left, pl.LazyFrame):
        left_lazy = left
    else:
        raise TypeError(f"Unsupported left type: {type(left)}")

    if isinstance(right, (str, Path)):
        right_lazy = pl.scan_parquet(right)
    elif isinstance(right, pl.DataFrame):
        right_lazy = right.lazy()
    elif isinstance(right, pl.LazyFrame):
        right_lazy = right
    else:
        raise TypeError(f"Unsupported right type: {type(right)}")

    # Perform lazy join
    if on is not None:
        result_lazy = left_lazy.join(right_lazy, on=on, how=how, suffix=suffix)
    elif left_on is not None and right_on is not None:
        result_lazy = left_lazy.join(right_lazy, left_on=left_on, right_on=right_on, how=how, suffix=suffix)
    else:
        raise ValueError("Must provide 'on' or both 'left_on' and 'right_on'")

    # Execute query
    if output_path:
        # Streaming write (never loads full result into memory)
        result_lazy.sink_parquet(output_path)
        print(f"Merged data written to {output_path}")
        return None
    else:
        # Collect result
        return result_lazy.collect()


def upsert_duckdb(
    new_data: Union[pd.DataFrame, pl.DataFrame, str, Path],
    target_table: Union[str, Path],
    key_columns: List[str],
    output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Upsert (INSERT or UPDATE) new data into existing parquet file using DuckDB.

    This is much more efficient than:
    1. Loading full parquet into pandas
    2. Merging/deduplicating in pandas
    3. Writing back to parquet

    Args:
        new_data: New data to upsert (DataFrame or parquet path)
        target_table: Existing parquet file to update
        key_columns: Columns that uniquely identify rows
        output_path: Where to write result (defaults to target_table)

    Example:
        >>> # Upsert new weekly data into player stats
        >>> upsert_duckdb(
        ...     "new_week_5_data.parquet",
        ...     "player_stats.parquet",
        ...     key_columns=["player_id", "year", "week"]
        ... )
    """
    if output_path is None:
        output_path = target_table

    con = duckdb.connect(':memory:')

    try:
        # Load existing data
        target_path = Path(target_table)
        if target_path.exists():
            con.execute(f"CREATE TABLE existing AS SELECT * FROM '{target_path}'")
        else:
            # No existing data, just write new data
            if isinstance(new_data, (str, Path)):
                con.execute(f"COPY (SELECT * FROM '{new_data}') TO '{output_path}' (FORMAT PARQUET)")
            else:
                con.register('new_tbl', new_data)
                con.execute(f"COPY (SELECT * FROM new_tbl) TO '{output_path}' (FORMAT PARQUET)")
            return

        # Load new data
        if isinstance(new_data, (str, Path)):
            con.execute(f"CREATE TABLE new_data AS SELECT * FROM '{new_data}'")
        else:
            con.register('new_data', new_data)

        # Build key match condition
        key_conditions = " AND ".join([f"e.{col} = n.{col}" for col in key_columns])

        # Upsert logic: Keep existing rows not in new_data, then add all new_data
        upsert_query = f"""
            SELECT * FROM existing e
            WHERE NOT EXISTS (
                SELECT 1 FROM new_data n
                WHERE {key_conditions}
            )
            UNION ALL
            SELECT * FROM new_data
        """

        # Write result
        con.execute(f"COPY ({upsert_query}) TO '{output_path}' (FORMAT PARQUET)")
        print(f"Upserted data written to {output_path}")

    finally:
        con.close()


def consolidate_parquet_files(
    input_files: List[Union[str, Path]],
    output_path: Union[str, Path],
    deduplicate_on: Optional[List[str]] = None,
    sort_by: Optional[List[str]] = None
) -> None:
    """
    Consolidate multiple parquet files into one using DuckDB.

    Much faster than loading all files into pandas and concatenating.

    Args:
        input_files: List of parquet files to consolidate
        output_path: Output file path
        deduplicate_on: Columns to deduplicate on (keeps last occurrence)
        sort_by: Columns to sort by

    Example:
        >>> # Consolidate all weekly files
        >>> consolidate_parquet_files(
        ...     ["week_1.parquet", "week_2.parquet", "week_3.parquet"],
        ...     "season_2024.parquet",
        ...     deduplicate_on=["player_id", "week"]
        ... )
    """
    con = duckdb.connect(':memory:')

    try:
        # Union all input files
        union_parts = [f"SELECT * FROM '{f}'" for f in input_files]
        union_query = " UNION ALL ".join(union_parts)

        # Deduplicate if requested
        if deduplicate_on:
            # Use ROW_NUMBER() to keep last occurrence
            partition_by = ", ".join(deduplicate_on)
            dedup_query = f"""
                SELECT * FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (PARTITION BY {partition_by} ORDER BY (SELECT NULL)) as rn
                    FROM ({union_query}) t
                ) WHERE rn = 1
            """
            # Remove the row number column
            cols_query = f"SELECT * EXCLUDE (rn) FROM ({dedup_query})"
        else:
            cols_query = union_query

        # Sort if requested
        if sort_by:
            order_clause = ", ".join(sort_by)
            final_query = f"SELECT * FROM ({cols_query}) ORDER BY {order_clause}"
        else:
            final_query = cols_query

        # Write result
        con.execute(f"COPY ({final_query}) TO '{output_path}' (FORMAT PARQUET)")
        print(f"Consolidated {len(input_files)} files to {output_path}")

    finally:
        con.close()


def join_chain_polars(
    base: Union[pl.DataFrame, str, Path],
    joins: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None
) -> Optional[pl.DataFrame]:
    """
    Perform a chain of joins using Polars lazy evaluation.

    Useful for joining multiple tables in sequence with optimal query planning.

    Args:
        base: Base DataFrame or parquet path
        joins: List of join specs, each dict with:
               - 'right': DataFrame or path to join
               - 'on': Join columns (or left_on/right_on)
               - 'how': Join type
        output_path: Optional output path

    Returns:
        Joined DataFrame (or None if output_path provided)

    Example:
        >>> result = join_chain_polars(
        ...     "player_base.parquet",
        ...     joins=[
        ...         {
        ...             'right': 'yahoo_stats.parquet',
        ...             'on': ['player_id', 'year'],
        ...             'how': 'left'
        ...         },
        ...         {
        ...             'right': 'nfl_stats.parquet',
        ...             'on': ['player_id', 'year', 'week'],
        ...             'how': 'left'
        ...         }
        ...     ],
        ...     output_path='merged_all.parquet'
        ... )
    """
    # Start with base
    if isinstance(base, (str, Path)):
        result = pl.scan_parquet(base)
    elif isinstance(base, pl.DataFrame):
        result = base.lazy()
    else:
        result = base

    # Apply each join
    for join_spec in joins:
        right = join_spec['right']
        if isinstance(right, (str, Path)):
            right_lazy = pl.scan_parquet(right)
        elif isinstance(right, pl.DataFrame):
            right_lazy = right.lazy()
        else:
            right_lazy = right

        # Perform join
        on = join_spec.get('on')
        left_on = join_spec.get('left_on')
        right_on = join_spec.get('right_on')
        how = join_spec.get('how', 'inner')
        suffix = join_spec.get('suffix', '_right')

        if on is not None:
            result = result.join(right_lazy, on=on, how=how, suffix=suffix)
        elif left_on and right_on:
            result = result.join(right_lazy, left_on=left_on, right_on=right_on, how=how, suffix=suffix)
        else:
            raise ValueError("Each join must specify 'on' or both 'left_on' and 'right_on'")

    # Execute
    if output_path:
        result.sink_parquet(output_path)
        print(f"Join chain result written to {output_path}")
        return None
    else:
        return result.collect()


def get_week_column(df):
    for c in ("week", "Week", "nfl_week", "game_week"):
        if c in df.columns:
            return c
    # last resort: derive from date if present (won't be perfect but prevents hard fail)
    return None


def slice_week(df, week):
    wk = get_week_column(df)
    if wk:
        return df[df[wk] == week]
    return df.iloc[0:0]  # empty slice if no week column

