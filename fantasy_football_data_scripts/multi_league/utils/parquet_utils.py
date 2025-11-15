"""
Partitioned parquet writing and reading utilities.

Ensures all large datasets are written partitioned by year (and week if large)
so that joins/reads scan less data.

Usage:
    from parquet_utils import write_partitioned, read_partitioned

    # Write partitioned by year
    write_partitioned(
        df,
        "player.parquet",
        partition_cols=["year"]
    )

    # Read only 2024 data
    df = read_partitioned(
        "player.parquet",
        filters=[("year", "==", 2024)]
    )
"""
from __future__ import annotations

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def write_partitioned(
    df: Union[pd.DataFrame, pl.DataFrame],
    output_path: Union[str, Path],
    partition_cols: List[str],
    use_polars: bool = True,
    compression: str = "snappy",
    **kwargs
) -> None:
    """
    Write DataFrame partitioned by specified columns.

    Automatically detects if output should be partitioned by year and/or week
    based on data size and column availability.

    Args:
        df: DataFrame to write (Pandas or Polars)
        output_path: Output path (will be a directory for partitioned data)
        partition_cols: Columns to partition by (e.g., ["year"] or ["year", "week"])
        use_polars: Use Polars for writing (faster)
        compression: Compression codec (snappy, gzip, zstd)
        **kwargs: Additional arguments for write_parquet

    Example:
        >>> write_partitioned(df, "player.parquet", partition_cols=["year"])
        # Creates: player.parquet/year=2024/data.parquet
        #          player.parquet/year=2023/data.parquet
    """
    output_path = Path(output_path)

    # Validate partition columns exist
    if isinstance(df, pd.DataFrame):
        missing = [col for col in partition_cols if col not in df.columns]
    else:
        missing = [col for col in partition_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Partition columns not found in DataFrame: {missing}")

    # Convert to Polars if requested and not already
    if use_polars and isinstance(df, pd.DataFrame):
        df_write = pl.from_pandas(df)
    elif not use_polars and isinstance(df, pl.DataFrame):
        df_write = df.to_pandas()
    else:
        df_write = df

    # Write partitioned
    if use_polars and isinstance(df_write, pl.DataFrame):
        # Polars partitioned write
        df_write.write_parquet(
            output_path,
            compression=compression,
            partition_by=partition_cols,
            **kwargs
        )
    else:
        # Pandas partitioned write using pyarrow
        import pyarrow.parquet as pq
        import pyarrow as pa

        table = pa.Table.from_pandas(df_write)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=partition_cols,
            compression=compression,
            **kwargs
        )

    print(f"Wrote partitioned parquet to {output_path}")
    print(f"  Partitions: {partition_cols}")
    print(f"  Total rows: {len(df):,}")


def read_partitioned(
    input_path: Union[str, Path],
    filters: Optional[List[tuple]] = None,
    columns: Optional[List[str]] = None,
    use_polars: bool = True,
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Read partitioned parquet with optional filters.

    Args:
        input_path: Path to partitioned parquet directory
        filters: List of filters as tuples (column, operator, value)
                 e.g., [("year", "==", 2024), ("week", ">=", 5)]
        columns: Specific columns to read (None = all)
        use_polars: Return Polars DataFrame (faster)
        **kwargs: Additional arguments for read_parquet

    Returns:
        DataFrame (Pandas or Polars depending on use_polars)

    Example:
        >>> # Read only 2024 data
        >>> df = read_partitioned(
        ...     "player.parquet",
        ...     filters=[("year", "==", 2024)]
        ... )
        >>>
        >>> # Read specific columns from 2024, weeks 1-5
        >>> df = read_partitioned(
        ...     "player.parquet",
        ...     filters=[("year", "==", 2024), ("week", "<=", 5)],
        ...     columns=["player_name", "points", "week"]
        ... )
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {input_path}")

    if use_polars:
        # Polars scan (lazy read with filters pushed down)
        lazy_df = pl.scan_parquet(input_path, **kwargs)

        # Apply filters
        if filters:
            for col, op, val in filters:
                if op == "==":
                    lazy_df = lazy_df.filter(pl.col(col) == val)
                elif op == "!=":
                    lazy_df = lazy_df.filter(pl.col(col) != val)
                elif op == ">":
                    lazy_df = lazy_df.filter(pl.col(col) > val)
                elif op == ">=":
                    lazy_df = lazy_df.filter(pl.col(col) >= val)
                elif op == "<":
                    lazy_df = lazy_df.filter(pl.col(col) < val)
                elif op == "<=":
                    lazy_df = lazy_df.filter(pl.col(col) <= val)
                elif op == "in":
                    lazy_df = lazy_df.filter(pl.col(col).is_in(val))
                else:
                    raise ValueError(f"Unsupported filter operator: {op}")

        # Select columns
        if columns:
            lazy_df = lazy_df.select(columns)

        # Collect (execute query)
        return lazy_df.collect()

    else:
        # Pandas read using pyarrow
        import pyarrow.parquet as pq

        # Convert filters to pyarrow format
        pyarrow_filters = None
        if filters:
            pyarrow_filters = []
            for col, op, val in filters:
                if op == "==":
                    pyarrow_filters.append((col, "=", val))
                elif op == "!=":
                    pyarrow_filters.append((col, "!=", val))
                elif op == ">":
                    pyarrow_filters.append((col, ">", val))
                elif op == ">=":
                    pyarrow_filters.append((col, ">=", val))
                elif op == "<":
                    pyarrow_filters.append((col, "<", val))
                elif op == "<=":
                    pyarrow_filters.append((col, "<=", val))
                elif op == "in":
                    pyarrow_filters.append((col, "in", val))

        dataset = pq.ParquetDataset(
            str(input_path),
            filters=pyarrow_filters,
            **kwargs
        )

        return dataset.read(columns=columns).to_pandas()


def auto_partition_cols(df: Union[pd.DataFrame, pl.DataFrame]) -> List[str]:
    """
    Automatically determine optimal partition columns.

    Logic:
    - If 'year' column exists and df has >100k rows, partition by year
    - If df has >1M rows and has 'week' column, partition by year and week
    - Otherwise, no partitioning

    Args:
        df: DataFrame to analyze

    Returns:
        List of column names to partition by
    """
    if isinstance(df, pl.DataFrame):
        row_count = len(df)
        columns = df.columns
    else:
        row_count = len(df)
        columns = df.columns.tolist()

    partition_cols = []

    # Check for year column
    if 'year' in columns and row_count > 100_000:
        partition_cols.append('year')

    # Check for week column (if already partitioning by year and >1M rows)
    if 'week' in columns and row_count > 1_000_000 and 'year' in partition_cols:
        partition_cols.append('week')

    return partition_cols


def write_auto_partitioned(
    df: Union[pd.DataFrame, pl.DataFrame],
    output_path: Union[str, Path],
    force_partition_cols: Optional[List[str]] = None,
    **kwargs
) -> None:
    """
    Write DataFrame with automatic partition detection.

    Analyzes the DataFrame and automatically determines optimal partitioning
    strategy based on size and available columns.

    Args:
        df: DataFrame to write
        output_path: Output path
        force_partition_cols: Override automatic detection
        **kwargs: Additional arguments for write_partitioned

    Example:
        >>> # Automatically partition if beneficial
        >>> write_auto_partitioned(large_df, "player.parquet")
    """
    output_path = Path(output_path)

    # Determine partition columns
    if force_partition_cols is not None:
        partition_cols = force_partition_cols
    else:
        partition_cols = auto_partition_cols(df)

    if partition_cols:
        print(f"Auto-partitioning by: {partition_cols}")
        write_partitioned(df, output_path, partition_cols, **kwargs)
    else:
        # Write single file (no partitioning needed)
        print(f"Writing single parquet file (no partitioning needed)")
        if isinstance(df, pl.DataFrame):
            df.write_parquet(output_path, **kwargs)
        else:
            df.to_parquet(output_path, **kwargs)


def get_partition_info(parquet_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about partitioned parquet dataset.

    Args:
        parquet_path: Path to parquet file or directory

    Returns:
        Dictionary with partition information

    Example:
        >>> info = get_partition_info("player.parquet")
        >>> print(f"Partitioned by: {info['partition_cols']}")
        >>> print(f"Partition count: {info['partition_count']}")
    """
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Path does not exist: {parquet_path}")

    try:
        dataset = pq.ParquetDataset(str(parquet_path))

        info = {
            'path': str(parquet_path),
            'is_partitioned': len(dataset.partitions.levels) > 0,
            'partition_cols': [level.name for level in dataset.partitions.levels] if dataset.partitions else [],
            'partition_count': len(dataset.fragments) if hasattr(dataset, 'fragments') else 0,
            'total_rows': None,  # Expensive to compute
            'schema': dataset.schema
        }

        return info

    except Exception as e:
        # Single file parquet
        return {
            'path': str(parquet_path),
            'is_partitioned': False,
            'partition_cols': [],
            'partition_count': 1,
            'error': str(e)
        }


def consolidate_partitions(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    new_partition_cols: Optional[List[str]] = None
) -> None:
    """
    Re-partition an existing parquet dataset.

    Useful for:
    - Converting non-partitioned to partitioned
    - Changing partition scheme (e.g., year only -> year+week)
    - Consolidating over-partitioned data

    Args:
        input_path: Existing parquet path
        output_path: New output path
        new_partition_cols: New partition columns (None = remove partitioning)

    Example:
        >>> # Convert to partitioned format
        >>> consolidate_partitions(
        ...     "old_player.parquet",
        ...     "new_player.parquet",
        ...     new_partition_cols=["year"]
        ... )
    """
    print(f"Reading from {input_path}...")
    df = read_partitioned(input_path, use_polars=True)

    print(f"Re-partitioning with {new_partition_cols}...")
    if new_partition_cols:
        write_partitioned(df, output_path, partition_cols=new_partition_cols)
    else:
        # Write as single file
        df.write_parquet(output_path)

    print(f"Consolidation complete: {output_path}")
