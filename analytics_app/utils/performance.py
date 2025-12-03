"""
Performance utilities for optimizing Streamlit app
"""

import functools
import logging
import time
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and log performance of operations"""

    def __init__(self):
        self.timings = {}

    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""

        class Timer:
            def __init__(self, name, monitor):
                self.name = name
                self.monitor = monitor
                self.start = None

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                duration = time.time() - self.start
                self.monitor.timings[self.name] = duration
                if duration > 1.0:
                    logger.warning(f"⚠️ {self.name} took {duration:.2f}s")

        return Timer(operation_name, self)


# Global performance monitor
perf_monitor = PerformanceMonitor()


def lazy_import(module_path: str):
    """
    Lazy import decorator to speed up initial app load
    Only import modules when they're actually needed
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Import the module only when function is called
            parts = module_path.rsplit(".", 1)
            if len(parts) == 2:
                module_name, attr = parts
                module = __import__(module_name, fromlist=[attr])
                imported = getattr(module, attr)
            else:
                imported = __import__(module_path)

            # Call the original function with imported module
            return func(imported, *args, **kwargs)

        return wrapper

    return decorator


def smart_cache(
    ttl: int = 600, show_spinner: bool = False, key_func: Optional[Callable] = None
):
    """
    Enhanced caching with session state fallback
    - Uses st.cache_data as primary cache
    - Falls back to session_state for user-specific data
    - Supports custom key functions for fine-grained control
    """

    def decorator(func):
        # Get the cached version
        cached_func = st.cache_data(ttl=ttl, show_spinner=show_spinner)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = f"_cache_{func.__name__}_{key_func(*args, **kwargs)}"
            else:
                cache_key = f"_cache_{func.__name__}"

            # Try session state first (user-specific)
            if cache_key in st.session_state:
                cached_time = st.session_state.get(f"{cache_key}_time", 0)
                if time.time() - cached_time < ttl:
                    return st.session_state[cache_key]

            # Call cached function
            with perf_monitor.time_operation(f"Cache miss: {func.__name__}"):
                result = cached_func(*args, **kwargs)

            # Store in session state
            st.session_state[cache_key] = result
            st.session_state[f"{cache_key}_time"] = time.time()

            return result

        return wrapper

    return decorator


def progressive_loader(message: str = "Loading..."):
    """
    Show progressive loading states with spinner
    Automatically times the operation
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with st.spinner(message):
                with perf_monitor.time_operation(func.__name__):
                    return func(*args, **kwargs)

        return wrapper

    return decorator


def batch_operations(items: list, batch_size: int = 100, operation: Callable = None):
    """
    Process items in batches to avoid overwhelming the UI
    Useful for rendering large lists of components
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_num = i // batch_size + 1

        status_text.text(f"Processing batch {batch_num}/{total_batches}...")
        progress_bar.progress(min(1.0, (i + batch_size) / len(items)))

        if operation:
            batch_results = [operation(item) for item in batch]
            results.extend(batch_results)

    progress_bar.empty()
    status_text.empty()

    return results


class DataLoader:
    """
    Smart data loader with caching and lazy loading
    Only loads data when tab is active
    """

    @staticmethod
    def load_if_active(tab_name: str, loader_func: Callable, *args, **kwargs) -> Any:
        """
        Only load data if the tab is currently active
        Stores in session state to avoid reloading on tab switch
        """
        cache_key = f"_data_{tab_name}"

        # Check if data is already loaded
        if cache_key in st.session_state:
            return st.session_state[cache_key]

        # Load data
        with st.spinner(f"Loading {tab_name} data..."):
            data = loader_func(*args, **kwargs)
            st.session_state[cache_key] = data
            return data

    @staticmethod
    def invalidate_cache(tab_name: str):
        """Invalidate cached data for a specific tab"""
        cache_key = f"_data_{tab_name}"
        if cache_key in st.session_state:
            del st.session_state[cache_key]


def optimize_dataframe(df, max_rows: int = 10000):
    """
    Optimize DataFrame for display
    - Limit rows
    - Convert to appropriate dtypes
    - Remove unnecessary columns
    """
    if df is None or df.empty:
        return df

    # Limit rows
    if len(df) > max_rows:
        st.info(f"Showing first {max_rows:,} of {len(df):,} rows")
        df = df.head(max_rows)

    # Optimize dtypes
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
        elif df[col].dtype == "float64":
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def debounce(wait_time: float = 0.5):
    """
    Debounce function calls to avoid excessive reruns
    Useful for search inputs and filters
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"_debounce_{func.__name__}"
            last_call_key = f"{key}_last_call"

            current_time = time.time()
            last_call = st.session_state.get(last_call_key, 0)

            if current_time - last_call < wait_time:
                return None

            st.session_state[last_call_key] = current_time
            return func(*args, **kwargs)

        return wrapper

    return decorator
