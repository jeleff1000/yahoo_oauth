"""
Enhanced data caching layer with intelligent invalidation
Sits between app and data_access to provide smart caching
"""
import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, Callable
import time
import hashlib
import json


class SmartCache:
    """
    Intelligent caching system that:
    - Uses both Streamlit cache and session state
    - Tracks data freshness
    - Supports partial cache invalidation
    - Monitors cache hit rates
    """

    def __init__(self):
        self._init_session_state()

    def _init_session_state(self):
        """Initialize session state for caching"""
        # Use setdefault to avoid overwriting existing values
        if not hasattr(st.session_state, "_cache_store"):
            st.session_state._cache_store = {}
        if not hasattr(st.session_state, "_cache_timestamps"):
            st.session_state._cache_timestamps = {}
        if not hasattr(st.session_state, "_cache_stats"):
            st.session_state._cache_stats = {"hits": 0, "misses": 0}

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key from function name and arguments"""
        # Convert args and kwargs to a hashable string
        key_data = {
            "func": func_name,
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str, ttl: int = 600) -> Optional[Any]:
        """
        Get cached data if it exists and hasn't expired
        """
        # Ensure initialized
        self._init_session_state()

        if key not in st.session_state._cache_store:
            st.session_state._cache_stats["misses"] += 1
            return None

        # Check if expired
        timestamp = st.session_state._cache_timestamps.get(key, 0)
        if time.time() - timestamp > ttl:
            # Expired - remove from cache
            del st.session_state._cache_store[key]
            del st.session_state._cache_timestamps[key]
            st.session_state._cache_stats["misses"] += 1
            return None

        st.session_state._cache_stats["hits"] += 1
        return st.session_state._cache_store[key]

    def set(self, key: str, value: Any):
        """Store data in cache with current timestamp"""
        # Ensure initialized
        self._init_session_state()

        st.session_state._cache_store[key] = value
        st.session_state._cache_timestamps[key] = time.time()

    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries
        If pattern is provided, only invalidate keys matching the pattern
        """
        # Ensure initialized
        self._init_session_state()

        if pattern is None:
            # Clear all
            st.session_state._cache_store.clear()
            st.session_state._cache_timestamps.clear()
        else:
            # Clear matching pattern
            keys_to_remove = [
                k for k in st.session_state._cache_store.keys()
                if pattern in k
            ]
            for k in keys_to_remove:
                del st.session_state._cache_store[k]
                if k in st.session_state._cache_timestamps:
                    del st.session_state._cache_timestamps[k]

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        # Ensure initialized
        self._init_session_state()

        stats = st.session_state._cache_stats.copy()
        total = stats["hits"] + stats["misses"]
        if total > 0:
            stats["hit_rate"] = (stats["hits"] / total) * 100
        else:
            stats["hit_rate"] = 0
        stats["cached_items"] = len(st.session_state._cache_store)
        return stats


# Global cache instance
smart_cache = SmartCache()


def cached_data_loader(
    ttl: int = 600,
    show_spinner: bool = True,
    spinner_text: str = "Loading data..."
):
    """
    Decorator for caching data loading functions
    Combines Streamlit cache_data with session-based caching
    """
    def decorator(func: Callable):
        # Create streamlit cached version
        st_cached = st.cache_data(ttl=ttl, show_spinner=False)(func)

        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = smart_cache._generate_key(func.__name__, args, kwargs)

            # Try to get from smart cache first (session state)
            cached_value = smart_cache.get(cache_key, ttl=ttl)
            if cached_value is not None:
                return cached_value

            # Cache miss - load data
            if show_spinner:
                with st.spinner(spinner_text):
                    result = st_cached(*args, **kwargs)
            else:
                result = st_cached(*args, **kwargs)

            # Store in smart cache
            smart_cache.set(cache_key, result)

            return result

        # Expose cache control methods
        wrapper.invalidate = lambda: smart_cache.invalidate(func.__name__)
        wrapper.cache_stats = smart_cache.get_stats

        return wrapper

    return decorator


# Convenience functions for common data operations
def get_or_load(
    cache_key: str,
    loader_func: Callable,
    ttl: int = 600,
    force_reload: bool = False
) -> Any:
    """
    Get data from cache or load it using the provided function
    """
    if not force_reload:
        cached = smart_cache.get(cache_key, ttl=ttl)
        if cached is not None:
            return cached

    # Load fresh data
    data = loader_func()
    smart_cache.set(cache_key, data)
    return data


def invalidate_tab_cache(tab_name: str):
    """Invalidate all cache entries for a specific tab"""
    smart_cache.invalidate(pattern=tab_name.lower())


def get_cache_size_mb() -> float:
    """Estimate cache size in MB"""
    import sys

    # Ensure initialized
    if not hasattr(st.session_state, "_cache_store"):
        return 0.0

    total_size = 0
    for value in st.session_state._cache_store.values():
        try:
            total_size += sys.getsizeof(value)
            if isinstance(value, pd.DataFrame):
                total_size += value.memory_usage(deep=True).sum()
        except:
            pass
    return total_size / (1024 * 1024)


def render_cache_stats():
    """Render cache statistics (for debugging)"""
    stats = smart_cache.get_stats()
    cache_size = get_cache_size_mb()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Cache Stats")
    st.sidebar.metric("Hit Rate", f"{stats['hit_rate']:.1f}%")
    st.sidebar.metric("Cached Items", stats['cached_items'])
    st.sidebar.metric("Cache Size", f"{cache_size:.2f} MB")

    if st.sidebar.button("Clear Cache"):
        smart_cache.invalidate()
        st.cache_data.clear()
        st.rerun()
