"""
One-time helper to clear Streamlit caches as requested.
Run this once in the same environment where you run the Streamlit app.

Windows (cmd.exe):
    python clear_streamlit_cache.py

Note: This must run in an environment with Streamlit installed.
"""
import streamlit as st

if __name__ == "__main__":
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        print("Streamlit caches cleared: cache_data and cache_resource")
    except Exception as e:
        print(f"Failed to clear Streamlit caches: {e}")

