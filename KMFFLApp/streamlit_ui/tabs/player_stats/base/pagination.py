import streamlit as st
import pandas as pd


class PaginationManager:
    """Manages pagination state and controls for dataframes."""

    def __init__(self, key_prefix: str):
        self.key_prefix = key_prefix
        self.offset_key = f"{key_prefix}_offset"
        self.limit_key = f"{key_prefix}_limit"

        # Initialize session state if not exists
        if self.offset_key not in st.session_state:
            st.session_state[self.offset_key] = 0
        if self.limit_key not in st.session_state:
            st.session_state[self.limit_key] = 100

    @st.fragment
    def display_controls(self, df: pd.DataFrame):
        """Display pagination controls and return current page info."""
        # Get total count from dataframe attributes or length
        total_count = getattr(df, 'attrs', {}).get('total_count', len(df))
        current_offset = st.session_state[self.offset_key]
        current_limit = st.session_state[self.limit_key]

        total_pages = (total_count + current_limit - 1) // current_limit
        current_page = (current_offset // current_limit) + 1

        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

        with col1:
            if st.button("⏮️ First", key=f"{self.key_prefix}_first",
                        disabled=(current_page == 1)):
                st.session_state[self.offset_key] = 0
                st.rerun()

        with col2:
            if st.button("◀️ Prev", key=f"{self.key_prefix}_prev",
                        disabled=(current_page == 1)):
                st.session_state[self.offset_key] = max(0, current_offset - current_limit)
                st.rerun()

        with col3:
            st.markdown(f"**Page {current_page} of {total_pages}** ({total_count:,} total records)")

        with col4:
            if st.button("Next ▶️", key=f"{self.key_prefix}_next",
                        disabled=(current_page >= total_pages)):
                st.session_state[self.offset_key] = current_offset + current_limit
                st.rerun()

        with col5:
            if st.button("Last ⏭️", key=f"{self.key_prefix}_last",
                        disabled=(current_page >= total_pages)):
                st.session_state[self.offset_key] = (total_pages - 1) * current_limit
                st.rerun()

        st.markdown("---")

        return {
            'current_page': current_page,
            'total_pages': total_pages,
            'total_count': total_count,
            'offset': current_offset,
            'limit': current_limit
        }

    def get_current_offset(self) -> int:
        """Get current offset value."""
        return st.session_state[self.offset_key]

    def get_current_limit(self) -> int:
        """Get current limit value."""
        return st.session_state[self.limit_key]

    def reset(self):
        """Reset pagination to first page."""
        st.session_state[self.offset_key] = 0