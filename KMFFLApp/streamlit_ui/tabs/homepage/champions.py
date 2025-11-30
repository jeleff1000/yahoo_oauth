import streamlit as st
from .hall_of_fame.hall_of_fame_homepage import HallOfFameViewer

@st.fragment
def display_champions(df_dict):
    hall_of_fame_viewer = HallOfFameViewer(df_dict)
    hall_of_fame_viewer.display()