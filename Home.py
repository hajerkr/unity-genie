import streamlit as st
import pandas as pd
import os
import flywheel
from dotenv import load_dotenv
import numpy as np
import pathlib
from pathlib import Path

# Load variables from .env into environment
load_dotenv(override=True)

import sys
import subprocess

try:
    import moviepy.editor as mp
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
    import moviepy
    import moviepy.editor as mp


st.set_page_config(page_title="Neuroimaging Pipeline", layout="wide")

# --- Session state initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None


# --- Login screen ---
def login_screen():
    st.title("🔐 Welcome to the Flywheel App")
    st.write("Please enter your Flywheel API key to continue.")

    api_key = st.text_input("Flywheel API Key", type="password")

    if st.button("Log in"):
        if not api_key:
            st.warning("Please enter an API key.")
        else:
            try:
                fw = flywheel.Client(api_key)
                # Simple validation – check that client works
                user = fw.get_current_user()
                st.success(f"Logged in as: {user.firstname} {user.lastname}")
                st.session_state.authenticated = True
                st.session_state.api_key = api_key
                st.rerun()
            except Exception as e:
                st.error("Invalid API key or connection error. Please try again.")


# --- Main app ---
def main_app():
    # ---- Streamlit App ----
    # set the page layout to wide
    work_dir = Path(__file__).parent/'./docs/'
    # st.logo(os.path.join(work_dir,"logo.jpg"), size='large')
    
    st.image(os.path.join(work_dir,"logo.jpg"), width=200)  # custom size instead of 'use_column_width'
    st.title("🧠 ULF MRI Processing Dashboard")
    # st.snow()
    # Sidebar for navigation
    st.write("Choose a module:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Data Download"):
            st.switch_page("pages/1_Data_Download.py")

    with col2:
        if st.button("🧹 Outlier Detection and Cleaning"):
            st.switch_page("pages/2_Cleaning_Outlier_Detection.py")

    with col3:
        if st.button("🧩 Segmentation QC"):
            st.switch_page("pages/3_Segmentation_QC.py")

# --- Page logic ---
if not st.session_state.authenticated:
    login_screen()
else:
    main_app()


