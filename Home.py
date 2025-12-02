import streamlit as st
import pandas as pd
import os
from flywheel.client import Client
from dotenv import load_dotenv
import numpy as np
import pathlib
from pathlib import Path
from utils.authentication import login_screen

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

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📥 Data Download"):
            st.switch_page("pages/1_Data_Download.py")

    with col2:
        if st.button("🧹 Outlier Detection/Cleaning"):
            st.switch_page("pages/2_Cleaning_Outlier_Detection.py")

    with col3:
        if st.button("🧩 Segmentation QC"):
            st.switch_page("pages/3_QC_Segmentation.py")
    with col4:
        if st.button("📊 Analysis and Visualization"):
            st.switch_page("pages/4_Analysis_Visualization.py")


if not st.session_state.authenticated:
    #Get API from env 
    api_key = None #os.getenv("FW_CLI_API_KEY")
    if api_key:
        st.session_state.api_key = api_key
        st.session_state.fw = Client(api_key)
        st.session_state.authenticated = True
        st.success("Authenticated using API key from environment.")
        main_app()
    else:
        st.session_state.fw = login_screen()
else:
    user = st.session_state.fw.get_current_user()
    st.success(f"Logged in as: {user.firstname} {user.lastname}")
    st.warning("For security reasons, please do not share your API key with anyone.")
    st.info("You can now navigate to other pages using the buttons at the top. Do not refresh the page or you will need to log in again.")
    main_app()



