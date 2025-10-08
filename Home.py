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
    import moviepy.editor as mp



# ---- Streamlit App ----
# set the page layout to wide
work_dir = Path(__file__).parent/'./docs/'
# st.logo(os.path.join(work_dir,"logo.jpg"), size='large')
st.set_page_config(page_title="Neuroimaging Pipeline", layout="wide")
st.image(os.path.join(work_dir,"logo.jpg"), width=200)  # custom size instead of 'use_column_width'
st.title("🧠 ULF MRI Processing Dashboard")

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