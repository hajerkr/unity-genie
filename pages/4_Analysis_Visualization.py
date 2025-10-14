import streamlit as st
from datetime import date

st.set_page_config(page_title="Coming Soon", layout="centered")

st.markdown("""
    <style>
        .centered {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='centered'>🚧 DataLine Integration   \nComing Soon 🚧</h2>", unsafe_allow_html=True)
st.markdown("<p class='centered'>We’re working on bringing interactive data exploration directly into this app.</p>", unsafe_allow_html=True)

st.divider()
st.markdown(f"<p class='centered'>Last updated: {date.today().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)
