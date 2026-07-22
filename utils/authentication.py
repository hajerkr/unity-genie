import streamlit as st
import flywheel
import os
from fw_client import FWClient

# --- Login screen ---
def login_screen():
    st.title(" Welcome to the UNITY App")
    st.write("🔐 Please enter your Flywheel API key to continue.")
    #Add link to instructions on how to get an API key
    st.markdown(
        "Don't have an API key? [Click here for instructions on how to get one](https://docs.flywheel.io/admin/users_and_access/howto_create_api_keys/#generate-an-api-key-enhanced)."
    )

    api_key = st.text_input("Flywheel API Key", type="password")

    if st.button("Log in"):
        if not api_key:
            st.warning("Please enter an API key.")
        else:
            try:
                st.session_state.fw = flywheel.Client(api_key)
                # Simple validation – check that client works
                st.session_state.fwclient = FWClient(
                    api_key=api_key,
                    timeout=100,
                )
                
                st.session_state.authenticated = True
                st.session_state.api_key = api_key
                st.rerun()
                return st.session_state.fw
            except Exception as e:
                st.error("Invalid API key or connection error. Please try again.")
    
    