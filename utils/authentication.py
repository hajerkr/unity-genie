import streamlit as st
import flywheel
import os

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