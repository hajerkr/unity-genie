

import streamlit as st
import flywheel
import os
import re
import uuid
import shutil
import time
from pathlib import Path
from datetime import datetime
import traceback

_STALE_SESSION_MAX_AGE_SECONDS = 6 * 60 * 60  # 6 hours

def get_session_data_dir() -> Path:
    """Return this browser session's private data directory, creating it if needed."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    base_dir = Path(__file__).resolve().parent.parent / "data"
    session_dir = base_dir / st.session_state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    _sweep_stale_session_dirs(base_dir, keep=st.session_state.session_id)
    return session_dir

def _sweep_stale_session_dirs(base_dir: Path, keep: str) -> None:
    """Best-effort removal of other sessions' leftover folders older than the max age."""
    if not base_dir.exists():
        return
    now = time.time()
    for entry in base_dir.iterdir():
        if entry.name == keep or not entry.is_dir():
            continue
        try:
            if now - entry.stat().st_mtime > _STALE_SESSION_MAX_AGE_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass

def is_complete(asys,gearname,latest_version=False):
    try:
        asys=asys.reload()
    except Exception as e:
        print(f"Error reloading analysis {asys.id}: {e}")
        
    if gearname =="gambas" and getattr(asys, 'gear_info', None) is None:
   
            print(f"Analysis {asys.id} has no gear_info, checking label for gambas-batch...")
            #Look at analysis container containing "gambas-batch" in the label
            print(asys.label)
            return (
                "gambas" in asys.label and ("0.4.17" in asys.label or "0.4.14" in asys.label)
                and len(asys.files) > 0
            )
    else:
        
        return (
            asys.gear_info is not None
            and asys.gear_info.get('name') == gearname
            and asys.job is not None
            and asys.job.get('state') == 'complete'
            )
        