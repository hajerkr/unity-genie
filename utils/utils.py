

import streamlit as st
import flywheel
import os
import re
from datetime import datetime
import traceback

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
        