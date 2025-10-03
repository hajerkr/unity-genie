import os
import flywheel
from pathlib import Path
import pathvalidate as pv
import pandas as pd
from datetime import datetime
import pytz
import time
import argparse
import subprocess
from dotenv import load_dotenv
import flywheel
import streamlit as st

"""   
Pull Results from Flywheel
Specify the keyword (to look for in the file of interest), project name, gear, and it version

"""

dotenv.load_dotenv()

def download_derivatives(project_id, gear, gear_versions, keywords, timestampFilter):
    """
    Download derivative files from Flywheel analyses based on specified criteria.
    Parameters:
    - project_id: The Flywheel project ID to search within.
    - derivative_type: The type of derivative to look for (e.g., 'recon-all-clinical', 'minimorph').
    Returns:
    - outdir: Path to the compiled CSV file containing results.
    """

    api_key = os.getenv('FW_CLI_API_KEY')
    # Connect to your Flywheel instance
    if timestampFilter is None:
        timestampFilter = datetime(2020, 1, 1, 0, 0, 0, 0, pytz.UTC)


    fw = flywheel.Client(api_key=api_key)
    print(f"User: {fw.get_current_user().firstname} {fw.get_current_user().lastname}")
    project = fw.projects.find_first(f'label={project_id}')
    st.info(f"Project: {project_id} Subjects n = {len(project.subjects())}\nSessions n = {len(project.sessions())}...")

    # Create a work directory in our local "home" directory
    work_dir = Path(Path.home()/'../data/', platform='auto')
    # If it doesn't exist, create it
    if not work_dir.exists():
        work_dir.mkdir(parents = True)
    # Create a custom path for our project (we may run this on other projects in the future) and create if it doesn't exist
    project_path = pv.sanitize_filepath(work_dir/project.label/gear, platform='auto')
    if not project_path.exists():
        project_path.mkdir(parents = True)
    # Preallocate lists
    df = pd.DataFrame()

    progress = st.progress(0)
    status = st.empty()

    # --- Find the results --- #
    sessions = project.sessions()

    # Iterate through all subjects in the project and find the results
    #Delete sessions where session.subject.label starts with 137-
    sessions = [ses for ses in sessions if not ses.subject.label.startswith('137-')]
    #Add slider for number of sessions to process
    # max_sessions = st.sidebar.slider("Select max number of sessions to fetch:", min_value=1, max_value=len(sessions), value=len(sessions))
    # sessions = sessions[:max_sessions]

    for i, session in enumerate(sessions[:10]):
        session = session.reload()
        ses_label = session.label
        sub_label = session.subject.label
        print(sub_label, ses_label)
        status.text(f"Fetching {sub_label} - {ses_label}...")
        for analysis in session.analyses:
            #print("Analyses ran on this subject: ", analysis.gear_info.name, analysis.gear_info.version)
            if analysis.gear_info is not None and analysis.gear_info.name == gear and analysis.created > timestampFilter and analysis.gear_info.version in gear_versions: # and analysis.gear_info.version == gearVersion and analysis.get("job").get("state") == "complete"
                # print("pulling: ", gear, gearVersion)
                #st.write(f"Found {gear} {analysis.gear_info.version} for {sub_label} - {ses_label}")
                for analysis_file in analysis.files:
                    for keyword in keywords:
                        if keyword in analysis_file.name:
                            file = analysis_file
                            file = file.reload()
                            # Sanitize our filename and parent path
                            download_dir = pv.sanitize_filepath(project_path/sub_label/ses_label,platform='auto')   
                            fileName = file.name #(analysis.gear_info.name + "_" + analysis.label + ".csv")

                            # Create the path
                            if not download_dir.exists():
                                download_dir.mkdir(parents=True)
                            download_path = download_dir/fileName
                            print(download_path)

                            # Download the file
                            print('Downloading file', ses_label, analysis.label)
                            file.download(download_path)

                            # Add subject to dataframe
                            with open(download_path) as csv_file:
                                results = pd.read_csv(csv_file, index_col=None, header=0) 
                                #Insert session.tags as a new column after 'session'
                                results.insert(3, 'session_QC', session.tags if session.tags else 'n/a')
                                #Instead of blindly appending, ensure differences in columns are handled
                                #Drop "Unnamed" columns if they exist
                                results = results.loc[:, ~results.columns.str.contains('^Unnamed')]                                
                                #Add analysis.gear_info.version in the results dataframe after the 'session' column
                                results.insert(0, f'{gear}_v', analysis.gear_info.version)
                                df = pd.concat([df, results], ignore_index=True)  # handles different columns

                            #Delete the file after reading it in
                            os.remove(download_path)

        progress.progress((i+1)/len(sessions))
                                
    # --- Save output --- #

    try:
        outname = project.label.replace(' ', '_')
        outname = outname.replace('(', '')
        outname = outname.replace(')', '')
        filename = (outname + "-" + gear + ".csv")
        # write DataFrame to an excel sheet 
        #df = pd.concat(df, axis=0, ignore_index=True)
        #If df is empty, add a message and exit
        if df.empty:
            st.warning("No results found for the specified criteria.")
            return None
        
        if 'input_gear_v' in df.columns:
            cols = df.columns.tolist()
            cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('input_gear_v')))
            df = df[cols]
        if 'scanner_software_v' in results.columns:
            cols = df.columns.tolist()
            cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('scanner_software_v')))
            df = df[cols]

        outdir = os.path.join(project_path, filename)
        df.to_csv(outdir, index=False)
        print(f"Results saved to {outdir}")
        return outdir

        #UPLOADS A FILE TO THE PROJECT INFORMATION TAB
        #project.upload_file(outdir)
    except Exception as e:
        st.warning(f"Exception occurred: {e}")
        