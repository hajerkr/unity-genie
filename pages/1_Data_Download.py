import os
import flywheel
from pathlib import Path
import pathvalidate as pv
import pandas as pd
from datetime import datetime, date
import pytz
import time
import argparse
import subprocess
import flywheel
import streamlit as st
import os
import numpy as np
import yaml
from packaging import version
from dotenv import load_dotenv
from utils.authentication import login_screen
import concurrent.futures
import traceback

# import threading
# from concurrent.futures import ThreadPoolExecutor
# from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


def download_session_data(project, session, project_path, segtool, input_source, fw_session_info, keywords, work_dir, tool_map):
    """
    Returns a single DataFrame row (or few rows) representing this session,
    with all gear/keyword CSVs merged horizontally.
    Returns None if nothing was found.
    """
    session = fw.get(session)
    session = session.reload()
    ses_label = session.label
    sub_label = session.subject.label
    session_df = pd.DataFrame()  # will accumulate horizontally across gears/keywords

    analyses = [
        a for a in session.analyses
        if a.reload().gear_info is not None and a.reload().gear_info.name in segtool and a.reload().job.get('state') == 'complete'
    ]

    mrr_analyses = []
    gambas_analyses = []
    
    def get_latest(analyses, input_keyword):
        keywords = [input_keyword] if isinstance(input_keyword, str) else input_keyword
        candidates = [a for a in analyses if any(kw in a.inputs[0].name for kw in keywords)]
        return [max(candidates, key=lambda a: a.created)] if candidates else []

    for segmentation_tool in segtool:
        tool_analyses = [a for a in analyses if a.gear_info.name == segmentation_tool]  # scoped

        if input_source == "MRR":
            mrr_analyses.append(get_latest(tool_analyses, "mrr"))

        elif input_source == "Enhanced (Gambas)":
            gambas_analyses.append(get_latest(tool_analyses, ["gambas", "ResCNN"]))

        else:
            mrr_analyses.append(get_latest(tool_analyses, "mrr"))
            gambas_analyses.append(get_latest(tool_analyses, ["gambas", "ResCNN"]))

            # gambas_analyses = [a for a in analyses if "gambas" in a.inputs[0].name]
            #print(f"Session {ses_label} - {mrr_filtered.label} MRR analyses, {gambas_filtered.label} Gambas analyses")
            # mrr_analyses.append(mrr_filtered)
            # gambas_analyses.append(gambas_filtered)
    
    
    #Add the mrr_analyses and gambas_analyses in one list
    analyses_filtered = mrr_analyses + gambas_analyses    
    #Delete empty lists from analyses_filtered
    analyses_filtered = [a for a in analyses_filtered if a]  # filter out empty lists
    analyses_filtered = [a for sublist in analyses_filtered for a in sublist if a]  # flatten and filter out any remaining None values
    
    print(f"Session {ses_label}: segtool={segtool}, keywords={keywords}, input_source={input_source}")
    print(f"Session {ses_label}: total analyses={len(analyses)}, filtered={len(analyses_filtered)}")

    try:
        if fw_session_info == "Yes":
            session_tags = session.tags if session.tags else []
            session_df['session_tags'] = ' | '.join(session_tags) if session_tags else 'n/a'
        
            for key, value in session.info.items():
                session_df[f'{key}'] = value

        for analysis in analyses_filtered:
            # print(type(analysis),type(analysis))
            analysis = analysis.reload()
            gear = analysis.gear_info.name
            volumetric_cols = tool_map[gear]
            matched_files = [f for f in analysis.files if any(kw in f.name for kw in keywords)]

            for analysis_file in matched_files:
                # matched = next((kw for kw in keywords if kw in analysis_file.name), None)
                # if matched is None:
                #     continue

                # # Download
                # keyword = matched
                # print(f"####### {keyword} #######")
                file = analysis_file.reload()
                download_dir = pv.sanitize_filepath(project_path / sub_label / ses_label, platform='auto')
                download_dir.mkdir(parents=True, exist_ok=True)
                download_path = download_dir / file.name
                print(f"Downloading {file.name} to {download_path}...")
                file.download(download_path)

                df = pd.read_csv(download_path)
                df["project"]    = project.label
                df["subject"]    = sub_label
                df["sex"]        = session.info.get('childBiologicalSex', 'n/a')
                df["session"]    = ses_label
                df["childTimepointAge_months"]   = session.info.get('childTimepointAge_months', df.get("age", "n/a"))

                # Clean and enrich
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df.insert(3, 'session_qc', session.tags[-1] if session.tags else 'n/a')
                df["age_source"] = "custom_info"
                
                
                #df.insert(0, 'gear_v', analysis.gear_info.version)

                # Prefix volumetric columns and tag with gear metadata
                if gear == "minimorph":
                    df["analysis_id_mm"]   = analysis.id
                    df["gear_v_minimorph"] = analysis.gear_info.version
                    df.rename(columns={col: f'mm_{col}' for col in volumetric_cols if col in df.columns}, inplace=True)
                elif gear == "supersynth":
                    df["analysis_id_ss"]   = analysis.id
                    df["gear_v_supersynth"] = analysis.gear_info.version
                    df.rename(columns={col: f'ss_{col}' for col in volumetric_cols if col in df.columns}, inplace=True)
                
                else:
                    df["analysis_id_ra"]   = analysis.id
                    df["gear_v_recon_all"] = analysis.gear_info.version
                    df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
                    df.rename(columns={col: f'ra_{col}' for col in volumetric_cols if col in df.columns}, inplace=True)

                # Merge horizontally into session_df
                #print(session_df, session_df.shape)

                if session_df.empty:
                    session_df = df
                else:
                    # drop duplicate identity cols before merging
                    print("Merging new data into session_df...")
                    merge_keys = ['subject', 'session']
                    cols_to_exclude = [c for c in session_df.columns if c not in merge_keys]
                    new_cols = [c for c in df.columns if c not in cols_to_exclude]

                    session_df = pd.merge(
                        session_df,
                        df[new_cols],
                        on=merge_keys,
                        how='outer'
                    )

                    print(session_df.shape)
                    print(session_df.values)

                # else:
                #     print(f"Session {ses_label} - merging {file.name} into session_df")
                #     id_cols = ['subject', 'session']  # adjust to match your actual col names
                #     merge_cols = [c for c in id_cols if c in session_df.columns and c in df.columns]

                #     # Drop columns from df that already exist in session_df (excluding the key)
                #     duplicate_cols = [c for c in df.columns if c in session_df.columns and c not in merge_cols]
                #     df_to_merge = df.drop(columns=duplicate_cols)
                #     #Concat the dataframes horizontally, keeping all rows (outer join)
                #     session_df = pd.merge(session_df, df_to_merge, how='outer')

                os.remove(download_path)

        #print(session_df.values)
        #Turn dataframe into list 
        #If user wants fw_session_info , pull session tags, and session custom information and add it to the csv
        
        #print(f"Finished processing session {ses_label} for subject {sub_label}. Final shape of session_df: {session_df.shape} with {len(session_df.columns.tolist())} columns")
        #Drop column gear_v
        session_df.drop(columns=['gear_v',"age_source","template_age"], inplace=True, errors='ignore')
        
        return session_df if not session_df.empty else None
    
    except Exception as e:
        print(f"EXCEPTION in session {ses_label} / {sub_label}: {e}\n{traceback.format_exc()}")
        return None
        # st.warning(f"Error processing session {ses_label} for subject {sub_label}: {e} ,  {traceback.format_exc()}")
        # return None


def download_derivatives(project_id, segtool, input_source, fw_session_info , keywords, timestampFilter, fw):
    project = fw.projects.find_first(f'label={project_id}')
    st.info(f"Project: {project_id}  \nSubjects: {len(project.subjects())}  \nSessions: {len(project.sessions())}")

    data_dir.mkdir(parents=True, exist_ok=True)
    project_path = pv.sanitize_filepath(data_dir / project.label, platform='auto')
    project_path.mkdir(parents=True, exist_ok=True)

    sessions = [s.id for s in project.sessions() if not s.subject.label.startswith('137-')]
    if debug:
        sessions = sessions[:5]
    progress = st.progress(0)
    status = st.empty()
    all_frames = []
    max_workers = min(4, len(sessions))  # Limit number of threads to avoid overwhelming the system

    with open(os.path.join(work_dir, '..', "utils", "vol_columns.yml"), "r") as f:
        tool_map = yaml.load(f, Loader=yaml.SafeLoader)
    print(segtool)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_session = {
            executor.submit(
                download_session_data,
                project, session_id, project_path, segtool, input_source, fw_session_info, keywords, work_dir, tool_map
            ): session_id
            for session_id in sessions
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_session)):
            session = future_to_session[future]
            try:
                session_df = future.result()
                print(f"Session {session} processed, session_df shape: {session_df.shape if session_df is not None else 'None'}")
                if session_df is not None:
                    all_frames.append(session_df)   # one df per session, stack vertically at the end
                    print(all_frames)
            except Exception as e:
                session = fw.get(session)
                ses = session.reload()
                ses_label = ses.label
                sub_label = session.subject.label
                
                st.warning(f"Failed {sub_label} - {ses_label}: {e} ,  {traceback.format_exc()}")

            progress.progress((i + 1) / len(sessions))
            status.text(f"Completed {i + 1}/{len(sessions)}")

    if not all_frames:
        st.warning("No results found.")
        return None

    combined = pd.concat(all_frames, ignore_index=True)  # vertical stack, one block per session

    # Deduplicate: keep latest gear version per subject/session/acquisition/gear
    for gear_col, gear_name in [('gear_v_recon_all', 'recon-all-clinical'), ('gear_v_minimorph', 'minimorph')]:
        if gear_col in combined.columns:
            key_cols = ['subject', 'session', 'acquisition']
            combined = (
                combined
                .sort_values(gear_col, key=lambda s: s.map(lambda v: version.parse(v) if pd.notna(v) else version.parse("0")), ascending=False)
                .drop_duplicates(subset=key_cols, keep='first')
            )

    # Reorder convenience columns to front
    for col in ['scanner_software_v', 'input_gear_v']:
        if col in combined.columns and 'acquisition' in combined.columns:
            cols = combined.columns.tolist()
            cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index(col)))
            combined = combined[cols]

    st.session_state.df = combined

    # Save
    segtool_str = '-'.join(segtool)
    outname = project.label.replace(' ', '_').replace('(', '').replace(')', '')
    outdir = project_path / f"{outname}-{segtool_str}.csv"
    combined.to_csv(outdir, index=False)
    return str(outdir)
        

def assemble_csv(derivatives, out_csv="derivatives_summary.csv"):
    """
    Assemble a CSV from a list of derivative file paths.
    """


    st.session_state.df = []
    projects = []
    # For simplicity, let's assume derivatives is a list of file paths
    for deriv in derivatives:
        print(f"Found derivative: {deriv}")
        df_proj = pd.read_csv(deriv)
        #Get project unique values from project column
        projects.append(df_proj['project'].unique())
        st.session_state.df.append(df_proj)
        #Concatenate to the st.session_state.df, some columns are in common, some are not, combination should handle this

    st.session_state.df = pd.concat(st.session_state.df , axis=0, ignore_index=True)
    #Reorder columns to have project, subject, session, acquisition at the front, then columns starting with ra, mm, then the rest
    #Drop age and sex columns if they exist
    st.session_state.df.drop(columns=['age','sex','gear_v'], inplace=True, errors='ignore')

    cols = st.session_state.df.columns.tolist()
    front_cols = ['project', 'subject', 'session', 'childTimepointAge_months', 'childBiologicalSex', 'studyTimepoint', 'session_qc', 'acquisition']
    cols = st.session_state.df.columns.tolist()

    ra_cols = [col for col in cols if col.startswith('ra_')]
    mm_cols = [col for col in cols if col.startswith('mm_')]
    spoken_for = set(front_cols + ra_cols + mm_cols)
    other_cols = [col for col in cols if col not in spoken_for]

    new_order = front_cols + ra_cols + mm_cols + other_cols

    # Add any missing front_cols as NaN
    for col in front_cols:
        if col not in st.session_state.df.columns:
            st.session_state.df[col] = np.nan

    st.session_state.df = st.session_state.df[new_order]

    #If columns in new_order are not in the dataframe, add them with NaN values
    for col in new_order:
        if col not in st.session_state.df.columns:
            st.session_state.df[col] = np.nan
    st.session_state.df = st.session_state.df[new_order]


    #Add projects separated by _ to the filename
    #Get all the unique values in df["project"] and concatenate them with _ in between
    unique_projects = st.session_state.df["project"].unique()
    print(unique_projects)
    project_str = '_'.join(unique_projects)

    #Add timestamp to the filename
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    out_csv = f"derivatives_summary_{project_str}_{time_str}.csv"
    
    return st.session_state.df, out_csv
    


st.title("📥 Data Download")
st.write("Download and compile derivatives from multiple projects into a single CSV file.")
st.write("Select the projects and derivative types from the sidebar :point_left:, then click 'Fetch derivatives'.")

# Sidebar inputs
st.sidebar.header("Settings")
#Project IDs is a list generated using fw.projects() by getting the project label

# --- Session state initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Now you can access them like this:
API_KEY = os.getenv("FW_CLI_API_KEY")

if (API_KEY == None or API_KEY == "") and st.session_state.authenticated == False:
    
    #Display message to enter API KEY in Home page
    st.warning("Please enter your Flywheel API key in the Home page to continue.")
    st.stop()
fw = flywheel.Client(st.session_state.api_key if st.session_state.authenticated else API_KEY)
data_dir = Path(__file__).parent/'../data/'
work_dir = Path(__file__).parent

@st.cache_data(ttl=600)
def get_projects():
    return [p.label for p in fw.projects()]

projects = get_projects()

project_ids = st.sidebar.multiselect("Select Projects", projects)

#Add tickboxes if recon all was selected, to select area, thickness, and volume
#Check boxes 
minimorph = st.sidebar.checkbox("Minimorph", value=False)
recon_all = st.sidebar.checkbox("Recon-all-clinical", value=False)
supersynth = st.sidebar.checkbox("Supersynth", value=False)
recon_any = st.sidebar.checkbox("Recon-any", value=False)

# derivative_type = st.sidebar.radio("Segmentation Tool:", ["recon-all-clinical", "minimorph","Both"])
#Add radio button yes or no to include flywheel session information in the download
st.session_state.input_source = st.sidebar.radio("Structural Image Segmented:", ["MRR", "Enhanced (Gambas)"], index=0)
st.session_state.fw_session_info = st.sidebar.radio("Include Flywheel Session Info (tags, custom info) in download?", ["No", "Yes"], index=0)

# if st.sidebar.button("Fetch derivatives"):
#     with st.spinner("Fetching derivatives..."):
#         lastV = fw.get_all_gears(
#             filter=f"gear.name='{derivative_type}'",
#             sort="created:desc",
#             limit=5,
#             all_versions=True,
#             exhaustive=True
#         )
#         st.success("Fetched gear versions!")

#Add number slider for number of versions to include
gear_vesions = 5
# gear_versions = st.sidebar.slider("Last gear versions:", min_value=1, max_value=10, value=3)

#Add date picker for after date
after_date = None
#after_date = st.sidebar.date_input("Select date (only fetch analyses after this date):", value=None)

derivative_type = []
keywords = []
if recon_all:
    derivative_type.append("recon-all-clinical")

    st.sidebar.markdown("**Select outputs to download:**")
    area = st.sidebar.checkbox("Area", value=False)
    thickness = st.sidebar.checkbox("Thickness", value=False)
    volume = st.sidebar.checkbox("Volume", value=True)

    if area:
        keywords.append("area")
    if thickness:
        keywords.append("thickness")
    if volume:
        keywords.append("volume")
    
    if not keywords:
        st.sidebar.warning("Please select at least one output type.")

if minimorph:
    derivative_type.append("minimorph")
    keywords.extend(["volumes"])
if supersynth:
    derivative_type.append("supersynth")
    keywords.extend(["volumes"])
if recon_any:
    derivative_type.append("recon-any")
    

#Add debugger button to only fetch a handful of sessions for testing
debug = st.sidebar.checkbox("Debug mode (fetch fewer sessions)", value=False)

if st.sidebar.button("Fetch derivatives"):
    
    derivative_paths = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, proj in enumerate(project_ids):
        st.write(f"Fetching {", ".join(derivative_type)} for {proj}...")
        # Connect to your Flywheel instance
        if after_date is None:
            after_date = datetime(2020, 1, 1, 0, 0, 0, 0, pytz.UTC)

        # fw = flywheel.Client(api_key=api_key)
        # project = fw.projects.find_first(f'label={project_id}')

        # # --- Find the results --- #
        # sessions = project.sessions()

        # # Iterate through all subjects in the project and find the results
        # #Delete sessions where session.subject.label starts with 137-
        # sessions = [ses for ses in sessions if not ses.subject.label.startswith('137-')]
        # #Add slider for number of sessions to process
        # max_sessions = st.sidebar.slider("Select max number of sessions to fetch:", min_value=1, max_value=len(sessions), value=len(sessions))
        # sessions = sessions[:max_sessions]
        st.info(f"Download for {proj} started in background.")
        # print(st.session_state)
        
        derivatives = download_derivatives(proj, derivative_type,st.session_state.input_source,st.session_state.fw_session_info, keywords, after_date,fw)
        #derivatives.to_csv(f"/Users/Hajer/unity/debugging/derivatives_{proj}.csv", index=False)
        if derivatives:
            derivative_paths.extend([derivatives])
        progress.progress((i+1)/len(project_ids))
    

    if not derivative_paths:
        st.error("No derivatives found. Please check your selections and try again.")
        # st.stop()
    else:    
        st.session_state.df, outdir = assemble_csv(derivative_paths)
        
        st.success("Download complete!")
        st.dataframe(st.session_state.df)
        
        # Provide CSV download
        st.session_state.df.to_csv(outdir, index=False)
        if os.path.exists(outdir):
            with open(outdir, "rb") as f:
                st.download_button("Download CSV", f, file_name=outdir)

            #If there is only one project in the dataframe, find the project in flywheel and upload the file
            unique_projects = st.session_state.df["project"].unique()
            if len(unique_projects) == 1:
                project_name = unique_projects[0]
                project = fw.projects.find_first(f'label={project_name}')
                print(f"Uploading {outdir} to Flywheel")
                project.upload_file(outdir)
                st.success(f"Uploaded {outdir} to Flywheel at {project.label}!")
            else:
                st.info("Multiple projects in the CSV, skipping Flywheel upload.")


