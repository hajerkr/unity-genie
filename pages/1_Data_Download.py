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
import flywheel
import streamlit as st
import os
import numpy as np
import yaml
from packaging import version
from dotenv import load_dotenv

def download_derivatives(project_id, segtool, gear_versions, keywords, timestampFilter,fw):
    """
    Download derivative files from Flywheel analyses based on specified criteria.
    Parameters:
    - project_id: The Flywheel project ID to search within.
    - derivative_type: The type of derivative to look for (e.g., 'recon-all-clinical', 'minimorph').
    Returns:
    - outdir: Path to the compiled CSV file containing results.
    """

    # config = dotenv_values(os.path.join(Path(__file__).parent/".."), ".env")
      
    # api_key =config['FW_CLI_API_KEY']
    # Connect to your Flywheel instance
    if timestampFilter is None:
        timestampFilter = datetime(2020, 1, 1, 0, 0, 0, 0, pytz.UTC)


    
    print(f"User: {fw.get_current_user().firstname} {fw.get_current_user().lastname}")
    project = fw.projects.find_first(f'label={project_id}')
    st.info(f"Project: {project_id} Subjects n = {len(project.subjects())}\nSessions n = {len(project.sessions())}...")

    # Create a work directory in our local "home" directory
    #Pass the directory relative to the script
    data_dir = Path(__file__).parent/'../data/'
    work_dir = Path(__file__).parent
    #work_dir = Path(Path.home()/'../data/', platform='auto')
    # If it doesn't exist, create it
    if not data_dir.exists():
        data_dir.mkdir(parents = True)
    # Create a custom path for our project (we may run this on other projects in the future) and create if it doesn't exist
    project_path = pv.sanitize_filepath(data_dir/project.label, platform='auto')
    if not project_path.exists():
        project_path.mkdir(parents = True)
    # Preallocate lists
    df = pd.DataFrame() #for recon-all
    df2 = pd.DataFrame() #for minimorph

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

    
        

    for i, session in enumerate(sessions):
        session = session.reload()
        ses_label = session.label
        sub_label = session.subject.label
        print(sub_label, ses_label)
        status.text(f"Fetching {sub_label} - {ses_label}...")
        #Iterate through analyses, looking for both gears if derivative_type is "both"
        analyses = [a for a in session.analyses if a.gear_info is not None and a.gear_info.name in segtool and a.created > timestampFilter] #and a.gear_info.version in gear_versions
        for analysis in analyses:
            # if analysis.gear_info is not None and analysis.gear_info.name == segtool and analysis.created > timestampFilter and analysis.gear_info.version in gear_versions: # and analysis.gear_info.version == gearVersion and analysis.get("job").get("state") == "complete"
            # print("pulling: ", segtool, gearVersion)
            #st.write(f"Found {segtool} {analysis.gear_info.version} for {sub_label} - {ses_label}")
            gear = analysis.gear_info.name

            with open(os.path.join(work_dir, '..', "utils","columns.yml"),"r") as f:
                tool_map = yaml.load(f, Loader=yaml.SafeLoader)
                volumetric_cols = tool_map[gear]
                
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
                        results = pd.read_csv(download_path) 
                        #Insert session.tags as a new column after 'session'
                        print(session.tags)
                        results.insert(3, 'session_qc', session.tags[-1] if session.tags else 'n/a')
                        #If age is empty or NaN, replace with subject.age
                        if 'age' in results.columns:
                            if any(age not in [None, 0, "0"] for age in [session.info.get('childTimepointAge_months',None), session.info.get('age_at_scan_months',None)]):
                                age_source = 'custom_info'
                                age =  session.info.get('childTimepointAge_months')
                                
                                results['age'] = (
                                                results['age']
                                                .replace([0, "0", ""], np.nan)  # normalize "empty" values
                                                .fillna(age)                    # fill NaNs with session age
                                            )
                                #For those change, set 'age_source' with value 'custom_info'
                                
                                #Account for if age_source does not exist
                                if 'age_source' not in results.columns:
                                    results['age_source'] = 'n/a'
                                results['age_source'] = results.apply(lambda row: age_source if row['age'] == age else row['age_source'], axis=1)

                        results["age"] = session.info.get('childTimepointAge_months', results.get("age","n/a"))
                        results["sex"] = session.info.get('childBiologicalSex', 'n/a')
                        results["project"] = project.label
                        #Instead of blindly appending, ensure differences in columns are handled
                        #Drop "Unnamed" columns if they exist
                        results = results.loc[:, ~results.columns.str.contains('^Unnamed')]                                
                        #Add analysis.gear_info.version in the results dataframe after the 'session' column
                        results.insert(0, f'gear_v', analysis.gear_info.version) #analysis.gear_info.name + "/"+
                        #If segmentation tool is minimorph, add prefix mm_ to the volumetric columns
                        if gear == "minimorph":
                            results.rename(columns={col: f'mm_{col}' for col in volumetric_cols if col in results.columns}, inplace=True)
                        else:
                            results.columns = results.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
                            results.rename(columns={col: f'ra_{col}' for col in volumetric_cols if col in results.columns}, inplace=True)

                        #st.dataframe(results)
                        if gear == "recon-all-clinical":
                            #If session and acquisition combo already exists in df, only keep the latest version based on the gear version
                            if not df.empty:
                                existing = df[(df['subject'] == sub_label) & (df['session'] == ses_label) & (df['acquisition'] == results['acquisition'].iloc[0])]
                                if not existing.empty:
                                    existing_version = existing[f'gear_v'].iloc[0]
                                    if version.parse(analysis.gear_info.version) <= version.parse(existing_version):
                                        status.text(f"Skipping {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} as existing version {existing_version} is newer or equal to {analysis.gear_info.version}")
                                        continue
                                    else:
                                        #Drop the existing row
                                        df = df.drop(existing.index)
                                        status.text(f"Replacing {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} with newer version {analysis.gear_info.version}")
                            
                            df = pd.concat([df, results], ignore_index=True)
                            #st.dataframe(df)

                        elif gear == "minimorph":
                            if not df2.empty:
                                existing = df2[(df2['subject'] == sub_label) & (df2['session'] == ses_label) & (df2['acquisition'] == results['acquisition'].iloc[0])]
                                if not existing.empty:
                                    existing_version = existing[f'gear_v'].iloc[0]
                                    if version.parse(analysis.gear_info.version) <= version.parse(existing_version) : #existing_version >= analysis.gear_info.version:
                                        status.text(f"Skipping {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} as existing version {existing_version} is newer or equal to {analysis.gear_info.version}")
                                        
                                        continue
                                    else:
                                        #Drop the existing row
                                        df2 = df2.drop(existing.index)
                                        status.text(f"Replacing {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} with newer version {analysis.gear_info.version} (previously {existing_version})")
                            df2 = pd.concat([df2, results], ignore_index=True)

                        #Delete the file after reading it in
                        os.remove(download_path)

        progress.progress((i+1)/len(sessions))
                                
    # --- Save output --- #
    # segtool = segtool.join("-") if isinstance(segtool, list) else segtool
    segtool = '-'.join(segtool)
    try:
        outname = project.label.replace(' ', '_')
        outname = outname.replace('(', '')
        outname = outname.replace(')', '')
        filename = (outname + "-" + segtool + ".csv")
        # write DataFrame to an excel sheet 
        #df = pd.concat(df, axis=0, ignore_index=True)
        #If df is empty, add a message and exit
        if df.empty and df2.empty:
            st.warning("No results found for the specified criteria.")
            return None
        
        if not df.empty:
            if 'input_gear_v' in df.columns:
                cols = df.columns.tolist()
                cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('input_gear_v')))
                df = df[cols]
            if 'scanner_software_v' in results.columns:
                cols = df.columns.tolist()
                cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('scanner_software_v')))
                df = df[cols]
        if not df2.empty:
            if 'input_gear_v' in df2.columns:
                cols = df2.columns.tolist()
                cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('input_gear_v')))
                df2 = df2[cols]
            if 'scanner_software_v' in results.columns:
                cols = df2.columns.tolist()
                cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('scanner_software_v')))
                df2 = df2[cols]

        #If we have two dataframes, merge
        if not df.empty and not df2.empty:
            #For every row for column "gear_v" in df2, add a prefix "minimorph/" to the value
            df2['gear_v'] = df2['gear_v'].apply(lambda x: 'minimorph/' + x if not x.startswith('minimorph/') else x)
            df['gear_v'] = df['gear_v'].apply(lambda x: 'recon-all-clinical/' + x if not x.startswith('recon-all-clinical/') else x)
            #Rename gear_v to recon_all_v gear in df , and minimorph_v in df2
            df = df.rename(columns={'gear_v': 'recon_all_v'})
            df2 = df2.rename(columns={'gear_v': 'minimorph_v'})

            df = pd.merge(df, df2, how='outer', on=['subject', 'session', 'acquisition', 'session_qc', 'age', 'age_source','sex','input_gear_v'])
        elif not df2.empty:
            df = df2
        

        outdir = os.path.join(project_path, filename)
        df.to_csv(outdir, index=False)
        print(f"Results saved to {outdir}")
        return outdir

        #UPLOADS A FILE TO THE PROJECT INFORMATION TAB
        #project.upload_file(outdir)
    except Exception as e:
        st.warning(f"Exception occurred: {e}")
        

def assemble_csv(derivatives, out_csv="derivatives_summary.csv"):
    """
    Assemble a CSV from a list of derivative file paths.
    """
    df = []
    # For simplicity, let's assume derivatives is a list of file paths
    for deriv in derivatives:
        print(f"Found derivative: {deriv}")
        df_proj = pd.read_csv(deriv)
        df.append(df_proj)


    df = pd.concat(df, axis=0, ignore_index=True)
    outdir = os.path.join("../data", out_csv)
    df.to_csv(outdir)
    df.to_csv(out_csv, index=False)
    return df



st.title("📥 Data Download")
st.write("Download and compile derivatives from multiple projects into a single CSV file.")
st.write("Select the projects and derivative types from the sidebar :point_left:, then click 'Fetch derivatives'.")

# Sidebar inputs
st.sidebar.header("Settings")
#Project IDs is a list generated using fw.projects() by getting the project label


# Now you can access them like this:
API_KEY = os.getenv("FW_CLI_API_KEY")
if API_KEY is None:
    raise ValueError("API_KEY not found. Please add it to your .env file.")

fw = flywheel.Client(api_key=API_KEY)

@st.cache_data(ttl=600)
def get_projects():
    return [p.label for p in fw.projects()]

projects = get_projects()

project_ids = st.sidebar.multiselect("Select Projects", projects)
derivative_type = st.sidebar.radio("Segmentaion Tool:", ["recon-all-clinical", "minimorph","both"])

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
gear_versions = st.sidebar.slider("Last gear versions:", min_value=1, max_value=10, value=3)
#Add date picker for after date
after_date = st.sidebar.date_input("Select date (only fetch analyses after this date):", value=None)

#Add tickboxes if recon all was selected, to select area, thickness, and volume
if derivative_type == "recon-all-clinical" or derivative_type == "both":
    st.sidebar.markdown("**Select outputs to download:**")
    area = st.sidebar.checkbox("Area", value=False)
    thickness = st.sidebar.checkbox("Thickness", value=False)
    volume = st.sidebar.checkbox("Volume", value=True)
    
    keywords = []
    if area:
        keywords.append("area")
    if thickness:
        keywords.append("thickness")
    if volume:
        keywords.append("volume")
    
    if not keywords:
        st.sidebar.warning("Please select at least one output type.")

else:
    keywords = ["volume"]

if derivative_type == "both":
    derivative_type = ["recon-all-clinical", "minimorph"]
else:
    derivative_type = [derivative_type]

if st.sidebar.button("Fetch derivatives"):
    
    all_derivatives = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, proj in enumerate(project_ids):
        st.write(f"Fetching {derivative_type} for {proj}...")
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
        
        derivatives = download_derivatives(proj, derivative_type,gear_versions, keywords, after_date,fw)
        if derivatives:
            all_derivatives.extend([derivatives])
        progress.progress((i+1)/len(project_ids))
    

    if not all_derivatives:
        st.error("No derivatives found. Please check your selections and try again.")
        # st.stop()
    else:    
        df = assemble_csv(all_derivatives)
        
        st.success("Download complete!")
        st.dataframe(df)
        
        # Provide CSV download
        csv_path = "derivatives_summary.csv" #Add today's date to the filename

        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button("Download CSV", f, file_name=csv_path)


