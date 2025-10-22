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
from utils.authentication import login_screen
import concurrent.futures
# import threading
# from concurrent.futures import ThreadPoolExecutor
# from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


# def download_session_data(project,session,project_path,segtool,timestampFilter,work_dir,keywords):
    # session = session.reload()
    # ses_label = session.label
    # sub_label = session.subject.label
    # print(sub_label, ses_label)
    # #status.text(f"Fetching {sub_label} - {ses_label}...")
    # #Iterate through analyses, looking for both gears if derivative_type is "both"
    # analyses = [a for a in session.analyses if a.gear_info is not None and a.gear_info.name in segtool and a.created > timestampFilter] #and a.gear_info.version in gear_versions
    # if st.session_state.input_source == "MRR":
    #     #Exclude analyses where session.acquisition.label contains "gambas" (case insensitive)
    #     analyses = [a for a in analyses if all("gambas" not in acq.label.lower() for acq in session.acquisitions())]
    # elif st.session_state.input_source == "Enhanced (Gambas)":
    #     #Only include analyses where session.acquisition.label contains "gambas" (case insensitive)
    #     analyses = [a for a in analyses if any("gambas" in acq.label.lower() for acq in session.acquisitions())]

    # for analysis in analyses:
    #     # if analysis.gear_info is not None and analysis.gear_info.name == segtool and analysis.created > timestampFilter and analysis.gear_info.version in gear_versions: # and analysis.gear_info.version == gearVersion and analysis.get("job").get("state") == "complete"
    #     # print("pulling: ", segtool, gearVersion)
    #     #st.write(f"Found {segtool} {analysis.gear_info.version} for {sub_label} - {ses_label}")
    #     gear = analysis.gear_info.name

    #     with open(os.path.join(work_dir, '..', "utils","columns.yml"),"r") as f:
    #         tool_map = yaml.load(f, Loader=yaml.SafeLoader)
    #         volumetric_cols = tool_map[gear]
    #     print(st.session_state.df)
    #     for analysis_file in analysis.files:
    #         for keyword in keywords:
    #             if keyword in analysis_file.name:
    #                 file = analysis_file
    #                 file = file.reload()
    #                 # Sanitize our filename and parent path
    #                 download_dir = pv.sanitize_filepath(project_path/sub_label/ses_label,platform='auto')   
    #                 fileName = file.name #(analysis.gear_info.name + "_" + analysis.label + ".csv")

    #                 # Create the path
    #                 if not download_dir.exists():
    #                     download_dir.mkdir(parents=True)
    #                 download_path = download_dir/fileName
    #                 print(download_path)

    #                 # Download the file
    #                 print('Downloading file', ses_label, analysis.label)
    #                 file.download(download_path)

    #                 # Add subject to dataframe
    #                 results = pd.read_csv(download_path) 
    #                 #Insert session.tags as a new column after 'session'
    #                 print(session.tags)
    #                 results.insert(3, 'session_qc', session.tags[-1] if session.tags else 'n/a')
    #                 #If age is empty or NaN, replace with subject.age
    #                 # age_source = 'custom_info'
    #                 # age =  session.info.get('childTimepointAge_months',"n/a")
    #                 # if 'age' in results.columns:
    #                 #     if any(age not in [None, 0, "0"] for age in [session.info.get('childTimepointAge_months',None), session.info.get('age_at_scan_months',None)]):
    #                 #         age_source = 'custom_info'
    #                 #         age =  session.info.get('childTimepointAge_months')
                            
    #                 #         results['age'] = (
    #                 #                         results['age']
    #                 #                         .replace([0, "0", ""], np.nan)  # normalize "empty" values
    #                 #                         .fillna(age)                    # fill NaNs with session age
    #                 #                     )
    #                 #         #For those change, set 'age_source' with value 'custom_info'
                            
    #                 #         #Account for if age_source does not exist
    #                 #         if 'age_source' not in results.columns:
    #                 #             results['age_source'] = 'n/a'
    #                 #         results['age_source'] = results.apply(lambda row: age_source if row['age'] == age else row['age_source'], axis=1)
    #                 results["age_source"] = "custom_info"
    #                 results["age"] = session.info.get('childTimepointAge_months', results.get("age","n/a"))
    #                 results["sex"] = session.info.get('childBiologicalSex', 'n/a')
    #                 results["project"] = project.label
    #                 #Get analysis id
                    
    #                 #Instead of blindly appending, ensure differences in columns are handled
    #                 #Drop "Unnamed" columns if they exist
    #                 results = results.loc[:, ~results.columns.str.contains('^Unnamed')]                                
    #                 #Add analysis.gear_info.version in the results dataframe after the 'session' column
    #                 results.insert(0, f'gear_v', analysis.gear_info.version) #analysis.gear_info.name + "/"+
    #                 #If segmentation tool is minimorph, add prefix mm_ to the volumetric columns
    #                 if gear == "minimorph":
    #                     results["analysis_id_mm"] = analysis.id
    #                     results.rename(columns={col: f'mm_{col}' for col in volumetric_cols if col in results.columns}, inplace=True)
    #                 else:
    #                     results["analysis_id_ra"] = analysis.id
    #                     results.columns = results.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
    #                     results.rename(columns={col: f'ra_{col}' for col in volumetric_cols if col in results.columns}, inplace=True)

    #                 #st.dataframe(results)
    #                 if gear == "recon-all-clinical":
    #                     #If session and acquisition combo already exists in st.session_state.df, only keep the latest version based on the gear version
    #                     if not st.session_state.df.empty:
    #                         existing = st.session_state.df[(st.session_state.df['subject'] == sub_label) & (st.session_state.df['session'] == ses_label) & (st.session_state.df['acquisition'] == results['acquisition'].iloc[0])]
    #                         if not existing.empty:
    #                             existing_version = existing[f'gear_v'].iloc[0]
    #                             if version.parse(analysis.gear_info.version) <= version.parse(existing_version):
    #                                 #status.text(f"Skipping {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} as existing version {existing_version} is newer or equal to {analysis.gear_info.version}")
    #                                 print(f"Skipping {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} as existing version {existing_version} is newer or equal to {analysis.gear_info.version}")

    #                                 continue
    #                             else:
    #                                 #Drop the existing row
    #                                 st.session_state.df = st.session_state.df.drop(existing.index)
    #                                 print(f"Replacing {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} with newer version {analysis.gear_info.version}")
    #                                 #status.text(f"Replacing {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} with newer version {analysis.gear_info.version}")
    #                     #Merge results with st.session_state.df on project, subject, session, acquisition, age, age_source 
    #                     if st.session_state.df.empty:
    #                         st.session_state.df = results

    #                     st.session_state.df = pd.merge(st.session_state.df, results, how='outer', on=results.columns.intersection(st.session_state.df.columns).tolist())
    #                     #st.session_state.df = pd.concat([st.session_state.df, results], ignore_index=True)
    #                     #st.dataframe(st.session_state.df)

    #                 elif gear == "minimorph":
    #                     if not st.session_state.df2.empty:
    #                         existing = st.session_state.df2[(st.session_state.df2['subject'] == sub_label) & (st.session_state.df2['session'] == ses_label) & (st.session_state.df2['acquisition'] == results['acquisition'].iloc[0])]
    #                         if not existing.empty:
    #                             existing_version = existing[f'gear_v'].iloc[0]
    #                             if version.parse(analysis.gear_info.version) <= version.parse(existing_version) : #existing_version >= analysis.gear_info.version:
    #                                 #status.text(f"Skipping {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} as existing version {existing_version} is newer or equal to {analysis.gear_info.version}")
    #                                 print(f"Skipping {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} as existing version {existing_version} is newer or equal to {analysis.gear_info.version}")

    #                                 continue
    #                             else:
    #                                 #Drop the existing row
    #                                 st.session_state.df2 = st.session_state.df2.drop(existing.index)
    #                                 #status.text(f"Replacing {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} with newer version {analysis.gear_info.version} (previously {existing_version})")
    #                                 print(f"Replacing {sub_label} - {ses_label} - {results['acquisition'].iloc[0]} with newer version {analysis.gear_info.version} (previously {existing_version})")

    #                     st.session_state.df2 = pd.concat([st.session_state.df2, results], ignore_index=True)

    #                 #Delete the file after reading it in
    #                 os.remove(download_path)

def download_derivatives(project_id, segtool, input_source, keywords, timestampFilter,fw):
    """
    Download derivative files from Flywheel analyses based on specified criteria.
    Parameters:
    - project_id: The Flywheel project ID to search within.
    - derivative_type: The type of derivative to look for (e.g., 'recon-all-clinical', 'minimorph').
    Returns:
    - outdir: Path to the compiled CSV file containing results.
    """

    # config = dotenv_values(os.path.join(Path(__file__).parent/".."), ".env")

    if timestampFilter is None:
        timestampFilter = datetime(2020, 1, 1, 0, 0, 0, 0, pytz.UTC)


    
    print(f"User: {fw.get_current_user().firstname} {fw.get_current_user().lastname}")
    project = fw.projects.find_first(f'label={project_id}')
    st.info(f"Project: {project_id}  \nSubjects n = {len(project.subjects())}  \nSessions n = {len(project.sessions())}")

    # Create a work directory in our local "home" directory
    #Pass the directory relative to the script
    
    #work_dir = Path(Path.home()/'../data/', platform='auto')
    # If it doesn't exist, create it
    if not data_dir.exists():
        data_dir.mkdir(parents = True)
    # Create a custom path for our project (we may run this on other projects in the future) and create if it doesn't exist
    project_path = pv.sanitize_filepath(data_dir/project.label, platform='auto')
    if not project_path.exists():
        project_path.mkdir(parents = True)
    # Preallocate lists
    st.session_state.df = pd.DataFrame() #for all analysis results

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

    max_workers = 4  # Adjust based on your system/API limits
    results = []

    #### Multithreading download ####
    
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # Submit all tasks
    #     future_to_session = {
    #         executor.submit(download_session_data, project,session,project_path,segtool,timestampFilter,work_dir,keywords): session 
    #         for session in sessions
    #     }
        
    #     # Process completed tasks
    #     for i, future in enumerate(concurrent.futures.as_completed(future_to_session)):
    #         result = future.result()
    #         results.append(result)
            
            # Update progress
            # progress.progress((i + 1) / len(sessions))
            # status.text(f"Completed {i + 1}/{len(sessions)}: {result}")
    #### End multithreading ####

    ############ Single thread download ############

    for i, session in enumerate(sessions[:5]):
        session = session.reload()
        ses_label = session.label
        sub_label = session.subject.label
        print(sub_label, ses_label)
        status.text(f"Fetching {sub_label} - {ses_label}...")
        #Iterate through analyses, looking for both gears if derivative_type is "both"
        analyses = [a for a in session.analyses if a.gear_info is not None and a.gear_info.name in segtool and a.created > timestampFilter] #and a.gear_info.version in gear_versions
        if input_source == "MRR":
            #Exclude analyses where session.acquisition.label contains "gambas" (case insensitive)
            analyses = [a for a in analyses if "gambas" not in a.label.lower()]
        elif input_source == "Enhanced (Gambas)":
            #Only include analyses where session.acquisition.label contains "gambas" (case insensitive)
            analyses = [a for a in analyses if "gambas" in a.label.lower()]

        #Grab the last created analysis for each segtool
        analyses_filtered = []
        for tool in segtool:
            tool_analyses = [a for a in analyses if a.gear_info.name == tool]
            if tool_analyses:
                latest_analysis = max(tool_analyses, key=lambda a: a.created)
                analyses_filtered.append(latest_analysis)
                print("Selected latest analysis for tool ", tool, " created on ", latest_analysis.created, " version ", latest_analysis.gear_info.version)


        # Initialize session-level results dataframe
        session_results = pd.DataFrame()
        analysis_results = pd.DataFrame()
        for analysis in analyses_filtered:
            # if analysis.gear_info is not None and analysis.gear_info.name == segtool and analysis.created > timestampFilter and analysis.gear_info.version in gear_versions: # and analysis.gear_info.version == gearVersion and analysis.get("job").get("state") == "complete"
            # print("pulling: ", segtool, gearVersion)
            #st.write(f"Found {segtool} {analysis.gear_info.version} for {sub_label} - {ses_label}")
            gear = analysis.gear_info.name

            with open(os.path.join(work_dir, '..', "utils","columns.yml"),"r") as f:
                tool_map = yaml.load(f, Loader=yaml.SafeLoader)
                volumetric_cols = tool_map[gear]

            # Initialize analysis-level results dataframe
            
            for analysis_file in analysis.files:
                for keyword in keywords:
                    if keyword in analysis_file.name:
                        if st.session_state.fw_session_info == "Yes":
                            session = session.reload()
                            session_info = session.info
                            session_info_df = pd.DataFrame([session_info])
                            


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
                        df = pd.read_csv(download_path) 
                        
                        #Insert session.tags as a new column after 'session'
                        print(session.tags)
                        df.insert(3, 'session_qc', session.tags[-1] if session.tags else 'n/a')
                        #If age is empty or NaN, replace with subject.age
                        if 'age' in df.columns:
                            if any(age not in [None, 0, "0"] for age in [session.info.get('childTimepointAge_months',None), session.info.get('age_at_scan_months',None)]):
                                age_source = 'custom_info'
                                age =  session.info.get('childTimepointAge_months')
                                
                                df['age'] = (
                                                df['age']
                                                .replace([0, "0", ""], np.nan)  # normalize "empty" values
                                                .fillna(age)                    # fill NaNs with session age
                                            )
                                #For those change, set 'age_source' with value 'custom_info'
                                
                                #Account for if age_source does not exist
                                if 'age_source' not in df.columns:
                                    df['age_source'] = 'n/a'
                                df['age_source'] = df.apply(lambda row: age_source if row['age'] == age else row['age_source'], axis=1)

                        df["childTimepointAge_months"] = session.info.get('childTimepointAge_months', df.get("age","n/a"))
                        df["childBiologicalSex"] = session.info.get('childBiologicalSex', 'n/a')
                        

                        df["project"] = project.label
                        #Get analysis id
                        
                        #Instead of blindly appending, ensure differences in columns are handled
                        #Drop "Unnamed" columns if they exist
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]                                
                        #Add analysis.gear_info.version in the results dataframe after the 'session' column
                        df.insert(0, f'gear_v', analysis.gear_info.version) #analysis.gear_info.name + "/"+
                        #If segmentation tool is minimorph, add prefix mm_ to the volumetric columns
                        if gear == "minimorph":
                            df["analysis_id_mm"] = analysis.id
                            df["gear_v_minimorph"]= analysis.gear_info.version
                            df.rename(columns={col: f'mm_{col}' for col in volumetric_cols if col in df.columns}, inplace=True)
                        else:
                            df["analysis_id_ra"] = analysis.id
                            df["gear_v_recon_all"]= analysis.gear_info.version
                            df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
                            df.rename(columns={col: f'ra_{col}' for col in volumetric_cols if col in df.columns}, inplace=True)

                        #st.dataframe(results)
                        # if gear == "recon-all-clinical":
                        #     #If session and acquisition combo already exists in st.session_state.df, only keep the latest version based on the gear version
                        #     if not st.session_state.df.empty:
                        #         existing = st.session_state.df[(st.session_state.df['subject'] == sub_label) & (st.session_state.df['session'] == ses_label) & (st.session_state.df['acquisition'] == df['acquisition'].iloc[0])]
                        #         if not existing.empty:
                        #             existing_version = existing[f'gear_v'].iloc[0]
                        #             if version.parse(analysis.gear_info.version) <= version.parse(existing_version):
                        #                 status.text(f"Skipping {sub_label} - {ses_label} - {df['acquisition'].iloc[0]} as existing version {existing_version} is newer or equal to {analysis.gear_info.version}")
                        #                 continue
                        #             else:
                        #                 #Drop the existing row
                        #                 st.session_state.df = st.session_state.df.drop(existing.index)
                        #                 status.text(f"Replacing {sub_label} - {ses_label} - {df['acquisition'].iloc[0]} with newer version {analysis.gear_info.version}")
                            
                            #Add session.info key-value pairs to the dataframe if selected
                            #Turn dictionary into dataframe
                            
                                #results = pd.concat([session_info_df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

                            # if st.session_state.df.empty:
                            #     st.session_state.df = df
                            # else:
                            #     results = results.combine_first(df)

                            # if st.session_state.fw_session_info == "Yes":
                            #     #Merge with results dataframe
                            #     results = results.combine_first(session_info_df)

                            #Add columns current results of this analysis

                            # common_cols = results.columns.intersection(st.session_state.df.columns).tolist()
                            # for col in common_cols:
                            #     st.session_state.df[col] = st.session_state.df[col].astype(object)
                            #     results[col] = results[col].astype(object)

                            #st.session_state.df = pd.merge(st.session_state.df, results, how='outer', on=results.columns.intersection(st.session_state.df.columns).tolist())
                            #st.session_state.df = pd.concat([st.session_state.df, results], ignore_index=True)
                            #st.dataframe(st.session_state.df)

                        # elif gear == "minimorph":
                        #     if not st.session_state.df2.empty:
                        #         existing = st.session_state.df2[(st.session_state.df2['subject'] == sub_label) & (st.session_state.df2['session'] == ses_label) & (st.session_state.df2['acquisition'] == df['acquisition'].iloc[0])]
                        #         if not existing.empty:
                        #             existing_version = existing[f'gear_v'].iloc[0]
                        #             if version.parse(analysis.gear_info.version) <= version.parse(existing_version) : #existing_version >= analysis.gear_info.version:
                        #                 status.text(f"Skipping {sub_label} - {ses_label} - {df['acquisition'].iloc[0]} as existing version {existing_version} is newer or equal to {analysis.gear_info.version}")
                                        
                        #                 continue
                        #             else:
                        #                 #Drop the existing row
                        #                 st.session_state.df2 = st.session_state.df2.drop(existing.index)
                        #                 status.text(f"Replacing {sub_label} - {ses_label} - {df['acquisition'].iloc[0]} with newer version {analysis.gear_info.version} (previously {existing_version})")
                            
                        #     st.session_state.df2 = pd.concat([st.session_state.df2, results], ignore_index=True)

                        
                        analysis_results = analysis_results.combine_first(df)
                        if st.session_state.fw_session_info == "Yes":
                            #Merge with results dataframe
                            analysis_results = analysis_results.combine_first(session_info_df)
                        #Delete the file after reading it in
                        os.remove(download_path)
            
            # Combine analysis results with session results
            if not analysis_results.empty:
                session_results = session_results.combine_first(analysis_results)
        
        # Concatenate session results to main dataframe only once per session
        if not analysis_results.empty:
            st.session_state.df = pd.concat([st.session_state.df, analysis_results], ignore_index=True)

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
        #st.session_state.df = pd.concat(st.session_state.df, axis=0, ignore_index=True)
        #If st.session_state.df is empty, add a message and exit
        if st.session_state.df.empty:
            st.warning("No results found for the specified criteria.")
            return None
        
        if not st.session_state.df.empty:
            if 'input_gear_v' in st.session_state.df.columns:
                cols = st.session_state.df.columns.tolist()
                cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('input_gear_v')))
                st.session_state.df = st.session_state.df[cols]
            if 'scanner_software_v' in st.session_state.df.columns:
                cols = st.session_state.df.columns.tolist()
                cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('scanner_software_v')))
                st.session_state.df = st.session_state.df[cols]
        # if not st.session_state.df2.empty:
        #     if 'input_gear_v' in st.session_state.df2.columns:
        #         cols = st.session_state.df2.columns.tolist()
        #         cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('input_gear_v')))
        #         st.session_state.df2 = st.session_state.df2[cols]
        #     if 'scanner_software_v' in results.columns:
        #         cols = st.session_state.df2.columns.tolist()
        #         cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index('scanner_software_v')))
        #         st.session_state.df2 = st.session_state.df2[cols]

        #If we have two dataframes, merge
        # if not st.session_state.df.empty and not st.session_state.df2.empty:
        #     #For every row for column "gear_v" in st.session_state.df2, add a prefix "minimorph/" to the value
        #     #st.session_state.df2['gear_v'] = st.session_state.df2['gear_v'].apply(lambda x: 'minimorph/' + x if not x.startswith('minimorph/') else x)
        #     #st.session_state.df['gear_v'] = st.session_state.df['gear_v'].apply(lambda x: 'recon-all-clinical/' + x if not x.startswith('recon-all-clinical/') else x)
        #     #Rename gear_v to recon_all_v gear in st.session_state.df , and minimorph_v in st.session_state.df2
        #     st.session_state.df = st.session_state.df.rename(columns={'gear_v': 'recon_all_v'})
        #     st.session_state.df2 = st.session_state.df2.rename(columns={'gear_v': 'minimorph_v'})
        #     #Get common columns between the two dataframes
            
        #     common_cols = st.session_state.df.columns.intersection(st.session_state.df2.columns).tolist()
        #     print(common_cols)
        #     st.session_state.df = pd.merge(st.session_state.df, st.session_state.df2, how='outer', on= common_cols )#['subject', 'session', 'acquisition', 'session_qc', 'age', 'age_source','sex','input_gear_v'])
        # elif not st.session_state.df2.empty:
        #     st.session_state.df = st.session_state.df2
        

        outdir = os.path.join(project_path, filename)
        st.session_state.df.to_csv(outdir, index=False)
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
    front_cols = ['project', 'subject', 'session', 'childTimepointAge_months','childBiologicalSex','studyTimepoint','session_qc','acquisition']
    ra_cols = [col for col in cols if col.startswith('ra_')]
    mm_cols = [col for col in cols if col.startswith('mm_')]
    other_cols = [col for col in cols if col not in front_cols + ra_cols + mm_cols]
    new_order = front_cols + ra_cols + mm_cols + other_cols

    st.session_state.df = st.session_state.df[new_order]


    #Add projects separated by _ to the filename
    project_str = "_".join([item for sublist in projects for item in sublist])
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
derivative_type = st.sidebar.radio("Segmentation Tool:", ["recon-all-clinical", "minimorph","Both"])
#Add radio button yes or no to include flywheel session information in the download
st.session_state.fw_session_info = st.sidebar.radio("Include Flywheel Session Info (tags, custom info) in download?", ["Yes", "No"], index=0)
st.session_state.input_source = st.sidebar.radio("Structural Image Segmented:", ["MRR", "Enhanced (Gambas)", "Both"], index=0)
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
after_date = st.sidebar.date_input("Select date (only fetch analyses after this date):", value=None)

#Add tickboxes if recon all was selected, to select area, thickness, and volume
if derivative_type == "recon-all-clinical" or derivative_type == "Both":
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

if derivative_type == "Both":
    derivative_type = ["recon-all-clinical", "minimorph"]
else:
    derivative_type = [derivative_type]

if st.sidebar.button("Fetch derivatives"):
    
    derivative_paths = []
    
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
        st.info(f"Download for {proj} started in background.")
        # print(st.session_state)
        derivatives = download_derivatives(proj, derivative_type,st.session_state.input_source, keywords, after_date,fw)
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


