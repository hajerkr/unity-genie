import streamlit as st
from flywheel.client import Client
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
        gear = fw.gears.find_first(f"gear.name={gearname}")
        #Get gear version
        gear_version = gear.gear.version if gear else "Unknown"
        return (
            asys.gear_info is not None
            and asys.gear_info.get('name') == gearname
            and asys.job is not None
            and asys.job.get('state') == 'complete'
            #ensure last gear version is ran
            and (not latest_version or asys.gear_info.get('version') == gear_version)
        )
    
    # asys=asys.reload()
    # return (
    #     asys.gear_info is not None
    #     and gearname in asys.gear_info.get('name') 
    #     and asys.job is not None
    #     and asys.job.get('state') == 'complete'
    # )

def is_failed(asys,gearname, latest_version=False):
    asys=asys.reload()
    gear = fw.gears.find_first(f"gear.name={gearname}")
    #Get gear version
    gear_version = gear.gear.version if gear else "Unknown"
    return (
        asys.gear_info is not None
        and gearname in asys.gear_info.get('name')
        and asys.job is not None
        and asys.job.get('state') == 'failed'
        and (not latest_version or asys.gear_info.get('version') == gear_version)
    )

def is_pending(asys,gearname):
    asys=asys.reload()
    return (
        asys.gear_info is not None
        and gearname in asys.gear_info.get('name')
        and asys.job is not None
        and asys.job.get('state') in ['pending', 'running']
    )
def run_gambas_jobs(fw, project):
    job_list = []
    processed_sessions = 0
    skipped_sessions = 0
    failed_sessions = 0
    status = st.empty()
    failed_sessions_list = []
    for session in project.sessions():
        #If it hasn't ran gambas yet, run it
        # Find the most recent gambas analysis
        gambas_file = find_latest_gambas_file(session)
        if gambas_file:
            status.text(f"GAMBAS already run for {session.label}, skipping.")
            skipped_sessions += 1
            continue

        job_id = submit_job(fw, session,"gambas")
        if job_id:
            job_list.append(job_id)
            processed_sessions += 1
            status.text(f"🚀 Submitted GAMBAS job (ID: {job_id}) for session {session.label}")
        else:
            failed_sessions += 1
            failed_sessions_list.append(session.label)
            status.text(f"❌ Failed to submit GAMBAS job for session {session.label}")
    


    # Summary
    # Flatten job_list if it contains nested lists
    if any(isinstance(i, list) for i in job_list):
        job_list = [job for sublist in job_list for job in sublist]

    st.info(f"\n📊 Summary:  \n   ✅ Jobs submitted: {processed_sessions}\n   ⏭️ Sessions skipped: {skipped_sessions}\n   ❌ Sessions failed: {failed_sessions}\n   📋 Total job IDs: {len(job_list)}")
    #Return CSV with skipped sessions
    if failed_sessions_list.append(session.label):
        skipped_sessions_str = "\n".join(failed_sessions_list.append(session.label))
        st.download_button(
            label="Download Failed Sessions",
            data=skipped_sessions_str,
            file_name="skipped_sessions_gambas.txt",
            mime="text/plain"
        )
    
    return job_list


def submit_job(fw, session,gearname):

    
    EXCLUDE_PATTERNS = ['Segmentation', 'Align', 'Mapping',"Localizer","Gray_White"]
    INCLUDE_PATTERN = 'T2'
    # PLANE_TYPES = ['AXI']
    GEAR_PLANE_TYPES = {"gambas": ['AXI'], "qa": ['AXI','SAG','COR'],"mriqc": ['AXI','SAG','COR']}
    status = st.empty()
    job_ids = []
    # Look at every acquisition in the session
    for acquisition in session.acquisitions.iter():
        inputs = {}
        print(f"Checking acquisition: {acquisition.label}...")
        acquisition = acquisition.reload()
        for file in acquisition.files:
        # We only want anatomical Nifti's
            if file.type == 'nifti' and INCLUDE_PATTERN in file.name:
                if all(pattern not in file.name for pattern in EXCLUDE_PATTERNS):
                    for plane in GEAR_PLANE_TYPES.get(gearname):
                        if plane in file.name:
                            inputs["input"] = file
                            break
        if inputs:                           
            try:
            # The destination for this analysis will be on the session
                dest = acquisition
                gear = fw.lookup(f'gears/{gearname}')
                time_fmt = '%d-%m-%Y_%H-%M-%S'
                analysis_tag = gearname
                analysis_label = f'{analysis_tag}_{datetime.now().strftime(time_fmt)}'
                if gearname == "mriqc": #Not analysis gear, so do not include analysis label
                    inputs["nifti"] = inputs.pop("input") #Rename input to nifti for mriqc
                    job_id = gear.run(
    
                        inputs=inputs,
                        destination=dest,
                        tags=['batch','qc'],
                        config={
                                "measurement": "auto-detect",
                                "save_derivatives": True,
                                "save_outputs": True,
                                "verbose_reports": True,
                                "include_rating_widget": True
                            }
                    )
                else:
                    
                    job_id = gear.run(
                        analysis_label=analysis_label,
                        inputs=inputs,
                        destination=dest,
                        tags=['batch'],
                        config={
                        
                            # "prefix": analysis_tag,
                        }
                    )
                
                
                job_ids.append(job_id)
                # return job_id
                
            except Exception as e:
                status.text(f"WARNING: Job cannot be sent for {dest.label}. Error: {e}")
    return job_ids

def run_jobs(fw, project, gearname, gambas=False, include_pattern=None,analysis_tag=None):
    """
    Run seg jobs on the most recent 'gambas' (or MRR) analysis for each session
    if segementation hasn't already been completed.
    """
    
    # Configuration
    
    gear = fw.lookup(f'gears/{gearname}')
    # gear_version = '0.4.8'
 
    # Initialize job tracking
    job_list = []
    processed_sessions = 0
    skipped_sessions = 0
    failed_sessions = 0
    skipped_sessions_list = []
    status = st.empty()
    st.info(f"🚀 Starting {gearname} job submission.  \n📁 Processing project: {project.label}")
    project_ = fw.projects.find_first(f'label={project.label}')
    project = project_.reload()
    # Loop through sessions
    if st.session_state.debug_mode:
        sessions = project.sessions()[:4]
        #st.info("⚠️ Debug Mode: Processing only first 4 sessions.")
    else:
        sessions = project.sessions()

    for session in sessions:
        if not (session.subject.label.startswith("137-")): #Ensure this does not run on the phantom - waste of resource and nonsense results
            # for session in subject.sessions():
                session = session.reload()
                session_id = f"{project.label}/{session.subject.label}/{session.label}"
                status.text(f"\n🔍 Checking session: {session_id} for subject {session.subject.label}")
                print(f"\n🔍 Checking session: {session_id} for subject {session.subject.label}")
                try:
                    # Check if gear already completed specifically for this input, or has a submitted job already
                    if has_completed_asys(session, gearname, gambas=gambas):
                        status.text(f"✅ {gearname} with this input already complete, skipping {session_id}")
                        print(f"✅ {gearname} with this input already complete for session {session_id}, skipping.")
                        skipped_sessions += 1
                        
                        continue
                    elif has_pending_asys(session, gearname, gambas=gambas):
                        status.text(f"⏳ {gearname} with this input already pending/running, skipping {session_id}")
                        print(f"⏳ {gearname} with this input already pending/running for session {session_id}, skipping.")
                        skipped_sessions += 1
                        
                        continue
                    elif has_failed_asys(session, gearname, gambas=gambas):
                        status.text(f"❌ {gearname} with this input has previously failed, skipping {session_id}")
                        print(f"❌ {gearname} with this input has previously failed for session {session_id}, skipping.")
                        skipped_sessions += 1
                        
                        continue
                    
                    if gearname in ["mriqc"]:
                        job_id =  submit_job(fw, session, gearname)
                        job_list.extend(job_id)
                        processed_sessions += 1
                        status.text(f"🚀 Submitted {gearname} job (ID: {job_id}) for session {session_id}")
                        print(f"🚀 Submitted {gearname} job (ID: {job_id}) for session {session_id}")
                        
                    ### GAMBAS CHECKS ####
                    elif gambas:
                        # Find the most recent gambas analysis
                        gambas_file = find_latest_gambas_file(session)
                        if not gambas_file:
                            status.text(f"⚠️ No suitable gambas file found. Submitting a gambas job for session {session_id}...")
                            #Add a function to run gambas if nothing has been found

                            
                            job_id = submit_job(fw, session,"gambas")
                            try:
                                if job_id:
                                    job_list.append(job_id)
                                    status.text(f"🚀 Submitting GAMBAS Job : Check Jobs Log")
                                else:
                                    skipped_sessions += 1
                            except Exception as e:
                                    status.text(f"WARNING: Job cannot be sent. Error: {e}")
                            
                            processed_sessions += 1
                            
                            continue
        
                        elif gambas_file:
                            print(f"✅ Found gambas file: {gambas_file.name}")
                            # Submit seg job
                            job_id = submit_seg_job(gear, session, gambas=True, input_file=gambas_file, analysis_tag=analysis_tag)
                            job_list.append(job_id)
                            processed_sessions += 1

                            print(f"🚀 Submitted {gearname} job (ID: {job_id})")
                            
                            
                    ### MRR CHECKS ####
                    elif not gambas and gearname != 'freesurfer-recon-all':
                        # Submit seg job without MRR input
                        
                        inputfile = None
                        mrr_results = fw.search(
                            {"structured_query": f"session._id = {session.id} AND analysis.label CONTAINS mrr",
                            "return_type":"analysis"}
                        )

                        mrr_matches = [r.analysis.reload() for r in mrr_results if is_complete(r.analysis,"mrr")]
                        
                       
                        # if not mrr_matches:
                        #     for acq in session.acquisitions():
                        #         acq = acq.reload()
                        #         mrr_matches = [asys for asys in acq.analyses if is_complete(asys,"mrr")]
                        #         if mrr_matches:
                        #             break

                        if not mrr_matches:
                            status.text(f"⚠️ No suitable MRR analysis found. Skipping session {session_id}.")
                            print(f"⚠️ No suitable MRR analysis found for session {session_id}. Skipping.")
                            skipped_sessions += 1
                            continue
                    
                        last_run_date = max([asys.created for asys in mrr_matches])                        
                        last_run_analysis = [asys for asys in mrr_matches if asys.created == last_run_date]
                        last_run_analysis = last_run_analysis[0]
                        # print(last_run_analysis.label)
                        # print(len(last_run_analysis.files))
                        for file in last_run_analysis.files:  
                            # print(file.name)
                            if re.search(r'mrr.*\.nii.gz', file.name):
                                inputfile = file

                        job_id = submit_seg_job(gear, session, gambas=False, input_file=inputfile,analysis_tag=analysis_tag)
                        job_list.append(job_id)
                        processed_sessions += 1
                        status.text(f"🚀 Submitted {gearname} job")
                        print(f"🚀 Submitted {gearname} job (ID: {job_id})")

                    elif gearname == 'freesurfer-recon-all':                     
                        
                        # Find T1w acquisition based on label string
                        inputfile = None
                        for acquisition in session.acquisitions():
                            acquisition = acquisition.reload()
                            if t1w_label_string in acquisition.label:
                                for file in acquisition.files:
                                    if file.type == 'nifti':
                                        inputfile = file
                                        print(f"✅ Found T1w file: {inputfile.name} in acquisition {acquisition.label}")
                                        break
                            if inputfile:
                                break
                        if not inputfile:
                            status.text(f"⚠️ No suitable T1w acquisition found with label containing '{t1w_label_string}'. Skipping session.")
                            skipped_sessions += 1
                            #Need to log this
                            skipped_sessions_list.append(session.label)
                            continue
                        # Submit seg job with T1w input
                        job_id = submit_seg_job(gear, session, gambas=False, input_file=inputfile,analysis_tag=analysis_tag)
                        
                except Exception as e:
                    print(f"Exception caught for session {session_id}: ", traceback.format_exc())
                    status.text(f"❌ Error processing session {session_id}: {str(e)}")
                    failed_sessions += 1
                    continue
        
    # Summary
    st.info(f"\n📊 Summary:  \n   ✅ Jobs submitted: {processed_sessions}\n   ⏭️ Sessions skipped: {skipped_sessions}\n   ❌ Sessions failed: {failed_sessions}\n   📋 Total job IDs: {len(job_list)}")
    #Return CSV with skipped sessions
    if skipped_sessions_list:
        skipped_sessions_str = "\n".join(skipped_sessions_list)
        st.download_button(
            label="Download Skipped Sessions",
            data=skipped_sessions_str,
            file_name="skipped_sessions.txt",
            mime="text/plain"
        )
    
    return job_list

 
def has_completed_asys(session, gearname,gambas=False):
    """
    Check if session already has a completed segmentation analysis of the target version.
    """

    query = ""
    if gambas:
        query = f"session._id = {session.id} AND (analysis.label CONTAINS {gearname} AND analysis.label CONTAINS gambas)"
    else:
        query = f"session._id = {session.id} AND (analysis.label CONTAINS {gearname} AND NOT(analysis.label CONTAINS gambas))"

    gear_results = fw.search(
        {"structured_query": query,
        "return_type":"analysis"}
    )
    gear_matches = [r.analysis.reload() for r in gear_results if is_complete(r.analysis,gearname, st.session_state.latest_version )]

    if gear_matches:
        return True
    
    # for analysis in session.analyses:
    #     if not analysis.gear_info:
    #         continue
            
    #     gear_name = analysis.gear_info.get('name', '').lower()
    #     #gear_version = analysis.gear_info.get('version', '')
        
    #     if gear.gear.name in gear_name : #and gear_version == target_version:
    #         job_state = analysis.job.get('state') if analysis.job else None
    #         if gambas and 'gambas' in analysis.label.lower() and job_state == 'complete':
    #             return True
    #         elif not gambas and 'gambas' not in analysis.label.lower() and job_state == 'complete':
    #             return True
    return False

def has_pending_asys(session, gearname, gambas=False):
    """
    Check if session already has a pending segmentation analysis of the target version.
    """

    query = ""
    if gambas:
        query = f"session._id = {session.id} AND (analysis.label CONTAINS {gearname} AND analysis.label CONTAINS gambas)"
    else:
        query = f"session._id = {session.id} AND (analysis.label CONTAINS {gearname} AND NOT(analysis.label CONTAINS gambas))"


    gear_results = fw.search(
        {"structured_query": query,
        "return_type":"analysis"}
    )
    gear_matches = [r.analysis.reload() for r in gear_results if is_pending(r.analysis,gearname)]

    if gear_matches:
        return True
    
    # for analysis in session.analyses:
    #     if not analysis.gear_info:
    #         continue
            
    #     gear_name = analysis.gear_info.get('name', '').lower()
    #     #gear_version = analysis.gear_info.get('version', '')
        
    #     if gear.gear.name in gear_name : #and gear_version == target_version:
    #         job_state = analysis.job.get('state') if analysis.job else None
    #         if gambas and 'gambas' in analysis.label.lower() and job_state in ['pending', 'running']:
    #             return True
    #         elif not gambas and 'gambas' not in analysis.label.lower() and job_state in ['pending', 'running']:
    #             return True
    return False

def has_failed_asys(session, gearname, gambas=False):
    """
    Check if session already has a failed segmentation analysis of the target version.
    """

    query = ""
    if gambas:
        query = f"session._id = {session.id} AND (analysis.label CONTAINS {gearname} AND analysis.label CONTAINS gambas)"
    else:
        query = f"session._id = {session.id} AND (analysis.label CONTAINS {gearname} AND NOT(analysis.label CONTAINS gambas))"


    gear_results = fw.search(
        {"structured_query": query,
        "return_type":"analysis"}
    )
    gear_matches = [r.analysis.reload() for r in gear_results if is_failed(r.analysis,gearname,st.session_state.latest_version)]

    if gear_matches:
        return True
    
    # for analysis in session.analyses:
    #     if not analysis.gear_info:
    #         continue
            
    #     gear_name = analysis.gear_info.get('name', '').lower()
    #     #gear_version = analysis.gear_info.get('version', '')
        
    #     if gear.gear.name in gear_name : #and gear_version == target_version:
    #         job_state = analysis.job.get('state') if analysis.job else None
    #         if gambas and 'gambas' in analysis.label.lower() and job_state == 'failed':
    #             return True
    #         elif not gambas and 'gambas' not in analysis.label.lower() and job_state == 'failed':
    #             return True
    return False
 
def find_latest_gambas_file(session):
    """
    Find the most recent gambas analysis file in the session or its acquisitions.
    Returns the first suitable gambas file ending with 'rec-axi_T2w_gambas.nii.gz' from the latest gambas analysis, or None if not found.
    """
    print(f"   Checking analyses in session")
    
    gambas_analyses = []
    gambas_results = fw.search(
        {"structured_query": f"session._id = {session.id} AND analysis.label CONTAINS gambas",
        "return_type":"analysis"}
    )
    gambas_analyses = [r.analysis for r in gambas_results if is_complete(r.analysis,"gambas")]
    latest_gambas = gambas_analyses[-1] if gambas_analyses else None
    
    if not gambas_analyses:
        print("   No gambas analyses found in session or acquisitions")
        return None
    
    print(f"   Found {len(gambas_analyses)} gambas analysis(es) total")
    
    # Use the latest gambas analysis (assuming they're ordered chronologically)
    # latest_gambas = gambas_analyses[-1]
    print(f"   Using latest gambas analysis: {latest_gambas.label}")
    
    # Debug: Print all files in the analysis
    
    
    #Try to reload the analysis
    try:
        latest_gambas = latest_gambas.reload()
        print(f"   Reloaded analysis successfully.")
    except Exception as e:
        print(f"   Warning: Could not reload analysis. Potentially manually created analysis container. Error: {e}")

    print(f"   Files in analysis: {[f.name for f in latest_gambas.files]}")
    # Find gambas output files - specifically look for files ending with "rec-axi_T2w_gambas.nii.gz"
    pattern = re.compile(r"(gambas|ResCNN)\.nii\.gz$", re.IGNORECASE)

    gambas_files = [
        f for f in latest_gambas.files
        if pattern.search(f.name)
    ]

    if not gambas_files: 
        print(f"   No files ending with 'rec-axi(_run-XX)_T2w_gambas.nii.gz' found in analysis {latest_gambas.label}")
        return None
    
    print(f"   Found {len(gambas_files)} gambas file(s): {[f.name for f in gambas_files]}")
    
    # Return the first matching file
    return gambas_files[0]
 
def is_gambas_analysis(analysis):
    """
    Check if an analysis is a gambas analysis by checking gear name or analysis label.
    """
    # Check gear name - must be exactly 'gambas' gear
    if analysis.gear_info and analysis.gear_info.get('name'):
        gear_name = analysis.gear_info.get('name').lower()
        if gear_name == 'gambas':
            return True
    
    # If no gear_info, check the analysis label for gambas version pattern
    elif analysis.label:
        label = analysis.label.lower()
        # Look for patterns like 'gambas/0.4.14' or 'gambas/0.4.17'
        if "gambas" in label and ("0.4.17" in label or "0.4.14" in label):
        # if pattern.search(label):
            return True
    
    return False
 
def submit_seg_job(gear, session, input_file, gambas=False,analysis_tag=None):
    """
    Submit a segmentation analysis job for the given session and gambas file.
    """
    inputs = {'input': input_file}
    analysis_label = ''
    gear_name = gear.gear.name
    # Create a unique analysis label with timestamp and gambas identifier
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if gambas:
        analysis_label = f'{gear_name}_gambas_{timestamp}'
    else:
        analysis_label = f'{gear_name}_mrr-axireg_{timestamp}'

    #Set up config according to the gear submitted
    config = {}
    if gear_name =="minimorph":
        if "36M" in session.subject.label:
                config = {"age": '24M'}
        elif "24M" in session.subject.label:
                config = {"age": '24M'}
        elif "12M" in session.subject.label:
                config = {"age": '12M'}
        elif "6M" in session.subject.label:
                config = {"age": '6M'}
        elif "3M" in session.subject.label:
                config = {"age": '3M'}
        else:
            config= {"age": "None"}
            
    elif gear_name == "infant-freesurfer":
        config = {
        "newborn": False,
        "age": None
        }
   
    print(config)
    # Submit the job
    job_id = gear.run(
        analysis_label=analysis_label,
        inputs=inputs,
        destination=session,
        tags=['batch','analysis',analysis_tag] if analysis_tag else ['batch','analysis'],
        config=config
    )
    
    return job_id
 
def check_job_status(fw, job_ids):
    """
    Check the status of submitted jobs.
    """
    print(f"\n🔍 Checking status of {len(job_ids)} jobs:")
    
    status_counts = {}
    for job_id in job_ids:
        try:
            job = fw.get_job(job_id)
            state = job.state
            status_counts[state] = status_counts.get(state, 0) + 1
            print(f"   Job {job_id}: {state}")
        except Exception as e:
            print(f"   Job {job_id}: Error - {str(e)}")
            status_counts['error'] = status_counts.get('error', 0) + 1
    
    print(f"\n📊 Job Status Summary:")
    for state, count in status_counts.items():
        print(f"   {state}: {count}")
 

# if __name__ == "__main__":
#     # Run the main function
#     print("Running main function")
#     job_list = run_recon_all_jobs(projectfw)
    
#     # Optionally check job status after submission
#     if job_list:
#         print(f"\n⏳ Waiting a moment before checking job status...")
#         import time
#         time.sleep(5)
        
        
#         check_job_status(fw, job_list)


   
def run_circumference_gear(fw, project, session=None):

    job_list = []
    processed_sessions = 0
    skipped_sessions = 0
    failed_sessions = 0
    status = st.empty()
    gear =  fw.lookup('gears/circumference')
    
    # Initialize gear_job_list
    job_list = list()
    analysis_tag = 'circumference'
    #Print a note that this gear will use GAMBAS as input
    st.info("⚠️ Note: The Circumference gear requires GAMBAS outputs as input. Ensure that GAMBAS has been run on the sessions.")

    for session in project.sessions():
        session = session.reload()
        if session is not None:
            inputfile = None
            status.text(f"Parsing...  {session.label}")

            inputs = {}

            analyses = session.analyses

            # If there are no analyses containers, we know that this gear was not run
            if len(analyses) == 0:
                run = 'False'
                status = 'NA'
                print('No analysis containers')
            else:
                try:

                    gambas_results = fw.search(
                        {"structured_query": f"session._id = {session.id} AND analysis.label CONTAINS gambas",
                        "return_type":"analysis"}
                    )

                    gambas_matches = [r.analysis.reload() for r in gambas_results if is_complete(r.analysis,"gambas")]
                    # gambas_matches =  [asys for asys in analyses if "gambas" in asys.label]

                    for matches in [gambas_matches]:
                        print(f'Found {len(matches)}')
                        # If there are no matches, the gear didn't run
                        if len(matches) == 0:
                            run = 'False'
                            status = 'NA'
                        # If there is one match, that's our target
                        elif len(matches) == 1:
                            run = 'True'
                            #status = matches[0].job.get('state')
                            print(status)

                            for file in matches[0].files:  
                                if re.search('mrr.nii.gz', file.name):
                                    inputfile = file
                                    print(inputfile.name)
                                    analysis_tag = 'circumference-mrr-axireg'
                                elif re.search('ResCNN.nii.gz', file.name):
                                    inputfile = file
                                    print(inputfile.name)
                                    analysis_tag = 'circumference-gambas'
                                elif re.search('T2w_gambas.nii.gz', file.name):
                                    inputfile = file
                                    print(inputfile.name)
                                    analysis_tag = 'circumference-gambas'
                                    
                        else:
                                #consider gambas as well as recon-all
                                last_run_date = max([asys.created for asys in matches])
                                last_run_analysis = [asys for asys in matches if asys.created == last_run_date]

                                # There should only be one exact match
                                last_run_analysis = last_run_analysis[0]

                                run = 'True'
                                #status = last_run_analysis.job.get('state')
                                for file in matches[0].files:  
                                    if re.search('mrr.nii.gz', file.name):
                                        inputfile = file
                                        print(inputfile.name)
                                        analysis_tag = 'circumference-mrr-axireg'
                                    elif re.search('ResCNN.nii.gz', file.name):
                                        inputfile = file
                                        print(inputfile.name)
                                        analysis_tag = 'circumference-gambas'

                                    elif re.search('T2w_gambas.nii.gz', file.name):
                                        inputfile = file
                                        print(inputfile.name)
                                        analysis_tag = 'circumference-gambas'
                    if inputfile:
                        inputs["input"]= inputfile
                        print("Input file" , inputfile.name)

                        try:
                            # The destination for this analysis will be on the session
                            target_template = 'None'
                            dest = session
                            time_fmt = '%d-%m-%Y_%H-%M-%S'

                            analysis_label = f'{analysis_tag}_{datetime.now().strftime(time_fmt)}'
                            job_id = gear.run(
                                analysis_label=analysis_label,
                                inputs=inputs,
                                destination=dest,
                                tags=['batch','analysis','circumference'],
                                config={
                                    # "target_template": target_template,
                                    "prefix": analysis_tag
                                }
                            )
                            job_list.append(job_id)
                            print("Submitting Job: Check Jobs Log", dest.label)
                            processed_sessions += 1
                        except Exception as e:
                            print(f"WARNING: Job cannot be sent for {dest.label}. Error: {e}")

                except Exception as e:
                        print(f"Exception caught for {session.label}: ", e)
     # Summary
    st.info(f"\n📊 Summary:  \n   ✅ Jobs submitted: {processed_sessions}\n   ⏭️ Sessions skipped: {skipped_sessions}\n   ❌ Sessions failed: {failed_sessions}\n   📋 Total job IDs: {len(job_list)}")

    return job_list
    
# --- Session state initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None

API_KEY = os.getenv("FW_CLI_API_KEY")

if (API_KEY == None or API_KEY == "") and st.session_state.authenticated == False:
    
    #Display message to enter API KEY in Home page
    st.warning("Please enter your Flywheel API key in the Home page to continue.")
    st.stop()
fw = Client(st.session_state.api_key if st.session_state.authenticated else API_KEY)


#Have a drop down to select batch runs 
#Add a message to not refresh the page while the batch job is running
st.markdown("⚠️ **Please do not refresh the page while the batch job is running to avoid interruptions.**")

st.title("📦 Batch Runs")
st.write("Select a gear to batch run to view details:")
# gear_list = fw.gears()
# gear_names = [gear.gear.name for gear in gear_list]
#Order them alphabetically

#For now only keep circumference, freesurfer-recon-all-clinical, gambas, minimorph
gear_names = ["QA","MRIQC","Circumference", "Freesurfer-recon-all", "Infant-freesurfer", "BIBSNET (baby-and-infant-brain-segmentation)","Recon-all-clinical","Recon-any","GAMBAS", 'Minimorph',"SuperSynth"]
gear_names.sort()
selected_gear_name = st.selectbox("Select Gear", gear_names)
selected_gear = next((gear for gear in gear_names if gear == selected_gear_name), None)
#Radio button for gambas or MRR input if applicable

# Only show radio button if selected gear is not QA
if selected_gear_name not in ["QA","MRIQC","Freesurfer-recon-all"] :
    input_type = st.radio("Select input type for segmentation gears:", ("MRR", "GAMBAS"), index=0 if "mrr" in selected_gear.lower() else 1)
else:
    input_type = None  # or set a default value if your downstream code needs it


#Dropdown to select project
project_list = fw.projects()
project_names = [project.label for project in project_list]
selected_project_name = st.selectbox("Select Project", project_names)
selected_project = next((project for project in project_list if project.label == selected_project_name), None)
if selected_project is None:
    st.warning("Please select a valid project.")
    st.stop()
st.info(f"Project: {selected_project.label} Subjects n = {len(selected_project.subjects())}  \nSessions n = {len(selected_project.sessions())}")
fw_project = fw.projects.find_first(f'label={selected_project.label}')

if selected_gear == "Freesurfer-recon-all":
    #This only takes T1w images
    #Have user enter i a textbox the string to look for in the acquisition label
    t1w_label_string = st.text_input("Enter string to identify T1w acquisition labels in your project (RMS, MPR, T1w):", value="MPRAGE")

#Add checkbox "debug" to only run on first 2 sessions
st.session_state.debug_mode =  False
n_sessions_debug = 4
debug_mode = st.checkbox(f"Debug Mode (Run on first {n_sessions_debug} sessions only)", value=False)
#Add checkbox to ensure latest version is ran
latest_version = st.checkbox("Use latest version of gear", value=False)
st.session_state.latest_version = latest_version

if debug_mode:
    st.warning(f"⚠️ Debug Mode is ON: The batch job will only run on the first {n_sessions_debug} sessions of the selected project.")
    st.session_state.debug_mode = True
#If you select the gear and project, and click a button, run the batch job
if st.button("Run Batch Job"):
    st.success(f"Running batch job for gear: {selected_gear} on project: {selected_project.label}, on {input_type} input")
    #Prepare dataframe to log job submissions (session variable)
    st.session_state.job_log = []
    
    if input_type == "GAMBAS":
        input_type = 1
    else:
        input_type = 0

    if selected_gear == "Circumference":
        run_circumference_gear(fw, fw_project)

    elif selected_gear == "Recon-all-clinical":
        
        job_list = run_jobs( fw, fw_project,'recon-all-clinical', gambas=input_type)
        if job_list:
            st.success(f"Submitted {len(job_list)} recon-all-clinical jobs.")
            check_job_status(fw, job_list)
        else:
            st.info("No recon-all-clinical jobs were submitted.")

    elif selected_gear == "Recon-any":
        job_list = run_jobs( fw, fw_project,'recon-any', gambas=input_type)
        if job_list:
            st.success(f"Submitted {len(job_list)} recon-any jobs.")
            check_job_status(fw, job_list)
        else:
            st.info("No recon-any jobs were submitted.")

    elif selected_gear == "Infant-freesurfer":
        #Add a true / false checkbox to 
        job_list = run_jobs(fw, fw_project, 'infant-freesurfer', gambas=input_type)
        if job_list:
            st.success(f"Submitted {len(job_list)} infant-freesurfer jobs.")
            check_job_status(fw, job_list)
        else:
            st.info("No infant-freesurfer jobs were submitted.")

    elif selected_gear == "BIBSNET (baby-and-infant-brain-segmentation)":
        job_list = run_jobs(fw, fw_project, 'baby-and-infant-brain-segmentation', gambas=input_type)
        if job_list:
            st.success(f"Submitted {len(job_list)} BIBSNET jobs.")
            check_job_status(fw, job_list)
        else:
            st.info("No BIBSNET jobs were submitted.")

    elif selected_gear == "Freesurfer-recon-all":
        #This only takes T1w images
        #Have user enter i a textbox the string to look for in the acquisition label
        #t1w_label_string = st.text_input("Enter string to identify T1w acquisition labels in your project (RMS, MPR, T1w):", value="MPRAGE")
        if t1w_label_string.strip() is not "":
            job_list = run_jobs(fw, fw_project, 'freesurfer-recon-all', gambas=False)

    elif selected_gear == "GAMBAS":
        job_list = run_gambas_jobs(fw, fw_project)

    elif selected_gear=="SuperSynth":
        job_list = run_jobs(fw, fw_project, 'supersynth', gambas=input_type, analysis_tag='gpuplus')

    elif selected_gear in ["QA","MRIQC"]:
        #Hide input type buttons
        job_list = run_jobs(fw, fw_project, selected_gear.lower(), gambas=input_type, analysis_tag=selected_gear.lower())
    
    else:
        job_list = run_jobs(fw, fw_project, selected_gear.lower(), gambas=input_type)
