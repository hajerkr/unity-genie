import streamlit as st
import flywheel
import os
import re
from datetime import datetime

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

        job_id = submit_gambas(fw, session)
        if job_id:
            job_list.append(job_id)
            processed_sessions += 1
            status.text(f"🚀 Submitted GAMBAS job (ID: {job_id}) for session {session.label}")
        else:
            failed_sessions += 1
            failed_sessions_list.append(session.label)
            status.text(f"❌ Failed to submit GAMBAS job for session {session.label}")
    


    # Summary
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


def submit_gambas(fw, session):

    inputs = {}
    EXCLUDE_PATTERNS = ['Segmentation', 'Align', 'Mapping']
    INCLUDE_PATTERN = 'T2'
    PLANE_TYPES = ['AXI']
    status = st.empty()
    # Look at every acquisition in the session
    for acquisition in session.acquisitions.iter():
        acquisition = acquisition.reload()
        for file in acquisition.files:
        # We only want anatomical Nifti's
            if file.type == 'nifti' and INCLUDE_PATTERN in file.name:
                if all(pattern not in file.name for pattern in EXCLUDE_PATTERNS):
                    for plane in PLANE_TYPES:
                        if plane in file.name:
                            inputs["input"] = file
                            break
    if inputs:                           
        try:
        # The destination for this analysis will be on the session
            dest = session
            gear_gambas = fw.lookup('gears/gambas')
            time_fmt = '%d-%m-%Y_%H-%M-%S'
            analysis_tag = 'gambas'
            analysis_label = f'{analysis_tag}_{datetime.now().strftime(time_fmt)}'
            job_id = gear_gambas.run(
                analysis_label=analysis_label,
                inputs=inputs,
                destination=dest,
                tags=['batch'],
                config={
                
                    # "prefix": analysis_tag,
                }
            )

            return job_id
            
        except Exception as e:
            status.text(f"WARNING: Job cannot be sent for {dest.label}. Error: {e}")


# def run_segmentation(project, gearname, gambas=False):
#     gear =  fw.lookup(f'gears/{gearname}')
    
#     # Initialize gear_job_list
#     job_list = list()

#     for subject in project.subjects():
#         subject = subject.reload()
#         for session in subject.sessions():
#             session = session.reload()
#             segmentation = [asys for asys in session.analyses if asys.gear_info is not None and asys.gear_info.get('name') == gearname and asys.job.get('state') == 'complete']

#             if segmentation:
#                 status.text(f"Skipping {session.label} as {gearname} has already been run.")
                
#             analysis_tag = gearname
#             inputfile = None
#             status.text("Parsing... ", subject.label, session.label)

#             inputs = {}
#             analyses = session.analyses

#             try:
#                 mrr_matches = [asys for asys in analyses if asys.gear_info is not None and asys.gear_info.get('name') == "mrr" and asys.job.get('state') == 'complete' and "mrr_T2w_MNI1.5mm" in asys.label]
#                 #gambas_matches =  [asys for asys in analyses if "gambas" in asys.label]
#                 if gearname == "freesurfer-recon-all":
#                     #Get nifti file in acquisition ending in _ses-GE_T2w
#                     print("Finding the MPRAGE acq")
#                     #Get aqcuiqisitons that have T1w and end with RMS
#                     for acquisition in session.acquisitions():
#                         acquisition = acquisition.reload()
#                         for file in acquisition.files:
#                             if acquisition.label.endswith('RMS') and file.type == 'nifti':
#                                 inputfile = file
#                                 print(inputfile.name)
#                                 analysis_tag = f'{gearname}-T1w-RMS'
                    
#                     # acquisition = session.acquisitions.find_first(f"label={subject.label}_{session.label}_acq-iso_T1w")
#                     # if acquisition is None:
#                     #     print(f"No acquisition found for {subject.label}_{session.label}_acq-iso_T1w")
#                     #     continue
#                     # acquisition = acquisition.reload()
#                     # for file in acquisition.files:
#                     #     if file.type == "nifti":
#                     #         inputfile = file
#                     #         print(inputfile.name)
#                     #         analysis_tag = f'{gearname}-iso_T1w'

#                 else:
                        
#                     for matches in [mrr_matches]:
#                         # If there are no matches, the gear didn't run
#                         if len(matches) == 0:
#                             run = 'False'
                        
#                         # If there is one match, that's our target
#                         elif len(matches) == 1:
#                             run = 'True'
#                             #status = matches[0].job.get('state')
                            

#                             for file in matches[0].files:  
#                                 if re.search(r'mrr.*\.nii.gz', file.name):
#                                     inputfile = file
#                                     print(inputfile.name)
#                                     analysis_tag = f'{gearname}-mrr-axireg'
#                                 elif re.search(r'ResCNN.*\.nii.gz', file.name):
#                                     inputfile = file
#                                     print(inputfile.name)
#                                     analysis_tag = f'{gearname}-gambas-1.5mm'
                                    
#                         else:
#                                 #consider gambas as well as recon-all
#                                 last_run_date = max([asys.created for asys in matches])
#                                 last_run_analysis = [asys for asys in matches if asys.created == last_run_date]

#                                 # There should only be one exact match
#                                 last_run_analysis = last_run_analysis[0]

#                                 run = 'True'
#                                 #status = last_run_analysis.job.get('state')
                                
#                                 for file in matches[0].files:  
#                                     if re.search(r'mrr.*\.nii.gz', file.name):
#                                         inputfile = file
#                                         print(inputfile.name)
#                                         analysis_tag = f'{gearname}-mrr-axireg'
#                                     elif re.search(r'ResCNN.*\.nii.gz', file.name):
#                                         inputfile = file
#                                         print(inputfile.name)
#                                         analysis_tag = f'{gearname}-gambas-1.5mm'
                                
#                     if inputfile:
#                         inputs["anatomical"]= inputfile
#                         print("Input file" , inputfile.name)

#                         try:
#                             # The destination for this analysis will be on the session
#                             target_template = 'None'
#                             dest = session
#                             time_fmt = '%d-%m-%Y_%H-%M-%S'

#                             analysis_label = f'{analysis_tag}_{datetime.now().strftime(time_fmt)}'
#                             if not recon_all:
#                                 job_id = gear.run(
#                                     analysis_label=analysis_label,
#                                     inputs=inputs,
#                                     destination=dest,
#                                     tags=['priority', gearname],
#                                     config={
#                                         #"target_template": "3M",
#                                         # "age": "3M"
#                                     }
#                                 )
#                                 job_list.append(job_id)

#                             print("Submitting Job: Check Jobs Log", dest.label)
#                         except Exception as e:
#                             print(f"WARNING: Job cannot be sent for {dest.label}. Error: {e}")

#             except Exception as e:
#                     print(f"Exception caught for {subject.label}: ", e)


def run_seg_jobs(fw, project, gearname, gambas=False, include_pattern=None,analysis_tag=None):
    """
    Run recon-all jobs on the most recent gambas analysis for each session
    if recon-all hasn't already been completed.
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
    # Loop through subjects and sessions
    if st.session_state.debug_mode:
        subjects = project.subjects()[:4]
        st.info("⚠️ Debug Mode: Processing only first 4 subjects.")
    else:
        subjects = project.subjects()
    for subject in subjects:
        if not (subject.label.startswith("137-")): #Ensure this does not run on the phantom - waste of resource and nonsense results
            for session in subject.sessions():
                session = session.reload()
                session_id = f"{project.label}/{subject.label}/{session.label}"
                status.text(f"\n🔍 Checking session: {session_id} for subject {subject.label}")
                try:
                    # Check if gear already completed specifically for gambas input
                    ### GAMBAS CHECKS ####
                    if gambas:
                        if has_completed_seg(session, gear, gambas=True):
                            status.text(f"✅ {gearname} with gambas input already complete, skipping")
                            skipped_sessions += 1
                            
                            continue
                    
                        # Find the most recent gambas analysis
                        gambas_file = find_latest_gambas_file(session)
                        if not gambas_file:
                            status.text(f"⚠️ No suitable gambas file found. Submitting a gambas job...")
                            #Add a function to run gambas if nothing has been found
                            job_id = submit_gambas(fw, session)
                            try:
                                job_list.append(job_id)
                                status.text(f"🚀 Submitting GAMBAS Job : Check Jobs Log")
                            except Exception as e:
                                    status.text(f"WARNING: Job cannot be sent. Error: {e}")

                            skipped_sessions += 1
                            
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
                        mrr_matches = [asys for asys in session.analyses if asys.gear_info is not None and asys.gear_info.get('name') == "mrr" and asys.job.get('state') == 'complete']
                        
                        if not mrr_matches:
                            status.text(f"⚠️ No suitable MRR analysis found. Skipping session.")
                            skipped_sessions += 1
                            continue

                        last_run_date = max([asys.created for asys in mrr_matches])
                        last_run_analysis = [asys for asys in mrr_matches if asys.created == last_run_date]
                        last_run_analysis = last_run_analysis[0]
                        
                        for file in last_run_analysis[0].files:  
                            if re.search(r'mrr.*\.nii.gz', file.name):
                                inputfile = file

                        job_id = submit_seg_job(gear, session, gambas=False, input_file=inputfile)
                        job_list.append(job_id)
                        processed_sessions += 1
                        print(f"🚀 Submitted {gearname} job (ID: {job_id})")

                    elif gearname == 'freesurfer-recon-all':
                        if has_completed_seg(session, gear, gambas=False):
                            status.text(f"✅ {gearname} already complete, skipping")
                            skipped_sessions += 1
                            continue
                        
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
                        job_id = submit_seg_job(gear, session, gambas=False, input_file=inputfile)

                   

                        
                except Exception as e:
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

 
def has_completed_seg(session, gear,gambas=False):
    """
    Check if session already has a completed recon-all analysis of the target version.
    """
    for analysis in session.analyses:
        if not analysis.gear_info:
            continue
            
        gear_name = analysis.gear_info.get('name', '').lower()
        #gear_version = analysis.gear_info.get('version', '')
        
        if gear.gear.name in gear_name : #and gear_version == target_version:
            job_state = analysis.job.get('state') if analysis.job else None
            if gambas and 'gambas' in analysis.label.lower() and job_state == 'complete':
                return True
            elif not gambas and 'gambas' not in analysis.label.lower() and job_state == 'complete':
                return True
    return False
 
def find_latest_gambas_file(session):
    """
    Find the most recent gambas analysis file in the session or its acquisitions.
    Returns the first suitable gambas file ending with 'rec-axi_T2w_gambas.nii.gz' from the latest gambas analysis, or None if not found.
    """
    print(f"   Checking {len(session.analyses)} analyses in session")
    
    # Debug: Print all session analyses
    for i, analysis in enumerate(session.analyses):
        gear_name = analysis.gear_info.get('name', 'No gear name') if analysis.gear_info else 'No gear_info'
        print(f"   Session Analysis {i}: label='{analysis.label}', gear='{gear_name}'")
    
    # Find all gambas analyses in this session
    gambas_analyses = []
    for analysis in session.analyses:
        if is_gambas_analysis(analysis):
            gambas_analyses.append(analysis)
            print(f"   Found gambas in session: {analysis.label}")
    
    # Also check acquisitions for gambas analyses
    print(f"   Checking acquisitions for gambas analyses...")
    for acquisition in session.acquisitions():
        acquisition = acquisition.reload()
        print(f"   Acquisition: {acquisition.label} has {len(acquisition.analyses)} analyses")
        
        for i, analysis in enumerate(acquisition.analyses):
            gear_name = analysis.gear_info.get('name', 'No gear name') if analysis.gear_info else 'No gear_info'
            print(f"     Acquisition Analysis {i}: label='{analysis.label}', gear='{gear_name}'")
            
            if is_gambas_analysis(analysis):
                gambas_analyses.append(analysis)
                print(f"   Found gambas in acquisition: {analysis.label}")
    
    if not gambas_analyses:
        print("   No gambas analyses found in session or acquisitions")
        return None
    
    print(f"   Found {len(gambas_analyses)} gambas analysis(es) total")
    
    # Use the latest gambas analysis (assuming they're ordered chronologically)
    latest_gambas = gambas_analyses[-1]
    print(f"   Using latest gambas analysis: {latest_gambas.label}")
    
    # Debug: Print all files in the analysis
    print(f"   Files in analysis: {[f.name for f in latest_gambas.files]}")
    
    # Find gambas output files - specifically look for files ending with "rec-axi_T2w_gambas.nii.gz"
    pattern = re.compile(r"rec-axi.*_T2w_(gambas|ResCNN)\.nii\.gz$")

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
        config= {"age": None}
    elif gear_name == "infant-freesurfer":
        config = {
        "newborn": False,
        "age": None
        }
   
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
                    mrr_matches = [asys for asys in analyses if asys.gear_info is not None and asys.gear_info.get('name') == "mrr" and asys.job.get('state') == 'complete']
                    gambas_matches =  [asys for asys in analyses if "gambas" in asys.label]

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
fw = flywheel.Client(st.session_state.api_key if st.session_state.authenticated else API_KEY)


#Have a drop down to select batch runs 
#Add a message to not refresh the page while the batch job is running
st.markdown("⚠️ **Please do not refresh the page while the batch job is running to avoid interruptions.**")

st.title("📦 Batch Runs")
st.write("Select a gear to batch run to view details:")
# gear_list = fw.gears()
# gear_names = [gear.gear.name for gear in gear_list]
#Order them alphabetically

#For now only keep circumference, freesurfer-recon-all-clinical, gambas, minimorph
gear_names = ["Circumference", "Freesurfer-recon-all", "Infant-freesurfer", "BIBSNET (baby-and-infant-brain-segmentation)","Recon-all-clinical","GAMBAS", 'Minimorph',"SuperSynth"]
gear_names.sort()
selected_gear_name = st.selectbox("Select Gear", gear_names)
selected_gear = next((gear for gear in gear_names if gear == selected_gear_name), None)
#Radio button for gambas or MRR input if applicable
input_type = st.radio("Select input type for segmentation gears:", ("MRR", "GAMBAS"), index=0 if "mrr" in selected_gear.lower() else 1)

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

#Add checkbox "debug" to only run on first 2 subjects
st.session_state.debug_mode =  False
debug_mode = st.checkbox("Debug Mode (Run on first 4 subjects only)", value=False)
if debug_mode:
    st.warning("⚠️ Debug Mode is ON: The batch job will only run on the first 2 subjects of the selected project.")
    st.session_state.debug_mode = True
#If you select the gear and project, and click a button, run the batch job
if st.button("Run Batch Job"):
    st.success(f"Running batch job for gear: {selected_gear} on project: {selected_project.label}")
    #Prepare dataframe to log job submissions (session variable)
    st.session_state.job_log = []
    

    if selected_gear == "Circumference":
        run_circumference_gear(fw, fw_project)

    elif selected_gear == "Recon-all-clinical":
        job_list = run_seg_jobs( fw, fw_project,'recon-all-clinical', gambas=input_type)
        if job_list:
            st.success(f"Submitted {len(job_list)} recon-all-clinical jobs.")
            check_job_status(fw, job_list)
        else:
            st.info("No recon-all-clinical jobs were submitted.")
    # elif selected_gear == "Recon-all-clinical (MRR input)":
    #     job_list = run_seg_jobs(fw, fw_project, 'recon-all-clinical', gambas=False)
    #     if job_list:
    #         st.success(f"Submitted {len(job_list)} recon-all-clinical jobs.")
    #         check_job_status(fw, job_list)
    #     else:
    #         st.info("No recon-all-clinical jobs were submitted.")
    elif selected_gear == "infant-freesurfer":
        #Add a true / false checkbox to 
        job_list = run_seg_jobs(fw, fw_project, 'infant-freesurfer', gambas=input_type)
        if job_list:
            st.success(f"Submitted {len(job_list)} infant-freesurfer jobs.")
            check_job_status(fw, job_list)
        else:
            st.info("No infant-freesurfer jobs were submitted.")

    elif selected_gear == "BIBSNET (baby-and-infant-brain-segmentation)":
        job_list = run_seg_jobs(fw, fw_project, 'baby-and-infant-brain-segmentation', gambas=input_type)
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
            job_list = run_seg_jobs(fw, fw_project, 'freesurfer-recon-all', gambas=False)

    elif selected_gear == "GAMBAS":
        job_list = run_gambas_jobs(fw, fw_project)

    elif selected_gear=="Supersynth":
        job_list = run_seg_jobs(fw, fw_project, 'supersynth', gambas=input_type, analysis_tag='gpuplus')


