import streamlit as st
import flywheel
import os
import re
from datetime import datetime
import traceback
import logging

logging.basicConfig(level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)

def find_input(session,input_gear):

    gear_results = fw.analyses.find(f"parents.session={session.id},gear_info.name={input_gear}")
    
    if not gear_results or input_gear == "gambas":
        analyses = fw.analyses.find(f"parents.session={session.id},label=~^{input_gear}")
        gear_results.extend(analyses)
        
    print(f"Found {len(gear_results)}")
    gear_matches = [r for r in gear_results if is_complete(r,input_gear)]
    
    latest_match = get_latest_match(gear_matches)
    
    return latest_match

def safe_is_status(asys, status_list):
    try:
        return is_status(asys, asys.gear_info.name, status_list)
    except Exception as e:
        logger.exception(f"Exception checking status for {asys}: {e}")
        return False

def is_complete(asys,gearname,latest_version=False):
    try:
       
        asys=asys.reload()
        if getattr(asys, 'gear_info', None) is not None:
            #If asys has attribute gear_info
            gear = fw.gears.find_first(f"gear.name={gearname}")
            #Get gear version
            gear_version = gear.gear.version if gear else "Unknown"
            #print(asys.gear_info, asys.gear_info.get('name'), asys.gear_info.get('name') == gearname, asys.job.get('state') )
            return (
                asys.gear_info is not None
                and asys.gear_info.get('name') == gearname
                and asys.job is not None
                and asys.job.get('state') == 'complete'
                #ensure last gear version is ran
                and (not latest_version or asys.gear_info.get('version') == gear_version)
            )
        
        elif gearname =="gambas":
    
                logger.info(f"Analysis {asys.id} has no gear_info, checking label for gambas-batch...")
                #Look at analysis container containing "gambas-batch" in the label
                logger.info(asys.label)
                return (
                    "gambas" in asys.label and ("0.4.17" in asys.label or "0.4.14" in asys.label)
                    and len(asys.files) > 0
                )
    except Exception as e:
        logger.info(f"Error reloading analysis {asys.id}: {e}")
        return False
    
    # asys=asys.reload()
    # return (
    #     asys.gear_info is not None
    #     and gearname in asys.gear_info.get('name') 
    #     and asys.job is not None
    #     and asys.job.get('state') == 'complete'
    # )

def is_status(asys,gearname, status, latest_version=False):
    # try:
    #     asys=asys.reload()
    #     gear = fw.gears.find_first(f"gear.name={gearname}")
    #     #Get gear version
    #     gear_version = gear.gear.version if gear else "Unknown"
    #     return (
    #         asys.gear_info is not None
    #         and gearname in asys.gear_info.get('name')
    #         and asys.job is not None
    #         and asys.job.get('state') in status
    #         and (not latest_version or asys.gear_info.get('version') == gear_version)
    #     )
    # except Exception as e:
    #     logger.exception(f"Error checking {status} status for analysis {asys.id}: {e}")
    #     return False
    try:
        
        if getattr(asys, 'gear_info', None) is not None:
            asys=asys.reload()
            
            #If asys has attribute gear_info
            gear = fw.gears.find_first(f"gear.name={gearname}")
            #Get gear version
            gear_version = gear.gear.version if gear else "Unknown"
            #print(asys.gear_info, asys.gear_info.get('name'), asys.gear_info.get('name') == gearname, asys.job.get('state') )
            return (
                asys.gear_info is not None
                and asys.gear_info.get('name') == gearname
                and asys.job is not None
                and asys.job.get('state') == 'complete'
                #ensure last gear version is ran
                and (not latest_version or asys.gear_info.get('version') == gear_version)
            )
        
        elif gearname =="gambas":
    
                logger.info(f"Analysis {asys.id} has no gear_info, checking label for gambas-batch...")
                #Look at analysis container containing "gambas-batch" in the label
                logger.info(asys.label)
                return (
                    "gambas" in asys.label and ("0.4.17" in asys.label or "0.4.14" in asys.label)
                    and len(asys.files) > 0
                )
    except Exception as e:
        logger.info(f"Error reloading analysis {asys.id}: {e}")
        return False
    
    
# def is_failed(asys,gearname, latest_version=False):

#     try:
#         asys=asys.reload()
#         gear = fw.gears.find_first(f"gear.name={gearname}")
#         #Get gear version
#         gear_version = gear.gear.version if gear else "Unknown"
#         return (
#             asys.gear_info is not None
#             and gearname in asys.gear_info.get('name')
#             and asys.job is not None
#             and asys.job.get('state') == 'failed'
#             and (not latest_version or asys.gear_info.get('version') == gear_version)
#         )
#     except Exception as e:
#         logger.exception(f"Error checking failed status for analysis {asys.id}: {e}")
#         return False

# def is_pending(asys,gearname, latest_version=False):
#     try:
#         asys=asys.reload()
#         return (
#             asys.gear_info is not None
#             and gearname in asys.gear_info.get('name')
#             and asys.job is not None
#             and asys.job.get('state') in ['pending', 'running']
#     )
#     except Exception as e:
#         logger.exception(f"Error checking pending status for analysis {asys.id}: {e}")
#         return False
def run_gambas_jobs(fw, project):
    job_list = []
    processed_sessions = 0
    skipped_sessions = 0
    failed_sessions = 0
    status = st.empty()
    failed_sessions_list = []
    for session in sorted(project.sessions(), key=lambda s: s.created, reverse=True):
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
            logger.info(f"🚀 Submitted GAMBAS job (ID: {job_id}) for session {session.label}")
            status.text(f"🚀 Submitted gambas job for session {session.label}")
        else:
            failed_sessions += 1
            failed_sessions_list.append((session.subject.label, session.label))
            status.text(f"❌ Failed to submit GAMBAS job for session {session.label}")
    


    # Summary
    # Flatten job_list if it contains nested lists
    if any(isinstance(i, list) for i in job_list):
        job_list = [job for sublist in job_list for job in sublist]

    st.info(f"\n📊 Summary:  \n   ✅ Jobs submitted: {processed_sessions}\n   ⏭️ Sessions skipped: {skipped_sessions}\n   ❌ Sessions failed: {failed_sessions}\n   📋 Total job IDs: {len(job_list)}")
    #Return CSV with skipped sessions
    if failed_sessions_list:
        skipped_sessions_str = "\n".join([", ".join(item) for item in failed_sessions_list])
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
    GEAR_PLANE_TYPES = {"gambas": ['AXI'], "qa": ['AXI','SAG','COR'],"mriqc": ['AXI','SAG','COR'],"mrr": ["AXI"]}
    INPUT_DICT = {"gambas": "input", "qa": "input", "mriqc": "nifti", "mrr": "axi","freesurfer-recon-all":"anatomical"}
    status = st.empty()
    job_ids = []
    tags = {"mriqc":"qc","gambas":"gpu"}
    # Look at every acquisition in the session
    for acquisition in session.acquisitions.find(f'label=~{INCLUDE_PATTERN}'):
        inputs = {}
        logger.info(f"Checking acquisition: {acquisition.label}...")

        match = next(
        (f for f in acquisition.files
         if f.type == 'nifti'
         and INCLUDE_PATTERN.lower() in f.name.lower()
         and not any(p.lower() in f.name.lower() for p in EXCLUDE_PATTERNS)
         and any(plane.lower() in f.name.lower() for plane in GEAR_PLANE_TYPES.get(gearname, []))),
        None
        )

        if match:
            inputs = {INPUT_DICT[gearname] : match}
        if inputs:                           
            try:
            # The destination for this analysis will be on the session
                dest = acquisition
                gear = fw.lookup(f'gears/{gearname}')
                time_fmt = '%d-%m-%Y_%H-%M-%S'
                analysis_tag = gearname
                analysis_label = f'{analysis_tag}_{datetime.now().strftime(time_fmt)}'
                if gearname == "mriqc": #Not analysis gear, so do not include analysis label
                    job_id = gear.run(
    
                        inputs=inputs,
                        destination=dest,
                        tags=['batch',tags[gearname]],
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
                        tags=['batch',tags.get(gearname, 'analysis')],
                        config={
                        
                            # "prefix": analysis_tag,
                        }
                    )
                
                
                job_ids.append(job_id)
                # return job_id
                
            except Exception as e:
                status.text(f"WARNING: Job cannot be sent for {dest.label}. Error: {e}")
    return job_ids

def get_analyses(session,gearname,input_type=None,inputfile=None):
    """Get analyses for a session based on gear name.

    Args:
        session (FW session object): The Flywheel session object to query.
        gearname (str): The name of the gear to filter analyses by.

    Returns:
        list? : A list of analyses that match the specified gear name for the given session.
    """
    analyses = fw.analyses.find(f"parents.session={session.id},gear_info.name={gearname.lower()}")
    
    analyses = sorted(analyses, key=lambda x: x.created, reverse=True)
    matching_analyses = []
    logger.info(f"Found {len(analyses)} {[asys.label for asys in analyses]}")
    for analysis in analyses:
        if input_type is not None:
            #Get the latest analysis created
            
            if analysis and analysis.inputs:
                input_analysis_id = analysis.inputs[0].parents.analysis
                #Find the gear of this analysis
                input_gear_asys = fw.analyses.find_first(f"_id={input_analysis_id}")
                if input_gear_asys and input_type.lower() in input_gear_asys.label: #Previously was if input_gear_asys and input_gear_asys.gear_info.name == input_type.lower() . Changed this because doesn't work for some gambas analysis containers that are not tied to a gear (gambas batch)
                    logger.info(f"Match found with input.")
                    matching_analyses.append(analysis)
                else:
                    logger.info(f"Gear name mismatch.")
        
        elif inputfile is not None:
            if analysis and analysis.inputs:
                #If it is the same if
                if analysis.inputs[0].id == inputfile.id:
                    logger.info(f"Found matching analysis of this gear with this input")
                    matching_analyses.append(analysis)
        else:
            logger.info(f"Analysis {analysis.label} has no inputs or inputs are not linked to an analysis.")
            matching_analyses.append(analysis)
    
    
    return matching_analyses


def has_failed_asys(session, gearname, input_type=None,inputfile=None):
    """
    Check if session already has a failed segmentation analysis of the target version.
    """

    gear_results = get_analyses(session, gearname,input_type)
    gear_matches = [r.reload() for r in gear_results if is_status(r,gearname, ["failed"], st.session_state.latest_version)]

    if len(gear_matches) > 2: #If there are more than 2 failed analyses, likely something is wrong and no need to keep trying and failing, so we will skip to save resources
        return True

    return False

def get_latest_match(gear_matches):
    
    gear_matches = [asys for asys in gear_matches if safe_is_status(asys, ["complete"])]
    if not gear_matches:
        return None
    gear_matches = sorted(gear_matches, key=lambda x: x.created, reverse=True)
    return gear_matches[0]
                        
   

def find_latest_gambas_file(session):
    """
    Find the most recent gambas analysis file in the session or its acquisitions.
    Returns the first suitable gambas file ending with 'rec-axi_T2w_gambas.nii.gz' from the latest gambas analysis, or None if not found.
    """
    logger.info(f"   Checking analyses in session")
    gambas_results = get_analyses(session, "gambas")
    #Get the acquisition analyses in this session
    
    gambas_analyses = [r for r in gambas_results if is_status(r,"gambas",["complete"])]
    latest_gambas = get_latest_match(gambas_analyses)
    
    if not gambas_analyses:
        logger.info("   No gambas analyses found in session or acquisitions")
        return None
    
    logger.info(f"   Found {len(gambas_analyses)} gambas analysis(es) total")
    logger.info(f"   Using latest gambas analysis: {latest_gambas.label}")
        
    #Try to reload the analysis
    try:
        latest_gambas = latest_gambas.reload()
        logger.info(f"   Reloaded analysis successfully.")
    except Exception as e:
        logger.exception(f"   Warning: Could not reload analysis. Potentially manually created analysis container. Error: {e}")

    logger.info(f"   Files in analysis: {[f.name for f in latest_gambas.files]}")
    # Find gambas output files - specifically look for files ending with "rec-axi_T2w_gambas.nii.gz"
    pattern_matching = re.compile(r"(gambas|ResCNN)\.nii\.gz$", re.IGNORECASE)

    gambas_files = [
        f for f in latest_gambas.files
        if pattern_matching.search(f.name)
    ]

    if not gambas_files: 
        logger.info(f"   No files ending with 'rec-axi(_run-XX)_T2w_gambas.nii.gz' found in analysis {latest_gambas.label}")
        return None
    
    logger.info(f"   Found {len(gambas_files)} gambas file(s): {[f.name for f in gambas_files]}")
    
    # Return the first matching file
    return gambas_files[0]
 
# def is_gambas_analysis(analysis):
#     """
#     Check if an analysis is a gambas analysis by checking gear name or analysis label.
#     """
#     # Check gear name - must be exactly 'gambas' gear
#     if analysis.gear_info and analysis.gear_info.get('name'):
#         gear_name = analysis.gear_info.get('name').lower()
#         if gear_name == 'gambas':
#             return True
    
#     # If no gear_info, check the analysis label for gambas version pattern
#     elif analysis.label:
#         label = analysis.label.lower()
#         # Look for patterns like 'gambas/0.4.14' or 'gambas/0.4.17'
#         if "gambas" in label and ("0.4.17" in label or "0.4.14" in label):
#         # if pattern.search(label):
#             return True
    
#     return False
 
def submit_job_input(gear, session, input_type, input_file, analysis_tag=None):
    """
    Submit a segmentation analysis job for the given session and gambas file.
    """
    
    
    analysis_label = ''
    gear_name = gear.gear.name
    manifest_input = "anatomical" if gear_name == "freesurfer-recon-all" else 'input'    
    inputs = {manifest_input: input_file}
    
    # Create a unique analysis label with timestamp and gambas identifier
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")    
    analysis_label =f'{gear_name}_{input_file.name.replace(".nii.gz","")}_{timestamp}' if st.session_state.acq_label else f'{gear_name}_{input_type.lower()}_{timestamp}'

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
    logger.info(f"\n🔍 Checking status of {len(job_ids)} jobs:")
    
    status_counts = {}
    for job_id in job_ids:
        try:
            job = fw.get_job(job_id)
            state = job.state
            status_counts[state] = status_counts.get(state, 0) + 1
            logger.info(f"   Job {job_id}: {state}")
        except Exception as e:
            logger.exception(f"   Job {job_id}: Error - {str(e)}")
            status_counts['error'] = status_counts.get('error', 0) + 1
    
    logger.info(f"\n📊 Job Status Summary:")
    for state, count in status_counts.items():
        logger.info(f"   {state}: {count}")


def run_circumference_gear(fw, project, input_type):

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
    #st.info("⚠️ Note: The Circumference gear requires GAMBAS outputs as input. Ensure that GAMBAS has been run on the sessions.")

    for session in project.sessions():
        session = session.reload()
        if session is not None and not session.info.get('childTimepointHC_MRI_cm', False):
            inputfile = None
            status.text(f"Parsing...  {session.label} for input {input_type}")
            inputs = {}
            logger.info(input_type == "MRR")
            pattern_mapping = {
                    "MRR": re.compile(r"mrr.*\.nii\.gz", re.IGNORECASE),
                    "GAMBAS": re.compile(r"(gambas|ResCNN)\.nii\.gz$",re.IGNORECASE),
                }
            pattern_matching = pattern_mapping.get(input_type, None)
            
            analysis_match = find_input(session, input_type.lower())
            if analysis_match:
                for file in analysis_match.files:
                    if pattern_matching.search(file.name):
                        inputfile = file
                        analysis_tag = f'{analysis_tag}_{input_type.lower()}'
                
            # if input_type == "MRR":
            #     logger.info("Looking for MRR input")
            #     mrr_match = find_input(session,"mrr")
                
            #     print(f"Found {len(mrr_matches)}")
            #     # if mrr_matches:
            #     #     latest_match = get_latest_match(mrr_matches)
            #     #     latest_match = latest_match.reload()
            #     #     print(latest_match.label)
            #     #     #Find the file
            #     #     print(len(latest_match.files))
            #     if mrr_match:
            #         for file in mrr_match.files: 
            #             print 
            #             if re.search(r"mrr.*\.nii\.gz", file.name):
            #                 inputfile = file
                            
            #                 logger.info(inputfile.name)
            #                 analysis_tag = f'{analysis_tag}-mrr-axireg'

            # elif input_type == "GAMBAS":
            #     #inputfile = find_latest_gambas_file(session)
            #     gambas_match = find_input(session,"gambas")
            #     analysis_tag = f'{analysis_tag}-gambas'

            if inputfile:
                inputs["input"]= inputfile
                logger.info(f"Input file {inputfile.name}")

                try:
                    # The destination for this analysis will be on the session
                    dest = session
                    time_fmt = '%d-%m-%Y_%H-%M-%S'

                    analysis_label = f'{analysis_tag}_{datetime.now().strftime(time_fmt)}'
                    job_id = gear.run(
                        analysis_label=analysis_label,
                        inputs=inputs,
                        destination=dest,
                        tags=['batch','analysis','circumference'],
                        config={
                            "prefix": analysis_tag
                        }
                    )
                    job_list.append(job_id)
                    logger.info(f"Submitting Job: Check Jobs Log {dest.label} ")
                    processed_sessions += 1
                except Exception as e:
                    logger.warning(f"WARNING: Job cannot be sent for {dest.label}. Error: {e}")
        else:
            status.text(f"⚠️ Session {session.label} has childTimepointHC_MRI_cm, skipping.")
            skipped_sessions += 1
            continue
       
     # Summary
    st.info(f"\n📊 Summary:  \n   ✅ Jobs submitted: {processed_sessions}\n   ⏭️ Sessions skipped: {skipped_sessions}\n   ❌ Sessions failed: {failed_sessions}\n   📋 Total job IDs: {len(job_list)}")

    return job_list

def run_jobs(fw, project, gearname, input_type=False, acq_label_string=None,analysis_tag=None):
    """
    Run seg jobs on the most recent 'gambas' (or MRR) analysis for each session
    if segmentation hasn't already been completed.
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
        sessions =  sorted(project.sessions(), key=lambda s: s.created, reverse=True)
        sessions = sessions[:4]
        #st.info("⚠️ Debug Mode: Processing only first 4 sessions.")
    else:
        sessions = project.sessions()

    for session in sessions:
        
        skip_session = False
        if not (session.subject.label.startswith("137-")): #Ensure this does not run on the phantom - waste of resource and nonsense results
            # for session in subject.sessions():
            try:
                session = session.reload()
                session_id = f"{project.label}/{session.subject.label}/{session.label}"
                status.text(f"\n🔍 Checking session: {session_id} for subject {session.subject.label}")
                logger.info(f"\n🔍 Checking session: {session_id} for subject {session.subject.label}")
                #First get analyses for this session and gearname, with this input type
                if acq_label_string is None:
                    past_failed_analyses_count = 0
                    ran_analyses = get_analyses(session, gearname, input_type=input_type)
                    #Check if any of the analyses are complete, pending, or failed
                    for analysis in ran_analyses:
                        #Check for complete
                        if is_status(analysis, gearname, ["complete"], latest_version=st.session_state.latest_version):
                            status.text(f"✅ {gearname} with this input already complete for session {session_id}, skipping.")
                            logger.info(f"✅ {gearname} with this input already complete for session {session_id}, skipping.")
                            skipped_sessions += 1
                            skip_session = True
                            break
                        #Check for pending/running
                        elif is_status(analysis, gearname, ["pending","running"], latest_version=st.session_state.latest_version):
                            status.text(f"⏳ {gearname} with this input already pending/running for session {session_id}, skipping.")
                            logger.info(f"⏳ {gearname} with this input already pending/running for session {session_id}, skipping.")
                            skipped_sessions += 1
                            skip_session = True
                            break
                        
                        #Check for failed
                        elif is_status(analysis, gearname, ["failed"], latest_version=st.session_state.latest_version):
                            past_failed_analyses_count += 1
                            if past_failed_analyses_count > 1:
                                status.text(f"❌ {gearname} with this input has previously failed twice for session {session_id}, skipping.")
                                logger.info(f"❌ {gearname} with this input has previously failed twice for session {session_id}, skipping.")
                                skipped_sessions += 1
                                skip_session = True
                                break
                #################
                if skip_session:
                    continue
                if gearname in ["mriqc","mrr"]:
                    job_id =  submit_job(fw, session, gearname)
                    job_list.extend(job_id)
                    processed_sessions += 1
                    if job_id:
                        status.text(f"🚀 Submitted {gearname} job for session {session_id}")
                        logger.info(f"🚀 Submitted {gearname} job (ID: {job_id}) for session {session_id}")
                    
                ### GAMBAS CHECKS ####
                elif input_type == "GAMBAS":
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
                        logger.info(f"✅ Found gambas file: {gambas_file.name}")
                        # Submit seg job
                        job_id = submit_job_input(gear, session, input_type, input_file=gambas_file, analysis_tag=analysis_tag)
                        job_list.append(job_id)
                        processed_sessions += 1

                        logger.info(f"🚀 Submitted {gearname} job (ID: {job_id})")
                        
                        
                ### MRR CHECKS ####
                elif input_type == "MRR" or input_type == "SuperField" or "SuperSynth(etic)" in input_type :
                    pattern_matching = re.compile(r"mrr.*\.nii\.gz",re.IGNORECASE) if input_type == "MRR" else re.compile(r".*T2.*\.nii\.gz",re.IGNORECASE) if input_type == "SuperField" else re.compile(r".*desc-synth_T1w.nii.gz",re.IGNORECASE) if input_type == "SuperSynth(etic) T1w" else re.compile(r".*desc-synth_T2w.nii.gz",re.IGNORECASE) if input_type == "SuperSynth(etic) T2w" else None
                    inputfile = None
                    #This finds the latest match already
                    gear_match = find_input(session, input_type.lower()) if input_type != "SuperSynth(etic) T1w" and input_type != "SuperSynth(etic) T2w" else find_input(session, "supersynth")
                    if not gear_match:
                        skip_session=True
                        skipped_sessions+= 1
                        status.text(f"⚠️ No suitable {input_type} file found.")
                        logger.info(f"SKIPPED SESSION {session_id}")
                        continue
                  
                    for file in gear_match.files:  
                        if pattern_matching.search(file.name):
                            inputfile = file

                    job_id = submit_job_input(gear, session, input_type, input_file=inputfile,analysis_tag=analysis_tag)
                    job_list.append(job_id)
                    processed_sessions += 1
                    status.text(f"🚀 Submitted {gearname} job")
                    logger.info(f"🚀 Submitted {gearname} job (ID: {job_id})")
                    
                
                ## Example: freesurfer-recon-all
                elif input_type == "Other (Acquisition)" and acq_label_string:
                    # Find corresponding acquisition based on provided label string
                    inputfile = None
                    logger.info("Other Acquisition")
                    for acquisition in session.acquisitions():
                        logger.info(acquisition.label)
                        acquisition = acquisition.reload()
                        if any(label.lower() in acquisition.label.lower() for label in acq_label_string) and not any(exclude_label.lower() in acquisition.label.lower() for exclude_label in exclude_acq_label_string):
                            for file in acquisition.files:
                                if file.type == 'nifti':
                                    inputfile = file
                                    status.text(f"✅ Found file: {inputfile.name} in acquisition {acquisition.label}")
                                    break
                    if inputfile:
                        #See if this analysis has already completed or is pending, or failed twice on this same input
                        ran_analyses =  get_analyses(session, gearname, input_type=None,inputfile=inputfile)
                        
                        for analysis in ran_analyses:
                            
                            if is_status(analysis, gearname, ["complete"], latest_version=st.session_state.latest_version):
                                status.text(f"✅ {gearname} with this input already complete for session {session_id}, skipping.")
                                logger.info(f"✅ {gearname} with this input already complete for session {session_id}, skipping.")
                                skipped_sessions += 1
                                skip_session = True
                                break
                            #Check for pending/running
                            elif is_status(analysis, gearname, ["pending","running"], latest_version=st.session_state.latest_version):
                                status.text(f"⏳ {gearname} with this input already pending/running for session {session_id}, skipping.")
                                logger.info(f"⏳ {gearname} with this input already pending/running for session {session_id}, skipping.")
                                skipped_sessions += 1
                                skip_session = True
                                break
                            
                            #Check for failed
                            elif is_status(analysis, gearname, ["failed"], latest_version=st.session_state.latest_version):
                                status.text(f"❌ {gearname} with this input has previously failed for session {session_id}, skipping.")
                                logger.info(f"❌ {gearname} with this input has previously failed for session {session_id}, skipping.")
                                skipped_sessions += 1
                                skip_session = True
                                break
                            
                    if not inputfile:
                        status.text(f"⚠️ No suitable acquisition found with label containing {' or '.join(acq_label_string)}. Skipping session.")
                        skipped_sessions += 1
                        #Need to log this
                        skipped_sessions_list.append(session.label)
                        continue
                    if skip_session:
                        logger.info("Skipping session")
                        continue
                    # Submit seg job with T1w input
                    job_id = submit_job_input(gear, session, input_type, input_file=inputfile,analysis_tag=analysis_tag)
                    job_list.append(job_id)
                    processed_sessions += 1
                    status.text(f"🚀 Submitted {gearname} job")
                    logger.info(f"🚀 Submitted {gearname} job (ID: {job_id})")
                        
            except Exception as e:
                logger.exception(f"Exception caught for session {session_id}:  {traceback.format_exc()}")
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
            file_name=f"skipped_sessions_{project.label}_{gearname}.txt",
            mime="text/plain"
        )
    
    return job_list

def submit_and_report(fw, job_list, label):
    if job_list:
        st.success(f"Submitted {len(job_list)} {label} jobs.")
        check_job_status(fw, job_list)
    else:
        st.info(f"No {label} jobs were submitted.")
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
st.write("Select a gear to batch run:")
# gear_list = fw.gears()
# gear_names = [gear.gear.name for gear in gear_list]
#Order them alphabetically

#For now only keep circumference, freesurfer-recon-all-clinical, gambas, minimorph
#gear_names = ["QA","MRIQC","Circumference", "MRR", "Freesurfer-recon-all", "Infant-freesurfer", "iBEAT", "BIBSNET (baby-and-infant-brain-segmentation)","Recon-all-clinical","Recon-any","GAMBAS", 'Minimorph',"SuperSynth"]
#gear_names.sort()
#selected_gear_name = st.selectbox("Select Gear", gear_names)
# Define categories for the dropdown options
gear_categories = {
    "Quality Assurance": ["QA", "MRIQC"],
    "Reconstruction": ["MRR"], #add CISO maybe
    "Segmentation": ["Freesurfer-recon-all", "Infant-freesurfer", "iBEAT", "BIBSNET (baby-and-infant-brain-segmentation)","Minimorph", "SuperSynth","Recon-all-clinical", "Recon-any"],
    "Advanced Processing": ["GAMBAS"], #Add superfield later
    "Other": ["Circumference"]
}

# Create a selectbox for categories
selected_category = st.selectbox("Select Gear Category", list(gear_categories.keys()))

# Populate the gear dropdown based on the selected category
selected_gear_name = st.selectbox("Select Gear", gear_categories[selected_category])
#selected_gear = next((gear for gear in gear_names if gear == selected_gear_name), None)
#Radio button for gambas or MRR input if applicable

# Only show radio button if selected gear is not QA
if selected_gear_name not in ["QA","MRIQC","MRR","GAMBAS"] :
    input_type = st.radio("Select input source to use:", ("MRR", "GAMBAS", "SuperField","SuperSynth(etic) T1w", "SuperSynth(etic) T2w", "Other (Acquisition)"), index=0 if "mrr" in selected_gear_name.lower() else 1 if "gambas" in selected_gear_name.lower() else 2 if "superfield" in selected_gear_name.lower() else 3 if "t1w" in selected_gear_name.lower() else 4 if "t2w" in selected_gear_name.lower() else 5)
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
st.info(f"Project: {selected_project.label}\nSubjects N = {len(selected_project.subjects())}\nSessions N = {len(selected_project.sessions())}")
fw_project = fw.projects.find_first(f'label={selected_project.label}')
acq_label_string, exclude_acq_label_string = None, None

if selected_gear_name == "Freesurfer-recon-all" or input_type == "Other (Acquisition)":
    #Note: user should be as specific as possible to avoid accidentally selecting the wrong acquisition. For example, if you have multiple T1w acquisitions, you might want to use "MPRAGE" or "T1w" instead of just "T1".
    st.info("⚠️ Note: Please ensure that the strings you enter are specific enough to uniquely identify the desired acquisitions. Check your project first.")

    acq_label_strings = st.text_input("Enter strings to identify the acquisition labels in your project (comma-separated, e.g., MPRAGE,T1w):", value="MPRAGE")
    acq_label_string = [label.strip() for label in acq_label_strings.split(",") if label.strip()]
    input_type = "Other (Acquisition)"
    
    #Add strings to exclude
    exclude_acq_label_strings = st.text_input("Enter strings to EXCLUDE acquisition labels (comma-separated, e.g., T1w,T2w):", value="")
    exclude_acq_label_string = [label.strip() for label in exclude_acq_label_strings.split(",") if label.strip()]
    
st.session_state.acq_label = acq_label_string
st.session_state.exclude_acq_label = exclude_acq_label_string

#Add checkbox "debug" to only run on first 2 sessions
st.session_state.debug_mode =  False
n_sessions_debug = 4
debug_mode = st.checkbox(f"Debug Mode (Run on first {n_sessions_debug} sessions only)", value=False)
#Add checkbox to ensure latest version is ran
latest_version = st.checkbox(
    "Enforce latest version of gear",
    value=False,
    help="When unchecked, jobs won't be resubmitted if any version of this gear was previously run. "
         "When checked, it ensures that the latest version of the gear has been run."
)
st.session_state.latest_version = latest_version

if debug_mode:
    st.warning(f"⚠️ Debug Mode is ON: The batch job will only run on the first {n_sessions_debug} sessions of the selected project.")
    st.session_state.debug_mode = True
#If you select the gear and project, and click a button, run the batch job
gear_name_mapping = {}
gear_name_mapping = {
    "QA": "qa",
    "MRIQC": "mriqc",
    "Circumference": "circumference",
    "MRR": "mrr",
    "Freesurfer-recon-all": "freesurfer-recon-all",
    "Infant-freesurfer": "infant-freesurfer",
    "iBEAT": "ibeat2",
    "BIBSNET (baby-and-infant-brain-segmentation)": "baby-and-infant-brain-segmentation",
    "Recon-all-clinical": "recon-all-clinical",
    "Recon-any": "recon-any",
    "GAMBAS": "gambas",
    "Minimorph": "minimorph",
    "SuperSynth": "supersynth",
    "SynthSR":"synthsr"
}
if st.button("Run Batch Job"):
    st.success(f"Running batch jobs for gear: {selected_gear_name} \nProject: {selected_project.label}, on {input_type if input_type is not None else acq_label_string} input")
    #Prepare dataframe to log job submissions (session variable)
    st.session_state.job_log = []

    

    # gear_name -> (gear_slug, extra kwargs for run_jobs)
    GEAR_CONFIG = {
        "Recon-all-clinical": ('recon-all-clinical', {'acq_label_string': acq_label_string}),
        "Recon-any": ('recon-any', {'acq_label_string': acq_label_string}),
        "Infant-freesurfer": ('infant-freesurfer', {}),
        "BIBSNET (baby-and-infant-brain-segmentation)": ('baby-and-infant-brain-segmentation', {}),
        "SuperSynth": ('supersynth', {'acq_label_string': acq_label_string, 'analysis_tag': 'gpuplus'}),
        "iBEAT": ('ibeat2', {'analysis_tag': 'gpu'}),
        "MRR": (None, {'input_type': None}),  # special-cased below for input_type=None
    }

    if selected_gear_name == "Circumference":
        run_circumference_gear(fw, fw_project, input_type=input_type)

    elif selected_gear_name == "GAMBAS":
        job_list = run_gambas_jobs(fw, fw_project)
        submit_and_report(fw, job_list, "GAMBAS")

    elif selected_gear_name in ("Freesurfer-recon-all", "SynthSR"):
        # These only take T1w images
        if acq_label_string:
            job_list = run_jobs(fw, fw_project, selected_gear_name.lower(),
                                input_type=input_type, acq_label_string=acq_label_string)
            submit_and_report(fw, job_list, selected_gear_name)

    elif selected_gear_name in ("QA", "MRIQC"):
        job_list = run_jobs(fw, fw_project, selected_gear_name.lower(),
                            input_type=input_type, analysis_tag=selected_gear_name.lower())
        submit_and_report(fw, job_list, selected_gear_name)

    elif selected_gear_name == "MRR":
        job_list = run_jobs(fw, fw_project, 'mrr', input_type=None, analysis_tag='mrr')
        submit_and_report(fw, job_list, "MRR")

    elif selected_gear_name in GEAR_CONFIG:
        gear_slug, extra_kwargs = GEAR_CONFIG[selected_gear_name]
        job_list = run_jobs(fw, fw_project, gear_slug, input_type=input_type, **extra_kwargs)
        submit_and_report(fw, job_list, selected_gear_name)

    else:
        job_list = run_jobs(fw, fw_project, selected_gear_name.lower(), input_type=input_type)
        submit_and_report(fw, job_list, selected_gear_name)