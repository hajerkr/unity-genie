import streamlit as st
import flywheel
import os


   
def run_circumference_gear(project, session=None):
    gear =  fw.lookup('gears/circumference')
    
    # Initialize gear_job_list
    job_list = list()
    analysis_tag = 'circumference'

    for session in project.sessions():
        session = session.reload()
        if session is not None:
            inputfile = None
            print("Parsing... ", session.label)

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
                        except Exception as e:
                            print(f"WARNING: Job cannot be sent for {dest.label}. Error: {e}")

                except Exception as e:
                        print(f"Exception caught for {session.label}: ", e)

# --- Session state initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None

API_KEY = os.getenv("FW_CLI_API_KEY")

if API_KEY == None or API_KEY == "" and st.session_state.authenticated == False:
    
    #Display message to enter API KEY in Home page
    st.warning("Please enter your Flywheel API key in the Home page to continue.")
    st.stop()
fw = flywheel.Client(st.session_state.api_key if st.session_state.authenticated else API_KEY)


#Have a drop down to select batch runs 
st.title("📦 Batch Runs")
st.write("Select a gear to batch run to view details:")
# gear_list = fw.gears()
# gear_names = [gear.gear.name for gear in gear_list]
#Order them alphabetically

#For now only keep circumference, freesurfer-recon-all-clinical, gambas, minimorph
gear_names = ['circumference', "freesurfer-recon-all", "recon-all-clinical", 'gambas', 'minimorph']
gear_names.sort()
selected_gear_name = st.selectbox("Select Gear", gear_names)
selected_gear = next((gear for gear in gear_names if gear == selected_gear_name), None)

#Dropdown to select project
project_list = fw.projects()
project_names = [project.label for project in project_list]
selected_project_name = st.selectbox("Select Project", project_names)
selected_project = next((project for project in project_list if project.label == selected_project_name), None)
if selected_project is None:
    st.warning("Please select a valid project.")
    st.stop()
st.info(f"Project: {selected_project.label} Subjects n = {len(selected_project.subjects())}\nSessions n = {len(selected_project.sessions())}...")

if selected_gear == "circumference":
    fw_project = fw.projects.find_first(f'label={selected_project.label}')
    run_circumference_gear(fw_project)

