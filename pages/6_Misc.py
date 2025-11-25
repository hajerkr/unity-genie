import flywheel
from fw_client import FWClient
import pandas as pd
import streamlit as st
import os
from pathlib import Path
import pathvalidate as pv

def load_flywheel_data(project_label):
    # fwclient = FWClient(
    # api_key=st.session_state.api_key,
    # timeout=100,
    # )
    #fw = flywheel.Client(api_key=st.session_state.api_key)

    protocol_label = 'QC'
    #project_label = "AKU MiNE Low Field"

    project = st.session_state.fw.projects.find_one(f'label={project_label}')
    project = project.reload()

    dest_proj_id = project.id
    protocol_filter = f"parents.project={dest_proj_id}"
    if protocol_label:
        protocol_filter += f",label={protocol_label}"

    protocol = st.session_state.fwclient.get(
        f"/api/read_task_protocols?filter=parents.project={dest_proj_id},label={protocol_label}"
    ).get("results")[0]


    my_filter = f"protocol_id={protocol._id},status=Todo"
    reader_tasks_todo = st.session_state.fwclient.get(f"/api/readertasks/project/{dest_proj_id}?filter={my_filter}").to_flat().get("results")
    

    my_filter = f"protocol_id={protocol._id},status=In_progress"
    reader_tasks_inprogress = st.session_state.fwclient.get(f"/api/readertasks/project/{dest_proj_id}?filter={my_filter}").to_flat().get("results")
    st.info(f"Tasks In Progress: {len(reader_tasks_inprogress)}  \nTasks To Do: {len(reader_tasks_todo)}")
        
    # Get files that have not been QC'ed
    # These should be the files that still have the "read" tag

    #For every task, get the subejct id, session id, acquisition id, and file id (name)
    # Combine all tasks (Todo and In_progress) into a single list
    all_tasks = reader_tasks_todo + reader_tasks_inprogress

    # Initialize an empty list to store task details
    task_details = []

    # Iterate over all tasks and collect details
    for task in all_tasks:
        task_info = {
            # "Task ID": task._id,
            "Project": project_label,
            "Subject ID": task.parent_info.subject.label,
            "Session ID": task.parent_info.session.label,
            "Session Timestamp": st.session_state.fw.get(task.parent_info.session._id).reload().timestamp,
            "Acquisition ID": task.parent_info.acquisition.label if task.parent_info.acquisition else None,
            "Status": task.status
        }
        task_details.append(task_info)

    # Convert the list of task details into a DataFrame
    tasks_df = pd.DataFrame(task_details)

    # Display the DataFrame
    st.dataframe(tasks_df.head(4))
    return tasks_df
    


st.title("⌗ Flywheel QC Tasks")
#Video : /Users/Hajer/unity/fw-notebooks/QC/output_video.mp4

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
else:
    st.session_state.api_key = API_KEY

#fw = flywheel.Client(st.session_state.api_key if st.session_state.authenticated else API_KEY)
data_dir = Path(__file__).parent/'../data/'

#Add dropdown to select project
@st.cache_data(ttl=600)
def get_projects():
    return [p.label for p in st.session_state.fw.projects()]

projects = get_projects()

project_label = st.selectbox("Select Project", projects)
selected_project = st.session_state.fw.projects.find_first(f'label={project_label}')

if st.button("Fetch Data"):
    with st.spinner("Loading data from Flywheel..."):
        project_data = load_flywheel_data(project_label)
        st.success("Data loaded successfully!")
        #Download button for the data as CSV

        if not data_dir.exists():
            data_dir.mkdir(parents = True)

        project_path = pv.sanitize_filepath(data_dir/project_label, platform='auto')
        if not project_path.exists():
            project_path.mkdir(parents = True)

        outdir = os.path.join(project_path, f"{project_path}_pending_qc_tasks.csv")
        project_data.to_csv(outdir, index=False)
        if os.path.exists(outdir):
            with open(outdir, "rb") as f:
                st.download_button("Download CSV", f, file_name=outdir)

    st.subheader(f"Project: {project_label}")
    st.write(project_data)