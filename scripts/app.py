import streamlit as st
import pandas as pd
import os
from pull_results import *
import flywheel

dotenv.load_dotenv()


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

# ---- Streamlit App ----
# set the page layout to wide
st.set_page_config(layout="wide")
st.logo("../docs/logo.jpg", size='large')
st.title(":streamlit: UNITY Derivative Downloader")
st.write("Download and compile derivatives from multiple projects into a single CSV file.")
st.write("Select the projects and derivative types from the sidebar :point_left:, then click 'Fetch derivatives'.")

# Sidebar inputs
st.sidebar.header("Settings")
#Project IDs is a list generated using fw.projects() by getting the project label
api_key = os.getenv('FW_CLI_API_KEY')
fw = flywheel.Client(api_key=api_key)
projects = [proj.label for proj in fw.projects()]
# Dropdown for project selection (can select multiple)
project_ids = st.sidebar.multiselect("Select Projects", projects)

derivative_type = st.sidebar.radio("Derivative type:", ["recon-all-clinical", "minimorph"])

#Add date range selector
after_date = st.sidebar.date_input("Analysis after date: ", value=None)
# start_date = st.sidebar.date_input("Start date")
# end_date = st.sidebar.date_input("End date")
#Add slider for last n versions of the gear
versions = st.sidebar.slider("Select number of recent gear versions to include:", min_value=1, max_value=10, value=4)

#Get the selected project IDs
print("Selected projects: ", project_ids)

lastV = fw.get_all_gears(filter=f"gear.name='{derivative_type}'", sort="created:desc", limit=versions, all_versions=True, exhaustive=True)
gear_versions = [] 
for gear_version in lastV:  
    #Add gear version to a list
    gear_versions = [gear_version.gear.version for gear_version in lastV]


#Add tickboxes if recon all was selected, to select area, thickness, and volume
if derivative_type == "recon-all-clinical":
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
#Add a stop button to interrupt the process


if st.sidebar.button("Fetch derivatives"):
    
    all_derivatives = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, proj in enumerate(project_ids):
        st.write(f"Fetching {derivative_type} for {proj}...")
        api_key = get_env_from_zshrc('FW_CLI_API_KEY')
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
        
        derivatives = download_derivatives(proj, derivative_type,gear_versions, keywords, after_date)
        if derivatives:
            all_derivatives.extend([derivatives])
        progress.progress((i+1)/len(project_ids))
    
    # Assemble into CSV
    print(all_derivatives)

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
