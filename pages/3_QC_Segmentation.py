import nibabel as nib
from nilearn import image, plotting
import nibabel.freesurfer as fs
from nilearn.image import resample_to_img
import matplotlib.colors as mcolors
from nibabel.freesurfer import read_annot
import imageio
from pathlib import Path
from matplotlib.colors import ListedColormap
from IPython.display import Image
import io
from datetime import datetime
from IPython.display import display, clear_output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

import matplotlib.image as mpimg
import re
#from dotenv import load_dotenv
import flywheel
import logging
log = logging.getLogger(__name__)
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
from tqdm import tqdm
from datetime import datetime
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt

import skimage
import plotly.express as px
from skimage.transform import resize
from ipywidgets import interact, FloatSlider, IntSlider

import os
import sys
import subprocess

import streamlit as st
from IPython.display import Image as IPImage, display
from PIL import Image
# from utils.giftomp4 import *
import moviepy.editor as mp
from utils.authentication import login_screen

def convert_gif_to_mp4(gif_path):

    clip = mp.VideoFileClip((gif_path))
    video_path = os.path.splitext(gif_path)[0]+'.mp4'
    clip.write_videofile(video_path)
    
    print(f"Converted {gif_path} to {video_path}")
    return video_path

def preprocess_nifti(nifti_path, target_height=300):
    # load data
    data = nib.load(nifti_path).get_fdata()
    data = np.nan_to_num(data)
    xyz = data.shape

    # Normalise entire data arrays
    min_val, max_val = np.min(data), np.max(data)
    if max_val > min_val:
        data = (255 * (data - min_val) / (max_val - min_val))
    else:
        data = np.zeros_like(data)  # Black image if constant
        
    # Determine orientation and choose slices correctly
    if 'SAG' in nifti_path.upper():
        orientation = "sag"
        plane = (1, 2)
        ax = 0
    elif 'COR' in nifti_path.upper():
        orientation = "cor"
        plane = (0, 2)
        ax = 1
    elif 'AXI' in nifti_path.upper():
        orientation = "axi"
        plane = (0, 1)
        ax = 2
    
    # Resize the image to target height
    w, h = xyz[plane[0]], xyz[plane[1]]
    scale_factor = target_height / h
    new_width = int(w * scale_factor)
    
    # Create new shape and resize
    new_shape = list(xyz)
    new_shape[plane[0]], new_shape[plane[1]] = new_width, target_height
    data = skimage.transform.resize(data, new_shape, mode='constant', preserve_range=True)
    
    return data, plane[0], ax


def nifti_overlay_gif_3planes(native_path, seg_path,
                              out_gif="overlay_3planes.gif",
                              out_mp4="overlay_3planes.mp4",
                              target_height=200, alpha=0.4, cmap="viridis",
                              fps=5, layout="horizontal"):

    native = nib.load(native_path).get_fdata().astype(np.float32)
    seg = nib.load(seg_path).get_fdata().astype(np.float32)
    if native.shape != seg.shape:
        raise ValueError("Shape mismatch between native and segmentation")

    cmap_fn = plt.get_cmap(cmap)

    planes = {
        "axial":    [native[:, :, i] for i in range(native.shape[2])],
        "coronal":  [native[:, i, :] for i in range(native.shape[1])],
        "sagittal": [native[i, :, :] for i in range(native.shape[0])]
    }
    seg_planes = {
        "axial":    [seg[:, :, i] for i in range(seg.shape[2])],
        "coronal":  [seg[:, i, :] for i in range(seg.shape[1])],
        "sagittal": [seg[i, :, :] for i in range(seg.shape[0])]
    }

    min_len = min(len(planes["axial"]), len(planes["coronal"]), len(planes["sagittal"]))
    frames = []

    for i in range(min_len):
        images = []
        for plane in ["axial", "coronal", "sagittal"]:
            sl = planes[plane][i]
            seg_sl = seg_planes[plane][i]

            scale = target_height / sl.shape[0]
            new_shape = (target_height, int(sl.shape[1]*scale))
            sl_r = resize(sl, new_shape, preserve_range=True, order=1)
            seg_r = resize(seg_sl, new_shape, preserve_range=True, order=0)

            if sl_r.max() > sl_r.min():
                sl_disp = ((sl_r - sl_r.min()) / (sl_r.max() - sl_r.min()) * 255).astype(np.uint8)
            else:
                sl_disp = np.zeros_like(sl_r).astype(np.uint8)

            bg = np.stack([sl_disp]*3, axis=-1)

            mask = seg_r > 0
            if mask.any():
                seg_norm = seg_r / (seg_r.max() if seg_r.max() > 0 else 1.0)
                seg_rgb = (cmap_fn(seg_norm)[:, :, :3] * 255).astype(np.uint8)
                bg[mask] = ((1 - alpha) * bg[mask] + alpha * seg_rgb[mask]).astype(np.uint8)

            images.append(bg)

        # rotate + match heights
        rotated_images = []
        heights = []

        for img in images:
            pil_img = Image.fromarray(img)
            pil_img_rot = pil_img.rotate(90, expand=True)
            rotated_images.append(pil_img_rot)
            heights.append(pil_img_rot.height)

        max_height = max(heights)
        resized_images = []
        for img in rotated_images:
            new_width = int(img.width * (max_height / img.height))
            resized_img = img.resize((new_width, max_height))
            resized_images.append(np.array(resized_img))

        combined = np.concatenate(resized_images, axis=1)
        frames.append(combined.astype(np.uint8))

    # --- Save GIF ---
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    print(f"✅ Saved GIF to {out_gif}")

    # --- Convert to MP4 ---
    out_mp4 = convert_gif_to_mp4(out_gif)
    print(f"✅ Saved MP4 to {out_mp4}")

    return out_mp4


def download_analysis_files(asys,sub_label,ses_label,str_pattern,download_dir):
    download_dir = Path(f'{download_dir}/{sub_label}/{ses_label}/')
    download_dir.mkdir(parents=True,exist_ok=True)
    input_file= asys.inputs[0]
    download_path = os.path.join(download_dir , input_file.name)
    fw.download_input_from_analysis( asys.id, input_file.name, download_path)
    print("Downloaded input file:",download_path)
    
    print([file.name for file in asys.files])
    for file in asys.files:            
        if file.name.endswith('nii.gz') : #re.search(str_pattern, file.name) and #or re.search(rf'{input_gear}.*\.nii.gz', file.name):

        #if file.name.endswith('aparc+aseg.nii.gz') or file.name.endswith('synthSR.nii.gz') or re.search('ResCNN.*\.nii.gz', file.name) or re.search('mrr_fast.*\.nii.gz', file.name) or re.search('mrr-axireg.*\.nii.gz', file.name) or re.search('.*\.zip', file.name):
            parcellation = file
            print("Found ", file.name)
            if file :
                
                download_path = os.path.join(download_dir , parcellation.name)
                
                parcellation.download(download_path)
                print('Downloaded parcellation ',download_path)
                

def get_data(sub_label, ses_label, asys, seg_gear, input_gear, v,download_dir,project,api_key):
    
    from fw_client import FWClient
    #api_key = os.environ.get('FW_CLI_API_KEY')
    fw_ = FWClient(api_key=api_key)

    # project = fw.projects.find_first(f'label={project_label}')
    # display(f"Project: {project.label}")
    # project = project.reload()

    subject = project.subjects.find_first(f'label="{sub_label}"')
    subject = subject.reload()
    sub_label = subject.label
    
    session = subject.sessions.find_first(f'label="{ses_label}"')
    session = session.reload()
    ses_label = session.label
    print(seg_gear, input_gear)
    analyses = session.analyses

    seg_parc_map = {"recon-all-clinical":"aparc+aseg","minimorph":"segmentation"}
    str_pattern = seg_parc_map[seg_gear]

    # If there are no analyses containers, we know that this gear was not run
    if len(analyses) == 0:
        run = 'False'
        status = 'NA'
        print('No analysis containers')
    else:
        try:
            
            #Get the asys by id
            if asys is not None:
                asys = fw.get_analysis(asys)
                download_analysis_files(asys,sub_label,ses_label,str_pattern,download_dir)

            elif input_gear.startswith("gambas"):
                seg_gear = seg_gear + "_gambas"
                print("Looking for anaylyses from ", seg_gear)
            else:
                matches = [asys for asys in analyses if asys.label.startswith(seg_gear) and asys.job.get('state') == "complete"]
                print("Matches: ", len(matches),[asys.label for asys in matches] )
                # If there are no matches, the gear didn't run
                if len(matches) == 0 and asys is None:
                    run = 'False'
                    status = 'NA'
                    print(f"Did not find any matched, {seg_gear} did not run.")
                # If there is one match, that's our target
                elif len(matches) == 1:
                    run = 'True'
                    #status = matches[0].job.get('state')
                    #print(status)
                    #print("Inputs ", matches[0])
                    asys=matches[0]
                    download_analysis_files(asys,sub_label,ses_label,str_pattern,download_dir)


                else:
                    last_run_date = max([asys.created for asys in matches])
                    last_run_analysis = [asys for asys in matches if asys.created == last_run_date and asys.job.get('state') == "complete"]

                    # There should only be one exact match
                    last_run_analysis = last_run_analysis[0]

                    run = 'True'
                    #status = last_run_analysis.job.get('state')
                    asys=last_run_analysis
                    download_analysis_files(asys,sub_label,ses_label,str_pattern,download_dir)

        except Exception as e:
            print(f"Exception caught for {sub_label} {ses_label}: ", e)
            


def load_ratings(RATINGS_FILE,metrics):
    if os.path.exists(RATINGS_FILE):
        return pd.read_csv(RATINGS_FILE)
    
    return pd.DataFrame(columns=["User", "Timestamp", "Project", "Subject", "Session"] + metrics)
    
# Function to simplify acquisition labels
def simplify_label(label):
    # Initialize empty result
    result = []
    
    # Check for orientation
    if 'AXI' in label.upper():
        result.append('AXI')
    elif 'COR' in label.upper():
        result.append('COR')
    elif 'SAG' in label.upper():
        result.append('SAG')
        
    elif 'Localizer' in label:
        result.append('LOC')
        
    # Check for T1/T2
    if 'T1' in label.upper():
        result.append('T1')
    elif 'T2' in label.upper():
        result.append('T2')
        
    # Check for fast vs. standard labels
    if 'FAST' in label.upper():
        result.append('FAST')
    elif 'STANDARD' in label.upper():
        result.append('STANDARD')
        
    # Check for Gray_White
    if 'Gray' in label.upper():
        result.append('GrayWhite')
        
    # Return combined result or original label if no matches
    return '_'.join(result) if result else label

def save_rating(ratings_file, responses,project,metrics):
    df = load_ratings(ratings_file,metrics)
    print(df.columns)
    print(responses, len(responses))
    new_entry = pd.DataFrame([responses], 
                              columns=df.columns)
    
    df = pd.concat([df, new_entry], ignore_index=True)
    # df.loc[:, 'Acquisition'] = df['Acquisition'].apply(simplify_label)
    
    df.to_csv(ratings_file, index=False)
    log.info(f"\nSaved rating: {ratings_file}")
    custom_name = ratings_file.split('/')[-1]
    #project.upload_file(ratings_file, filename=custom_name)


    
def find_csv_file(directory, username):
    username_cleaned = username.replace(" ", "")
    
    for root, _, files in os.walk(directory):
        for file in files:
            if username_cleaned in file and file.endswith(".csv"):
                return os.path.join(root, file)  # Return the first matching file found

    return None  # No matching file found


def qc_subject(row, segmentation_tool, metrics):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    input_gear, gear_v = None, None #row["input_gear_v"].split("/")[0], row["input_gear_v"].split("/")[1]
    sub_label, ses_label = row["subject"],row["session"]
    asys = row.get("analysis_id", None)
    project_label = row["project"].strip()
    print(project_label)

    project = fw.projects.find_first(f'label={project_label}')
    project = project.reload()
    
    download_dir = os.path.join(Path(__file__).parent,"..","data")
    with st.spinner("Downloading data..."):
        st.write('Data will be temporarily downloaded to the local machine.')
        get_data(sub_label, ses_label, asys, segmentation_tool, input_gear, gear_v, download_dir, project, st.session_state.api_key)
    #Create the output video
    segmentation_path , native_scan_path = None, None
    files = os.listdir(path=f"{download_dir}/{sub_label}/{ses_label}")

    for file in files:
        print(file)
        if file.endswith(f'{segmentation_suffix[segmentation_tool]}.nii.gz'):
            segmentation_path = os.path.join(f"{download_dir}/{sub_label}/{ses_label}",file)
        if file.endswith('synthSR.nii.gz'):
            print("Found synthSR")
            native_scan_path = os.path.join(f"{download_dir}/{sub_label}/{ses_label}",file)
            #As soon as we find segmentation and native scan, we can stop    
        elif file.endswith('.nii.gz') and not file.endswith(f'{segmentation_suffix[segmentation_tool]}.nii.gz') and st.session_state.segmentation_tool != "recon-all-clinical":   
            
            native_scan_path = os.path.join(f"{download_dir}/{sub_label}/{ses_label}",file)

    if segmentation_path is None or native_scan_path is None:
        st.error(f"Missing files for {sub_label} {ses_label}. Skipping...")
        return

    print("NATIVE SCAN AND SEGMENTATION SCAN: ", native_scan_path, segmentation_path)
    out_mp4 = nifti_overlay_gif_3planes(native_scan_path, segmentation_path,
        out_gif=f"{download_dir}/{sub_label}/{ses_label}/overlay_3planes.gif",
        out_mp4=f"{download_dir}/{sub_label}/{ses_label}/overlay_3planes.mp4",
        target_height=200, alpha=0.4, cmap="viridis",
        fps=5, layout="horizontal")
    
    #When the video is ready, delete the local nifti files to save space
    os.remove(native_scan_path)
    os.remove(segmentation_path)
    st.write(f"### Subject: {sub_label} Session: {ses_label}")

    st.session_state.responses = [st.session_state.username, timestamp, project_label, sub_label, ses_label] 
    
    st.session_state.df_outliers.drop(columns=['is_outlier'])
    filtered_cols = [col for col in st.session_state.df_outliers.columns if (
                                            (col.endswith("_zscore") or col.endswith("_cov"))
                                            and not col.startswith("n_roi_outliers")
                                        )]
    # 2. From those, keep only columns where the value is 1
    outlier_rois = [col for col in filtered_cols if row[col] == 1]
    #Get rid of the suffix in the names (mm_ and ra_)
    outlier_rois = [re.sub(r'^(mm_|ra_)', '', col) for col in outlier_rois]

    # print("Outlier regions for this subject (using z score and cov):")
    # print(f"{set(outlier_rois)}")
    outliers = list(set(outlier_rois))

    #Display the video
    st.video(out_mp4)
    st.write ('⚠️ Regions with outlier metrics for this subject (using z score and cov):')
    st.write(f"{outliers}")

   
    with st.form("qc_form", clear_on_submit=True):
        print(st.session_state.responses)
        # --- Display all questions harcoded ---
        st.write("Answer the questions below:")
        metric1 = "Are the main brain structures correctly segmented (e.g., gray/white matter, ventricles)? (Y/N)"
        response1 = st.radio(f"**Q1: {metric1}**", ["Yes", "No"], key="q_0")
        st.session_state.responses.append(response1)
        metric2 = "Are there any major errors or artifacts? (No/Minor/Major)"
        response2 = st.radio(f"**Q2: {metric2}**", ["No", "Minor", "Major"], key="q_1")
        st.session_state.responses.append(response2)
        metric3 = "Overall segmentation quality (Good/Poor)"
        response3 = st.radio(f"**Q3: {metric3}**", ["Good", "Poor"], key="q_2")
        st.session_state.responses.append(response3)
        metric4 = "Include in analysis? (Y/N)"
        response4 = st.radio(f"**Q4: {metric4}**", ["Yes", "No"], key="q_3")
        st.session_state.responses.append(response4)
        metric5 = "Comments"
        response5 = st.text_input(f"**Q5: {metric5}**", key="q_4")
        st.session_state.responses.append(response5)

    

        submitted = st.form_submit_button('Save/Next subject')

        if submitted:
            # st.session_state.responses  = responses
            st.success("All questions answered! ✅")
            ratings_file = os.path.join(download_dir, f'Parcellation_QC_{st.session_state.username.replace(" ","_")}.csv')
            save_rating(ratings_file, st.session_state.responses, project, metrics)
            # Advance to next row and refresh UI without losing state
            if "row" not in st.session_state:
                st.session_state.row = 0
            

            current = st.session_state.row == len(st.session_state.df_outliers.index) - 1
            print('Current index', st.session_state.row, current, len(st.session_state.df_outliers.index))
            if current:
                #Check if the file exists
                #Show an alert 

                st.success(f"You have completed the QC for all subjects! 🎉 Data were saved under {os.path.join(download_dir, f'Parcellation_QC_{st.session_state.username.replace(" ","_")}.csv')}")
                ratings_df = load_ratings(ratings_file, metrics)
                return ratings_df
                st.balloons()
                st.stop() 
            else:
                st.session_state.row = min(st.session_state.row + 1, len(st.session_state.df_outliers.index) - 1)
                st.rerun()

# Now you can access them like this:
API_KEY = os.getenv("FW_CLI_API_KEY")
if API_KEY is None:
    login_screen()
    # if "api_key" not in st.session_state:
    #     st.session_state.api_key = st.text_input("Enter Flywheel API key:", type="password")

    # if API_KEY:
    #     st.session_state.api_key = API_KEY
        
    # else:
    #     raise ValueError("API_KEY not found. Please add it to your .env file.")

# try:
#     fw = flywheel.Client(api_key=st.session_state.api_key)
# except Exception as e:
#     raise ValueError("Invalid API key or connection error. Please try again.") from e

st.title("🧠 Segmentation QC Demo")
#Video : /Users/Hajer/unity/fw-notebooks/QC/output_video.mp4

#Upload the list of outliers as a csv
uploaded_outliers = st.file_uploader("Upload outlier file", type=["csv"])
#The other option is to retrieve the outliers from flywheel : Not yet Implemented

# 2. Visualizing the data: Generate overlay videos for QC
segmentation_tool = st.radio("Segmentation source:", ["recon-all-clinical", "minimorph"])
segmentation_suffix = {"minimorph":"segmentation","recon-all-clinical":"aparc+aseg"}

#Ask for input: username
st.session_state.username = st.text_input("Enter your name or initials:")
if st.session_state.username:
    st.success(f"Hello, {st.session_state.username}! You can proceed with the QC.")

st.session_state.df_outliers = None
if uploaded_outliers is not None:
    st.session_state.df_outliers = pd.read_csv(uploaded_outliers)
    st.write(f"Outliers DataFrame: {st.session_state.df_outliers.shape[0]} rows and {st.session_state.df_outliers.shape[1]} columns")
    st.dataframe(st.session_state.df_outliers)

if segmentation_tool and uploaded_outliers is not None and st.session_state.username: #st.button("Start QC") and 
    st.session_state.segmentation_tool = segmentation_tool
    st.write("Starting QC process...")
    st.write('Instructions: Review the video below and provide your feedback. You can rate the segmentation quality and add comments as needed.')
    
    

    metrics =["Are the main brain structures correctly segmented (e.g., gray/white matter, ventricles)? (Y/N)",
          "Are there any major errors or artifacts? (No/Minor/Major)",
          "Overall segmentation quality (Good/Poor)", "Include in analysis? (Y/N)",
          "Comments"]
    

    if "row" not in st.session_state:
        print('Initializing row index...')
        st.session_state.row = 0

    def next_row():
        st.session_state.row += 1

    def prev_row():
        st.session_state.row -= 1

    # col1, col2, _, _ = st.columns([0.1, 0.17, 0.1, 0.63])

    # if st.session_state.row < len(st.session_state.df_outliers.index)-1:
    #     col2.button(">", on_click=next_row)
    # else:
    #     col2.write("") 

    # if st.session_state.row > 0:
    #     col1.button("<", on_click=prev_row)
    # else:
    #     col1.write("") 
    
    current_row = st.session_state.df_outliers.iloc[st.session_state.row]
    print('Current row', current_row)
    # st.write(current_row)
    #Make sure to stop when we reach the last row
    
    qc_subject(current_row, segmentation_tool, metrics)
    
    
    # if "current_subject_index" not in st.session_state:
    #     st.session_state["current_subject_index"] = 0

    
    # if st.session_state["current_subject_index"] != len(st.session_state.df_outliers):
    #     subject = subjects[st.session_state["current_subject_index"]]
    # qc_subject(st.session_state.df_outliers.iloc[st.session_state["current_subject_index"]])

    # for _, row in st.session_state.df_outliers.iterrows():
    #     content_placeholder = st.empty()
    #     qc_subject(row)
    #     # if i < len(metrics) - 2:  # Only ask if not the last metric
    #     #Have a prompt to continue or exit
    #     if st.button("Continue to next subject"):
    #         st.write("Continuing to next subject...")
    #         clear_output(wait=True)
    #         continue
    #     else:
    #         st.write("QC process paused. You can resume later.")
    #         break
