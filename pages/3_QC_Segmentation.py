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


def convert_gif_to_mp4(gif_path):

    clip = mp.VideoFileClip((f))
    video_path = os.path.splitext(f)[0]+'.mp4'
    clip.write_videofile(video_path)
    
    print(f"Converted {f} to {video_path}")
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
                

def get_data(sub_label, ses_label, seg_gear, input_gear, v,download_dir,project_label,api_key):
    
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
            if input_gear.startswith("gambas"):
                seg_gear = seg_gear + "_gambas"
                print("Looking for anaylyses from ", seg_gear)

            matches = [asys for asys in analyses if asys.label.startswith(seg_gear) and asys.job.get('state') == "complete"]
            print("Matches: ", len(matches),[asys.label for asys in matches] )
            # If there are no matches, the gear didn't run
            if len(matches) == 0:
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
    log.info("QC responses have been uploaded to the project information tab.")


    
def find_csv_file(directory, username):
    username_cleaned = username.replace(" ", "")
    
    for root, _, files in os.walk(directory):
        for file in files:
            if username_cleaned in file and file.endswith(".csv"):
                return os.path.join(root, file)  # Return the first matching file found

    return None  # No matching file found


import streamlit as st

today = datetime.now()

# Now you can access them like this:
API_KEY = os.getenv("FW_CLI_API_KEY")
if API_KEY is None:
    raise ValueError("API_KEY not found. Please add it to your .env file.")

fw = flywheel.Client(api_key=API_KEY)

st.title("🧠 Segmentation QC Demo")
#Video : /Users/Hajer/unity/fw-notebooks/QC/output_video.mp4

#Upload the list of outliers as a csv
uploaded_outliers = st.file_uploader("Upload outlier file", type=["csv"])
#The other option is to retrieve the outliers from flywheel : Not yet Implemented

# 2. Visualizing the data: Generate overlay videos for QC
segmentation_tool = st.radio("Segmentation source:", ["recon-all-clinical", "minimorph"])
segmentation_suffix = {"minimorph":"segmentation","recon-all-clinical":"aparc+aseg"}

#Ask for input: username
username = st.text_input("Enter your name or initials:")
if username:
    st.success(f"Hello, {username}! You can proceed with the QC.")

df_outliers = None
if uploaded_outliers is not None:
    df_outliers = pd.read_csv(uploaded_outliers)
    st.write("Outliers DataFrame:")
    st.dataframe(df_outliers)

if st.button("Start QC") and segmentation_tool and uploaded_outliers is not None and username:
    st.write("Starting QC process...")
    st.write('Instructions: Review the video below and provide your feedback. You can rate the segmentation quality and add comments as needed.')
    st.write('###1. Getting the data: Data will be temporarily downloaded to the local machine.')
    

    metrics =["Are the main brain structures correctly segmented (e.g., gray/white matter, ventricles)? (Y/N)",
          "Are there any major errors or artifacts? (No/Minor/Major)",
          "Overall segmentation quality (Good/Poor)", "Include in analysis? (Y/N)",
          "Comments"]
    
    for _, row in df_outliers.iterrows():
        input_gear, gear_v = row["input_gear_v"].split("/")[0], row["input_gear_v"].split("/")[1]
        sub_label, ses_label = row["subject"],row["session"]
        project_label = row["project"].strip()
        print(project_label)

        project = fw.projects.find_first(f'label={project_label}')
        project = project.reload()
        
        download_dir = os.path.join(Path(__file__).parent,"..","data")
        get_data(sub_label, ses_label, segmentation_tool,input_gear, gear_v,download_dir, project, API_KEY)
        #Create the output video
        segmentation_path , native_scan_path = None, None
        files = os.listdir(path=f"{download_dir}/{sub_label}/{ses_label}")

        for file in files:
            if file.endswith(f'{segmentation_suffix[segmentation_tool]}.nii.gz'):
                segmentation_path = os.path.join(f"{download_dir}/{sub_label}/{ses_label}",file)
            else:
                native_scan_path = os.path.join(f"{download_dir}/{sub_label}/{ses_label}",file)

        if segmentation_path is None or native_scan_path is None:
            st.error(f"Missing files for {sub_label} {ses_label}. Skipping...")
            continue


        nifti_overlay_gif_3planes(native_scan_path, segmentation_path,
            out_gif=f"{download_dir}/{sub_label}/{ses_label}/overlay_3planes.gif",
            out_mp4=f"{download_dir}/{sub_label}/{ses_label}/overlay_3planes.mp4",
            target_height=200, alpha=0.4, cmap="viridis",
            fps=5, layout="horizontal")
        
        #When the video is ready, delete the local nifti files to save space
        os.remove(native_scan_path)
        os.remove(segmentation_path)
        st.write(f"### Subject: {sub_label} Session: {ses_label}")

        #Display the video
        st.video(out_mp4)
        st.write("Answer the questions below:")

        for metric in metrics[:-1]:
            response = ""
            #Provide a few buttons for each option
            if "Y/N" in metric:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Y - {metric}", key=f"{sub_label}_{ses_label}_{metric}_Y"):
                        response = "Y"
                with col2:
                    if st.button(f"N - {metric}", key=f"{sub_label}_{ses_label}_{metric}_N"):
                        response = "N"
            elif "No/Minor/Major" in metric:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"No - {metric}", key=f"{sub_label}_{ses_label}_{metric}_No"):
                        response = "No"
                with col2:
                    if st.button(f"Minor - {metric}", key=f"{sub_label}_{ses_label}_{metric}_Minor"):
                        response = "Minor"
                with col3:
                    if st.button(f"Major - {metric}", key=f"{sub_label}_{ses_label}_{metric}_Major"):
                        response = "Major"
            elif "Good/Poor" in metric:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Good - {metric}", key=f"{sub_label}_{ses_label}_{metric}_Good"):
                        response = "Good"
                with col2:
                    if st.button(f"Poor - {metric}", key=f"{sub_label}_{ses_label}_{metric}_Poor"):
                        response = "Poor"

            # Store responses for paired image set
            responses.append(response.upper())

        # Have user input any comments to elaborate on artefact presence
        #Comment section with user input
        comments = st.text_input(f"{metrics[-1]}: ")
        responses.append(comments)
        # responses.append(timestamp)

        print(responses)
        ratings_file = os.path.join(download_dir,f'Parcellation_QC_{username.replace(" ","")}.csv')
        save_rating(ratings_file,responses,project,metrics)

        # if i < len(metrics) - 2:  # Only ask if not the last metric
        #Have a prompt to continue or exit
        if st.button("Continue to next subject"):
            st.write("Continuing to next subject...")
            clear_output(wait=True)
            continue
        else:
            st.write("QC process paused. You can resume later.")
            break
