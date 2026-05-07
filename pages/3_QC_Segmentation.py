#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import flywheel
import os
import re
import logging
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import random
import pandas as pd
import streamlit as st
import moviepy.editor as mp

from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.transform import resize
from skimage import exposure
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_ratings_filename():
    return (
        f"Segmentation_QC_{st.session_state.segmentation_tool}_"
        f"{st.session_state.username.replace(' ', '_')}.csv"
    )


def convert_gif_to_mp4(gif_path):
    clip = mp.VideoFileClip(gif_path)
    video_path = os.path.splitext(gif_path)[0] + ".mp4"
    clip.write_videofile(video_path, logger=None)
    return video_path


# ---------------------------------------------------------------------------
# VISUALISATION — MP4  (scrolling through all slices)
# ---------------------------------------------------------------------------

def nifti_overlay_gif_3planes(
    native_path, seg_path,
    out_gif="overlay_3planes.gif",
    out_mp4="overlay_3planes.mp4",
    target_height=200, alpha=0.4, cmap="viridis",
    fps=5, layout="horizontal",
):
    native = nib.load(native_path).get_fdata().astype(np.float32)
    seg    = nib.load(seg_path).get_fdata().astype(np.float32)

    if native.shape != seg.shape:
        raise ValueError("Shape mismatch between native and segmentation")

    cmap_fn = plt.get_cmap(cmap)

    planes = {
        "axial":    [native[:, :, i] for i in range(native.shape[2])],
        "coronal":  [native[:, i, :] for i in range(native.shape[1])],
        "sagittal": [native[i, :, :] for i in range(native.shape[0])],
    }
    seg_planes = {
        "axial":    [seg[:, :, i] for i in range(seg.shape[2])],
        "coronal":  [seg[:, i, :] for i in range(seg.shape[1])],
        "sagittal": [seg[i, :, :] for i in range(seg.shape[0])],
    }

    min_len = min(len(planes["axial"]), len(planes["coronal"]), len(planes["sagittal"]))
    frames  = []

    for i in range(min_len):
        images = []
        for plane in ["axial", "coronal", "sagittal"]:
            sl     = planes[plane][i]
            seg_sl = seg_planes[plane][i]

            scale     = target_height / sl.shape[0]
            new_shape = (target_height, int(sl.shape[1] * scale))
            sl_r      = resize(sl,     new_shape, preserve_range=True, order=1)
            seg_r     = resize(seg_sl, new_shape, preserve_range=True, order=0)

            if sl_r.max() > sl_r.min():
                sl_disp = ((sl_r - sl_r.min()) / (sl_r.max() - sl_r.min()) * 255).astype(np.uint8)
            else:
                sl_disp = np.zeros_like(sl_r, dtype=np.uint8)

            bg   = np.stack([sl_disp] * 3, axis=-1)
            mask = seg_r > 0
            if mask.any():
                seg_norm = seg_r / (seg_r.max() if seg_r.max() > 0 else 1.0)
                seg_rgb  = (cmap_fn(seg_norm)[:, :, :3] * 255).astype(np.uint8)
                bg[mask] = ((1 - alpha) * bg[mask] + alpha * seg_rgb[mask]).astype(np.uint8)

            images.append(bg)

        rotated = [Image.fromarray(img).rotate(90, expand=True) for img in images]
        max_h   = max(r.height for r in rotated)
        resized = [
            np.array(r.resize((int(r.width * (max_h / r.height)), max_h)))
            for r in rotated
        ]
        frames.append(np.concatenate(resized, axis=1).astype(np.uint8))

    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    out_mp4 = convert_gif_to_mp4(out_gif)
    return out_mp4


# ---------------------------------------------------------------------------
# VISUALISATION — PNG  (middle-slice contour overlay, black background)
# ---------------------------------------------------------------------------

def generate_png_qc(native_path, seg_path, out_png):
    """
    Render a 3×3 grid (rows = slice positions 1/3, 1/2, 2/3;
    cols = Sagittal, Coronal, Axial) with per-label contour overlays
    on a contrast-enhanced grayscale background.
    """
    anat_data  = nib.load(native_path).get_fdata()
    atlas_data = nib.load(seg_path).get_fdata()

    # Contrast-enhance anatomy
    anat_norm     = (anat_data - anat_data.min()) / (anat_data.max() - anat_data.min() + 1e-8)
    anat_enhanced = exposure.equalize_adapthist(anat_norm, clip_limit=0.005)

    # Build shuffled per-label colour palette so adjacent IDs differ visually
    unique_labels = np.unique(atlas_data)
    unique_labels = unique_labels[unique_labels != 0]
    cmap_base     = plt.get_cmap("tab20")
    label_colors  = [cmap_base(i % 20) for i in range(len(unique_labels))]
    random.seed(50)
    random.shuffle(label_colors)

    # Three evenly-spaced slice positions
    fractions = [1/3, 1/2, 2/3]
    shapes    = anat_data.shape  # (S, C, A)

    fig, axes = plt.subplots(len(fractions), 3, figsize=(18, 15), constrained_layout=True)
    fig.patch.set_facecolor("black")

    for row_idx, frac in enumerate(fractions):
        s_idx = int(shapes[0] * frac)
        c_idx = int(shapes[1] * frac)
        a_idx = int(shapes[2] * frac)

        views = [
            (np.rot90(anat_enhanced[s_idx, :, :], 3),
             np.rot90(atlas_data[s_idx, :, :], 3),   "Sagittal"),
            (np.rot90(anat_enhanced[:, c_idx, :], 3),
             np.rot90(atlas_data[:, c_idx, :], 3),   "Coronal"),
            (np.rot90(anat_enhanced[:, :, a_idx], 3),
             np.rot90(atlas_data[:, :, a_idx], 3),   "Axial"),
        ]

        for col_idx, (anat_sl, atlas_sl, title) in enumerate(views):
            ax = axes[row_idx, col_idx]
            ax.imshow(anat_sl, cmap="gray", origin="lower")

            for l_idx, lv in enumerate(unique_labels):
                mask = (atlas_sl == lv).astype(float)
                if np.any(mask):
                    ax.contour(mask, levels=[0.5], colors=[label_colors[l_idx]], linewidths=3.5)

            # Position label on first column
            # if col_idx == 0:
            #     ax.text(
            #         -20, anat_sl.shape[0] / 2,
            #         f"Pos: {frac:.2f}",
            #         color="yellow", va="center", ha="right",
            #         fontsize=12, fontweight="bold",
            #     )

            # # Column title on first row only
            # if row_idx == 0:
            #     ax.set_title(title, color="white", fontweight="bold", fontsize=16)

            ax.axis("off")

    plt.savefig(out_png, facecolor="black", dpi=150)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# FLYWHEEL DOWNLOAD
# ---------------------------------------------------------------------------

def download_analysis_files(asys, sub_label, ses_label, str_pattern, download_dir):
    dl_dir = Path(f"{download_dir}/{sub_label}/{ses_label}/")
    dl_dir.mkdir(parents=True, exist_ok=True)

    input_file    = asys.inputs[0]
    download_path = str(dl_dir / input_file.name)
    fw.download_input_from_analysis(asys.id, input_file.name, download_path)

    for file in asys.files:
        if file.name.endswith("nii.gz"):
            file.download(str(dl_dir / file.name))


def get_data(sub_label, ses_label, asys_id, seg_gear, download_dir, project):
    subject     = project.subjects.find_first(f"label={sub_label}").reload()
    session     = subject.sessions.find_first(f"label={ses_label}").reload()
    seg_parc    = {"recon-all-clinical": "aparc+aseg", "minimorph": "segmentation"}
    str_pattern = seg_parc.get(seg_gear, "segmentation")
    try:
        asys = fw.get_analysis(asys_id).reload()
        download_analysis_files(asys, sub_label, ses_label, str_pattern, download_dir)
    except Exception as e:
        print(f"Exception for {sub_label} {ses_label}: {e}")


def get_analysis_id_columns(segmentation_tool, available_columns=None):
    suffix = "ra" if segmentation_tool == "recon-all-clinical" else "mm"
    preferred = [
        f"MRR_analysis_id_{suffix}",
        f"GAMBAS_analysis_id_{suffix}",
        f"analysis_id_{suffix}",
    ]
    if available_columns is None:
        return preferred

    cols = list(available_columns)
    detected = [
        c for c in cols
        if "analysis_id" in c.lower()
        and re.search(rf"(^|[_-]){suffix}($|[_-])", c.lower())
    ]
    ordered = [c for c in preferred if c in cols]
    for c in detected:
        if c not in ordered:
            ordered.append(c)
    return ordered


def process_subject_row(row, segmentation_tool, segmentation_suffix, download_dir, fw, api_key, viz_mode):
    """Download NIfTI files and generate MP4 or PNG for one subject."""
    try:
        sub_label, ses_label = row["subject"], row["session"]
        project_label        = row["project"].strip()
        asys_id = None
        for col in get_analysis_id_columns(segmentation_tool, row.index):
            val = row.get(col, None)
            if pd.notna(val):
                asys_id = val
                break
        if asys_id is None:
            return f"{sub_label}-{ses_label}: missing analysis id"

        project = fw.projects.find_first(f"label={project_label}").reload()
        get_data(sub_label, ses_label, asys_id, segmentation_tool, download_dir, project)

        subject_dir = os.path.join(download_dir, sub_label, ses_label)
        if not os.path.exists(subject_dir):
            return f"{sub_label}-{ses_label}: folder missing"

        seg_path, native_path = None, None
        for fname in os.listdir(subject_dir):
            full = os.path.join(subject_dir, fname)
            if fname.endswith(f"{segmentation_suffix[segmentation_tool]}.nii.gz"):
                seg_path = full
            elif fname.endswith("synthSR.nii.gz"):
                native_path = full
            elif fname.endswith(".nii.gz") and segmentation_tool != "recon-all-clinical":
                native_path = full

        if seg_path is None or native_path is None:
            return f"{sub_label}-{ses_label}: missing files (seg={seg_path}, native={native_path})"

        if viz_mode == "mp4":
            nifti_overlay_gif_3planes(
                native_path, seg_path,
                out_gif=os.path.join(subject_dir, "overlay_3planes.gif"),
                out_mp4=os.path.join(subject_dir, "overlay_3planes.mp4"),
                target_height=200, alpha=0.4, cmap="viridis", fps=5,
            )
        else:  # png
            generate_png_qc(
                native_path, seg_path,
                out_png=os.path.join(subject_dir, "qc_slices.png"),
            )

        os.remove(native_path)
        os.remove(seg_path)
        return f"{sub_label}-{ses_label}: done"

    except Exception as e:
        return f"{row.get('subject','?')}-{row.get('session','?')}: ERROR → {e}"


# ---------------------------------------------------------------------------
# RATINGS
# ---------------------------------------------------------------------------

def load_ratings(ratings_file, metrics, download=False):
    if os.path.exists(ratings_file):
        df = pd.read_csv(ratings_file)
        if download:
            st.dataframe(df)
        return df
    return pd.DataFrame(columns=["user", "timestamp", "project", "subject", "session"] + metrics)


def save_rating(ratings_file, responses, project, metrics):
    df        = load_ratings(ratings_file, metrics)
    new_entry = pd.DataFrame([responses], columns=df.columns)
    df        = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ratings_file, index=False)
    # NOTE: do NOT upload here — uploading is handled in qc_subject() using
    # st.session_state.asys, which is the single container created at session
    # start. Uploading here previously created a new analysis container on
    # every save, causing the duplicates you saw in Flywheel.


def check_previous_reviews(project, username):
    project   = fw.projects.find_first("label=CapeTown26-sprint").reload()
    filtered  = [
        a for a in project.analyses
        if username.replace(" ", "_").lower() in a.label.lower()
    ]
    reviewed, user_asys_id, old_path = False, "", ""

    if filtered:
        latest       = filtered[-1]
        user_asys_id = latest.id
        csv_files    = [f for f in latest.files if f.name.endswith(".csv")]
        if csv_files:
            dl_dir = os.path.join(Path(__file__).parent, "..", "data")
            os.makedirs(dl_dir, exist_ok=True)
            for csv_file in csv_files:
                old_path = os.path.join(dl_dir, csv_file.name)
                try:
                    latest.download_file(csv_file.name, old_path)
                    reviewed = True
                    df = pd.read_csv(old_path)
                    if not df.empty and {"subject", "session", "user"}.issubset(df.columns):
                        st.session_state.asys = latest
                except Exception as e:
                    print("check_previous_reviews error:", e)
    else:
        analysis = project.add_analysis(
            label=(
                f"Segmentation_QC_{st.session_state.segmentation_tool}_"
                f"{st.session_state.username.replace(' ', '_')}"
            )
        )
        user_asys_id          = analysis.id
        st.session_state.asys = analysis

    return reviewed, user_asys_id, old_path


# ---------------------------------------------------------------------------
# PER-SUBJECT QC FORM
# ---------------------------------------------------------------------------

def qc_subject(row, segmentation_tool, metrics, media_placeholder):
    timestamp     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sub_label     = row["subject"]
    ses_label     = row["session"]
    project_label = row["project"].strip()
    download_dir  = os.path.join(Path(__file__).parent, "..", "data")
    viz_mode      = st.session_state.get("viz_mode", "mp4")

    st.write(f"### Subject: `{sub_label}`  |  Session: `{ses_label}`")
    st.session_state.responses = [
        st.session_state.username, timestamp, project_label, sub_label, ses_label
    ]

    # Outlier regions
    df_out = st.session_state.df_outliers
    df_out.drop(columns=["is_outlier"], errors="ignore", inplace=True)
    filtered_cols = [
        c for c in df_out.columns
        if (c.endswith("_zscore") or c.endswith("_cov"))
        and not c.startswith("n_roi_outliers")
    ]
    outlier_rois = [re.sub(r"^(mm_|ra_)", "", c) for c in filtered_cols if row[c] == 1]
    outliers     = list(set(outlier_rois))

    # ---- Display media -----------------------------------------------------
    try:
        media_placeholder.empty()

        if viz_mode == "mp4":
            mp4_path = os.path.join(download_dir, sub_label, ses_label, "overlay_3planes.mp4")
            with media_placeholder.container():
                st.video(mp4_path)
        else:
            png_path = os.path.join(download_dir, sub_label, ses_label, "qc_slices.png")
            with media_placeholder.container():
                st.image(png_path, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading media for {sub_label} {ses_label}: {e}")
        st.stop()

    st.write("⚠️ **Regions with outlier metrics** (z-score / CoV):")
    st.write(outliers if outliers else "_None flagged_")

    # ---- QC form -----------------------------------------------------------
    with st.form("qc_form", clear_on_submit=True):
        st.write("Answer the questions below:")

        responses = []
        qs = [
            ("Are the main brain structures correctly segmented (e.g., gray/white matter, ventricles)?",
             ["Yes", "No"]),
            ("Are there any major errors or artifacts?",
             ["No", "Minor", "Major"]),
            ("Overall segmentation quality",
             ["Good", "Poor"]),
            ("Include in analysis?",
             ["Yes", "No"]),
        ]
        for i, (question, opts) in enumerate(qs):
            r = st.radio(f"**Q{i+1}: {question}**", opts, key=f"q_{i}")
            responses.append(r)

        comment = st.text_input("**Q5: Comments**", key="q_4")
        responses.append(comment)

        submitted = st.form_submit_button("Save / Next subject ▶")

        if submitted:
            st.session_state.responses.extend(responses)
            st.success("Saved! ✅")

            ratings_file = os.path.join(download_dir, get_ratings_filename())
            save_rating(ratings_file, st.session_state.responses, None, metrics)
            st.session_state.asys.upload_file(ratings_file)

            at_end = st.session_state.row == len(df_out.index) - 1
            if at_end:
                st.success("QC complete for all subjects! 🎉")
                st.balloons()
                load_ratings(ratings_file, metrics, download=True)
            else:
                st.session_state.row = min(st.session_state.row + 1, len(df_out.index) - 1)
                st.rerun()


# ===========================================================================
# STREAMLIT PAGE LAYOUT
# ===========================================================================

st.set_page_config(page_title="Segmentation QC", page_icon="🧠", layout="wide")
st.title("🧠 Segmentation QC")

# --- Session state defaults -------------------------------------------------
_defaults = {
    "authenticated": False,
    "api_key":        None,
    "row":            None,
    "df_outliers":    None,
    "uploaded_outliers_sig": "",
    "username":       "",
    "segmentation_tool": None,
    "viz_mode":       "mp4",
    "data_prepared":  False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Auth -------------------------------------------------------------------
API_KEY = os.getenv("FW_CLI_API_KEY")
if not API_KEY and not st.session_state.authenticated:
    st.warning("Please enter your Flywheel API key in the Home page to continue.")
    st.stop()

fw = flywheel.Client(st.session_state.api_key if st.session_state.authenticated else API_KEY)

segmentation_suffix = {
    "minimorph":          "segmentation",
    "recon-all-clinical": "aparc+aseg",
}

# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ QC Settings")

    segmentation_tool = st.radio(
        "Segmentation source",
        ["recon-all-clinical", "minimorph"],
    )
    st.session_state.segmentation_tool = segmentation_tool

    viz_mode = st.radio(
        "Visualisation mode",
        ["mp4", "png"],
        format_func=lambda x: (
            "🎬 Scrolling video (MP4)" if x == "mp4"
            else "🖼️ Middle-slice PNG (faster)"
        ),
        key="viz_mode",
        help=(
            "MP4: scrolls through every slice across all three planes.\n"
            "PNG: renders the middle slice of each plane with coloured contour overlays — "
            "generates faster and is easier to compare across subjects."
        ),
    )

    username = st.text_input("Your name / initials", value=st.session_state.username)
    if username:
        st.session_state.username = username.replace(" ", "_")
        st.success(f"Hello, {username}!")

# --- File upload ------------------------------------------------------------
uploaded_outliers = st.file_uploader("📂 Upload outlier CSV", type=["csv"])

if uploaded_outliers is not None:
    upload_sig = f"{uploaded_outliers.name}:{uploaded_outliers.size}"
    should_reload = (
        st.session_state.df_outliers is None
        or st.session_state.uploaded_outliers_sig != upload_sig
    )

    if should_reload:
        df_uploaded = pd.read_csv(uploaded_outliers)
        df_uploaded.columns = [
            str(c).replace("\ufeff", "").strip().replace(" ", "_")
            for c in df_uploaded.columns
        ]
        st.session_state.df_outliers = df_uploaded
        st.session_state.uploaded_outliers_sig = upload_sig
        st.session_state.row = 0
        st.session_state.data_prepared = False   # reset so download re-runs if file changes

    st.write(
        f"Loaded **{st.session_state.df_outliers.shape[0]}** subjects × "
        f"**{st.session_state.df_outliers.shape[1]}** columns"
    )
    st.dataframe(st.session_state.df_outliers)

# --- Main flow --------------------------------------------------------------
ready = (
    segmentation_tool
    and uploaded_outliers is not None
    and st.session_state.username
    and st.session_state.df_outliers is not None
)

if ready:
    metrics = [
        "Are the main brain structures correctly segmented (e.g., gray/white matter, ventricles)?",
        "Are there any major errors or artifacts?",
        "Overall segmentation quality",
        "Include in analysis?",
        "Comments",
    ]

    mode_label = "Scrolling video (MP4)" if viz_mode == "mp4" else "Middle-slice PNG"
    st.info(f"📋 Review the **{mode_label}** below and complete the form for each subject.")

    # --- One-time download + render pass ------------------------------------
    if not st.session_state.data_prepared:
        project_labels = st.session_state.df_outliers["project"].unique()
        if len(project_labels) == 1:
            project = fw.projects.find_first(f"label={project_labels[0]}").reload()

            reviewed, _, old_path = check_previous_reviews(project, st.session_state.username)
            if reviewed:
                st.warning("Previous ratings found — skipping already-reviewed subjects.")
                prev_df     = pd.read_csv(old_path)
                rated_pairs = prev_df[["subject", "session"]].apply(tuple, axis=1).tolist()
                st.session_state.df_outliers["_ss"] = (
                    st.session_state.df_outliers[["subject", "session"]].apply(tuple, axis=1)
                )
                st.session_state.df_outliers = (
                    st.session_state.df_outliers[
                        ~st.session_state.df_outliers["_ss"].isin(rated_pairs)
                    ].drop(columns=["_ss"])
                )
                st.write(f"Remaining: **{st.session_state.df_outliers.shape[0]}** subjects")

            asys_id_cols = get_analysis_id_columns(
                segmentation_tool,
                st.session_state.df_outliers.columns,
            )
            if not asys_id_cols:
                analysis_like_cols = [
                    c for c in st.session_state.df_outliers.columns
                    if "analysis_id" in c.lower()
                ]
                st.error(
                    "Missing analysis-id columns for selected segmentation tool. "
                    f"Expected one of: {', '.join(get_analysis_id_columns(segmentation_tool))}. "
                    f"Found analysis-id-like columns: {', '.join(analysis_like_cols) if analysis_like_cols else 'none'}"
                )
                st.stop()
            st.session_state.df_outliers.dropna(subset=asys_id_cols, how="all", inplace=True)

            rows         = [r for _, r in st.session_state.df_outliers.iterrows()]
            total        = len(rows)
            prog_bar     = st.progress(0)
            status_txt   = st.empty()
            log_box      = st.empty()
            results      = []
            download_dir = os.path.join(Path(__file__).parent, "..", "data")

            with st.spinner(f"Downloading and rendering {mode_label} for {total} subjects…"):
                with ThreadPoolExecutor(max_workers=min(6, total)) as executor:
                    futures = [
                        executor.submit(
                            process_subject_row,
                            row, segmentation_tool, segmentation_suffix,
                            download_dir, fw, st.session_state.api_key,
                            viz_mode,
                        )
                        for row in rows
                    ]
                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    results.append(result)
                    prog_bar.progress(i / total)
                    status_txt.text(f"Processed {i}/{total}")
                    log_box.text("\n".join(results[-10:]))

            st.session_state.data_prepared = True

    # --- Per-subject QC UI --------------------------------------------------
    df = st.session_state.df_outliers
    n  = len(df.index)
    st.progress((st.session_state.row + 1) / n,
                text=f"Subject {st.session_state.row + 1} of {n}")

    current_row       = df.iloc[st.session_state.row]
    media_placeholder = st.empty()
    qc_subject(current_row, segmentation_tool, metrics, media_placeholder)