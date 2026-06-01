import os
import traceback
from pathlib import Path
from datetime import datetime

import flywheel
import numpy as np
import pandas as pd
import pathvalidate as pv
import streamlit as st
import yaml
from packaging import version


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_latest(analyses, input_keyword):
    """Return list containing the most recently created analysis whose first
    input filename contains any of the given keywords."""
    kws = [input_keyword] if isinstance(input_keyword, str) else input_keyword
    candidates = [a for a in analyses if a.inputs and any(kw in a.inputs[0].name for kw in kws)]
    for a in analyses:
        input_name = a.inputs[0].name if a.inputs else 'n/a'
        print(f"    Analysis {a.id} input name: {input_name}")
    return [max(candidates, key=lambda a: a.created)] if candidates else []


def _build_session_df(fw, project, session, analyses_list,
                      fw_session_info, keywords, tool_map,
                      project_path):
    """
    Download files for a pre-filtered list of analyses and
    merge them horizontally into a single DataFrame row.
    Returns an empty DataFrame if nothing matched.
    """
    ses_label = session.label
    sub_label = session.subject.label
    session_df = pd.DataFrame()

    if fw_session_info == "Yes":
        session_tags = session.tags if session.tags else []
        session_df['session_tags'] = ' | '.join(session_tags) if session_tags else 'n/a'
        for key, value in session.info.items():
            session_df[key] = value

    for analysis in analyses_list:
        analysis = analysis.reload()
        gear = analysis.gear_info.name
        volumetric_cols = tool_map.get(gear, [])

        matched_files = [f for f in analysis.files if any(kw in f.name for kw in keywords)]
        print(f"    gear={gear}: {len(matched_files)} file(s) matched")

        for analysis_file in matched_files:
            file = analysis_file.reload()
            download_dir = pv.sanitize_filepath(project_path / sub_label / ses_label, platform='auto')
            download_dir.mkdir(parents=True, exist_ok=True)
            download_path = download_dir / file.name

            print(f"    Downloading {file.name} ...")
            file.download(download_path)

            df = pd.read_csv(download_path)
            df["project"] = project.label
            df["subject"] = sub_label
            df["sex"] = session.info.get('childBiologicalSex', 'n/a')
            df["session"] = ses_label
            df["childTimepointAge_months"] = session.info.get('childTimepointAge_months', df.get("age", "n/a"))

            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.insert(3, 'session_qc', session.tags[-1] if session.tags else 'n/a')
            df["age_source"] = "custom_info"

            # Prefix volumetric columns by gear (your requested casing style)
            if gear == "minimorph":
                df["analysis_id_mm"] = analysis.id
                df["gear_v_minimorph"] = analysis.gear_info.version
                df.rename(columns={c: f"MM_{c}" for c in volumetric_cols if c in df.columns}, inplace=True)

            elif gear == "supersynth":
                df["analysis_id_ss"] = analysis.id
                df["gear_v_supersynth"] = analysis.gear_info.version
                df.rename(columns={c: f"ss_{c}" for c in volumetric_cols if c in df.columns}, inplace=True)

            else:  # recon-all-clinical / recon-any
                df["analysis_id_ra"] = analysis.id
                df["gear_v_recon_all"] = analysis.gear_info.version
                df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
                volumetric_cols_norm = [c.replace(' ', '_').replace('-', '_').lower() for c in volumetric_cols]
                df.rename(columns={c: f"RA_{c}" for c in volumetric_cols_norm if c in df.columns}, inplace=True)

            # Merge horizontally into session_df
            if session_df.empty:
                session_df = df
            else:
                merge_keys = ['subject', 'session']
                new_cols = merge_keys + [c for c in df.columns if c not in session_df.columns and c not in merge_keys]
                session_df = pd.merge(session_df, df[new_cols], on=merge_keys, how='outer')

            # Clean local temp file
            try:
                os.remove(download_path)
            except Exception:
                pass

    session_df.drop(columns=['gear_v', 'age_source', 'template_age'], inplace=True, errors='ignore')
    return session_df


def download_session_data(fw, project, session_id, project_path,
                          segtool, input_source, fw_session_info,
                          keywords, tool_map):
    """
    Returns:
      - input_source == "Both": dict {'mrr': df_or_None, 'gambas': df_or_None}
      - otherwise:              a single DataFrame, or None
    """
    session = fw.get(session_id).reload()
    ses_label = session.label
    sub_label = session.subject.label

    # Collect completed analyses from session + acquisition level
    analyses = [
        a for a in session.analyses
        if a.reload().gear_info is not None
        and a.reload().gear_info.name in segtool
        and a.reload().job.get('state') == 'complete'
    ]

    for acq in session.acquisitions():
        acq = acq.reload()
        acq_analyses = [
            a for a in acq.analyses
            if a.reload().gear_info is not None
            and a.reload().gear_info.name in segtool
            and a.reload().job.get('state') == 'complete'
        ]
        analyses.extend(acq_analyses)

    mrr_analyses = []
    gambas_analyses = []

    for segmentation_tool in segtool:
        tool_analyses = [a for a in analyses if a.gear_info.name == segmentation_tool]

        if input_source in ("MRR", "Both"):
            mrr_analyses.extend(get_latest(tool_analyses, "mrr"))

        if input_source in ("Enhanced (Gambas)", "Both"):
            gambas_analyses.extend(get_latest(tool_analyses, ["gambas", "ResCNN"]))

    try:
        if input_source == "Both":
            print(f"  [{sub_label} / {ses_label}] analyses found={len(analyses)} | MRR={len(mrr_analyses)} | Gambas={len(gambas_analyses)}")
            mrr_df = _build_session_df(fw, project, session, mrr_analyses, fw_session_info, keywords, tool_map, project_path)
            gambas_df = _build_session_df(fw, project, session, gambas_analyses, fw_session_info, keywords, tool_map, project_path)
            return {
                "mrr": mrr_df if not mrr_df.empty else None,
                "gambas": gambas_df if not gambas_df.empty else None
            }

        analyses_filtered = mrr_analyses + gambas_analyses
        print(f"  [{sub_label} / {ses_label}] analyses found={len(analyses)} | filtered={len(analyses_filtered)}")
        session_df = _build_session_df(fw, project, session, analyses_filtered, fw_session_info, keywords, tool_map, project_path)
        return session_df if not session_df.empty else None

    except Exception:
        print(f"EXCEPTION in [{sub_label} / {ses_label}]:\n{traceback.format_exc()}")
        if input_source == "Both":
            return {"mrr": None, "gambas": None}
        return None


def _finalise_project_frames(frames, label_suffix, segtool, project, project_path):
    """Dedup, reorder, save and return path for one frame list."""
    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    for gear_col in ['gear_v_recon_all', 'gear_v_minimorph']:
        if gear_col in combined.columns:
            key_cols = [c for c in ['subject', 'session', 'acquisition'] if c in combined.columns]
            combined = (
                combined
                .sort_values(
                    gear_col,
                    key=lambda s: s.map(lambda v: version.parse(v) if pd.notna(v) else version.parse("0")),
                    ascending=False
                )
                .drop_duplicates(subset=key_cols, keep='first')
            )

    for col in ['scanner_software_v', 'input_gear_v']:
        if col in combined.columns and 'acquisition' in combined.columns:
            cols = combined.columns.tolist()
            cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index(col)))
            combined = combined[cols]

    segtool_str = '-'.join(segtool)
    outname = project.label.replace(' ', '_').replace('(', '').replace(')', '')
    outpath = project_path / f"{outname}-{segtool_str}-{label_suffix}.csv"
    combined.to_csv(outpath, index=False)
    return str(outpath)


def download_derivatives(project_id, segtool, input_source, fw_session_info, keywords, fw, data_dir, tool_map, debug=False):
    project = fw.projects.find_first(f"label={project_id}")
    if project is None:
        st.warning(f"Project '{project_id}' not found.")
        return None

    st.info(f"Project: {project.label}  \nSubjects: {len(project.subjects())}  \nSessions: {len(project.sessions())}")

    project_path = pv.sanitize_filepath(data_dir / project.label, platform='auto')
    project_path.mkdir(parents=True, exist_ok=True)

    sessions = [s.id for s in project.sessions() if not s.subject.label.startswith('137')]
    if debug:
        sessions = np.random.RandomState(seed=42).choice(
            sessions, size=min(20, len(sessions)), replace=False
        )
        sessions = list(sessions)

    progress_s = st.progress(0)
    status = st.empty()

    mrr_frames = []
    gambas_frames = []
    all_frames = []

    for i, session_id in enumerate(sessions, 1):
        try:
            result = download_session_data(
                fw, project, session_id, project_path,
                segtool, input_source, fw_session_info,
                keywords, tool_map
            )
        except Exception:
            st.error(f"Error processing session {session_id}: {traceback.format_exc()}")
            continue

        if input_source == "Both":
            if result["mrr"] is not None:
                mrr_frames.append(result["mrr"])
            if result["gambas"] is not None:
                gambas_frames.append(result["gambas"])
        else:
            if result is not None:
                all_frames.append(result)

        progress_s.progress(i / max(1, len(sessions)))
        status.text(f"Completed {i}/{len(sessions)}")

    if input_source == "Both":
        mrr_path = _finalise_project_frames(mrr_frames, "MRR", segtool, project, project_path)
        gambas_path = _finalise_project_frames(gambas_frames, "Gambas", segtool, project, project_path)
        return (mrr_path, gambas_path)

    label = input_source.replace(' ', '_').replace('(', '').replace(')', '')
    single_path = _finalise_project_frames(all_frames, label, segtool, project, project_path)
    return single_path


def assemble_csv(derivative_paths, label=""):
    frames = []
    for deriv in derivative_paths:
        if deriv is None:
            continue
        frames.append(pd.read_csv(deriv))

    if not frames:
        return None, None

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined.drop(columns=['age', 'sex', 'gear_v'], inplace=True, errors='ignore')

    front_cols = ['project', 'subject', 'session', 'childTimepointAge_months',
                  'childBiologicalSex', 'studyTimepoint', 'session_qc', 'acquisition']
    cols = combined.columns.tolist()

    ra_cols = [c for c in cols if c.startswith('RA_')]
    mm_cols = [c for c in cols if c.startswith('MM_')]
    ss_cols = [c for c in cols if c.startswith('ss_')]

    spoken_for = set(front_cols + ra_cols + mm_cols + ss_cols)
    other_cols = [c for c in cols if c not in spoken_for]
    new_order = front_cols + ra_cols + mm_cols + ss_cols + other_cols

    for col in front_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    combined = combined[new_order]

    unique_projects = combined["project"].dropna().unique()
    project_str = '_'.join(unique_projects) if len(unique_projects) else "unknown_project"
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    label_str = f"_{label}" if label else ""

    out_csv = f"derivatives_summary_{project_str}{label_str}_{time_str}.csv"
    combined.to_csv(out_csv, index=False)
    return combined, out_csv


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.title("📥 Data Download")
st.write("Download and compile derivatives from multiple projects into CSV file(s).")
st.write("Select projects and derivative types from the sidebar, then click **Fetch derivatives**.")

# Session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None

API_KEY = os.getenv("FW_CLI_API_KEY")
if (API_KEY is None or API_KEY == "") and st.session_state.authenticated is False:
    st.warning("Please enter your Flywheel API key in the Home page to continue.")
    st.stop()

fw = flywheel.Client(st.session_state.api_key if st.session_state.authenticated else API_KEY)

# Paths
data_dir = Path(__file__).resolve().parent / "../data/"
data_dir.mkdir(parents=True, exist_ok=True)

# vol_columns.yml pathing (portable)
yml_path = (Path(__file__).resolve().parent / "../utils/vol_columns.yml").resolve()
if not yml_path.exists():
    yml_path = (Path(__file__).resolve().parent / "utils/vol_columns.yml").resolve()

if not yml_path.exists():
    st.error(f"vol_columns.yml not found. Tried:\n- {yml_path}")
    st.stop()

with open(yml_path, "r") as f:
    tool_map = yaml.load(f, Loader=yaml.SafeLoader)

@st.cache_data(ttl=600)
def get_projects():
    return [p.label for p in fw.projects()]

projects = get_projects()

st.sidebar.header("Settings")
project_ids = st.sidebar.multiselect("Select Projects", projects)

minimorph = st.sidebar.checkbox("Minimorph", value=False)
recon_all = st.sidebar.checkbox("Recon-all-clinical", value=False)
supersynth = st.sidebar.checkbox("Supersynth", value=False)
recon_any = st.sidebar.checkbox("Recon-any", value=False)

st.session_state.input_source = st.sidebar.radio(
    "Structural Image Segmented:",
    ["MRR", "Enhanced (Gambas)", "Both"],
    index=0
)

st.session_state.fw_session_info = st.sidebar.radio(
    "Include Flywheel Session Info (tags, custom info) in download?",
    ["No", "Yes"],
    index=0
)

derivative_type = []
keywords = []

if recon_all:
    derivative_type.append("recon-all-clinical")
    st.sidebar.markdown("**Select outputs to download (Recon-all-clinical):**")
    area = st.sidebar.checkbox("Area", value=False)
    thickness = st.sidebar.checkbox("Thickness", value=False)
    volume = st.sidebar.checkbox("Volume", value=True)

    if area:
        keywords.append("area")
    if thickness:
        keywords.append("thickness")
    if volume:
        keywords.append("volume")

    if not (area or thickness or volume):
        st.sidebar.warning("Please select at least one recon-all output type.")

if minimorph:
    derivative_type.append("minimorph")
    keywords.append("volumes")

if supersynth:
    derivative_type.append("supersynth")
    keywords.append("volumes")

if recon_any:
    derivative_type.append("recon-any")

debug = st.sidebar.checkbox("Debug mode (random up to 20 sessions, seed=42)", value=False)

if st.sidebar.button("Fetch derivatives"):
    if not project_ids:
        st.error("Please select at least one project.")
        st.stop()
    if not derivative_type:
        st.error("Please select at least one tool.")
        st.stop()
    if not keywords:
        st.error("Please select at least one keyword/output type.")
        st.stop()

    progress_p = st.progress(0)
    single_paths = []
    mrr_paths = []
    gambas_paths = []

    for i, proj in enumerate(project_ids, 1):
        st.write(f"Fetching {', '.join(derivative_type)} for **{proj}** ...")

        result = download_derivatives(
            project_id=proj,
            segtool=derivative_type,
            input_source=st.session_state.input_source,
            fw_session_info=st.session_state.fw_session_info,
            keywords=keywords,
            fw=fw,
            data_dir=data_dir,
            tool_map=tool_map,
            debug=debug
        )

        if st.session_state.input_source == "Both":
            if result is not None:
                mrr_path, gambas_path = result
                if mrr_path:
                    mrr_paths.append(mrr_path)
                if gambas_path:
                    gambas_paths.append(gambas_path)
        else:
            if result:
                single_paths.append(result)

        progress_p.progress(i / len(project_ids))

    # Final assembly + downloads
    if st.session_state.input_source == "Both":
        if not mrr_paths and not gambas_paths:
            st.error("No derivatives found.")
            st.stop()

        st.subheader("MRR Output")
        if mrr_paths:
            mrr_df, mrr_out = assemble_csv(mrr_paths, label="MRR")
            if mrr_df is not None:
                st.dataframe(mrr_df)
                with open(mrr_out, "rb") as f:
                    st.download_button("Download MRR CSV", f, file_name=mrr_out, key="download_mrr")
        else:
            st.info("No MRR results found.")

        st.subheader("Enhanced (Gambas) Output")
        if gambas_paths:
            gambas_df, gambas_out = assemble_csv(gambas_paths, label="Gambas")
            if gambas_df is not None:
                st.dataframe(gambas_df)
                with open(gambas_out, "rb") as f:
                    st.download_button("Download Gambas CSV", f, file_name=gambas_out, key="download_gambas")
        else:
            st.info("No Gambas results found.")

        st.success("Download complete!")

    else:
        if not single_paths:
            st.error("No derivatives found.")
            st.stop()

        final_df, out_csv = assemble_csv(single_paths)
        st.session_state.df = final_df
        st.success("Download complete!")
        st.dataframe(final_df)

        with open(out_csv, "rb") as f:
            st.download_button("Download CSV", f, file_name=out_csv)