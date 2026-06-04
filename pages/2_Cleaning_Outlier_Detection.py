#Impute age if it's missing, using the shared utils
import pandas as pd
import numpy as np
import streamlit as st
import numpy as np

# from shared.utils.curate_output import demo
# from statannot import add_stat_annotation  
import plotly.graph_objects as go
import plotly.subplots as subplots
import datetime
from collections import Counter

import json
import flywheel

from pathlib import Path
import os
import logging
import datetime
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import seaborn as sns
import yaml

pd.set_option('display.max_columns', None)

st.title("🧹 Cleaning / Outlier Detection")
st.write("Tools for cleaning and detecting outliers.")

work_dir = Path(__file__).parent

# ---------------------------------------------------------------------------
# Per-tool input prefix configuration.
# Maps each segmentation tool to the input prefixes it supports.
# Tools with only one prefix skip the intersection step (single-input tools
# are flagged based on that one input alone).
# ---------------------------------------------------------------------------
TOOL_INPUT_PREFIXES = {}

INPUT_PREFIXES = ["GAMBAS", "MRR"]
TOOL_PREFIX_MAP = {}
CFG_DICT = {}
CONFIG = {}
def covariance_difference(data):
    """
    Compute the difference between each point and the covariate prediction.

    Parameters:
    - data: DataFrame with standardized values.

    Returns:
    - DataFrame with differences between actual z-score and predicted z-score.
    """

    # Need at least 2 rows and 2 columns for meaningful covariance-based adjustment
    if data.shape[0] < 2 or data.shape[1] < 2:
        return data.copy()
    
    # Drop columns that are entirely NaN
    data = data.dropna(axis=1, how="all")
    if data.shape[0] < 2 or data.shape[1] < 2:
        return data.copy()
    
    newdata = np.array([]).reshape(0, data.shape[0])

    for i in range(len(data.T)):
        copy = data.copy()

        # If after dropping one column there are no predictors left, skip
        if copy.shape[1] < 2:
            return data.copy()

        cov_matrix_i = np.cov(copy.T)

        # Protect against scalar covariance outputs
        if np.ndim(cov_matrix_i) < 2:
            return data.copy()

        cov_column_i = cov_matrix_i.T[i, :]
        cov_column_i = np.delete(cov_column_i, i).reshape(-1, 1)
        copy = copy.drop(copy.columns[i], axis=1)

        if copy.shape[1] == 0:
            return data.copy()

        newdata = np.vstack([
            newdata,
            np.dot(copy.fillna(0), cov_column_i).squeeze() / copy.shape[1]
        ])

    return data - newdata.T


def threshold_outlier_detection(data, skip_covariance=False, thresholds=None):
    
    """
    Detect outliers using covariance or z-score method.

    Parameters:
    - data: DataFrame with standardized values.
    - skip_covariance: if True, use raw z-scores instead of covariance-adjusted values.
    - thresholds: dict mapping column name → threshold value.

    Returns:
    - DataFrame with outlier counts and per-column outlier flags.
    """
    thresholds = {k: float(v) for k, v in (thresholds or {}).items()} or {}
    
   

    method = "cov"
    if not skip_covariance:
        cov_differences = covariance_difference(data)
    else:
        cov_differences = data.copy()
        method = "zscore"

    for key, value in thresholds.items():
        print(key, type(value))
        print(cov_differences[key].map(type).value_counts().head())

    outlier_header = "n_roi_outliers_" + method

    outliers_df = pd.DataFrame()
    for key, value in thresholds.items():
        if key not in cov_differences.columns:
            raise ValueError(f"Key '{key}' not found in the DataFrame columns.")

        if outlier_header not in cov_differences.columns:
            cov_differences[outlier_header] = cov_differences[key].apply(
                lambda x: 1 if x > value or x < -value else 0
            )
            #outliers_df = cov_differences.map(lambda x: 1 if x > value or x < -value else 0)
            outliers_df[f"{key}_outlier_{method}"] = cov_differences[key].apply(
                        lambda x: 1 if x > value or x < -value else 0
                    )
        else:
            outliers = cov_differences[key].apply(lambda x: 1 if x > value or x < -value else 0)
            # outliers_df = cov_differences.map(lambda x: 1 if x > value or x < -value else 0)
            outliers_df[f"{key}_outlier_{method}"] = cov_differences[key].apply(
                    lambda x: 1 if x > value or x < -value else 0
                )
            cov_differences[outlier_header] = cov_differences[outlier_header] + outliers

    # outliers_df.rename(
    #     {col: f"{col}_outlier_{method}" for col in outliers_df.columns if col in list(thresholds.keys())},
    #     inplace=True,
    #     axis=1,
    # )
    outliers_df = outliers_df.loc[:, ~outliers_df.columns.duplicated()]
    outliers_df_unique = outliers_df.loc[:, ~outliers_df.columns.isin(cov_differences.columns)]
    cov_differences = pd.concat([cov_differences, outliers_df_unique], axis=1)

    return cov_differences.reset_index(drop=True)


def compute_cost_prefix_sums(x):
    """
    Precompute prefix sums for efficient cost calculation.
    
    Args:
        x: array of values
    
    Returns:
        prefix_sum: cumulative sum of x
        prefix_sum_sq: cumulative sum of x squared
    """
    x = np.array(x, dtype=float)
    prefix_sum = np.cumsum(x)
    prefix_sum_sq = np.cumsum(x**2)
    return prefix_sum, prefix_sum_sq

def interval_cost(prefix_sum, prefix_sum_sq, i, j, alpha=0.0):
    """
    Calculate cost of binning elements x[i:j] together.
    Cost = variance + penalty for small bins.
    
    Args:
        prefix_sum, prefix_sum_sq: precomputed prefix sums
        i, j: bin range (i inclusive, j exclusive)
        alpha: penalty coefficient for small bins
    
    Returns:
        cost: total cost for this bin
    """
    n = j - i
    if n <= 0:
        return np.inf

    s = prefix_sum[j-1] - (prefix_sum[i-1] if i > 0 else 0)
    s2 = prefix_sum_sq[j-1] - (prefix_sum_sq[i-1] if i > 0 else 0)
    mean = s / n

    # Variance-based cost
    base_cost = s2 - 2 * mean * s + n * mean**2
    penalty = alpha / n if alpha > 0 else 0.0

    return base_cost + penalty


def optimal_binning_min_size(x, k, min_bin_size, alpha=0.0, beta=25, gamma=50):
    """
    Find optimal k bins for data x using dynamic programming.
    
    Args:
        x: data to bin
        k: number of bins
        min_bin_size: minimum elements per bin
        alpha: small bin penalty
        beta: width penalty coefficient
        gamma: boundary penalty coefficient
    
    Returns:
        boundaries: list of (start_idx, end_idx) tuples
        cost: total optimization cost
        sorted_ages: sorted input data
        bins: list of (min_val, max_val) tuples per bin
    """
    x = sorted(x)
    n = len(x)
    if n == 0:
        raise ValueError("Input data is empty.")
        st.warning("Input data is empty. Returning no bins.")
    
    

    if n < k * min_bin_size:
        raise ValueError(
            f"Not enough samples ({n}) for {k} bins with min size {min_bin_size}"
        )

    prefix_sum, prefix_sum_sq = compute_cost_prefix_sums(x)
    
    # DP tables: dp[j] = min cost to bin first j elements
    dp_prev = np.full(n + 1, np.inf)
    dp_curr = np.full(n + 1, np.inf)
    backtrack = np.full((k + 1, n + 1), -1, dtype=int)
    dp_prev[0] = 0.0

    # Fill DP table for each bin count
    for bins in range(1, k + 1):
        dp_curr[:] = np.inf

        for j in range(1, n + 1):
            # Valid range for previous bin endpoint
            i_min = max(bins - 1, j - (n - (k - bins) * min_bin_size))
            i_max = j - min_bin_size
            if i_max < i_min:
                continue

            for i in range(i_min, i_max + 1):
                prev_cost = dp_prev[i]
                if not np.isfinite(prev_cost):
                    continue

                # Width penalty: penalize narrow bins
                if i > 0:
                    bin_width = x[j-1] - x[i]
                    width_penalty = beta / max(bin_width, 1e-9)
                else:
                    width_penalty = 0.0

                cost = (
                    prev_cost
                    + interval_cost(prefix_sum, prefix_sum_sq, i, j, alpha)
                    + width_penalty
                    + gamma * (i - 1)
                )

                if cost < dp_curr[j]:
                    dp_curr[j] = cost
                    backtrack[bins, j] = i

        dp_prev, dp_curr = dp_curr, dp_prev

    if not np.isfinite(dp_prev[n]):
        raise RuntimeError("No feasible binning found with given constraints.")

    # Reconstruct bin boundaries via backtracking
    boundaries = []
    curr = n
    for bins in range(k, 0, -1):
        prev = backtrack[bins, curr]
        if prev < 0:
            raise RuntimeError("Backtracking failed; constraints likely inconsistent.")
        boundaries.append((prev, curr))
        curr = prev

    boundaries.reverse()
    bins_list = [(x[i], x[j-1]) for i, j in boundaries]
    return boundaries, dp_prev[n], x, bins_list


def find_optimal_number_of_bins(x, min_k, max_k, min_bin_size, alpha=0.0, beta=25, gamma=50):
    """
    Search for optimal number of bins in range [min_k, max_k].
    
    Returns:
        best_boundaries, best_cost, best_sorted, best_bins
    """
    best_k = None
    best_cost = np.inf
    best_boundaries = None
    best_sorted = None
    best_bins = None

    for k in range(min_k, max_k + 1):
        try:
            _, cost, _, _ = optimal_binning_min_size(x, k, min_bin_size, alpha=alpha, beta=beta, gamma=gamma)
            if best_k is None or cost < best_cost:
                best_k = k
                best_boundaries, _, best_sorted, best_bins = optimal_binning_min_size(x, k, min_bin_size, alpha=alpha, beta=beta, gamma=gamma)
                best_cost = cost
        except (ValueError, RuntimeError):
            pass
    
    return best_boundaries, best_cost, best_sorted, best_bins

def plot_bins(df, bin_column, age_column):
    """
    Histogram plot showing binning of the age column.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for bin in df[bin_column].unique():
        if bin is np.nan:
            continue
        bin_data = df[df[bin_column] == bin][age_column]
        ax.hist(bin_data, bins=20, alpha=0.5, label=f"Bin {bin}")
    ax.set_xlabel("Age in Months")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution by Bin")
    plt.tight_layout()

    return fig



def create_age_bins(df, column_name, min_k = 2, max_k = 10, alpha=0, beta=10, gamma=3):
    """
    Create binned age column in dataframe using optimal binning.
    """
    vals = df[column_name].values
    vals = vals[~np.isnan(vals)]
    min_bin_size = len(vals) // (max_k * 2)
    
    boundaries, _, _, bins = find_optimal_number_of_bins(vals, min_k, max_k, min_bin_size, alpha=alpha, beta=beta, gamma=gamma)
    if boundaries is None:
        raise ValueError("Failed to find optimal bins for age. Please check the age distribution and try adjusting parameters.")
    df["binned_age"] = pd.cut(df[column_name], bins=[-np.inf] + [b[1] for b in bins] + [np.inf], labels=False)
    return df

def outlier_detection(
    df: pd.DataFrame,
    age_column: str,
    volumetric_columns: list,
    misc_columns: list,
    cov_thresholds=None,
    zscore_thresholds=None,
) -> pd.DataFrame:
    """
    Detect outliers in a DataFrame using covariance and z-score methods.

    Parameters:
    - df: input DataFrame.
    - age_column: column to group by for per-age-group normalisation.
    - volumetric_columns: columns to run outlier detection on.
    - misc_columns: additional columns to carry through into the output.
    - cov_thresholds: thresholds for covariance-based detection.
    - zscore_thresholds: thresholds for z-score-based detection.

    Returns:
    - (df_flagged, outliers_df) tuple.
    """

    df = df.copy()
    outliers = pd.DataFrame()

    df[volumetric_columns] = df[volumetric_columns].replace(
        ["NA", "N/A", "na", "nan", ""], np.nan
    )
    df[volumetric_columns] = df[volumetric_columns].apply(pd.to_numeric, errors="coerce")

    cov_thresholds = cov_thresholds or {}
    zscore_thresholds = zscore_thresholds or {}

    df = df.copy()
    outliers = pd.DataFrame()

    missing_cols = [col for col in volumetric_columns if col not in df.columns]
    if missing_cols:
        st.error(
            f"The following required volumetric columns are missing from the data: "
            f"{', '.join(missing_cols)}. Please check your input data."
        )
        return df, outliers
    
    # Attempt to create age bins, but if it fails, proceed with original age column
    try:
        df = create_age_bins(df, age_column)
        bin_column = "binned_age"
        st.write("Age binning visualization:")
        fig = plot_bins(df, bin_column, age_column)
        st.pyplot(fig)
    except ValueError as e:
        st.warning(f"Error occurred while creating age bins: {e}")
        st.write("Proceeding with original age column without binning.")
        bin_column = age_column  # fallback to original age column if binning fails

    for age in df[bin_column].unique():
        age_df = df[df[bin_column] == age]
        outliers_grouped = pd.DataFrame()
        if not age_df.empty:
            if len(age_df) < 3:
                continue
            z_scores = (age_df[volumetric_columns] - age_df[volumetric_columns].mean()) / age_df[volumetric_columns].std()

            
            outliers_grouped = threshold_outlier_detection(z_scores, thresholds=cov_thresholds)
            zscore_outliers = threshold_outlier_detection(z_scores, skip_covariance=True, thresholds=zscore_thresholds)

            zscore_columns = [col for col in zscore_outliers.columns if col.endswith("_zscore")]
            for col in zscore_columns:
                outliers_grouped[col] = zscore_outliers[col].values

            for col in misc_columns:
                if col in age_df.columns:
                    outliers_grouped[col] = age_df[col].values
                else:
                    outliers_grouped[col] = np.nan

            #If there are more than 2 outliers in either method, flag the session as an outlier
            outliers_grouped = outliers_grouped[
                    (outliers_grouped["n_roi_outliers_cov"] >= 2) | (outliers_grouped["n_roi_outliers_zscore"] >= 2)
                ]

        if not outliers.empty:
            outliers = pd.concat([outliers, outliers_grouped], ignore_index=True)
            first_cols = ["CohortName", "StudyID", "studyTimepoint", "age_in_months"]
            if "input_gear_v" in df.columns:
                first_cols.append("input_gear_v")
            other_cols = [col for col in outliers.columns if col not in first_cols]
            outliers = outliers[first_cols + other_cols]
        else:
            print("Outliers is empty....", age)
            outliers = outliers_grouped

    if outliers.empty:
        st.warning("No outliers detected based on the provided thresholds.")
        return df, outliers

    outliers["is_outlier"] = True
    #tag_only = outliers[["StudyID", "is_outlier"]].drop_duplicates()
    #df = df.copy().merge(tag_only, how="left", on="StudyID")

    tag_only = outliers[["StudyID", "studyTimepoint", "is_outlier"]].drop_duplicates()
    df = df.copy().merge(tag_only, how="left", on=["StudyID", "studyTimepoint"])
    
    df["is_outlier"] = df["is_outlier"].fillna(0).astype(bool)

    first_cols = ["CohortName", "StudyID", "studyTimepoint", "is_outlier", "n_roi_outliers_zscore", "n_roi_outliers_cov"]
    if "input_gear_v" in df.columns:
        first_cols.append("input_gear_v")
    other_cols = [col for col in outliers.columns if col not in first_cols]
    outliers = outliers[first_cols + other_cols]

    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    outliers.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    return df, outliers


def plot_outlier_trend(outliers_df, keyword):
    cov_cols = [
        col for col in outliers_df.columns if col.endswith("_outlier_cov") and not col.startswith("n_roi_")
    ]
    zscore_cols = [
        col for col in outliers_df.columns if col.endswith("_outlier_zscore") and not col.startswith("n_roi_")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    if cov_cols:
        cov_counts = outliers_df[cov_cols].sum(axis=0).sort_values(ascending=True)
        sns.barplot(
            y=cov_counts.index,
            x=cov_counts.values,
            hue=cov_counts.index,
            palette="Blues_d",
            ax=axes[0],
            legend=False
        )
        axes[0].set_title("Outlier Frequency by ROI (Covariance)")
        axes[0].set_xlabel("Number of Outliers")
        axes[0].set_ylabel("ROI")
    else:
        axes[0].text(0.5, 0.5, "No covariance outlier columns", ha="center", va="center")
        axes[0].set_title("Outlier Frequency by ROI (Covariance)")
        axes[0].axis("off")

    if zscore_cols:
        zscore_counts = outliers_df[zscore_cols].sum(axis=0).sort_values(ascending=True)
        sns.barplot(
            y=zscore_counts.index,
            x=zscore_counts.values,
            palette="Reds_d",
            ax=axes[1]
        )
        axes[1].set_title("Outlier Frequency by ROI (Z-score)")
        axes[1].set_xlabel("Number of Outliers")
        axes[1].set_ylabel("ROI")
    else:
        axes[1].text(0.5, 0.5, "No z-score outlier columns", ha="center", va="center")
        axes[1].set_title("Outlier Frequency by ROI (Z-score)")
        axes[1].axis("off")

    plt.suptitle(keyword, fontsize=16)
    plt.tight_layout()
    #return fig

def find_amygdala_columns(columns):
    #Candidates should be any columns that contain 'amygdala'
    candidates = [col for col in columns if "amygdala" in col.lower()]
    return candidates


def cleaning_procedure(df_outlier_flag, MM_vols, RA_vols):

    st.info(f"Original size .. {df_outlier_flag.shape}")
    dup_keys = CONFIG["duplicate_keys"]
    dup_keys = [c for c in dup_keys if c in df_outlier_flag.columns]
    # dup_keys = ["CohortName", "StudyID", "studyTimepoint", "childTimepointAge_months", "MRR_acquisition", "GAMBAS_acquisition"]
    #volume_cols = [c for c in df_outlier_flag.columns if c.startswith("mm") or c.startswith("ra")]

    tool_stems = list(TOOL_PREFIX_MAP.values())
    volume_cols = [
        c for c in df_outlier_flag.columns
        if any(
            c.startswith(f"{stem}_") or c.startswith(f"GAMBAS_{stem}_") or c.startswith(f"MRR_{stem}_")
            for stem in tool_stems
        )
    ]

    # is_outlier is the cross-tool intersection already computed in process_outliers.
    if "is_outlier" not in df_outlier_flag.columns:
        df_outlier_flag["is_outlier"] = False
    df_outlier_flag["is_outlier"] = df_outlier_flag["is_outlier"].astype(bool)

    n_outliers = df_outlier_flag["is_outlier"].sum()
    st.write(f"1. Removing {n_outliers} sessions flagged as outliers (cross-tool intersection)")
    df_filtered = df_outlier_flag[~df_outlier_flag["is_outlier"]].copy()

    df_filtered["is_duplicate"] = df_filtered.duplicated(subset=dup_keys, keep=False)
    st.write("N duplicates (project-subject-session-age-acquisition):", df_filtered[df_filtered["is_duplicate"] == True].shape)

    st.write("2. Filter out rows where amygdala volumes are below threshold (if amygdala segmentation is present)")
    amygdala_candidates = find_amygdala_columns(df_filtered.columns)
    #For all candidates, do the thresholding and keep only those that pass. If no candidates, skip this step.
    if amygdala_candidates:
        st.write(f"Amygdala candidates found: {', '.join(amygdala_candidates)}")
        #each candidate should be checked for the threshold, and a new column should be created for each candidate with the pass/fail result. Then we can filter to keep only rows that pass all candidates.
        df_filtered["pass_amygdala"] = True
        for candidate in amygdala_candidates:
            df_filtered["pass_amygdala"] &= df_filtered[candidate] >= 250
        df_filtered = df_filtered[df_filtered["pass_amygdala"]]
        st.write("N rows after amygdala threshold filter:", df_filtered.shape)

    else:
        st.write("No amygdala candidates found. Skipping amygdala threshold filter.")

    st.write("3. From those that passed the conditions, pick the one with fewest zeros in volumetric columns (failed segmentations)")
    candidates = df_filtered.copy()
    candidates["zero_count"] = (candidates[volume_cols] == 0).sum(axis=1)
    candidates = candidates.sort_values(dup_keys + ["zero_count"])
    candidates_cleaned = candidates.drop_duplicates(subset=dup_keys, keep="first").copy()
    df_final = pd.concat([candidates_cleaned], ignore_index=True)

    st.write("After cleaning duplicate rows ..", df_final.shape)
    return df_final


# ---------------------------------------------------------------------------
# NEW HELPER — run outlier detection for one input prefix and return the set
# of outlier session keys.
# ---------------------------------------------------------------------------
def _detect_outliers_for_input(
    df_filt: pd.DataFrame,
    base_volumetric_cols: list,
    outlier_thresholds: dict,
    columns_to_keep_base: list,
):
    available_vols = [c for c in base_volumetric_cols if c in df_filt.columns]
    available_thresholds = {
        k: v for k, v in outlier_thresholds.items() if k in df_filt.columns
    }

    missing = [c for c in base_volumetric_cols if c not in df_filt.columns]
    if missing:
        st.warning(
            f"{len(missing)} volumetric columns missing "
            f"(e.g. {missing[0]}). Proceeding with {len(available_vols)} available columns."
        )

    if len(available_vols) < 2:
        st.warning("Fewer than 2 volumetric columns available. Skipping this tool.")
        return set(), set(), pd.DataFrame()

    columns_to_keep = [c for c in (columns_to_keep_base + available_vols) if c in df_filt.columns]
    df_sub = df_filt[df_filt[available_vols].notna().any(axis=1)].copy()

    if df_sub.empty:
        return set(), set(), pd.DataFrame()

    evaluated_sessions = set(zip(df_sub["StudyID"], df_sub["studyTimepoint"]))

    _, outliers_df = outlier_detection(
        df_sub[columns_to_keep],
        age_column="age_in_months",
        volumetric_columns=available_vols,
        misc_columns=columns_to_keep,
        cov_thresholds=available_thresholds,
        zscore_thresholds=available_thresholds,
    )

    if outliers_df.empty:
        return set(), evaluated_sessions, pd.DataFrame()

    outlier_sessions = set(zip(outliers_df["StudyID"], outliers_df["studyTimepoint"]))
    #st.info(f"  → {len(outlier_sessions)} outlier sessions detected")
    return outlier_sessions, evaluated_sessions, outliers_df


@st.cache_data
def process_outliers(df, df_demo, keywords, group_str="all"):
    st.write(f"### Group: {group_str}")

    # ------------------------------------------------------------------
    # Merge demographics once
    # ------------------------------------------------------------------
    if df_demo is not None:
        merge_keys = CONFIG["demographic_merge_keys"]
        merge_keys = [c for c in merge_keys if c in df.columns and c in df_demo.columns]

        df_merged = df.merge(
            df_demo,
            on=merge_keys,
            how="left",
            suffixes=("", "_from_demo"),
        )
    else:
        st.warning("No additional demographic file uploaded. Missing ages will remain as NaN.")
        df_merged = df.copy()
        df_merged["age_from_demo"] = np.nan

    df_merged["childTimepointAge_months"] = df_merged["childTimepointAge_months"].combine_first(
        df_merged.get("age_from_demo")
    )
    df_merged = df_merged.drop(columns=["age_from_demo"], errors="ignore")
    df_merged["age_in_months"] = df_merged["childTimepointAge_months"].apply(
        lambda x: int(np.ceil(x)) if pd.notnull(x) else np.nan
    )

    df_out = df_merged.copy()

    columns_to_keep_base = [
        "CohortName", "StudyID", "studyTimepoint", "age_in_months",
        "childBiologicalSex", "MRR_acquisition", "GAMBAS_acquisition", "session_qc",
    ]
    columns_to_keep_base = [c for c in columns_to_keep_base if c in df_merged.columns]

    if "input_gear_v" in df_merged.columns:
        columns_to_keep_base.append("input_gear_v")

    # Store outlier sessions per tool, then assign columns after the loop
    tool_outlier_sessions = {}

    for segmentation_tool in keywords:
        st.write(f"Tool: {segmentation_tool}")
        st.info(f"Running outlier detection for {segmentation_tool}...")

        outlier_thresholds = CFG_DICT[segmentation_tool]["thresholds"]
        base_volumetric_cols = CFG_DICT[segmentation_tool]["volumetric_cols"]

        outlier_sessions, evaluated_sessions, outliers_df = _detect_outliers_for_input(
            df_merged,
            base_volumetric_cols,
            outlier_thresholds,
            columns_to_keep_base,
        )

        tool_outlier_sessions[segmentation_tool] = outlier_sessions
        n_confirmed = len(outlier_sessions)

        st.info(f"Confirmed outliers for {segmentation_tool}: {n_confirmed}")

        if not outliers_df.empty:
            plot_df = outliers_df.copy()
            plot_df = plot_df[
                plot_df.apply(
                    lambda row: (row["StudyID"], row["studyTimepoint"]) in outlier_sessions,
                    axis=1,
                )
            ].copy()

            if not plot_df.empty:
                fig = plot_outlier_trend(plot_df, segmentation_tool)
                st.pyplot(fig, use_container_width=False)

    # ------------------------------------------------------------------
    # Add one flag column per selected tool
    # ------------------------------------------------------------------
    for segmentation_tool in keywords:
        outlier_flag_col = f"is_{TOOL_PREFIX_MAP[segmentation_tool]}_outlier"
        session_set = tool_outlier_sessions.get(segmentation_tool, set())

        df_out[outlier_flag_col] = df_out.apply(
            lambda row: (row["StudyID"], row["studyTimepoint"]) in session_set,
            axis=1,
        )

        st.write(f"  → {outlier_flag_col}: {int(df_out[outlier_flag_col].sum())} flagged")

    active_flag_cols = [
        f"is_{TOOL_PREFIX_MAP[tool]}_outlier"
        for tool in keywords
        if f"is_{TOOL_PREFIX_MAP[tool]}_outlier" in df_out.columns
    ]

    if active_flag_cols:
        df_out["any_outlier"] = df_out[active_flag_cols].any(axis=1)
    else:
        df_out["any_outlier"] = False

    all_outliers_df = df_out[df_out["any_outlier"]].copy()

    os.makedirs(os.path.join(work_dir.parent, "data"), exist_ok=True)
    final_outliers_path = os.path.join(work_dir.parent, "data", f"{group_str}_all_outliers.csv")
    all_outliers_df.to_csv(final_outliers_path, index=False)
    st.write("Last 10 columns of outliers dataframe:", all_outliers_df.iloc[:, -10:].head())
    return df_out, final_outliers_path, all_outliers_df


def main():
    global TOOL_PREFIX_MAP, CFG_DICT, CONFIG, TOOL_INPUT_PREFIXES
 
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    with open(os.path.join(work_dir.parent, "utils", "thresholds.yml"), "r") as f:
        CFG_DICT = yaml.load(f, Loader=yaml.SafeLoader)
        TOOL_PREFIX_MAP = {
            tool: cfg["prefix"]
            for tool, cfg in CFG_DICT.items()
        }

        TOOL_INPUT_PREFIXES = {
            tool: cfg["inputs"]
            for tool, cfg in CFG_DICT.items()
        }
    with open(os.path.join(work_dir.parent, "utils", "config.yml"), "r") as f:
        CONFIG = yaml.load(f, Loader=yaml.SafeLoader)

    #stratify = st.checkbox("Stratify cleaning by project?", value=True)
    #Input box to enter column to stratify by (default "CohortName")
    stratify_col = st.text_input("Enter column name to stratify by (default: 'project')", value="CohortName")
    unique_groups = []

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head())

        if stratify_col not in df.columns:
            st.error(f"Column '{stratify_col}' not found in uploaded CSV.")
            return

        if stratify_col:
            unique_groups = df[stratify_col].unique()
            print(unique_groups)
            st.info(f"Stratifying cleaning by {stratify_col}. Projects found: {', '.join(map(str, unique_groups))}")
        else:
            st.info(f"Not stratifying cleaning by project. Projects found: {', '.join(map(str, unique_groups))}")

    # segmentation_tool = st.radio(
    #     "Upload derivatives from:",
    #     ["recon-all-clinical", "minimorph", "supersynth", "all"],
    # )
    #Check box instead of radio button
    st.text("Choose which segmentation tools to include in outlier detection (intersection across selected tools):")
    
    # recon_all_clinical = st.checkbox("Include recon-all-clinical (GAMBAS and MRR)", value=True)
    # minimorph = st.checkbox("Include minimorph (MRR only)", value=True)
    # supersynth = st.checkbox("Include supersynth (MRR only)", value=False)

    # segmentation_tool = []
    # if supersynth:
    #     segmentation_tool.append("supersynth")
    # if recon_all_clinical:
    #     segmentation_tool.append("recon-all-clinical")
    # if minimorph:
    #     segmentation_tool.append("minimorph")

    # keywords = []
    # for tool in segmentation_tool:
    #     if tool in TOOL_INPUT_PREFIXES:
    #         keywords.append(tool)

    keywords = []

    for tool_name, cfg in CFG_DICT.items():
        label = cfg.get("label", tool_name)
        default = cfg.get("default_selected", False)

        if st.checkbox(label, value=default, key=f"{tool_name}"):
            keywords.append(tool_name)

    # if segmentation_tool == "all":
    #     keywords = list(TOOL_INPUT_PREFIXES.keys())
    # else:
    #     keywords = [segmentation_tool]

    uploaded_demo = st.file_uploader("Upload demographic CSV file (optional)", type=["csv"])
    df_demo = None
    final_outliers_path  = ""
    if uploaded_demo is not None:
        df_demo = pd.read_csv(uploaded_demo)
        st.success("Demographic file uploaded successfully!")
        st.dataframe(df_demo.head())

        required_columns = {"CohortName", "StudyID", "studyTimepoint", "childTimepointAge_months"}
        if not required_columns.issubset(df_demo.columns):
            st.error(
                f"Demographic file must contain the following columns (name-sensitive): "
                f"{', '.join(required_columns)}"
            )


        
    if st.button("Detect Outliers") and uploaded_file is not None:
        st.session_state["processed_df"] = pd.DataFrame()
        all_outlier_dfs = []
        final_outliers_path = ""

        if stratify_col and len(unique_groups) > 1:
            progress = st.progress(0)
            for group in unique_groups:
                st.info(f"Processing group: {group}")
                df_group = df[df[stratify_col] == group]
                group_str = str(group)

                processed, group_outliers_path, group_outliers = process_outliers(
                    df_group, df_demo, keywords, group_str
                )

                st.session_state["processed_df"] = pd.concat(
                    [st.session_state["processed_df"], processed], ignore_index=True
                )
                all_outlier_dfs.append(group_outliers)

                progress.progress((np.where(unique_groups == group)[0][0] + 1) / len(unique_groups))

            final_outliers_df = pd.concat(all_outlier_dfs, ignore_index=True) if all_outlier_dfs else pd.DataFrame()
            final_outliers_path = os.path.join(work_dir.parent, "data", "all_groups_all_outliers.csv")
            final_outliers_df.to_csv(final_outliers_path, index=False)

        else:
            st.info("Processing all groups together...")
            processed, final_outliers_path, outliers = process_outliers(df, df_demo, keywords)
            st.session_state["processed_df"] = processed

        os.makedirs(os.path.join(work_dir.parent, "data"), exist_ok=True)
        processed_path = os.path.join(work_dir.parent, "data", "allData_outlierFlagged.csv")
        st.session_state["processed_df"].to_csv(processed_path, index=False)
        st.session_state["processed_path"] = processed_path
        st.session_state["final_outlier_paths"] = final_outliers_path
        

    if st.session_state.get("processed_path") and os.path.exists(st.session_state["processed_path"]):
        with open(st.session_state["processed_path"], "rb") as f:
            st.download_button(
                label="Download annotated dataset",
                data=f,
                file_name="allData_outlierFlagged.csv",
                key="dl_processed",
            )

    if st.session_state.get("final_outlier_paths") and os.path.exists(st.session_state["final_outlier_paths"]):
        with open(st.session_state["final_outlier_paths"], "rb") as f:
            st.download_button(
                label="Download all outliers",
                data=f,
                file_name=os.path.basename(st.session_state["final_outlier_paths"]),
                key="dl_all_outliers",
            )
    # ------------------------------------------------------------------
    # Step 2: Clean — same logic as before (intersect across tools)
    # ------------------------------------------------------------------
    st.write("### Clean Outliers")
    st.write("Click the button below to download a clean dataset with outliers removed based on the following criteria:")
    st.markdown(
        """
    - Condition 1: RA outlier only, and all MM_vols are NA
    - Condition 2: MM outlier only, MM_vols populated, all RA_vols NA
    - Condition 3: Both MM and RA are outliers
    - Condition 4: RA, MM and SS are outliers
    - From those that pass amygdala threshold (if RA_left_amygdala and RA_right_amygdala are present),
      pick the one with fewest zeros (failed segmentations)
    """
    )

    if st.button("Clean Outliers") and "processed_df" in st.session_state:
        st.info("Cleaning outliers...")
        df_ = st.session_state["processed_df"]

        MM_vols = [col for col in df_.columns if col.startswith("mm_")]
        RA_area = [col for col in df_.columns if col.startswith("ra_") and col.endswith("_area")]
        RA_thickness = [col for col in df_.columns if col.startswith("ra_") and col.endswith("_thickness")]
        RA_vols = [col for col in df_.columns if col.startswith("ra_") and col not in (RA_area + RA_thickness)]

        clean_df = cleaning_procedure(df_, MM_vols, RA_vols)
        st.success("Outliers cleaned.")
        st.write(f"Size of dataframe after cleaning: {clean_df.shape}")

        clean_path = os.path.join(work_dir.parent, "data", "alldata_cleaned.csv")
        cols = clean_df.columns.tolist()
        front_cols = [
            "CohortName", "StudyID", "studyTimepoint", "childTimepointAge_months",
            "childBiologicalSex", "studyTimepoint", "session_qc", "MRR_acquisition","GAMBAS_acquisition",
        ]
        ra_cols = [col for col in cols if col.startswith("ra_")]
        mm_cols = [col for col in cols if col.startswith("mm_")]
        other_cols = [col for col in cols if col not in front_cols + ra_cols + mm_cols]
        new_order = front_cols + ra_cols + mm_cols + other_cols
        clean_df = clean_df[[c for c in new_order if c in clean_df.columns]]
        clean_df.to_csv(clean_path, index=False)

        st.dataframe(clean_df.head())
        if os.path.exists(clean_path):
            with open(clean_path, "rb") as f:
                st.download_button("Download clean CSV", f, file_name=clean_path)


if __name__ == "__main__":
    main()