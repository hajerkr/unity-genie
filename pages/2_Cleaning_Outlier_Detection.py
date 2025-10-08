#Impute age if it's missing, using the shared utils
import pandas as pd
import streamlit as st

from shared.utils.curate_output import demo

import streamlit as st
from statannot import add_stat_annotation  
import plotly.graph_objects as go
import plotly.subplots as subplots
import datetime
from collections import Counter

import json

import flywheel
import pandas as pd
import numpy as np
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

def covariance_difference(data):
    """
    Compute the difference between each point and the covariate prediction.

    Parameters:
    - data: DataFrame with standardized values.
    - method: Method to use for outlier detection ('covariance' or 'linear_regression').

    Returns:
    - DataFrame with an differences between actual z-score and predicted z-score.
    """


    #compute covariance matrix
    newdata = np.array([]).reshape(0, data.shape[0])
    

    # compute the "predicted" values for each column based on differences from mean and covariance matrix
    for i in range(len(data.T)):

        copy = data.copy()

        # include only covariance that does not include the i-th column
        cov_matrix_i = np.cov(copy.T)

        #return cov_matrix_i
        cov_column_i = cov_matrix_i.T[i,:]
        cov_column_i = np.delete(cov_column_i, i).reshape(-1,1)



        #drop the i-th column of the data
        copy = copy.drop(copy.columns[i], axis=1)


        # compute the mean of the i-th column based on the covariance with the other columns
        # the i-th column is the dot product of the covariance with the other columns
        #  divided by the number of columns to normalize
        newdata = np.vstack([newdata, np.dot(copy, cov_column_i).squeeze()/copy.shape[1]])

    # return the difference between the original data and the predicted values
    
    return data - newdata.T

def threshold_outlier_detection(data,skip_covariance = False, thresholds: dict = {}):
    """
    Detect outliers using covariance method.
    
    Parameters:
    - data: DataFrame with standardized values.
    - kwargs: Additional parameters for the method.
    
    Returns:
    - DataFrame with outliers detected.
    """
   
    method = "cov"
    if not skip_covariance:
        cov_differences = covariance_difference(data)
    else:
        cov_differences = data.copy()

        method = "zscore"
    
    outlier_header = "n_roi_outliers_"+ method

    #cov_differences.set_index(data.index.values, inplace=True)
    #cov_differences["subject"] = data["subject"].values
    outliers_df = pd.DataFrame() 
    for key, value in thresholds.items():
        if key not in cov_differences.columns:
            raise ValueError(f"Key '{key}' not found in the DataFrame columns.")
        
        if outlier_header not in cov_differences.columns:
            cov_differences[outlier_header] = cov_differences[key].apply(lambda x: 1 if x > value or x < -value else 0)
            outliers_df = cov_differences.map(lambda x: 1 if x > value or x < -value else 0)
            #cov_differences["outliers"] = cov_differences[key].apply(lambda x: 1 if x > value or x < -value else 0)
        else:
            #outliers = cov_differences[key].apply(lambda x: 1 if x > value or x < -value else 0)
            outliers = cov_differences[key].apply(lambda x: 1 if x > value or x < -value else 0)
            outliers_df = cov_differences.map(lambda x: 1 if x > value or x < -value else 0)
            
            cov_differences[outlier_header] = cov_differences[outlier_header] + outliers
        
    # display(outliers_df)
    #concatenate outliers_df and cov_differences
    #add suffix to outliers_df columns
    outliers_df.rename({col: f"{col}_outlier_{method}" for col in outliers_df.columns if col in list(thresholds.keys())}, inplace=True, axis=1)
    #drop dupmicate columns
    outliers_df = outliers_df.loc[:,~outliers_df.columns.duplicated()]


    # Drop duplicate columns from df2 before concat
    outliers_df_unique = outliers_df.loc[:, ~outliers_df.columns.isin(cov_differences.columns)]

    # Concatenate
    cov_differences = pd.concat([cov_differences, outliers_df_unique], axis=1)

    # cov_differences = pd.concat([cov_differences, outliers_df], axis=1)
    # print("Columns of cov_differences", cov_differences.columns)
    # display(cov_differences)
    return cov_differences.reset_index(drop=True)
        

def outlier_detection(df: pd.DataFrame, age_column: str, volumetric_columns: list, misc_columns: list, cov_thresholds: dict = {},zscore_thresholds: dict = {}) -> pd.DataFrame:
    """
    Detect outliers in a DataFrame using various methods.
    
    Parameters:
    - df: pd.DataFrame, the input data.
    - age_column: str, the name of the age column to analyze.
    - volumetric_columns: list, the names of the volumetric columns to analyze.
    - misc_columns: list, list of additional columns to include in the output.
    - method: str, the method to use for outlier detection. Options are 'zscore', 'pca', 'lof', 'isolation_forest'.
    - plot: bool, whether to plot the results.
    - explained: bool, whether to compute explainable differences.
    - kwargs: dict, additional parameters for the chosen method.
    -   For 'zscore': {'threshold': float} - the z-score threshold for outlier detection.
    -   For 'lof': {'n_neighbors': int} - the number of neighbors for
        Local Outlier Factor.
    -   For 'isolation_forest': {'contamination': float} - the proportion of outliers in the data.
    -   For 'pca': {'n_components': int, 'threshold': float} - number of PCA components and threshold for outlier detection.
    -   For 'mahalanobis': {'threshold': float} - the threshold for Mahalanobis distance outlier detection.
    -   For 'explain_method': {'method': str} - the method to use for explainable differences, options are 'covariance' or 'linear_regression'.
    
    Returns:
    - outliers: pd.DataFrame, the detected outliers.
    """
    df = df.copy()
    outliers = pd.DataFrame()
    z_score_agg = pd.DataFrame()
    #Perform outlier detection for each age group
    for age in df[age_column].unique():
        age_df = df[df[age_column] == age]
        # print(f"Is age_df {age} empty", age_df.empty)
        outliers_grouped = pd.DataFrame()
        if not age_df.empty:
            # perform z-score normalization
            z_scores = (age_df[volumetric_columns] - age_df[volumetric_columns].mean()) / age_df[volumetric_columns].std()

            # perform outlier detection based on the covariance
            outliers_grouped = threshold_outlier_detection(z_scores, thresholds=cov_thresholds)
            zscore_outliers = threshold_outlier_detection(z_scores, skip_covariance=True, thresholds=zscore_thresholds)
            #Get all columns ending in _zscore from zscore_outliers and add them to outliers_grouped
            zscore_columns = [col for col in zscore_outliers.columns if col.endswith('_zscore')]
            # Add the z-score columns to outliers_grouped
            for col in zscore_columns:
                outliers_grouped[col] = zscore_outliers[col].values

            # outliers_grouped["n_roi_outliers_zscore"] = zscore_outliers[ "n_roi_outliers_zscore"]
            for col in misc_columns:
                if col in age_df.columns:
                    outliers_grouped[col] = age_df[col].values
                else:
                    outliers_grouped[col] = np.nan
            
            # filter to keep only rows with outliers
            #outliers_grouped = outliers_grouped[outliers_grouped["outliers"] > 0]
            outliers_grouped = outliers_grouped[(outliers_grouped["n_roi_outliers_cov"] > 0) | (outliers_grouped["n_roi_outliers_zscore"] > 0)]
           
        if not outliers.empty:
            outliers = pd.concat([outliers, outliers_grouped], ignore_index=True)
            first_cols = ["project", "subject", "session","age_in_months"]
            if "input gear v" in df.columns:
                 first_cols.append("input gear v")
            # Any other columns that are present
            
            other_cols = [col for col in outliers.columns if col not in first_cols]
            
            # Reorder
            outliers = outliers[first_cols + other_cols]
            
        else:
            print("Outliers is empty....",age)
            # continue
            outliers = outliers_grouped
        
        #outliers.to_csv(f"outliers_{age}.csv", index=False)
        # Aggregate z-scores for all ages

        # Not used, but can be useful for further analysis
        # z_scores['subject'] = age_df['subject']

        # if not z_score_agg.empty:
        #     # Aggregate z-scores for all ages
        #     z_score_agg = pd.concat([z_score_agg, z_scores], ignore_index=True)
        # else:
        #     z_score_agg = z_scores.copy()
    
    #flag outliers
    outliers["is_outlier"] = True
    tag_only = outliers[["subject", "is_outlier"]].drop_duplicates()
    df = df.copy().merge(tag_only, how='left', on='subject')
    df['is_outlier'] = df['is_outlier'].fillna(0).astype(bool)
    
    first_cols = ["project", "subject", "session","is_outlier","n_roi_outliers_zscore","n_roi_outliers_cov"]
    if "input gear v" in df.columns:
         first_cols.append("input gear v")
    # Any other columns that are present
    other_cols = [col for col in outliers.columns if col not in first_cols]
    # Reorder
    outliers = outliers[first_cols + other_cols]

    # cleanup
    df.drop(columns = ["Unnamed: 0"], inplace=True, errors='ignore')
    outliers.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    
    return df, outliers



def plot_outlier_trend(outliers_df,keyword):

    # Get columns for each method
    cov_cols = [col for col in outliers_df.columns if col.endswith("_outlier_cov") and not col.startswith("n_roi_")]
    zscore_cols = [col for col in outliers_df.columns if col.endswith("_outlier_zscore") and not col.startswith("n_roi_")]

    # Compute counts
    cov_counts = outliers_df[cov_cols].sum(axis=0).sort_values(ascending=True)
    zscore_counts = outliers_df[zscore_cols].sum(axis=0).sort_values(ascending=True)

    # Setup subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Plot 1: Covariance-based outliers
    sns.barplot(y=cov_counts.index, x=cov_counts.values, hue=cov_counts.index, palette="Blues_d", ax=axes[0],legend=False)
    axes[0].set_title("Outlier Frequency by ROI (Covariance)")
    axes[0].set_xlabel("Number of Outliers")
    axes[0].set_ylabel("ROI")
    # Plot 2: Z-score-based outliers
    sns.barplot(y=zscore_counts.index, x=zscore_counts.values, palette="Reds_d", ax=axes[1])
    axes[1].set_title("Outlier Frequency by ROI (Z-score)")
    axes[1].set_xlabel("Number of Outliers")
    axes[1].set_ylabel("ROI")  # Hide duplicate y-axis label

    plt.suptitle(keyword, fontsize=16)
    plt.tight_layout()
    #   

    return fig



def cleaning_procedure(df_outlier_flag, MM_vols, RA_vols):
    
    # Step 1: Load the outlier-flagged data
    #df_outlier_flag = pd.read_csv('/Users/Hajer/unity/GF Sprint25/August_London/Data/UNITY-dataTemplate_MRI_outlierFlag.csv')
    st.info(f"Original size .. {df_outlier_flag.shape}")

    dup_keys = ["project", "subject", "session","age",'acquisition']
    volume_cols = [c for c in df_outlier_flag.columns if c.startswith("mm") or c.startswith("ra")]
    #If either of is_recon-all-clinical_outlier or is_minimorph_outlier are missing, add them with False values
    if "is_recon-all-clinical_outlier" not in df_outlier_flag.columns:
        df_outlier_flag["is_recon-all-clinical_outlier"] = False
    if "is_minimorph_outlier" not in df_outlier_flag.columns:
        df_outlier_flag["is_minimorph_outlier"] = False

    # Condition 1: RA outlier only, and all MM_vols are NA
    cond1 = (
        (df_outlier_flag["is_recon-all-clinical_outlier"]) &
        (~df_outlier_flag["is_minimorph_outlier"]) &
        (df_outlier_flag[MM_vols].isna().all(axis=1))
    )

    # Condition 2: MM outlier only, MM_vols populated, all RA_vols NA
    cond2 = (
        (df_outlier_flag["is_minimorph_outlier"]) &
        (~df_outlier_flag["is_recon-all-clinical_outlier"]) &
        (df_outlier_flag[MM_vols].notna().all(axis=1))
        & (df_outlier_flag[RA_vols].isna().all(axis=1))
    )

    # Condition 3: Both MM and RA are outliers
    cond3 = (
        df_outlier_flag["is_minimorph_outlier"]
        & df_outlier_flag["is_recon-all-clinical_outlier"]
    )
    st.write("1. Filter out rows based on outlier conditions:")

    st.write("a. Condition 1: RA outlier, no MM data, N=", cond1.sum())
    st.write("b. Condition 2: MM outlier, no RA data, N=", cond2.sum())
    st.write("c. Condition 3: RA outlier and MM outlier, N=", cond3.sum())

    df_filtered = df_outlier_flag[~(cond1 | cond2 | cond3)].copy()
    df_outlier_flag["is_outlier"] = (cond1 | cond2 | cond3)


    #Tag the duplicates
    df_filtered["is_duplicate"] = df_filtered.duplicated(subset=dup_keys, keep=False)
    st.write("N duplicates (project-subject-session-age-acquisition):", df_filtered[df_filtered["is_duplicate"]==True].shape)

    # df_filtered[df_filtered["is_duplicate"]].to_csv('/Users/Hajer/unity/GF Sprint25/August_London/Data/UNITY-dataTemplate_MRI_duplicates.csv',index=False)
    st.write("2. Filter out rows where amygdala volumes are below threshold (if amygdala segmentation is present)")

    #Step 2: Apply amygdala threshold filter if RA_left_amygdala and RA_right_amygdala are present
    if "ra_left_amygdala" in df_filtered.columns and "ra_right_amygdala" in df_filtered.columns:
        df_filtered["pass_amygdala"] = np.nan  # start empty
        df_filtered.loc[df_filtered["is_duplicate"], "pass_amygdala"] = (
            (df_filtered["ra_left_amygdala"] >= 250) &
            (df_filtered["ra_right_amygdala"] >= 250)
        )

        df_filtered = df_filtered[df_filtered["pass_amygdala"].isna() | (df_filtered["pass_amygdala"] == True)]

        st.write("N rows after amygdala threshold filter:", df_filtered.shape)
    else:
        st.warning("RA_left_amygdala and/or RA_right_amygdala columns not found. Skipping amygdala threshold filter.")
        df_filtered["pass_amygdala"] = np.nan

    # Step 3: From those that pass amygdala, pick the one with fewest zeros
    # Only candidates where pass_amygdala is True
    st.write("3. From those that passsed the conditions, pick the one with fewest zeros in volumetric columns (failed segmentations)")
    candidates = df_filtered.copy()
    candidates["zero_count"] = (candidates[volume_cols] == 0).sum(axis=1)

    # Sort so lowest zero_count per duplicate group comes first
    candidates = candidates.sort_values(dup_keys + ["zero_count"])

    # Keep one per group (lowest zero_count)
    candidates_cleaned = candidates.drop_duplicates(subset=dup_keys, keep="first").copy()

    # Now combine with the NaN pass_amygdala rows
    # na_rows = df_filtered[df_filtered["pass_amygdala"].isna()].copy()
    df_final = pd.concat([candidates_cleaned], ignore_index=True)


    # candidates = df_filtered[df_filtered["pass_amygdala"]].copy()
    # candidates["zero_count"] = (candidates[volume_cols] == 0).sum(axis=1)
    # candidates = candidates.sort_values(dup_keys + ["zero_count"])
    # #display(candidates)
    # df_final = candidates.drop_duplicates(subset=dup_keys, keep="first").copy() #Keep the first (lower count of zero)


    # st.write("Before cleaning duplicate rows .." , df_filtered.shape)
    st.write("After cleaning duplicate rows .." , df_final.shape)

    return df_final
    # df_final.to_csv("/Users/Hajer/unity/GF Sprint25/August_London/Data/PRISMA_clean.csv",index=False)
    #df_filtered.to_csv("/Users/Hajer/unity/GF Sprint25/August_London/Data/UNITY-dataTemplate_MRI_withDups.csv",index=False)


@st.cache_data
def process_outliers(df, df_demo, keywords):
    key_cols = ["subject", "session", "project"]

    for segmentation_tool in keywords:
        with open(os.path.join(work_dir, "..","utils","thresholds.yml"), 'r') as f:
            threshold_dictionary = yaml.load(f, Loader=yaml.SafeLoader)
            outlier_thresholds = threshold_dictionary[segmentation_tool]['thresholds']
            volumetric_cols = threshold_dictionary[segmentation_tool]['volumetric_cols']


        #reformat column headers
        #df.columns = df.columns.str.replace('_', ' ').str.replace('-', ' ').str.lower()

        #Make a new age column that has rounded up age
        #handle nans when applying lambda
        #From dfs read in the "age_at_scan_months" and session and subject from , and fill in the age in the df dataframe by matching subject and session ids

        # Merge to bring in age from demo
        if df_demo is not None:
            df_merged = df.merge(
                df_demo[['subject', 'session', 'age']], 
                on=['subject', 'session'], 
                how='left', 
                suffixes=('', '_from_demo')
            )
        else:
            st.warning("No demographic file uploaded. Missing ages will remain as NaN.")
            df_merged = df.copy()
            df_merged['age_from_demo'] = np.nan


        # Fill missing ages using demo
        df_merged['age'] = df_merged['age'].combine_first(df_merged['age_from_demo'])

        # drop the helper column
        df_merged = df_merged.drop(columns=['age_from_demo'])
        print(f"Number of NA ages: {df['age'].isna().sum()}")

        df_merged['age_in_months'] = df_merged['age'].apply(lambda x: int(np.ceil(x)) if pd.notnull(x) else np.nan)
        # df_merged.rename(columns={"icv":"total_intracranial"},inplace=True)
            

        columns_to_keep = ['project', 'subject','session', 'age_in_months', 'sex','acquisition','session_qc']  + volumetric_cols
        if "input_gear_v" in df.columns:
            columns_to_keep.insert(6, "input_gear_v")


        st.write("Size of dataframe before outlier detection:", df_merged.shape)
        #st.write('Columns are the following:', df_merged.columns.tolist())
        #Filter out those that failed visual QC?
        #Filter out from the outliers_df those sessions that are in the failed_qc
        #Filter rows where "Session QC" column containts T2w_QC_failed

        #failed_qc = df_merged[df_merged['session_QC'].str.contains('T2w_QC_failed', na=False)]
        #df_filtered should contain only those that did not fail QC
        df_filt = df_merged[df_merged["session_qc"] != 'T2w_QC_failed'] 

        #st.write(f"N sessions that failed QC: {df_merged[df_merged['session_qc'] == 'T2w_QC_failed'].value_counts()}")
        # sns.countplot(x='failed_qc', data=df_merged, palette='Set1')
        # plt.title("QC-failed")

        # break

        st.write(f"Length of dataframe after filtering out failed AXI-T2 QC: {len(df_filt)} / {len(df_merged)}")
        #Running outlier detection
        st.info("Running outlier detection using covariance and z-score methods...")

        df_, outliers_df = outlier_detection(df_filt[columns_to_keep], age_column = 'age_in_months',volumetric_columns=volumetric_cols, misc_columns= columns_to_keep, cov_thresholds = outlier_thresholds, zscore_thresholds = outlier_thresholds)
        st.info(f"N outliers found: {len(outliers_df)}")
        fig = plot_outlier_trend(outliers_df, segmentation_tool)
        st.pyplot(fig, use_container_width=False) 
        outliers_path = os.path.join(work_dir,"..","data",f"{segmentation_tool}_outliers.csv")
        

        if outliers_df.empty:
            st.error("No outliers found. Nothing to clean.")
            # st.stop()
        else:    
            outliers_df.to_csv(outliers_path, index=False)
            
            st.success("Download complete! File can be found in the data folder.")
            st.dataframe(outliers_df.head())

            # Merge outliers with original dataframe to keep flag
            merged = df.merge(
                outliers_df[key_cols+volumetric_cols].drop_duplicates(),
                on=key_cols+volumetric_cols,
                how="left",
                indicator=True
            )

            #display(merged)

            if f"is_{segmentation_tool}_outlier" not in df:
                df[f"is_{segmentation_tool}_outlier"] = False


            df.loc[merged["_merge"] == "both", f"is_{segmentation_tool}_outlier"] = True

    df.to_csv(os.path.join(work_dir,"..","data",f"allData_outlierFlagged.csv"), index=False)
    return df

def main ():

    # Upload a CSV file with the data to clean
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    #Add check box to ask if cleaning needs to be stratified by project or not
    stratify = st.checkbox("Stratify cleaning by project?", value=True)
   
    if uploaded_file is not None:
        # Read CSV into pandas DataFrame
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Show first few rows
        st.dataframe(df.head())

        unique_projects = df.project.unique()
        if stratify:
            st.info(f"Stratifying cleaning by project. Projects found: {', '.join(unique_projects)}")
        else:
            st.info(f"Not stratifying cleaning by project. Projects found: {', '.join(unique_projects)}")

    #Add radio buttons inderneath to select the segmentation tool
    segmentation_tool = st.radio("Upload derivatives from:", ["recon-all-clinical", "minimorph","both"])
    keywords = []

    if segmentation_tool == "both":
        keywords = ["recon-all-clinical", "minimorph"]
    else:
        keywords = [segmentation_tool]

    #Upload optional demographic data file
    uploaded_demo = st.file_uploader("Upload demographic CSV file (optional)", type=["csv"])
    df_demo = None
    if uploaded_demo is not None:
        df_demo = pd.read_csv(uploaded_demo)
        st.success("Demographic file uploaded successfully!")
        st.dataframe(df_demo.head())

        #if file does not have headers "subject", "session", "age", add a warning
        required_columns = {"subject", "session", "age"}
        
        if not required_columns.issubset(df_demo.columns):
            st.error(f"Demographic file must contain the following columns (name-sensitive): {', '.join(required_columns)}")
            

    #Only enable the button if a file is uploaded
    if st.button("Detect Outliers") and uploaded_file is not None:

        if stratify and len(unique_projects) > 1:
            progress = st.progress(0)
            status = st.empty()

            for project in unique_projects:
                st.info(f"Processing project: {project}")
                df_project = df[df["project"] == project]
                processed = process_outliers(df_project, df_demo, keywords)
                if "processed_df" in st.session_state:
                    st.session_state["processed_df"] = pd.concat([st.session_state["processed_df"], processed], ignore_index=True)
                else:
                    st.session_state["processed_df"] = processed

                progress.progress((np.where(unique_projects == project)[0][0] + 1) / len(unique_projects))
        else:
            st.info("Processing all projects together...")
            processed = process_outliers(df, df_demo, keywords)
            st.session_state["processed_df"] = processed
        # st.stop()
        # processed = process_outliers(df, df_demo, keywords)
        # st.session_state["processed_df"] = processed
            
        
    #Step 2: Get a cleaned dataset
    st.write("### Clean Outliers")
    #Add button to clean outliers
    #Get a cleaned dataset
    st.write('Click the button below to download a clean dataset with outliers removed based on the following criteria:')
    st.markdown("""
    - Condition 1: RA outlier only, and all MM_vols are NA
    - Condition 2: MM outlier only, MM_vols populated, all RA_vols NA
    - Condition 3: Both MM and RA are outliers
    - From those that pass amygdala threshold (if RA_left_amygdala and RA_right_amygdala are present), pick the one with fewest zeros (failed segmentations)
    """)

    if st.button("Clean Outliers") and "processed_df" in st.session_state:
        st.info("Cleaning outliers...")
        df_ = st.session_state["processed_df"]

        MM_vols = [col for col in df_.columns if col.startswith("mm_")]
        RA_area = [col for col in df_.columns if col.startswith("ra_") and col.endswith('_area')]
        RA_thickness = [col for col in df_.columns if col.startswith("ra_") and col.endswith('_thickness')]

        RA_vols = [col for col in df.columns if col.startswith("ra_") and not(col in (RA_area + RA_thickness))]
        
        clean_df = cleaning_procedure(df_, MM_vols, RA_vols)
        st.success("Outliers cleaned.")
        st.write(f"Size of dataframe after cleaning: {clean_df.shape}")
        #Download cleaned dataframe
        clean_path = os.path.join(work_dir,"..", "data",f"alldata_cleaned.csv")
        clean_df.to_csv(clean_path, index=False)
        st.success("Download complete.")


#call main
if __name__ == "__main__":
    main()