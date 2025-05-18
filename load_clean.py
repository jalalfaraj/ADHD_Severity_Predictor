#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:09:01 2025

@author: jalalfaraj
"""

import pandas as pd
import numpy as np
import os

# File Paths
DATA_PATH = '/Users/user/Desktop/Data Science Projects/ADHD_Divorce/Data/Data_2022-2023.csv'
RESULTS_PATH = '/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/'

# Ensure Results Directory Exists
os.makedirs(RESULTS_PATH, exist_ok=True)

# Load the Data
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    print(f'Initial Data Loaded: {data.shape[0]} rows, {data.shape[1]} columns')

    # Filtering Only Relevant Values for ADHD and Anxiety (Severity 1, 2, 3)
    data = data[(data['ADHDSevInd_2223'].isin([1, 2, 3])) & (data['anxiety_2223'].isin([1, 2, 3]))]

    # Removing Missing Values (99, 95, 90) in All Relevant Columns
    data.replace([99, 95, 90], np.nan, inplace=True)

    # Dropping rows only if critical columns are missing
    data.dropna(subset=['ADHDSevInd_2223', 'anxiety_2223'], inplace=True)

    # Renaming Columns for Clarity
    data.rename(columns={'ADHDSevInd_2223': 'ADHD_Severity', 'anxiety_2223': 'Anxiety_Severity'}, inplace=True)

    # Identifying Relevant Columns based on Codebook
    relevant_columns = [
        'ADHD_Severity', 'Anxiety_Severity', 'age5_2223', 'sex_2223', 'MotherMH_2223', 'FatherMH_2223',
        'smoking_2223', 'ScreenTime_2223', 'HrsSleep_2223', 'ACEdivorce_2223', 'PhysAct_2223',
        'bully_2223', 'bullied_2223', 'argue_2223', 'MakeFriend_2223', 'SchlEngage_2223', 
        'SmkInside_2223', 'vape_2223', 'ShareIdeas_2223', 'MealTogether_2223', 'FamResilience_2223',
        'ACE2more_2223', 'ParAggrav_2223', 'EmSupport_2223', 'ParCoping_2223', 'BedTime_2223', 
        'SchlMiss_2223', 'AftSchAct_2223'
    ]

    data = data[relevant_columns]

    # Handling Missing Values:
    # Ordinal Columns (Categorical with meaningful order)
    ordinal_cols = [
        'SmkInside_2223', 'MealTogether_2223', 'ParAggrav_2223', 'ParCoping_2223', 
        'MotherMH_2223', 'FatherMH_2223', 'PhysAct_2223', 'bully_2223', 'bullied_2223',
        'argue_2223', 'MakeFriend_2223', 'age5_2223', 'ScreenTime_2223', 'BedTime_2223'
    ]

    # Binary Variables
    binary_cols = ['HrsSleep_2223', 'ACEdivorce_2223', 'EmSupport_2223', 'vape_2223']

    # Non-Ordinal Categorical Columns
    categorical_cols = ['sex_2223', 'smoking_2223', 'ShareIdeas_2223', 
                        'SchlEngage_2223', 'SchlMiss_2223', 'AftSchAct_2223']

    # Creating Missing Indicator Columns for All Variables
    for col in ordinal_cols + binary_cols + categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col + '_Missing'] = data[col].isnull().astype(int)
    
    # Imputing Missing Values:
    for col in ordinal_cols + binary_cols + categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Final Check for Remaining NaNs
    rows_with_nan = data[data.isnull().any(axis=1)]
    if not rows_with_nan.empty:
        print("\n⚠️ Rows with NaN Values Found After Cleaning:")
        print(rows_with_nan)
    else:
        print("\n✅ No Rows with NaN Values Found in the Final Cleaned Data.")

    # Save the Cleaned Data (No Encoding or Scaling)
    cleaned_data_path = RESULTS_PATH + 'cleaned_data.csv'
    data.to_csv(cleaned_data_path, index=False)
    print(f'Cleaned Data Saved: {cleaned_data_path}')

    # Save a Shortened Version (50 Rows)
    sample_data_path = RESULTS_PATH + 'head_cleaned_data.csv'
    data.head(50).to_csv(sample_data_path, index=False)
    print(f'Sample Data (50 Rows) Saved: {sample_data_path}')

    return data

# Test the Cleaning Function
if __name__ == '__main__':
    cleaned_data = load_and_clean_data(DATA_PATH)
    print(f'Final Cleaned Data: {cleaned_data.shape[0]} rows, {cleaned_data.shape[1]} columns')
