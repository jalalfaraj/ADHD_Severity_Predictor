#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:09:25 2025

@author: jalalfaraj
"""

import pandas as pd
import numpy as np
import os

# File Paths
CLEANED_DATA_PATH = '/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/cleaned_data.csv'
RESULTS_PATH = '/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/'

# Ensure Results Directory Exists
os.makedirs(RESULTS_PATH, exist_ok=True)

# Load the Cleaned Data
def load_cleaned_data(file_path):
    data = pd.read_csv(file_path)
    print(f'Loaded Cleaned Data: {data.shape[0]} rows, {data.shape[1]} columns')
    return data

# Feature Engineering Function
def feature_engineering(data):
    print("\n✅ Starting Feature Engineering...")

    # Feature 1: Adequate_Sleep (Binary)
    def calculate_adequate_sleep(row):
        if row['age5_2223'] in [1] and 12 <= row['HrsSleep_2223'] <= 16:
            return 1
        elif row['age5_2223'] in [2] and 11 <= row['HrsSleep_2223'] <= 14:
            return 1
        elif row['age5_2223'] in [3] and 10 <= row['HrsSleep_2223'] <= 13:
            return 1
        elif row['age5_2223'] in [4] and 9 <= row['HrsSleep_2223'] <= 12:
            return 1
        elif row['age5_2223'] in [5] and 8 <= row['HrsSleep_2223'] <= 10:
            return 1
        else:
            return 0

    data['Adequate_Sleep'] = data.apply(calculate_adequate_sleep, axis=1)

    # Feature 2: Family Stability Score (Ordinal)
    data['Family_Stability_Score'] = (
        data['MealTogether_2223'] +     # Higher value means more family meals (more stability)
        (4 - data['ParAggrav_2223']) +  # Higher value means less parental aggravation (more stability)
        data['ParCoping_2223']          # Higher value means better parental coping (more stability)
    )

    # Feature 3: One-Hot Encoding Non-Ordinal Categorical Variables
    categorical_cols = ['sex_2223', 'smoking_2223', 'ShareIdeas_2223', 
                        'SchlEngage_2223', 'SchlMiss_2223', 'AftSchAct_2223']

    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Saving the Fully Engineered Data
    engineered_data_path = RESULTS_PATH + 'engineered_data.csv'
    data.to_csv(engineered_data_path, index=False)
    print(f'\n✅ Fully Engineered Data Saved: {engineered_data_path}')

    # Save a Shortened Version (50 Rows)
    sample_data_path = RESULTS_PATH + 'head_engineered_data.csv'
    data.head(50).to_csv(sample_data_path, index=False)
    print(f'Sample Data (50 Rows) Saved: {sample_data_path}')

    return data

# Run Feature Engineering Process
if __name__ == '__main__':
    cleaned_data = load_cleaned_data(CLEANED_DATA_PATH)
    engineered_data = feature_engineering(cleaned_data)
    print(f'\n✅ Final Engineered Data: {engineered_data.shape[0]} rows, {engineered_data.shape[1]} columns')
