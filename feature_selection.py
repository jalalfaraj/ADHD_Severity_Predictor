#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:10:04 2025

@author: jalalfaraj
"""
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def automated_feature_selection(data_path, target=str()):
    """
    Automated Feature Selection Function with Enhanced Debugging.

    Parameters:
    - data_path: str - The path to the cleaned dataset
    - target: str - The target column name (default 'ADHD_Severity')

    Returns:
    - pd.DataFrame - The dataset with selected features
    """
    print("\n🔍 Starting Automated Feature Selection...")

    # Loading the cleaned dataset
    try:
        data = pd.read_csv(data_path)
        print(f"\n✅ Loaded cleaned dataset from: {data_path}")
    except FileNotFoundError:
        raise ValueError(f"❌ The specified data path '{data_path}' does not exist.")

    # Verify target column exists
    if target not in data.columns:
        raise ValueError(f"❌ Target column '{target}' does not exist in the dataset.")
    
    # Exclude Anxiety_Severity if it exists
    if 'Anxiety_Severity' in data.columns:
        data.drop(columns=['Anxiety_Severity'], inplace=True)
        print("\n✅ Excluded 'Anxiety_Severity' from features.")

    # Separate features and target variable
    X = data.drop(columns=[target], errors='ignore')
    y = data[target]

    # Debugging: Checking if the target variable has missing values
    if y.isnull().sum() > 0:
        print(f"⚠️ Missing values in target column '{target}': {y.isnull().sum()}. Filling with mode.")
        y.fillna(y.mode()[0], inplace=True)

    # Handling Missing Values in Features
    X.fillna(X.median(numeric_only=True), inplace=True)
    X.fillna(X.mode().iloc[0], inplace=True)

    # Debugging: Checking for empty features
    if X.empty:
        raise ValueError("❌ No features available after separating target.")

    # Automatically Encoding Categorical Columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        X[col] = label_encoder.fit_transform(X[col])
    print(f"\n✅ Categorical columns encoded: {list(categorical_cols)}")

    # Scaling features for better performance
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Debugging: Verifying X_scaled is not empty
    if X_scaled.empty:
        raise ValueError("❌ No valid features available after scaling.")

    # Debugging: Checking for zero variance columns (constant features)
    zero_variance_features = X_scaled.columns[X_scaled.var() == 0].tolist()
    if zero_variance_features:
        print(f"\n⚠️ Removing zero variance features: {zero_variance_features}")
        X_scaled.drop(columns=zero_variance_features, inplace=True)

    ### 1. Correlation Analysis (Spearman)
    correlation_matrix = X_scaled.corr(method='spearman')
    
    print(f"\n🔍 Initial Feature Count: {X_scaled.shape[1]}")
    
    high_correlation_features = [
        column for column in correlation_matrix.columns 
        if any(abs(correlation_matrix[column]) > 0.85) and column != target
    ]

    # Debugging: Listing the highly correlated features
    print("\n✅ Highly Correlated Features Identified (Correlation > 0.85):")
    print(high_correlation_features)

    # Dropping highly correlated features if more than 2 features remain
    if len(high_correlation_features) > 0 and X_scaled.shape[1] - len(high_correlation_features) > 2:
        X_scaled.drop(columns=high_correlation_features, inplace=True, errors='ignore')
        print(f"\n✅ Highly Correlated Features Removed: {high_correlation_features}")
    else:
        print("\n❌ Not removing highly correlated features to avoid empty DataFrame.")

    # Debugging: Displaying remaining features after correlation filtering
    print(f"\n✅ Remaining Features after Correlation Filtering: {X_scaled.shape[1]}")
    print(X_scaled.columns.tolist())

    ### 2. Mutual Information (MI)
    try:
        print("\n🔍 Debugging X_scaled before Mutual Information Calculation:")
        print(f"Shape: {X_scaled.shape}")
        print(f"Any NaN values: {X_scaled.isnull().sum().sum()}")

        if X_scaled.empty:
            raise ValueError("❌ X_scaled is empty after preprocessing.")
        
        mi = mutual_info_classif(X_scaled, y)
        mi_scores = pd.Series(mi, index=X_scaled.columns).sort_values(ascending=False)
        mi_selected_features = mi_scores[mi_scores > 0.01].index.tolist()
        print("\n✅ Selected Features by Mutual Information (MI > 0.01):")
        print(mi_selected_features)
    except ValueError as e:
        raise ValueError(f"❌ Error in Mutual Information Calculation: {e}")

    ### 3. Recursive Feature Elimination (RFE) with Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    rfe = RFE(lr_model, n_features_to_select=10)
    rfe.fit(X_scaled, y)
    rfe_selected_features = X_scaled.columns[rfe.support_].tolist()
    print("\n✅ Top 10 Features Selected by RFE (Logistic Regression):")
    print(rfe_selected_features)

    ### 4. Random Forest Feature Importance
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_scaled, y)
    rf_feature_importance = pd.Series(rf_model.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
    rf_selected_features = rf_feature_importance[rf_feature_importance > 0.01].index.tolist()
    print("\n✅ Selected Features by Random Forest Importance (Importance > 0.01):")
    print(rf_selected_features)

    ### Automated Combined Feature Selection
    combined_features = set(mi_selected_features) & set(rfe_selected_features) & set(rf_selected_features)
    final_selected_features = list(combined_features)
    print("\n✅ Final Selected Features after Combining All Methods:")
    print(final_selected_features)

    # Creating final dataset with selected features
    X_final = X_scaled[final_selected_features]
    print("\n✅ Final Feature Selection Complete.")

    # Save the final selected features
    final_path = '/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/selected_features.csv'
    X_final.to_csv(final_path, index=False)
    print(f"\n✅ Final dataset with selected features saved at: {final_path}")

    return X_final

# Usage Example
data_path = '/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/cleaned_data.csv'
X_final = automated_feature_selection(data_path, target='ADHD_Severity')

