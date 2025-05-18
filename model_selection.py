#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:12:09 2025

@author: jalalfaraj
"""

# ADHD Severity Prediction Model (0-2)

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Load Data
data = pd.read_csv('/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/ADHD_Severity_Expanded.csv')

# Step 2: Preparing Features and Target
selected_features = [
    'SchlEngage_2223', 'sex_2223', 'argue_2223', 
    'age5_2223', 'ParAggrav_2223', 'bullied_2223', 
    'ACE2more_2223', 'MakeFriend_2223'
]
X = data[selected_features]
y = data['ADHD_Severity_Expanded'] - 1  # Convert to 0-2 for model training

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

model_performance = {}
trained_models = {}

def evaluate_model(model_name, model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n🔍 {model_name} Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='viridis', fmt='d', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    model_performance[model_name] = accuracy
    return model

# Training and Evaluating All Models
for model_name, model in models.items():
    print(f"\n🔧 Training {model_name}...")
    model.fit(X_train_scaled, y_train)
    trained_models[model_name] = evaluate_model(model_name, model, X_test_scaled, y_test)

# Identifying Best Model
best_model_name = max(model_performance, key=model_performance.get)
print(f"\n✅ Best Model: {best_model_name} with Accuracy: {model_performance[best_model_name]:.4f}")
best_model = trained_models[best_model_name]

# Step 6: Saving the Best Model and Scaler
with open('/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/best_adhd_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Feature Importance (If Random Forest or XGBoost)
if best_model_name in ["Random Forest", "XGBoost"]:
    importance = best_model.feature_importances_
    feature_importance = pd.Series(importance, index=selected_features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar', color='skyblue')
    plt.title(f'Feature Importance ({best_model_name})')
    plt.show()

print("\n✅ Best model and scaler saved as 'best_adhd_model.pkl' and 'scaler.pkl'")
