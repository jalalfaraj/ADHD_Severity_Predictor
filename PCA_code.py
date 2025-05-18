#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:10:26 2025

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D

# Load your cleaned dataset
data_path = '/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/cleaned_data.csv'
data = pd.read_csv(data_path)

# Selecting the specified features for EDA
selected_features = [
    'SchlEngage_2223', 'sex_2223', 'argue_2223', 
    'age5_2223', 'ParAggrav_2223', 'bullied_2223', 
    'ACE2more_2223', 'MakeFriend_2223', 'ADHD_Severity'
]

eda_data = data[selected_features].dropna()  # Dropping NaNs for simplicity
X = eda_data.drop(columns=['ADHD_Severity'])
y = eda_data['ADHD_Severity']

# Scaling the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: PCA with Variance Explanation
pca = PCA(n_components=3)
pca_data = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

# Plotting Variance Explained by PCA
plt.figure(figsize=(8, 5))
plt.bar(range(1, 4), explained_variance * 100, color='skyblue')
plt.title('Variance Explained by Each PCA Component (%)')
plt.xlabel('PCA Components')
plt.ylabel('Variance Explained (%)')
plt.show()

# Step 2: 2D PCA Projection with Contour
pca_df_2d = pd.DataFrame(data=pca_data[:, :2], columns=['PC1', 'PC2'])
pca_df_2d['ADHD_Severity'] = y

plt.figure(figsize=(10, 6))
sns.kdeplot(x='PC1', y='PC2', data=pca_df_2d, cmap='viridis', fill=True, alpha=0.3)
sns.scatterplot(data=pca_df_2d, x='PC1', y='PC2', hue='ADHD_Severity', palette='viridis', alpha=0.6)
plt.title('2D PCA with Contour Overlay')
plt.show()

# Step 3: 3D PCA Projection
pca_df_3d = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2', 'PC3'])
pca_df_3d['ADHD_Severity'] = y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_df_3d['PC1'], pca_df_3d['PC2'], pca_df_3d['PC3'], 
                     c=y, cmap='viridis', alpha=0.7)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA Projection')
fig.colorbar(scatter, ax=ax, label='ADHD_Severity')
plt.show()

# Step 4: t-SNE Projection
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_data = tsne.fit_transform(X_scaled)

# Creating a DataFrame for t-SNE results
tsne_df = pd.DataFrame(data=tsne_data, columns=['TSNE1', 'TSNE2'])
tsne_df['ADHD_Severity'] = y

# Visualizing t-SNE
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='ADHD_Severity', palette='viridis', alpha=0.7)
plt.title('t-SNE Projection of Selected Features (2D)')
plt.show()

# Step 5: Random Forest Feature Importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_scaled, y)
rf_feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Displaying Feature Importance
plt.figure(figsize=(10, 6))
rf_feature_importance.plot(kind='bar', color='skyblue')
plt.title('Feature Importance (Random Forest) - Selected Features')
plt.ylabel('Importance')
plt.show()

print("\n✅ Feature Importance from Random Forest:")
print(rf_feature_importance)
