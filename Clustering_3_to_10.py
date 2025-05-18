#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:11:33 2025

@author: jalalfaraj
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load your cleaned dataset
data_path = '/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/cleaned_data.csv'
data = pd.read_csv(data_path)

# Selecting the specified features for clustering
selected_features = [
    'SchlEngage_2223', 'sex_2223', 'argue_2223', 
    'age5_2223', 'ParAggrav_2223', 'bullied_2223', 
    'ACE2more_2223', 'MakeFriend_2223', 'ADHD_Severity'
]

eda_data = data[selected_features].dropna()
X = eda_data.drop(columns=['ADHD_Severity'])
y = eda_data['ADHD_Severity']

# Scaling the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: PCA Transformation (3 Components for Clustering)
pca = PCA(n_components=3)
pca_data = pca.fit_transform(X_scaled)

# Step 2: KMeans Clustering (3 clusters for finer ADHD severity)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(pca_data)

# Adding cluster labels to the original data
eda_data['ADHD_Severity_Expanded'] = cluster_labels + 1  # To make clusters from 1 to 3

# Step 3: Visualizing Clusters in 3D PCA Space
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2', 'PC3'])
pca_df['Cluster'] = cluster_labels + 1

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], 
                     c=pca_df['Cluster'], cmap='viridis', alpha=0.7)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA with KMeans Clusters (3 Levels)')
fig.colorbar(scatter, ax=ax, label='Cluster (Severity)')
plt.show()

# Step 4: Visualizing Clusters in 2D PCA Space with Contour
pca_df_2d = pd.DataFrame(data=pca_data[:, :2], columns=['PC1', 'PC2'])
pca_df_2d['Cluster'] = cluster_labels + 1

plt.figure(figsize=(10, 6))
sns.kdeplot(x='PC1', y='PC2', data=pca_df_2d, cmap='viridis', fill=True, alpha=0.3)
sns.scatterplot(data=pca_df_2d, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.7)
plt.title('2D PCA with KMeans Clusters (3 Levels)')
plt.show()

# Step 5: Save the updated dataset with new ADHD severity levels
eda_data.to_csv('/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/ADHD_Severity_Expanded.csv', index=False)
print("\n✅ Updated dataset with expanded ADHD severity saved as 'ADHD_Severity_Expanded.csv'")
