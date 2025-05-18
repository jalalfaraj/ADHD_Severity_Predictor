#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:11:33 2025

@author: jalalfaraj
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the updated dataset with expanded ADHD severity
data = pd.read_csv('/Users/user/Desktop/Data Science Projects/ADHD_Divorce/results/ADHD_Severity_Expanded.csv')

# Step 1: Calculating Cluster Descriptive Statistics
cluster_profile = data.groupby('ADHD_Severity_Expanded').mean().T
print("\n✅ Cluster Profile (Mean Values by Feature):")
print(cluster_profile)

# Step 2: Visualizing Cluster Sizes
cluster_sizes = data['ADHD_Severity_Expanded'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette='viridis')
plt.title('Cluster Sizes (ADHD Severity Levels 1-3)')
plt.xlabel('ADHD Severity Level (Cluster)')
plt.ylabel('Number of Data Points')
plt.show()

# Step 3: Radar Plot (Spider Plot) of Cluster Feature Means
from math import pi

# Preparing data for Radar Plot
categories = list(cluster_profile.index)
num_categories = len(categories)

# Radar Plot Setup
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
angles += angles[:1]

# Plotting each cluster
for cluster in range(1, 4):
    values = list(cluster_profile[cluster])
    values += values[:1]  # Ensuring the loop closes
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'Severity {cluster}')
    ax.fill(angles, values, alpha=0.2)

# Customizing Radar Plot
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Radar Plot - Feature Profiles by ADHD Severity Level (1-3)')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.show()
