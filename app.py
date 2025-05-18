#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:12:26 2025

@author: jalalfaraj
"""

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = pickle.load(open('best_adhd_model.pkl'', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit App Title and Description
st.title('ADHD Severity Prediction (1-3)')
st.markdown("""
This application predicts the **ADHD severity** of a child based on various behavioral and environmental factors. 
- The prediction scale ranges from **1 to 3**, where:
  - **1 = Safe (Low Severity)**
  - **3 = Very Compromised (High Severity)**
- The app provides guidance on what factors to focus on for improvement, excluding factors that cannot be changed (like age and sex).
""")

# User Inputs for the 8 Selected Features
SchlEngage = st.slider("School Engagement (1-5)", 1, 5, 3)
sex = st.selectbox("Child's Sex (0 = Female, 1 = Male)", [0, 1])
argue = st.slider("Argument Frequency (1-5)", 1, 5, 3)
age = st.slider("Child's Age (5-18)", 5, 18, 10)

ParAggrav = st.slider("Parental Aggravation", 1, 4, 2, format="%d", 
                      help="1: Never, 2: Rarely, 3: Sometimes, 4: Usually or always")

ACE2more = st.selectbox("Parental Relationship Status", 
                        [0, 1], format_func=lambda x: "Married" if x == 0 else "Divorced",
                        help="Select if the parents are married or divorced")

bullied = st.slider("Bullying Frequency", 1, 5, 2, format="%d",
                   help=("1: Never (in the past 12 months), "
                         "2: 1-2 times (in the past 12 months), "
                         "3: 1-2 times per month, "
                         "4: 1-2 times per week, "
                         "5: Almost every day"))

MakeFriend = st.selectbox("Difficulty Making Friends", 
                          [1, 2, 3], 
                          format_func=lambda x: 
                          "No difficulty" if x == 1 else 
                          "A little difficulty" if x == 2 else 
                          "A lot of difficulty")

# Prediction Button
if st.button("Predict ADHD Severity"):
    # Preparing input data for prediction
    input_data = pd.DataFrame([{
        'SchlEngage_2223': SchlEngage,
        'sex_2223': sex,
        'argue_2223': argue,
        'age5_2223': age,
        'ParAggrav_2223': ParAggrav,
        'bullied_2223': bullied,
        'ACE2more_2223': ACE2more,
        'MakeFriend_2223': MakeFriend
    }])

    # Scaling the input data
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    severity = int(prediction + 1)  # Convert from 0-9 to 1-10

    st.subheader(f"Predicted ADHD Severity: {severity} (1-3)")

    # Compromised Level Output
    if severity >= 3:
        st.error("🔴 High Risk: Very Compromised (3))")
        st.markdown("""
        - **Recommendation:** Focus on improving school engagement, reducing argument frequency, 
        and addressing social skills (like making friends). Seek professional help if needed.
        """)
    elif severity >= 2:
        st.warning("🟠 Moderate Risk: Compromised (2)")
        st.markdown("""
        - **Recommendation:** Maintain positive school engagement and minimize argument frequency. 
        - **Social Skills:** Encourage positive interactions to reduce bullying and social isolation.
        """)
    else:
        st.success("🟢 Low Risk: Safe (1)")
        st.markdown("""
        - **Recommendation:** Continue maintaining a supportive and structured environment. 
        - **Monitor:** Stay consistent with positive reinforcement and social support.
        """)

