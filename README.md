# ADHD Severity Prediction Project

This repository provides an end-to-end machine learning pipeline to predict ADHD severity in children using behavioral, social, and environmental factors. The project is designed to help parents and educators identify the risk of ADHD in children and take early intervention steps.

## 📁 Project Structure

├── app.py # Streamlit application for ADHD severity prediction

├── Clustering_3_to_10.py # Code for clustering ADHD severity into 3 levels

├── feature_engineering.py # Feature engineering and transformation

├── feature_selection.py # Feature selection using multiple methods

├── load_clean.py # Data loading and cleaning

├── model_selection.py # Model training and evaluation (Logistic Regression, RF, XGBoost)

├── PCA_code.py # PCA transformation and visualization

├── Visualizing_Clusters.py # Visualizing cluster distribution

├── README.md # Project documentation

---

## 🚀 Project Overview
This project aims to predict the severity of ADHD (Attention Deficit Hyperactivity Disorder) in children using a series of behavioral and environmental factors. It provides:
- A machine learning pipeline for feature selection, clustering, and model training.
- An interactive Streamlit app for parents to assess their child's ADHD severity.

### ✅ Key Features:
- Data cleaning and preprocessing using `load_clean.py`.
- Feature engineering and transformation using `feature_engineering.py`.
- Automated feature selection using `feature_selection.py`.
- Cluster analysis of ADHD severity into **3 levels** (Low, Moderate, High) using `Clustering_3.py`.
- Machine learning model training and evaluation using `model_selection.py`.
- Principal Component Analysis (PCA) for dimensionality reduction using `PCA_code.py`.
- Streamlit app for real-time ADHD severity prediction (`app.py`).

---

## ⚡ How It Works
1. Data is cleaned, transformed, and standardized using `load_clean.py` and `feature_engineering.py`.
2. The most important features are selected using `feature_selection.py`.
3. Children are clustered into **3 severity levels** using KMeans (`Clustering_3.py`):
   - **0: Low Severity**
   - **1: Moderate Severity**
   - **2: High Severity**
4. Machine learning models (Logistic Regression, Random Forest, XGBoost) are trained (`model_selection.py`).
5. The best-performing model is saved and used in the Streamlit app (`app.py`).
6. Parents can enter their child’s details in the app to receive a predicted ADHD severity score (1-3) along with actionable recommendations.

---

## 🌐 How to Run the Streamlit App Locally
1. Make sure you have **Python 3.7+** installed.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Access the app at:

arduino
Copy
Edit
http://localhost:8501
🚀 How to Deploy to Streamlit Cloud

✅ Step 1: Fork or Clone This Repository
bash
Copy
Edit
git clone https://github.com/your-username/ADHD-Prediction.git

✅ Step 2: Sign Up on Streamlit Cloud
Go to Streamlit Cloud.

Sign in with your GitHub account.

✅ Step 3: Create a New App
Click "New App" in Streamlit Cloud.

Connect it to your GitHub repository.

Set the app entry point to:

Copy
Edit
app.py

✅ Step 4: Set Up Dependencies
Ensure your requirements.txt file includes:

nginx
Copy
Edit
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost

✅ Step 5: Deploy
Click "Deploy".

Your app will be live, and you will receive a public URL (like https://your-username-your-repo.streamlit.app/).


---
📊 Example Use Case
Parents enter their child’s details (school engagement, argument frequency, bullying experience, etc.).

The app predicts ADHD severity (1-3) and provides actionable recommendations.

Parents can see how their child's values compare to the average values of non-compromised children.


---
💡 How to Extend This Project
Add new behavioral or medical features to improve prediction accuracy.

Integrate additional models (like LightGBM or CatBoost).

Use GridSearchCV for hyperparameter optimization in model_selection.py.

Deploy the Streamlit app on Streamlit Cloud or any cloud provider (AWS, Azure, GCP).


---
Make sure your requirements.txt is correctly formatted and includes all necessary packages.


Acknowledgements
Data source: https://www.childhealthdata.org/
