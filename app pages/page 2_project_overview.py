import streamlit as st
import pandas as pd

def page_project_overview_body(data):

    st.title("Project Overview")

    # Project background
    st.markdown("### 📌 Background")
    st.write(
        "Cardiovascular disease is one of the leading causes of death worldwide. "
        "Early detection and risk prediction can help healthcare providers take "
        "preventive action and guide treatment. "
        "This project uses machine learning techniques to analyse patient data "
        "and predict the likelihood of heart disease."
    )

    # Dataset summary
    st.markdown("### 📊 Dataset Summary")
    st.write(
        "The dataset was sourced from the UCI Heart Disease dataset (via Kaggle). "
        "It contains clinical and demographic information for patients, "
        "with a target variable indicating presence (1) or absence (0) of heart disease."
    )

    # Key dataset metrics
    num_patients = data.shape[0]
    num_features = data.shape[1] - 1  # excluding target
    disease_rate = (data['target'].mean() * 100).round(1)

    st.metric("Total Patients", num_patients)
    st.metric("Features", num_features)
    st.metric("Heart Disease Rate", f"{disease_rate}%")

    # Dataset preview
    st.markdown("#### 🗂️ Sample Data")
    st.dataframe(data.head(10))

    # Glossary of terms
    st.markdown("### 📖 Key Medical Terms")
    st.info(
        f"* **cp**: Chest pain type (4 values)\n"
        f"* **trestbps**: Resting blood pressure (mm Hg)\n"
        f"* **chol**: Serum cholesterol (mg/dl)\n"
        f"* **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)\n"
        f"* **restecg**: Resting electrocardiographic results (0, 1, 2)\n"
        f"* **thalach**: Maximum heart rate achieved\n"
        f"* **exang**: Exercise induced angina (1 = yes, 0 = no)\n"
        f"* **oldpeak**: ST depression induced by exercise relative to rest\n"
        f"* **slope**: Slope of the peak exercise ST segment\n"
        f"* **ca**: Number of major vessels (0–3) colored by fluoroscopy\n"
        f"* **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)"
    )
