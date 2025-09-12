import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_style("whitegrid")

def page_project_overview_body(data: pd.DataFrame):
    st.title("📊 Project Overview & Dataset Insights")

    # ---- Why: Problem Context ----
    st.markdown("### Why this matters")
    st.info(
        "Cardiovascular disease is the **leading cause of death worldwide**. "
        "Early detection of high-risk patients can save lives, reduce hospitalizations, "
        "and lower healthcare costs. "
        "This project aims to leverage patient clinical and lifestyle data to identify patients at risk of heart disease."
    )

    # ---- So What: Dataset Summary & Insights ----
    st.markdown("### Dataset Summary & Insights")
    
    # Key metrics
    num_patients = data.shape[0]
    num_features = data.shape[1] - 1  # Exclude target
    disease_rate = (data['HeartDisease'].mean() * 100).round(1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", num_patients)
    col2.metric("Number of Features", num_features)
    col3.metric("Heart Disease Rate", f"{disease_rate}%")

    # Key observations
    st.write(
        "* Most patients are aged 50–65.\n"
        "* High cholesterol and elevated blood pressure appear more frequently in patients with heart disease.\n"
        "* Early patterns in this dataset can guide predictive modeling and risk stratification."
    )

    # Dataset preview
    st.markdown("#### Sample Data")
    st.dataframe(data.head(10))

    # Optional visual: top categorical features vs HeartDisease
    st.markdown("#### Feature Overview vs Disease Outcome")
    categorical_features = [col for col in data.columns if data[col].dtype == "object" or data[col].nunique() < 10]
    
    if categorical_features:
        selected_feature = st.selectbox(
            "Select a feature to visualize",
            options=categorical_features
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(
            data=data,
            x=selected_feature,
            hue='HeartDisease',
            order=data[selected_feature].value_counts().index,
            palette="viridis"
        )
        plt.title(f"{selected_feature} distribution by Heart Disease")
        st.pyplot(fig)
    
    # ---- Now What: How this page guides decisions ----
    st.markdown("### How to use this page")
    st.info(
        "* Understand the dataset structure and key metrics.\n"
        "* Observe patterns in high-risk vs low-risk patients.\n"
        "* Prepare for deeper analysis in the Exploratory Data Analysis page.\n"
        "* Get context for model training and later patient risk predictions."
    )


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
