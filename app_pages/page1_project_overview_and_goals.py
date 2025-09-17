import streamlit as st
import pandas as pd


def page_project_overview_and_goals_body(data: pd.DataFrame):

    st.title(
        "💡 Heart Disease Prediction Dashboard - Executive Summary & Overview"
    )

    # ---- WHY: Problem / Business Context ----
    st.markdown("### Why: The Challenge")
    st.info(
        "Cardiovascular disease is the **leading cause of death worldwide**. "
        "Healthcare providers aim to identify high-risk patients early to "
        "prevent adverse outcomes, reduce hospitalizations, and lower "
        "healthcare costs."
    )

    # ---- Client Requirements ----
    st.subheader("🎯 Client Requirements")
    st.write(
        "- Provide a dashboard for clinicians to explore "
        "heart disease risk factors.\n"
        "- Deliver predictive ML model(s) to identify high-risk patients.\n"
        "- Include interpretability so clinicians understand key features.\n"
        "- Provide actionable recommendations for patient monitoring "
        "and prevention."
    )
    st.subheader("💡 Project Hypotheses")
    st.write(
        "**Hypothesis 1 (Analytics-driven):** Certain clinical "
        "features such as age, "
        "cholesterol, resting blood pressure, and chest pain type are "
        "significantly associated with heart disease.\n\n"
        "**Hypothesis 2 (ML-driven):** A machine learning model using "
        "these clinical "
        "features can accurately predict heart disease risk with "
        "ROC-AUC ≥ 0.80 and F1 ≥ 0.80."
    )

    # ---- So What: Dataset Summary & Insights ----
    st.markdown("### Dataset Summary & Insights")

    # Key dataset metrics (KPIs)
    total_patients = data.shape[0]
    num_features = data.shape[1] - 1  # excluding target
    disease_rate = (data["HeartDisease"].mean() * 100).round(1)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Patients", total_patients)
    kpi2.metric("Number of Features", num_features)
    kpi3.metric("Heart Disease Rate", f"{disease_rate}%")

    # Key observations / narrative insights
    st.markdown("#### Key Observations from Dataset")
    st.write(
        "- Most patients are aged **50–65**.\n"
        "- **High cholesterol** and **elevated blood pressure** appear "
        "more frequently in patients with heart disease.\n"
        "- Early patterns in this dataset suggest risk stratification is "
        "possible using clinical features.\n"
        "- Helps guide predictive modeling and preventive interventions."
    )

    # Highlight most important features
    st.markdown("#### Key Risk Factors Identified")
    st.write(
        "- **Age**: Older patients have higher risk.\n"
        "- **Chest Pain Type (cp)**: Certain types strongly correlate with "
        "heart disease.\n"
        "- **Cholesterol & Blood Pressure**: High values increase risk.\n"
        "- **Exercise-Induced Angina (exang)**: Positive cases show higher "
        "likelihood of heart disease.\n"
        "- **ST depression (oldpeak) & Maximum Heart Rate (thalach)**: Key "
        "predictors from numeric correlations.\n"
        "- **Thalassemia (thal) & Number of major vessels (ca)**: Strong "
        "categorical associations with disease outcome."
    )

    # Optional: Show a small table preview
    if st.checkbox("Preview Sample Data"):
        st.dataframe(data.head(5))

    # ---- NOW WHAT: Recommended Actions ----
    st.markdown("### Now What: Recommended Actions")
    st.success(
        "Based on these insights, healthcare providers could:\n"
        "1. Identify high-risk patients and prioritize preventive "
        "interventions.\n"
        "2. Design monitoring protocols focusing on key clinical "
        "indicators.\n"
        "3. Leverage predictive models to recommend lifestyle changes "
        "or further testing.\n"
        "4. Use insights to allocate resources effectively and improve "
        "patient outcomes."
    )

    # ---- Link to Full Documentation ----
    st.markdown(
        "*For additional information, please visit and read the "
        "[Project README file](https://github.com/aaronalonsotorrens/"
        "Heart_disease_risk_predictor).*"
    )

    # ---- Key Medical Terms / Glossary ----
    st.markdown("### 📖 Key Medical Terms")
    st.info(
        "* **cp**: Chest pain type (4 values)\n"
        "* **trestbps**: Resting blood pressure (mm Hg)\n"
        "* **chol**: Serum cholesterol (mg/dl)\n"
        "* **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)\n"
        "* **restecg**: Resting electrocardiographic results (0, 1, 2)\n"
        "* **thalach**: Maximum heart rate achieved\n"
        "* **exang**: Exercise induced angina (1 = yes, 0 = no)\n"
        "* **oldpeak**: ST depression induced by exercise relative to rest\n"
        "* **slope**: Slope of the peak exercise ST segment\n"
        "* **ca**: Number of major vessels (0–3) colored by fluoroscopy\n"
        "* **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = "
        "reversible defect)"
    )
