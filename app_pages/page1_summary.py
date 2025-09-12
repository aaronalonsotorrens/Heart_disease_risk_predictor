import streamlit as st
import pandas as pd

def page_summary_body(data: pd.DataFrame):
    """
    Page 1: Executive Summary with storytelling flow.
    """
    st.title("💡 Heart Disease Prediction Dashboard - Executive Summary")

    # ---- WHY: Problem / Business Context ----
    st.markdown("### Why: The Challenge")
    st.info(
        "Cardiovascular disease is one of the leading causes of death globally. "
        "Healthcare providers aim to identify high-risk patients early to prevent adverse outcomes."
    )

    # ---- So What: Data Insights ----
    st.markdown("### So What: Key Data Insights")
    
    # Basic KPIs
    total_patients = data.shape[0]
    num_features = data.shape[1] - 1  # excluding HeartDisease
    disease_rate = (data['HeartDisease'].mean() * 100).round(1)
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Patients", total_patients)
    kpi2.metric("Number of Features", num_features)
    kpi3.metric("Heart Disease Rate", f"{disease_rate}%")
    
    # Highlight most important features (quick preview from correlation or domain knowledge)
    st.markdown("#### Key Risk Factors Identified")
    st.write(
        "- **Age**: Older patients have higher risk.\n"
        "- **Chest Pain Type (cp)**: Certain types strongly correlate with heart disease.\n"
        "- **Cholesterol & Blood Pressure**: High values increase risk.\n"
        "- **Exercise-Induced Angina**: Positive cases have higher likelihood of heart disease."
    )

    # ---- NOW WHAT: Recommended Actions ----
    st.markdown("### Now What: Next Steps / Actions")
    st.success(
        "Based on the dataset insights, healthcare providers could:\n"
        "1. Identify high-risk patients and prioritize preventive interventions.\n"
        "2. Design patient monitoring protocols focusing on key clinical indicators.\n"
        "3. Use predictive models to recommend lifestyle changes or further testing."
    )

    # ---- Link to Full Documentation ----
    st.markdown(
        "*For additional information, please visit and read the "
        "[Project README file](https://github.com/your-username/your-heart-disease-repo).*"
    )

    # Optional: Show a small table preview
    if st.checkbox("Preview Sample Data"):
        st.dataframe(data.head(5))
