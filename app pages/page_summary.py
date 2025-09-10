import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # Text based on README file - "Dataset Content" section
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **patient** is a person who has provided clinical and lifestyle information.\n"
        f"* A **prospect** is a new patient whose risk is not yet known.\n"
        f"* A **high-risk patient** is someone predicted to have a high likelihood of "
        f"developing heart disease based on the dataset features.\n"
        f"we consider **age** and other clinical factors as key indicators.\n\n"
        f"**Project Dataset**\n"
        f"* The dataset represents patients with different attributes collected "
        f"during medical examination and lifestyle assessment. It contains information such as:\n"
        f"    - Demographics (age, sex)\n"
        f"    - Clinical measures (blood pressure, cholesterol, fasting blood sugar)\n"
        f"    - Heart-related test results (ECG, maximum heart rate, exercise-induced angina)\n"
        f"    - Target variable: presence or absence of heart disease"
    )

    # Link to README file so users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/your-username/your-heart-disease-repo)."
    )

    # Copied/adapted from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client (a healthcare provider) is interested in understanding the "
        f"patterns from the patient dataset to learn the most relevant variables "
        f"correlated with heart disease.\n"
        f"* 2 - The client is interested in determining whether or not a given prospect "
        f"(new patient) is at risk of developing heart disease. "
        f"If so, the client is interested to know the risk band (low, medium, high). "
        f"Based on that, the healthcare provider could suggest potential preventive measures "
        f"or treatments."
    )