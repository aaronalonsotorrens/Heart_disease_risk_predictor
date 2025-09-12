import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.model_utils import enhanced_predict, load_pipeline, preprocess_input

# Load best pipeline
pipeline_best = load_pipeline(
    "/workspaces/Heart_disease_risk_predictor/outputs/models/deployment/best_model_pipeline.pkl"
)

def page_inference_tool_body():
    st.title("🩺 Patient Risk Prediction Tool")
    st.info(
        "Input patient clinical data to predict heart disease risk using the best trained model. "
        "The tool returns predicted class, probability, risk band, recommendations, and top contributing features."
    )

    # ---- Feature Reference (collapsible) ----
    with st.expander("📘 Feature Reference"):
        st.markdown(
            "* **age**: Patient age in years\n"
            "* **sex**: Biological sex (0 = Female, 1 = Male)\n"
            "* **cp (Chest Pain Type):**\n"
            "   - 1 = Typical angina\n"
            "   - 2 = Atypical angina\n"
            "   - 3 = Non-anginal pain\n"
            "   - 4 = Asymptomatic\n"
            "* **trestbps**: Resting blood pressure (mm Hg)\n"
            "* **chol**: Serum cholesterol (mg/dl)\n"
            "* **fbs (Fasting Blood Sugar):** 1 = True (>120 mg/dl), 0 = False\n"
            "* **restecg (Resting ECG results):**\n"
            "   - 0 = Normal\n"
            "   - 1 = ST-T wave abnormality\n"
            "   - 2 = Left ventricular hypertrophy\n"
            "* **thalach**: Maximum heart rate achieved\n"
            "* **exang (Exercise Induced Angina):** 1 = Yes, 0 = No\n"
            "* **oldpeak**: ST depression induced by exercise relative to rest\n"
            "* **slope (Slope of peak exercise ST segment):**\n"
            "   - 1 = Upsloping\n"
            "   - 2 = Flat\n"
            "   - 3 = Downsloping\n"
            "* **ca**: Number of major vessels (0–4) colored by fluoroscopy\n"
            "* **thal (Thalassemia):**\n"
            "   - 3 = Normal\n"
            "   - 6 = Fixed defect\n"
            "   - 7 = Reversible defect"
        )

    # ---- Input widgets ----
    age = st.number_input("Age", 0, 120, 55)
    sex = st.radio("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", [1,2,3,4])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 50, 250, 120)
    chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [0,1])
    restecg = st.selectbox("Resting ECG (restecg)", [0,1,2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina (exang)", [0,1])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [1,2,3])
    ca = st.selectbox("Number of Major Vessels (ca)", [0,1,2,3,4])
    thal = st.selectbox("Thalassemia (thal)", [3,6,7])

    patient_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    # ---- Example Patients ----
    if st.checkbox("Use Example Patients"):
        sample_patient = {
            "age": 55, "sex": 1, "cp": 3, "trestbps": 240, "chol": 220, "fbs": 0,
            "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
        }
        high_risk_patient = {
            "age": 68, "sex": 1, "cp": 4, "trestbps": 180, "chol": 300, "fbs": 1,
            "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 3.0, "slope": 3, "ca": 2, "thal": 7
        }
        low_risk_patient = {
            "age": 45, "sex": 0, "cp": 1, "trestbps": 110, "chol": 180, "fbs": 0,
            "restecg": 0, "thalach": 180, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
        }

        patient_choice = st.radio(
            "Choose Example Patient:",
            ["Sample Patient", "High-Risk Patient", "Low-Risk Patient"]
        )
        patient_data = pd.DataFrame([
            sample_patient if patient_choice=="Sample Patient"
            else high_risk_patient if patient_choice=="High-Risk Patient"
            else low_risk_patient
        ])

    # ---- Run Prediction ----
    if st.button("Run Prediction"):
        processed_data = preprocess_input(patient_data, pipeline_best)
        result = enhanced_predict(pipeline_best, processed_data)

        st.success("✅ Prediction Results")
        st.write(f"**Prediction:** {result['Prediction']}")
        st.write(f"**Probability:** {result['Probability']:.1f}%")
        st.write(f"**Risk Band:** {result['Risk Band']}")
        st.write(f"**Recommendation:** {result['Recommendation']}")


        # ---- Top Contributions Table (optional) ----
        top_contrib = result.get("Top Contributions")
        if top_contrib is not None and not top_contrib.empty:
            st.markdown("**Top Contributing Features:**")
            st.table(top_contrib)

            # ---- Simple Bar Plot ----
            top_features = top_contrib.head(5)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(top_features["Feature"], top_features["Contribution"], color='royalblue')
            ax.set_xlabel("Contribution")
            ax.set_title("Top Feature Contributions")
            st.pyplot(fig)

