import streamlit as st
import pandas as pd
from src.model_utils import enhanced_predict, load_pipeline  # assuming you have these helper functions

# Load the best trained pipeline
pipeline_best = load_pipeline("/workspaces/Heart_disease_risk_predictor/outputs/models/deployment/best_model_pipeline.pkl")

def page_inference_tool_body():

    st.write("### 🩺 Patient Risk Prediction Tool")
    st.info(
        "Input patient clinical data to predict heart disease risk using the best trained model. "
        "The tool returns predicted class, probability, risk band, recommendations, and top contributing features."
    )

    st.markdown("#### ⚙️ Enter Patient Data")

    # ---- Input widgets for patient data ----
    age = st.number_input("Age", min_value=0, max_value=120, value=55)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[1,2,3,4])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0,1])
    restecg = st.selectbox("Resting ECG (restecg)", options=[0,1,2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.radio("Exercise Induced Angina (exang)", options=[0,1])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1,2,3])
    ca = st.selectbox("Number of Major Vessels (ca)", options=[0,1,2,3,4])
    thal = st.selectbox("Thalassemia (thal)", options=[3,6,7])

    # ---- Convert inputs to DataFrame ----
    patient_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # ---- Predefined sample patients ----
    if st.checkbox("Use Example Patients"):
        sample_patient = {
            "age": 55, "sex": 1, "cp": 3, "trestbps": 240, "chol": 220, "fbs": 0,
            "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
        }
        high_risk_patient = {
            "age": 68, "sex": 1, "cp": 4, "trestbps": 180, "chol": 300, "fbs": 1,
            "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 3.0, "slope": 3, "ca": 2, "thal": 7
        }
        patient_choice = st.radio("Choose Example Patient:", ["Sample Patient", "High-Risk Patient"])
        if patient_choice == "Sample Patient":
            patient_data = pd.DataFrame([sample_patient])
        else:
            patient_data = pd.DataFrame([high_risk_patient])

    # ---- Run prediction ----
    if st.button("Run Prediction"):
        result = enhanced_predict(pipeline_best, patient_data)

        st.success("✅ Prediction Results")
        st.write(f"**Prediction:** {result['Prediction']}")
        st.write(f"**Probability:** {result['Probability']:.2f}%")
        st.write(f"**Risk Band:** {result['Risk Band']}")
        st.write(f"**Recommendation:** {result['Recommendation']}")

        if "Top Contributions" in result:
            st.markdown("**Top Contributing Features:**")
            st.table(result["Top Contributions"])
