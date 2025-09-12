import streamlit as st
import pandas as pd
from src.model_utils import enhanced_predict, load_pipeline

# Load the best trained pipeline
pipeline_best = load_pipeline(
    "/workspaces/Heart_disease_risk_predictor/outputs/models/deployment/best_model_pipeline.pkl"
)

def preprocess_input(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Create engineered features and align input columns with pipeline."""
    df = df.copy()
    
    # Engineered features
    df["thalch"] = df["thalach"]
    df["chol_age_ratio"] = df["chol"] / df["age"]
    df["oldpeak_thalach_ratio"] = df["oldpeak"] / df["thalach"]
    df["age_trestbps"] = df["age"] * df["trestbps"]
    df["thalch_oldpeak"] = df["thalach"] * df["oldpeak"]
    df["age_group"] = pd.cut(df["age"], bins=[0,30,40,50,60,70,80,120], labels=False)
    
    # Ensure all columns expected by pipeline exist
    pipeline_features = pipeline.feature_names_in_
    for col in pipeline_features:
        if col not in df.columns:
            df[col] = 0  # placeholder for missing columns
    
    # Keep only the columns pipeline expects
    df = df[pipeline_features]
    return df

def page_inference_tool_body():
    st.title("🩺 Patient Risk Prediction Tool")
    st.info(
        "Input patient clinical data to predict heart disease risk using the best trained model. "
        "The tool returns predicted class, probability, risk band, recommendations, and top contributing features."
    )

    # ---- Input widgets ----
    age = st.number_input("Age", min_value=0, max_value=120, value=55)
    sex = st.radio("Sex", options=[0,1], format_func=lambda x: "Female" if x==0 else "Male")
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
        patient_choice = st.radio("Choose Example Patient:", ["Sample Patient", "High-Risk Patient"])
        patient_data = pd.DataFrame([sample_patient if patient_choice=="Sample Patient" else high_risk_patient])

    # ---- Run Prediction ----
    if st.button("Run Prediction"):
        # Preprocess input to match pipeline
        patient_data_processed = preprocess_input(patient_data, pipeline_best)
        result = enhanced_predict(pipeline_best, patient_data_processed)

        st.success("✅ Prediction Results")
        st.write(f"**Prediction:** {result['Prediction']}")
        st.write(f"**Probability:** {result['Probability']:.2f}%")
        st.write(f"**Risk Band:** {result['Risk Band']}")
        st.write(f"**Recommendation:** {result['Recommendation']}")

        if "Top Contributions" in result and result["Top Contributions"] is not None:
            st.markdown("**Top Contributing Features:**")
            st.table(result["Top Contributions"])
