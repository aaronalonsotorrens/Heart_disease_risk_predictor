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
            ["High-Risk Patient", "Low-Risk Patient"]
        )
        patient_data = pd.DataFrame([
            high_risk_patient if patient_choice=="High-Risk Patient"
            else low_risk_patient
        ])

    # ---- Run Prediction ----
    if st.button("Run Prediction"):
        # Rebuild patient_data here to ensure latest inputs are used
        if st.checkbox("Use Example Patients"):
            patient_choice = st.radio(
                "Choose Example Patient:",
                ["High-Risk Patient", "Low-Risk Patient"]
            )
            patient_data = pd.DataFrame([
                high_risk_patient if patient_choice=="High-Risk Patient"
                else low_risk_patient
            ])
        else:
            # Build from user inputs
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

    # ---- Preprocess & Predict ----
    processed_data = preprocess_input(patient_data, pipeline_best)
    result = enhanced_predict(pipeline_best, processed_data)

    # ---- Display Results ----
    st.success("✅ Prediction Results")
    st.write(f"**Prediction:** {result['Prediction']}")
    st.write(f"**Probability:** {result['Probability']*100:.2f}%")
    st.write(f"**Risk Band:** {result['Risk Band']}")
    st.write(f"**Recommendation:** {result['Recommendation']}")

    with st.expander("⚠️ Advanced Input (Full 22 Features)"):
        st.warning(
            "Optional advanced entry for testing all model features. "
            "This includes engineered and dataset-specific fields. "
            "If results indicate elevated risk, please consult a healthcare professional."
        )

        # ---- Core numeric inputs ----
        adv_id = st.number_input("Patient ID", 0, 9999, 0, key="adv_id")
        adv_age = st.number_input("Age", 0, 120, 55, key="adv_age")
        adv_trestbps = st.number_input("Resting Blood Pressure (trestbps)", 50, 250, 120, key="adv_trestbps")
        adv_chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200, key="adv_chol")
        adv_thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 60, 220, 150, key="adv_thalach")
        adv_oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1, key="adv_oldpeak")

        # ---- Engineered features manually input ----
        adv_chol_age_ratio = st.number_input("Cholesterol / Age Ratio (chol_age_ratio)", 0.0, 20.0, 3.5, 0.1)
        adv_oldpeak_thalach_ratio = st.number_input("Oldpeak / Thalach Ratio (oldpeak_thalach_ratio)", 0.0, 1.0, 0.01, 0.01)
        adv_age_trestbps = st.number_input("Age * Trestbps (age_trestbps)", 0, 50000, 9000)
        adv_thalch_oldpeak = st.number_input("Thalach * Oldpeak (thalch_oldpeak)", 0, 5000, 450)
        adv_age_group = st.number_input("Age Group (age_group, 0-5)", 0, 5, 4)

        # ---- One-hot encoded categorical features ----
        adv_sex_male = st.radio("Sex: Male?", [0,1], key="adv_sex_male")
        adv_dataset = st.selectbox(
            "Dataset Source",
            ["Default", "Hungary", "Switzerland", "VA Long Beach"],
            key="adv_dataset"
        )
        adv_cp = st.selectbox(
            "Chest Pain Type",
            ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
            key="adv_cp"
        )
        adv_fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0,1], key="adv_fbs")
        adv_restecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"],
            key="adv_restecg"
        )
        adv_exang = st.radio("Exercise Induced Angina", [0,1], key="adv_exang")

        # ---- Build patient dictionary ----
        adv_patient = pd.DataFrame([{
            "id": adv_id,
            "age": adv_age,
            "trestbps": adv_trestbps,
            "chol": adv_chol,
            "thalch": adv_thalach,
            "oldpeak": adv_oldpeak,
            "sex_Male": adv_sex_male,
            "dataset_Hungary": 1 if adv_dataset=="Hungary" else 0,
            "dataset_Switzerland": 1 if adv_dataset=="Switzerland" else 0,
            "dataset_VA Long Beach": 1 if adv_dataset=="VA Long Beach" else 0,
            "cp_typical angina": 1 if adv_cp=="Typical angina" else 0,
            "cp_atypical angina": 1 if adv_cp=="Atypical angina" else 0,
            "cp_non-anginal": 1 if adv_cp=="Non-anginal pain" else 0,
            "fbs_True": adv_fbs,
            "restecg_normal": 1 if adv_restecg=="Normal" else 0,
            "restecg_st-t abnormality": 1 if adv_restecg=="ST-T abnormality" else 0,
            "exang_True": adv_exang,
            "chol_age_ratio": adv_chol_age_ratio,
            "oldpeak_thalach_ratio": adv_oldpeak_thalach_ratio,
            "age_trestbps": adv_age_trestbps,
            "thalch_oldpeak": adv_thalch_oldpeak,
            "age_group": adv_age_group
        }])

        # ---- Advanced Example Patients ----
        if st.checkbox("Use Advanced Example Patients"):
            adv_high_risk_patient = {
                "id": 1,
                "age": 68, "trestbps": 180, "chol": 300, "thalch": 120, "oldpeak": 3.0,
                "sex_Male": 1,
                "dataset_Hungary": 0, "dataset_Switzerland": 0, "dataset_VA Long Beach": 1,
                "cp_typical angina": 0, "cp_atypical angina": 0, "cp_non-anginal": 0,
                "fbs_True": 1, "restecg_normal": 0, "restecg_st-t abnormality": 0, "exang_True": 1,
                "chol_age_ratio": 300/68,
                "oldpeak_thalach_ratio": 3.0/120,
                "age_trestbps": 68*180,
                "thalch_oldpeak": 120*3.0,
                "age_group": 4
            }

            adv_very_high_risk_patient = {
                "id": 2,
                "age": 75, "trestbps": 210, "chol": 380, "thalch": 100, "oldpeak": 4.8,
                "sex_Male": 1,
                "dataset_Hungary": 0, "dataset_Switzerland": 0, "dataset_VA Long Beach": 1,
                "cp_typical angina": 0, "cp_atypical angina": 0, "cp_non-anginal": 0,
                "fbs_True": 1, "restecg_normal": 0, "restecg_st-t abnormality": 1, "exang_True": 1,
                "chol_age_ratio": 380/75,
                "oldpeak_thalach_ratio": 4.8/100,
                "age_trestbps": 75*210,
                "thalch_oldpeak": 100*4.8,
                "age_group": 5
            }

            adv_patient_choice = st.radio(
                "Choose Advanced Example Patient:",
                ["High-Risk Patient", "Very High-Risk Patient"]
            )

            # Build DataFrame from chosen example patient
            adv_patient = pd.DataFrame([
                adv_high_risk_patient if adv_patient_choice == "High-Risk Patient"
                else adv_very_high_risk_patient
            ])

        # ---- Run Advanced Prediction ----
        if st.button("Run Advanced Prediction", key="run_adv_pred"):
            result_adv = enhanced_predict(pipeline_best, adv_patient)

            st.success("✅ Advanced Prediction Results")
            st.write(f"**Prediction:** {result_adv['Prediction']}")
            st.write(f"**Probability:** {result_adv['Probability']*100:.2f}%")
            st.write(f"**Risk Band:** {result_adv['Risk Band']}")
            st.write(f"**Recommendation:** {result_adv['Recommendation']}")