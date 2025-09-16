import streamlit as st
import pandas as pd
from src.model_utils import enhanced_predict, load_pipeline, preprocess_input

# Load best pipeline
pipeline_best = load_pipeline(
    "outputs/models/"
    "deployment/best_model_pipeline.pkl"
)


def page_heart_risk_predictor_tool_body():
    st.title("🩺 Patient Risk Prediction Tool")

    st.info(
        (
            "Input patient clinical data to predict heart disease risk "
            "using the best trained model. The tool returns predicted "
            "class, probability, risk band, recommendations, and top "
            "contributing features."
        )
    )

    # ---- Feature Reference (collapsible) ----
    with st.expander("📘 Feature Reference"):
        st.markdown(
            (
                "* **age**: Patient age in years\n"
                "* **sex**: Biological sex (0 = Female, 1 = Male)\n"
                "* **cp (Chest Pain Type):**\n"
                "  - 1 = Typical angina\n"
                "  - 2 = Atypical angina\n"
                "  - 3 = Non-anginal pain\n"
                "  - 4 = Asymptomatic\n"
                "* **trestbps**: Resting blood pressure (mm Hg)\n"
                "* **chol**: Serum cholesterol (mg/dl)\n"
                "* **fbs (Fasting Blood Sugar):** 1 = True (>120 mg/dl), "
                "0 = False\n"
                "* **restecg (Resting ECG results):**\n"
                "  - 0 = Normal\n"
                "  - 1 = ST-T wave abnormality\n"
                "  - 2 = Left ventricular hypertrophy\n"
                "* **thalach**: Maximum heart rate achieved\n"
                "* **exang (Exercise Induced Angina):** 1 = Yes, 0 = No\n"
                "* **oldpeak**: ST depression induced by exercise relative "
                "to rest\n"
                "* **slope (Slope of peak exercise ST segment):**\n"
                "  - 1 = Upsloping\n"
                "  - 2 = Flat\n"
                "  - 3 = Downsloping\n"
                "* **ca**: Number of major vessels (0–4) colored "
                "by fluoroscopy\n"
                "* **thal (Thalassemia):**\n"
                "  - 3 = Normal\n"
                "  - 6 = Fixed defect\n"
                "  - 7 = Reversible defect"
            )
        )

    # ---- Input widgets ----
    age = st.number_input("Age (years)", 0, 120, 55)
    sex = st.radio(
        "Sex",
        [0, 1],
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)",
        key="sex",
    )
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical angina (1)",
            2: "Atypical angina (2)",
            3: "Non-anginal pain (3)",
            4: "Asymptomatic (4)",
        }[x],
        key="cp",
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)", 50, 250, 120
    )
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio(
        "Fasting Blood Sugar > 120 mg/dl (fbs)",
        [0, 1],
        format_func=lambda x: "False (0)" if x == 0 else "True (1)",
        key="fbs",
    )
    restecg = st.selectbox(
        "Resting ECG (restecg)",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Normal (0)",
            1: "ST-T wave abnormality (1)",
            2: "Left ventricular hypertrophy (2)",
        }[x],
        key="restecg",
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved", 60, 220, 150
    )
    exang = st.radio(
        "Exercise Induced Angina (exang)",
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        key="exang",
    )
    oldpeak = st.number_input(
        "ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Upsloping (1)",
            2: "Flat (2)",
            3: "Downsloping (3)",
        }[x],
        key="slope",
    )

    ca = st.selectbox(
        "Number of Major Vessels Colored by Fluoroscopy",
        [0, 1, 2, 3, 4],
        format_func=lambda x: f"{x} vessel(s)",
        key="ca",
    )

    thal = st.selectbox(
        "Thalassemia (thal)",
        [3, 6, 7],
        format_func=lambda x: {
            3: "Normal (3)",
            6: "Fixed defect (6)",
            7: "Reversible defect (7)",
        }[x],
        key="thal",
    )

    # ---- Example Patients ----
    use_example = st.checkbox(
        "Use Example Patients", key="use_example_patients"
    )
    if use_example:
        high_risk_patient = {
            "age": 68, "sex": 1, "cp": 4, "trestbps": 180, "chol": 300,
            "fbs": 1, "restecg": 2, "thalach": 120, "exang": 1,
            "oldpeak": 3.0, "slope": 3, "ca": 2, "thal": 7,
        }
        low_risk_patient = {
            "age": 45, "sex": 0, "cp": 1, "trestbps": 110, "chol": 180,
            "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0,
            "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3,
        }

        patient_choice = st.radio(
            "Choose Example Patient:",
            ["High-Risk Patient", "Low-Risk Patient"],
            key="example_patient_choice",
        )

        patient_data = pd.DataFrame(
            [
                (
                    high_risk_patient
                    if patient_choice == "High-Risk Patient"
                    else low_risk_patient
                )
            ]
        )
    else:
        patient_data = pd.DataFrame(
            [
                {
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
                    "thal": thal,
                }
            ]
        )

    # ---- Run Prediction ----
    if st.button("Run Prediction", key="run_pred"):
        processed_data = preprocess_input(patient_data, pipeline_best)
        result = enhanced_predict(pipeline_best, processed_data)
        st.success("✅ Prediction Results")
        st.write(f"**Prediction:** {result['Prediction']}")
        st.write(f"**Probability:** {result['Probability']*100:.2f}%")
        st.write(f"**Risk Band:** {result['Risk Band']}")
        st.write(f"**Recommendation:** {result['Recommendation']}")

    # ---- Model Input Explanation ----
    st.info(
        (
            "ℹ️ **Model Input Explanation:**\n\n"
            "Default features focus on core clinical variables. "
            "Designed to flag patients needing attention (risk ~50%).\n\n"
            "For precise assessment, enable **Advanced Input**. "
            "Includes all features for improved prediction and probability "
            "resolution."
        )
    )
    # ---- Fully Advanced Input (All 22 features) ----
    with st.expander("⚠️ Full Advanced Input (All Features)"):
        st.warning(
            "Manually set all features. "
            "Use with caution for testing extreme/high-risk patients."
        )

        # ---- Core Clinical Features (same style as simple version) ----
        adv_id = st.number_input("Patient ID", 0, 9999, 999, key="adv_id")
        adv_age = st.number_input("Age (years)", 0, 120, 70, key="adv_age")

        adv_sex = st.radio(
            "Sex",
            [0, 1],
            format_func=lambda x: "Female (0)" if x == 0 else "Male (1)",
            key="adv_sex",
        )

        adv_cp = st.selectbox(
            "Chest Pain Type (cp)",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical angina (1)",
                2: "Atypical angina (2)",
                3: "Non-anginal pain (3)",
                4: "Asymptomatic (4)",
            }[x],
            key="adv_cp",
        )

        adv_trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)", 50, 250, 180, key="adv_trestbps"
        )
        adv_chol = st.number_input(
            "Serum Cholesterol (mg/dl)", 100, 600, 300, key="adv_chol"
        )

        adv_fbs = st.radio(
            "Fasting Blood Sugar > 120 mg/dl (fbs)",
            [0, 1],
            format_func=lambda x: "False (0)" if x == 0 else "True (1)",
            key="adv_fbs",
        )

        adv_restecg = st.selectbox(
            "Resting ECG (restecg)",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal (0)",
                1: "ST-T wave abnormality (1)",
                2: "Left ventricular hypertrophy (2)",
            }[x],
            key="adv_restecg",
        )

        adv_thalach = st.number_input(
            "Maximum Heart Rate Achieved", 60, 220, 100, key="adv_thalach"
        )

        adv_exang = st.radio(
            "Exercise Induced Angina (exang)",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            key="adv_exang",
        )

        adv_oldpeak = st.number_input(
            "ST Depression (oldpeak)", 0.0, 10.0, 4.0, 0.1, key="adv_oldpeak"
        )

        adv_slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping (1)",
                2: "Flat (2)",
                3: "Downsloping (3)",
            }[x],
            key="adv_slope",
        )

        adv_ca = st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy",
            [0, 1, 2, 3, 4],
            format_func=lambda x: f"{x} vessel(s)",
            key="adv_ca",
        )

        adv_thal = st.selectbox(
            "Thalassemia (thal)",
            [3, 6, 7],
            format_func=lambda x: {
                3: "Normal (3)",
                6: "Fixed defect (6)",
                7: "Reversible defect (7)",
            }[x],
            key="adv_thal",
        )

        # ---- Dataset Placeholders ----
        adv_dataset_hungary = st.selectbox(
            "Residency in Hungary dataset?",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            index=0,
            key="adv_dataset_hungary",
        )

        adv_dataset_switzerland = st.selectbox(
            "Residency in Switzerland dataset?",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            index=0,
            key="adv_dataset_switzerland",
        )

        adv_dataset_va = st.selectbox(
            "Residency in VA Long Beach dataset?",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            index=0,
            key="adv_dataset_va",
        )

        # ---- Engineered Features ----
        adv_thalch = adv_thalach
        adv_chol_age_ratio = round(min(adv_chol / max(adv_age, 1), 10), 3)
        adv_oldpeak_thalach_ratio = round(
            min(adv_oldpeak / max(adv_thalach, 1), 10), 3
        )
        adv_age_trestbps = adv_age * adv_trestbps
        adv_thalch_oldpeak = adv_thalach * adv_oldpeak
        adv_age_group = pd.cut(
            [adv_age], bins=[0, 30, 40, 50, 60, 70, 80, 120], labels=False
        )[0]

        # Assemble into a DataFrame
        full_patient = pd.DataFrame(
            [
                {
                    "id": adv_id,
                    "age": adv_age,
                    "trestbps": adv_trestbps,
                    "chol": adv_chol,
                    "thalch": adv_thalch,
                    "oldpeak": adv_oldpeak,
                    # One-hot encoded / boolean features
                    "sex_Male": int(adv_sex == 1),
                    "fbs_True": int(adv_fbs == 1),
                    "exang_True": int(adv_exang == 1),
                    "cp_typical angina": int(adv_cp == 1),
                    "cp_atypical angina": int(adv_cp == 2),
                    "cp_non-anginal": int(adv_cp == 3),
                    "cp_asymptomatic": int(adv_cp == 4),
                    "restecg_normal": int(adv_restecg == 0),
                    "restecg_st-t abnormality": int(adv_restecg == 1),
                    "restecg_lv_hypertrophy": int(adv_restecg == 2),
                    "slope": adv_slope,
                    "ca": adv_ca,
                    "thal": adv_thal,
                    "dataset_Hungary": int(adv_dataset_hungary),
                    "dataset_Switzerland": int(adv_dataset_switzerland),
                    "dataset_VA Long Beach": int(adv_dataset_va),
                    # Engineered features
                    "chol_age_ratio": adv_chol_age_ratio,
                    "oldpeak_thalach_ratio": adv_oldpeak_thalach_ratio,
                    "age_trestbps": adv_age_trestbps,
                    "thalch_oldpeak": adv_thalch_oldpeak,
                    "age_group": adv_age_group,
                }
            ]
        )

        # Prediction button
        if st.button("Run Full Advanced Prediction"):
            processed_data = preprocess_input(full_patient, pipeline_best)
            result = enhanced_predict(pipeline_best, processed_data)
            st.success("✅ Full Advanced Prediction Results")
            st.write(f"**Prediction:** {result['Prediction']}")
            st.write(f"**Probability:** {result['Probability']*100:.2f}%")
            st.write(f"**Risk Band:** {result['Risk Band']}")
            st.write(f"**Recommendation:** {result['Recommendation']}")
