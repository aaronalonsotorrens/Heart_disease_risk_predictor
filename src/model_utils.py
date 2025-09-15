import joblib
import pandas as pd
import numpy as np


def load_pipeline(path: str):
    """
    Load a saved model pipeline from disk.
    """
    return joblib.load(path)


def preprocess_input(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Create engineered features, perform necessary one-hot encodings and align
    input columns with the pipeline's expected feature names.


    This function is defensive: it only creates features when the source
    columns exist and fills missing expected columns with zeros so the
    pipeline receives the exact schema it expects.
    """
    df = df.copy()

    # --- Engineered features (only if source cols exist) ---
    if "thalach" in df.columns:
        # keep original name used in some notebooks
        df["thalch"] = df["thalach"]

    if all(col in df.columns for col in ["chol", "age"]):
        # avoid division by zero
        df["chol_age_ratio"] = df["chol"] / df["age"].replace(0, np.nan)
        df["chol_age_ratio"] = df["chol_age_ratio"].fillna(0).clip(upper=10)

    if all(col in df.columns for col in ["oldpeak", "thalach"]):
        df["oldpeak_thalach_ratio"] = df["oldpeak"] / df["thalach"].replace(
            0, np.nan
        )
        df["oldpeak_thalach_ratio"] = (
            df["oldpeak_thalach_ratio"].fillna(0).clip(upper=10)
        )

    if all(col in df.columns for col in ["age", "trestbps"]):
        df["age_trestbps"] = df["age"] * df["trestbps"]

    if all(col in df.columns for col in ["thalach", "oldpeak"]):
        df["thalch_oldpeak"] = df["thalach"] * df["oldpeak"]

    # Age groups (same binning used during training)
    if "age" in df.columns:
        df["age_group"] = (
            pd.cut(
                df["age"],
                bins=[0, 30, 40, 50, 60, 70, 80, 120],
                labels=False,
            )
            .astype(float)
            .fillna(0)
        )

    # --- One-hot / binary encodings (only if source cols exist) ---
    if "sex" in df.columns:
        df["sex_Male"] = (df["sex"] == 1).astype(int)

    if "cp" in df.columns:
        df["cp_typical angina"] = (df["cp"] == 1).astype(int)
        df["cp_atypical angina"] = (df["cp"] == 2).astype(int)
        df["cp_non-anginal"] = (df["cp"] == 3).astype(int)
        df["cp_asymptomatic"] = (df["cp"] == 4).astype(int)

    if "fbs" in df.columns:
        df["fbs_True"] = (df["fbs"] == 1).astype(int)

    if "restecg" in df.columns:
        df["restecg_normal"] = (df["restecg"] == 0).astype(int)
        df["restecg_st-t abnormality"] = (df["restecg"] == 1).astype(int)
        df["restecg_lv_hypertrophy"] = (df["restecg"] == 2).astype(int)

    if "exang" in df.columns:
        df["exang_True"] = (df["exang"] == 1).astype(int)

    if "thal" in df.columns:
        df["thal_normal"] = (df["thal"] == 3).astype(int)
        df["thal_fixed"] = (df["thal"] == 6).astype(int)
        df["thal_reversible"] = (df["thal"] == 7).astype(int)

    # Provide dataset one-hot placeholders (training used these columns)
    for source in ["Hungary", "Switzerland", "VA Long Beach"]:
        col = f"dataset_{source}"
        if col not in df.columns:
            df[col] = 0

    # --- Final alignment with pipeline expected features ---
    # pipeline.feature_names_in_ is the canonical order used when training
    pipeline_features = list(pipeline.feature_names_in_)

    for col in pipeline_features:
        if col not in df.columns:
            # fill missing columns with 0 so the pipeline receives a complete schema
            df[col] = 0

    # Reorder and return only the features the pipeline expects
    df = df[pipeline_features]

    return df


def enhanced_predict(pipeline, df: pd.DataFrame):
    """
    Make predictions for heart disease risk with enhanced output.

    Returns a dictionary with Prediction, Probability, Risk Band, Recommendation.
    """
    prob = pipeline.predict_proba(df)[:, 1][0]
    pred = int(prob >= 0.5)

    # Risk band
    if prob < 0.3:
        risk_band = "Low"
        recommendation = "Maintain healthy lifestyle."
    elif prob < 0.5:
        risk_band = "Medium"
        recommendation = "Consult a doctor and monitor risk factors."
    elif prob < 0.6:
        risk_band = "High"
        recommendation = "Seek medical attention and improve healthy lifestyle"
    else:
        risk_band = "Very High"
        recommendation = "Immediate medical attention recommended."

    # Optional: top contributing features
    try:
        import shap

        explainer = shap.Explainer(pipeline.named_steps["classifier"])
        shap_values = explainer(df)
        top_contrib = (
            pd.DataFrame(
                {"Feature": df.columns, "Contribution": shap_values.values[0]}
            )
            .sort_values("Contribution", key=abs, ascending=False)
            .head(5)
        )
    except Exception:
        top_contrib = None

    return {
        "Prediction": pred,
        "Probability": round(prob, 3),
        "Risk Band": risk_band,
        "Recommendation": recommendation,
        "Top Contributions": top_contrib,
    }
