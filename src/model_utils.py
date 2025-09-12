import joblib
import pandas as pd
import numpy as np

def load_pipeline(path: str):
    """
    Load a saved model pipeline from disk.
    """
    return joblib.load(path)

def preprocess_input(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Create engineered features and align input columns with pipeline."""
    df = df.copy()
    
    # Example engineered features
    df["thalch"] = df["thalach"]
    df["chol_age_ratio"] = df["chol"] / df["age"]
    df["oldpeak_thalach_ratio"] = df["oldpeak"] / df["thalach"]
    df["age_trestbps"] = df["age"] * df["trestbps"]
    df["thalch_oldpeak"] = df["thalach"] * df["oldpeak"]
    df["age_group"] = pd.cut(df["age"], bins=[0,30,40,50,60,70,80,120], labels=False)
    
    # Ensure all columns expected by the pipeline exist
    pipeline_features = pipeline.feature_names_in_
    for col in pipeline_features:
        if col not in df.columns:
            df[col] = 0  # placeholder for missing columns
    
    # Keep only the columns pipeline expects
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
    elif prob < 0.7:
        risk_band = "Medium"
        recommendation = "Consult a doctor and monitor risk factors."
    else:
        risk_band = "High"
        recommendation = "Immediate medical attention recommended."

    # Optional: top contributing features
    try:
        import shap
        explainer = shap.Explainer(pipeline.named_steps['classifier'])
        shap_values = explainer(df)
        top_contrib = pd.DataFrame({
            'Feature': df.columns,
            'Contribution': shap_values.values[0]
        }).sort_values('Contribution', key=abs, ascending=False).head(5)
    except Exception:
        top_contrib = None

    return {
        "Prediction": pred,
        "Probability": round(prob, 3),
        "Risk Band": risk_band,
        "Recommendation": recommendation,
        "Top Contributions": top_contrib
    }
