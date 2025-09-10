import joblib
import pandas as pd
import numpy as np

def load_pipeline(path: str):
    """
    Load a saved model pipeline from disk.
    """
    return joblib.load(path)


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
