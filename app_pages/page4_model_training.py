import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, RocCurveDisplay

# Load saved models & performance summary
baseline_models = joblib.load("/workspaces/Heart_disease_risk_predictor/outputs/models/baseline_models.pkl")  
performance_df = pd.read_csv("/workspaces/Heart_disease_risk_predictor/outputs/models/model_performance_summary.csv")

# Load test splits
X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
    "/workspaces/Heart_disease_risk_predictor/outputs/models/splits.pkl"
)

def page_model_training_body():
    st.write("### Model Training: Baseline vs Tuned")

    st.info(
        "* This page compares **baseline models** (Notebook 4) with **tuned models** (Notebook 5). "
        "The goal is to evaluate which performs best for predicting heart disease risk."
    )

    # Performance Summary Table
    st.write("#### Performance Summary (Validation Set)")
    st.dataframe(performance_df.style.highlight_max(subset=["ROC-AUC", "F1"], color='lightgreen'))

    st.write("---")

    # Select model to visualize
    model_choice = st.selectbox(
        "Select a baseline model to visualize on the test set:",
        list(baseline_models.keys())
    )

    pipeline = baseline_models[model_choice]

    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    st.write(f"#### Confusion Matrix: {model_choice}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write(f"#### ROC Curve: {model_choice}")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    st.pyplot(fig)

