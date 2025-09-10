import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, RocCurveDisplay

# Load trained pipelines and performance summary
baseline_pipeline = joblib.load("models/baseline_pipeline.joblib")
tuned_pipeline = joblib.load("models/tuned_pipeline.joblib")
performance_df = pd.read_csv("models/model_performance_summary.csv")  # includes metrics: accuracy, recall, precision, AUC

def page_model_training_body():
    st.write("### Model Training: Baseline vs Tuned")

    st.info(
        f"* Here we compare the **baseline models** with the **tuned models** "
        f"trained on the cleaned dataset. The goal is to identify the best performing "
        f"classifier for predicting disease/churn (adapted to your project)."
    )

    # Performance Summary Table
    st.write("#### Performance Summary")
    st.dataframe(performance_df.style.highlight_max(subset=["Recall", "AUC"], color='lightgreen'))

    st.write("---")

    # Radio button to choose model for further analysis
    selected_model = st.radio(
        "Select model to visualize details:",
        ("Baseline", "Tuned")
    )

    if selected_model == "Baseline":
        pipeline = baseline_pipeline
    else:
        pipeline = tuned_pipeline

    # Load test data for visualizations
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability for positive class

    st.write(f"#### Confusion Matrix: {selected_model} Model")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write(f"#### ROC Curve: {selected_model} Model")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    st.pyplot(fig)

    st.write(f"#### Feature Importance: {selected_model} Model")
    # Only works if pipeline has feature_importances_ (tree-based model)
    try:
        feature_importances = pipeline.named_steps['classifier'].feature_importances_
        features = X_test.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
    except AttributeError:
        st.warning("Feature importance not available for this model type.")
    
    st.success(
        f"* Using these visualizations, users can compare baseline and tuned models, "
        f"identify which features drive predictions, and understand the trade-offs "
        f"between recall, precision, and AUC."
    )
