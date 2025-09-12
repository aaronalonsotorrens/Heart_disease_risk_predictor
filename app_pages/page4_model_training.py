import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, RocCurveDisplay

sns.set_style("whitegrid")

# ---- Load models and data ----
baseline_models = joblib.load("/workspaces/Heart_disease_risk_predictor/outputs/models/baseline_models.pkl")  
performance_df = pd.read_csv("/workspaces/Heart_disease_risk_predictor/outputs/models/model_performance_summary.csv")
X_train, X_val, X_test, y_train, y_val, y_test = joblib.load("/workspaces/Heart_disease_risk_predictor/outputs/models/splits.pkl")

# Page title
st.write("### Model Training: Baseline vs Tuned Models")

# ---- Why ----
st.info(
    "Goal: Evaluate different models to identify the best approach for predicting heart disease risk. "
    "Selecting the right model is crucial to make reliable predictions for patients."
)

# ---- So What ----
st.write("#### Model Performance on Validation Set")
st.dataframe(
    performance_df.style.highlight_max(subset=["ROC-AUC", "F1"], color='lightgreen')
)
st.write(
    "Observation: Models with higher ROC-AUC and F1 scores are better at distinguishing between patients with and without heart disease."
)

# ---- Select a model for deeper analysis ----
model_choice = st.selectbox("Select a baseline model to visualize on the test set:", list(baseline_models.keys()))
pipeline = baseline_models[model_choice]

# ---- Confusion Matrix ----
st.write(f"#### Confusion Matrix: {model_choice}")
y_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
st.info(
    "Interpretation: The confusion matrix shows how many patients were correctly classified vs misclassified. "
    "Focus on minimizing false negatives (high-risk patients predicted as low-risk)."
)

# ---- ROC Curve ----
st.write(f"#### ROC Curve: {model_choice}")
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
st.pyplot(fig)
st.info(
    "Interpretation: The ROC curve shows the trade-off between sensitivity (recall) and specificity. "
    "A higher area under the curve (AUC) indicates better model performance."
)

# ---- Now What ----
st.success(
    "Next Steps: Use insights from these baseline models to guide hyperparameter tuning, feature selection, "
    "and advanced pipelines. Focus on models that balance high recall and precision for patient safety."
)

