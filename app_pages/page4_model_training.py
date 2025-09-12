import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, RocCurveDisplay

sns.set_style("whitegrid")

def page_model_training_body():
    """Model Training page with baseline vs tuned model comparison."""

    # ---- Load models and performance data ----
    baseline_models = joblib.load("/workspaces/Heart_disease_risk_predictor/outputs/models/baseline_models.pkl")  
    performance_df = pd.read_csv("/workspaces/Heart_disease_risk_predictor/outputs/models/model_performance_summary.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        "/workspaces/Heart_disease_risk_predictor/outputs/models/splits.pkl"
    )

    st.title("Model Training: Baseline vs Tuned Models")

    # ---- Why ----
    st.info(
        "Goal: Evaluate different models to identify the best approach for predicting heart disease risk. "
        "Selecting the right model is crucial to make reliable predictions for patients."
    )

    # ---- Model Performance Table ----
    st.write("#### Model Performance on Validation Set")
    st.dataframe(
        performance_df.style.highlight_max(subset=["ROC-AUC", "F1"], color='lightgreen')
    )

    # ---- Select a model for deeper analysis ----
    model_choice = st.selectbox("Select a baseline model to visualize on the test set:", list(baseline_models.keys()))
    pipeline = baseline_models[model_choice]

    # ---- Dynamic explanatory notes ----
    model_notes = {
        "Logistic Regression": (
            "Logistic Regression is simple and interpretable. "
            "ROC-AUC is ~0.81–0.82; it may underperform with non-linear feature interactions. "
            "SMOTE helps improve F1 slightly by balancing classes."
        ),
        "Random Forest": (
            "Random Forest captures non-linear patterns well and has high ROC-AUC (~0.88). "
            "It balances sensitivity and specificity, generally reducing false negatives."
        ),
        "XGBoost": (
            "XGBoost is a gradient boosting tree model. Tuned versions achieve ROC-AUC ~0.89–0.90, "
            "excellent at handling complex feature interactions."
        ),
        "LightGBM": (
            "LightGBM is a fast, gradient boosting model. Tuned LightGBM has ROC-AUC ~0.90, "
            "high F1, and performs well with tree-based feature splits."
        )
    }

    st.info(model_notes.get(model_choice, "Model selected. Examine confusion matrix and ROC curve below."))

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
        "Interpretation: Focus on minimizing false negatives (high-risk patients predicted as low-risk)."
    )

    # ---- ROC Curve ----
    st.write(f"#### ROC Curve: {model_choice}")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    st.pyplot(fig)
    st.info(
        "Interpretation: Curves near top-left indicate better performance. "
        "Higher AUC confirms the model's ability to distinguish heart disease cases."
    )

    # ---- Now What ----
    st.success(
        "### 4️⃣ Next Steps\n"
        "- Use insights from baseline models to guide **hyperparameter tuning** and **feature selection**.\n"
        "- Focus on models that **maximize recall** to ensure high-risk patients are not missed.\n"
        "- Consider advanced pipelines and ensemble methods for potentially higher predictive accuracy.\n"
        "- Integrate these models into patient risk assessment tools with confidence informed by these evaluations."
    )