import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, RocCurveDisplay

sns.set_style("whitegrid")


def page_model_development_and_evaluation_body():
    """Model Development and Evaluation"""

    # ---- Load models and performance data ----
    baseline_models = joblib.load(
        "outputs/models/baseline_models.pkl"
    )
    performance_df = pd.read_csv(
        "outputs/models/baseline_model_performance.csv"
    )
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        "outputs/models/splits.pkl"
    )

    st.title("Model Development page and Evaluation")
    # ---- Why / Goal ----
    st.info(
        ("Goal: Evaluate different **baseline models** to identify the best "
         "starting point for predicting heart disease risk. Selecting a "
         "strong baseline is crucial before tuning to ensure reliable "
         "predictions and minimize clinical risk. By examining metrics like "
         "ROC-AUC and F1, we can understand how well each baseline model "
         "distinguishes high-risk patients and balances false "
         "positives/negatives. Later, all models will be evaluated on the "
         "test set to guide further tuning and final model selection.")
    )

    # ---- Filter performance_df to show only baseline models ----
    baseline_model_names = list(baseline_models.keys())
    baseline_perf_df = performance_df[
        performance_df["Model"].isin(baseline_model_names)
    ]

    # ---- Baseline Model Performance Table ----
    st.write("#### Baseline Model Performance on Validation Set")
    st.dataframe(
        baseline_perf_df.style.highlight_max(
            subset=["ROC-AUC", "F1"], color="lightgreen"
        )
    )

    st.write(
        ("Observation: This table shows **only baseline models** evaluated on "
         "the validation set. Tuned models will be explored on Page 5. All "
         "models (baseline and tuned) are later evaluated on the test set for "
         "further tuning and final selection.")
    )

    # ---- Select a model for deeper analysis ----
    model_choice = st.selectbox(
        "Select a baseline model to visualize on the test set:",
        list(baseline_models.keys()),
    )
    pipeline = baseline_models[model_choice]

    # ---- Dynamic explanatory notes per model ----
    model_notes = {
        "Logistic Regression": (
            "Logistic Regression is a linear model that is easy to interpret. "
            "It achieves ROC-AUC ~0.81–0.82, showing good discrimination but"
            "may underperform on complex non-linear patterns. SMOTE "
            "(Synthetic Minority Over-sampling Technique) can slightly "
            "improve F1 by balancing class distribution, making the model"
            "more sensitive to rare high-risk cases."
        ),
        "Random Forest": (
            "Random Forest captures non-linear relationships and interactions "
            "effectively. With ROC-AUC ~0.88 and F1 ~0.84, it shows strong "
            "overall performance. The confusion matrix often reveals few "
            "false negatives, which is crucial for identifying "
            "high-risk patients."
        ),
        "XGBoost": (
            "XGBoost is a gradient boosting model optimized for complex "
            "feature interactions. Tuned versions achieve ROC-AUC ~0.89–0.90. "
            "It balances precision and recall well, making it robust for risk "
            "prediction in clinical datasets."
        ),
        "LightGBM": (
            "LightGBM is a fast, gradient boosting tree-based model. Tuned "
            "LightGBM achieves ROC-AUC ~0.90 and high F1. Its efficient "
            "handling of feature splits allows strong performance with large "
            "feature sets, highlighting key predictors effectively."
        ),
    }

    st.info(
        model_notes.get(
            model_choice,
            "Model selected. Examine confusion matrix and ROC curve below.",
        )
    )

    # ---- Confusion Matrix ----
    st.write(f"#### Confusion Matrix: {model_choice}")
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.info(
        ("Interpretation: Each cell shows counts of predictions vs actual "
         "outcomes. Pay special attention to the false negatives (patients "
         "with heart disease predicted as healthy) as these are clinically "
         "critical to minimize. High true positives and low false negatives "
         "indicate the model is reliable for identifying high-risk patients.")
    )

    # ---- ROC Curve ----
    st.write(f"#### ROC Curve: {model_choice}")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    st.pyplot(fig)
    st.info(
        ("Interpretation: The ROC curve illustrates the trade-off between "
         "sensitivity (recall) and specificity. Curves closer to the top-left "
         "indicate better performance. Higher area under the curve (AUC) "
         "confirms the models ability to distinguish patients with "
         "and without heart disease.")
    )

    # ---- Now What ----
    st.success(
        ("### 4️⃣ Next Steps\n"
         "- Use insights from these baseline models to guide "
         "**hyperparameter tuning** and **feature selection**.\n"
         "- Focus on models that **maximize recall** to reduce missed "
         "high-risk patients.\n"
         "- Evaluate tree-based models for feature importance to identify key "
         "clinical predictors.\n"
         "- Later, compare baseline vs tuned models (Page 5) to measure "
         "improvements in predictive performance.\n"
         "- Integrate the most reliable models into patient risk assessment "
         "workflows with confidence informed by these evaluations.")
    )
