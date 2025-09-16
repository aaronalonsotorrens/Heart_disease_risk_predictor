import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, RocCurveDisplay

sns.set_style("whitegrid")


def get_feature_importance(pipeline, X_test):
    """Extract feature importance from tree-based pipelines."""
    final_estimator = pipeline[-1]  # last step in pipeline
    preprocessor = pipeline[:-1]  # all steps except final estimator

    if hasattr(final_estimator, "feature_importances_"):
        importances = final_estimator.feature_importances_
        try:
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = X_test.columns
        except Exception:
            feature_names = [f"feat_{i}" for i in range(len(importances))]

        fi_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        )
        fi_df = fi_df.sort_values("Importance", ascending=False)
        return fi_df
    else:
        return None


def page_model_tuning_and_insights_body():
    """Streamlit Page 4: Tuned Models Comparison & Feature Importance."""

    st.title("Tuned Models: Comparison & Feature Importance")

    # ---- Load models ----
    tuned_models = {
        "Logistic Regression": joblib.load(
            "outputs/models/"
            "tuned_models/best_logistic_regression_pipeline.pkl"
        ),
        "Random Forest": joblib.load(
            "outputs/models/"
            "tuned_models/best_random_forest_pipeline.pkl"
        ),
        "XGBoost": joblib.load(
            "outputs/models/"
            "tuned_models/best_xgboost_pipeline.pkl"
        ),
        "LightGBM": joblib.load(
            "outputs/models/"
            "tuned_models/best_lightgbm_pipeline.pkl"
        ),
    }

    performance_df = pd.read_csv(
        "outputs/models/"
        "model_performance_summary.csv"
    )

    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        "outputs/models/splits.pkl"
    )

    # ---- Hyperparameter Tuning Info ----
    st.info(
        ("**Hyperparameter tuning (GridSearchCV)** was applied to "
         "systematically optimize model parameters for each model. "
         "5-fold cross-validation on training data selected parameters that "
         "maximize F1. "
         "This improves ROC-AUC and reduces false negatives, which is "
         "critical for clinical risk prediction.")
    )

    # ---- Tuned Model Performance Table ----
    tuned_model_names = list(tuned_models.keys())
    tuned_perf_df = performance_df[
        performance_df["Model"].isin(tuned_model_names)
    ]

    st.subheader("Validation Set Performance: Tuned Models")
    st.dataframe(
        tuned_perf_df.style.highlight_max(
            subset=["ROC-AUC", "F1"], color="lightgreen"
        )
    )

    st.write(
        ("Observation: Metrics show improvements from tuning. "
         "Tree-based models benefit most from optimized hyperparameters.")
    )

    # ---- Select a model for deeper analysis ----
    model_choice = st.selectbox(
        "Select a tuned model to visualize on the test set:", tuned_model_names
    )
    pipeline = tuned_models[model_choice]

    # ---- Dynamic Notes per Model ----
    model_notes = {
        "Logistic Regression": (
            "Regularization tuning slightly improved balance between "
            "precision and recall. "
            "F1 = 0.7792, ROC-AUC = 0.8137. "
            "Performance is lower than tree-based models. "
            "Provides an interpretable baseline."
        ),
        "Random Forest": (
            "Tuning max_depth, min_samples_split, and n_estimators increased "
            "F1 to 0.8447 and ROC-AUC to 0.8829. "
            "Predictions are stable and discrimination is strong. "
            "Model reliably identifies high-risk patients."
        ),
        "XGBoost": (
            "Optimized learning rate and max_depth led to ROC-AUC = 0.8905 "
            "and F1 = 0.8428. "
            "Slight improvement over Random Forest. "
            "F1 remains strong and reliable."
        ),
        "LightGBM": (
            "Tuned parameters improved ROC-AUC to 0.8972 (highest reported). "
            "F1 = 0.8375 is slightly lower than RF/XGBoost. "
            "Best for distinguishing patients with heart disease. "
            "Minor F1 tradeoff is acceptable."
        ),
    }

    st.info(
        model_notes.get(
            model_choice,
            "Model selected. Examine confusion matrix and ROC curve below.",
        )
    )

    # ---- Confusion Matrix ----
    st.subheader(f"Confusion Matrix: {model_choice} (Test Set)")
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.info(
        ("**Interpretation**: Low false negatives indicate reliable "
         "identification of high-risk patients.")
    )

    # ---- ROC Curve ----
    st.subheader(f"ROC Curve: {model_choice} (Test Set)")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    st.pyplot(fig)

    st.info(
        ("**Interpretation**: Curves closer to top-left indicate better "
         "discrimination. Higher AUC confirms strong predictive ability.")
    )

    # ---- Feature Importance (Tree Models Only) ----
    if model_choice in ["Random Forest", "XGBoost", "LightGBM"]:
        fi_df = get_feature_importance(pipeline, X_test)
        if fi_df is not None:
            fi_df["Importance_norm"] = (
                fi_df["Importance"] / fi_df["Importance"].max()
            )

            st.subheader(f"Feature Importance: {model_choice} (Normalized)")

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(
                data=fi_df,
                y="Feature",
                x="Importance_norm",
                palette="viridis",
                ax=ax,
            )
            ax.set_xlabel("Normalized Importance (0-1)")
            ax.set_ylabel("Feature")
            plt.title(f"{model_choice} - Feature Importance (Normalized)")
            st.pyplot(fig)

        st.info(
            (
                "Observation: Feature importances are normalized "
                "(0 = least, 1 = most). Compare features only within the "
                "same model. Higher bars indicate greater influence. "
                "If 'num_id' appears highly ranked, it is an artifact. "
                "It does not contribute predictive power. "
                "Ablation tests confirm removing 'num_id' does not "
                "affect accuracy."
            )
        )
    else:
        st.info(
            ("Logistic Regression does not provide tree-based feature "
             "importance. Coefficients indicate magnitude and direction "
             "of effect. Large positive/negative coefficients correspond "
             "to stronger influence. Not directly comparable to tree-based "
             "importance scores.")
        )

    # ---- Next Steps ----
    st.success(
        ("### Next Steps\n"
         "- Compare tuned vs baseline metrics to quantify improvement.\n"
         "- Focus on models that maximize recall for clinical safety.\n"
         "- Use feature importance to guide feature selection"
         " and interpretability.\n"
         "- Deploy the best-performing tuned model in production "
         "with consistent preprocessing.")
    )
