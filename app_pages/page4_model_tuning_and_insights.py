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
    preprocessor = pipeline[:-1]    # all steps except final estimator
    
    if hasattr(final_estimator, "feature_importances_"):
        importances = final_estimator.feature_importances_
        # Attempt to get feature names from preprocessor
        try:
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = X_test.columns
        except:
            feature_names = [f"feat_{i}" for i in range(len(importances))]
        
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False)
        return fi_df
    else:
        return None


def page_model_tuning_and_insights_body():
    """Streamlit Page 5: Tuned Models Comparison & Feature Importance."""
    
    st.title("Tuned Models: Comparison & Feature Importance")
    
    # ---- Load models, splits, and performance summary ----
    tuned_models = {
        "Logistic Regression": joblib.load("/workspaces/Heart_disease_risk_predictor/outputs/models/tuned_models/best_logistic_regression_pipeline.pkl"),
        "Random Forest": joblib.load("/workspaces/Heart_disease_risk_predictor/outputs/models/tuned_models/best_random_forest_pipeline.pkl"),
        "XGBoost": joblib.load("/workspaces/Heart_disease_risk_predictor/outputs/models/tuned_models/best_xgboost_pipeline.pkl"),
        "LightGBM": joblib.load("/workspaces/Heart_disease_risk_predictor/outputs/models/tuned_models/best_lightgbm_pipeline.pkl")
    }

    performance_df = pd.read_csv("/workspaces/Heart_disease_risk_predictor/outputs/models/model_performance_summary.csv")
    
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        "/workspaces/Heart_disease_risk_predictor/outputs/models/splits.pkl"
    )

    # ---- General Hyperparameter Tuning Info ----
    st.info(
        "**Hyperparameter tuning (GridSearchCV)** was applied to systematically optimize model parameters for each model. "
        "We used 5-fold cross-validation on the training data to select the combination of parameters that maximizes F1 score. "
        "This improves discrimination (ROC-AUC) and reduces false negatives, which is critical for clinical risk prediction."
    )

    # ---- Filter performance_df to show only tuned models ----
    tuned_model_names = list(tuned_models.keys())
    tuned_perf_df = performance_df[performance_df['Model'].isin(tuned_model_names)]

    # ---- Tuned Model Performance Table ----
    st.subheader("Validation Set Performance: Tuned Models")
    st.dataframe(
        tuned_perf_df.style.highlight_max(subset=["ROC-AUC", "F1"], color='lightgreen')
    )
    st.write(
        "Observation: Metrics show improvements from tuning. Tree-based models benefit most from optimized hyperparameters."
    )

    # ---- Select a model for deeper analysis ----
    model_choice = st.selectbox("Select a tuned model to visualize on the test set:", tuned_model_names)
    pipeline = tuned_models[model_choice]

    # ---- Dynamic explanatory notes per model ----
    model_notes = {
        "Logistic Regression": (
            "Regularization tuning slightly improved balance between precision and recall, giving F1 = 0.7792 and ROC-AUC = 0.8137. "
            "Performance is lower than tree-based models but provides an interpretable baseline."
        ),
        "Random Forest": (
            "Tuning max_depth, min_samples_split, and n_estimators increased F1 to 0.8447 and ROC-AUC to 0.8829, "
            "giving stable predictions and good discrimination."
        ),
        "XGBoost": (
            "Optimized learning rate and max_depth led to ROC-AUC = 0.8905 and F1 = 0.8428, "
            "slightly improving discrimination over Random Forest while maintaining strong F1."
        ),
        "LightGBM": (
            "Tuned parameters improved ROC-AUC to 0.8972 (highest among models) while F1 = 0.8375 is slightly lower than RF/XGBoost. "
            "Best for distinguishing patients with heart disease but F1 tradeoff is minor."
        )
    }

    st.info(model_notes.get(model_choice, "Model selected. Examine confusion matrix and ROC curve below."))

    # ---- Confusion Matrix ----
    st.subheader(f"Confusion Matrix: {model_choice} (Test Set)")
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.info(
        "Interpretation: Low false negatives indicate reliable identification of high-risk patients."
    )

    # ---- ROC Curve ----
    st.subheader(f"ROC Curve: {model_choice} (Test Set)")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    st.pyplot(fig)
    st.info(
        "Interpretation: Curves closer to top-left indicate better discrimination; higher AUC confirms strong predictive ability."
    )

    # ---- Feature Importance (Tree Models Only) ----
    if model_choice in ["Random Forest", "XGBoost", "LightGBM"]:
        fi_df = get_feature_importance(pipeline, X_test)
        if fi_df is not None:
            # Normalize importance to 0-1 scale
            fi_df["Importance_norm"] = fi_df["Importance"] / fi_df["Importance"].max()
            
            st.subheader(f"Feature Importance: {model_choice} (Normalized)")
            
            # Plot using normalized values
            fig, ax = plt.subplots(figsize=(8,6))
            sns.barplot(
                data=fi_df,
                y="Feature",
                x="Importance_norm",
                palette="viridis",
                ax=ax
            )
            ax.set_xlabel("Normalized Importance (0-1)")
            ax.set_ylabel("Feature")
            plt.title(f"{model_choice} - Feature Importance (Normalized)")
            st.pyplot(fig)
            
        st.info(
            "Observation: Feature importances are normalized within the model (0 = least important, 1 = most important). "
            "Only compare features within the same model, not between models. "
            "Higher bars indicate greater influence on predictions. "
            "Note: If 'num_id' appears highly ranked, this is an artifact — it does not contribute predictive power. "
            "The ablation test carried out during development confirms that removing 'num_id' has no effect on model accuracy."
        )
    else:
        st.info(
            "Logistic Regression does not provide a tree-based feature importance because it is a linear model. "
            "Instead, each coefficient indicates the magnitude and direction of a feature's effect on the prediction. "
            "Large positive or negative coefficients correspond to stronger influence, but these are not directly comparable to tree-based importance scores."
        )

    # ---- Next Steps ----
    st.success(
        "### Next Steps\n"
        "- Compare tuned vs baseline metrics to quantify improvement.\n"
        "- Focus on models that maximize recall for clinical safety.\n"
        "- Use feature importance to guide feature selection and model interpretability.\n"
        "- Deploy the best-performing tuned model in production with consistent preprocessing."
    )
