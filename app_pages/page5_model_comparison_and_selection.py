import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def page_model_comparison_and_selection_body():
    st.title("Advanced Experiments & Best Pipeline Selection")

    st.info(
        (
            "This page evaluates the previously tuned models on the "
            "**holdout test set**. No new training or hyperparameter tuning "
            "is performed here. We compare performance metrics across "
            "pipelines to select the best one for deployment."
        )
    )

    # ---- Load Model Comparison Data ----
    model_comparison_df = pd.read_csv(
        "outputs/models/final_results.csv"
    )
    model_comparison_df.columns = (
        model_comparison_df.columns.str.lower().str.replace("-", "_")
    )

    # ---- Why → So What → Now What storytelling ----
    st.subheader("Why: Model Selection Matters")
    st.write(
        (
            "Different pipelines produce varying predictive performance. "
            "Selecting the best pipeline ensures high recall for detecting "
            "patients at risk, reducing the likelihood of missed diagnoses."
        )
    )

    st.subheader("So What: Comparative Performance Metrics")
    st.info(
        (
            "These metrics are evaluated on the **holdout test set**, which "
            "contains data that the models have **never seen during training "
            "or hyperparameter tuning**. This ensures the reported "
            "performance reflects how the model will behave on new, "
            "real-world patients rather than memorizing the training data."
        )
    )
    st.dataframe(model_comparison_df)

    # ---- Highlight Best Model ----
    best_model = model_comparison_df.loc[
        model_comparison_df["recall"].idxmax()
    ]
    st.success(
        (
            f"**Best Pipeline Selected:** {best_model['model']}\n\n"
            f"Performance Metrics:\n"
            f"Accuracy = {best_model['accuracy']:.2f}\n"
            f"Recall = {best_model['recall']:.2f}\n"
            f"Precision = {best_model['precision']:.2f}\n"
            f"F1 = {best_model['f1']:.2f}\n"
            f"AUC = {best_model['roc_auc']:.2f}"
        )
    )

    # ---- Now What: Actionable Insights ----
    st.subheader("Now What: Actionable Insights")
    st.write(
        (
            "This pipeline is ready to predict heart disease risk for new "
            "patients. It balances recall and precision, ensuring high-risk "
            "patients are flagged while minimizing false positives. Metrics "
            "from the holdout test set give confidence that these predictions "
            "generalize to unseen patients."
        )
    )

    # ---- Visualize Metrics Across Pipelines ----
    st.write("#### Model Performance Comparison Across Pipelines")
    melted_df = model_comparison_df.melt(
        id_vars="model",
        value_vars=["accuracy", "recall", "precision", "f1", "roc_auc"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=melted_df,
        x="model",
        y="value",
        hue="variable",
        palette="viridis"
    )
    plt.ylabel("Metric Value")
    plt.xlabel("Model / Pipeline")
    plt.title("Model Performance Comparison (Holdout Test Set)")
    st.pyplot(fig)

    st.info(
        (
            "This final pipeline is now ready for deployment in the inference "
            "tool to predict patient risk. Consistent preprocessing ensures "
            "that predictions remain reliable for new, unseen patients."
        )
    )

    # ---- Hypothesis 2 Validation ----
    st.subheader("✅ Hypothesis 2 Validation")
    st.write(
        f"We evaluated all tuned models on the holdout test set to assess "
        f"predictive performance:\n"
        f"- Best model: {best_model['model']}\n"
        f"- ROC-AUC = {best_model['roc_auc']:.2f}\n"
        f"- F1 = {best_model['f1']:.2f}\n\n"
        "This confirms Hypothesis 2: our ML pipeline achieves strong "
        "predictive performance, successfully identifying high-risk "
        "patients based on clinical features."
    )

    # ---- Next Steps ----
    st.success(
        (
            "### Next Steps\n"
            "- Integrate `best_model_pipeline.pkl` into an API or "
            "application.\n"
            "- Implement input validation and standardized preprocessing.\n"
            "- Add logging and monitoring for predictions in production.\n"
            "- Consider explainability tools (SHAP, permutation importance) "
            "for end-user interpretability.\n"
            "- Document the full workflow for reproducibility and regulatory "
            "compliance if required."
        )
    )
