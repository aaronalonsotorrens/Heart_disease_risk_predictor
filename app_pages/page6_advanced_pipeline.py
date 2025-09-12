import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def page_advanced_experiments_body():
    st.write("### Advanced Experiments & Best Pipeline Selection")

    st.info(
        "This page presents advanced experiments conducted for model optimization, "
        "including hyperparameter tuning, ensemble methods, and pipeline comparison. "
        "We highlight the best performing pipeline chosen for deployment."
    )

    # ---- Load Model Comparison Data ----
    model_comparison_df = pd.read_csv(
        "/workspaces/Heart_disease_risk_predictor/outputs/models/final_results.csv"
    )
    model_comparison_df.columns = model_comparison_df.columns.str.lower().str.replace("-", "_")

    # Why → So What → Now What storytelling
    st.subheader("Why: Model Selection Matters")
    st.write(
        "Different pipelines produce varying predictive performance. "
        "Selecting the best pipeline ensures high recall for detecting patients at risk, "
        "reducing the likelihood of missed diagnoses."
    )

    st.subheader("So What: Comparative Performance Metrics")
    st.dataframe(model_comparison_df)

    best_model = model_comparison_df.loc[model_comparison_df['recall'].idxmax()]
    st.success(
        f"**Best Pipeline Selected:** {best_model['model']}\n\n"
        f"Performance Metrics: Accuracy = {best_model['accuracy']:.2f}, "
        f"Recall = {best_model['recall']:.2f}, Precision = {best_model['precision']:.2f}, "
        f"AUC = {best_model['roc_auc']:.2f}"
    )

    st.subheader("Now What: Actionable Insights")
    st.write(
        "This pipeline is ready to predict heart disease risk for new patients. "
        "It balances recall and precision, ensuring high-risk patients are flagged while "
        "minimizing false positives."
    )

    # Visualize Metrics Across Pipelines
    st.write("#### Model Performance Comparison Across Pipelines")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=model_comparison_df.melt(
            id_vars='model',
            value_vars=['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
        ),
        x='model',
        y='value',
        hue='variable',
        palette='viridis'
    )
    plt.ylabel("Metric Value")
    plt.xlabel("Model / Pipeline")
    plt.title("Model Performance Comparison")
    st.pyplot(fig)

    # Optional: Display Pipeline Steps
    st.subheader("Pipeline Steps")
    st.text(
        "Example pipeline steps:\n"
        "1. Data preprocessing: imputation + scaling\n"
        "2. Feature selection: top k features\n"
        "3. Classifier: XGBoost with tuned hyperparameters\n"
        "4. Cross-validation: Stratified K-Fold\n"
        "5. Final fit on full training data"
    )
    st.info(
        "This final pipeline is now ready for deployment in the inference tool to predict patient risk."
    )
