import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_style("whitegrid")

def page_advanced_experiments_body():
    st.write("### Advanced Experiments & Best Pipeline Selection")

    st.info(
        "This page presents the advanced experiments conducted for model optimization, "
        "including hyperparameter tuning, ensemble methods, and pipeline comparison. "
        "We highlight the best performing pipeline chosen for deployment."
    )

    # ---- Load Model Comparison Data ----
    model_comparison_df = pd.read_csv(
        "/workspaces/Heart_disease_risk_predictor/outputs/models/final_results.csv"
    )
    # Normalize column names: lowercase + replace "-" with "_"
    model_comparison_df.columns = (
        model_comparison_df.columns.str.lower().str.replace("-", "_")
    )
    # Now columns are: model, dataset, accuracy, precision, recall, f1, roc_auc

    st.write("#### Model Comparison Summary")
    st.dataframe(model_comparison_df)

    # ---- Highlight Best Pipeline ----
    best_model = model_comparison_df.loc[model_comparison_df['recall'].idxmax()]
    st.success(
        f"**Best Pipeline Selected:** {best_model['model']} \n\n"
        f"Performance Metrics: Accuracy = {best_model['accuracy']:.2f}, "
        f"Recall = {best_model['recall']:.2f}, Precision = {best_model['precision']:.2f}, "
        f"AUC = {best_model['roc_auc']:.2f}"
    )

    st.write("---")

    # ---- Visualize Metrics per Model ----
    st.write("#### Comparison of Key Metrics Across Pipelines")
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

    st.write(
        "* The bar chart above shows the performance of each model across multiple metrics. "
        "The selected best pipeline maximizes recall while maintaining good precision and AUC."
    )

    # ---- Optional: Display Pipeline Steps ----
    st.write("#### Chosen Pipeline Steps")
    st.text(
        "Example pipeline steps:\n"
        "1. Data preprocessing: imputation + scaling\n"
        "2. Feature selection: top k features\n"
        "3. Classifier: XGBoost with tuned hyperparameters\n"
        "4. Cross-validation: Stratified K-Fold\n"
        "5. Final fit on full training data"
    )
    st.info(
        "This final pipeline is ready for deployment in the inference tool."
    )
