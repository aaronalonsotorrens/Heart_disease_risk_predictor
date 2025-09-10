import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_style("whitegrid")

def page_ablation_study_body():
    st.write("### Ablation Study: Feature Impact on Model Performance")
    
    st.info(
        "This page shows how removing certain features (Notebook 5.9) affects "
        "model performance. It highlights which variables are most important."
    )
    
    # Load Ablation Results (saved in Notebook 5.9)
    baseline_ablation_df = pd.read_csv(
        "/workspaces/Heart_disease_risk_predictor/outputs/models/baseline_ablation.csv"
    )
    tuned_ablation_df = pd.read_csv(
        "/workspaces/Heart_disease_risk_predictor/outputs/models/tuned_ablation.csv"
    )

    # Choose which results to display
    ablation_choice = st.radio("Choose Ablation Results to Display:", ["Baseline", "Tuned"])

    if ablation_choice == "Baseline":
        st.write("**Baseline Model Ablation Results**")
        ablation_df = baseline_ablation_df
    else:
        st.write("**Tuned Model Ablation Results**")
        ablation_df = tuned_ablation_df

    st.dataframe(ablation_df)

    st.write("---")

    # Visualize Δ Accuracy
    st.write("#### Impact on Accuracy per Feature Dropped")
    
    ablation_df['delta_accuracy'] = ablation_df['Accuracy_with_id'] - ablation_df['Accuracy_without_id']
    ablation_df_sorted = ablation_df.sort_values("delta_accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=ablation_df_sorted,
        y="Model",
        x="delta_accuracy",
        palette="viridis"
    )
    plt.xlabel("Δ Accuracy (with_id - without_id)")
    plt.ylabel("Model")
    plt.title(f"Feature Impact on {ablation_choice} Model Accuracy")
    st.pyplot(fig)

    st.write(
        "* Models with the largest Δ Accuracy are most sensitive to feature removal. "
        "This highlights the most predictive variables in the dataset."
    )
