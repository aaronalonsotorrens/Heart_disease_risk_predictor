import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def page_ablation_study_body():
    st.write("### Ablation Study: Feature Impact on Model Performance")
    
    # ---- Why ----
    st.info(
        "Purpose: Identify which features have the most influence on model performance. "
        "Understanding feature importance helps refine models and ensures we focus on the variables that matter most for patient risk prediction."
    )
    
    # ---- Load Ablation Results ----
    baseline_ablation_df = pd.read_csv("/workspaces/Heart_disease_risk_predictor/outputs/models/baseline_ablation.csv")
    tuned_ablation_df = pd.read_csv("/workspaces/Heart_disease_risk_predictor/outputs/models/tuned_ablation.csv")
    
    # ---- User choice ----
    ablation_choice = st.radio("Choose Ablation Results to Display:", ["Baseline", "Tuned"])
    if ablation_choice == "Baseline":
        st.write("**Baseline Model Ablation Results**")
        ablation_df = baseline_ablation_df
    else:
        st.write("**Tuned Model Ablation Results**")
        ablation_df = tuned_ablation_df
    
    st.dataframe(ablation_df)
    
    # ---- So What ----
    st.write("#### Δ Accuracy by Feature Removal")
    ablation_df['delta_accuracy'] = ablation_df['Accuracy_with_id'] - ablation_df['Accuracy_without_id']
    ablation_df_sorted = ablation_df.sort_values("delta_accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=ablation_df_sorted,
        y="Model",
        x="delta_accuracy",
        palette="viridis"
    )
    plt.xlabel("Δ Accuracy (with vs without feature)")
    plt.ylabel("Model")
    plt.title(f"Feature Impact on {ablation_choice} Model Accuracy")
    st.pyplot(fig)

    st.write(
    "* Models with the largest Δ Accuracy are most sensitive to feature removal. "
    "For example, if removing **cholesterol** or **chest pain type (cp)** drastically reduces accuracy, "
    "these are critical predictors to measure accurately. "
    "This highlights the most predictive variables in the dataset and informs which clinical measurements "
    "should be prioritized in patient assessments."
    )

    st.info(
        "Observation: Features with the largest Δ Accuracy are the most important for predictions. "
        "This helps prioritize which variables should be retained in the model and which could be removed without hurting performance."
    )

    # ---- Now What ----
    st.success(
        "Next Steps: Use this feature importance insight to guide feature selection, improve model efficiency, "
        "and ensure that the predictive model is both interpretable and clinically meaningful."
    )
