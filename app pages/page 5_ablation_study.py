import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_style("whitegrid")

def page_ablation_study_body():
    st.write("### Ablation Study: Feature Impact on Model Performance")
    
    st.info(
        "This page shows how removing certain features from the dataset affects the "
        "performance of our ML models. It helps to understand which variables are critical "
        "for predictive accuracy."
    )
    
    # ---- Load Ablation Data ----
    # Example: ablation results saved from notebooks
    baseline_ablation_df = pd.read_csv("data/baseline_ablation.csv")  # columns: feature_dropped, accuracy
    tuned_ablation_df = pd.read_csv("data/tuned_ablation.csv")        # columns: feature_dropped, accuracy

    # ---- Choose model type ----
    ablation_choice = st.radio("Choose Ablation Results to Display:", ["Baseline", "Tuned"])

    if ablation_choice == "Baseline":
        st.write("**Baseline Model Ablation Results**")
        ablation_df = baseline_ablation_df
    else:
        st.write("**Tuned Model Ablation Results**")
        ablation_df = tuned_ablation_df

    st.dataframe(ablation_df)

    st.write("---")

    # ---- Visualize Δ Accuracy ----
    st.write("#### Impact on Accuracy per Feature Dropped")
    
    # Sort by accuracy drop
    ablation_df['delta_accuracy'] = ablation_df['accuracy'].max() - ablation_df['accuracy']
    ablation_df_sorted = ablation_df.sort_values("delta_accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=ablation_df_sorted,
        y="feature_dropped",
        x="delta_accuracy",
        palette="viridis"
    )
    plt.xlabel("Δ Accuracy (drop compared to max)")
    plt.ylabel("Feature Dropped")
    plt.title(f"Feature Impact on {ablation_choice} Model Accuracy")
    st.pyplot(fig)

    st.write(
        "* The bar chart above shows which features had the largest impact on model accuracy. "
        "Features with the highest Δ Accuracy are the most important for predictive performance."
    )
