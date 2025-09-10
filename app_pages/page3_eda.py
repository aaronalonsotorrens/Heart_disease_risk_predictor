import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset (replace with your cleaned CSV / data loader)
@st.cache_data
def load_data():
    df = pd.read_csv("/workspaces/Heart_disease_risk_predictor/outputs/datasets/cleaned/heart_disease_cleaned.csv")  # Example path
    return df

df = load_data()

def page_eda_body():
    st.write("### Exploratory Data Analysis")
    st.info(
        "This page provides insights into the dataset features, distributions, "
        "and their relationships with the target variable (Heart Disease)."
    )

    # Dataset preview
    if st.checkbox("Preview Dataset"):
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
        st.dataframe(df.head(10))

    st.write("---")

    # Feature distribution comparison
    st.write("#### Feature Distributions vs Disease Outcome")

    target_var = "target"  # Replace with your target column name
    features = [col for col in df.columns if col != target_var]

    selected_feature = st.selectbox(
        "Select a feature to visualize against Disease Outcome", features
    )

    if df[selected_feature].dtype == "object" or df[selected_feature].nunique() < 10:
        # Categorical feature plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x=selected_feature, hue=target_var,
                      order=df[selected_feature].value_counts().index)
        plt.title(f"{selected_feature} vs {target_var}")
        st.pyplot(fig)
    else:
        # Numerical feature plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x=selected_feature, hue=target_var, kde=True, element="step")
        plt.title(f"{selected_feature} vs {target_var}")
        st.pyplot(fig)

    st.write("---")

    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.write("Correlation heatmap between features:")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        st.pyplot(fig)

    st.write("---")

    # Optional interactive Plotly scatter for numerical features
    numeric_features = df.select_dtypes(include="number").columns.tolist()
    if st.checkbox("Interactive Scatter Plot"):
        x_axis = st.selectbox("X-axis", numeric_features, index=0)
        y_axis = st.selectbox("Y-axis", numeric_features, index=1)

        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=target_var,
            title=f"{y_axis} vs {x_axis} colored by {target_var}",
            height=500
        )
        st.plotly_chart(fig)
