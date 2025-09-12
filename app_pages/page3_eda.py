import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_style("whitegrid")

def page_eda_body(data: pd.DataFrame):
    """
    Exploratory Data Analysis (EDA) page.
    Accepts a DataFrame 'data' with a 'HeartDisease' column.
    """

    HeartDisease_var = "HeartDisease"

    # Identify numeric and categorical features
    numeric_features = data.select_dtypes(include="number").columns.tolist()
    if HeartDisease_var in numeric_features:
        numeric_features.remove(HeartDisease_var)
    categorical_features = [col for col in data.columns if col not in numeric_features + [HeartDisease_var]]

    # ---- Page Title ----
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.info(
        "EDA helps us understand patterns in the dataset, identify key risk factors, "
        "and guide predictive modeling for patient heart disease risk."
    )

    # ---- Feature Distribution ----
    st.write("### 1️⃣ Feature Distributions vs Heart Disease Outcome")
    selected_feature = st.selectbox("Select a feature to explore:", numeric_features + categorical_features)
    st.write(f"Analyzing **{selected_feature}** and its relationship with heart disease:")

    # Visualization based on type
    if selected_feature in categorical_features or data[selected_feature].nunique() < 10:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=data,
            x=selected_feature,
            hue=HeartDisease_var,
            order=data[selected_feature].value_counts().index
        )
        plt.title(f"{selected_feature} vs {HeartDisease_var}")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=data,
            x=selected_feature,
            hue=HeartDisease_var,
            kde=True,
            element="step"
        )
        plt.title(f"{selected_feature} distribution by Heart Disease outcome")

    st.pyplot(fig)

    # ---- Interpretation ----
    if selected_feature in numeric_features:
        corr_value = data[selected_feature].corr(data[HeartDisease_var])
        strength = "weak" if abs(corr_value)<0.2 else "moderate" if abs(corr_value)<0.5 else "strong"
        st.info(
            f"Observation: **{selected_feature}** has a correlation of **{corr_value:.2f}** ({strength}) "
            f"with Heart Disease. This means higher values of {selected_feature} are "
            f"{'associated with higher risk' if corr_value>0 else 'associated with lower risk'}."
        )
    else:
        st.info(
            f"Observation: Categories in **{selected_feature}** show different disease prevalence. "
            f"Focus on categories with higher incidence for potential risk signals."
        )

    st.write("---")

    # ---- Correlation Heatmap (Top Features) ----
    st.write("### 2️⃣ Correlation Heatmap: Top Features")
    top_features = ["age", "chol", "trestbps", "exang"]  # select meaningful columns from cleaned data
    available_features = [f for f in top_features if f in data.columns]
    corr = data[available_features + [HeartDisease_var]].corr()

    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
        st.pyplot(fig)
        st.info(
            "These features were selected based on prior feature distribution analysis. "
            "Strong correlations (positive or negative) with Heart Disease indicate key predictors used in modeling."
        )

    st.write("---")

    # ---- Interactive Exploration ----
    st.write("### 3️⃣ Interactive Feature Exploration")
    if st.checkbox("Enable Interactive Plot"):
        x_axis = st.selectbox("X-axis", numeric_features, index=0)
        y_axis = st.selectbox("Y-axis", numeric_features, index=1)
        
        fig = px.scatter(
            data,
            x=x_axis,
            y=y_axis,
            color=HeartDisease_var,
            title=f"{y_axis} vs {x_axis} colored by {HeartDisease_var}",
            height=500
        )
        st.plotly_chart(fig)
        st.info(
            "Interactive exploration: observe how two numeric features jointly relate to disease outcomes. "
            "Try exploring top risk factors such as Age, Cholesterol, and Maximum Heart Rate."
        )

    # ---- Takeaway ----
    st.success(
        "EDA Takeaway: Focus on variables with strong associations to heart disease. "
        "Feature distributions, correlations, and interactive exploration together inform model selection, "
        "feature engineering, and patient risk assessment."
    )
