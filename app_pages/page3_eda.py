import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_style("whitegrid")

# ---- Load dataset ----
@st.cache_data
def load_raw_data():
    df = pd.read_csv("/workspaces/Heart_disease_risk_predictor/inputs/datasets/raw/heart_disease_uci.csv")
    return df

raw_data = load_raw_data()

def page_eda_body(raw_data: pd.DataFrame):
    """
    Exploratory Data Analysis (EDA) page using raw heart disease dataset.
    """
    df = raw_data.copy()

    # Identify target column
    if "num" in df.columns:
        HeartDisease_var = "num"  # raw dataset target
        df[HeartDisease_var] = df[HeartDisease_var].astype(str)  # for plotting
    else:
        st.error("Target column 'num' not found in raw dataset!")
        return

    features = [col for col in df.columns if col != HeartDisease_var]

    # Page title and purpose
    st.title("Exploratory Data Analysis (EDA)")
    st.info(
        "EDA helps us understand the dataset, identify key risk factors for heart disease, "
        "and detect patterns that will guide modeling and patient risk prediction."
    )

    # ---- Dataset preview ----
    with st.expander("Preview Dataset"):
        st.write(f"The dataset contains **{df.shape[0]} patients** and **{df.shape[1]-1} features** (excluding target).")
        st.dataframe(df.head(10))

    # ---- Feature distribution analysis ----
    st.write("#### Feature Distributions vs Heart Disease Outcome")
    selected_feature = st.selectbox("Select a feature to explore:", features)
    st.write(f"Analyzing **{selected_feature}** and its relationship with heart disease:")

    if df[selected_feature].dtype == "object" or df[selected_feature].nunique() < 10:
        # Categorical → countplot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=df,
            x=selected_feature,
            hue=HeartDisease_var,
            order=df[selected_feature].value_counts().index
        )
        plt.title(f"{selected_feature} vs Heart Disease")
        st.pyplot(fig)
    else:
        # Numerical → histogram + KDE
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=df,
            x=selected_feature,
            hue=HeartDisease_var,
            kde=True,
            element="step"
        )
        plt.title(f"{selected_feature} vs Heart Disease")
        st.pyplot(fig)

        # Add correlation warning for numerical features
        corr_value = df[selected_feature].astype(float).corr(pd.to_numeric(df[HeartDisease_var]))
        st.info(
            f"Observation: **{selected_feature}** has a linear correlation of **{corr_value:.2f}** with heart disease. "
            "⚠️ Note: Linear correlation may underestimate importance if the relationship is non-linear or threshold-based."
        )

    st.write("---")

    # ---- Correlation heatmap (top features only) ----
    if st.checkbox("Show Correlation Heatmap (Top Features)"):
        st.write(
            "Correlation heatmap shows linear relationships between features and the target. "
            "We focus on top features to highlight key predictors."
        )
        corr = df.corr()
        # Select top 5 features most correlated with target
        top_features = corr[HeartDisease_var].abs().sort_values(ascending=False).index[1:6].tolist()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[top_features + [HeartDisease_var]].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
        st.pyplot(fig)
        st.info(f"Top features shown: {', '.join(top_features)}")

    st.write("---")

    # ---- Interactive scatter plot ----
    if st.checkbox("Interactive Scatter Plot"):
        numeric_features = df.select_dtypes(include="number").columns.tolist()
        x_axis = st.selectbox("X-axis", numeric_features, index=0)
        y_axis = st.selectbox("Y-axis", numeric_features, index=1)
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=HeartDisease_var,
            title=f"{y_axis} vs {x_axis} colored by Heart Disease",
            height=500
        )
        st.plotly_chart(fig)
        st.info("Explore combinations of features to identify patterns related to heart disease.")

    st.success(
        "EDA Takeaway: Focus on variables with strong associations to heart disease. "
        "These insights inform model selection, feature engineering, and patient risk assessment."
    )
