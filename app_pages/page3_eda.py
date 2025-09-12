import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set Seaborn style
sns.set_style("whitegrid")

# Load dataset (replace with your cleaned CSV / data loader)
@st.cache_data
def load_data():
    df = pd.read_csv(
        "/workspaces/Heart_disease_risk_predictor/outputs/datasets/cleaned/heart_disease_cleaned.csv"
    )
    return df

df = load_data()
HeartDisease_var = "HeartDisease"
features = [col for col in df.columns if col != HeartDisease_var]

# Page title and purpose
st.write("### Exploratory Data Analysis (EDA)")
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
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df,
        x=selected_feature,
        hue=HeartDisease_var,
        order=df[selected_feature].value_counts().index
    )
    plt.title(f"{selected_feature} vs {HeartDisease_var}")
    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df,
        x=selected_feature,
        hue=HeartDisease_var,
        kde=True,
        element="step"
    )
    plt.title(f"{selected_feature} vs {HeartDisease_var}")
    st.pyplot(fig)

# ---- Insight for feature ----
corr_value = df[selected_feature].corr(df[HeartDisease_var]) if df[selected_feature].dtype != 'object' else None
if corr_value is not None:
    st.info(f"Observation: **{selected_feature}** has a correlation of **{corr_value:.2f}** with heart disease. Higher values indicate {'higher' if corr_value>0 else 'lower'} risk.")
else:
    st.info(f"Observation: Categories in **{selected_feature}** show differences in disease prevalence. Focus on categories with higher incidence.")

st.write("---")

# ---- Correlation heatmap ----
if st.checkbox("Show Correlation Heatmap"):
    st.write("The correlation heatmap helps identify relationships between features and the target, highlighting potentially important predictors.")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    st.pyplot(fig)

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
        title=f"{y_axis} vs {x_axis} colored by {HeartDisease_var}",
        height=500
    )
    st.plotly_chart(fig)
    st.info(f"Interactive exploration: see how combinations of **{x_axis}** and **{y_axis}** relate to disease outcomes.")

st.success(
    "EDA Takeaway: Focus on variables with strong associations to heart disease. "
    "These insights inform model selection, feature engineering, and patient risk assessment."
)
