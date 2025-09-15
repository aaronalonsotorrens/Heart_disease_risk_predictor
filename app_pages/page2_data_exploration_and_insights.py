import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_style("whitegrid")


def page_data_exploration_and_insights_body(data: pd.DataFrame):
    """
    Exploratory Data Analysis (EDA) page.
    Accepts a DataFrame 'data' with a 'HeartDisease' column.
    """

    HeartDisease_var = "HeartDisease"

    # Identify numeric and categorical features
    numeric_features = data.select_dtypes(include="number").columns.tolist()
    if HeartDisease_var in numeric_features:
        numeric_features.remove(HeartDisease_var)

    categorical_features = [
        col
        for col in data.columns
        if col not in numeric_features + [HeartDisease_var]
    ]

    # ---- Page Title ----
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.info(
        "EDA helps us understand patterns in the dataset, identify key "
        "risk factors, and guide predictive modeling for patient heart "
        "disease risk."
    )

    # ---- Feature Distribution ----
    st.write("### 1️⃣ Feature Distributions vs Heart Disease Outcome")
    st.write(
        "These plots show how each feature (e.g., Age, Cholesterol, Chest "
        "Pain Type) is distributed for patients with and without heart "
        "disease.\n\n"
        "👉 This helps detect **risk factors** (features that look different "
        "across groups), and also reveals **imbalances** in categories or "
        "ranges."
    )

    selected_feature = st.selectbox(
        "Select a feature to explore:",
        numeric_features + categorical_features,
    )
    st.write(
        f"Analyzing **{selected_feature}** and its relationship with heart "
        "disease:"
    )

    # Visualization based on type
    if (
        selected_feature in categorical_features
        or data[selected_feature].nunique() < 10
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=data,
            x=selected_feature,
            hue=HeartDisease_var,
            order=data[selected_feature].value_counts().index,
        )
        plt.title(
            f"{selected_feature} vs {HeartDisease_var}"
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=data,
            x=selected_feature,
            hue=HeartDisease_var,
            kde=True,
            element="step",
        )
        plt.title(
            f"{selected_feature} distribution by Heart Disease outcome"
        )

    st.pyplot(fig)

    # ---- Interpretation ----
    if selected_feature in numeric_features:
        corr_value = data[selected_feature].corr(data[HeartDisease_var])
        if abs(corr_value) < 0.2:
            strength = "weak"
        elif abs(corr_value) < 0.5:
            strength = "moderate"
        else:
            strength = "strong"

        association = (
            "associated with higher risk"
            if corr_value > 0
            else "associated with lower risk"
        )

        observation = (
            f"Observation: **{selected_feature}** has a correlation of "
            f"**{corr_value:.2f}** ({strength}) with Heart Disease. "
            f"This means higher values of {selected_feature} are "
            f"{association}."
        )

        st.info(observation)
        st.warning(
            "⚠️ Note: Linear correlation may underestimate importance "
            "for non-linear or threshold effects (e.g., Age > 50). "
            "Binary targets can also dilute correlation values."
        )
    else:
        st.info(
            f"Observation: Categories in **{selected_feature}** show "
            "different disease prevalence. Focus on categories with "
            "higher incidence for potential risk signals."
        )

    st.write("---")

    # ---- Correlation Heatmap (Top Features) ----
    st.write("### 2️⃣ Correlation Heatmap: Top Features")
    st.write(
        "The heatmap highlights **linear relationships** between numeric "
        "features and Heart Disease. Strong correlations (positive or "
        "negative) suggest these features could be important predictors."
    )
    st.write(
        "👉 Example: High resting blood pressure (trestbps) or "
        "cholesterol (chol) levels may correlate with higher risk. "
        "But remember: correlation does **not** always imply causation."
    )

    top_features = ["age", "chol", "trestbps", "exang"]
    available_features = [f for f in top_features if f in data.columns]
    corr = data[available_features + [HeartDisease_var]].corr()

    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            ax=ax,
        )
        st.pyplot(fig)
        st.info(
            "These correlations help us **rank features** for model training. "
            "For example, highly correlated features are strong candidates "
            "for inclusion, while redundant features may be dropped to "
            "avoid overfitting."
        )

    st.write("---")

    # ---- Interactive Exploration ----
    st.write("### 3️⃣ Interactive Feature Exploration")
    st.write(
        "Scatterplots allow us to **compare two numeric features together** "
        "and see how they interact. By coloring points by Heart Disease "
        "outcome, we can spot patterns that single-feature plots "
        "may miss. \n\n"
        "👉 Example: Patients with **high cholesterol and high blood "
        "pressure** might cluster into the higher-risk group."
    )

    if st.checkbox("Enable Interactive Plot"):
        x_axis = st.selectbox("X-axis", numeric_features, index=0)
        y_axis = st.selectbox("Y-axis", numeric_features, index=1)

        fig = px.scatter(
            data,
            x=x_axis,
            y=y_axis,
            color=HeartDisease_var,
            title=(
                f"{y_axis} vs {x_axis} colored by {HeartDisease_var}"
            ),
            height=500,
        )
        st.plotly_chart(fig)
        st.info(
            "Interactive exploration is especially useful for detecting "
            "**feature interactions**, which linear correlations alone might "
            "miss."
        )

    # ---- Takeaway ----
    st.success(
        "### 🧾 EDA Takeaways\n"
        "- **Feature Distributions** reveal which variables differ most "
        "between patients with and without heart disease. This helps "
        "identify candidate risk factors.\n"
        "- **Correlation Analysis** highlights linear relationships and "
        "helps us prioritize features for training.\n"
        "- **Interactive Exploration** uncovers interactions between "
        "features that may be more predictive together than alone.\n\n"
        "➡️ These insights guide **feature engineering**, **model "
        "selection**, and ultimately improve the predictive power of our "
        "models. For example, strong predictors (like chest pain type or "
        "exercise-induced angina) will be emphasized in training, while "
        "weak or redundant variables may be dropped."
    )
