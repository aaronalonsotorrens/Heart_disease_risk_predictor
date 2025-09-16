import streamlit as st
import pandas as pd

from app_pages.multipage import MultiPage
from app_pages.page1_project_overview_and_goals import (
    page_project_overview_and_goals_body,
)
from app_pages.page2_data_exploration_and_insights import (
    page_data_exploration_and_insights_body,
)
from app_pages.page3_model_development_and_evaluation import (
    page_model_development_and_evaluation_body,
)
from app_pages.page4_model_tuning_and_insights import (
    page_model_tuning_and_insights_body,
)
from app_pages.page5_model_comparison_and_selection import (
    page_model_comparison_and_selection_body,
)
from app_pages.page6_heart_risk_predictor_tool import (
    page_heart_risk_predictor_tool_body,
)

# ---- Streamlit Page Config ----
# Must be the first Streamlit command
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="🖥️",
)

# ---- DATA LOADING / CACHING ----


@st.cache_data
def load_data(path: str):
    """
    Load and cache dataset.

    Returns a pandas DataFrame.
    """
    return pd.read_csv(path)

data_path = (
    "outputs/datasets/cleaned/heart_disease_cleaned.csv"
)
data = load_data(data_path)

# ---- CREATE MULTI-PAGE APP ----
app = MultiPage("Heart Disease Prediction Dashboard")

# Add pages in logical order
app.add_page(
    "📌 Project Overview & Goals",
    lambda: page_project_overview_and_goals_body(data),
)
app.add_page(
    "📊 Data Exploration & Insights",
    lambda: page_data_exploration_and_insights_body(data),
)
app.add_page(
    "⚙️ Model Development & Evaluation",
    page_model_development_and_evaluation_body,
)
app.add_page(
    "🔧 Model Tuning & Insights",
    page_model_tuning_and_insights_body,
)
app.add_page(
    "📈 Model Comparison & Selection",
    page_model_comparison_and_selection_body,
)
app.add_page(
    "🩺 Heart Risk Predictor Tool",
    page_heart_risk_predictor_tool_body,
)

# ---- RUN APP ----
app.run()
