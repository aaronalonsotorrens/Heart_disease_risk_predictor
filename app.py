import streamlit as st
import pandas as pd
from app_pages.multipage import MultiPage

# Must be first Streamlit command
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="🖥️"
)

# DATA LOADING / CACHING

@st.cache_data
def load_data(path: str):
    """Load and cache dataset."""
    return pd.read_csv(path)

data_path = "/workspaces/Heart_disease_risk_predictor/outputs/datasets/cleaned/heart_disease_cleaned.csv"
data = load_data(data_path)

# Load raw dataset
raw_data = pd.read_csv("/workspaces/Heart_disease_risk_predictor/inputs/datasets/raw/heart_disease_uci.csv")

# IMPORT PAGE FUNCTIONS

from app_pages.page1_summary import page_summary_body
from app_pages.page2_project_overview import page_project_overview_body
from app_pages.page3_eda import page_eda_body
from app_pages.page4_model_training import page_model_training_body
from app_pages.page5_ablation_study import page_ablation_study_body
from app_pages.page6_advanced_pipeline import page_advanced_experiments_body
from app_pages.page7_inference import page_inference_tool_body

# CREATE MULTI-PAGE APP

app = MultiPage("Heart Disease Prediction Dashboard")

# Add pages in logical order
app.add_page("Summary", lambda: page_summary_body(data))  # first landing page
app.add_page("Project Overview", lambda: page_project_overview_body(data))
app.add_page("Exploratory Data Analysis", lambda: page_eda_body(raw_data)) 
app.add_page("Model Training", page_model_training_body)
app.add_page("Ablation Study", page_ablation_study_body)
app.add_page("Advanced Experiments", page_advanced_experiments_body)
app.add_page("Inference Tool", page_inference_tool_body)

# RUN APP

app.run()