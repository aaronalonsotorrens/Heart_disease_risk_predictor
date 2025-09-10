import streamlit as st
from app_pages.multipage import MultiPage

# Import your page modules
from app_pages.page1_summary import page_summary_body
from app_pages.page2_project_overview import page_project_overview_body
from app_pages.page3_eda import page_eda_body
from app_pages.page4_model_training import page_model_training_body
from app_pages.page5_ablation_study import page_ablation_study_body
from app_pages.page6_advanced_pipeline import page_advanced_experiments_body
from app_pages.page7_inference import page_inference_tool_body


import pandas as pd

# Load dataset for Page 1
data = pd.read_csv("/workspaces/Heart_disease_risk_predictor/outputs/datasets/cleaned/heart_disease_cleaned.csv")

# Create app
app = MultiPage("Heart Disease Prediction Dashboard")

# Add pages
app.add_page("Project Overview", lambda: page_project_overview_body(data))
app.add_page("Exploratory Data Analysis", page_eda_body)
app.add_page("Model Training", page_model_training_body)
app.add_page("Ablation Study", page_ablation_study_body)
app.add_page("Advanced Experiments", page_advanced_experiments_body)
app.add_page("Inference Tool", page_inference_tool_body)

# Run app
app.run()