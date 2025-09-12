# Heart Disease Risk Prediction Project

## Dataset Content

The dataset is sourced from the [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).  
We created a fictitious scenario where a medical analytics company wants to predict heart disease risk for patients.  

Each row represents a patient, and each column contains a patient attribute, including:

- **Patient health info**: age, sex, blood pressure, cholesterol, etc.  
- **Symptoms & diagnostic measures**: chest pain type, resting ECG results, maximum heart rate, exercise-induced angina, ST depression, slope of ST segment, major vessels colored by fluoroscopy, thalassemia.  
- **Target variable**: presence of heart disease.  

| Variable  | Meaning | Units/Values |
|-----------|---------|--------------|
| age       | Patient age | Years |
| sex       | Patient gender | 1 = Male, 0 = Female |
| cp        | Chest pain type | 1-4 (typical/atypical/asymptomatic/other) |
| trestbps  | Resting blood pressure | mm Hg |
| chol      | Serum cholesterol | mg/dl |
| fbs       | Fasting blood sugar > 120 mg/dl | 1 = True, 0 = False |
| restecg   | Resting ECG results | 0-2 |
| thalach   | Maximum heart rate achieved | bpm |
| exang     | Exercise-induced angina | 1 = Yes, 0 = No |
| oldpeak   | ST depression induced by exercise relative to rest | Numeric value |
| slope     | Slope of peak exercise ST segment | 1-3 |
| ca        | Number of major vessels colored by fluoroscopy | 0-3 |
| thal      | Thalassemia | 3 = normal, 6 = fixed defect, 7 = reversible defect |
| target    | Heart disease presence | 0 = No disease, 1 = Disease |

---

## Project Terms & Jargon

- **Patient**: an individual undergoing medical evaluation.  
- **Prospect patient**: a new patient whose risk is to be predicted.  
- **Heart disease positive**: a patient diagnosed with heart disease (target = 1).  
- **Heart disease negative**: a patient without heart disease (target = 0).  

---

## Business Requirements

As a Data Analyst at **Code Institute Consulting**, you are requested by a medical analytics client to provide actionable insights and data-driven predictions regarding heart disease risk.  

The client has a dataset with patient health attributes and wants the following:

1. Understand patterns in patient data to identify the most relevant features correlated with heart disease.  
2. Predict whether a prospect patient has heart disease based on their health attributes.  
3. Identify subgroups (clusters) of patients to profile risk patterns and assist clinical decision-making.  

---

## Hypotheses and Validation

- **Hypothesis 1**: Older patients and those with high cholesterol are more likely to have heart disease.  
  - *Validation*: Correlation study, univariate analysis, and feature importance analysis.  

- **Hypothesis 2**: Certain types of chest pain are strong indicators of heart disease risk.  
  - *Validation*: Compare heart disease prevalence across chest pain types using plots and statistical tests.  

This segment sets the context, dataset understanding, and project framing.

# Heart Disease Risk Prediction Project

## Mapping Business Requirements to ML Tasks

**Business Requirement 1:** Understand patterns in patient data and identify features correlated with heart disease.  
**Approach:**

- Data visualization (histograms, boxplots, bar charts) to explore patient attributes.
- Correlation analysis (Pearson and Spearman) to identify features most related to heart disease.

**Business Requirement 2:** Predict heart disease for a prospect patient and identify patient clusters.  
**Approach:**

- Classification model: Predict presence or absence of heart disease (binary classification).
- Clustering model: Identify patient subgroups with similar health profiles.
- Regression (optional): Predict a continuous risk score if required (probability of heart disease).

---

## ML Business Case

### 1. Predict Heart Disease — Classification Model

- **Goal:** Predict whether a patient has heart disease (`target = 1`) using patient attributes.
- **Input features:** All attributes except `target` and any patient identifier.
- **Output:** Binary flag (0 = No disease, 1 = Disease) with probability.
- **Performance metrics:**
  - Recall ≥ 0.8 for detecting disease.
  - Precision ≥ 0.8 to minimize false positives.
- **Training data:** Filtered patient records (~300–500 rows).

### 2. Predict Risk Score — Regression Pipeline (Optional)

- **Goal:** Predict a continuous risk score for heart disease.
- **Models tested:** Linear Regression, Random Forest Regressor.
- **Metrics:** R² ≥ 0.7 on train/test sets.
- **Note:** Optional; primary focus is classification.

### 3. Cluster Analysis Pipeline

- **Goal:** Group patients with similar health profiles.
- **Model type:** KMeans clustering.
- **Metrics:** Silhouette score ≥ 0.45; fewer than 15 clusters for interpretability.
- **Output:** Cluster labels appended to patient records.

---

## Notebooks Overview

**Notebook 1 — Data Inspection & Cleaning

- Load dataset, check types, missing values, and outliers.
- Encode categorical variables, scale numerical features.
- Output: Cleaned dataset ready for analysis.

**Notebook 2 — Exploratory Data Analysis (EDA)

- Visualize distributions (age, cholesterol, blood pressure, chest pain type).
- Correlation and univariate analysis.
- Output: Insights into key risk factors.

**Notebook 3 — Feature Engineering & Preprocessing

- Transform variables, one-hot encode categorical features.
- Split data into training, validation, and test sets.
- Output: Preprocessed dataset and train/test splits.

**Notebook 4 — Heart Disease Classification Pipeline

- Build and train classification models: Logistic Regression, Random Forest, XGBoost.
- Evaluate with Recall, Precision, F1-score, ROC-AUC.
- Analyze feature importance.
- Output: Trained classifier and evaluation report.

**Notebook 5 — Predict Risk Score (Optional Regression)

- Build regression model for continuous risk score.
- Evaluate using R², RMSE, MAE.
- Output: Trained regressor and evaluation report.

**Notebook 6 — Final Evaluation & Deployment Prep

- Evaluate tuned models on test set.
- Visualize feature importance for tree-based models.
- Save final pipelines for deployment.
- Optional: Cluster analysis.

**Notebook 7 — Dashboard Integration (Streamlit)

- Combine all analysis and ML predictions into interactive dashboard.
- Pages for dataset overview, predictive analysis, feature importance, risk scores, and clusters.

---

## Dashboard Design (Streamlit)

**Page 1: Quick Project Summary

- Project overview, dataset details, business requirements, terminology.

**Page 2: Patient Data Analysis

- Interactive dataset preview, correlation heatmaps, variable distributions.

**Page 3: Heart Disease Prediction

- Form for prospect patient inputs.
- Output: Prediction, probability, optional risk score, cluster assignment.

**Page 4: Hypothesis Validation

- Summarize and validate hypotheses from EDA and ML.

**Page 5: Classification Results

- Display confusion matrix, ROC curve, feature importance.

**Page 6: Risk Score Prediction (Optional)

- Show regression outputs if implemented.
- Evaluate R², RMSE, MAE.

**Page 7: Clustering Results

- Visualize clusters (PCA or scatter plots).
- Cluster distribution across disease status.
- Patient subgroup profiles.

## ML Pipelines

The project implements three machine learning pipelines: **Classification**, **Regression (optional)**, and **Clustering**.  
All pipelines are modular and reproducible using scikit-learn.

---

### 1. Heart Disease Classification Pipeline

- **Goal**: Predict whether a patient has heart disease.  
- **Models tested**: Logistic Regression, Random Forest, XGBoost.  
- **Pipeline steps**:
  1. Preprocessing (scaling, encoding categorical variables).
  2. Feature selection (correlation analysis, feature importance).
  3. Model training with hyperparameter tuning (GridSearchCV/RandomizedSearchCV).
  4. Evaluation on test set.  
- **Metrics used**:
  - Recall (positive class = disease).
  - Precision.
  - F1-score.
  - ROC-AUC.  
- **Outcome**:
  - Logistic Regression → strong interpretability baseline.  
  - Random Forest & XGBoost → higher recall and ROC-AUC, better for deployment.

---

### 2. Risk Score Prediction Pipeline (Optional Regression)

- **Goal**: Predict a continuous heart disease risk score.  
- **Models tested**: Linear Regression, Random Forest Regressor.  
- **Pipeline steps**:
  1. Preprocessing (scaling, encoding).
  2. Train regression models.
  3. Evaluate predictions.  
- **Metrics used**:
  - R² score.
  - RMSE (Root Mean Squared Error).
  - MAE (Mean Absolute Error).  
- **Outcome**:
  - Regression provides interpretability but classification pipeline chosen as primary.

---

### 3. Patient Clustering Pipeline

- **Goal**: Identify patient subgroups with similar health profiles.  
- **Model tested**: KMeans clustering.  
- **Pipeline steps**:
  1. Preprocessing (scaling).
  2. Dimensionality reduction (PCA for visualization).
  3. KMeans clustering with optimal k (evaluated by silhouette score).  
- **Metrics used**:
  - Silhouette score.
  - Cluster interpretability (distribution of disease cases per cluster).  
- **Outcome**:
  - Subgroups emerged (e.g., older patients with high cholesterol vs younger, lower-risk patients).

---

## Dashboard Design (Streamlit)

The Streamlit app provides a user-friendly way to explore data and generate predictions.

---

### Page 1: Quick Project Summary

- Explains project goals, dataset details, and business requirements.
- Defines project-specific jargon.

### Page 2: Patient Data Analysis

- Interactive dataset preview.
- Visualizations of distributions and correlations.
- Insights into key risk factors.

### Page 3: Heart Disease Prediction

- Form inputs: Age, Sex, Chest Pain Type, BP, Cholesterol, Max HR, etc.
- **Predict button**: runs classification pipeline.
- **Outputs**:
  - Prediction (Yes/No).
  - Probability score.
  - Risk score (optional).
  - Assigned cluster.

### Page 4: Hypothesis Validation

- Lists hypotheses from EDA.
- Shows whether validated (e.g., ✅ "older patients have higher risk").

### Page 5: Classification Results

- Summarizes model performance.
- Displays:
  - Confusion matrix.
  - ROC curve.
  - Precision-Recall curve.
- Lists top predictive features.

### Page 6: Risk Score Prediction (Optional)

- Regression results with plots:
  - Predicted vs actual risk score.
- Error metrics (R², RMSE, MAE).

### Page 7: Cluster Analysis

- Shows scatterplots of clusters (PCA reduced).
- Displays cluster distribution across disease status.
- Provides subgroup profiles.

---

## Deployment & Usage

### Local Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/heart-disease-risk-prediction.git
   cd heart-disease-risk-prediction

Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run Jupyter notebooks:

bash
Copy code
jupyter notebook
Launch the Streamlit dashboard:

bash
Copy code
streamlit run app.py
Codespaces / Cloud IDE Setup
Fork this repository.

Click the green Code button → Create Codespace on Main.

Inside Codespaces:

bash
Copy code
pip install -r requirements.txt
Confirm Python kernel:

bash
Copy code
python --version
Launch the app:

bash
Copy code
streamlit run app.py

## Project Conclusions

- Tuned classification models achieved strong generalization performance on the holdout test set.
- Tree-based models (Random Forest, XGBoost) slightly outperformed Logistic Regression in ROC-AUC and Recall.
- Feature importance analyses identified key predictive variables, such as:
  - Age
  - Chest pain type
  - Maximum heart rate achieved
  - Cholesterol levels
- Clustering revealed meaningful patient subgroups for clinical insights:
  - High-risk older patients with elevated cholesterol.
  - Lower-risk younger patients.
- The final best model pipeline ensures reproducible preprocessing and prediction.

---

## Next Steps / Future Work

1. **Deployment & Integration**
   - Embed `best_model_pipeline.pkl` into a clinical decision support API or web app.
   - Implement input validation, standardization, and exception handling.

2. **Monitoring & Logging**
   - Track model predictions in production.
   - Monitor for concept drift and data quality issues.

3. **Explainability & Interpretability**
   - Integrate SHAP or permutation importance to explain individual predictions.
   - Generate clinician-friendly feature importance reports.

4. **Model Enhancement**
   - Explore ensemble stacking of classifiers.
   - Include additional patient features or lab results.
   - Evaluate other algorithms (LightGBM, CatBoost) for further performance improvement.

5. **Regulatory & Documentation**
   - Maintain a fully reproducible workflow.
   - Document all preprocessing, training, and evaluation steps.
   - Ensure compliance with healthcare data standards.

---

## Appendices

### Glossary

- **Patient**: Individual undergoing medical evaluation.
- **Prospect Patient**: New patient whose heart disease risk is predicted.
- **Heart Disease Positive**: Patient diagnosed with heart disease (target = 1).
- **Heart Disease Negative**: Patient without heart disease (target = 0).
- **Pipeline**: Preprocessing + feature engineering + model training sequence.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
- **Recall**: True positive rate; ability to detect disease cases.

### References

UCI Machine Learning Repository: Heart Disease Dataset
  
<https://archive.ics.uci.edu/ml/datasets/Heart+Disease>

Scikit-learn documentation: <https://scikit-learn.org/stable/documentation.html>

SHAP Explainability: <https://github.com/slundberg/shap>

Streamlit Documentation: <https://docs.streamlit.io/>

### Dataset License

- The dataset is publicly available from UCI ML Repository.
- License: Open access for research and educational purposes.
- In this project, a fictitious scenario is used to demonstrate ML workflows.

---

## Project Summary

This project provides a complete end-to-end workflow for predicting heart disease risk:

- Data cleaning & EDA.
- Feature engineering & preprocessing.
- Classification & optional regression pipelines.
- Patient clustering for subgroup analysis.
- Interactive Streamlit dashboard for end users.
- Deployment-ready model pipeline and documentation.

It is designed for reproducibility, interpretability, and real-world clinical applicability.
