# 🫀 Heart Disease Risk Predictor

## 1. Project Overview

The **Heart Disease Risk Predictor** is a data-driven project designed to predict the likelihood of heart disease in patients based on clinical and demographic features. This tool is intended to support **public health workers, clinicians, and healthcare stakeholders** in identifying high-risk individuals early, enabling preventative interventions and informed decision-making.

**Objectives:**

- Build a robust predictive model for heart disease risk.
- Provide clear, interpretable insights into key risk factors.
- Develop a reusable workflow for preprocessing, modeling, and evaluation.
- Prepare for eventual deployment in an interactive dashboard for real-time predictions.

---

## 2. Business Context & Case

Heart disease remains one of the leading causes of morbidity and mortality worldwide. Early detection of high-risk patients can significantly reduce complications and healthcare costs.  

**Business Goals:**

1. Identify patients at high risk of heart disease.
2. Understand which factors contribute most to risk.
3. Provide actionable insights for preventive healthcare interventions.
4. Deliver reproducible, interpretable machine learning pipelines.

**Success Criteria:**

- Achieve **≥80% accuracy** on classification tasks.
- Maintain **high recall** to minimize false negatives.
- Ensure explainability using feature importance and SHAP/LIME.
- Maintain clean, documented, and reusable code pipelines.

**ML Tasks Mapped to Business Case:**

- **Classification:** Predict whether a patient is at risk (`HeartDisease` 0/1).
- **Regression (Optional):** Estimate a risk score or severity index.
- **Clustering (Optional):** Group patients with similar profiles for population-level insights.

---

## 3. Kanban Board – User Stories & Epics

The project workflow is organized using a **Kanban methodology** to track progress and prioritize tasks.

### Epic 1 – Information Gathering & Data Collection

| Priority | User Story |
|----------|------------|
| [M] | As a data scientist, I want to download and store the UCI Heart Disease dataset from Kaggle so that I can work with a clean, local copy for analysis. |
| [M] | As a project team member, I want to document all feature definitions and their units so that stakeholders and developers understand the meaning of each input variable. |
| [S] | As a data scientist, I want to explore dataset structure (rows, columns, data types) so that I can plan preprocessing steps effectively. |

### Epic 2 – Data Visualization, Cleaning & Preparation

| Priority | User Story |
|----------|------------|
| [M] | As a data scientist, I want to visualise distributions of all features so that I can detect anomalies, missing values, and outliers. |
| [M] | As a data scientist, I want to perform correlation analysis and feature importance scoring so that I can identify the most relevant variables for predicting heart disease. |
| [M] | As a data scientist, I want to handle missing/invalid values and outliers so that my dataset is ready for model training without bias. |
| [S] | As a data scientist, I want to create a cleaned, preprocessed dataset file so that I can reuse it across different model experiments. |

### Epic 3 – Model Training, Optimization & Validation

| Priority | User Story |
|----------|------------|
| [M] | As a machine learning engineer, I want to train baseline models (Logistic Regression, Random Forest) so that I can compare performance and select a strong candidate. |
| [M] | As a machine learning engineer, I want to perform hyperparameter tuning with cross-validation so that I can improve model accuracy and reduce overfitting. |
| [M] | As a machine learning engineer, I want to track model performance metrics (accuracy, ROC-AUC, recall, precision) so that I can ensure the model meets the 80% accuracy target and low false negatives. |
| [S/C] | As a machine learning engineer, I want to use SHAP or LIME for explainability so that stakeholders understand how the model makes predictions. (Should have, but becomes Could have if time is tight) |

### Epic 4 – Dashboard Planning, Designing & Development

| Priority | User Story |
|----------|------------|
| [M] | As a frontend developer, I want to design a user-friendly Streamlit interface so that public health workers can easily navigate the dashboard. |
| [M] | As a public health worker, I want to input patient data into a form so that I can receive a real-time prediction of heart disease risk. |
| [S] | As a public health worker, I want to see the most important features influencing a prediction so that I can better understand the patient’s risk factors. |
| [S] | As a stakeholder, I want to view a project summary and dataset overview in the dashboard so that I can quickly grasp the purpose and scope of the tool. |

### Epic 5 – Dashboard Deployment & Release

| Priority | User Story |
|----------|------------|
| [M] | As a DevOps engineer, I want to deploy the Streamlit app online so that stakeholders can access the tool without installing anything. |
| [S] | As a project manager, I want to implement version control for the model and dashboard code so that I can manage updates and rollbacks safely. |
| [C] | As a stakeholder, I want to have a technical page on the dashboard showing metrics and pipeline details so that I can verify the robustness of the solution. |

## 4. Dataset Content

The **Heart Disease dataset** is sourced from the **UCI Machine Learning Repository** via Kaggle. The dataset contains clinical and demographic data for patients, which will be used to predict the presence of heart disease.  

Each row represents a patient, and each column represents a clinical or demographic attribute.

### 4.1 Feature Definitions

| Variable | Meaning | Units / Categories |
|----------|---------|------------------|
| `age` | Patient age | Years |
| `sex` | Patient gender | 1 = Male, 0 = Female |
| `cp` | Chest pain type | 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal pain, 3: Asymptomatic |
| `trestbps` | Resting blood pressure | mm Hg |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar | 1 = >120 mg/dl, 0 = ≤120 mg/dl |
| `restecg` | Resting electrocardiographic results | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved | bpm |
| `exang` | Exercise-induced angina | 1 = Yes, 0 = No |
| `oldpeak` | ST depression induced by exercise relative to rest | Depression level (numeric) |
| `slope` | Slope of peak exercise ST segment | 0: Upsloping, 1: Flat, 2: Downsloping |
| `ca` | Number of major vessels colored by fluoroscopy | 0–3 |
| `thal` | Thalassemia | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect |
| `target` | Heart disease diagnosis | 1 = Presence, 0 = Absence |

---

### 4.2 Project Terms & Jargon

- **Patient:** Individual under clinical evaluation.  
- **Feature / Variable:** An attribute measured for each patient.  
- **Target:** The outcome variable to be predicted (`target`).  
- **Risk Factor:** A feature correlated with the presence of heart disease.  
- **Prospect Patient:** A new patient whose heart disease risk is unknown.  

---

# 🫀 Heart Disease Risk Predictor — Notebook Workflow

## Notebook 1 — Data Upload & Initial Inspection

**Purpose:**  
Introduce and load the heart disease dataset for exploration and preprocessing.

**Key Features:**

- Upload dataset from local files or Kaggle.
- Inspect structure, rows, columns, and data types.
- Identify missing values and invalid entries.
- Initial observations guide preprocessing strategy.

**Outcome:**  
Dataset loaded, cleaned of major inconsistencies, and ready for deeper preprocessing.

---

## Notebook 2 — Data Preprocessing & Feature Engineering

**Purpose:**  
Transform raw data into a machine-learning-ready format.

**Key Features:**

- Handle missing values using median/mode imputation.
- Encode categorical variables (one-hot or binary encoding).
- Scale numeric features to standardize magnitude.
- Engineer new features (e.g., age groups, cholesterol ratios, interaction terms).
- Detect and treat outliers to reduce bias.

**Outcome:**  
Cleaned, numeric, fully-preprocessed dataset saved for model training.

---

## Notebook 3 — Exploratory Data Analysis (EDA) & Feature Selection

**Purpose:**  
Understand feature distributions, relationships, and importance before modeling.

**Key Features:**

- Visualize distributions and correlations.
- Identify predictive features via statistical tests and domain knowledge.
- Examine multicollinearity to reduce redundancy.
- Rank features by relevance for candidate models.

**Outcome:**  
Informed selection of features for model training, ensuring interpretability and predictive power.

---

## Notebook 4 — Model Training & Baseline Performance

**Purpose:**  
Train multiple ML models and establish baseline performance.

**Key Features:**

- Models used: Logistic Regression, Random Forest, XGBoost, LightGBM.
- Split data into training and validation sets.
- Evaluate baseline metrics: accuracy, precision, recall, F1, ROC-AUC.
- Identify strengths and weaknesses of each model for further tuning.

**Outcome:**  
Baseline performance recorded; models selected for hyperparameter optimization.

---

## Notebook 5 — Hyperparameter Tuning & Advanced Experiments

**Purpose:**  
Optimize model performance using systematic tuning and cross-validation.

**Key Features:**

- GridSearchCV or randomized search for hyperparameter optimization.
- Cross-validation to ensure robust performance estimates.
- Compare tuned models across metrics and select top performers.
- Optional experiments: feature subset selection, engineered feature impact.

**Outcome:**  
Tuned models ready for final evaluation, with improved generalization over baseline.

---

## Notebook 6 — Final Evaluation & Deployment Prep

**Purpose:**  
Assess generalization of tuned models and prepare the best pipeline for deployment.

**Key Features:**

- Evaluate models on holdout test set using classification metrics and ROC-AUC.
- Visualize confusion matrices and ROC curves.
- Analyze feature importance for tree-based models.
- Compare models to select the best-performing pipeline.
- Save evaluation results and deployment-ready pipeline.

**Outcome:**  
Best model identified and saved; evaluation results documented for reproducibility and reporting.

---

## Notebook 7 — Model Deployment & Inference

**Purpose:**  
Enable deployment and reproducible inference on new patient data.

**Key Features:**

- Load deployment-ready pipeline from Notebook 6.
- Provide helper functions for input alignment and inference.
- Enhanced predictions with probabilities, risk bands (Low/Medium/High), recommendations, and top feature contributions.
- Test predictions on simplified clinical inputs and full feature sets.
- Backup deployment pipeline for production use.

**Outcome:**  
Pipeline ready for API/web integration; interpretable and actionable predictions validated.

# 🫀 Heart Disease Risk Predictor — Prediction Model Details

## Classification Model — Heart Disease Risk

We aim to develop a **machine learning model to predict the likelihood of heart disease** in patients based on clinical and demographic features. The target variable is **categorical** and contains **2 classes**:

- `0` = No heart disease  
- `1` = Presence of heart disease  

This is a **supervised, 2-class, single-label classification model**.

**Goal:**  
Provide healthcare practitioners and public health stakeholders with reliable insights to identify high-risk patients early and inform preventive interventions.

---

### Model Success Metrics

- **Recall ≥ 80%** for detecting patients with heart disease (`1`) on both train and test sets.  
- Precision for predicting no heart disease (`0`) should also be **≥ 80%** to avoid overestimating risk.  

**Failure Criteria:**

- If more than 30% of high-risk patients are misclassified as low-risk over a follow-up period (indicating poor clinical utility).  
- If precision for low-risk patients falls below 80%, leading to unnecessary interventions.

### Model Output

- A **binary flag**: `0` (no heart disease) or `1` (heart disease).  
- **Probability score** for each patient indicating the risk of heart disease.  

**Input Data Collection:**

- Online: patient completes a clinical form with features such as age, sex, blood pressure, cholesterol, chest pain type, maximum heart rate, etc.  
- Offline: clinician or public health worker collects input during patient interview and enters data into the tool.  

**Inference:** Predictions are generated **on the fly**, not in batch mode.

### Heuristics

- No prior approach exists for predicting heart disease risk in a single-step ML pipeline with interpretable results for clinical use.

### Training Data

- Source: UCI Heart Disease dataset via Kaggle (~1,000 patient records).  
- Target: `target` (heart disease presence).  
- Features: Clinical and demographic variables, excluding identifiers.  
- Preprocessing includes handling missing values, categorical encoding, feature scaling, and engineered features such as cholesterol ratios and age groups.

### Pipeline Details

1. **Baseline Models:** Logistic Regression, Random Forest, XGBoost, LightGBM.  
2. **Tuning:** Hyperparameter optimization using GridSearchCV and cross-validation.  
3. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC.  
4. **Feature Analysis:** Importance ranking and optional SHAP/LIME explanations.  
5. **Deployment:** Saved pipeline enables real-time inference in web or API-based applications.

### Optional Extensions

- **Multi-class Risk Band Classification:** Patients can be categorized into `Low`, `Medium`, or `High` risk based on predicted probability thresholds.  
- **Feature Contribution Reporting:** Top contributing variables for each prediction are returned to enhance clinical interpretability.

# Heart Disease Prediction Dashboard

## Overview

This Streamlit dashboard predicts heart disease risk using clinical patient data. It includes EDA, model training, ablation studies, advanced experiments, and an inference tool for real-time predictions.

---

## Dashboard Design (Streamlit App Pages)

### Page 1: Quick Project Summary

**Before the analysis:**  
We intended this page to provide a high-level summary of the project to new users and stakeholders.  

**After the analysis:**  
The page provides:

- Quick project summary
- Project terms & jargon
- Dataset description
- Business requirements:
  - Predict heart disease risk
  - Identify high-risk patients
  - Provide actionable clinical recommendations

**Outcome:**  
Users understand the purpose of the project, the data being used, and the key goals for predictive modeling.

---

### Page 2: Exploratory Data Analysis (EDA)

**Before the analysis:**  
We aimed to answer: “Which clinical variables are associated with heart disease?”  

**After the analysis:**  
The page provides:

- Checkbox: Inspect dataset (rows, columns, first 10 records)
- Display correlations between features and heart disease
- Individual plots showing distribution of key features (e.g., age, cholesterol, max heart rate)
- Parallel plots to examine multivariate relationships

**Outcome:**  
Insights guide feature selection and inform the ML models.

---

### Page 3: Model Training

**Before the analysis:**  
We planned to train several machine learning models and compare performance metrics.  

**After the analysis:**  
The page provides:

- ML pipeline steps for each model (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Performance metrics: Accuracy, F1, ROC-AUC
- Observations and considerations after training

**Outcome:**  
Identifies the best-performing models for further evaluation.

---

### Page 4: Ablation Study

**Before the analysis:**  
We wanted to assess the contribution of each feature and verify the impact of engineered features.  

**After the analysis:**  
The page provides:

- Tuned model performance comparison
- Confusion matrix and ROC curve for selected models
- Tree-based feature importance (Random Forest, XGBoost, LightGBM)
- Dynamic notes explaining model-specific observations

**Outcome:**  
Users see which features are most influential and validate feature engineering.

---

### Page 5: Advanced Experiments

**Before the analysis:**  
Goal: Compare tuned models on a holdout test set to select the best-performing pipeline.  

**After the analysis:**  
The page provides:

- Comparative performance metrics on the holdout test set
- Highlight of the best model based on recall and overall performance
- Barplots visualizing metrics across models
- Actionable insights for deployment

**Outcome:**  
Provides confidence in the final model pipeline for real-world deployment.

---

### Page 6: Inference Tool (Core Patient Prediction)

**Before the analysis:**  
We intended this page for real-time predictions of heart disease risk using core clinical features.  

**After the analysis:**  
The page provides:

- Input widgets for main clinical features
- Optional example patients (high-risk / low-risk)
- "Run Prediction" button
- Outputs: Predicted class, probability, risk band, clinical recommendation
- Top contributing features (via SHAP, if available)

**Outcome:**  
Enables quick, actionable risk assessment for individual patients.

---

### Page 7: Inference Tool (Patient Risk Prediction – Simple & Advanced)

**Before the analysis:**  
We wanted a page that allows quick, actionable predictions for typical users while also providing an option to explore advanced inputs for extreme or high-risk cases.

**After the analysis:**  
The page provides:

- **Simple Input Section** for core clinical features:
  - Age, sex, chest pain type, blood pressure, cholesterol, max heart rate, ST depression, exercise-induced angina, slope, major vessels, thalassemia
  - “Run Prediction” button
  - Outputs: predicted class, probability, risk band, clinical recommendation, top contributing features (via SHAP if available)
- **Optional Advanced Input Section** (expandable):
  - All 22 features including engineered variables (`chol_age_ratio`, `oldpeak_thalach_ratio`, etc.)
  - Dataset placeholders for source tracking
  - “Run Full Advanced Prediction” button
  - Same outputs as simple mode

**Outcome:**  

- Quick and intuitive predictions for standard cases improve user experience.  
- Advanced input allows in-depth exploration, testing extreme or rare patient scenarios, and validating the model’s predictions.  
- Provides a flexible interface catering to both casual users and clinical/data specialists.

---

## Next Steps / Deployment Considerations

- Integrate the selected pipeline into an API or production application
- Standardize preprocessing and input validation
- Add logging and monitoring for predictions
- Consider interpretability tools (SHAP, permutation importance)
- Document full workflow for reproducibility and clinical compliance
