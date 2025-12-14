# Project_3.Employee-Attrition
Machine Learning–based Employee Attrition Prediction with Streamlit Deployment
This project focuses on predicting employee attrition using machine learning techniques.
The goal is to help organizations identify employees who are at risk of leaving and take proactive retention measures.
Employee attrition is a critical challenge for organizations as it increases recruitment
costs and impacts productivity. This project aims to build a predictive model that
classifies whether an employee is likely to leave the organization based on demographic,
job-related, and compensation features. 

**PREPROCESSING:**
- Removed irrelevant columns
- Handled categorical features using Label Encoding
- Applied StandardScaler for numerical feature scaling
- Checked data distribution using KDE plots
- 
**- MODELS USED:**
- Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- XGBoost
- 
**MODEL EVALUATION :**
- Evaluation metrics: F1-score, ROC-AUC, Precision, Recall
- Hyperparameter tuning using GridSearchCV
- Best model selected based on F1-score
- 
**- Deployment**
- Deployed the trained model using Streamlit
- 
**- Results & Business Insights**
- Monthly Income, Age, Total Working Years, and OverTime were key drivers of attrition
- Employees with frequent overtime and lower income showed higher attrition risk  
