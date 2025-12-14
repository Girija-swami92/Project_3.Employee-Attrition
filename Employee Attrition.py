import pandas as pd
import numpy as np
import streamlit as st
import joblib

model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\GUVI\Projects\Models\LogisticRegression\LogisticRegression.pkl")
scaler = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\GUVI\Projects\Models\LogisticRegression\Scaler.pkl")
encoder = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\GUVI\Projects\Models\LogisticRegression\Encoders.pkl")
model1 = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\GUVI\Projects\Models\LRP\LRP_model.pkl")


R = st.sidebar.radio("🫂 Employee Analysis ", ( "🎯 Employee Attrition", "🏅 Performance Rating"))

if R == "🎯 Employee Attrition":

    st.title("🔍 Attrition Prediction")

    age = st.number_input("Age", 18, 60, key="a_age")
    department = st.selectbox("Department", encoder["Department"].classes_, key="a_dept")
    distance = st.number_input("Distance From Home", 1, 50, key="a_dist")
    gender = st.selectbox("Gender", encoder["Gender"].classes_, key="a_gender")
    job_role = st.selectbox("Job Role", encoder["JobRole"].classes_, key="a_job")
    monthly_income = st.number_input("Monthly Income", 1000, 300000, key="a_income")
    overtime = st.selectbox("OverTime", encoder["OverTime"].classes_, key="a_ot")
    percent_hike = st.number_input("Percent Salary Hike", 0, 30, key="a_hike")
    total_working_years = st.number_input("Total Working Years", 0, 40, key="a_twy")
    years_at_company = st.number_input("Years at Company", 0, 40, key="a_yac")

    # Encode
    job_role = encoder["JobRole"].transform([job_role])[0]
    department = encoder["Department"].transform([department])[0]
    overtime = encoder["OverTime"].transform([overtime])[0]
    gender = encoder["Gender"].transform([gender])[0]

    input_df = pd.DataFrame([[ 
        age,
        encoder["BusinessTravel"].transform(["Travel_Rarely"])[0],
        800,
        department,
        distance,
        3,
        encoder["EducationField"].transform(["Life Sciences"])[0],
        3,
        gender,
        60,
        3,
        2,
        job_role,
        3,
        encoder["MaritalStatus"].transform(["Married"])[0],
        monthly_income,
        15000,
        2,
        overtime,
        percent_hike,
        3,
        3,
        1,
        total_working_years,
        2,
        3,
        years_at_company,
        2,
        1,
        2
    ]], columns=[
        "Age","BusinessTravel","DailyRate","Department",
        "DistanceFromHome","Education","EducationField",
        "EnvironmentSatisfaction","Gender","HourlyRate",
        "JobInvolvement","JobLevel","JobRole","JobSatisfaction",
        "MaritalStatus","MonthlyIncome","MonthlyRate",
        "NumCompaniesWorked","OverTime","PercentSalaryHike",
        "PerformanceRating","RelationshipSatisfaction",
        "StockOptionLevel","TotalWorkingYears",
        "TrainingTimesLastYear","WorkLifeBalance",
        "YearsAtCompany","YearsInCurrentRole",
        "YearsSinceLastPromotion","YearsWithCurrManager"
    ])

    input_scaled = scaler.transform(input_df)

    if st.button("Predict Attrition", key="predict_attrition"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Employee likely to leave (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Employee likely to stay (Probability: {1 - prob:.2f})")


elif R == "🏅 Performance Rating":

    st.title("🏅 Performance Rating Prediction")

    age = st.number_input("Age", 18, 60, key="p_age")
    department = st.selectbox("Department", encoder["Department"].classes_, key="p_dept")
    distance = st.number_input("Distance From Home", 1, 50, key="p_dist")
    gender = st.selectbox("Gender", encoder["Gender"].classes_, key="p_gender")
    job_role = st.selectbox("Job Role", encoder["JobRole"].classes_, key="p_job")
    monthly_income = st.number_input("Monthly Income", 1000, 300000, key="p_income")
    overtime = st.selectbox("OverTime", encoder["OverTime"].classes_, key="p_ot")
    percent_hike = st.number_input("Percent Salary Hike", 0, 30, key="p_hike")
    total_working_years = st.number_input("Total Working Years", 0, 40, key="p_twy")
    years_at_company = st.number_input("Years at Company", 0, 40, key="p_yac")

    # Encode
    job_role = encoder["JobRole"].transform([job_role])[0]
    department = encoder["Department"].transform([department])[0]
    overtime = encoder["OverTime"].transform([overtime])[0]
    gender = encoder["Gender"].transform([gender])[0]

    input_df = pd.DataFrame([[ 
        age,
        encoder["BusinessTravel"].transform(["Travel_Rarely"])[0],
        800,
        department,
        distance,
        3,
        encoder["EducationField"].transform(["Life Sciences"])[0],
        3,
        gender,
        60,
        3,
        2,
        job_role,
        3,
        encoder["MaritalStatus"].transform(["Married"])[0],
        monthly_income,
        15000,
        2,
        overtime,
        percent_hike,
        3,
        1,
        total_working_years,
        2,
        3,
        years_at_company,
        2,
        1,
        2
    ]], columns=[
        "Age","BusinessTravel","DailyRate","Department",
        "DistanceFromHome","Education","EducationField",
        "EnvironmentSatisfaction","Gender","HourlyRate",
        "JobInvolvement","JobLevel","JobRole","JobSatisfaction",
        "MaritalStatus","MonthlyIncome","MonthlyRate",
        "NumCompaniesWorked","OverTime","PercentSalaryHike",
        "RelationshipSatisfaction","StockOptionLevel",
        "TotalWorkingYears","TrainingTimesLastYear",
        "WorkLifeBalance","YearsAtCompany","YearsInCurrentRole",
        "YearsSinceLastPromotion","YearsWithCurrManager"
    ])

    if st.button("Predict Performance Rating", key="predict_performance"):
        prediction = model1.predict(input_df)[0]
        st.success(f"⭐ Predicted Performance Rating: {prediction}")