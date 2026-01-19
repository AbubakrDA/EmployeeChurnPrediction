import streamlit as st
import pandas as pd
import joblib
import requests

st.set_page_config(page_title="Employee Churn Prediction", layout="wide")

st.title("📊 Employee Churn Prediction Dashboard")
st.markdown("""
Predict the likelihood of an employee leaving the company based on their performance and other factors.
""")

# Load model directly for convenience in Streamlit (or could use API)
@st.cache_resource
def load_model():
    return joblib.load('churn_model.joblib')

model = load_model()

# Sidebar for inputs
st.sidebar.header("Input Employee Features")

def user_input_features():
    satisfaction_level = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.sidebar.slider("Last Evaluation", 0.0, 1.0, 0.5)
    number_project = st.sidebar.number_input("Number of Projects", 2, 10, 3)
    average_montly_hours = st.sidebar.number_input("Average Monthly Hours", 90, 310, 200)
    time_spend_company = st.sidebar.number_input("Time Spent at Company (Years)", 1, 10, 3)
    work_accident = st.sidebar.selectbox("Work Accident", [0, 1])
    promotion_last_5years = st.sidebar.selectbox("Promotion in Last 5 Years", [0, 1])
    
    department = st.sidebar.selectbox("Department", 
        ['sales', 'accounting', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing', 'RandD'])
    
    salary = st.sidebar.selectbox("Salary Level", ['low', 'medium', 'high'])
    
    data = {
        'satisfaction_level': satisfaction_level,
        'last_evaluation': last_evaluation,
        'number_project': number_project,
        'average_montly_hours': average_montly_hours,
        'time_spend_company': time_spend_company,
        'Work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years,
        'Departments ': department, # Match trailing space
        'salary': salary
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Display input parameters
st.subheader("Input Parameters")
st.write(input_df)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ **Warning**: This employee is likely to leave.")
        st.write(f"Churn Probability: **{probability[0]:.2%}**")
    else:
        st.success(f"✅ **Safe**: This employee is likely to stay.")
        st.write(f"Churn Probability: **{probability[0]:.2%}**")

st.markdown("---")
st.info("Note: Prediction is based on a Random Forest model trained on the HR Dataset.")
