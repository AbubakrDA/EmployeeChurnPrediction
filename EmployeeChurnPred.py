import streamlit as st
import joblib
import pandas as pd

# Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_churn_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("Employee Churn Prediction")
st.write("Enter employee details to predict if they are likely to leave the company.")

# Input Form
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
        last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
        number_project = st.number_input("Number of Projects", min_value=2, max_value=10, value=3)
        average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350, value=150)
        
    with col2:
        time_spend_company = st.number_input("Time Spent at Company (Years)", min_value=1, max_value=10, value=3)
        work_accident = st.selectbox("Work Accident", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
    departments = st.selectbox("Department", ['sales', 'accounting', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing', 'RandD'])
    salary = st.selectbox("Salary Level", ['low', 'medium', 'high'])

    submit_button = st.form_submit_button("Predict Churn")

if submit_button and model:
    # Create DataFrame
    input_data = pd.DataFrame([{
        'satisfaction_level': satisfaction_level,
        'last_evaluation': last_evaluation,
        'number_project': number_project,
        'average_montly_hours': average_montly_hours,
        'time_spend_company': time_spend_company,
        'Work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years,
        'Departments': departments,
        'salary': salary
    }])
    
    try:
        prediction = model.predict(input_data)[0]
        # prediction_proba = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.error(f"Prediction: Employee is likely to LEAVE.")
        else:
            st.success(f"Prediction: Employee is likely to STAY.")
            
    except Exception as e:
        st.error(f"Prediction Failed: {e}")
