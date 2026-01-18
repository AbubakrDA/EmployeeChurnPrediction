import streamlit as st
import joblib
import pandas as pd
import io

# Set Page Config
st.set_page_config(page_title="Employee Churn Prediction", layout="wide")

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

st.title("🛡️ Employee Churn Prediction")
st.write("Predict employee turnover using our trained machine learning model.")

# Create tabs for Single and Batch prediction
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab1:
    st.header("👤 Single Employee Prediction")
    st.write("Adjust the details below to predict a single employee's status.")
    
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

    if submit_button:
        if model:
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
                proba = model.predict_proba(input_data)[0][1] if hasattr(model, 'predict_proba') else None
                
                if prediction == 1:
                    st.error(f"### ⚠️ Prediction: Employee is likely to LEAVE.")
                else:
                    st.success(f"### ✅ Prediction: Employee is likely to STAY.")
                
                if proba is not None:
                    st.write(f"**Confidence Score (Churn Probability):** {proba:.2%}")
                    st.progress(proba)
                    
            except Exception as e:
                st.error(f"Prediction Failed: {e}")
        else:
            st.warning("Model not found. Please train the model first.")

with tab2:
    st.header("📄 Batch Prediction from CSV")
    st.write("Upload a CSV file with employee data to get bulk predictions.")
    
    st.info("""
    **Required Columns:** 
    `satisfaction_level`, `last_evaluation`, `number_project`, `average_montly_hours`, 
    `time_spend_company`, `Work_accident`, `promotion_last_5years`, `Departments`, `salary`
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            # Robust column handling: strip whitespace from column names
            input_df.columns = input_df.columns.str.strip()
            
            st.write("### Preview of Uploaded Data")
            st.dataframe(input_df.head())
            
            if model:
                # Validate columns
                required_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 
                                 'average_montly_hours', 'time_spend_company', 'Work_accident', 
                                 'promotion_last_5years', 'Departments', 'salary']
                
                missing_cols = [col for col in required_cols if col not in input_df.columns]
                
                if missing_cols:
                    st.error(f"The following columns are missing: {', '.join(missing_cols)}")
                else:
                    if st.button("Generate Batch Predictions"):
                        # Get only necessary columns for the model
                        predict_df = input_df[required_cols].copy()
                        
                        predictions = model.predict(predict_df)
                        input_df['Churn_Prediction'] = ["Leave" if p == 1 else "Stay" for p in predictions]
                        
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(predict_df)[:, 1]
                            input_df['Churn_Probability'] = [f"{p:.2%}" for p in probabilities]
                        
                        st.subheader("Results")
                        st.dataframe(input_df)
                        
                        # Download results
                        csv_results = input_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv_results,
                            file_name="employee_churn_results.csv",
                            mime="text/csv",
                        )
            else:
                st.warning("Model not found. Please train the model first.")
                
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
