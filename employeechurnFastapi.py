from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Employee Churn Prediction API")

# Load the saved model pipeline
model = joblib.load('churn_model.joblib')

class EmployeeData(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    Work_accident: int
    promotion_last_5years: int
    department: str
    salary: str

@app.post("/predict")
def predict_churn(data: EmployeeData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([{
        'satisfaction_level': data.satisfaction_level,
        'last_evaluation': data.last_evaluation,
        'number_project': data.number_project,
        'average_montly_hours': data.average_montly_hours,
        'time_spend_company': data.time_spend_company,
        'Work_accident': data.Work_accident,
        'promotion_last_5years': data.promotion_last_5years,
        'Departments ': data.department,  # Match the trailing space in the model
        'salary': data.salary
    }])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]
    
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0])
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Employee Churn Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
