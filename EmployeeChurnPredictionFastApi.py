# create fastapi app for employee churn prediction

import pandas as pd
from fastapi import FastAPI
import joblib
import uvicorn
import numpy as np
from pydantic import BaseModel

app = FastAPI()

class EmployeeInput(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    Work_accident: int
    promotion_last_5years: int
    Departments: str
    salary: str

# load the model
model = joblib.load('churn_model.joblib')

@app.get('/')
def read_root():
    return {'message': 'Welcome to the Employee Churn Prediction API'}  

@app.post('/predict')
def predict(data: EmployeeInput):
    # Map input to DataFrame and fix column name mismatch for 'Departments '
    input_data = data.model_dump()
    input_data['Departments '] = input_data.pop('Departments')
    df = pd.DataFrame([input_data])
    
    prediction = model.predict(df)

    if prediction[0] == 1:
        return {'prediction': 'Churn'}
    else:
        return {'prediction': 'Stay'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
