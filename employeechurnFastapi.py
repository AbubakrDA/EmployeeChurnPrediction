from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# Define Input Schema
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

app = FastAPI(title="Employee Churn Prediction API")

# Load Model
MODEL_PATH = "best_churn_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.get("/")
def home():
    return {"message": "Welcome to Employee Churn Prediction API"}

@app.post("/predict")
def predict_churn(input_data: EmployeeInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model could not be loaded.")
    
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Predict
        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None
        
        result = "Left" if prediction == 1 else "Stayed"
        
        return {
            "prediction_label": result,
            "prediction_score": int(prediction),
            "probability_churn": float(prob) if prob is not None else "N/A"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
