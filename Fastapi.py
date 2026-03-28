from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
import os
import time

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "best_churn_model.pkl")

# --- App Initialization ---
app = FastAPI(
    title="Employee Churn Prediction API",
    description="REST API to predict employee turnover using a trained Machine Learning model.",
    version="1.0.0"
)

# --- Data Models ---
class EmployeeInput(BaseModel):
    satisfaction_level: float = Field(..., ge=0, le=1, description="Satisfaction level (0.0 to 1.0)")
    last_evaluation: float = Field(..., ge=0, le=1, description="Last evaluation score (0.0 to 1.0)")
    number_project: int = Field(..., ge=2, le=10, description="Number of projects assigned")
    average_montly_hours: int = Field(..., ge=50, le=350, description="Average monthly working hours")
    time_spend_company: int = Field(..., ge=1, le=10, description="Years spent at the company")
    Work_accident: int = Field(..., ge=0, le=1, description="1 if work accident occurred, 0 otherwise")
    promotion_last_5years: int = Field(..., ge=0, le=1, description="1 if promoted in last 5 years, 0 otherwise")
    Departments: str = Field(..., description="Department name (e.g., sales, technical, etc.)")
    salary: str = Field(..., description="Salary level (low, medium, or high)")

    class Config:
        schema_extra = {
            "example": {
                "satisfaction_level": 0.38,
                "last_evaluation": 0.53,
                "number_project": 2,
                "average_montly_hours": 157,
                "time_spend_company": 3,
                "Work_accident": 0,
                "promotion_last_5years": 0,
                "Departments": "sales",
                "salary": "low"
            }
        }

class PredictionOutput(BaseModel):
    prediction_label: str
    prediction_score: int
    probability_churn: float
    processing_time_ms: float

# --- Model Loading ---
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")

# --- Endpoints ---
@app.get("/", tags=["General"])
async def root():
    """Health check and welcome message."""
    return {
        "status": "online",
        "message": "Employee Churn Prediction API is running",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(data: EmployeeInput):
    """
    Predict whether an employee will leave or stay.
    
    - **satisfaction_level**: 0.0 to 1.0
    - **last_evaluation**: 0.0 to 1.0
    - **number_project**: 2 to 10
    - **average_montly_hours**: 50 to 350
    - **time_spend_company**: 1 to 10
    - **Work_accident**: 0 or 1
    - **promotion_last_5years**: 0 or 1
    - **Departments**: sales, accounting, hr, technical, support, management, IT, product_mng, marketing, RandD
    - **salary**: low, medium, high
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    start_time = time.time()
    
    try:
        # Convert input Pydantic model to DataFrame for the scikit-learn pipeline
        input_df = pd.DataFrame([data.dict()])
        
        # Perform prediction
        prediction = model.predict(input_df)[0]
        
        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            churn_prob = float(probs[1])
        else:
            churn_prob = 1.0 if prediction == 1 else 0.0
            
        result_label = "Left" if prediction == 1 else "Stayed"
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionOutput(
            prediction_label=result_label,
            prediction_score=int(prediction),
            probability_churn=churn_prob,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("Fastapi:app", host="0.0.0.0", port=8000, reload=True)
