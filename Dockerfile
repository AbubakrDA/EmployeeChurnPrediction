FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Explicitly installing libraries used in the project
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    joblib \
    fastapi \
    uvicorn \
    streamlit \
    xgboost \
    pydantic

# Copy project files
COPY . .

# Expose ports for API (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command: Run API
# You can override this to run Streamlit: "streamlit", "run", "EmployeeChurnPred.py"
CMD ["uvicorn", "employeechurnFastapi:app", "--host", "0.0.0.0", "--port", "8000"]
