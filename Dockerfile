# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports for API (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Create a startup script to run both services locally as a fallback
RUN echo '#!/bin/bash\n\
uvicorn Fastapi:app --host 0.0.0.0 --port 8000 &\n\
streamlit run EmployeeChurnPred.py --server.port 8501 --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Default command: Run API
# Overridden by docker-compose for UI service
# Use CMD ["./start.sh"] if you want to run both in a single container
CMD ["uvicorn", "Fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
