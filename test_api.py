import requests

url = "http://localhost:8000/predict"
data = {
    "satisfaction_level": 0.65,
    "last_evaluation": 0.80,
    "number_project": 4,
    "average_montly_hours": 200,
    "time_spend_company": 3,
    "Work_accident": 0,
    "promotion_last_5years": 0,
    "Departments": "sales",
    "salary": "medium"
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
