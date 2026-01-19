import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('HR_Dataset.csv')

# Preprocessing: Handle duplicates
df = df.drop_duplicates()

# Split into features and target
X = df.drop(columns=['left'])
y = df['left']

# Define the ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['satisfaction_level', 'last_evaluation', 'number_project', 
                               'average_montly_hours', 'time_spend_company', 'Work_accident', 
                               'promotion_last_5years']),
    ('nominal', OneHotEncoder(), ['Departments ']),
    ('ordinal', OrdinalEncoder(), ['salary'])
], remainder='passthrough')

# Define the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])

# Fit the model on the entire dataset
print("Training the final model...")
model_pipeline.fit(X, y)

# Save the model
model_filename = 'churn_model.joblib'
joblib.dump(model_pipeline, model_filename)
print(f"Model saved as {model_filename}")
