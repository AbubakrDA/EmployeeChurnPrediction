import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Optional imports
models_available = {}

# RandomForest is standard
models_available['RandomForest'] = RandomForestClassifier(random_state=42)

try:
    from xgboost import XGBClassifier
    models_available['XGBoost'] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
except ImportError:
    print("XGBoost not installed. Skipping.")

try:
    from lightgbm import LGBMClassifier
    models_available['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)
except ImportError:
    print("LightGBM not installed. Skipping.")

try:
    from catboost import CatBoostClassifier
    models_available['CatBoost'] = CatBoostClassifier(random_state=42, verbose=0)
except ImportError:
    print("CatBoost not installed. Skipping.")


def main():
    # 1. Load Data
    file_path = 'HR_Dataset.csv'
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # 2. Clean Column Names
    df.columns = df.columns.str.strip()
    print("Columns:", df.columns.tolist())

    # 3. Define Features and Target
    target = 'left'
    X = df.drop(columns=[target])
    y = df[target]

    # 4. Define Feature Groups
    numeric_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                        'average_montly_hours', 'time_spend_company']
    categorical_features = ['Departments']
    ordinal_features = ['salary']
    ordinal_categories = [['low', 'medium', 'high']]
    passthrough_features = ['Work_accident', 'promotion_last_5years']

    # 5. Build Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
            ('pass', 'passthrough', passthrough_features)
        ],
        remainder='drop' 
    )

    # 6. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 7. Train and Evaluate each Model
    if not models_available:
        print("No models available to train.")
        return

    best_accuracy = 0.0
    best_model = None
    best_model_name = ""

    for name, model in models_available.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        try:
            pipeline.fit(X_train, y_train)
            
            print(f"Evaluating {name}...")
            y_pred = pipeline.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = pipeline
                best_model_name = name

        except Exception as e:
            print(f"Failed to train/evaluate {name}: {e}")

    # 8. Save Best Model
    if best_model:
        model_filename = "best_churn_model.pkl"
        joblib.dump(best_model, model_filename)
        print(f"\nBest model was {best_model_name} with accuracy {best_accuracy:.4f}.")
        print(f"Model saved to {model_filename}")

        # 9. Test Sample Prediction
        print("\n--- Testing Sample Prediction ---")
        sample_data = pd.DataFrame([{
            'satisfaction_level': 0.1,
            'last_evaluation': 0.9,
            'number_project': 6,
            'average_montly_hours': 250,
            'time_spend_company': 4,
            'Work_accident': 0,
            'promotion_last_5years': 0,
            'Departments': 'sales',
            'salary': 'low'
        }])
        
        try:
            prediction = best_model.predict(sample_data)[0]
            pred_label = "Left" if prediction == 1 else "Stayed"
            print("Sample Input:")
            print(sample_data.to_string(index=False))
            print(f"Prediction: {pred_label}")
        except Exception as e:
            print(f"Prediction failed on sample data: {e}")

if __name__ == "__main__":
    main()
