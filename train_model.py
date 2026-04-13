import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import joblib
import os
import json
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Set an experiment name (creates it if it doesn't exist)
experiment_name = "predictive-maintenance"
mlflow.set_experiment(experiment_name)
print(f"MLflow Version: {mlflow.__version__}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")







df = pd.read_csv('data/ai4i2020.csv')

failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

df.columns = df.columns.str.strip()

# Define correctly
failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Drop safely
df = df.drop(columns=failure_columns + ['UDI', 'Product ID'], errors='ignore')

# Encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# Split
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Save feature names
feature_names = X.columns.tolist()

# Define parameters
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "class_weight": "balanced"
}

# Start an MLflow run
with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Log parameters
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Log model with signature and input example
    signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
    input_example = X_train_scaled[:5]
    
    # Log the model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="predictive-maintenance-model" # This registers it!
    )
    
    # Log preprocessing objects as artifacts
    os.makedirs("temp_artifacts", exist_ok=True)
    joblib.dump(scaler, "temp_artifacts/scaler.pkl")
    joblib.dump(le, "temp_artifacts/label_encoder.pkl")
    with open("temp_artifacts/feature_names.txt", "w") as f:
        f.write(','.join(feature_names))
    
    mlflow.log_artifacts("temp_artifacts", artifact_path="preprocessing")
    
    # Clean up temp artifacts
    import shutil
    shutil.rmtree("temp_artifacts")
    
    print(f"Run finished. Metrics: {metrics}")
