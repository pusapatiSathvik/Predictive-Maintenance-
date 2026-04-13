# app/model.py
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import os
from mlflow.tracking import MlflowClient

# Global variables
_model = None
_scaler = None
_label_encoder = None
_feature_names = None

# MLflow configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "predictive-maintenance"   # Must match the experiment name used in training
MODEL_NAME = "predictive-maintenance-model"  # Optional if you registered the model

def load_model_from_mlflow():
    """
    Load the latest trained model from the local MLflow tracking server.
    It uses the latest run of the specified experiment.
    """
    global _model, _scaler, _label_encoder, _feature_names
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Find the experiment ID
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Have you run train_model.py?")
    
    # Get the latest run (by start time)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{EXPERIMENT_NAME}'.")
    
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    artifact_uri = latest_run.info.artifact_uri
    print(f"Loading model from run_id: {run_id}")
    
    # Load the sklearn model
    _model = mlflow.sklearn.load_model(f"{artifact_uri}/model")
    
    # Load preprocessing artifacts (scaler, encoder, feature names)
    # MLflow logs artifacts under 'preprocessing' directory
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{artifact_uri}/preprocessing",
        dst_path="."
    )
    _scaler = joblib.load(os.path.join(local_path, "scaler.pkl"))
    _label_encoder = joblib.load(os.path.join(local_path, "label_encoder.pkl"))
    with open(os.path.join(local_path, "feature_names.txt"), "r") as f:
        _feature_names = f.read().strip().split(',')
    
    print("Model and preprocessing artifacts loaded successfully.")

def load_model():
    """Load model (called on FastAPI startup)."""
    if _model is None:
        load_model_from_mlflow()

def predict(input_data: dict) -> tuple:
    """Make prediction from input dictionary."""
    global _model, _scaler, _label_encoder, _feature_names
    if _model is None:
        load_model()
    
    # Convert input to DataFrame with correct column order
    df = pd.DataFrame([input_data])
    # Rename to match training feature names
    df.columns = ['Type', 'Air_temperature_K', 'Process_temperature_K',
                  'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
    df = df.rename(columns={
        'Air_temperature_K': 'Air temperature [K]',
        'Process_temperature_K': 'Process temperature [K]',
        'Rotational_speed_rpm': 'Rotational speed [rpm]',
        'Torque_Nm': 'Torque [Nm]',
        'Tool_wear_min': 'Tool wear [min]'
    })
    
    # Encode 'Type' column
    df['Type'] = _label_encoder.transform(df['Type'])
    
    # Ensure correct feature order
    df = df[_feature_names]
    
    # Scale features
    scaled = _scaler.transform(df)
    
    # Predict probability and class
    proba = _model.predict_proba(scaled)[0][1]  # Probability of failure
    pred = int(proba > 0.5)
    
    return pred, proba