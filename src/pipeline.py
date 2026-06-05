import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_metrics
from src.model_training import train_workload_forecaster, detect_zombie_servers
from src.evaluation import evaluate_forecaster

def run_full_pipeline():
    """
    Orchestrates the entire FinOpsLens Machine Learning lifecycle.
    1. Ingests & Preprocesses data
    2. Trains the forecasting and clustering models
    3. Evaluates model performance
    """
    print("Starting the FinOpsLens ML Pipeline Execution...\n")
    
    print("[STEP 1/4] Processing Cloud Infrastructure Logs...")
    preprocess_metrics()
    
    print("\n[STEP 2/4] Training XGBoost Workload Forecaster...")
    model, X_test, y_test = train_workload_forecaster()
    
    print("\n[STEP 3/4] Evaluating Forecaster Accuracy...")
    metrics = evaluate_forecaster(model, X_test, y_test)
    
    print("\n[STEP 4/4] Executing DBSCAN Anomaly Detection...")
    zombie_profiles = detect_zombie_servers()
    
    print("\nFull FinOpsLens ML Pipeline executed successfully! All outputs generated.")
    return model, zombie_profiles

if __name__ == "__main__":
    run_full_pipeline() 