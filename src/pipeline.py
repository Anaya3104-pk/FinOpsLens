import os
import sys

# Ensure the root directory is in the python path so imports find 'src' smoothly
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
    print("🎬 Starting the FinOpsLens ML Pipeline Execution...\n")
    
    # Step 1: Ingest and clean resource telemetry
    print("[STEP 1/4] Processing Cloud Infrastructure Logs...")
    preprocess_metrics()
    
    # Step 2: Train Supervised load forecaster
    print("\n[STEP 2/4] Training XGBoost Workload Forecaster...")
    model, X_test, y_test = train_workload_forecaster()
    
    # Step 3: Run evaluation metrics on the forecaster
    print("\n[STEP 3/4] Evaluating Forecaster Accuracy...")
    metrics = evaluate_forecaster(model, X_test, y_test)
    
    # Step 4: Execute unsupervised anomaly detection to catch zombie assets
    print("\n[STEP 4/4] Executing DBSCAN Anomaly Detection...")
    zombie_profiles = detect_zombie_servers()
    
    print("\n🏆 Full FinOpsLens ML Pipeline executed successfully! All outputs generated.")
    return model, zombie_profiles

if __name__ == "__main__":
    run_full_pipeline() 