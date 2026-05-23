import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecaster(model, X_test, y_test):
    """
    Evaluates the trained XGBoost model using unseen test data 
    and prints key regression performance metrics.
    """
    print("\n📊 Evaluating Workload Forecaster Performance...")
    
    # 1. Generate predictions using our test features
    predictions = model.predict(X_test)
    
    # 2. Calculate Evaluation Metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # 3. Print a clean, readable report
    print("-" * 45)
    print(f"📈 Mean Absolute Error (MAE):  {mae:.2f}% CPU")
    print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}% CPU")
    print("-" * 45)
    
    # An MAE under 5% is excellent for synthetic operational log data
    if mae < 5.0:
        print("💡 Summary: The model is highly accurate. Ready for FinOps autoscaling recommendations.")
    else:
        print("💡 Summary: Model predictions have variance. Consider further hyperparameter tuning.")
        
    return {"MAE": mae, "RMSE": rmse}

if __name__ == "__main__":
    # This block allows testing when running evaluation.py directly
    import pickle
    import os
    from model_training import train_workload_forecaster
    
    if os.path.exists('artifacts/xgboost_forecaster.pkl'):
        # Reload existing model and generate a test split to evaluate
        with open('artifacts/xgboost_forecaster.pkl', 'rb') as f:
            model = pickle.load(f)
        df = pd.read_csv("Data/cleaned_cloud_metrics.csv")
        features = ['hour', 'day_of_week', 'is_weekend', 'cpu_lag_1hr']
        split_idx = int(len(df) * 0.8)
        X_test = df[features].iloc[split_idx:]
        y_test = df['cpu_utilization'].iloc[split_idx:]
        
        evaluate_forecaster(model, X_test, y_test)
    else:
        print("🔄 Model artifact not found. Let's run a quick training cycle first...")
        model, X_test, y_test = train_workload_forecaster()
        evaluate_forecaster(model, X_test, y_test)