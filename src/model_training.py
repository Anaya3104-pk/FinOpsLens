import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import pickle

def train_workload_forecaster(data_path="Data/cleaned_cloud_metrics.csv"):
    """Trains an XGBoost Regressor to predict the next hour's CPU utilization."""
    df = pd.read_csv(data_path)
    
    # 1. Define Features and Target
    # We use time context and recent momentum to forecast behavior
    features = ['hour', 'day_of_week', 'is_weekend', 'cpu_lag_1hr']
    target = 'cpu_utilization'
    
    X = df[features]
    y = df[target]
    
    # 2. Train-Test Split (Chronological split for time-series data)
    # Using the last 20% of data chronologically for evaluation
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print("🚀 Training Supervised XGBoost Forecaster...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model artifact
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/xgboost_forecaster.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("💾 Saved XGBoost model to artifacts/xgboost_forecaster.pkl")
    
    return model, X_test, y_test

def detect_zombie_servers(data_path="Data/cleaned_cloud_metrics.csv"):
    """Uses DBSCAN to cluster servers and flag high-cost, low-utilization outliers."""
    df = pd.read_csv(data_path)
    
    # 1. Profile each server by taking their historical averages
    # This condenses hourly telemetry into a single distinct behavioral footprint per server
    server_profiles = df.groupby('server_id').agg({
        'cpu_utilization': 'mean',
        'memory_usage': 'mean',
        'cost_per_hour': 'mean'
    }).reset_index()
    
    # 2. Scale features (Crucial step for distance-based clustering algorithms like DBSCAN)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(server_profiles[['cpu_utilization', 'cost_per_hour']])
    
    print("🔍 Running Unsupervised DBSCAN Outlier Detection...")
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    dbscan = DBSCAN(eps=1.0, min_samples=2)
    server_profiles['cluster'] = dbscan.fit_predict(scaled_features)
    
    # 3. Label identified anomalies
    # In DBSCAN, points labeled -1 do not belong to any dense cluster (Noise)
    # If a noise point has extremely low average CPU usage, it's flagged as a Zombie Resource
    server_profiles['is_zombie'] = np.where(
        (server_profiles['cluster'] == -1) & (server_profiles['cpu_utilization'] < 10), 
        True, 
        False
    )
    
    # Save the profiling results for use in the dashboard later
    server_profiles.to_csv("Data/zombie_analysis_output.csv", index=False)
    
    zombies = server_profiles[server_profiles['is_zombie'] == True]['server_id'].tolist()
    print(f"🚨 DBSCAN Analysis Complete. Flagged Zombie Servers: {zombies}")
    
    return server_profiles

if __name__ == "__main__":
    # Test execution block to verify functionality
    if os.path.exists("Data/cleaned_cloud_metrics.csv"):
        train_workload_forecaster()
        detect_zombie_servers()
    else:
        print("❌ Preprocessed data file not found. Please execute src/data_preprocessing.py first.")