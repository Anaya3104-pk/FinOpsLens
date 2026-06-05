import pandas as pd
import numpy as np
import os
import pickle
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def train_workload_forecaster(data_path="Data/cleaned_cloud_metrics.csv"):
    """Trains multiple regression models and saves their MAE, RMSE, and R2 scores."""
    df = pd.read_csv(data_path)
    features = ['hour', 'day_of_week', 'is_weekend', 'cpu_lag_1hr']
    target = 'cpu_utilization'
    
    X = df[features]
    y = df[target]
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    comparison_results = []
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        comparison_results.append({
            "Model": name,
            "MAE (% CPU)": round(mae, 2),
            "RMSE (% CPU)": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        trained_models[name] = model

    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv("Data/regression_comparison.csv", index=False)
    
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/xgboost_forecaster.pkl', 'wb') as f:
        pickle.dump(trained_models["XGBoost"], f)
        
    print("Regression Model Comparison with R2 Complete!")
    return trained_models["XGBoost"], X_test, y_test

def detect_zombie_servers(data_path="Data/cleaned_cloud_metrics.csv"):
    """Runs DBSCAN to detect zombie servers based on density clustering."""
    df = pd.read_csv(data_path)
    
    server_profiles = df.groupby('server_id').agg({
        'cpu_utilization': 'mean',
        'memory_usage': 'mean',
        'cost_per_hour': 'mean'
    }).reset_index()
    
    features = ['cpu_utilization', 'memory_usage', 'cost_per_hour']
    X = server_profiles[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dbscan = DBSCAN(eps=1.0, min_samples=2)
    server_profiles['cluster'] = dbscan.fit_predict(X_scaled)
    server_profiles['is_zombie'] = (server_profiles['cluster'] == -1) & (server_profiles['cpu_utilization'] < 10.0)
    
    server_profiles.to_csv("Data/zombie_analysis_output.csv", index=False)
    print("DBSCAN Complete.")
    return server_profiles