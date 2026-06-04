import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(output_path="Data/cloud_metrics.csv"):
    """Generates 30 days of realistic hourly cloud resource metrics for 5 distinct servers."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    start_date = datetime(2026, 5, 1)
    total_hours = 24 * 30
    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]
    
    servers = {
        'srv_prod_web_01':   {'type': 'm5.xlarge', 'cost_per_hr': 0.192},
        'srv_prod_web_02':   {'type': 'm5.xlarge', 'cost_per_hr': 0.192},
        'srv_worker_01':     {'type': 'c5.large',  'cost_per_hr': 0.085},
        'srv_worker_02':     {'type': 'c5.large',  'cost_per_hr': 0.085},
        'srv_zombie_idle':   {'type': 'r5.xlarge', 'cost_per_hr': 0.252} 
    }
    
    data_rows = []
    for dt in timestamps:
        hour = dt.hour
        day_of_week = dt.weekday()
        
        for srv_id, srv_meta in servers.items():
            noise = np.random.normal(0, 2)
            
            if 'web' in srv_id:
                base_cpu = 25 + 15 * np.sin(2 * np.pi * (hour - 6) / 24) + 20 * np.sin(2 * np.pi * (hour - 15) / 12)
                if day_of_week >= 4: base_cpu += 10
                cpu = max(5, min(95, base_cpu + noise))
                memory = max(20, min(90, cpu * 0.8 + 15 + np.random.normal(0, 3)))
                network_in = max(10, cpu * 1.5 + np.random.normal(0, 5))
                
            elif 'worker' in srv_id:
                cpu = max(35, min(65, 45 + np.random.normal(0, 4)))
                memory = max(50, min(70, 60 + np.random.normal(0, 2)))
                network_in = max(5, 12 + np.random.normal(0, 2))
                
            elif 'zombie' in srv_id:
                cpu = max(1, min(4, 2.1 + np.random.normal(0, 0.4)))
                memory = max(8, min(12, 10.0 + np.random.normal(0, 0.5)))
                network_in = max(0.1, 0.5 + np.random.normal(0, 0.1))
                
            network_out = network_in * 1.2
            
            data_rows.append({
                'timestamp': dt, 'server_id': srv_id, 'instance_type': srv_meta['type'],
                'cost_per_hour': srv_meta['cost_per_hr'], 'cpu_utilization': round(cpu, 2),
                'memory_usage': round(memory, 2), 'network_in_mb': round(network_in, 2), 'network_out_mb': round(network_out, 2)
            })
            
    df = pd.DataFrame(data_rows)
    df.to_csv(output_path, index=False)
    print(f"Raw operational metrics compiled to {output_path}")
    return df

def preprocess_metrics(input_path="Data/cloud_metrics.csv", output_path="Data/cleaned_cloud_metrics.csv"):
    """Transforms raw server metrics into distinct mathematical features for models."""
    if not os.path.exists(input_path):
        df = generate_synthetic_data(input_path)
    else:
        df = pd.read_csv(input_path)
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.weekday
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    df['cpu_lag_1hr'] = df.groupby('server_id')['cpu_utilization'].shift(1)
    df = df.dropna().copy()
    
    df.to_csv(output_path, index=False)
    print(f"Features engineered and written to {output_path}")
    return df

if __name__ == "__main__":
    preprocess_metrics()