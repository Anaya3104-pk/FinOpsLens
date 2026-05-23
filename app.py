import streamlit as pd_st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configurations to give it an enterprise SaaS product look
pd_st.set_page_config(page_title="FinOpsLens Dashboard", page_icon="🔍", layout="wide")

# 1. Title and Header Section
pd_st.title("🔍 FinOpsLens: Automated Cloud Cost & Resource Optimizer")
pd_st.markdown("### Real-Time Infrastructure Performance Analytics & Anomaly Detection")
pd_st.divider()

# Ensure backend data is available before running
if not os.path.exists("Data/cleaned_cloud_metrics.csv") or not os.path.exists("Data/zombie_analysis_output.csv"):
    pd_st.error("❌ Operational metric logs not detected. Please run 'python main.py' in your terminal first to generate data and model states.")
else:
    # 2. Ingest Data Layers
    df = pd.read_csv("Data/cleaned_cloud_metrics.csv")
    zombie_df = pd.read_csv("Data/zombie_analysis_output.csv")
    
    with open('artifacts/xgboost_forecaster.pkl', 'rb') as f:
        forecaster_model = pickle.load(f)
        
    # 3. Calculate Financial Metrics (FinOps Math)
    total_hours = df['timestamp'].nunique()
    total_current_cost = df['cost_per_hour'].sum()
    
    # Calculate savings from eliminating the Zombie Server completely
    zombie_ids = zombie_df[zombie_df['is_zombie'] == True]['server_id'].tolist()
    zombie_hourly_waste = zombie_df[zombie_df['is_zombie'] == True]['cost_per_hour'].sum()
    total_zombie_waste = zombie_hourly_waste * (total_hours / len(zombie_df['server_id'].unique()))
    
    potential_savings = total_zombie_waste
    optimized_cost = total_current_cost - potential_savings

    # 4. Top-Tier KPI Metrics Row
    col1, col2, col3 = pd_st.columns(3)
    col1.metric(label="Total Current Spend (30 Days)", value=f"${total_current_cost:,.2f}")
    col2.metric(label="Potential FinOps Savings", value=f"${potential_savings:,.2f}", delta=f"-{(potential_savings/total_current_cost)*100:.1f}% Waste", delta_color="inverse")
    col3.metric(label="Optimized Monthly Target Cost", value=f"${optimized_cost:,.2f}")
    
    pd_st.divider()
    
    # 5. Two-Column Analytical Layout
    left_col, right_col = pd_st.columns([1, 1])
    
    with left_col:
        pd_st.subheader("🚨 Unsupervised Anomaly Desk: Zombie Servers")
        pd_st.markdown("The **DBSCAN Clustering Model** isolated these instances because they maintain active baseline rental costs but present near-zero utilization density.")
        
        # Display clean breakdown of zombie assets
        display_zombie = zombie_df.copy()
        display_zombie.columns = ['Server ID', 'Avg CPU %', 'Avg Memory %', 'Hourly Cost ($)', 'DBSCAN Cluster', 'Is Waste']
        pd_st.dataframe(display_zombie.style.background_gradient(subset=['Avg CPU %'], cmap='Reds_r'), use_container_width=True)
        
        for zombie in zombie_ids:
            pd_st.warning(f"⚠️ **Action Recommended:** Terminate **{zombie}** immediately. Continuous idle state detected. Monthly waste: **${total_zombie_waste:,.2f}**")

    with right_col:
        pd_st.subheader("🔮 Supervised Scaling Engine: Workload Forecaster")
        pd_st.markdown("Select an active cloud server to run our **XGBoost Regressor** across the next 24 hours of operational telemetry.")
        
        # Dropdown to filter by server
        available_servers = [s for s in df['server_id'].unique() if s not in zombie_ids]
        selected_server = pd_st.selectbox("Select Active Production Instance:", available_servers)
        
        # Filter server data and isolate the last 24 records for visual comparison
        srv_data = df[df['server_id'] == selected_server].tail(24).copy()
        
        # Use our features matrix to generate interactive predictions
        features = ['hour', 'day_of_week', 'is_weekend', 'cpu_lag_1hr']
        srv_data['predicted_cpu'] = forecaster_model.predict(srv_data[features])
        
        # Clean timestamps for neat chart rendering
        srv_data['Time'] = pd.to_datetime(srv_data['timestamp']).dt.strftime('%H:%M')
        chart_data = srv_data.set_index('Time')[['cpu_utilization', 'predicted_cpu']]
        chart_data.columns = ['Actual CPU Load (%)', 'XGBoost Predicted Load (%)']
        
        # Render beautiful line chart overlay
        pd_st.line_chart(chart_data, use_container_width=True)
        
        # Compute dynamic advice based on future workload threshold maximums
        max_predicted_load = srv_data['predicted_cpu'].max()
        if max_predicted_load < 30.0:
            pd_st.info(f"💡 **FinOps Scaling Advice:** Predicted peak load is only {max_predicted_load:.1f}%. You can safely downgrade this instance type to save up to 50% extra cost.")
        else:
            pd_st.success(f"💡 **FinOps Scaling Advice:** Instance size is appropriate. Predicted spikes up to {max_predicted_load:.1f}% require current computing limits.")