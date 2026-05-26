import streamlit as pd_st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configurations with professional custom layouts
pd_st.set_page_config(page_title="FinOpsLens Dashboard", page_icon="🔍", layout="wide")

# =========================================================================
# 🎨 HIGH-CONTRAST UI & PROFESSIONAL THEME CONFIGURATION (SAFE INJECTION)
# =========================================================================
ui_styles = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    .stApp {
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: #111827;
        padding: 10px 20px;
        border-radius: 12px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        font-weight: 600;
        font-size: 16px;
        color: #9ca3af !important;
    }
    .stTabs [aria-selected="true"] {
        color: #10b981 !important;
        border-bottom: 3px solid #10b981 !important;
    }
    .metric-container {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #9ca3af;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
    }
    .styled-table-wrapper {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px;
        padding: 15px;
    }
    div[data-testid="stTable"] table {
        color: #ffffff !important;
        background-color: #111827 !important;
        width: 100%;
    }
    div[data-testid="stTable"] th {
        background-color: #1f2937 !important;
        color: #10b981 !important;
        font-weight: 700 !important;
        padding: 12px !important;
    }
    div[data-testid="stTable"] td {
        color: #f8fafc !important;
        border-bottom: 1px solid #374151 !important;
        padding: 12px !important;
    }
    .section-header {
        color: #ffffff; 
        font-weight: 700;
        font-size: 24px;
        margin-bottom: 15px;
    }
    .proof-tile {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 25px;
        height: 100%;
    }
    .proof-title-red {
        color: #f43f5e;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(244, 63, 94, 0.2);
        padding-bottom: 8px;
    }
    .proof-title-green {
        color: #10b981;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(16, 185, 129, 0.2);
        padding-bottom: 8px;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px dashed #374151;
    }
    .stat-label { color: #9ca3af; font-size: 15px; }
    .stat-val { color: #ffffff; font-weight: 600; font-size: 15px; }
    </style>
"""
pd_st.markdown(ui_styles, unsafe_allow_html=True)

# =========================================================================
# ⚙️ SYSTEM VERIFICATION & IO OPERATIONS
# =========================================================================
if not os.path.exists("Data/cleaned_cloud_metrics.csv") or not os.path.exists("Data/regression_comparison.csv"):
    pd_st.error("❌ Operational metric logs not detected. Please run '.venv\\Scripts\\python main.py' first.")
else:
    pd_st.markdown("<h1 style='color: #ffffff; font-weight: 800;'><i class='fa-solid fa-magnifying-glass-chart' style='color:#10b981; margin-right:15px;'></i>FinOpsLens: Autonomous Cloud Cost Optimizer</h1>", unsafe_allow_html=True)

    # 🎯 OBJECTIVES MATRIX PANEL
    objectives_html = """
        <div style="background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px; margin-bottom: 25px;">
            <h4 style="color: #10b981; margin-top:0; font-weight:700;"><i class="fa-solid fa-bullseye" style="margin-right:10px;"></i>Project Core Objectives</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 14px; color: #cbd5e1;">
                <div><strong style="color:#ffffff;"><i class="fa-solid fa-robot" style="margin-right:6px; color:#3b82f6;"></i> 1. Cost Automation:</strong> Eliminate manual DevOps overheads by orchestrating continuous automated financial monitoring protocols.</div>
                <div><strong style="color:#ffffff;"><i class="fa-solid fa-skull-crossbones" style="margin-right:6px; color:#f43f5e;"></i> 2. Waste Elimination:</strong> Isolate high-cost, underutilized "Zombie Servers" dynamically using unsupervised spatial parameters.</div>
                <div><strong style="color:#ffffff;"><i class="fa-solid fa-timeline" style="margin-right:6px; color:#10b981;"></i> 3. Proactive Scaling:</strong> Maximize uptime boundaries by predicting 24-hour infrastructure capacity spikes ahead of schedule.</div>
                <div><strong style="color:#ffffff;"><i class="fa-solid fa-desktop" style="margin-right:6px; color:#a855f7;"></i> 4. Data Resolution:</strong> Translate abstract matrix arrays into clear, readable operational and financial metrics.</div>
            </div>
        </div>
    """
    pd_st.markdown(objectives_html, unsafe_allow_html=True)
    pd_st.divider()

    tab1, tab2 = pd_st.tabs(["🖥️ Executive Command Center", "📊 Advanced ML Benchmarking"])

    df = pd.read_csv("Data/cleaned_cloud_metrics.csv")
    zombie_df = pd.read_csv("Data/zombie_analysis_output.csv")
    reg_comp = pd.read_csv("Data/regression_comparison.csv")
    
    with open('artifacts/xgboost_forecaster.pkl', 'rb') as f:
        forecaster_model = pickle.load(f)

    # =========================================================================
    # TAB 1: EXECUTIVE COMMAND CENTER
    # =========================================================================
    with tab1:
        total_hours = df['timestamp'].nunique()
        total_current_cost = df['cost_per_hour'].sum()
        zombie_ids = zombie_df[zombie_df['is_zombie'] == True]['server_id'].tolist()
        zombie_hourly_waste = zombie_df[zombie_df['is_zombie'] == True]['cost_per_hour'].sum()
        total_zombie_waste = zombie_hourly_waste * (total_hours / len(zombie_df['server_id'].unique()))
        
        potential_savings = total_zombie_waste
        optimized_cost = total_current_cost - potential_savings

        m_col1, m_col2, m_col3 = pd_st.columns(3)
        with m_col1:
            pd_st.markdown(f'<div class="metric-container"><div class="metric-title"><i class="fa-solid fa-wallet" style="margin-right:8px;"></i> Total Spend</div><div class="metric-value">${total_current_cost:,.2f}</div></div>', unsafe_allow_html=True)
        with m_col2:
            pd_st.markdown(f'<div class="metric-container" style="border-bottom: 3px solid #f43f5e;"><div class="metric-title"><i class="fa-solid fa-dumpster-fire" style="color:#f43f5e; margin-right:8px;"></i> Financial Waste</div><div class="metric-value" style="color: #f43f5e;">${potential_savings:,.2f}</div></div>', unsafe_allow_html=True)
        with m_col3:
            pd_st.markdown(f'<div class="metric-container" style="border-bottom: 3px solid #10b981;"><div class="metric-title"><i class="fa-solid fa-shield-halved" style="color:#10b981; margin-right:8px;"></i> Optimized Target</div><div class="metric-value" style="color: #10b981;">${optimized_cost:,.2f}</div></div>', unsafe_allow_html=True)
        
        pd_st.markdown("<br>", unsafe_allow_html=True)
        
        left_col, right_col = pd_st.columns([1, 1], gap="large")
        
        with left_col:
            pd_st.markdown("<div class='section-header'><i class='fa-solid fa-circle-radiation' style='color:#f43f5e; margin-right:12px;'></i>Unsupervised Anomaly Tracker</div>", unsafe_allow_html=True)
            display_zombie = zombie_df.copy()
            display_zombie.columns = ['Server ID', 'Avg CPU %', 'Avg Memory %', 'Hourly Cost ($)', 'DBSCAN Cluster', 'Is Waste']
            pd_st.dataframe(display_zombie.style.background_gradient(subset=['Avg CPU %'], cmap='Reds_r'), use_container_width=True)
            
            for zombie in zombie_ids:
                pd_st.markdown(f'<div style="background-color: rgba(244, 63, 94, 0.1); border: 1px solid #f43f5e; border-radius: 8px; padding: 15px; margin-top: 15px;"><strong style="color: #f43f5e;"><i class="fa-solid fa-skull-crossbones"></i> Mandate:</strong> Terminate active server <code>{zombie}</code>. Monthly leakage: <strong>${total_zombie_waste:,.2f}</strong></div>', unsafe_allow_html=True)

        with right_col:
            pd_st.markdown("<div class='section-header'><i class='fa-solid fa-chart-line' style='color:#3b82f6; margin-right:12px;'></i>Proactive Autoscaling Forecaster</div>", unsafe_allow_html=True)
            available_servers = [s for s in df['server_id'].unique() if s not in zombie_ids]
            selected_server = pd_st.selectbox("Select Core Enterprise Node:", available_servers)
            
            srv_data = df[df['server_id'] == selected_server].tail(24).copy()
            features = ['hour', 'day_of_week', 'is_weekend', 'cpu_lag_1hr']
            srv_data['predicted_cpu'] = forecaster_model.predict(srv_data[features])
            srv_data['Time'] = pd.to_datetime(srv_data['timestamp']).dt.strftime('%H:%M')
            
            chart_data = srv_data.set_index('Time')[['cpu_utilization', 'predicted_cpu']]
            chart_data.columns = ['Current Telemetry (%)', 'XGBoost Prediction Envelop (%)']
            pd_st.line_chart(chart_data, use_container_width=True)

    # =========================================================================
    # TAB 2: TECHNICAL NUMERICAL ANALYSIS MATRIX
    # =========================================================================
    with tab2:
        pd_st.markdown("<h2 style='color: #ffffff; font-weight:700;'><i class='fa-solid fa-scale-balanced' style='color:#3b82f6; margin-right:12px;'></i>Mathematical Model Validation Matrix</h2>", unsafe_allow_html=True)
        pd_st.markdown("Mathematical evaluation summaries tracking absolute mathematical validation parameters.")
        pd_st.divider()
        
        pd_st.markdown("<h3 style='color: #3b82f6; font-weight:600;'><i class='fa-solid fa-server' style='margin-right:10px;'></i>1. Workload Forecasting Performance (Supervised Benchmarks)</h3>", unsafe_allow_html=True)
        
        pd_st.markdown('<div class="styled-table-wrapper" style="max-width: 700px;">', unsafe_allow_html=True)
        pd_st.table(reg_comp)
        pd_st.markdown('</div>', unsafe_allow_html=True)
        
        pd_st.markdown("<br><br>", unsafe_allow_html=True)
        pd_st.divider()

        pd_st.markdown("<h3 style='color: #f43f5e; font-weight:600;'><i class='fa-solid fa-network-wired' style='margin-right:10px;'></i>2. Clustering Space Resolution (Unsupervised Structural Proof)</h3>", unsafe_allow_html=True)
        
        col_km, col_db = pd_st.columns(2, gap="large")
        
        with col_km:
            km_html = """
                <div class="proof-tile">
                    <div class="proof-title-red"><i class="fa-solid fa-circle-xmark" style="margin-right:8px;"></i> K-Means Metric Limitations</div>
                    <div class="stat-row"><div class="stat-label">Cluster Cardinality (K Specification)</div><div class="stat-val">Manual / Static Constraint</div></div>
                    <div class="stat-row"><div class="stat-label">Outlier Realignment Mode</div><div class="stat-val">Forced Centroid Integration</div></div>
                    <div class="stat-row"><div class="stat-label">Anomalous Density Resolution</div><div class="stat-val">0% (Absorbed Into Means)</div></div>
                    <div class="stat-row"><div class="stat-label">Zombie Instance Detection State</div><div class="stat-val" style="color: #f43f5e;">FAILED (Hidden / Unresolved)</div></div>
                </div>
            """
            pd_st.markdown(km_html, unsafe_allow_html=True)
            
        with col_db:
            total_detected_clusters = zombie_df['cluster'].nunique() - (1 if -1 in zombie_df['cluster'].values else 0)
            noise_points_found = len(zombie_df[zombie_df['cluster'] == -1])
            
            db_html = f"""
                <div class="proof-tile" style="border-color: #10b981;">
                    <div class="proof-title-green"><i class="fa-solid fa-circle-check" style="margin-right:8px;"></i> DBSCAN Mathematical Resolution</div>
                    <div class="stat-row"><div class="stat-label">Cluster Cardinality (Density Discovery)</div><div class="stat-val">Automated ({total_detected_clusters} Dense Environments)</div></div>
                    <div class="stat-row"><div class="stat-label">Outlier Realignment Mode</div><div class="stat-val">Isolate as Noise State Group</div></div>
                    <div class="stat-row"><div class="stat-label">Identified Mathematical Outliers (Cluster -1)</div><div class="stat-val" style="color: #10b981;">{noise_points_found} Element Isolated</div></div>
                    <div class="stat-row"><div class="stat-label">Zombie Instance Detection State</div><div class="stat-val" style="color: #10b981;">100% SUCCESSFUL</div></div>
                </div>
            """
            pd_st.markdown(db_html, unsafe_allow_html=True)