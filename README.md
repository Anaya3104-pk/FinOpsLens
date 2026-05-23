# 🔍 FinOpsLens: Automated Cloud Cost & Resource Optimizer

**FinOpsLens** is an intelligent, data-driven cloud financial operations (FinOps) platform that targets multi-billion dollar cloud resource wastage. By combining **Supervised Time-Series Forecasting** with **Unsupervised Anomaly Detection**, the system proactively predicts scaling demands while simultaneously identifying and isolating high-cost, underutilized "Zombie Servers."

---

## 🚀 Key Features
* **Supervised Workload Forecasting:** Utilizes an **XGBoost Regressor** to analyze cyclical traffic behaviors (e.g., peak lunch/dinner rushes, weekend surges) and predict computing loads for the next 24 hours with an exceptional **2.48% Mean Absolute Error (MAE)**.
* **Unsupervised Waste Detection:** Leverages **DBSCAN (Density-Based Spatial Clustering)** to profile monthly server footprints. It isolates low-density, high-cost operational anomalies as noise clusters (-1), flagging them instantly as structural waste.
* **Automated Decision Engine:** Features a rules-based cost engine that translates ML outputs into real-time dollars saved, actionable rightsizing steps, and direct termination recommendations.
* **Interactive Executive Dashboard:** A production-grade UI engineered with **Streamlit** that features interactive workload charts, real-time cost tracking, and warning desks.

---

## 🏗️ System Architecture & Data Pipeline

The pipeline routes raw telemetry seamlessly through distinct processing layers:

1. **Ingestion & Feature Engineering (`src/data_preprocessing.py`):** Converts raw infrastructure metrics into mathematical inputs by creating temporal indices (`hour`, `day_of_week`) and sequential tracking dimensions (`cpu_lag_1hr`).
2. **Parallel Model Execution (`src/model_training.py`):**
   * Trains the predictive forecasting model chronologically.
   * Scales spatial telemetry using `StandardScaler` and clusters server signatures with DBSCAN.
3. **Validation & Quality Gate (`src/evaluation.py`):** Quantifies predictive errors via MAE and RMSE before allowing recommendations to flow to the interface layer.
4. **Interactive Visualization Layer (`app.py`):** Renders actionable financial KPI metrics and graph overlays.

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Frontend UI:** Streamlit

---

## 🏃‍♂️ Installation & Quickstart

### 1. Clone the Workspace
```bash
git clone [https://github.com/Anaya3104-pk/FinOpsLens.git](https://github.com/Anaya3104-pk/FinOpsLens.git)
cd FinOpsLens