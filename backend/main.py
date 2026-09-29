from pathlib import Path
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBRegressor


# Project root: D:\FinOpsLens
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "Data"
MODEL_PATH = ROOT / "artifacts" / "xgboost_forecaster.json"


app = FastAPI(
    title="FinOpsLens ML API",
    version="1.0.0"
)


# Allow the Next.js frontend to communicate with the API locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_data():
    df = pd.read_csv(DATA / "cleaned_cloud_metrics.csv")
    zombie_df = pd.read_csv(DATA / "zombie_analysis_output.csv")
    regression = pd.read_csv(DATA / "regression_comparison.csv")

    return df, zombie_df, regression


@lru_cache(maxsize=1)
def load_model():
    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    return model


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "FinOpsLens ML API"
    }


@app.get("/api/dashboard")
def dashboard():
    df, zombie_df, regression = load_data()

    zombie_mask = zombie_df["is_zombie"] == True

    zombie_hourly_waste = float(
        zombie_df.loc[zombie_mask, "cost_per_hour"].sum()
    )

    total_hours = int(df["timestamp"].nunique())

    server_count = max(
        len(zombie_df["server_id"].unique()),
        1
    )

    total_current_cost = float(
        df["cost_per_hour"].sum()
    )

    potential_savings = (
        zombie_hourly_waste *
        (total_hours / server_count)
    )

    return {
        "total_hours": total_hours,
        "total_current_cost": total_current_cost,
        "potential_savings": potential_savings,
        "optimized_cost": total_current_cost - potential_savings,

        "zombie_servers": zombie_df.to_dict(
            orient="records"
        ),

        "servers": [
            str(server)
            for server in df["server_id"].unique()
            if server not in set(
                zombie_df.loc[
                    zombie_mask,
                    "server_id"
                ].tolist()
            )
        ],

        "regression": regression.to_dict(
            orient="records"
        ),
    }


@app.get("/api/predict")
def predict(server: str):
    df, zombie_df, _ = load_data()
    model = load_model()

    zombie_ids = set(
        zombie_df.loc[
            zombie_df["is_zombie"] == True,
            "server_id"
        ].tolist()
    )

    if server in zombie_ids:
        raise HTTPException(
            status_code=400,
            detail="Zombie server is not available for forecasting."
        )

    srv = df[
        df["server_id"] == server
    ].tail(24).copy()

    if srv.empty:
        raise HTTPException(
            status_code=404,
            detail="Server not found."
        )

    features = [
        "hour",
        "day_of_week",
        "is_weekend",
        "cpu_lag_1hr"
    ]

    srv["predicted_cpu"] = model.predict(
        srv[features]
    )

    srv["Time"] = pd.to_datetime(
        srv["timestamp"]
    ).dt.strftime("%H:%M")

    forecast = []

    for _, row in srv.iterrows():
        forecast.append(
            {
                "Time": row["Time"],
                "Current Telemetry (%)": round(
                    float(row["cpu_utilization"]),
                    2
                ),
                "XGBoost Prediction Envelop (%)": round(
                    float(row["predicted_cpu"]),
                    2
                ),
            }
        )

    return {
        "server": server,
        "forecast": forecast
    }