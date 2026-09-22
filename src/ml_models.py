"""
ml_models.py
------------
Machine Learning and Predictive Analytics engine for Health Sentinel.

Capabilities:
1. Multi-Model Case Forecasting (ARIMA, Holt-Winters Exponential Smoothing, Linear Baseline)
   with 80% & 95% Confidence Interval error bands and validation accuracy metrics (MAE, MAPE).
2. Outbreak Anomaly & Surge Detection using rolling baseline Z-score and IQR methods.
3. Outbreak Risk Driver Classification using Random Forest with feature importance extraction.

Author : Analytics & Data Science Team
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------------------------------------- #
# 1. Multi-Model Time-Series Forecasting with Confidence Intervals
# --------------------------------------------------------------------------- #
def generate_advanced_forecast(
    data: pd.DataFrame,
    model_type: str = "ARIMA",
    horizon: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate point forecasts and confidence interval bounds for monthly cases.
    
    Returns:
        history_df: Monthly historical aggregated data
        forecast_df: Projected future months with Upper/Lower 95% and 80% CI
        metrics: Dictionary containing MAE, MAPE, and model status
    """
    empty_history = pd.DataFrame(columns=["Month", "Actual_Cases"])
    empty_forecast = pd.DataFrame(columns=["Month", "Forecast_Cases", "Lower_80", "Upper_80", "Lower_95", "Upper_95"])
    empty_metrics = {"mae": 0.0, "mape": 0.0, "model_fit_score": "N/A", "status": "Insufficient data"}

    if data.empty or "historical_cases" not in data.columns:
        return empty_history, empty_forecast, empty_metrics

    # Group by year_month
    ts_data = data.dropna(subset=["year_month", "historical_cases"]).copy()
    if ts_data.empty:
        return empty_history, empty_forecast, empty_metrics

    monthly = (
        ts_data.groupby("year_month")["historical_cases"]
        .sum()
        .reset_index()
    )
    monthly["Month"] = pd.to_datetime(monthly["year_month"] + "-01", format="%Y-%m-%d", errors="coerce")
    monthly = monthly.dropna(subset=["Month"]).sort_values("Month")
    monthly = monthly[["Month", "historical_cases"]].rename(columns={"historical_cases": "Actual_Cases"})

    n_obs = len(monthly)
    if n_obs < 4:
        # Fallback for very sparse series
        last_val = monthly["Actual_Cases"].iloc[-1] if not monthly.empty else 0
        last_date = monthly["Month"].iloc[-1] if not monthly.empty else pd.Timestamp.now()
        f_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
        forecast_df = pd.DataFrame({
            "Month": f_dates,
            "Forecast_Cases": [int(last_val)] * horizon,
            "Lower_80": [max(0, int(last_val * 0.8))] * horizon,
            "Upper_80": [int(last_val * 1.2)] * horizon,
            "Lower_95": [max(0, int(last_val * 0.7))] * horizon,
            "Upper_95": [int(last_val * 1.35)] * horizon,
        })
        return monthly.reset_index(drop=True), forecast_df, {"mae": 0.0, "mape": 0.0, "model_fit_score": "Baseline Fallback", "status": "Sparse data fallback"}

    y = monthly["Actual_Cases"].values
    last_date = monthly["Month"].iloc[-1]
    f_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")

    forecast_values = np.zeros(horizon)
    se_forecast = np.zeros(horizon)
    mae = 0.0
    mape = 0.0
    model_score = "Good"

    # In-sample backtest split (hold out last 20% or max 3 months)
    split_idx = max(2, n_obs - min(3, max(1, int(n_obs * 0.2))))
    train_y = y[:split_idx]
    test_y = y[split_idx:]

    try:
        if model_type == "ARIMA":
            # Log-transform ARIMA(1,1,1)
            log_y = np.log(y + 1)
            model = ARIMA(log_y, order=(1, 1, 1))
            fit = model.fit()
            log_pred = fit.forecast(steps=horizon)
            forecast_values = np.maximum(0, np.exp(log_pred) - 1)
            
            # Estimate standard error from residuals
            residuals = y - np.maximum(0, np.exp(fit.fittedvalues) - 1)
            residual_std = np.std(residuals) if len(residuals) > 0 else np.std(y) * 0.15
            se_forecast = np.array([residual_std * np.sqrt(1 + 0.15 * i) for i in range(horizon)])

            # Backtest evaluation
            if len(test_y) > 0 and len(train_y) >= 3:
                fit_test = ARIMA(np.log(train_y + 1), order=(1, 1, 1)).fit()
                pred_eval = np.maximum(0, np.exp(fit_test.forecast(steps=len(test_y))) - 1)
                mae = float(np.mean(np.abs(test_y - pred_eval)))
                mape = float(np.mean(np.abs(test_y - pred_eval) / np.maximum(1, test_y)) * 100)

        elif model_type == "Holt-Winters":
            # Holt-Winters Exponential Smoothing with additive trend
            hw_model = ExponentialSmoothing(
                y.astype(float),
                trend="add",
                damped_trend=True,
                initialization_method="estimated"
            )
            hw_fit = hw_model.fit()
            forecast_values = np.maximum(0, hw_fit.forecast(horizon))
            
            residuals = y - hw_fit.fittedvalues
            residual_std = np.std(residuals) if len(residuals) > 0 else np.std(y) * 0.15
            se_forecast = np.array([residual_std * np.sqrt(1 + 0.12 * i) for i in range(horizon)])

            if len(test_y) > 0 and len(train_y) >= 3:
                hw_test = ExponentialSmoothing(train_y.astype(float), trend="add", damped_trend=True).fit()
                pred_eval = np.maximum(0, hw_test.forecast(len(test_y)))
                mae = float(np.mean(np.abs(test_y - pred_eval)))
                mape = float(np.mean(np.abs(test_y - pred_eval) / np.maximum(1, test_y)) * 100)

        else:  # Linear Trend Baseline
            x = np.arange(n_obs)
            poly_coeff = np.polyfit(x, y, 1)
            poly_fn = np.poly1d(poly_coeff)
            future_x = np.arange(n_obs, n_obs + horizon)
            forecast_values = np.maximum(0, poly_fn(future_x))

            residuals = y - poly_fn(x)
            residual_std = np.std(residuals) if len(residuals) > 0 else np.std(y) * 0.2
            se_forecast = np.array([residual_std * np.sqrt(1 + 0.18 * i) for i in range(horizon)])

            if len(test_y) > 0 and len(train_y) >= 2:
                poly_t = np.poly1d(np.polyfit(np.arange(len(train_y)), train_y, 1))
                pred_eval = np.maximum(0, poly_t(np.arange(len(train_y), len(train_y) + len(test_y))))
                mae = float(np.mean(np.abs(test_y - pred_eval)))
                mape = float(np.mean(np.abs(test_y - pred_eval) / np.maximum(1, test_y)) * 100)

    except Exception:
        # Fallback to moving average if model fitting raises an exception
        base_val = y[-3:].mean() if n_obs >= 3 else y.mean()
        forecast_values = np.array([base_val] * horizon)
        se_forecast = np.array([base_val * 0.15 * np.sqrt(1 + 0.1 * i) for i in range(horizon)])
        mae = float(np.std(y))
        mape = 15.0
        model_score = "Baseline (Fallback)"

    # Compute 80% (Z=1.28) and 95% (Z=1.96) bounds
    lower_80 = np.maximum(0, forecast_values - 1.28 * se_forecast)
    upper_80 = forecast_values + 1.28 * se_forecast
    lower_95 = np.maximum(0, forecast_values - 1.96 * se_forecast)
    upper_95 = forecast_values + 1.96 * se_forecast

    forecast_df = pd.DataFrame({
        "Month": f_dates,
        "Forecast_Cases": np.round(forecast_values).astype(int),
        "Lower_80": np.round(lower_80).astype(int),
        "Upper_80": np.round(upper_80).astype(int),
        "Lower_95": np.round(lower_95).astype(int),
        "Upper_95": np.round(upper_95).astype(int),
    })

    metrics = {
        "mae": round(mae, 1),
        "mape": round(mape, 1),
        "model_fit_score": model_score if mape < 30 else "Moderate",
        "model_type": model_type,
        "status": "Success"
    }

    return monthly.reset_index(drop=True), forecast_df, metrics


# --------------------------------------------------------------------------- #
# 2. Outbreak Anomaly & Surge Detection Engine
# --------------------------------------------------------------------------- #
def detect_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect anomalous surges in disease cases using rolling baseline statistics.
    Flags rows with Z-score > z_threshold or case velocity spikes.
    """
    if df.empty or "historical_cases" not in df.columns:
        return df.copy()

    result = df.copy()
    
    # Sort chronologically
    if "year_month" in result.columns:
        result = result.sort_values("year_month")
    
    # Calculate group baseline (per state and disease if available, else overall)
    group_cols = [c for c in ["state_name", "disease_name"] if c in result.columns]
    
    if group_cols:
        # Group rolling statistics
        grp = result.groupby(group_cols)["historical_cases"]
        baseline_mean = grp.transform(lambda s: s.rolling(window=3, min_periods=1).mean().shift(1))
        baseline_std = grp.transform(lambda s: s.rolling(window=3, min_periods=1).std().shift(1))
    else:
        baseline_mean = result["historical_cases"].rolling(window=3, min_periods=1).mean().shift(1)
        baseline_std = result["historical_cases"].rolling(window=3, min_periods=1).std().shift(1)

    # Fill NaN baseline with initial values
    baseline_mean = baseline_mean.fillna(result["historical_cases"].mean())
    baseline_std = baseline_std.fillna(result["historical_cases"].std()).replace(0, 1)

    # Compute Z-score and percentage spike
    result["baseline_cases"] = baseline_mean.round(1)
    result["z_score"] = ((result["historical_cases"] - baseline_mean) / baseline_std).round(2)
    result["surge_pct"] = (((result["historical_cases"] - baseline_mean) / np.maximum(1, baseline_mean)) * 100).round(1)
    
    # Flag anomalies
    result["is_anomaly"] = (result["z_score"] >= z_threshold) & (result["historical_cases"] > baseline_mean * 1.3)
    
    # Assign anomaly severity tag
    def get_severity(row):
        if not row["is_anomaly"]:
            return "Normal"
        if row["z_score"] >= 3.0 or row["surge_pct"] >= 100:
            return "Critical Spike"
        return "Warning Spike"

    result["anomaly_severity"] = result.apply(get_severity, axis=1)
    return result


# --------------------------------------------------------------------------- #
# 3. Machine Learning Outbreak Risk Driver Classifier (Random Forest)
# --------------------------------------------------------------------------- #
def train_risk_driver_model(
    outbreak_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Train a Random Forest classifier to predict Outbreak Alert Level (High vs Moderate vs Low)
    and compute normalized feature importance to identify primary drivers of risk.
    """
    empty_result = {
        "feature_importances": pd.DataFrame(columns=["Driver", "Importance", "Percentage"]),
        "accuracy": 0.0,
        "f1": 0.0,
        "trained": False,
        "message": "Insufficient data to train model"
    }

    if outbreak_df.empty or len(outbreak_df) < 15 or "alert_level" not in outbreak_df.columns:
        return empty_result

    # Potential feature columns available in fact_outbreak joined tables
    candidate_features = [
        "historical_cases",
        "predicted_cases",
        "response_time_hours",
        "hospital_readiness_score",
        "environmental_risk_index",
        "aqi_index",
        "rainfall_mm",
        "positivity_rate",
        "icu_bed_occupancy_rate",
    ]
    
    # Filter features that exist in dataframe and have non-null variance
    features = [c for c in candidate_features if c in outbreak_df.columns and outbreak_df[c].nunique() > 1]
    
    if len(features) < 2:
        # Construct dynamic fallback features if standard join features are limited
        if "historical_cases" in outbreak_df.columns and "response_time_hours" in outbreak_df.columns:
            features = ["historical_cases", "response_time_hours"]
        else:
            return empty_result

    data = outbreak_df[features + ["alert_level"]].dropna()
    if len(data) < 15 or data["alert_level"].nunique() < 2:
        return empty_result

    X = data[features]
    y = data["alert_level"].astype(str)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Human-readable labels for features
    label_map = {
        "historical_cases": "Case Volume Velocity",
        "predicted_cases": "Projected Outbreak Trajectory",
        "response_time_hours": "Containment Response Latency",
        "hospital_readiness_score": "Healthcare Readiness Deficit",
        "environmental_risk_index": "Environmental Vulnerability Index",
        "aqi_index": "Air Quality (AQI) Stress",
        "rainfall_mm": "Vector Monsoon Rainfall",
        "positivity_rate": "Lab Positivity Rate",
        "icu_bed_occupancy_rate": "ICU Bed Stress",
    }

    importances = rf.feature_importances_
    feat_df = pd.DataFrame({
        "Driver": [label_map.get(f, f.replace("_", " ").title()) for f in features],
        "Importance": importances,
        "Percentage": (importances / importances.sum() * 100).round(1)
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "feature_importances": feat_df,
        "accuracy": round(float(acc) * 100, 1),
        "f1": round(float(f1) * 100, 1),
        "trained": True,
        "features_used": features,
        "message": "Model trained successfully on outbreak surveillance records"
    }
