"""FastAPI prediction router — demand forecasting + dynamic pricing."""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    StationForecastRequest,
    StationForecastResponse,
    ForecastPoint,
    HotspotItem,
    HotspotResponse,
)
from src.pricing.pricing_engine import DynamicPricingEngine

router = APIRouter(prefix="/api/v1", tags=["预测服务"])
engine = DynamicPricingEngine()

# Load pricing analysis data (lazy)
_pricing_df: Optional[pd.DataFrame] = None


def _load_pricing_data() -> pd.DataFrame:
    global _pricing_df
    if _pricing_df is None:
        base = os.path.join(os.path.dirname(__file__), "..", "..", "..")
        csv_path = os.path.join(base, "data", "processed", "pricing_analysis.csv")
        if os.path.exists(csv_path):
            _pricing_df = pd.read_csv(csv_path)
        else:
            _pricing_df = pd.DataFrame()
    return _pricing_df


def _time_features(ts: datetime) -> dict:
    """Extract time features matching the silver layer schema."""
    hour = ts.hour
    weekday = (ts.weekday() + 1) % 7 + 1  # 1=Sun..7=Sat Spark dayofweek
    return {
        "month": ts.month,
        "day": ts.day,
        "weekday": weekday,
        "weekofyear": int(ts.strftime("%U")),
        "dayofyear": int(ts.strftime("%j")),
        "hour": hour,
    }


def _predict_demand(
    station_id: str, ts: datetime, is_holiday: bool
) -> tuple[float, float]:
    """Lightweight demand prediction using historical averages from pricing data.

    In production, this would call the PySpark ML model.
    Here we use pre-computed pricing_analysis.csv for accurate results.
    """
    pdf = _load_pricing_data()
    if pdf.empty:
        # Fallback: simple heuristic
        hour = ts.hour
        weekday = ts.weekday()
        base = 10 + 15 * (1 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0)
        is_weekend = weekday >= 5
        base = base * 0.6 if is_weekend else base
        return (base, base * 0.9)

    # Match station + hour from historical data
    hour = ts.hour
    mask = (pdf["station_id"] == station_id) & (
        pdf["event_hour"].str[:13] == ts.strftime("%Y-%m-%d %H")
    )
    matched = pdf.loc[mask, ["bike_demand", "dock_demand"]]

    if len(matched) > 0:
        return (
            float(matched["bike_demand"].mean()),
            float(matched["dock_demand"].mean()),
        )

    # Broader: same hour, any station
    hour_mask = pdf["event_hour"].str.contains(f" {hour:02d}:")
    return (
        float(pdf.loc[hour_mask, "bike_demand"].mean()),
        float(pdf.loc[hour_mask, "dock_demand"].mean()),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """预测单个站点在指定时间的需求 + 定价建议"""
    try:
        ts = datetime.strptime(request.timestamp, "%Y-%m-%d %H:%M")
    except ValueError:
        raise HTTPException(400, "时间格式错误。请使用 'yyyy-MM-dd HH:mm' 格式")

    bike_dem, dock_dem = _predict_demand(request.station_id, ts, request.is_holiday)

    result = engine.predict_price(
        station_id=request.station_id,
        timestamp=request.timestamp,
        predicted_bike_demand=bike_dem,
        predicted_dock_demand=dock_dem,
        available_bikes_now=request.available_bikes,
        available_docks_now=request.available_docks,
    )

    return PredictionResponse(
        station_id=result.station_id,
        timestamp=result.timestamp,
        predicted_bike_demand=result.predicted_bike_demand,
        predicted_dock_demand=result.predicted_dock_demand,
        pricing_multiplier=result.pricing_multiplier,
        suggested_price_usd=result.suggested_price_usd,
        pricing_zone=result.pricing_zone.value,
        reason=result.reason,
    )


@router.post("/station/{station_id}/forecast", response_model=StationForecastResponse)
async def forecast_station(request: StationForecastRequest):
    """为指定站点生成未来 24 小时的逐时需求预测 + 定价预报"""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    forecast = []

    for h in range(24):
        ts = now + timedelta(hours=h)
        ts_str = ts.strftime("%Y-%m-%d %H:%M")

        bike_dem, dock_dem = _predict_demand(request.station_id, ts, False)

        result = engine.predict_price(
            station_id=request.station_id,
            timestamp=ts_str,
            predicted_bike_demand=bike_dem,
            predicted_dock_demand=dock_dem,
            available_bikes_now=request.available_bikes,
            available_docks_now=request.available_docks,
        )

        forecast.append(
            ForecastPoint(
                timestamp=ts_str,
                predicted_bike_demand=result.predicted_bike_demand,
                predicted_dock_demand=result.predicted_dock_demand,
                pricing_multiplier=result.pricing_multiplier,
                suggested_price_usd=result.suggested_price_usd,
                pricing_zone=result.pricing_zone.value,
            )
        )

        # Simulate state evolution
        request.available_bikes = max(
            0, request.available_bikes - int(bike_dem) + int(dock_dem)
        )
        request.available_bikes = min(request.capacity, request.available_bikes)
        request.available_docks = request.capacity - request.available_bikes

    return StationForecastResponse(
        station_id=request.station_id,
        capacity=request.capacity,
        forecast=forecast,
    )


@router.get("/hotspots", response_model=HotspotResponse)
async def get_hotspots():
    """返回当前需要涨价和降价的 Top-5 站点列表"""
    pdf = _load_pricing_data()

    if pdf.empty:
        return HotspotResponse(surge_stations=[], discount_stations=[])

    # Aggregate by station
    station_agg = (
        pdf.groupby("station_id")
        .agg(
            avg_price=("suggested_price", "mean"),
            avg_multiplier=("pricing_multiplier", "mean"),
            surge_pct=("pricing_zone", lambda x: (x == "Surge").sum() / len(x) * 100),
        )
        .reset_index()
    )

    surge = station_agg.nlargest(5, "avg_multiplier")
    discount = station_agg.nsmallest(5, "avg_multiplier")

    return HotspotResponse(
        surge_stations=[
            HotspotItem(
                station_id=str(r["station_id"]),
                avg_price=round(r["avg_price"], 2),
                avg_multiplier=round(r["avg_multiplier"], 2),
                surge_pct=round(r["surge_pct"], 1),
            )
            for _, r in surge.iterrows()
        ],
        discount_stations=[
            HotspotItem(
                station_id=str(r["station_id"]),
                avg_price=round(r["avg_price"], 2),
                avg_multiplier=round(r["avg_multiplier"], 2),
                surge_pct=round(r["surge_pct"], 1),
            )
            for _, r in discount.iterrows()
        ],
    )
