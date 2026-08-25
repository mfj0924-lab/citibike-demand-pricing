"""
Supply-Demand Gap Analyzer.

For each station-hour:
  bike_gap = available_bikes - predicted_bike_demand
  dock_gap = available_docks - predicted_dock_demand

Negative gap → shortage (surge needed).
Positive gap → surplus (discount possible).

Outputs a CSV with pricing recommendations for all station-hours.
"""

import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark_session
from src.utils.io_utils import read_parquet


def estimate_available_bikes(capacity: int) -> int:
    """Estimate current available bikes as ~50% of capacity.

    In production, this would come from GBFS real-time station_status API.
    For offline analysis, assume half the docks are occupied on average.
    """
    if capacity <= 0:
        return 15  # default for unknown stations
    return max(1, capacity // 2)


def estimate_available_docks(capacity: int) -> int:
    """Estimate empty docking slots."""
    if capacity <= 0:
        return 15
    return max(1, capacity - estimate_available_bikes(capacity))


def calculate_pricing_multiplier(
    bike_gap: float,
    dock_gap: float,
    available_bikes: int,
    available_docks: int,
    base_multiplier: float = 1.0,
) -> float:
    """Calculate a pricing multiplier based on supply-demand gaps.

    Four scenarios:
      1. Double scarce (bike shortage + dock shortage) → surge 1.5-2.0x
      2. Bike shortage only → surge 1.2-1.5x
      3. Dock shortage only → surge 1.1-1.3x
      4. Both abundant → discount 0.7-0.9x
      5. Balanced → normal 1.0x
    """
    # Normalize gaps to [-1, 1] range
    bike_shortage = -bike_gap / max(available_bikes, 1) if bike_gap < 0 else 0.0
    dock_shortage = -dock_gap / max(available_docks, 1) if dock_gap < 0 else 0.0
    bike_surplus = bike_gap / max(available_bikes, 1) if bike_gap > 0 else 0.0
    dock_surplus = dock_gap / max(available_docks, 1) if dock_gap > 0 else 0.0

    if bike_shortage > 0.5 and dock_shortage > 0.5:
        # Double scarce: heavy surge
        multiplier = 1.3 + (bike_shortage + dock_shortage) * 0.5
        multiplier = min(multiplier, 2.0)
    elif bike_shortage > 0.3:
        # Bike shortage: moderate surge (encourage users to go elsewhere)
        multiplier = 1.0 + bike_shortage * 0.6
        multiplier = min(multiplier, 1.5)
    elif dock_shortage > 0.3:
        # Dock congestion: mild surge (discourage returns here)
        multiplier = 1.0 + dock_shortage * 0.4
        multiplier = min(multiplier, 1.3)
    elif bike_surplus > 0.5 and dock_surplus > 0.5:
        # Both abundant: discount to attract users
        multiplier = 1.0 - min(bike_surplus, dock_surplus) * 0.25
        multiplier = max(multiplier, 0.7)
    elif bike_surplus > 0.5:
        # Bike surplus: mild discount
        multiplier = 1.0 - bike_surplus * 0.15
        multiplier = max(multiplier, 0.8)
    else:
        # Balanced
        multiplier = 1.0

    return round(multiplier, 2)


def generate_pricing_analysis(
    silver_parquet: str,
    output_csv: str,
    station_info_json: str = None,
) -> pd.DataFrame:
    """Generate pricing recommendations for all station-hours in silver layer.

    Uses GBFS capacity for station size estimation.
    Returns a pandas DataFrame suitable for visualization.
    """
    spark = get_spark_session("pricing_analysis", "3g")

    df = read_parquet(spark, silver_parquet)

    # Select needed columns
    cols_needed = ["station_id", "event_hour", "bike_demand", "dock_demand", "capacity"]
    pdf = df.select(*cols_needed).toPandas()
    spark.stop()

    # Estimate current state
    pdf["available_bikes"] = pdf["capacity"].apply(estimate_available_bikes)
    pdf["available_docks"] = pdf["capacity"].apply(estimate_available_docks)

    # Calculate gaps
    pdf["bike_gap"] = pdf["available_bikes"] - pdf["bike_demand"]
    pdf["dock_gap"] = pdf["available_docks"] - pdf["dock_demand"]

    # Apply pricing
    multipliers = []
    for _, row in pdf.iterrows():
        m = calculate_pricing_multiplier(
            row["bike_gap"],
            row["dock_gap"],
            row["available_bikes"],
            row["available_docks"],
        )
        multipliers.append(m)

    pdf["pricing_multiplier"] = multipliers
    pdf["suggested_price"] = (pdf["pricing_multiplier"] * 4.49).round(
        2
    )  # CitiBike e-bike base price

    # Classify pricing zone
    conditions = [
        pdf["pricing_multiplier"] > 1.2,
        pdf["pricing_multiplier"] > 1.0,
        pdf["pricing_multiplier"] < 0.9,
    ]
    choices = ["Surge", "Mild Surge", "Discount"]
    pdf["pricing_zone"] = np.select(conditions, choices, default="Normal")

    # Save
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    pdf.to_csv(output_csv, index=False)
    print(f"  Pricing analysis saved: {len(pdf):,} rows → {output_csv}")

    # Summary
    for zone in ["Surge", "Mild Surge", "Normal", "Discount"]:
        count = (pdf["pricing_zone"] == zone).sum()
        pct = count / len(pdf) * 100
        avg_price = pdf.loc[pdf["pricing_zone"] == zone, "suggested_price"].mean()
        print(f"  {zone:12s}: {count:8,} ({pct:5.1f}%)  avg ${avg_price:.2f}")

    return pdf


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "..")

    pdf = generate_pricing_analysis(
        silver_parquet=os.path.join(
            base, "data", "processed", "silver", "hourly_demand.parquet"
        ),
        output_csv=os.path.join(base, "data", "processed", "pricing_analysis.csv"),
    )
