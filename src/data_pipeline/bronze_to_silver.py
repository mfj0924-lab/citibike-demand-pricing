"""
Bronze → Silver Transformer (Enhanced): aggregate trips with rich features.

v2 adds:
- Spatial capacity matching (lat/lng proximity to GBFS)
- New trip-level features: electric_ratio, member_ratio, avg_trip_duration_min
- Station active_hours count
- Prepares date column for weather join (if weather CSV available)
"""

import os, sys
from dataclasses import dataclass
from typing import List

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark_session
from src.data_pipeline.abstract_transformer import AbstractTransformer
from src.data_pipeline.spatial_matcher import match_stations_from_dataframe
from src.utils.io_utils import read_parquet, write_parquet
from src.utils.logger import logging  # noqa: F401

_US_HOLIDAYS: List[str] = [
    "2025-01-01",
    "2025-01-20",
    "2025-02-17",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-10-13",
    "2025-11-11",
    "2025-11-27",
    "2025-12-25",
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-05-25",
    "2026-06-19",
    "2026-07-04",
    "2026-09-07",
    "2026-10-12",
    "2026-11-11",
    "2026-11-26",
    "2026-12-25",
    "2027-01-01",
    "2027-01-18",
    "2027-02-15",
    "2027-05-31",
    "2027-06-19",
    "2027-07-04",
    "2027-09-06",
    "2027-10-11",
    "2027-11-11",
    "2027-11-25",
    "2027-12-25",
]


@dataclass
class BronzeToSilverConfig:
    trips_parquet: str = ""
    stations_parquet: str = ""
    station_info_json: str = ""
    weather_csv: str = ""
    hourly_demand_parquet: str = ""


class BronzeToSilverTransformer(AbstractTransformer):
    """Aggregate trips into hourly demand per station (enhanced v2)."""

    def __init__(self, spark: SparkSession, config: BronzeToSilverConfig):
        super().__init__(spark, config)

    # ── Spatial capacity matching ─────────────────────────

    def _build_capacity_broadcast(self):
        """Match bronze stations to GBFS stations by lat/lng, return broadcast dict."""
        stations_pd = (
            read_parquet(self.spark, self.config.stations_parquet)
            .select("station_id", "latitude", "longitude")
            .toPandas()
        )

        cap_map = match_stations_from_dataframe(
            stations_pd, self.config.station_info_json, max_distance_m=100.0
        )
        return self.spark.sparkContext.broadcast(cap_map)

    def _add_capacity_udf(self, df: DataFrame, cap_bc) -> DataFrame:
        """Add capacity column using broadcast dict."""
        from pyspark.sql.types import IntegerType

        @F.udf(IntegerType())
        def get_cap(sid):
            return cap_bc.value.get(str(sid), 0)

        return df.withColumn("capacity", get_cap(col("station_id")))

    # ── Event splitting with extra features ───────────────

    def _split_trips_into_events(self, df: DataFrame) -> DataFrame:
        """Split each trip into start (借出) and end (还入) events.

        Adds columns for richer aggregation:
        - is_electric: 1 if electric bike, 0 otherwise
        - is_member: 1 if member, 0 otherwise
        - trip_duration_sec: trip length in seconds (start events only)
        """
        trip_duration = (
            F.unix_timestamp("ended_at") - F.unix_timestamp("started_at")
        ).cast(DoubleType())

        is_electric = F.when(col("rideable_type") == "electric_bike", 1).otherwise(0)
        is_member = F.when(col("member_casual") == "member", 1).otherwise(0)

        start_events = df.select(
            col("started_at").alias("event_time"),
            col("start_station_id").alias("station_id"),
            F.lit(1).cast(IntegerType()).alias("bike_departure"),
            F.lit(0).cast(IntegerType()).alias("bike_arrival"),
            is_electric.cast(IntegerType()).alias("is_electric"),
            is_member.cast(IntegerType()).alias("is_member"),
            trip_duration.alias("trip_duration_sec"),
        )
        end_events = df.select(
            col("ended_at").alias("event_time"),
            col("end_station_id").alias("station_id"),
            F.lit(0).cast(IntegerType()).alias("bike_departure"),
            F.lit(1).cast(IntegerType()).alias("bike_arrival"),
            F.lit(0).cast(IntegerType()).alias("is_electric"),
            F.lit(0).cast(IntegerType()).alias("is_member"),
            F.lit(0.0).cast(DoubleType()).alias("trip_duration_sec"),
        )
        return start_events.union(end_events)

    def _aggregate_hourly(self, events: DataFrame) -> DataFrame:
        """Group by (station_id, event_hour) with all features."""
        hourly = (
            events.withColumn("event_hour", F.date_trunc("hour", col("event_time")))
            .groupBy("station_id", "event_hour")
            .agg(
                F.sum("bike_departure").alias("bike_demand"),
                F.sum("bike_arrival").alias("dock_demand"),
                F.sum("is_electric").alias("electric_count"),
                F.sum("is_member").alias("member_count"),
                F.sum("trip_duration_sec").alias("total_duration_sec"),
                F.count("bike_departure").alias("total_events"),
            )
        )
        # Compute ratios
        hourly = (
            hourly.withColumn(
                "electric_ratio",
                F.when(
                    col("bike_demand") > 0, col("electric_count") / col("bike_demand")
                ).otherwise(0),
            )
            .withColumn(
                "member_ratio",
                F.when(
                    col("bike_demand") > 0, col("member_count") / col("bike_demand")
                ).otherwise(0),
            )
            .withColumn(
                "avg_trip_duration_min",
                F.when(
                    col("bike_demand") > 0,
                    col("total_duration_sec") / col("bike_demand") / 60.0,
                ).otherwise(0),
            )
            .drop(
                "electric_count", "member_count", "total_duration_sec", "total_events"
            )
        )
        return hourly

    # ── Time features (unchanged) ─────────────────────────

    def _add_time_features(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumn("year", F.year("event_hour"))
            .withColumn("month", F.month("event_hour"))
            .withColumn("day", F.dayofmonth("event_hour"))
            .withColumn("weekday", F.dayofweek("event_hour"))
            .withColumn("weekofyear", F.weekofyear("event_hour"))
            .withColumn("dayofyear", F.dayofyear("event_hour"))
            .withColumn("hour", F.hour("event_hour"))
            .withColumn(
                "is_weekend", F.when(col("weekday").isin([1, 7]), 1).otherwise(0)
            )
            .withColumn(
                "is_rush_hour",
                F.when(
                    (col("hour").between(7, 9)) | (col("hour").between(17, 19)), 1
                ).otherwise(0),
            )
        )

    @staticmethod
    def _cyclic_encode(df: DataFrame, col_name: str, period: int) -> DataFrame:
        rad = 2.0 * 3.141592653589793 * F.col(col_name) / period
        return df.withColumn(f"{col_name}_sin", F.sin(rad)).withColumn(
            f"{col_name}_cos", F.cos(rad)
        )

    def _add_holiday_flag(self, df: DataFrame) -> DataFrame:
        holiday_df = self.spark.createDataFrame(
            [(h, 1) for h in _US_HOLIDAYS], schema="holiday_date string, is_holiday int"
        )
        return (
            df.withColumn("event_date_str", F.date_format("event_hour", "yyyy-MM-dd"))
            .join(
                holiday_df, F.col("event_date_str") == F.col("holiday_date"), how="left"
            )
            .fillna({"is_holiday": 0})
            .drop("event_date_str", "holiday_date")
        )

    # ── Weather data join ─────────────────────────────────

    def _add_weather(self, df: DataFrame) -> DataFrame:
        """Join with weather CSV (Open-Meteo format)."""
        wpath = self.config.weather_csv
        if not wpath or not os.path.exists(wpath):
            print("  (no weather CSV, skipping)")
            return df

        weather_pd = pd.read_csv(wpath, skiprows=3)  # skip Open-Meteo header
        weather_pd = weather_pd.rename(
            columns={
                "time": "date",
                "temperature_2m_mean (°C)": "temp_c",
                "precipitation_sum (mm)": "precip_mm",
                "wind_speed_10m_max (km/h)": "wind_kmh",
            }
        )
        # Only keep date + weather columns
        weather_pd = weather_pd[["date", "temp_c", "precip_mm", "wind_kmh"]]
        print(f"  Weather rows: {len(weather_pd):,}")

        weather = self.spark.createDataFrame(weather_pd)
        df = df.withColumn("event_date", F.to_date("event_hour"))
        df = df.join(weather, df.event_date == weather.date, how="left")
        missing = df.filter(F.col("temp_c").isNull()).count()
        print(
            f"  Weather matched: {df.count() - missing:,} / {df.count():,} rows, "
            f"{missing:,} missing"
        )
        return df.drop("event_date", "date")

    # ── Main ──────────────────────────────────────────────

    def transform(self):
        logging.info("=== Bronze → Silver Transformer (v2 enhanced) ===")
        cfg = self.config

        # 1. Build capacity lookup (spatial match)
        print("\n  [1] Spatial capacity matching...")
        cap_bc = self._build_capacity_broadcast()

        # 2. Read & split
        print("\n  [2] Reading trips and splitting into events...")
        df = read_parquet(self.spark, cfg.trips_parquet)
        print(f"  Trip rows: {df.count():,}")
        df = df.coalesce(4)
        events = self._split_trips_into_events(df)
        del df

        # 3. Aggregate
        print("\n  [3] Aggregating hourly...")
        hourly = self._aggregate_hourly(events)
        hourly = hourly.coalesce(4)
        print(f"  Hourly rows: {hourly.count():,}")

        # 4. Time features
        hourly = self._add_time_features(hourly)

        # 5. Cyclic encoding
        hourly = self._cyclic_encode(hourly, "hour", 24)
        hourly = self._cyclic_encode(hourly, "weekday", 7)

        # 6. Holiday
        hourly = self._add_holiday_flag(hourly)

        # 7. Capacity (spatial match)
        print("\n  [4] Adding station capacity (spatial matching)...")
        hourly = self._add_capacity_udf(hourly, cap_bc)

        # 8. Weather
        print("\n  [5] Joining weather data...")
        hourly = self._add_weather(hourly)

        # 9. Write
        print(f"\n  [6] Writing to {cfg.hourly_demand_parquet}...")
        write_parquet(hourly, cfg.hourly_demand_parquet)
        print(f"  Done.")
        logging.info("Bronze → Silver (v2) done.")


if __name__ == "__main__":
    spark = get_spark_session("bronze_to_silver", "4g")
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    bronze_root = os.path.join(base, "data", "processed", "bronze")
    silver_root = os.path.join(base, "data", "processed", "silver")

    weather_csv = os.path.join(base, "data", "unstructured", "nyc_weather_2026.csv")
    if not os.path.exists(weather_csv):
        weather_csv = ""  # skip if not downloaded

    config = BronzeToSilverConfig(
        trips_parquet=os.path.join(bronze_root, "trips.parquet"),
        stations_parquet=os.path.join(bronze_root, "stations.parquet"),
        station_info_json=os.path.join(base, "data", "raw", "station_information.json"),
        weather_csv=weather_csv,
        hourly_demand_parquet=os.path.join(silver_root, "hourly_demand.parquet"),
    )

    transformer = BronzeToSilverTransformer(spark, config)
    transformer.transform()
    spark.stop()
