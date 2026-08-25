"""
Raw → Bronze Transformer: clean timestamps, split station info, handle nulls.

Simplified from Shakleen's version:
- Only one timestamp format (yyyy-MM-dd HH:mm:ss.SSS) for 2023+ data
- Splits station metadata from trip data
- Drops rows with null station IDs
- Writes bronze/trips.parquet and bronze/stations.parquet
"""

import os
import sys
from dataclasses import dataclass
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, to_timestamp, min as spark_min

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark_session
from src.data_pipeline.abstract_transformer import AbstractTransformer
from src.utils.io_utils import read_parquet, write_parquet
from src.utils.logger import logging  # noqa: F401


@dataclass
class RawToBronzeConfig:
    bronze_root: str = ""
    raw_trips_parquet: str = ""
    trips_parquet: str = ""
    stations_parquet: str = ""


class RawToBronzeTransformer(AbstractTransformer):
    """Clean raw trips → produce clean trips + station dimension table."""

    def __init__(self, spark: SparkSession, config: RawToBronzeConfig):
        super().__init__(spark, config)

    def _fix_timestamps(self, df: DataFrame) -> DataFrame:
        """Convert started_at / ended_at strings to timestamps."""
        return (
            df
            .withColumn("started_at", to_timestamp("started_at", "yyyy-MM-dd HH:mm:ss.SSS"))
            .withColumn("ended_at", to_timestamp("ended_at", "yyyy-MM-dd HH:mm:ss.SSS"))
        )

    def _drop_duplicate_ride_ids(self, df: DataFrame) -> DataFrame:
        """Keep the first occurrence of each ride_id."""
        return df.dropDuplicates(["ride_id"])

    def _extract_station_table(self, df: DataFrame) -> DataFrame:
        """Extract a deduplicated station dimension table from trip data.

        Collects station info from both start and end columns, groups by
        station_id to get one row per unique station.
        """
        start_stations = (
            df.select(
                col("start_station_id").alias("station_id"),
                col("start_station_name").alias("name"),
                col("start_lat").alias("latitude"),
                col("start_lng").alias("longitude"),
            )
        )
        end_stations = (
            df.select(
                col("end_station_id").alias("station_id"),
                col("end_station_name").alias("name"),
                col("end_lat").alias("latitude"),
                col("end_lng").alias("longitude"),
            )
        )
        all_stations = start_stations.union(end_stations)

        # Deduplicate: keep the name/lat/lng from the FIRST occurrence of each ID
        station_table = all_stations.groupBy("station_id").agg(
            spark_min("name").alias("name"),
            spark_min("latitude").alias("latitude"),
            spark_min("longitude").alias("longitude"),
        )
        return station_table

    def _strip_station_cols_from_trips(self, df: DataFrame) -> DataFrame:
        """Remove station name/lat/lng columns, keep only station IDs."""
        return df.drop(
            "start_station_name", "end_station_name",
            "start_lat", "start_lng", "end_lat", "end_lng",
        )

    def transform(self):
        logging.info("=== Raw → Bronze Transformer ===")
        cfg = self.config

        # Read
        df = read_parquet(self.spark, cfg.raw_trips_parquet)
        before = df.count()
        print(f"  Input rows: {before:,}")

        # Fix timestamps
        df = self._fix_timestamps(df)

        # Drop duplicate ride_ids
        df = self._drop_duplicate_ride_ids(df)

        # Drop rows missing either station ID
        df = df.dropna(subset=["start_station_id", "end_station_id"], how="any")

        # Extract station table BEFORE stripping columns
        station_df = self._extract_station_table(df)

        # Strip station name/lat/lng from trip table
        df = self._strip_station_cols_from_trips(df)

        after = df.count()
        station_count = station_df.count()
        print(f"  Output trip rows: {after:,}")
        print(f"  Unique stations: {station_count:,}")

        write_parquet(df, cfg.trips_parquet)
        write_parquet(station_df, cfg.stations_parquet)
        logging.info("Raw → Bronze done.")


if __name__ == "__main__":
    spark = get_spark_session("raw_to_bronze", "4g")
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    bronze_root = os.path.join(base, "data", "processed", "bronze")

    config = RawToBronzeConfig(
        bronze_root=bronze_root,
        raw_trips_parquet=os.path.join(bronze_root, "raw_trips.parquet"),
        trips_parquet=os.path.join(bronze_root, "trips.parquet"),
        stations_parquet=os.path.join(bronze_root, "stations.parquet"),
    )

    transformer = RawToBronzeTransformer(spark, config)
    transformer.transform()
    spark.stop()
