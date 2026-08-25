"""
Data Ingestor: Read raw CitiBike CSV files → write to Bronze Parquet.

Simplified from Shakleen's version — only handles 2023+ CitiBike schema.
Reads all CSV files from data/raw/, standardizes column names, writes to
data/processed/bronze/raw_trips.parquet.
"""

import os
import sys
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark_session
from src.utils.io_utils import write_parquet
from src.utils.logger import logging  # noqa: F401


class DataIngestor:
    """Read raw CSVs → write Bronze Parquet."""

    def __init__(self, spark: SparkSession, raw_dir: str, bronze_dir: str):
        self.spark = spark
        self.raw_dir = raw_dir
        self.bronze_dir = bronze_dir

    def _list_csv_paths(self) -> list:
        """List absolute paths of all CSV files in raw_dir (Python-side glob)."""
        import glob
        pattern = os.path.join(self.raw_dir, "*.csv")
        paths = glob.glob(pattern)
        if not paths:
            raise FileNotFoundError(f"No CSV files found matching {pattern}")
        # Convert to file:// URIs (works on Windows without winutils)
        return [os.path.abspath(p) for p in sorted(paths)]

    def read_all_csvs(self) -> DataFrame:
        """Read all CSV files from data/raw/ into a single DataFrame."""
        csv_paths = self._list_csv_paths()
        logging.info("Reading %d CSV files from %s", len(csv_paths), self.raw_dir)
        print(f"  Found {len(csv_paths)} CSV files")

        df = (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(csv_paths)
        )
        return df

    def add_file_name_column(self, df: DataFrame) -> DataFrame:
        """Add a file_name column extracted from the input file path."""
        df = df.withColumn("file_path", input_file_name())
        return df.withColumn(
            "file_name",
            regexp_extract("file_path", r"[^\\/]+$", 0)
        ).drop("file_path")

    def run(self) -> None:
        """Execute the full ingestion pipeline."""
        logging.info("=== Data Ingestor: Raw CSV → Bronze Parquet ===")
        df = self.read_all_csvs()
        total = df.count()
        logging.info("Total rows read: %d", total)
        print(f"  Rows read: {total:,}")

        df = self.add_file_name_column(df)
        os.makedirs(self.bronze_dir, exist_ok=True)
        output_path = os.path.join(self.bronze_dir, "raw_trips.parquet")
        write_parquet(df, output_path)
        logging.info("Written to %s", output_path)
        print(f"  Written to: {output_path}")


if __name__ == "__main__":
    spark = get_spark_session("data_ingestor", "4g")
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    raw_dir = os.path.join(base, "data", "raw")
    bronze_dir = os.path.join(base, "data", "processed", "bronze")

    ingestor = DataIngestor(spark, raw_dir, bronze_dir)
    ingestor.run()
    spark.stop()
