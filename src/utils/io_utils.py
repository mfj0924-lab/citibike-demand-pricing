"""I/O utilities for reading and writing Parquet files.

Uses pandas + pyarrow to write Parquet (avoids winutils.exe requirement on Windows).
PySpark is used for reading.
"""

import os
import shutil
import pickle
from typing import Any
from pyspark.sql import DataFrame, SparkSession


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    """Read a Parquet directory into a Spark DataFrame."""
    return spark.read.parquet(path)


def write_parquet(df: DataFrame, path: str, mode: str = "overwrite") -> None:
    """Write a Spark DataFrame to Parquet using Spark's native writer."""
    if mode == "overwrite" and os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    df.write.mode(mode).parquet(path)
    print(f"  Written to: {path}")


def save_pickle(obj: Any, path: str) -> None:
    """Save a Python object to a pickle file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load a Python object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
