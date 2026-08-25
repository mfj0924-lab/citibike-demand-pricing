"""PySpark Session configuration for local development.

Sets up a SparkSession with JDK 17 and local[*] master mode.
Import get_spark_session() wherever you need a SparkSession.
"""

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "bike_pricing", driver_memory: str = "4g") -> SparkSession:
    """Create or return a local PySpark session.

    Args:
        app_name: Spark application name.
        driver_memory: JVM heap size for driver (e.g. '4g', '8g').

    Returns:
        Configured SparkSession in local mode.
    """
    java_home = os.getenv("JAVA_HOME")
    if not java_home:
        raise EnvironmentError("请先安装 JDK 17 并设置 JAVA_HOME。")

    python_executable = os.getenv("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_PYTHON", python_executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_executable)

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    )

    hadoop_home = os.getenv("HADOOP_HOME")
    if hadoop_home:
        hadoop_bin = (Path(hadoop_home) / "bin").as_posix()
        builder = builder.config(
            "spark.driver.extraJavaOptions",
            f"-Dfile.encoding=UTF-8 -Djava.library.path={hadoop_bin}",
        )
    else:
        builder = builder.config(
            "spark.driver.extraJavaOptions", "-Dfile.encoding=UTF-8"
        )

    return builder.getOrCreate()


if __name__ == "__main__":
    spark = get_spark_session("test")
    print(f"Spark version: {spark.version}")
    print(f"Java home: {os.environ.get('JAVA_HOME', 'NOT SET')}")
    spark.stop()
