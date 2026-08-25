"""
Silver → Gold Transformer: standardize, encode, PCA, produce training features.

Adapted from Shakleen — replaces Delta Lake with Parquet, adds PCA step.
Pipeline: VectorAssembler → StandardScaler → OneHotEncoder → (PCA) → final columns.
"""

import os
import sys
from dataclasses import dataclass
from typing import List

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    PCA,
)
from pyspark.sql import DataFrame, SparkSession

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark_session
from src.data_pipeline.abstract_transformer import AbstractTransformer
from src.utils.io_utils import read_parquet, write_parquet
from src.utils.logger import logging  # noqa: F401


@dataclass
class SilverToGoldConfig:
    hourly_demand_parquet: str = ""
    gold_features_parquet: str = ""
    pipeline_model_dir: str = ""
    pca_k: int = 8  # number of PCA components to keep


class SilverToGoldTransformer(AbstractTransformer):
    """Normalize + encode + PCA → ready-for-ML dataset."""

    def __init__(self, spark: SparkSession, config: SilverToGoldConfig):
        super().__init__(spark, config)

    def _get_feature_cols(self, df: DataFrame) -> List[str]:
        """Return the list of numeric feature columns to use."""
        exclude = {
            "station_id",
            "event_hour",
            "bike_demand",
            "dock_demand",
            "year",  # leave year as a feature
        }
        # Prefer numeric columns only
        numeric_types = {"int", "bigint", "double", "float"}
        cols = []
        for c, t in df.dtypes:
            if c not in exclude and any(nt in t for nt in numeric_types):
                cols.append(c)
        return cols

    def _build_pipeline(self, feature_cols: List[str]) -> List:
        """Build PySpark ML pipeline stages."""
        stages = []

        # Stage 0: VectorAssembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="raw_features",
            handleInvalid="skip",
        )
        stages.append(assembler)

        # Stage 1: StandardScaler
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True,
        )
        stages.append(scaler)

        # Stage 2: PCA (optional — skip when k=0)
        if self.config.pca_k > 0:
            pca = PCA(
                k=min(self.config.pca_k, len(feature_cols)),
                inputCol="scaled_features",
                outputCol="pca_features",
            )
            stages.append(pca)

        return stages

    def _apply_pipeline(self, df: DataFrame, stages: List) -> DataFrame:
        """Fit and transform using the pipeline, save the model."""
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)

        # Save pipeline model
        model_dir = self.config.pipeline_model_dir
        model.write().overwrite().save(model_dir)
        logging.info("Pipeline model saved to %s", model_dir)

        return model.transform(df)

    def _print_pca_summary(self, df: DataFrame):
        """Log PCA explained variance if PCA was applied."""
        if self.config.pca_k <= 0:
            return
        # PySpark PCA stores explainedVariance in the model
        # We can approximate by computing variance of pca_features columns
        print("  PCA applied — components saved in pipeline model.")
        # Note: PySpark PCA does not expose explained_variance_ratio easily.
        # We'll compute it in the Jupyter notebook for the report.

    def transform(self):
        logging.info("=== Silver → Gold Transformer ===")
        cfg = self.config

        df = read_parquet(self.spark, cfg.hourly_demand_parquet)
        print(f"  Input rows: {df.count():,}")

        feature_cols = self._get_feature_cols(df)
        print(f"  Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

        stages = self._build_pipeline(feature_cols)
        df = self._apply_pipeline(df, stages)
        self._print_pca_summary(df)

        # Keep final training columns
        feature_out = f"pca_features" if cfg.pca_k > 0 else "scaled_features"
        keep_cols = [
            "station_id",
            "event_hour",
            feature_out,
            "bike_demand",
            "dock_demand",
        ]
        df_out = df.select(*[c for c in keep_cols if c in df.columns])

        write_parquet(df_out, cfg.gold_features_parquet)
        print(f"  Output rows: {df_out.count():,}")
        print(f"  Written to: {cfg.gold_features_parquet}")
        logging.info("Silver → Gold done.")


if __name__ == "__main__":
    spark = get_spark_session("silver_to_gold", "4g")
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    silver_root = os.path.join(base, "data", "processed", "silver")
    gold_root = os.path.join(base, "data", "processed", "gold")
    models_root = os.path.join(base, "models")

    config = SilverToGoldConfig(
        hourly_demand_parquet=os.path.join(silver_root, "hourly_demand.parquet"),
        gold_features_parquet=os.path.join(gold_root, "train_features.parquet"),
        pipeline_model_dir=os.path.join(models_root, "gold_pipeline"),
        pca_k=5,
    )

    transformer = SilverToGoldTransformer(spark, config)
    transformer.transform()
    spark.stop()
