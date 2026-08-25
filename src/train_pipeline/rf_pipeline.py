"""
Random Forest regression pipeline with Hyperopt tuning.
"""

import os, sys
from dataclasses import dataclass
from typing import Dict, Any

from pyspark.ml.regression import RandomForestRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark_session
from src.train_pipeline.abstract_pipeline import AbstractPipeline, PipelineConfig


@dataclass
class RFPipelineConfig(PipelineConfig):
    model_artifact_path: str = ""
    # Default RF params (overridden by Hyperopt)
    num_trees: int = 50
    max_depth: int = 10
    max_bins: int = 24
    min_instances_per_node: int = 10
    subsampling_rate: float = 0.8


class RFPipeline(AbstractPipeline):
    def __init__(self, spark, config: RFPipelineConfig):
        super().__init__(spark, config)

    def get_search_space(self) -> Dict[str, Any]:
        from hyperopt import hp
        return {
            "numTrees": hp.quniform("numTrees", 20, 150, 10),
            "maxDepth": hp.quniform("maxDepth", 5, 25, 1),
            "maxBins": hp.quniform("maxBins", 16, 48, 1),
            "minInstancesPerNode": hp.quniform("minInstancesPerNode", 2, 50, 1),
            "subsamplingRate": hp.uniform("subsamplingRate", 0.5, 1.0),
        }

    def get_regressor(self, label_name: str, predict_name: str, params: Dict[str, Any] = None):
        if params is None:
            params = {}
        return RandomForestRegressor(
            featuresCol=self.cfg.feature_column_name,
            labelCol=label_name,
            predictionCol=predict_name,
            seed=self.cfg.seed,
            numTrees=int(params.get("numTrees", self.cfg.num_trees)),
            maxDepth=int(params.get("maxDepth", self.cfg.max_depth)),
            maxBins=int(params.get("maxBins", self.cfg.max_bins)),
            minInstancesPerNode=int(params.get("minInstancesPerNode", self.cfg.min_instances_per_node)),
            subsamplingRate=float(params.get("subsamplingRate", self.cfg.subsampling_rate)),
        )


if __name__ == "__main__":
    spark = get_spark_session("rf_train", "8g")
    base = os.path.join(os.path.dirname(__file__), "..", "..")

    config = RFPipelineConfig(
        silver_parquet=os.path.join(base, "data", "processed", "silver", "hourly_demand.parquet"),
        model_artifact_path=os.path.join(base, "models", "rf"),
    )

    pipeline = RFPipeline(spark, config)
    pipeline.run()
    spark.stop()
