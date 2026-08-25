"""
Abstract training pipeline — adapted from Shakleen.

Base class for RF and GBT pipelines. Manages:
- Train/val/test split (80/10/10)
- Hyperopt search space definition (abstract)
- Model training, evaluation, and saving
"""

import os
import json
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

from src.utils.io_utils import read_parquet


@dataclass
class PipelineConfig:
    silver_parquet: str = ""
    model_artifact_path: str = ""
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    seed: int = 42
    feature_column_name: str = "features"
    feature_cols: List[str] = field(default_factory=list)


class AbstractPipeline:
    """Base class for PySpark ML regression pipelines."""

    def __init__(self, spark: SparkSession, config: PipelineConfig):
        self.spark = spark
        self.cfg = config

    # ── Feature columns ─────────────────────────────────

    @staticmethod
    def _numeric_feature_cols(df) -> List[str]:
        exclude = {"station_id", "event_hour", "bike_demand", "dock_demand",
                   "year", "capacity", "pca_features", "scaled_features", "raw_features"}
        numeric = {"int", "bigint", "double", "float"}
        return [c for c, t in df.dtypes if c not in exclude and any(n in t for n in numeric)]

    # ── Data loading & splitting ────────────────────────

    def load_data(self) -> Tuple:
        """Load silver data, split into train/val/test by randomSplit."""
        from pyspark.ml.feature import VectorAssembler

        df = read_parquet(self.spark, self.cfg.silver_parquet)

        if not self.cfg.feature_cols:
            self.cfg.feature_cols = self._numeric_feature_cols(df)

        assembler = VectorAssembler(
            inputCols=self.cfg.feature_cols,
            outputCol=self.cfg.feature_column_name,
            handleInvalid="skip",
        )
        df = assembler.transform(df)

        # Cache for repeated access in training
        df = df.select("event_hour", self.cfg.feature_column_name, "bike_demand", "dock_demand")
        df = df.cache()

        total = df.count()
        train, val, test = df.randomSplit(
            [self.cfg.train_fraction, self.cfg.val_fraction,
             1.0 - self.cfg.train_fraction - self.cfg.val_fraction],
            seed=self.cfg.seed,
        )
        train = train.cache()
        val = val.cache()

        print(f"  Total: {total:,}  Train: {train.count():,}  Val: {val.count():,}  Test: {test.count():,}")
        return train, val, test

    # ── Hyperopt integration ────────────────────────────

    @abstractmethod
    def get_search_space(self) -> Dict[str, Any]:
        """Return the Hyperopt search space."""
        ...

    @abstractmethod
    def get_regressor(self, label_name: str, predict_name: str, params: Dict[str, Any] = None):
        """Create an RF or GBT regressor with given parameters."""
        ...

    # ── Training ────────────────────────────────────────

    def train_and_evaluate(
        self, train, val, test,
        label_name: str, predict_name: str,
        model_name: str,
        params: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """Train model, evaluate on val/test, save."""
        if params is None:
            params = {}

        regressor = self.get_regressor(label_name, predict_name, params)
        t0 = time.time()
        model = regressor.fit(train)
        elapsed = time.time() - t0
        print(f"    {model_name} trained in {elapsed:.1f}s")

        # Evaluate
        evaluator_rmse = RegressionEvaluator(
            labelCol=label_name, predictionCol=predict_name, metricName="rmse"
        )
        evaluator_r2 = RegressionEvaluator(
            labelCol=label_name, predictionCol=predict_name, metricName="r2"
        )
        evaluator_mae = RegressionEvaluator(
            labelCol=label_name, predictionCol=predict_name, metricName="mae"
        )

        val_rmse = evaluator_rmse.evaluate(model.transform(val))
        val_r2 = evaluator_r2.evaluate(model.transform(val))
        val_mae = evaluator_mae.evaluate(model.transform(val))

        test_rmse = evaluator_rmse.evaluate(model.transform(test))
        test_r2 = evaluator_r2.evaluate(model.transform(test))
        test_mae = evaluator_mae.evaluate(model.transform(test))

        # Feature importance
        if hasattr(model, "featureImportances"):
            importances = model.featureImportances.toArray()
            feat_imp = dict(zip(self.cfg.feature_cols, importances.tolist()))
            top5 = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            print(f"    Top-5 features: {[(f, round(i, 4)) for f, i in top5]}")

        # Save
        artifact_dir = os.path.join(self.cfg.model_artifact_path, model_name)
        model.write().overwrite().save(artifact_dir)

        results = {
            "model": model_name,
            "val_rmse": val_rmse, "val_r2": val_r2, "val_mae": val_mae,
            "test_rmse": test_rmse, "test_r2": test_r2, "test_mae": test_mae,
            "train_time_s": elapsed,
            "params": params,
        }
        print(f"    Val  RMSE={val_rmse:.2f}  R2={val_r2:.4f}  MAE={val_mae:.2f}")
        print(f"    Test RMSE={test_rmse:.2f}  R2={test_r2:.4f}  MAE={test_mae:.2f}")
        return results

    def run(self):
        """Full training run: load, train 2 tasks, compare."""
        print("=" * 60)
        print(f"  Training: {self.__class__.__name__}")
        print("=" * 60)

        train, val, test = self.load_data()

        results = []
        for task_label, task_name in [("bike_demand", "bike"), ("dock_demand", "dock")]:
            print(f"\n  --- {task_name} ---")
            r = self.train_and_evaluate(
                train, val, test,
                label_name=task_label,
                predict_name=f"pred_{task_name}",
                model_name=f"{task_name}_model",
            )
            results.append(r)

        # Summary
        print("\n" + "=" * 60)
        print("  Summary")
        for r in results:
            print(f"  {r['model']}: Test RMSE={r['test_rmse']:.2f}, R2={r['test_r2']:.4f}")

        # Save summary
        summary_path = os.path.join(
            self.cfg.model_artifact_path, "training_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Summary saved to {summary_path}")

        return results
