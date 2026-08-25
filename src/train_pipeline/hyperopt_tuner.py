"""
Hyperopt Bayesian optimization for PySpark ML model hyperparameters.

Runs N trials, each training on a subset of data, evaluating RMSE on a validation
subset. Uses Tree-structured Parzen Estimator (TPE) for efficient search.
"""

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Dict, Any

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.io_utils import read_parquet


@dataclass
class HyperoptConfig:
    silver_parquet: str = ""
    output_json: str = ""
    max_evals: int = 20       # number of Hyperopt trials
    train_fraction: float = 0.8
    random_seed: int = 42


def run_hyperopt_tuning(
    spark: SparkSession,
    config: HyperoptConfig,
    model_type: str = "rf",
    label_col: str = "bike_demand",
) -> Dict[str, Any]:
    """Run Hyperopt Bayesian optimization for RF or GBT on a given target.

    Args:
        spark: Active SparkSession.
        config: HyperoptConfig with paths and settings.
        model_type: 'rf' or 'gbt'.
        label_col: 'bike_demand' or 'dock_demand'.

    Returns:
        Dict with best params and metrics.
    """
    # Load and prepare data
    df = read_parquet(spark, config.silver_parquet)

    # Feature columns
    exclude = {"station_id", "event_hour", "bike_demand", "dock_demand",
               "year", "capacity"}
    numeric = {"int", "bigint", "double", "float"}
    feature_cols = [c for c, t in df.dtypes if c not in exclude and any(n in t for n in numeric)]

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="skip"
    )
    df = assembler.transform(df)

    # Sample for faster tuning (500k rows), then randomSplit
    total = df.count()
    sample_fraction = min(1.0, 500000 / total)
    df_sample = df.sample(fraction=sample_fraction, seed=config.random_seed).cache()
    df_sample = df_sample.select("event_hour", "features", label_col)

    train, val = df_sample.randomSplit(
        [config.train_fraction, 1.0 - config.train_fraction], seed=config.random_seed
    )
    train = train.cache()
    val = val.cache()
    print(f"  Total sampled: {df_sample.count():,}  Train: {train.count():,}  Val: {val.count():,}  Features: {len(feature_cols)}")

    # Define search space
    if model_type == "rf":
        space = {
            "numTrees": hp.quniform("numTrees", 20, 150, 10),
            "maxDepth": hp.quniform("maxDepth", 5, 25, 1),
            "maxBins": hp.quniform("maxBins", 16, 48, 1),
            "minInstancesPerNode": hp.quniform("minInstancesPerNode", 2, 50, 1),
            "subsamplingRate": hp.uniform("subsamplingRate", 0.5, 1.0),
        }
        RegressorClass = RandomForestRegressor
    else:
        space = {
            "maxIter": hp.quniform("maxIter", 20, 100, 10),
            "maxDepth": hp.quniform("maxDepth", 5, 20, 1),
            "maxBins": hp.quniform("maxBins", 16, 48, 1),
            "minInstancesPerNode": hp.quniform("minInstancesPerNode", 2, 50, 1),
            "subsamplingRate": hp.uniform("subsamplingRate", 0.5, 1.0),
        }
        RegressorClass = GBTRegressor

    trials = Trials()
    trial_results = []

    def objective(params):
        trial_num = len(trial_results) + 1

        # Convert float params to int where needed
        for k in ["numTrees", "maxDepth", "maxBins", "minInstancesPerNode", "maxIter"]:
            if k in params:
                params[k] = int(params[k])

        if model_type == "rf":
            regressor = RegressorClass(
                featuresCol="features",
                labelCol=label_col,
                predictionCol="prediction",
                seed=config.random_seed,
                numTrees=int(params.get("numTrees", 100)),
                maxDepth=int(params.get("maxDepth", 10)),
                maxBins=int(params.get("maxBins", 32)),
                minInstancesPerNode=int(params.get("minInstancesPerNode", 5)),
                subsamplingRate=float(params.get("subsamplingRate", 0.8)),
            )
        elif model_type == "gbt":
            regressor = RegressorClass(
                featuresCol="features",
                labelCol=label_col,
                predictionCol="prediction",
                seed=config.random_seed,
                maxIter=int(params.get("maxIter", 20)),
                maxDepth=int(params.get("maxDepth", 10)),
                maxBins=int(params.get("maxBins", 32)),
                minInstancesPerNode=int(params.get("minInstancesPerNode", 5)),
                subsamplingRate=float(params.get("subsamplingRate", 0.8)),
            )

        t0 = time.time()
        model = regressor.fit(train)
        elapsed = time.time() - t0

        evaluator = RegressionEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="rmse"
        )
        rmse = evaluator.evaluate(model.transform(val))
        r2_eval = RegressionEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="r2"
        )
        r2 = r2_eval.evaluate(model.transform(val))

        result = {**params, "rmse": rmse, "r2": r2, "time_s": elapsed, "trial": trial_num}
        trial_results.append(result)
        print(f"  Trial {trial_num:2d}  RMSE={rmse:.4f}  R2={r2:.4f}  {elapsed:.1f}s  "
              f"params={json.dumps({k: v for k, v in params.items()}, default=str)}")

        return {"loss": rmse, "status": STATUS_OK}

    # Run optimization
    print(f"\n  Running Hyperopt ({config.max_evals} trials, {model_type}, {label_col})...")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=config.max_evals,
        trials=trials,
        rstate=None,
    )

    best_params = {k: int(v) if k in ["numTrees", "maxDepth", "maxBins",
                 "minInstancesPerNode", "maxIter"] else float(v)
                 for k, v in best.items()}

    best_trial = min(trial_results, key=lambda x: x["rmse"])
    print(f"\n  Best:  RMSE={best_trial['rmse']:.4f}  R2={best_trial['r2']:.4f}")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")

    # Save results
    output = {
        "model_type": model_type,
        "label_col": label_col,
        "best_params": best_params,
        "best_rmse": best_trial["rmse"],
        "best_r2": best_trial["r2"],
        "trials": trial_results,
        "feature_cols": feature_cols,
    }

    if config.output_json:
        os.makedirs(os.path.dirname(config.output_json) or ".", exist_ok=True)
        with open(config.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Results saved to {config.output_json}")

    return output


if __name__ == "__main__":
    from config.spark_config import get_spark_session
    spark = get_spark_session("hyperopt", "5g")
    base = os.path.join(os.path.dirname(__file__), "..", "..")

    silver = os.path.join(base, "data", "processed", "silver", "hourly_demand.parquet")
    models_dir = os.path.join(base, "models")

    for model_type in ["rf", "gbt"]:
        for label in ["bike_demand", "dock_demand"]:
            print(f"\n{'='*60}")
            print(f"  Hyperopt: {model_type} → {label}")
            print(f"{'='*60}")
            config = HyperoptConfig(
                silver_parquet=silver,
                output_json=os.path.join(models_dir, f"hyperopt_{model_type}_{label}.json"),
                max_evals=8,
            )
            run_hyperopt_tuning(spark, config, model_type=model_type, label_col=label)

    spark.stop()
