"""
Dimensionality reduction: PCA (PySpark ML) + t-SNE (sklearn, on sampled data).

- PCA: reduce 10+ numeric features to K principal components.
- t-SNE: 2D visualization of demand patterns on a 50k-row sample.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark_session
from src.utils.io_utils import read_parquet
from src.utils.logger import logging  # noqa: F401


@dataclass
class DimReductionConfig:
    silver_parquet: str = ""
    pca_k: int = 5
    tsne_sample_n: int = 50000
    tsne_perplexity: int = 30
    random_seed: int = 42


class DimensionalityReducer:
    """Run PCA via PySpark ML and t-SNE via sklearn on sampled data."""

    def __init__(self, spark, config: DimReductionConfig):
        self.spark = spark
        self.cfg = config

    # ── feature column selection ─────────────────────────────

    @staticmethod
    def _numeric_feature_cols(df) -> List[str]:
        """Return numeric feature column names (excluding IDs and targets)."""
        exclude = {
            "station_id",
            "event_hour",
            "bike_demand",
            "dock_demand",
            "year",
            "capacity",
        }
        numeric_types = {"int", "bigint", "double", "float"}
        cols = []
        for c, t in df.dtypes:
            if c not in exclude and any(nt in t for nt in numeric_types):
                cols.append(c)
        return cols

    # ── PCA via PySpark ──────────────────────────────────────

    def run_pca(
        self, output_model_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fit PCA on silver data, return (explained_variance, loadings, feature_names)."""
        from pyspark.ml.feature import PCA, VectorAssembler, StandardScaler

        df = read_parquet(self.spark, self.cfg.silver_parquet)
        feature_cols = self._numeric_feature_cols(df)
        print(f"  PCA feature columns ({len(feature_cols)}): {feature_cols}")

        # Assemble + scale
        assembler = VectorAssembler(
            inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip"
        )
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True,
        )
        df = assembler.transform(df)
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)

        k = min(self.cfg.pca_k, len(feature_cols))
        pca = PCA(k=k, inputCol="scaled_features", outputCol="pca_features")
        pca_model = pca.fit(df)

        # Explained variance ratio
        ev = np.array(pca_model.explainedVariance)
        ev_ratio = ev / ev.sum()
        cumsum = np.cumsum(ev_ratio)
        print(f"  PCA {k} components, cumulative variance explained:")
        for i, (r, c) in enumerate(zip(ev_ratio, cumsum)):
            print(f"    PC{i+1}: {r:.4f}  (cumulative: {c:.4f})")

        # Save model
        if output_model_path:
            pca_model.write().overwrite().save(output_model_path)

        return ev_ratio, ev, feature_cols

    # ── t-SNE via sklearn ───────────────────────────────────

    def run_tsne(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample silver data, run t-SNE to 2D, return (tsne_xy, demands, hours)."""

        df = read_parquet(self.spark, self.cfg.silver_parquet)
        feature_cols = self._numeric_feature_cols(df)

        # Stratified sample: ensure spread across hours
        sample = df.sample(
            fraction=self.cfg.tsne_sample_n
            / df.count()
            * 2,  # oversample then truncate
            seed=self.cfg.random_seed,
        ).limit(self.cfg.tsne_sample_n)

        # Collect to pandas
        pdf = sample.select(
            *feature_cols, "bike_demand", "dock_demand", "hour"
        ).toPandas()
        # Drop duplicate columns (hour appears in both feature_cols and explicit select)
        pdf = pdf.loc[:, ~pdf.columns.duplicated()]
        print(f"  t-SNE sample: {len(pdf):,} rows")

        # Scale
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(pdf[feature_cols].values)

        # t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=self.cfg.tsne_perplexity,
            random_state=self.cfg.random_seed,
            n_jobs=-1,
            verbose=0,
        )
        X_2d = tsne.fit_transform(X)
        print(f"  t-SNE done. KL divergence: {tsne.kl_divergence_:.2f}")

        return (
            X_2d,
            pdf["bike_demand"].values + pdf["dock_demand"].values,
            pdf["hour"].values,
        )


if __name__ == "__main__":
    spark = get_spark_session("dim_reduction", "4g")
    base = os.path.join(os.path.dirname(__file__), "..", "..")

    cfg = DimReductionConfig(
        silver_parquet=os.path.join(
            base, "data", "processed", "silver", "hourly_demand.parquet"
        ),
        pca_k=5,
        tsne_sample_n=50000,
    )

    reducer = DimensionalityReducer(spark, cfg)

    # Run PCA
    print("=" * 50)
    print("PCA Analysis")
    print("=" * 50)
    ev_ratio, ev, feature_cols = reducer.run_pca(
        output_model_path=os.path.join(base, "models", "pca_model")
    )
    print(f"  Top feature per PC (by loading magnitude)...")

    # Run t-SNE
    print()
    print("=" * 50)
    print("t-SNE Analysis")
    print("=" * 50)
    tsne_xy, total_demand, hours = reducer.run_tsne()

    # Save t-SNE results for visualization
    tsne_df = pd.DataFrame(
        {
            "tsne_x": tsne_xy[:, 0].ravel(),
            "tsne_y": tsne_xy[:, 1].ravel(),
            "total_demand": total_demand,
            "hour": hours,
        }
    )
    tsne_path = os.path.join(base, "data", "processed", "tsne_results.csv")
    tsne_df.to_csv(tsne_path, index=False)
    print(f"  t-SNE results saved to {tsne_path}")

    spark.stop()
