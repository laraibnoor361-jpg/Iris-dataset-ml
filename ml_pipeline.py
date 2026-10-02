"""
ml_pipeline.py
--------------
Core machine-learning logic for the Iris KMeans clustering project:
data loading, preprocessing, feature scaling, model training,
evaluation, prediction and artifact persistence.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "kmeans_iris_model.pkl"

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
DEFAULT_K = 3
RANDOM_STATE = 42

# Maps many common Iris column spellings (e.g. the Kaggle "Iris.csv" file)
# to the canonical column names used in this project.
_COLUMN_ALIASES = {
    "sepallength": "sepal_length",
    "sepallengthcm": "sepal_length",
    "sepalwidth": "sepal_width",
    "sepalwidthcm": "sepal_width",
    "petallength": "petal_length",
    "petallengthcm": "petal_length",
    "petalwidth": "petal_width",
    "petalwidthcm": "petal_width",
    "species": "species",
    "variety": "species",
    "class": "species",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical names and drop unrelated ones (e.g. Id)."""
    renamed = {}
    for col in df.columns:
        key = re.sub(r"[^a-z]", "", str(col).lower())
        if key in _COLUMN_ALIASES:
            renamed[col] = _COLUMN_ALIASES[key]
    df = df.rename(columns=renamed)
    keep = [c for c in FEATURES + [TARGET] if c in df.columns]
    return df[keep].copy()


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load and preprocess the Iris dataset."""
    df = pd.read_csv(path)
    df = standardize_columns(df)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=FEATURES).drop_duplicates().reset_index(drop=True)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if TARGET in df.columns:
        df[TARGET] = (
            df[TARGET].astype(str).str.strip().str.replace("Iris-", "", regex=False).str.lower()
        )
    return df


def build_pipeline(n_clusters: int = DEFAULT_K) -> Pipeline:
    """Feature scaling + KMeans in a single scikit-learn pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE),
            ),
        ]
    )


def _name_clusters(labels: np.ndarray, species: pd.Series, n_clusters: int) -> dict:
    """Give every cluster a readable name based on its majority species."""
    names = {}
    for cluster in range(n_clusters):
        members = species[labels == cluster]
        if len(members) == 0:
            names[cluster] = f"Cluster {cluster}"
        else:
            majority = members.value_counts().idxmax()
            names[cluster] = f"Cluster {cluster} (mostly {majority})"
    return names


def train_model(df: pd.DataFrame, n_clusters: int = DEFAULT_K) -> dict:
    """Train the pipeline and return every artifact needed for inference/visuals."""
    if len(df) <= n_clusters:
        raise ValueError("Need more samples than clusters to train KMeans.")

    X = df[FEATURES]
    pipeline = build_pipeline(n_clusters)
    pipeline.fit(X)
    labels = pipeline.predict(X)

    scaled = pipeline.named_steps["scaler"].transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(scaled)

    metrics = {
        "n_samples": int(len(df)),
        "n_clusters": int(n_clusters),
        "inertia": float(pipeline.named_steps["kmeans"].inertia_),
        "silhouette": float(silhouette_score(scaled, labels)) if n_clusters > 1 else 0.0,
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
    }

    if TARGET in df.columns:
        cluster_names = _name_clusters(labels, df[TARGET], n_clusters)
        majority_hits = sum(
            (df[TARGET][labels == c].value_counts().max() if (labels == c).any() else 0)
            for c in range(n_clusters)
        )
        metrics["purity"] = float(majority_hits / len(df))
    else:
        cluster_names = {c: f"Cluster {c}" for c in range(n_clusters)}

    return {
        "pipeline": pipeline,
        "pca": pca,
        "features": FEATURES,
        "cluster_names": cluster_names,
        "metrics": metrics,
    }


def elbow_analysis(df: pd.DataFrame, k_max: int = 8) -> pd.DataFrame:
    """Inertia and silhouette score for k = 2..k_max (used for the elbow plot)."""
    scaled = StandardScaler().fit_transform(df[FEATURES])
    k_max = max(2, min(k_max, len(df) - 1))
    rows = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaled)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(scaled, km.labels_)),
            }
        )
    return pd.DataFrame(rows)


def assign_clusters(artifacts: dict, df: pd.DataFrame) -> np.ndarray:
    """Cluster labels for every row of a dataframe."""
    return artifacts["pipeline"].predict(df[artifacts["features"]])


def predict_cluster(artifacts: dict, sepal_length, sepal_width, petal_length, petal_width) -> dict:
    """Predict the cluster for one flower and return distances to all centroids."""
    row = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=artifacts["features"],
    )
    pipeline = artifacts["pipeline"]
    cluster = int(pipeline.predict(row)[0])
    distances = pipeline.named_steps["kmeans"].transform(
        pipeline.named_steps["scaler"].transform(row)
    )[0]
    return {
        "cluster": cluster,
        "name": artifacts["cluster_names"].get(cluster, f"Cluster {cluster}"),
        "distances": distances,
        "row": row,
    }


def save_artifacts(artifacts: dict, path=MODEL_PATH) -> Path:
    joblib.dump(artifacts, path)
    return Path(path)


def load_artifacts(path=MODEL_PATH) -> dict:
    return joblib.load(path)
