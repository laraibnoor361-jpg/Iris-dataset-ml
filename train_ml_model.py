"""
train_ml_model.py
-----------------
Loads the dataset, trains the KMeans model and saves the artifacts with joblib.

Usage:
    python train_ml_model.py
    python train_ml_model.py --data dataset.csv --k 3
"""

import argparse

from ml_pipeline import DATA_PATH, DEFAULT_K, MODEL_PATH, load_data, save_artifacts, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Iris KMeans model.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to the CSV dataset")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of clusters")
    parser.add_argument("--output", default=str(MODEL_PATH), help="Where to save the model")
    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    df = load_data(args.data)
    print(f"Samples: {len(df)} | Features: 4")

    print(f"Training KMeans with k={args.k} ...")
    artifacts = train_model(df, n_clusters=args.k)

    metrics = artifacts["metrics"]
    print("\nTraining finished")
    print(f"  Inertia          : {metrics['inertia']:.3f}")
    print(f"  Silhouette score : {metrics['silhouette']:.3f}")
    if "purity" in metrics:
        print(f"  Cluster purity   : {metrics['purity']:.1%}")
    print("  Cluster names    :")
    for cluster, name in artifacts["cluster_names"].items():
        print(f"    {cluster} -> {name}")

    saved_to = save_artifacts(artifacts, args.output)
    print(f"\nModel artifacts saved to: {saved_to}")


if __name__ == "__main__":
    main()
