"""
test_ml_model.py
----------------
Loads the saved KMeans model and predicts the cluster for new Iris measurements.
Run `python train_ml_model.py` first.

Usage:
    python test_ml_model.py
"""

import sys

from ml_pipeline import MODEL_PATH, load_artifacts, predict_cluster

SAMPLES = [
    {"label": "Typical setosa", "values": (5.0, 3.4, 1.5, 0.2)},
    {"label": "Typical versicolor", "values": (6.0, 2.8, 4.4, 1.4)},
    {"label": "Typical virginica", "values": (6.8, 3.0, 5.7, 2.1)},
    {"label": "Small flower", "values": (4.8, 3.1, 1.6, 0.2)},
    {"label": "Large flower", "values": (7.4, 2.9, 6.3, 2.0)},
]


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run: python train_ml_model.py")
        return 1

    artifacts = load_artifacts()
    print(f"Loaded model ({artifacts['metrics']['n_clusters']} clusters)\n")
    print(f"{'Sample':<20}{'Measurements':<26}Predicted cluster")
    print("-" * 78)

    for sample in SAMPLES:
        result = predict_cluster(artifacts, *sample["values"])
        assert 0 <= result["cluster"] < artifacts["metrics"]["n_clusters"]
        print(f"{sample['label']:<20}{str(sample['values']):<26}{result['name']}")

    print("\nAll predictions completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
