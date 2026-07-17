import json
import logging
import os
import pickle
from typing import Any, Dict, List

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

from project_config import FEATURES_PATH, MODEL_METADATA_PATH, MODEL_PATH, ROOT_DIR, ensure_project_dirs


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def precision_at_k(actual: List[str], recommended: List[str], k: int = 5) -> float:
    if not actual:
        return 0.0
    return len(set(recommended[:k]).intersection(actual)) / k


def recall_at_k(actual: List[str], recommended: List[str], k: int = 5) -> float:
    if not actual:
        return 0.0
    return len(set(recommended[:k]).intersection(actual)) / len(actual)


def ndcg_at_k(actual: List[str], recommended: List[str], k: int = 5) -> float:
    if not actual:
        return 0.0
    gains = [1.0 if item in set(actual) else 0.0 for item in recommended[:k]]
    dcg = sum(gain / np.log2(idx + 2) for idx, gain in enumerate(gains))
    ideal_gains = [1.0] * min(len(actual), k)
    idcg = sum(gain / np.log2(idx + 2) for idx, gain in enumerate(ideal_gains))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def train_model() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Training recommendation model")
    features = pd.read_csv(FEATURES_PATH, dtype={"user_id": str, "product_id": str})
    features["rating"] = pd.to_numeric(features["rating"], errors="coerce")

    eligible_test_rows = features[features["rating"] >= 4.0]
    test_indices = eligible_test_rows.groupby("user_id").tail(1).index
    test_features = features.loc[test_indices].copy()
    train_features = features.drop(index=test_indices).copy()

    all_users = features["user_id"].drop_duplicates().tolist()
    all_products = features["product_id"].drop_duplicates().tolist()
    user_item_matrix = train_features.pivot_table(index="user_id", columns="product_id", values="rating", fill_value=0.0)
    user_item_matrix = user_item_matrix.reindex(index=all_users, columns=all_products, fill_value=0.0)
    matrix = user_item_matrix.to_numpy()
    nmf = NMF(n_components=3, random_state=42, max_iter=500)
    user_factors = nmf.fit_transform(matrix)
    item_factors = nmf.components_.T
    predictions = user_factors @ item_factors.T

    user_to_idx = {str(user_id): idx for idx, user_id in enumerate(user_item_matrix.index)}
    item_to_idx = {str(product_id): idx for idx, product_id in enumerate(user_item_matrix.columns)}
    test_actual: List[float] = []
    test_predictions: List[float] = []
    for row in test_features.itertuples(index=False):
        user_idx = user_to_idx.get(str(row.user_id))
        item_idx = item_to_idx.get(str(row.product_id))
        if user_idx is not None and item_idx is not None:
            test_actual.append(float(row.rating))
            test_predictions.append(float(predictions[user_idx, item_idx]))

    rmse = float(np.sqrt(mean_squared_error(test_actual, test_predictions)))
    relevant_by_user = test_features.groupby("user_id")["product_id"].apply(lambda values: [str(value) for value in values]).to_dict()
    top_k = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    ndcg_scores: List[float] = []
    for user_idx, user_id in enumerate(user_item_matrix.index):
        user_scores = predictions[user_idx].copy()
        user_scores[matrix[user_idx] > 0] = -np.inf
        recommended = [str(user_item_matrix.columns[idx]) for idx in np.argsort(user_scores)[::-1][:5]]
        relevant = relevant_by_user.get(str(user_id), [])
        if relevant:
            precision_scores.append(precision_at_k(relevant, recommended, k=5))
            recall_scores.append(recall_at_k(relevant, recommended, k=5))
            ndcg_scores.append(ndcg_at_k(relevant, recommended, k=5))
        top_k.append({"user_id": str(user_id), "recommendations": recommended})

    metrics = {
        "rmse": rmse,
        "precision_at_5": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall_at_5": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "ndcg_at_5": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
    }
    metadata = {
        "model": "NMF",
        "parameters": {"n_components": 3, "random_state": 42},
        "evaluation": {
            "method": "per-user positive interaction holdout",
            "training_rows": int(len(train_features)),
            "test_rows": int(len(test_features)),
        },
        "metrics": metrics,
        "recommendations": top_k,
    }

    with MODEL_PATH.open("wb") as handle:
        pickle.dump({"nmf": nmf, "user_item_matrix": user_item_matrix, "user_factors": user_factors, "item_factors": item_factors}, handle)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT_DIR / "mlruns").resolve().as_uri()))
    mlflow.set_experiment("recomart-recommendation")
    with mlflow.start_run(run_name="recomart-nmf-run") as run:
        metadata["mlflow_run_id"] = run.info.run_id
        serializable_metadata = to_serializable(metadata)
        MODEL_METADATA_PATH.write_text(json.dumps(serializable_metadata, indent=2), encoding="utf-8")
        mlflow.log_params(metadata["parameters"])
        mlflow.log_param("evaluation_method", metadata["evaluation"]["method"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(MODEL_METADATA_PATH))
        mlflow.log_artifact(str(FEATURES_PATH))

    logger.info("Model training completed")
    return metadata


if __name__ == "__main__":
    train_model()


