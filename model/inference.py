import argparse
import json
import pickle
from typing import Dict, List

import numpy as np

from project_config import MODEL_PATH


def _load_artifact() -> Dict[str, object]:
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


def recommend_for_user(user_id: str, k: int = 5) -> Dict[str, object]:
    artifact = _load_artifact()
    user_item_matrix = artifact["user_item_matrix"]
    user_factors = artifact["user_factors"]
    item_factors = artifact["item_factors"]

    user_index_as_str = user_item_matrix.index.astype(str)
    user_to_idx = {value: idx for idx, value in enumerate(user_index_as_str)}

    if user_id not in user_to_idx:
        return {
            "user_id": user_id,
            "status": "user_not_found",
            "recommendations": [],
            "message": "User not present in training matrix",
        }

    user_idx = user_to_idx[user_id]
    scores = user_factors[user_idx] @ item_factors.T
    rated_items = user_item_matrix.iloc[user_idx].to_numpy() > 0
    scores[rated_items] = -np.inf
    available_items = int((~rated_items).sum())
    top_indices = np.argsort(scores)[::-1][:min(k, available_items)]
    recommendations: List[str] = [str(user_item_matrix.columns[idx]) for idx in top_indices]
    return {
        "user_id": str(user_id),
        "status": "ok",
        "k": int(k),
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RecoMart inference interface")
    parser.add_argument("--user-id", required=True, help="User identifier")
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()

    result = recommend_for_user(args.user_id, args.k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
