import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from project_config import (
    FEATURES_PATH,
    FEATURE_STORE_METADATA_PATH,
    FEATURE_STORE_VERSIONS_DIR,
    ensure_project_dirs,
)


class FeatureStore:
    def __init__(self, feature_path: str = str(FEATURES_PATH)) -> None:
        self.feature_path = Path(feature_path)
        self.metadata_path = FEATURE_STORE_METADATA_PATH
        self.versions_dir = FEATURE_STORE_VERSIONS_DIR
        ensure_project_dirs()

    def save_features(self, features: pd.DataFrame, version: str = "v1") -> Dict[str, Any]:
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        versioned_feature_path = version_dir / self.feature_path.name

        features.to_csv(versioned_feature_path, index=False)

        existing_versions: List[str] = []
        if self.metadata_path.exists():
            try:
                previous = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                existing_versions = list(previous.get("available_versions", []))
            except json.JSONDecodeError:
                existing_versions = []
        if version not in existing_versions:
            existing_versions.append(version)

        metadata = {
            "version": version,
            "available_versions": existing_versions,
            "feature_columns": features.columns.tolist(),
            "feature_definitions": {
                "user_id": {
                    "source": "prepared_interactions.csv",
                    "transformation": "Direct user identifier from interactions",
                },
                "product_id": {
                    "source": "prepared_interactions.csv",
                    "transformation": "Direct product identifier from interactions",
                },
                "rating": {
                    "source": "prepared_interactions.csv",
                    "transformation": "Direct user-product rating",
                },
                "price_scaled": {
                    "source": "products_api.json",
                    "transformation": "Standardized product price",
                },
                "popularity_scaled": {
                    "source": "products_api.json",
                    "transformation": "Standardized product popularity score",
                },
                "category_code": {
                    "source": "products_api.json",
                    "transformation": "Categorical encoding of product category",
                },
                "user_activity_frequency": {
                    "source": "prepared_interactions.csv",
                    "transformation": "Count of interactions per user",
                },
                "user_avg_rating": {
                    "source": "prepared_interactions.csv",
                    "transformation": "Average rating per user",
                },
                "item_avg_rating": {
                    "source": "prepared_interactions.csv",
                    "transformation": "Average rating per product",
                },
                "item_popularity": {
                    "source": "prepared_interactions.csv",
                    "transformation": "Count of interactions per product",
                },
            },
            "rows": int(len(features)),
            "latest_path": str(self.feature_path),
            "versioned_path": str(versioned_feature_path),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def _resolve_feature_path(self, version: str | None) -> Path:
        if not version or version == "latest":
            return self.feature_path
        version_path = self.versions_dir / version / self.feature_path.name
        if version_path.exists():
            return version_path
        raise FileNotFoundError(f"No features found for version: {version}")

    def get_feature_vector(self, user_id: str, product_id: str, version: str = "latest") -> Dict[str, Any]:
        features = pd.read_csv(
            self._resolve_feature_path(version),
            dtype={"user_id": str, "product_id": str},
        )
        subset = features[
            (features["user_id"] == str(user_id))
            & (features["product_id"] == str(product_id))
        ]
        if subset.empty:
            return {"status": "not_found"}
        row = subset.iloc[0].to_dict()
        row["version"] = version
        return row

    def list_versions(self) -> List[str]:
        if not self.metadata_path.exists():
            return []
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return list(metadata.get("available_versions", []))


def run_feature_store() -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Registering feature store")
    store = FeatureStore()
    features = pd.read_csv(FEATURES_PATH)
    version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    return store.save_features(features, version=version)


if __name__ == "__main__":
    run_feature_store()
