import logging
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from project_config import (
    EDA_PLOTS_DIR,
    PROCESSED_INTERACTIONS_PATH,
    RAW_DIR,
    ROOT_DIR,
    ensure_project_dirs,
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_latest_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = RAW_DIR / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interactions_path = ROOT_DIR / manifest["files"][0]["path"]
    products_path = ROOT_DIR / manifest["files"][1]["path"]
    interactions = pd.read_csv(interactions_path)
    products = pd.read_json(products_path)
    return interactions, products


def prepare_data() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Preparing data")

    interactions, products = load_latest_data()

    interactions = interactions.dropna(subset=["user_id", "product_id", "rating"])
    merged = interactions.merge(products, on="product_id", how="left")
    merged["event_timestamp"] = pd.to_datetime(merged["event_timestamp"], errors="coerce")
    merged["category"] = merged["category"].fillna("Unknown")
    merged["price"] = pd.to_numeric(merged["price"], errors="coerce")
    merged["popularity_score"] = pd.to_numeric(merged["popularity_score"], errors="coerce")
    merged["price"] = merged["price"].fillna(merged["price"].median())
    merged["popularity_score"] = merged["popularity_score"].fillna(merged["popularity_score"].median())
    merged["category_code"] = merged["category"].astype("category").cat.codes
    price_std = merged["price"].std(ddof=0)
    popularity_std = merged["popularity_score"].std(ddof=0)
    merged["price_scaled"] = 0.0 if price_std == 0 else (merged["price"] - merged["price"].mean()) / price_std
    merged["popularity_scaled"] = 0.0 if popularity_std == 0 else (merged["popularity_score"] - merged["popularity_score"].mean()) / popularity_std
    merged = merged.dropna(subset=["product_id"])

    PROCESSED_INTERACTIONS_PATH.write_text(merged.to_csv(index=False), encoding="utf-8")

    plt.figure(figsize=(8, 4))
    merged["rating"].hist(bins=[1, 2, 3, 4, 5], color="#4C78A8")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "rating_distribution.png")

    plt.figure(figsize=(8, 6))
    corr = merged[["rating", "price_scaled", "popularity_scaled", "category_code"]].corr()
    plt.imshow(corr, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "feature_correlation.png")

    plt.figure(figsize=(10, 5))
    item_popularity = merged.groupby("product_id").size().sort_values(ascending=False).head(20)
    item_popularity.plot(kind="bar", color="#4C78A8")
    plt.title("Top 20 Items by Interaction Count")
    plt.xlabel("Product ID")
    plt.ylabel("Interaction Count")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "item_popularity.png")

    interaction_matrix = pd.crosstab(merged["user_id"], merged["product_id"])
    sparsity = 1 - (interaction_matrix.gt(0).sum().sum() / interaction_matrix.size)
    plt.figure(figsize=(10, 6))
    plt.imshow(interaction_matrix.gt(0), aspect="auto", interpolation="nearest", cmap="Greys")
    plt.title(f"User-Item Interaction Sparsity ({sparsity:.2%})")
    plt.xlabel("Products")
    plt.ylabel("Users")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "interaction_sparsity.png")

    logger.info("Data preparation completed")
    return {"prepared_rows": int(len(merged)), "path": str(PROCESSED_INTERACTIONS_PATH)}


if __name__ == "__main__":
    prepare_data()
