import logging
import sqlite3
from typing import Dict, Any

import pandas as pd

from project_config import DATABASE_DIR, FEATURES_PATH, PROCESSED_INTERACTIONS_PATH, ensure_project_dirs


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def engineer_features() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Engineering features")
    df = pd.read_csv(PROCESSED_INTERACTIONS_PATH)

    user_activity = df.groupby("user_id").size().rename("user_activity_frequency")
    user_rating_avg = df.groupby("user_id")["rating"].mean().rename("user_avg_rating")
    item_rating_avg = df.groupby("product_id")["rating"].mean().rename("item_avg_rating")
    item_popularity = df.groupby("product_id").size().rename("item_popularity")

    feature_frame = df[["user_id", "product_id", "rating", "price_scaled", "popularity_scaled", "category_code"]].copy()
    feature_frame = feature_frame.merge(user_activity, on="user_id", how="left")
    feature_frame = feature_frame.merge(user_rating_avg, on="user_id", how="left")
    feature_frame = feature_frame.merge(item_rating_avg, on="product_id", how="left")
    feature_frame = feature_frame.merge(item_popularity, on="product_id", how="left")

    feature_frame.to_csv(FEATURES_PATH, index=False)

    database_path = DATABASE_DIR / "recomart_features.db"
    schema_path = DATABASE_DIR / "schema.sql"
    user_features = feature_frame[["user_id", "user_activity_frequency", "user_avg_rating"]].drop_duplicates("user_id").copy()
    item_features = feature_frame[["product_id", "item_avg_rating", "item_popularity"]].drop_duplicates("product_id").copy()
    interaction_features = feature_frame[["user_id", "product_id", "rating", "price_scaled", "popularity_scaled", "category_code"]].copy()
    user_features["user_id"] = user_features["user_id"].astype(str)
    item_features["product_id"] = item_features["product_id"].astype(str)
    interaction_features["user_id"] = interaction_features["user_id"].astype(str)
    interaction_features["product_id"] = interaction_features["product_id"].astype(str)

    with sqlite3.connect(database_path) as connection:
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        connection.execute("DELETE FROM user_features")
        connection.execute("DELETE FROM item_features")
        connection.execute("DELETE FROM interaction_features")
        user_features.to_sql("user_features", connection, if_exists="append", index=False)
        item_features.to_sql("item_features", connection, if_exists="append", index=False)
        interaction_features.to_sql("interaction_features", connection, if_exists="append", index=False)

    logger.info("Feature engineering completed")
    return {"feature_rows": int(len(feature_frame)), "path": str(FEATURES_PATH), "database_path": str(database_path)}


if __name__ == "__main__":
    engineer_features()
