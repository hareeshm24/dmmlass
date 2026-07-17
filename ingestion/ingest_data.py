import json
import logging
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from project_config import (
    RAW_DIR,
    RAW_INTERACTIONS_PATH,
    RAW_PRODUCTS_PATH,
    ROOT_DIR,
    ensure_project_dirs,
)
from ingestion.load_movielens import load_movielens_dataset

LOG_PATH = ROOT_DIR / "logs" / "ingestion.log"
MANIFEST_PATH = RAW_DIR / "ingestion_manifest.json"
NEAR_REAL_TIME_MANIFEST_PATH = RAW_DIR / "near_real_time_ingestion_manifest.json"
PRODUCTS_API_URL = "https://dummyjson.com/products?limit=0&select=id,title,category,price,rating"


def execute_with_retry(func: Any, *args: Any, retries: int = 3, backoff_seconds: int = 2, **kwargs: Any) -> Any:
    logger = logging.getLogger(__name__)
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.warning("Attempt %s/%s for %s failed: %s", attempt, retries, func.__name__, exc)
            if attempt == retries:
                raise
            time.sleep(backoff_seconds * attempt)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )


def write_partitioned_file(data: Any, source: str, file_name: str) -> Path:
    date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    destination_dir = RAW_DIR / source / date_prefix
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / file_name
    if isinstance(data, pd.DataFrame):
        if file_name.endswith(".json"):
            data.to_json(destination_path, orient="records", indent=2)
        else:
            data.to_csv(destination_path, index=False)
    else:
        destination_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return destination_path


def load_interactions(source_path: Path) -> pd.DataFrame:
    return pd.read_csv(source_path)


def load_products(source_path: Path) -> List[Dict[str, Any]]:
    with source_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_products_from_api(api_url: str = PRODUCTS_API_URL) -> List[Dict[str, Any]]:
    request = urllib.request.Request(api_url, headers={"User-Agent": "RecoMart/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    products = payload.get("products", [])
    if not isinstance(products, list) or not products:
        raise ValueError("REST API response did not contain products")

    return [
        {
            "product_id": f"api-{product['id']}",
            "name": product["title"],
            "category": product["category"],
            "price": product["price"],
            "popularity_score": product["rating"],
        }
        for product in products
    ]


def build_manifest(paths: List[Tuple[str, Path]]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {"ingested_at": datetime.now(timezone.utc).isoformat(), "files": []}
    for source, path in paths:
        manifest["files"].append({"source": source, "path": str(path.relative_to(ROOT_DIR))})
    return manifest


def run_ingestion() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting ingestion")

    dataset_dir = RAW_DIR / "public" / "ml-100k" / "ml-100k"
    if dataset_dir.exists() and (dataset_dir / "u.data").exists():
        interactions, products = execute_with_retry(load_movielens_dataset)
        logger.info("Using MovieLens public dataset")
    else:
        interactions = execute_with_retry(load_interactions, RAW_INTERACTIONS_PATH)
        products = execute_with_retry(load_products, RAW_PRODUCTS_PATH)
        logger.info("Using local sample dataset")

    interaction_path = execute_with_retry(write_partitioned_file, interactions, "ratings", "interactions.csv")
    product_path = execute_with_retry(write_partitioned_file, products, "api", "products.json")

    manifest_paths = [("ratings", interaction_path), ("api", product_path)]
    external_product_path = None
    try:
        external_products = execute_with_retry(load_products_from_api)
        external_product_path = execute_with_retry(
            write_partitioned_file,
            external_products,
            "api",
            "external_products.json",
        )
        manifest_paths.append(("external_api", external_product_path))
        logger.info("Fetched %s products from REST API", len(external_products))
    except Exception as exc:
        logger.error("REST API ingestion failed after retries: %s", exc)

    manifest = build_manifest(manifest_paths)
    execute_with_retry(MANIFEST_PATH.write_text, json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Ingestion completed successfully")
    result = {"interactions": interaction_path, "products": product_path, "manifest": MANIFEST_PATH}
    if external_product_path is not None:
        result["external_products"] = external_product_path
    return result


def run_near_real_time_ingestion() -> Dict[str, Path]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting near-real-time REST API ingestion")

    external_products = execute_with_retry(load_products_from_api)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    external_product_path = execute_with_retry(
        write_partitioned_file,
        external_products,
        "near_real_time_api",
        f"external_products_{timestamp}.json",
    )

    manifest = build_manifest([("external_api", external_product_path)])
    execute_with_retry(
        NEAR_REAL_TIME_MANIFEST_PATH.write_text,
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Near-real-time REST API ingestion completed successfully with %s products",
        len(external_products),
    )
    return {
        "external_products": external_product_path,
        "manifest": NEAR_REAL_TIME_MANIFEST_PATH,
    }


if __name__ == "__main__":
    run_ingestion()
