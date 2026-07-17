import logging
from pathlib import Path
from typing import Dict, Any

from feature_engineering.engineer_features import engineer_features
from feature_store.feature_store import run_feature_store
from ingestion.ingest_data import run_ingestion
from model.train_model import train_model
from preprocessing.prepare_data import prepare_data
from validation.validate_data import run_validation
from project_config import LOGS_DIR, ensure_project_dirs


def run_pipeline() -> Dict[str, Any]:
    ensure_project_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOGS_DIR / "pipeline.log"), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info("Starting end-to-end pipeline")
    ingestion_output = run_ingestion()
    validation_output = run_validation()
    preprocessing_output = prepare_data()
    features_output = engineer_features()
    feature_store_output = run_feature_store()
    model_output = train_model()
    logger.info("Pipeline completed")
    return {
        "ingestion": ingestion_output,
        "validation": validation_output,
        "preprocessing": preprocessing_output,
        "features": features_output,
        "feature_store": feature_store_output,
        "model": model_output,
    }


if __name__ == "__main__":
    run_pipeline()
