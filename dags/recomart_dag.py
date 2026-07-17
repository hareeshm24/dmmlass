from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.decorators import task

ROOT_START = datetime(2026, 7, 1)

default_args = {
    "owner": "recomart",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="recomart_end_to_end",
    default_args=default_args,
    start_date=ROOT_START,
    schedule_interval="@daily",
    catchup=False,
    tags=["recomart"],
) as dag:

    @task
    def ingest() -> Dict[str, Any]:
        # imports inside tasks to avoid top-level Airflow import issues
        from ingestion.ingest_data import run_ingestion

        out = run_ingestion()
        # return manifest path and file list for visibility in the UI
        return {"manifest": str(out.get("manifest")), "files": {k: str(v) for k, v in out.items()}}

    @task
    def validate(ingest_result: Dict[str, Any]) -> Dict[str, Any]:
        from validation.validate_data import run_validation

        return run_validation()

    @task
    def prepare(validate_result: Dict[str, Any]) -> Dict[str, Any]:
        from preprocessing.prepare_data import prepare_data

        return prepare_data()

    @task
    def features(prepare_result: Dict[str, Any]) -> Dict[str, Any]:
        from feature_engineering.engineer_features import engineer_features

        return engineer_features()

    @task
    def register_features(features_result: Dict[str, Any]) -> Dict[str, Any]:
        from feature_store.feature_store import run_feature_store

        return run_feature_store()

    @task
    def train(register_result: Dict[str, Any]) -> Dict[str, Any]:
        from model.train_model import train_model

        return train_model()

    ingestion = ingest()
    validation = validate(ingestion)
    preparation = prepare(validation)
    engineered = features(preparation)
    fstore = register_features(engineered)
    model = train(fstore)

    ingestion >> validation >> preparation >> engineered >> fstore >> model


with DAG(
    dag_id="recomart_near_real_time_ingestion",
    default_args=default_args,
    start_date=ROOT_START,
    schedule_interval="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["recomart", "near-real-time"],
) as near_real_time_dag:

    @task
    def ingest_external_api() -> Dict[str, Any]:
        from ingestion.ingest_data import run_near_real_time_ingestion

        out = run_near_real_time_ingestion()
        return {"manifest": str(out["manifest"]), "external_products": str(out["external_products"])}

    @task
    def validate_external_api(ingest_result: Dict[str, Any]) -> Dict[str, Any]:
        import pandas as pd

        from validation.validate_data import validate_products

        products = pd.read_json(ingest_result["external_products"])
        report = validate_products(products)
        if report["issues"]:
            raise ValueError("; ".join(report["issues"]))
        return report

    external_api_ingestion = ingest_external_api()
    external_api_validation = validate_external_api(external_api_ingestion)

    external_api_ingestion >> external_api_validation

