# RecoMart Recommendation System

This repository contains a modular end-to-end recommendation pipeline for RecoMart. It includes:

- batch data ingestion from the MovieLens 100k public dataset and REST API product ingestion
- validation with automated quality checks and PDF reporting
- preprocessing and EDA plot generation
- feature engineering for collaborative-style recommendations
- a simple feature store with metadata
- model training using NMF and experiment metadata
- daily batch orchestration and five-minute near-real-time REST API ingestion with Airflow

## 1) Python environment setup

Recommended Python versions:
- `3.10` or `3.11` for full local stack (including Airflow package import)
- `3.12+` can run core pipeline, tests, DVC, and MLflow parts, but Airflow should be run via Docker Compose

From project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:
- `requirements.txt` installs Airflow only on supported Python versions (`<3.12`).
- If Airflow is skipped due to Python version, use Docker for Airflow UI/DAG execution.

## 2) Download public dataset and verify REST API access

```bash
python -m ingestion.download_public_dataset
```

Expected dataset location:
- `data/raw/public/ml-100k/ml-100k/u.data`
- `data/raw/public/ml-100k/ml-100k/u.item`

Run batch ingestion, including the supplemental REST API fetch:

```bash
python -m ingestion.ingest_data
```

Run one near-real-time REST API ingestion cycle locally:

```bash
python -c "from ingestion.ingest_data import run_near_real_time_ingestion; print(run_near_real_time_ingestion())"
```

## 3) Run the full pipeline

```bash
python -m orchestration.pipeline
```

## 4) Validate outputs

Check logs:
- `logs/ingestion.log`
- `logs/pipeline.log`

Check data artifacts:
- `data/raw/ingestion_manifest.json` (latest batch partition paths)
- `data/raw/near_real_time_api/<year>/<month>/<day>/external_products_<timestamp>.json`
- `data/raw/near_real_time_ingestion_manifest.json` (latest REST API micro-batch)
- `data/processed/prepared_interactions.csv`
- `data/features/feature_table.csv`
- `data/features/feature_store_metadata.json`
- `data/models/model_metadata.json`
- `data/models/recommendation_model.pkl`

Check reports:
- `reports/validation_report.json`
- `reports/data_quality_report.pdf`
- `reports/plots/rating_distribution.png`
- `reports/plots/feature_correlation.png`
- `reports/plots/item_popularity.png`
- `reports/plots/interaction_sparsity.png`

Run tests:

```bash
python -m pytest -q
```

Validate DVC stage reproducibility:

```bash
python -m dvc repro
```

Optional Airflow import check (when using Python 3.10/3.11):

```bash
python -c "import airflow; print('airflow_import_ok')"
```

## 5) Inference interface

After training, run batch inference for a user via:

```bash
python -m model.inference --user-id 1 --k 5
```

This returns top-k recommended product ids in JSON format.

## Key outputs

- Raw data partitions in data/raw/
- Prepared datasets in data/processed/
- Engineered features in data/features/
- Model artifacts in data/models/
- Logs in logs/
- Reports in reports/

## Local Airflow & MLflow (quick start)

Start a local Airflow UI and MLflow server using Docker Compose (this uses `airflow standalone` for simplicity):

```bash
cd RecoMart-Recommendation-System
docker-compose up --build
```

- Airflow UI: http://localhost:8080 — use `recomart_end_to_end` for the daily batch pipeline and `recomart_near_real_time_ingestion` for REST API ingestion every five minutes.
- MLflow UI: http://localhost:5000 — view experiments and logged artifacts.

Notes:
- The DAG file is located at `dags/recomart_dag.py`. It defines the daily end-to-end pipeline and a separate five-minute REST API ingestion and validation DAG.
- The ingestion task returns a manifest path and file list to XCom so you can inspect which datasets were used for that run.

## Data versioning and lineage workflow

DVC tracks the raw data, validation reports, prepared data, engineered features, versioned feature-store files, SQLite feature database, and trained model artifacts declared in `dvc.yaml`.

Run or reproduce the complete versioned pipeline:

```bash
python -m dvc repro
```

Inspect stage lineage and changed artifacts:

```bash
python -m dvc dag
python -m dvc status
```

Commit the pipeline definition and generated lock-file hashes with Git:

```bash
git add .dvc/config dvc.yaml dvc.lock README.md
git commit -m "Track RecoMart data pipeline with DVC"
```

When data or transformation code changes, rerun `python -m dvc repro` and commit the updated `dvc.lock`. This records the data source, pipeline stage dependencies, output hashes, and transformation lineage for each reproducible version.
