# RecoMart Demo Walkthrough

## Duration
5-10 minutes

## Suggested structure
1. Introduce the project goal and business value.
   - RecoMart uses customer-product interactions to generate personalized recommendations.
   - The pipeline supports reproducible data processing, model training, and experiment tracking.

2. Show the repository structure and explain the modular pipeline.
   - `ingestion/`
   - `validation/`
   - `preprocessing/`
   - `feature_engineering/`
   - `feature_store/`
   - `model/`
   - `dags/`
   - `reports/`

3. Demonstrate batch and near-real-time ingestion.
   ```bash
   python -m ingestion.ingest_data
   python -c "from ingestion.ingest_data import run_near_real_time_ingestion; print(run_near_real_time_ingestion())"
   ```
   Show:
   - batch data under `data/raw/ratings/` and `data/raw/api/`
   - REST API micro-batches under `data/raw/near_real_time_api/`
   - `data/raw/ingestion_manifest.json`
   - `data/raw/near_real_time_ingestion_manifest.json`

4. Run the complete pipeline locally using:
   ```bash
   python -m orchestration.pipeline
   ```
   Alternatively, reproduce the versioned pipeline using:
   ```bash
   python -m dvc repro
   ```

5. Highlight the key outputs:
   - raw data in `data/raw/`
   - prepared data in `data/processed/`
   - feature table and versions in `data/features/`
   - SQLite feature database in `database/recomart_features.db`
   - validation reports and EDA plots in `reports/`
   - trained model and metadata in `data/models/`

6. Demonstrate feature-store retrieval and inference.
   ```bash
   python -c "from feature_store.feature_store import FeatureStore; print(FeatureStore().get_feature_vector('1', '1'))"
   python -m model.inference --user-id 1 --k 5
   ```

7. Explain orchestration, lineage, and experiment tracking.
   - Show the successful `recomart_end_to_end` Airflow DAG.
   - Show the `recomart_near_real_time_ingestion` DAG scheduled every five minutes.
   - Show the DVC lineage using:
     ```bash
     python -m dvc dag
     python -m dvc status
     ```
   - Show the MLflow run ID, parameters, metrics, and artifacts.

8. Present the model results.
   - RMSE
   - Precision@5
   - Recall@5
   - NDCG@5
   - Explain that evaluation uses a per-user holdout and excludes previously rated products.

9. Conclude with limitations and future scope.
   - Kafka-based event streaming
   - cloud object storage
   - hybrid recommendation models
   - automated model deployment and monitoring

## Talking points
- The pipeline covers batch and near-real-time ingestion, validation, preprocessing, EDA, feature engineering, structured database storage, feature-store versioning, model training, inference, and orchestration.
- Batch interactions use the MovieLens 100k dataset, while supplemental product data is fetched from a REST API with retry and logging.
- The model uses an NMF-based collaborative filtering approach.
- Validation checks schema, missing values, duplicates, ranges, timestamps, prices, and popularity values.
- The feature store records feature sources, transformations, and available versions.
- DVC records pipeline dependencies, output versions, and reproducible lineage.
- MLflow records parameters, metrics, the model artifact, model metadata, and the feature table.

## Expected evidence
- Successful Airflow DAG status for `recomart_end_to_end`.
- Successful Airflow task status for `recomart_near_real_time_ingestion`.
- MLflow experiment page showing the run ID, parameters, metrics, and artifacts.
- DVC DAG output and `Data and pipelines are up to date` status.
- Generated plots in `reports/plots/`.
- Validation report PDF in `reports/`.
- Feature-store metadata and version folders in `data/features/`.
- SQLite row counts from `database/recomart_features.db`.
- Model metadata JSON and trained model in `data/models/`.
- Successful recommendation output from `model.inference`.

## Final recording checklist
- Keep the recording between 5 and 10 minutes.
- Use readable terminal and browser zoom levels.
- Do not expose passwords, tokens, or private links.
- Show the complete end-to-end flow rather than only source code.
- Confirm the Google Drive video link is accessible to any BITS ID.
