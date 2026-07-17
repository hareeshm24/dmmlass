from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
FEATURE_STORE_VERSIONS_DIR = FEATURES_DIR / "versions"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"
DATABASE_DIR = ROOT_DIR / "database"

RAW_INTERACTIONS_PATH = RAW_DIR / "ratings" / "interactions.csv"
RAW_PRODUCTS_PATH = RAW_DIR / "api" / "products_api.json"
PROCESSED_INTERACTIONS_PATH = PROCESSED_DIR / "prepared_interactions.csv"
FEATURES_PATH = FEATURES_DIR / "feature_table.csv"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_PATH = MODELS_DIR / "recommendation_model.pkl"
FEATURE_STORE_METADATA_PATH = FEATURES_DIR / "feature_store_metadata.json"
VALIDATION_REPORT_JSON = REPORTS_DIR / "validation_report.json"
VALIDATION_REPORT_PDF = REPORTS_DIR / "data_quality_report.pdf"
EDA_PLOTS_DIR = REPORTS_DIR / "plots"


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        FEATURES_DIR,
        FEATURE_STORE_VERSIONS_DIR,
        MODELS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
        EDA_PLOTS_DIR,
        DATABASE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
