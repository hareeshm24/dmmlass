import json
import os
import sqlite3
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "RecoMart_Assignment_Report.pdf"
PLOTS_DIR = ROOT / "reports" / "plots"
MODEL_METADATA_PATH = ROOT / "data" / "models" / "model_metadata.json"
VALIDATION_REPORT_PATH = ROOT / "reports" / "validation_report.json"
FEATURE_STORE_METADATA_PATH = ROOT / "data" / "features" / "feature_store_metadata.json"
INGESTION_MANIFEST_PATH = ROOT / "data" / "raw" / "ingestion_manifest.json"
NEAR_REAL_TIME_MANIFEST_PATH = ROOT / "data" / "raw" / "near_real_time_ingestion_manifest.json"
DATABASE_PATH = ROOT / "database" / "recomart_features.db"
SCREENSHOTS_DIR = ROOT / "reports" / "screenshots"
TEAM_NAME = os.getenv("RECOMART_TEAM_NAME", "Group 49")
TEAM_MEMBER_DETAILS = os.getenv(
    "RECOMART_TEAM_MEMBER_DETAILS",
    "1. K Sharnika - 2025AB05310;2. Akshay Govindrao Telang - 2025DA04003;3. Debosmita Bose - 2025AB05316;4. Sachin Sharma - 2025DA04006;5. Marati Hareesh Babu - 2025AB05317",
)
SUBMISSION_DATE = os.getenv("RECOMART_SUBMISSION_DATE", date.today().strftime("%d.%m.%Y"))
VIDEO_LINK = os.getenv("RECOMART_VIDEO_LINK", "Not provided")
ZIP_LINK = os.getenv("RECOMART_ZIP_LINK", "Not provided")
REPOSITORY_LINK = os.getenv("RECOMART_REPOSITORY_LINK", "https://github.com/hareeshm24/DMML-Assignment")


def build_report() -> None:
    doc = SimpleDocTemplate(str(REPORT_PATH), pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("End-to-End Data Management Pipeline for a Recommendation System", styles["Title"]))
    story.append(Paragraph("RecoMart Recommendation System", styles["Heading1"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"Team: {TEAM_NAME}", styles["BodyText"]))
    story.append(Paragraph("Team Member Details:", styles["BodyText"]))
    team_members = [member.strip() for member in TEAM_MEMBER_DETAILS.split(";") if member.strip()]
    for member in team_members:
        story.append(Paragraph(member, styles["BodyText"]))
    story.append(Paragraph(f"Submission Date: {SUBMISSION_DATE}", styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("1. Problem Statement", styles["Heading2"]))
    story.append(Paragraph("RecoMart wants to create a scalable recommendation engine that uses user-item interactions and product metadata to provide personalized product suggestions and improve customer engagement and cross-selling opportunities.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("2. Objectives", styles["Heading2"]))
    story.append(Paragraph("- Support batch ingestion from MovieLens files and near-real-time product ingestion from a REST API into timestamp-partitioned raw storage", styles["BodyText"]))
    story.append(Paragraph("- Validate schema, missing values, duplicates, ranges, and timestamp formats", styles["BodyText"]))
    story.append(Paragraph("- Prepare data and generate rating-distribution, popularity, sparsity, and correlation analysis", styles["BodyText"]))
    story.append(Paragraph("- Engineer recommendation features and store them in CSV and SQLite", styles["BodyText"]))
    story.append(Paragraph("- Maintain versioned feature retrieval and DVC-based data lineage", styles["BodyText"]))
    story.append(Paragraph("- Train and evaluate an NMF recommendation model and expose a command-line inference interface", styles["BodyText"]))
    story.append(Paragraph("- Orchestrate the pipeline with Airflow and track experiments with MLflow", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("3. Methodology / Pipeline", styles["Heading2"]))
    story.append(Paragraph("The implemented workflow supports daily batch ingestion and five-minute near-real-time REST API ingestion, followed by validation -> preparation and EDA -> feature engineering and SQLite storage -> versioned feature store -> NMF model training and evaluation -> inference. DVC records stage dependencies and output versions, Airflow orchestrates both ingestion modes, and MLflow records model parameters, metrics, and artifacts.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("4. Implementation Details", styles["Heading2"]))
    story.append(Paragraph("- Ingestion scripts process the MovieLens 100k public dataset for batch interactions and product metadata, fetch supplemental product data from a REST API with retries, and write timestamp-partitioned batch data under data/raw/api and independent micro-batches under data/raw/near_real_time_api with separate manifests.", styles["BodyText"]))
    story.append(Paragraph("- Validation checks verify required schemas, missing values, duplicate rows, rating ranges, timestamps, prices, and popularity values, then generate JSON and PDF data-quality reports.", styles["BodyText"]))
    story.append(Paragraph("- Preparation joins product metadata with interactions, handles missing values, encodes categories, safely standardizes numeric attributes, and generates the required EDA plots.", styles["BodyText"]))
    story.append(Paragraph("- Feature engineering creates user activity frequency, user and item average ratings, and item popularity features, and stores transformed feature tables in SQLite using database/schema.sql.", styles["BodyText"]))
    story.append(Paragraph("- The custom feature store records feature definitions, sources, transformations, available versions, and versioned retrieval paths for training and inference.", styles["BodyText"]))
    story.append(Paragraph("- DVC tracks the pipeline stages, dependencies, generated outputs, and reproducible lineage through dvc.yaml and dvc.lock.", styles["BodyText"]))
    story.append(Paragraph("- The NMF model uses a per-user positive-interaction holdout, excludes previously rated products from recommendations, and reports RMSE, Precision@5, Recall@5, and NDCG@5.", styles["BodyText"]))
    story.append(Paragraph("- Airflow provides the daily recomart_end_to_end DAG and the five-minute recomart_near_real_time_ingestion DAG with validation and failure logs, while MLflow stores the experiment run ID, parameters, metrics, model metadata, model artifact, and feature table.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    try:
        model_meta = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        model_meta = None
    try:
        validation_report = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        validation_report = None
    try:
        feature_store_meta = json.loads(FEATURE_STORE_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        feature_store_meta = None
    try:
        ingestion_manifest = json.loads(INGESTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        ingestion_manifest = None
    try:
        near_real_time_manifest = json.loads(NEAR_REAL_TIME_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        near_real_time_manifest = None

    database_counts = None
    if DATABASE_PATH.exists():
        try:
            with sqlite3.connect(DATABASE_PATH) as connection:
                database_counts = {
                    "user_features": connection.execute("SELECT COUNT(*) FROM user_features").fetchone()[0],
                    "item_features": connection.execute("SELECT COUNT(*) FROM item_features").fetchone()[0],
                    "interaction_features": connection.execute("SELECT COUNT(*) FROM interaction_features").fetchone()[0],
                }
        except Exception:
            database_counts = None

    story.append(Paragraph("5. Results and Outputs", styles["Heading2"]))
    if model_meta:
        metrics_data = [["Metric", "Value"], ["Model", model_meta.get("model")], ["NMF Components", model_meta.get("parameters", {}).get("n_components")], ["RMSE", round(model_meta.get("metrics", {}).get("rmse", 0.0), 4)], ["Precision@5", round(model_meta.get("metrics", {}).get("precision_at_5", 0.0), 4)], ["Recall@5", round(model_meta.get("metrics", {}).get("recall_at_5", 0.0), 4)], ["NDCG@5", round(model_meta.get("metrics", {}).get("ndcg_at_5", 0.0), 4)]]
    else:
        metrics_data = [["Metric", "Value"], ["Model", "N/A"], ["NMF Components", "N/A"], ["RMSE", "N/A"], ["Precision@5", "N/A"], ["Recall@5", "N/A"], ["NDCG@5", "N/A"]]
    table = Table(metrics_data, colWidths=[2.2 * inch, 3.2 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))

    if model_meta and model_meta.get("evaluation"):
        evaluation = model_meta.get("evaluation", {})
        story.append(Paragraph(f"Evaluation method: {evaluation.get('method', 'N/A')}; training rows = {evaluation.get('training_rows', 'N/A')}; test rows = {evaluation.get('test_rows', 'N/A')}.", styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))

    if validation_report:
        validation_data = [["Dataset", "Rows", "Missing", "Duplicates", "Issues"]]
        for dataset in validation_report.get("datasets", []):
            metrics = dataset.get("metrics", {})
            validation_data.append([
                dataset.get("dataset", "N/A"),
                metrics.get("rows", "N/A"),
                metrics.get("missing_values", "N/A"),
                metrics.get("duplicate_rows", "N/A"),
                len(dataset.get("issues", [])),
            ])
        validation_table = Table(validation_data, colWidths=[1.35 * inch, 0.8 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch], repeatRows=1)
        validation_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(Paragraph(f"Validation summary: {validation_report.get('summary', {}).get('datasets_checked', 0)} datasets checked, passed = {validation_report.get('summary', {}).get('passed', False)}", styles["BodyText"]))
        story.append(validation_table)
    else:
        story.append(Paragraph("Validation summary: Not available", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    if ingestion_manifest:
        story.append(Paragraph(f"Latest batch ingestion partition: {ingestion_manifest.get('date_prefix', 'N/A')}; files recorded = {len(ingestion_manifest.get('files', []))}.", styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))

    if near_real_time_manifest:
        story.append(Paragraph(f"Latest near-real-time REST API ingestion: {near_real_time_manifest.get('ingested_at', 'N/A')}; files recorded = {len(near_real_time_manifest.get('files', []))}.", styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))

    if feature_store_meta:
        story.append(Paragraph(f"Feature store version: {feature_store_meta.get('version', 'N/A')}; available versions = {len(feature_store_meta.get('available_versions', []))}; rows = {feature_store_meta.get('rows', 'N/A')}; documented features = {len(feature_store_meta.get('feature_definitions', {}))}.", styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))

    if database_counts:
        database_data = [["SQLite Table", "Rows"]] + [[name, count] for name, count in database_counts.items()]
        database_table = Table(database_data, colWidths=[2.8 * inch, 1.2 * inch], repeatRows=1)
        database_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(Paragraph("Structured feature storage:", styles["BodyText"]))
        story.append(database_table)
        story.append(Spacer(1, 0.15 * inch))

    # Add MLflow and sample recommendations if available
    if model_meta and model_meta.get("mlflow_run_id"):
        story.append(Paragraph("Experiment Tracking", styles["Heading2"]))
        story.append(Paragraph(f"MLflow run id: {model_meta.get('mlflow_run_id')}", styles["BodyText"]))
        story.append(Paragraph(f"Local MLflow database: {ROOT / 'mlflow.db'}", styles["BodyText"]))
        story.append(Paragraph(f"Local artifact path: {ROOT / 'mlruns'}", styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Sample Recommendations (top 5 per user):", styles["Heading3"]))
        # show up to 5 users
        recs = model_meta.get("recommendations", [])
        for item in recs[:5]:
            if isinstance(item, dict):
                uid = item.get("user_id")
                rlist = [str(value) for value in item.get("recommendations", [])]
                story.append(Paragraph(f"- {uid}: {', '.join(rlist[:5])}", styles["BodyText"]))
        story.append(Spacer(1, 0.15 * inch))
    story.append(Spacer(1, 0.15 * inch))

    story.append(PageBreak())
    story.append(Paragraph("6. EDA Plots and Execution Evidence", styles["Heading2"]))
    plot_files = [
        ("Rating distribution", PLOTS_DIR / "rating_distribution.png"),
        ("Feature correlation", PLOTS_DIR / "feature_correlation.png"),
        ("Item popularity", PLOTS_DIR / "item_popularity.png"),
        ("Interaction sparsity", PLOTS_DIR / "interaction_sparsity.png"),
    ]
    plot_count = 0
    for title, plot_path in plot_files:
        if plot_path.exists():
            story.append(Paragraph(title, styles["Heading3"]))
            story.append(Image(str(plot_path), width=4.8 * inch, height=2.7 * inch))
            story.append(Spacer(1, 0.1 * inch))
            plot_count += 1
    if plot_count == 0:
        story.append(Paragraph("EDA plots were not available when this report was generated.", styles["BodyText"]))

    screenshot_files = [
        ("Airflow batch and near-real-time DAG execution", SCREENSHOTS_DIR / "airflow_success.png"),
        ("MLflow experiment run", SCREENSHOTS_DIR / "mlflow_run.png"),
        ("DVC pipeline lineage", SCREENSHOTS_DIR / "dvc_dag.png"),
    ]
    screenshot_count = 0
    for title, screenshot_path in screenshot_files:
        if screenshot_path.exists():
            story.append(Paragraph(title, styles["Heading3"]))
            story.append(Image(str(screenshot_path), width=5.8 * inch, height=3.2 * inch))
            story.append(Spacer(1, 0.1 * inch))
            screenshot_count += 1
    if screenshot_count == 0:
        story.append(Paragraph("Airflow, MLflow, and DVC screenshots were not available when this report was generated. Add airflow_success.png, mlflow_run.png, and dvc_dag.png under reports/screenshots before final submission.", styles["BodyText"]))

    story.append(Paragraph("7. Conclusion and Future Scope", styles["Heading2"]))
    story.append(Paragraph("The pipeline provides a modular and reproducible foundation for recommendation-system data management, including batch and near-real-time REST API ingestion, validation, EDA, structured feature storage, feature versioning, holdout evaluation, inference, orchestration, and experiment tracking. Future work can add Kafka-based event streaming, cloud object storage, richer hybrid recommendation features, automated model promotion, and deployment through a production inference API.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("8. Submission Links", styles["Heading2"]))
    story.append(Paragraph(f"Google Drive Link to Video Walkthrough: {VIDEO_LINK}", styles["BodyText"]))
    story.append(Paragraph(f"Google Drive Link to Deliverables ZIP: {ZIP_LINK}", styles["BodyText"]))
    story.append(Paragraph(f"Repository: {REPOSITORY_LINK}", styles["BodyText"]))

    try:
        doc.build(story)
    except PermissionError:
        alt_path = REPORT_PATH.with_name(REPORT_PATH.stem + "_v2" + REPORT_PATH.suffix)
        alt_doc = SimpleDocTemplate(str(alt_path), pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
        alt_doc.build(story)


if __name__ == "__main__":
    build_report()
