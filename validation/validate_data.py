import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from project_config import (
    RAW_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    VALIDATION_REPORT_JSON,
    VALIDATION_REPORT_PDF,
    ensure_project_dirs,
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_latest_raw_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = RAW_DIR / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interactions_path = ROOT_DIR / manifest["files"][0]["path"]
    products_path = ROOT_DIR / manifest["files"][1]["path"]
    interactions = pd.read_csv(interactions_path)
    products = pd.read_json(products_path)
    return interactions, products


def validate_interactions(df: pd.DataFrame) -> Dict[str, Any]:
    issues: List[str] = []
    required_columns = {
        "user_id",
        "product_id",
        "rating",
        "event_timestamp",
        "purchase_flag",
        "session_id",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        issues.append(f"Schema mismatch - missing interaction columns: {', '.join(missing_columns)}")

    parsed_timestamp = pd.to_datetime(df["event_timestamp"], errors="coerce") if "event_timestamp" in df.columns else pd.Series(pd.NaT, index=df.index)
    rating_numeric = pd.to_numeric(df["rating"], errors="coerce") if "rating" in df.columns else pd.Series(dtype=float)
    metrics = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_rating_values": int((rating_numeric.isna() | ~rating_numeric.between(1, 5)).sum()) if "rating" in df.columns else int(len(df)),
        "invalid_timestamps": int(parsed_timestamp.isna().sum()) if "event_timestamp" in df.columns else int(len(df)),
    }

    if metrics["missing_values"] > 0:
        issues.append("Missing values detected in interactions data")
    if metrics["duplicate_rows"] > 0:
        issues.append("Duplicate interaction rows detected")
    if "rating" in df.columns and rating_numeric.between(1, 5).all():
        metrics["rating_range_ok"] = True
    else:
        issues.append("Ratings outside allowed range 1-5")
        metrics["rating_range_ok"] = False
    if "user_id" in df.columns and df["user_id"].isna().any():
        issues.append("Missing user_id values")
    if "product_id" in df.columns and df["product_id"].isna().any():
        issues.append("Missing product_id values")
    if "event_timestamp" in df.columns and parsed_timestamp.isna().any():
        issues.append("Invalid or missing event_timestamp values")
    return {"dataset": "interactions", "metrics": metrics, "issues": issues}


def validate_products(df: pd.DataFrame) -> Dict[str, Any]:
    issues: List[str] = []
    required_columns = {"product_id", "name", "category", "price", "popularity_score"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        issues.append(f"Schema mismatch - missing product columns: {', '.join(missing_columns)}")

    price_numeric = pd.to_numeric(df.get("price"), errors="coerce") if "price" in df.columns else pd.Series(dtype=float)
    popularity_numeric = pd.to_numeric(df.get("popularity_score"), errors="coerce") if "popularity_score" in df.columns else pd.Series(dtype=float)
    invalid_price_values = price_numeric.isna() | price_numeric.le(0)
    invalid_popularity_values = popularity_numeric.isna() | ~popularity_numeric.between(0, 5)
    metrics = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_price_values": int(invalid_price_values.sum()) if "price" in df.columns else int(len(df)),
        "invalid_popularity_values": int(invalid_popularity_values.sum()) if "popularity_score" in df.columns else int(len(df)),
    }
    if metrics["missing_values"] > 0:
        issues.append("Missing values detected in products data")
    if metrics["duplicate_rows"] > 0:
        issues.append("Duplicate product rows detected")
    if "product_id" in df.columns and df["product_id"].isna().any():
        issues.append("Missing product_id values")
    if "category" in df.columns and df["category"].isna().any():
        issues.append("Missing category values")
    if "price" in df.columns and invalid_price_values.any():
        issues.append("Invalid price values")
    if "popularity_score" in df.columns and invalid_popularity_values.any():
        issues.append("Invalid popularity_score values")
    return {"dataset": "products", "metrics": metrics, "issues": issues}


def generate_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    report = {"summary": {"datasets_checked": len(results), "passed": all(not item["issues"] for item in results)}}
    report["datasets"] = results
    return report


def write_pdf(report: Dict[str, Any]) -> None:
    document = SimpleDocTemplate(str(VALIDATION_REPORT_PDF), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("RecoMart Data Quality Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Summary: {report['summary']}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    for dataset in report["datasets"]:
        story.append(Paragraph(dataset["dataset"].capitalize(), styles["Heading2"]))
        rows = [["Metric", "Value"]] + [[key.replace("_", " ").title(), str(value)] for key, value in dataset["metrics"].items() if key != "columns"]
        table = Table(rows, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(table)
        story.append(Spacer(1, 12))
        if dataset["issues"]:
            story.append(Paragraph("Issues: " + "; ".join(dataset["issues"]), styles["BodyText"]))
        else:
            story.append(Paragraph("Issues: None", styles["BodyText"]))
        story.append(Spacer(1, 12))
    document.build(story)


def run_validation() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting validation")
    interactions, products = load_latest_raw_files()
    results = [validate_interactions(interactions), validate_products(products)]
    report = generate_report(results)
    VALIDATION_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_pdf(report)
    if not report["summary"]["passed"]:
        logger.error("Validation failed")
        raise ValueError("Data validation failed. See reports/validation_report.json for details.")
    logger.info("Validation completed")
    return report


if __name__ == "__main__":
    run_validation()
