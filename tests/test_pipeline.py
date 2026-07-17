import pandas as pd

from ingestion import ingest_data
from validation.validate_data import validate_interactions, validate_products


def test_validate_interactions_accepts_valid_data():
    df = pd.DataFrame(
        {
            "user_id": ["U001", "U002"],
            "product_id": ["P001", "P002"],
            "rating": [5, 4],
            "event_timestamp": ["2026-07-01 10:21:00", "2026-07-02 10:21:00"],
            "purchase_flag": [1, 0],
            "session_id": ["U001-session", "U002-session"],
        }
    )
    report = validate_interactions(df)
    assert report["metrics"]["rows"] == 2
    assert report["issues"] == []


def test_validate_products_detects_missing_category():
    df = pd.DataFrame({"product_id": ["P001"], "category": [None]})
    report = validate_products(df)
    assert any("Missing category values" in issue for issue in report["issues"])


def test_write_partitioned_file_writes_json_for_product_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_data, "RAW_DIR", tmp_path)
    df = pd.DataFrame({"product_id": ["P001"], "category": ["Books"]})

    output_path = ingest_data.write_partitioned_file(df, "api", "products.json")

    assert output_path.exists()
    loaded = pd.read_json(output_path)
    assert loaded["product_id"].tolist() == ["P001"]
