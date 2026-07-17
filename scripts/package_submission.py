import shutil
from pathlib import Path
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = ROOT / "reports" / "generate_submission_report.py"
OUTPUT_ZIP = ROOT / "RecoMart_Submission.zip"


def build_pdf() -> None:
    # run the report builder
    try:
        sys.path.insert(0, str(ROOT))
        from reports.generate_submission_report import build_report

        build_report()
    except Exception as e:
        print(f"Warning: failed to build PDF report: {e}")


def collect_files(zip_path: Path) -> None:
    include = [
        ".dvc",
        ".dvcignore",
        ".gitignore",
        "README.md",
        "DM4ML-Assignment(S2-25).pdf",
        "dags",
        "data/raw/ratings",
        "data/raw/api",
        "data/raw/near_real_time_api",
        "data/raw/ingestion_manifest.json",
        "data/raw/near_real_time_ingestion_manifest.json",
        "data/processed",
        "data/features",
        "data/models",
        "database",
        "feature_engineering",
        "feature_store",
        "ingestion",
        "model",
        "orchestration",
        "preprocessing",
        "reports",
        "scripts",
        "tests",
        "validation",
        "docker-compose.yml",
        "dvc.lock",
        "dvc.yaml",
        "main.py",
        "mlflow.db",
        "mlruns",
        "project_config.py",
        "requirements.txt",
    ]

    excluded_names = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv"}
    excluded_suffixes = {".pyc", ".pyo"}

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in include:
            p = ROOT / rel
            if not p.exists():
                print(f"Skipping missing file: {rel}")
                continue

            paths = [p] if p.is_file() else sorted(path for path in p.rglob("*") if path.is_file())
            for file_path in paths:
                if file_path.resolve() == zip_path.resolve():
                    continue
                relative_path = file_path.relative_to(ROOT)
                if any(part in excluded_names for part in relative_path.parts):
                    continue
                if file_path.suffix in excluded_suffixes:
                    continue
                z.write(file_path, arcname=str(relative_path))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    print("Building PDF report...")
    build_pdf()
    print(f"Creating submission ZIP at {OUTPUT_ZIP}")
    collect_files(OUTPUT_ZIP)
    print("Done. Verify RecoMart_Submission.zip in repository root.")


if __name__ == "__main__":
    main()
