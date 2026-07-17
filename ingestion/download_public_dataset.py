import sys
import time
import urllib.request
import zipfile
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import RAW_DIR, ensure_project_dirs

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
TARGET_DIR = RAW_DIR / "public" / "ml-100k"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _download_with_retry(archive_path: Path, retries: int = 3, backoff_seconds: int = 2) -> None:
    logger = logging.getLogger(__name__)
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading MovieLens archive (attempt %s/%s)", attempt, retries)
            urllib.request.urlretrieve(DATASET_URL, archive_path)
            return
        except Exception as exc:
            logger.warning("Download attempt %s failed: %s", attempt, exc)
            if attempt == retries:
                raise
            time.sleep(backoff_seconds * attempt)


def _extract_with_retry(archive_path: Path, retries: int = 3, backoff_seconds: int = 2) -> None:
    logger = logging.getLogger(__name__)
    for attempt in range(1, retries + 1):
        try:
            logger.info("Extracting MovieLens archive (attempt %s/%s)", attempt, retries)
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(TARGET_DIR)
            return
        except Exception as exc:
            logger.warning("Extract attempt %s failed: %s", attempt, exc)
            if attempt == retries:
                raise
            time.sleep(backoff_seconds * attempt)


def download_movie_lens_100k() -> Path:
    setup_logging()
    logger = logging.getLogger(__name__)
    ensure_project_dirs()
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TARGET_DIR / "ml-100k.zip"

    if not archive_path.exists():
        _download_with_retry(archive_path)
    else:
        logger.info("Using existing archive at %s", archive_path)

    if not (TARGET_DIR / "u.data").exists():
        _extract_with_retry(archive_path)
    else:
        logger.info("MovieLens dataset already extracted at %s", TARGET_DIR)

    return TARGET_DIR


if __name__ == "__main__":
    download_movie_lens_100k()
