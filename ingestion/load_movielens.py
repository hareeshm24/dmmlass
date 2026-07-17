import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import RAW_DIR, RAW_INTERACTIONS_PATH, RAW_PRODUCTS_PATH, ensure_project_dirs


def load_movielens_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    RAW_INTERACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PRODUCTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = RAW_DIR / "public" / "ml-100k" / "ml-100k"
    ratings_path = dataset_dir / "u.data"
    items_path = dataset_dir / "u.item"

    ratings_df = pd.read_csv(
        ratings_path,
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        header=None,
    )
    items_df = pd.read_csv(
        items_path,
        sep='|',
        encoding='latin-1',
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western'],
        header=None,
    )

    ratings_df['product_id'] = ratings_df['item_id'].astype(str)
    ratings_df['user_id'] = ratings_df['user_id'].astype(str)
    ratings_df['event_timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_df['purchase_flag'] = 1
    ratings_df['session_id'] = ratings_df['user_id'] + '-session'
    ratings_df = ratings_df[['user_id', 'product_id', 'rating', 'event_timestamp', 'purchase_flag', 'session_id']]

    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']
    items_df['category'] = items_df[genre_columns].idxmax(axis=1).replace('unknown', 'Unknown')
    item_stats = ratings_df.groupby('product_id', as_index=False).agg(
        rating_count=('rating', 'count'),
        average_rating=('rating', 'mean'),
    )
    max_rating_count = item_stats['rating_count'].max()
    item_stats['popularity_score'] = (
        0.7 * (item_stats['rating_count'] / max_rating_count) * 5
        + 0.3 * item_stats['average_rating']
    ).round(2)

    products_df = pd.DataFrame({
        'product_id': items_df['movie_id'].astype(str),
        'name': items_df['title'],
        'category': items_df['category'],
        'price': (4.99 + (items_df['movie_id'] % 20) * 0.75 + items_df[genre_columns].sum(axis=1) * 0.25).round(2),
    })
    products_df = products_df.merge(
        item_stats[['product_id', 'popularity_score']],
        on='product_id',
        how='left',
    )
    products_df['popularity_score'] = products_df['popularity_score'].fillna(0.0)

    ratings_df.to_csv(RAW_INTERACTIONS_PATH, index=False)
    products_df.to_json(RAW_PRODUCTS_PATH, orient='records', indent=2)
    return ratings_df, products_df


if __name__ == "__main__":
    load_movielens_dataset()
