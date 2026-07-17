CREATE TABLE IF NOT EXISTS user_features (
    user_id TEXT PRIMARY KEY,
    user_activity_frequency INTEGER,
    user_avg_rating REAL
);

CREATE TABLE IF NOT EXISTS item_features (
    product_id TEXT PRIMARY KEY,
    item_avg_rating REAL,
    item_popularity INTEGER
);

CREATE TABLE IF NOT EXISTS interaction_features (
    user_id TEXT,
    product_id TEXT,
    rating REAL,
    price_scaled REAL,
    popularity_scaled REAL,
    category_code INTEGER
);
