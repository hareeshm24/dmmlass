# RecoMart Recommendation System Assignment Summary

## Problem formulation
The project builds a recommendation pipeline for RecoMart to improve product discovery and customer engagement using interaction and product metadata.

## Pipeline stages
1. Ingestion: reads interactions from CSV and product metadata from JSON.
2. Validation: checks missing values, duplicates, and rating ranges.
3. Preparation: joins and cleans datasets and produces EDA plots.
4. Feature engineering: creates user and item activity features.
5. Feature store: persists versioned feature metadata.
6. Model training: fits an NMF-based recommendation model and records metrics.

## Outputs
- Raw partitions under data/raw/
- Prepared data under data/processed/
- Feature tables under data/features/
- Model artifacts under data/models/
- Validation reports under reports/
