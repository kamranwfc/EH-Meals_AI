"""
Popularity prediction module for EH Meals (Option A - practical predictor).

- Loads /mnt/data/meals_data.json
- Builds simple features from current data
- Creates a target proxy that blends order_count and recency
- Trains a RandomForestRegressor to predict "predicted_popularity"
- Persists model to /mnt/data/popularity_model.joblib
- Exposes a convenience function `predict_popularity_scores(meals)` that returns
  a dict {meal_uuid: predicted_score}

This is intentionally simple and robust for production until real historical
order logs are available.
"""
from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime, timezone
import json
import math
import os
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

MODEL_PATH = 'data/popularity_model.joblib'
DATA_PATH = 'data/meals_data.json'


def parse_datetime(dt_str: Optional[str]):
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            if fmt.endswith("%z"):
                return datetime.strptime(dt_str, fmt)
            else:
                return datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except Exception:
        return None


def build_features(df: pd.DataFrame):
    now = datetime.now(timezone.utc)
    # order_count
    df['order_count'] = df['order_count'].fillna(0).astype(float)
    # days since last ordered (large for never-ordered)
    def days_since(x):
        if pd.isna(x) or x is None:
            return 9999.0
        dt = parse_datetime(x)
        if dt is None:
            return 9999.0
        return (now - dt).total_seconds() / 86400.0
    df['days_since_last'] = df['last_ordered'].apply(days_since)
    # diet count
    df['diet_count'] = df['meal_diets'].apply(lambda arr: len(arr) if isinstance(arr, list) else 0)
    # is_featured
    df['is_featured'] = df['is_featured'].fillna(0).astype(int)
    # cafeteria
    df['cafeteria'] = df['cafeteria'].fillna(0).astype(int)
    # meal type mapping (breakfast=0, lunch_dinner=1, add_on_item=2, other=3)
    def map_type(t):
        if not isinstance(t, dict):
            return 3
        name = t.get('name', '').lower()
        if name == 'breakfast':
            return 0
        if name == 'lunch_dinner':
            return 1
        if name == 'add_on_item':
            return 2
        return 3
    df['meal_type_code'] = df['meal_type'].apply(map_type)
    return df


def build_target(df: pd.DataFrame):
    # Proxy target: order_count adjusted by recency multiplier
    # More recent meals are slightly more likely to be popular; this builds a label
    def recency_mult(days):
        if days <= 7:
            return 1.5
        if days <= 30:
            return 1.2
        if days <= 90:
            return 1.05
        return 1.0
    df['recency_mult'] = df['days_since_last'].apply(lambda d: recency_mult(d))
    df['target_popularity'] = df['order_count'] * df['recency_mult']
    # add small smoothing
    df['target_popularity'] = df['target_popularity'] + 1.0
    return df


def train_and_persist_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    with open(DATA_PATH, 'r') as fh:
        raw = json.load(fh)
    df = pd.DataFrame(raw)
    df = build_features(df)
    df = build_target(df)

    features = ['order_count', 'days_since_last', 'diet_count', 'is_featured', 'cafeteria', 'meal_type_code']
    X = df[features].fillna(0)
    y = df['target_popularity']

    if len(df) < 5:
        raise RuntimeError('Not enough meals to train a robust model. Need >=5')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    joblib.dump({'model': model, 'features': features}, MODEL_PATH)
    return {'rmse': rmse, 'trained_on': len(X_train), 'tested_on': len(X_test), 'model_path': MODEL_PATH}


def predict_popularity_scores(meals: List[dict]) -> Dict[str, float]:
    # Loads model from disk and predicts scores for given meal dicts (raw format from JSON)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('Model not trained. Run train_and_persist_model() first.')
    artefact = joblib.load(MODEL_PATH)
    model = artefact['model']
    features = artefact['features']

    df = pd.DataFrame(meals)
    df = build_features(df)
    X = df[features].fillna(0)
    preds = model.predict(X)
    # normalize predictions to 0-1
    minp = float(preds.min())
    maxp = float(preds.max())
    if math.isclose(maxp, minp):
        norm = [0.5 for _ in preds]
    else:
        norm = [(p - minp) / (maxp - minp) for p in preds]
    return {row['uuid']: float(norm[idx]) for idx, row in df.reset_index().iterrows()} 
