# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "time_signature",
    "valence"
]

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_and_select(df):
    df = df[NUMERIC_FEATURES + ["popularity"]].copy()
    df = df.fillna(0)  # replaces dropna() to avoid empty rows
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def make_target(df, threshold=60):
    df = df.copy()
    df["target"] = (df["popularity"] > threshold).astype(int)
    return df

def split_and_scale(df, test_size=0.2, random_state=42):
    X = df[NUMERIC_FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler
