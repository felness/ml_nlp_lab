"""Обучение ML-моделей кредитного скоринга на датасете UCI Adult."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

_HERE = Path(__file__).parent
MODEL_PATH = _HERE / "model.pkl"
REPORT_PATH = _HERE.parent.parent / "report.md"

NUMERICAL_FEATURES = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_FEATURES = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "report": classification_report(y_test, y_pred, target_names=["<=50K (rejected)", ">50K (approved)"]),
    }


def train_and_save() -> dict[str, dict]:
    """
    Returns:
        Словарь имя_модели -> метрики.
    """
    adult = fetch_ucirepo(id=2)
    X: pd.DataFrame = adult.data.features.copy()
    y_raw: pd.DataFrame = adult.data.targets.copy()

    y: pd.Series = (y_raw.iloc[:, 0].str.strip().isin([">50K", ">50K."])).astype(int)

    X = X.drop(columns=["education"], errors="ignore")
    X = X.replace("?", "Unknown").replace(" ?", "Unknown")

    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[train] Обучающая: {len(X_train)}, Тестовая: {len(X_test)}")

    candidates = {
        "LogisticRegression": Pipeline([
            ("pre", _build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("pre", _build_preprocessor()),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)),
        ]),
    }

    results: dict[str, dict] = {}
    best_pipeline: Pipeline | None = None
    best_f1 = -1.0
    best_name = ""

    for name, pipeline in candidates.items():
        print(f"[train] Обучаем {name}...")
        pipeline.fit(X_train, y_train)
        metrics = _evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        print(f"[train] {name}: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_pipeline = pipeline
            best_name = name

    joblib.dump({"model": best_pipeline, "model_name": best_name, "feature_names": NUMERICAL_FEATURES + CATEGORICAL_FEATURES}, MODEL_PATH)
    print(f"[train] Сохранена модель {best_name} → {MODEL_PATH}")
    return results


if __name__ == "__main__":
    train_and_save()
