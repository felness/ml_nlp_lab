"""Оценка перформанса LLM (Qwen2.5:0.5B) на размеченных данных UCI Adult.
"""

from __future__ import annotations

import argparse
import time

import httpx
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

_EDUCATION_MAP = {
    "Bachelors": "bachelor's degree", "Masters": "master's degree",
    "Doctorate": "doctorate", "Prof-school": "professional school",
    "Assoc-acdm": "associate degree (academic)", "Assoc-voc": "associate degree (vocational)",
    "Some-college": "some college", "HS-grad": "high school graduate",
    "11th": "11th grade", "10th": "10th grade", "9th": "9th grade",
    "7th-8th": "7th-8th grade", "5th-6th": "5th-6th grade",
    "1st-4th": "1st-4th grade", "Preschool": "no formal education",
}


def _row_to_text(row: pd.Series) -> str:
    """
    Args:
        row: Строка датасета UCI Adult.

    Returns:
        Текстовое описание на английском для передачи в LLM.
    """
    sex = "Male" if str(row.get("sex", "")).strip() == "Male" else "Female"
    age = int(row.get("age", 30))
    education = _EDUCATION_MAP.get(str(row.get("education", "")).strip(), str(row.get("education", "")))
    marital = str(row.get("marital-status", "")).strip()
    occupation = str(row.get("occupation", "")).strip()
    hours = int(row.get("hours-per-week", 40))
    workclass = str(row.get("workclass", "")).strip()
    capital_gain = float(row.get("capital-gain", 0))
    capital_loss = float(row.get("capital-loss", 0))
    country = str(row.get("native-country", "")).strip()
    edu_num = int(row.get("education-num", 9))
    estimated_income = 20_000 + edu_num * 2_500 + max(0, hours - 40) * 500

    return (
        f"{sex}, {age} years old, estimated annual income ${estimated_income:,.0f}, "
        f"education: {education} (level {edu_num}/16), marital status: {marital}, "
        f"occupation: {occupation}, workclass: {workclass}, works {hours} hours/week, "
        f"capital gain: ${capital_gain:.0f}, capital loss: ${capital_loss:.0f}, "
        f"native country: {country}."
    )


def _call_analyze(description: str, base_url: str, timeout: float = 120.0) -> str:
    """Отправить описание на /analyze и вернуть решение LLM.

    Args:
        description: Текстовое описание заявителя.
        base_url: Базовый URL сервиса.
        timeout: Таймаут запроса в секундах.

    Returns:
        'APPROVED', 'REJECTED' или 'UNKNOWN'.
    """
    try:
        resp = httpx.post(f"{base_url}/analyze", json={"description": description}, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("decision", "UNKNOWN")
    except Exception as exc:
        print(f"  [ошибка] {exc}")
        return "UNKNOWN"


def evaluate(n_samples: int = 50, base_url: str = "http://127.0.0.1:8000") -> dict:
    """Оценить перформанс LLM на выборке из тестовых данных UCI Adult.

    Args:
        n_samples: Количество примеров для проверки.
        base_url: Базовый URL MCP-сервиса.

    Returns:
        Словарь с accuracy, precision, recall, f1 и classification_report.
    """
    adult = fetch_ucirepo(id=2)
    X: pd.DataFrame = adult.data.features.copy()
    y_raw: pd.DataFrame = adult.data.targets.copy()

    y_binary = (y_raw.iloc[:, 0].str.strip().isin([">50K", ">50K."])).astype(int)

    X = X.drop(columns=["education"], errors="ignore")
    X = X.replace("?", "Unknown").replace(" ?", "Unknown")

    _, X_test, _, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

    X_sample = X_test.head(n_samples)
    y_sample = y_test.head(n_samples)

    print(f"[evaluate] Оцениваем {n_samples} примеров через {base_url}/analyze...")
    y_pred_labels: list[str] = []
    y_true_labels: list[int] = []

    for i, (idx, row) in enumerate(X_sample.iterrows()):
        true_label = int(y_sample.loc[idx])
        description = _row_to_text(row)
        print(f"  [{i+1}/{n_samples}] истина={'APPROVED' if true_label == 1 else 'REJECTED'}", end=" → ")
        decision = _call_analyze(description, base_url)
        print(decision)
        y_pred_labels.append(decision)
        y_true_labels.append(true_label)
        time.sleep(1)

    n_unknown = y_pred_labels.count("UNKNOWN")
    y_pred_binary = [1 if d == "APPROVED" else 0 for d in y_pred_labels]

    metrics = {
        "accuracy": float(accuracy_score(y_true_labels, y_pred_binary)),
        "precision": float(precision_score(y_true_labels, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_labels, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true_labels, y_pred_binary, zero_division=0)),
        "report": classification_report(
            y_true_labels, y_pred_binary,
            target_names=["<=50K (REJECTED)", ">50K (APPROVED)"],
            zero_division=0,
        ),
    }

    print(f"\n[evaluate] Accuracy={metrics['accuracy']:.4f} Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
    print(f"[evaluate] UNKNOWN: {n_unknown}/{n_samples}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка перформанса LLM на UCI Adult")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000")
    args = parser.parse_args()
    evaluate(n_samples=args.n_samples, base_url=args.url)
