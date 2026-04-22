from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("credit-scoring-server")

_MODEL_PATH = Path(__file__).parent / "ml" / "model.pkl"


@mcp.tool()
def calculate_credit_score(
    age: int,
    annual_income: float,
    debt_ratio: float,
    employment_years: int,
    num_credit_lines: int = 5,
) -> dict:
    """Рассчитать кредитный скор (300–850) по финансовым данным заявителя.

    Args:
        age: Возраст в годах.
        annual_income: Годовой доход в USD.
        debt_ratio: Соотношение долга к доходу [0.0, 1.0].
        employment_years: Лет непрерывного стажа.
        num_credit_lines: Число открытых кредитных линий.

    Returns:
        Словарь с score (300–850) и rating (poor/fair/good/excellent).
    """
    score = 300

    if 25 <= age <= 60:
        score += 100
    elif age < 25:
        score += 50
    else:
        score += 75

    if annual_income >= 100_000:
        score += 200
    elif annual_income >= 50_000:
        score += 150
    elif annual_income >= 30_000:
        score += 100
    else:
        score += 50

    if debt_ratio <= 0.2:
        score += 150
    elif debt_ratio <= 0.4:
        score += 100
    elif debt_ratio <= 0.6:
        score += 50

    score += min(employment_years * 10, 100)

    if 3 <= num_credit_lines <= 7:
        score += 100
    elif num_credit_lines < 3:
        score += 50
    else:
        score += 75

    score = min(score, 850)

    if score >= 700:
        rating = "excellent"
    elif score >= 650:
        rating = "good"
    elif score >= 600:
        rating = "fair"
    else:
        rating = "poor"

    return {"score": score, "rating": rating}


@mcp.tool()
def assess_risk_by_metadata(
    education_level: str,
    marital_status: str,
    occupation: str,
    hours_per_week: int,
) -> dict:
    """Оценить кредитный риск по персональным метаданным заявителя.

    Args:
        education_level: Уровень образования (bachelors, masters, hs-grad и др.).
        marital_status: Семейное положение (married, single, divorced и др.).
        occupation: Профессия (exec-managerial, sales, other-service и др.).
        hours_per_week: Рабочих часов в неделю.

    Returns:
        Словарь с risk_level (low/medium/high), risk_score и деталями.
    """
    education_weights: dict[str, int] = {
        "doctorate": -3, "prof-school": -3, "masters": -2, "bachelors": -1,
        "assoc-voc": 0, "assoc-acdm": 0, "some-college": 0,
        "hs-grad": 1, "12th": 1, "11th": 2, "10th": 2,
        "9th": 2, "7th-8th": 3, "5th-6th": 3, "1st-4th": 3, "preschool": 3,
    }
    marital_weights: dict[str, int] = {
        "married-civ-spouse": -2, "married-af-spouse": -2,
        "divorced": 1, "never-married": 0, "separated": 1,
        "widowed": 0, "married-spouse-absent": 0,
        "married": -2, "single": 0,
    }
    occupation_weights: dict[str, int] = {
        "exec-managerial": -2, "prof-specialty": -2, "tech-support": -1,
        "protective-serv": -1, "armed-forces": -1, "sales": 0,
        "craft-repair": 0, "transport-moving": 1, "other-service": 2,
        "handlers-cleaners": 2, "farming-fishing": 2,
        "machine-op-inspct": 1, "priv-house-serv": 2,
    }

    def _normalize(s: str) -> str:
        return s.lower().strip().replace(" ", "-")

    risk_score = 0
    risk_score += education_weights.get(_normalize(education_level), 0)
    risk_score += marital_weights.get(_normalize(marital_status), 0)
    risk_score += occupation_weights.get(_normalize(occupation), 0)

    if hours_per_week >= 40:
        risk_score -= 1
    elif hours_per_week < 20:
        risk_score += 2

    if risk_score <= -2:
        level = "low"
    elif risk_score <= 1:
        level = "medium"
    else:
        level = "high"

    return {
        "risk_level": level,
        "risk_score": risk_score,
        "details": {
            "education": education_level,
            "marital_status": marital_status,
            "occupation": occupation,
            "hours_per_week": hours_per_week,
        },
    }


@mcp.tool()
def predict_creditworthiness_ml(
    age: int,
    education_num: int,
    hours_per_week: int,
    capital_gain: float = 0.0,
    capital_loss: float = 0.0,
    workclass: str = "Private",
    marital_status: str = "Never-married",
    occupation: str = "Other-service",
    relationship: str = "Not-in-family",
    sex: str = "Male",
    race: str = "White",
    native_country: str = "United-States",
    fnlwgt: int = 100_000,
) -> dict:
    """Предсказать кредитоспособность

    Args:
        age: Возраст в годах.
        education_num: Уровень образования числом 1–16.
        hours_per_week: Рабочих часов в неделю.
        capital_gain: Прирост капитала в USD.
        capital_loss: Убыток капитала в USD.
        workclass: Тип занятости.
        marital_status: Семейное положение.
        occupation: Профессия.
        relationship: Тип семейных отношений.
        sex: Male или Female.
        race: Раса по данным переписи.
        native_country: Страна происхождения.
        fnlwgt: Вес переписи

    Returns:
        Словарь с prediction (approved/rejected), вероятностями и confidence.
    """
    if not _MODEL_PATH.exists():
        return {"error": "ML-модель не обучена.", "prediction": "unknown"}

    model_data = joblib.load(_MODEL_PATH)
    model = model_data["model"]

    row = pd.DataFrame([{
        "age": age, "fnlwgt": fnlwgt, "education-num": education_num,
        "capital-gain": capital_gain, "capital-loss": capital_loss,
        "hours-per-week": hours_per_week, "workclass": workclass,
        "marital-status": marital_status, "occupation": occupation,
        "relationship": relationship, "race": race,
        "sex": sex, "native-country": native_country,
    }])

    proba = model.predict_proba(row)[0]
    prediction = model.predict(row)[0]

    return {
        "prediction": "approved" if int(prediction) == 1 else "rejected",
        "probability_approved": round(float(proba[1]), 4),
        "probability_rejected": round(float(proba[0]), 4),
        "confidence": round(float(max(proba)), 4),
        "model_used": model_data.get("model_name", "unknown"),
    }
