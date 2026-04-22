from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from NLP.ml_nlp_lab.mcp_service.mcp_client import analyze_credit_application
from NLP.ml_nlp_lab.mcp_service.mcp_server import mcp
from fastmcp import Client
from NLP.ml_nlp_lab.mcp_service.ml.train import train_and_save
from NLP.ml_nlp_lab.mcp_service.ml.evaluate_llm import evaluate

log = logging.getLogger("uvicorn.error")

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:0.5b")

_MODEL_PATH = Path(__file__).parent / "ml" / "model.pkl"
_REPORT_PATH = Path(__file__).parent.parent / "report.md"

app = FastAPI(
    title="Credit Scoring MCP Service",
    description="LLM + Classical ML credit analysis via FastMCP tools",
    version="1.0.0",
)


async def _wait_for_ollama(retries: int = 20, delay: float = 5.0) -> None:
    """wait готовности Ollama API с ретраями"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, retries + 1):
            try:
                r = await client.get(f"{OLLAMA_URL}/api/tags")
                if r.status_code == 200:
                    log.info("Ollama готова.")
                    return
            except Exception:
                pass
            log.info("Ожидание Ollama (%d/%d)...", attempt, retries)
            await asyncio.sleep(delay)
    log.warning("Ollama не ответила в отведённое время.")


async def _ensure_model_pulled() -> None:
    """LLM-модель в Ollama если она ещё не кэширована."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            names = {m["name"] for m in r.json().get("models", [])}
            if MODEL_NAME not in names and f"{MODEL_NAME}:latest" not in names:
                log.info("Загрузка модели %s ...", MODEL_NAME)
                await client.post(f"{OLLAMA_URL}/api/pull", json={"name": MODEL_NAME}, timeout=300.0)
                log.info("Модель %s загружена.", MODEL_NAME)
        except Exception as exc:
            log.warning("Не удалось загрузить модель: %s", exc)


def _train_ml_model() -> None:
    if _MODEL_PATH.exists():
        log.info("ML-модель уже обучена: %s", _MODEL_PATH)
        return
    log.info("Обучение ML-модели (первый запуск ~60 с)...")
    train_and_save()


@app.on_event("startup")
async def startup() -> None:
    await _wait_for_ollama()
    await _ensure_model_pulled()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _train_ml_model)
    log.info("MCP-сервис готов.")


class AnalyzeRequest(BaseModel):

    description: str


class AnalyzeResponse(BaseModel):

    decision: str
    explanation: str
    tools_used: list[dict]
    iterations: int


@app.post("/analyze", response_model=AnalyzeResponse, summary="Анализ кредитной заявки")
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Запустить полный агентный анализ кредитной заявки через LLM + MCP.

    Args:
        request: Тело с полем description.

    Returns:
        Структурированный результат с решением и трейсами тулов
    """
    if not request.description.strip():
        raise HTTPException(status_code=422, detail="Поле description не может быть пустым")
    try:
        result = await analyze_credit_application(request.description)
        return AnalyzeResponse(**result)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Нет соединения с Ollama: {OLLAMA_URL}")
    except Exception as exc:
        log.exception("Ошибка анализа")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/train", summary="Обучить ML-модель")
async def train_model() -> dict:
    """старт обучение на UCI Adult: LR + GradientBoosting, сохранить лучшую модель."""
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, train_and_save)
        return {"status": "ok", "metrics": results}
    except Exception as exc:
        log.exception("Ошибка обучения")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate", summary="Оценить перформанс LLM на UCI Adult")
async def evaluate_llm(n_samples: int = 50) -> dict:
    """Прогнать N примеров из датасета через /analyze и вернуть метрики LLM.

    Args:
        n_samples: Количество примеров для оценки.

    Returns:
        Метрики accuracy, precision, recall, f1.
    """
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, lambda: evaluate(n_samples, "http://127.0.0.1:8000"))
        return {"status": "ok", "metrics": results}
    except Exception as exc:
        log.exception("Ошибка оценки LLM")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/report", response_class=PlainTextResponse, summary="Отчёт о производительности ML")
async def get_report() -> str:
    if not _REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Отчёт не найден. Вызовите POST /train.")
    return _REPORT_PATH.read_text(encoding="utf-8")


@app.get("/tools", summary="Список MCP-инструментов")
async def list_tools() -> dict:
    """Вернуть все инструменты, зарегистрированные на MCP-сервере."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    return {
        "tools": [
            {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
            for t in tools
        ]
    }


@app.get("/health", summary="Проверка работоспособности")
async def health() -> dict:
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok",
        "ollama_reachable": ollama_ok,
        "ml_model_trained": _MODEL_PATH.exists(),
        "model": MODEL_NAME,
    }
