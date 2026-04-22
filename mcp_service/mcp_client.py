from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from fastmcp import Client

from NLP.ml_nlp_lab.mcp_service.mcp_server import mcp

log = logging.getLogger(__name__)

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
MAX_ITERATIONS: int = 8

SYSTEM_PROMPT = """\
Ты кредитный аналитик банка. Твоя задача — оценить заявку на кредит.

ПРАВИЛА — соблюдай строго:
1. Вызови инструмент `calculate_credit_score` с финансовыми данными заявителя.
2. Вызови инструмент `assess_risk_by_metadata` с данными об образовании, \
семейном положении, профессии и рабочих часах в неделю.
3. После получения результатов инструментов напиши финальный ответ на русском языке.
4. Финальный ответ ОБЯЗАТЕЛЬНО должен содержать одну из двух строк:
   РЕШЕНИЕ: APPROVED
   РЕШЕНИЕ: REJECTED
   Скор ниже 600 или уровень риска "high" означает REJECTED. Иначе — APPROVED.
5. Не запрашивай дополнительную информацию. Принимай решение на основе имеющихся данных.\
"""


async def _get_ollama_tools() -> list[dict]:
    """Получить список инструментов MCP и привести к формату Ollama."""
    async with Client(mcp) as client:
        tools = await client.list_tools()

    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in tools
    ]


async def _call_mcp_tool(tool_name: str, tool_args: dict) -> Any:
    """Выполнить инструмент на MCP-сервере и вернуть результат.

    Args:
        tool_name: Имя инструмента.
        tool_args: Аргументы вызова.

    Returns:
        Распарсенный JSON, текст или словарь с ошибкой.
    """
    try:
        async with Client(mcp) as client:
            content_list = await client.call_tool(tool_name, tool_args)

        if content_list:
            raw = content_list[0]
            text = getattr(raw, "text", str(raw))
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return {"error": "пустой ответ инструмента"}
    except Exception as exc:
        log.error("Инструмент %s завершился с ошибкой: %s", tool_name, exc)
        return {"error": str(exc)}


def _extract_decision(text: str) -> str:

    upper = text.upper()

    if "РЕШЕНИЕ: APPROVED" in upper or "DECISION: APPROVED" in upper:
        return "APPROVED"
    if "РЕШЕНИЕ: REJECTED" in upper or "DECISION: REJECTED" in upper:
        return "REJECTED"

    reject_signals = [
        "REJECTED", "DENIED", "DECLINED", "NOT APPROVED",
        "HIGH RISK", "POOR CREDIT", "ОТКАЗАНО", "ОТКАЗАТЬ",
        "ВЫСОКИЙ РИСК", "ПЛОХОЙ КРЕДИТ", "НЕ ПОДХОДИТ",
        "НЕ РЕКОМЕНДУЕМ", "ОТКЛОНИТЬ", "ОТКЛОНЕНА",
    ]
    approve_signals = [
        "APPROVED", "CREDITWORTHY", "EXCELLENT CREDIT",
        "ОДОБРЕНО", "ОДОБРИТЬ", "КРЕДИТОСПОСОБЕН",
        "ПОДХОДИТ ДЛЯ КРЕДИТОВАНИЯ", "ПОДХОДИТ ДЛЯ",
        "МОЖЕТ БЫТЬ ВЫДАН", "РЕКОМЕНДУЕМ ОДОБРИТЬ",
        "ЗАЯВКА ОДОБРЕНА", "ЗАЯВКА ПОДХОДИТ",
    ]

    for signal in reject_signals:
        if signal in upper:
            return "REJECTED"
    for signal in approve_signals:
        if signal in upper:
            return "APPROVED"

    return "UNKNOWN"


async def analyze_credit_application(description: str) -> dict:
    """Запустить агентный цикл анализа кредитной заявки.
    Returns:
        Словарь с decision, explanation, tools_used, iterations.
    """
    tools = await _get_ollama_tools()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Проанализируй следующую кредитную заявку:\n\n{description}"},
    ]

    tools_used: list[dict] = []

    async with httpx.AsyncClient(timeout=180.0) as http:
        for iteration in range(1, MAX_ITERATIONS + 1):
            log.info("Итерация LLM %d/%d", iteration, MAX_ITERATIONS)

            resp = await http.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": MODEL_NAME, "messages": messages, "tools": tools, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()

            message: dict = data.get("message", {})
            tool_calls: list[dict] = message.get("tool_calls") or []

            if not tool_calls:
                final_text: str = message.get("content", "")
                return {
                    "decision": _extract_decision(final_text),
                    "explanation": final_text,
                    "tools_used": tools_used,
                    "iterations": iteration,
                }

            messages.append(message)

            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name: str = func.get("name", "")
                tool_args: dict = func.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                log.info("Вызов инструмента: %s(%s)", tool_name, tool_args)
                result = await _call_mcp_tool(tool_name, tool_args)
                tools_used.append({"tool": tool_name, "args": tool_args, "result": result})
                messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    log.warning("Лимит итераций достигнут без финального ответа LLM")
    return {
        "decision": "UNKNOWN",
        "explanation": "Анализ не завершён в пределах допустимого числа итераций.",
        "tools_used": tools_used,
        "iterations": MAX_ITERATIONS,
    }
