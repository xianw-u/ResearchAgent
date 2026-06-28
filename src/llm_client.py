from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMConfig:
    enabled: bool
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: int = 60

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.api_key.strip() and self.base_url.strip() and self.model.strip())


def generate_grounded_answer(
    question: str,
    sources: list[dict[str, Any]],
    config: LLMConfig,
    language: str = "English",
    chat_history: list[dict[str, Any]] | None = None,
) -> str:
    if not config.is_ready:
        raise ValueError("LLM is enabled but API key, base URL, or model is missing.")
    if not sources:
        raise ValueError("No retrieved sources are available for grounded generation.")

    prompt = _build_prompt(question, sources, language, chat_history or [])
    if config.provider == "Anthropic-compatible":
        return _call_anthropic_compatible(prompt, config)
    return _call_openai_compatible(prompt, config)


def _build_prompt(
    question: str,
    sources: list[dict[str, Any]],
    language: str,
    chat_history: list[dict[str, Any]],
) -> str:
    context_lines = []
    for idx, source in enumerate(sources, start=1):
        context_lines.append(f"[Source {idx} | p.{source['page']}]\n{source['text']}")
    context = "\n\n".join(context_lines)
    history = _format_chat_history(chat_history)

    answer_language = "Simplified Chinese" if language == "简体中文" else "English"
    return f"""You are ResearchAgent, a rigorous research assistant.

Answer the user's question using only the retrieved paper context below.
Use the chat history to resolve follow-up references such as "it", "this method", or "the module".
If the context is insufficient, say that the current PDF evidence is insufficient.
Write the answer in {answer_language}.
Include page citations like [p.3] where appropriate.

Recent chat history:
{history}

Question:
{question}

Retrieved context:
{context}
"""


def _format_chat_history(chat_history: list[dict[str, Any]], max_turns: int = 6) -> str:
    if not chat_history:
        return "No previous conversation."
    recent = chat_history[-max_turns:]
    lines = []
    for message in recent:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No previous conversation."


def _call_openai_compatible(prompt: str, config: LLMConfig) -> str:
    endpoint = _openai_endpoint(config.base_url)
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "You answer with grounded citations from the supplied context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=config.timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_anthropic_compatible(prompt: str, config: LLMConfig) -> str:
    endpoint = _anthropic_endpoint(config.base_url)
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "max_tokens": 1200,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=config.timeout)
    response.raise_for_status()
    data = response.json()
    content = data.get("content", [])
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "\n".join(part for part in text_parts if part).strip()
    if isinstance(content, str):
        return content.strip()
    raise ValueError("Unexpected Anthropic-compatible response format.")


def _openai_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _anthropic_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"
