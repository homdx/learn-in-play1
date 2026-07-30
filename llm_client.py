"""
llm_client.py

Минимальный клиент для чат-комплишенов, который умеет говорить и с Ollama
(``/api/chat``), и с Jan / любым OpenAI-совместимым сервером (``/v1/chat/
completions``). Логика и формат payload'ов взяты (в упрощённом виде) из
tools/llm_stream.py проекта https://github.com/homdx/jan-auto-agent
(ветка collect-fix-2) — чтобы конфиги и способ обращения к серверу были
такими же, как там.

Использование:

    from llm_client import LLMClient

    client = LLMClient.from_config(cfg, "player")   # секция [player] в config.ini
    text = client.chat(system="...", user="...")
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Убирает блоки <think>...</think> (qwen3-style reasoning) из ответа."""
    if not text:
        return text
    out = _THINK_RE.sub("", text)
    if "</think>" in out:
        out = out.rsplit("</think>", 1)[-1]
    elif "<think>" in out:
        out = out.split("<think>", 1)[0]
    out = out.replace("<think>", "").replace("</think>", "")
    return out.strip()


def strip_json_fence(text: str) -> str:
    """Снимает ```json ... ``` / ``` ... ``` обёртку, если она есть."""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text


def _make_unverified_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _ollama_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/api/chat"):
        return base
    if base.endswith("/api"):
        return f"{base}/chat"
    return f"{base}/api/chat"


class LLMClient:
    """
    Обёртка над одним профилем API (см. [api] / [api_local] / [api_remote]
    в config.ini). Поддерживает api_format = "ollama" (для `ollama serve`)
    и api_format = "openai" (для Jan и вообще любого OpenAI-совместимого
    сервера, включая удалённые).
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 api_format: str = "ollama", verify_ssl: bool = True,
                 num_ctx: int = 0, think: "bool | None" = None,
                 timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_format = api_format
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        self._ssl_context = None if verify_ssl else _make_unverified_context()

    @classmethod
    def from_config(cls, cfg, section: str = None) -> "LLMClient":
        """
        Строит клиента из configparser.ConfigParser, следуя схеме
        agents.ini: [api].active указывает, какую секцию api_<active>
        использовать (например api_local для Ollama, api_remote для Jan
        / удалённого сервера).
        """
        active = cfg.get("api", "active", fallback="local")
        api_section = f"api_{active}"
        verify_ssl = cfg.getboolean("api", "verify_ssl", fallback=True)

        base_url = cfg.get(api_section, "base_url")
        api_key = cfg.get(api_section, "api_key", fallback="not-needed")
        model = cfg.get(api_section, "model")
        api_format = cfg.get(api_section, "api_format", fallback="ollama")
        num_ctx = cfg.getint(api_section, "num_ctx", fallback=0)
        think_raw = cfg.get(api_section, "think", fallback=None)
        think = None if think_raw is None else cfg.getboolean(api_section, "think")
        timeout = cfg.getint("api", "timeout_seconds", fallback=120)

        return cls(base_url=base_url, api_key=api_key, model=model,
                    api_format=api_format, verify_ssl=verify_ssl,
                    num_ctx=num_ctx, think=think, timeout=timeout)

    def _build_request(self, system: str, user: str, temperature: float,
                        max_tokens: int):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.api_format == "ollama":
            url = _ollama_chat_url(self.base_url)
            options = {"temperature": temperature, "num_predict": max_tokens}
            if self.num_ctx:
                options["num_ctx"] = self.num_ctx
            payload = {"model": self.model, "messages": messages,
                       "stream": False, "options": options}
            if self.think is not None:
                payload["think"] = self.think
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages,
                "stream": False,
            }
        return url, headers, payload

    def _extract_content(self, raw: dict) -> str:
        if self.api_format == "ollama":
            return raw["message"]["content"].strip()
        choices = raw.get("choices") or []
        if not choices:
            raise ValueError(
                f"Пустой ответ LLM (choices=[]) - возможно, запрос был "
                f"отфильтрован сервером. Ключи ответа: {list(raw.keys())}"
            )
        return choices[0]["message"]["content"].strip()

    def chat(self, system: str, user: str, temperature: float = 0.4,
             max_tokens: int = 400) -> str:
        """Блокирующий вызов чат-комплишена. Возвращает текст ответа
        (с уже вырезанным <think>...</think>, если он был)."""
        url, headers, payload = self._build_request(system, user, temperature, max_tokens)
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout,
                                         context=self._ssl_context) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code} от {url}: {detail or e.reason}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Не удалось подключиться к {url} ({e.reason}). "
                f"Проверьте, что сервер запущен (ollama serve / Jan API server) "
                f"и base_url в config.ini указан верно."
            ) from None

        text = self._extract_content(raw)
        return strip_think(text)

    def chat_json(self, system: str, user: str, temperature: float = 0.4,
                  max_tokens: int = 400) -> dict:
        """Как chat(), но парсит ответ как JSON (снимая ``` обёртку при необходимости)."""
        text = self.chat(system, user, temperature=temperature, max_tokens=max_tokens)
        cleaned = strip_json_fence(text)
        return json.loads(cleaned)
