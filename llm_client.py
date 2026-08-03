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


class LLMUnavailable(RuntimeError):
    """
    FIX-17: сервер моделей недоступен подряд слишком много раз.

    Это НЕ обычная ошибка вызова: обычную агент гасит и уходит в заглушку
    (аварийная ставка, "просто ставлю", пустая реплика). Именно поэтому в
    реальном прогоне падение ollama осталось незамеченным — игра доиграла
    два раунда за одну секунду на одних заглушках, записала пять фиктивных
    ставок по 1 монете в раунд и объявила партию законченной.

    Это исключение агенты НЕ гасят, а прокидывают наверх: раунд обрывается
    без сохранения, а не играется вслепую. Благодаря чекпойнтам Фазы 0 и
    фазы диалогов его можно доиграть после починки сервера.
    """


def _is_timeout(exc) -> bool:
    """
    Истёкший таймаут — это медленный сервер, а не мёртвый.

    RETRY-1: раньше таймаут увеличивал счётчик предохранителя наравне с
    разрывом связи, и шесть подряд роняли весь прогон с LLMUnavailable —
    хотя ollama жив и просто долго думает над промптом в несколько тысяч
    токенов. На CPU-ноутбуке это штатный режим, а не авария.
    """
    if isinstance(exc, TimeoutError):
        return True
    # socket.timeout -> OSError с характерным текстом на старых версиях
    return isinstance(exc, OSError) and "timed out" in str(exc).lower()


def _note_failure(exc):
    if _is_timeout(exc):
        return
    _Breaker.failures += 1
    if _Breaker.failures >= _Breaker.threshold:
        raise LLMUnavailable(
            f"{_Breaker.failures} LLM-вызовов подряд завершились ошибкой "
            f"(последняя: {exc}). Раунд прерван, чтобы не доигрывать его на "
            f"аварийных заглушках."
        ) from None


class _Breaker:
    """
    FIX-17: состояние выключателя. Живёт ОТДЕЛЬНО от LLMClient намеренно.

    Это здоровье сервера моделей, а не свойство класса-обёртки: экземпляров
    клиента много (по одному на агента), сервер один. Держать счётчик
    атрибутом LLMClient оказалось вдобавок хрупко — обращение по глобальному
    имени внутри собственных методов ломается, если имя в модуле подменить
    (например, заглушкой в тестах), а `cls._failures += 1` из подкласса
    молча заводит отдельный счётчик на подклассе вместо общего.
    """
    failures = 0
    threshold = 6


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
                 timeout: int = 120, retries: int = 1, on_retry=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_format = api_format
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        # RETRY-1: сколько ДОПОЛНИТЕЛЬНЫХ попыток делать при сбое.
        # 0 — прежнее поведение (одна попытка, дальше заглушка).
        self.retries = max(0, int(retries))
        # Необязательный колбэк log(str): клиент не знает про логгер игры,
        # но вызывающий может подставить свой, чтобы повторы были видны.
        self.on_retry = on_retry
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
        retries = cfg.getint("api", "retries", fallback=1)

        return cls(base_url=base_url, api_key=api_key, model=model,
                    api_format=api_format, verify_ssl=verify_ssl,
                    num_ctx=num_ctx, think=think, timeout=timeout,
                    retries=retries)

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

    @classmethod
    def configure_breaker(cls, threshold: int):
        _Breaker.threshold = max(1, int(threshold))
        _Breaker.failures = 0

    @classmethod
    def reset_breaker(cls):
        _Breaker.failures = 0

    def chat_json(self, system: str, user: str, temperature: float = 0.4,
                  max_tokens: int = 400) -> dict:
        """
        Как chat(), но парсит ответ как JSON (снимая ``` обёртку при
        необходимости).

        RETRY-1: при сбое делает ещё self.retries попыток. Повторяем всё,
        что может пройти со второго раза — таймаут, битый JSON, временную
        ошибку сервера. НЕ повторяем LLMUnavailable: предохранитель уже
        решил, что сервера нет, и долбиться в него бессмысленно.

        Цена повтора на CPU-ноутбуке — ещё один таймаут стены, до 15 минут.
        Это осознанный размен: пропущенный вызов молча превращается в
        заглушку ("иду ставить", "оставляю старый чек-лист"), и в логе
        неотличим от осознанного решения игрока. Лучше подождать, чем
        получить раунд, где молчание игрока — на самом деле сбой связи.
        """
        # getattr, а не self.retries: подклассы-заглушки в тестах и в
        # stub-режиме создаются без вызова __init__, и жёсткое обращение к
        # атрибуту ломало бы их на ровном месте. Без него — прежнее
        # поведение: одна попытка.
        retries = getattr(self, "retries", 0)
        attempts = retries + 1
        last = None
        for attempt in range(1, attempts + 1):
            try:
                text = self.chat(system, user, temperature=temperature,
                                 max_tokens=max_tokens)
                cleaned = strip_json_fence(text)
                result = json.loads(cleaned)
            except LLMUnavailable:
                raise
            except Exception as e:
                last = e
                if attempt < attempts:
                    on_retry = getattr(self, "on_retry", None)
                    if on_retry:
                        kind = "timeout" if _is_timeout(e) else type(e).__name__
                        on_retry(f"LLM call failed ({kind}: {e}), "
                                 f"retry {attempt}/{retries}")
                    continue
                # FIX-17: разрыв связи и битый JSON считаем по-разному. Битый
                # JSON означает, что сервер жив и отвечает — модель просто не
                # попала в формат, это штатная ситуация, а не повод рвать
                # раунд. RETRY-1 добавил к этому таймаут: медленный сервер
                # тоже живой (см. _is_timeout).
                if not isinstance(e, (json.JSONDecodeError, ValueError)):
                    _note_failure(e)
                raise
            _Breaker.failures = 0
            return result
        raise last   # недостижимо, но пусть будет явно
