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

import inspect
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request

# ── DEBUG ────────────────────────────────────────────────────────────────────
# Включается переменной окружения:  LLM_DEBUG=1 python run_game_v2.py ...
# Или жёстко:  _LLM_DEBUG = True
_LLM_DEBUG: bool = os.environ.get("LLM_DEBUG", "0").strip() not in ("0", "", "false", "no")

def _dbg(*args):
    if _LLM_DEBUG:
        import sys, datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[LLM_DEBUG {ts}]", *args, file=sys.stderr, flush=True)
# ─────────────────────────────────────────────────────────────────────────────

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


_RETRY_AFTER_MS_RE = re.compile(r"(?:try again|retry)\s+in\s+([\d.]+)\s*ms", re.IGNORECASE)
_RETRY_AFTER_S_RE = re.compile(r"(?:try again|retry)\s+in\s+([\d.]+)\s*s(?:econds?)?\b", re.IGNORECASE)


def _parse_retry_after(e: urllib.error.HTTPError, detail: str):
    """RATE-2: сколько РЕАЛЬНО ждать перед повтором 429 — а не фиксированные
    "60 секунд по умолчанию", когда сервер прямым текстом сказал точную
    цифру.

    Реальный случай (Groq): второй подряд 429 на тот же запрос падал уже не
    в RATE-1 (тот срабатывает только один раз на вызов, см. `_retried_429`
    ниже), а в общий HTTP-RETRY, который слепо спал `error_retry_wait_sec`
    (60 сек) — хотя тело ответа содержало `"Please try again in 820ms"`.
    Groq не всегда шлёт заголовок `Retry-After` (виден не в каждом 429), но
    почти всегда пишет точное время текстом в теле ошибки — берём его как
    запасной источник, если заголовка нет.

    RATE-4: Google Gemini (`generativelanguage.googleapis.com`) на 429
    пишет ДРУГОЙ глагол — не "try again in", а "Please retry in
    57.062042596s." — старые регулярки требовали ровно фразу "try again
    in" и не совпадали вообще, поэтому пауза не парсилась и код всегда
    ждал фиксированные 60 сек вместо честных ~57 из ответа сервера.
    Регулярки расширены на оба варианта: "try again in" И "retry in".

    Возвращает секунды (float) или None, если число не нашлось нигде —
    тогда вызывающий сам решает, чем заменить (обычно фиксированным
    error_retry_wait_sec).
    """
    if e.headers:
        retry_after = e.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.1, float(retry_after))
            except ValueError:
                pass
    m = _RETRY_AFTER_MS_RE.search(detail)
    if m:
        try:
            return max(0.1, float(m.group(1)) / 1000.0)
        except ValueError:
            pass
    m = _RETRY_AFTER_S_RE.search(detail)
    if m:
        try:
            return max(0.1, float(m.group(1)))
        except ValueError:
            pass
    return None


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


def _chat_takes_json_mode(fn) -> bool:
    """Принимает ли данная реализация chat() kwarg json_mode.

    Результат кэшируется по объекту функции: chat_json вызывается на
    каждый ход каждого агента, а inspect.signature() не бесплатен.
    Функция с **kwargs считается принимающей — так подписаны обёртки.
    """
    key = getattr(fn, "__func__", fn)
    cached = _JSON_MODE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        params = inspect.signature(fn).parameters
        ok = ("json_mode" in params
              or any(p.kind is inspect.Parameter.VAR_KEYWORD
                     for p in params.values()))
    except (TypeError, ValueError):
        ok = False
    _JSON_MODE_CACHE[key] = ok
    return ok


_JSON_MODE_CACHE: dict = {}


class _ServerCaps:
    """
    Что УМЕЕТ сервер моделей. Живёт на уровне модуля по той же причине,
    что и _Breaker: агентов много, каждый строит свой LLMClient
    (agent_v2.py:729), а сервер за ними один.

    Первая версия держала этот флаг на экземпляре — и была бесполезна.
    Игрок, поймавший 400 на "format": "json", отключал поле себе, а
    четверо остальных приходили со своими клиентами и ловили тот же 400
    заново, каждый раунд. Знание, добытое ценой запроса, обязано быть
    общим.

    Ключ — base_url, а не глобальный флаг: в одном прогоне можно ходить
    и на локальную ollama (format понимает), и на удалённый шлюз (не
    понимает), и вывод про один сервер не должен калечить другой.
    """
    no_json_format: dict[str, bool] = {}

    @classmethod
    def rejects_json_format(cls, base_url: str) -> bool:
        return cls.no_json_format.get(base_url, False)

    @classmethod
    def mark_json_format_rejected(cls, base_url: str):
        cls.no_json_format[base_url] = True

    @classmethod
    def reset(cls):
        cls.no_json_format.clear()


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
                 timeout: int = 120, retries: int = 1, on_retry=None,
                 json_format: bool = True,
                 error_retries: int = 0, error_retry_wait_sec: int = 60,
                 max_retry_after_sec: int = 180):
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
        # HTTP-RETRY: сколько ДОПОЛНИТЕЛЬНЫХ попыток делать на ЛЮБОЙ HTTP-код
        # ошибки от сервера (402 "кончились кредиты", 5xx, и т.п.), помимо
        # уже существующего отдельного разбора 429 и 400+json_mode. Каждая
        # попытка ждёт error_retry_wait_sec секунд перед повтором ТОГО ЖЕ
        # запроса без изменений — в отличие от RETRY-1 (chat_json), который
        # меняет temperature/промпт и не ждёт вовсе. Раньше 402/5xx сразу
        # роняли раунд без единой попытки переждать: HF отдаёт 402 даже на
        # кратковременный сбой биллинга у них, не только при реально
        # исчерпанной квоте, так что слепой отказ без повтора терял раунды
        # там, где повтор через минуту мог бы пройти. По умолчанию 0
        # (прежнее поведение — падать сразу): прямое LLMClient(...), как в
        # тестах и заглушках, не должно внезапно засыпать на минуту при
        # каждой HTTP-ошибке. from_config() ниже включает 2 попытки по
        # умолчанию — это путь реальных прогонов игры.
        self.error_retries = max(0, int(error_retries))
        self.error_retry_wait_sec = max(1, int(error_retry_wait_sec))
        # RATE-3: реальный случай — Groq на 429 по ДНЕВНОМУ лимиту (TPD, а
        # не TPM) прислал Retry-After=1754 сек (29 минут) и по логике этого
        # же прогона это ещё вырастет вплоть до нескольких ЧАСОВ (TPD
        # сбрасывается раз в сутки, а не раз в минуту). RATE-1/HTTP-RETRY
        # честно спали ровно столько, сколько сказал сервер — блокируя ВЕСЬ
        # прогон игры на этот срок на одном запросе, вместо того чтобы
        # быстро сдаться и дать пулу (если настроен) переключиться на
        # другой сервер/модель, или игре — форсировать ставку и продолжить.
        # Потолок: если сервер просит ждать дольше max_retry_after_sec —
        # не спим, поднимаем исключение сразу. Короткие TPM-паузы (обычно
        # секунды-десятки секунд) под потолок не попадают и ждут как раньше.
        self.max_retry_after_sec = max(1, int(max_retry_after_sec))
        # Необязательный колбэк log(str): клиент не знает про логгер игры,
        # но вызывающий может подставить свой, чтобы повторы были видны.
        self.on_retry = on_retry
        self._ssl_context = None if verify_ssl else _make_unverified_context()
        # EOS-2/COMPAT: json_format=false в [api] — это утверждение про
        # СЕРВЕР, поэтому оно и пишется в общий кэш, а не в экземпляр.
        if not json_format:
            _ServerCaps.mark_json_format_rejected(self.base_url)

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
        # HTTP-RETRY: см. докстринг __init__. 2 попытки по 60 сек — то же,
        # что раньше было только у 429, теперь распространено на 402/5xx.
        error_retries = cfg.getint("api", "error_retries", fallback=2)
        error_retry_wait_sec = cfg.getint("api", "error_retry_wait_sec",
                                          fallback=60)
        # RATE-3: потолок разумного ожидания на 429 — см. докстринг
        # __init__. 180 сек по умолчанию: перекрывает типичные TPM-паузы
        # (обычно секунды-десятки секунд), но не даёт повторам блокировать
        # весь прогон на десятки минут/часы, когда сервер прислал время
        # ДО СБРОСА ДНЕВНОГО/МЕСЯЧНОГО лимита, а не минутного.
        max_retry_after_sec = cfg.getint("api", "max_retry_after_sec",
                                         fallback=180)
        # Шлюзы перед удалённым сервером могут не знать поля format —
        # можно отключить заранее, не дожидаясь первого 400.
        json_format = cfg.getboolean("api", "json_format", fallback=True)

        return cls(base_url=base_url, api_key=api_key, model=model,
                    api_format=api_format, verify_ssl=verify_ssl,
                    num_ctx=num_ctx, think=think, timeout=timeout,
                    retries=retries, json_format=json_format,
                    error_retries=error_retries,
                    error_retry_wait_sec=error_retry_wait_sec,
                    max_retry_after_sec=max_retry_after_sec)

    def _build_request(self, system: str, user: str, temperature: float,
                        max_tokens: int, json_mode: bool = False):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # UA-1: urllib без явного User-Agent шлёт "Python-urllib/3.x" —
            # реальный случай: Groq (за Cloudflare) вернул HTTP 403 "error
            # code: 1010" — это Cloudflare-специфичная блокировка ИМЕННО ПО
            # СИГНАТУРЕ клиента (известные строки скрипт-библиотек вроде
            # python-urllib/python-requests/curl/scrapy банятся до всякого
            # JS-челленджа), не касается настоящих ботов с реальным
            # поведением. Честная, но не библиотечная строка UA обычно
            # достаточна, чтобы пройти именно этот фильтр — TLS/JA3-
            # фингерпринтинг она не подделывает и не решает более
            # продвинутую защиту, но 1010 чаще всего именно про строку UA.
            "User-Agent": "learn-in-play1-llm-client/1.0 (+https://github.com/homdx/learn-in-play1)",
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
            # EOS-2: грамматика JSON запрещает EOS первым токеном. Пустой
            # ответ (eval_count=1, eval_duration отсутствует) физически
            # невозможен: сэмплер обязан начать с '{'.
            if json_mode and not self._json_format_off():
                payload["format"] = "json"
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages,
                "stream": False,
            }
            if json_mode and not self._json_format_off():
                payload["response_format"] = {"type": "json_object"}
            # THINK-1: Qwen3 и другие hybrid-thinking модели за vLLM/SGLang
            # умеют выключать <think> через chat_template_kwargs.
            # enable_thinking. Без этого модель тратит ВЕСЬ max_tokens на
            # рассуждения и обрывается по 'length' с пустым content —
            # ответ уходит целиком в reasoning_content, JSON так и не
            # начинается (см. _extract_content ниже). think=false в
            # конфиге теперь действует и для api_format=openai, не только
            # для ollama.
            if self.think is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": self.think}
            # THINK-3: chat_template_kwargs — это vLLM/SGLang-специфика.
            # OpenRouter (и любой gateway перед ним, включая gpt-oss —
            # реальный случай: openai/gpt-oss-20b:free через OpenRouter
            # сжёг все 700 max_tokens на скрытые рассуждения в отдельном
            # канале и вернул content="" с finish_reason='length', а
            # chat_template_kwargs.enable_thinking этот канал не трогает
            # вообще) использует СВОЙ унифицированный параметр — вложенный
            # объект reasoning (см. openrouter.ai/docs/use-cases/reasoning-
            # tokens), плюс есть отдельный плоский reasoning_effort у
            # нативного OpenAI o1/o3/gpt-5. Отправляем оба поля сразу: то,
            # что конкретный провайдер не понимает, он молча игнорирует —
            # дешевле, чем гадать, какое поле сработает именно здесь.
            if self.think is False:
                payload["reasoning_effort"] = "low"
                payload["reasoning"] = {"effort": "low", "exclude": True}
        _dbg(f"_build_request: url={url!r}, model={self.model!r}, "
             f"api_format={self.api_format!r}, think={self.think!r}, "
             f"num_ctx={self.num_ctx}, max_tokens={max_tokens}, temp={temperature}, "
             f"json_mode={json_mode}")
        _dbg(f"  payload keys: {list(payload.keys())}")
        return url, headers, payload

    def _extract_content(self, raw: dict) -> str:
        _dbg(f"_extract_content: api_format={self.api_format!r}")
        _dbg(f"  raw keys: {list(raw.keys())}")
        if self.api_format == "ollama":
            msg = raw.get("message", {})
            _dbg(f"  ollama message keys: {list(msg.keys())}")
            content = msg.get("content", "")
            # Некоторые thinking-модели в Ollama кладут reasoning в message.thinking
            # а content оставляют пустым
            thinking = msg.get("thinking", "")
            _dbg(f"  content repr (first 200): {content[:200]!r}")
            _dbg(f"  thinking present: {bool(thinking)}, len={len(thinking)}")

            # DIAG-CTX: prompt_eval_count/eval_count — единственный способ
            # отличить "модель подумала пустым <think> блоком" от "промпт
            # занял весь num_ctx, генерировать было некуда". Раньше _dbg
            # печатал только list(raw.keys()) — по нему оба случая выглядят
            # одинаково пусто, а причины и фикс у них разные.
            prompt_eval_count = raw.get("prompt_eval_count")
            eval_count = raw.get("eval_count")
            has_eval_duration = "eval_duration" in raw
            _dbg(f"  prompt_eval_count={prompt_eval_count}, "
                 f"eval_count={eval_count}, num_ctx={self.num_ctx}, "
                 f"has_eval_duration={has_eval_duration}")

            if not content.strip() and not thinking:
                if eval_count == 0 or (eval_count is not None and not has_eval_duration):
                    near_limit = (
                        self.num_ctx and prompt_eval_count is not None
                        and prompt_eval_count >= self.num_ctx - 64
                    )
                    _dbg(
                        "  WARNING: 0 completion tokens generated "
                        f"(prompt_eval_count={prompt_eval_count}/{self.num_ctx}). "
                        + ("Prompt has filled the context window — this is "
                           "context overflow, NOT a thinking-only glitch. "
                           "Shrink dsyn/checklist/history sizes or raise num_ctx."
                           if near_limit else
                           "eval_count=0 but prompt is not near num_ctx limit — "
                           "investigate server-side (stop token / filter?).")
                    )
                elif thinking:
                    _dbg("  WARNING: content is empty but thinking is not — "
                         "model returned only a <think> block, no JSON output!")
            return content.strip()
        choices = raw.get("choices") or []
        _dbg(f"  openai choices count: {len(choices)}")
        if not choices:
            _dbg(f"  ERROR: empty choices. Full raw: {json.dumps(raw)[:500]}")
            raise ValueError(
                f"Пустой ответ LLM (choices=[]) - возможно, запрос был "
                f"отфильтрован сервером. Ключи ответа: {list(raw.keys())}"
            )
        msg = choices[0].get("message", {})
        _dbg(f"  openai message keys: {list(msg.keys())}")
        content = msg.get("content", "") or ""
        # OpenAI-совместимые серверы с thinking иногда кладут рассуждения в
        # отдельное поле, а content остаётся пустым. Разные экосистемы
        # называют его по-разному: vLLM/SGLang — "reasoning_content",
        # OpenRouter (в т.ч. gpt-oss через него, реальный случай) —
        # "reasoning". Раньше проверялось только первое имя, и
        # LLM_DEBUG печатал "reasoning_content present: False" даже когда
        # в raw-ответе лежало 3000+ символов рассуждений под "reasoning" —
        # диагностика откровенно врала о причине пустого content.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        finish_reason = choices[0].get("finish_reason", "")
        _dbg(f"  finish_reason: {finish_reason!r}")
        _dbg(f"  content repr (first 200): {content[:200]!r}")
        _dbg(f"  reasoning present: {bool(reasoning)}, len={len(reasoning)}")
        if not content.strip() and reasoning:
            _dbg("  WARNING: content is empty but reasoning is not — "
                 "model returned only thinking, no actual JSON output!")
        return content.strip()

    def chat(self, system: str, user: str, temperature: float = 0.4,
             max_tokens: int = 400, json_mode: bool = False,
             _retried_429: bool = False, _http_retry_n: int = 0) -> str:
        """Блокинг-вызов чат-комплишена. Возвращает текст ответа
        (с уже вырезанным <think>...</think>, если он был)."""
        url, headers, payload = self._build_request(system, user, temperature,
                                                    max_tokens, json_mode)
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
            # RATE-1: 429 у router.huggingface.co (и вообще у большинства
            # OpenAI-совместимых шлюзов) значит "подожди и повтори", а не
            # "сервер недоступен" — путать его с обрывом связи (RETRY-1)
            # неверно: там повтор МЕНЯЕТ temperature/промпт, здесь нужно
            # просто подождать и повторить ТОТ ЖЕ запрос без изменений.
            # Ждём Retry-After из ответа, если сервер его прислал, иначе
            # 60 сек по умолчанию. Повторяем только один раз здесь —
            # дальше решает обычный RETRY-1 в chat_json().
            if e.code == 429 and not _retried_429:
                wait_s = _parse_retry_after(e, detail)
                if wait_s is None:
                    wait_s = 60.0
                if wait_s > self.max_retry_after_sec:
                    # RATE-3: сервер просит ждать дольше разумного потолка —
                    # почти наверняка дневной/месячный лимит, а не минутный.
                    # Не спим часами внутри одного вызова — поднимаем
                    # исключение сразу, пусть решает вызывающий уровень
                    # (пул переключится на другой сервер, игра форсирует
                    # решение), а не блокируем весь прогон на этот срок.
                    msg = (f"HTTP 429 от {url}: сервер просит ждать "
                          f"{wait_s:.0f} сек — это дольше потолка "
                          f"{self.max_retry_after_sec} сек (похоже на "
                          f"дневной/месячный лимит, а не минутный), не жду")
                    on_retry = getattr(self, "on_retry", None)
                    if on_retry:
                        on_retry(msg)
                    _dbg(f"chat(): {msg}")
                    raise RuntimeError(msg) from None
                on_retry = getattr(self, "on_retry", None)
                if on_retry:
                    on_retry(f"HTTP 429 от {url} (rate limited), "
                             f"жду {wait_s:.1f} сек и повторяю тот же запрос")
                _dbg(f"chat(): HTTP 429, sleeping {wait_s:.1f}s before retry "
                     f"(detail={detail[:200]!r})")
                time.sleep(wait_s)
                return self.chat(system, user, temperature=temperature,
                                 max_tokens=max_tokens, json_mode=json_mode,
                                 _retried_429=True)
            # EOS-2/COMPAT: 400 на запросе с "format": "json" почти всегда
            # означает шлюз, который валидирует тело строго и режет поля,
            # которых нет в его схеме. Локальная ollama такое поле
            # принимает, удалённый прокси — нет, поэтому поломка вылезает
            # только на api_remote и выглядит как «работало, перестало».
            #
            # Отступаем один раз и запоминаем на клиенте: дальше запросы
            # идут в прежнем виде. Теряем грамматическую защиту от EOS —
            # но она была страховкой, а не условием работы, и падать
            # вместо этого нельзя.
            if json_mode and e.code == 400 and not self._json_format_off():
                _ServerCaps.mark_json_format_rejected(self.base_url)
                _dbg("chat(): HTTP 400 with format=json — server rejects the "
                     "field, retrying without it and disabling it for this "
                     "client")
                return self.chat(system, user, temperature=temperature,
                                 max_tokens=max_tokens, json_mode=False)
            # HTTP-RETRY: любой другой код ошибки от сервера — 402 "кончились
            # кредиты", 5xx, и т.п. Раньше это сразу роняло раунд (см.
            # max_consecutive_failures = 1 в [api]) без единой попытки
            # переждать. У router.huggingface.co 402 иногда отдаётся и на
            # кратковременный сбой биллинга на их стороне, не только когда
            # квота реально исчерпана — так что фиксированная пауза и повтор
            # того же запроса стоят своей цены, даже если для настоящего
            # исчерпания месячной квоты они не помогут и раунд всё равно
            # прервётся, просто на 1-2 минуты позже.
            #
            # RATE-2: если это ВТОРОЙ подряд 429 на тот же вызов (RATE-1
            # выше уже использовал свой единственный повтор и снова получил
            # 429), не спим слепо error_retry_wait_sec — Groq в этот момент
            # обычно прямо пишет реальное время в теле ответа ("Please try
            # again in 820ms"), и ждать вместо этого фиксированную минуту —
            # чистая потеря времени. Для остальных кодов (402/5xx), где
            # такой точной цифры обычно нет, остаётся error_retry_wait_sec.
            if _http_retry_n < self.error_retries:
                wait_s = self.error_retry_wait_sec
                if e.code == 429:
                    precise = _parse_retry_after(e, detail)
                    if precise is not None:
                        wait_s = precise
                # RATE-3: тот же потолок, что и в RATE-1 выше — см. докстринг
                # __init__. Именно ЗДЕСЬ был реальный зависон на 1754 сек в
                # логе (второй подряд 429 на дневной лимит Groq попал уже в
                # эту ветку, а не в RATE-1).
                if wait_s > self.max_retry_after_sec:
                    msg = (f"HTTP {e.code} от {url}: сервер просит ждать "
                          f"{wait_s:.0f} сек — это дольше потолка "
                          f"{self.max_retry_after_sec} сек, не жду")
                    on_retry = getattr(self, "on_retry", None)
                    if on_retry:
                        on_retry(msg)
                    _dbg(f"chat(): {msg}")
                    raise RuntimeError(msg) from None
                on_retry = getattr(self, "on_retry", None)
                msg = (f"HTTP {e.code} от {url} ({detail or e.reason}), жду "
                       f"{wait_s:.1f} сек и повторяю тот же запрос (попытка "
                       f"{_http_retry_n + 1}/{self.error_retries})")
                if on_retry:
                    on_retry(msg)
                _dbg(f"chat(): {msg}")
                time.sleep(wait_s)
                return self.chat(system, user, temperature=temperature,
                                 max_tokens=max_tokens, json_mode=json_mode,
                                 _retried_429=_retried_429,
                                 _http_retry_n=_http_retry_n + 1)
            raise RuntimeError(f"HTTP {e.code} от {url}: {detail or e.reason}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Не удалось подключиться к {url} ({e.reason}). "
                f"Проверьте, что сервер запущен (ollama serve / Jan API server) "
                f"и base_url в config.ini указан верно."
            ) from None

        text = self._extract_content(raw)
        _dbg(f"chat(): raw text len={len(text)}, repr(first 300): {text[:300]!r}")
        stripped = strip_think(text)
        _dbg(f"chat(): after strip_think len={len(stripped)}, repr(first 300): {stripped[:300]!r}")
        if not stripped and text:
            _dbg("chat(): WARNING — strip_think() returned empty string! "
                 "The model likely returned ONLY a <think> block with no content after it.")
        return stripped

    def _json_format_off(self) -> bool:
        """getattr — для заглушек, созданных без __init__ (нет base_url)."""
        return _ServerCaps.rejects_json_format(getattr(self, "base_url", ""))

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
                # EOS-2 (заменяет EOS-1). EOS-1 понижал temperature на
                # повторе — и этим гарантировал повтор сбоя: если EOS уже
                # argmax первого токена, то чем ниже температура, тем
                # вернее он выпадет снова. В логе это видно прямо: три
                # попытки подряд с temp 0.8 → 0.48 → 0.288, все три
                # eval_count=1, вторая и третья за 0.8 с (промпт уже в
                # KV-кэше, тот же префикс → тот же argmax).
                #
                # Теперь наоборот: температуру повышаем и заодно портим
                # хвост промпта. Изменение последнего токена промпта
                # сбивает и распределение, и переиспользование кэша —
                # попытка перестаёт быть точной копией предыдущей.
                attempt_temp = temperature
                attempt_user = user
                if attempt > 1:
                    attempt_temp = min(temperature + 0.15 * (attempt - 1), 1.0)
                    attempt_user = (
                        user + "\n\n(Ответь ТОЛЬКО объектом JSON, "
                        "начиная с символа '{'. Попытка "
                        f"{attempt}.)"
                    )
                # EOS-2/STUB: заглушки в тестах и stub-режиме подменяют
                # chat() своей функцией со старой сигнатурой, без
                # json_mode. Передавать новый kwarg вслепую — значит
                # ронять их TypeError'ом на ровном месте, причём внутри
                # except-ветки он выглядел бы как «модель не ответила».
                # Спрашиваем сигнатуру, а не ловим TypeError: настоящий
                # TypeError из тела chat() так не будет проглочен.
                if _chat_takes_json_mode(self.chat):
                    text = self.chat(system, attempt_user,
                                     temperature=attempt_temp,
                                     max_tokens=max_tokens, json_mode=True)
                else:
                    text = self.chat(system, attempt_user,
                                     temperature=attempt_temp,
                                     max_tokens=max_tokens)
                _dbg(f"chat_json() attempt {attempt}: text len={len(text)}, "
                     f"temperature={attempt_temp:.3f}")
                cleaned = strip_json_fence(text)
                _dbg(f"chat_json() after strip_json_fence len={len(cleaned)}, "
                     f"repr(first 300): {cleaned[:300]!r}")
                if not cleaned:
                    raise json.JSONDecodeError(
                        "chat_json got empty string after stripping — "
                        "model sampled EOS as first token (0-1 completion "
                        "tokens; see LLM_DEBUG prompt_eval_count/eval_count "
                        "for context-overflow vs sampling-glitch diagnosis)",
                        "", 0)
                result = json.loads(cleaned)
            except LLMUnavailable:
                raise
            except Exception as e:
                last = e
                if attempt < attempts:
                    on_retry = getattr(self, "on_retry", None)
                    # HTTP-RETRY: пауза перед повтором. Раньше применялась
                    # ТОЛЬКО к HTTPError внутри chat() (402/5xx) — а битый
                    # JSON / EOS первым токеном (тот самый сбой HF из лога)
                    # шёл через ЭТОТ цикл и ретраился МГНОВЕННО, без паузы,
                    # хотя настройка error_retries/error_retry_wait_sec уже
                    # была в конфиге и подразумевала паузу на ЛЮБОЙ сбой, не
                    # только HTTP-код. getattr — та же причина, что и у
                    # `retries` выше: заглушки без __init__ не должны падать
                    # тут, а просто не паузить (0 попыток паузы по умолчанию).
                    error_retries = getattr(self, "error_retries", 0)
                    wait_s = getattr(self, "error_retry_wait_sec", 60)
                    if error_retries > 0:
                        msg = (f"LLM call failed ({type(e).__name__}: {e}), "
                               f"жду {wait_s} сек и повторяю (попытка "
                               f"{attempt}/{retries})")
                        if on_retry:
                            on_retry(msg)
                        _dbg(f"chat_json(): {msg}")
                        time.sleep(wait_s)
                    elif on_retry:
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
