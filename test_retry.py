"""
test_retry.py — повтор вызова и обработка таймаута (RETRY-1).

Что проверяем:
  * при сбое делается ещё `retries` попыток, и успех со второй попытки
    возвращает нормальный результат;
  * таймаут НЕ увеличивает счётчик предохранителя — медленный сервер это
    не мёртвый сервер (на CPU-ноутбуке 5-8 минут на вызов — штатный режим);
  * настоящий разрыв связи по-прежнему считается и по-прежнему роняет
    прогон на пороге;
  * LLMUnavailable не ретраится — предохранитель уже всё решил;
  * подкласс-заглушка без __init__ не ломается (старое поведение: одна
    попытка).
"""

import json
import unittest

import llm_client
from llm_client import LLMClient, LLMUnavailable


class Base(unittest.TestCase):
    def setUp(self):
        LLMClient.reset_breaker()
        LLMClient.configure_breaker(6)

    def tearDown(self):
        LLMClient.reset_breaker()

    @staticmethod
    def client(retries=1, **kw):
        return LLMClient(base_url="http://x", api_key="k", model="m",
                         retries=retries, **kw)


class TestRetry(Base):

    def test_succeeds_on_second_attempt(self):
        c = self.client(retries=1)
        calls = []

        def fake_chat(system, user, temperature=0.4, max_tokens=400):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("timed out")
            return '{"action": "talk"}'

        c.chat = fake_chat
        self.assertEqual(c.chat_json("s", "u"), {"action": "talk"})
        self.assertEqual(len(calls), 2)

    def test_no_retry_when_disabled(self):
        c = self.client(retries=0)
        calls = []

        def fake_chat(*a, **k):
            calls.append(1)
            raise TimeoutError("timed out")

        c.chat = fake_chat
        with self.assertRaises(TimeoutError):
            c.chat_json("s", "u")
        self.assertEqual(len(calls), 1)

    def test_retries_exhausted_reraises_last_error(self):
        c = self.client(retries=2)
        calls = []

        def fake_chat(*a, **k):
            calls.append(1)
            raise ValueError("bad json")

        c.chat = fake_chat
        with self.assertRaises(ValueError):
            c.chat_json("s", "u")
        self.assertEqual(len(calls), 3)

    def test_retry_is_logged(self):
        seen = []
        c = self.client(retries=1, on_retry=seen.append)
        calls = []

        def fake_chat(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("timed out")
            return "{}"

        c.chat = fake_chat
        c.chat_json("s", "u")
        self.assertEqual(len(seen), 1)
        self.assertIn("timeout", seen[0])
        self.assertIn("retry 1/1", seen[0])

    def test_broken_json_is_retried_too(self):
        c = self.client(retries=1)
        calls = []

        def fake_chat(*a, **k):
            calls.append(1)
            return "not json at all" if len(calls) == 1 else '{"ok": 1}'

        c.chat = fake_chat
        self.assertEqual(c.chat_json("s", "u"), {"ok": 1})

    def test_unavailable_is_not_retried(self):
        c = self.client(retries=3)
        calls = []

        def fake_chat(*a, **k):
            calls.append(1)
            raise LLMUnavailable("server down")

        c.chat = fake_chat
        with self.assertRaises(LLMUnavailable):
            c.chat_json("s", "u")
        self.assertEqual(len(calls), 1)


class TestTimeoutNotCountedAsOutage(Base):

    def test_timeout_does_not_trip_breaker(self):
        """
        Ключевое: 600-секундные таймауты на медленном ноутбуке раньше
        накапливались и на шестом роняли весь прогон.
        """
        c = self.client(retries=0)
        c.chat = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out"))
        for _ in range(20):
            with self.assertRaises(TimeoutError):
                c.chat_json("s", "u")
        self.assertEqual(llm_client._Breaker.failures, 0)

    def test_socket_timeout_shape_also_ignored(self):
        c = self.client(retries=0)
        c.chat = lambda *a, **k: (_ for _ in ()).throw(OSError("The read operation timed out"))
        for _ in range(10):
            with self.assertRaises(OSError):
                c.chat_json("s", "u")
        self.assertEqual(llm_client._Breaker.failures, 0)

    def test_real_connection_error_still_counts(self):
        c = self.client(retries=0)
        c.chat = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("Не удалось подключиться"))
        for _ in range(5):
            with self.assertRaises(RuntimeError):
                c.chat_json("s", "u")
        self.assertEqual(llm_client._Breaker.failures, 5)
        with self.assertRaises(LLMUnavailable):
            c.chat_json("s", "u")

    def test_broken_json_still_does_not_count(self):
        c = self.client(retries=0)
        c.chat = lambda *a, **k: "garbage"
        for _ in range(10):
            with self.assertRaises(Exception):
                c.chat_json("s", "u")
        self.assertEqual(llm_client._Breaker.failures, 0)

    def test_success_resets_counter_after_retry(self):
        c = self.client(retries=1)
        calls = []

        def fake_chat(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("Не удалось подключиться")
            return "{}"

        c.chat = fake_chat
        c.chat_json("s", "u")
        self.assertEqual(llm_client._Breaker.failures, 0)


class TestHttpErrorRetry(unittest.TestCase):
    """HTTP-RETRY: пауза и повтор ТОГО ЖЕ запроса на ЛЮБОЙ код ошибки —
    в первую очередь 402 (кончились кредиты у router.huggingface.co),
    но и 5xx тоже."""

    def setUp(self):
        llm_client._Breaker.failures = 0
        llm_client._ServerCaps.reset()
        self._orig_sleep = llm_client.time.sleep
        self.slept = []
        llm_client.time.sleep = lambda s: self.slept.append(s)

    def tearDown(self):
        llm_client.time.sleep = self._orig_sleep

    def _err(self, code, body=b"error"):
        import urllib.error, io
        return urllib.error.HTTPError("https://api.ai", code, "err", {}, io.BytesIO(body))

    def _patch(self, responses):
        seq = list(responses)

        def fake_urlopen(req, timeout=None, context=None):
            nxt = seq.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        llm_client.urllib.request.urlopen = fake_urlopen

    def _ok(self, text='{"ok":1}'):
        import io, json as _json
        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return R(_json.dumps({"message": {"role": "a", "content": text},
                             "eval_count": 1}).encode())

    def test_constructor_default_is_no_retry(self):
        """Регрессия: прямое создание LLMClient(...) НЕ должно ждать —
        иначе весь test_json_format_fallback.py спал бы по минуте."""
        c = LLMClient("https://api.ai", "k", "m", retries=0)
        self._patch([self._err(402)])
        with self.assertRaises(RuntimeError):
            c.chat_json("s", "u")
        self.assertEqual(self.slept, [])

    def test_402_retried_when_configured(self):
        c = LLMClient("https://api.ai", "k", "m", retries=0,
                     error_retries=2, error_retry_wait_sec=60)
        self._patch([self._err(402), self._err(402), self._ok()])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        self.assertEqual(self.slept, [60, 60])

    def test_402_exhausts_retries_and_raises(self):
        c = LLMClient("https://api.ai", "k", "m", retries=0,
                     error_retries=2, error_retry_wait_sec=60)
        self._patch([self._err(402), self._err(402), self._err(402)])
        with self.assertRaises(RuntimeError):
            c.chat_json("s", "u")
        self.assertEqual(self.slept, [60, 60])

    def test_5xx_also_retried(self):
        """Не только 402 — любой код ошибки от сервера."""
        c = LLMClient("https://api.ai", "k", "m", retries=0,
                     error_retries=1, error_retry_wait_sec=60)
        self._patch([self._err(503), self._ok()])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        self.assertEqual(self.slept, [60])

    def test_from_config_defaults_to_two_retries_of_60s(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote"},
                       "api_remote": {"base_url": "https://api.ai", "model": "m"}})
        c = LLMClient.from_config(cfg)
        self.assertEqual(c.error_retries, 2)
        self.assertEqual(c.error_retry_wait_sec, 60)

    def test_from_config_reads_custom_error_retry_settings(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "error_retries": "3",
                              "error_retry_wait_sec": "30"},
                       "api_remote": {"base_url": "https://api.ai", "model": "m"}})
        c = LLMClient.from_config(cfg)
        self.assertEqual(c.error_retries, 3)
        self.assertEqual(c.error_retry_wait_sec, 30)

    def test_retry_is_logged(self):
        seen = []
        c = LLMClient("https://api.ai", "k", "m", retries=0,
                     error_retries=1, error_retry_wait_sec=60, on_retry=seen.append)
        self._patch([self._err(402), self._ok()])
        c.chat_json("s", "u")
        self.assertTrue(seen)
        self.assertIn("402", seen[0])
        self.assertIn("60", seen[0])

    def test_empty_completion_is_also_paused_and_retried(self):
        """Ключевой кейс: 'model sampled EOS as first token' — это
        JSONDecodeError, ретраится ДРУГИМ циклом (chat_json, RETRY-1), не
        через HTTPError-ветку в chat(). Пауза должна применяться и тут —
        настройка error_retries/error_retry_wait_sec касается ЛЮБОГО сбоя,
        не только HTTP-кода."""
        c = LLMClient("https://api.ai", "k", "m", retries=2,
                     error_retries=2, error_retry_wait_sec=60)
        self._patch([self._ok(text=""), self._ok(text=""), self._ok('{"ok":1}')])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        self.assertEqual(self.slept, [60, 60])

    def test_empty_completion_without_error_retries_is_instant(self):
        """Регрессия: без настройки (конструктор по умолчанию) — как
        раньше, повтор без всякой паузы."""
        c = LLMClient("https://api.ai", "k", "m", retries=2)
        self._patch([self._ok(text=""), self._ok(text=""), self._ok('{"ok":1}')])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        self.assertEqual(self.slept, [])


class TestUserAgentHeader(unittest.TestCase):
    """UA-1: реальный случай — Groq (за Cloudflare) вернул HTTP 403
    "error code: 1010" на дефолтный User-Agent от urllib
    ("Python-urllib/3.x"). Явный, не библиотечный UA должен уходить в
    КАЖДОМ запросе, для обоих api_format — иначе регрессия тихо вернёт
    именно этот баг."""

    def test_openai_format_sends_custom_user_agent(self):
        c = LLMClient("https://api.groq.com/openai/v1", "k",
                      "llama-3.1-8b-instant", api_format="openai")
        _, headers, _ = c._build_request("sys", "usr", 0.4, 400, json_mode=True)
        self.assertIn("User-Agent", headers)
        self.assertNotIn("python-urllib", headers["User-Agent"].lower())
        self.assertNotIn("urllib", headers["User-Agent"].lower())

    def test_ollama_format_sends_custom_user_agent(self):
        c = LLMClient("http://localhost:11434", "ollama", "qwen3:8b",
                      api_format="ollama")
        _, headers, _ = c._build_request("sys", "usr", 0.4, 400, json_mode=True)
        self.assertIn("User-Agent", headers)
        self.assertNotIn("urllib", headers["User-Agent"].lower())


class TestRateLimitPreciseWait(unittest.TestCase):
    """RATE-2: реальный случай (Groq) — второй подряд 429 на тот же вызов
    падал уже не в RATE-1 (у него только одна попытка на вызов), а в общий
    HTTP-RETRY, который слепо спал error_retry_wait_sec (60 сек), хотя тело
    ответа содержало точное время: "Please try again in 820ms"."""

    def setUp(self):
        llm_client._Breaker.failures = 0
        llm_client._ServerCaps.reset()
        self._orig_sleep = llm_client.time.sleep
        self.slept = []
        llm_client.time.sleep = lambda s: self.slept.append(s)

    def tearDown(self):
        llm_client.time.sleep = self._orig_sleep

    def _err_429(self, body: str, headers: dict = None):
        import urllib.error, io, email.message
        h = email.message.Message()
        for k, v in (headers or {}).items():
            h[k] = v
        return urllib.error.HTTPError("https://api.groq.com/x", 429, "Too Many Requests",
                                      h, io.BytesIO(body.encode()))

    def _patch(self, responses):
        seq = list(responses)

        def fake_urlopen(req, timeout=None, context=None):
            nxt = seq.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        llm_client.urllib.request.urlopen = fake_urlopen

    def _ok(self, text='{"ok":1}'):
        import io, json as _json
        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return R(_json.dumps({"choices": [{"message": {"content": text},
                                          "finish_reason": "stop"}]}).encode())

    def test_first_429_uses_retry_after_header(self):
        c = LLMClient("https://api.groq.com/openai/v1", "k", "m",
                     api_format="openai", retries=0)
        self._patch([self._err_429("rate limited", headers={"Retry-After": "19"}),
                    self._ok()])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [19.0])

    def test_second_consecutive_429_uses_body_text_not_fixed_60s(self):
        """Ключевой регрессионный тест: ВТОРОЙ подряд 429 (RATE-1 уже
        использовал свою одну попытку) должен ждать 0.82с из текста тела
        ответа, а НЕ фиксированные error_retry_wait_sec=60."""
        c = LLMClient("https://api.groq.com/openai/v1", "k", "m",
                     api_format="openai", retries=0,
                     error_retries=2, error_retry_wait_sec=60)
        body = ('{"error":{"message":"Rate limit reached... '
               'Please try again in 820ms. Need more tokens?",'
               '"type":"tokens","code":"rate_limit_exceeded"}}')
        self._patch([
            self._err_429("rate limited", headers={"Retry-After": "19"}),  # RATE-1: 19s
            self._err_429(body),  # второй 429, БЕЗ заголовка — только текст
            self._ok(),
        ])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [19.0, 0.82])

    def test_second_429_without_any_timing_info_falls_back_to_fixed_wait(self):
        """Если сервер не дал ни заголовка, ни текста с цифрой — используем
        error_retry_wait_sec как и раньше, а не падаем и не ждём 0 секунд."""
        c = LLMClient("https://api.groq.com/openai/v1", "k", "m",
                     api_format="openai", retries=0,
                     error_retries=2, error_retry_wait_sec=45)
        self._patch([
            self._err_429("rate limited", headers={"Retry-After": "19"}),
            self._err_429("rate limited, no timing info at all"),
            self._ok(),
        ])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [19.0, 45])

    def test_non_429_error_still_uses_fixed_wait_not_body_parsing(self):
        """402/5xx не должны пытаться парсить 'try again in' из тела —
        это специфика формата ошибок 429 у Groq, не общий контракт."""
        import urllib.error, io
        c = LLMClient("https://api.groq.com/openai/v1", "k", "m",
                     api_format="openai", retries=0,
                     error_retries=1, error_retry_wait_sec=45)
        err_402 = urllib.error.HTTPError(
            "u", 402, "x", {}, io.BytesIO(b"Please try again in 5ms, no credits"))
        self._patch([err_402, self._ok()])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [45])

    def test_rate1_refuses_to_sleep_past_the_cap(self):
        """RATE-3: реальный случай — Groq на дневном лимите (TPD, не TPM)
        прислал Retry-After=1754 сек (29 минут), и по логике того же
        прогона это росло бы вплоть до нескольких часов (TPD сбрасывается
        раз в сутки). Раньше RATE-1 послушно спал бы все 1754 секунды,
        блокируя весь прогон игры на одном запросе."""
        c = LLMClient("https://api.groq.com/openai/v1", "k", "m",
                     api_format="openai", retries=0,
                     max_retry_after_sec=180)
        self._patch([self._err_429("rate limited", headers={"Retry-After": "1754"})])
        with self.assertRaises((RuntimeError, Exception)):
            c.chat_json("s", "u")
        self.assertEqual(self.slept, [], "не должен был спать вообще — сразу упасть")

    def test_second_consecutive_429_also_respects_the_cap(self):
        """Тот же потолок должен работать и во ВТОРОМ (общем HTTP-RETRY)
        месте — именно там произошло реальное зависание в логе, так как
        RATE-1 уже использовал свою одну попытку до этого."""
        c = LLMClient("https://api.groq.com/openai/v1", "k", "m",
                     api_format="openai", retries=0,
                     error_retries=2, error_retry_wait_sec=60,
                     max_retry_after_sec=180)
        self._patch([
            self._err_429("rate limited", headers={"Retry-After": "19"}),   # RATE-1: ok, 19s
            self._err_429("rate limited", headers={"Retry-After": "1754"}),  # потолок!
        ])
        with self.assertRaises((RuntimeError, Exception)):
            c.chat_json("s", "u")
        self.assertEqual(self.slept, [19.0], "второй сон не должен был случиться")

    def test_wait_just_under_the_cap_still_sleeps_normally(self):
        """Регрессия: обычные короткие TPM-паузы (секунды/десятки секунд)
        не должны ломаться потолком — только аномально длинные."""
        c = LLMClient("https://api.groq.com/openai/v1", "k", "m",
                     api_format="openai", retries=0,
                     max_retry_after_sec=180)
        self._patch([self._err_429("rate limited", headers={"Retry-After": "170"}),
                    self._ok()])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [170.0])

    def test_from_config_reads_max_retry_after_sec(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "max_retry_after_sec": "30"},
                       "api_remote": {"base_url": "https://api.ai", "model": "m"}})
        c = LLMClient.from_config(cfg)
        self.assertEqual(c.max_retry_after_sec, 30)

    def test_from_config_default_cap_is_180(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote"},
                       "api_remote": {"base_url": "https://api.ai", "model": "m"}})
        c = LLMClient.from_config(cfg)
        self.assertEqual(c.max_retry_after_sec, 180)


class TestGeminiRetryAfterParsing(unittest.TestCase):
    """RATE-4: Google Gemini (generativelanguage.googleapis.com) на 429
    формулирует паузу по-другому, чем Groq/OpenAI: не "Please try again in
    820ms", а "Please retry in 57.062042596s." — старые регулярки искали
    буквально фразу "try again in" и не совпадали вообще ни с чем, из-за
    чего код всегда ждал фиксированные error_retry_wait_sec (60 сек) вместо
    честных ~57 из ответа сервера. Реальный случай из лога:

        "Quota exceeded for metric: .../generate_content_free_tier_requests,
        limit: 20, model: gemini-3.6-flash
        Please retry in 57.062042596s."
    """

    def setUp(self):
        llm_client._Breaker.failures = 0
        llm_client._ServerCaps.reset()
        self._orig_sleep = llm_client.time.sleep
        self.slept = []
        llm_client.time.sleep = lambda s: self.slept.append(s)

    def tearDown(self):
        llm_client.time.sleep = self._orig_sleep

    def _err_429(self, body: str, headers: dict = None):
        import urllib.error, io, email.message
        h = email.message.Message()
        for k, v in (headers or {}).items():
            h[k] = v
        return urllib.error.HTTPError(
            "https://generativelanguage.googleapis.com/x", 429,
            "Too Many Requests", h, io.BytesIO(body.encode()))

    def _patch(self, responses):
        seq = list(responses)

        def fake_urlopen(req, timeout=None, context=None):
            nxt = seq.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        llm_client.urllib.request.urlopen = fake_urlopen

    def _ok(self, text='{"ok":1}'):
        import io, json as _json
        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return R(_json.dumps({"choices": [{"message": {"content": text},
                                          "finish_reason": "stop"}]}).encode())

    GEMINI_BODY = (
        '{\n'
        '  "error": {\n'
        '    "code": 429,\n'
        '    "message": "You exceeded your current quota, please check your '
        'plan and billing details. For more information on this error, head '
        'to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor '
        'your current usage, head to: https://ai.dev/rate-limit. \\n'
        '* Quota exceeded for metric: '
        'generativelanguage.googleapis.com/generate_content_free_tier_requests, '
        'limit: 20, model: gemini-3.6-flash\\n'
        'Please retry in 57.062042596s.",\n'
        '    "status": "RESOURCE_EXHAUSTED"\n'
        '  }\n'
        '}'
    )

    def test_first_429_parses_gemini_retry_in_seconds_from_body(self):
        """RATE-1 (первая попытка на вызов): без заголовка Retry-After,
        текст тела — единственный источник, и он должен быть распознан."""
        c = LLMClient("https://generativelanguage.googleapis.com/v1beta/openai",
                     "k", "gemini-3.6-flash", api_format="openai", retries=0)
        self._patch([self._err_429(self.GEMINI_BODY), self._ok()])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [57.062042596])

    def test_second_consecutive_429_also_parses_gemini_format(self):
        """Тот же регресс, что RATE-2 чинил для Groq: ВТОРОЙ подряд 429
        (после того как RATE-1 уже использовал свою единственную попытку)
        идёт через общий HTTP-RETRY — и должен ждать распарсенные ~57с из
        тела Gemini, а не фиксированные error_retry_wait_sec=60, как было
        в реальном логе ("жду 60.0 сек" вместо честных ~57)."""
        c = LLMClient("https://generativelanguage.googleapis.com/v1beta/openai",
                     "k", "gemini-3.6-flash", api_format="openai", retries=0,
                     error_retries=2, error_retry_wait_sec=60)
        self._patch([
            self._err_429(self.GEMINI_BODY),  # RATE-1 использует попытку
            self._err_429(self.GEMINI_BODY),  # второй 429 — тоже Gemini-формат
            self._ok(),
        ])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [57.062042596, 57.062042596])

    def test_gemini_retry_after_header_still_takes_priority_if_present(self):
        """Если Gemini когда-нибудь начнёт слать Retry-After header — он
        должен побеждать текст тела, как и для всех остальных провайдеров
        (заголовок всегда приоритетнее текста, см. _parse_retry_after)."""
        c = LLMClient("https://generativelanguage.googleapis.com/v1beta/openai",
                     "k", "gemini-3.6-flash", api_format="openai", retries=0)
        self._patch([
            self._err_429(self.GEMINI_BODY, headers={"Retry-After": "12"}),
            self._ok(),
        ])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [12.0])

    def test_gemini_format_does_not_break_existing_groq_ms_format(self):
        """Регресс-щит: расширение регулярки под 'retry in' не должно
        задеть старый путь 'try again in ...ms' (Groq)."""
        c = LLMClient("https://api.groq.com/openai/v1", "k", "m",
                     api_format="openai", retries=0)
        body = ('{"error":{"message":"Rate limit reached. '
               'Please try again in 820ms.","type":"tokens"}}')
        self._patch([self._err_429(body), self._ok()])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [0.82])

    def test_gemini_style_retry_in_seconds_without_decimal(self):
        """Целое число секунд (без дробной части) тоже должно парситься —
        не только Gemini-style дробные секунды из реального лога."""
        c = LLMClient("https://generativelanguage.googleapis.com/v1beta/openai",
                     "k", "gemini-3.6-flash", api_format="openai", retries=0)
        body = '{"error":{"message":"Quota exceeded. Please retry in 15s.","code":429}}'
        self._patch([self._err_429(body), self._ok()])
        c.chat_json("s", "u")
        self.assertEqual(self.slept, [15.0])

    def test_gemini_cap_still_applies_via_max_retry_after_sec(self):
        """RATE-3: потолок ожидания должен работать и для Gemini-формата —
        аномально долгая квота (например суточный лимит) не должна
        подвешивать прогон на часы, как и для Groq."""
        c = LLMClient("https://generativelanguage.googleapis.com/v1beta/openai",
                     "k", "gemini-3.6-flash", api_format="openai", retries=0,
                     max_retry_after_sec=180)
        body = '{"error":{"message":"Quota exceeded. Please retry in 3600s.","code":429}}'
        self._patch([self._err_429(body)])
        with self.assertRaises((RuntimeError, Exception)):
            c.chat_json("s", "u")
        self.assertEqual(self.slept, [], "не должен был спать вообще — сразу упасть")


class TestStubSubclassCompatibility(Base):

    def test_subclass_without_init_still_works(self):
        """Заглушки в тестах и stub-режиме создаются без __init__."""
        class Stub(LLMClient):
            def __init__(self):
                pass

            def chat(self, *a, **k):
                return '{"ok": 1}'

        self.assertEqual(Stub().chat_json("s", "u"), {"ok": 1})

    def test_subclass_without_init_makes_single_attempt(self):
        class Stub(LLMClient):
            def __init__(self):
                self.calls = 0

            def chat(self, *a, **k):
                self.calls += 1
                raise ValueError("bad")

        st = Stub()
        with self.assertRaises(ValueError):
            st.chat_json("s", "u")
        self.assertEqual(st.calls, 1)


class TestConfig(Base):

    def test_from_config_reads_retries_and_timeout(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_string("""
[api]
active = local
timeout_seconds = 900
retries = 2
[api_local]
base_url = http://localhost:11434
model = qwen3:8b
""")
        c = LLMClient.from_config(cfg)
        self.assertEqual(c.timeout, 900)
        self.assertEqual(c.retries, 2)

    def test_retries_default_is_one(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_string("""
[api]
active = local
[api_local]
base_url = http://localhost:11434
model = qwen3:8b
""")
        self.assertEqual(LLMClient.from_config(cfg).retries, 1)

    def test_shipped_config_has_15_minute_timeout(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read("config_v2.ini")
        self.assertEqual(cfg.getint("api", "timeout_seconds"), 900)
        self.assertGreaterEqual(cfg.getint("api", "retries"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
