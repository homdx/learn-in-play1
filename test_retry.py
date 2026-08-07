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
