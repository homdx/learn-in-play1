"""
test_think_param_fallback.py — THINK-4/THINK-5: реальный случай (Groq,
qwen/qwen3.6-27b) — chat_template_kwargs / reasoning_effort / reasoning не
тихо игнорируются неподдерживающим сервером (как рассчитывал комментарий
THINK-3), а валят ВЕСЬ запрос отдельным HTTP 400:

    {"error":{"message":"property 'chat_template_kwargs' is unsupported",
              "type":"invalid_request_error"}}

Старый код (THINK-4, первая версия фикса) ловил отказ и отключал ВСЕ ТРИ
поля ОДНИМ общим флагом. Это была слишком грубая реакция: chat_template_
kwargs — vLLM/SGLang-специфика, никогда не входившая в OpenAI-спеку, а
reasoning_effort/reasoning — штатные OpenAI/OpenRouter поля, которые тот
же сервер вполне может принимать нормально и которыми реально подавляется
thinking. Бандлинг выбрасывал рабочий механизм вместе с неработающим.

THINK-5 (эта версия) отключает ТОЛЬКО то поле, которое реально названо в
тексте ошибки сервера — остальные два продолжают отправляться как обычно.

Эти тесты контрфактические: часть падает на бандлинг-версии (THINK-4),
проходит на раздельной (THINK-5).

    python3 test_think_param_fallback.py [-v]
"""

import io
import json
import unittest
import urllib.error

import llm_client
from llm_client import LLMClient


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _err400(body: bytes):
    return urllib.error.HTTPError(
        "https://api.groq.com/openai/v1/chat/completions", 400,
        "Bad Request", {}, io.BytesIO(body))


def _ok(text='{"ok": 1}'):
    return _Resp(json.dumps({
        "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                 "total_tokens": 15},
    }).encode())


CHAT_TEMPLATE_KWARGS_400 = (
    b'{"error":{"message":"property \'chat_template_kwargs\' is '
    b'unsupported","type":"invalid_request_error"}}'
)
REASONING_EFFORT_400 = (
    b'{"error":{"message":"Unsupported parameter: \'reasoning_effort\' is '
    b'not supported with this model."}}'
)


class TestThinkFieldFallback(unittest.TestCase):
    def setUp(self):
        llm_client._Breaker.failures = 0
        llm_client._ServerCaps.reset()

    def _patch(self, responses):
        seq = list(responses)

        def fake_urlopen(req, timeout=None, context=None):
            nxt = seq.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        llm_client.urllib.request.urlopen = fake_urlopen

    def _client(self, think=False, **kw):
        return LLMClient("https://api.groq.com/openai/v1", "k",
                         "qwen/qwen3.6-27b", api_format="openai",
                         think=think, retries=0, **kw)

    def test_chat_template_kwargs_rejection_recovers_without_hanging(self):
        c = self._client(think=False)
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(CHAT_TEMPLATE_KWARGS_400),
            _ok(),
        ])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})

    def test_rejecting_chat_template_kwargs_does_not_disable_reasoning_effort(self):
        """КЛЮЧЕВОЙ тест THINK-5: отказ на chat_template_kwargs не должен
        трогать reasoning_effort/reasoning — они у ЭТОГО сервера могли
        никогда не быть протестированы и вполне рабочие."""
        c = self._client(think=False)
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(CHAT_TEMPLATE_KWARGS_400),
            _ok(),
        ])
        c.chat_json("s", "u")
        self.assertTrue(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "chat_template_kwargs"))
        self.assertFalse(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "reasoning_effort"))
        self.assertFalse(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "reasoning"))

        _, _, payload = c._build_request("s", "u", 0.4, 400, json_mode=False)
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertIn("reasoning_effort", payload)
        self.assertIn("reasoning", payload)

    def test_server_is_remembered_across_calls(self):
        c = self._client(think=False)
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(CHAT_TEMPLATE_KWARGS_400),
            _ok(),
        ])
        c.chat_json("s", "u")
        self.assertTrue(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "chat_template_kwargs"))

        self._patch([_ok()])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})

    def test_build_request_omits_only_the_specifically_rejected_field(self):
        c = self._client(think=False)
        llm_client._ServerCaps.mark_think_field_rejected(
            c.base_url, "chat_template_kwargs")
        _, _, payload = c._build_request("s", "u", 0.4, 400, json_mode=False)
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertIn("reasoning_effort", payload)
        self.assertIn("reasoning", payload)

    def test_reasoning_effort_rejection_tracked_independently(self):
        c = self._client(think=False)
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(REASONING_EFFORT_400),
            _ok(),
        ])
        c.chat_json("s", "u")
        self.assertTrue(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "reasoning_effort"))
        self.assertFalse(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "chat_template_kwargs"))
        self.assertFalse(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "reasoning"))

    def test_all_three_fields_can_be_rejected_independently_over_multiple_calls(self):
        c = self._client(think=False)
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(CHAT_TEMPLATE_KWARGS_400),
            _err400(REASONING_EFFORT_400),
            _err400(b'{"error":{"message":"unknown field \\"reasoning\\""}}'),
            _ok(),
        ])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        for f in ("chat_template_kwargs", "reasoning_effort", "reasoning"):
            self.assertTrue(llm_client._ServerCaps.think_field_rejected(
                c.base_url, f))

    def test_unrelated_400_does_not_trigger_think_field_fallback(self):
        c = LLMClient("https://api.groq.com/openai/v1", "k", "m",
                     api_format="openai", think=None, retries=0,
                     error_retries=1, error_retry_wait_sec=0)
        self._patch([
            _err400(b'{"error":{"message":"invalid model id"}}'),
            _ok(),
        ])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        for f in ("chat_template_kwargs", "reasoning_effort", "reasoning"):
            self.assertFalse(
                llm_client._ServerCaps.think_field_rejected(c.base_url, f))

    def test_ollama_format_never_triggers_this_fallback(self):
        c = LLMClient("http://localhost:11434", "ollama", "qwen3:8b",
                     api_format="ollama", think=False, retries=0,
                     error_retries=1, error_retry_wait_sec=0)
        self._patch([
            _err400(CHAT_TEMPLATE_KWARGS_400),
            _Resp(json.dumps({"message": {"role": "a", "content": '{"ok":1}'},
                             "eval_count": 5}).encode()),
        ])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        self.assertFalse(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "chat_template_kwargs"))

    def test_think_none_never_sends_the_fields_so_never_triggers(self):
        """Это тот самый случай из последнего реального лога — think=None,
        модель бесконечно жуёт <think>, но это НЕ баг из этого файла."""
        c = self._client(think=None)
        _, _, payload = c._build_request("s", "u", 0.4, 400, json_mode=False)
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("reasoning", payload)


REASONING_EFFORT_VALUE_400 = (
    b'{"error":{"message":"`reasoning_effort` must be one of `none` or '
    b'`default`","type":"invalid_request_error"}}'
)


class TestReasoningEffortValueCorrection(unittest.TestCase):
    """THINK-6: реальный случай (Groq, qwen/qwen3.6-27b, подтверждено
    руками через curl) — reasoning_effort="low" отвергается НЕ как
    неподдерживаемое поле, а как неверное ЗНАЧЕНИЕ: сервер сам говорит
    "`reasoning_effort` must be one of `none` or `default`". "none" реально
    подавляет thinking (проверено curl'ом), "default" — не подавляет.
    Нельзя просто выбросить поле (как для chat_template_kwargs) — оно
    рабочее, просто со своим набором значений у ЭТОГО сервера. И нельзя
    зашить "none" глобально — нативный OpenAI o1/o3/gpt-5 ждёт "low"/
    "medium"/"high", "none" там был бы неверным значением."""

    def setUp(self):
        llm_client._Breaker.failures = 0
        llm_client._ServerCaps.reset()

    def _patch(self, responses):
        seq = list(responses)

        def fake_urlopen(req, timeout=None, context=None):
            nxt = seq.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        llm_client.urllib.request.urlopen = fake_urlopen

    def _client(self, base_url="https://api.groq.com/openai/v1"):
        return LLMClient(base_url, "k", "qwen/qwen3.6-27b",
                         api_format="openai", think=False, retries=0)

    def test_value_error_switches_to_suggested_none_not_dropped_entirely(self):
        c = self._client()
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(REASONING_EFFORT_VALUE_400),
            _ok(),
        ])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        # Поле НЕ должно быть выброшено целиком — это была ошибка значения,
        # не отказ поля.
        self.assertFalse(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "reasoning_effort"))
        self.assertEqual(
            llm_client._ServerCaps.reasoning_effort_for(c.base_url), "none")

    def test_corrected_value_used_on_subsequent_requests(self):
        c = self._client()
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(REASONING_EFFORT_VALUE_400),
            _ok(),
        ])
        c.chat_json("s", "u")
        _, _, payload = c._build_request("s", "u", 0.4, 400, json_mode=False)
        self.assertEqual(payload.get("reasoning_effort"), "none")

    def test_other_server_unaffected_still_uses_low(self):
        """base_url — ключ. Нативный OpenAI (гипотетически на другом
        base_url) не должен получить "none" из-за того, что Groq его не
        принял — у каждого сервера своя допустимая раскладка значений."""
        c_groq = self._client()
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(REASONING_EFFORT_VALUE_400),
            _ok(),
        ])
        c_groq.chat_json("s", "u")

        c_other = self._client(base_url="https://api.openai.com/v1")
        _, _, payload = c_other._build_request("s", "u", 0.4, 400,
                                                json_mode=False)
        self.assertEqual(payload.get("reasoning_effort"), "low")

    def test_default_preferred_over_nothing_when_none_not_offered(self):
        """Если сервер предлагает набор БЕЗ "none" (гипотетический другой
        валидатор) — берём первое предложенное значение, а не выбрасываем
        поле совсем."""
        c = self._client()
        body = (b'{"error":{"message":"`reasoning_effort` must be one of '
               b'`default` or `high`"}}')
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(body),
            _ok(),
        ])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        self.assertEqual(
            llm_client._ServerCaps.reasoning_effort_for(c.base_url), "default")

    def test_plain_unsupported_field_message_still_drops_field_entirely(self):
        """Регресс-щит: формулировка БЕЗ 'must be one of' (поле реально не
        существует у сервера, не просто неверное значение) — должна
        по-прежнему приводить к полному отключению поля, а не пытаться
        парсить несуществующую подсказку."""
        c = self._client()
        self._patch([
            _err400(b'{"error":{"message":"json_object is unsupported"}}'),
            _err400(b'{"error":{"message":"Unsupported parameter: '
                    b'\'reasoning_effort\' is not supported with this '
                    b'model."}}'),
            _ok(),
        ])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        self.assertTrue(llm_client._ServerCaps.think_field_rejected(
            c.base_url, "reasoning_effort"))
        self.assertEqual(
            llm_client._ServerCaps.reasoning_effort_for(c.base_url), "low")


if __name__ == "__main__":
    unittest.main(verbosity=2)
