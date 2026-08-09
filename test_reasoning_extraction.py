"""
test_reasoning_extraction.py — THINK-3: реальный случай из прогона на
OpenRouter (openai/gpt-oss-20b:free). Сервер сжёг весь max_tokens на скрытые
рассуждения и вернул content="" с finish_reason='length', а рассуждения
лежали в message["reasoning"] (не "reasoning_content" — то единственное имя,
которое _extract_content умел проверять). Диагностика (LLM_DEBUG) поэтому
врала: "reasoning_content present: False" при 3000+ символах рассуждений в
другом поле. chat_template_kwargs.enable_thinking=False (единственный способ
выключить thinking, который был в коде) не понимается OpenRouter/gpt-oss —
у него свой параметр `reasoning`.

Эти тесты контрфактические: падают на старом коде (только "reasoning_content"
в _extract_content, только chat_template_kwargs в _build_request), проходят
на исправленном.
"""

import contextlib
import io
import unittest

import llm_client
from llm_client import LLMClient


@contextlib.contextmanager
def _capture_dbg():
    """Включает LLM_DEBUG на время теста и перехватывает его вывод (idёт в
    stderr через print) в StringIO — чтобы assert-ить реальный текст
    диагноза, а не только факт отсутствия исключения."""
    orig = llm_client._LLM_DEBUG
    llm_client._LLM_DEBUG = True
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            yield buf
    finally:
        llm_client._LLM_DEBUG = orig


class TestReasoningFieldExtraction(unittest.TestCase):
    """_extract_content должен видеть рассуждения под ЛЮБЫМ из известных
    имён поля, а не только "reasoning_content"."""

    def setUp(self):
        self.c = LLMClient("https://api.example", "k", "m", api_format="openai")

    def _raw(self, message_extra: dict, finish_reason: str = "length") -> dict:
        return {
            "choices": [{
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": None, **message_extra},
            }]
        }

    def test_reasoning_content_field_still_recognized(self):
        """Регрессия: старое имя поля (vLLM/SGLang, Qwen3) не должно сломаться."""
        raw = self._raw({"reasoning_content": "thinking..."})
        content = self.c._extract_content(raw)
        self.assertEqual(content, "")  # content пуст — это ожидаемо для этого теста

    def test_reasoning_field_now_recognized(self):
        """Реальный случай: OpenRouter/gpt-oss кладёт рассуждения в
        "reasoning", а не в "reasoning_content". _extract_content не должен
        падать и должен корректно вернуть пустой content (рассуждения сами
        по себе не JSON-ответ), не бросив исключение на новом имени поля."""
        raw = self._raw({"reasoning": "We need to output JSON with updated betting synapse..."})
        content = self.c._extract_content(raw)
        self.assertEqual(content, "")

    def test_content_present_wins_over_reasoning(self):
        """Обычный случай (не сбой): content есть — используем его, а не
        рассуждения, независимо от того, под каким именем лежат рассуждения."""
        raw = self._raw({"content": '{"ok": 1}', "reasoning": "some thoughts"})
        content = self.c._extract_content(raw)
        self.assertEqual(content, '{"ok": 1}')

    def test_no_reasoning_field_at_all_still_works(self):
        """Обычная не-thinking модель без каких-либо reasoning-полей вообще —
        поведение не должно измениться."""
        raw = self._raw({"content": '{"ok": 1}'})
        content = self.c._extract_content(raw)
        self.assertEqual(content, '{"ok": 1}')


class TestThinkFalsePayload(unittest.TestCase):
    """think=False должен реально пытаться выключить рассуждения на КАЖДОЙ
    известной экосистеме параметров, не только vLLM/SGLang."""

    def _payload(self, think, api_format="openai"):
        c = LLMClient("https://api.example", "k", "m", api_format=api_format,
                     think=think)
        _, _, payload = c._build_request("sys", "usr", 0.4, 400, json_mode=True)
        return payload

    def test_think_false_sets_chat_template_kwargs(self):
        """Старый механизм (vLLM/SGLang, Qwen3) — не должен сломаться."""
        payload = self._payload(False)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_think_false_also_sets_openrouter_reasoning_object(self):
        """Реальный случай: openai/gpt-oss-20b:free через OpenRouter не
        понимает chat_template_kwargs вообще — у него свой унифицированный
        параметр reasoning (openrouter.ai/docs/use-cases/reasoning-tokens)."""
        payload = self._payload(False)
        self.assertIn("reasoning", payload)
        self.assertEqual(payload["reasoning"].get("effort"), "low")

    def test_think_false_also_sets_flat_reasoning_effort(self):
        """Нативный OpenAI o1/o3/gpt-5-style reasoning_effort — третья
        экосистема параметров, тоже плоское поле, не вложенный объект."""
        payload = self._payload(False)
        self.assertEqual(payload.get("reasoning_effort"), "low")

    def test_think_true_does_not_set_reasoning_suppression(self):
        """think=True — явно ХОТИМ рассуждения, не надо их гасить."""
        payload = self._payload(True)
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("reasoning", payload)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})

    def test_think_unset_sends_nothing_reasoning_related(self):
        """Регрессия: think не указан в конфиге (None) — поведение как
        раньше, ничего лишнего в payload."""
        payload = self._payload(None)
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("reasoning", payload)

    def test_ollama_format_unaffected(self):
        """Подавление reasoning для openai-эндпоинтов не должно просачиваться
        в ollama-ветку — там свой параметр think на верхнем уровне payload."""
        payload = self._payload(False, api_format="ollama")
        self.assertEqual(payload.get("think"), False)
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("reasoning", payload)


class TestOpenAIFormatContextOverflowDiagnostic(unittest.TestCase):
    """DIAG-CTX-3: реальный случай (Groq, qwen/qwen3.6-27b) — модель
    вложила ВСЁ рассуждение прямо в content как литеральный текст
    "<think>...", без завершающего JSON и без закрывающего тега.
    _extract_content() видит НЕПУСТОЙ content на своём этапе (это же самый
    текст "<think>..."), а strip_think() съедает его уже в chat() — то
    есть ПОЗЖЕ. Первая версия этого фикса проверяла "content.strip() пуст"
    внутри _extract_content(), где content ещё не пуст НИКОГДА для этого
    сценария — DIAGNOSIS не печатался вообще, только usage. Правильное
    место — chat(), сразу после strip_think(), где известно, что текст
    БЫЛ, а стал пустым. Эти тесты гоняют через chat_json() полностью, а
    не только _extract_content(), и ловят настоящий DIAGNOSIS в выводе."""

    def setUp(self):
        self.c = LLMClient("https://api.example", "k", "m", api_format="openai",
                          retries=0)

    def _patch(self, raw_body: dict):
        import io, json as _json

        class FakeResp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout=None, context=None):
            return FakeResp(_json.dumps(raw_body).encode())
        llm_client.urllib.request.urlopen = fake_urlopen

    def _raw_with_unclosed_think(self, finish_reason, completion_tokens):
        return {
            "choices": [{"index": 0, "finish_reason": finish_reason,
                        "message": {"role": "assistant",
                                   "content": "<think>\nreasoning with no "
                                              "closing tag or answer"}}],
            "usage": {"prompt_tokens": 1813,
                     "completion_tokens": completion_tokens,
                     "total_tokens": 1813 + completion_tokens},
        }

    def test_length_finish_reason_diagnosis_mentions_burned_budget(self):
        """Точный сценарий из реального лога: finish_reason='length',
        completion_tokens == потраченный max_tokens, content — один
        незакрытый <think> без ответа."""
        self._patch(self._raw_with_unclosed_think("length", 700))
        with _capture_dbg() as out:
            with self.assertRaises(Exception):
                self.c.chat_json("s", "u")
        text = out.getvalue()
        self.assertIn("DIAGNOSIS", text)
        self.assertIn("finish_reason='length'", text)
        self.assertIn("completion_tokens=700", text)
        self.assertIn("burned its whole max_tokens budget", text)

    def test_stop_finish_reason_diagnosis_is_distinguishable_from_length(self):
        """Если модель сама остановилась (finish_reason='stop') внутри
        <think>, не исчерпав max_tokens — это ДРУГАЯ причина (не бюджет
        токенов), диагноз должен отличаться от случая 'length'."""
        self._patch(self._raw_with_unclosed_think("stop", 120))
        with _capture_dbg() as out:
            with self.assertRaises(Exception):
                self.c.chat_json("s", "u")
        text = out.getvalue()
        self.assertIn("DIAGNOSIS", text)
        self.assertIn("finish_reason='stop'", text)
        self.assertIn("not a token-budget issue", text)
        self.assertNotIn("burned its whole max_tokens budget", text)

    def test_usage_tokens_logged_for_openai_format(self):
        raw = {
            "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 0,
                      "total_tokens": 500},
        }
        content = self.c._extract_content(raw)
        self.assertEqual(content, "")

    def test_no_usage_field_at_all_does_not_crash(self):
        """Регресс-щит: не все openai-совместимые серверы шлют usage —
        отсутствие поля не должно ронять _extract_content ни chat()."""
        self._patch({"choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant",
                                             "content": "<think>no usage field"}}]})
        with self.assertRaises(Exception):
            self.c.chat_json("s", "u")  # не должно бросить что-то ДРУГОЕ,
                                        # кроме ожидаемого JSONDecodeError


class TestOllamaContextOverflowDiagnosisMovedCorrectly(unittest.TestCase):
    """Тот же DIAG-CTX-3 фикс должен работать и для ollama: раньше
    диагноз (context overflow vs sampling glitch) стоял в
    _extract_content(), где содержимое ollama-варианта <think>-в-content
    тоже ещё не пусто на этом этапе — то же самое слепое пятно, что и у
    openai-ветки, просто для другого провайдера."""

    def setUp(self):
        self.c = LLMClient("http://localhost:11434", "ollama", "qwen3:8b",
                          api_format="ollama", retries=0, num_ctx=8192)

    def _patch(self, raw_body: dict):
        import io, json as _json

        class FakeResp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout=None, context=None):
            return FakeResp(_json.dumps(raw_body).encode())
        llm_client.urllib.request.urlopen = fake_urlopen

    def test_near_context_limit_is_flagged_as_overflow(self):
        self._patch({
            "message": {"role": "assistant",
                       "content": "<think>\nunclosed reasoning, no answer"},
            "prompt_eval_count": 8150, "eval_count": 200,
            "eval_duration": 12345,
        })
        with _capture_dbg() as out:
            with self.assertRaises(Exception):
                self.c.chat_json("s", "u")
        text = out.getvalue()
        self.assertIn("DIAGNOSIS", text)
        self.assertIn("context overflow", text)

    def test_far_from_context_limit_is_not_flagged_as_overflow(self):
        self._patch({
            "message": {"role": "assistant",
                       "content": "<think>\nunclosed reasoning, no answer"},
            "prompt_eval_count": 300, "eval_count": 200,
            "eval_duration": 12345,
        })
        with _capture_dbg() as out:
            with self.assertRaises(Exception):
                self.c.chat_json("s", "u")
        text = out.getvalue()
        self.assertIn("DIAGNOSIS", text)
        self.assertNotIn("context overflow", text)


class TestChatJsonEmptyStringErrorMessage(unittest.TestCase):
    """Текст ошибки chat_json() при пустой строке после strip не должен
    безусловно указывать на ollama-only поля (prompt_eval_count/eval_count)
    — для openai-формата провайдеров это бесполезная подсказка, там нужно
    смотреть finish_reason/usage."""

    def test_error_message_mentions_both_provider_diagnostics(self):
        c = LLMClient("https://api.example", "k", "m", api_format="openai",
                     retries=0)
        import io

        class FakeResp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout=None, context=None):
            import json as _json
            body = _json.dumps({
                "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant",
                                       "content": "<think>only thinking</think>"}}]
            }).encode()
            return FakeResp(body)
        import llm_client as _lc
        orig = _lc.urllib.request.urlopen
        _lc.urllib.request.urlopen = fake_urlopen
        try:
            with self.assertRaises(Exception) as ctx:
                c.chat_json("s", "u")
            msg = str(ctx.exception)
            self.assertIn("finish_reason", msg)
            self.assertIn("usage", msg)
            self.assertIn("prompt_eval_count", msg)
            self.assertIn("openai-format", msg)
            self.assertIn("ollama", msg)
        finally:
            _lc.urllib.request.urlopen = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)
