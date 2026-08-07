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

import unittest

from llm_client import LLMClient


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
