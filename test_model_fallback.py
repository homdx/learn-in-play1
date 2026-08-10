"""
test_model_fallback.py — MODEL-FALLBACK-1: список моделей через запятую в
[api_remote]/[api_local] model=..., с автоматическим переключением на
следующую модель, когда текущая исчерпывает СВОИ собственные ретраи.

Идея запроса: "if I setup models for api remote or local as array then
after all limits reached try other model and to finish last model. done
if all attempt connection failed" — то есть:
  1. model может быть массивом (в конфиге — строкой через запятую).
  2. Когда лимиты/ошибки на текущей модели исчерпаны — пробуем следующую.
  3. Пробуем ВСЕ модели по очереди, до последней.
  4. Если ВСЕ модели тоже отказали — считаем вызов завершённым неудачей
     (пробрасываем последнюю ошибку наверх, как раньше делала одна модель).

    python3 test_model_fallback.py [-v]
"""

import io
import json
import unittest
import urllib.error

import llm_client
from llm_client import LLMClient, LLMUnavailable


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _ok(text='{"ok": 1}'):
    body = {
        "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                 "total_tokens": 15},
    }
    return _Resp(json.dumps(body).encode())


def _err_conn():
    return urllib.error.URLError("Connection refused")


class ModelFallbackHarness(unittest.TestCase):
    def setUp(self):
        llm_client._Breaker.failures = 0
        llm_client._ServerCaps.reset()
        self._orig_sleep = llm_client.time.sleep
        llm_client.time.sleep = lambda s: None   # тесты не должны реально ждать

    def tearDown(self):
        llm_client.time.sleep = self._orig_sleep

    def _client(self, model, **kw):
        return LLMClient("https://api.example/v1", "k", model,
                         api_format="openai", retries=0, error_retries=0,
                         **kw)

    def _patch(self, responses):
        seq = list(responses)

        def fake_urlopen(req, timeout=None, context=None):
            nxt = seq.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        llm_client.urllib.request.urlopen = fake_urlopen


class TestModelListParsing(ModelFallbackHarness):

    def test_single_model_string_gives_a_list_of_one(self):
        c = self._client("model-a")
        self.assertEqual(c.models, ["model-a"])
        self.assertEqual(c.model, "model-a")

    def test_comma_separated_models_are_split(self):
        c = self._client("model-a, model-b , model-c")
        self.assertEqual(c.models, ["model-a", "model-b", "model-c"])
        self.assertEqual(c.model, "model-a", "первая модель — активная по умолчанию")

    def test_no_behaviour_change_for_a_single_model(self):
        """Один элемент без запятой — поведение НЕ должно отличаться от
        версии кода без этой фичи (regression guard)."""
        c = self._client("solo-model")
        self._patch([_ok()])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"ok": 1})
        self.assertEqual(c.model, "solo-model")

    def test_duplicate_model_names_are_deduplicated(self):
        """MODEL-FALLBACK-3: реальный случай — конфиг пользователя дал
        'gemini-3.6-flash, gemini-3.6-flash, gemini-3.5-flash' (дубликат
        по опечатке при копировании). Без дедупликации "переключение на
        следующую модель" честно переключалось на строку с ТЕМ ЖЕ именем
        — лишний полный цикл ретраев (~7 минут на free tier Gemini) на
        заведомо мёртвой модели, прежде чем дойти до реально другой."""
        c = self._client("gemini-3.6-flash, gemini-3.6-flash, gemini-3.5-flash")
        self.assertEqual(c.models, ["gemini-3.6-flash", "gemini-3.5-flash"],
                         "дубликат должен быть выброшен, порядок сохранён")
        self.assertEqual(c.model, "gemini-3.6-flash")

    def test_deduplication_preserves_first_occurrence_order(self):
        c = self._client("b, a, b, c, a")
        self.assertEqual(c.models, ["b", "a", "c"])

    def test_all_identical_models_collapse_to_one(self):
        """Крайний случай: список из одних дубликатов — должен схлопнуться
        в единственную модель, а не пытаться 'переключаться' сам на себя."""
        c = self._client("solo, solo, solo")
        self.assertEqual(c.models, ["solo"])


class TestFallbackOnExhaustedRetries(ModelFallbackHarness):

    def test_second_model_used_after_first_exhausts_its_retries(self):
        c = self._client("model-a,model-b")
        self._patch([_err_conn(), _ok('{"from":"b"}')])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"from": "b"})
        self.assertEqual(c.model, "model-b", "активная модель должна была переключиться")

    def test_all_models_tried_in_order_before_giving_up(self):
        c = self._client("model-a,model-b,model-c")
        self._patch([_err_conn(), _err_conn(), _ok('{"from":"c"}')])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"from": "c"})
        self.assertEqual(c.model, "model-c")

    def test_all_models_failing_raises_the_last_error(self):
        """'done if all attempt connection failed' — если ни одна модель
        не ответила, вызов завершается неудачей (исключение прокидывается
        наверх), а не тихой заглушкой внутри клиента."""
        c = self._client("model-a,model-b")
        self._patch([_err_conn(), _err_conn()])
        with self.assertRaises(Exception):
            c.chat_json("s", "u")

    def test_model_actually_used_in_the_request_payload(self):
        c = self._client("model-a,model-b")
        sent_models = []

        def fake_urlopen(req, timeout=None, context=None):
            body = json.loads(req.data.decode())
            sent_models.append(body.get("model"))
            if len(sent_models) == 1:
                raise _err_conn()
            return _ok('{"ok":1}')
        llm_client.urllib.request.urlopen = fake_urlopen

        c.chat_json("s", "u")
        self.assertEqual(sent_models, ["model-a", "model-b"])


class TestFallbackOnLLMUnavailable(ModelFallbackHarness):
    """Шесть подряд сбоев ОДНОЙ модели поднимают LLMUnavailable
    (предохранитель) — но это не должно мешать попробовать СЛЕДУЮЩУЮ
    модель списка, раз лимиты у провайдеров (Groq TPM/TPD и т.п.) обычно
    считаются ПО МОДЕЛИ, а не по серверу целиком."""

    def test_breaker_trip_on_first_model_still_tries_second(self):
        c = self._client("model-a,model-b")
        threshold = llm_client._Breaker.threshold
        # Один сбой на модель = один инкремент предохранителя (RETRY-1
        # ретраит ВНУТРИ модели без вызова _note_failure, он срабатывает
        # только на итоговом исчерпании). Подводим счётчик к порогу-1,
        # чтобы ровно сбой model-a перевалил через threshold и поднял
        # LLMUnavailable — именно тот случай, который должен всё равно
        # дать шанс model-b.
        llm_client._Breaker.failures = threshold - 1
        self._patch([_err_conn(), _ok('{"from":"b"}')])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"from": "b"})
        self.assertEqual(c.model, "model-b")

    def test_llm_unavailable_on_last_model_still_propagates(self):
        c = self._client("model-a,model-b")
        threshold = llm_client._Breaker.threshold
        llm_client._Breaker.failures = threshold - 1
        # Оба сбоя — оба перешагивают порог (после успешного model-a-сбоя
        # предохранитель сбросится ТОЛЬКО при успехе, а его тут нет, так
        # что счётчик продолжит расти и на model-b тоже поднимет LLMUnavailable).
        self._patch([_err_conn(), _err_conn()])
        with self.assertRaises(LLMUnavailable):
            c.chat_json("s", "u")


class TestOnRetryCallbackMentionsModelSwitch(ModelFallbackHarness):

    def test_on_retry_is_called_when_switching_models(self):
        messages = []
        c = self._client("model-a,model-b", on_retry=messages.append)
        self._patch([_err_conn(), _ok()])
        c.chat_json("s", "u")
        self.assertTrue(any("model-b" in m for m in messages),
                        f"нет упоминания переключения модели: {messages}")


class TestFromConfigReadsModelList(unittest.TestCase):

    def test_comma_separated_model_in_ini_becomes_a_list(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_dict({
            "api": {"active": "remote"},
            "api_remote": {"base_url": "https://api.groq.com/openai/v1",
                          "model": "qwen/qwen3.6-27b, llama-3.3-70b-versatile",
                          "api_format": "openai"},
        })
        c = LLMClient.from_config(cfg)
        self.assertEqual(c.models, ["qwen/qwen3.6-27b",
                                    "llama-3.3-70b-versatile"])

    def test_single_model_in_ini_unaffected(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_dict({
            "api": {"active": "local"},
            "api_local": {"base_url": "http://localhost:11434",
                         "model": "qwen3:8b"},
        })
        c = LLMClient.from_config(cfg)
        self.assertEqual(c.models, ["qwen3:8b"])


class TestModelPositionPersistsAcrossCalls(ModelFallbackHarness):
    """MODEL-FALLBACK-2: реальный случай — три модели в списке, первая
    (Gemini free tier) исчерпала СУТОЧНУЮ квоту. Каждый новый вызов
    chat_json() — это ОТДЕЛЬНОЕ решение агента (план раунда, следующий
    ход, реплика), не один вызов на весь раунд. Раньше каждый такой
    вызов заново начинал перебор с models[0] — то есть заново тратил
    ~6-7 минут на попытки мёртвой модели, прежде чем снова дойти до
    рабочей. Со стороны это выглядело как "ходит по кругу", и было им
    буквально. Эти тесты проверяют, что ВТОРОЙ вызов chat_json() того же
    (или другого) клиента на тот же base_url начинает СРАЗУ с последней
    рабочей модели, а не с нуля."""

    def test_second_call_starts_from_last_successful_model(self):
        c = self._client("model-a,model-b,model-c")
        # Первый вызов: model-a мертва, model-b отвечает — успех на b.
        self._patch([_err_conn(), _ok('{"call":1}')])
        r1 = c.chat_json("s", "u")
        self.assertEqual(r1, {"call": 1})
        self.assertEqual(c.model, "model-b")

        # Второй вызов — ДОЛЖЕН начаться сразу с model-b, а не с model-a
        # заново. Патчим ОДИН ответ: если бы код полез в model-a первой,
        # этого единственного ответа не хватило бы и тест бы упал с
        # IndexError на пустой очереди ответов.
        self._patch([_ok('{"call":2}')])
        r2 = c.chat_json("s", "u")
        self.assertEqual(r2, {"call": 2})
        self.assertEqual(c.model, "model-b",
                         "второй вызов должен был начать сразу с model-b")

    def test_position_shared_across_different_client_instances(self):
        """Позиция хранится в _ServerCaps по base_url — общая для всех
        клиентов на этот сервер, не только для одного экземпляра
        LLMClient (у каждого игрока свой клиент, но один и тот же
        удалённый сервер и один и тот же список моделей)."""
        c1 = self._client("model-a,model-b")
        self._patch([_err_conn(), _ok('{"who":1}')])
        c1.chat_json("s", "u")

        c2 = self._client("model-a,model-b")   # новый клиент, тот же base_url
        self.assertEqual(c2.base_url, c1.base_url)
        self._patch([_ok('{"who":2}')])   # только ОДИН ответ — сразу model-b
        result = c2.chat_json("s", "u")
        self.assertEqual(result, {"who": 2})
        self.assertEqual(c2.model, "model-b")

    def test_full_pass_failure_resets_position_to_zero(self):
        """Если ВСЕ модели прохода провалились — следующий внешний вызов
        не должен залипнуть на последней модели прохода (там нечего
        пробовать дальше, вернее начать заново с первой)."""
        c = self._client("model-a,model-b")
        self._patch([_err_conn(), _err_conn()])
        with self.assertRaises(Exception):
            c.chat_json("s", "u")
        self.assertEqual(llm_client._ServerCaps.get_model_index(c.base_url), 0)

        # Следующий вызов начинает с model-a заново (могло пройти время,
        # квота могла восстановиться) — не с несуществующего "следующего
        # после последней".
        self._patch([_ok('{"recovered": true}')])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"recovered": True})
        self.assertEqual(c.model, "model-a")

    def test_wraps_around_when_starting_position_is_not_the_first(self):
        """Если стартуем с model-c (последняя по списку) и она тоже
        отказала — перебор должен закольцеваться на model-a, а не
        остановиться, будто список кончился."""
        c = self._client("model-a,model-b,model-c")
        llm_client._ServerCaps.set_model_index(c.base_url, 2)   # model-c
        self._patch([_err_conn(), _ok('{"wrapped": true}')])
        result = c.chat_json("s", "u")
        self.assertEqual(result, {"wrapped": True})
        self.assertEqual(c.model, "model-a",
                         "после model-c по кругу должна идти model-a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
