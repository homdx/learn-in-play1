"""EOS-2/COMPAT: шлюз отвергает "format": "json" → откат без падения."""
import io
import json
import unittest
import urllib.error

import llm_client
from llm_client import LLMClient


def _err400():
    return urllib.error.HTTPError(
        "https://api.ai/api/chat", 400, "Bad Request", {},
        io.BytesIO(b'{"status":400,"error":"Bad Request"}'))


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _ok(text='{"message":"hi"}'):
    return _Resp(json.dumps({"message": {"role": "a", "content": text},
                             "eval_count": 5}).encode())


class TestFormatFallback(unittest.TestCase):
    def setUp(self):
        llm_client._Breaker.failures = 0
        llm_client._ServerCaps.reset()
        self.c = LLMClient("https://api.ai", "k", "m", retries=2)
        self.payloads = []

    def _patch(self, responses):
        seq = list(responses)

        def fake_urlopen(req, timeout=None, context=None):
            self.payloads.append(json.loads(req.data.decode()))
            nxt = seq.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        llm_client.urllib.request.urlopen = fake_urlopen

    def test_400_falls_back_and_succeeds(self):
        self._patch([_err400(), _ok()])
        self.assertEqual(self.c.chat_json("s", "u"), {"message": "hi"})
        self.assertIn("format", self.payloads[0])
        self.assertNotIn("format", self.payloads[1])

    def test_flag_sticks_for_later_calls(self):
        """Второй вызов не должен снова нарываться на тот же 400."""
        self._patch([_err400(), _ok(), _ok()])
        self.c.chat_json("s", "u")
        self.c.chat_json("s", "u")
        self.assertNotIn("format", self.payloads[2])

    def test_other_400_still_raises(self):
        """КОНТРФАКТ: если format уже отключён, 400 — настоящая ошибка."""
        c = LLMClient("https://api.ai", "k", "m", retries=0, json_format=False)
        self._patch([_err400()])
        with self.assertRaises(RuntimeError):
            c.chat_json("s", "u")
        self.assertNotIn("format", self.payloads[0])

    def test_non_400_not_swallowed(self):
        e = urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"x"))
        self._patch([e])
        c = LLMClient("https://api.ai", "k", "m", retries=0)
        with self.assertRaises(RuntimeError):
            c.chat_json("s", "u")

    def test_config_can_disable_upfront(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "json_format": "false"},
                       "api_remote": {"base_url": "https://api.ai",
                                      "model": "m"}})
        c = LLMClient.from_config(cfg)
        self._patch([_ok()])
        c.chat_json("s", "u")
        self.assertNotIn("format", self.payloads[0])


class TestCapabilityIsShared(unittest.TestCase):
    """Главное: вывод про сервер общий для всех агентов.

    Каждый агент строит свой LLMClient (agent_v2.py:729). Пока флаг жил
    на экземпляре, пятеро игроков ловили один и тот же 400 пять раз —
    ровно то, что видно в логе.
    """

    def setUp(self):
        llm_client._Breaker.failures = 0
        llm_client._ServerCaps.reset()
        self.payloads = []

    def _patch(self, responses):
        seq = list(responses)

        def fake_urlopen(req, timeout=None, context=None):
            self.payloads.append(json.loads(req.data.decode()))
            nxt = seq.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        llm_client.urllib.request.urlopen = fake_urlopen

    def test_second_agent_does_not_repeat_the_400(self):
        a = LLMClient("https://api.ai", "k", "m", retries=2)
        b = LLMClient("https://api.ai", "k", "m", retries=2)
        self._patch([_err400(), _ok(), _ok()])
        a.chat_json("s", "u")          # платит за открытие
        b.chat_json("s", "u")          # должен пользоваться готовым
        self.assertNotIn("format", self.payloads[2],
                         "второй агент снова отправил format")
        self.assertEqual(len(self.payloads), 3,
                         "второй агент потратил лишний запрос на тот же 400")

    def test_other_server_unaffected(self):
        """КОНТРФАКТ: вывод про шлюз не должен калечить локальную ollama."""
        remote = LLMClient("https://api.ai", "k", "m", retries=2)
        local = LLMClient("http://localhost:11434", "k", "m", retries=2)
        self._patch([_err400(), _ok(), _ok()])
        remote.chat_json("s", "u")
        local.chat_json("s", "u")
        self.assertIn("format", self.payloads[2],
                      "локальный сервер зря лишили format")

    def test_config_optout_is_also_shared(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "json_format": "false"},
                       "api_remote": {"base_url": "https://api.ai",
                                      "model": "m"}})
        LLMClient.from_config(cfg)                       # только объявили
        other = LLMClient("https://api.ai", "k", "m")    # другой агент
        self._patch([_ok()])
        other.chat_json("s", "u")
        self.assertNotIn("format", self.payloads[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
