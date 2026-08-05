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


if __name__ == "__main__":
    unittest.main(verbosity=2)
