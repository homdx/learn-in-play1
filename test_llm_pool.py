"""Тесты llm_pool (POOL-1). python3 test_llm_pool.py"""
import configparser
import threading
import time
import unittest

import llm_pool
from llm_client import LLMUnavailable
from llm_pool import LLMPool, PooledClient, run_parallel


class FakeClient:
    def __init__(self, name, delay=0.0, fail_times=0, exc=None):
        self.base_url = f"http://{name}"
        self.model = name
        self.calls = 0
        self.delay = delay
        self.fail_times = fail_times
        self.exc = exc or RuntimeError("boom")

    def chat_json(self, *a, **kw):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.exc
        return {"from": self.model}


class TestPoolBasics(unittest.TestCase):
    def test_single_client_needs_no_pool(self):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "local"},
                       "api_local": {"base_url": "http://a", "model": "m"}})
        c = llm_pool.build_client(cfg)
        self.assertNotIsInstance(c, PooledClient)

    def test_pool_section_list(self):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "pool": "api_remote, api_r2"},
                       "api_remote": {"base_url": "http://a", "model": "m1"},
                       "api_r2": {"base_url": "http://b", "model": "m2"}})
        c = llm_pool.build_client(cfg)
        self.assertIsInstance(c, PooledClient)
        self.assertEqual(c.pool.size, 2)

    def test_missing_section_is_loud(self):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "pool": "api_remote, nope"},
                       "api_remote": {"base_url": "http://a", "model": "m"}})
        with self.assertRaises(ValueError):
            llm_pool.build_client(cfg)

    def test_pool_clients_default_to_http_error_retry(self):
        """HTTP-RETRY: без явной настройки клиент пула всё равно должен
        получить error_retries/error_retry_wait_sec — иначе пауза перед
        повтором на 402/5xx не применяется вовсе (терялась в обход
        LLMClient.from_config, см. _client_for_section)."""
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "pool": "api_remote, api_r2"},
                       "api_remote": {"base_url": "http://a", "model": "m1"},
                       "api_r2": {"base_url": "http://b", "model": "m2"}})
        clients, _ = llm_pool.clients_from_config(cfg)
        for c in clients:
            self.assertEqual(c.error_retries, 2)
            self.assertEqual(c.error_retry_wait_sec, 60)

    def test_pool_clients_read_custom_error_retry_settings(self):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "pool": "api_remote",
                              "error_retries": "5", "error_retry_wait_sec": "15"},
                       "api_remote": {"base_url": "http://a", "model": "m1"}})
        clients, _ = llm_pool.clients_from_config(cfg)
        self.assertEqual(clients[0].error_retries, 5)
        self.assertEqual(clients[0].error_retry_wait_sec, 15)

    def test_pool_clients_default_to_max_retry_after_cap(self):
        """RATE-3: без потолка второй подряд 429 на дневной лимит (TPD, не
        TPM) блокировал бы весь пул, вместо мгновенного failover на другой
        сервер — а пул строит клиентов в обход LLMClient.from_config(),
        так что дефолт конструктора применялся бы всегда, не значение из
        конфига."""
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "pool": "api_remote, api_r2"},
                       "api_remote": {"base_url": "http://a", "model": "m1"},
                       "api_r2": {"base_url": "http://b", "model": "m2"}})
        clients, _ = llm_pool.clients_from_config(cfg)
        for c in clients:
            self.assertEqual(c.max_retry_after_sec, 180)

    def test_pool_clients_read_custom_max_retry_after_cap(self):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"api": {"active": "remote", "pool": "api_remote",
                              "max_retry_after_sec": "30"},
                       "api_remote": {"base_url": "http://a", "model": "m1"}})
        clients, _ = llm_pool.clients_from_config(cfg)
        self.assertEqual(clients[0].max_retry_after_sec, 30)


class TestLease(unittest.TestCase):
    def test_two_calls_run_concurrently_on_two_servers(self):
        a, b = FakeClient("a", delay=0.15), FakeClient("b", delay=0.15)
        pc = PooledClient(LLMPool([a, b], ["a", "b"]))
        t0 = time.monotonic()
        run_parallel([lambda: pc.chat_json(), lambda: pc.chat_json()], workers=2)
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.28, "вызовы выполнились последовательно")
        self.assertEqual((a.calls, b.calls), (1, 1))

    def test_one_server_serialises(self):
        a = FakeClient("a", delay=0.1)
        pc = PooledClient(LLMPool([a], ["a"]))
        t0 = time.monotonic()
        run_parallel([lambda: pc.chat_json(), lambda: pc.chat_json()], workers=2)
        self.assertGreater(time.monotonic() - t0, 0.19,
                           "один сервер обслужил два вызова одновременно")

    def test_endpoint_not_shared_while_busy(self):
        seen = []
        lock = threading.Lock()

        class Spy(FakeClient):
            def chat_json(self, *a, **kw):
                with lock:
                    seen.append(("in", self.model))
                time.sleep(0.05)
                with lock:
                    seen.append(("out", self.model))
                return {}

        s = Spy("a")
        pc = PooledClient(LLMPool([s], ["a"]))
        run_parallel([lambda: pc.chat_json() for _ in range(3)], workers=3)
        # при корректной аренде in/out строго чередуются
        self.assertEqual([x[0] for x in seen],
                         ["in", "out"] * 3)


class TestFailover(unittest.TestCase):
    def test_failed_call_retries_on_other_server(self):
        bad = FakeClient("bad", fail_times=1)
        good = FakeClient("good")
        pc = PooledClient(LLMPool([bad, good], ["bad", "good"]))
        self.assertEqual(pc.chat_json(), {"from": "good"})
        self.assertEqual((bad.calls, good.calls), (1, 1))

    def test_no_failover_when_disabled(self):
        bad = FakeClient("bad", fail_times=5)
        good = FakeClient("good")
        pc = PooledClient(LLMPool([bad, good], ["bad", "good"]), max_failover=0)
        with self.assertRaises(RuntimeError):
            pc.chat_json()
        self.assertEqual(good.calls, 0)

    def test_unavailable_is_not_failed_over(self):
        """Предохранитель глобален — другой сервер его не лечит."""
        bad = FakeClient("bad", fail_times=1, exc=LLMUnavailable("down"))
        good = FakeClient("good")
        pc = PooledClient(LLMPool([bad, good], ["bad", "good"]))
        with self.assertRaises(LLMUnavailable):
            pc.chat_json()
        self.assertEqual(good.calls, 0)

    def test_repeated_failures_park_the_endpoint(self):
        bad = FakeClient("bad", fail_times=99)
        good = FakeClient("good")
        pc = PooledClient(LLMPool([bad, good], ["bad", "good"]))
        for _ in range(llm_pool.ENDPOINT_FAIL_THRESHOLD):
            pc.chat_json()
        before = bad.calls
        pc.chat_json()
        self.assertEqual(bad.calls, before,
                         "выведенный из ротации сервер снова получил вызов")

    def test_healthy_call_resets_fail_counter(self):
        ep = llm_pool._Endpoint(FakeClient("a"), "a")
        ep.note_fail()
        ep.note_ok()
        self.assertEqual(ep.fails, 0)


class TestRunParallel(unittest.TestCase):
    def test_sequential_when_single_worker(self):
        order = []
        tasks = [lambda i=i: order.append(i) for i in range(4)]
        run_parallel(tasks, workers=1)
        self.assertEqual(order, [0, 1, 2, 3])

    def test_results_keep_input_order(self):
        tasks = [lambda i=i: i for i in range(5)]
        self.assertEqual(run_parallel(tasks, workers=3), [0, 1, 2, 3, 4])

    def test_one_failure_does_not_cancel_others(self):
        def boom():
            raise ValueError("x")
        errs = []
        out = run_parallel([lambda: 1, boom, lambda: 3], workers=3,
                           on_error=errs.append)
        self.assertEqual(out, [1, None, 3])
        self.assertEqual(len(errs), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
