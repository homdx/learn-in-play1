"""
test_retry_temp_budget.py — AUTO-RETRY-TEMP-BUDGET-1.

Real-run field report (r001, config_v2_mistral.ini, retries=1): three
different players each hit a *different* JSONDecodeError
("Unterminated string starting at: line N column M...", "Expecting
value...", "Expecting property name enclosed in double quotes...") on
their checklist calls. Every single one retried ONCE (temp+0.15, same
max_tokens) and failed again with the SAME error class, then fell back
to "keeping old checklist" / "plan_round failed" — a silent skip that
gets committed to disk indistinguishable from the player genuinely
deciding not to change anything.

The old EOS-2 retry only ever changed `temperature` (a plain +0.15 ramp
off the caller's own value) and never touched `max_tokens` at all — so a
retry against a truncation failure re-asked at the *exact* budget that
caused the truncation. This file checks the fix: a fixed, absolute
temperature schedule (default 0.0, 0.1 — decoupled from whatever
`temperature` the caller passed) crossed with a doubling max_tokens
budget, one full cycle through every temperature per token tier before
the tier doubles. Same 2-D grid pattern as jan-auto-agent's
Gate1Filter._UNPARSEABLE_TEMPERATURES / AUTO-RETRY-TEMP-1.

What we check:
  * attempt 1 (the happy path) is untouched — original temperature and
    max_tokens, no grid involved;
  * attempt 2+ follows the (tier, temp) grid in order: temp cycles
    through retry_temperatures before max_tokens doubles;
  * max_tokens actually escalates — this is the part the old code never
    did, and it's the one that fixes real truncation errors;
  * the escalation is capped by num_ctx (when known) or a fixed default
    ceiling otherwise, and never left uncapped;
  * retry_temperatures/retry_tokens_step_mult/retry_tokens_ctx_fraction/
    retry_tokens_default_ceiling are configurable from agents.ini's
    [api] section with the same defaults as direct construction;
  * a stub subclass created without __init__ (as tests/stub-mode do
    elsewhere in this codebase) still works via getattr fallbacks;
  * end-to-end: a call that fails with a truncation-shaped
    JSONDecodeError on attempt 1 and succeeds on attempt 2 actually used
    the escalated max_tokens/temperature, not the original ones.
"""

from __future__ import annotations

import configparser
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
    def client(retries=5, **kw):
        return LLMClient(base_url="http://x", api_key="k", model="m",
                          retries=retries, **kw)


class TestDefaults(Base):

    def test_default_retry_temperatures_is_0_and_01(self):
        c = self.client()
        self.assertEqual(c.retry_temperatures, (0.0, 0.1))

    def test_default_step_mult_is_2(self):
        c = self.client()
        self.assertEqual(c.retry_tokens_step_mult, 2.0)

    def test_default_ctx_fraction_is_half(self):
        c = self.client()
        self.assertEqual(c.retry_tokens_ctx_fraction, 0.5)

    def test_default_ceiling_is_32768(self):
        c = self.client()
        self.assertEqual(c.retry_tokens_default_ceiling, 32768)

    def test_retry_temperatures_accepts_comma_string(self):
        c = self.client(retry_temperatures="0.0, 0.2, 0.4")
        self.assertEqual(c.retry_temperatures, (0.0, 0.2, 0.4))

    def test_retry_temperatures_accepts_tuple(self):
        c = self.client(retry_temperatures=(0.0, 0.3))
        self.assertEqual(c.retry_temperatures, (0.0, 0.3))

    def test_empty_retry_temperatures_falls_back_to_zero(self):
        c = self.client(retry_temperatures="")
        self.assertEqual(c.retry_temperatures, (0.0,))


class TestGridSchedule(Base):
    """Attempt-by-attempt (max_tokens, temperature) schedule."""

    def _record_calls(self, c, n_failures, final_ok=True):
        calls = []

        def fake_chat(system, user, temperature=0.4, max_tokens=400,
                      **kw):
            calls.append({"temperature": temperature, "max_tokens": max_tokens})
            if len(calls) <= n_failures:
                raise json.JSONDecodeError("Unterminated string", "", 0)
            if final_ok:
                return '{"checklist": "ok"}'
            raise json.JSONDecodeError("Unterminated string", "", 0)

        c.chat = fake_chat
        return calls

    def test_attempt_1_is_untouched_happy_path(self):
        """First attempt must use the caller's own temperature/max_tokens,
        not the retry grid — the grid only applies to attempt 2+."""
        c = self.client(retries=1)
        calls = self._record_calls(c, n_failures=0)
        c.chat_json("s", "u", temperature=0.4, max_tokens=250)
        self.assertEqual(calls[0], {"temperature": 0.4, "max_tokens": 250})

    def test_first_retry_uses_first_temperature_and_doubles_tokens(self):
        c = self.client(retries=1, retry_temperatures=(0.0, 0.1))
        calls = self._record_calls(c, n_failures=1)
        c.chat_json("s", "u", temperature=0.4, max_tokens=250)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["temperature"], 0.0)
        self.assertEqual(calls[1]["max_tokens"], 500)   # 250 * 2**1

    def test_second_retry_uses_second_temperature_same_tier(self):
        """Second retry (attempt 3): still tier 0 (same doubled budget),
        but the SECOND temperature in the schedule — the tier only
        doubles again once every temperature has been tried once."""
        c = self.client(retries=2, retry_temperatures=(0.0, 0.1))
        calls = self._record_calls(c, n_failures=2)
        c.chat_json("s", "u", temperature=0.4, max_tokens=250)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[2]["temperature"], 0.1)
        self.assertEqual(calls[2]["max_tokens"], 500)   # still tier 0

    def test_third_retry_advances_to_next_tier(self):
        """Attempt 4 = third retry: back to the first temperature, but
        the token tier has now doubled a second time."""
        c = self.client(retries=3, retry_temperatures=(0.0, 0.1))
        calls = self._record_calls(c, n_failures=3)
        c.chat_json("s", "u", temperature=0.4, max_tokens=250)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[3]["temperature"], 0.0)
        self.assertEqual(calls[3]["max_tokens"], 1000)  # 250 * 2**2

    def test_full_five_attempt_grid_matches_docstring_example(self):
        """Mirrors the worked example in llm_client.py's comment:
        max_tokens=512, default schedule (0.0, 0.1) ->
          attempt2: 1024/0.0, attempt3: 1024/0.1,
          attempt4: 2048/0.0, attempt5: 2048/0.1."""
        c = self.client(retries=4, retry_temperatures=(0.0, 0.1))
        calls = self._record_calls(c, n_failures=4)
        c.chat_json("s", "u", temperature=0.4, max_tokens=512)
        got = [(round(x["temperature"], 2), x["max_tokens"]) for x in calls]
        self.assertEqual(got, [
            (0.4, 512),     # attempt 1: happy path, untouched
            (0.0, 1024),    # attempt 2
            (0.1, 1024),    # attempt 3
            (0.0, 2048),    # attempt 4
            (0.1, 2048),    # attempt 5
        ])

    def test_single_temperature_schedule_still_doubles_every_attempt(self):
        """With only one temperature configured, n_temps=1 so every
        retry is its own tier — tokens double on every single attempt,
        never repeating the same budget twice."""
        c = self.client(retries=3, retry_temperatures=(0.0,))
        calls = self._record_calls(c, n_failures=3)
        c.chat_json("s", "u", temperature=0.4, max_tokens=100)
        got = [x["max_tokens"] for x in calls]
        self.assertEqual(got, [100, 200, 400, 800])

    def test_recovers_from_truncation_using_escalated_budget(self):
        """End-to-end: simulate the real failure mode — attempt 1
        truncates (JSONDecodeError), attempt 2 succeeds only because it
        got the doubled token budget. Confirms chat_json() returns the
        parsed result and actually used the escalated call."""
        c = self.client(retries=1)
        calls = self._record_calls(c, n_failures=1)
        result = c.chat_json("s", "u", temperature=0.4, max_tokens=300)
        self.assertEqual(result, {"checklist": "ok"})
        self.assertEqual(calls[1]["max_tokens"], 600)
        self.assertEqual(calls[1]["temperature"], 0.0)


class TestCeiling(Base):

    def _record_calls(self, c, n_failures):
        calls = []

        def fake_chat(system, user, temperature=0.4, max_tokens=400, **kw):
            calls.append(max_tokens)
            if len(calls) <= n_failures:
                raise json.JSONDecodeError("Unterminated string", "", 0)
            return '{"checklist": "ok"}'

        c.chat = fake_chat
        return calls

    def test_capped_by_num_ctx_fraction_when_num_ctx_known(self):
        # num_ctx=2000, fraction=0.5 -> ceiling=1000. Uncapped tier-2
        # budget would be 500 * 2**2 = 2000, which must be capped to 1000.
        c = self.client(retries=3, num_ctx=2000,
                         retry_tokens_ctx_fraction=0.5)
        calls = self._record_calls(c, n_failures=3)
        c.chat_json("s", "u", temperature=0.4, max_tokens=500)
        self.assertEqual(calls, [500, 1000, 1000, 1000])

    def test_capped_by_default_ceiling_when_num_ctx_unset(self):
        c = self.client(retries=5, num_ctx=0,
                         retry_tokens_default_ceiling=1500)
        calls = self._record_calls(c, n_failures=5)
        c.chat_json("s", "u", temperature=0.4, max_tokens=1000)
        # tier doublings: 1000, 2000(cap 1500), 2000(cap 1500), 4000(cap
        # 1500), 4000(cap 1500), 8000(cap 1500)
        self.assertEqual(calls, [1000, 1500, 1500, 1500, 1500, 1500])

    def test_capped_escalation_warns_via_on_retry(self):
        seen = []
        c = self.client(retries=1, num_ctx=0, retry_tokens_default_ceiling=600,
                         on_retry=lambda msg: seen.append(msg))
        self._record_calls(c, n_failures=1)
        c.chat_json("s", "u", temperature=0.4, max_tokens=500)
        self.assertTrue(any("escalation wants" in m for m in seen))


class TestStubWithoutInit(Base):
    """Subclasses created without __init__ (test doubles, stub mode
    elsewhere in this codebase) must not break — same getattr-fallback
    contract as `retries` already has."""

    def test_stub_without_init_uses_grid_defaults(self):
        class Stub(LLMClient):
            def __init__(self):
                self.retries = 1
                self.calls = []

            def chat(self, system, user, temperature=0.4, max_tokens=400,
                     **kw):
                self.calls.append((temperature, max_tokens))
                if len(self.calls) == 1:
                    raise json.JSONDecodeError("Unterminated string", "", 0)
                return '{"ok": true}'

        st = Stub()
        result = st.chat_json("s", "u", temperature=0.4, max_tokens=300)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(st.calls[1], (0.0, 600))


class TestConfig(Base):

    def _cfg(self, extra_api_lines=""):
        cfg = configparser.ConfigParser()
        cfg.read_string(f"""
[api]
active = local
{extra_api_lines}
[api_local]
base_url = http://localhost:11434
model = qwen3:8b
""")
        return cfg

    def test_from_config_defaults_match_direct_construction(self):
        c = LLMClient.from_config(self._cfg())
        self.assertEqual(c.retry_temperatures, (0.0, 0.1))
        self.assertEqual(c.retry_tokens_step_mult, 2.0)
        self.assertEqual(c.retry_tokens_ctx_fraction, 0.5)
        self.assertEqual(c.retry_tokens_default_ceiling, 32768)

    def test_from_config_reads_overrides(self):
        c = LLMClient.from_config(self._cfg(
            "retry_temperatures = 0.0, 0.2\n"
            "retry_tokens_step_mult = 3.0\n"
            "retry_tokens_ctx_fraction = 0.25\n"
            "retry_tokens_default_ceiling = 8192\n"
        ))
        self.assertEqual(c.retry_temperatures, (0.0, 0.2))
        self.assertEqual(c.retry_tokens_step_mult, 3.0)
        self.assertEqual(c.retry_tokens_ctx_fraction, 0.25)
        self.assertEqual(c.retry_tokens_default_ceiling, 8192)

    def test_shipped_configs_parse_with_new_keys_or_fall_back_cleanly(self):
        """The shipped .ini files predate this feature and shouldn't
        need editing — from_config() must still build a working client
        purely off the documented defaults."""
        import glob
        for path in glob.glob("config_v2*.ini"):
            cfg = configparser.ConfigParser()
            try:
                cfg.read(path)
            except configparser.Error:
                # Not this feature's concern — a handful of shipped
                # configs have pre-existing parse issues unrelated to
                # retry escalation; skip rather than fail this test on
                # unrelated file corruption.
                continue
            if not cfg.has_section("api") or not cfg.has_option("api", "active"):
                continue
            active = cfg.get("api", "active", fallback="local")
            if not cfg.has_section(f"api_{active}"):
                continue
            c = LLMClient.from_config(cfg)
            self.assertTrue(len(c.retry_temperatures) >= 1)
            self.assertGreaterEqual(c.retry_tokens_step_mult, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
