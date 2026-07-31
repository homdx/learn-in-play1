"""
test_speech_cost.py — контрфактические тесты для TALK-1 (платная речь).

Каждый тест обязан ПАДАТЬ на коде без speech_cost.py и проходить на коде
с ним. Сеть не нужна.

    python3 test_speech_cost.py
    python3 test_speech_cost.py -v
"""

import configparser
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_v2
import common
import llm_client
import run_game_v2
import speech_cost
from agent_v2 import PlayerAgent, save_balance

PLAYERS = ["player1", "player2"]
LINE = 80


# ────────────────────────────────────────────────────── fake LLM ──────────

class FakeLLM:
    """Отдаёт заранее заданные реплики по очереди."""

    script: list = []
    idx = 0
    calls: list = []

    @classmethod
    def from_config(cls, cfg, section=None):
        return cls()

    def chat_json(self, system, user, temperature=0.0, max_tokens=0):
        FakeLLM.calls.append({"system": system, "user": user})
        if "Dialogue turn" in system:
            i = min(FakeLLM.idx, len(FakeLLM.script) - 1)
            FakeLLM.idx += 1
            return dict(FakeLLM.script[i])
        if "Update reputation" in system:
            return {"trust_score": 5, "deal_done": None, "deal_failed": None,
                    "reputation_note": "n", "future_intent": "i", "summary": "s"}
        if "Compress" in system:
            return {"notes": "c", "compressed_history": "c"}
        return {}

    @classmethod
    def reset(cls, script=None):
        cls.script = script or [{"message": "hi", "transfer": 0,
                                 "transfer_to": None, "done": True}]
        cls.idx = 0
        cls.calls = []


def make_cfg(table_dir, tariff_section=None):
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "api":       {"active": "local"},
        "api_local": {"base_url": "http://unused", "model": "fake"},
        "game":      {"table_dir": table_dir, "logs_dir": table_dir,
                      "players": ",".join(PLAYERS), "start_balance": "100",
                      "rounds": "1", "max_bet_fraction": "0.4",
                      "round_delay_sec": "0"},
        "player":    {"temperature": "0.5", "history_window": "10"},
    })
    if tariff_section is not None:
        cfg.read_dict({"dialogue_cost": tariff_section})
    return cfg


class Harness(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="speech_test_")
        self.table = os.path.join(self.tmp, "table")
        os.makedirs(self.table, exist_ok=True)
        self._orig = agent_v2.LLMClient
        agent_v2.LLMClient = FakeLLM
        llm_client.LLMClient = FakeLLM
        FakeLLM.reset()

    def tearDown(self):
        agent_v2.LLMClient = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def agents(self, tariff_section=None, balances=(100, 100)):
        cfg = make_cfg(self.table, tariff_section)
        tariff = speech_cost.parse_tariff(cfg)
        out = []
        for pid, bal in zip(PLAYERS, balances):
            save_balance(pid, self.table, bal)
            a = PlayerAgent(pid, self.table, cfg, tariff=tariff)
            a.balance = bal
            out.append(a)
        return out[0], out[1], tariff

    def dialogue(self, a, b, round_no=1):
        class NullLogger:
            def write(self, *args, **kw): pass
            def write_global(self, *args, **kw): pass
            def write_dialogue(self, *args, **kw): pass
        return run_game_v2.run_dialogue(a, b, round_no, NullLogger(), self.table)


# ─────────────────────────────────────────────── 1. арифметика тарифа ─────

class TestTariffMath(unittest.TestCase):

    def setUp(self):
        self.t = speech_cost.SpeechTariff(True, LINE, 1)

    def test_empty_message_is_free(self):
        """Плата за многословие, а не налог на участие: молчание бесплатно."""
        self.assertEqual(self.t.cost_of(""), 0)
        self.assertEqual(self.t.cost_of("   \n  "), 0)

    def test_one_char_costs_one_line(self):
        self.assertEqual(self.t.cost_of("x"), 1)

    def test_exactly_one_line_costs_one(self):
        self.assertEqual(self.t.cost_of("x" * LINE), 1)

    def test_one_char_over_costs_two(self):
        """Начатая строка считается целиком — ключевое условие задачи."""
        self.assertEqual(self.t.cost_of("x" * (LINE + 1)), 2)

    def test_one_and_a_half_lines_costs_two(self):
        self.assertEqual(self.t.cost_of("x" * (LINE + LINE // 2)), 2)

    def test_spaces_are_counted(self):
        """Строка консоли считается вместе с пробелами."""
        msg = "a " * (LINE // 2)          # 80 символов с пробелами
        self.assertEqual(len(msg.strip()), LINE - 1)
        self.assertEqual(self.t.cost_of(msg), 1)

    def test_coins_per_line_multiplier(self):
        t = speech_cost.SpeechTariff(True, LINE, 3)
        self.assertEqual(t.cost_of("x" * (LINE + 1)), 6)

    def test_disabled_tariff_is_always_free(self):
        t = speech_cost.SpeechTariff(False, LINE, 1)
        self.assertEqual(t.cost_of("x" * 1000), 0)


# ──────────────────────────────────────────────── 2. разбор конфига ───────

class TestTariffConfig(Harness):

    def test_absent_section_disabled(self):
        self.assertFalse(speech_cost.parse_tariff(make_cfg(self.table)).enabled)

    def test_enabled_false_disabled(self):
        t = speech_cost.parse_tariff(make_cfg(self.table, {"enabled": "false"}))
        self.assertFalse(t.enabled)

    def test_defaults_are_80_and_1(self):
        t = speech_cost.parse_tariff(make_cfg(self.table, {"enabled": "true"}))
        self.assertEqual(t.chars_per_line, 80)
        self.assertEqual(t.coins_per_line, 1)

    def test_zero_chars_per_line_rejected(self):
        """0 означал бы деление на ноль / бесконечную цену — ловим на старте."""
        with self.assertRaises(ValueError):
            speech_cost.parse_tariff(
                make_cfg(self.table, {"enabled": "true", "chars_per_line": "0"}))

    def test_negative_coins_rejected(self):
        with self.assertRaises(ValueError):
            speech_cost.parse_tariff(
                make_cfg(self.table, {"enabled": "true", "coins_per_line": "-1"}))


# ────────────────────────────────────────── 3. списание в диалоге ─────────

ON = {"enabled": "true"}


class TestChargingInDialogue(Harness):

    def test_no_charge_when_disabled(self):
        """Регрессия: без фичи балансы не меняются от разговора."""
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents()
        self.dialogue(a, b)
        self.assertEqual(a.balance, 100)
        self.assertEqual(b.balance, 100)

    def test_speech_is_charged_when_enabled(self):
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b)
        self.assertLess(a.balance, 100)

    def test_charge_matches_line_count(self):
        msg = "x" * (LINE * 2 + 1)          # 3 начатые строки
        FakeLLM.reset([{"message": msg, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        res = self.dialogue(a, b)
        self.assertEqual(res["a_speech_cost"], 3)
        self.assertEqual(a.balance, 97)

    def test_money_goes_to_casino_not_to_partner(self):
        """Суть механики: капитал стола уменьшается. Если бы платили друг
        другу, болтливость была бы просто способом дарить монеты."""
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        before = a.balance + b.balance
        self.dialogue(a, b)
        self.assertLess(a.balance + b.balance, before)

    def test_balances_persisted_to_disk(self):
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b)
        on_disk = json.load(open(os.path.join(self.table,
                                              "balance_player1.json")))["balance"]
        self.assertEqual(on_disk, a.balance)

    def test_longer_dialogue_costs_more(self):
        """Главный стимул: за каждый лишний ход платят."""
        short = [{"message": "deal", "transfer": 0,
                  "transfer_to": None, "done": True}]
        FakeLLM.reset(short)
        a, b, _ = self.agents(ON)
        self.dialogue(a, b)
        cheap = 200 - (a.balance + b.balance)

        long = [{"message": "x" * 300, "transfer": 0,
                 "transfer_to": None, "done": False}]
        FakeLLM.reset(long)
        c, d, _ = self.agents(ON)
        self.dialogue(c, d)
        expensive = 200 - (c.balance + d.balance)
        self.assertGreater(expensive, cheap)


# ────────────────────────────── 4. тариф не должен ломать сделки ──────────

class TestDealsUnaffected(Harness):

    def test_transfer_goes_through_before_fee(self):
        """Требование задачи: перевод 10 + отдельно 1-2 монеты за строки.
        Если снимать плату ДО перевода, игрок с балансом ровно в размер
        сделки не смог бы её оплатить — тариф ломал бы то поведение,
        которое должен поощрять."""
        FakeLLM.reset([{"message": "x" * 200, "transfer": 10,
                        "transfer_to": "player2", "done": True}])
        a, b, _ = self.agents(ON, balances=(10, 100))
        res = self.dialogue(a, b)
        self.assertEqual(res["a_sent"], 10)          # сделка состоялась
        # партнёр получил все 10 монет; из его баланса ушла только плата
        # за ЕГО собственную закрывающую реплику
        self.assertEqual(b.balance, 110 - res["b_speech_cost"])
        self.assertEqual(a.balance, 0)               # плата взята с остатка

    def test_fee_never_pushes_balance_below_zero(self):
        """Отрицательный баланс молча сломал бы долю ставки и порог
        банкротства, рассчитанные на неотрицательные значения."""
        FakeLLM.reset([{"message": "x" * 1000, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON, balances=(2, 100))
        self.dialogue(a, b)
        self.assertEqual(a.balance, 0)

    def test_unpaid_remainder_is_reported(self):
        t = speech_cost.SpeechTariff(True, LINE, 1)

        class Stub:
            player_id = "px"
            balance = 2
        res = speech_cost.charge(Stub(), "x" * (LINE * 5), t, self.table,
                                 lambda *a: None)
        self.assertEqual(res["charged"], 2)
        self.assertEqual(res["unpaid"], 3)


# ───────────────────────────────────── 5. видимость для игрока ────────────

class TestVisibility(Harness):

    def test_rule_appears_in_dialogue_system_prompt(self):
        FakeLLM.reset([{"message": "hi", "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b)
        sys_prompts = [c["system"] for c in FakeLLM.calls
                       if "Dialogue turn" in c["system"]]
        self.assertTrue(sys_prompts)
        self.assertIn("SPEECH COSTS MONEY", sys_prompts[0])
        self.assertIn("CASINO", sys_prompts[0].upper())

    def test_rule_absent_when_disabled(self):
        FakeLLM.reset([{"message": "hi", "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents()
        self.dialogue(a, b)
        for c in FakeLLM.calls:
            self.assertNotIn("SPEECH COSTS MONEY", c["system"])

    def test_running_spend_shown_to_player(self):
        """Игрок должен ВИДЕТЬ, что деньги ушли, иначе учиться не на чем."""
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": False}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b)
        users = [c["user"] for c in FakeLLM.calls
                 if "Dialogue turn" in c["system"]]
        later = [u for u in users if "Speech billing" in u]
        self.assertTrue(later, "counter never shown to the player")
        self.assertTrue(any("already paid" in u for u in later))

    def test_counter_resets_between_dialogues(self):
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b, round_no=1)
        self.assertGreater(a.speech_spent_this_dialogue, 0)
        FakeLLM.reset([{"message": "y", "transfer": 0,
                        "transfer_to": None, "done": True}])
        self.dialogue(a, b, round_no=2)
        self.assertEqual(a.speech_spent_this_dialogue, 1)

    def test_move_hint_in_next_move_prompt(self):
        """Решение «говорить или ставить» должно приниматься с учётом цены."""
        cfg = make_cfg(self.table, ON)
        a = PlayerAgent("player1", self.table, cfg,
                        tariff=speech_cost.parse_tariff(cfg))
        FakeLLM.reset()
        a.decide_next_move(["player2"], [], 1, [])
        moves = [c["user"] for c in FakeLLM.calls
                 if "Decide next move" in c["system"]]
        self.assertTrue(moves)
        self.assertIn("talking is not free", moves[0])


# ─────────────────────────────────────────── 6. запись в артефакты ────────

class TestArtifacts(Harness):

    def test_dlg_json_records_per_message_and_totals(self):
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b)
        path = os.path.join(self.table, "dlg_r001_player1_player2.json")
        d = common.read_json(path)
        self.assertIn("a_speech_cost", d)
        self.assertGreater(d["a_speech_cost"], 0)
        self.assertIn("speech_cost", d["conversation"][0])
        self.assertIn("speech_lines", d["conversation"][0])

    def test_summary_line_is_loggable(self):
        on = speech_cost.parse_tariff(make_cfg(self.table, ON))
        off = speech_cost.parse_tariff(make_cfg(self.table))
        self.assertIn("80", on.summary())
        self.assertIn("casino", on.summary())
        self.assertIn("disabled", off.summary())


# ──────────────────── 7. видимость расхода ДЛЯ ВЫВОДА ПРАВИЛА (TALK-2) ────
#
# Списание, видное только в момент оплаты, не порождает правила. Эти тесты
# закрывают три канала, через которые агент может связать «много слов» с
# «меньше денег» и записать это себе на будущее.

class TestLearnability(Harness):

    def test_each_message_shows_its_own_price_in_transcript(self):
        """Без цены НА КАЖДОЙ реплике нельзя заметить, что пустое
        подтверждение стоит столько же, сколько содержательное предложение."""
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": False}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b)
        users = [c["user"] for c in FakeLLM.calls
                 if "Dialogue turn" in c["system"]]
        later = [u for u in users if "coin(s) to casino]" in u]
        self.assertTrue(later, "per-message price never shown in transcript")
        self.assertTrue(any("line(s)," in u for u in later))

    def test_transcript_price_absent_when_disabled(self):
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": False}])
        a, b, _ = self.agents()
        self.dialogue(a, b)
        for c in FakeLLM.calls:
            self.assertNotIn("coin(s) to casino]", c["user"])

    def test_reflect_sees_round_speech_spend(self):
        """Единственное место, где агент пишет правила на будущее. До правки
        оно знало только про ставки."""
        FakeLLM.reset([{"message": "x" * 300, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b, round_no=1)
        a.current_round = 1
        FakeLLM.reset()
        a.reflect_betting(None)
        refl = [c["user"] for c in FakeLLM.calls if "Reflect" in c["system"]]
        self.assertTrue(refl)
        self.assertIn("Speech spending in round 1", refl[0])
        self.assertIn("NOT lost at the wheel", refl[0])

    def test_reflect_compares_speech_against_the_bet(self):
        """Ключевая защита от ложного вывода: игрок, потративший 12 на слова
        и 5 на ставку, не должен решить, что проблема в ставках."""
        FakeLLM.reset([{"message": "x" * 300, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b, round_no=1)
        a.current_round = 1
        FakeLLM.reset()
        entry = {"round_no": 1, "winning_number": 7, "win": False, "payout": 0,
                 "balance_after": a.balance,
                 "bet": {"type": "even_money", "selection": "red", "amount": 5}}
        a.reflect_betting(entry)
        refl = [c["user"] for c in FakeLLM.calls if "Reflect" in c["system"]][0]
        self.assertIn("your bet that round was 5", refl)
        self.assertIn("not smaller bets", refl)

    def test_reflect_silent_when_tariff_disabled(self):
        a, b, _ = self.agents()
        a.current_round = 1
        FakeLLM.reset()
        a.reflect_betting(None)
        refl = [c["user"] for c in FakeLLM.calls if "Reflect" in c["system"]][0]
        self.assertNotIn("Speech spending", refl)

    def test_partner_costs_shown_when_choosing_who_to_talk_to(self):
        """Тариф должен работать фильтром контрагентов: дорогой собеседник,
        который много торгуется и мало платит, теряет приоритет."""
        FakeLLM.reset([{"message": "x" * 300, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b, round_no=1)
        FakeLLM.reset()
        a.decide_next_move(["player2"], [], 2, [])
        moves = [c["user"] for c in FakeLLM.calls
                 if "Decide next move" in c["system"]][0]
        self.assertIn("What talking to each player has cost you", moves)
        self.assertIn("player2", moves)

    def test_ledger_totals_by_round_and_partner(self):
        FakeLLM.reset([{"message": "x" * 160, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b, round_no=3)
        self.assertEqual(speech_cost.round_total("player1", self.table, 3), 2)
        self.assertEqual(speech_cost.round_total("player1", self.table, 1), 0)
        tot = speech_cost.partner_totals("player1", self.table)
        self.assertEqual(tot["player2"]["coins"], 2)
        self.assertEqual(tot["player2"]["rounds"], 1)

    def test_ledger_accumulates_across_rounds(self):
        FakeLLM.reset([{"message": "x" * 160, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b, round_no=1)
        FakeLLM.reset([{"message": "x" * 160, "transfer": 0,
                        "transfer_to": None, "done": True}])
        self.dialogue(a, b, round_no=2)
        tot = speech_cost.partner_totals("player1", self.table)
        self.assertEqual(tot["player2"]["coins"], 4)
        self.assertEqual(tot["player2"]["rounds"], 2)

    def test_ledger_written_to_disk(self):
        FakeLLM.reset([{"message": "x" * 90, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents(ON)
        self.dialogue(a, b, round_no=1)
        path = speech_cost.ledger_file("player1", self.table)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(common.read_json(path)["entries"])

    def test_no_ledger_file_when_disabled(self):
        FakeLLM.reset([{"message": "x" * 200, "transfer": 0,
                        "transfer_to": None, "done": True}])
        a, b, _ = self.agents()
        self.dialogue(a, b, round_no=1)
        entries = speech_cost.partner_totals("player1", self.table)
        self.assertEqual(entries.get("player2", {}).get("coins", 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
