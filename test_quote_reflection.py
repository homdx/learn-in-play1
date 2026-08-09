"""
test_quote_reflection.py — тесты для QUOTE-1: рефлексия-2 над архивом
разговоров с конкретным партнёром + выбор точной цитаты.

Проверяем:
  1. Тумблер (game.quote_reflection_enabled) — по умолчанию выключено,
     ни одного LLM-вызова, ни байта в промпте.
  2. dialogue_archive.full_history() — читает ОБЕ стороны, ВСЕ раунды,
     по конкретной паре игроков; пусто, если разговоров не было.
  3. reflect_on_partner_dialogues(): "нет разговоров" — без LLM-вызова;
     выбор цитаты по номеру — код сам подставляет ТОЧНЫЙ текст, а не то,
     что вернула модель; невалидный/null индекс — без цитаты.
  4. Цитата пробрасывается в промпт dialogue_turn через параметр quote_note.
  5. run_dialogue() вызывает рефлексию-2 РОВНО ОДИН раз на сторону за
     диалог (у инициатора на turn=0, у отвечающего на его первом ответе),
     а не на каждый последующий ход.

    python3 test_quote_reflection.py [-v]
"""

import configparser
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import llm_client
import agent_v2
import dialogue_archive
import run_game_v2
from agent_v2 import PlayerAgent
from llm_client import LLMUnavailable


PLAYERS = ["player1", "player2", "player3"]


def make_cfg(table_dir, logs_dir, quote_reflection_enabled=False,
            max_bets_per_round=1):
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "api":       {"active": "local"},
        "api_local": {"base_url": "http://unused", "model": "fake"},
        "game":      {"table_dir": table_dir, "logs_dir": logs_dir,
                      "players": ",".join(PLAYERS),
                      "start_balance": "100", "rounds": "1",
                      "round_delay_sec": "0",
                      "max_bets_per_round": str(max_bets_per_round),
                      "quote_reflection_enabled": str(quote_reflection_enabled).lower()},
        "player":    {"temperature": "0.5", "max_tokens": "100",
                      "history_window": "10"},
        "memory":    {"quote_reflection_max_lines": "30"},
    })
    return cfg


class FakeLLM:
    behaviour = {}
    calls = []

    @classmethod
    def from_config(cls, cfg, section=None):
        return cls()

    def chat_json(self, system, user, temperature=0.0, max_tokens=0):
        FakeLLM.calls.append({"system": system, "user": user})
        if "Review your own past dialogue history" in system:
            fn = FakeLLM.behaviour.get("quote_reflect")
            if fn is None:
                raise LLMUnavailable("no quote_reflect behaviour configured")
            return fn(user)
        if "Place casino bet" in system:
            return {"type": "even_money", "selection": "red", "amount": 1}
        return {}


def write_dlg(table_dir, round_no, pid_a, pid_b, conversation):
    path = os.path.join(table_dir, f"dlg_r{round_no:03d}_{pid_a}_{pid_b}.json")
    common.write_json(path, {
        "round": round_no, "pid_a": pid_a, "pid_b": pid_b,
        "conversation": conversation, "a_sent": 0, "b_sent": 0,
        "a_speech_cost": 0, "b_speech_cost": 0,
    })


class QuoteHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="casino_quote_test_")
        self.table = os.path.join(self.tmp, "table")
        self.logs = os.path.join(self.tmp, "logs")
        os.makedirs(self.table, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)
        self._orig_client = agent_v2.LLMClient
        agent_v2.LLMClient = FakeLLM
        llm_client.LLMClient = FakeLLM
        FakeLLM.behaviour = {}
        FakeLLM.calls = []

    def tearDown(self):
        agent_v2.LLMClient = self._orig_client
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_agent(self, pid="player1", quote_reflection_enabled=False):
        cfg = make_cfg(self.table, self.logs,
                       quote_reflection_enabled=quote_reflection_enabled)
        return PlayerAgent(pid, self.table, cfg)


# ═══════════════════════════ 1. ТУМБЛЕР ═══════════════════════════════════

class TestToggle(QuoteHarness):

    def test_disabled_by_default_no_llm_call_no_text(self):
        agent = self.make_agent(quote_reflection_enabled=False)
        write_dlg(self.table, 1, "player1", "player2",
                 [{"from": "player1", "message": "hi", "transfer": 0}])
        result = agent.reflect_on_partner_dialogues("player2", round_no=2)
        self.assertEqual(result, "")
        self.assertEqual(len(FakeLLM.calls), 0)

    def test_enabled_makes_llm_call_when_history_exists(self):
        agent = self.make_agent(quote_reflection_enabled=True)
        write_dlg(self.table, 1, "player1", "player2",
                 [{"from": "player2", "message": "I'll cover your bet", "transfer": 0}])
        FakeLLM.behaviour = {"quote_reflect": lambda u: {"quote_index": None, "note": ""}}
        agent.reflect_on_partner_dialogues("player2", round_no=2)
        self.assertEqual(len(FakeLLM.calls), 1)


# ═══════════════════ 2. dialogue_archive.full_history() ═══════════════════

class TestFullHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="casino_archive_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_when_never_talked(self):
        self.assertEqual(dialogue_archive.full_history(self.tmp, "player1", "player2"), [])

    def test_reads_both_sides_single_round(self):
        write_dlg(self.tmp, 1, "player1", "player2", [
            {"from": "player1", "message": "offer 10"},
            {"from": "player2", "message": "accept"},
        ])
        hist = dialogue_archive.full_history(self.tmp, "player1", "player2")
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["from"], "player1")
        self.assertEqual(hist[0]["message"], "offer 10")
        self.assertEqual(hist[1]["from"], "player2")

    def test_reads_across_multiple_rounds_chronologically(self):
        write_dlg(self.tmp, 2, "player1", "player2",
                 [{"from": "player1", "message": "round two"}])
        write_dlg(self.tmp, 1, "player1", "player2",
                 [{"from": "player1", "message": "round one"}])
        hist = dialogue_archive.full_history(self.tmp, "player1", "player2")
        self.assertEqual([h["round_no"] for h in hist], [1, 2])
        self.assertEqual(hist[0]["message"], "round one")

    def test_ignores_dialogues_with_other_players(self):
        write_dlg(self.tmp, 1, "player1", "player3",
                 [{"from": "player3", "message": "unrelated"}])
        hist = dialogue_archive.full_history(self.tmp, "player1", "player2")
        self.assertEqual(hist, [])

    def test_order_independent_of_who_is_a_or_b_in_filename(self):
        write_dlg(self.tmp, 1, "player2", "player1",
                 [{"from": "player2", "message": "hello"}])
        hist = dialogue_archive.full_history(self.tmp, "player1", "player2")
        self.assertEqual(len(hist), 1)


# ═══════════════════ 3. reflect_on_partner_dialogues() ═════════════════════

class TestReflectOnPartnerDialogues(QuoteHarness):

    def test_no_history_returns_fixed_text_without_llm_call(self):
        agent = self.make_agent(quote_reflection_enabled=True)
        result = agent.reflect_on_partner_dialogues("player2", round_no=1)
        self.assertIn("never had a dialogue with player2", result)
        self.assertEqual(len(FakeLLM.calls), 0)

    def test_quote_index_selects_exact_stored_text(self):
        agent = self.make_agent(quote_reflection_enabled=True)
        write_dlg(self.table, 1, "player1", "player2", [
            {"from": "player1", "message": "I need a hedge"},
            {"from": "player2", "message": "I'll cover your 10 on black if it loses"},
        ])
        FakeLLM.behaviour = {"quote_reflect": lambda u: {
            "quote_index": 1, "note": "they promised to cover my loss"}}
        result = agent.reflect_on_partner_dialogues("player2", round_no=2)
        self.assertIn("I'll cover your 10 on black if it loses", result)
        self.assertIn("player2", result)
        self.assertIn("round 1", result)

        saved = agent_v2.load_quote_notes("player1", self.table)
        self.assertEqual(saved["player2"]["quote_text"],
                         "I'll cover your 10 on black if it loses")
        self.assertEqual(saved["player2"]["quote_from"], "player2")
        self.assertEqual(saved["player2"]["quote_round"], 1)

    def test_llm_cannot_alter_quote_text_even_if_note_lies_about_it(self):
        """Модель не может подменить сохранённый текст цитаты — только
        ВЫБРАТЬ номер строки; сам текст код берёт из архива, а не из
        ответа модели, даже если она вписывает другой текст в 'note'."""
        agent = self.make_agent(quote_reflection_enabled=True)
        write_dlg(self.table, 1, "player1", "player2", [
            {"from": "player2", "message": "I will NOT cover any bet"},
        ])
        FakeLLM.behaviour = {"quote_reflect": lambda u: {
            "quote_index": 0,
            "note": "they promised to cover my bet no matter what"  # ложь про сам факт
        }}
        agent.reflect_on_partner_dialogues("player2", round_no=2)
        saved = agent_v2.load_quote_notes("player1", self.table)
        # текст цитаты — РЕАЛЬНЫЙ, из архива, вне зависимости от вранья в note
        self.assertEqual(saved["player2"]["quote_text"], "I will NOT cover any bet")

    def test_null_quote_index_saves_no_quote(self):
        agent = self.make_agent(quote_reflection_enabled=True)
        write_dlg(self.table, 1, "player1", "player2",
                 [{"from": "player1", "message": "hi"}])
        FakeLLM.behaviour = {"quote_reflect": lambda u: {
            "quote_index": None, "note": "nothing worth bringing up"}}
        result = agent.reflect_on_partner_dialogues("player2", round_no=2)
        saved = agent_v2.load_quote_notes("player1", self.table)
        self.assertIsNone(saved["player2"]["quote_text"])
        self.assertIn("nothing worth bringing up", result)

    def test_out_of_range_quote_index_treated_as_no_quote(self):
        agent = self.make_agent(quote_reflection_enabled=True)
        write_dlg(self.table, 1, "player1", "player2",
                 [{"from": "player1", "message": "hi"}])
        FakeLLM.behaviour = {"quote_reflect": lambda u: {
            "quote_index": 999, "note": "x"}}
        agent.reflect_on_partner_dialogues("player2", round_no=2)
        saved = agent_v2.load_quote_notes("player1", self.table)
        self.assertIsNone(saved["player2"]["quote_text"])

    def test_llm_failure_degrades_to_empty_string_not_crash(self):
        agent = self.make_agent(quote_reflection_enabled=True)
        write_dlg(self.table, 1, "player1", "player2",
                 [{"from": "player1", "message": "hi"}])
        FakeLLM.behaviour = {"quote_reflect": lambda u: (_ for _ in ()).throw(
            ValueError("malformed"))}
        result = agent.reflect_on_partner_dialogues("player2", round_no=2)
        self.assertEqual(result, "")

    def test_incoming_message_included_for_responder_reflection(self):
        agent = self.make_agent(quote_reflection_enabled=True)
        write_dlg(self.table, 1, "player1", "player2",
                 [{"from": "player2", "message": "old claim"}])
        captured = {}
        def fn(u):
            captured["user"] = u
            return {"quote_index": None, "note": ""}
        FakeLLM.behaviour = {"quote_reflect": fn}
        agent.reflect_on_partner_dialogues("player2", round_no=2,
                                           incoming_message="new claim from them")
        self.assertIn("new claim from them", captured["user"])

    def test_per_partner_notes_are_independent(self):
        agent = self.make_agent(quote_reflection_enabled=True)
        write_dlg(self.table, 1, "player1", "player2",
                 [{"from": "player2", "message": "quote for p2"}])
        write_dlg(self.table, 1, "player1", "player3",
                 [{"from": "player3", "message": "quote for p3"}])
        FakeLLM.behaviour = {"quote_reflect": lambda u: {"quote_index": 0, "note": "x"}}
        agent.reflect_on_partner_dialogues("player2", round_no=2)
        agent.reflect_on_partner_dialogues("player3", round_no=2)
        saved = agent_v2.load_quote_notes("player1", self.table)
        self.assertEqual(saved["player2"]["quote_text"], "quote for p2")
        self.assertEqual(saved["player3"]["quote_text"], "quote for p3")


# ═══════════ QUOTE-2: заявление про ТРЕТЬЕГО игрока ════════════════════════

class TestCrossPlayerClaimVerification(QuoteHarness):
    """
    Сценарий: player4 говорит player2 "ты обещал player5 ставить чёрное".
    Это заявление про архив {player2, player5}, а НЕ про архив
    {player2, player4} (текущий собеседник). Рефлексия-2 у отвечающего
    должна САМА определить это и подтянуть архив с правильным игроком.
    """

    def make_cfg3(self, quote_reflection_enabled=True):
        cfg = configparser.ConfigParser()
        cfg.read_dict({
            "api":       {"active": "local"},
            "api_local": {"base_url": "http://unused", "model": "fake"},
            "game":      {"table_dir": self.table, "logs_dir": self.logs,
                          "players": "player1,player2,player4,player5",
                          "start_balance": "100", "rounds": "1",
                          "round_delay_sec": "0", "max_bets_per_round": "1",
                          "quote_reflection_enabled": str(quote_reflection_enabled).lower()},
            "player":    {"temperature": "0.5", "max_tokens": "100"},
            "memory":    {"quote_reflection_max_lines": "30"},
        })
        return cfg

    def test_claim_about_third_player_pulls_correct_archive(self):
        agent2 = PlayerAgent("player2", self.table, self.make_cfg3())

        # player2's REAL past dialogue with player5: no such promise.
        write_dlg(self.table, 2, "player2", "player5", [
            {"from": "player2", "message": "I'm betting red this round"},
            {"from": "player5", "message": "ok, I'll do 1st dozen"},
        ])
        # player2's dialogue with player4 (current speaker) is unrelated.
        write_dlg(self.table, 1, "player2", "player4",
                 [{"from": "player4", "message": "nice weather"}])

        calls = {"n": 0}
        def dispatch(**kw):
            calls["n"] += 1
            system, user = kw["system"], kw["user"]
            if "Identify which player" in system:
                self.assertIn("player5", user)
                return {"archive_with": "player5"}
            if "Review your own past dialogue history" in system:
                self.assertIn("player5", user)
                self.assertNotIn("nice weather", user)  # НЕ архив с player4
                return {"quote_index": 0, "note": "I never promised player5 that"}
            return {}
        agent2.client.chat_json = lambda **kw: dispatch(**kw)

        result = agent2.reflect_on_partner_dialogues(
            "player4", round_no=3,
            incoming_message="you promised player5 to bet black")

        self.assertEqual(calls["n"], 2)   # ровно +1 вызов сверх QUOTE-1
        self.assertIn("I'm betting red this round", result)   # реальная цитата
        self.assertIn("player5", result)
        self.assertNotIn("nice weather", result)

        saved = agent_v2.load_quote_notes("player2", self.table)
        self.assertIn("player5", saved)              # заметка легла под player5, не player4
        self.assertEqual(saved["player5"]["quote_text"], "I'm betting red this round")
        self.assertEqual(saved["player5"]["raised_by"], "player4")

    def test_claim_about_current_partner_stays_on_current_partner(self):
        """Если заявление НЕ про третье лицо — поведение как в QUOTE-1,
        архив остаётся с текущим собеседником."""
        agent2 = PlayerAgent("player2", self.table, self.make_cfg3())
        write_dlg(self.table, 1, "player2", "player4",
                 [{"from": "player4", "message": "you owe me 5 coins"}])

        def dispatch(**kw):
            system = kw["system"]
            if "Identify which player" in system:
                return {"archive_with": None}
            if "Review your own past dialogue history" in system:
                return {"quote_index": 0, "note": "checking"}
            return {}
        agent2.client.chat_json = lambda **kw: dispatch(**kw)

        result = agent2.reflect_on_partner_dialogues(
            "player4", round_no=2, incoming_message="you still owe me 5 coins")
        self.assertIn("you owe me 5 coins", result)
        saved = agent_v2.load_quote_notes("player2", self.table)
        self.assertIn("player4", saved)
        self.assertIsNone(saved["player4"]["raised_by"])

    def test_hallucinated_third_player_name_falls_back_to_current_partner(self):
        """Модель называет игрока, которого нет за столом — код должен
        откатиться на текущего собеседника, а не читать несуществующий файл."""
        agent2 = PlayerAgent("player2", self.table, self.make_cfg3())
        write_dlg(self.table, 1, "player2", "player4",
                 [{"from": "player4", "message": "real line with player4"}])

        def dispatch(**kw):
            system = kw["system"]
            if "Identify which player" in system:
                return {"archive_with": "player99"}   # не существует
            if "Review your own past dialogue history" in system:
                return {"quote_index": 0, "note": "x"}
            return {}
        agent2.client.chat_json = lambda **kw: dispatch(**kw)

        result = agent2.reflect_on_partner_dialogues(
            "player4", round_no=2, incoming_message="you promised player99 stuff")
        self.assertIn("real line with player4", result)
        saved = agent_v2.load_quote_notes("player2", self.table)
        self.assertIn("player4", saved)
        self.assertNotIn("player99", saved)

    def test_no_archive_with_named_third_player_reports_no_dialogue(self):
        """Заявление про игрока, с которым РЕАЛЬНО не было диалогов —
        должно вернуть детерминированный факт, без выдумывания цитаты."""
        agent2 = PlayerAgent("player2", self.table, self.make_cfg3())
        # player2 и player5 никогда не разговаривали.
        def dispatch(**kw):
            system = kw["system"]
            if "Identify which player" in system:
                return {"archive_with": "player5"}
            return {}
        agent2.client.chat_json = lambda **kw: dispatch(**kw)

        result = agent2.reflect_on_partner_dialogues(
            "player4", round_no=2,
            incoming_message="you promised player5 to bet black")
        self.assertIn("never had a dialogue with player5", result)

    def test_initiator_side_never_triggers_cross_player_lookup(self):
        """У инициатора incoming_message=None — определение 'чей архив
        нужен' не запускается вообще, ноль лишних вызовов."""
        agent2 = PlayerAgent("player2", self.table, self.make_cfg3())
        write_dlg(self.table, 1, "player2", "player4",
                 [{"from": "player4", "message": "hi"}])
        calls = {"n": 0}
        def dispatch(**kw):
            calls["n"] += 1
            return {"quote_index": None, "note": ""}
        agent2.client.chat_json = lambda **kw: dispatch(**kw)

        agent2.reflect_on_partner_dialogues("player4", round_no=2)
        self.assertEqual(calls["n"], 1)   # ровно как в QUOTE-1, без доп. вызова


# ═══════════════════ 4. Проброс в dialogue_turn ═══════════════════════════

class TestDialogueTurnReceivesQuoteNote(QuoteHarness):

    def test_quote_note_appears_in_dialogue_turn_prompt(self):
        agent = self.make_agent(quote_reflection_enabled=True)
        captured = {}
        def fake_chat_json(**kw):
            captured["user"] = kw.get("user", "")
            return {"message": "hi", "transfer": 0, "transfer_to": None, "done": False}
        agent.client.chat_json = lambda **kw: fake_chat_json(**kw)

        agent.dialogue_turn(
            partner_id="player2", partner_balance=100, conversation=[],
            round_no=1, is_initiator=True,
            quote_note="QUOTE NOTE: they promised to cover your bet\n\n"
        )
        self.assertIn("QUOTE NOTE: they promised to cover your bet", captured["user"])

    def test_empty_quote_note_adds_nothing(self):
        agent = self.make_agent(quote_reflection_enabled=False)
        captured = {}
        def fake_chat_json(**kw):
            captured["user"] = kw.get("user", "")
            return {"message": "hi", "transfer": 0, "transfer_to": None, "done": False}
        agent.client.chat_json = lambda **kw: fake_chat_json(**kw)

        agent.dialogue_turn(
            partner_id="player2", partner_balance=100, conversation=[],
            round_no=1, is_initiator=True, quote_note=""
        )
        self.assertNotIn("QUOTE NOTE", captured["user"])


# ═══════════════════ 5. Вызов ровно один раз за диалог ═════════════════════

class TestRunDialogueCallsReflectionOncePerSide(QuoteHarness):
    """
    Проверяем, что run_dialogue() дёргает reflect_on_partner_dialogues()
    РОВНО ОДИН раз для инициатора (перед turn=0) и РОВНО ОДИН раз для
    отвечающего (перед его первым ответом), а не на каждый последующий ход.
    """

    def test_reflection_called_exactly_once_per_side(self):
        cfg_a = make_cfg(self.table, self.logs, quote_reflection_enabled=True)
        cfg_b = make_cfg(self.table, self.logs, quote_reflection_enabled=True)
        agent_a = PlayerAgent("player1", self.table, cfg_a)
        agent_b = PlayerAgent("player2", self.table, cfg_b)

        calls = {"a": 0, "b": 0}
        orig_reflect_a = agent_a.reflect_on_partner_dialogues
        orig_reflect_b = agent_b.reflect_on_partner_dialogues

        def counted_a(partner_id, round_no, incoming_message=None):
            calls["a"] += 1
            return ""
        def counted_b(partner_id, round_no, incoming_message=None):
            calls["b"] += 1
            return ""
        agent_a.reflect_on_partner_dialogues = counted_a
        agent_b.reflect_on_partner_dialogues = counted_b

        # Многоходовый, но короткий диалог: обе стороны используют
        # 'done' быстро, чтобы проверить именно счётчик рефлексий, а не
        # длину диалога.
        turn_counter = {"a": 0, "b": 0}
        def fake_dialogue_turn_a(**kw):
            turn_counter["a"] += 1
            done = turn_counter["a"] >= 2
            return {"message": f"a-turn-{turn_counter['a']}", "transfer": 0,
                    "transfer_to": None, "done": done}
        def fake_dialogue_turn_b(**kw):
            turn_counter["b"] += 1
            done = turn_counter["b"] >= 1
            return {"message": f"b-turn-{turn_counter['b']}", "transfer": 0,
                    "transfer_to": None, "done": done}
        agent_a.dialogue_turn = lambda **kw: fake_dialogue_turn_a(**kw)
        agent_b.dialogue_turn = lambda **kw: fake_dialogue_turn_b(**kw)

        run_game_v2.run_dialogue(agent_a, agent_b, round_no=1,
                                 logger=_NullLogger(), table_dir=self.table)

        self.assertEqual(calls["a"], 1)
        self.assertEqual(calls["b"], 1)


class _NullLogger:
    def write(self, *a, **kw): pass
    def write_global(self, *a, **kw): pass
    def write_dialogue(self, *a, **kw): pass


if __name__ == "__main__":
    unittest.main()
