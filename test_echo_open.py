"""
test_echo_open.py — эхо-фильтр (ECHO-1) и правило открывающей реплики (OPEN-1).

Оба выросли из одного прогона:

  * ECHO-1. Игрок вернул реплику собеседника слово в слово, включая фразу
    "ты уже заплатил мне 50 монет" — то есть подтвердил получение денег,
    которые сам же и отдал. `_detect_loop` этого не ловит: он смотрит на
    транскрипт ДО генерации, а не на то, что модель вернула.

  * OPEN-1. Игрок открыл диалог словами "обсудим за 50 монет" и в этой же
    реплике перевёл половину капитала — до всякого ответа. Ответом было
    "я стратегиями не торгую".
"""

import os
import shutil
import tempfile
import unittest

import common
import promise_ledger
import run_game_v2
from agent_v2 import PlayerAgent


class FakeLogger:
    def __init__(self):
        self.lines = []

    def write(self, pid, msg):
        self.lines.append((pid, msg))

    def write_global(self, msg):
        self.lines.append(("GAME", msg))


REAL_ECHO = ("The ledger confirms no bet on red in round 1. You've already paid "
             "me 50 coins, so let's settle this. If you insist on verifying, "
             "I'll take the hit. No more spins, no more claims. Goodbye.")


class TestEchoDetector(unittest.TestCase):

    def test_verbatim_echo_of_partner_is_caught(self):
        conv = [{"from": "player2", "message": REAL_ECHO}]
        self.assertTrue(PlayerAgent._is_echo(REAL_ECHO, conv, "player2"))

    def test_near_verbatim_echo_is_caught(self):
        conv = [{"from": "player2", "message": REAL_ECHO}]
        almost = REAL_ECHO.replace("Goodbye.", "Bye.")
        self.assertTrue(PlayerAgent._is_echo(almost, conv, "player2"))

    def test_own_previous_message_is_also_caught(self):
        """
        ECHO-2: самоповтор раньше проходил между всеми проверками.
        `_detect_loop` смотрит транскрипт ДО генерации, когда повтора ещё
        нет, а `_is_echo` сравнивал только с чужой репликой. В прогоне игрок
        дословно повторил собственную фразу — совпадение 1.00, и никто не
        сработал.
        """
        conv = [{"from": "player1", "message": "some unrelated question"},
                {"from": "player2", "message": REAL_ECHO},
                {"from": "player1", "message": "another unrelated line"}]
        self.assertTrue(PlayerAgent._is_echo(REAL_ECHO, conv, "player1"))

    def test_real_self_repeat_from_the_log(self):
        line = "I'm on black for 10. You cover red and green. Let's go."
        conv = [{"from": "player2", "message": line},
                {"from": "player1", "message": "Deal confirmed, red and green."}]
        self.assertTrue(PlayerAgent._is_echo(line, conv, "player1"))

    def test_normal_reply_passes(self):
        conv = [{"from": "player2", "message": "I don't trade strategies. "
                                              "But I can help you recover a debt."}]
        msg = "Not interested in debts. What is your read on even_money bets?"
        self.assertFalse(PlayerAgent._is_echo(msg, conv, "player2"))

    def test_agreeing_in_own_words_is_not_an_echo(self):
        """Согласие с предложением не должно гаситься как эхо."""
        conv = [{"from": "player2", "message": "Pay 5 coins now and I will name "
                                              "the dozen before the spin."}]
        msg = "Agreed. Sending 5 now — name it."
        self.assertFalse(PlayerAgent._is_echo(msg, conv, "player2"))

    def test_empty_conversation_is_never_an_echo(self):
        self.assertFalse(PlayerAgent._is_echo("anything", [], "player2"))

    def test_empty_messages_do_not_crash(self):
        conv = [{"from": "player2", "message": ""}]
        self.assertFalse(PlayerAgent._is_echo("", conv, "player2"))
        self.assertFalse(PlayerAgent._is_echo("hello", conv, "player2"))

    def test_echo_path_returns_loop_break_and_zero_transfer(self):
        src = open("agent_v2.py", encoding="utf-8").read()
        i = src.index("echoed {partner_id}'s own message back verbatim")
        window = src[i:i + 400]
        self.assertIn('"loop_break": True', window)
        self.assertIn('"transfer": 0', window)

    def test_echo_check_skipped_on_closing_turn(self):
        """
        Закрывающий ход часто пересказывает согласованное партнёром — это
        не копирование, а подтверждение сделки.
        """
        src = open("agent_v2.py", encoding="utf-8").read()
        self.assertIn("if not closing_turn and self._is_echo(", src)


class TestOpeningTransfer(unittest.TestCase):

    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.lg = FakeLogger()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _promise(self, pid, partner, amount, created_round, status="open",
                 direction="i_owe"):
        promise_ledger.save_promises(pid, self.base, [{
            "id": 1, "direction": direction, "counterparty": partner,
            "amount": amount, "due_round": created_round + 1,
            "created_round": created_round, "description": "", "status": status,
        }])

    def cap(self, requested, conversation=None, round_no=2, free=10):
        return run_game_v2._cap_opening_transfer(
            None, "player1", "player2", requested,
            conversation if conversation is not None else [],
            self.base, round_no, self.lg, free)

    def test_mid_dialogue_transfer_is_untouched(self):
        conv = [{"from": "player2", "message": "pay me 50"}]
        self.assertEqual(self.cap(50, conversation=conv), 50)
        self.assertEqual(self.lg.lines, [])

    def test_small_opening_transfer_passes_without_any_promise(self):
        """Чаевые или маленький аванс остаются возможными."""
        self.assertEqual(self.cap(5), 5)
        self.assertEqual(self.lg.lines, [])

    def test_the_real_50_coin_case_is_capped(self):
        self.assertEqual(self.cap(50), 10)
        self.assertIn("CAPPED", self.lg.lines[0][1])
        self.assertIn("no open promise", self.lg.lines[0][1])

    def test_settlement_of_an_earlier_promise_passes_in_full(self):
        """Расчёт по договорённости прошлого раунда — законный ход."""
        self._promise("player1", "player2", 40, created_round=1)
        self.assertEqual(self.cap(40, round_no=2), 40)
        self.assertEqual(self.lg.lines, [])

    def test_transfer_above_the_promise_is_capped_to_it(self):
        self._promise("player1", "player2", 40, created_round=1)
        self.assertEqual(self.cap(90, round_no=2), 40)
        self.assertIn("open promise to player2 is 40c", self.lg.lines[0][1])

    def test_promise_from_the_same_round_does_not_count(self):
        """Обещание этого раунда не могло возникнуть до первого диалога."""
        self._promise("player1", "player2", 40, created_round=2)
        self.assertEqual(self.cap(40, round_no=2), 10)

    def test_settled_promise_does_not_count(self):
        self._promise("player1", "player2", 40, created_round=1, status="settled")
        self.assertEqual(self.cap(40, round_no=2), 10)

    def test_promise_in_the_other_direction_does_not_count(self):
        """'Мне должны' — не основание платить самому."""
        self._promise("player1", "player2", 40, created_round=1,
                      direction="owed_to_me")
        self.assertEqual(self.cap(40, round_no=2), 10)

    def test_promise_to_a_third_party_does_not_count(self):
        self._promise("player1", "player3", 40, created_round=1)
        self.assertEqual(self.cap(40, round_no=2), 10)

    def test_zero_transfer_is_left_alone(self):
        self.assertEqual(self.cap(0), 0)
        self.assertEqual(self.lg.lines, [])

    def test_allowance_is_configurable(self):
        self.assertEqual(self.cap(30, free=30), 30)
        self.assertEqual(self.cap(30, free=0), 0)

    def test_round_one_has_no_valid_settlement(self):
        """
        В первом раунде предыдущих раундов не существует, поэтому никакое
        обещание не может быть "с прошлого раунда" — остаётся только
        свободная сумма.
        """
        self._promise("player1", "player2", 40, created_round=0)
        self.assertEqual(self.cap(40, round_no=1), 10)


class TestWiring(unittest.TestCase):

    def test_opening_rule_runs_before_the_round_budget(self):
        """
        Иначе предоплата за воздух съедала бы бюджет раунда, даже будучи
        затем обнулённой.
        """
        src = open("run_game_v2.py", encoding="utf-8").read()
        i = src.index("_cap_opening_transfer(agent_a")
        j = src.index("_cap_transfer(agent_a", i)
        self.assertLess(i, j)

    def test_conversation_is_still_empty_at_that_point(self):
        """Реплика A добавляется в транскрипт ПОСЛЕ расчёта перевода."""
        src = open("run_game_v2.py", encoding="utf-8").read()
        i = src.index("_cap_opening_transfer(agent_a")
        j = src.index('conversation.append(', i)
        self.assertLess(i, j)

    def test_config_ships_the_allowance(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read("config_v2.ini")
        self.assertGreaterEqual(cfg.getint("player", "opening_transfer_free"), 0)




class TestDsynGrossFlows(unittest.TestCase):
    """
    DSYN-1: репутационная синапса должна помнить обороты, а не только нетто.

    Реальный случай: player1 и player2 прогнали друг через друга 54 монеты в
    четыре приёма (31 туда, 23 обратно), а в синапсе осело "отдал 8, получил
    0". Нетто было верным, обороты — потеряны.
    """

    def setUp(self):
        self.base = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _agent(self, pid="player1"):
        import agent_v2

        class Stub(PlayerAgent):
            def __init__(self, pid, base):
                self.player_id = pid
                self.base_dir = base
                self.dsyn_recent = 12
                self.deals_shown = 6
                self.fails_shown = 4
                self.tok_update_dsyn = 350
                self.temperature = 0.4

            def _log(self, msg):
                pass

            def _dsyn_llm(self, *a, **k):
                raise RuntimeError("no llm in test")

        st = Stub(pid, self.base)
        # у update_dsyn единственный внешний вызов — LLM; заглушаем его так,
        # чтобы сработала штатная ветка "LLM failed" и осталась только
        # детерминированная арифметика
        st.client = type("C", (), {
            "chat_json": staticmethod(
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub")))
        })()
        return st

    def _rep(self, pid, partner):
        import agent_v2
        return agent_v2.load_dsyn(pid, self.base)["reputation"][partner]

    def test_gross_flows_are_recorded_when_passed(self):
        a = self._agent()
        a.update_dsyn("player2", [{"from": "player1", "message": "m"}],
                      -8, 1, sent=31, received=23)
        rep = self._rep("player1", "player2")
        self.assertEqual(rep["total_sent"], 31)
        self.assertEqual(rep["total_received"], 23)
        self.assertEqual(rep["net"], -8)

    def test_the_other_side_mirrors_it(self):
        b = self._agent("player2")
        b.update_dsyn("player1", [{"from": "player2", "message": "m"}],
                      8, 1, sent=23, received=31)
        rep = self._rep("player2", "player1")
        self.assertEqual(rep["total_sent"], 23)
        self.assertEqual(rep["total_received"], 31)
        self.assertEqual(rep["net"], 8)

    def test_old_net_only_call_still_works(self):
        """Обратная совместимость: без оборотов — прежнее поведение."""
        a = self._agent()
        a.update_dsyn("player2", [{"from": "player1", "message": "m"}], -8, 1)
        rep = self._rep("player1", "player2")
        self.assertEqual(rep["total_sent"], 8)
        self.assertEqual(rep["total_received"], 0)
        self.assertEqual(rep["net"], -8)

    def test_flows_accumulate_across_rounds(self):
        a = self._agent()
        a.update_dsyn("player2", [{"from": "player1", "message": "m"}],
                      -8, 1, sent=31, received=23)
        a.update_dsyn("player2", [{"from": "player1", "message": "m"}],
                      5, 2, sent=10, received=15)
        rep = self._rep("player1", "player2")
        self.assertEqual(rep["total_sent"], 41)
        self.assertEqual(rep["total_received"], 38)
        self.assertEqual(rep["net"], -3)

    def test_orchestrator_passes_both_directions(self):
        src = open("run_game_v2.py", encoding="utf-8").read()
        self.assertIn("sent=a_total_sent, received=b_total_sent", src)
        self.assertIn("sent=b_total_sent, received=a_total_sent", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
