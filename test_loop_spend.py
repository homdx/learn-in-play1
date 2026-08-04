"""
test_loop_spend.py — обрыв по петле, бюджет переводов, антифантомный раунд.

Все три исправления выросли из одного прогона, где в ПЕРВОМ раунде игрок
отдал 135 монет из стартовых ста по событиям, которых не было:

  * LOOP-1  — детектор петли срабатывал, но партнёр получал закрывающий
              ход и переводил в нём деньги (трижды за раунд, по 30 монет);
  * SPEND-1 — суммарные переводы за раунд ничем не ограничивались;
  * ROUND-1 — в первом раунде обсуждался "долг за раунд 4", и пустой
              публичный журнал этому никак не мешал.
"""

import unittest

import agent_v2
from agent_v2 import _current_round_notice
import run_game_v2


class FakeLogger:
    def __init__(self):
        self.lines = []

    def write(self, pid, msg):
        self.lines.append((pid, msg))

    def write_global(self, msg):
        self.lines.append(("GAME", msg))


class Agent:
    """Минимальная заглушка: только поля, которые трогает _cap_transfer."""

    def __init__(self, balance=100, budget=None, sent=0):
        self.balance = balance
        if budget is not None:
            self.transfer_budget_this_round = budget
        self.sent_this_round = sent


class TestLoopBreak(unittest.TestCase):

    @staticmethod
    def _looping_conversation():
        line = ("Round 4, straight bet on 17, lost 30 coins. "
                "He told me he would put it on red. Check the ledger.")
        return [
            {"from": "player1", "message": "I'll pay 30 coins for a full strategy plan."},
            {"from": "player2", "message": line},
            {"from": "player1", "message": "I'll pay 30 coins for a full strategy plan."},
            {"from": "player2", "message": line},
        ]

    def test_detector_still_fires_on_repetition(self):
        self.assertTrue(agent_v2.PlayerAgent._detect_loop(self._looping_conversation()))

    def test_stub_carries_loop_break_flag(self):
        """Заглушка должна отличаться от обычного done."""
        src = open("agent_v2.py", encoding="utf-8").read()
        i = src.index('loop detected in dialogue with')
        window = src[i:i + 1100]
        self.assertIn('"loop_break": True', window)
        self.assertIn('"transfer": 0', window)

    def test_orchestrator_breaks_on_loop_for_both_sides(self):
        src = open("run_game_v2.py", encoding="utf-8").read()
        self.assertEqual(src.count('turn_a.get("loop_break")'), 1)
        self.assertEqual(src.count('turn_b.get("loop_break")'), 1)

    def test_loop_break_checked_before_granting_closing_turn(self):
        """
        Порядок критичен: если done проверяется первым, closing_for уже
        выставлен и партнёр получает ход с правом перевода.
        """
        src = open("run_game_v2.py", encoding="utf-8").read()
        for who, other in (("a", "b"), ("b", "a")):
            i = src.index(f'turn_{who}.get("loop_break")')
            j = src.index(f'closing_for = pid_{other}', i)
            k = src.index(f'turn_{who}.get("done")', i)
            self.assertLess(i, k, f"loop_break должен проверяться до done ({who})")
            self.assertLess(i, j)


class TestTransferBudget(unittest.TestCase):

    def test_no_budget_means_no_cap(self):
        lg = FakeLogger()
        self.assertEqual(run_game_v2._cap_transfer(Agent(), "p1", 90, lg), 90)
        self.assertEqual(lg.lines, [])

    def test_within_budget_passes_untouched(self):
        lg = FakeLogger()
        a = Agent(balance=100, budget=50)
        self.assertEqual(run_game_v2._cap_transfer(a, "p1", 30, lg), 30)
        self.assertEqual(lg.lines, [])

    def test_over_budget_is_capped_not_dropped(self):
        lg = FakeLogger()
        a = Agent(balance=100, budget=50, sent=30)
        self.assertEqual(run_game_v2._cap_transfer(a, "p1", 40, lg), 20)
        self.assertIn("CAPPED", lg.lines[0][1])

    def test_exhausted_budget_blocks_completely(self):
        lg = FakeLogger()
        a = Agent(balance=100, budget=50, sent=50)
        self.assertEqual(run_game_v2._cap_transfer(a, "p1", 25, lg), 0)
        self.assertIn("BLOCKED", lg.lines[0][1])

    def test_real_case_135_coins_is_stopped_at_50(self):
        """
        Воспроизведение случая: три перевода 75 + 30 + 30 при старте 100.
        С бюджетом 50% проходит ровно 50.
        """
        lg = FakeLogger()
        a = Agent(balance=100, budget=50)
        total = 0
        for want in (75, 30, 30):
            got = run_game_v2._cap_transfer(a, "p1", want, lg)
            a.sent_this_round += got
            total += got
        self.assertEqual(total, 50)

    def test_incoming_coins_do_not_widen_the_budget(self):
        """
        Бюджет считается от баланса на начало раунда. Иначе цепочка
        "получил 30 — отдал больше" воспроизводила бы ту же дыру.
        """
        lg = FakeLogger()
        a = Agent(balance=100, budget=50, sent=50)
        a.balance = 200          # в диалоге пришли монеты
        self.assertEqual(run_game_v2._cap_transfer(a, "p1", 100, lg), 0)

    def test_budget_is_set_from_round_start_balance(self):
        src = open("run_game_v2.py", encoding="utf-8").read()
        self.assertIn("max_transfer_fraction_round", src)
        self.assertIn("transfer_budget_this_round", src)
        i = src.index("_ag.sent_this_round = 0")
        j = src.index("int(_ag.balance * _frac)", i)
        self.assertLess(j - i, 300)

    def test_config_ships_with_a_limit(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read("config_v2.ini")
        frac = cfg.getfloat("player", "max_transfer_fraction_round")
        self.assertTrue(0 < frac < 1)


class TestPhantomRoundNotice(unittest.TestCase):

    def test_round_one_says_zero_rounds_played(self):
        txt = _current_round_notice(1)
        self.assertIn("ZERO rounds have been played", txt)
        self.assertIn("ledger is EMPTY", txt)
        self.assertIn("fabricated", txt)

    def test_round_one_denies_round_zero_explicitly(self):
        """В логах фигурировали и 'round 0', и 'round 4' — оба в первом раунде."""
        self.assertIn("no round 0", _current_round_notice(1))

    def test_later_round_states_the_played_range(self):
        txt = _current_round_notice(5)
        self.assertIn("1..4", txt)
        self.assertIn("above 4", txt)

    def test_second_round_range_is_singular_but_present(self):
        self.assertIn("1..1", _current_round_notice(2))

    def test_none_round_still_returns_empty(self):
        self.assertEqual(_current_round_notice(None), "")

    def test_notice_is_used_in_dialogue_prompt(self):
        src = open("agent_v2.py", encoding="utf-8").read()
        self.assertGreaterEqual(src.count("_current_round_notice(round_no)"), 1)




class TestTransfersPerDialogue(unittest.TestCase):
    """
    XFER-1: ограничение по ЧИСЛУ переводов в одном диалоге.

    Два реальных случая. Игрок трижды за диалог платил за одну и ту же
    "схему ставок" (10 + 10 + 20) и не получил ничего. И: тот, кто ТРЕБОВАЛ
    30 монет, сам их перевёл, а собеседник перевёл 30 обратно — шестьдесят
    монет за два хода при нулевом итоге.
    """

    def setUp(self):
        self.lg = FakeLogger()

    def cap(self, requested, made, limit=2):
        return run_game_v2._cap_dialogue_transfers(
            None, "player1", requested, made, limit, self.lg)

    def test_first_transfer_passes(self):
        self.assertEqual(self.cap(20, made=0), 20)
        self.assertEqual(self.lg.lines, [])

    def test_second_transfer_passes(self):
        self.assertEqual(self.cap(20, made=1), 20)
        self.assertEqual(self.lg.lines, [])

    def test_third_transfer_is_blocked(self):
        self.assertEqual(self.cap(20, made=2), 0)
        self.assertIn("BLOCKED", self.lg.lines[0][1])
        self.assertIn("limit 2", self.lg.lines[0][1])

    def test_the_real_10_10_20_case(self):
        """Третий платёж за ту же 'схему' не проходит."""
        made, total = 0, 0
        for want in (10, 10, 20):
            got = self.cap(want, made=made)
            if got:
                made += 1
                total += got
        self.assertEqual(total, 20)
        self.assertEqual(made, 2)

    def test_zero_limit_disables_the_rule(self):
        self.assertEqual(self.cap(50, made=9, limit=0), 50)
        self.assertEqual(self.lg.lines, [])

    def test_counter_is_wired_for_both_sides(self):
        src = open("run_game_v2.py", encoding="utf-8").read()
        self.assertIn("a_transfers += 1", src)
        self.assertIn("b_transfers += 1", src)
        self.assertEqual(src.count("_cap_dialogue_transfers("), 3)

    def test_counter_resets_per_dialogue(self):
        """Счётчик локальный для диалога, а не для раунда."""
        src = open("run_game_v2.py", encoding="utf-8").read()
        i = src.index("a_transfers  = 0")
        j = src.index("def run_dialogue")
        self.assertGreater(i, j, "счётчик должен инициализироваться внутри диалога")

    def test_it_runs_before_the_round_budget(self):
        src = open("run_game_v2.py", encoding="utf-8").read()
        i = src.index("_cap_dialogue_transfers(\n            agent_a")
        j = src.index("_cap_transfer(agent_a", i)
        self.assertLess(i, j)

    def test_shipped_limit_is_two(self):
        self.assertEqual(run_game_v2.MAX_TRANSFERS_PER_DIALOGUE, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
