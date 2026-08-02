"""
test_open_bets.py — открытые ставки внутри раунда (BET-1).

Проверяем ровно то, на чём механика может тихо сломаться:
  * множество открытых ставок читается из тех же bet-файлов, что видит
    крупье (иначе агенты и колесо разойдутся в том, что стоит на столе);
  * блок промпта строится всегда, в том числе когда стол ещё пуст;
  * своя ставка помечена отдельно — агент должен видеть, что уже связан;
  * битый или недописанный bet-файл не роняет раунд;
  * после того как крупье убрал ставки, стол снова пуст.
"""

import json
import os
import shutil
import tempfile
import unittest

import common
import open_bets


def write_bet(base, pid, bet):
    common.write_json(common.bet_file(pid, base), bet)


EVEN = {"type": "even_money", "selection": "red", "amount": 10}
STRAIGHT = {"type": "straight", "numbers": [17], "amount": 15}


class TestRead(unittest.TestCase):

    def setUp(self):
        self.base = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_empty_table(self):
        self.assertEqual(open_bets.read(self.base), [])

    def test_reads_placed_bets(self):
        write_bet(self.base, "player1", EVEN)
        write_bet(self.base, "player2", STRAIGHT)
        got = dict(open_bets.read(self.base))
        self.assertEqual(set(got), {"player1", "player2"})
        self.assertEqual(got["player1"]["amount"], 10)
        self.assertEqual(got["player2"]["numbers"], [17])

    def test_matches_croupier_view(self):
        """Агенты и крупье обязаны видеть одно и то же множество ставок."""
        import croupier_v2
        write_bet(self.base, "player1", EVEN)
        write_bet(self.base, "player3", STRAIGHT)
        mine = {pid for pid, _ in open_bets.read(self.base)}
        theirs = {pid for pid, _, _ in croupier_v2.collect_bets(self.base)}
        self.assertEqual(mine, theirs)

    def test_ignores_unrelated_files(self):
        write_bet(self.base, "player1", EVEN)
        common.write_json(os.path.join(self.base, "balance_player1.json"),
                          {"balance": 50})
        common.write_json(os.path.join(self.base, "result_player1.json"),
                          {"payout": 0})
        self.assertEqual([p for p, _ in open_bets.read(self.base)], ["player1"])

    def test_corrupt_bet_file_is_skipped_not_fatal(self):
        """Обрыв на записи не должен ронять раунд у следующего игрока."""
        write_bet(self.base, "player1", EVEN)
        with open(common.bet_file("player2", self.base), "w") as f:
            f.write('{"type": "even_mon')       # недописанный JSON
        got = [p for p, _ in open_bets.read(self.base)]
        self.assertEqual(got, ["player1"])

    def test_bet_without_amount_is_skipped(self):
        write_bet(self.base, "player1", {"type": "even_money",
                                         "selection": "red"})
        self.assertEqual(open_bets.read(self.base), [])

    def test_table_clears_after_croupier(self):
        write_bet(self.base, "player1", EVEN)
        os.remove(common.bet_file("player1", self.base))
        self.assertEqual(open_bets.read(self.base), [])


class TestFormat(unittest.TestCase):

    def setUp(self):
        self.base = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_empty_block_still_renders(self):
        txt = open_bets.format_for_prompt(self.base, "player1")
        self.assertIn("nobody has placed a bet yet", txt)
        self.assertTrue(txt.endswith("\n\n"))

    def test_ends_with_blank_line_for_concatenation(self):
        write_bet(self.base, "player1", EVEN)
        txt = open_bets.format_for_prompt(self.base, "player2")
        self.assertTrue(txt.endswith("\n\n"))

    def test_own_bet_marked(self):
        write_bet(self.base, "player1", EVEN)
        write_bet(self.base, "player2", STRAIGHT)
        txt = open_bets.format_for_prompt(self.base, "player2")
        self.assertIn("YOU: straight(17) for 15 coins", txt)
        self.assertIn("player1: even_money(red) for 10 coins", txt)
        self.assertNotIn("player2:", txt)

    def test_no_self_pid_marks_nothing(self):
        write_bet(self.base, "player1", EVEN)
        txt = open_bets.format_for_prompt(self.base)
        self.assertNotIn("YOU:", txt)

    def test_balances_are_not_leaked(self):
        """Публична ставка, а не карман: балансов в блоке быть не должно."""
        write_bet(self.base, "player1", EVEN)
        common.write_json(os.path.join(self.base, "balance_player1.json"),
                          {"balance": 1234})
        txt = open_bets.format_for_prompt(self.base, "player2")
        self.assertNotIn("1234", txt)

    def test_describe_multi_number_bet(self):
        write_bet(self.base, "player1", {"type": "corner",
                                         "numbers": [1, 2, 4, 5],
                                         "amount": 4})
        txt = open_bets.format_for_prompt(self.base, "player2")
        self.assertIn("corner(1,2,4,5) for 4 coins", txt)

    def test_lie_detection_hint_present_only_when_bets_exist(self):
        empty = open_bets.format_for_prompt(self.base, "player1")
        self.assertNotIn("they lied to you", empty)
        write_bet(self.base, "player1", EVEN)
        filled = open_bets.format_for_prompt(self.base, "player2")
        self.assertIn("they lied to you", filled)


class TestOrchestratorWiring(unittest.TestCase):
    """Проверяем, что патч действительно вклеен в те места, где нужен."""

    def test_place_bet_helper_exists(self):
        import run_game_v2
        self.assertTrue(hasattr(run_game_v2, "_place_bet_for_player"))

    def test_prompt_paths_use_open_bets(self):
        src = open("agent_v2.py", encoding="utf-8").read()
        self.assertEqual(src.count("open_bets.format_for_prompt"), 4,
                         "ожидаются 4 точки: plan_round, decide_next_move, "
                         "dialogue_turn, decide_bet")

    def test_checkpoint_advances_past_player_who_already_bet(self):
        """
        После ставки указатель обязан смотреть на СЛЕДУЮЩЕГО игрока, иначе
        рестарт после Ctrl+C вернёт нас в ход уже отыгравшего и он получит
        вторую порцию диалогов сверх лимита.
        """
        src = open("run_game_v2.py", encoding="utf-8").read()
        self.assertIn("player_index + 1", src)
        # именно ВЫЗОВ внутри цикла, а не определение функции выше по файлу
        call = "\n            _place_bet_for_player(agent, pid, base_dir, round_no, logger)"
        i = src.index(call)
        j = src.index("player_index + 1", i)
        self.assertLess(j - i, 700, "сдвиг указателя должен идти сразу за ставкой")




class TestReflectionInTurn(unittest.TestCase):
    """
    BET-2: рефлексия переехала из общей Phase 0 в начало хода игрока, чтобы
    он осмыслял прошлый раунд, уже видя ставки сходивших до него.
    """

    def setUp(self):
        self.base = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    class FakeAgent:
        def __init__(self):
            self.reflected_with = "not called"
            self.current_round = None

        def reflect_betting(self, entry):
            self.reflected_with = entry

    class FakeLogger:
        def __init__(self):
            self.lines = []

        def write(self, pid, msg):
            self.lines.append((pid, msg))

    def _write_history(self, pid, rounds):
        import agent_v2
        common.write_json(agent_v2.history_file(pid, self.base),
                          {"rounds": rounds})

    def test_reflects_with_entry_of_previous_round(self):
        import run_game_v2
        self._write_history("player1", [
            {"round_no": 4, "win": False, "payout": 0, "balance_after": 70},
            {"round_no": 5, "win": True, "payout": 20, "balance_after": 90},
        ])
        agent, logger = self.FakeAgent(), self.FakeLogger()
        run_game_v2._reflect_for_player(agent, "player1", self.base, 5, logger)
        self.assertEqual(agent.reflected_with["payout"], 20)
        self.assertEqual(agent.current_round, 5)

    def test_reflects_anyway_when_player_did_not_bet(self):
        """FIX-5 не должен потеряться при переезде: банкрот тоже учится."""
        import run_game_v2
        self._write_history("player1", [{"round_no": 3, "win": False,
                                         "payout": 0, "balance_after": 0}])
        agent, logger = self.FakeAgent(), self.FakeLogger()
        run_game_v2._reflect_for_player(agent, "player1", self.base, 5, logger)
        self.assertIsNone(agent.reflected_with)
        self.assertTrue(any("reflecting anyway" in m for _, m in logger.lines))

    def test_no_history_file_at_all(self):
        import run_game_v2
        agent, logger = self.FakeAgent(), self.FakeLogger()
        run_game_v2._reflect_for_player(agent, "playerX", self.base, 5, logger)
        self.assertIsNone(agent.reflected_with)

    def test_first_round_reflects_without_complaining(self):
        """prev_round=0 — рефлексировать не о чем, но и жаловаться не о чем."""
        import run_game_v2
        agent, logger = self.FakeAgent(), self.FakeLogger()
        run_game_v2._reflect_for_player(agent, "player1", self.base, 0, logger)
        self.assertIsNone(agent.reflected_with)
        self.assertEqual(logger.lines, [])

    def test_phase0_no_longer_reflects(self):
        src = open("run_game_v2.py", encoding="utf-8").read()
        i = src.index("settle_pending_results(agents, players, base_dir, round_no - 1")
        window = src[i:i + 300]
        self.assertIn("reflect=False", window)

    def test_reflection_runs_before_planning(self):
        src = open("run_game_v2.py", encoding="utf-8").read()
        i = src.index("_reflect_for_player(agent, pid, base_dir, round_no - 1, logger)")
        j = src.index("plan_round(round_no, avail)", i)
        self.assertLess(j - i, 600,
                        "рефлексия должна стоять непосредственно перед plan_round")

    def test_reflection_runs_after_checkpoint_is_saved(self):
        """
        Чекпойнт пишется на входе в ход игрока, ДО рефлексии: после Ctrl+C
        восстановление начинается именно с рефлексии этого игрока.
        """
        src = open("run_game_v2.py", encoding="utf-8").read()
        i = src.index("_reflect_for_player(agent, pid, base_dir, round_no - 1, logger)")
        save = src.rindex("base_dir, round_no, player_index, talked_to", 0, i)
        self.assertLess(i - save, 800)


if __name__ == "__main__":
    unittest.main(verbosity=2)
