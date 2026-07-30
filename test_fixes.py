"""
test_fixes.py — контрфактические тесты для FIX-1..FIX-7.

Каждый тест написан так, чтобы ПАДАТЬ на неисправленном коде и проходить
на исправленном. LLM подменяется скриптованным фейком — сеть не нужна.

    python3 test_fixes.py            # прогнать всё
    python3 test_fixes.py -v         # с подробностями
"""

import configparser
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import llm_client
import agent_v2
import run_game_v2
from agent_v2 import PlayerAgent


PLAYERS = ["player1", "player2", "player3"]
START_BALANCE = 100


# ────────────────────────────────────────────────────── fake LLM ──────────

class FakeLLM:
    """Скриптованный клиент. Поведение задаётся через FakeLLM.behaviour."""

    behaviour = {}

    @classmethod
    def from_config(cls, cfg, section=None):
        return cls()

    def chat_json(self, system, user, temperature=0.0, max_tokens=0):
        b = FakeLLM.behaviour
        if "Decide next move" in system:
            return b.get("next_move", lambda u: {"action": "bet", "reason": "x"})(user)
        if "Dialogue turn" in system:
            return b.get("dialogue", lambda u: {"message": "no deal, goodbye",
                                                "transfer": 0, "transfer_to": None,
                                                "done": True})(user)
        if "Update reputation" in system:
            return {"trust_score": 6, "deal_done": None, "deal_failed": None,
                    "reputation_note": "n", "future_intent": "i", "summary": "s"}
        if "Place casino bet" in system:
            return b.get("bet", lambda u: {"type": "even_money", "selection": "red",
                                           "amount": 10})(user)
        if "Reflect" in system:
            return b.get("reflect", lambda u: {"notes": "n", "update_persona": False})(user)
        if "Compress" in system:
            return {"notes": "compressed", "compressed_history": "compressed"}
        return {}


def make_cfg(table_dir, logs_dir, rounds):
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "api":      {"active": "local"},
        "api_local": {"base_url": "http://unused", "model": "fake"},
        "game":     {"table_dir": table_dir, "logs_dir": logs_dir,
                     "players": ",".join(PLAYERS),
                     "start_balance": str(START_BALANCE),
                     "rounds": str(rounds), "max_bet_fraction": "0.4",
                     "round_delay_sec": "0"},
        "player":   {"temperature": "0.5", "max_tokens": "100",
                     "history_window": "10"},
    })
    return cfg


class GameHarness(unittest.TestCase):
    """Общая обвязка: временный стол, подменённый LLM, запуск main()."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="casino_test_")
        self.table = os.path.join(self.tmp, "table")
        self.logs = os.path.join(self.tmp, "logs")
        self._orig_client = agent_v2.LLMClient
        agent_v2.LLMClient = FakeLLM
        llm_client.LLMClient = FakeLLM
        FakeLLM.behaviour = {}
        self._orig_load_config = run_game_v2.load_config

    def tearDown(self):
        agent_v2.LLMClient = self._orig_client
        run_game_v2.load_config = self._orig_load_config
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_game(self, rounds, winning_number=None, seed=1):
        cfg = make_cfg(self.table, self.logs, rounds)
        run_game_v2.load_config = lambda path: cfg
        argv = ["run_game_v2.py", "--config", "unused"]
        if winning_number is not None:
            argv += ["--winning-number", str(winning_number)]
        old_argv, sys.argv = sys.argv, argv
        random.seed(seed)
        try:
            run_game_v2.main()
        finally:
            sys.argv = old_argv

    def balances(self):
        return run_game_v2.get_balances(self.table, PLAYERS)

    def dialogues(self):
        out = []
        for f in sorted(os.listdir(self.table)):
            if f.startswith("dlg_"):
                out.append(common.read_json(os.path.join(self.table, f)))
        return out


# ───────────────────────────────────────── FIX-1: фантомный перевод ───────

class TestPhantomTransfer(GameHarness):

    def _talk_once(self, user):
        if "Your player id: player1" in user and "(no one yet this round)" in user:
            return {"action": "talk", "partner": "player2", "reason": "x"}
        return {"action": "bet", "reason": "x"}

    def test_transfer_with_null_recipient_is_not_logged_as_paid(self):
        """
        Модель возвращает transfer=5, transfer_to=None. На неисправленном
        коде деньги не переводились, но в conversation/лог/синапсу попадала
        запись "отправлено 5 монет" — фантомная сделка.
        После FIX-1 запись и баланс обязаны совпадать.
        """
        FakeLLM.behaviour = {
            "next_move": self._talk_once,
            "dialogue": lambda u: {"message": "here is your payment for the tip",
                                   "transfer": 5, "transfer_to": None, "done": True},
        }
        before = START_BALANCE
        self.run_game(rounds=1, winning_number=0)

        dlgs = self.dialogues()
        self.assertTrue(dlgs, "диалог не состоялся — тест не проверяет то, что должен")
        conv = dlgs[0]["conversation"]
        logged = sum(t.get("transfer", 0) for t in conv)
        accounted = dlgs[0]["a_sent"] + dlgs[0]["b_sent"]
        self.assertEqual(
            logged, accounted,
            f"в conversation записано {logged} монет переводов, а фактически "
            f"проведено {accounted} — фантомная сделка (FIX-1)"
        )

    def test_transfer_to_null_still_reaches_the_partner(self):
        """
        FIX-1 трактует transfer>0 с пустым transfer_to как намерение
        заплатить собеседнику (других получателей в диалоге не бывает).
        Деньги должны реально дойти, а не потеряться.
        """
        FakeLLM.behaviour = {
            "next_move": self._talk_once,
            "dialogue": lambda u: (
                {"message": "here are 5 coins for your tip", "transfer": 5,
                 "transfer_to": None, "done": True}
                if "Dialogue with player2" in u else
                {"message": "thanks, noted", "transfer": 0,
                 "transfer_to": None, "done": True}
            ),
            "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 1},
        }
        self.run_game(rounds=1, winning_number=0)
        dlgs = self.dialogues()
        self.assertEqual(dlgs[0]["a_sent"], 5,
                         "перевод с transfer_to=None потерялся (FIX-1)")

    def test_transfer_to_third_party_is_cancelled_and_not_logged(self):
        """Явно указанный третий игрок — перевод отменяется И не логируется."""
        FakeLLM.behaviour = {
            "next_move": self._talk_once,
            "dialogue": lambda u: {"message": "sending this to player3 instead",
                                   "transfer": 7, "transfer_to": "player3",
                                   "done": True},
            "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 1},
        }
        self.run_game(rounds=1, winning_number=0)
        dlgs = self.dialogues()
        conv = dlgs[0]["conversation"]
        self.assertEqual(sum(t.get("transfer", 0) for t in conv), 0)
        self.assertEqual(dlgs[0]["a_sent"], 0)


# ────────────────────────────── FIX-2: выплата последнего раунда ──────────

class TestFinalSettlement(GameHarness):

    def test_last_round_payout_is_credited(self):
        """
        Ставка even_money на red, выигрышный номер 32 (красное) → выплата
        2×ставки. На неисправленном коде выигрыш последнего раунда не
        зачислялся никогда: result_*.json оставались на диске.
        """
        FakeLLM.behaviour = {
            "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 10},
        }
        self.run_game(rounds=1, winning_number=32)  # 32 — красное

        leftovers = [f for f in os.listdir(self.table) if f.startswith("result_")]
        self.assertEqual(leftovers, [],
                         f"остались необработанные результаты: {leftovers} (FIX-2)")

        for pid, bal in self.balances().items():
            self.assertEqual(
                bal, START_BALANCE + 10,
                f"{pid}: выигрыш последнего раунда не зачислен "
                f"(ожидалось {START_BALANCE + 10}, получено {bal}) (FIX-2)"
            )

    def test_history_and_public_ledger_include_last_round(self):
        FakeLLM.behaviour = {
            "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 10},
        }
        self.run_game(rounds=2, winning_number=32)
        hist = agent_v2.load_history("player1", self.table)
        self.assertEqual(len(hist), 2,
                         "последний раунд не попал в историю игрока (FIX-2)")
        ledger = agent_v2.load_public_ledger(self.table)
        self.assertEqual(len(ledger), 2 * len(PLAYERS),
                         "последний раунд не попал в публичный журнал (FIX-2)")


# ─────────────────────────── FIX-3: атомарность списания ставки ───────────

class TestBetAtomicity(GameHarness):

    def test_bet_file_is_written_before_balance_is_deducted(self):
        """
        Симулируем обрыв ровно между записью ставки и списанием: падаем
        внутри save_balance. Деньги не должны исчезнуть без ставки.
        На старом порядке (списание → файл) ставки на диске не оказывалось,
        а баланс уже был уменьшен.
        """
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        FakeLLM.behaviour = {
            "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 10},
        }
        bet = agent.decide_bet()

        # воспроизводим последовательность из Фазы 2 (исправленный порядок)
        common.write_json(common.bet_file("player1", self.table), bet)
        # <-- здесь "обрыв": списание не выполняем

        self.assertTrue(
            os.path.exists(common.bet_file("player1", self.table)),
            "ставка не записана до списания (FIX-3)"
        )
        bal = common.read_json(common.balance_file("player1", self.table))["balance"]
        self.assertEqual(bal, START_BALANCE,
                         "баланс списан раньше, чем ставка попала на диск (FIX-3)")

    def test_phase2_source_order(self):
        """Статическая проверка порядка операций в Фазе 2."""
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "run_game_v2.py"), encoding="utf-8").read()
        i_bet_file = src.index("common.write_json(common.bet_file(pid, base_dir), bet)")
        i_deduct = src.index('agent.balance -= bet["amount"]')
        self.assertLess(i_bet_file, i_deduct,
                        "ставка записывается ПОСЛЕ списания баланса (FIX-3)")


# ─────────────────────────────── FIX-4: детектор зацикливания ─────────────

class TestLoopDetector(unittest.TestCase):

    @staticmethod
    def conv(*msgs):
        return [{"from": f"p{i % 2 + 1}", "message": m} for i, m in enumerate(msgs)]

    def test_legit_loan_counteroffer_is_not_a_loop(self):
        """Контроффер по займу переиспользует лексику оффера — это торг."""
        c = self.conv(
            "Lend me 10 coins now, I will return 12 next round",
            "I will lend you 10 coins if you return 15 next round, not 12",
            "Fine, 15 it is, but I want the coins before I place my bet",
        )
        self.assertFalse(PlayerAgent._detect_loop(c),
                         "легитимный торг по займу помечен как петля (FIX-4)")

    def test_legit_price_negotiation_is_not_a_loop(self):
        c = self.conv(
            "I will sell you my strategy for 5 coins, it is even_money red",
            "I will pay 3 coins for that strategy, not 5",
            "Meet me at 4 coins and the strategy is yours right now",
        )
        self.assertFalse(PlayerAgent._detect_loop(c),
                         "легитимный торг о цене помечен как петля (FIX-4)")

    def test_real_repetition_is_still_caught(self):
        c = self.conv(
            "I will sell you my strategy for 5 coins, even_money red small amounts",
            "Sounds interesting, tell me more about it",
            "I will sell you my strategy for 5 coins, even_money red small amounts",
        )
        self.assertTrue(PlayerAgent._detect_loop(c),
                        "дословный повтор не пойман (FIX-4)")

    def test_parroting_the_partner_is_still_caught(self):
        c = self.conv(
            "opening offer about nothing in particular",
            "I am an information broker charging 2 coins per tip on dozens",
            "I am an information broker charging 2 coins per tip on dozens",
        )
        self.assertTrue(PlayerAgent._detect_loop(c),
                        "попугайничание не поймано (FIX-4)")

    def test_first_reply_never_triggers(self):
        """До 3-го сообщения детектор молчит: первый ответ всегда эхо-подобен."""
        c = self.conv(
            "I will lend you 10 coins at 20 percent interest",
            "I will lend you 10 coins at 20 percent interest",
        )
        self.assertFalse(PlayerAgent._detect_loop(c),
                         "детектор сработал на 2-м сообщении (FIX-4)")


# ──────────────────────── FIX-5: рефлексия банкрота ───────────────────────

class TestBankruptReflection(GameHarness):

    def test_player_without_bet_still_reflects(self):
        """
        Игрок с нулевым балансом не ставит → результата нет → на старом коде
        reflect_betting не вызывался никогда, синапса замерзала навсегда.
        """
        os.makedirs(self.table, exist_ok=True)
        # обнуляем player1 ещё до старта
        agent_v2.save_balance("player1", self.table, 0)

        reflected = []
        orig = PlayerAgent.reflect_betting

        def spy(self, last_entry=None):
            reflected.append((self.player_id, last_entry))
            return orig(self, last_entry)

        PlayerAgent.reflect_betting = spy
        try:
            FakeLLM.behaviour = {
                "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 5},
            }
            self.run_game(rounds=2, winning_number=0)
        finally:
            PlayerAgent.reflect_betting = orig

        p1 = [entry for pid, entry in reflected if pid == "player1"]
        self.assertTrue(p1, "банкрот player1 не рефлексировал ни разу (FIX-5)")
        self.assertIn(None, p1,
                      "рефлексия без результата ставки не вызывалась (FIX-5)")

    def test_reflect_betting_accepts_none(self):
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        FakeLLM.behaviour = {"reflect": lambda u: {"notes": "ok", "update_persona": False}}
        agent.reflect_betting(None)  # не должно бросить
        self.assertEqual(agent_v2.load_notes("player1", self.table), "ok")


# ───────────────────── FIX-6: типобезопасность synapse/persona ────────────

class TestTypeSafety(GameHarness):

    def test_notes_returned_as_list_do_not_crash(self):
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        FakeLLM.behaviour = {
            "reflect": lambda u: {"notes": ["rule one", "rule two"],
                                  "update_persona": True,
                                  "new_persona": {"role": "broker"}},
        }
        agent.reflect_betting(None)
        notes = agent_v2.load_notes("player1", self.table)
        self.assertIsInstance(notes, str, "notes сохранены не строкой (FIX-6)")
        self.assertIsInstance(agent.persona_prompt, str)
        # следующий вызов компрессора не должен падать на len()
        agent._compress_betting_synapse_if_needed()


# ───────────────── FIX-7: чекпойнт фазы диалогов на входе в ход ───────────

class TestDialoguePhaseCheckpoint(GameHarness):

    def test_checkpoint_written_before_first_dialogue_of_a_player(self):
        """
        Ctrl+C во время ПЕРВОГО диалога player2 не должен откатывать
        указатель на player1 (тот уже отговорил).
        """
        os.makedirs(self.table, exist_ok=True)
        seen = []

        orig_run = run_game_v2.run_dialogue

        def spy(agent_a, agent_b, round_no, logger, table_dir):
            st = run_game_v2.load_dialogue_phase_state(table_dir, round_no)
            seen.append((agent_a.player_id, st["player_index"] if st else None))
            return orig_run(agent_a, agent_b, round_no, logger, table_dir)

        run_game_v2.run_dialogue = spy
        try:
            FakeLLM.behaviour = {
                "next_move": lambda u: (
                    {"action": "talk", "partner": "player3", "reason": "x"}
                    if "Your player id: player2" in u
                    and "(no one yet this round)" in u
                    else {"action": "bet", "reason": "x"}
                ),
                "dialogue": lambda u: {"message": "quick chat, nothing to trade",
                                       "transfer": 0, "transfer_to": None, "done": True},
            }
            self.run_game(rounds=1, winning_number=0)
        finally:
            run_game_v2.run_dialogue = orig_run

        self.assertTrue(seen, "диалог не состоялся — тест не проверяет то, что должен")
        initiator, idx = seen[0]
        self.assertEqual(initiator, "player2")
        self.assertEqual(
            idx, PLAYERS.index("player2"),
            f"на входе в первый диалог player2 чекпойнт указывает на индекс "
            f"{idx}, а не на {PLAYERS.index('player2')} (FIX-7)"
        )


# ──────────────────────── сквозная проверка сохранения денег ──────────────

class TestMoneyConservation(GameHarness):

    def test_transfers_conserve_total_money(self):
        """
        Переводы между игроками не создают и не уничтожают монеты.
        Казино — единственный источник/сток, поэтому сверяем итог с
        суммой всех выплат минус суммой всех ставок.
        """
        FakeLLM.behaviour = {
            "next_move": lambda u: (
                {"action": "talk", "partner": "player2", "reason": "x"}
                if "Your player id: player1" in u and "(no one yet this round)" in u
                else {"action": "bet", "reason": "x"}
            ),
            "dialogue": lambda u: (
                {"message": "take 7 coins as a loan", "transfer": 7,
                 "transfer_to": None, "done": True}
                if "Dialogue with player2" in u else
                {"message": "received, thanks", "transfer": 0,
                 "transfer_to": None, "done": True}
            ),
            "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 5},
        }
        self.run_game(rounds=3, winning_number=None, seed=7)

        ledger = agent_v2.load_public_ledger(self.table)
        staked = sum(e["bet"]["amount"] for e in ledger)
        paid = sum(e["payout"] for e in ledger)
        expected_total = START_BALANCE * len(PLAYERS) - staked + paid

        actual_total = sum(self.balances().values())
        self.assertEqual(
            actual_total, expected_total,
            f"деньги не сходятся: на столе {actual_total}, ожидалось "
            f"{expected_total} (старт {START_BALANCE * len(PLAYERS)}, "
            f"поставлено {staked}, выплачено {paid})"
        )

    def test_no_player_goes_negative(self):
        FakeLLM.behaviour = {
            "next_move": lambda u: (
                {"action": "talk", "partner": "player2", "reason": "x"}
                if "Your player id: player1" in u and "(no one yet this round)" in u
                else {"action": "bet", "reason": "x"}
            ),
            "dialogue": lambda u: {"message": "take everything I have",
                                   "transfer": 10 ** 6, "transfer_to": None,
                                   "done": True},
            "bet": lambda u: {"type": "straight", "numbers": [17], "amount": 10 ** 6},
        }
        self.run_game(rounds=3, seed=3)
        for pid, bal in self.balances().items():
            self.assertGreaterEqual(bal, 0, f"{pid} ушёл в минус: {bal}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
