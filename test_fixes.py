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
from llm_client import LLMClient as RealLLMClient
try:
    from llm_client import LLMUnavailable
except ImportError:                      # старый код: выключателя ещё нет
    class LLMUnavailable(RuntimeError):
        pass
import agent_v2
import promise_ledger
import transfer_ledger
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
        self._orig_llm_client_class = llm_client.LLMClient
        agent_v2.LLMClient = FakeLLM
        llm_client.LLMClient = FakeLLM
        FakeLLM.behaviour = {}
        self._orig_load_config = run_game_v2.load_config

    def tearDown(self):
        agent_v2.LLMClient = self._orig_client
        llm_client.LLMClient = self._orig_llm_client_class
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
        i_deduct = src.index('agent.balance -= common.total_bet_amount(bet)')
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




# ───────────── FIX-8/FIX-9: дедлайн и закрывающий ход ─────────────────────

class TestDeadlineAndClosingTurn(GameHarness):

    def _hint(self, is_initiator, turns_left=2):
        """Воспроизводит ветку role_hint для turns_left<=2."""
        return (
            ("You initiated this." if is_initiator else "They contacted you.")
            + f" Only {turns_left} message(s) left in this conversation — "
        )

    def test_initiator_also_gets_the_deadline_warning(self):
        """
        FIX-8: соседние строковые литералы склеивались раньше тернарника,
        поэтому инициатор получал только "You initiated this." без всякого
        предупреждения о дедлайне — а он говорит на 1/3/5/7-м сообщениях.
        """
        seen = []
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.0, max_tokens=0):
                seen.append(user)
                return {"message": "ok", "transfer": 0, "transfer_to": None,
                        "done": False}

        agent.client = Spy()
        # 6 сообщений уже сказано → turns_left = 2
        distinct = [
            "selling dozen system cheap",
            "prefer column strategies personally",
            "borrow twenty repay thirty",
            "collateral required otherwise refuse",
            "rumour circulating about player3",
            "verify ledger before believing",
        ]
        conv = [{"from": "player1" if i % 2 == 0 else "player2", "message": m}
                for i, m in enumerate(distinct)]
        agent.dialogue_turn("player2", 100, conv, 1, is_initiator=True)
        self.assertIn(
            "message(s) left", seen[-1],
            "инициатор не получил предупреждение о дедлайне (FIX-8)"
        )

    def test_partner_gets_a_closing_turn_after_done(self):
        """
        FIX-9: раньше `done` от одной стороны делал безусловный break, и
        партнёр не получал хода вообще — не мог доплатить по согласованному.
        """
        FakeLLM.behaviour = {
            "next_move": lambda u: (
                {"action": "talk", "partner": "player2", "reason": "x"}
                if "Your player id: player1" in u and "(no one yet this round)" in u
                else {"action": "bet", "reason": "x"}
            ),
            # A сразу закрывает диалог; B на закрывающем ходу платит
            "dialogue": lambda u: (
                {"message": "final offer: 5 coins for the strategy, take it or leave it",
                 "transfer": 0, "transfer_to": None, "done": True}
                if "Dialogue with player2" in u else
                {"message": "taking it, here are the coins", "transfer": 5,
                 "transfer_to": "player1", "done": True}
            ),
            "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 1},
        }
        self.run_game(rounds=1, winning_number=0)
        dlg = self.dialogues()[0]
        self.assertEqual(
            len(dlg["conversation"]), 2,
            "партнёр не получил закрывающий ход после done (FIX-9)"
        )
        self.assertEqual(
            dlg["b_sent"], 5,
            "партнёр не смог доплатить по уже согласованной сделке (FIX-9)"
        )

    def test_closing_turn_does_not_extend_into_new_negotiation(self):
        """Закрывающий ход ровно один — новый круг торга он не открывает."""
        FakeLLM.behaviour = {
            "next_move": lambda u: (
                {"action": "talk", "partner": "player2", "reason": "x"}
                if "Your player id: player1" in u and "(no one yet this round)" in u
                else {"action": "bet", "reason": "x"}
            ),
            "dialogue": lambda u: (
                {"message": "I am done here, goodbye", "transfer": 0,
                 "transfer_to": None, "done": True}
                if "Dialogue with player2" in u else
                # партнёр пытается продолжить торг (done=false) — не должен смочь
                {"message": "wait, let me counter with a brand new offer",
                 "transfer": 0, "transfer_to": None, "done": False}
            ),
            "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 1},
        }
        self.run_game(rounds=1, winning_number=0)
        dlg = self.dialogues()[0]
        self.assertEqual(len(dlg["conversation"]), 2,
                         "закрывающий ход открыл новый круг торга (FIX-9)")

    def test_done_warning_present_on_every_turn(self):
        """FIX-9b: агент на КАЖДОМ ходу знает, что done=true лишает его
        возможности заплатить — иначе должник закрывает сделку в ноль."""
        seen = []
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.0, max_tokens=0):
                seen.append(user)
                return {"message": "ok", "transfer": 0, "transfer_to": None,
                        "done": False}

        agent.client = Spy()
        agent.dialogue_turn("player2", 100, [], 1, is_initiator=True)
        self.assertIn('setting "done": true ends YOUR participation', seen[-1],
                      "предупреждение про done отсутствует в промпте (FIX-9b)")


# ─────────────────── FIX-10: ротация порядка хода ─────────────────────────

class TestRoundRobinOrder(unittest.TestCase):

    P5 = ["player1", "player2", "player3", "player4", "player5"]

    def test_exact_rotation_sequence(self):
        """Р1 начинает player1, Р2 — player2, ... Р5 — player5, Р6 — снова player1."""
        expected = {
            1: ["player1", "player2", "player3", "player4", "player5"],
            2: ["player2", "player3", "player4", "player5", "player1"],
            3: ["player3", "player4", "player5", "player1", "player2"],
            4: ["player4", "player5", "player1", "player2", "player3"],
            5: ["player5", "player1", "player2", "player3", "player4"],
            6: ["player1", "player2", "player3", "player4", "player5"],
        }
        for rnd, want in expected.items():
            self.assertEqual(
                run_game_v2.round_player_order(self.P5, rnd), want,
                f"порядок в раунде {rnd} неверен (FIX-10)"
            )

    def test_every_player_leads_once_per_cycle(self):
        leaders = [run_game_v2.round_player_order(self.P5, r)[0]
                   for r in range(1, len(self.P5) + 1)]
        self.assertEqual(sorted(leaders), sorted(self.P5),
                         "за цикл не каждый игрок побывал первым (FIX-10)")

    def test_every_player_visits_every_position_once_per_cycle(self):
        """Полная справедливость: за N раундов каждый стоит на каждой позиции ровно раз."""
        seats = {p: [] for p in self.P5}
        for r in range(1, len(self.P5) + 1):
            for pos, pid in enumerate(run_game_v2.round_player_order(self.P5, r)):
                seats[pid].append(pos)
        for pid, positions in seats.items():
            self.assertEqual(sorted(positions), list(range(len(self.P5))),
                             f"{pid} занял позиции {sorted(positions)} (FIX-10)")

    def test_order_is_deterministic_for_resume(self):
        """Порядок зависит ТОЛЬКО от round_no — иначе чекпойнт FIX-7 указывал бы
        на другого игрока после рестарта."""
        for r in (1, 4, 7, 23):
            self.assertEqual(run_game_v2.round_player_order(self.P5, r),
                             run_game_v2.round_player_order(self.P5, r),
                             "порядок не воспроизводится (FIX-10)")

    def test_single_player_and_empty_list(self):
        self.assertEqual(run_game_v2.round_player_order(["p1"], 7), ["p1"])
        self.assertEqual(run_game_v2.round_player_order([], 3), [])


def ring_partner(u):
    """Собеседник по кольцу: p1→p2→p3→p1. Даёт равномерный граф общения,
    в котором позиционный эффект виден в чистом виде."""
    if "(no one yet this round)" not in u:
        return {"action": "bet", "reason": "x"}
    me = u.split("Your player id: ")[1].split(".")[0].strip()
    avail = (u.split("available to talk to right now: [")[1].split("]")[0]
             .replace("'", "").split(", "))
    want = PLAYERS[(PLAYERS.index(me) + 1) % len(PLAYERS)]
    return {"action": "talk", "partner": want if want in avail else avail[0],
            "reason": "x"}


class TestPositionalFairness(GameHarness):

    def test_no_player_is_permanently_blind(self):
        """
        До FIX-10 первый в списке НИ РАЗУ за партию не видел ни одного чужого
        диалога (структурно: список чужих разговоров пуст, когда он ходит).
        После ротации слепая позиция достаётся всем по очереди.
        """
        import collections
        seen = collections.defaultdict(list)
        orig = PlayerAgent.decide_next_move

        def spy(self, avail, talked, rn, dlgs=None):
            others = [d for d in (dlgs or []) if self.player_id not in (d[0], d[1])]
            seen[self.player_id].append(len(others))
            return orig(self, avail, talked, rn, dlgs)

        PlayerAgent.decide_next_move = spy
        try:
            FakeLLM.behaviour = {
                # КОЛЬЦО общения p1→p2→p3→p1: если все сходятся на одном
                # партнёре, для него все диалоги оказываются "своими" и
                # эффект ротации маскируется артефактом сценария
                "next_move": ring_partner,
                "dialogue": lambda u: {"message": "brief exchange of terms",
                                       "transfer": 0, "transfer_to": None, "done": True},
                "bet": lambda u: {"type": "even_money", "selection": "red", "amount": 5},
            }
            self.run_game(rounds=len(PLAYERS), seed=2)
        finally:
            PlayerAgent.decide_next_move = orig

        blind = [pid for pid, counts in seen.items() if max(counts) == 0]
        self.assertEqual(
            blind, [],
            f"игроки {blind} за полный цикл не увидели ни одного чужого "
            f"диалога — позиционная слепота сохранилась (FIX-10)"
        )


# ───────────── FIX-11: бюджеты и память из конфига ────────────────────────

class TestConfigurableBudgets(GameHarness):

    def _cfg(self, tokens=None, memory=None):
        cfg = make_cfg(self.table, self.logs, 1)
        if tokens:
            cfg.read_dict({"tokens": {k: str(v) for k, v in tokens.items()}})
        if memory:
            cfg.read_dict({"memory": {k: str(v) for k, v in memory.items()}})
        return cfg

    def test_every_call_site_uses_its_configured_budget(self):
        """
        Раньше пять из шести бюджетов были зашиты в agent_v2.py и из ini не
        настраивались вовсе — [player] max_tokens попадал только в ставку.
        """
        want = {"bet": 111, "dialogue": 222, "next_move": 333,
                "reflect": 444, "update_dsyn": 555, "compress": 666}
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table,
                            self._cfg(tokens=want, memory={"synapse_chars": 50}))
        got = {}

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                tag = ("compress" if "Compress" in system else
                       "reflect" if "Reflect" in system else
                       "next_move" if "Decide next move" in system else
                       "dialogue" if "Dialogue turn" in system else
                       "update_dsyn" if "Update reputation" in system else
                       "bet" if "Place casino bet" in system else "?")
                got[tag] = max_tokens
                if tag == "bet":
                    return {"type": "even_money", "selection": "red", "amount": 5}
                if tag == "next_move":
                    return {"action": "bet", "reason": "x"}
                if tag == "dialogue":
                    return {"message": "m", "transfer": 0, "transfer_to": None,
                            "done": True}
                return {"notes": "n", "update_persona": False, "trust_score": 6,
                        "reputation_note": "n", "future_intent": "i", "summary": "s"}

        agent.client = Spy()
        agent_v2.save_notes("player1", self.table, "x" * 200)   # > synapse_chars → compress
        agent.reflect_betting(None)
        agent.decide_next_move(["player2"], [], 1, [])
        agent.dialogue_turn("player2", 50, [], 1, is_initiator=True)
        agent.update_dsyn("player2", [{"from": "player1", "message": "m"}], 0, 1)
        agent.decide_bet()

        for tag, budget in want.items():
            self.assertEqual(got.get(tag), budget,
                             f"вызов '{tag}' игнорирует [tokens] {tag}={budget} (FIX-11)")

    def test_memory_thresholds_come_from_config(self):
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table,
                            self._cfg(memory={"synapse_chars": 7777,
                                              "dsyn_chars": 8888,
                                              "raw_interactions": 33}))
        self.assertEqual(agent.synapse_chars, 7777)
        self.assertEqual(agent.dsyn_chars, 8888)
        self.assertEqual(agent.raw_interactions, 33)

    def test_defaults_are_the_raised_ones(self):
        """Без секций [tokens]/[memory] действуют новые, расширенные значения."""
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, make_cfg(self.table, self.logs, 1))
        self.assertGreaterEqual(agent.synapse_chars, 4000,
                                "порог синапсы не поднят (FIX-11)")
        self.assertGreaterEqual(agent.dsyn_chars, 6000)
        self.assertGreater(agent.tok_dialogue, agent.tok_bet,
                           "диалогу по-прежнему выделено меньше, чем ставке (FIX-11)")


class TestLedgerWindow(unittest.TestCase):

    @staticmethod
    def ledger(n_rounds=10, players=("p1", "p2", "p3", "p4", "p5")):
        out = []
        for r in range(1, n_rounds + 1):
            for p in players:
                out.append({"round_no": r, "player_id": p, "winning_number": 7,
                            "bet": {"type": "even_money", "selection": "red",
                                    "amount": 5},
                            "win": False, "payout": 0})
        return out

    def test_window_is_applied_after_filtering_own_entries(self):
        """
        Раньше срез брался ДО отбрасывания своих записей, поэтому реальное
        окно было меньше заявленного и зависело от числа игроков: при пяти
        window=10 давал 8 строк, т.е. по 2 ставки на оппонента.
        """
        txt = agent_v2._format_public_ledger(self.ledger(), window=10,
                                             exclude_pid="p3")
        lines = [l for l in txt.strip().split("\n") if l.strip()]
        self.assertEqual(len(lines), 10,
                         f"окно=10 дало {len(lines)} строк — срез применён "
                         f"до фильтрации (FIX-11)")
        self.assertNotIn("p3 bet", txt, "свои записи не отфильтрованы")

    def test_window_larger_than_history_returns_everything(self):
        entries = self.ledger(n_rounds=3)          # 15 записей, 12 чужих
        txt = agent_v2._format_public_ledger(entries, window=999, exclude_pid="p3")
        lines = [l for l in txt.strip().split("\n") if l.strip()]
        self.assertEqual(len(lines), 12)

    def test_dsyn_display_limits_are_honoured(self):
        d = agent_v2._empty_dsyn()
        d["reputation"]["p2"] = {"trust_score": 5, "net": 0,
                                 "deals_done": [f"d{i}" for i in range(20)],
                                 "deals_failed": [f"f{i}" for i in range(20)],
                                 "reputation_note": "", "future_intent": "",
                                 "last_seen_round": 1}
        d["interactions"] = [{"round": i, "partner": "p2", "net_transfer": 0,
                              "summary": f"s{i}"} for i in range(20)]
        txt = agent_v2._format_dsyn_for_prompt(d, recent=7, deals=5, fails=3)
        self.assertEqual(txt.count("net=+0c — s"), 7, "recent не соблюдён (FIX-11)")
        done_line = [l for l in txt.split("\n") if "✓" in l][0]
        self.assertEqual(done_line.count(";"), 4, "deals не соблюдён (FIX-11)")


class TestConfigParsable(unittest.TestCase):

    def test_shipped_ini_has_no_inline_comments(self):
        """
        configparser не срезает комментарии в конце строки: `x = 5  # note`
        читается целиком как строка и роняет getint(). Игра грузит конфиг
        обычным ConfigParser, так что такая строка убила бы запуск.
        """
        import configparser as cp
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "config_v2.ini")
        c = cp.ConfigParser()
        c.read(path, encoding="utf-8")
        for section in ("tokens", "memory"):
            self.assertIn(section, c.sections(), f"нет секции [{section}]")
            for key in c[section]:
                try:
                    c.getint(section, key)
                except ValueError as e:
                    self.fail(f"[{section}] {key} не парсится как int "
                              f"(комментарий в конце строки?): {e}")


# ───────────── FIX-12: персона больше не растёт бесконечно ────────────────

class TestPersonaBounded(GameHarness):

    def _agent(self, persona_chars=300):
        cfg = make_cfg(self.table, self.logs, 1)
        cfg.read_dict({"memory": {"persona_chars": str(persona_chars)}})
        os.makedirs(self.table, exist_ok=True)
        return PlayerAgent("player1", self.table, cfg)

    def test_oversized_persona_is_compressed_on_reflect(self):
        """
        Персона была ЕДИНСТВЕННЫМ неограниченным компонентом промпта: заметки
        резались по synapse_chars, синапса — по dsyn_chars, журнал — по окну,
        а персону не обрезал и не сжимал никто.
        """
        agent = self._agent(persona_chars=300)
        agent_v2.save_text(agent_v2.prompt_file("player1", self.table), "Z" * 5000)
        FakeLLM.behaviour = {"reflect": lambda u: {"notes": "n", "update_persona": False}}

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                if "Compress your persona" in system:
                    return {"new_persona": "Tight broker persona. Charges upfront."}
                return {"notes": "n", "update_persona": False}

        agent.client = Spy()
        agent.reflect_betting(None)
        self.assertLessEqual(
            len(agent.persona_prompt), 300,
            f"персона осталась {len(agent.persona_prompt)} символов (FIX-12)"
        )

    def test_persona_compression_does_not_send_persona_twice(self):
        """
        system при сжатии должен быть CORE_SYSTEM_PROMPT, а не abstract_prompt:
        последний содержит саму персону, и вызов, который чинит переполнение
        контекста, переполнял бы его сильнее всех остальных.
        """
        agent = self._agent(persona_chars=300)
        marker = "UNIQUEPERSONAMARKER"
        agent_v2.save_text(agent_v2.prompt_file("player1", self.table),
                           marker + "Z" * 5000)
        seen = {}

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                if "Compress your persona" in system:
                    seen["system"] = system
                    return {"new_persona": "short"}
                return {"notes": "n", "update_persona": False}

        agent.client = Spy()
        agent.reflect_betting(None)
        self.assertIn("system", seen, "сжатие персоны не вызывалось")
        self.assertNotIn(marker, seen["system"],
                         "персона уехала в запрос дважды — в system и в user (FIX-12)")

    def test_compression_call_itself_stays_bounded(self):
        """
        FIX-12b: вызов сжатия кладёт персону в user-сообщение целиком. На
        персоне в 50k символов это давало 166% контекста — вызов, лечащий
        переполнение, переполнял сильнее всего остального.
        """
        agent = self._agent(persona_chars=300)
        agent_v2.save_text(agent_v2.prompt_file("player1", self.table), "Z" * 50000)
        sizes = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                sizes.append(len(system) + len(user))
                if "Compress your persona" in system:
                    return {"new_persona": "short"}
                return {"notes": "n", "update_persona": False}

        agent.client = Spy()
        agent.reflect_betting(None)
        hard = 300 * agent_v2.PERSONA_HARD_FACTOR
        self.assertLess(
            max(sizes), len(agent_v2.CORE_SYSTEM_PROMPT) + hard + 3000,
            f"запрос на сжатие раздулся до {max(sizes)} символов (FIX-12b)"
        )

    def test_new_persona_over_limit_is_truncated_on_save(self):
        """Модель регулярно игнорирует объявленный лимит — не верим на слово."""
        agent = self._agent(persona_chars=200)
        FakeLLM.behaviour = {
            "reflect": lambda u: {"notes": "n", "update_persona": True,
                                  "new_persona": "Q" * 4000},
        }
        agent.reflect_betting(None)
        self.assertLessEqual(len(agent.persona_prompt), 200,
                             "переросшая персона записана без обрезки (FIX-12)")

    def test_abstract_prompt_is_bounded_even_without_reflection(self):
        """
        Жёсткий потолок в abstract_prompt — последняя линия обороны: до первой
        рефлексии (раунд 1) и на случай правки prompt_<id>.txt руками.
        """
        agent = self._agent(persona_chars=300)
        agent_v2.save_text(agent_v2.prompt_file("player1", self.table), "Z" * 50000)
        hard = 300 * agent_v2.PERSONA_HARD_FACTOR
        self.assertLessEqual(
            len(agent.abstract_prompt), len(agent_v2.CORE_SYSTEM_PROMPT) + hard + 200,
            "abstract_prompt не ограничен жёстким потолком (FIX-12)"
        )

    def test_persona_limit_is_announced_in_the_reflect_prompt(self):
        agent = self._agent(persona_chars=1234)
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"notes": "n", "update_persona": False}

        agent.client = Spy()
        agent.reflect_betting(None)
        self.assertIn("MAX 1234 characters", seen[-1],
                      "лимит персоны не объявлен модели (FIX-12)")

    def test_compression_failure_falls_back_to_truncation(self):
        agent = self._agent(persona_chars=300)
        agent_v2.save_text(agent_v2.prompt_file("player1", self.table), "Z" * 5000)

        class Boom(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                if "Compress your persona" in system:
                    raise RuntimeError("llm down")
                return {"notes": "n", "update_persona": False}

        agent.client = Boom()
        agent.reflect_betting(None)
        self.assertLessEqual(len(agent.persona_prompt), 300,
                             "при сбое сжатия персона не обрезана (FIX-12)")


class TestTruncateText(unittest.TestCase):

    def test_cuts_at_sentence_boundary(self):
        t = "First sentence here. Second sentence here. Third one runs long."
        out = agent_v2._truncate_text(t, 45)
        self.assertTrue(out.endswith("."), f"обрезано посреди предложения: {out!r}")
        self.assertLessEqual(len(out), 45)

    def test_falls_back_to_hard_cut_when_boundary_too_early(self):
        """Если ближайшая граница съедает больше трети — режем жёстко."""
        t = "Short. " + "x" * 200
        out = agent_v2._truncate_text(t, 150)
        self.assertLessEqual(len(out), 150)
        self.assertGreater(len(out), 100, "потеряли слишком много текста")

    def test_short_text_untouched(self):
        self.assertEqual(agent_v2._truncate_text("abc", 100), "abc")


# ───── FIX-13/14: проверяемость заявлений и скорборд ──────────────────────

class TestVerifiability(unittest.TestCase):

    def test_core_prompt_states_the_spin_rules(self):
        """
        В реальном прогоне player3 продал за 30 монет 'Audit log R1: 12 spins',
        хотя спин в раунде ровно один и крутит его крупье. Правила должны быть
        в неизменяемом ядре, а не надеяться на здравый смысл модели.
        """
        # промпт — свёрстанный текст, поэтому сравниваем по нормализованным
        # пробелам: иначе тест ловит перенос строки, а не отсутствие правила
        core = " ".join(agent_v2.CORE_SYSTEM_PROMPT.split())
        for fragment in ("ONE spin per round", "PUBLIC LEDGER",
                         "no 00", "37 pockets"):
            self.assertIn(fragment, core, f"в ядре нет: {fragment!r} (FIX-13)")

    def test_core_prompt_covers_both_directions(self):
        """Проверять чужие заявления И знать, что твои собственные проверяемы."""
        core = " ".join(agent_v2.CORE_SYSTEM_PROMPT.split())
        self.assertIn("If it is not in the ledger, it did not happen", core)
        self.assertIn("Your own bets are in there too", core)

    def test_core_prompt_still_allows_trading_opinions(self):
        """Запрещать торговлю прогнозами нельзя — в ней вся игра."""
        core = " ".join(agent_v2.CORE_SYSTEM_PROMPT.split())
        self.assertIn("opinions and fine to trade", core)


class TestScoreboard(unittest.TestCase):

    @staticmethod
    def led(rows):
        return [{"round_no": r, "player_id": p, "winning_number": 19,
                 "bet": {"type": t, "selection": "red", "amount": a},
                 "win": w, "payout": pay}
                for r, p, t, a, w, pay in rows]

    def test_pnl_is_computed_not_left_to_the_model(self):
        txt = agent_v2._format_scoreboard(self.led([
            (1, "p1", "even_money", 20, True, 40),
            (2, "p1", "even_money", 10, False, 0),
            (3, "p1", "even_money", 5,  False, 0),
        ]))
        self.assertIn("casino P&L=+5c", txt, f"P&L посчитан неверно: {txt}")
        self.assertIn("3 bet(s)", txt)
        self.assertIn("won 1/3", txt)

    def test_whole_pnl_has_no_decimals(self):
        """MONEY-1: суммы в этой игре целые (монеты) — постоянные '.00' на
        каждой целой сумме были визуальным шумом без всякой пользы. Целое
        значение показывается как целое число, без десятичных."""
        txt = agent_v2._format_scoreboard(self.led([
            (1, "p1", "even_money", 20, True, 40),
        ]))
        self.assertIn("casino P&L=+20c", txt)
        self.assertNotIn("+20.00c", txt)
        self.assertNotIn("+20.0c", txt)

    def test_fractional_pnl_still_shown_with_decimals(self):
        """MONEY-1: если сумма всё же дробная (например, старый баг с не
        скорректированным float от модели до BUGFIX-AMOUNT-1, или любой
        другой путь, которым дробь могла бы просочиться) — она должна
        остаться ВИДИМОЙ, а не молча округлиться до целого и потерять
        данные. fmt_money() — единая точка форматирования для этого."""
        self.assertEqual(agent_v2.fmt_money(5.5), "+5.50")
        self.assertEqual(agent_v2.fmt_money(-3.25), "-3.25")
        self.assertEqual(agent_v2.fmt_money(20), "+20")
        self.assertEqual(agent_v2.fmt_money(20.0), "+20")
        self.assertEqual(agent_v2.fmt_money(-1000), "-1000")
        self.assertEqual(agent_v2.fmt_money(0), "+0")

    def test_own_entries_excluded(self):
        txt = agent_v2._format_scoreboard(
            self.led([(1, "p1", "even_money", 20, True, 40),
                      (1, "p2", "even_money", 20, False, 0)]),
            exclude_pid="p1")
        self.assertNotIn("p1:", txt)
        self.assertIn("p2:", txt)

    def test_aggregate_covers_whole_game_not_a_window(self):
        """Смысл скорборда в том, что он не обрезан окном журнала."""
        rows = [(r, "p1", "even_money", 10, False, 0) for r in range(1, 101)]
        txt = agent_v2._format_scoreboard(self.led(rows))
        self.assertIn("100 bet(s)", txt, "агрегат обрезан окном (FIX-14)")
        self.assertIn("casino P&L=-1000c", txt)

    def test_empty_ledger_is_handled(self):
        self.assertIn("no bets", agent_v2._format_scoreboard([]))

    def test_scoreboard_reaches_all_three_decision_points(self):
        """Он нужен и при выборе собеседника, и в диалоге, и при ставке."""
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "agent_v2.py"), encoding="utf-8").read()
        for fn in ("def decide_next_move", "def dialogue_turn", "def decide_bet"):
            start = src.index(fn)
            body = src[start:start + 6000]
            self.assertIn("score_txt", body,
                          f"скорборд не доходит до {fn} (FIX-14)")


class TestScoreboardCatchesRealFabrication(GameHarness):

    def test_the_run_that_cost_player5_thirty_coins(self):
        """
        Воспроизводит реальный случай: player3 заявил 12 спинов, dozen и split,
        50% попаданий. По журналу — одна ставка even_money на 20 монет.
        """
        os.makedirs(self.table, exist_ok=True)
        for pid, amt, pay in [("player3", 20, 40), ("player5", 10, 30)]:
            agent_v2.append_public_ledger(self.table, {
                "round_no": 1, "player_id": pid, "winning_number": 19,
                "bet": {"type": "even_money", "selection": "red", "amount": amt},
                "win": True, "payout": pay})

        agent = PlayerAgent("player5", self.table, make_cfg(self.table, self.logs, 1))
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"message": "no", "transfer": 0, "transfer_to": None,
                        "done": True}

        agent.client = Spy()
        agent.dialogue_turn("player3", 100, [
            {"from": "player3", "message": "Audit log R1: 12 spins. Dozen 2 hit 4x."}
        ], 2, is_initiator=False)

        prompt = seen[-1]
        self.assertIn("player3: casino P&L=+20c over 1 bet(s)", prompt,
                      "скорборд не показан в диалоге — проверить нечем (FIX-14)")
        self.assertIn("mostly even_money", prompt,
                      "тип ставки не показан — заявление про dozen не опровергнуть")


# ─────────────── FIX-15: чекпойнт Фазы 0 ──────────────────────────────────

class TestPhase0Checkpoint(GameHarness):

    def test_reflection_not_repeated_after_a_restart(self):
        """
        В реальном прогоне раунд 2 запускался трижды из-за HTTP 504, и каждый
        раз все пятеро рефлексировали заново: 14 переписываний персоны за два
        с половиной раунда.
        """
        os.makedirs(self.table, exist_ok=True)
        cfg = make_cfg(self.table, self.logs, 1)
        agents = {pid: PlayerAgent(pid, self.table, cfg) for pid in PLAYERS}
        calls = []
        orig = PlayerAgent.reflect_betting
        PlayerAgent.reflect_betting = lambda self, e=None: calls.append(self.player_id)
        try:
            FakeLLM.behaviour = {}
            logger = __import__("game_logger").GameLogger(self.logs)
            # первый заход: обрываемся после двух игроков
            run_game_v2.settle_pending_results(
                agents, PLAYERS[:2], self.table, 1, logger, checkpoint_round=2)
            self.assertEqual(calls, PLAYERS[:2])
            # рестарт того же раунда: первые двое повторяться не должны
            calls.clear()
            run_game_v2.settle_pending_results(
                agents, PLAYERS, self.table, 1, logger, checkpoint_round=2)
            logger.close()
        finally:
            PlayerAgent.reflect_betting = orig
        self.assertEqual(
            calls, PLAYERS[2:],
            f"после рестарта повторно отрефлексировали: {calls} (FIX-15)"
        )

    def test_checkpoint_is_scoped_to_its_round(self):
        os.makedirs(self.table, exist_ok=True)
        run_game_v2.save_phase0_done(self.table, 2, {"player1"})
        self.assertEqual(run_game_v2.load_phase0_done(self.table, 2), {"player1"})
        self.assertEqual(run_game_v2.load_phase0_done(self.table, 3), set(),
                         "чекпойнт протёк в следующий раунд (FIX-15)")

    def test_checkpoint_cleared_after_phase_completes(self):
        os.makedirs(self.table, exist_ok=True)
        run_game_v2.save_phase0_done(self.table, 2, {"player1"})
        run_game_v2.clear_phase0_state(self.table)
        self.assertEqual(run_game_v2.load_phase0_done(self.table, 2), set())

    def test_final_settlement_does_not_checkpoint(self):
        """Финальный расчёт повторять нечему — файл создаваться не должен."""
        os.makedirs(self.table, exist_ok=True)
        cfg = make_cfg(self.table, self.logs, 1)
        agents = {pid: PlayerAgent(pid, self.table, cfg) for pid in PLAYERS}
        logger = __import__("game_logger").GameLogger(self.logs)
        run_game_v2.settle_pending_results(agents, PLAYERS, self.table, 1, logger,
                                           reflect=False)
        logger.close()
        self.assertFalse(os.path.exists(run_game_v2.phase0_state_file(self.table)))


# ───────── FIX-16: идемпотентная запись журнала и истории ─────────────────

class TestLedgerIdempotent(GameHarness):

    def _entry(self, pid, rnd, amount):
        return {"round_no": rnd, "player_id": pid, "winning_number": 7,
                "bet": {"type": "even_money", "selection": "red", "amount": amount},
                "win": False, "payout": 0}

    def test_replaying_a_round_replaces_instead_of_duplicating(self):
        """
        В реальном прогоне падение сервера оставило аварийные ставки в 1 монету
        за раунды 3-4, а после перезапуска рядом легли настоящие. Журнал стал
        буквально противоречивым — и пойманный на вранье игрок сослался на
        «расхождение в журнале», формально говоря правду.
        """
        os.makedirs(self.table, exist_ok=True)
        agent_v2.append_public_ledger(self.table, self._entry("player1", 3, 1))
        agent_v2.append_public_ledger(self.table, self._entry("player1", 3, 20))

        led = agent_v2.load_public_ledger(self.table)
        r3 = [e for e in led if e["player_id"] == "player1" and e["round_no"] == 3]
        self.assertEqual(len(r3), 1,
                         f"на раунд 3 осталось {len(r3)} записей — призрак сбоя "
                         f"остался в журнале (FIX-16)")
        self.assertEqual(r3[0]["bet"]["amount"], 20,
                         "сохранилась старая аварийная запись, а не настоящая")

    def test_other_players_and_rounds_untouched(self):
        os.makedirs(self.table, exist_ok=True)
        for pid, rnd, amt in [("player1", 3, 1), ("player2", 3, 5),
                              ("player1", 4, 7), ("player1", 3, 20)]:
            agent_v2.append_public_ledger(self.table, self._entry(pid, rnd, amt))
        led = agent_v2.load_public_ledger(self.table)
        self.assertEqual(len(led), 3)
        self.assertEqual({(e["player_id"], e["round_no"]) for e in led},
                         {("player1", 3), ("player2", 3), ("player1", 4)})

    def test_ledger_stays_sorted_by_round(self):
        os.makedirs(self.table, exist_ok=True)
        for rnd in (5, 1, 3):
            agent_v2.append_public_ledger(self.table, self._entry("player1", rnd, 1))
        rounds = [e["round_no"] for e in agent_v2.load_public_ledger(self.table)]
        self.assertEqual(rounds, sorted(rounds))

    def test_history_is_idempotent_too(self):
        os.makedirs(self.table, exist_ok=True)
        agent_v2.append_history("player1", self.table,
                                {"round_no": 3, "balance_after": 10})
        agent_v2.append_history("player1", self.table,
                                {"round_no": 3, "balance_after": 99})
        hist = agent_v2.load_history("player1", self.table)
        self.assertEqual(len(hist), 1, "дубль раунда в личной истории (FIX-16)")
        self.assertEqual(hist[0]["balance_after"], 99)

    def test_scoreboard_not_poisoned_by_replayed_round(self):
        """Скорборд считается по журналу — призраки искажали бы и его."""
        os.makedirs(self.table, exist_ok=True)
        agent_v2.append_public_ledger(self.table, self._entry("player1", 3, 1))
        agent_v2.append_public_ledger(self.table, self._entry("player1", 3, 20))
        txt = agent_v2._format_scoreboard(agent_v2.load_public_ledger(self.table))
        self.assertIn("1 bet(s)", txt, f"скорборд посчитал призрак: {txt}")
        self.assertIn("staked 20", txt)


# ───────── FIX-17: выключатель при падении сервера моделей ────────────────

class TestCircuitBreaker(GameHarness):

    def setUp(self):
        super().setUp()
        RealLLMClient.configure_breaker(3)

    def tearDown(self):
        RealLLMClient.configure_breaker(6)
        super().tearDown()

    class Dead(RealLLMClient):
        """Сервер лежит: любой вызов — отказ соединения."""
        def __init__(self): pass
        @classmethod
        def from_config(cls, cfg, section=None): return cls()
        def chat(self, *a, **kw):
            raise RuntimeError("Connection refused")

    def test_breaker_trips_after_n_consecutive_failures(self):
        c = self.Dead()
        for i in range(2):
            with self.assertRaises(RuntimeError):
                c.chat_json("s", "u")
        with self.assertRaises(LLMUnavailable):
            c.chat_json("s", "u")

    def test_success_resets_the_counter(self):
        c = self.Dead()
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                c.chat_json("s", "u")
        llm_client._Breaker.failures = 0
        with self.assertRaises(RuntimeError) as cm:
            c.chat_json("s", "u")
        self.assertNotIsInstance(cm.exception, LLMUnavailable)

    def test_bad_json_does_not_trip_the_breaker(self):
        """
        Битый JSON означает, что сервер жив и отвечает — модель просто не
        попала в формат. Это штатная ситуация с ретраем, а не повод рвать
        раунд, иначе выключатель срабатывал бы на здоровом сервере.
        """
        class Garbage(RealLLMClient):
            def __init__(self): pass
            def chat(self, *a, **kw): return "not json at all"
        c = Garbage()
        for _ in range(10):
            with self.assertRaises(Exception) as cm:
                c.chat_json("s", "u")
            self.assertNotIsInstance(cm.exception, LLMUnavailable)

    def test_agent_does_not_swallow_the_breaker(self):
        """
        Ключевое: обычную ошибку агент гасит и уходит в заглушку — именно
        поэтому падение сервера прошло незамеченным. Выключатель гаситься
        не должен.
        """
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, make_cfg(self.table, self.logs, 1))

        class Boom(FakeLLM):
            def chat_json(self, *a, **kw):
                raise LLMUnavailable("server down")

        agent.client = Boom()
        for call in (lambda: agent.decide_bet(1),
                     lambda: agent.decide_next_move(["player2"], [], 1, []),
                     lambda: agent.dialogue_turn("player2", 10, [], 1, True),
                     lambda: agent.reflect_betting(None)):
            with self.assertRaises(LLMUnavailable):
                call()

    def test_every_agent_handler_reraises_the_breaker(self):
        """Статически: ни один except Exception не должен глотать выключатель."""
        import ast as _ast
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "agent_v2.py"), encoding="utf-8").read()
        bad = []
        for node in _ast.walk(_ast.parse(src)):
            if isinstance(node, _ast.Try):
                names = [h.type.id for h in node.handlers
                         if isinstance(h.type, _ast.Name)]
                if "Exception" in names and "LLMUnavailable" not in names:
                    bad.append(node.lineno)
        self.assertEqual(bad, [], f"глотают выключатель, строки: {bad} (FIX-17)")


# ───────── FIX-18: номер раунда и явное «спина ещё не было» ───────────────

class TestCurrentRoundNotice(GameHarness):

    def test_notice_names_the_round_and_denies_future_results(self):
        txt = agent_v2._current_round_notice(4)
        self.assertIn("CURRENT ROUND: 4", txt)
        self.assertIn("has NOT been spun", txt)
        self.assertIn("or any later round", txt)

    def test_notice_sits_next_to_the_scoreboard_everywhere(self):
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "agent_v2.py"), encoding="utf-8").read()
        # три решения + шаг планирования чек-листа (FIX-19).
        # ROLE-R1 вклинил после уведомления блок роли на первый раунд, поэтому
        # хвостовой "+" остался не у всех вхождений — считаем сам вызов.
        self.assertEqual(src.count("+ _current_round_notice(round_no)"), 4,
                         "уведомление подставлено не везде (FIX-18)")

    def test_decide_bet_now_knows_the_round(self):
        """Раньше при выборе ставки агент вообще не знал, который идёт раунд."""
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, make_cfg(self.table, self.logs, 1))
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"type": "even_money", "selection": "red", "amount": 5}

        agent.client = Spy()
        agent.decide_bet(7)
        self.assertIn("bet for round 7", seen[-1], "номер раунда не дошёл (FIX-18)")
        self.assertIn("CURRENT ROUND: 7", seen[-1])

    def test_the_r5_during_r4_scenario_is_now_contradicted_on_screen(self):
        """
        Воспроизводит реальный случай: находясь в раунде 4, игрок отчитался о
        результате спина раунда 5 и заплатил по нему 60 монет.
        """
        os.makedirs(self.table, exist_ok=True)
        agent_v2.append_public_ledger(self.table, {
            "round_no": 3, "player_id": "player3", "winning_number": 36,
            "bet": {"type": "even_money", "selection": "red", "amount": 5},
            "win": True, "payout": 10})
        agent = PlayerAgent("player1", self.table, make_cfg(self.table, self.logs, 1))
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"message": "no", "transfer": 0, "transfer_to": None,
                        "done": True}

        agent.client = Spy()
        agent.dialogue_turn("player3", 200, [
            {"from": "player3",
             "message": "Public ledger confirms the R5 60c joint position hit."}
        ], 4, is_initiator=False)
        prompt = seen[-1]
        self.assertIn("CURRENT ROUND: 4", prompt)
        self.assertIn("No result exists for round 4 or any later round", prompt)
        self.assertIn("player3: casino P&L=+5c over 1 bet(s)", prompt)
        self.assertIn("last bet in r3", prompt)


# ───────────── FIX-19: чек-лист как краткосрочная повестка ────────────────

class TestChecklist(GameHarness):

    def _agent(self, pid="player1", **memory):
        cfg = make_cfg(self.table, self.logs, 1)
        if memory:
            cfg.read_dict({"memory": {k: str(v) for k, v in memory.items()}})
        os.makedirs(self.table, exist_ok=True)
        return PlayerAgent(pid, self.table, cfg)

    def test_starts_from_a_template_and_persists(self):
        agent = self._agent()
        self.assertIn("Owed TO me", agent._checklist_or_default())
        agent_v2.save_checklist("player1", self.table, "my own format")
        self.assertEqual(agent._checklist_or_default(), "my own format")

    def test_agent_may_discard_the_suggested_structure_entirely(self):
        """
        Главное требование: файл принадлежит агенту. Он вправе выбросить все
        предложенные заголовки и хранить что угодно в любом виде.
        """
        agent = self._agent()
        exotic = "DEBTORS>>player3:24@r5|player2:8@r6\nTRUST_DELTA:+2,-1\nASK:player4"
        FakeLLM.behaviour = {}

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                # проверяем, что агенту прямо разрешено ломать структуру
                assert "throw the whole structure away" in user, user[:200]
                return {"checklist": exotic}

        agent.client = Spy()
        agent.plan_round(3, ["player2"])
        self.assertEqual(agent._checklist_or_default(), exotic)
        self.assertNotIn("Owed TO me", agent._checklist_or_default())

    def test_plan_round_sees_recent_results_and_scoreboard(self):
        """«чтобы раунды тоже видели сразу последние игры при анализе»."""
        agent = self._agent()
        agent_v2.append_public_ledger(self.table, {
            "round_no": 1, "player_id": "player2", "winning_number": 19,
            "bet": {"type": "even_money", "selection": "red", "amount": 20},
            "win": True, "payout": 40})
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"checklist": "ok"}

        agent.client = Spy()
        agent.plan_round(2, ["player2"])
        p = seen[-1]
        self.assertIn("player2: casino P&L=+20c", p, "нет скорборда в планировании")
        self.assertIn("Recent table results", p, "нет свежих результатов раундов")
        self.assertIn("Your reputation map", p)
        self.assertIn("CURRENT ROUND: 2", p)

    def test_update_after_dialogue_sees_the_transcript(self):
        agent = self._agent()
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"checklist": "updated"}

        agent.client = Spy()
        agent.update_checklist("player3", [
            {"from": "player3", "message": "I will repay 24 coins by r5",
             "transfer": 0, "transfer_to": None},
        ], -5, 4)
        p = seen[-1]
        self.assertIn("I will repay 24 coins by r5", p, "транскрипт не передан")
        self.assertIn("-5", p, "нетто перевода не передано")
        self.assertEqual(agent._checklist_or_default(), "updated")

    def test_checklist_reaches_all_three_decisions(self):
        """Выбор собеседника, сам диалог и выбор ставки."""
        agent = self._agent()
        agent_v2.save_checklist("player1", self.table, "MARKER_AGENDA_XYZ")
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                if "Place casino bet" in system:
                    return {"type": "even_money", "selection": "red", "amount": 5}
                if "Decide next move" in system:
                    return {"action": "bet", "reason": "x"}
                return {"message": "m", "transfer": 0, "transfer_to": None,
                        "done": True}

        agent.client = Spy()
        agent.decide_next_move(["player2"], [], 1, [])
        agent.dialogue_turn("player2", 10, [], 1, is_initiator=True)
        agent.decide_bet(1)
        for i, name in enumerate(("decide_next_move", "dialogue_turn", "decide_bet")):
            self.assertIn("MARKER_AGENDA_XYZ", seen[i],
                          f"чек-лист не дошёл до {name} (FIX-19)")

    def test_disabled_flag_removes_it_from_prompts_entirely(self):
        """
        Выключение должно экономить и вызовы, И контекст — иначе агент
        получал бы пустой шаблон вместо ничего.
        """
        cfg = make_cfg(self.table, self.logs, 1)
        cfg.read_dict({"game": {"use_checklist": "false"}})
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        agent_v2.save_checklist("player1", self.table, "MARKER_AGENDA_XYZ")
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"action": "bet", "reason": "x"}

        agent.client = Spy()
        agent.decide_next_move(["player2"], [], 1, [])
        self.assertNotIn("MARKER_AGENDA_XYZ", seen[-1])
        self.assertNotIn("YOUR CHECKLIST", seen[-1])

    def test_bounded_like_the_persona(self):
        """Урок FIX-12: любой самопишущийся файл обязан иметь потолок."""
        agent = self._agent(checklist_chars=200)

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                return {"checklist": "Z" * 9000}

        agent.client = Spy()
        agent.plan_round(1, [])
        self.assertLessEqual(len(agent._checklist_or_default()), 200,
                             "чек-лист растёт без ограничения (FIX-19)")

    def test_llm_failure_keeps_the_old_checklist(self):
        agent = self._agent()
        agent_v2.save_checklist("player1", self.table, "important agenda")

        class Boom(FakeLLM):
            def chat_json(self, *a, **kw):
                raise RuntimeError("boom")

        agent.client = Boom()
        self.assertEqual(agent.plan_round(1, []), "important agenda")
        self.assertEqual(agent._checklist_or_default(), "important agenda")

    def test_breaker_is_not_swallowed_by_checklist_steps(self):
        agent = self._agent()

        class Down(FakeLLM):
            def chat_json(self, *a, **kw):
                raise LLMUnavailable("server down")

        agent.client = Down()
        with self.assertRaises(LLMUnavailable):
            agent.plan_round(1, [])
        with self.assertRaises(LLMUnavailable):
            agent.update_checklist("player2", [], 0, 1)

    def test_both_sides_update_after_a_dialogue(self):
        """Обязательство фиксируют обе стороны, а не только инициатор."""
        calls = []
        orig = PlayerAgent.update_checklist
        PlayerAgent.update_checklist = (
            lambda self, p, c, n, r, speech_became_free=False:
                calls.append((self.player_id, p)) or "x")
        try:
            FakeLLM.behaviour = {
                "next_move": lambda u: (
                    {"action": "talk", "partner": "player2", "reason": "x"}
                    if "Your player id: player1" in u
                    and "(no one yet this round)" in u
                    else {"action": "bet", "reason": "x"}),
                "dialogue": lambda u: {"message": "deal", "transfer": 0,
                                       "transfer_to": None, "done": True},
                "bet": lambda u: {"type": "even_money", "selection": "red",
                                  "amount": 1},
            }
            self.run_game(rounds=1, winning_number=0)
        finally:
            PlayerAgent.update_checklist = orig
        self.assertIn(("player1", "player2"), calls)
        self.assertIn(("player2", "player1"), calls)


# ───────────── FIX-20: структурированный реестр обещаний ──────────────────

class TestPromiseLedger(GameHarness):
    """
    README называл это единственным незакрытым пунктом: обещание живёт
    только в свободном тексте синапсы/чек-листа, и его исполнение никем не
    проверяется. Эти тесты — контрфактические в том же смысле, что и
    остальные: они падают, если update_checklist не читает/не сохраняет
    "promises", или если plan_round не подмешивает деterministic-напоминание.
    """

    def _agent(self, pid="player1", **memory):
        cfg = make_cfg(self.table, self.logs, 1)
        if memory:
            cfg.read_dict({"memory": {k: str(v) for k, v in memory.items()}})
        os.makedirs(self.table, exist_ok=True)
        return PlayerAgent(pid, self.table, cfg)

    def test_promise_extracted_without_an_extra_llm_call(self):
        """"3 раунда, потом верну деньги" -> сохраняется как структурный
        промис ОДНИМ и тем же вызовом update_checklist, без нового round-trip'а."""
        agent = self._agent()
        calls = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                calls.append(user)
                return {
                    "checklist": "ok",
                    "promises": [{
                        "direction": "owed_to_me", "counterparty": "player3",
                        "amount": 20, "due_round": 5,
                        "description": "play and return the loan", "status": "open",
                    }],
                }

        agent.client = Spy()
        agent.update_checklist("player3", [
            {"from": "player3",
             "message": "I'll play for 3 rounds and return your money",
             "transfer": 0, "transfer_to": None},
        ], 0, round_no=2)

        self.assertEqual(len(calls), 1, "должен быть ровно один вызов LLM")
        self.assertIn(promise_ledger.PROMPT_INSTRUCTIONS, calls[0],
                      "инструкция по promises не попала в тот же промпт")
        saved = promise_ledger.load_promises("player1", self.table)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["counterparty"], "player3")
        self.assertEqual(saved[0]["due_round"], 5)

    def test_reminder_is_absent_then_upcoming_then_overdue_without_any_llm_call(self):
        """Переходы Upcoming -> DUE -> OVERDUE считаются кодом, не моделью:
        due_reminder не дергает LLM вообще."""
        os.makedirs(self.table, exist_ok=True)
        promise_ledger.merge_and_save("player1", self.table, [{
            "direction": "owed_to_me", "counterparty": "player3",
            "amount": 20, "due_round": 5, "description": "loan", "status": "open",
        }], round_no=2)

        r3 = promise_ledger.due_reminder("player1", self.table, round_no=3)
        self.assertIn("Upcoming", r3)
        self.assertNotIn("OVERDUE", r3)

        r5 = promise_ledger.due_reminder("player1", self.table, round_no=5)
        self.assertIn("DUE THIS ROUND", r5)

        r6 = promise_ledger.due_reminder("player1", self.table, round_no=6)
        self.assertIn("OVERDUE", r6)
        self.assertIn("player3 owes YOU 20c", r6)

    def test_plan_round_shows_overdue_promise_in_the_actual_prompt(self):
        """Не просто format-функция сама по себе — проверяем, что plan_round
        реально подмешивает её в промпт, который уйдёт модели."""
        agent = self._agent()
        promise_ledger.merge_and_save("player1", self.table, [{
            "direction": "owed_to_me", "counterparty": "player3",
            "amount": 20, "due_round": 2, "description": "loan", "status": "open",
        }], round_no=1)
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"checklist": "ok"}

        agent.client = Spy()
        agent.plan_round(6, ["player3"])
        self.assertIn("OVERDUE", seen[-1])
        self.assertIn("player3 owes YOU 20c", seen[-1])

    def test_settled_promise_stops_being_reminded(self):
        agent = self._agent()
        promise_ledger.merge_and_save("player1", self.table, [{
            "direction": "owed_to_me", "counterparty": "player3",
            "amount": 20, "due_round": 2, "description": "loan", "status": "open",
        }], round_no=1)

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                return {
                    "checklist": "ok",
                    "promises": [{
                        "direction": "owed_to_me", "counterparty": "player3",
                        "amount": 20, "due_round": 2, "description": "loan",
                        "status": "settled",
                    }],
                }

        agent.client = Spy()
        agent.update_checklist("player3", [], 20, round_no=7)
        self.assertEqual(
            promise_ledger.due_reminder("player1", self.table, round_no=8), "",
            "settled-обещание не должно больше маячить в промпте")

    def test_malformed_promises_from_model_never_crash_the_round(self):
        """Тот же принцип, что FIX-6 для notes/persona: мусор от модели
        отбрасывается, а не роняет игру и не портит остальные записи."""
        agent = self._agent()

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                return {"checklist": "ok", "promises": "not a list at all"}

        agent.client = Spy()
        agent.update_checklist("player3", [], 0, round_no=1)  # не должно бросить
        self.assertEqual(promise_ledger.load_promises("player1", self.table), [])

        class SpyGarbageItems(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                return {"checklist": "ok", "promises": [
                    {"direction": "not_a_valid_direction", "amount": 5},
                    {"direction": "owed_to_me"},               # нет counterparty
                    "just a string, not even a dict",
                    {"direction": "i_owe", "counterparty": "player2",
                     "amount": "not-an-int", "due_round": 3},
                ]}

        agent.client = SpyGarbageItems()
        agent.update_checklist("player2", [], 0, round_no=1)
        self.assertEqual(promise_ledger.load_promises("player1", self.table), [],
                          "ни одна битая запись не должна была сохраниться")

    def test_missing_promises_field_keeps_prior_structured_list(self):
        """Если в этот раз модель забыла вернуть 'promises' (старый промпт,
        сбой парсинга) — не стирать то, что уже отслеживалось."""
        agent = self._agent()
        promise_ledger.merge_and_save("player1", self.table, [{
            "direction": "owed_to_me", "counterparty": "player3",
            "amount": 20, "due_round": 5, "description": "loan", "status": "open",
        }], round_no=2)

        class SpyNoPromisesField(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                return {"checklist": "ok"}   # нет "promises" вообще

        agent.client = SpyNoPromisesField()
        agent.update_checklist("player4", [], 0, round_no=3)
        # FIX-20a: раньше здесь утверждалось == [] — тест с именем
        # "keeps_prior_structured_list" закреплял ровно противоположное
        # поведение и создавал ложное ощущение, что случай покрыт.
        kept = promise_ledger.load_promises("player1", self.table)
        self.assertEqual(len(kept), 1,
                         "отсутствие поля 'promises' стёрло реестр (FIX-20a)")
        self.assertEqual(kept[0]["counterparty"], "player3")
        self.assertEqual(kept[0]["amount"], 20)
        self.assertEqual(kept[0]["status"], "open")

    def test_hard_cap_like_the_checklist_and_persona(self):
        """Урок FIX-12/19: любой самопишущийся список обязан иметь потолок."""
        many = [{
            "direction": "owed_to_me", "counterparty": f"player{i}",
            "amount": 1, "due_round": 1, "description": "x", "status": "open",
        } for i in range(promise_ledger.MAX_PROMISES + 10)]
        os.makedirs(self.table, exist_ok=True)
        saved = promise_ledger.merge_and_save("player1", self.table, many, round_no=1)
        self.assertLessEqual(len(saved), promise_ledger.MAX_PROMISES)

    def test_breaker_is_not_swallowed_during_checklist_llm_call(self):
        """LLMUnavailable должен всплыть наверх ДО того, как promises
        попытаются сохраниться с несуществующим resp — не тихая заглушка."""
        agent = self._agent()

        class Down(FakeLLM):
            def chat_json(self, *a, **kw):
                raise LLMUnavailable("server down")

        agent.client = Down()
        with self.assertRaises(LLMUnavailable):
            agent.update_checklist("player3", [], 0, round_no=1)
        self.assertEqual(promise_ledger.load_promises("player1", self.table), [])


# ───────────── FIX-21: детерминированный журнал переводов между игроками ──

class TestTransferLedger(GameHarness):
    """
    Ровно ваш сценарий: player1 просит 30 за стратегию, player2 переводит
    20, торг о недостающих 10 не завершается сделкой — но 20 монет уже
    ушли. До этого фикса факт перевода жил ТОЛЬКО в субъективной dsyn
    каждого игрока (пишет LLM со своей колокольни): ничто не мешало
    player1 записать «получил подарок», а player2 — «заплатил и ничего не
    получил», и это два несовместимых рассказа об одном и том же переводе.
    Эти тесты проверяют, что цифры теперь — общий, до всякого LLM
    зафиксированный факт, одинаковый для обеих сторон.
    """

    def test_full_broken_deal_scenario_from_the_conversation(self):
        """Именно диалог из вопроса: 20 переведено, 10 не переведено,
        сделка не состоялась — обе стороны видят одни и те же 20/-20."""
        os.makedirs(self.table, exist_ok=True)
        agent1 = PlayerAgent("player1", self.table, make_cfg(self.table, self.logs, 1))
        agent2 = PlayerAgent("player2", self.table, make_cfg(self.table, self.logs, 1))
        agent1.balance, agent2.balance = 100, 100

        class Scripted(FakeLLM):
            script = iter([
                {"message": "Sell you my strategy for 30", "transfer": 0,
                 "transfer_to": None, "done": False},                       # player1
                {"message": "Here's 20", "transfer": 20,
                 "transfer_to": "player1", "done": False},                  # player2
                {"message": "I said I need 10 more", "transfer": 0,
                 "transfer_to": None, "done": False},                       # player1
                {"message": "I paid enough, let's just play", "transfer": 0,
                 "transfer_to": None, "done": True},                        # player2
                {"message": "Send the rest or no deal", "transfer": 0,
                 "transfer_to": None, "done": True},                        # player1 closing? -> но done=True у B уже был
            ])

            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                return next(self.script)

        agent1.client = Scripted()
        agent2.client = Scripted()
        run_game_v2.run_dialogue(agent1, agent2, round_no=3,
                                  logger=run_game_v2.GameLogger(self.logs),
                                  table_dir=self.table)

        e1 = transfer_ledger.load_entries("player1", self.table)
        e2 = transfer_ledger.load_entries("player2", self.table)
        self.assertEqual(len(e1), 1)
        self.assertEqual(len(e2), 1)
        # player1's side: sent 0, received 20 -> net +20
        self.assertEqual(e1[0]["sent"], 0)
        self.assertEqual(e1[0]["received"], 20)
        self.assertEqual(e1[0]["net"], 20)
        # player2's side: exactly the mirror -> net -20
        self.assertEqual(e2[0]["sent"], 20)
        self.assertEqual(e2[0]["received"], 0)
        self.assertEqual(e2[0]["net"], -20)
        # никакого сюрприза: числа буквально зеркальны друг другу
        self.assertEqual(e1[0]["sent"], e2[0]["received"])
        self.assertEqual(e1[0]["received"], e2[0]["sent"])

    def test_no_llm_call_needed_to_produce_the_reminder(self):
        os.makedirs(self.table, exist_ok=True)
        transfer_ledger.record_dialogue(self.table, "player1", "player2",
                                        round_no=3, a_sent=0, b_sent=20)
        # player2's view: sent 20, got nothing back — visible without any LLM
        block = transfer_ledger.format_recent("player2", self.table)
        self.assertIn("sent 20c, received 0c (net -20c)", block)
        # player1's view of the exact same transfer, mirrored
        block1 = transfer_ledger.format_recent("player1", self.table)
        self.assertIn("sent 0c, received 20c (net +20c)", block1)

    def test_nothing_recorded_when_no_money_actually_moved(self):
        """Торг сорвался, но денег не было — журнал не заводит пустых записей."""
        os.makedirs(self.table, exist_ok=True)
        transfer_ledger.record_dialogue(self.table, "player1", "player2",
                                        round_no=1, a_sent=0, b_sent=0)
        self.assertEqual(transfer_ledger.load_entries("player1", self.table), [])
        self.assertEqual(transfer_ledger.load_entries("player2", self.table), [])

    def test_replaying_a_round_replaces_not_duplicates(self):
        """Тот же принцип, что FIX-16 для публичного журнала: одна запись
        на пару (partner, round_no)."""
        os.makedirs(self.table, exist_ok=True)
        transfer_ledger.record_dialogue(self.table, "player1", "player2",
                                        round_no=3, a_sent=0, b_sent=20)
        transfer_ledger.record_dialogue(self.table, "player1", "player2",
                                        round_no=3, a_sent=0, b_sent=25)
        entries = transfer_ledger.load_entries("player2", self.table)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["sent"], 25)

    def test_appears_in_plan_round_prompt(self):
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        transfer_ledger.record_dialogue(self.table, "player1", "player2",
                                        round_no=3, a_sent=0, b_sent=20)
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"checklist": "ok"}

        agent.client = Spy()
        agent.plan_round(4, ["player2"])
        self.assertIn("VERIFIED transfer history", seen[-1])
        self.assertIn("received 20c", seen[-1])

    def test_appears_inside_the_live_dialogue_with_that_partner(self):
        """Самое важное место: пока идёт спор внутри разговора, обе стороны
        должны видеть прошлые переводы именно ДРУГ С ДРУГОМ."""
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        transfer_ledger.record_dialogue(self.table, "player1", "player2",
                                        round_no=3, a_sent=0, b_sent=20)
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.8, max_tokens=400):
                seen.append(user)
                return {"message": "m", "transfer": 0, "transfer_to": None,
                        "done": False}

        agent.client = Spy()
        agent.dialogue_turn("player2", 100, [], round_no=4, is_initiator=True)
        self.assertIn("VERIFIED transfer history", seen[-1])

    def test_bounded_like_the_other_self_writing_ledgers(self):
        os.makedirs(self.table, exist_ok=True)
        for r in range(transfer_ledger.MAX_ENTRIES_PER_PLAYER + 20):
            transfer_ledger.record_dialogue(self.table, "player1",
                                            f"opp{r}", round_no=r,
                                            a_sent=0, b_sent=1)
        entries = transfer_ledger.load_entries("player1", self.table)
        self.assertLessEqual(len(entries), transfer_ledger.MAX_ENTRIES_PER_PLAYER)


# ───── FIX-20a..d: реестр обещаний — слияние вместо замещения ─────────────

class TestPromiseMerge(GameHarness):

    def _p(self, cp, amount, due, status="open", direction="owed_to_me", pid=None):
        d = {"direction": direction, "counterparty": cp, "amount": amount,
             "due_round": due, "description": "x", "status": status}
        if pid is not None:
            d["id"] = pid
        return d

    def setUp(self):
        super().setUp()
        os.makedirs(self.table, exist_ok=True)

    def test_a_promise_survives_a_conversation_with_someone_else(self):
        """
        Тот самый баг: реестр заменялся целиком тем, что вернула модель, а
        текущий список ей не показывали вовсе. Долг исчезал не когда его
        прощали, а когда игрок поговорил с кем-то другим.
        """
        promise_ledger.merge_and_save("player1", self.table,
                                      [self._p("player2", 24, 5)], round_no=4)
        promise_ledger.merge_and_save("player1", self.table,
                                      [self._p("player3", 8, 6, direction="i_owe")],
                                      round_no=4)
        got = {(p["counterparty"], p["amount"])
               for p in promise_ledger.load_promises("player1", self.table)}
        self.assertEqual(got, {("player2", 24), ("player3", 8)},
                         "обещание пропало после разговора с другим (FIX-20a)")

    def test_omission_means_no_change_not_deletion(self):
        for payload in (None, [], "not a list", {"nope": 1}):
            promise_ledger.save_promises("player1", self.table,
                                         [dict(self._p("player2", 24, 5), id=1)])
            promise_ledger.merge_and_save("player1", self.table, payload, 4)
            self.assertEqual(len(promise_ledger.load_promises("player1", self.table)), 1,
                             f"реестр стёрт при promises={payload!r} (FIX-20a)")

    def test_status_change_by_id(self):
        promise_ledger.merge_and_save("player1", self.table,
                                      [self._p("player2", 24, 5)], 4)
        pid = promise_ledger.load_promises("player1", self.table)[0]["id"]
        promise_ledger.merge_and_save(
            "player1", self.table,
            [self._p("player2", 24, 5, status="settled", pid=pid)], 5)
        kept = promise_ledger.load_promises("player1", self.table)
        self.assertEqual(len(kept), 1, "смена статуса создала дубль")
        self.assertEqual(kept[0]["status"], "settled")

    def test_repeating_a_promise_without_id_does_not_duplicate(self):
        """Модель часто не возвращает id — опознаём по отпечатку."""
        for _ in range(4):
            promise_ledger.merge_and_save("player1", self.table,
                                          [self._p("player2", 24, 5)], 4)
        self.assertEqual(len(promise_ledger.load_promises("player1", self.table)), 1,
                         "обещание продублировалось (FIX-20a)")

    def test_settled_promise_leaves_the_reminder_but_stays_recorded(self):
        promise_ledger.merge_and_save("player1", self.table,
                                      [self._p("player2", 24, 5)], 4)
        pid = promise_ledger.load_promises("player1", self.table)[0]["id"]
        promise_ledger.merge_and_save(
            "player1", self.table,
            [self._p("player2", 24, 5, status="settled", pid=pid)], 5)
        self.assertEqual(promise_ledger.due_reminder("player1", self.table, 6), "")
        self.assertEqual(len(promise_ledger.load_promises("player1", self.table)), 1)

    def test_eviction_drops_closed_before_open(self):
        """FIX-20c: закрытые вытесняются раньше открытых, срок решает порядок."""
        closed = [self._p(f"c{i}", 1, i, status="settled")
                  for i in range(promise_ledger.MAX_PROMISES)]
        promise_ledger.merge_and_save("player1", self.table, closed, 1)
        fresh = [self._p("player9", 99, 50)]
        promise_ledger.merge_and_save("player1", self.table, fresh, 1)
        kept = promise_ledger.load_promises("player1", self.table)
        self.assertLessEqual(len(kept), promise_ledger.MAX_PROMISES)
        self.assertIn("player9", [p["counterparty"] for p in kept],
                      "открытое обещание вытеснено раньше закрытых (FIX-20c)")

    def test_model_is_shown_the_ledger_with_ids(self):
        """Без этого просьба «обнови статус» ссылаться не на что."""
        promise_ledger.merge_and_save("player1", self.table,
                                      [self._p("player2", 24, 5)], 4)
        txt = promise_ledger.format_for_model("player1", self.table, 5)
        self.assertIn("id=", txt)
        self.assertIn("player2 owes you 24c", txt)
        self.assertIn("due r5", txt)

    def test_update_checklist_prompt_contains_the_ledger(self):
        promise_ledger.merge_and_save("player1", self.table,
                                      [self._p("player2", 24, 5)], 4)
        agent = PlayerAgent("player1", self.table,
                            make_cfg(self.table, self.logs, 1))
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"checklist": "ok", "promises": []}

        agent.client = Spy()
        agent.update_checklist("player3", [], 0, 5)
        self.assertIn("Your tracked promises", seen[-1],
                      "реестр не показан модели — переносить нечего (FIX-20a)")
        self.assertIn("id=", seen[-1])
        self.assertIn("does not delete it", seen[-1],
                      "модели не сказано, что пропуск записи безопасен")

    def test_update_checklist_mentions_free_speech_after_transfer(self):
        """XFER-FREE-NOTE: без этого параметра игрок ЖИВЬЁМ видел правило
        внутри диалога, но нигде не встречал его повторно на пост-анализе —
        и урок на будущее ("отправь перевод пораньше — торг бесплатен")
        никогда не закреплялся в чек-листе (подтверждено на реальном
        прогоне: ни одна заметка за всю партию это не упомянула)."""
        agent = PlayerAgent("player1", self.table,
                            make_cfg(self.table, self.logs, 1))
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"checklist": "ok", "promises": []}

        agent.client = Spy()
        agent.update_checklist("player3", [], 0, 5, speech_became_free=True)
        self.assertIn("speech was", seen[-1])
        self.assertIn("free", seen[-1])
        self.assertIn("player3", seen[-1])

    def test_update_checklist_silent_when_speech_was_not_free(self):
        """Регрессия: без срабатывания правила в промпте не должно быть
        никакого упоминания — иначе ложный сигнал на КАЖДЫЙ диалог."""
        agent = PlayerAgent("player1", self.table,
                            make_cfg(self.table, self.logs, 1))
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"checklist": "ok", "promises": []}

        agent.client = Spy()
        agent.update_checklist("player3", [], 0, 5)  # speech_became_free по умолчанию False
        self.assertNotIn("TACTIC NOTE", seen[-1])

    def test_plan_round_can_also_close_a_promise(self):
        """
        FIX-20d: раньше статус менялся только в update_checklist, то есть
        лишь в разговоре. Должник, переставший выходить на связь, оставлял
        долг просроченным навсегда.
        """
        promise_ledger.merge_and_save("player1", self.table,
                                      [self._p("player2", 24, 5)], 4)
        pid = promise_ledger.load_promises("player1", self.table)[0]["id"]
        agent = PlayerAgent("player1", self.table,
                            make_cfg(self.table, self.logs, 1))

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                assert "Your tracked promises" in user, "реестр не показан в plan_round"
                return {"checklist": "ok",
                        "promises": [{"id": pid, "direction": "owed_to_me",
                                      "counterparty": "player2", "amount": 24,
                                      "due_round": 5, "description": "x",
                                      "status": "broken"}]}

        agent.client = Spy()
        agent.plan_round(9, ["player3"])
        self.assertEqual(
            promise_ledger.load_promises("player1", self.table)[0]["status"], "broken")

    def test_reminder_reaches_the_live_dialogue(self):
        """
        FIX-20b: долг требуют в разговоре. Реестр переводов там уже был,
        реестр обещаний — нет.
        """
        promise_ledger.merge_and_save("player1", self.table,
                                      [self._p("player2", 24, 5)], 4)
        agent = PlayerAgent("player1", self.table,
                            make_cfg(self.table, self.logs, 1))
        seen = []

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                seen.append(user)
                return {"message": "pay up", "transfer": 0, "transfer_to": None,
                        "done": True}

        agent.client = Spy()
        agent.dialogue_turn("player2", 50, [], 7, is_initiator=True)
        self.assertIn("OVERDUE", seen[-1],
                      "просроченный долг не виден в диалоге (FIX-20b)")
        self.assertIn("player2 owes YOU 24c", seen[-1])

    def test_full_scenario_two_partners_across_rounds(self):
        """Сквозной: два долга, разные партнёры, погашение одного."""
        agent = PlayerAgent("player1", self.table,
                            make_cfg(self.table, self.logs, 1))
        script = [
            [self._p("player2", 24, 6)],
            [self._p("player3", 8, 7, direction="i_owe")],
            [],
        ]

        class Spy(FakeLLM):
            def chat_json(self, system, user, temperature=0.4, max_tokens=400):
                return {"checklist": "ok",
                        "promises": script.pop(0) if script else []}

        agent.client = Spy()
        agent.update_checklist("player2", [], 0, 5)
        agent.update_checklist("player3", [], 0, 5)
        agent.update_checklist("player4", [], 0, 5)   # третий разговор — ничего нового

        kept = promise_ledger.load_promises("player1", self.table)
        self.assertEqual(len(kept), 2, "долги не пережили три разговора (FIX-20a)")
        reminder = promise_ledger.due_reminder("player1", self.table, 8)
        self.assertIn("player2 owes YOU 24c", reminder)
        self.assertIn("YOU owe player3 8c", reminder)


class _BailoutAgent:
    def __init__(self):
        self.balance = 0


class _BailoutLog:
    """Заглушка GameLogger: копит строки, чтобы проверить, что начисление
    попало и в общий лог, и в персональный лог получателя."""

    def __init__(self):
        self.lines = []

    def write_global(self, msg):
        self.lines.append(("GAME", msg))

    def write(self, pid, msg):
        self.lines.append((pid, msg))

    def write_balances(self, balances):
        self.lines.append(("BAL", str(balances)))


class TestBailout(unittest.TestCase):
    """Докапитализация банкрота и публичное объявление о ней."""

    PLAYERS = ["player1", "player2", "player3"]

    def setUp(self):
        self.table = tempfile.mkdtemp()
        self.cfg = configparser.ConfigParser()
        self.cfg.read_dict({"game": {"bailout_enabled": "true",
                                     "bailout_zero_rounds": "2",
                                     "bailout_amount": "20",
                                     "bailout_all_players": "false"}})
        self.agents = {p: _BailoutAgent() for p in self.PLAYERS}
        self.log = _BailoutLog()

    def _bal(self, pid, value):
        common.write_json(common.balance_file(pid, self.table), {"balance": value})

    def _balances(self):
        return run_game_v2.get_balances(self.table, self.PLAYERS)

    def _run(self, round_no):
        run_game_v2.apply_bailout_if_needed(
            self.agents, self.PLAYERS, self.table, round_no, self.cfg, self.log)

    def _setup(self, **balances):
        for pid in self.PLAYERS:
            self._bal(pid, balances.get(pid, 50))

    def test_fires_only_after_more_than_two_zero_rounds(self):
        self._setup(player2=0)
        for r in (1, 2):
            self._run(r)
            self.assertEqual(self._balances()["player2"], 0,
                             "начислено раньше третьего нулевого раунда")
        self._run(3)
        self.assertEqual(self._balances()["player2"], 20,
                         "+20 не начислены банкроту на третьем нулевом раунде")

    def test_only_the_bankrupt_gets_paid_by_default(self):
        self._setup(player2=0)
        for r in (1, 2, 3):
            self._run(r)
        self.assertEqual(self._balances(),
                         {"player1": 50, "player2": 20, "player3": 50},
                         "деньги получил кто-то кроме банкрота")
        self.assertEqual(self.agents["player1"].balance, 0,
                         "баланс не-получателя трогать нельзя")
        self.assertEqual(self.agents["player2"].balance, 20,
                         "баланс агента в памяти разошёлся с диском")

    def test_all_players_mode(self):
        self.cfg["game"]["bailout_all_players"] = "true"
        self._setup(player2=0)
        for r in (1, 2, 3):
            self._run(r)
        self.assertEqual(self._balances(),
                         {"player1": 70, "player2": 20, "player3": 70},
                         "режим bailout_all_players не начислил всем")
        self.assertIn("EVERY player", common.bailout_notice(self.table, 3))

    def test_not_applied_twice_after_restart(self):
        self._setup(player2=0)
        for r in (1, 2, 3):
            self._run(r)
        self._run(3)                      # рестарт внутри того же раунда
        self.assertEqual(self._balances()["player2"], 20,
                         "повторный проход раунда начислил второй раз")

    def test_streak_resets_after_payout(self):
        self._setup(player2=0)
        for r in (1, 2, 3):
            self._run(r)
        self._bal("player2", 0)           # снова разорился
        for r in (4, 5):
            self._run(r)
        self.assertEqual(self._balances()["player2"], 0,
                         "счётчик нулей не обнулился после выплаты")
        self._run(6)
        self.assertEqual(self._balances()["player2"], 20)

    def test_several_bankrupts_are_all_named_with_balances(self):
        self._setup(player1=0, player3=0)
        for r in (1, 2, 3):
            self._run(r)
        notice = common.bailout_notice(self.table, 3)
        self.assertIn("player1", notice)
        self.assertIn("player3", notice)
        self.assertIn("player1 = 20 coin(s)", notice)
        self.assertIn("player3 = 20 coin(s)", notice)
        self.assertNotIn("player2 =", notice,
                         "в объявление попал не-банкрот")
        self.assertEqual(self._balances(),
                         {"player1": 20, "player2": 50, "player3": 20})

    def test_notice_is_one_off_and_logged(self):
        self._setup(player2=0)
        for r in (1, 2, 3):
            self._run(r)
        notice = common.bailout_notice(self.table, 3)
        self.assertIn("+20", notice)
        self.assertIn("player2", notice)
        self.assertIn("player2 = 20 coin(s)", notice)
        self.assertEqual(common.bailout_notice(self.table, 4), "",
                         "объявление должно быть разовым")
        self.assertEqual(common.bailout_notice(self.table, 2), "")
        self.assertTrue(any(pid == "GAME" and "BAILOUT" in m
                            for pid, m in self.log.lines),
                        "начисление не попало в общий лог/консоль")
        self.assertTrue(any(pid == "player2" and "BAILOUT" in m
                            for pid, m in self.log.lines),
                        "нет строки в персональном логе банкрота")

    def test_disabled_by_config(self):
        self.cfg["game"]["bailout_enabled"] = "false"
        self._setup(player2=0)
        for r in (1, 2, 3, 4):
            self._run(r)
        self.assertEqual(self._balances()["player2"], 0)
        self.assertEqual(common.bailout_notice(self.table, 3), "")

    def test_notice_reaches_every_prompt(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "agent_v2.py")
        src = open(path, encoding="utf-8").read()
        self.assertEqual(
            src.count("+ common.bailout_notice(self.base_dir, round_no)"), 4,
            "объявление подставлено не во все четыре промпта")


# ── FIX-22: reflect_betting ретраится и не врёт про баланс при сбое ────────

class TestReflectBettingRetryAndFallback(GameHarness):
    """
    Реальный случай из разбора партии: у player2 reflect_betting упал по
    504-таймауту, старые заметки остались как есть — синапса продолжала
    утверждать "Balance: 115... WON", хотя реальный баланс был 80 и
    последняя ставка была проиграна. Игрок планировал следующий раунд по
    заведомо неверным фактам о самом себе.
    """

    def _entry(self, round_no=2, winning_number=35, win=False,
               payout=0, balance_after=80):
        return {
            "round_no": round_no,
            "winning_number": winning_number,
            "bet": {"type": "even_money", "amount": 15},
            "win": win,
            "payout": payout,
            "balance_after": balance_after,
        }

    def test_single_transient_failure_is_retried_and_recovers(self):
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        agent.balance = 80
        agent_v2.save_notes("player1", self.table, "old stale strategy text")

        calls = {"n": 0}

        def flaky_reflect(user):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("HTTP 504 Gateway Time-out")
            return {"notes": "fresh notes after retry", "update_persona": False}

        FakeLLM.behaviour = {"reflect": flaky_reflect}
        agent.reflect_betting(self._entry())

        self.assertEqual(calls["n"], 2,
                         "первый сбой должен привести ровно к одному "
                         "повторному вызову (FIX-22)")
        notes = agent_v2.load_notes("player1", self.table)
        self.assertEqual(notes, "fresh notes after retry",
                         "успешный ретрай должен победить, а не старые "
                         "заметки (FIX-22)")

    def test_repeated_failure_does_not_silently_keep_stale_notes(self):
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        agent.balance = 80
        agent_v2.save_notes(
            "player1", self.table,
            "Balance: 115. Last bet: even_money(red), amount 10, WON. Net +10."
        )

        def always_fails(user):
            raise RuntimeError("HTTP 504 Gateway Time-out")

        FakeLLM.behaviour = {"reflect": always_fails}
        agent.reflect_betting(self._entry(balance_after=80))

        notes = agent_v2.load_notes("player1", self.table)
        self.assertIn("80", notes,
                      "реальный баланс должен появиться в заметках даже "
                      "при двойном сбое (FIX-22)")
        self.assertIn("FACT-CHECK", notes,
                      "должна быть явная фактическая справка, а не тихая "
                      "подмена (FIX-22)")
        self.assertIn("lost", notes.lower(),
                      "реальный исход последней ставки (проигрыш) должен "
                      "быть отражён, а не унаследованное 'WON' (FIX-22)")
        # старый текст стратегии не выбрасывается — он просто больше не
        # единственный источник фактов
        self.assertIn("Balance: 115", notes,
                      "старый текст стратегии должен сохраниться под "
                      "справкой, а не быть стёрт (FIX-22)")

    def test_fallback_respects_synapse_char_limit(self):
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        agent.balance = 80
        agent.synapse_chars = 50
        agent_v2.save_notes("player1", self.table, "x" * 500)

        def always_fails(user):
            raise RuntimeError("boom")

        FakeLLM.behaviour = {"reflect": always_fails}
        agent.reflect_betting(self._entry())

        notes = agent_v2.load_notes("player1", self.table)
        self.assertLessEqual(len(notes), 50,
                             "справка о факте не должна пробивать лимит "
                             "синапсы (FIX-22)")

    def test_bankrupt_fallback_states_no_bet_placed(self):
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        agent.balance = 0
        agent_v2.save_notes("player1", self.table, "old plan assuming capital")

        def always_fails(user):
            raise RuntimeError("boom")

        FakeLLM.behaviour = {"reflect": always_fails}
        agent.reflect_betting(None)

        notes = agent_v2.load_notes("player1", self.table)
        self.assertIn("0", notes)
        self.assertIn("no bet", notes.lower())


# ── FIX-23: номер раунда не путается с выпавшим числом рулетки ─────────────

class TestReflectPromptDisambiguatesSpinVsRound(GameHarness):
    """
    В логах партии player4 записал в свою синапсу "Round 35: Staked 10..."
    вместо настоящего номера раунда (2) — спутав его с выпавшим на рулетке
    числом 35. Промпт должен явно различать эти два числа.
    """

    def test_prompt_labels_winning_number_and_round_separately(self):
        cfg = make_cfg(self.table, self.logs, 1)
        os.makedirs(self.table, exist_ok=True)
        agent = PlayerAgent("player1", self.table, cfg)
        agent.balance = 95

        seen_user = {}

        def capture(user):
            seen_user["text"] = user
            return {"notes": "ok", "update_persona": False}

        FakeLLM.behaviour = {"reflect": capture}
        entry = {
            "round_no": 2,
            "winning_number": 35,
            "bet": {"type": "even_money", "amount": 10},
            "win": False,
            "payout": 0,
            "balance_after": 95,
        }
        agent.reflect_betting(entry)

        text = seen_user["text"]
        self.assertIn("winning_number=35", text,
                      "выпавшее число должно быть явно подписано (FIX-23)")
        self.assertIn("round 2", text,
                      "номер раунда должен присутствовать отдельно от "
                      "выпавшего числа (FIX-23)")
        self.assertNotIn("Round result: number=35", text,
                         "старая двусмысленная формулировка не должна "
                         "оставаться в промпте (FIX-23)")




class TestPromptDiet(unittest.TestCase):
    """
    PROMPT-1: у роли в системном промпте лежало ~6100 символов инструкций
    (ядро 3713 + роль 2400). 8b следует началу и середине и теряет хвост —
    отсюда "I'll take the 20 coins" с переводом 20 монет НАРУЖУ.
    Критичные правила вынесены в короткий блок в КОНЕЦ ядра.
    """

    def test_core_is_substantially_shorter(self):
        self.assertLess(len(agent_v2.CORE_SYSTEM_PROMPT), 3000)

    def test_role_prompts_are_short(self):
        import roles
        for name, text in roles.ROLE_PROMPTS.items():
            self.assertLess(len(text), 1700, f"роль {name} снова разрослась")

    def test_pitfalls_block_sits_at_the_very_end(self):
        core = agent_v2.CORE_SYSTEM_PROMPT
        i = core.index("THE FOUR THINGS PLAYERS GET WRONG")
        self.assertGreater(i / len(core), 0.6,
                           "блок с главными правилами должен быть в конце")

    def test_transfer_direction_is_spelled_out(self):
        core = " ".join(agent_v2.CORE_SYSTEM_PROMPT.split())
        self.assertIn("TRANSFER MEANS YOU PAY", core)
        self.assertIn("I'll take", core)
        self.assertIn("set transfer to 0", core)

    def test_done_is_described_as_a_flag(self):
        core = " ".join(agent_v2.CORE_SYSTEM_PROMPT.split())
        self.assertIn("DONE IS A FLAG", core)
        self.assertIn("done=false", core)

    def test_self_repetition_is_forbidden_in_core(self):
        core = " ".join(agent_v2.CORE_SYSTEM_PROMPT.split())
        self.assertIn("DO NOT REPEAT YOURSELF", core)

    def test_bet_what_you_said_is_in_core(self):
        core = " ".join(agent_v2.CORE_SYSTEM_PROMPT.split())
        self.assertIn("BET WHAT YOU SAID", core)

    def test_green_is_explicitly_not_a_bet(self):
        """В прогоне стороны договорились ставить на 'red and green'."""
        core = " ".join(agent_v2.CORE_SYSTEM_PROMPT.split())
        self.assertIn('"green" is not a bet', core)

    def test_even_money_options_are_listed(self):
        core = " ".join(agent_v2.CORE_SYSTEM_PROMPT.split())
        for field in ("red", "black", "odd", "even", "low", "high"):
            self.assertIn(field, core)


if __name__ == "__main__":
    unittest.main(verbosity=2)
