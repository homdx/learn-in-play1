"""
test_multi_bet.py — тесты для MULTI-BET-1: несколько ставок за раунд.

Два обязательных блока (по требованию задачи):
  1. Обязательная аварийная заглушка "1 монета на red", если LLM глючит
     при размещении ставки — ДОЛЖНА срабатывать одинаково и при
     max_bets_per_round=1, и при max_bets_per_round>1.
  2. Корректная работа нескольких ставок одновременно: конфиг
     max_bets_per_round, парсинг/валидация/оценка списка ставок,
     списание суммарного баланса, обработка крупье.

    python3 test_multi_bet.py            # прогнать всё
    python3 test_multi_bet.py -v         # с подробностями
"""

import configparser
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import llm_client
import agent_v2
import croupier_v2
import open_bets
import run_game_v2
from agent_v2 import PlayerAgent
from llm_client import LLMUnavailable


PLAYERS = ["player1", "player2", "player3"]
START_BALANCE = 100


def make_cfg(table_dir, logs_dir, max_bets_per_round=1, start_balance=START_BALANCE,
             max_bet_fraction=0.4):
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "api":       {"active": "local"},
        "api_local": {"base_url": "http://unused", "model": "fake"},
        "game":      {"table_dir": table_dir, "logs_dir": logs_dir,
                      "players": ",".join(PLAYERS),
                      "start_balance": str(start_balance),
                      "rounds": "1", "max_bet_fraction": str(max_bet_fraction),
                      "round_delay_sec": "0",
                      "max_bets_per_round": str(max_bets_per_round)},
        "player":    {"temperature": "0.5", "max_tokens": "100",
                      "history_window": "10"},
    })
    return cfg


class FakeLLM:
    """Скриптованный клиент. Поведение задаётся через FakeLLM.behaviour."""

    behaviour = {}

    @classmethod
    def from_config(cls, cfg, section=None):
        return cls()

    def chat_json(self, system, user, temperature=0.0, max_tokens=0):
        b = FakeLLM.behaviour
        if "Place casino bet" in system:
            fn = b.get("bet")
            if fn is None:
                raise LLMUnavailable("no bet behaviour configured")
            return fn(user)
        return {}


class MultiBetHarness(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="casino_multibet_test_")
        self.table = os.path.join(self.tmp, "table")
        self.logs = os.path.join(self.tmp, "logs")
        os.makedirs(self.table, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)
        self._orig_client = agent_v2.LLMClient
        agent_v2.LLMClient = FakeLLM
        llm_client.LLMClient = FakeLLM
        FakeLLM.behaviour = {}

    def tearDown(self):
        agent_v2.LLMClient = self._orig_client
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_agent(self, pid="player1", max_bets_per_round=1, start_balance=START_BALANCE,
                   max_bet_fraction=0.4):
        cfg = make_cfg(self.table, self.logs, max_bets_per_round=max_bets_per_round,
                       start_balance=start_balance, max_bet_fraction=max_bet_fraction)
        return PlayerAgent(pid, self.table, cfg)


# ═══════════════════════════ 1. ОБЯЗАТЕЛЬНАЯ ЗАГЛУШКА ══════════════════════

class TestMandatoryFallbackBet(MultiBetHarness):
    """
    Если LLM глючит (даёт мусор дважды подряд, либо кидает исключение,
    отличное от LLMUnavailable), decide_bet() ОБЯЗАН вернуть ровно
    одну ставку even_money/red на 1 монету — независимо от того,
    включён режим нескольких ставок или нет.
    """

    def test_fallback_is_one_coin_on_red_single_bet_mode(self):
        FakeLLM.behaviour = {"bet": lambda u: {"garbage": True}}  # невалидная ставка
        agent = self.make_agent(max_bets_per_round=1)
        bet = agent.decide_bet(round_no=1)

        self.assertEqual(bet["type"], "even_money")
        self.assertEqual(bet["selection"], "red")
        self.assertEqual(bet["amount"], 1)
        self.assertEqual(bet["player_id"], "player1")
        # заглушка — это плоская ставка, НЕ контейнер "bets"
        self.assertNotIn("bets", bet)

    def test_fallback_is_one_coin_on_red_multi_bet_mode(self):
        """
        Даже когда max_bets_per_round>1, при сбое LLM игрок получает
        РОВНО ОДНУ простую ставку 1 на красное, а не список ставок.
        """
        FakeLLM.behaviour = {"bet": lambda u: {"garbage": True}}
        agent = self.make_agent(max_bets_per_round=3)
        bet = agent.decide_bet(round_no=1)

        self.assertEqual(bet["type"], "even_money")
        self.assertEqual(bet["selection"], "red")
        self.assertEqual(bet["amount"], 1)
        self.assertNotIn("bets", bet)

    def test_fallback_triggers_on_repeated_exception(self):
        """Модель дважды подряд бросает произвольное исключение (не
        LLMUnavailable) — тоже должно кончиться заглушкой, а не падением."""
        calls = {"n": 0}

        def flaky(u):
            calls["n"] += 1
            raise ValueError("model returned malformed output")

        FakeLLM.behaviour = {"bet": flaky}
        agent = self.make_agent(max_bets_per_round=1)
        bet = agent.decide_bet(round_no=1)

        self.assertEqual(calls["n"], 2)  # ровно 2 попытки, как в коде
        self.assertEqual(bet, {"type": "even_money", "selection": "red",
                               "amount": 1, "player_id": "player1"})

    def test_fallback_respects_zero_balance(self):
        """Если баланс уже 0, заглушка не должна ставить больше, чем есть."""
        FakeLLM.behaviour = {"bet": lambda u: {"garbage": True}}
        agent = self.make_agent(max_bets_per_round=1, start_balance=0)
        agent.balance = 0
        bet = agent.decide_bet(round_no=1)
        self.assertEqual(bet["amount"], 0)

    def test_fallback_bet_is_processed_correctly_by_croupier(self):
        """Сквозной тест: заглушка реально проходит через крупье как
        обычная валидная ставка even_money/red."""
        common.write_json(common.bet_file("player1", self.table),
                          {"type": "even_money", "selection": "red",
                           "amount": 1, "player_id": "player1"})
        winning_number, results = croupier_v2.run_round(self.table, winning_number=1)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r["win"])          # 1 — красное число
        self.assertEqual(r["payout"], 2)   # 1 (ставка) + 1*1 (1:1)


# ═══════════════════════════ 2. НЕСКОЛЬКО СТАВОК ═══════════════════════════

class TestCommonMultiBetHelpers(unittest.TestCase):
    """Юнит-тесты чистых функций common.py для контейнера ставок."""

    def test_normalize_legacy_single_bet(self):
        bet = {"type": "straight", "numbers": [7], "amount": 5}
        self.assertEqual(common.normalize_bet_container(bet), [bet])

    def test_normalize_multi_bet(self):
        subs = [{"type": "straight", "numbers": [7], "amount": 5},
                {"type": "even_money", "selection": "black", "amount": 3}]
        bet = {"bets": subs}
        self.assertEqual(common.normalize_bet_container(bet), subs)

    def test_normalize_rejects_empty_bets_list(self):
        with self.assertRaises(ValueError):
            common.normalize_bet_container({"bets": []})

    def test_normalize_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            common.normalize_bet_container("not a dict")

    def test_total_bet_amount_single(self):
        self.assertEqual(common.total_bet_amount(
            {"type": "straight", "numbers": [7], "amount": 5}), 5)

    def test_total_bet_amount_multi(self):
        bet = {"bets": [
            {"type": "straight", "numbers": [7], "amount": 5},
            {"type": "straight", "numbers": [11], "amount": 3},
            {"type": "even_money", "selection": "black", "amount": 2},
        ]}
        self.assertEqual(common.total_bet_amount(bet), 10)

    def test_validate_bets_ok(self):
        bet = {"bets": [
            {"type": "straight", "numbers": [7], "amount": 5},
            {"type": "even_money", "selection": "red", "amount": 2},
        ]}
        subs = common.validate_bets(bet, max_bets=3, balance=100)
        self.assertEqual(len(subs), 2)

    def test_validate_bets_rejects_too_many(self):
        bet = {"bets": [
            {"type": "straight", "numbers": [1], "amount": 1},
            {"type": "straight", "numbers": [2], "amount": 1},
            {"type": "straight", "numbers": [3], "amount": 1},
        ]}
        with self.assertRaises(ValueError):
            common.validate_bets(bet, max_bets=2)

    def test_validate_bets_rejects_over_balance(self):
        bet = {"bets": [
            {"type": "straight", "numbers": [1], "amount": 60},
            {"type": "straight", "numbers": [2], "amount": 60},
        ]}
        with self.assertRaises(ValueError):
            common.validate_bets(bet, balance=100)

    def test_validate_bets_propagates_sub_bet_errors(self):
        bet = {"bets": [{"type": "not_a_real_type", "amount": 5}]}
        with self.assertRaises(ValueError):
            common.validate_bets(bet)

    def test_evaluate_bets_multi_mixed_win_loss(self):
        bet = {"bets": [
            {"type": "straight", "numbers": [7], "amount": 5},   # выиграет на 7
            {"type": "straight", "numbers": [11], "amount": 3},  # проиграет
        ]}
        any_win, total_payout, per_bet = common.evaluate_bets(bet, winning_number=7)
        self.assertTrue(any_win)
        self.assertEqual(total_payout, 5 + 5 * 35)   # только первая ставка сыграла
        self.assertEqual(len(per_bet), 2)
        self.assertTrue(per_bet[0]["win"])
        self.assertFalse(per_bet[1]["win"])

    def test_evaluate_bets_multi_all_lose(self):
        bet = {"bets": [
            {"type": "straight", "numbers": [1], "amount": 5},
            {"type": "straight", "numbers": [2], "amount": 5},
        ]}
        any_win, total_payout, per_bet = common.evaluate_bets(bet, winning_number=3)
        self.assertFalse(any_win)
        self.assertEqual(total_payout, 0)

    def test_evaluate_bets_single_bet_still_works(self):
        bet = {"type": "even_money", "selection": "red", "amount": 10}
        any_win, total_payout, per_bet = common.evaluate_bets(bet, winning_number=1)
        self.assertTrue(any_win)
        self.assertEqual(total_payout, 20)
        self.assertEqual(len(per_bet), 1)

    def test_describe_bets_multi(self):
        bet = {"bets": [
            {"type": "straight", "numbers": [7], "amount": 5},
            {"type": "even_money", "selection": "black", "amount": 2},
        ]}
        desc = common.describe_bets(bet)
        self.assertIn("straight([7]) amount=5", desc)
        self.assertIn("even_money(black) amount=2", desc)
        self.assertIn(";", desc)  # несколько ставок разделены


class TestAgentDecidesMultipleBets(MultiBetHarness):
    """decide_bet() при max_bets_per_round > 1 должен парсить и
    нормализовать список ставок из ответа LLM."""

    def test_agent_places_several_bets_within_budget(self):
        def multi_bet_response(user):
            return {"bets": [
                {"type": "straight", "numbers": [7], "amount": 5},
                {"type": "straight", "numbers": [11], "amount": 5},
                {"type": "even_money", "selection": "black", "amount": 10},
            ], "reasoning": "spreading risk"}

        FakeLLM.behaviour = {"bet": multi_bet_response}
        agent = self.make_agent(max_bets_per_round=3)
        bet = agent.decide_bet(round_no=1)

        self.assertIn("bets", bet)
        self.assertEqual(len(bet["bets"]), 3)
        self.assertEqual(common.total_bet_amount(bet), 20)
        self.assertEqual(bet["player_id"], "player1")

    def test_agent_scales_down_bets_exceeding_balance(self):
        """Если сумма запрошенных ставок больше баланса, ставки должны
        быть пропорционально уменьшены, а не отброшены целиком."""
        def greedy_bet_response(user):
            return {"bets": [
                {"type": "straight", "numbers": [1], "amount": 80},
                {"type": "straight", "numbers": [2], "amount": 80},
            ]}

        FakeLLM.behaviour = {"bet": greedy_bet_response}
        agent = self.make_agent(max_bets_per_round=2, start_balance=100)
        agent.balance = 100
        bet = agent.decide_bet(round_no=1)

        self.assertIn("bets", bet)
        total = common.total_bet_amount(bet)
        self.assertLessEqual(total, 100)
        self.assertGreater(total, 0)

    def test_agent_rejects_too_many_bets_and_retries_then_falls_back(self):
        """LLM просит больше ставок, чем разрешено конфигом, оба раза —
        итог должен быть аварийной заглушкой, а не превышением лимита."""
        def too_many_bets(user):
            return {"bets": [
                {"type": "straight", "numbers": [1], "amount": 1},
                {"type": "straight", "numbers": [2], "amount": 1},
                {"type": "straight", "numbers": [3], "amount": 1},
            ]}

        FakeLLM.behaviour = {"bet": too_many_bets}
        agent = self.make_agent(max_bets_per_round=2)
        bet = agent.decide_bet(round_no=1)

        self.assertNotIn("bets", bet)
        self.assertEqual(bet["type"], "even_money")
        self.assertEqual(bet["selection"], "red")
        self.assertEqual(bet["amount"], 1)

    def test_single_bet_mode_unaffected_by_bets_key(self):
        """max_bets_per_round=1 (по умолчанию) игнорирует 'bets' в ответе
        и требует старый плоский формат — полная обратная совместимость."""
        def old_style_response(user):
            return {"type": "straight", "numbers": [7], "amount": 5}

        FakeLLM.behaviour = {"bet": old_style_response}
        agent = self.make_agent(max_bets_per_round=1)
        bet = agent.decide_bet(round_no=1)

        self.assertNotIn("bets", bet)
        self.assertEqual(bet["type"], "straight")
        self.assertEqual(bet["numbers"], [7])
        self.assertEqual(bet["amount"], 5)


class TestCroupierMultiBetRound(MultiBetHarness):
    """Сквозные тесты обработки нескольких ставок крупье + списание
    баланса в run_game_v2._place_bet_for_player."""

    def test_croupier_resolves_multiple_bets_for_one_player(self):
        common.write_json(common.bet_file("player1", self.table), {
            "bets": [
                {"type": "straight", "numbers": [7], "amount": 5},
                {"type": "straight", "numbers": [11], "amount": 5},
                {"type": "even_money", "selection": "black", "amount": 10},
            ],
            "player_id": "player1",
        })
        winning_number, results = croupier_v2.run_round(self.table, winning_number=7)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r["win"])
        # straight(7) hits: 5 + 5*35 = 180; other two bets lose.
        self.assertEqual(r["payout"], 180)
        self.assertEqual(len(r["bets"]), 3)
        # файл ставки должен быть удалён после обработки
        self.assertFalse(os.path.exists(common.bet_file("player1", self.table)))

    def test_croupier_handles_mix_of_single_and_multi_bet_players(self):
        common.write_json(common.bet_file("player1", self.table),
                          {"type": "even_money", "selection": "red", "amount": 10,
                           "player_id": "player1"})
        common.write_json(common.bet_file("player2", self.table), {
            "bets": [
                {"type": "straight", "numbers": [1], "amount": 5},
                {"type": "straight", "numbers": [2], "amount": 5},
            ],
            "player_id": "player2",
        })
        winning_number, results = croupier_v2.run_round(self.table, winning_number=1)
        by_pid = {r["player_id"]: r for r in results}
        self.assertTrue(by_pid["player1"]["win"])   # 1 — красное
        self.assertTrue(by_pid["player2"]["win"])   # straight(1) сыграл
        self.assertEqual(by_pid["player2"]["payout"], 5 + 5 * 35)

    def test_croupier_invalidates_bet_container_with_bad_sub_bet(self):
        common.write_json(common.bet_file("player1", self.table), {
            "bets": [
                {"type": "straight", "numbers": [7], "amount": 5},
                {"type": "straight", "numbers": [999], "amount": 5},  # вне диапазона
            ],
            "player_id": "player1",
        })
        winning_number, results = croupier_v2.run_round(self.table, winning_number=7)
        r = results[0]
        self.assertFalse(r["win"])
        self.assertEqual(r["payout"], 0)

    def test_place_bet_for_player_deducts_total_of_all_bets(self):
        def multi_bet_response(user):
            return {"bets": [
                {"type": "straight", "numbers": [7], "amount": 5},
                {"type": "even_money", "selection": "black", "amount": 15},
            ]}

        FakeLLM.behaviour = {"bet": multi_bet_response}
        agent = self.make_agent(max_bets_per_round=2, start_balance=100)
        agent.balance = 100

        class DummyLogger:
            def write(self, pid, msg):
                pass

        placed = run_game_v2._place_bet_for_player(
            agent, "player1", self.table, round_no=1, logger=DummyLogger())
        self.assertTrue(placed)
        self.assertEqual(agent.balance, 100 - 20)

        bet_on_disk = common.read_json(common.bet_file("player1", self.table))
        self.assertEqual(common.total_bet_amount(bet_on_disk), 20)


class TestFractionalBetAmountBug(MultiBetHarness):
    """
    BUGFIX-AMOUNT-1: LLM иногда возвращает amount дробным (5.7) или
    строкой ("5"). Раньше это значение "путешествовало" дробным через
    баланс/выплаты/JSON много раундов подряд. validate_bet() теперь
    приводит amount к int РОВНО ОДИН РАЗ и мутирует сам bet-словарь.
    """

    def test_validate_bet_coerces_float_amount_in_place(self):
        bet = {"type": "even_money", "selection": "red", "amount": 5.7}
        common.validate_bet(bet)
        self.assertEqual(bet["amount"], 5)
        self.assertIsInstance(bet["amount"], int)

    def test_validate_bet_coerces_string_amount(self):
        bet = {"type": "straight", "numbers": [7], "amount": "12"}
        common.validate_bet(bet)
        self.assertEqual(bet["amount"], 12)
        self.assertIsInstance(bet["amount"], int)

    def test_validate_bet_rejects_non_numeric_amount(self):
        bet = {"type": "straight", "numbers": [7], "amount": "not-a-number"}
        with self.assertRaises(ValueError):
            common.validate_bet(bet)

    def test_evaluate_bet_payout_is_int_after_fractional_input(self):
        bet = {"type": "even_money", "selection": "red", "amount": 5.7}
        common.validate_bet(bet)   # normally called before evaluate in real flow
        win, payout = common.evaluate_bet(bet, winning_number=1)
        self.assertTrue(win)
        self.assertIsInstance(payout, int)
        self.assertEqual(payout, 10)  # 5 + 5*1, NOT 11.4

    def test_multi_bet_sub_amounts_coerced_by_validate_bets(self):
        bet = {"bets": [
            {"type": "straight", "numbers": [7], "amount": 5.9},
            {"type": "even_money", "selection": "black", "amount": "3"},
        ]}
        common.validate_bets(bet)
        subs = common.normalize_bet_container(bet)
        self.assertEqual(subs[0]["amount"], 5)
        self.assertEqual(subs[1]["amount"], 3)
        self.assertIsInstance(subs[0]["amount"], int)
        self.assertIsInstance(subs[1]["amount"], int)

    def test_croupier_never_writes_fractional_balance_from_fractional_bet(self):
        """
        Сквозной тест: файл ставки на диске содержит дробное amount (как
        если бы LLM это записал до фикса или другой процесс сериализовал
        число как float) — итоговый payout и total_bet_amount ОБЯЗАНЫ
        быть целыми, а не дробными, после обработки крупье.
        """
        common.write_json(common.bet_file("player1", self.table),
                          {"type": "even_money", "selection": "red",
                           "amount": 7.3, "player_id": "player1"})
        winning_number, results = croupier_v2.run_round(self.table, winning_number=1)
        r = results[0]
        self.assertTrue(r["win"])
        self.assertIsInstance(r["payout"], int)
        self.assertEqual(r["payout"], 14)  # 7 + 7*1, not 14.6

    def test_agent_decide_bet_single_mode_coerces_fractional_llm_amount(self):
        """decide_bet() (одиночный режим) не должен пропускать дробную
        сумму дальше в bet["amount"] без округления."""
        FakeLLM.behaviour = {"bet": lambda u: {"type": "even_money",
                                               "selection": "red",
                                               "amount": 9.9}}
        agent = self.make_agent(max_bets_per_round=1)
        bet = agent.decide_bet(round_no=1)
        self.assertIsInstance(bet["amount"], int)
        self.assertEqual(bet["amount"], 9)

    def test_agent_decide_bet_multi_mode_coerces_fractional_llm_amounts(self):
        def multi_fractional(u):
            return {"bets": [
                {"type": "straight", "numbers": [7], "amount": 4.6},
                {"type": "even_money", "selection": "black", "amount": 3.2},
            ]}

        FakeLLM.behaviour = {"bet": multi_fractional}
        agent = self.make_agent(max_bets_per_round=2)
        bet = agent.decide_bet(round_no=1)
        for sub in bet["bets"]:
            self.assertIsInstance(sub["amount"], int)
        self.assertEqual(common.total_bet_amount(bet), 4 + 3)


# ═════════════════════ 3. ПОСТ-ДИАЛОГОВАЯ СИНАПСА И ЧЕК-ЛИСТ ═══════════════

class TestPostDialogueSynapseExists(MultiBetHarness):
    """
    Проверка, что 'синапс после разговора' действительно существует и
    пишется на диск: (a) диалоговая синапса (dsyn, репутация о партнёре)
    через update_dsyn(), и (b) чек-лист (краткосрочная повестка,
    буквально обновляемая ПОСЛЕ каждого диалога) через update_checklist().
    """

    def test_dsyn_is_saved_to_disk_after_dialogue(self):
        FakeLLM.behaviour = {}  # update_dsyn ветка не завязана на "bet" ключ
        agent = self.make_agent(max_bets_per_round=1)
        dsyn_path = agent_v2.dsyn_file(agent.player_id, self.table)
        self.assertFalse(os.path.exists(dsyn_path))

        conversation = [{"from": "player1", "message": "let's talk business"}]
        agent.update_dsyn("player2", conversation, net_transfer=0, round_no=1)

        self.assertTrue(os.path.exists(dsyn_path),
                        "dsyn не была сохранена на диск после диалога")
        data = common.read_json(dsyn_path)
        self.assertIn("player2", data.get("reputation", {}))

    def test_checklist_is_saved_to_disk_after_dialogue(self):
        agent = self.make_agent(max_bets_per_round=1)

        def checklist_update(u):
            return {"checklist": "player2 owes me 5 by r2", "promises": []}

        FakeLLM.behaviour = {"bet": lambda u: {"garbage": True}}  # unrelated branch
        # patch chat_json behaviour generically via a small wrapper
        agent.client.chat_json = lambda **kw: checklist_update(kw.get("user", ""))

        checklist_path = agent_v2.checklist_file(agent.player_id, self.table) \
            if hasattr(agent_v2, "checklist_file") else None

        conversation = [{"from": "player1", "message": "deal agreed"}]
        new_text = agent.update_checklist("player2", conversation, net_transfer=0,
                                          round_no=1)
        self.assertIn("owes me 5", new_text)

        # перечитываем через официальный загрузчик, а не строим путь руками —
        # так тест не завязан на внутреннее имя файла.
        reloaded = agent_v2.load_checklist(agent.player_id, self.table)
        self.assertIn("owes me 5", reloaded)


class TestChecklistCapacityForMultiBet(MultiBetHarness):
    """
    MULTI-BET-2: в режиме нескольких ставок за раунд чек-лист должен
    получать БОЛЬШИЙ потолок автоматически (больше "строк"/символов),
    потому что игроку нужно следить и за своими, и за чужими ставками —
    если только пользователь явно не задал checklist_chars в конфиге,
    в этом случае явное значение в приоритете.
    """

    def test_single_bet_mode_keeps_default_checklist_cap(self):
        agent = self.make_agent(max_bets_per_round=1)
        self.assertEqual(agent.checklist_chars, agent_v2.DEF_CHECKLIST_CHARS)

    def test_multi_bet_mode_raises_checklist_cap_automatically(self):
        agent = self.make_agent(max_bets_per_round=3)
        self.assertEqual(agent.checklist_chars,
                         agent_v2.DEF_CHECKLIST_CHARS_MULTI_BET)
        self.assertGreater(agent.checklist_chars, agent_v2.DEF_CHECKLIST_CHARS)

    def test_explicit_checklist_chars_config_wins_over_multi_bet_default(self):
        cfg = make_cfg(self.table, self.logs, max_bets_per_round=3)
        if not cfg.has_section("memory"):
            cfg.add_section("memory")
        cfg.set("memory", "checklist_chars", "1234")
        agent = PlayerAgent("player1", self.table, cfg)
        self.assertEqual(agent.checklist_chars, 1234)

    def test_explicit_checklist_chars_multi_bet_override_respected(self):
        cfg = make_cfg(self.table, self.logs, max_bets_per_round=3)
        if not cfg.has_section("memory"):
            cfg.add_section("memory")
        cfg.set("memory", "checklist_chars_multi_bet", "5000")
        agent = PlayerAgent("player1", self.table, cfg)
        self.assertEqual(agent.checklist_chars, 5000)

    def test_update_checklist_prompt_includes_bet_tracking_hint_when_multi(self):
        agent = self.make_agent(max_bets_per_round=3)
        captured = {}

        def fake_chat_json(**kw):
            captured["user"] = kw.get("user", "")
            return {"checklist": "x", "promises": []}

        agent.client.chat_json = lambda **kw: fake_chat_json(**kw)
        agent.update_checklist("player2", [{"from": "player1", "message": "hi"}],
                               net_transfer=0, round_no=1)
        self.assertIn("track", captured["user"].lower())
        self.assertIn("numbers/selections", captured["user"])

    def test_update_checklist_prompt_omits_bet_tracking_hint_when_single(self):
        agent = self.make_agent(max_bets_per_round=1)
        captured = {}

        def fake_chat_json(**kw):
            captured["user"] = kw.get("user", "")
            return {"checklist": "x", "promises": []}

        agent.client.chat_json = lambda **kw: fake_chat_json(**kw)
        agent.update_checklist("player2", [{"from": "player1", "message": "hi"}],
                               net_transfer=0, round_no=1)
        self.assertNotIn("numbers/selections", captured["user"])


class TestMultiBetStrategyHintEarlierStages(MultiBetHarness):
    """
    MULTI-BET-3: раньше про несколько ставок узнавали ТОЛЬКО внутри
    decide_bet(), то есть уже в момент самой ставки — на стадиях
    планирования (plan_round, перед диалогами) и рефлексии
    (reflect_betting, сразу после раунда) об этой возможности не было ни
    слова, так что стратегия на несколько ставок физически не могла
    сложиться заранее. Хинт должен появляться там ТОЛЬКО когда
    max_bets_per_round > 1, и отсутствовать в одиночном режиме.
    """

    def test_plan_round_includes_hint_when_multi_bet_enabled(self):
        agent = self.make_agent(max_bets_per_round=3)
        captured = {}
        agent.client.chat_json = lambda **kw: (
            captured.setdefault("user", kw.get("user", "")),
            {"checklist": "x", "promises": []}
        )[1]
        agent.plan_round(round_no=1, available_partners=["player2"])
        self.assertIn("UP TO 3 SEPARATE bets", captured["user"])

    def test_plan_round_omits_hint_when_single_bet_mode(self):
        agent = self.make_agent(max_bets_per_round=1)
        captured = {}
        agent.client.chat_json = lambda **kw: (
            captured.setdefault("user", kw.get("user", "")),
            {"checklist": "x", "promises": []}
        )[1]
        agent.plan_round(round_no=1, available_partners=["player2"])
        self.assertNotIn("SEPARATE bets", captured["user"])

    def test_reflect_betting_includes_hint_when_multi_bet_enabled(self):
        agent = self.make_agent(max_bets_per_round=3)
        captured = {}
        agent.client.chat_json = lambda **kw: (
            captured.setdefault("user", kw.get("user", "")),
            {"notes": "x", "update_persona": False}
        )[1]
        agent.reflect_betting(last_entry=None, round_no=1)
        self.assertIn("UP TO 3 SEPARATE bets", captured["user"])

    def test_reflect_betting_omits_hint_when_single_bet_mode(self):
        agent = self.make_agent(max_bets_per_round=1)
        captured = {}
        agent.client.chat_json = lambda **kw: (
            captured.setdefault("user", kw.get("user", "")),
            {"notes": "x", "update_persona": False}
        )[1]
        agent.reflect_betting(last_entry=None, round_no=1)
        self.assertNotIn("SEPARATE bets", captured["user"])

    def test_hint_mentions_correct_max_for_this_agent(self):
        agent = self.make_agent(max_bets_per_round=5)
        hint = agent._multi_bet_strategy_hint()
        self.assertIn("UP TO 5", hint)

    def test_hint_empty_string_in_single_bet_mode(self):
        agent = self.make_agent(max_bets_per_round=1)
        self.assertEqual(agent._multi_bet_strategy_hint(), "")


class TestLedgerStressManyMultiBetEntries(unittest.TestCase):
    """
    Реестр раньше видел не больше одной ставки на (игрок, раунд). Теперь
    записи могут содержать контейнер с несколькими под-ставками (до
    max_bets_per_round). Проверяем, что весь код форматирования —
    публичный леджер, скорборд, "уже размещённые ставки" — переживает
    большой реестр со смесью 1/несколько ставок и старого/нового формата
    без исключений и без потери данных.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="casino_ledger_stress_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _random_bet(self, rng, max_bets=4):
        types = ["straight", "split", "street", "corner", "sixline",
                "dozen", "column", "even_money"]
        n = rng.randint(1, max_bets)
        subs = []
        for _ in range(n):
            t = rng.choice(types)
            if t == "dozen":
                subs.append({"type": t, "selection": rng.choice(["1st12", "2nd12", "3rd12"]),
                            "amount": rng.randint(1, 20)})
            elif t == "column":
                subs.append({"type": t, "selection": rng.choice(["col1", "col2", "col3"]),
                            "amount": rng.randint(1, 20)})
            elif t == "even_money":
                subs.append({"type": t, "selection": rng.choice(
                    ["red", "black", "even", "odd", "low", "high"]),
                    "amount": rng.randint(1, 20)})
            else:
                size = common.BET_SIZE[t]
                subs.append({"type": t, "numbers": rng.sample(range(37), size),
                            "amount": rng.randint(1, 20)})
        if n == 1 and rng.random() < 0.5:
            return subs[0]           # иногда честный старый плоский формат
        return {"bets": subs}

    def test_large_mixed_ledger_survives_all_formatting_paths(self):
        import random as _random
        rng = _random.Random(42)
        players = PLAYERS
        entries_written = 0

        for round_no in range(1, 41):        # 40 раундов x 5 игроков = 200 записей
            for pid in players:
                bet = self._random_bet(rng)
                winning_number = rng.randint(0, 36)
                subs = common.validate_bets(bet)     # не должно бросать
                win, payout, per_bet = common.evaluate_bets(bet, winning_number)
                self.assertIsInstance(payout, int)
                agent_v2.append_public_ledger(self.tmp, {
                    "round_no": round_no, "player_id": pid,
                    "winning_number": winning_number, "bet": bet,
                    "win": win, "payout": payout,
                })
                entries_written += 1

        ledger = agent_v2.load_public_ledger(self.tmp)
        self.assertEqual(len(ledger), entries_written)

        # Публичный леджер (окно 25) — не должен бросать и должен вернуть
        # непустой, разумного размера текст.
        txt_ledger = agent_v2._format_public_ledger(ledger, window=25,
                                                     exclude_pid="player1")
        self.assertIsInstance(txt_ledger, str)
        self.assertNotEqual(txt_ledger, "(no public results yet)")
        self.assertIn("amount=", txt_ledger)

        # Скорборд — арифметика по всем 200 записям не должна падать и
        # должна дать по строке на каждого НЕ-исключённого игрока.
        txt_score = agent_v2._format_scoreboard(ledger, exclude_pid="player1")
        for pid in players:
            if pid != "player1":
                self.assertIn(pid, txt_score)
        self.assertIn("casino P&L=", txt_score)

        # "Уже размещённые ставки этого раунда" — записываем свежие
        # мульти-ставки для всех игроков и форматируем.
        for pid in players:
            common.write_json(common.bet_file(pid, self.tmp), self._random_bet(rng))
        txt_open = open_bets.format_for_prompt(self.tmp, "player1")
        self.assertIn("Bets ALREADY PLACED", txt_open)
        for pid in players:
            if pid != "player1":
                self.assertIn(pid, txt_open)

    def test_ledger_entry_stays_one_per_player_per_round_even_with_multi_bets(self):
        """FIX-16 (замещение записи при переигровке раунда) не должен
        сломаться из-за контейнера с несколькими ставками."""
        bet_v1 = {"bets": [{"type": "straight", "numbers": [1], "amount": 5},
                           {"type": "straight", "numbers": [2], "amount": 5}]}
        bet_v2 = {"bets": [{"type": "straight", "numbers": [3], "amount": 5},
                           {"type": "straight", "numbers": [4], "amount": 5},
                           {"type": "straight", "numbers": [5], "amount": 5}]}
        agent_v2.append_public_ledger(self.tmp, {
            "round_no": 1, "player_id": "player1", "winning_number": 0,
            "bet": bet_v1, "win": False, "payout": 0,
        })
        agent_v2.append_public_ledger(self.tmp, {
            "round_no": 1, "player_id": "player1", "winning_number": 3,
            "bet": bet_v2, "win": True, "payout": 40,
        })
        ledger = agent_v2.load_public_ledger(self.tmp)
        self.assertEqual(len(ledger), 1)   # переигровка ЗАМЕНИЛА, не добавила
        self.assertEqual(len(ledger[0]["bet"]["bets"]), 3)


if __name__ == "__main__":
    unittest.main()
