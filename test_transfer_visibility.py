"""TVIS-1: видимость несостоявшихся переводов. python3 test_transfer_visibility.py"""
import unittest

import agent_v2
import run_game_v2 as R


class L:
    def write(self, *a, **k): pass


class Ag:
    def __init__(self, balance, budget=None, sent=0):
        self.balance = balance
        self.transfer_budget_this_round = budget
        self.sent_this_round = sent


class TestOutcome(unittest.TestCase):
    def _out(self, agent, raw, delivered, **kw):
        after_balance = kw.get("after_balance", min(raw, agent.balance))
        return R._transfer_outcome(
            agent, raw, after_balance,
            kw.get("after_open", after_balance),
            kw.get("after_count", kw.get("after_open", after_balance)),
            kw.get("final", delivered), delivered,
            kw.get("to", "p2"), "p2")

    def test_full_transfer_reports_nothing(self):
        self.assertIsNone(self._out(Ag(50), 20, 20))

    def test_no_transfer_reports_nothing(self):
        self.assertIsNone(self._out(Ag(50), 0, 0))

    def test_zero_balance_is_reported(self):
        att = self._out(Ag(0), 20, 0)
        self.assertEqual(att["reason"], "balance")
        self.assertEqual((att["requested"], att["delivered"]), (20, 0))

    def test_round_budget_is_reported(self):
        """Сценарий из вопроса: начал раунд с 0, получил 15, шлёт 20."""
        a = Ag(15, budget=0, sent=0)
        att = self._out(a, 20, 0, after_balance=15, after_open=15,
                        after_count=15, final=0)
        self.assertEqual(att["reason"], "round_budget")
        self.assertIn("0", att["detail"])

    def test_smaller_retry_still_reported(self):
        """И вторая, меньшая попытка тоже не проходит молча."""
        a = Ag(15, budget=0)
        att = self._out(a, 5, 0, after_balance=5, after_open=5,
                        after_count=5, final=0)
        self.assertEqual(att["reason"], "round_budget")

    def test_wrong_recipient_reported(self):
        att = self._out(Ag(50), 20, 0, to="p3")
        self.assertEqual(att["reason"], "wrong_recipient")

    def test_partial_delivery_reported(self):
        att = self._out(Ag(50), 20, 8, after_balance=20, after_open=20,
                        after_count=20, final=8)
        self.assertEqual(att["delivered"], 8)


class TestNote(unittest.TestCase):
    def _turn(self, **kw):
        t = {"from": "p1", "message": "m", "transfer": 0, "transfer_to": None,
             "attempt": None}
        t.update(kw)
        return t

    def test_sender_sees_reason(self):
        t = self._turn(attempt={"requested": 20, "delivered": 0,
                                "reason": "round_budget",
                                "detail": "бюджет переводов на раунд 0"})
        note = agent_v2.format_transfer_note(t, "p1")
        self.assertIn("20", note)
        self.assertIn("бюджет", note)
        self.assertIn("НЕ оплачена", note)

    def test_receiver_sees_the_failure_too(self):
        """Ключевое: получатель тоже не должен считать сделку оплаченной."""
        t = self._turn(attempt={"requested": 20, "delivered": 0,
                                "reason": "balance", "detail": "на счету было 0"})
        note = agent_v2.format_transfer_note(t, "p2")
        self.assertIn("p1", note)
        self.assertIn("НЕ оплачено", note)

    def test_receiver_is_not_told_the_reason(self):
        t = self._turn(attempt={"requested": 20, "delivered": 0,
                                "reason": "balance", "detail": "на счету было 0"})
        self.assertNotIn("на счету", agent_v2.format_transfer_note(t, "p2"))

    def test_successful_transfer_annotated_as_before(self):
        note = agent_v2.format_transfer_note(
            self._turn(transfer=7, transfer_to="p2"), "p1")
        self.assertIn("7", note)
        self.assertNotIn("НЕ", note)

    def test_plain_turn_has_no_note(self):
        self.assertEqual(agent_v2.format_transfer_note(self._turn(), "p1"), "")

    def test_old_entries_without_attempt_field(self):
        """Диалоги, записанные до TVIS-1, не должны ломать рендер."""
        t = {"from": "p1", "message": "m", "transfer": 3, "transfer_to": "p2"}
        self.assertIn("3", agent_v2.format_transfer_note(t, "p1"))


class TestBudgetNote(unittest.TestCase):
    def _agent(self, balance, budget, sent=0):
        a = agent_v2.PlayerAgent.__new__(agent_v2.PlayerAgent)
        a.balance = balance
        a.transfer_budget_this_round = budget
        a.sent_this_round = sent
        return a

    def test_spendable_respects_budget(self):
        self.assertEqual(self._agent(15, 0)._spendable(), 0)
        self.assertEqual(self._agent(100, 50)._spendable(), 50)
        self.assertEqual(self._agent(10, 50)._spendable(), 10)
        self.assertEqual(self._agent(100, 50, sent=30)._spendable(), 20)

    def test_no_budget_configured_means_balance(self):
        self.assertEqual(self._agent(42, None)._spendable(), 42)

    def test_zero_budget_note_is_explicit(self):
        note = self._agent(15, 0)._transfer_budget_note()
        self.assertIn("CANNOT", note)

    def test_note_explains_start_of_round_basis(self):
        note = self._agent(100, 50)._transfer_budget_note()
        self.assertIn("START", note)


class TestDsynGross(unittest.TestCase):
    """DSYN-2: брутто должно доходить до текста промпта, а не только до файла."""

    def _prompt(self, net, sent=None, received=None):
        import json as _j
        captured = {}

        class Spy:
            def chat_json(s, system, user, **kw):
                captured["user"] = user
                return {"trust_score": 5, "deal_done": None, "deal_failed": None,
                        "reputation_note": "n", "future_intent": "i",
                        "summary": "s"}

        class NullLog:
            def write(self, *a, **k): pass

        a = agent_v2.PlayerAgent.__new__(agent_v2.PlayerAgent)
        a.player_id, a.base_dir = "p1", self.tmp
        a.client = Spy()
        a.balance = 100
        a.log = NullLog()
        a.tariff = None
        a.temperature = 0.4
        a.persona_chars = 2000
        a.tok_update_dsyn = 700
        a.update_dsyn("p2", [], net, 1, sent=sent, received=received)
        # update_dsyn ловит любое исключение и молча уходит в заглушку,
        # поэтому пустой захват означает, что промпт не собрался вовсе —
        # это провал теста, а не «нет числа в строке».
        self.assertIn("user", captured, "chat_json не был вызван")
        return captured["user"]

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def test_gross_both_directions_present(self):
        txt = self._prompt(-10, sent=15, received=5)
        self.assertIn("15", txt)
        self.assertIn("5", txt)
        self.assertIn("-10", txt)

    def test_falls_back_to_net_when_gross_absent(self):
        txt = self._prompt(-10)
        self.assertIn("10", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
