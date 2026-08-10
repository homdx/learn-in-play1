"""
RECON-1: a real coin transfer should auto-settle the matching open
promise on both sides, without waiting for either player's LLM to
mention it again in a future checklist update.

Reproduces the exact bug found in a saved game: player4's
promises_player4.json kept an "owed_to_me" entry open for a 3-coin
payment from player1 that had already arrived two turns earlier in
the same dialogue.
"""
import tempfile
import unittest

import promise_ledger as pl


class TestAutoSettleFromTransfer(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()

    def test_real_bug_reproduction_player4_player1(self):
        # player4's own ledger: player1 still "owes" 3 coins for a call,
        # exactly as found in the saved game (status stuck at "open").
        pl.save_promises("player4", self.base, [
            {"id": 5, "direction": "owed_to_me", "counterparty": "player1",
             "amount": 3, "due_round": 4, "description": "payment for next call",
             "status": "open"},
        ])

        # The 3-coin transfer from player1 to player4 actually happens.
        pl.auto_settle_from_transfer(self.base, "player1", "player4", 3, round_no=3)

        promises = pl.load_promises("player4", self.base)
        self.assertEqual(promises[0]["status"], "settled",
                          "payment already received must not stay 'open'")

    def test_mirrors_onto_sender_i_owe_side_too(self):
        pl.save_promises("player1", self.base, [
            {"id": 1, "direction": "i_owe", "counterparty": "player4",
             "amount": 3, "due_round": 4, "description": "call payment",
             "status": "open"},
        ])
        pl.auto_settle_from_transfer(self.base, "player1", "player4", 3, round_no=3)
        promises = pl.load_promises("player1", self.base)
        self.assertEqual(promises[0]["status"], "settled")

    def test_partial_payment_does_not_falsely_settle(self):
        pl.save_promises("player4", self.base, [
            {"id": 5, "direction": "owed_to_me", "counterparty": "player1",
             "amount": 10, "due_round": 4, "description": "big debt",
             "status": "open"},
        ])
        pl.auto_settle_from_transfer(self.base, "player1", "player4", 3, round_no=3)
        promises = pl.load_promises("player4", self.base)
        self.assertEqual(promises[0]["status"], "open",
                          "a partial payment must not close the whole debt")

    def test_settles_oldest_first_across_multiple_debts(self):
        pl.save_promises("player4", self.base, [
            {"id": 5, "direction": "owed_to_me", "counterparty": "player1",
             "amount": 3, "due_round": 4, "description": "older",
             "status": "open"},
            {"id": 6, "direction": "owed_to_me", "counterparty": "player1",
             "amount": 3, "due_round": 6, "description": "newer",
             "status": "open"},
        ])
        pl.auto_settle_from_transfer(self.base, "player1", "player4", 3, round_no=5)
        promises = {p["id"]: p for p in pl.load_promises("player4", self.base)}
        self.assertEqual(promises[5]["status"], "settled")
        self.assertEqual(promises[6]["status"], "open")

    def test_unrelated_counterparty_is_untouched(self):
        pl.save_promises("player4", self.base, [
            {"id": 5, "direction": "owed_to_me", "counterparty": "player2",
             "amount": 3, "due_round": 4, "description": "unrelated",
             "status": "open"},
        ])
        pl.auto_settle_from_transfer(self.base, "player1", "player4", 3, round_no=3)
        promises = pl.load_promises("player4", self.base)
        self.assertEqual(promises[0]["status"], "open")

    def test_zero_or_negative_amount_is_a_no_op(self):
        pl.save_promises("player4", self.base, [
            {"id": 5, "direction": "owed_to_me", "counterparty": "player1",
             "amount": 3, "due_round": 4, "description": "x", "status": "open"},
        ])
        pl.auto_settle_from_transfer(self.base, "player1", "player4", 0, round_no=3)
        self.assertEqual(pl.load_promises("player4", self.base)[0]["status"], "open")


if __name__ == "__main__":
    unittest.main()
