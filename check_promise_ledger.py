import tempfile, os
import promise_ledger as pl

with tempfile.TemporaryDirectory() as d:
    pid = "player1"

    # Round 2: player3 says "I'll play and return your money in 3 rounds" —
    # this is what the model would return from update_checklist's extra field.
    model_promises = [{
        "direction": "owed_to_me",
        "counterparty": "player3",
        "amount": 20,
        "due_round": 5,          # promised for 3 rounds from now (r2 -> r5)
        "description": "will play and return the 20c loan",
        "status": "open",
    }]
    saved = pl.merge_and_save(pid, d, model_promises, round_no=2)
    assert saved[0]["due_round"] == 5

    # Round 3, 4: not due yet -> shown as "Upcoming", not nagging
    r3 = pl.due_reminder(pid, d, round_no=3)
    assert "Upcoming" in r3 and "OVERDUE" not in r3
    print("r3 reminder:\n", r3)

    # Round 5: due now
    r5 = pl.due_reminder(pid, d, round_no=5)
    assert "DUE THIS ROUND" in r5
    print("r5 reminder:\n", r5)

    # Round 6: player3 never paid -> overdue, agent CANNOT lose track of it
    r6 = pl.due_reminder(pid, d, round_no=6)
    assert "OVERDUE" in r6 and "player3 owes YOU 20c" in r6
    print("r6 reminder:\n", r6)

    # Round 7: agent later marks it settled via another update_checklist call
    settled = [dict(model_promises[0], status="settled")]
    pl.merge_and_save(pid, d, settled, round_no=7)
    r7 = pl.due_reminder(pid, d, round_no=7)
    assert r7 == ""   # nothing open left -> no reminder block at all
    print("r7 reminder (should be empty):", repr(r7))

    # Malformed model output must never crash the round
    pl.merge_and_save(pid, d, "not a list", round_no=8)
    pl.merge_and_save(pid, d, [{"direction": "bogus"}], round_no=8)

print("OK — all promise_ledger checks passed")
