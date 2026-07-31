"""
transfer_ledger.py — deterministic, symmetric record of coin transfers
between players (FIX-21).

The scenario this closes: player1 asks 30 for a strategy, player2 sends 20,
they argue over the remaining 10, and the dialogue ends with no agreement —
but the 20 coins already moved. Until now that fact lived only inside each
player's OWN dialogue synapse (`update_dsyn`), written by an LLM call that
reads the transcript from that player's point of view. Nothing stopped
player1's synapse from recording "received a gift" while player2's recorded
"paid in full and got nothing" — two irreconcilable stories about the same
transfer, with no shared ground truth either side could point to.

`run_dialogue()` in run_game_v2.py already computes the exact amounts
(`a_total_sent`, `b_total_sent`) straight from the actual balance-changing
transfers in the conversation — this is not a new fact, just one that was
previously discarded after `update_dsyn` ran. This module persists it
BEFORE any LLM sees it, identically for both sides, so neither player's
synapse can quietly disagree with what actually happened to the coins.

No LLM call, no interpretation of who was "right" — that judgment (was this
a broken deal, a gift, a loan) is still left entirely to the agents and
their personas. This only guarantees the raw numbers are the same fact for
both of them, the same way the public casino ledger already is for bets.
"""

from __future__ import annotations

import os
import common

MAX_ENTRIES_PER_PLAYER = 60   # oldest entries dropped first, like the raw ledger windows


def ledger_file(pid, base_dir):
    return os.path.join(base_dir, f"transfers_{pid}.json")


def load_entries(pid, base_dir) -> list[dict]:
    path = ledger_file(pid, base_dir)
    if os.path.exists(path):
        return common.read_json(path).get("entries", [])
    return []


def _save(pid, base_dir, entries: list[dict]):
    entries = entries[-MAX_ENTRIES_PER_PLAYER:]
    common.write_json(ledger_file(pid, base_dir), {"entries": entries})


def _upsert(pid, base_dir, partner, round_no, sent, received):
    """One entry per (partner, round_no), like FIX-16's public ledger —
    replaying a round replaces the old entry instead of duplicating it."""
    entries = load_entries(pid, base_dir)
    entries = [e for e in entries
               if not (e.get("partner") == partner and e.get("round_no") == round_no)]
    entries.append({
        "round_no": round_no,
        "partner": partner,
        "sent": sent,
        "received": received,
        "net": received - sent,
    })
    entries.sort(key=lambda e: e["round_no"])
    _save(pid, base_dir, entries)


def record_dialogue(base_dir, pid_a, pid_b, round_no, a_sent, b_sent):
    """Call once per finished dialogue, right where net_a/net_b are already
    computed in run_dialogue(). Writes the SAME numbers into both players'
    ledgers — there is exactly one true record of what moved, not one per
    player's interpretation of it."""
    if a_sent == 0 and b_sent == 0:
        return   # nothing moved, nothing to record
    _upsert(pid_a, base_dir, pid_b, round_no, sent=a_sent, received=b_sent)
    _upsert(pid_b, base_dir, pid_a, round_no, sent=b_sent, received=a_sent)


def format_recent(pid, base_dir, partner: str | None = None, window: int = 10) -> str:
    """Deterministic reminder block for plan_round — no LLM involved in
    producing it. If `partner` is given, restricts to that pairing (useful
    right before talking to them again); otherwise shows the most recent
    transfers with anyone."""
    entries = load_entries(pid, base_dir)
    if partner is not None:
        entries = [e for e in entries if e["partner"] == partner]
    if not entries:
        return ""
    tail = entries[-window:]
    lines = ["VERIFIED transfer history (actual coins moved, not what either "
             "side later claimed):"]
    for e in tail:
        lines.append(
            f"  r{e['round_no']}: with {e['partner']} — you sent {e['sent']}c, "
            f"received {e['received']}c (net {e['net']:+d}c)"
        )
    return "\n".join(lines) + "\n\n"
