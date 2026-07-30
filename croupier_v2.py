"""
croupier_v2.py — same as original croupier but with logging and returns (winning_number, results).
"""

import glob
import os
import re

import common


def collect_bets(base_dir):
    bets = []
    pattern = os.path.join(base_dir, "bet_*.json")
    for path in sorted(glob.glob(pattern)):
        m = re.match(r"bet_(.+)\.json$", os.path.basename(path))
        if not m:
            continue
        pid = m.group(1)
        bet = common.read_json(path)
        bets.append((pid, bet, path))
    return bets


def run_round(base_dir, winning_number=None, logger=None):
    bets = collect_bets(base_dir)
    if not bets:
        if logger:
            logger.write_global("No bets — round skipped.")
        return None, []

    if winning_number is None:
        winning_number = common.spin_wheel()

    color = "green (zero)"
    if winning_number in common.RED_NUMBERS:
        color = "red"
    elif winning_number in common.BLACK_NUMBERS:
        color = "black"

    msg = f"Wheel spins… number={winning_number} ({color}). Processing {len(bets)} bet(s)."
    if logger:
        logger.write_global(msg)
    else:
        print(msg)

    results = []
    for pid, bet, bet_path in bets:
        try:
            common.validate_bet(bet)
            win, payout = common.evaluate_bet(bet, winning_number)
        except ValueError as e:
            win, payout = False, 0
            msg = f"Invalid bet ({e}), no payout."
            if logger:
                logger.write(pid, msg)
            else:
                print(f"[{pid}] {msg}")

        result = {
            "player_id": pid,
            "winning_number": winning_number,
            "bet": bet,
            "win": win,
            "payout": payout,
        }
        common.write_json(common.result_file(pid, base_dir), result)
        os.remove(bet_path)

        status = f"WON payout={payout}" if win else "lost"
        msg2 = f"bet={bet['type']} amount={bet['amount']} → {status}"
        if logger:
            logger.write(pid, msg2)
        else:
            print(f"[{pid}] {msg2}")

        results.append(result)

    if logger:
        logger.write_global("Round complete. Results written.")
    return winning_number, results


def print_balances(base_dir, players):
    for pid in players:
        path = common.balance_file(pid, base_dir)
        if os.path.exists(path):
            bal = common.read_json(path)["balance"]
            print(f"  {pid}: {bal}")
