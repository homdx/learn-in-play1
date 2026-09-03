"""
croupier_v2.py — same as original croupier but with logging and returns (winning_number, results).
"""

import glob
import logging
import os
import re

import common

logger = logging.getLogger(__name__)


def collect_bets(base_dir):
    bets = []
    pattern = os.path.join(base_dir, "bet_*.json")
    for path in sorted(glob.glob(pattern)):
        m = re.match(r"bet_(.+)\.json$", os.path.basename(path))
        if not m:
            continue
        pid = m.group(1)
        try:
            bet = common.read_json(path)
        except (OSError, ValueError) as e:
            logger.warning("Skipping corrupted/unreadable bet file %s: %s", path, e)
            continue
        bets.append((pid, bet, path))
    return bets


def run_round(base_dir, winning_number=None, logger=None):
    bets = collect_bets(base_dir)
    if not bets:
        if logger:
            logger.write_global("No bets — round skipped.")
        return None, []

    # BET-1: перед спином повторяем (ротируем) в лог ставку каждого
    # игрока — так итоговый лог раунда содержит сводку ставок сразу
    # перед спином, а не только в момент их размещения диалоговой фазой.
    for pid, bet, _bet_path in bets:
        recap_msg = f"bet recap before spin: {common.describe_bets(bet)}"
        if logger:
            logger.write(pid, recap_msg)
        else:
            print(f"[{pid}] {recap_msg}")

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
        per_bet = []
        try:
            common.validate_bets(bet)
            win, payout, per_bet = common.evaluate_bets(bet, winning_number)
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
            "bets": per_bet,
            "win": win,
            "payout": payout,
        }
        common.write_json(common.result_file(pid, base_dir), result)
        os.remove(bet_path)

        status = f"WON payout={payout}" if win else "lost"
        msg2 = f"{common.describe_bets(bet)} → {status}"
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
