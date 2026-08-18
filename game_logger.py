"""
game_logger.py — writes all events to:
  logs/full_game.txt      (complete log for future analysis)
  logs/player_<id>.txt    (per-player log)
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class GameLogger:
    def __init__(self, logs_dir: str = "./logs"):
        os.makedirs(logs_dir, exist_ok=True)
        self.logs_dir = logs_dir
        self._players: dict[str, object] = {}
        try:
            self._full = open(os.path.join(logs_dir, "full_game.txt"), "a", encoding="utf-8")
        except OSError as e:
            logger.warning("Could not open full_game.txt for writing: %s", e)
            self._full = None

    def _player_file(self, pid):
        if pid not in self._players:
            path = os.path.join(self.logs_dir, f"player_{pid}.txt")
            try:
                self._players[pid] = open(path, "a", encoding="utf-8")
            except OSError as e:
                logger.warning("Could not open log file for player %s: %s", pid, e)
                self._players[pid] = None
        return self._players[pid]

    def _stamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _safe_write(self, f, line, label="log"):
        """Write+flush to a file handle, swallowing OSError (disk full,
        permission revoked mid-run, etc.) so a logging failure never
        crashes the game loop."""
        if f is None:
            return
        try:
            f.write(line)
            f.flush()
        except OSError as e:
            logger.warning("Failed writing to %s: %s", label, e)

    def write(self, pid: str, msg: str):
        line = f"[{self._stamp()}][{pid}] {msg}\n"
        self._safe_write(self._full, line, "full_game.txt")
        pf = self._player_file(pid)
        self._safe_write(pf, line, f"player_{pid}.txt")
        print(line, end="")

    def write_global(self, msg: str):
        line = f"[{self._stamp()}][GAME] {msg}\n"
        self._safe_write(self._full, line, "full_game.txt")
        print(line, end="")

    def write_dialogue(self, round_no: int, pid_a: str, pid_b: str, turn: int,
                       speaker: str, message: str, transfer: int = 0, to: str = None):
        transfer_note = f" [→ {transfer} coins to {to}]" if transfer > 0 else ""
        line = (f"[{self._stamp()}][DLG r{round_no:03d} {pid_a}↔{pid_b} "
                f"turn{turn}] {speaker}: {message}{transfer_note}\n")
        self._safe_write(self._full, line, "full_game.txt")
        for pid in (pid_a, pid_b):
            pf = self._player_file(pid)
            self._safe_write(pf, line, f"player_{pid}.txt")
        print(line, end="")

    def write_round_header(self, round_no: int, total: int):
        sep = "=" * 60
        line = f"\n{sep}\n[{self._stamp()}][GAME] ROUND {round_no}/{total}\n{sep}\n"
        self._safe_write(self._full, line, "full_game.txt")
        print(line, end="")

    def write_balances(self, balances: dict):
        parts = "  ".join(f"{pid}={bal}" for pid, bal in balances.items())
        line = f"[{self._stamp()}][BALANCES] {parts}\n"
        self._safe_write(self._full, line, "full_game.txt")
        print(line, end="")

    def close(self):
        if self._full is not None:
            try:
                self._full.close()
            except OSError as e:
                logger.warning("Error closing full_game.txt: %s", e)
        for pid, f in self._players.items():
            if f is None:
                continue
            try:
                f.close()
            except OSError as e:
                logger.warning("Error closing log file for player %s: %s", pid, e)
