"""
game_logger.py — writes all events to:
  logs/full_game.txt      (complete log for future analysis)
  logs/player_<id>.txt    (per-player log)
"""

import os
from datetime import datetime


class GameLogger:
    def __init__(self, logs_dir: str = "./logs"):
        os.makedirs(logs_dir, exist_ok=True)
        self.logs_dir = logs_dir
        self._full = open(os.path.join(logs_dir, "full_game.txt"), "a", encoding="utf-8")
        self._players: dict[str, object] = {}

    def _player_file(self, pid):
        if pid not in self._players:
            path = os.path.join(self.logs_dir, f"player_{pid}.txt")
            self._players[pid] = open(path, "a", encoding="utf-8")
        return self._players[pid]

    def _stamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def write(self, pid: str, msg: str):
        line = f"[{self._stamp()}][{pid}] {msg}\n"
        self._full.write(line)
        self._full.flush()
        pf = self._player_file(pid)
        pf.write(line)
        pf.flush()
        print(line, end="")

    def write_global(self, msg: str):
        line = f"[{self._stamp()}][GAME] {msg}\n"
        self._full.write(line)
        self._full.flush()
        print(line, end="")

    def write_dialogue(self, round_no: int, pid_a: str, pid_b: str, turn: int,
                       speaker: str, message: str, transfer: int = 0, to: str = None):
        transfer_note = f" [→ {transfer} coins to {to}]" if transfer > 0 else ""
        line = (f"[{self._stamp()}][DLG r{round_no:03d} {pid_a}↔{pid_b} "
                f"turn{turn}] {speaker}: {message}{transfer_note}\n")
        self._full.write(line)
        self._full.flush()
        for pid in (pid_a, pid_b):
            pf = self._player_file(pid)
            pf.write(line)
            pf.flush()
        print(line, end="")

    def write_round_header(self, round_no: int, total: int):
        sep = "=" * 60
        line = f"\n{sep}\n[{self._stamp()}][GAME] ROUND {round_no}/{total}\n{sep}\n"
        self._full.write(line)
        self._full.flush()
        print(line, end="")

    def write_balances(self, balances: dict):
        parts = "  ".join(f"{pid}={bal}" for pid, bal in balances.items())
        line = f"[{self._stamp()}][BALANCES] {parts}\n"
        self._full.write(line)
        self._full.flush()
        print(line, end="")

    def close(self):
        self._full.close()
        for f in self._players.values():
            f.close()
