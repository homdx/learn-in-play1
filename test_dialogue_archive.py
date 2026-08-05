"""Тесты dialogue_archive (VERIFY-1). Запуск: python3 test_dialogue_archive.py"""
import json
import os
import shutil
import tempfile

import dialogue_archive as da


def _dlg(table, r, a, b, turns):
    path = os.path.join(table, f"dlg_r{r:03d}_{a}_{b}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"round": r, "pid_a": a, "pid_b": b,
                   "conversation": [{"from": f, "message": m,
                                     "transfer": 0, "transfer_to": None}
                                    for f, m in turns],
                   "a_sent": 0, "b_sent": 0}, fh)
    return path


def setup():
    t = tempfile.mkdtemp()
    _dlg(t, 3, "player1", "player4",
         [("player1", "what are you on?"), ("player4", "black, 10 coins")])
    _dlg(t, 5, "player4", "player1",
         [("player4", "buy my call for 10c"), ("player1", "no")])
    _dlg(t, 5, "player2", "player3", [("player2", "player4 owes me")])
    return t


def run():
    t = setup()
    fails = []

    def check(name, cond):
        print(("  ok   " if cond else "  FAIL ") + name)
        if not cond:
            fails.append(name)

    # 1. индекс находит все файлы, порядок имён в файле не важен
    check("rounds_together symmetric",
          da.rounds_together(t, "player1", "player4") == [3, 5]
          and da.rounds_together(t, "player4", "player1") == [3, 5])

    # 2. пары, которая не разговаривала, в архиве нет
    check("never met -> no_record",
          da.lookup(t, "player1", "player5")["status"] == da.STATUS_NO_RECORD)

    # 3. разговор был, но не в том раунде — тоже опровержение
    r = da.lookup(t, "player1", "player4", round_no=4)
    check("wrong round -> no_record", r["status"] == da.STATUS_NO_RECORD
          and r["rounds"] == [3, 5])

    # 4. дословная реплика поднимается
    r = da.lookup(t, "player1", "player4", round_no=3)
    check("found exact line", r["status"] == da.STATUS_FOUND
          and r["lines"] == [(3, "black, 10 coins")])

    # 5. КОНТРФАКТ: если бы фильтр по speaker не работал, сюда попала бы
    #    реплика player1 — проверяем, что чужих реплик нет
    r = da.lookup(t, "player1", "player4")
    check("speaker filter excludes listener",
          all("what are you on?" not in m for _, m in r["lines"]))

    # 6. молчавший игрок: файл есть, реплик нет
    _dlg(t, 6, "player1", "player3", [("player1", "hi")])
    check("silent speaker -> no_lines",
          da.lookup(t, "player1", "player3", 6)["status"] == da.STATUS_NO_LINES)

    # 7. заглушка "(…)" уликой не является
    _dlg(t, 7, "player1", "player3", [("player3", "(…)")])
    check("stub turn is not evidence",
          da.lookup(t, "player1", "player3", 7)["status"] == da.STATUS_NO_LINES)

    # 8. id с подчёркиванием не разъезжается по разделителю
    t2 = tempfile.mkdtemp()
    _dlg(t2, 2, "player_1", "player_2", [("player_1", "deal")])
    check("underscored ids survive parsing",
          da.rounds_together(t2, "player_1", "player_2") == [2])

    # 9. битый файл не роняет и не считается разговором
    with open(os.path.join(t2, "dlg_r004_player_1_player_2.json"), "w") as fh:
        fh.write("{not json")
    check("corrupt file ignored",
          da.rounds_together(t2, "player_1", "player_2") == [2])

    # 10. пустой/несуществующий стол
    check("missing dir is empty",
          da.rounds_together("/nonexistent-table", "a", "b") == [])

    # 11. текст улики называет статус явно
    txt = da.format_evidence(t, "player1", "player5", None, claim="he promised red")
    check("no_record wording", "NEVER had a conversation" in txt)
    txt = da.format_evidence(t, "player1", "player4", 3)
    check("found wording quotes line", "black, 10 coins" in txt)

    # 12. подготовка разделяет "есть запись" и "записи нет"
    prep = da.format_preparation(t, "player1", ["player4", "player5"], round_no=6)
    check("prep marks quotable partner", "round(s) 3, 5" in prep)
    check("prep warns on missing record", "NO conversation on record" in prep)

    # 13. КОНТРФАКТ: подготовка не показывает раунды >= текущего
    prep5 = da.format_preparation(t, "player1", ["player4"], round_no=5)
    check("prep excludes current round", "3" in prep5 and "5" not in prep5.split("player4:")[1].split("\n")[0])

    # 14. лимит улик
    long_turns = [("player4", f"line {i}") for i in range(10)]
    t3 = tempfile.mkdtemp()
    _dlg(t3, 1, "player1", "player4", long_turns)
    check("evidence capped",
          len(da.lookup(t3, "player1", "player4", 1)["lines"]) == da.MAX_EVIDENCE_LINES)

    for d in (t, t2, t3):
        shutil.rmtree(d, ignore_errors=True)

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(run())
