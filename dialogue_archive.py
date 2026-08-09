"""
dialogue_archive.py — читалка журнала диалогов (VERIFY-1).

Движок уже пишет каждый диалог целиком: run_game_v2.py сохраняет
``dlg_r{NNN}_{pid_a}_{pid_b}.json`` со списком ``conversation``. До сих пор
эти файлы были write-only: их никто не открывал. Модуль превращает их в
источник УЛИК.

Зачем это нужно. Роли black_liar/white_liar строятся на утверждениях о
третьем игроке, которого нет в комнате: «player4 обещал мне red», «я дал
player3 дюжину в раунде 3». Проверить такое утверждение по публичному
журналу нельзя — там ставки, а не разговоры. Собеседнику остаётся верить
на слово, и ложь про третьих лиц не стоит вруну ничего.

Здесь появляется третий вариант между «верю» и «не верю»: посмотреть
запись. Модуль отвечает на два вопроса, и оба — механически, без LLM:

  1. Был ли вообще разговор между X и Y в раунде N? Если файла нет,
     утверждение «он мне сказал» опровергнуто целиком и бесплатно. Это
     ловит самый частый тип выдумки — ссылку на встречу, которой не было.
  2. Если разговор был — что именно X там говорил? Реальные реплики
     отдаются дословно, и дальше стороны спорят уже о смысле сказанного,
     а не о самом факте.

Модуль намеренно НИЧЕГО не решает про правдивость. Он возвращает
статус (``no_record`` / ``no_lines`` / ``found``) и настоящий текст.
Вердикт по существу — дело того, кто читает: собеседника или арбитра.
Разделение важное: ``no_record`` — это факт файловой системы, его нельзя
оспорить, а «смысл сказанного не совпал» — это уже мнение.

Про кого утверждение — модуль тоже не угадывает. Имя игрока и номер
раунда приходят снаружи структурным полем ``claim_about``, которое
заполняет сам говорящий (см. roles.py). Вытаскивать «о ком речь» из
прозы регуляркой — источник ошибок и лишний вызов модели там, где
достаточно попросить заполнить поле.
"""

from __future__ import annotations

import json
import os
import re

# dlg_r006_player1_player4.json
_DLG_RE = re.compile(r"^dlg_r(\d{3,})_(.+?)_(.+?)\.json$")

# Сколько реплик одного игрока отдавать как улику. Ограничение не
# косметическое: улика уходит в промпт, а промпт на 30b стоит минуты.
MAX_EVIDENCE_LINES = 4
# Длинную реплику режем — для сверки утверждения хватает начала, а
# целиком она вытесняет из контекста чек-лист.
MAX_EVIDENCE_CHARS = 400

STATUS_NO_RECORD = "no_record"   # файла нет: разговора не было
STATUS_NO_LINES = "no_lines"     # файл есть, но игрок в нём не говорил
STATUS_FOUND = "found"           # есть дословные реплики


def _norm(pid) -> str:
    return str(pid or "").strip().lower()


def iter_dialogue_files(table_dir: str):
    """(round_no, pid_a, pid_b, path) по всем dlg-файлам стола.

    Имена игроков берутся ИЗ СОДЕРЖИМОГО файла, а имя файла даёт только
    номер раунда. Соблазн разобрать имя целиком велик — оно выглядит
    самоописательным, — но разделитель '_' встречается и внутри id:
    ``dlg_r002_player_1_player_2.json`` любой регуляркой делится не там,
    где нужно, и пара молча теряется. Номер раунда однозначен (цифры
    сразу после 'r'), поэтому он остаётся за именем.

    Битый json пропускается: разговор, который нельзя прочитать, уликой
    служить не может, но и ронять из-за него раунд незачем.
    """
    try:
        names = os.listdir(table_dir)
    except OSError:
        return
    for name in sorted(names):
        m = _DLG_RE.match(name)
        if not m:
            continue
        try:
            round_no = int(m.group(1))
        except ValueError:
            continue
        path = os.path.join(table_dir, name)
        data = load_conversation(path)
        if data is None:
            continue
        pid_a, pid_b = data.get("pid_a"), data.get("pid_b")
        if not pid_a or not pid_b:
            continue
        yield round_no, str(pid_a), str(pid_b), path


def load_conversation(path: str) -> dict | None:
    """Читает один dlg-файл. Возвращает None на битом/пустом.

    Битый файл — не повод падать посреди раунда: улики просто не будет,
    и собеседник останется при своём мнении, как раньше.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("conversation"), list):
        return None
    return data


def rounds_together(table_dir: str, pid_x: str, pid_y: str) -> list[int]:
    """Раунды, в которых X и Y действительно разговаривали.

    Порядок имён в имени файла произвольный (инициатором мог быть любой),
    поэтому сравнение — по множеству.
    """
    want = {_norm(pid_x), _norm(pid_y)}
    out = []
    for round_no, a, b, _path in iter_dialogue_files(table_dir):
        if {_norm(a), _norm(b)} == want:
            out.append(round_no)
    return sorted(set(out))


def _lines_by(data: dict, speaker: str) -> list[str]:
    out = []
    for turn in data.get("conversation", []):
        if not isinstance(turn, dict):
            continue
        if _norm(turn.get("from")) != _norm(speaker):
            continue
        msg = str(turn.get("message", "")).strip()
        if not msg or msg == "(…)":
            continue
        if len(msg) > MAX_EVIDENCE_CHARS:
            msg = msg[:MAX_EVIDENCE_CHARS].rstrip() + " …"
        out.append(msg)
    return out


def lookup(table_dir: str, listener: str, speaker: str,
           round_no: int | None = None) -> dict:
    """
    Поднять из архива, что ``speaker`` говорил игроку ``listener``.

    round_no=None — искать по всем раундам (утверждение без даты).

    Возвращает словарь:
        status    — no_record | no_lines | found
        rounds    — раунды, где эти двое разговаривали
        lines     — [(round_no, текст)], дословно, не более MAX_EVIDENCE_LINES
    """
    rounds = rounds_together(table_dir, listener, speaker)
    if not rounds:
        return {"status": STATUS_NO_RECORD, "rounds": [], "lines": []}
    if round_no is not None and int(round_no) not in rounds:
        # Разговоры были, но не в заявленном раунде — для утверждения
        # «он сказал это в раунде 5» это такое же опровержение.
        return {"status": STATUS_NO_RECORD, "rounds": rounds, "lines": []}

    targets = [int(round_no)] if round_no is not None else rounds
    want = {_norm(listener), _norm(speaker)}
    lines: list[tuple[int, str]] = []
    for r, a, b, path in iter_dialogue_files(table_dir):
        if r not in targets or {_norm(a), _norm(b)} != want:
            continue
        data = load_conversation(path)
        if data is None:
            continue
        for msg in _lines_by(data, speaker):
            lines.append((r, msg))

    if not lines:
        return {"status": STATUS_NO_LINES, "rounds": rounds, "lines": []}
    # Свежие реплики информативнее старых: режем с начала, а не с конца.
    lines = lines[-MAX_EVIDENCE_LINES:]
    return {"status": STATUS_FOUND, "rounds": rounds, "lines": lines}


# ── блоки для промпта ────────────────────────────────────────────────────

def full_history(table_dir: str, pid_x: str, pid_y: str,
                 max_chars: int = MAX_EVIDENCE_CHARS) -> list[dict]:
    """
    QUOTE-1: ВСЯ переписка pid_x↔pid_y — реплики ОБЕИХ сторон, по ВСЕМ
    раундам, в хронологическом порядке. В отличие от lookup() (который
    берёт слова только ОДНОЙ стороны, для проверки чужого утверждения о
    третьем лице), это нужно для собственной рефлексии игрока над своими
    же прошлыми разговорами с конкретным партнёром — там важен весь обмен
    репликами, а не только то, что сказал говорящий.

    Возвращает [{"round_no": int, "from": str, "message": str}, ...].
    Пустой список — если эти двое никогда не разговаривали (файл-уровневый
    факт, тот же принцип, что и STATUS_NO_RECORD в lookup()).
    """
    want = {_norm(pid_x), _norm(pid_y)}
    out = []
    for r, a, b, path in iter_dialogue_files(table_dir):
        if {_norm(a), _norm(b)} != want:
            continue
        data = load_conversation(path)
        if data is None:
            continue
        for turn in data.get("conversation", []):
            if not isinstance(turn, dict):
                continue
            msg = str(turn.get("message", "")).strip()
            if not msg or msg == "(…)":
                continue
            if len(msg) > max_chars:
                msg = msg[:max_chars].rstrip() + " …"
            out.append({"round_no": r, "from": turn.get("from"), "message": msg})
    out.sort(key=lambda e: e["round_no"])
    return out


def format_evidence(table_dir: str, listener: str, speaker: str,
                    round_no: int | None, claim: str = "") -> str:
    """
    Улика для промпта: что архив говорит про заявление ``claim``.

    Уходит ОБЕИМ сторонам одинаковым текстом. В этом весь смысл: врун
    видит ту же запись, что и его собеседник, и на следующем ходу
    выбирает — настаивать или признать ошибку. Разные версии архива у
    сторон превратили бы проверку в ещё один повод спорить о фактах.
    """
    res = lookup(table_dir, listener, speaker, round_no)
    where = f" in round {round_no}" if round_no is not None else ""
    head = ("\n=== TABLE RECORD (pulled by the casino, not by either of you) ===\n"
            + (f"Claim under check: \"{claim}\"\n" if claim else ""))

    if res["status"] == STATUS_NO_RECORD:
        if res["rounds"]:
            body = (f"{listener} and {speaker} did talk — in round(s) "
                    f"{', '.join(map(str, res['rounds']))} — but NOT{where}. "
                    f"The conversation being claimed did not happen.\n")
        else:
            body = (f"{listener} and {speaker} have NEVER had a conversation "
                    f"at this table. No such exchange exists.\n")
        body += ("This is a file-level fact, not an opinion: it cannot be "
                 "argued with.\n")
    elif res["status"] == STATUS_NO_LINES:
        body = (f"{speaker} and {listener} spoke{where}, but {speaker} said "
                f"nothing that was recorded. Nothing supports the claim.\n")
    else:
        body = f"What {speaker} actually said to {listener}, word for word:\n"
        for r, msg in res["lines"]:
            body += f"  [r{r}] {speaker}: {msg}\n"
        body += ("Judge for yourself whether this matches the claim. The "
                 "words are exact; what they meant is still arguable.\n")

    return head + body + "=== END TABLE RECORD ===\n\n"


def format_preparation(table_dir: str, pid: str, others: list[str],
                       round_no: int | None = None) -> str:
    """
    Блок подготовки для ролей: с кем из остальных у тебя РЕАЛЬНО были
    разговоры и в каких раундах.

    Это половина «гибкости», которую роль должна иметь. Зная, что запись
    существует, роль может сослаться на неё и выиграть проверку; зная,
    что записи нет, — понимает, что ссылаться на неё опасно, и либо
    выдумывает осознанно (принимая риск), либо говорит о текущем раунде,
    где архива ещё нет ни у кого.
    """
    rows = []
    for other in others:
        if _norm(other) == _norm(pid):
            continue
        rounds = [r for r in rounds_together(table_dir, pid, other)
                  if round_no is None or r < int(round_no)]
        rows.append((other, rounds))
    if not rows:
        return ""

    out = ("\n=== YOUR OWN RECORD (what the casino can pull up about you) ===\n"
           "Anything you say about a past conversation can be checked "
           "against this. If you are challenged, the real transcript is "
           "shown to BOTH of you.\n")
    for other, rounds in rows:
        if rounds:
            out += (f"  {other}: you really talked in round(s) "
                    f"{', '.join(map(str, rounds))} — quotable, will hold up.\n")
        else:
            out += (f"  {other}: NO conversation on record — any \"he told me\" "
                    f"about {other} collapses the moment it is checked.\n")
    out += "=== END YOUR OWN RECORD ===\n\n"
    return out
