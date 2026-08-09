"""
agent_v2.py — autonomous LLM player with:
  * Abstract English prompt (player can rewrite it each round)
  * BETTING SYNAPSE  (notes_<id>.json)  — casino strategy, auto-compressed
  * DIALOGUE SYNAPSE (dsyn_<id>.json)   — per-player reputation map + recent raw turns
  * Auto-compression: when synapse text exceeds MAX_SYNAPSE_CHARS the LLM
    compresses it into a shorter summary, keeping the key facts
  * Player-to-player dialogue (max 2 partners / max 4 turns each)
  * Coin transfers during dialogues
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

import common
import open_bets
import promise_ledger
import roles
import speech_cost
import transfer_ledger
import llm_pool
from llm_client import LLMClient, LLMUnavailable

# ── thresholds (DEFAULTS — переопределяются секциями [memory]/[tokens]) ───
#
# FIX-11: раньше это были жёсткие константы, рассчитанные на куда меньший
# контекст. При num_ctx=8192 замер показал пик использования 31%: агенты
# сжимали собственную память в момент, когда свободны были две трети окна.
# Синапса — это и есть всё, что агент накапливает между раундами, поэтому
# порог 1200 символов работал прямо против развития агентов.
MAX_SYNAPSE_CHARS    = 4000  # compress betting synapse above this
MAX_DSYN_CHARS       = 6000  # compress dialogue synapse above this
MAX_RAW_INTERACTIONS = 12    # keep last N raw interaction entries before compressing

# сколько показывать в промпте (тоже упиралось в тесный контекст)
DEF_LEDGER_WINDOW_MOVE = 40  # строк публичного журнала при выборе собеседника
DEF_LEDGER_WINDOW_BET  = 60  # строк публичного журнала при выборе ставки
DEF_DSYN_RECENT        = 8   # последних сырых взаимодействий в синапсе
DEF_DEALS_SHOWN        = 6   # успешных сделок на игрока в репутационной карте
DEF_FAILS_SHOWN        = 4   # сорванных сделок на игрока

# FIX-12: до этого персона была ЕДИНСТВЕННЫМ неограниченным компонентом
# промпта. Заметки резались по synapse_chars, диалоговая синапса — по
# dsyn_chars, журнал — по окну, история — по history_window, а персона не
# ограничивалась ничем и не сжималась никогда. При этом направление роста —
# по умолчанию вверх: модель переписывает персону, видя её же в собственном
# системном промпте, а на задаче "перепиши этот текст" LLM почти всегда
# расширяет. Каждая версия строилась на предыдущей, более длинной.
#
# Отказ был двойне тихий: ошибки нет, а Ollama режет промпт С НАЧАЛА —
# первым выпадал CORE_SYSTEM_PROMPT с правилами игры, и агент "забывал",
# что диалоги и платные услуги вообще существуют.
DEF_PERSONA_CHARS      = 2500  # порог сжатия персоны
DEF_CHECKLIST_CHARS    = 2000  # потолок чек-листа (FIX-19)
PERSONA_HARD_FACTOR    = 2     # жёсткий потолок = persona_chars * этого

# ── бюджеты токенов на каждый тип вызова LLM ──────────────────────────────
# FIX-11: были зашиты по месту вызова; в конфиге настраивался ТОЛЬКО bet,
# то есть один вызов из шести. И распределены наоборот: 600 доставалось
# ставке (ей нужно выдать три поля JSON), а диалогу — где генерируется весь
# содержательный текст — только 300.
DEF_TOKENS = {
    "bet":         250,
    "dialogue":    600,
    "next_move":   350,
    "reflect":     700,
    "update_dsyn": 350,
    "compress":    500,
    "checklist":   700,
}

# ─────────────────────────────────────────────────────── file helpers ──────

def notes_file(pid, base_dir):
    return os.path.join(base_dir, f"notes_{pid}.json")

def history_file(pid, base_dir):
    return os.path.join(base_dir, f"history_{pid}.json")

def dsyn_file(pid, base_dir):
    return os.path.join(base_dir, f"dsyn_{pid}.json")

def public_ledger_file(base_dir):
    """Общий журнал результатов ВСЕХ игроков — видимый каждому, в отличие
    от истории (history_<id>.json), которая у каждого только своя."""
    return os.path.join(base_dir, "public_results.json")

def load_public_ledger(base_dir):
    path = public_ledger_file(base_dir)
    if os.path.exists(path):
        return common.read_json(path).get("entries", [])
    return []

def append_public_ledger(base_dir, entry):
    """
    FIX-16: одна запись на пару (игрок, раунд). При переигрывании раунда
    старая запись ЗАМЕЩАЕТСЯ, а не дописывается рядом.

    Раньше падение сервера моделей посреди партии оставляло в журнале
    призраков: аварийные ставки в 1 монету за раунды 3 и 4, а после
    перезапуска рядом ложились настоящие записи тех же раундов. Журнал
    становился буквально противоречивым — по две записи на раунд, — и это
    не косметика: он основа всей проверяемости. Агент, пойманный на вранье,
    получал законное основание сослаться на «расхождение в журнале», и в
    реальном прогоне ровно это и произошло.
    """
    entries = load_public_ledger(base_dir)
    pid, rnd = entry.get("player_id"), entry.get("round_no")
    if pid is not None and rnd is not None:
        entries = [e for e in entries
                   if not (e.get("player_id") == pid and e.get("round_no") == rnd)]
    entries.append(entry)
    entries.sort(key=lambda e: (e.get("round_no") or 0, e.get("player_id") or ""))
    common.write_json(public_ledger_file(base_dir), {"entries": entries})

def checklist_file(pid, base_dir):
    return os.path.join(base_dir, f"checklist_{pid}.md")


# FIX-19: стартовый шаблон чек-листа.
#
# Это ПОДСКАЗКА, а не схема. Файл принадлежит агенту целиком: он вправе
# выбросить любой раздел, придумать свои и хранить что угодно — вплоть до
# формата, который никто из нас не предвидел. Ничего здесь не парсится
# кодом, ни один раздел не обязателен.
#
# Зачем он нужен. У агента уже есть три вида памяти, и все долгосрочные:
# персона (кто я), игровая синапса (как ставить), репутационная карта (кто
# чего стоит). Краткосрочной повестки не было НИ ОДНОЙ: «в этом раунде
# стребовать с player3 24 монеты», «я обещал player5 параметры», «перед
# займом проверить его P&L». Поэтому договорённости и растворялись между
# раундами — их негде было держать.
DEFAULT_CHECKLIST = """## Agenda this round
(nothing yet)

## Owed TO me
(nothing yet)

## I owe / I promised
(nothing yet)

## Verify before paying
(nothing yet)

## Notes
(nothing yet)
"""


def load_checklist(pid, base_dir):
    return load_text(checklist_file(pid, base_dir), DEFAULT_CHECKLIST)


def save_checklist(pid, base_dir, text):
    save_text(checklist_file(pid, base_dir), text)


def checklist_history_file(pid, base_dir):
    return os.path.join(base_dir, f"checklist_history_{pid}.log")


def append_checklist_history(pid, base_dir, event: str,
                             prompt_system: str, prompt_user: str,
                             checklist_text: str):
    """
    Полная, только-дописываемая хронология чек-листа игрока: дата/время,
    что произошло (планирование раунда / обновление после диалога — и с
    кем), весь промпт, который ушёл модели, и итоговый текст чек-листа.

    Этот файл игрой никогда не читается и не перезаписывается — это
    внешний журнал для последующего разбора партии (FIX-19b: раньше
    checklist_<pid>.md хранил только ПОСЛЕДНЮЮ версию, и промежуточные
    состояния/промпты были невосстановимы после перезаписи).
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 80
    block = (
        f"{sep}\n"
        f"[{stamp}] {pid} | {event}\n"
        f"{'-' * 80}\n"
        f"PROMPT (system):\n{prompt_system}\n\n"
        f"PROMPT (user):\n{prompt_user}\n"
        f"{'-' * 80}\n"
        f"NEW CHECKLIST:\n{checklist_text}\n"
        f"{sep}\n\n"
    )
    with open(checklist_history_file(pid, base_dir), "a", encoding="utf-8") as f:
        f.write(block)


def append_checklist_history_failure(pid, base_dir, event: str, reason: str):
    """Короткая запись о неудачной попытке переписать чек-лист — старый
    текст сохраняется, но факт попытки и причина остаются в хронологии."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 80
    block = (
        f"{sep}\n"
        f"[{stamp}] {pid} | {event} — FAILED ({reason}), checklist unchanged\n"
        f"{sep}\n\n"
    )
    with open(checklist_history_file(pid, base_dir), "a", encoding="utf-8") as f:
        f.write(block)


def prompt_file(pid, base_dir):
    return os.path.join(base_dir, f"prompt_{pid}.txt")


def fieldnotes_file(pid, base_dir):
    return os.path.join(base_dir, f"fieldnotes_{pid}.md")


# ROLE-N: сколько символов полевых заметок держим. Осознанно мало: это
# рабочий блокнот приёмов, а не вторая персона. При переполнении вытесняется
# самая старая запись.
FIELDNOTES_CHARS = 2500

# ROLE-P: сколько наблюдений роль может добавить за один раунд. Одного было
# мало: заметка вида "метод простаивает" фиксирует диагноз, но не то, ЧТО
# именно попробовали, с кем и по какой цене — а именно этих подробностей и
# не хватало, чтобы в следующий раунд зайти иначе.
FIELDNOTES_PER_ROUND = 4


def load_fieldnotes(pid, base_dir) -> list[str]:
    """Полевые заметки роли — по одной строке на запись, старые первыми."""
    text = load_text(fieldnotes_file(pid, base_dir), "")
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def append_fieldnotes(pid, base_dir, notes, cap: int = FIELDNOTES_CHARS,
                      limit: int = FIELDNOTES_PER_ROUND) -> list[str]:
    """ROLE-P: дозапись НЕСКОЛЬКИХ наблюдений за раунд, по одному на строку."""
    if isinstance(notes, str):
        notes = [notes]
    kept = load_fieldnotes(pid, base_dir)
    for note in list(notes or [])[:limit]:
        kept = append_fieldnote(pid, base_dir, note, cap)
    return kept


def append_fieldnote(pid, base_dir, note: str, cap: int = FIELDNOTES_CHARS) -> list[str]:
    """
    ROLE-N: дозапись наблюдения о том, ЧТО СРАБОТАЛО и что нет.

    Только дозапись: прошлые строки не редактируются и не переписываются
    моделью. Это принципиально — переписывание превратило бы блокнот в
    очередную персону, а роль в этой игре заперта именно потому, что модель,
    редактируя текст роли, неизбежно его смягчает.

    Зачем вообще: в реальном прогоне Прокурор четыре раза подряд заходил с
    одинаково построенным обвинением и четыре раза получал один и тот же
    отказ ("приватные обещания не в реестре"). Нигде не было места, где
    осел бы вывод о МЕТОДЕ: чек-лист живёт один раунд, синапса хранит
    доверие к людям, а персона заперта. Теперь есть.

    Дубликаты не копим: повторное наблюдение всплывает в конец списка, а не
    добавляется второй строкой.
    """
    note = " ".join(str(note or "").split())
    if not note:
        return load_fieldnotes(pid, base_dir)
    notes = [n for n in load_fieldnotes(pid, base_dir) if n.lower() != note.lower()]
    notes.append(note)
    # вытесняем самые старые, пока не влезем в лимит
    while notes and len("\n".join(notes)) > cap:
        notes.pop(0)
    save_text(fieldnotes_file(pid, base_dir), "\n".join(notes))
    return notes


def load_text(path, default=""):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return default

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_notes(pid, base_dir):
    path = notes_file(pid, base_dir)
    if os.path.exists(path):
        # FIX-6: старые прогоны могли записать не-строку — приводим на чтении
        return _as_text(common.read_json(path).get("notes", ""))
    return ""

def save_notes(pid, base_dir, notes):
    common.write_json(notes_file(pid, base_dir), {"notes": notes})

def load_history(pid, base_dir):
    path = history_file(pid, base_dir)
    if os.path.exists(path):
        return common.read_json(path).get("rounds", [])
    return []

def append_history(pid, base_dir, entry):
    """FIX-16: то же для личной истории — одна запись на раунд."""
    rounds = load_history(pid, base_dir)
    rnd = entry.get("round_no")
    if rnd is not None:
        rounds = [r for r in rounds if r.get("round_no") != rnd]
    rounds.append(entry)
    rounds.sort(key=lambda r: r.get("round_no") or 0)
    common.write_json(history_file(pid, base_dir), {"rounds": rounds})

def load_balance(pid, base_dir, start_balance):
    path = common.balance_file(pid, base_dir)
    if os.path.exists(path):
        return common.read_json(path)["balance"]
    common.write_json(path, {"balance": start_balance})
    return start_balance

def save_balance(pid, base_dir, balance):
    common.write_json(common.balance_file(pid, base_dir), {"balance": balance})


# ─────────────────────────────────────────── dialogue synapse structure ───
#
# dsyn format:
# {
#   "reputation": {
#     "player2": {
#       "trust_score": 1-10,
#       "total_sent": N,
#       "total_received": N,
#       "net": N,                  # received - sent (positive = profitable)
#       "deals_done": ["sold strategy r2 for 5c", ...],
#       "deals_failed": ["promised loan r3 never came"],
#       "reputation_note": "reliable, sells cheap tips",
#       "future_intent": "offer loan next round",
#       "last_seen_round": 4
#     },
#     ...
#   },
#   "interactions": [   # last MAX_RAW_INTERACTIONS raw entries
#     {"round": N, "partner": "p2", "net_transfer": N, "summary": "...", "timestamp": "..."}
#   ],
#   "compressed_history": "compact text summary of older interactions"
# }

def _truncate_text(text: str, limit: int) -> str:
    """Обрезка по границе предложения/строки, а не посреди слова."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(". "), head.rfind(".\n"), head.rfind("\n"))
    # откатываемся к границе, только если не теряем больше трети текста
    if cut > limit * 0.66:
        return head[:cut + 1].rstrip()
    return head.rstrip()


def _as_text(value, fallback=""):
    """FIX-6: LLM возвращает то строку, то список, то dict. Приводим к
    строке, не роняя игру и не записывая на диск структуру, на которой
    потом упадёт len()."""
    if isinstance(value, str):
        return value
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _stamp_stale_notes(old_notes: str, last_entry, real_balance: int,
                       char_limit: int) -> str:
    """FIX-22: когда reflect_betting дважды не смог обновить синапсу, мы
    больше не отдаём старый текст как есть — он мог описывать другой баланс
    и другой исход ставки (см. player2 в разборе партии: синапса говорила
    "Balance: 115... WON", хотя реально было 80 и проигрыш). Приклеиваем
    сверху короткую, гарантированно верную фактическую справку, оставляя
    сам текст стратегии нетронутым внизу — под общим лимитом синапсы."""
    if last_entry is None:
        fact = f"[FACT-CHECK: balance is {real_balance}, no bet was placed last round.]"
    else:
        outcome = "WON" if last_entry.get("win") else "lost"
        fact = (
            f"[FACT-CHECK: balance is {real_balance} (not what the strategy text "
            f"below may say). Last bet {outcome}, payout={last_entry.get('payout')}.] "
        )
    stale = f"{fact}\n{old_notes or ''}"
    return stale[:char_limit]


def _empty_dsyn():
    return {"reputation": {}, "interactions": [], "compressed_history": ""}

def load_dsyn(pid, base_dir):
    path = dsyn_file(pid, base_dir)
    if os.path.exists(path):
        data = common.read_json(path)
        # migrate old format
        if "reputation" not in data:
            data = _empty_dsyn()
            data["interactions"] = common.read_json(path).get("interactions", [])
        return data
    return _empty_dsyn()

def save_dsyn(pid, base_dir, data):
    common.write_json(dsyn_file(pid, base_dir), data)


# ─────────────────────────────────────────────────────── formatting ────────

def _format_history(rounds, window):
    if not rounds:
        return "(no round history yet)"
    tail = rounds[-window:]
    lines = []
    for r in tail:
        bet = r["bet"]
        outcome = "WON" if r["win"] else "lost"
        bd = bet.get("numbers", bet.get("selection"))
        lines.append(
            f"  num={r['winning_number']} bet={bet['type']}({bd}) "
            f"amount={bet['amount']} → {outcome} payout={r['payout']} "
            f"balance_after={r['balance_after']}"
        )
    return "\n".join(lines)


def _format_public_ledger(entries: list[dict], window: int = 20, exclude_pid: str = None) -> str:
    """Компактная сводка последних `window` реальных исходов ставок ПО ВСЕМ
    игрокам (проверяемые факты — не то, что кто-то сказал в диалоге)."""
    if not entries:
        return "(no public results yet)"
    # FIX-11: раньше срез брался ДО фильтрации (`entries[-window:]`, затем
    # отброс своих записей), поэтому фактическое окно было меньше заявленного
    # и плавало: при 5 игроках window=10 давал ~8 строк, т.е. по 2 ставки на
    # оппонента — по такой выборке нельзя судить, выигрывает он или нет.
    # Фильтруем сначала, срезаем потом: окно означает то, что написано.
    visible = [e for e in entries
               if not (exclude_pid and e.get("player_id") == exclude_pid)]
    lines = []
    for e in visible[-window:]:
        bet = e.get("bet", {}) or {}
        bd = bet.get("numbers", bet.get("selection"))
        status = "WON" if e.get("win") else "lost"
        lines.append(
            f"  r{e.get('round_no', '?')}: {e.get('player_id')} bet {bet.get('type')}({bd}) "
            f"amount={bet.get('amount')} on number={e.get('winning_number')} → {status} "
            f"payout={e.get('payout', 0)}"
        )
    return "\n".join(lines) if lines else "(no public results yet)"


def _dialogue_running_totals(conversation, pid, partner_id) -> str:
    """
    Тот же приём, что и в _current_round_notice(): вместо того чтобы
    заставлять модель складывать в уме разрозненные пометки
    "[sent N coins to X]", разбросанные по репликам диалога, считаем сумму
    сами и кладём готовое число вплотную к транскрипту.

    Реальный случай, ради которого это добавлено: в одном диалоге игрок
    трижды "доплачивал" в уже растущий, по словам партнёра, общий пул
    (20 → 20 → 10 → 20, итого 70 монет), хотя все переводы были прямо в
    транскрипте — сложить их вручную из прозы модель не смогла.
    """
    my_sent = sum(t["transfer"] for t in conversation
                 if t.get("from") == pid and t.get("transfer_to") == partner_id)
    their_sent = sum(t["transfer"] for t in conversation
                     if t.get("from") == partner_id and t.get("transfer_to") == pid)
    if my_sent == 0 and their_sent == 0:
        return ""
    return (
        f"RUNNING TOTAL in THIS conversation so far: you have sent {partner_id} "
        f"{my_sent} coin(s); {partner_id} has sent you {their_sent} coin(s). "
        f"Check this before sending more — do not re-pay something you already "
        f"sent just because it was rephrased or re-summarised in the text above.\n"
    )


def _current_round_notice(round_no) -> str:
    """
    FIX-18: одна строка ВПЛОТНУЮ к данным вместо абзаца в правилах.

    Запрет уже был в CORE_SYSTEM_PROMPT ("один спин за раунд, крутит крупье,
    в самом конце раунда"), он был активен — и его проигнорировали: в
    реальном прогоне игрок, находясь в раунде 4, отчитался о результате
    спина раунда 5 и заплатил по нему 60 монет. Скорборд при этом показывал
    `last bet in r3`, то есть противоречие было прямо на экране у обоих.
    Поэтому здесь не ещё один абзац прозы, а конкретный факт про ЭТОТ раунд,
    поставленный вплотную к таблице, по которой его проверяют.
    """
    if round_no is None:
        return ""
    played = round_no - 1
    if played <= 0:
        # ROUND-1: пустой журнал — самый опасный случай, и прежний текст его
        # не покрывал. В реальном прогоне в ПЕРВОМ раунде игрок предъявил
        # долг за "раунд 4", трое собеседников приняли это как факт, и 135
        # монет сменили владельца по событию, которого не было. Формулировка
        # "нет результата для раунда N и позже" технически это запрещала, но
        # ничего не говорила о том, что сыграно НОЛЬ раундов и сослаться
        # физически не на что.
        return (
            f"CURRENT ROUND: {round_no}. ZERO rounds have been played so far. "
            f"The public ledger is EMPTY. There is no round 0, no earlier "
            f"session, no prior history of any kind — not for you, not for "
            f"anyone at this table. ANY reference to a past bet, a past "
            f"result, a past debt or a past promise is fabricated, no matter "
            f"how confidently it is stated or how often it is repeated. Do "
            f"not pay for one, do not settle one, and do not repeat one.\n"
        )
    return (
        f"CURRENT ROUND: {round_no}. Rounds actually played so far: 1..{played} "
        f"— and nothing else exists. The wheel has NOT been spun for round "
        f"{round_no} yet; it spins after all dialogue and all bets are in. "
        f"No result exists for round {round_no} or any later round, for you or "
        f"for anyone else. Any figure quoted as an outcome of round {round_no}+ "
        f"is invented, including by you. If a claim names a round above "
        f"{played}, it is fabricated — check the number before you answer it.\n"
    )


def _format_scoreboard(entries: list[dict], exclude_pid: str = None) -> str:
    """
    FIX-14: агрегат по ВСЕМУ публичному журналу, посчитанный в Python.

    Раньше агенту показывали только сырые строки журнала, и чтобы ответить
    «выигрывает этот игрок или проигрывает», он должен был отфильтровать их
    по игроку, просуммировать ставки, просуммировать выплаты и вычесть —
    и всё это без места на рассуждение (think=false, JSON-only ответ). При
    прежнем окне в 10 строк на пятерых он к тому же видел по 2 ставки на
    оппонента, чего не хватило бы и идеальному счётчику.

    Тот же приём уже применён к диалоговой синапсе: `net`, `total_sent` и
    `total_received` считает код, а не модель. Здесь — то же самое для
    казино. Новой информации это не добавляет: журнал и так публичен,
    убирается только арифметический барьер.

    Баланс сюда намеренно НЕ включён — он содержит итог приватных сделок,
    и его публикация выдала бы третьим лицам исход чужих диалогов.
    """
    if not entries:
        return "(no bets have been resolved yet)"

    agg: dict[str, dict] = {}
    for e in entries:
        pid = e.get("player_id")
        if not pid or (exclude_pid and pid == exclude_pid):
            continue
        a = agg.setdefault(pid, {"staked": 0, "won": 0, "bets": 0, "hits": 0,
                                 "last": None, "types": {}})
        bet = e.get("bet", {}) or {}
        a["staked"] += bet.get("amount", 0) or 0
        a["won"]    += e.get("payout", 0) or 0
        a["bets"]   += 1
        a["hits"]   += 1 if e.get("win") else 0
        a["last"]    = e.get("round_no", a["last"])
        t = bet.get("type", "?")
        a["types"][t] = a["types"].get(t, 0) + 1

    if not agg:
        return "(no bets have been resolved yet)"

    lines = []
    for pid in sorted(agg):
        a = agg[pid]
        pnl = a["won"] - a["staked"]
        fav = max(a["types"].items(), key=lambda kv: kv[1])[0] if a["types"] else "?"
        lines.append(
            f"  {pid}: casino P&L={pnl:+.2f}c over {a['bets']} bet(s) "
            f"(staked {a['staked']}, returned {a['won']}, won {a['hits']}/{a['bets']}), "
            f"mostly {fav}, last bet in r{a['last']}"
        )
    return "\n".join(lines)


def _format_dsyn_for_prompt(data: dict, recent: int = DEF_DSYN_RECENT,
                            deals: int = DEF_DEALS_SHOWN,
                            fails: int = DEF_FAILS_SHOWN) -> str:
    """
    Builds a compact, token-efficient summary of the dialogue synapse
    for injection into LLM prompts.
    """
    parts = []

    # 1. compressed history from older rounds
    if data.get("compressed_history"):
        parts.append(f"[Compressed older history]\n{data['compressed_history']}")

    # 2. per-player reputation table
    rep = data.get("reputation", {})
    if rep:
        lines = ["[Player reputation map]"]
        for pid, r in rep.items():
            lines.append(
                f"  {pid}: trust={r.get('trust_score',5)}/10 "
                f"net={r.get('net',0):+.2f}c "
                f"last_seen=r{r.get('last_seen_round','?')} "
                f"note='{r.get('reputation_note','')}' "
                f"intent='{r.get('future_intent','')}'"
            )
            done = r.get("deals_done", [])
            fail = r.get("deals_failed", [])
            if done:
                lines.append(f"    ✓ {'; '.join(done[-deals:])}")
            if fail:
                lines.append(f"    ✗ {'; '.join(fail[-fails:])}")
        parts.append("\n".join(lines))

    # 3. last few raw interactions
    raw = data.get("interactions", [])
    if raw:
        lines = ["[Recent interactions]"]
        for itx in raw[-recent:]:
            lines.append(
                f"  r{itx.get('round','?')} {itx.get('partner','?')}: "
                f"net={itx.get('net_transfer',0):+.2f}c — {itx.get('summary','')}"
            )
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "(no interactions with other players yet)"


# ─────────────────────────────────────────────────────── JSON safe parse ──

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

def _safe_json(text: str) -> dict:
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return json.loads(text)


# ───────────────────────────────────────── core (immutable) system prompt ─
#
# This part is HARD-CODED and ALWAYS prepended to every LLM call, regardless
# of what the player writes into its own editable persona/strategy prompt.
# It defines the game's goal and mechanics that must never be lost:
#   - the objective (grow capital)
#   - the three available levers (betting, paid services, negotiation)
#   - that dialogue with other players and paid services ARE part of the game
#   - the JSON-only output contract
# A player can invent any persona it wants on top of this, but it can never
# accidentally overwrite or "forget" these core rules, because they are not
# stored in the player-editable prompt file at all — they live in code.

CORE_SYSTEM_PROMPT = """\
=== CORE GAME RULES (fixed — you cannot change this part) ===
You are an AI player in a multi-round casino game against other AI players.
Your only goal: END THE GAME WITH MORE COINS THAN YOU STARTED WITH.

Every round you can do three things:
  1. BET at the roulette table.
  2. TALK to other players before betting.
  3. TRADE with them for coins — sell tips or strategy, lend at interest,
     broker deals, collect debts. Coin transfers between players are a normal
     part of this game, not an exception.

You keep two memories that persist all game: a BETTING SYNAPSE (your own
strategy notes) and a DIALOGUE SYNAPSE (what you think of each player). Use
them. You may adopt any persona you like; all three tools stay available.

--- FACTS YOU CANNOT ARGUE WITH ---
The wheel is European: 37 pockets, 0 to 36, no 00. The only bet types that
exist: straight, split, street, corner, sixline, dozen, column, even_money.
even_money means red, black, odd, even, low or high — "green" is not a bet.

ONE spin per round. The croupier spins it, at the very end, after all talk
and all bets. Nobody can run a test session, a backtest or extra spins.

Every bet and every outcome goes into a PUBLIC LEDGER you are shown each
round. It cannot be forged, and it cuts both ways:
  - Someone describes a past result? If it is not in the ledger, it did not
    happen. Paying for unverifiable data is the fastest way to lose coins.
  - Your own bets are in there too. Promise one bet and place another, and
    everyone sees it.
Predictions about the future are opinions and fine to trade. Claims about
what already happened are facts, and all of them are checkable.
--- END FACTS ---

--- THE FOUR THINGS PLAYERS GET WRONG (re-read before every answer) ---
1. TRANSFER MEANS YOU PAY. `transfer` moves coins OUT of your balance INTO
   your partner's. If you are ASKING to be paid, DEMANDING payment, or saying
   "I'll take", "send me", "you owe me" — set transfer to 0. Only set it
   above 0 when you have decided to hand your own coins over.
2. DONE IS A FLAG, NOT A SENTENCE. Writing "let's wrap up" while leaving
   done=false keeps the conversation going and you will repeat yourself.
   When you are finished, set done=true.
3. DO NOT REPEAT YOURSELF. Never send a message that says the same thing as
   one you already sent in this dialogue. If you have nothing new, set
   done=true instead. Every line costs coins.
4. BET WHAT YOU SAID. If you told someone your bet, place that exact bet.
--- END ---

Respond with valid JSON only, matching the requested schema. No markdown, no
text outside the JSON object.
=== END CORE GAME RULES ===
"""


# ─────────────────────────────────── default editable persona/strategy ────
# This is the ONLY part stored in prompt_<id>.txt and the ONLY part a player
# can rewrite via reflect_betting()'s "new_persona". It sits on top
# of CORE_SYSTEM_PROMPT, never replaces it.

DEFAULT_PERSONA_PROMPT = """\
You have no fixed persona yet. Pick a starting identity and strategy that
feels promising, and refine it round by round based on what actually works.
"""


# ─────────────────────────────────────────────────────── PlayerAgent ──────

def format_transfer_note(turn: dict, viewer_pid: str) -> str:
    """
    TVIS-1: пометка о переводе в транскрипте — для ОБЕИХ сторон.

    До этого нулевой перевод не отображался никак: аннотация рисовалась
    только при transfer > 0, поэтому реплика "отправляю 20 монет", по
    которой не ушло ничего, выглядела как обычная фраза. Отправитель
    считал, что заплатил, получатель — что ему заплатили, и оба заносили
    несостоявшуюся сделку в синапсу как состоявшуюся.

    Формулировки разные намеренно. Отправителю нужно ПОЧЕМУ — иначе он
    повторит ту же попытку меньшей суммой и снова не поймёт, что мешает.
    Получателю нужен сам факт: ему обещали и не отдали, а причина — не
    его забота и не повод считать партнёра честным.
    """
    sent = turn.get("transfer", 0) or 0
    frm = turn.get("from")
    parts = []
    if sent > 0:
        parts.append(f" [передано {sent} монет → {turn.get('transfer_to')}]")

    att = turn.get("attempt")
    if att and att.get("requested", 0) > 0:
        req = att["requested"]
        got = att.get("delivered", 0)
        detail = att.get("detail") or ""
        if frm == viewer_pid:
            head = (f"НЕ ПРОШЛО: ты попытался отправить {req}, "
                    f"дошло {got}")
            parts.append(f" [{head}{' — ' + detail if detail else ''}. "
                         f"Деньги остались у тебя; сделка НЕ оплачена]")
        else:
            parts.append(f" [{frm} пытался отправить {req}, реально дошло "
                         f"{got} — обещанное НЕ оплачено]")
    return "".join(parts)


class PlayerAgent:
    def __init__(self, player_id, base_dir, cfg, logger=None,
                 roles_assignment=None, tariff=None):
        self.player_id = player_id
        self.base_dir = base_dir
        self.cfg = cfg
        self.log = logger

        # POOL-1: один сервер → обычный LLMClient, как раньше. Несколько
        # (ключ pool в [api]) → обёртка над ОБЩИМ на процесс пулом.
        self.client = llm_pool.shared_client(
            cfg, factory=lambda: LLMClient.from_config(cfg))
        # RETRY-1: повтор должен быть виден в логе игрока, иначе разница
        # между "модель молчит" и "модель ответила со второй попытки"
        # теряется при разборе прогона.
        self.client.on_retry = self._log

        self.temperature    = cfg.getfloat("player", "temperature",    fallback=0.75)
        self.history_window = cfg.getint("player",   "history_window", fallback=10)

        # FIX-11: бюджеты токенов — секция [tokens], с откатом на старое
        # поведение ([player] max_tokens для ставки) если секции нет.
        def _tok(name):
            return cfg.getint("tokens", name, fallback=DEF_TOKENS[name])
        self.tok_bet         = cfg.getint("tokens", "bet",
                                          fallback=cfg.getint("player", "max_tokens",
                                                              fallback=DEF_TOKENS["bet"]))
        self.tok_dialogue    = _tok("dialogue")
        self.tok_next_move   = _tok("next_move")
        self.tok_reflect     = _tok("reflect")
        self.tok_update_dsyn = _tok("update_dsyn")
        self.tok_compress    = _tok("compress")
        self.tok_checklist   = _tok("checklist")

        # FIX-11: объём памяти агента — секция [memory]
        def _mem(name, default):
            return cfg.getint("memory", name, fallback=default)
        self.synapse_chars     = _mem("synapse_chars",      MAX_SYNAPSE_CHARS)
        self.dsyn_chars        = _mem("dsyn_chars",         MAX_DSYN_CHARS)
        self.raw_interactions  = _mem("raw_interactions",   MAX_RAW_INTERACTIONS)
        self.ledger_win_move   = _mem("ledger_window_move", DEF_LEDGER_WINDOW_MOVE)
        self.ledger_win_bet    = _mem("ledger_window_bet",  DEF_LEDGER_WINDOW_BET)
        self.dsyn_recent       = _mem("dsyn_recent_shown",  DEF_DSYN_RECENT)
        self.deals_shown       = _mem("deals_shown",        DEF_DEALS_SHOWN)
        self.fails_shown       = _mem("fails_shown",        DEF_FAILS_SHOWN)
        self.persona_chars     = _mem("persona_chars",      DEF_PERSONA_CHARS)
        self.checklist_chars   = _mem("checklist_chars",    DEF_CHECKLIST_CHARS)
        # FIX-19: при выключенном чек-листе он не должен ни обновляться, НИ
        # попадать в промпты — иначе выключение экономило бы вызовы, но не
        # контекст, и агент получал бы пустой шаблон вместо ничего.
        self.use_checklist = cfg.getboolean("game", "use_checklist", fallback=True)

        self.max_bet_fraction = cfg.getfloat("game", "max_bet_fraction", fallback=0.4)
        self.start_balance    = cfg.getint("game",   "start_balance",    fallback=100)

        self.balance = load_balance(player_id, base_dir, self.start_balance)

        # ROLE-1: назначаемая роль из [roles]. Разбор конфига идемпотентен и
        # дешёвый, поэтому агент может сделать его сам — это позволяет
        # создавать PlayerAgent вне run_game_v2 (тесты, отдельные утилиты) без
        # обязательной передачи готового RoleAssignment.
        if roles_assignment is not None:
            self.roles = roles_assignment
        else:
            players = [p.strip() for p in
                       cfg.get("game", "players", fallback="").split(",") if p.strip()]
            if player_id not in players:
                players = players + [player_id]
            self.roles = roles.parse_roles(cfg, players)
        self.role           = self.roles.role_of(player_id)
        self.persona_locked = self.roles.is_locked(player_id)

        # TALK-1: тариф на речь. Разбирается тем же способом, что и роли —
        # агент может создать его сам, чтобы PlayerAgent оставался
        # конструируемым вне run_game_v2 (тесты, утилиты).
        self.tariff = (tariff if tariff is not None
                       else speech_cost.parse_tariff(cfg))
        # сколько монет игрок уже отдал казино за слова в ТЕКУЩЕМ диалоге;
        # обнуляется в начале каждого диалога из run_dialogue
        self.speech_spent_this_dialogue = 0
        # ROLE-P: дефолт на случай, если агент создан не через run_game_v2
        # (тесты, прямой вызов) — там, где флаг реально решается по ролям
        # и конфигу, run_game_v2 перезаписывает его до первого разговора.
        self.speech_is_free = False

        pf = prompt_file(player_id, base_dir)
        seeded = roles.seed_prompt_file(pf, self.roles, player_id, save_text)
        if seeded:
            self._log(f"role={self.role} persona_locked={self.persona_locked}")
        elif not os.path.exists(pf):
            save_text(pf, DEFAULT_PERSONA_PROMPT)

    @property
    def persona_prompt(self):
        """The player-editable part only (persona/strategy)."""
        return load_text(prompt_file(self.player_id, self.base_dir), DEFAULT_PERSONA_PROMPT)

    @property
    def abstract_prompt(self):
        """
        Full system prompt sent to the LLM: immutable core game rules
        (hard-coded, always present) + whatever persona/strategy text the
        player has written for itself. The core section can never be
        dropped, edited, or overwritten by the player — only this
        persona/strategy section is stored in prompt_<id>.txt and subject
        to rewriting via reflect_betting().
        """
        # FIX-12: жёсткий потолок как последняя линия обороны. Сжатие персоны
        # происходит в Фазе 0 (reflect_betting), но до неё — первый раунд, а
        # файл prompt_<id>.txt может быть отредактирован и вручную. Здесь
        # обрезка без вызова LLM гарантирует, что персона физически не может
        # вытеснить CORE_SYSTEM_PROMPT из контекста ни при каком раскладе.
        persona = self.persona_prompt
        hard_cap = self.persona_chars * PERSONA_HARD_FACTOR
        if len(persona) > hard_cap:
            self._log(f"persona {len(persona)} chars exceeds hard cap {hard_cap} "
                      f"— truncated for this call")
            persona = _truncate_text(persona, hard_cap)
        return (
            CORE_SYSTEM_PROMPT
            + "\n=== YOUR PERSONA & STRATEGY (editable by you each round) ===\n"
            + persona
            + "\n=== END PERSONA & STRATEGY ===\n"
            + self._fieldnotes_block()
        )

    def _fieldnotes_block(self) -> str:
        """
        ROLE-N: блокнот приёмов. Только для ролей с запертой персоной —
        остальные и так переписывают себя каждый раунд.
        """
        if not getattr(self, "role", None):
            return ""
        notes = load_fieldnotes(self.player_id, self.base_dir)
        if not notes:
            return ""
        body = "\n".join(f"  - {n}" for n in notes)
        return (
            "\n=== FIELD NOTES (what you have learned about working this table) ===\n"
            + body
            + "\nThese are your own observations from earlier rounds. Use them:\n"
            "do not re-run an approach that is listed here as rejected — vary the\n"
            "angle, the target or the price. They refine HOW you work your method;\n"
            "they never replace it.\n=== END FIELD NOTES ===\n"
        )

    def _spendable(self) -> int:
        """Сколько РЕАЛЬНО можно отправить прямо сейчас: min(баланс, остаток бюджета)."""
        limit = getattr(self, "transfer_budget_this_round", None)
        if limit is None:
            return self.balance
        left = max(0, limit - getattr(self, "sent_this_round", 0))
        return min(self.balance, left)

    def _transfer_budget_note(self) -> str:
        limit = getattr(self, "transfer_budget_this_round", None)
        if limit is None:
            return ""
        spent = getattr(self, "sent_this_round", 0)
        left = max(0, limit - spent)
        note = (f"Round transfer budget: {left} of {limit} left "
                f"(already sent {spent} this round).\n")
        if left == 0:
            note += ("You CANNOT move any coins this round — anything you "
                     "promise to pay now will not go through, and your "
                     "partner will see the attempt fail. Offer something "
                     "other than money, or agree to pay next round.\n")
        elif left < self.balance:
            note += ("This budget is computed from your balance at the START "
                     "of the round; coins received during dialogues do NOT "
                     "raise it.\n")
        return note

    def _log(self, msg):
        if self.log:
            self.log.write(self.player_id, msg)
        else:
            print(f"[{self.player_id}] {msg}")

    # ── synapse compression ──────────────────────────────────────────────

    def _compress_betting_synapse_if_needed(self):
        """If betting notes > MAX_SYNAPSE_CHARS, ask LLM to compress them."""
        notes = load_notes(self.player_id, self.base_dir)
        if len(notes) <= self.synapse_chars:
            return notes
        self._log(f"betting synapse too long ({len(notes)} chars) → compressing…")
        user_msg = (
            f"Your current betting synapse is too long ({len(notes)} chars) and must be "
            f"compressed to under {self.synapse_chars} chars.\n\n"
            f"Current synapse:\n{notes}\n\n"
            f"Compress it: keep the most important strategic rules, key observations, "
            f"current role, bet preferences. Remove redundant or outdated entries.\n"
            f"Return ONLY JSON: {{\"notes\": \"compressed synapse text\"}}"
        )
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + "\nTASK: Compress your betting synapse.",
                user=user_msg, temperature=0.3, max_tokens=self.tok_compress
            )
            compressed = _as_text(resp.get("notes"), notes)[:self.synapse_chars]
            save_notes(self.player_id, self.base_dir, compressed)
            self._log(f"betting synapse compressed: {len(notes)} → {len(compressed)} chars")
            return compressed
        except LLMUnavailable:
            raise            # FIX-17: выключатель наверх, не в заглушку
        except Exception as e:
            self._log(f"betting synapse compression failed ({e})")
            return notes[:self.synapse_chars]

    def _compress_dialogue_synapse_if_needed(self):
        """
        If raw interactions list is too long, ask LLM to summarise older
        entries into compressed_history, keeping only last MAX_RAW_INTERACTIONS raw.
        """
        dsyn = load_dsyn(self.player_id, self.base_dir)
        raw = dsyn.get("interactions", [])
        text_size = len(json.dumps(dsyn))

        if text_size <= self.dsyn_chars and len(raw) <= self.raw_interactions:
            return dsyn

        old_raw = raw[:-self.raw_interactions] if len(raw) > self.raw_interactions else []
        keep_raw = raw[-self.raw_interactions:]

        if not old_raw:
            return dsyn

        self._log(f"dialogue synapse too large ({text_size} chars), compressing {len(old_raw)} old entries…")
        old_text = "\n".join(
            f"r{e.get('round','?')} partner={e.get('partner','?')} "
            f"net={e.get('net_transfer',0):+.2f} summary={e.get('summary','')}"
            for e in old_raw
        )
        existing_compressed = dsyn.get("compressed_history", "")

        user_msg = (
            f"Compress the following old interaction log into a short paragraph "
            f"(max 400 chars). Merge with any existing compressed history. "
            f"Keep: net money flows, deal outcomes, trust lessons, debts.\n\n"
            f"Existing compressed history:\n{existing_compressed or '(none)'}\n\n"
            f"New entries to add:\n{old_text}\n\n"
            f"Return ONLY JSON: {{\"compressed_history\": \"merged compact text\"}}"
        )
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + "\nTASK: Compress dialogue synapse history.",
                user=user_msg, temperature=0.3, max_tokens=self.tok_compress
            )
            new_compressed = _as_text(resp.get("compressed_history"), existing_compressed)[:500]
        except LLMUnavailable:
            raise            # FIX-17: выключатель наверх, не в заглушку
        except Exception as e:
            self._log(f"dialogue synapse compression failed ({e})")
            new_compressed = existing_compressed

        dsyn["compressed_history"] = new_compressed
        dsyn["interactions"] = keep_raw
        save_dsyn(self.player_id, self.base_dir, dsyn)
        self._log(f"dialogue synapse compressed: kept {len(keep_raw)} raw, "
                  f"compressed_history={len(new_compressed)} chars")
        return dsyn

    def _compress_persona_if_needed(self):
        """
        FIX-12: если персона переросла persona_chars, просим модель ужать её,
        сохранив роль и стратегию. При неудаче — обрезка по границе
        предложения.

        ВАЖНО: system здесь CORE_SYSTEM_PROMPT, а НЕ self.abstract_prompt.
        abstract_prompt содержит саму персону, поэтому при сжатии раздутая
        персона уехала бы в запрос ДВАЖДЫ — и именно тот вызов, который
        должен чинить переполнение контекста, переполнял бы его сильнее всех
        остальных.
        """
        persona = self.persona_prompt
        if len(persona) <= self.persona_chars:
            return persona

        # ROLE-1: у запертой персоны сжатие не просто бессмысленно — оно
        # ВРАЖДЕБНО. Сжатие делается моделью, а модель, переписывая текст
        # роли, неизбежно смягчает и обобщает его; через два-три сжатия от
        # инструкции остаётся пересказ. Заперта — значит заперта, включая
        # «безобидные» перезаписи. От переполнения контекста защищает
        # жёсткая обрезка в abstract_prompt, она не пишет на диск.
        if self.persona_locked:
            self._log(f"persona locked (role={self.role}) — skipping compression "
                      f"of {len(persona)} chars; hard cap still applies per call")
            return persona

        # FIX-12b: сам вызов сжатия кладёт персону в user-сообщение целиком.
        # На персоне в 50k символов это давало 166% контекста — то есть вызов,
        # призванный вылечить переполнение, переполнял сильнее всего, что
        # было до него. Сначала жёстко режем до потолка, потом просим ужать:
        # смысла отправлять модели 50k символов, чтобы получить 2.5k, нет.
        hard_cap = self.persona_chars * PERSONA_HARD_FACTOR
        if len(persona) > hard_cap:
            self._log(f"persona {len(persona)} chars → hard-truncated to {hard_cap} "
                      f"before compression")
            persona = _truncate_text(persona, hard_cap)

        self._log(f"persona too long ({len(persona)} chars) → compressing…")
        user_msg = (
            f"Your persona/strategy text has grown to {len(persona)} characters "
            f"and must be compressed to under {self.persona_chars}.\n\n"
            f"Current persona/strategy:\n{persona}\n\n"
            f"Compress it: keep your identity, your core strategic rules and your "
            f"current stance toward other players. Drop repetition, narration and "
            f"anything already superseded. Do not invent new strategy here — this "
            f"is compression, not revision.\n"
            f"Return ONLY JSON: {{\"new_persona\": \"compressed persona text\"}}"
        )
        try:
            resp = self.client.chat_json(
                system=CORE_SYSTEM_PROMPT + "\nTASK: Compress your persona/strategy text.",
                user=user_msg, temperature=0.3, max_tokens=self.tok_compress
            )
            compressed = _truncate_text(_as_text(resp.get("new_persona"), persona),
                                        self.persona_chars)
        except LLMUnavailable:
            raise            # FIX-17: выключатель наверх, не в заглушку
        except Exception as e:
            self._log(f"persona compression failed ({e}), truncating")
            compressed = _truncate_text(persona, self.persona_chars)

        save_text(prompt_file(self.player_id, self.base_dir), compressed)
        self._log(f"persona compressed: {len(persona)} → {len(compressed)} chars")
        return compressed

    # ── checklist: краткосрочная повестка агента ─────────────────────────

    _CHECKLIST_OWNERSHIP = (
        "This file is YOURS. The headings above are only a starting suggestion — "
        "you may delete any of them, rename them, invent your own, or throw the "
        "whole structure away and keep something completely different. Nothing in "
        "it is parsed by the game; it is shown back to you verbatim before you "
        "choose who to talk to, during every conversation, and when you place your "
        "bet. Keep whatever actually helps you act, drop whatever you never look at."
    )

    def _checklist_or_default(self):
        text = load_checklist(self.player_id, self.base_dir)
        return _truncate_text(text, self.checklist_chars)

    def _checklist_block(self, label: str) -> str:
        """Блок для промпта; пустая строка, если чек-лист отключён."""
        if not self.use_checklist:
            return ""
        return f"{label}\n---\n{self._checklist_or_default()}\n---\n\n"

    def plan_round(self, round_no: int, available_partners: list) -> str:
        """
        FIX-19, шаг 1: планирование ПЕРЕД фазой диалогов.

        Агент видит свой чек-лист, свежие результаты раундов, скорборд и
        репутационную карту — и переписывает повестку: с кого что стребовать,
        что он сам должен, что проверить прежде чем платить.
        """
        checklist = self._checklist_or_default()
        ledger    = load_public_ledger(self.base_dir)
        history   = load_history(self.player_id, self.base_dir)
        dsyn      = load_dsyn(self.player_id, self.base_dir)

        user_msg = (
            f"You are {self.player_id}. You are about to start the dialogue phase "
            f"of round {round_no}.\n"
            f"Your balance: {self.balance}. "
            f"Players you can talk to: {available_partners}\n\n"
            + common.bailout_notice(self.base_dir, round_no)
            + _current_round_notice(round_no)
            + self._first_round_role_block(round_no) +
            f"Casino scoreboard — VERIFIED totals of other players:\n"
            f"{_format_scoreboard(ledger, exclude_pid=self.player_id)}\n\n"
            f"Recent table results (raw ledger):\n"
            f"{_format_public_ledger(ledger, window=self.ledger_win_move, exclude_pid=self.player_id)}\n\n"
            f"Your own recent rounds:\n{_format_history(history, self.history_window)}\n\n"
            f"Your reputation map:\n"
            f"{_format_dsyn_for_prompt(dsyn, self.dsyn_recent, self.deals_shown, self.fails_shown)}\n\n"
            + transfer_ledger.format_recent(self.player_id, self.base_dir)
            + promise_ledger.due_reminder(self.player_id, self.base_dir, round_no)
            + open_bets.format_for_prompt(self.base_dir, self.player_id) +
            f"Your checklist as it stands:\n---\n{checklist}\n---\n\n"
            f"{self._CHECKLIST_OWNERSHIP}\n\n"
            f"Rewrite it for this round. Carry forward anything still open, delete "
            f"what is settled or dead, and add what you intend to do THIS round: who "
            f"you mean to approach and for what, what you owe and to whom, what you "
            f"are owed, and what you must verify against the ledger before paying "
            f"anyone. Be concrete — name players and amounts. Max "
            f"{self.checklist_chars} characters.\n"
            # FIX-20d: правка реестра доступна и здесь. Раньше статус можно
            # было сменить только в update_checklist, то есть лишь поговорив
            # с кем-то: если должник перестал выходить на связь, пометить
            # долг broken было физически нечем, и он висел просроченным вечно.
            + promise_ledger.format_for_model(self.player_id, self.base_dir, round_no)
            + "\n" +
            f"{promise_ledger.PROMPT_INSTRUCTIONS}\n"
            f"Return ONLY JSON: {{\"checklist\": \"the full new text\", "
            f"\"promises\": [...]}}"
        )
        system_prompt = self.abstract_prompt + "\nTASK: Plan your round. JSON only."
        try:
            resp = self.client.chat_json(
                system=system_prompt,
                user=user_msg, temperature=0.4, max_tokens=self.tok_checklist
            )
            new = _truncate_text(_as_text(resp.get("checklist"), checklist),
                                 self.checklist_chars)
        except LLMUnavailable:
            append_checklist_history_failure(
                self.player_id, self.base_dir,
                f"PLAN_ROUND r{round_no}", "LLM unavailable")
            raise            # FIX-17: выключатель наверх, не в заглушку
        except Exception as e:
            self._log(f"plan_round failed ({e}), keeping old checklist")
            append_checklist_history_failure(
                self.player_id, self.base_dir, f"PLAN_ROUND r{round_no}", str(e))
            return checklist

        save_checklist(self.player_id, self.base_dir, new)
        promise_ledger.merge_and_save(          # FIX-20d
            self.player_id, self.base_dir, resp.get("promises"), round_no)
        self._log(f"CHECKLIST PLANNED r{round_no} ({len(new)} chars):\n{new}")
        append_checklist_history(
            self.player_id, self.base_dir, f"PLAN_ROUND r{round_no}",
            system_prompt, user_msg, new)
        return new

    def update_checklist(self, partner_id: str, conversation: list[dict],
                         net_transfer: int, round_no: int,
                         speech_became_free: bool = False) -> str:
        """
        FIX-19, шаг 2: обновление ПОСЛЕ каждого диалога, пока разговор свеж.

        Отдельно от update_dsyn намеренно: та пишет долгосрочную репутацию
        («можно ли ему верить»), эта — что конкретно теперь надо сделать
        («он обязался вернуть 24 к концу r5 — стребовать»).

        XFER-FREE-NOTE: `speech_became_free` — сработало ли в ЭТОМ диалоге
        правило "речь бесплатна после перевода" (см. speech_cost.py). Само
        правило игрок и так видел ЖИВЬЁМ внутри диалога — этот параметр не
        сообщает ему ничего нового про факт, а специально ПОВТОРЯЕТ его
        здесь, на пост-анализе, чтобы игрок мог сделать из него урок на
        будущее ("отправь мелкий перевод пораньше — весь дальнейший торг в
        этом разговоре бесплатен"), а не просто воспользоваться моментом и
        забыть. Без этого повторения ни в одной заметке/чек-листе за всю
        партию это никогда не всплывало (проверено на реальном прогоне).
        """
        checklist = self._checklist_or_default()
        conv_txt = "\n".join(
            f"  {t['from']}: {t['message']}"
            + format_transfer_note(t, self.player_id)
            for t in conversation
        )
        free_speech_note = ""
        if speech_became_free:
            free_speech_note = (
                f"\nTACTIC NOTE: partway through this conversation, a coin transfer "
                f"happened between you and {partner_id}, and after that speech was "
                f"free for BOTH of you for the rest of it — the casino's per-line "
                f"tariff no longer applied. This is a repeatable lever: sending or "
                f"agreeing to a transfer early in a future dialogue buys unlimited "
                f"free negotiation for the rest of THAT conversation. Consider "
                f"whether to use this deliberately next time instead of only "
                f"noticing it after the fact.\n"
            )
        user_msg = (
            f"You are {self.player_id}. Round {round_no}. Your conversation with "
            f"{partner_id} just ended.\n"
            f"Net coins moved between you this conversation: {net_transfer:+.2f} "
            f"(positive = you received).\nYour balance is now {self.balance}.\n"
            f"{free_speech_note}\n"
            f"Transcript:\n{conv_txt}\n\n"
            f"Your checklist:\n---\n{checklist}\n---\n\n"
            # FIX-20a: реестр показываем модели. Раньше её просили «перенести
            # все открытые обещания», не показав ни одного — выполнить это
            # было нельзя, и список каждый раз пересочинялся с нуля.
            + promise_ledger.format_for_model(self.player_id, self.base_dir, round_no)
            + "\n" +
            f"{self._CHECKLIST_OWNERSHIP}\n\n"
            f"Update it in light of what just happened. Record any concrete "
            f"commitment made — by them to you, and by YOU to them — with the "
            f"player name, the amount and the round it is due. A bet either of "
            f"you places at the casino table is not such a commitment — it is "
            f"owed to the wheel, not to each other; only record actual transfers "
            f"or agreed payout splits. Tick off anything "
            f"that was actually settled just now. Note anything they claimed that "
            f"you have not yet checked against the ledger. Max "
            f"{self.checklist_chars} characters.\n"
            f"{promise_ledger.PROMPT_INSTRUCTIONS}\n"
            f"Return ONLY JSON: {{\"checklist\": \"the full new text\", "
            f"\"promises\": [...]}}"
        )
        system_prompt = self.abstract_prompt + "\nTASK: Update your checklist. JSON only."
        event = f"UPDATE_CHECKLIST r{round_no} after dialogue with {partner_id}"
        try:
            resp = self.client.chat_json(
                system=system_prompt,
                user=user_msg, temperature=0.4, max_tokens=self.tok_checklist
            )
            new = _truncate_text(_as_text(resp.get("checklist"), checklist),
                                 self.checklist_chars)
        except LLMUnavailable:
            append_checklist_history_failure(
                self.player_id, self.base_dir, event, "LLM unavailable")
            raise
        except Exception as e:
            self._log(f"update_checklist failed ({e}), keeping old checklist")
            append_checklist_history_failure(self.player_id, self.base_dir, event, str(e))
            return checklist

        save_checklist(self.player_id, self.base_dir, new)
        # FIX-20: same call, one extra field — no additional LLM round trip.
        # Malformed/missing "promises" is handled inside promise_ledger and
        # never raises.
        promise_ledger.merge_and_save(
            self.player_id, self.base_dir, resp.get("promises"), round_no)
        self._log(f"CHECKLIST UPDATED after {partner_id} ({len(new)} chars)")
        append_checklist_history(
            self.player_id, self.base_dir, event, system_prompt, user_msg, new)
        return new

    # ── apply result ─────────────────────────────────────────────────────

    def apply_result(self, result, round_no=None):
        win    = result.get("win", False)
        payout = result.get("payout", 0)
        if win:
            self.balance += payout
        save_balance(self.player_id, self.base_dir, self.balance)
        entry = {
            "round_no": round_no,
            "winning_number": result.get("winning_number"),
            "bet": result.get("bet"),
            "win": win,
            "payout": payout,
            "balance_after": self.balance,
        }
        append_history(self.player_id, self.base_dir, entry)

        # публичная запись — видна ВСЕМ игрокам, чтобы можно было проверить
        # реальный исход чужой ставки, а не просто верить словам в диалоге
        append_public_ledger(self.base_dir, {
            "round_no": round_no,
            "player_id": self.player_id,
            "winning_number": result.get("winning_number"),
            "bet": result.get("bet"),
            "win": win,
            "payout": payout,
        })

        status = "WON" if win else "lost"
        self._log(f"last round: num={entry['winning_number']} {status} "
                  f"payout={payout} balance={self.balance}")
        return entry

    # ── reflect: update betting synapse ─────────────────────────────────

    def reflect_betting(self, last_entry=None, round_no=None):
        round_no = round_no or getattr(self, "current_round", None)
        notes    = self._compress_betting_synapse_if_needed()
        persona  = self._compress_persona_if_needed()   # FIX-12
        history  = load_history(self.player_id, self.base_dir)
        hist_text = _format_history(history, self.history_window)

        # FIX-5: last_entry=None означает, что в прошлом раунде игрок не
        # ставил (обычно — баланс 0). Раньше рефлексия в этом случае просто
        # не вызывалась, и банкрот навсегда замораживал свою синапсу и
        # персону — переставал эволюционировать ровно тогда, когда это
        # было нужнее всего. Теперь он получает свой ход на переосмысление
        # и может переключиться на диалоговую экономику (займы, продажа
        # стратегий), чтобы вернуться в игру.
        if last_entry is None:
            result_line = (
                f"You did NOT place a bet last round (balance was {self.balance}). "
                f"Betting alone will not get you out of this. Rethink your approach: "
                f"the dialogue economy (loans, selling your strategy or information, "
                f"brokering) is available to you every round and does not require "
                f"capital up front.\n\n"
            )
        else:
            outcome = "WON" if last_entry["win"] else "lost"
            # FIX-23: раньше поле называлось просто "number=", без уточнения,
            # что это номер ВЫПАВШЕГО ЧИСЛА рулетки (0-36), а не номер раунда.
            # На практике одна из моделей (player4) начала путать их и писала
            # в свою синапсу "Round 35: ..." вместо настоящего номера раунда —
            # ошибка тихо копилась раунд за раундом. Явно называем и подписываем
            # оба числа, чтобы модели было не за что зацепиться для путаницы.
            round_no = last_entry.get("round_no", round_no)
            round_tag = f"round {round_no}" if round_no is not None else "this round"
            result_line = (
                f"Result of {round_tag} (note: this is the ROUND number, separate "
                f"from the roulette wheel's winning number below): "
                f"winning_number={last_entry['winning_number']}, "
                f"bet={last_entry['bet']['type']} amount={last_entry['bet']['amount']} → "
                f"{outcome}, payout={last_entry['payout']}, "
                f"balance={last_entry['balance_after']}.\n\n"
            )

        # ROLE-1: у запертой персоны просить new_persona нельзя. Дело не в
        # экономии токенов: если попросить и потом отбросить, модель каждый
        # раунд формулирует «новую себя», видит, что мир её не принял, и
        # начинает описывать этот конфликт в СИНАПСЕ — то есть роль всё равно
        # размывается, только через другой файл. Проще не спрашивать.
        if self.persona_locked:
            # ROLE-N: персону по-прежнему не отдаём на правку, но просим одно
            # наблюдение о приёме — что сработало, а что нет. Это единственное
            # место, где вывод о МЕТОДЕ может осесть: чек-лист живёт один
            # раунд, синапса хранит доверие к людям, персона заперта.
            existing = load_fieldnotes(self.player_id, self.base_dir)
            seen = ("\n".join(f"  - {n}" for n in existing)
                    if existing else "  (none yet)")
            persona_block = (
                f"Your persona/strategy text (fixed for this game — you cannot "
                f"edit it, and you are not being asked to):\n{persona}\n\n"
                f"Your field notes so far:\n{seen}\n\n"
                f"Update your betting synapse, and add up to "
                f"{FIELDNOTES_PER_ROUND} field notes about how your METHOD is "
                f"landing at this table: an approach that was rejected and the "
                f"exact reason given, a price nobody paid, a player who only "
                f"responds to a certain kind of offer, an angle you have not "
                f"tried yet and intend to. Be specific — name the player, the "
                f"price, the wording that failed. One observation per entry, "
                f"under 200 characters each. Only NEW ones: if you have nothing "
                f"to add, return an empty list.\n"
                f"A field note refines HOW you work your method. It never "
                f"questions the method itself and never proposes abandoning it.\n"
                f"Return ONLY JSON:\n"
                f"{{\"notes\": \"updated strategy (max {self.synapse_chars} chars)\",\n"
                f" \"field_notes\": [\"new observation\", \"...\"]}}"
            )
        else:
            persona_block = (
                f"Current persona/strategy text (the ONLY part of your prompt you can edit):\n"
                f"{persona}\n\n"
                f"Update your betting synapse. You may also rewrite your persona/strategy text "
                f"(NOT the core game rules — those are fixed and always apply regardless of what "
                f"you write here; betting, paid services, and negotiation with other players remain "
                f"available to you no matter how you rewrite this section).\n"
                f"Return ONLY JSON:\n"
                f"{{\"notes\": \"updated strategy (max {self.synapse_chars} chars)\",\n"
                f" \"update_persona\": true/false,\n"
                f" \"new_persona\": \"full new persona/strategy text, MAX {self.persona_chars} "
                f"characters (only if update_persona=true). Keep it tight: this text is "
                f"prepended to every prompt you receive, so bloat here crowds out your "
                f"synapses and the public ledger.\"}}"
            )

        # TALK-2: расход на речь ставится РЯДОМ со ставкой, до синапсы. Без
        # этой строки рефлексия видела только ставки, и падение баланса от
        # болтовни списывалось бы на них — игрок стал бы ставить осторожнее
        # вместо того, чтобы меньше говорить.
        speech_block = ""
        if round_no is not None:
            speech_block = speech_cost.format_reflect_summary(
                self.player_id, self.base_dir, round_no, self.tariff,
                bet_amount=(last_entry["bet"]["amount"] if last_entry else None))

        user_msg = (
            result_line
            + speech_block +
            f"Current betting synapse:\n{notes or '(empty)'}\n\n"
            f"Round history:\n{hist_text}\n\n"
            + persona_block
        )
        # FIX-22: раньше один-единственный сбой (504 от прокси, битый JSON,
        # что угодно) молча оставлял СТАРУЮ синапсу как есть — а в ней мог
        # быть вчерашний баланс и "last bet WON", хотя на самом деле только
        # что был проигрыш. Игрок в следующем раунде принимал решения по
        # заведомо неверным фактам о самом себе. Живой пример из партии:
        # player2 вошёл в R3 с синапсой "Balance: 115... WON", хотя баланс
        # был 80 и последняя ставка была проиграна.
        #
        # Лечим в два слоя:
        #  1) один ретрай перед тем как сдаться — 504 и битый JSON нередко
        #     разовые, повторный запрос часто проходит;
        #  2) если и ретрай не помог, не отдаём старый текст молча: клеим
        #     сверху короткую фактическую справку (реальный баланс и исход
        #     последнего раунда), чтобы следующий раунд агент планировал
        #     хотя бы от правильных цифр, даже если сама стратегия внутри
        #     синапсы устарела.
        new_notes = None
        last_err = None
        for attempt in range(2):
            try:
                resp = self.client.chat_json(
                    system=self.abstract_prompt + "\nTASK: Reflect on last round. Update betting synapse.",
                    user=user_msg, temperature=0.5, max_tokens=self.tok_reflect
                )
                # FIX-6: модель периодически возвращает notes/new_persona
                # списком или dict-ом. Раньше это молча записывалось на диск и
                # падало позже, вне try — на len(notes) в компрессоре синапсы.
                new_notes = _as_text(resp.get("notes"), notes)[:self.synapse_chars]

                # ROLE-N: одно наблюдение за раунд, дозаписью.
                if self.persona_locked:
                    raw = resp.get("field_notes", resp.get("field_note"))
                    if isinstance(raw, str):
                        raw = [raw]
                    fresh = [_as_text(n, "")[:250].strip()
                             for n in (raw or []) if _as_text(n, "").strip()]
                    if fresh:
                        kept = append_fieldnotes(self.player_id, self.base_dir, fresh)
                        for n in fresh[:FIELDNOTES_PER_ROUND]:
                            self._log(f"FIELD NOTE (+1, {len(kept)} kept): {n}")

                if self.persona_locked and resp.get("update_persona"):
                    # ROLE-1: модель всё равно иногда возвращает поле, которого
                    # у неё не просили. Пишем это в лог, а не игнорируем молча:
                    # частота попыток переписать роль — сама по себе результат
                    # эксперимента (насколько персона «давит» на агента).
                    self._log(f"persona locked (role={self.role}) — "
                              f"ignored attempted rewrite")
                elif resp.get("update_persona") and resp.get("new_persona"):
                    # FIX-12: модель регулярно превышает объявленный лимит —
                    # обрезаем по границе предложения, а не доверяем на слово.
                    raw_persona = _as_text(resp["new_persona"], persona)
                    new_persona = _truncate_text(raw_persona, self.persona_chars)
                    if len(raw_persona) > len(new_persona):
                        self._log(f"new persona {len(raw_persona)} chars > limit "
                                  f"{self.persona_chars} — truncated")
                    save_text(prompt_file(self.player_id, self.base_dir), new_persona)
                    self._log(f"PERSONA REWRITTEN ({len(new_persona)} chars):\n{new_persona}")
                last_err = None
                break
            except LLMUnavailable:
                raise            # FIX-17: выключатель наверх, не в заглушку
            except Exception as e:
                last_err = e
                if attempt == 0:
                    self._log(f"reflect_betting failed ({e}), retrying once")
                    continue

        if last_err is not None:
            self._log(f"reflect_betting failed twice ({last_err}), "
                      f"keeping old notes with a corrected fact header")
            new_notes = _stamp_stale_notes(notes, last_entry, self.balance,
                                          self.synapse_chars)

        save_notes(self.player_id, self.base_dir, new_notes)
        self._log(f"betting synapse → {new_notes[:120]}{'…' if len(new_notes) > 120 else ''}")
        return new_notes

    # ── iterative next-move decision (talk to X / go bet) ────────────────

    def decide_next_move(self, available_players: list[str], talked_to: list[str],
                         round_no: int, dialogues_this_round: list[tuple] = None) -> dict:
        """
        Called in a loop BEFORE betting. The player looks at its own reputation
        map (who is available, what happened with them before, in this round
        and in past rounds) and decides:
          - talk to a specific player it hasn't fully exhausted yet, or
          - stop talking and go place a bet at the casino.
        This is called again after every completed dialogue, so the player
        can chain multiple conversations in the same round if it wants to.
        """
        dsyn     = self._compress_dialogue_synapse_if_needed()
        dsyn_txt = _format_dsyn_for_prompt(dsyn, self.dsyn_recent,
                                           self.deals_shown, self.fails_shown)
        hist_txt = _format_history(load_history(self.player_id, self.base_dir), 4)
        notes    = load_notes(self.player_id, self.base_dir)

        talked_txt = ", ".join(talked_to) if talked_to else "(no one yet this round)"

        others_txt = "(no other dialogues yet this round)"
        if dialogues_this_round:
            others = []
            for item in dialogues_this_round:
                a, b = item[0], item[1]
                had_transfer = item[2] if len(item) > 2 else False
                if self.player_id in (a, b):
                    continue
                tag = " (a transfer happened — could be 1 coin, could be 100)" if had_transfer else " (no money changed hands, as far as you know)"
                others.append(f"  {a} ↔ {b}{tag}")
            if others:
                others_txt = "\n".join(others)

        ledger     = load_public_ledger(self.base_dir)
        score_txt  = _format_scoreboard(ledger, exclude_pid=self.player_id)
        public_txt = _format_public_ledger(ledger, window=self.ledger_win_move,
                                           exclude_pid=self.player_id)

        user_msg = (
            f"Round {round_no}. Your player id: {self.player_id}. Your balance: {self.balance}.\n\n"
            f"Betting synapse:\n{notes or '(empty)'}\n\n"
            f"Dialogue synapse / reputation map of other players:\n{dsyn_txt}\n\n"
            f"Recent casino history:\n{hist_txt}\n\n"
            + common.bailout_notice(self.base_dir, round_no)
            + _current_round_notice(round_no)
            + self._first_round_role_block(round_no) +
            f"Casino scoreboard — VERIFIED totals of OTHER players over the WHOLE "
            f"game, computed from the public ledger (this is fact, not opinion, and "
            f"not what anyone told you):\n{score_txt}\n\n"
            f"Public results — the raw ledger lines behind that scoreboard "
            f"(use to fact-check specific claims made in dialogue):\n{public_txt}\n\n"
            f"Dialogues already held by OTHER players this round (you were not part of "
            f"these — you can go ask one of them about it if it seems relevant):\n{others_txt}\n\n"
            + open_bets.format_for_prompt(self.base_dir, self.player_id) +
            f"Players available to talk to right now: {available_players}\n"
            + self._checklist_block("YOUR CHECKLIST (your own agenda — act on it):") +
            f"Players you already talked to THIS round: {talked_txt}\n\n"
            f"IMPORTANT: your reputation map is built from what OTHER PLAYERS TOLD YOU "
            f"or what you observed. Players may lie, exaggerate, spread false rumors about "
            f"others, or make promises they don't keep. Do not treat anything a player says "
            f"about themselves or about a third player as verified fact — cross-check it "
            f"against your own past interactions (deals_done / deals_failed / net transfers) "
            f"before trusting it. A high trust_score should come from actual deals and money "
            f"that changed hands, not from claims made in conversation.\n\n"
            + self.tariff.move_hint(getattr(self, "speech_is_free", False))
            + speech_cost.format_partner_costs(self.player_id, self.base_dir,
                                               self.tariff) +
            f"Decide your NEXT move:\n"
            f"  - talk to one specific available player (pick who and why), OR\n"
            f"  - stop talking this round and go place a bet at the casino.\n\n"
            f"Return ONLY JSON:\n"
            f"{{\"action\": \"talk\" or \"bet\",\n"
            f" \"partner\": \"pid\" (required if action=talk, must be from available list),\n"
            f" \"reason\": \"short reason for this choice\"}}"
        )
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + "\nTASK: Decide next move: talk to a player or go bet.",
                user=user_msg, temperature=0.7, max_tokens=self.tok_next_move
            )
            action  = resp.get("action", "bet")
            partner = resp.get("partner")
            # RETRY-2: LLM иногда возвращает "reason" не строкой (dict/list/None),
            # а вызывающий код в run_game_v2.py делает reason[:80] — на не-строке
            # это падает с "unhashable type: 'slice'" (для dict) или TypeError
            # (для None/list). Приводим к строке здесь же, у источника.
            reason = resp.get("reason", "")
            if not isinstance(reason, str):
                reason = str(reason)
            if action == "talk" and partner in available_players:
                return {"action": "talk", "partner": partner, "reason": reason}
            return {"action": "bet", "partner": None, "reason": reason}
        except LLMUnavailable:
            raise            # FIX-17: выключатель наверх, не в заглушку
        except Exception as e:
            # RETRY-1: заглушка "иду ставить" в логе неотличима от осознанного
            # решения замолчать. Помечаем явно, иначе при разборе прогона
            # технический сбой читается как стратегия игрока.
            self._log(f"decide_next_move failed ({e}) — FORCED to bet, "
                      f"dialogues skipped this round")
            return {"action": "bet", "partner": None,
                    "reason": "FORCED: decide_next_move failed"}

    # ── one dialogue turn ─────────────────────────────────────────────────

    # FIX-4: служебные слова раздували пересечение множеств и убивали
    # нормальный торг. "Lend me 10 coins now, I will return 12 next round"
    # vs "I will lend you 10 coins if you return 15 next round, not 12" —
    # идеальный контроффер — детектировался как петля на 2-м сообщении.
    _STOPWORDS = frozenset("""
        a an and are as at be been but by can coins could do does for from get
        give had has have he her him his i if in is it its me my next no not of
        on one or our out round she so that the their them then there they this
        to up us was we what when which who will with would you your yours it's
        i'll i'd we'll you'll don't won't let's just now ok okay
    """.split())

    @classmethod
    def _content_words(cls, message: str) -> set:
        words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", message.lower())
        return {w for w in words if w not in cls._STOPWORDS and len(w) > 1}

    def _first_round_role_block(self, round_no, stage: str = "open") -> str:
        """
        ROLE-R1: инструкция роли на первый раунд, пока журнал пуст.
        Со второго раунда исчезает — дальше роль работает по своему методу.

        Стадия важна: первая версия давала роли одну фразу, и Прокурор
        произнёс её дословно дважды подряд — детектор петли оборвал диалог.
        Теперь после собственной реплики роль получает другой текст: не
        повторять предложение, а вытянуть намерение собеседника.
        """
        if round_no != 1:
            return ""
        return roles.first_round_opening(getattr(self, "role", None), stage)

    def _r1_stage(self, conversation: list[dict]) -> str:
        """Уже говорил в этом диалоге — значит, продолжение, а не открытие."""
        if any(t.get("from") == self.player_id for t in (conversation or [])):
            return "followup"
        return "open"

    @classmethod
    def _is_echo(cls, message: str, conversation: list[dict], partner_id: str,
                 threshold: float = 0.85) -> bool:
        """
        ECHO-1: сгенерированная реплика почти дословно повторяет ПОСЛЕДНЮЮ
        реплику партнёра.

        `_detect_loop` смотрит на транскрипт ДО генерации и потому этот
        случай пропускает: он ловит, что разговор ходит по кругу, но не то,
        что модель вернула чужой текст как свой. В реальном прогоне игрок
        слово в слово воспроизвёл реплику собеседника, включая фразу "ты
        уже заплатил мне 50 монет" — то есть подтвердил получение денег,
        которые сам же и отдал. Такая реплика хуже, чем бесполезная: она
        попадает в транскрипт как факт и в синапсу как обязательство.

        Порог выше, чем у _detect_loop (0.85 против 0.8): здесь речь не о
        похожести темы, а о копировании текста.
        """
        if not conversation:
            return False
        mine = cls._content_words(message)
        if not mine:
            return False

        # ECHO-2: сравниваем и с последней репликой партнёра, и со СВОЕЙ
        # предыдущей. Самоповтор раньше проходил между всеми проверками:
        # `_detect_loop` смотрит транскрипт ДО генерации, когда повтора ещё
        # нет, а `_is_echo` сравнивал только с чужой репликой. В реальном
        # прогоне игрок дословно повторил собственную фразу ("I'm on black
        # for 10. You cover red and green. Let's go.") — совпадение 1.00, и
        # ни один детектор не сработал.
        candidates = []
        last = conversation[-1]
        if last.get("from") == partner_id:
            candidates.append(last.get("message", ""))
        for turn in reversed(conversation):
            if turn.get("from") != partner_id:
                candidates.append(turn.get("message", ""))
                break

        for text in candidates:
            theirs = cls._content_words(text)
            if not theirs:
                continue
            if len(mine & theirs) / max(len(mine), len(theirs)) >= threshold:
                return True
        return False

    _NUM_RE = re.compile(r"\d+")

    @classmethod
    def _terms_moved(cls, conversation: list[dict]) -> bool:
        """
        ECHO-3: идёт ли торг прямо сейчас.

        Признак движения — новое число в последней реплике, которого не было
        раньше в разговоре: 20 → 25 → 22 это встречные предложения, а не
        петля. Обрывать такой обмен нельзя, даже если лексика повторяется
        почти дословно: в торге она и обязана повторяться, меняются только
        цифры.

        Проверка намеренно грубая. Ложно разрешить один лишний ход дешевле,
        чем зарубить сделку на середине: в реальном прогоне детектор убивал
        две трети диалогов.
        """
        if len(conversation) < 2:
            return False
        latest = set(cls._NUM_RE.findall(conversation[-1].get("message", "")))
        if not latest:
            return False
        earlier = set()
        for turn in conversation[:-1]:
            earlier |= set(cls._NUM_RE.findall(turn.get("message", "")))
        return bool(latest - earlier)

    @classmethod
    def _detect_loop(cls, conversation: list[dict], threshold: float = 0.75, window: int = 4,
                     immediate_threshold: float = 0.8) -> bool:
        """
        Returns True if the newest message is suspiciously similar to ANY of
        the last `window` messages (catches drift/repetition over several
        turns), OR if it's already fairly similar to the IMMEDIATELY preceding
        message from the other side (a lower, more sensitive threshold,
        since an early near-echo of the other player's exact offer is a
        stronger signal of parroting than of a legitimate counter-offer).
        """
        # FIX-4: первый ответ на оффер естественно переиспользует его
        # лексику ("10 coins", "strategy", "loan") — это торг, а не петля.
        # Немедленную проверку включаем только с 3-го сообщения.
        if len(conversation) < 3:
            return False
        latest = cls._content_words(conversation[-1]["message"])
        if not latest:
            return False

        # sensitive check against the single immediately preceding message
        prev_words = cls._content_words(conversation[-2]["message"])
        if prev_words:
            overlap = len(latest & prev_words) / max(len(latest), len(prev_words))
            if overlap >= immediate_threshold:
                return True

        for prev in conversation[-(window + 1):-1]:
            prev_words = cls._content_words(prev["message"])
            if not prev_words:
                continue
            overlap = len(latest & prev_words) / max(len(latest), len(prev_words))
            if overlap >= threshold:
                return True
        return False

    def dialogue_turn(self, partner_id: str, partner_balance: int,
                      conversation: list[dict], round_no: int,
                      is_initiator: bool, closing_turn: bool = False,
                      dialogue_free: bool = False,
                      max_messages: int = 8) -> dict:
        # XFER-FREE: `dialogue_free` — уже прошёл перевод в ЭТОМ диалоге
        # (взводится в run_dialogue ДО этого вызова). Отдельно от
        # `speech_is_free` (роль, действует во всех диалогах игрока):
        # здесь бесплатность разговорная и разовая, поэтому reason
        # передаётся в тексты промпта, чтобы игрок не путал одно с другим.
        role_free = getattr(self, "speech_is_free", False)
        free_now  = role_free or dialogue_free
        free_reason = "role" if role_free else "transfer"

        # FIX-9: закрывающий ход после того, как партнёр объявил done.
        # Единственная его цель — дать доиграть уже согласованную сделку
        # (в первую очередь заплатить), а не открыть новый круг торга.
        # Детектор петель здесь не применяется: партнёр уже завершил
        # разговор, и повторение его формулировок — не зацикливание.
        # ECHO-3: петля при активном торге — почти всегда ложная тревога.
        # Лексика оффера повторяется по существу дела, а меняются числа;
        # обрыв на этом месте убивает сделку в момент сближения позиций.
        if (not closing_turn and self._detect_loop(conversation)
                and not self._terms_moved(conversation)):
            self._log(f"loop detected in dialogue with {partner_id}, ending early")
            # LOOP-1: `loop_break` отличает обрыв по петле от нормального
            # `done`. Разница в том, что происходит дальше: после `done`
            # партнёр получает закрывающий ход (FIX-9), чтобы доплатить по
            # УЖЕ СОГЛАСОВАННОЙ сделке. Но зацикленный спор — это спор без
            # согласованной сделки, и закрывающий ход в нём превращался в
            # подарок: в реальном прогоне после трёх срабатываний детектора
            # партнёр каждый раз использовал его, чтобы перевести 30 монет —
            # включая вымогателя, который в этом ходе заплатил собственной
            # жертве. Оркестратор обязан оборвать диалог немедленно.
            return {
                "message": "Let's wrap this up here — we're going in circles.",
                "transfer": 0, "transfer_to": None, "done": True,
                "loop_break": True
            }

        dsyn     = load_dsyn(self.player_id, self.base_dir)
        dsyn_txt = _format_dsyn_for_prompt(dsyn, self.dsyn_recent,
                                           self.deals_shown, self.fails_shown)
        score_txt = _format_scoreboard(load_public_ledger(self.base_dir),
                                       exclude_pid=self.player_id)
        # FIX: dialogue_turn was the only prompt-building path that never
        # showed the agent its OWN recent round outcomes (decide_next_move
        # and plan_round both already do). Without it, a player mid-dialogue
        # has no way to fact-check its own claims about its own last result
        # against anything — leading to confidently wrong statements like
        # "Black lost" when the player's own ledger entry says it won.
        hist_txt = _format_history(load_history(self.player_id, self.base_dir), 4)
        # TALK-2: цена КАЖДОЙ реплики прямо в транскрипте. Агрегата в конце
        # промпта было мало: без цены на каждой строке нельзя заметить, что
        # "Agreement confirmed and locked" стоило столько же, сколько
        # содержательное предложение — а именно из этого выводится правило.
        conv_txt = "\n".join(
            f"  {t['from']}: {t['message']}"
            + format_transfer_note(t, self.player_id)
            + speech_cost.format_transcript_cost(t, self.tariff)
            for t in conversation
        )
        turns_left = max_messages - len(conversation)

        # Remind the player of its OWN earlier stance in this conversation,
        # so it counters the partner's offer instead of drifting into
        # repeating or adopting the partner's position (e.g. a buyer
        # accidentally echoing the seller's price/role back to them).
        own_earlier = next((t["message"] for t in reversed(conversation) if t["from"] == self.player_id), None)
        stance_hint = (
            f"\nYour own earlier position in this conversation was: \"{own_earlier}\"\n"
            f"Stay consistent with YOUR OWN goal and role from that message — "
            f"react to {partner_id}'s latest offer, don't repeat or adopt their wording/role.\n"
            if own_earlier else ""
        )

        if closing_turn:
            role_hint = (
                f"{partner_id} has ENDED this conversation. This is your FINAL "
                f"message — there will be no further turns and no chance to act "
                f"on anything agreed here. Do NOT open a new topic or negotiate "
                f"further. If you agreed to pay {partner_id} anything in this "
                f"conversation, set 'transfer' to that amount NOW — otherwise the "
                f"deal simply does not happen and {partner_id} will record you as "
                f"someone who agreed and did not pay. Otherwise just close politely."
            )
        elif len(conversation) == 0:
            role_hint = (
                "You STARTED this conversation — make a specific, concrete opening "
                "offer or request (name a price, a bet type, or an action). "
                "Do not just say hello or express generic interest."
            )
        elif turns_left <= 2:
            # FIX-8: соседние строковые литералы склеивались РАНЬШЕ тернарника,
            # поэтому выражение читалось как `A if cond else (B + C)` — и
            # инициатор получал только "You initiated this." без всякого
            # предупреждения о дедлайне. А инициатор говорит на 1/3/5/7-м
            # сообщениях, т.е. сторона, начавшая торг, не узнавала, что
            # время вышло, и упиралась в обрыв на 8-м сообщении.
            role_hint = (
                ("You initiated this." if is_initiator else "They contacted you.")
                + f" Only {turns_left} message(s) left in this conversation — "
                  f"accept, reject, or counter the offer NOW, or say goodbye. "
                  f"Do not introduce new topics. If you owe money under a deal "
                  f"agreed here, send it in THIS message: there will be no "
                  f"later turn to do it."
            )
        else:
            role_hint = (
                ("You initiated this." if is_initiator else "They contacted you.") +
                " Advance the conversation: accept, reject, or make a concrete counter-offer. "
                "Do not repeat anything already said above."
            )

        user_msg = (
            f"You are {self.player_id}. "
            f"Dialogue with {partner_id} (their balance ≈{partner_balance}). "
            f"Round {round_no}. {role_hint}\n"
            f"{stance_hint}\n"
            + self._checklist_block(
                "YOUR CHECKLIST (your own agenda — this is what you came to do):") +
            f"Dialogue synapse for {partner_id}:\n{dsyn_txt}\n\n"
            + common.bailout_notice(self.base_dir, round_no)
            + _current_round_notice(round_no)
            + self._first_round_role_block(round_no, self._r1_stage(conversation)) +
            f"Casino scoreboard — VERIFIED totals from the public ledger. If "
            f"{partner_id} describes their own results, check them here before "
            f"paying for anything:\n{score_txt}\n\n"
            f"Your own recent rounds (fact-check any claim you or {partner_id} "
            f"make about YOUR results against this — do not rely on memory or "
            f"guesswork about what you won or lost):\n{hist_txt}\n\n"
            f"Conversation so far:\n{conv_txt or '(just started)'}\n\n"
            + _dialogue_running_totals(conversation, self.player_id, partner_id)
            # FIX-20b: раньше due_reminder стоял ТОЛЬКО в plan_round, хотя
            # долг требуют именно в разговоре — и реестр переводов здесь уже
            # был, а реестр обещаний нет. Асимметрия исправлена.
            + promise_ledger.due_reminder(self.player_id, self.base_dir, round_no)
            + open_bets.format_for_prompt(self.base_dir, self.player_id)
            + transfer_ledger.format_recent(self.player_id, self.base_dir,
                                             partner=partner_id)
            + self.tariff.status_text(self.speech_spent_this_dialogue, self.balance,
                                      free_now, free_reason)
            # TVIS-1: "Max transfer" — это ещё НЕ то, что реально пройдёт.
            # Поверх баланса стоит бюджет переводов на раунд, посчитанный
            # от баланса на НАЧАЛО раунда (SPEND-1). Игрок, начавший с нуля
            # и получивший монеты в диалоге, имеет бюджет 0 и не может
            # передать дальше ничего — если не сказать ему это прямо, он
            # будет соглашаться на сделки, которые движок обрежет молча.
            + f"Your balance: {self.balance}. Max transfer: {self._spendable()} coins.\n"
            + self._transfer_budget_note()
            + "\n"
            f"Note: 'transfer' only lets YOU send coins to {partner_id}. If you want THEM "
            f"to pay you, say so in your message and wait for their turn — do not send "
            f"coins yourself when you meant to ask for payment.\n\n"
            f"CRITICAL: setting \"done\": true ends YOUR participation immediately — you "
            f"get no further turn in this conversation. So if you are AGREEING to pay "
            f"{partner_id}, put the coins in 'transfer' in this SAME message. Saying "
            f"\"deal, I agree, I'll send it\" together with done=true means the money is "
            f"never sent, the deal does not happen, and {partner_id} records you as "
            f"someone who agreed and defaulted. Agree and pay in one move, or leave "
            f"done=false and pay on your next turn.\n\n"
            f"Your message must be different from anything already said above.\n\n"
            f"Return ONLY JSON:\n"
            f"{{\"message\": \"your text\",\n"
            f" \"transfer\": 0,\n"
            f" \"transfer_to\": null or \"{partner_id}\",\n"
            f" \"done\": false}}"
        )
        # ECHO-3: попугайскую реплику НЕ обрываем сразу — просим переписать.
        #
        # Прогон на 8b показал цену прежнего поведения: детектор убил 8 из 12
        # диалогов на одной машине и 15 из 22 на другой. Срабатывал он верно
        # — модель действительно копировала собеседника, — но наказание
        # доставалось обеим сторонам, включая ту, что вела себя нормально, и
        # торг обрывался, не начавшись.
        #
        # Теперь первый повтор стоит одного лишнего вызова LLM с явным
        # указанием, что именно не так. Диалог обрывается только если модель
        # скопировала снова: тогда ей действительно нечего сказать.
        echo_retry_hint = (
            "\n\nSTOP — YOUR PREVIOUS DRAFT WAS A COPY. It repeated, almost "
            "word for word, something already said in this conversation. That "
            "wastes a paid line and tells your partner nothing.\n"
            "Write something genuinely NEW instead. Pick one:\n"
            "  - accept their terms and, if you are paying, put the coins in "
            "'transfer' now;\n"
            "  - make a COUNTER-OFFER with a different number or a different "
            "field;\n"
            "  - ask one specific question you do not know the answer to;\n"
            "  - if you have nothing left to add, say so briefly and set "
            "done=true.\n"
            "Do NOT restate their message, and do NOT restate your own."
        )

        try:
            attempt = 0
            while True:
                resp = self.client.chat_json(
                    system=self.abstract_prompt
                           + self.tariff.rule_text(free_now, free_reason)
                           + f"\nTASK: Dialogue turn with {partner_id}. Be concrete, no filler, no repetition.",
                    user=user_msg + (echo_retry_hint if attempt else ""),
                    temperature=0.8 + 0.1 * attempt,
                    max_tokens=self.tok_dialogue
                )
                draft = str(resp.get("message", "…"))
                if closing_turn or not self._is_echo(draft, conversation, partner_id):
                    break
                attempt += 1
                if attempt > 1:
                    break
                self._log(f"draft echoed the conversation — asking for a new "
                          f"line (attempt {attempt + 1})")
            try:
                raw_transfer = int(float(resp.get("transfer", 0) or 0))
            except (TypeError, ValueError):
                raw_transfer = 0
            transfer = max(0, min(raw_transfer, self.balance))
            # FIX-1: единственный возможный получатель в этом диалоге —
            # сам партнёр. Модель регулярно возвращает transfer>0 с
            # transfer_to=null (или с опечаткой в id), и раньше такой
            # перевод молча пропадал, оставляя в логе фантомную сделку.
            # Считаем непустую сумму намерением заплатить партнёру;
            # явное указание ТРЕТЬЕГО игрока — по-прежнему отмена.
            raw_to = resp.get("transfer_to")
            if transfer > 0 and raw_to not in (None, "", partner_id):
                self._log(f"transfer_to={raw_to!r} is not the dialogue partner "
                          f"({partner_id}) — transfer cancelled")
                transfer = 0
            transfer_to = partner_id if transfer > 0 else None
            message = str(resp.get("message", "…"))

            # ECHO-1/ECHO-3: сюда доходит либо нормальная реплика, либо
            # повтор, выживший после переспроса. Второй раз — значит,
            # сказать действительно нечего, закрываем диалог через
            # loop_break (без закрывающего хода партнёру).
            #
            # Перевод из повторной реплики отменяем в любом случае: он
            # согласован не был, это часть скопированного чужого текста.
            # Именно так утекли 38 монет в реальном прогоне — копия чужой
            # фразы "let's split the risk" пришла вместе с переводом.
            if not closing_turn and self._is_echo(message, conversation, partner_id):
                self._log(f"echoed the conversation twice in a row "
                          f"(with {partner_id}) — ending dialogue")
                return {
                    "message": "Let's wrap this up here — we're going in circles.",
                    "transfer": 0, "transfer_to": None, "done": True,
                    "loop_break": True
                }

            return {
                "message": message,
                "transfer": transfer,
                "transfer_to": transfer_to,
                "done": bool(resp.get("done", False))
            }
        except LLMUnavailable:
            raise            # FIX-17: выключатель наверх, не в заглушку
        except Exception as e:
            self._log(f"dialogue_turn failed ({e})")
            return {"message": "(…)", "transfer": 0, "transfer_to": None, "done": True}

    # ── update dialogue synapse after conversation ────────────────────────

    def update_dsyn(self, partner_id: str, conversation: list[dict],
                    net_transfer: int, round_no: int,
                    sent: int = None, received: int = None,
                    speech_became_free: bool = False):
        """
        After a dialogue: ask LLM to update the reputation entry for partner_id
        and add a raw interaction record.

        DSYN-1: `sent` и `received` — фактические обороты диалога. Раньше сюда
        приходило только НЕТТО, и обороты восстанавливались из него как
        sent=max(0,-net), received=max(0,net) — то есть одна из двух граф по
        построению всегда была нулём.

        Пока деньги идут в одну сторону, разницы не видно. При встречных
        переводах она принципиальна: в реальном диалоге игроки прогнали друг
        через друга 54 монеты в четыре приёма (31 туда, 23 обратно), а в
        синапсе осело "отдал 8, получил 0". Именно эти два числа модель видит
        в промпте и по ним судит, сколько между ней и партнёром прошло, —
        так что терялся ровно тот факт, который стоило заметить.

        Аргументы необязательные: без них поведение прежнее, чтобы не ломать
        внешние вызовы, знающие только нетто.

        XFER-FREE-NOTE: `speech_became_free` — см. тот же параметр у
        update_checklist. Здесь смысл другой: это репутационная синапса ПРО
        ЭТОГО ПАРТНЁРА, так что урок формулируется как наблюдение о ЕГО
        поведении — партнёр мог перевести деньги именно чтобы разговорить
        игрока бесплатно, и это стоит учитывать при оценке его тактики.
        """
        dsyn = load_dsyn(self.player_id, self.base_dir)
        existing_rep = dsyn["reputation"].get(partner_id, {})

        score_txt = _format_scoreboard(load_public_ledger(self.base_dir),
                                       exclude_pid=self.player_id)
        # Если вызывающий не передал обороты (внешние вызовы, знающие
        # только нетто), восстанавливаем из него же — хуже, чем было, не
        # станет, а формат промпта остаётся единым.
        _gross_sent = max(0, -net_transfer) if sent is None else max(0, int(sent))
        _gross_recv = max(0, net_transfer) if received is None else max(0, int(received))
        conv_txt = "\n".join(
            f"  {t['from']}: {t['message']}"
            # TVIS-1: именно здесь рождалась reputation по неоплаченным
            # сделкам — синапса видела только состоявшиеся переводы, и
            # обещание без денег было неотличимо от исполненного.
            + format_transfer_note(t, self.player_id)
            for t in conversation
        )

        free_speech_note = ""
        if speech_became_free:
            free_speech_note = (
                f"\nNote: partway through this conversation, a transfer happened "
                f"between you and {partner_id}, which made speech free for BOTH "
                f"of you for the rest of it. Consider whether {partner_id} timed "
                f"that transfer deliberately to unlock free negotiation room, and "
                f"factor that into how you read their tactics.\n"
            )

        user_msg = (
            f"You are {self.player_id}. "
            f"Conversation with {partner_id} in round {round_no} just ended.\n"
            # DSYN-2: брутто в ТЕКСТ промпта, а не только в счётчики файла.
            # Числа sent/received приходили сюда с DSYN-1 и аккуратно
            # ложились в total_sent/total_received, но модель их не видела:
            # в промпте стояло одно свёрнутое нетто. А нетто -10 одинаково
            # описывает и "отдал 10 и всё", и цепочку "отдал 10 → вернули 5
            # → доотправил 5", где партнёр деньги ВОЗВРАЩАЛ и спорил о
            # сумме. Для trust_score это противоположные истории, и по
            # свёрнутому числу вторая читалась как выкачивание монет.
            f"Money moved this conversation: you sent {partner_id} "
            f"{_gross_sent} coin(s); {partner_id} sent you {_gross_recv} "
            f"coin(s); net for you {net_transfer:+.2f} "
            f"(positive=you received).\n"
            f"Judge by BOTH figures: coins returned or paid back are not the "
            f"same as coins never moved.\n"
            f"{free_speech_note}\n"
            f"Full conversation:\n{conv_txt}\n\n"
            f"Current reputation entry for {partner_id}:\n"
            f"{json.dumps(existing_rep, ensure_ascii=False) if existing_rep else '(first interaction)'}\n\n"
            f"Update the reputation record. Track what worked, what failed, debts, trust.\n"
            f"Return ONLY JSON:\n"
            f"{{\"trust_score\": 1-10,\n"
            f" \"deal_done\": \"short description of what was agreed/sold/bought or null\",\n"
            f" \"deal_failed\": \"what went wrong or null\",\n"
            f" \"reputation_note\": \"1-sentence updated note about this player\",\n"
            f" \"future_intent\": \"what you plan with them next round\",\n"
            f" \"summary\": \"1-sentence summary of this specific conversation\"}}"
        )
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + f"\nTASK: Update reputation for {partner_id}.",
                user=user_msg, temperature=0.4, max_tokens=self.tok_update_dsyn
            )
        except LLMUnavailable:
            raise            # FIX-17: выключатель наверх, не в заглушку
        except Exception as e:
            self._log(f"update_dsyn LLM failed ({e})")
            resp = {
                "trust_score": existing_rep.get("trust_score", 5),
                "reputation_note": existing_rep.get("reputation_note", ""),
                "future_intent": "",
                "summary": "(synapse update failed)"
            }

        # Update reputation map
        old = dsyn["reputation"].get(partner_id, {
            "trust_score": 5, "total_sent": 0, "total_received": 0, "net": 0,
            "deals_done": [], "deals_failed": [], "reputation_note": "",
            "future_intent": "", "last_seen_round": 0
        })
        # DSYN-1: если обороты переданы — пишем их; иначе прежняя реконструкция
        # из нетто (одна из граф окажется нулевой, см. docstring).
        actual_sent     = max(0, -net_transfer) if sent is None else max(0, int(sent))
        actual_received = max(0, net_transfer) if received is None else max(0, int(received))
        old["total_sent"]     += actual_sent
        old["total_received"] += actual_received
        old["net"]            += net_transfer
        old["trust_score"]     = resp.get("trust_score", old["trust_score"])
        # SCHEMA-2: тот же класс бага, что уже ловили в hallucination_test.py
        # (SCHEMA-1) — модель (реальный случай: Mistral) вернула
        # future_intent вложенным объектом вместо строки. Раньше это
        # писалось в dsyn как есть и падало ниже на `old['future_intent']
        # [:60]` с "TypeError: unhashable type: 'slice'" (срез dict'а —
        # Python пытается хэшировать slice-объект как ключ). _as_text —
        # тот же приём, что уже используется в этом файле для notes
        # (FIX-6), просто эта дорожка (update_dsyn) его не унаследовала.
        old["reputation_note"] = _as_text(resp.get("reputation_note"), old["reputation_note"])
        old["future_intent"]   = _as_text(resp.get("future_intent"), "")
        old["last_seen_round"] = round_no
        if resp.get("deal_done"):
            old["deals_done"].append(f"r{round_no}: {resp['deal_done']}")
        if resp.get("deal_failed"):
            old["deals_failed"].append(f"r{round_no}: {resp['deal_failed']}")
        dsyn["reputation"][partner_id] = old

        # Append raw interaction
        dsyn["interactions"].append({
            "round": round_no,
            "partner": partner_id,
            "net_transfer": net_transfer,
            "summary": _as_text(resp.get("summary"), ""),
            "timestamp": datetime.now().isoformat()
        })

        save_dsyn(self.player_id, self.base_dir, dsyn)
        self._log(
            f"dsyn updated for {partner_id}: trust={old['trust_score']}/10 "
            f"net_total={old['net']:+.2f}c intent='{old['future_intent'][:60]}'"
        )

    # ── decide bet ───────────────────────────────────────────────────────

    def decide_bet(self, round_no: int = None) -> dict:
        notes    = load_notes(self.player_id, self.base_dir)
        hist_txt = _format_history(load_history(self.player_id, self.base_dir), self.history_window)
        dsyn     = load_dsyn(self.player_id, self.base_dir)
        dsyn_txt = _format_dsyn_for_prompt(dsyn, self.dsyn_recent,
                                           self.deals_shown, self.fails_shown)
        ledger     = load_public_ledger(self.base_dir)
        score_txt  = _format_scoreboard(ledger, exclude_pid=self.player_id)
        public_txt = _format_public_ledger(ledger, window=self.ledger_win_bet,
                                           exclude_pid=self.player_id)
        max_amt  = max(1, int(self.balance * self.max_bet_fraction))

        user_msg = (
            f"Player: {self.player_id}  Balance: {self.balance}  "
            f"Recommended max bet: {max_amt}\n\n"
            f"Betting synapse:\n{notes or '(none yet — pick a starting strategy)'}\n\n"
            f"Dialogue synapse:\n{dsyn_txt}\n\n"
            f"You are choosing your bet for round {round_no}.\n\n"
            + self._checklist_block("YOUR CHECKLIST:") +
            f"Round history:\n{hist_txt}\n\n"
            + common.bailout_notice(self.base_dir, round_no)
            + _current_round_notice(round_no)
            + self._first_round_role_block(round_no, "bet") +
            f"Casino scoreboard — VERIFIED totals of OTHER players over the WHOLE "
            f"game, computed from the public ledger. If someone sold you a strategy, "
            f"this is where you see whether it has ever actually made them "
            f"money:\n{score_txt}\n\n"
            f"Public results — the raw ledger lines behind that scoreboard:\n{public_txt}\n\n"
            + open_bets.format_for_prompt(self.base_dir, self.player_id) +
            f"Place your bet. Return ONLY JSON:\n"
            f"{{\"type\": \"...\", \"numbers\": [...] OR \"selection\": \"...\", "
            f"\"amount\": N, \"reasoning\": \"short reason\"}}\n"
            f"Types: straight(1#,35:1) split(2#,17:1) street(3#,11:1) corner(4#,8:1) "
            f"sixline(6#,5:1) dozen(sel=1st12/2nd12/3rd12,2:1) "
            f"column(sel=col1/col2/col3,2:1) "
            f"even_money(sel=red/black/even/odd/low/high,1:1). "
            f"amount ≤ {self.balance}."
        )
        bet = None
        for _ in range(2):
            try:
                bet = self.client.chat_json(
                    system=self.abstract_prompt + "\nTASK: Place casino bet. JSON only.",
                    user=user_msg, temperature=self.temperature, max_tokens=self.tok_bet
                )
                bet.pop("reasoning", None)
                bet["player_id"] = self.player_id
                common.validate_bet(bet)
                if bet["amount"] > self.balance:
                    bet["amount"] = self.balance
                break
            except LLMUnavailable:
                raise        # FIX-17: выключатель наверх, не в заглушку
            except Exception as e:
                self._log(f"decide_bet attempt failed: {e}")
                bet = None

        if bet is None:
            bet = {"type": "even_money", "selection": "red",
                   "amount": min(self.balance, 1), "player_id": self.player_id}
            self._log(f"fallback bet used: {bet}")

        return bet
