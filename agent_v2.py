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
    return (
        f"CURRENT ROUND: {round_no}. The wheel has NOT been spun for round "
        f"{round_no} yet — it spins after all dialogue and all bets are in. "
        f"No result exists for round {round_no} or any later round, for you or "
        f"for anyone else. Any figure quoted as an outcome of round {round_no}+ "
        f"is invented, including by you.\n"
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
            f"  {pid}: casino P&L={pnl:+d}c over {a['bets']} bet(s) "
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
                f"net={r.get('net',0):+d}c "
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
                f"net={itx.get('net_transfer',0):+d}c — {itx.get('summary','')}"
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
You are an autonomous economic agent playing a multi-round casino game against
other AI players. Your ONLY objective is to GROW YOUR CAPITAL (coin balance)
by the end of the game. Three tools are always available to you, every round:

  1. ROULETTE BETTING — place bets at the casino table (straight, split,
     street, corner, sixline, dozen, column, even_money).
  2. PAID SERVICES TO OTHER PLAYERS — you may sell your betting strategy,
     sell information/tips, offer loans with interest, act as a broker, or
     run any other paid arrangement with another player, in exchange for
     coins.
  3. NEGOTIATION / DIALOGUE — you may talk to other players before betting:
     buy strategies or tips from them, borrow coins, form alliances, collect
     debts, or simply gather information. Dialogue and coin transfers between
     players are a core, always-available part of this game — never assume
     you can only interact with the casino.

You maintain two SEPARATE memories across rounds:
  - a BETTING SYNAPSE: your own casino strategy notes and observations.
  - a DIALOGUE SYNAPSE: a reputation map of other players (trust, deals done,
    deals failed, net money exchanged, your future intent per player).
Both persist and compound over the whole game — use them.

You are free to adopt and evolve ANY persona or strategic identity you like
(conservative banker, aggressive gambler, information broker, loan shark,
strategy coach, etc.) — but no matter which persona you choose, betting,
paid services, and negotiation with other players must all remain available
options to you every round.

Always respond with valid JSON only, exactly matching the schema requested
in each task. No markdown, no prose outside the JSON object.

--- WHAT IS VERIFIABLE (read this before you believe anyone, including yourself) ---
The wheel is European: 37 pockets, numbered 0 to 36. There is no 00. Valid bet
types are exactly: straight, split, street, corner, sixline, dozen, column,
even_money — nothing else exists.

There is EXACTLY ONE spin per round, and only the croupier performs it, at the
very end of the round, after all dialogue and all bets. You cannot spin the
wheel. You cannot run a "test session", a "validation run", a "backtest", or
"12 spins" — no such mechanism exists in this game. Neither can anyone else.

Every bet placed by every player, and its outcome, is written to a PUBLIC
LEDGER that you are shown each round. It is the only record of what actually
happened at the table, and it cannot be forged.

Two consequences, and they cut both ways:

  1. When another player describes results — hit rates, session logs, spins
     they ran, how their strategy performed — CHECK IT AGAINST THE PUBLIC
     LEDGER FIRST. If a claimed outcome is not there, it did not happen. If
     someone reports more spins than there have been rounds, or a bet type
     that does not exist, or a result that contradicts the ledger, they are
     fabricating it, and you should price their offer accordingly and record
     it in their reputation. Paying for unverifiable data is how capital is
     lost fastest in this game.

  2. Your OWN bets are in that same public ledger. Anything you claim about
     your own record can be checked by everyone else, immediately. If you
     promise to place a particular bet and place a different one, that is
     visible too.

Strategies, tips and predictions are opinions and cannot be verified in
advance — that is normal, and trading them is legitimate. Claims about what
ALREADY HAPPENED at the table are facts, and they are all checkable.
--- END VERIFIABILITY ---
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

class PlayerAgent:
    def __init__(self, player_id, base_dir, cfg, logger=None):
        self.player_id = player_id
        self.base_dir = base_dir
        self.cfg = cfg
        self.log = logger

        self.client = LLMClient.from_config(cfg)

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

        pf = prompt_file(player_id, base_dir)
        if not os.path.exists(pf):
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
        )

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
            f"net={e.get('net_transfer',0):+d} summary={e.get('summary','')}"
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
            + _current_round_notice(round_no) +
            f"Casino scoreboard — VERIFIED totals of other players:\n"
            f"{_format_scoreboard(ledger, exclude_pid=self.player_id)}\n\n"
            f"Recent table results (raw ledger):\n"
            f"{_format_public_ledger(ledger, window=self.ledger_win_move, exclude_pid=self.player_id)}\n\n"
            f"Your own recent rounds:\n{_format_history(history, self.history_window)}\n\n"
            f"Your reputation map:\n"
            f"{_format_dsyn_for_prompt(dsyn, self.dsyn_recent, self.deals_shown, self.fails_shown)}\n\n"
            f"Your checklist as it stands:\n---\n{checklist}\n---\n\n"
            f"{self._CHECKLIST_OWNERSHIP}\n\n"
            f"Rewrite it for this round. Carry forward anything still open, delete "
            f"what is settled or dead, and add what you intend to do THIS round: who "
            f"you mean to approach and for what, what you owe and to whom, what you "
            f"are owed, and what you must verify against the ledger before paying "
            f"anyone. Be concrete — name players and amounts. Max "
            f"{self.checklist_chars} characters.\n"
            f"Return ONLY JSON: {{\"checklist\": \"the full new text\"}}"
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
        self._log(f"CHECKLIST PLANNED r{round_no} ({len(new)} chars):\n{new}")
        append_checklist_history(
            self.player_id, self.base_dir, f"PLAN_ROUND r{round_no}",
            system_prompt, user_msg, new)
        return new

    def update_checklist(self, partner_id: str, conversation: list[dict],
                         net_transfer: int, round_no: int) -> str:
        """
        FIX-19, шаг 2: обновление ПОСЛЕ каждого диалога, пока разговор свеж.

        Отдельно от update_dsyn намеренно: та пишет долгосрочную репутацию
        («можно ли ему верить»), эта — что конкретно теперь надо сделать
        («он обязался вернуть 24 к концу r5 — стребовать»).
        """
        checklist = self._checklist_or_default()
        conv_txt = "\n".join(
            f"  {t['from']}: {t['message']}"
            + (f"  [{t['transfer']} coins to {t['transfer_to']}]" if t.get("transfer") else "")
            for t in conversation
        )
        user_msg = (
            f"You are {self.player_id}. Round {round_no}. Your conversation with "
            f"{partner_id} just ended.\n"
            f"Net coins moved between you this conversation: {net_transfer:+d} "
            f"(positive = you received).\nYour balance is now {self.balance}.\n\n"
            f"Transcript:\n{conv_txt}\n\n"
            f"Your checklist:\n---\n{checklist}\n---\n\n"
            f"{self._CHECKLIST_OWNERSHIP}\n\n"
            f"Update it in light of what just happened. Record any concrete "
            f"commitment made — by them to you, and by YOU to them — with the "
            f"player name, the amount and the round it is due. Tick off anything "
            f"that was actually settled just now. Note anything they claimed that "
            f"you have not yet checked against the ledger. Max "
            f"{self.checklist_chars} characters.\n"
            f"Return ONLY JSON: {{\"checklist\": \"the full new text\"}}"
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

    def reflect_betting(self, last_entry=None):
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
            result_line = (
                f"Round result: number={last_entry['winning_number']}, "
                f"bet={last_entry['bet']['type']} amount={last_entry['bet']['amount']} → "
                f"{outcome}, payout={last_entry['payout']}, "
                f"balance={last_entry['balance_after']}.\n\n"
            )

        user_msg = (
            result_line +
            f"Current betting synapse:\n{notes or '(empty)'}\n\n"
            f"Round history:\n{hist_text}\n\n"
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
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + "\nTASK: Reflect on last round. Update betting synapse.",
                user=user_msg, temperature=0.5, max_tokens=self.tok_reflect
            )
            # FIX-6: модель периодически возвращает notes/new_persona
            # списком или dict-ом. Раньше это молча записывалось на диск и
            # падало позже, вне try — на len(notes) в компрессоре синапсы.
            new_notes = _as_text(resp.get("notes"), notes)[:self.synapse_chars]
            if resp.get("update_persona") and resp.get("new_persona"):
                # FIX-12: модель регулярно превышает объявленный лимит —
                # обрезаем по границе предложения, а не доверяем на слово.
                raw_persona = _as_text(resp["new_persona"], persona)
                new_persona = _truncate_text(raw_persona, self.persona_chars)
                if len(raw_persona) > len(new_persona):
                    self._log(f"new persona {len(raw_persona)} chars > limit "
                              f"{self.persona_chars} — truncated")
                save_text(prompt_file(self.player_id, self.base_dir), new_persona)
                self._log(f"PERSONA REWRITTEN ({len(new_persona)} chars):\n{new_persona}")
        except LLMUnavailable:
            raise            # FIX-17: выключатель наверх, не в заглушку
        except Exception as e:
            self._log(f"reflect_betting failed ({e}), keeping old notes")
            new_notes = notes

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
            + _current_round_notice(round_no) +
            f"Casino scoreboard — VERIFIED totals of OTHER players over the WHOLE "
            f"game, computed from the public ledger (this is fact, not opinion, and "
            f"not what anyone told you):\n{score_txt}\n\n"
            f"Public results — the raw ledger lines behind that scoreboard "
            f"(use to fact-check specific claims made in dialogue):\n{public_txt}\n\n"
            f"Dialogues already held by OTHER players this round (you were not part of "
            f"these — you can go ask one of them about it if it seems relevant):\n{others_txt}\n\n"
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
            if action == "talk" and partner in available_players:
                return {"action": "talk", "partner": partner, "reason": resp.get("reason", "")}
            return {"action": "bet", "partner": None, "reason": resp.get("reason", "")}
        except LLMUnavailable:
            raise            # FIX-17: выключатель наверх, не в заглушку
        except Exception as e:
            self._log(f"decide_next_move failed ({e})")
            return {"action": "bet", "partner": None, "reason": ""}

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
                      is_initiator: bool, closing_turn: bool = False) -> dict:

        # FIX-9: закрывающий ход после того, как партнёр объявил done.
        # Единственная его цель — дать доиграть уже согласованную сделку
        # (в первую очередь заплатить), а не открыть новый круг торга.
        # Детектор петель здесь не применяется: партнёр уже завершил
        # разговор, и повторение его формулировок — не зацикливание.
        if not closing_turn and self._detect_loop(conversation):
            self._log(f"loop detected in dialogue with {partner_id}, ending early")
            return {
                "message": "Let's wrap this up here — we're going in circles.",
                "transfer": 0, "transfer_to": None, "done": True
            }

        dsyn     = load_dsyn(self.player_id, self.base_dir)
        dsyn_txt = _format_dsyn_for_prompt(dsyn, self.dsyn_recent,
                                           self.deals_shown, self.fails_shown)
        score_txt = _format_scoreboard(load_public_ledger(self.base_dir),
                                       exclude_pid=self.player_id)
        conv_txt = "\n".join(
            f"  {t['from']}: {t['message']}"
            + (f" [sent {t['transfer']} coins to {t['transfer_to']}]"
               if t.get("transfer", 0) > 0 else "")
            for t in conversation
        )
        turns_left = 8 - len(conversation)

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
            + _current_round_notice(round_no) +
            f"Casino scoreboard — VERIFIED totals from the public ledger. If "
            f"{partner_id} describes their own results, check them here before "
            f"paying for anything:\n{score_txt}\n\n"
            f"Conversation so far:\n{conv_txt or '(just started)'}\n\n"
            + _dialogue_running_totals(conversation, self.player_id, partner_id) +
            f"Your balance: {self.balance}. Max transfer: {self.balance} coins.\n\n"
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
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + f"\nTASK: Dialogue turn with {partner_id}. Be concrete, no filler, no repetition.",
                user=user_msg, temperature=0.8, max_tokens=self.tok_dialogue
            )
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
            return {
                "message": str(resp.get("message", "…")),
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
                    net_transfer: int, round_no: int):
        """
        After a dialogue: ask LLM to update the reputation entry for partner_id
        and add a raw interaction record.
        """
        dsyn = load_dsyn(self.player_id, self.base_dir)
        existing_rep = dsyn["reputation"].get(partner_id, {})

        score_txt = _format_scoreboard(load_public_ledger(self.base_dir),
                                       exclude_pid=self.player_id)
        conv_txt = "\n".join(
            f"  {t['from']}: {t['message']}"
            + (f" [+{t['transfer']} coins]" if t.get("transfer", 0) > 0 else "")
            for t in conversation
        )

        user_msg = (
            f"You are {self.player_id}. "
            f"Conversation with {partner_id} in round {round_no} just ended.\n"
            f"Net money for YOU: {net_transfer:+d} coins "
            f"(positive=you received, negative=you sent).\n\n"
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
        sent     = max(0, -net_transfer)
        received = max(0, net_transfer)
        old["total_sent"]     += sent
        old["total_received"] += received
        old["net"]            += net_transfer
        old["trust_score"]     = resp.get("trust_score", old["trust_score"])
        old["reputation_note"] = resp.get("reputation_note", old["reputation_note"])
        old["future_intent"]   = resp.get("future_intent", "")
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
            "summary": resp.get("summary", ""),
            "timestamp": datetime.now().isoformat()
        })

        save_dsyn(self.player_id, self.base_dir, dsyn)
        self._log(
            f"dsyn updated for {partner_id}: trust={old['trust_score']}/10 "
            f"net_total={old['net']:+d}c intent='{old['future_intent'][:60]}'"
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
            + _current_round_notice(round_no) +
            f"Casino scoreboard — VERIFIED totals of OTHER players over the WHOLE "
            f"game, computed from the public ledger. If someone sold you a strategy, "
            f"this is where you see whether it has ever actually made them "
            f"money:\n{score_txt}\n\n"
            f"Public results — the raw ledger lines behind that scoreboard:\n{public_txt}\n\n"
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
