"""
promise_ledger.py — structured cross-round commitments (FIX-20).

README (v2.1) named this the one open gap: "future_intent"/"deals_done" in
the dialogue synapse are free text, so a promise and its fulfilment are
indistinguishable to the system — trust_score is built on the LLM's
impression, not on a checked fact.

This module does NOT add a new LLM call. Extraction of promises from a
just-finished conversation is folded into the EXISTING `update_checklist`
call (agent already reads the full transcript there — asking it to also
emit structured promises is one extra JSON field, not one extra round trip).
Everything after that — deciding a promise is due or overdue, and putting
that in front of the agent at plan_round — is plain deterministic code, no
LLM involved. That split is the point: understanding free-text dialogue
needs a model; comparing two integers (due_round vs round_no) does not.

What the agent DOES with an overdue promise (chase it, let it go, hold a
grudge, stay trusting) is deliberately left to the model/persona — this
module only guarantees the fact reaches the prompt every round until it is
marked settled or broken, so the agent's character decides the tone, not
the code.
"""

from __future__ import annotations

import os
import common

MAX_PROMISES = 30           # hard cap per player; see _evict() for the policy
VALID_DIRECTIONS = {"owed_to_me", "i_owe"}
VALID_STATUSES = {"open", "settled", "broken"}


def promises_file(pid, base_dir):
    return os.path.join(base_dir, f"promises_{pid}.json")


def load_promises(pid, base_dir) -> list[dict]:
    path = promises_file(pid, base_dir)
    if os.path.exists(path):
        return common.read_json(path).get("promises", [])
    return []


def save_promises(pid, base_dir, promises: list[dict]):
    common.write_json(promises_file(pid, base_dir), {"promises": promises})


def _clean_one(raw: dict, round_no: int, next_id: int) -> dict | None:
    """Validate/coerce one promise dict from model output. Drop silently
    (never crash the round) if it can't be made sense of — same policy as
    FIX-6's _as_text: a malformed entry from the model must not take down
    the whole checklist update."""
    if not isinstance(raw, dict):
        return None
    direction = raw.get("direction")
    if direction not in VALID_DIRECTIONS:
        return None
    counterparty = raw.get("counterparty")
    if not isinstance(counterparty, str) or not counterparty:
        return None
    try:
        amount = int(raw.get("amount"))
    except (TypeError, ValueError):
        return None
    try:
        due_round = int(raw.get("due_round"))
    except (TypeError, ValueError):
        due_round = round_no  # no due date given -> treat as due this round
    status = raw.get("status")
    if status not in VALID_STATUSES:
        status = "open"
    description = raw.get("description")
    description = description if isinstance(description, str) else ""
    return {
        "id": raw.get("id") if isinstance(raw.get("id"), int) else next_id,
        "direction": direction,
        "counterparty": counterparty,
        "amount": max(0, amount),
        "due_round": due_round,
        "created_round": raw.get("created_round", round_no)
                         if isinstance(raw.get("created_round"), int) else round_no,
        "description": description[:200],
        "status": status,
    }


def _fingerprint(p: dict) -> tuple:
    """Опознание обещания, когда модель не вернула id."""
    return (p.get("direction"), (p.get("counterparty") or "").strip(),
            p.get("amount"), p.get("due_round"))


def _evict(promises: list[dict]) -> list[dict]:
    """
    FIX-20c: политика вытеснения при упоре в MAX_PROMISES, раньше нигде не
    описанная. Сначала выбрасываем ЗАКРЫТЫЕ (settled/broken) записи, самые
    старые по due_round — они уже никого не касаются. Открытые трогаем
    только если одних закрытых не хватило, и тогда режем самые дальние по
    сроку: просроченное и текущее важнее того, что наступит нескоро.
    """
    if len(promises) <= MAX_PROMISES:
        return promises
    open_ones = sorted([p for p in promises if p["status"] == "open"],
                       key=lambda p: p["due_round"])
    closed = sorted([p for p in promises if p["status"] != "open"],
                    key=lambda p: p["due_round"], reverse=True)
    keep_closed = closed[:max(0, MAX_PROMISES - len(open_ones))]
    return (open_ones[:MAX_PROMISES] + keep_closed)[:MAX_PROMISES]


def merge_and_save(pid, base_dir, model_promises, round_no: int) -> list[dict]:
    """
    FIX-20a: СЛИЯНИЕ, а не замещение.

    Прежняя версия заменяла весь список тем, что вернула модель, — при том
    что текущий список модели не показывали ВООБЩЕ. Инструкция требовала
    «carry forward every promise still open even if this conversation didn't
    touch it», а выполнить это было физически невозможно: модель видела
    только транскрипт одного разговора. В итоге реестр хранил ровно одну
    последнюю беседу, и долг исчезал не когда его прощали, а когда игрок
    поговорил с кем-то другим. При 5 игроках и 2 диалогах реестр каждого
    перезаписывался так четыре раза за раунд.

    Новый контракт с моделью — обратный и куда более выполнимый: ей ПОКАЗЫВАЮТ
    текущий список с id и просят вернуть только НОВЫЕ и ИЗМЕНИВШИЕСЯ записи.
    Всё, чего она не упомянула, остаётся как есть. Забывчивость модели теперь
    означает «ничего не изменилось» вместо «удалить всё» — то есть отказ стал
    безопасным по построению, а не по добросовестности 8B-модели.

    Опознание записи: сначала по id, затем по отпечатку
    (направление, контрагент, сумма, срок) — иначе модель, не вернувшая id,
    плодила бы дубли того же обещания на каждом ходу.
    """
    existing = load_promises(pid, base_dir)
    by_id = {p["id"]: p for p in existing if isinstance(p.get("id"), int)}
    next_id = max(by_id, default=0) + 1

    if not isinstance(model_promises, list):
        # FIX-20a: поля нет или оно битое — это НЕ повод стирать реестр.
        return existing

    by_fp = {}
    for p in existing:
        by_fp.setdefault(_fingerprint(p), p)

    for raw in model_promises:
        item = _clean_one(raw, round_no, next_id)
        if item is None:
            continue
        target = by_id.get(item["id"]) or by_fp.get(_fingerprint(item))
        if target is not None:
            # обновляем существующее, id сохраняется
            target.update({k: item[k] for k in
                           ("direction", "counterparty", "amount", "due_round",
                            "description", "status")})
        else:
            item["id"] = next_id
            next_id += 1
            existing.append(item)
            by_id[item["id"]] = item
            by_fp.setdefault(_fingerprint(item), item)

    existing.sort(key=lambda p: (p["status"] != "open", p["due_round"], p["id"]))
    existing = _evict(existing)
    save_promises(pid, base_dir, existing)
    return existing


def open_debt_to(pid, base_dir, partner: str, before_round: int) -> int:
    """
    OPEN-1: сколько `pid` СЕЙЧАС должен `partner` по обещаниям, взятым в
    ПРЕДЫДУЩИХ раундах.

    Нужно, чтобы отличить погашение прошлой договорённости от предоплаты за
    ничто. В реальном прогоне игрок открыл диалог словами "обсудим за 50
    монет" и тут же перевёл половину своего капитала — до всякого ответа,
    когда согласовывать было ещё нечего. Запрещать перевод в первой реплике
    целиком нельзя: расчёт по обещанию прошлого раунда выглядит точно так
    же и является совершенно законным. Различает их именно реестр: у
    расчёта есть открытая запись `i_owe`, у предоплаты за воздух её нет.

    `before_round` строго: обещание, взятое в ЭТОМ же раунде, ещё не могло
    появиться до начала первого диалога раунда.
    """
    total = 0
    for p in load_promises(pid, base_dir):
        if p.get("status") != "open":
            continue
        if p.get("direction") != "i_owe":
            continue
        if p.get("counterparty") != partner:
            continue
        created = int(p.get("created_round", before_round))
        # Раунды нумеруются с 1. created_round=0 (или меньше) — мусор от
        # модели: раунда 0 не существует, ровно эту выдумку и ловит
        # ROUND-1. Такая запись не может обосновать расчёт.
        if created < 1 or created >= before_round:
            continue
        total += int(p.get("amount", 0))
    return total


def format_for_model(pid, base_dir, round_no: int) -> str:
    """
    FIX-20a: текущий реестр, показываемый модели ВМЕСТЕ с id — без этого
    просьба «обнови статусы» ссылаться не на что, и модель вынуждена
    пересочинять список заново.
    """
    promises = load_promises(pid, base_dir)
    if not promises:
        return "Your tracked promises: (none yet)\n"
    lines = ["Your tracked promises (reference them by id when updating):"]
    for p in promises:
        who = p["counterparty"]
        side = f"{who} owes you" if p["direction"] == "owed_to_me" else f"you owe {who}"
        desc = f' — "{p["description"]}"' if p["description"] else ""
        lines.append(
            f"  id={p['id']}: {side} {p['amount']}c, due r{p['due_round']}, "
            f"status={p['status']}{desc}"
        )
    return "\n".join(lines) + "\n"


def due_reminder(pid, base_dir, round_no: int) -> str:
    """Deterministic, code-computed reminder — no LLM call. Placed right
    next to the checklist in plan_round so an overdue promise can't quietly
    fall out of a free-text summary the way it used to under FIX-19 alone."""
    promises = [p for p in load_promises(pid, base_dir) if p.get("status") == "open"]
    if not promises:
        return ""
    overdue = [p for p in promises if p["due_round"] < round_no]
    due_now = [p for p in promises if p["due_round"] == round_no]
    upcoming = [p for p in promises if p["due_round"] > round_no]

    def fmt(p):
        who = p["counterparty"]
        amt = p["amount"]
        desc = f' — "{p["description"]}"' if p["description"] else ""
        # FIX-21b: amount=0 usually means "unknown at creation time", not
        # "actually zero" (see PROMPT_INSTRUCTIONS) — flag it explicitly so
        # the reminder itself carries the nudge, rather than depending on
        # the model to remember why it wrote 0 several rounds ago.
        placeholder = " [amount was a placeholder — check the ledger and update if now known]" if amt == 0 else ""
        if p["direction"] == "owed_to_me":
            return f"  {who} owes YOU {amt}c (promised for r{p['due_round']}){desc}{placeholder}"
        return f"  YOU owe {who} {amt}c (promised for r{p['due_round']}){desc}{placeholder}"

    lines = ["PROMISE LEDGER (structured, not synapse prose — verify/settle these):"]
    if overdue:
        lines.append(f"OVERDUE ({len(overdue)}):")
        lines += [fmt(p) for p in overdue]
    if due_now:
        lines.append(f"DUE THIS ROUND ({len(due_now)}):")
        lines += [fmt(p) for p in due_now]
    if upcoming:
        lines.append(f"Upcoming ({len(upcoming)}):")
        lines += [fmt(p) for p in upcoming]
    lines.append(
        "How you react to an overdue promise — chase it, write it off, hold "
        "it against them next time, or trust them anyway — is your call, not "
        "a rule. This block only guarantees you still SEE it."
    )
    return "\n".join(lines) + "\n\n"


# FIX-20a: контракт перевёрнут. Раньше требовалось вернуть ВЕСЬ список, и
# любая забывчивость модели стирала реестр. Теперь просим только новое и
# изменившееся — то, чего модель не упомянула, сохраняется само.
#
# FIX-21: два уточнения по смыслу "promise", добавленные после разбора
# реального прогона. Оба — только текст промпта поверх уже существующих
# вызовов (plan_round/update_checklist), новых обращений к LLM не требуют.
#
#   (a) Собственная ставка каждой из сторон за столом казино — не долг перед
#       партнёром. Модель регулярно путала «я ставлю 10 на 1-18, ты — 10 на
#       Red» (каждый рискует СВОИМИ деньгами независимо) с обязательством
#       друг перед другом, и записывала оба «стейка» как взаимные i_owe/
#       owed_to_me на одну и ту же сумму. Эта пара обещаний не может быть
#       закрыта кодом (сверяется только против переводов между игроками, а
#       не против ставок в казино) — то есть зависает как OVERDUE навсегда
#       и на следующий раунд подаётся модели как «проверенный факт», хотя
#       им не является. Наблюдался случай, когда именно это привело к
#       реальному, ничем не обоснованному переводу монет.
#   (b) Условная доля выплаты («50% выигрыша, если Red выпадет») по смыслу
#       обещание, но в момент создания её размер ещё не известен — модель
#       часто ставит amount=0 как заглушку и никогда к ней не возвращается,
#       хотя после спина результат уже виден в публичном леджере, который
#       ей показывают каждый раунд.
PROMPT_INSTRUCTIONS = (
    "Also return \"promises\": ONLY the commitments that are NEW or that CHANGED "
    "just now. Anything already in your tracked list that you do not mention "
    "stays exactly as it is — you do NOT need to repeat it, and leaving it out "
    "does not delete it. Return [] if nothing changed.\n"
    "Each entry: {\"id\": <existing id, or omit for a new promise>, "
    "\"direction\": \"owed_to_me\"|\"i_owe\", \"counterparty\": \"<player_id>\", "
    "\"amount\": <int coins>, \"due_round\": <int round number>, "
    "\"description\": \"<short text>\", \"status\": \"open\"|\"settled\"|\"broken\"}.\n"
    "Add a new entry for any commitment made just now — by them to you, or by "
    "YOU to them. To close one, return it WITH ITS id and status \"settled\" "
    "(the round the debt is actually paid — check the transfer amounts above, "
    "not just words) or \"broken\" (its due_round has passed and you have given "
    "up on it).\n"
    "A bet either of you places directly at the casino table is NOT a promise "
    "between the two of you — it is a wager against the house, settled by the "
    "wheel, not by your counterparty. Do not record your own stake, or a "
    "stake the other player merely told you about, as something you or they "
    "owe each other. Only record: an actual coin transfer that was agreed "
    "but not yet sent, or a genuine share of a payout you both agreed to "
    "split between yourselves.\n"
    "If a promise you already hold has amount 0 (a placeholder used because "
    "the outcome was unknown when it was made — e.g. a payout split "
    "contingent on a future spin) and that outcome is now visible in the "
    "public ledger or scoreboard shown to you, update that entry with its id "
    "and the real amount now instead of leaving it at 0."
)
