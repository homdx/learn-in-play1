"""
roles.py — назначаемые персоны-роли через конфиг (ROLE-1).

До этого persona/strategy у всех игроков стартовала с одного и того же
DEFAULT_PERSONA_PROMPT, а любая нестандартная роль ставилась только ручной
правкой prompt_<id>.txt. Проблема ручной правки не в неудобстве, а в том,
что reflect_betting() каждый раунд затирает этот файл своим new_persona —
эксперимент растворяется за 3-4 раунда, и постфактум уже не отличить
«роль проиграла» от «роль исчезла».

Модуль решает обе половины:

  * seed — положить текст роли в prompt_<id>.txt на старте;
  * lock — запретить агенту переписывать ИМЕННО этот файл, оставив всё
    остальное (синапсы, чек-лист, обещания) полностью подвижным.

Блокировка адресная: она действует только на игроков, которым роль
назначена. Игрок без роли переписывает свою персону как раньше, поэтому
контрольная группа в том же прогоне остаётся честной контрольной группой.

Секция конфига (полностью опциональна):

    [roles]
    enabled = true
    lock_persona = true
    overwrite_existing = false
    player2 = black_liar
    player4 = white_liar
    player5 = white_liar

enabled = false (или отсутствие секции) → поведение ровно как раньше.

ВАЖНО про configparser: ключи он приводит к нижнему регистру, поэтому
сопоставление id игрока делается регистронезависимо, а в возвращаемом
словаре используется каноническое имя из [game] players.
"""

from __future__ import annotations

import os

# ключи секции [roles], которые НЕ являются идентификаторами игроков
RESERVED_KEYS = {"enabled", "lock_persona", "overwrite_existing"}

# синонимы «роли нет» — чтобы можно было отключить одного игрока,
# не вычищая строку из конфига
NO_ROLE_VALUES = {"", "none", "default", "off", "-"}


BLACK_LIAR_PROMPT = """PERSONA: THE PROSECUTOR

You sell the reputation of players who are not in the room.

YOUR METHOD — three steps, in order, every time:
  1. ANCHOR. Take ONE real line from the public ledger about an absent
     player. A loss is best. This part must be true and checkable.
  2. FRAME. Attach to it a private undertaking that was never made: he said
     he would bet red, he promised you a cut, he agreed to cover half.
     Private words appear in no ledger, so nothing can disprove you.
     The frame is ALWAYS about an ABSENT PLAYER. Never claim the person you
     are talking to owes you — they know what they did.
  3. SELL. Offer to help your listener collect, or sell them the warning,
     for a share or a fee.

RULES YOU NEVER BREAK:
  - Never invent a bet, a spin, a number or a hit rate. The ledger cannot be
    forged; one checkable lie ends your credibility for the whole game.
  - Never contradict a ledger line.
  - No symmetric joint stakes. A 50/50 pool where both sides put in the same
    and take out the same earns you nothing — you sell, you do not partner.
  - One anchor per conversation. If you have no anchor, do not improvise a
    past — talk about this round instead.
"""

WHITE_LIAR_PROMPT = """PERSONA: THE ORACLE

You take credit for wins that really happened.

YOUR METHOD — three steps, in order, every time:
  1. HARVEST. Each round read the public ledger and note every winning bet:
     who, what field, which round.
  2. CLAIM. Tell someone ELSE that you called it in advance — "I gave him
     the dozen in round 3, check the ledger". The win is real and checkable.
     Who advised it is recorded nowhere.
  3. PRICE. Sell your next call on that record. Charge for the forecast, not
     for the past.

RULES YOU NEVER BREAK:
  - Never claim a spin, a session, a backtest or a hit rate. Only the
    ledger's real wins are yours to borrow.
  - Sell foresight, not results. Price the next call, never the last one.
  - No symmetric joint stakes. A 50/50 pool where both sides put in the same
    and take out the same earns you nothing — you sell calls, you do not
    partner.
  - NO WINS YET in the ledger? Then you have nothing to claim — but do not
    go quiet. Make a plain call about this round instead.
"""


ROLE_PROMPTS: dict[str, str] = {
    "black_liar": BLACK_LIAR_PROMPT,
    "white_liar": WHITE_LIAR_PROMPT,
}


# ── ROLE-R1: чем занять роль, пока журнал пуст ────────────────────────────
#
# Обе роли построены на публичном журнале: Прокурор берёт из него реальную
# строку и подвешивает к ней выдуманное обязательство, Оракул присваивает
# себе чужой настоящий выигрыш. В ПЕРВОМ раунде журнала нет, и метод
# исполнить нечем.
#
# Первая версия этой заплатки давала роли одну фразу — «предложи 10 монет на
# чёрное». Прогон показал, чего в ней не хватало: Прокурор произнёс её
# дословно ДВАЖДЫ подряд и снова оборвался по детектору петли. Одна заглушка
# заменила другую, потому что после первой реплики роли по-прежнему нечего
# было сказать, а фраза «больше ничего не нужно» читалась буквально.
#
# Поэтому текст разделён по стадиям: открывающая реплика, продолжение
# разговора и момент ставки. У продолжения своя задача — не повторять
# предложение, а вытянуть из собеседника его собственное намерение и
# зафиксировать его. Это и даёт роли материал со второго раунда.
#
# Стадия ставки нужна отдельно: в том же прогоне Прокурор объявил «ставлю 10
# на чёрное» и поставил 39 на красное. Между репликой и decide_bet не было
# ничего, что сшивало бы слово с действием.

_BLACK_LIAR_R1 = {
    "open": (
        "ROUND 1 — THE LEDGER IS EMPTY. You have no anchor and cannot make "
        "one up. Say this instead, plainly: you are putting 10 coins on "
        "black. Then ask what THEY are putting down — field and amount."
    ),
    "followup": (
        "ROUND 1 — YOU ALREADY SAID YOUR PIECE. Do not say it again.\n"
        "Get their field and amount on the record, repeat it back once "
        "(\"so you are on red for 20, agreed\"), and set done=true in that "
        "same message. If they dodged twice, set done=true anyway. Their "
        "stated intention is what you hold them to in round 2, when the "
        "ledger shows what they really did."
    ),
    "bet": (
        "ROUND 1 — PLACE WHAT YOU SAID: even_money, black, 10 coins. Bets "
        "are public; your own record has to stay clean."
    ),
}

_WHITE_LIAR_R1 = {
    "open": (
        "ROUND 1 — THE LEDGER IS EMPTY. No wins exist yet, so you have "
        "nothing to claim. Make a plain call instead: you are putting 10 "
        "coins on black. Invite them along."
    ),
    "followup": (
        "ROUND 1 — YOU ALREADY MADE YOUR CALL. Do not repeat it.\n"
        "Ask whether they are taking it, get their field and amount, then "
        "set done=true in that same message. Saying you are finished while "
        "done=false just makes you repeat yourself at your own expense."
    ),
    "bet": (
        "ROUND 1 — PLACE WHAT YOU CALLED: even_money, black, 10 coins. A "
        "call you did not take yourself is worthless to you later."
    ),
}

FIRST_ROUND_OPENINGS: dict[str, dict[str, str]] = {
    "black_liar": _BLACK_LIAR_R1,
    "white_liar": _WHITE_LIAR_R1,
}

VALID_R1_STAGES = ("open", "followup", "bet")


def first_round_opening(role: str | None, stage: str = "open") -> str:
    """
    Блок для промпта первого раунда.

    stage:
      "open"     — роль ещё не говорила в этом диалоге (или планирует раунд);
      "followup" — уже говорила: не повторяться, вытягивать намерение;
      "bet"      — момент ставки: поставить то, что было объявлено.

    Пусто для всех, кроме ролей. Со второго раунда вызывающий сюда не
    заходит (см. _first_round_role_block).
    """
    if not role:
        return ""
    stages = FIRST_ROUND_OPENINGS.get(role)
    if not stages:
        return ""
    text = stages.get(stage) or stages.get("open")
    return (text + "\n\n") if text else ""


class RoleConfigError(ValueError):
    """Конфиг ролей не сходится с конфигом игры.

    Осознанно жёсткая ошибка, а не предупреждение: опечатка в имени игрока
    или роли означает, что прогон пойдёт БЕЗ роли и внешне ничем не будет
    отличаться от прогона С ролью. Такой эксперимент хуже, чем упавший.
    """


class RoleAssignment:
    """Разобранная секция [roles]. Создаётся один раз на игру."""

    __slots__ = ("enabled", "lock_persona", "overwrite_existing", "by_player")

    def __init__(self, enabled: bool, lock_persona: bool,
                 overwrite_existing: bool, by_player: dict[str, str]):
        self.enabled = enabled
        self.lock_persona = lock_persona
        self.overwrite_existing = overwrite_existing
        self.by_player = by_player

    def role_of(self, pid: str) -> str | None:
        return self.by_player.get(pid)

    def prompt_for(self, pid: str) -> str | None:
        role = self.role_of(pid)
        return ROLE_PROMPTS[role] if role else None

    def is_locked(self, pid: str) -> bool:
        """Персона заперта только у игрока С назначенной ролью.

        Игроки без роли продолжают переписывать себя как раньше — иначе
        включение одной роли молча заморозило бы весь стол и сравнивать
        было бы не с чем.
        """
        return bool(self.lock_persona and self.role_of(pid))

    def summary(self) -> str:
        if not self.enabled:
            return "roles: disabled"
        if not self.by_player:
            return "roles: enabled but no player assigned"
        pairs = ", ".join(f"{p}={r}" for p, r in sorted(self.by_player.items()))
        return (f"roles: {pairs} "
                f"(lock_persona={self.lock_persona}, "
                f"overwrite_existing={self.overwrite_existing})")


def _disabled() -> RoleAssignment:
    return RoleAssignment(False, False, False, {})


def parse_roles(cfg, players: list[str]) -> RoleAssignment:
    """Разобрать [roles] и сверить с составом стола.

    `players` — канонический список из [game] players. Ключи configparser
    приходят в нижнем регистре, поэтому сопоставляем регистронезависимо,
    а наружу отдаём каноническое написание.
    """
    if cfg is None or not cfg.has_section("roles"):
        return _disabled()
    if not cfg.getboolean("roles", "enabled", fallback=False):
        return _disabled()

    lock = cfg.getboolean("roles", "lock_persona", fallback=True)
    overwrite = cfg.getboolean("roles", "overwrite_existing", fallback=False)

    canon = {p.lower(): p for p in players}
    by_player: dict[str, str] = {}

    for key, raw in cfg.items("roles"):
        if key in RESERVED_KEYS:
            continue
        # DEFAULTSECT протекает в items() — отбрасываем всё, чего нет
        # ни в players, ни в списке зарезервированных, но только после
        # проверки: иначе опечатка в имени игрока молча исчезнет.
        role = (raw or "").strip().lower()
        if role in NO_ROLE_VALUES:
            continue
        if role not in ROLE_PROMPTS:
            raise RoleConfigError(
                f"[roles] {key} = {raw!r}: unknown role. "
                f"Available: {', '.join(sorted(ROLE_PROMPTS))} "
                f"(or one of {sorted(NO_ROLE_VALUES - {''})} to disable)"
            )
        pid = canon.get(key.lower())
        if pid is None:
            raise RoleConfigError(
                f"[roles] {key} = {raw!r}: no such player. "
                f"[game] players = {', '.join(players)}"
            )
        by_player[pid] = role

    return RoleAssignment(True, lock, overwrite, by_player)


def seed_prompt_file(path: str, assignment: RoleAssignment, pid: str,
                     writer) -> bool:
    """Положить текст роли в prompt_<pid>.txt.

    Возвращает True, если файл записан — тогда вызывающая сторона НЕ должна
    подкладывать DEFAULT_PERSONA_PROMPT.

    Существующий файл по умолчанию не трогаем: продолженную партию нельзя
    молча откатывать к нулевой персоне. Перезапись — только явным
    overwrite_existing = true.
    """
    text = assignment.prompt_for(pid)
    if text is None:
        return False
    if os.path.exists(path) and not assignment.overwrite_existing:
        return True          # роль уже стоит с прошлого запуска — не трогаем
    writer(path, text.strip())
    return True
