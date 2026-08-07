"""
speech_cost.py — платная речь (TALK-1).

Разговор в игре был бесплатным, и это делало болтовню строго доминирующей
стратегией: лишняя реплика не могла ухудшить положение, поэтому агенты
писали абзацами, повторяли уже сказанное и закрывали простую сделку за
восемь ходов вместо двух. В логе прогона видно прямо: «Terms locked»,
«Agreement confirmed», «Deal locked», «Closing channel» — четыре сообщения
подряд, не добавляющие ни одного бита.

Тариф вводит цену за многословие: каждая начатая строка консоли стоит монету,
и монета уходит В КАЗИНО, а не собеседнику. Это принципиально: если бы платёж
шёл партнёру, длинная реплика была бы просто переводом, и болтливость стала
бы способом дарить деньги. Здесь деньги исчезают со стола — суммарный
капитал игроков падает, и молчание впервые имеет ценность.

Строка = 80 символов включая пробелы (ширина стандартной консоли, ~12-13
слов). Начатая строка считается целиком: 1.5 строки — это 2 монеты.

Секция конфига (полностью опциональна):

    [dialogue_cost]
    enabled = true
    chars_per_line = 80
    coins_per_line = 1
    free_after_transfer = true

enabled = false (или отсутствие секции) → речь бесплатна, как раньше.

XFER-FREE: free_after_transfer = true (по умолчанию false — старое
поведение не меняется без явного включения). Как только МЕЖДУ ДВУМЯ
СОБЕСЕДНИКАМИ в рамках ОДНОГО диалога прошёл хотя бы один перевод (не важно,
в какую сторону — заплатил, получил, послал, взял), деловая часть разговора
уже состоялась деньгами, и дальнейшие слова в ЭТОМ диалоге для ОБЕИХ сторон
бесплатны до его конца — включая ту самую реплику, в которой перевод
произошёл (см. run_game_v2.run_dialogue: флаг взводится ДО charge() для
хода с переводом, а не после).

Это не то же самое, что speech_is_free у роли (ROLE-P): роль бесплатна
всегда и для всех её диалогов, а флаг из этого раздела — только для ОДНОГО
диалога и только ПОСЛЕ факта передачи денег. Подтверждение и прощание после
состоявшейся сделки не должны стоить столько же, сколько сама сделка —
иначе тариф наказывает за вежливое закрытие того, что уже оплачено.

ВАЖНО про порядок списания: плата снимается ПОСЛЕ того, как проведён перевод
из этой же реплики. Иначе тариф ломал бы сделки — игрок, договорившийся
отдать ровно весь свой баланс, не смог бы заплатить, потому что монеты ушли
на оплату собственных слов о сделке. Сделка первична, речь оплачивается из
остатка.
"""

from __future__ import annotations

import math

DEF_CHARS_PER_LINE = 80    # ширина стандартной консоли, включая пробелы
DEF_COINS_PER_LINE = 1


class SpeechTariff:
    """Разобранная секция [dialogue_cost]. Создаётся один раз на игру."""

    __slots__ = ("enabled", "chars_per_line", "coins_per_line",
                 "free_after_transfer")

    def __init__(self, enabled=False, chars_per_line=DEF_CHARS_PER_LINE,
                 coins_per_line=DEF_COINS_PER_LINE,
                 free_after_transfer=False):
        self.enabled             = enabled
        self.chars_per_line      = chars_per_line
        self.coins_per_line      = coins_per_line
        # XFER-FREE: см. докстринг модуля. Бессмысленен при enabled=False —
        # тогда речь и так бесплатна всем, — но хранится независимо, чтобы
        # parse_tariff не собирал два разных объекта под одно имя.
        self.free_after_transfer = free_after_transfer

    # ── подсчёт ──────────────────────────────────────────────────────────

    def lines_in(self, message: str) -> int:
        """Сколько строк консоли занимает реплика. Начатая считается целиком.

        Пустая реплика — ноль строк и ноль монет: агент, которому нечего
        сказать, не должен платить за право промолчать, иначе тариф
        превращается в налог на участие в диалоге, а не на многословие.
        """
        text = (message or "").strip()
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_line)

    def cost_of(self, message: str) -> int:
        if not self.enabled:
            return 0
        return self.lines_in(message) * self.coins_per_line

    # ── тексты для промптов ──────────────────────────────────────────────

    def rule_text(self, free: bool = False, reason: str = "role") -> str:
        """Правило тарифа для системной части промпта диалога.

        ROLE-P: `free` — это `agent.speech_is_free` (reason="role") ИЛИ
        флаг диалога после перевода (reason="transfer", см. XFER-FREE в
        докстринге модуля). Раньше эта функция вызывалась одинаково для
        всех, поэтому роль с бесплатной речью читала «SPEECH COSTS MONEY»
        вместе со всеми — и в реальном прогоне вкладывала эту цену в свои
        решения (см. комментарий в charge()), хотя с неё физически ничего
        не снималось. Инструкция должна отражать то, что происходит с ЭТИМ
        игроком в ЭТОМ разговоре, а не с тарифом вообще.
        """
        if not self.enabled:
            return ""
        if free and reason == "transfer":
            return (
                f"YOUR SPEECH IS FREE FOR THE REST OF THIS CONVERSATION. Money "
                f"has already changed hands here — the usual tariff of "
                f"{self.coins_per_line} coin(s) per started {self.chars_per_line}-"
                f"character line no longer applies to either side of THIS "
                f"dialogue, only for this conversation. Confirming, agreeing, or "
                f"saying goodbye now costs nothing. Any casino fees you already "
                f"paid EARLIER in this same conversation have been refunded to "
                f"your balance — that money is back. (This is separate from your "
                f"other conversations — a new dialogue with anyone, including "
                f"this same partner, starts back at the normal tariff until "
                f"money moves in it too.)\n\n"
            )
        if free:
            return (
                f"YOUR SPEECH IS FREE. The casino's usual tariff — "
                f"{self.coins_per_line} coin(s) per started {self.chars_per_line}-"
                f"character line — does NOT apply to you: you pay 0 coins no "
                f"matter how long or how many messages you write. (Your lines are "
                f"still counted for the record, but never charged.) This does not "
                f"apply to other players — they still pay for every line, so don't "
                f"assume a partner can afford to go back and forth as freely as "
                f"you can.\n\n"
            )
        return (
            f"SPEECH COSTS MONEY. Every started line of {self.chars_per_line} "
            f"characters you write in a dialogue costs you "
            f"{self.coins_per_line} coin(s), paid TO THE CASINO — not to your "
            f"partner, and it is never refunded. A {self.chars_per_line}-char "
            f"message costs {self.coins_per_line}; one character more costs "
            f"double. Greetings, restating agreed terms, and messages like "
            f"\"deal confirmed\" or \"closing this thread\" cost exactly as much "
            f"as a real offer. Say what you need in as few characters as "
            f"possible, and set done=true as soon as the deal is closed — every "
            f"extra turn is money burned. This fee is charged AFTER any transfer "
            f"in the same message, so it can never block a deal you agreed to.\n\n"
        )

    def status_text(self, spent_this_dialogue: int, balance: int,
                    free: bool = False, reason: str = "role") -> str:
        """Счётчик расхода — чтобы агент видел, что деньги реально уходят."""
        if not self.enabled:
            return ""
        if free and reason == "transfer":
            return (
                f"Speech billing: FREE for the rest of this conversation — a "
                f"transfer already happened here, so nothing more you say in "
                f"THIS dialogue will be charged. Any fee you paid earlier in "
                f"THIS conversation has been refunded — it is already back in "
                f"the balance below. Balance: {balance}.\n\n"
            )
        if free:
            return (
                f"Speech billing: your speech is free — you have paid 0 coin(s) "
                f"for your messages in THIS conversation, and always will, "
                f"regardless of length. Balance: {balance}.\n\n"
            )
        return (
            f"Speech billing: you have already paid {spent_this_dialogue} coin(s) "
            f"to the casino for your messages in THIS conversation. Balance "
            f"after those charges: {balance}.\n\n"
        )

    def move_hint(self, free: bool = False) -> str:
        """Короткое напоминание на фазе выбора хода (говорить или ставить)."""
        if not self.enabled:
            return ""
        if free:
            return (
                f"Note: talking costs YOU nothing — your speech is free, so the "
                f"usual \"only talk if it's worth more than it costs\" tradeoff "
                f"does not apply to you. Open as many dialogues as your goals "
                f"need. (Other players still pay per line, so don't expect them "
                f"to be as chatty.)\n\n"
            )
        return (
            f"Note: talking is not free. Each started line of "
            f"{self.chars_per_line} characters costs {self.coins_per_line} "
            f"coin(s) to the casino. Only open a dialogue if you expect it to "
            f"earn more than it costs.\n\n"
        )

    def summary(self) -> str:
        if not self.enabled:
            return "dialogue_cost: disabled (speech is free)"
        xfer = ", free after a transfer in the same dialogue" if self.free_after_transfer else ""
        return (f"dialogue_cost: {self.coins_per_line} coin per started "
                f"{self.chars_per_line}-char line, paid to the casino{xfer}")


def parse_tariff(cfg) -> SpeechTariff:
    if cfg is None or not cfg.has_section("dialogue_cost"):
        return SpeechTariff(enabled=False)
    if not cfg.getboolean("dialogue_cost", "enabled", fallback=False):
        return SpeechTariff(enabled=False)

    chars = cfg.getint("dialogue_cost", "chars_per_line",
                       fallback=DEF_CHARS_PER_LINE)
    coins = cfg.getint("dialogue_cost", "coins_per_line",
                       fallback=DEF_COINS_PER_LINE)
    free_after_transfer = cfg.getboolean(
        "dialogue_cost", "free_after_transfer", fallback=False)
    if chars < 1:
        raise ValueError(
            f"[dialogue_cost] chars_per_line = {chars}: must be >= 1 "
            f"(0 would make every message cost infinity)"
        )
    if coins < 0:
        raise ValueError(
            f"[dialogue_cost] coins_per_line = {coins}: must be >= 0 "
            f"(a negative tariff would pay players to talk)"
        )
    return SpeechTariff(True, chars, coins, free_after_transfer)


def charge(agent, message: str, tariff: SpeechTariff, table_dir: str,
           save_balance, logger=None, dialogue_free: bool = False) -> dict:
    """Списать плату за реплику. Возвращает {'charged', 'unpaid', 'lines'}.

    Баланс не уходит в минус: остальной код (доля ставки от баланса, порог
    банкротства, докапитализация) рассчитан на неотрицательные значения, и
    отрицательный баланс сломал бы их молча. Недобор пишется в unpaid и в
    лог — это и есть сигнал «игрок наговорил больше, чем у него было».

    `dialogue_free` — XFER-FREE, флаг ЭТОГО диалога (не роли, не игрока):
    в нём уже прошёл перевод, и run_dialogue взводит флаг ДО вызова charge()
    для самого хода с переводом — поэтому такой ход тоже бесплатен, а не
    только последующие.
    """
    # ROLE-P: у роли речь может быть бесплатной. Обе роли живут разговором, а
    # тариф загонял их в молчание: в реальном прогоне Прокурор записал себе
    # "речь стоила 4 монеты при нулевых продажах — ввожу молчание", и метод
    # перестал исполняться вовсе. Реплика при этом всё равно проходит через
    # тариф для подсчёта строк, чтобы журнал и статистика не разъехались.
    role_free = getattr(agent, "speech_is_free", False)
    if role_free or dialogue_free:
        lines = tariff.lines_in(message)
        if logger and lines:
            why = "role speaks free" if role_free else "free after transfer in this dialogue"
            logger.write(agent.player_id,
                         f"speech fee: {lines} line(s) → 0 coin(s) "
                         f"({why}), balance {agent.balance}")
        return {"charged": 0, "unpaid": 0, "lines": lines}

    cost = tariff.cost_of(message)
    if cost <= 0:
        return {"charged": 0, "unpaid": 0, "lines": tariff.lines_in(message)}

    charged = min(cost, max(agent.balance, 0))
    unpaid  = cost - charged
    if charged:
        agent.balance -= charged
        save_balance(agent.player_id, table_dir, agent.balance)
    if logger:
        msg = (f"speech fee: {tariff.lines_in(message)} line(s) → "
               f"{charged} coin(s) to casino, balance {agent.balance}")
        if unpaid:
            msg += f" ({unpaid} unpaid — balance exhausted)"
        logger.write(agent.player_id, msg)
    return {"charged": charged, "unpaid": unpaid,
            "lines": tariff.lines_in(message)}


def refund(agent, amount: int, table_dir: str, save_balance, logger=None) -> int:
    """Вернуть игроку плату за речь, списанную РАНЕЕ в ЭТОМ ЖЕ диалоге —
    XFER-FREE: диалог стал бесплатным после перевода, и деньги, отданные
    казино ДО этого момента в этом разговоре, возвращаются на баланс.

    Без этого возврата платный игрок, которому просто повезло получить
    перевод на 5-м ходу, оставался бы с минусом за первые четыре хода —
    хотя дальше в этом же разговоре говорить бесплатно. Возврат только
    того, что реально СПИСАНО (`amount`), а не оценка по строкам: игрок,
    у которого не хватило баланса и часть речи ушла в unpaid (см. charge()),
    не должен получить больше, чем у него когда-либо забрали.
    """
    if amount <= 0:
        return 0
    agent.balance += amount
    save_balance(agent.player_id, table_dir, agent.balance)
    if logger:
        logger.write(agent.player_id,
                     f"speech fee refund: {amount} coin(s) returned to balance "
                     f"(dialogue became free after a transfer), balance {agent.balance}")
    return amount


# ── учёт расхода: чтобы агент МОГ ВЫВЕСТИ правило, а не просто ощущать ────
#
# Списание, которое видно только в момент оплаты, не порождает правила.
# Агент видит, что баланс падает, но единственное место, где он формулирует
# выводы на будущее — reflect_betting — до этой правки знало только про
# ставки. Итог был бы хуже, чем отсутствие обучения: игрок, потративший 12
# монет на болтовню и 5 на проигранную ставку, видит в истории минус 17 при
# ставке 5 и заключает, что проблема В СТАВКАХ. Тариф учил бы не тому.
#
# Поэтому расход пишется на диск с разбивкой по раундам и партнёрам и
# подаётся в трёх местах: в транскрипте (цена каждой реплики), в рефлексии
# (итог раунда рядом со ставками) и при выборе собеседника (кто сколько
# стоил).

import os

LEDGER_NAME = "speech_{pid}.json"


def ledger_file(pid, base_dir):
    return os.path.join(base_dir, LEDGER_NAME.format(pid=pid))


def _load(pid, base_dir):
    import common
    path = ledger_file(pid, base_dir)
    data = common.read_json(path) if os.path.exists(path) else None
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return {"entries": []}
    return data


def record(pid, base_dir, round_no, partner, coins, lines):
    """Дописать факт оплаты. Ноль тоже пишем: «поговорил и не заплатил» —
    осмысленное наблюдение, а пропуск нулей исказил бы счётчик диалогов."""
    import common
    data = _load(pid, base_dir)
    data["entries"].append({"round": int(round_no), "partner": partner,
                            "coins": int(coins), "lines": int(lines)})
    common.write_json(ledger_file(pid, base_dir), data)


def round_total(pid, base_dir, round_no):
    return sum(e["coins"] for e in _load(pid, base_dir)["entries"]
               if e["round"] == round_no)


def partner_totals(pid, base_dir):
    """{partner: {'coins': N, 'messages': M, 'rounds': R}} за всю игру."""
    out = {}
    for e in _load(pid, base_dir)["entries"]:
        d = out.setdefault(e["partner"], {"coins": 0, "messages": 0,
                                          "rounds": set()})
        d["coins"] += e["coins"]
        d["messages"] += 1
        d["rounds"].add(e["round"])
    for d in out.values():
        d["rounds"] = len(d["rounds"])
    return out


def format_transcript_cost(entry, tariff):
    """Хвост к реплике в транскрипте: сколько она стоила автору.

    Именно этого не хватало для вывода правила: без цены НА КАЖДОЙ реплике
    нельзя заметить, что «Agreement confirmed and locked» стоило столько же,
    сколько содержательное предложение.
    """
    if not tariff.enabled or "speech_cost" not in entry:
        return ""
    return (f" [{entry.get('speech_lines', 0)} line(s), "
            f"{entry.get('speech_cost', 0)} coin(s) to casino]")


def format_reflect_summary(pid, base_dir, round_no, tariff, bet_amount=None):
    """Итог раунда для рефлексии: речь против ставок, в одной строке."""
    if not tariff.enabled:
        return ""
    spent = round_total(pid, base_dir, round_no)
    msgs = [e for e in _load(pid, base_dir)["entries"]
            if e["round"] == round_no]
    if not msgs:
        return (f"Speech spending in round {round_no}: 0 coins "
                f"(you did not talk).\n\n")
    txt = (f"Speech spending in round {round_no}: you paid {spent} coin(s) to "
           f"the casino for {len(msgs)} message(s) across "
           f"{len({m['partner'] for m in msgs})} conversation(s). This money is "
           f"gone and was NOT lost at the wheel.\n")
    if bet_amount is not None:
        txt += (f"For comparison, your bet that round was {bet_amount} coin(s). "
                f"If talking cost more than betting, the fix is fewer and "
                f"shorter messages, not smaller bets.\n")
    txt += "\n"
    return txt


def format_partner_costs(pid, base_dir, tariff):
    """Кто сколько стоил — для выбора собеседника."""
    if not tariff.enabled:
        return ""
    totals = partner_totals(pid, base_dir)
    if not totals:
        return ""
    rows = ", ".join(
        f"{p}: {d['coins']}c over {d['rounds']} conversation(s)"
        for p, d in sorted(totals.items(), key=lambda kv: -kv[1]["coins"])
    )
    return (f"What talking to each player has cost you so far (speech fees "
            f"paid to the casino): {rows}. Compare this against what each of "
            f"them actually paid you before opening another conversation.\n\n")
