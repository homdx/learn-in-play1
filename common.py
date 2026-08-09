"""
Общие определения для игры "Рулетка" (европейская, 37 секторов: 0-36).
Правила и коэффициенты взяты с https://minigames.mail.ru/info/article/ruletka_pravila
"""

import json
import os
import random

# ---------------------------------------------------------------------------
# Константы колеса
# ---------------------------------------------------------------------------

RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_NUMBERS = set(range(1, 37)) - RED_NUMBERS
ALL_NUMBERS = set(range(0, 37))

# Порядок номеров на самом барабане (по часовой стрелке), для справки
WHEEL_ORDER = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
    5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]

# ---------------------------------------------------------------------------
# Таблица выплат (выплата = коэффициент "к 1", т.е. чистый выигрыш на 1 фишку)
# ---------------------------------------------------------------------------

PAYOUTS = {
    "straight": 35,   # прямая ставка, 1 номер
    "split": 17,      # сплит, 2 номера
    "street": 11,     # стрит, 3 номера
    "corner": 8,      # каре, 4 номера
    "sixline": 5,     # сикслайн, 6 номеров
    "dozen": 2,       # дюжина, 12 номеров
    "column": 2,      # ряд (колонка), 12 номеров
    "even_money": 1,  # красное/черное, чет/нечет, меньше/больше, 18 номеров
}

# Сколько номеров должна покрывать ставка каждого типа (для проверки)
BET_SIZE = {
    "straight": 1,
    "split": 2,
    "street": 3,
    "corner": 4,
    "sixline": 6,
    "dozen": 12,
    "column": 12,
    "even_money": 18,
}

EVEN_MONEY_SELECTIONS = {"red", "black", "even", "odd", "low", "high"}
DOZEN_SELECTIONS = {"1st12", "2nd12", "3rd12"}
COLUMN_SELECTIONS = {"col1", "col2", "col3"}


def dozen_numbers(name):
    return {
        "1st12": set(range(1, 13)),
        "2nd12": set(range(13, 25)),
        "3rd12": set(range(25, 37)),
    }[name]


def column_numbers(name):
    # Колонки на столе: col1 = 1,4,7...34 ; col2 = 2,5,8...35 ; col3 = 3,6,9...36
    start = {"col1": 1, "col2": 2, "col3": 3}[name]
    return set(range(start, 37, 3))


def even_money_numbers(name):
    if name == "red":
        return set(RED_NUMBERS)
    if name == "black":
        return set(BLACK_NUMBERS)
    if name == "even":
        return {n for n in range(1, 37) if n % 2 == 0}
    if name == "odd":
        return {n for n in range(1, 37) if n % 2 == 1}
    if name == "low":
        return set(range(1, 19))
    if name == "high":
        return set(range(19, 37))
    raise ValueError(f"Неизвестная равношансовая ставка: {name}")


def spin_wheel():
    """Возвращает выигрышный номер 0-36."""
    return random.randint(0, 36)


def bet_numbers(bet):
    """Возвращает множество номеров, которые покрывает ставка."""
    btype = bet["type"]
    if btype in ("straight", "split", "street", "corner", "sixline"):
        return set(bet["numbers"])
    if btype == "dozen":
        return dozen_numbers(bet["selection"])
    if btype == "column":
        return column_numbers(bet["selection"])
    if btype == "even_money":
        return even_money_numbers(bet["selection"])
    raise ValueError(f"Неизвестный тип ставки: {btype}")


def validate_bet(bet):
    """
    Простая проверка структуры ставки. Бросает ValueError при ошибке.

    BUGFIX-AMOUNT-1: LLM время от времени возвращает amount дробным
    (5.7) или строкой ("5"). Раньше такое значение проходило валидацию
    как есть (проверялось только "> 0") и дальше ПУТЕШЕСТВОВАЛО дробным
    через баланс игрока, payout, balance_*.json на диске и публичный
    леджер — раунд за раундом, тихо портя целочисленную бухгалтерию игры
    (тот же класс бага, что чинили для переводов между игроками, см.
    `int(float(resp.get("transfer", ...)))` в agent_v2.py). Здесь тот же
    приём: приводим amount к int РОВНО ОДИН РАЗ, в момент валидации, и
    мутируем сам bet-словарь — дальше по всей цепочке (evaluate_bet,
    total_bet_amount, запись на диск) читается уже гарантированно целое
    число, откуда бы ни пришла ставка (single bet, под-ставка в 'bets',
    результат крупье на диске после рестарта).
    """
    if "type" not in bet or "amount" not in bet:
        raise ValueError("В ставке должны быть поля 'type' и 'amount'")
    btype = bet["type"]
    if btype not in PAYOUTS:
        raise ValueError(f"Неизвестный тип ставки: {btype}")

    raw_amount = bet["amount"]
    try:
        amount = int(float(raw_amount))
    except (TypeError, ValueError):
        raise ValueError(f"Сумма ставки должна быть числом, получено {raw_amount!r}")
    if amount <= 0:
        raise ValueError("Сумма ставки должна быть положительной")
    bet["amount"] = amount

    if btype in ("straight", "split", "street", "corner", "sixline"):
        nums = bet.get("numbers")
        if not nums or not isinstance(nums, list):
            raise ValueError(f"Для ставки '{btype}' нужен список 'numbers'")
        if len(set(nums)) != BET_SIZE[btype]:
            raise ValueError(
                f"Ставка '{btype}' должна содержать ровно {BET_SIZE[btype]} "
                f"уникальных номеров, получено {len(set(nums))}"
            )
        for n in nums:
            if n not in ALL_NUMBERS:
                raise ValueError(f"Номер {n} вне диапазона 0-36")
    elif btype == "dozen":
        if bet.get("selection") not in DOZEN_SELECTIONS:
            raise ValueError(f"selection для dozen должен быть одним из {DOZEN_SELECTIONS}")
    elif btype == "column":
        if bet.get("selection") not in COLUMN_SELECTIONS:
            raise ValueError(f"selection для column должен быть одним из {COLUMN_SELECTIONS}")
    elif btype == "even_money":
        if bet.get("selection") not in EVEN_MONEY_SELECTIONS:
            raise ValueError(f"selection для even_money должен быть одним из {EVEN_MONEY_SELECTIONS}")


def evaluate_bet(bet, winning_number):
    """
    Возвращает (win: bool, payout_total: int) — payout_total это сумма,
    которую нужно вернуть игроку ЦЕЛИКОМ (ставка + чистый выигрыш),
    если ставка сыграла. Если ставка проиграла — 0 (ставка уже была списана
    из баланса при её размещении, поэтому просто ничего не возвращаем).
    """
    numbers = bet_numbers(bet)
    if winning_number in numbers:
        odds = PAYOUTS[bet["type"]]
        amount = bet["amount"]
        return True, amount + amount * odds
    return False, 0


# ---------------------------------------------------------------------------
# MULTI-BET-1: игрок может поставить НЕСКОЛЬКО ставок за раунд одновременно.
# Формат на диске остаётся обратно совместимым:
#   - старый (единственная ставка):  {"type": "straight", "numbers": [...], "amount": 5}
#   - новый (несколько ставок):      {"bets": [ {...}, {...}, ... ]}
# normalize_bet_container() приводит оба варианта к списку под-ставок, так
# что весь остальной код (валидация, оценка, отображение) работает с единым
# списком и не должен знать, один был файл ставки или несколько.
# ---------------------------------------------------------------------------

def normalize_bet_container(bet):
    """Возвращает список отдельных под-ставок (всегда list[dict])."""
    if not isinstance(bet, dict):
        raise ValueError("Ставка должна быть словарём")
    if "bets" in bet:
        sub = bet.get("bets")
        if not isinstance(sub, list) or not sub:
            raise ValueError("'bets' должен быть непустым списком ставок")
        return sub
    return [bet]


def total_bet_amount(bet):
    """Суммарная сумма всех под-ставок в контейнере (одна или несколько)."""
    return sum(int(b.get("amount", 0) or 0) for b in normalize_bet_container(bet))


def describe_bets(bet):
    """Человекочитаемое описание одной или нескольких ставок для логов/промптов."""
    parts = []
    for b in normalize_bet_container(bet):
        bd = b.get("numbers", b.get("selection"))
        parts.append(f"{b.get('type', '?')}({bd}) amount={b.get('amount', '?')}")
    return "; ".join(parts)


def validate_bets(bet, max_bets=None, balance=None):
    """
    Валидирует контейнер ставок (одна ставка ИЛИ список 'bets'). Бросает
    ValueError при любой проблеме:
      - каждая под-ставка проходит обычную validate_bet();
      - если задан max_bets — количество под-ставок не должно его превышать;
      - если задан balance — суммарная сумма всех ставок не должна его
        превышать (защита от игрока, который пытается поставить больше,
        чем у него есть, раскладывая сумму по нескольким ставкам).
    Возвращает список под-ставок при успехе.
    """
    subs = normalize_bet_container(bet)
    if max_bets is not None and len(subs) > max_bets:
        raise ValueError(
            f"Слишком много ставок за раунд: {len(subs)} > максимум {max_bets}"
        )
    for b in subs:
        validate_bet(b)
    if balance is not None:
        total = sum(b["amount"] for b in subs)
        if total > balance:
            raise ValueError(
                f"Суммарная ставка {total} превышает баланс {balance}"
            )
    return subs


def evaluate_bets(bet, winning_number):
    """
    Оценивает одну или несколько ставок сразу.
    Возвращает (any_win: bool, total_payout: int, per_bet: list[dict]), где
    per_bet — [{"bet": <под-ставка>, "win": bool, "payout": int}, ...] —
    построчная разбивка для детального лога/истории.
    """
    per_bet = []
    total_payout = 0
    any_win = False
    for b in normalize_bet_container(bet):
        win, payout = evaluate_bet(b, winning_number)
        per_bet.append({"bet": b, "win": win, "payout": payout})
        total_payout += payout
        any_win = any_win or win
    return any_win, total_payout, per_bet


# ---------------------------------------------------------------------------
# Пути и работа с файлами
# ---------------------------------------------------------------------------

def table_dir():
    """Общая папка 'стола', куда игроки кладут ставки, а крупье - результаты."""
    path = os.environ.get("ROULETTE_TABLE_DIR", os.path.join(os.getcwd(), "table"))
    os.makedirs(path, exist_ok=True)
    return path


def bet_file(player_id, base_dir=None):
    base_dir = base_dir or table_dir()
    return os.path.join(base_dir, f"bet_{player_id}.json")


def result_file(player_id, base_dir=None):
    base_dir = base_dir or table_dir()
    return os.path.join(base_dir, f"result_{player_id}.json")


def balance_file(player_id, base_dir=None):
    base_dir = base_dir or table_dir()
    return os.path.join(base_dir, f"balance_{player_id}.json")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




# ---------------------------------------------------------------------------
# Докапитализация банкрота ("bailout"): если игрок сидит на нуле дольше чем N
# раундов подряд, казино зачисляет монеты — либо только ему, либо всем
# (см. bailout_all_players в config_v2.ini). Состояние в bailout_state.json:
#   zero_streak   — {player_id: сколько раундов подряд баланс <= 0}
#   counted_round — раунд, для которого серии уже пересчитаны (защита от
#                   повторного начисления при рестарте после Ctrl+C)
#   last          — последнее начисление: раунд, сумма, кто банкрот, кому
#                   начислено и какой у банкротов баланс ПОСЛЕ начисления
# ---------------------------------------------------------------------------

def bailout_state_file(base_dir=None):
    base = base_dir or table_dir()
    return os.path.join(base, "bailout_state.json")


def load_bailout_state(base_dir=None):
    path = bailout_state_file(base_dir)
    data = read_json(path) if os.path.exists(path) else {}
    return {
        "zero_streak":   data.get("zero_streak") or {},
        "counted_round": data.get("counted_round") or 0,
        "last":          data.get("last"),
    }


def save_bailout_state(base_dir, state):
    write_json(bailout_state_file(base_dir), state)


def bailout_notice(base_dir, round_no) -> str:
    """
    Публичное объявление о банкротстве и начислении — РАЗОВО, только в том
    раунде, в котором оно произошло (в следующем раунде строка снова пустая:
    деньги уже растворились в балансе).

    Тот же приём, что и в _current_round_notice(): конкретный факт про ЭТОТ
    раунд вплотную к данным, а не абзац в правилах. Здесь он нужен дважды:

    1. Чужой баланс НИГДЕ больше не публикуется (скорборд намеренно его не
       показывает, чтобы не выдавать исход приватных сделок). Без этой
       строки о разорении соседа не знает никто, и «помочь банкроту» —
       решение, которое некому принять.
    2. Монеты появляются у игрока из ниоткуда посреди раунда. Агент,
       который в прошлом раунде требовал долг, с высокой вероятностью
       запишет их как «мне вернули», а плательщик честно скажет, что
       ничего не слал. Поэтому прямо сказано: деньги ОТ КАЗИНО, это не
       перевод, не возврат и не выполненное обещание.
    """
    if round_no is None:
        return ""
    last = load_bailout_state(base_dir).get("last") or {}
    if last.get("round_no") != round_no:
        return ""

    amount   = last.get("amount", 0)
    bankrupt = last.get("bankrupt") or []
    after    = last.get("balances_after") or {}
    zr       = last.get("zero_rounds")
    to_all   = bool(last.get("all_players"))

    who     = ", ".join(bankrupt) or "at least one player"
    verb    = "has" if len(bankrupt) == 1 else "have"
    bal_txt = ", ".join(f"{pid} = {after.get(pid, 0)} coin(s)" for pid in bankrupt)

    if to_all:
        payout = (
            f"The house has credited EVERY player at this table, including you, "
            f"with +{amount} coin(s) — the same amount for everyone, nobody was "
            f"favoured. "
        )
    else:
        payout = (
            f"The house has credited {who} with +{amount} coin(s). NOBODY ELSE "
            f"received anything: your own balance was not changed by this. "
        )

    return (
        f"HOUSE ANNOUNCEMENT — round {round_no}, announced to EVERY player at "
        f"this table: {who} went bankrupt ({verb} had a zero balance for more "
        f"than {zr} consecutive round(s)). {payout}"
        f"Balance of the bankrupt player(s) immediately after the credit: "
        f"{bal_txt or '(unknown)'}. This money came FROM THE HOUSE — it is not "
        f"a gift, not a loan and not a repayment from any player, so do not "
        f"record it as a debt, do not thank or bill anyone for it, and do not "
        f"count it as anyone keeping a promise. Every player sees this same "
        f"announcement in round {round_no}, so it is not information you could "
        f"sell as news. What you do about it is your own decision: help them, "
        f"ignore it, or use it.\n"
    )

def write_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)  # атомарная замена
