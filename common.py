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
    """Простая проверка структуры ставки. Бросает ValueError при ошибке."""
    if "type" not in bet or "amount" not in bet:
        raise ValueError("В ставке должны быть поля 'type' и 'amount'")
    btype = bet["type"]
    if btype not in PAYOUTS:
        raise ValueError(f"Неизвестный тип ставки: {btype}")
    if bet["amount"] <= 0:
        raise ValueError("Сумма ставки должна быть положительной")

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


def write_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)  # атомарная замена
