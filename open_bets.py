"""
open_bets.py — открытые ставки текущего раунда (BET-1).

Раньше все ставки возникали одновременно, в отдельной фазе после того, как
отговорили ВСЕ игроки. Из-за этого внутри раунда не существовало ни одного
необратимого действия: любое заявление «поставлю 5 на чёрное» оставалось
словами вплоть до вращения, и проверить его можно было только постфактум,
когда реагировать уже поздно. Обещание, которое нельзя нарушить наблюдаемо,
не создаёт ни доверия, ни репутации — отсюда мёртвый ритуал взаимных
переводов «по 5 туда, по 5 обратно», державшийся три прогона подряд.

Теперь игрок ставит СРАЗУ после того, как закончил свои разговоры, и его
ставка становится видна всем, кто ходит после него. Появляется асимметрия
(первый ходит вслепую, последний видит всю картину) и, главное, появляется
фальсифицируемое обещание: пообещал в диалоге одно, поставил другое — и это
видно ДО того, как собеседник сделал свой ход.

Отдельного файла состояния не заводим. Ставка уже лежит на диске как
`bet_<pid>.json` с момента размещения и удаляется крупье после обработки
(croupier_v2.run_round → os.remove(bet_path)). Значит, множество
существующих bet-файлов — это ровно «ставки, сделанные в этом раунде и ещё
не сыгравшие». Это переживает Ctrl+C бесплатно и не может рассинхронизоваться
с реальностью, потому что источник один.

ВАЖНО про приватность: сумма и поле ставки публичны (как фишки на сукне),
а вот баланс игрока — нет; здесь мы его не раскрываем.
"""

from __future__ import annotations

import glob
import os
import re

import common


def _describe(bet: dict) -> str:
    """Человекочитаемое поле ставки: 'even_money(red)', 'straight(17)'."""
    sel = bet.get("selection")
    nums = bet.get("numbers")
    if sel:
        target = str(sel)
    elif nums:
        target = ",".join(str(n) for n in nums)
    else:
        target = "?"
    return f"{bet.get('type', '?')}({target})"


def read(base_dir: str) -> list:
    """
    [(pid, bet_dict), ...] — ставки, размещённые в этом раунде и ещё не
    обработанные крупье. Обход тот же, что у croupier_v2.collect_bets, чтобы
    оба модуля видели ровно одно и то же множество файлов. Битый или
    недописанный файл молча пропускаем: лучше не показать ставку, чем
    уронить раунд на полуслове.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(base_dir, "bet_*.json"))):
        m = re.match(r"bet_(.+)\.json$", os.path.basename(path))
        if not m:
            continue
        try:
            bet = common.read_json(path)
        except Exception:
            continue
        if isinstance(bet, dict) and bet.get("amount") is not None:
            out.append((m.group(1), bet))
    return out


def format_for_prompt(base_dir: str, self_pid: str = None) -> str:
    """
    Блок для промпта. Всегда возвращает текст с двумя переводами строки на
    конце, чтобы его можно было вклеивать в f-строки конкатенацией.

    Своя ставка показывается тоже — агент должен видеть, что он уже
    связан обязательством и изменить его не может.
    """
    placed = read(base_dir)
    header = (
        "Bets ALREADY PLACED this round (visible to everyone, like chips on "
        "the felt — these are FACTS, not claims, and they can no longer be "
        "changed):\n"
    )
    if not placed:
        return (header +
                "  (nobody has placed a bet yet this round — you are among the "
                "first to act, and you are acting blind)\n\n")

    lines = []
    for pid, bet in placed:
        mark = "  YOU: " if pid == self_pid else f"  {pid}: "
        lines.append(f"{mark}{_describe(bet)} for {bet.get('amount')} coins")
    tail = (
        "\nIf a player told you in conversation that they would bet one thing "
        "and the list above shows another, they lied to you — and you still "
        "have your own move to make.\n\n"
    )
    return header + "\n".join(lines) + "\n" + tail
