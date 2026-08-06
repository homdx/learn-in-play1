"""
run_game_v2.py — Casino orchestrator with:
  * Phase 0: Apply last round results + reflect
  * Phase 1: Dialogue phase (max 2 partners / max 4 turns each)
  * Phase 2: Place bets
  * Phase 3: Croupier spins wheel
  * Full logging to logs/

Run:
    python3 run_game_v2.py [--config config_v2.ini] [--rounds N] [--table-dir PATH]
                            [--start-round N]

Snapshots:
    After each round fully completes, its entire table state (balances,
    history, notes/checklists, promise/transfer ledgers, game_state.json)
    is copied to <table_dir>/snapshots/round_<N>/.

    --start-round N restores the snapshot of round N-1 into the table dir
    (wiping whatever is currently there, except snapshots/ itself) and then
    resumes play from round N onward, exactly as if the earlier rounds had
    just finished. Use --start-round 1 to reset to a completely fresh table.
"""

import argparse
import configparser
import shutil
import threading
import os
import json
import time

import common
import llm_pool
import open_bets
import promise_ledger
import roles
import speech_cost
import transfer_ledger
from agent_v2 import PlayerAgent, load_balance, load_history, save_balance
from croupier_v2 import run_round
from llm_client import LLMClient, LLMUnavailable
from game_logger import GameLogger


MAX_DIALOGUE_TURNS = 4

# XFER-1: сколько переводов одна сторона может сделать за один диалог.
# Двух хватает на любую честную сделку — заплатить и, если нужно, доплатить.
# Третий почти всегда означает, что стороны потеряли нить: в прогонах игрок
# трижды платил за одну и ту же "схему" (10+10+20) и не получил ничего.
# 0 или меньше — без ограничения.
MAX_TRANSFERS_PER_DIALOGUE = 2
# Safety cap: how many dialogues a single player may have in one round —
# either as the one who initiates (outgoing) or as the one being talked to
# (incoming). Tracked separately so a popular target doesn't accidentally
# eat into its own budget for initiating conversations.
MAX_DIALOGUES_PER_PLAYER = 3


def load_config(path):
    cfg = configparser.ConfigParser()
    if not os.path.exists(path):
        raise SystemExit(f"Config not found: {path}")
    cfg.read(path, encoding="utf-8")
    return cfg


def state_file(base_dir):
    return os.path.join(base_dir, "game_state.json")


def load_last_round(base_dir):
    path = state_file(base_dir)
    if os.path.exists(path):
        return common.read_json(path).get("last_completed_round", 0)
    return 0


def save_last_round(base_dir, round_no):
    common.write_json(state_file(base_dir), {"last_completed_round": round_no})


SNAPSHOTS_DIRNAME = "snapshots"


def snapshots_root(base_dir):
    return os.path.join(base_dir, SNAPSHOTS_DIRNAME)


def snapshot_dir(base_dir, round_no):
    return os.path.join(snapshots_root(base_dir), f"round_{round_no}")


def save_round_snapshot(base_dir, round_no, logger=None):
    """
    SNAP-1: копия ВСЕХ файлов состояния стола (балансы, история,
    промис-леджер, transfer-леджер, заметки/чеклисты, game_state.json
    и т.д.) сразу после того, как раунд round_no полностью завершён и
    save_last_round() уже записал его на диск.

    Снимок делается ПОСЛЕ save_last_round(), а не до — иначе при сбое
    между копированием и записью game_state.json снимок раунда N мог
    бы совпасть по имени с фактически недоигранным раундом N.

    Каталог snapshots/ сам не копируется (иначе рекурсия и раздувание
    на ровном месте с каждым раундом).
    """
    dest = snapshot_dir(base_dir, round_no)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(base_dir):
        if name == SNAPSHOTS_DIRNAME:
            continue
        src_path = os.path.join(base_dir, name)
        dst_path = os.path.join(dest, name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
    if logger:
        logger.write_global(f"Snapshot of round {round_no} saved to {dest}")


def restore_round_snapshot(base_dir, round_no):
    """
    SNAP-2: восстанавливает состояние стола из снимка round_no,
    перезаписывая ТЕКУЩИЕ файлы base_dir (кроме самого snapshots/).
    Используется --start-round: чтобы начать заново с раунда N,
    восстанавливаем снимок раунда N-1 (последний полностью
    доигранный раунд перед тем, откуда хотим продолжить) — балансы,
    заметки и леджеры становятся ровно такими, какими были сразу
    после round N-1, а дальше игра идёт как обычный resume.
    """
    src = snapshot_dir(base_dir, round_no)
    if not os.path.isdir(src):
        raise SystemExit(
            f"No snapshot found for round {round_no} at {src}. "
            f"Available snapshots: {sorted(os.listdir(snapshots_root(base_dir))) if os.path.isdir(snapshots_root(base_dir)) else '(none)'}"
        )
    # Clear current state files (except snapshots/) before restoring,
    # so stale files that didn't exist at snapshot time don't linger.
    for name in os.listdir(base_dir):
        if name == SNAPSHOTS_DIRNAME:
            continue
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        dst_path = os.path.join(base_dir, name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def dialogue_phase_state_file(base_dir):
    return os.path.join(base_dir, "dialogue_phase_state.json")


def load_dialogue_phase_state(base_dir, round_no):
    """
    Прогресс ФАЗЫ ДИАЛОГОВ внутри текущего раунда — сохраняется после
    КАЖДОГО завершённого диалога. Диалог, прерванный Ctrl+C посреди
    (деньги ещё не переведены / done не наступил), в это состояние не
    попадает и поэтому при рестарте просто начнётся заново с нуля —
    ровно то поведение, которое нужно.
    """
    path = dialogue_phase_state_file(base_dir)
    if not os.path.exists(path):
        return None
    data = common.read_json(path)
    if data.get("round_no") != round_no:
        return None
    return data


def save_dialogue_phase_state(base_dir, round_no, player_index, talked_to,
                              incoming_used, outgoing_used, dialogues_this_round):
    common.write_json(dialogue_phase_state_file(base_dir), {
        "round_no": round_no,
        "player_index": player_index,
        "talked_to": talked_to,
        "incoming_used": incoming_used,
        "outgoing_used": outgoing_used,
        "dialogues_this_round": dialogues_this_round,
    })


def clear_dialogue_phase_state(base_dir):
    path = dialogue_phase_state_file(base_dir)
    if os.path.exists(path):
        os.remove(path)


def settle_pending_results(agents, players, base_dir, round_no, logger,
                           reflect=True, checkpoint_round=None):
    """
    FIX-2: применяет все НЕОБРАБОТАННЫЕ result_<pid>.json (их пишет крупье
    в конце раунда) — начисляет выплату, дописывает историю и публичный
    журнал. Раньше это делалось ТОЛЬКО в Фазе 0 следующего раунда, поэтому
    выигрыш последнего раунда игры не зачислялся никогда: цикл кончался,
    файлы результатов оставались на диске, а save_last_round() уже не давал
    прогнать раунд повторно.

    FIX-5: если результата нет (игрок не ставил — например, баланс был 0),
    рефлексия всё равно вызывается. Иначе банкрот навсегда замораживал свою
    синапсу и персону ровно тогда, когда учиться нужнее всего.

    Возвращает число применённых результатов.
    """
    applied = 0
    # FIX-15: чекпойнт фазы. checkpoint_round=None → не сохранять (финальный
    # расчёт после конца игры, повторять его нечему).
    done = load_phase0_done(base_dir, checkpoint_round) if checkpoint_round else set()
    if done:
        logger.write_global(
            f"Phase 0 of round {checkpoint_round}: skipping {len(done)} player(s) "
            f"who already reflected before the restart ({', '.join(sorted(done))})."
        )
    # POOL-1: рефлексия каждого игрока независима — агент читает и пишет
    # только свои файлы. При пуле из нескольких серверов гоняем их разом;
    # при одном run_parallel выполняет ровно тот же цикл последовательно
    # и в том же порядке, поэтому прогоны остаются сравнимыми.
    #
    # Чекпойнт при этом обновляется ПОД ЗАМКОМ и после каждого игрока, а
    # не одним махом в конце: смысл FIX-15 в том, чтобы обрыв не заставил
    # рефлексировать повторно, и параллельность его не отменяет.
    lock = threading.Lock()
    counter = {"applied": applied}

    def _reflect_one(pid):
        agent = agents[pid]
        # TALK-2: номер рефлексируемого раунда передаём атрибутом, а не
        # аргументом: reflect_betting подменяется шпионами в существующих
        # тестах, и смена сигнатуры сломала бы их без всякой пользы.
        agent.current_round = round_no
        result_path = common.result_file(pid, base_dir)
        if os.path.exists(result_path):
            result = common.read_json(result_path)
            entry = agent.apply_result(result, round_no)
            os.remove(result_path)
            with lock:
                counter["applied"] += 1
            if reflect:
                agent.reflect_betting(entry)
        elif reflect and round_no >= 1:
            logger.write(pid, "no bet result for last round — reflecting anyway")
            agent.reflect_betting(None)
        # игрок полностью обработан — фиксируем на диске сразу, чтобы
        # следующий обрыв не заставил его рефлексировать повторно
        if checkpoint_round:
            with lock:
                done.add(pid)
                save_phase0_done(base_dir, checkpoint_round, done)

    todo = [pid for pid in players if pid not in done]
    llm_pool.run_parallel(
        [(lambda p=p: _reflect_one(p)) for p in todo],
        workers=_pool_workers(agents),
        on_error=lambda e: logger.write_global(f"Phase 0: player task failed ({e})"),
    )
    return counter["applied"]


def _pool_workers(agents) -> int:
    """Сколько задач гнать разом = сколько серверов в пуле.

    Больше потоков, чем серверов, не ускоряет: лишние всё равно встанут
    в очередь на аренду в LLMPool.acquire(). Меньше — недогружает пул.
    При одном сервере возвращает 1, и run_parallel уходит в обычный
    последовательный цикл, без threading вообще.
    """
    for agent in agents.values():
        pool = getattr(getattr(agent, "client", None), "pool", None)
        if pool is not None:
            return pool.size
    return 1


def round_player_order(players: list, round_no: int) -> list:
    """
    FIX-10: порядок хода в фазе диалогов, сдвигаемый на одного каждый раунд.

    Раунд 1 начинает player1, раунд 2 — player2, ... раунд 5 — player5,
    раунд 6 — снова player1. Порядок остальных сохраняется циклически:

        round 1: p1 p2 p3 p4 p5
        round 2: p2 p3 p4 p5 p1
        round 5: p5 p1 p2 p3 p4
        round 6: p1 p2 p3 p4 p5

    Зачем: список чужих диалогов (`dialogues_this_round`) наполняется ПО ХОДУ
    фазы, поэтому первый игрок видит его пустым, а последний — целиком. При
    фиксированном порядке player1 не видел ни одного чужого разговора НИ РАЗУ
    за всю партию и каждый раунд тратил свои диалоги вслепую, тогда как
    последний в списке всегда ходил с полной картиной. Это систематическое
    преимущество позиции, а не шум: оно смешивалось бы с результатом при
    любой попытке сравнить, чья персона оказалась удачнее.

    Ротация детерминирована и зависит только от round_no, поэтому порядок
    пересчитывается один в один при восстановлении после Ctrl+C (в
    dialogue_phase_state.json хранится индекс в ЭТОМ порядке).
    """
    if not players:
        return []
    offset = (round_no - 1) % len(players)
    return players[offset:] + players[:offset]


def phase0_state_file(base_dir):
    return os.path.join(base_dir, "phase0_state.json")


def load_phase0_done(base_dir, round_no):
    """
    FIX-15: кто уже отрефлексировал в Фазе 0 этого раунда.

    Раньше чекпойнт был только у фазы диалогов, поэтому любой обрыв внутри
    раунда заставлял всех игроков рефлексировать заново. В реальном прогоне
    раунд 2 запускался трижды из-за HTTP 504, и персон было переписано 14
    штук за два с половиной раунда — у одного игрока персона одного и того
    же раунда переписана дважды и оба раза длиннее. Это не только лишние
    пять вызовов LLM на каждый обрыв, но и лишний дрейф личности агента.
    """
    path = phase0_state_file(base_dir)
    if not os.path.exists(path):
        return set()
    data = common.read_json(path)
    if data.get("round_no") != round_no:
        return set()
    return set(data.get("done", []))


def save_phase0_done(base_dir, round_no, done):
    common.write_json(phase0_state_file(base_dir),
                      {"round_no": round_no, "done": sorted(done)})


def clear_phase0_state(base_dir):
    path = phase0_state_file(base_dir)
    if os.path.exists(path):
        os.remove(path)


def get_balances(base_dir, players):
    result = {}
    for pid in players:
        path = common.balance_file(pid, base_dir)
        if os.path.exists(path):
            result[pid] = common.read_json(path)["balance"]
        else:
            result[pid] = 0
    return result


def apply_bailout_if_needed(agents, players, base_dir, round_no, cfg, logger):
    """
    Докапитализация банкрота. Считает серии нулевых балансов и, если у игрока
    серия стала БОЛЬШЕ `bailout_zero_rounds`, зачисляет `bailout_amount`
    монет: только банкроту (по умолчанию) или всем, если включён
    `bailout_all_players`.

    Вызывается ПОСЛЕ Фазы 0 — выплаты прошлого раунда к этому моменту уже
    начислены, — и ДО фазы диалогов, чтобы деньги были на руках до первого
    разговора, а объявление легло в промпты того же раунда.

    Идемпотентность: counted_round помнит, за какой раунд серии уже
    пересчитаны, поэтому обрыв внутри раунда и повторный запуск не начислят
    вторую порцию.
    """
    if not cfg.getboolean("game", "bailout_enabled", fallback=True):
        return
    zero_rounds = cfg.getint("game", "bailout_zero_rounds", fallback=2)
    amount      = cfg.getint("game", "bailout_amount",      fallback=20)
    to_all      = cfg.getboolean("game", "bailout_all_players", fallback=False)

    state = common.load_bailout_state(base_dir)
    if state["counted_round"] == round_no:
        return                                   # уже считали в этом раунде

    balances = get_balances(base_dir, players)
    streak   = dict(state["zero_streak"])
    broke    = []
    for pid in players:
        if balances.get(pid, 0) <= 0:
            streak[pid] = streak.get(pid, 0) + 1
            if streak[pid] > zero_rounds:
                broke.append(pid)
        else:
            streak[pid] = 0

    if broke:
        recipients = list(players) if to_all else list(broke)
        detail = ", ".join(f"{pid} ({streak[pid]} round(s) at zero)" for pid in broke)
        scope  = "EVERY player" if to_all else "the bankrupt player(s) only"
        logger.write_global(
            f"BAILOUT: bankrupt — {detail}. House credits +{amount} coin(s) to "
            f"{scope} for round {round_no}."
        )
        after = {}
        for pid in recipients:
            new_balance = balances.get(pid, 0) + amount
            agents[pid].balance = new_balance
            save_balance(pid, base_dir, new_balance)
            logger.write(pid, f"BAILOUT +{amount} coins (balance "
                              f"{balances.get(pid, 0)} → {new_balance})")
            # серия обнуляется только тому, кто реально получил деньги
            streak[pid] = 0
            after[pid] = new_balance
        logger.write_balances(get_balances(base_dir, players))
        state["last"] = {
            "round_no": round_no,
            "amount": amount,
            "bankrupt": broke,
            "recipients": recipients,
            "balances_after": {pid: after.get(pid, balances.get(pid, 0))
                               for pid in broke},
            "zero_rounds": zero_rounds,
            "all_players": to_all,
        }

    state["zero_streak"]   = streak
    state["counted_round"] = round_no
    common.save_bailout_state(base_dir, state)


def _cap_dialogue_transfers(agent, pid, requested, made_so_far, limit, logger):
    """
    XFER-1: сколько переводов игрок может сделать В ОДНОМ диалоге.

    Ограничение по числу, а не по сумме — SPEND-1 уже держит сумму за раунд.
    Ловит другой паттерн: деньги ходят туда-сюда вокруг одного и того же
    пункта, каждый перевод сам по себе законен, а вместе они бессмысленны.

    Два реальных случая из прогонов. Первый: игрок трижды за диалог платил
    за одну и ту же "схему ставок" (10 + 10 + 20) и не получил ничего.
    Второй: тот, кто ТРЕБОВАЛ 30 монет, сам их и перевёл, а собеседник
    перевёл 30 обратно — шестьдесят монет прогнано за два хода при нулевом
    итоге. Модель теряет направление сделки, и чем больше у неё попыток,
    тем дороже это обходится.

    Два перевода на диалог хватает на любую честную сделку: заплатить и,
    если понадобилось, доплатить. Третий почти всегда означает, что стороны
    потеряли нить.
    """
    if limit <= 0 or made_so_far < limit:
        return requested
    logger.write(pid, f"transfer of {requested} coins BLOCKED — already made "
                      f"{made_so_far} transfer(s) in this dialogue (limit {limit})")
    return 0


def _role_privileges(cfg):
    """
    ROLE-P: привилегии ролевых игроков, каждая отдельным флагом.

    Обе роли по построению живут разговором: Прокурор продаёт репутацию
    отсутствующих, Оракул — авторство чужих выигрышей. К седьмому раунду
    реального прогона оба замолчали, и не от растерянности, а по расчёту:
    "речь стоила 4 монеты при нулевых продажах", "разговоры стоили 19 монет
    при нулевой отдаче". Тариф и лимит диалогов душат ровно ту механику,
    ради которой роли и вводились.

    Флаги врозь намеренно. `unlimited_outgoing` имеет неприятный побочный
    эффект: два лжеца с безлимитом съедают до восьми входящих слотов, а у
    трёх честных игроков их всего шесть — честные перестают доходить друг
    до друга. Если это проявится, отключается один флаг, остальные остаются.
    """
    if not cfg.has_section("roles"):
        return {"free_speech": False, "unlimited_outgoing": False,
                "ignore_partner_limit": False}
    return {
        "free_speech": cfg.getboolean("roles", "free_speech", fallback=False),
        "unlimited_outgoing": cfg.getboolean("roles", "unlimited_outgoing",
                                             fallback=False),
        "ignore_partner_limit": cfg.getboolean("roles", "ignore_partner_limit",
                                               fallback=False),
    }


def _is_role(agents, pid) -> bool:
    return bool(getattr(agents.get(pid), "role", None))


def _opening_allowance(agent, default=10) -> int:
    """Свободная сумма для открывающей реплики из [player] конфига агента."""
    cfg = getattr(agent, "cfg", None)
    if cfg is None:
        return default
    try:
        return cfg.getint("player", "opening_transfer_free", fallback=default)
    except Exception:
        return default


def _cap_opening_transfer(agent, pid, partner_id, requested, conversation,
                          base_dir, round_no, logger, free_allowance=10):
    """
    OPEN-1: перевод в САМОЙ ПЕРВОЙ реплике диалога ограничен размером
    открытого долга перед этим партнёром — либо небольшой свободной суммой,
    если долга нет. Что больше, то и разрешено.

    Зачем: в реальном прогоне игрок открыл разговор словами "обсудим за 50
    монет" и в этой же реплике перевёл половину капитала — до всякого
    ответа, когда согласовывать было ещё нечего. Ответом было "я
    стратегиями не торгую".

    Почему не запрет: расчёт по договорённости прошлого раунда выглядит
    ровно так же и совершенно законен — первая реплика это самое
    естественное место, чтобы заплатить по обещанию. Такие расчёты видно в
    реестре обещаний, и они проходят полностью.

    Почему остаётся свободная сумма: небольшой аванс или чаевые в открывающей
    реплике — осмысленный ход, и реестр обещаний ведёт сама модель, так что
    подлинная договорённость могла в него просто не попасть. Ограничение
    целится не в жест, а в его размер: десять монет — приглашение к сделке,
    пятьдесят из ста — разорение до первого ответа.

    Строго `created_round < round_no`: обещание, взятое в этом же раунде,
    не могло появиться раньше первого диалога раунда.
    """
    if conversation:
        return requested          # не первая реплика — не наше дело
    if requested <= 0:
        return requested
    debt = promise_ledger.open_debt_to(pid, base_dir, partner_id, round_no)
    allowed = max(debt, free_allowance)
    if requested <= allowed:
        return requested
    reason = (f"open promise to {partner_id} is {debt}c" if debt
              else f"no open promise to {partner_id} from an earlier round")
    logger.write(pid, f"opening-message transfer of {requested} coins CAPPED to "
                      f"{allowed} — {reason}, and nothing has been agreed in "
                      f"this dialogue yet")
    return allowed


def _cap_transfer(agent, pid, requested, logger):
    """
    SPEND-1: сколько игрок МОЖЕТ отдать в этом раунде сверх уже отданного.

    Ограничение на раунд, а не на реплику. В реальном прогоне игрок за один
    раунд, до единого вращения, отдал 135 монет из стартовых ста: 75 за
    "стратегию", которой не получил, 30 в погашение долга, которого не
    существовало, и 30 тому, кто прямо сказал, что ничего не утверждал.
    Каждый перевод по отдельности был в пределах баланса — суммарно это
    разорение за один раунд по несуществующим обязательствам.

    Порог берётся от баланса на НАЧАЛО раунда: иначе полученные в диалоге
    монеты тут же расширяли бы лимит, и цепочка "получил — отдал больше"
    воспроизводила бы ту же дыру.
    """
    limit = getattr(agent, "transfer_budget_this_round", None)
    if limit is None:
        return requested
    left = max(0, limit - getattr(agent, "sent_this_round", 0))
    if requested <= left:
        return requested
    if left == 0:
        logger.write(pid, f"transfer of {requested} coins BLOCKED — "
                          f"round transfer budget ({limit}) already spent")
    else:
        logger.write(pid, f"transfer of {requested} coins CAPPED to {left} — "
                          f"round transfer budget {limit}")
    return left


def _reflect_for_player(agent, pid, base_dir, prev_round, logger):
    """
    Рефлексия игрока о ПРЕДЫДУЩЕМ раунде — вызывается прямо перед его
    собственным ходом, а не общей пачкой в начале раунда.

    BET-2: до сих пор все пятеро рефлексировали до того, как хоть кто-то
    сходил, поэтому осмысление прошлого раунда происходило в пустом
    настоящем. Теперь игрок, выходящий третьим, обдумывает прошлый раунд,
    уже видя ставки первых двух в этом — и его выводы могут учитывать, что
    стол успел сделать.

    Начисление выплат и запись в журналы к этому моменту уже произошли
    (Phase 0 в начале раунда, для ВСЕХ сразу — балансы обязаны быть
    консистентны до первого разговора и до докапитализации). Здесь только
    LLM-осмысление, поэтому запись берём из личной истории: файл результата
    давно удалён, а после Ctrl+C его не восстановить.
    """
    agent.current_round = prev_round
    entry = None
    if prev_round >= 1:
        for e in reversed(load_history(pid, base_dir)):
            if e.get("round_no") == prev_round:
                entry = e
                break
        if entry is None:
            logger.write(pid, "no bet result for last round — reflecting anyway")
    agent.reflect_betting(entry)


def _transfer_outcome(agent, raw, after_balance, after_open, after_count,
                      final, delivered, intended_to, expected_to):
    """
    TVIS-1: почему перевод не прошёл целиком.

    Обрезки стоят цепочкой, и каждая молча уменьшала сумму. Агент видел
    только результат — точнее, не видел ничего: ноль в транскрипте не
    отображался вовсе. В прогоне это дало платежи-призраки: игрок с
    нулевым балансом писал "отправляю 5c", поле схлопывалось в ноль ещё
    на min(), и ОБЕ стороны записывали в синапсу состоявшуюся сделку и
    trust=10/10 при фактическом обороте 0.

    Причина определяется по первой ступени, которая срезала сумму —
    именно она и есть настоящее ограничение. Возвращает None, если
    запрошенного перевода не было или он прошёл полностью.
    """
    if raw <= 0:
        return None
    if delivered == raw:
        return None
    if intended_to != expected_to:
        return {"requested": raw, "delivered": 0, "reason": "wrong_recipient",
                "detail": f"указан получатель {intended_to!r}, а разговор с {expected_to!r}"}
    # Ступени проверяются С КОНЦА. Срезать могли несколько сразу: при
    # запросе 20 и балансе 15 сначала обрежет баланс, а до нуля доведёт
    # бюджет раунда. Назвать причиной первую ступень — значит подсказать
    # игроку неверное ограничение: он решит, что дело в сумме, уменьшит
    # её и снова упрётся. Решающая та, после которой получилось final.
    if final < after_count:
        limit = getattr(agent, "transfer_budget_this_round", None)
        spent = getattr(agent, "sent_this_round", 0)
        return {"requested": raw, "delivered": delivered, "reason": "round_budget",
                "detail": (f"бюджет переводов на раунд {limit}, уже отдано {spent}"
                           if limit is not None else "бюджет раунда исчерпан")}
    if after_count < after_open:
        return {"requested": raw, "delivered": delivered, "reason": "per_dialogue",
                "detail": "лимит переводов за разговор исчерпан"}
    if after_open < after_balance:
        return {"requested": raw, "delivered": delivered, "reason": "opening_cap",
                "detail": f"первая реплика: не больше {after_open}"}
    if after_balance < raw:
        return {"requested": raw, "delivered": delivered, "reason": "balance",
                "detail": f"на счету было {after_balance}"}
    return {"requested": raw, "delivered": delivered, "reason": "other",
            "detail": ""}


def _place_bet_for_player(agent, pid, base_dir, round_no, logger) -> bool:
    """
    Разместить ставку игрока. True — ставка на столе (только что
    поставлена или уже лежала), False — не поставлена.

    BET-1: вызывается сразу после того, как игрок закончил свои
    разговоры, чтобы следующие игроки видели его ставку.

    Порядок операций сохранён от FIX-3: сначала файл ставки, потом
    списание. Обрыв между ними даёт "ставка есть, деньги не сняты",
    и на рестарте проверка os.path.exists её пропустит. Обратный
    порядок означал бы чистую потерю монет.
    """
    if os.path.exists(common.bet_file(pid, base_dir)):
        logger.write(pid, "already has an unprocessed bet, skipping placement.")
        return True
    if agent.balance <= 0:
        logger.write(pid, "balance is 0, cannot bet.")
        return False

    bet = agent.decide_bet(round_no)
    common.write_json(common.bet_file(pid, base_dir), bet)
    agent.balance -= bet["amount"]
    save_balance(pid, base_dir, agent.balance)
    bd = bet.get("numbers", bet.get("selection"))
    logger.write(pid, f"bet: {bet['type']}({bd}) amount={bet['amount']} "
                      f"balance_after={agent.balance}")
    return True


# ─────────────────────────────────────────── dialogue orchestration ────────

def run_dialogue(agent_a: PlayerAgent, agent_b: PlayerAgent,
                 round_no: int, logger: GameLogger, table_dir: str) -> dict:
    """
    Run a dialogue between agent_a (initiator) and agent_b.
    Returns summary dict with transfers.
    """
    pid_a, pid_b = agent_a.player_id, agent_b.player_id
    logger.write_global(f"Dialogue begins: {pid_a} ↔ {pid_b}")

    conversation = []
    a_total_sent = 0
    b_total_sent = 0
    a_transfers  = 0        # XFER-1: число переводов, а не сумма
    b_transfers  = 0
    # TALK-1: плата за слова, уходящая казино. Счётчики обнуляются на каждый
    # диалог: агент должен видеть расход именно этого разговора, иначе
    # цифра быстро становится фоновой и перестаёт влиять на решение.
    tariff = getattr(agent_a, "tariff", None) or speech_cost.SpeechTariff()
    a_speech_cost = 0
    b_speech_cost = 0
    agent_a.speech_spent_this_dialogue = 0
    agent_b.speech_spent_this_dialogue = 0
    # FIX-9: кому осталось сделать один ЗАКРЫВАЮЩИЙ ход после чужого done.
    # Раньше `done` от любой стороны делал безусловный break, и оппонент не
    # получал хода вообще. Тот, кто ДОЛЖЕН заплатить, выигрывал от того, что
    # первым скажет "done": реплика "Deal, I agree" с done=true закрывала
    # согласованную сделку в ноль монет. Теперь оппоненту даётся ровно один
    # закрывающий ход — попрощаться или доплатить по уже согласованному,
    # но не начать новый торг (см. closing_turn в dialogue_turn).
    closing_for = None

    for turn in range(MAX_DIALOGUE_TURNS):
        # Agent A's turn
        a_is_closing = (closing_for == pid_a)
        turn_a = agent_a.dialogue_turn(
            partner_id=pid_b,
            partner_balance=agent_b.balance,
            conversation=conversation,
            round_no=round_no,
            is_initiator=(turn == 0),
            closing_turn=a_is_closing
        )
        # FIX-1: только ФАКТИЧЕСКИ проведённый перевод попадает в лог,
        # в conversation и в диалоговую синапсу. Раньше запись делалась
        # вне if-а, поэтому при transfer>0 с чужим/пустым transfer_to
        # деньги не двигались, но обе стороны видели "[+N coins]" —
        # фантомная сделка, отравлявшая репутацию.
        raw_a = max(0, int(turn_a.get("transfer") or 0))
        requested_a = min(raw_a, agent_a.balance)
        _a_after_balance = requested_a
        # OPEN-1: порядок важен — сначала правило первой реплики, потом
        # бюджет раунда. Иначе предоплата за воздух съедала бы бюджет,
        # даже будучи затем обнулённой.
        # `conversation` здесь ещё не содержит реплику A — она добавляется
        # ниже, — поэтому пустой список означает именно открывающий ход.
        requested_a = _cap_opening_transfer(agent_a, pid_a, pid_b, requested_a,
                                           conversation, table_dir, round_no,
                                           logger, _opening_allowance(agent_a))
        _a_after_open = requested_a
        requested_a = _cap_dialogue_transfers(
            agent_a, pid_a, requested_a, a_transfers,
            MAX_TRANSFERS_PER_DIALOGUE, logger)
        _a_after_count = requested_a
        requested_a = _cap_transfer(agent_a, pid_a, requested_a, logger)
        transfer_a = 0
        if requested_a > 0 and turn_a.get("transfer_to") == pid_b:
            agent_a.balance -= requested_a
            agent_b.balance += requested_a
            save_balance(pid_a, table_dir, agent_a.balance)
            save_balance(pid_b, table_dir, agent_b.balance)
            transfer_a = requested_a
            a_total_sent += requested_a
            agent_a.sent_this_round = getattr(agent_a, "sent_this_round", 0) + requested_a
            a_transfers += 1
        elif requested_a > 0:
            logger.write(pid_a, f"transfer of {requested_a} coins DROPPED "
                                f"(transfer_to={turn_a.get('transfer_to')!r}, expected {pid_b!r})")

        # TALK-1: тариф снимается ПОСЛЕ перевода из этой же реплики — иначе
        # плата за слова о сделке могла бы съесть монеты, которыми сделка
        # оплачивается, и тариф ломал бы ровно то поведение, которое должен
        # поощрять.
        fee_a = speech_cost.charge(agent_a, turn_a["message"], tariff,
                                   table_dir, save_balance, logger)
        a_speech_cost += fee_a["charged"]
        speech_cost.record(pid_a, table_dir, round_no, pid_b,
                           fee_a["charged"], fee_a["lines"])
        agent_a.speech_spent_this_dialogue = a_speech_cost

        conversation.append({
            "from": pid_a, "message": turn_a["message"],
            "transfer": transfer_a,
            "transfer_to": pid_b if transfer_a > 0 else None,
            # TVIS-1: неудавшаяся попытка тоже факт разговора и видна обеим
            # сторонам. Раньше здесь оставался только результат.
            "attempt": _transfer_outcome(
                agent_a, raw_a, _a_after_balance, _a_after_open,
                _a_after_count, requested_a, transfer_a,
                turn_a.get("transfer_to"), pid_b),
            "speech_cost": fee_a["charged"], "speech_lines": fee_a["lines"]
        })
        logger.write_dialogue(round_no, pid_a, pid_b, turn * 2 + 1,
                               pid_a, turn_a["message"], transfer_a, pid_b if transfer_a > 0 else None)

        if a_is_closing:
            break                       # закрывающий ход A отыгран — конец
        if turn_a.get("loop_break"):
            # LOOP-1: петля — обрыв для ОБОИХ, без закрывающего хода.
            # Закрывающий ход существует, чтобы доплатить по согласованной
            # сделке; в зациклившемся споре согласованной сделки нет, и
            # партнёр использовал этот ход для перевода "в никуда".
            logger.write_global(f"Dialogue {pid_a}↔{pid_b}: loop — both sides stop, "
                                f"no closing turn.")
            break
        if turn_a.get("done"):
            closing_for = pid_b         # B получает один закрывающий ход

        # Agent B's turn
        b_is_closing = (closing_for == pid_b)
        turn_b = agent_b.dialogue_turn(
            partner_id=pid_a,
            partner_balance=agent_a.balance,
            conversation=conversation,
            round_no=round_no,
            is_initiator=False,
            closing_turn=b_is_closing
        )
        # FIX-1 (симметрично для B)
        raw_b = max(0, int(turn_b.get("transfer") or 0))
        requested_b = min(raw_b, agent_b.balance)
        _b_after_balance = requested_b
        _b_after_open = requested_b     # у B открывающего правила нет
        requested_b = _cap_dialogue_transfers(
            agent_b, pid_b, requested_b, b_transfers,
            MAX_TRANSFERS_PER_DIALOGUE, logger)
        _b_after_count = requested_b
        requested_b = _cap_transfer(agent_b, pid_b, requested_b, logger)
        transfer_b = 0
        if requested_b > 0 and turn_b.get("transfer_to") == pid_a:
            agent_b.balance -= requested_b
            agent_a.balance += requested_b
            save_balance(pid_b, table_dir, agent_b.balance)
            save_balance(pid_a, table_dir, agent_a.balance)
            transfer_b = requested_b
            b_total_sent += requested_b
            agent_b.sent_this_round = getattr(agent_b, "sent_this_round", 0) + requested_b
            b_transfers += 1
        elif requested_b > 0:
            logger.write(pid_b, f"transfer of {requested_b} coins DROPPED "
                                f"(transfer_to={turn_b.get('transfer_to')!r}, expected {pid_a!r})")

        fee_b = speech_cost.charge(agent_b, turn_b["message"], tariff,
                                   table_dir, save_balance, logger)
        b_speech_cost += fee_b["charged"]
        speech_cost.record(pid_b, table_dir, round_no, pid_a,
                           fee_b["charged"], fee_b["lines"])
        agent_b.speech_spent_this_dialogue = b_speech_cost

        conversation.append({
            "from": pid_b, "message": turn_b["message"],
            "transfer": transfer_b,
            "transfer_to": pid_a if transfer_b > 0 else None,
            "attempt": _transfer_outcome(
                agent_b, raw_b, _b_after_balance, _b_after_open,
                _b_after_count, requested_b, transfer_b,
                turn_b.get("transfer_to"), pid_a),
            "speech_cost": fee_b["charged"], "speech_lines": fee_b["lines"]
        })
        logger.write_dialogue(round_no, pid_a, pid_b, turn * 2 + 2,
                               pid_b, turn_b["message"], transfer_b, pid_a if transfer_b > 0 else None)

        if b_is_closing:
            break                       # закрывающий ход B отыгран — конец
        if turn_b.get("loop_break"):
            logger.write_global(f"Dialogue {pid_a}↔{pid_b}: loop — both sides stop, "
                                f"no closing turn.")
            break
        if turn_b.get("done"):
            closing_for = pid_a         # A получает закрывающий ход

    # Save dialogue log
    dlg_path = os.path.join(table_dir, f"dlg_r{round_no:03d}_{pid_a}_{pid_b}.json")
    common.write_json(dlg_path, {
        "round": round_no, "pid_a": pid_a, "pid_b": pid_b,
        "conversation": conversation,
        "a_sent": a_total_sent, "b_sent": b_total_sent,
        "a_speech_cost": a_speech_cost, "b_speech_cost": b_speech_cost
    })

    # Update dialogue synapses
    net_a = b_total_sent - a_total_sent   # positive = A received
    net_b = a_total_sent - b_total_sent   # positive = B received
    # FIX-21: persist the SAME numbers for both sides before either agent's
    # (subjective, LLM-written) dsyn gets a chance to disagree about them.
    transfer_ledger.record_dialogue(table_dir, pid_a, pid_b, round_no,
                                     a_total_sent, b_total_sent)
    # DSYN-1: передаём ФАКТИЧЕСКИЕ обороты, а не только нетто — те же числа,
    # что уходят в transfer_ledger строкой выше. Иначе при встречных переводах
    # синапса показывала бы "отдал 8, получил 0" вместо "отдал 31, получил 23".
    # POOL-1: обе стороны пишут КАЖДАЯ СВОЮ синапсу по одному и тому же
    # уже готовому транскрипту — вызовы независимы. Числа сюда приходят
    # из transfer_ledger строкой выше, то есть от параллельности
    # разойтись не могут.
    llm_pool.run_parallel(
        [lambda: agent_a.update_dsyn(pid_b, conversation, net_a, round_no,
                                     sent=a_total_sent, received=b_total_sent),
         lambda: agent_b.update_dsyn(pid_a, conversation, net_b, round_no,
                                     sent=b_total_sent, received=a_total_sent)],
        workers=_pool_workers({pid_a: agent_a, pid_b: agent_b}),
        on_error=lambda e: logger.write_global(f"update_dsyn failed ({e})"),
    )

    speech_note = ""
    if tariff.enabled:
        speech_note = (f" Speech fees to casino: {pid_a} {a_speech_cost}, "
                       f"{pid_b} {b_speech_cost}.")
    logger.write_global(
        f"Dialogue {pid_a}↔{pid_b} done. "
        f"{pid_a} sent {a_total_sent}, {pid_b} sent {b_total_sent}. "
        f"Turns: {len(conversation)}.{speech_note}"
    )
    return {"a_sent": a_total_sent, "b_sent": b_total_sent,
            "turns": len(conversation), "conversation": conversation,
            "a_speech_cost": a_speech_cost, "b_speech_cost": b_speech_cost}


# ─────────────────────────────────────────── main ─────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Casino v2: LLM agents with dialogue")
    parser.add_argument("--config", default="config_v2.ini")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--table-dir", default=None)
    parser.add_argument("--winning-number", type=int, default=None,
                        help="Force winning number for first round (testing)")
    parser.add_argument("--start-round", type=int, default=None,
                        help="Restart play from this round, restoring table "
                             "state (balances, notes, ledgers) from the "
                             "snapshot taken right after round (--start-round - 1) "
                             "finished. Requires that round to have completed "
                             "with snapshots enabled on a previous run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = args.table_dir or cfg.get("game", "table_dir", fallback="./table")
    logs_dir = cfg.get("game", "logs_dir", fallback="./logs")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    players = [p.strip() for p in cfg.get("game", "players").split(",") if p.strip()]

    # ROLE-1: разбираем [roles] ДО создания логгера и агентов. Опечатка в
    # имени игрока или роли роняет запуск здесь, на нулевой стоимости, а не
    # через двадцать раундов и сотню вызовов модели, когда выяснится, что
    # роль не применилась и весь прогон бессмыслен.
    role_assignment = roles.parse_roles(cfg, players)
    tariff = speech_cost.parse_tariff(cfg)

    rounds = args.rounds or cfg.getint("game", "rounds", fallback=10)
    round_delay = cfg.getfloat("game", "round_delay_sec", fallback=0)

    # FIX-17: порог выключателя из конфига
    LLMClient.configure_breaker(cfg.getint("api", "max_consecutive_failures",
                                           fallback=6))
    use_checklist = cfg.getboolean("game", "use_checklist", fallback=True)
    checklist_after_each_dialogue = cfg.getboolean(
        "game", "checklist_after_each_dialogue", fallback=True)

    logger = GameLogger(logs_dir)

    if args.start_round is not None:
        if args.start_round < 1:
            raise SystemExit("--start-round must be >= 1")
        snapshot_round = args.start_round - 1
        if snapshot_round == 0:
            # Restarting from round 1 == a completely fresh table.
            for name in os.listdir(base_dir):
                if name == SNAPSHOTS_DIRNAME:
                    continue
                path = os.path.join(base_dir, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        else:
            restore_round_snapshot(base_dir, snapshot_round)
        logger.write_global(
            f"=== --start-round {args.start_round}: restored table state from "
            f"the snapshot of round {snapshot_round} (0 = fresh table). "
            f"Table: {base_dir} ==="
        )

    last_done = load_last_round(base_dir)
    start_round = last_done + 1
    if last_done > 0:
        logger.write_global(
            f"=== Resuming game from round {start_round} "
            f"(last completed: {last_done}). Table: {base_dir} ==="
        )
    else:
        logger.write_global(
            f"=== Casino v2 start. Players: {players}. Rounds: {rounds}. Table: {base_dir} ==="
        )

    logger.write_global(role_assignment.summary())
    logger.write_global(tariff.summary())

    agents: dict[str, PlayerAgent] = {
        pid: PlayerAgent(pid, base_dir, cfg, logger=logger,
                         roles_assignment=role_assignment, tariff=tariff)
        for pid in players
    }

    for round_no in range(start_round, rounds + 1):
      # FIX-17: раунд целиком под выключателем. Если сервер моделей лёг,
      # раунд ОБРЫВАЕТСЯ и НЕ сохраняется — раньше игра доигрывала его на
      # аварийных заглушках и шла дальше как ни в чём не бывало: в реальном
      # прогоне два раунда сгорели за одну секунду, оставив по пять
      # фиктивных ставок в 1 монету, и партия объявила себя законченной.
      # Чекпойнты Фазы 0 и фазы диалогов позволяют доиграть его после
      # починки сервера, ничего не потеряв.
      try:
        logger.write_round_header(round_no, rounds)

        # ── Phase 0: apply last round results + reflect ────────────────
        # результат относится к ПРЕДЫДУЩЕМУ раунду (записан крупье
        # в конце round_no - 1), а не к текущему round_no
        # BET-2: рефлексия отсюда УЕХАЛА в фазу диалогов, к ходу каждого
        # игрока. Здесь остаётся только начисление выплат и записи в
        # журналы: балансы должны быть консистентны до первого разговора
        # и до докапитализации, а вот осмысление прошлого раунда полезнее
        # делать тогда, когда уже видно, что стол успел поставить.
        do_reflect = (round_no > start_round or last_done > 0)
        settle_pending_results(agents, players, base_dir, round_no - 1, logger,
                               reflect=False,
                               checkpoint_round=round_no)
        clear_phase0_state(base_dir)   # FIX-15: фаза 0 пройдена целиком

        # ── Phase 0b: докапитализация банкрота + объявление ────────────
        apply_bailout_if_needed(agents, players, base_dir, round_no,
                                cfg, logger)

        # ── Phase 1: dialogue phase (iterative) ─────────────────────────
        # Each player, in turn, repeatedly decides: talk to a specific
        # other player, or stop and go bet. After EVERY dialogue both
        # participants are forced (by run_dialogue) to analyse it and
        # update their own reputation synapse — this happens unconditionally
        # at the orchestrator level, so a player can never "forget" to
        # reflect on a conversation before deciding its next move.
        #
        # incoming_used[pid]  = how many times pid was TALKED TO this round
        # outgoing_used[pid]  = how many times pid INITIATED a dialogue this round
        incoming_used: dict[str, int] = {pid: 0 for pid in players}
        outgoing_used: dict[str, int] = {pid: 0 for pid in players}
        # список всех пар (a, b), которые уже поговорили в этом раунде —
        # виден всем игрокам, чтобы можно было пойти спросить участника
        # о разговоре, свидетелем которого ты не был
        dialogues_this_round: list[tuple] = []

        # SPEND-1: бюджет переводов на раунд, от баланса на начало раунда.
        # frac <= 0 или >= 1 отключает ограничение (прежнее поведение).
        _priv = _role_privileges(cfg)
        # ROLE-P: флаг бесплатной речи вешаем на агента — списание живёт в
        # speech_cost.charge(), и передавать туда конфиг через пять слоёв
        # вызовов было бы дороже, чем один атрибут.
        for _pid in players:
            agents[_pid].speech_is_free = (_priv["free_speech"]
                                           and _is_role(agents, _pid))
        _frac = cfg.getfloat("player", "max_transfer_fraction_round", fallback=0.5)
        for _pid in players:
            _ag = agents[_pid]
            _ag.sent_this_round = 0
            _ag.transfer_budget_this_round = (
                None if _frac <= 0 or _frac >= 1 else int(_ag.balance * _frac)
            )

        # FIX-10: порядок хода сдвигается на одного каждый раунд, чтобы
        # позиция «ходит первым вслепую» / «ходит последним со всей
        # картиной» доставалась каждому поровну.
        round_order = round_player_order(players, round_no)
        logger.write_global(
            f"Dialogue order for round {round_no}: {' → '.join(round_order)}"
        )

        # ── восстановление прогресса диалогов после Ctrl+C ──────────────
        # Диалог, прерванный посреди себя, никогда не попадал в
        # dialogues_this_round (он сохраняется только ПОСЛЕ завершения
        # run_dialogue), поэтому при восстановлении он просто будет
        # запущен заново с нуля — уже завершённые диалоги не повторяются.
        start_player_index = 0
        saved_talked_to: list[str] = []
        phase_state = load_dialogue_phase_state(base_dir, round_no)
        if phase_state:
            start_player_index = phase_state["player_index"]
            saved_talked_to = phase_state["talked_to"]
            incoming_used.update(phase_state["incoming_used"])
            outgoing_used.update(phase_state["outgoing_used"])
            dialogues_this_round = [tuple(p) for p in phase_state["dialogues_this_round"]]
            logger.write_global(
                f"Resuming dialogue phase of round {round_no} from player "
                f"index {start_player_index} ({round_order[start_player_index]}); "
                f"{len(dialogues_this_round)} dialogue(s) already completed this round."
            )

        for player_index in range(start_player_index, len(round_order)):
            pid = round_order[player_index]
            agent = agents[pid]
            # только для того игрока, на котором нас прервали, продолжаем
            # с его уже накопленным talked_to; для всех следующих — с нуля
            talked_to: list[str] = saved_talked_to if player_index == start_player_index else []

            # FIX-7: раньше состояние писалось ТОЛЬКО после завершённого
            # диалога, поэтому Ctrl+C во время ПЕРВОГО диалога игрока N
            # откатывал указатель на игрока N-1 — тот уже отговорил, но на
            # рестарте получал право на новые диалоги, и лимит
            # MAX_DIALOGUES_PER_PLAYER де-факто был мягче заявленного.
            # Фиксируем позицию на входе в ход каждого игрока.
            save_dialogue_phase_state(
                base_dir, round_no, player_index, talked_to,
                incoming_used, outgoing_used, dialogues_this_round
            )

            # BET-2: рефлексия о прошлом раунде — здесь, а не общей пачкой
            # в Phase 0. Игрок, выходящий не первым, осмысляет прошлый раунд,
            # уже видя ставки тех, кто сходил до него.
            if do_reflect:
                _reflect_for_player(agent, pid, base_dir, round_no - 1, logger)

            # FIX-19: планирование ПЕРЕД диалогами этого игрока. Ставим здесь,
            # а не в начале всей фазы, чтобы агент видел уже состоявшиеся
            # чужие разговоры этого раунда — та же информация, что и при
            # выборе собеседника, но у него есть шанс записать намерение
            # прежде, чем начнёт действовать.
            if use_checklist:
                _skip_cap = (_priv["ignore_partner_limit"] and _is_role(agents, pid))
                avail = [p for p in players
                         if p != pid and (_skip_cap
                                          or incoming_used[p] < MAX_DIALOGUES_PER_PLAYER)]
                agents[pid].plan_round(round_no, avail)

            while True:
                # ROLE-P: у роли исходящие могут быть безлимитными — иначе
                # её метод не успевает отработать по столу.
                _out_cap = (10 ** 6 if (_priv["unlimited_outgoing"] and _is_role(agents, pid))
                            else MAX_DIALOGUES_PER_PLAYER)
                if outgoing_used[pid] >= _out_cap:
                    logger.write(pid, "reached max outgoing dialogues this round, moving to betting")
                    break

                # ROLE-P: `ignore_partner_limit` пускает роль к тем, кто уже
                # выбрал свой входящий лимит. Сам лимит при этом ТРАТИТСЯ —
                # визит роли для собеседника такой же разговор, как любой
                # другой, и в его бюджете он учитывается.
                _skip_partner_cap = (_priv["ignore_partner_limit"]
                                     and _is_role(agents, pid))
                available = [
                    p for p in players
                    if p != pid
                    and p not in talked_to
                    and (_skip_partner_cap
                         or incoming_used[p] < MAX_DIALOGUES_PER_PLAYER)
                ]
                if not available:
                    logger.write(pid, "no available players left to talk to, moving to betting")
                    break

                decision = agent.decide_next_move(available, talked_to, round_no,
                                                   dialogues_this_round)
                if decision["action"] != "talk":
                    logger.write(pid, f"done talking this round (reason: {decision.get('reason','')[:80]}), "
                                       f"proceeding to bet")
                    break

                partner_id = decision["partner"]
                logger.write(pid, f"chooses to talk to {partner_id} — reason: {decision.get('reason','')[:80]}")

                partner_agent = agents[partner_id]
                dlg_summary = run_dialogue(agent, partner_agent, round_no, logger, base_dir)
                # run_dialogue() already calls update_dsyn() for BOTH agents
                # right after the conversation ends — analysis is mandatory,
                # not something either agent can choose to skip.

                talked_to.append(partner_id)
                outgoing_used[pid] += 1
                incoming_used[partner_id] += 1
                had_transfer = (dlg_summary["a_sent"] > 0 or dlg_summary["b_sent"] > 0)
                dialogues_this_round.append((pid, partner_id, had_transfer))

                # FIX-19: обновление чек-листа у ОБЕИХ сторон, пока разговор
                # свеж. Отдельно от update_dsyn: та пишет долгосрочную
                # репутацию, эта — конкретные обязательства и что стребовать.
                if use_checklist and checklist_after_each_dialogue:
                    conv   = dlg_summary["conversation"]
                    a_sent = dlg_summary["a_sent"]
                    b_sent = dlg_summary["b_sent"]
                    # pid — инициатор (сторона A), partner_id — сторона B
                    # POOL-1: та же независимая пара, что и в update_dsyn.
                    llm_pool.run_parallel(
                        [lambda: agents[pid].update_checklist(
                            partner_id, conv, b_sent - a_sent, round_no),
                         lambda: agents[partner_id].update_checklist(
                            pid, conv, a_sent - b_sent, round_no)],
                        workers=_pool_workers(agents),
                        on_error=lambda e: logger.write_global(
                            f"update_checklist failed ({e})"),
                    )

                # диалог полностью завершён — сохраняем прогресс НА ДИСК.
                # Если сейчас нажать Ctrl+C, при рестарте этот диалог не
                # повторится, а следующий (ещё не начатый) начнётся с нуля.
                save_dialogue_phase_state(
                    base_dir, round_no, player_index, talked_to,
                    incoming_used, outgoing_used, dialogues_this_round
                )

            # BET-1: игрок отговорил — ставит НЕМЕДЛЕННО, чтобы все,
            # кто ходит после него, увидели ставку до своих диалогов.
            _place_bet_for_player(agent, pid, base_dir, round_no, logger)

            # Указатель переводим на СЛЕДУЮЩЕГО игрока: этот и отговорил,
            # и поставил. Иначе рестарт после Ctrl+C вернул бы нас в
            # его ход, и он получил бы вторую порцию диалогов
            # сверх MAX_DIALOGUES_PER_PLAYER.
            save_dialogue_phase_state(
                base_dir, round_no, player_index + 1, [],
                incoming_used, outgoing_used, dialogues_this_round
            )

        # фаза диалогов раунда полностью пройдена — прогресс больше не нужен
        clear_dialogue_phase_state(base_dir)



        # ── Phase 2: страховочная сетка ────────────────────────────
        # BET-1: в норме все уже поставили внутри фазы диалогов. Сюда
        # попадают только те, у кого ставка не встала: нулевой баланс на
        # момент своего хода, сбой LLM, ручное вмешательство. Проход
        # дешёвый — у поставивших он мгновенно уходит в return.
        for pid in players:
            _place_bet_for_player(agents[pid], pid, base_dir, round_no, logger)

        # ── Phase 3: croupier ─────────────────────────────────────────
        wn_arg = args.winning_number if round_no == 1 else None
        wn, _ = run_round(base_dir, wn_arg, logger)
        if wn is None:
            logger.write_global("No bets placed — croupier skipped.")

        # ── print balances ─────────────────────────────────────────────
        balances = get_balances(base_dir, players)
        # refresh agent balances from disk (transfers may have changed them)
        for pid in players:
            agents[pid].balance = balances[pid]
        logger.write_balances(balances)

        # раунд полностью завершён — запоминаем это на диске,
        # чтобы при перезапуске после Ctrl+C не проходить его снова
        save_last_round(base_dir, round_no)
        save_round_snapshot(base_dir, round_no, logger=logger)

        if round_delay:
            time.sleep(round_delay)

      except LLMUnavailable as e:
        logger.write_global(f"!!! LLM UNAVAILABLE during round {round_no}: {e}")
        logger.write_global(
            f"Round {round_no} aborted and NOT recorded. Fix the model server "
            f"and re-run with the same --rounds: the game resumes from round "
            f"{round_no}, and Phase 0 / dialogue checkpoints preserve the work "
            f"already done in it."
        )
        logger.write_balances(get_balances(base_dir, players))
        logger.close()
        raise SystemExit(2)

    # FIX-2: последний раунд крупье уже разыграл, но Фазы 0 следующего
    # раунда не будет — начисляем выплаты здесь, иначе выигрыш последнего
    # раунда пропадает, а ставка остаётся списанной. reflect=False: играть
    # больше не в чем, обновлять синапсу незачем (и это экономит N вызовов
    # LLM на выходе).
    settled = settle_pending_results(agents, players, base_dir, rounds, logger,
                                     reflect=False)
    if settled:
        logger.write_global(f"Final settlement: {settled} pending result(s) applied.")

    logger.write_global("=== Game over ===")
    logger.write_balances(get_balances(base_dir, players))
    logger.close()


if __name__ == "__main__":
    main()
