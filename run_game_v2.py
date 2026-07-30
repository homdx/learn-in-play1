"""
run_game_v2.py — Casino orchestrator with:
  * Phase 0: Apply last round results + reflect
  * Phase 1: Dialogue phase (max 2 partners / max 4 turns each)
  * Phase 2: Place bets
  * Phase 3: Croupier spins wheel
  * Full logging to logs/

Run:
    python3 run_game_v2.py [--config config_v2.ini] [--rounds N] [--table-dir PATH]
"""

import argparse
import configparser
import os
import json
import time

import common
from agent_v2 import PlayerAgent, load_balance, save_balance
from croupier_v2 import run_round
from game_logger import GameLogger


MAX_DIALOGUE_TURNS = 4
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
                           reflect=True):
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
    for pid in players:
        agent = agents[pid]
        result_path = common.result_file(pid, base_dir)
        if os.path.exists(result_path):
            result = common.read_json(result_path)
            entry = agent.apply_result(result, round_no)
            os.remove(result_path)
            applied += 1
            if reflect:
                agent.reflect_betting(entry)
        elif reflect and round_no >= 1:
            logger.write(pid, "no bet result for last round — reflecting anyway")
            agent.reflect_betting(None)
    return applied


def get_balances(base_dir, players):
    result = {}
    for pid in players:
        path = common.balance_file(pid, base_dir)
        if os.path.exists(path):
            result[pid] = common.read_json(path)["balance"]
        else:
            result[pid] = 0
    return result


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
    done = False

    for turn in range(MAX_DIALOGUE_TURNS):
        if done:
            break

        # Agent A's turn
        turn_a = agent_a.dialogue_turn(
            partner_id=pid_b,
            partner_balance=agent_b.balance,
            conversation=conversation,
            round_no=round_no,
            is_initiator=(turn == 0)
        )
        # FIX-1: только ФАКТИЧЕСКИ проведённый перевод попадает в лог,
        # в conversation и в диалоговую синапсу. Раньше запись делалась
        # вне if-а, поэтому при transfer>0 с чужим/пустым transfer_to
        # деньги не двигались, но обе стороны видели "[+N coins]" —
        # фантомная сделка, отравлявшая репутацию.
        requested_a = min(turn_a["transfer"], agent_a.balance)
        transfer_a = 0
        if requested_a > 0 and turn_a.get("transfer_to") == pid_b:
            agent_a.balance -= requested_a
            agent_b.balance += requested_a
            save_balance(pid_a, table_dir, agent_a.balance)
            save_balance(pid_b, table_dir, agent_b.balance)
            transfer_a = requested_a
            a_total_sent += requested_a
        elif requested_a > 0:
            logger.write(pid_a, f"transfer of {requested_a} coins DROPPED "
                                f"(transfer_to={turn_a.get('transfer_to')!r}, expected {pid_b!r})")

        conversation.append({
            "from": pid_a, "message": turn_a["message"],
            "transfer": transfer_a,
            "transfer_to": pid_b if transfer_a > 0 else None
        })
        logger.write_dialogue(round_no, pid_a, pid_b, turn * 2 + 1,
                               pid_a, turn_a["message"], transfer_a, pid_b if transfer_a > 0 else None)

        if turn_a.get("done"):
            done = True
            break

        # Agent B's turn
        turn_b = agent_b.dialogue_turn(
            partner_id=pid_a,
            partner_balance=agent_a.balance,
            conversation=conversation,
            round_no=round_no,
            is_initiator=False
        )
        # FIX-1 (симметрично для B)
        requested_b = min(turn_b["transfer"], agent_b.balance)
        transfer_b = 0
        if requested_b > 0 and turn_b.get("transfer_to") == pid_a:
            agent_b.balance -= requested_b
            agent_a.balance += requested_b
            save_balance(pid_b, table_dir, agent_b.balance)
            save_balance(pid_a, table_dir, agent_a.balance)
            transfer_b = requested_b
            b_total_sent += requested_b
        elif requested_b > 0:
            logger.write(pid_b, f"transfer of {requested_b} coins DROPPED "
                                f"(transfer_to={turn_b.get('transfer_to')!r}, expected {pid_a!r})")

        conversation.append({
            "from": pid_b, "message": turn_b["message"],
            "transfer": transfer_b,
            "transfer_to": pid_a if transfer_b > 0 else None
        })
        logger.write_dialogue(round_no, pid_a, pid_b, turn * 2 + 2,
                               pid_b, turn_b["message"], transfer_b, pid_a if transfer_b > 0 else None)

        if turn_b.get("done"):
            done = True

    # Save dialogue log
    dlg_path = os.path.join(table_dir, f"dlg_r{round_no:03d}_{pid_a}_{pid_b}.json")
    common.write_json(dlg_path, {
        "round": round_no, "pid_a": pid_a, "pid_b": pid_b,
        "conversation": conversation,
        "a_sent": a_total_sent, "b_sent": b_total_sent
    })

    # Update dialogue synapses
    net_a = b_total_sent - a_total_sent   # positive = A received
    net_b = a_total_sent - b_total_sent   # positive = B received
    agent_a.update_dsyn(pid_b, conversation, net_a, round_no)
    agent_b.update_dsyn(pid_a, conversation, net_b, round_no)

    logger.write_global(
        f"Dialogue {pid_a}↔{pid_b} done. "
        f"{pid_a} sent {a_total_sent}, {pid_b} sent {b_total_sent}. "
        f"Turns: {len(conversation)}"
    )
    return {"a_sent": a_total_sent, "b_sent": b_total_sent, "turns": len(conversation)}


# ─────────────────────────────────────────── main ─────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Casino v2: LLM agents with dialogue")
    parser.add_argument("--config", default="config_v2.ini")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--table-dir", default=None)
    parser.add_argument("--winning-number", type=int, default=None,
                        help="Force winning number for first round (testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = args.table_dir or cfg.get("game", "table_dir", fallback="./table")
    logs_dir = cfg.get("game", "logs_dir", fallback="./logs")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    players = [p.strip() for p in cfg.get("game", "players").split(",") if p.strip()]
    rounds = args.rounds or cfg.getint("game", "rounds", fallback=10)
    round_delay = cfg.getfloat("game", "round_delay_sec", fallback=0)

    logger = GameLogger(logs_dir)

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

    agents: dict[str, PlayerAgent] = {
        pid: PlayerAgent(pid, base_dir, cfg, logger=logger)
        for pid in players
    }

    for round_no in range(start_round, rounds + 1):
        logger.write_round_header(round_no, rounds)

        # ── Phase 0: apply last round results + reflect ────────────────
        # результат относится к ПРЕДЫДУЩЕМУ раунду (записан крупье
        # в конце round_no - 1), а не к текущему round_no
        settle_pending_results(agents, players, base_dir, round_no - 1, logger,
                               reflect=(round_no > start_round or last_done > 0))

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
                f"index {start_player_index} ({players[start_player_index]}); "
                f"{len(dialogues_this_round)} dialogue(s) already completed this round."
            )

        for player_index in range(start_player_index, len(players)):
            pid = players[player_index]
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

            while True:
                if outgoing_used[pid] >= MAX_DIALOGUES_PER_PLAYER:
                    logger.write(pid, "reached max outgoing dialogues this round, moving to betting")
                    break

                available = [
                    p for p in players
                    if p != pid
                    and p not in talked_to
                    and incoming_used[p] < MAX_DIALOGUES_PER_PLAYER
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

                # диалог полностью завершён — сохраняем прогресс НА ДИСК.
                # Если сейчас нажать Ctrl+C, при рестарте этот диалог не
                # повторится, а следующий (ещё не начатый) начнётся с нуля.
                save_dialogue_phase_state(
                    base_dir, round_no, player_index, talked_to,
                    incoming_used, outgoing_used, dialogues_this_round
                )

        # фаза диалогов раунда полностью пройдена — прогресс больше не нужен
        clear_dialogue_phase_state(base_dir)



        # ── Phase 2: place bets ────────────────────────────────────────
        for pid in players:
            agent = agents[pid]
            if os.path.exists(common.bet_file(pid, base_dir)):
                logger.write(pid, "already has an unprocessed bet, skipping placement.")
                continue
            if agent.balance <= 0:
                logger.write(pid, "balance is 0, cannot bet.")
                continue

            bet = agent.decide_bet()
            # FIX-3: сначала файл ставки, потом списание. Обрыв между двумя
            # операциями раньше означал "деньги списаны, ставки нет" —
            # чистая потеря. Теперь худший случай обратный: ставка есть,
            # списание не прошло, и на рестарте Фаза 2 её пропустит по
            # `if os.path.exists(bet_file)`. Деньги не исчезают.
            common.write_json(common.bet_file(pid, base_dir), bet)
            agent.balance -= bet["amount"]
            save_balance(pid, base_dir, agent.balance)
            bd = bet.get("numbers", bet.get("selection"))
            logger.write(pid, f"bet: {bet['type']}({bd}) amount={bet['amount']} "
                             f"balance_after={agent.balance}")

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

        if round_delay:
            time.sleep(round_delay)

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
