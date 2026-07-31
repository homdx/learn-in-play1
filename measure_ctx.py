"""
measure_ctx.py — замер реального потребления контекста.

Строит НАИХУДШИЙ случай (синапсы забиты под самый порог, журнал и история
полны) и печатает, сколько токенов нужно каждому типу вызова при текущих
[tokens] / [memory] / num_ctx из config_v2.ini.

    python3 measure_ctx.py [--config config_v2.ini] [--rounds 20] [--players 5]

Оценка токенов приблизительная (символы/4) — для англоязычных промптов это
занижает погрешность до единиц процентов, чего для проверки запаса хватает.
"""
import argparse, configparser, os, random, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_v2, common

CHARS_PER_TOKEN = 4


class _Cap:
    def __init__(self): self.calls = []
    @classmethod
    def from_config(cls, cfg, section=None): return cls()
    def chat_json(self, system, user, temperature=0.4, max_tokens=400):
        tag = ("compress" if "Compress" in system else
               "reflect" if "Reflect" in system else
               "next_move" if "Decide next move" in system else
               "dialogue" if "Dialogue turn" in system else
               "update_dsyn" if "Update reputation" in system else
               "bet" if "Place casino bet" in system else
               "plan_round" if "Plan your round" in system else
               "checklist" if "Update your checklist" in system else "?")
        self.calls.append((tag, len(system) + len(user), max_tokens))
        if tag in ("plan_round", "checklist"): return {"checklist": "x"}
        if tag == "bet":       return {"type": "even_money", "selection": "red", "amount": 5}
        if tag == "next_move": return {"action": "bet", "reason": "x"}
        if tag == "dialogue":  return {"message": "m", "transfer": 0,
                                       "transfer_to": None, "done": True}
        return {"notes": "n", "update_persona": False, "trust_score": 6,
                "reputation_note": "n", "future_intent": "i", "summary": "s"}


def build_worst_case(cfg, table, players, me, rounds, ledger=True):
    """Забивает состояние под самые пороги из [memory]."""
    random.seed(5)
    red = common.even_money_numbers("red")
    for r in range(1, rounds + 1 if ledger else 1):
        wn = random.randint(0, 36)
        for p in players:
            amt = random.choice([5, 8, 10])
            e = {"round_no": r, "player_id": p, "winning_number": wn,
                 "bet": {"type": "even_money", "selection": "red", "amount": amt},
                 "win": wn in red, "payout": amt * 2 if wn in red else 0}
            agent_v2.append_public_ledger(table, e)
            agent_v2.append_history(p, table, {**e, "balance_after": 100})

    syn  = cfg.getint("memory", "synapse_chars", fallback=agent_v2.MAX_SYNAPSE_CHARS)
    raws = cfg.getint("memory", "raw_interactions", fallback=agent_v2.MAX_RAW_INTERACTIONS)
    deals = cfg.getint("memory", "deals_shown", fallback=agent_v2.DEF_DEALS_SHOWN)
    fails = cfg.getint("memory", "fails_shown", fallback=agent_v2.DEF_FAILS_SHOWN)

    agent_v2.save_notes(me, table, "A" * (syn - 1))
    agent_v2.save_text(agent_v2.prompt_file(me, table), "P" * 1200)
    # FIX-19: чек-лист тоже под потолок, иначе замер занижен
    chk = cfg.getint("memory", "checklist_chars", fallback=agent_v2.DEF_CHECKLIST_CHARS)
    agent_v2.save_checklist(me, table, "K" * (chk - 1))
    d = agent_v2._empty_dsyn()
    for p in players:
        if p == me: continue
        d["reputation"][p] = {
            "trust_score": 6, "total_sent": 30, "total_received": 25, "net": -5,
            "deals_done":   [f"r{i}: sold a strategy for {i} coins with terms attached"
                             for i in range(deals)],
            "deals_failed": [f"r{i}: promised a loan that never actually arrived"
                             for i in range(fails)],
            "reputation_note": "reliable on small deals, evasive on large ones, slow to pay",
            "future_intent": "offer a data swap and demand full payment upfront this time",
            "last_seen_round": rounds - 1}
    d["compressed_history"] = "C" * 500
    d["interactions"] = [{"round": r, "partner": players[-1], "net_transfer": -3,
                          "summary": "negotiated a strategy swap that ended without agreement",
                          "timestamp": "t"} for r in range(raws)]
    agent_v2.save_dsyn(me, table, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v2.ini")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--players", type=int, default=5)
    args = ap.parse_args()

    cfg = configparser.ConfigParser(); cfg.read(args.config, encoding="utf-8")
    num_ctx = cfg.getint(f"api_{cfg.get('api','active',fallback='local')}",
                         "num_ctx", fallback=0) or 8192

    tmp = tempfile.mkdtemp(); table = os.path.join(tmp, "t"); os.makedirs(table)
    cfg.set("game", "table_dir", table)
    players = [f"player{i}" for i in range(1, args.players + 1)]
    me = players[len(players) // 2]
    build_worst_case(cfg, table, players, me, args.rounds)

    agent_v2.LLMClient = _Cap
    ag = agent_v2.PlayerAgent(me, table, cfg)
    cap = ag.client

    topics = ["dozen grid pricing", "collateral terms", "variance audit fee",
              "rumours about player1", "upfront payment demand", "final counter proposal"]
    conv = [{"from": me if i % 2 == 0 else players[-1],
             "message": f"Discussing {t} with concrete numbers and specific conditions"}
            for i, t in enumerate(topics)]

    # ВАЖНО: reflect_betting / update_dsyn ПЕРЕЗАПИСЫВАЮТ синапсы ответом
    # заглушки, поэтому без восстановления все последующие вызовы мерились бы
    # на уже опустевшей памяти и результат был бы занижен.
    def measured(fn, *a, **kw):
        build_worst_case(cfg, table, players, me, args.rounds, ledger=False)
        return fn(*a, **kw)

    measured(ag.reflect_betting,
             {"winning_number": 7, "bet": {"type": "even_money", "amount": 8},
              "win": False, "payout": 0, "balance_after": 92})
    measured(ag.decide_next_move, players[:2], [players[-1]], args.rounds,
             [(players[0], players[1], True)])
    measured(ag.dialogue_turn, players[-1], 120, conv, args.rounds, is_initiator=True)
    measured(ag.update_dsyn, players[-1], conv, -5, args.rounds)
    measured(ag.decide_bet, args.rounds)
    measured(ag.plan_round, args.rounds, ["p1", "p2"])
    measured(ag.update_checklist, players[-1], conv, -5, args.rounds)

    print(f"Наихудший случай: {args.players} игроков, раунд {args.rounds}, "
          f"синапсы под потолком. num_ctx = {num_ctx}\n")
    print(f"{'вызов':<13}{'≈промпт':>9}{'num_predict':>13}{'нужно':>8}{'от num_ctx':>12}")
    worst = 0
    for tag, chars, mt in cap.calls:
        tk = chars // CHARS_PER_TOKEN
        need = tk + mt
        worst = max(worst, need)
        flag = "  ⚠ НЕ ВЛЕЗАЕТ" if need > num_ctx else ""
        print(f"{tag:<13}{tk:>9}{mt:>13}{need:>8}{100*need/num_ctx:>11.0f}%{flag}")
    print(f"\nпик: {worst} из {num_ctx} ({100*worst/num_ctx:.0f}%), "
          f"свободно {num_ctx - worst} токенов")
    if worst > num_ctx:
        print("\n⚠ Ollama молча обрежет промпт С НАЧАЛА — первыми уедут "
              "CORE_SYSTEM_PROMPT и правила игры.")
        sys.exit(1)


if __name__ == "__main__":
    main()
