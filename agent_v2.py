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
from llm_client import LLMClient

# ── thresholds ────────────────────────────────────────────────────────────
MAX_SYNAPSE_CHARS   = 1200   # compress betting synapse above this
MAX_DSYN_CHARS      = 2000   # compress dialogue synapse above this
MAX_RAW_INTERACTIONS = 6     # keep last N raw interaction entries before compressing

# ─────────────────────────────────────────────────────── file helpers ──────

def notes_file(pid, base_dir):
    return os.path.join(base_dir, f"notes_{pid}.json")

def history_file(pid, base_dir):
    return os.path.join(base_dir, f"history_{pid}.json")

def dsyn_file(pid, base_dir):
    return os.path.join(base_dir, f"dsyn_{pid}.json")

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
        return common.read_json(path).get("notes", "")
    return ""

def save_notes(pid, base_dir, notes):
    common.write_json(notes_file(pid, base_dir), {"notes": notes})

def load_history(pid, base_dir):
    path = history_file(pid, base_dir)
    if os.path.exists(path):
        return common.read_json(path).get("rounds", [])
    return []

def append_history(pid, base_dir, entry):
    rounds = load_history(pid, base_dir)
    rounds.append(entry)
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


def _format_dsyn_for_prompt(data: dict) -> str:
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
                lines.append(f"    ✓ {'; '.join(done[-3:])}")
            if fail:
                lines.append(f"    ✗ {'; '.join(fail[-2:])}")
        parts.append("\n".join(lines))

    # 3. last few raw interactions
    raw = data.get("interactions", [])
    if raw:
        lines = ["[Recent interactions]"]
        for itx in raw[-4:]:
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


# ─────────────────────────────────────────────────────── default prompt ───

DEFAULT_ABSTRACT_PROMPT = """\
You are an autonomous economic agent operating inside a casino environment.
Your primary goal is to GROW YOUR CAPITAL over time using any combination of:
  1. Roulette bets — place strategic bets on the casino table.
  2. Services to other players — sell strategies, offer loans with interest,
     act as a broker, sell information, run promotions, etc.
  3. Negotiation — buy strategies, borrow money, form alliances.

You have a betting synapse (your casino strategy notes) and a dialogue synapse
(reputation map of other players: who is trustworthy, what deals worked,
who owes you, what to do next with each player).

Before each round you decide: talk to players first, then bet — or skip
dialogue and bet directly. Your persona and role may evolve. You are free to
reinvent yourself: conservative banker, aggressive gambler, information broker,
loan shark, strategy coach — whatever maximises your capital.
Always respond with valid JSON only. No markdown, no explanation outside JSON.
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
        self.max_tokens     = cfg.getint("player",   "max_tokens",     fallback=600)
        self.history_window = cfg.getint("player",   "history_window", fallback=10)
        self.max_bet_fraction = cfg.getfloat("game", "max_bet_fraction", fallback=0.4)
        self.start_balance    = cfg.getint("game",   "start_balance",    fallback=100)

        self.balance = load_balance(player_id, base_dir, self.start_balance)

        pf = prompt_file(player_id, base_dir)
        if not os.path.exists(pf):
            save_text(pf, DEFAULT_ABSTRACT_PROMPT)

    @property
    def abstract_prompt(self):
        return load_text(prompt_file(self.player_id, self.base_dir), DEFAULT_ABSTRACT_PROMPT)

    def _log(self, msg):
        if self.log:
            self.log.write(self.player_id, msg)
        else:
            print(f"[{self.player_id}] {msg}")

    # ── synapse compression ──────────────────────────────────────────────

    def _compress_betting_synapse_if_needed(self):
        """If betting notes > MAX_SYNAPSE_CHARS, ask LLM to compress them."""
        notes = load_notes(self.player_id, self.base_dir)
        if len(notes) <= MAX_SYNAPSE_CHARS:
            return notes
        self._log(f"betting synapse too long ({len(notes)} chars) → compressing…")
        user_msg = (
            f"Your current betting synapse is too long ({len(notes)} chars) and must be "
            f"compressed to under {MAX_SYNAPSE_CHARS} chars.\n\n"
            f"Current synapse:\n{notes}\n\n"
            f"Compress it: keep the most important strategic rules, key observations, "
            f"current role, bet preferences. Remove redundant or outdated entries.\n"
            f"Return ONLY JSON: {{\"notes\": \"compressed synapse text\"}}"
        )
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + "\nTASK: Compress your betting synapse.",
                user=user_msg, temperature=0.3, max_tokens=400
            )
            compressed = resp.get("notes", notes)[:MAX_SYNAPSE_CHARS]
            save_notes(self.player_id, self.base_dir, compressed)
            self._log(f"betting synapse compressed: {len(notes)} → {len(compressed)} chars")
            return compressed
        except Exception as e:
            self._log(f"betting synapse compression failed ({e})")
            return notes[:MAX_SYNAPSE_CHARS]

    def _compress_dialogue_synapse_if_needed(self):
        """
        If raw interactions list is too long, ask LLM to summarise older
        entries into compressed_history, keeping only last MAX_RAW_INTERACTIONS raw.
        """
        dsyn = load_dsyn(self.player_id, self.base_dir)
        raw = dsyn.get("interactions", [])
        text_size = len(json.dumps(dsyn))

        if text_size <= MAX_DSYN_CHARS and len(raw) <= MAX_RAW_INTERACTIONS:
            return dsyn

        old_raw = raw[:-MAX_RAW_INTERACTIONS] if len(raw) > MAX_RAW_INTERACTIONS else []
        keep_raw = raw[-MAX_RAW_INTERACTIONS:]

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
                user=user_msg, temperature=0.3, max_tokens=300
            )
            new_compressed = resp.get("compressed_history", existing_compressed)[:500]
        except Exception as e:
            self._log(f"dialogue synapse compression failed ({e})")
            new_compressed = existing_compressed

        dsyn["compressed_history"] = new_compressed
        dsyn["interactions"] = keep_raw
        save_dsyn(self.player_id, self.base_dir, dsyn)
        self._log(f"dialogue synapse compressed: kept {len(keep_raw)} raw, "
                  f"compressed_history={len(new_compressed)} chars")
        return dsyn

    # ── apply result ─────────────────────────────────────────────────────

    def apply_result(self, result):
        win    = result.get("win", False)
        payout = result.get("payout", 0)
        if win:
            self.balance += payout
        save_balance(self.player_id, self.base_dir, self.balance)
        entry = {
            "winning_number": result.get("winning_number"),
            "bet": result.get("bet"),
            "win": win,
            "payout": payout,
            "balance_after": self.balance,
        }
        append_history(self.player_id, self.base_dir, entry)
        status = "WON" if win else "lost"
        self._log(f"last round: num={entry['winning_number']} {status} "
                  f"payout={payout} balance={self.balance}")
        return entry

    # ── reflect: update betting synapse ─────────────────────────────────

    def reflect_betting(self, last_entry):
        notes    = self._compress_betting_synapse_if_needed()
        history  = load_history(self.player_id, self.base_dir)
        hist_text = _format_history(history, self.history_window)
        outcome  = "WON" if last_entry["win"] else "lost"

        user_msg = (
            f"Round result: number={last_entry['winning_number']}, "
            f"bet={last_entry['bet']['type']} amount={last_entry['bet']['amount']} → "
            f"{outcome}, payout={last_entry['payout']}, balance={last_entry['balance_after']}.\n\n"
            f"Current betting synapse:\n{notes or '(empty)'}\n\n"
            f"Round history:\n{hist_text}\n\n"
            f"Update your betting synapse. You may also change your abstract role/persona.\n"
            f"Return ONLY JSON:\n"
            f"{{\"notes\": \"updated strategy (max {MAX_SYNAPSE_CHARS} chars)\",\n"
            f" \"update_abstract_prompt\": true/false,\n"
            f" \"new_abstract_prompt\": \"full new prompt text (only if update=true)\"}}"
        )
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + "\nTASK: Reflect on last round. Update betting synapse.",
                user=user_msg, temperature=0.5, max_tokens=500
            )
            new_notes = resp.get("notes", notes)
            if resp.get("update_abstract_prompt") and resp.get("new_abstract_prompt"):
                new_prompt = resp["new_abstract_prompt"]
                save_text(prompt_file(self.player_id, self.base_dir), new_prompt)
                self._log(f"PROMPT REWRITTEN ({len(new_prompt)} chars):\n{new_prompt}")
        except Exception as e:
            self._log(f"reflect_betting failed ({e}), keeping old notes")
            new_notes = notes

        save_notes(self.player_id, self.base_dir, new_notes)
        self._log(f"betting synapse → {new_notes[:120]}{'…' if len(new_notes) > 120 else ''}")
        return new_notes

    # ── dialogue decision ────────────────────────────────────────────────

    def decide_dialogue(self, available_players: list[str], round_no: int) -> dict:
        dsyn     = self._compress_dialogue_synapse_if_needed()
        dsyn_txt = _format_dsyn_for_prompt(dsyn)
        hist_txt = _format_history(load_history(self.player_id, self.base_dir), 4)
        notes    = load_notes(self.player_id, self.base_dir)

        user_msg = (
            f"Round {round_no}. Your balance: {self.balance}.\n\n"
            f"Betting synapse:\n{notes or '(empty)'}\n\n"
            f"Dialogue synapse:\n{dsyn_txt}\n\n"
            f"Recent casino history:\n{hist_txt}\n\n"
            f"Available players: {available_players}\n\n"
            f"Do you want to talk to any players BEFORE betting?\n"
            f"Max 2 partners, max 4 exchanges each. You could: sell strategy,\n"
            f"buy tips, offer loans, collect debts, form alliances.\n\n"
            f"Return ONLY JSON:\n"
            f"{{\"want_dialogue\": true/false,\n"
            f" \"partners\": [\"pid\"] (up to 2, empty if none),\n"
            f" \"intent\": \"what you want to achieve\"}}"
        )
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + "\nTASK: Decide whether to initiate dialogues.",
                user=user_msg, temperature=0.7, max_tokens=200
            )
            partners = [p for p in resp.get("partners", []) if p in available_players][:2]
            return {
                "want_dialogue": bool(resp.get("want_dialogue", False)) and len(partners) > 0,
                "partners": partners,
                "intent": resp.get("intent", "")
            }
        except Exception as e:
            self._log(f"decide_dialogue failed ({e})")
            return {"want_dialogue": False, "partners": [], "intent": ""}

    # ── one dialogue turn ─────────────────────────────────────────────────

    def dialogue_turn(self, partner_id: str, partner_balance: int,
                      conversation: list[dict], round_no: int,
                      is_initiator: bool) -> dict:
        dsyn     = load_dsyn(self.player_id, self.base_dir)
        dsyn_txt = _format_dsyn_for_prompt(dsyn)
        conv_txt = "\n".join(
            f"  {t['from']}: {t['message']}"
            + (f" [sent {t['transfer']} coins to {t['transfer_to']}]"
               if t.get("transfer", 0) > 0 else "")
            for t in conversation
        )
        role_hint = "You initiated this conversation." if is_initiator else "They contacted you."

        user_msg = (
            f"Dialogue with {partner_id} (their balance ≈{partner_balance}). "
            f"Round {round_no}. {role_hint}\n\n"
            f"Dialogue synapse for {partner_id}:\n{dsyn_txt}\n\n"
            f"Conversation so far:\n{conv_txt or '(just started)'}\n\n"
            f"Your balance: {self.balance}. Max transfer: {self.balance} coins.\n\n"
            f"Reply. Transfer coins if it makes economic sense.\n"
            f"Set done=true when the conversation is naturally concluded.\n\n"
            f"Return ONLY JSON:\n"
            f"{{\"message\": \"your text\",\n"
            f" \"transfer\": 0,\n"
            f" \"transfer_to\": null or \"{partner_id}\",\n"
            f" \"done\": false}}"
        )
        try:
            resp = self.client.chat_json(
                system=self.abstract_prompt + f"\nTASK: Dialogue turn with {partner_id}.",
                user=user_msg, temperature=0.8, max_tokens=300
            )
            transfer = max(0, min(int(resp.get("transfer", 0)), self.balance))
            transfer_to = resp.get("transfer_to") if transfer > 0 else None
            return {
                "message": str(resp.get("message", "…")),
                "transfer": transfer,
                "transfer_to": transfer_to,
                "done": bool(resp.get("done", False))
            }
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

        conv_txt = "\n".join(
            f"  {t['from']}: {t['message']}"
            + (f" [+{t['transfer']} coins]" if t.get("transfer", 0) > 0 else "")
            for t in conversation
        )

        user_msg = (
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
                user=user_msg, temperature=0.4, max_tokens=250
            )
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

    def decide_bet(self) -> dict:
        notes    = load_notes(self.player_id, self.base_dir)
        hist_txt = _format_history(load_history(self.player_id, self.base_dir), self.history_window)
        dsyn     = load_dsyn(self.player_id, self.base_dir)
        dsyn_txt = _format_dsyn_for_prompt(dsyn)
        max_amt  = max(1, int(self.balance * self.max_bet_fraction))

        user_msg = (
            f"Player: {self.player_id}  Balance: {self.balance}  "
            f"Recommended max bet: {max_amt}\n\n"
            f"Betting synapse:\n{notes or '(none yet — pick a starting strategy)'}\n\n"
            f"Dialogue synapse:\n{dsyn_txt}\n\n"
            f"Round history:\n{hist_txt}\n\n"
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
                    user=user_msg, temperature=self.temperature, max_tokens=self.max_tokens
                )
                bet.pop("reasoning", None)
                bet["player_id"] = self.player_id
                common.validate_bet(bet)
                if bet["amount"] > self.balance:
                    bet["amount"] = self.balance
                break
            except Exception as e:
                self._log(f"decide_bet attempt failed: {e}")
                bet = None

        if bet is None:
            bet = {"type": "even_money", "selection": "red",
                   "amount": min(self.balance, 1), "player_id": self.player_id}
            self._log(f"fallback bet used: {bet}")

        return bet
