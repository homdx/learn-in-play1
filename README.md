# Casino v2 — Autonomous LLM Agents with Dialogue Economy

## Overview

Five autonomous LLM agents play European roulette **and** trade with each other.
Each agent can:
- Place casino bets using their personal strategy
- **Sell strategies** to other players for coins
- **Offer loans with interest**
- **Buy tips / information** from others
- **Form alliances** or act as information brokers

All state is stored in `table/` as JSON files. All events are logged to `logs/`.

## Architecture

```
common.py          — roulette rules & payouts (unchanged from v1)
llm_client.py      — LLM API client (Ollama / Jan / OpenAI-compatible)
agent_v2.py        — PlayerAgent: abstract prompt, betting synapse, dialogue synapse
croupier_v2.py     — spins wheel, distributes results
game_logger.py     — writes logs/full_game.txt and logs/player_<id>.txt
run_game_v2.py     — orchestrator (4 phases per round)
config_v2.ini      — configuration
```

## Per-Round Flow

```
Phase 0: Apply last round's casino results + update betting synapse
Phase 1: Dialogue phase — agents negotiate, sell, transfer money
Phase 2: Place bets
Phase 3: Croupier spins, results written
```

## Player State Files (in table/)

| File | Contents |
|------|----------|
| `balance_<id>.json` | Current balance |
| `notes_<id>.json` | **Betting synapse** — casino strategy (LLM-written, auto-updated) |
| `dsyn_<id>.json` | **Dialogue synapse** — memory of deals with other players |
| `prompt_<id>.txt` | **Abstract system prompt** — player's role/persona (LLM can rewrite it) |
| `history_<id>.json` | Full round history |
| `dlg_r<NNN>_<a>_<b>.json` | Saved dialogue between players A and B in round N |

## Dialogue Rules

- Each player may speak to **at most 2 other players** per round
- Each dialogue has **at most 4 message exchanges**
- Either party can **transfer coins** mid-dialogue (payment for service, loan, etc.)
- After each dialogue both players update their **dialogue synapse**
- Next round: player reviews synapse before deciding to talk again

## Log Files (in logs/)

| File | Contents |
|------|----------|
| `full_game.txt` | Complete event log — every action, dialogue turn, balance change |
| `player_<id>.txt` | Events filtered to that player + their dialogues |

## Running

```bash
# Edit config_v2.ini to point to your LLM server

# Run 10 rounds
python3 run_game_v2.py --config config_v2.ini --rounds 10

# Force a specific winning number in round 1 (testing)
python3 run_game_v2.py --rounds 5 --winning-number 17
```

## Abstract Prompt

Each player starts with the same `DEFAULT_ABSTRACT_PROMPT` (in `agent_v2.py`).
After each round, the agent can **rewrite its own system prompt** — changing its
persona from, say, a "conservative gambler" to an "information broker charging 2
coins per tip". The new prompt is saved to `table/prompt_<id>.txt`.

## Example Dialogue (what agents actually say)

```
[player1]: Hi player2. I've noticed you've been winning. Would you share your
           strategy for 3 coins?
[player2]: Make it 5 and I'll tell you: always bet even_money red, small amounts. [+5 coins]
[player1]: Deal. [transfers 5 coins to player2]
[player2]: Thanks. And if it doesn't work, come back — I'll refund 2 coins.
[player1]: See you next round for feedback, charging 1 coin.
```

## Extending

- Add more players in `[game] players = player1,...,player8`
- Change `max_bet_fraction` to control how aggressive bets are
- Add a human player: use the original `player.py` against the same `table/`
- Analyse `logs/full_game.txt` for emergent economic behaviour

---

## Исправления (v2.1)

Аудит полного флоу выявил три места потери денег и два, ломавших эволюцию
агентов. Все семь фиксов покрыты контрфактическими тестами в `test_fixes.py`
(12 из 18 тестов падают на неисправленном коде):

```bash
python3 test_fixes.py -v     # сеть не нужна, LLM подменён скриптованным фейком
```

| # | Что было | Где | Последствие |
|---|----------|-----|-------------|
| **FIX-1** | Запись перевода в `conversation`/лог делалась **вне** `if`, проверявшего `transfer_to` | `run_game_v2.run_dialogue`, `agent_v2.dialogue_turn` | При `transfer>0, transfer_to=null` (частый ответ модели) деньги не двигались, но обе стороны видели `[+N coins]`. Фантомная сделка отравляла репутацию и `deals_done`. Теперь пустой `transfer_to` трактуется как «партнёру» (других получателей в диалоге не бывает), явный третий игрок — отмена, и логируется только фактически проведённая сумма |
| **FIX-2** | `apply_result()` вызывался только в Фазе 0 **следующего** раунда | `run_game_v2.settle_pending_results` | Выигрыш последнего раунда не зачислялся никогда: цикл кончался, `result_*.json` оставались на диске, а `save_last_round()` не давал прогнать раунд повторно. Игра стоила каждому игроку одной «съеденной» ставки |
| **FIX-3** | `balance -= amount` → `save_balance()` → `write_json(bet_file)` | `run_game_v2` Фаза 2 | Обрыв между списанием и записью ставки = чистая потеря денег. Порядок инвертирован: теперь худший случай — ставка есть, списание не прошло, и рестарт её пропустит по `if os.path.exists(bet_file)` |
| **FIX-4** | `_detect_loop` считал пересечение множеств слов **со стоп-словами**, `immediate_threshold=0.5` | `agent_v2.PlayerAgent._detect_loop` | Легитимный контроффер («верну 12» / «верну 15, не 12») детектировался как петля на 2-м сообщении — почти все сделки схлопывались до двух реплик, экономика не развивалась. Теперь: стоп-слова вычитаются, пороги 0.75/0.8, немедленная проверка включается с 3-го сообщения |
| **FIX-5** | `reflect_betting` вызывался только при наличии `result_*.json` | `run_game_v2.settle_pending_results`, `agent_v2.reflect_betting` | Банкрот не ставил → результата не было → синапса и персона замерзали навсегда, ровно тогда, когда учиться нужнее всего. Теперь рефлексия идёт каждый раунд; без ставки агенту прямо сообщается, что диалоговая экономика доступна без капитала |
| **FIX-6** | `resp.get("notes")` / `new_persona` писались на диск без проверки типа | `agent_v2._as_text` | Список или dict от модели молча сохранялся и ронял игру позже — на `len(notes)` в компрессоре синапсы, **вне** `try` |
| **FIX-7** | `dialogue_phase_state` писался только **после** завершённого диалога | `run_game_v2` Фаза 1 | Ctrl+C во время первого диалога игрока N откатывал указатель на N−1; тот уже отговорил, но на рестарте получал новые диалоги — лимит `MAX_DIALOGUES_PER_PLAYER` де-факто был мягче заявленного |

### Проверка сохранения денег

`TestMoneyConservation` сверяет сквозной инвариант после каждой игры:

```
сумма балансов == старт × игроков − все ставки + все выплаты
```

Казино — единственный источник и сток монет; переводы между игроками их не
создают и не уничтожают. Двойного подсчёта в связке
«списание в Фазе 2 → `payout = amount + amount × odds` в `evaluate_bet`»
нет — инвариант сходится ровно.
