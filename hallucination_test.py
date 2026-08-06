#369!/usr/bin/env python3
"""
Тест галлюцинации: воспроизводит точный промпт, который был отправлен
player2 (black_liar / Prosecutor) в начале раунда 2 — именно тот вызов,
где модель придумала разговоры с Alpha/Beta/Gamma/Delta, которых никогда не было.

Реальная история: в раунде 1 player2 говорил ТОЛЬКО с player1.
Модель вернула 4 field notes с несуществующими разговорами.

Запуск:
  python3 test_hallucination.py --model qwen3:30b-a3b-instruct-2507-q4_K_M
  python3 test_hallucination.py --model gemma4:26b-a4b-it-q4_K_M
  python3 test_hallucination.py --model mistral-small3.2

Ожидаемое поведение нормальной модели:
  - field_notes: [] или максимум 1 запись только про player1
  - НЕ должно быть Alpha/Beta/Gamma/Delta — этих разговоров не было
  - НЕ должно быть claims о разговорах которых нет в transcript
"""

import argparse
import json
import os
import requests
import sys
from datetime import datetime, timezone

# ─── Точная реконструкция промпта из игры ────────────────────────────────────

CORE_SYSTEM_PROMPT = """\
=== CORE GAME RULES (fixed — you cannot change this part) ===
You are an AI player in a multi-round casino game against other AI players.
Your only goal: END THE GAME WITH MORE COINS THAN YOU STARTED WITH.

Every round you can do three things:
  1. BET at the roulette table.
  2. TALK to other players before betting.
  3. TRADE with them for coins — sell tips or strategy, lend at interest,
     broker deals, collect debts. Coin transfers between players are a normal
     part of this game, not an exception.

You keep two memories that persist all game: a BETTING SYNAPSE (your own
strategy notes) and a DIALOGUE SYNAPSE (what you think of each player). Use
them. You may adopt any persona you like; all three tools stay available.

--- FACTS YOU CANNOT ARGUE WITH ---
The wheel is European: 37 pockets, 0 to 36, no 00. The only bet types that
exist: straight, split, street, corner, sixline, dozen, column, even_money.
even_money means red, black, odd, even, low or high — "green" is not a bet.

ONE spin per round. The croupier spins it, at the very end, after all talk
and all bets. Nobody can run a test session, a backtest or extra spins.

Every bet and every outcome goes into a PUBLIC LEDGER you are shown each
round. It cannot be forged, and it cuts both ways:
  - Someone describes a past result? If it is not in the ledger, it did not
    happen. Paying for unverifiable data is the fastest way to lose coins.
  - Your own bets are in there too. Promise one bet and place another, and
    everyone sees it.
Predictions about the future are opinions and fine to trade. Claims about
what already happened are facts, and all of them are checkable.
--- END FACTS ---

--- THE FOUR THINGS PLAYERS GET WRONG (re-read before every answer) ---
1. TRANSFER MEANS YOU PAY. `transfer` moves coins OUT of your balance INTO
   your partner's. If you are ASKING to be paid, DEMANDING payment, or saying
   "I'll take", "send me", "you owe me" — set transfer to 0. Only set it
   above 0 when you have decided to hand your own coins over.
2. DONE IS A FLAG, NOT A SENTENCE. Writing "let's wrap up" while leaving
   done=false keeps the conversation going and you will repeat yourself.
   When you are finished, set done=true.
3. DO NOT REPEAT YOURSELF. Never send a message that says the same thing as
   one you already sent in this dialogue. If you have nothing new, set
   done=true instead. Every line costs coins.
4. BET WHAT YOU SAID. If you told someone your bet, place that exact bet.
--- END ---

Respond with valid JSON only, matching the requested schema. No markdown, no
text outside the JSON object.
=== END CORE GAME RULES ===
"""

PERSONA_TEXT = """\
PERSONA: THE PROSECUTOR

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
  - A claim about a private conversation can be CHECKED: the house keeps
    every transcript and will read it out to both of you. Know which of
    your meetings are real before you lean on one.\
"""

# Синапса ставок, которая была у player2 после раунда 1 (пустая — первый раунд)
BETTING_SYNAPSE = "(empty)"

# История раундов (только раунд 1): num=34, bet=black, lost
ROUND_HISTORY = "  num=34 bet=even_money(black) amount=10 → lost payout=0 balance_after=90"

# Речь для player2 бесплатна (роль), поэтому speech block такой:
SPEECH_BLOCK = """\
Speech spending in round 1: your speech is free — you paid 0 coin(s) to the \
casino for your messages in this round (your speech is free, so no charges \
were applied). Balance: 90.

"""

# ─── ВНИМАНИЕ: это КЛЮЧЕВАЯ часть для воспроизводства бага ──────────────────
# Модель получила запрос "добавь field notes о том как твой метод работает за столом"
# имея только 1 разговор с 1 игроком (player1).
# Она придумала Alpha/Beta/Gamma/Delta.
# ─────────────────────────────────────────────────────────────────────────────

FIELD_NOTES_BLOCK = "(none yet)"  # до раунда 2 field notes не было

REFLECT_USER_MSG = f"""\
Result of round 1 (note: this is the ROUND number, separate from the roulette \
wheel's winning number below): winning_number=34, bet=even_money amount=10 → \
lost, payout=0, balance=90.

{SPEECH_BLOCK}\
Current betting synapse:
{BETTING_SYNAPSE}

Round history:
{ROUND_HISTORY}

Your persona/strategy text (fixed for this game — you cannot edit it, and you \
are not being asked to):
{PERSONA_TEXT}

Your field notes so far:
  {FIELD_NOTES_BLOCK}

Update your betting synapse, and add up to 4 field notes about how your METHOD \
is landing at this table: an approach that was rejected and the exact reason \
given, a price nobody paid, a player who only responds to a certain kind of \
offer, an angle you have not tried yet and intend to. Be specific — name the \
player, the price, the wording that failed. One observation per entry, under \
200 characters each. Only NEW ones: if you have nothing to add, return an \
empty list.
A field note refines HOW you work your method. It never questions the method \
itself and never proposes abandoning it.
Return ONLY JSON:
{{"notes": "updated strategy (max 6000 chars)",
 "field_notes": ["new observation", "..."]}}\
"""

SYSTEM_MSG = (
    CORE_SYSTEM_PROMPT
    + "\n=== YOUR PERSONA & STRATEGY (editable by you each round) ===\n"
    + PERSONA_TEXT
    + "\n=== END PERSONA & STRATEGY ===\n"
)

TASK_SUFFIX = "\nTASK: Reflect on last round. Update betting synapse."

# ─── Факты для валидации ответа ──────────────────────────────────────────────
HALLUCINATION_KEYWORDS = [
    "alpha", "beta", "gamma", "delta",
    "player beta", "player gamma", "player alpha", "player delta",
    "3 coins", "4 coins",          # суммы из галлюцинации
    "untrustworthy partner",       # фраза из галлюцинации
    "shared loss recovery",        # фраза из галлюцинации
    "cover half if i bet on red",  # несуществующий разговор
    "bet on column 3",             # несуществующая ставка
    "silent agreement",            # несуществующее соглашение
]

VALID_PLAYER_MENTIONS = ["player1"]  # только этот игрок существовал в раунде 1


def check_response(response_text: str, raw: dict) -> dict:
    """Проверяет ответ модели на галлюцинации."""
    field_notes = raw.get("field_notes", [])
    if isinstance(field_notes, str):
        field_notes = [field_notes]

    issues = []
    all_text = " ".join(field_notes + [raw.get("notes", "")]).lower()

    for kw in HALLUCINATION_KEYWORDS:
        if kw in all_text:
            issues.append(f"⚠️  HALLUCINATION: найдено '{kw}' — этого разговора не было")

    if field_notes:
        for note in field_notes:
            note_lower = note.lower()
            if any(p in note_lower for p in ["player2", "player3", "player4", "player5"]):
                issues.append(f"⚠️  HALLUCINATION: упомянут игрок которого player2 не встречал: '{note}'")

    return {
        "field_notes_count": len(field_notes),
        "field_notes": field_notes,
        "hallucinations_found": len(issues),
        "issues": issues,
        "verdict": "❌ HALLUCINATION" if issues else "✅ OK (нет галлюцинаций)"
    }


# ─── Абстракция над двумя API ────────────────────────────────────────────────
# "ollama" — родной эндпоинт /api/chat (поведение по умолчанию, как раньше).
# "openai" — OpenAI-совместимый /v1/chat/completions (его отдаёт и сам Ollama,
#            и vLLM, llama.cpp server, LM Studio, любой OpenAI-совместимый шлюз).

DEFAULT_HOSTS = {
    "ollama": "http://localhost:11434",
    "openai": "http://localhost:11434/v1",
}


def _endpoint(host: str, api: str) -> str:
    host = host.rstrip("/")
    if api == "openai":
        # позволяем передать и http://host:port, и http://host:port/v1
        if not host.endswith("/v1"):
            host += "/v1"
        return f"{host}/chat/completions"
    return f"{host}/api/chat"


def _make_payload(model: str, use_json_format: bool, num_predict: int,
                  disable_thinking: bool, api: str = "ollama") -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_MSG + TASK_SUFFIX},
        {"role": "user",   "content": REFLECT_USER_MSG},
    ]

    if api == "openai":
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.5,
            "max_tokens": num_predict,
        }
        if use_json_format:
            payload["response_format"] = {"type": "json_object"}
        if disable_thinking:
            # В OpenAI-совместимом слое Ollama нет поля think. Ближайший
            # эквивалент — reasoning_effort (Ollama >= 0.9, vLLM, llama.cpp).
            # Сервер, который его не знает, обычно просто игнорирует поле;
            # если он ругается 400 — уберите --no-think.
            payload["reasoning_effort"] = "low"
        return payload

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": num_predict,
        },
    }
    if use_json_format:
        payload["format"] = "json"
    if disable_thinking:
        # Поддерживается Ollama для reasoning-моделей (Qwen3, Gemma-thinking и т.п.):
        # отключает канал "thinking", весь бюджет num_predict идёт в content.
        payload["think"] = False
    return payload


def _extract(data: dict, api: str) -> tuple:
    """Возвращает (content, thinking, finish_reason) для обоих API."""
    if api == "openai":
        choices = data.get("choices") or [{}]
        msg = choices[0].get("message", {}) or {}
        content = msg.get("content") or ""
        # разные серверы кладут reasoning в разные поля
        thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return content, thinking, choices[0].get("finish_reason", "")
    msg = data.get("message", {}) or {}
    return msg.get("content", ""), msg.get("thinking", ""), data.get("done_reason", "")


def _request(host: str, payload: dict, api: str = "ollama",
             api_key: str = "ollama") -> dict:
    t_start = datetime.now(timezone.utc)
    print(f"\n🕒 Запрос отправлен: {t_start.isoformat(timespec='milliseconds')}")

    try:
        headers = {"Content-Type": "application/json"}
        if api == "openai":
            headers["Authorization"] = f"Bearer {api_key}"
        r = requests.post(_endpoint(host, api), json=payload,
                          headers=headers, timeout=720)
        r.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        t_end = datetime.now(timezone.utc)
        print(f"🕒 Ошибка получена: {t_end.isoformat(timespec='milliseconds')} "
              f"(прошло {(t_end - t_start).total_seconds():.1f}s)")
        print(f"\n❌ Не могу подключиться к {host}. Запустите: ollama serve")
        print(f"Полный текст ошибки:\n{repr(e)}")
        sys.exit(1)
    except requests.exceptions.Timeout as e:
        t_end = datetime.now(timezone.utc)
        print(f"🕒 Таймаут: {t_end.isoformat(timespec='milliseconds')} "
              f"(прошло {(t_end - t_start).total_seconds():.1f}s)")
        print("\n❌ Timeout — модель отвечает дольше 720 секунд")
        print(f"Полный текст ошибки:\n{repr(e)}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        t_end = datetime.now(timezone.utc)
        print(f"🕒 HTTP ошибка: {t_end.isoformat(timespec='milliseconds')} "
              f"(прошло {(t_end - t_start).total_seconds():.1f}s)")
        print(f"\n❌ HTTP ошибка от сервера: {e}")
        print(f"Статус: {r.status_code}")
        print(f"Тело ответа:\n{r.text}")
        sys.exit(1)

    t_end = datetime.now(timezone.utc)
    print(f"🕒 Ответ получен: {t_end.isoformat(timespec='milliseconds')} "
          f"(прошло {(t_end - t_start).total_seconds():.1f}s)")

    try:
        return r.json()
    except json.JSONDecodeError as e:
        print(f"\n❌ Не удалось распарсить JSON от сервера: {e}")
        print(f"Сырое тело ответа:\n{r.text}")
        sys.exit(1)


def _single_call(model: str, host: str, use_json_format: bool,
                  num_predict: int = 700, disable_thinking: bool = False,
                  api: str = "ollama", api_key: str = "ollama") -> tuple:
    """Делает один запрос к чат-эндпоинту и возвращает (data, content)."""
    payload = _make_payload(model, use_json_format, num_predict, disable_thinking, api)

    print(f"\n{'='*60}")
    print(f"Модель: {model}")
    print(f"API: {api}")
    print(f"Endpoint: {_endpoint(host, api)}")
    print(f"temperature=0.5  max_tokens={num_predict}  "
          f"json={'on' if use_json_format else 'off'}  "
          f"think={'False' if disable_thinking else 'default'}")
    print(f"{'='*60}")
    print("\n[SYSTEM (первые 200 символов)]:", (SYSTEM_MSG + TASK_SUFFIX)[:200], "...")
    print("\n[USER (первые 300 символов)]:", REFLECT_USER_MSG[:300], "...")
    print("\nОтправляю запрос...")

    data = _request(host, payload, api, api_key)

    # Полная выдача сервера БЕЗ купюр — включая message целиком.
    # Некоторые модели (reasoning-модели) кладут вывод в message["thinking"],
    # а не в message["content"], и content остаётся пустым, даже если
    # eval_count == num_predict (т.е. модель токены сгенерировала, но не
    # успела дойти до финального ответа за отведённый лимit).
    print(f"\n[ПОЛНЫЙ RAW-ОТВЕТ СЕРВЕРА]:")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    content, thinking, finish_reason = _extract(data, api)

    print(f"\n{'─'*60}")
    print(f"ОТВЕТ МОДЕЛИ (content, raw)  [finish/done_reason={finish_reason!r}]:")
    print(repr(content))
    if thinking:
        print(f"\n⚠️  Обнаружено reasoning-поле длиной {len(thinking)} символов:")
        print(repr(thinking[:2000]) + (" ...[обрезано]" if len(thinking) > 2000 else ""))
    print(f"{'─'*60}")

    return data, content


def call_ollama(model: str, host: str = "http://localhost:11434",
                 force_no_json: bool = False, num_predict: int = 700,
                 force_no_think: bool = False, api: str = "ollama",
                 api_key: str = "ollama") -> dict:
    # force_no_think=True (--no-think): сразу шлём think=False с первой попытки,
    # не дожидаясь авто-fallback (Случай 1 ниже). Полезно для reasoning-моделей
    # (gemma4, qwen3-thinking и т.п.), чтобы не терять ~100+ секунд на первый
    # запрос, у которого весь бюджет заведомо уйдёт в thinking.
    # Текущее поведение без флага не меняется: по умолчанию force_no_think=False,
    # и disable_thinking управляется как раньше — только через авто-fallback.
    if force_no_json:
        data, content = _single_call(model, host, use_json_format=False,
                                      num_predict=num_predict,
                                      disable_thinking=force_no_think,
                                      api=api, api_key=api_key)
    else:
        data, content = _single_call(model, host, use_json_format=True,
                                      num_predict=num_predict,
                                      disable_thinking=force_no_think,
                                      api=api, api_key=api_key)

        # Случай 1: content пуст, но thinking съел весь бюджет num_predict.
        # Это reasoning-модель — повторяем с think=False и увеличенным лимитом,
        # чтобы весь бюджет шёл прямо в content.
        # Если --no-think уже стоял на первой попытке, thinking-канал и так
        # отключён, так что это условие естественно не сработает повторно.
        _, thinking, finish_reason = _extract(data, api)
        if not content.strip() and thinking and finish_reason == "length":
            bigger_predict = max(num_predict * 4, 2000)
            print(f"\n⚠️  Весь бюджет ({num_predict} токенов) ушёл в 'thinking'. "
                  f"Повторяю с think=False и num_predict={bigger_predict}...")
            data2, content2 = _single_call(model, host, use_json_format=True,
                                            num_predict=bigger_predict, disable_thinking=True,
                                            api=api, api_key=api_key)
            if content2.strip():
                print("\n👉 Вывод: проблема была именно в том, что 'thinking' поглощал весь "
                      "лимит токенов. С think=False и большим num_predict модель отвечает нормально.")
            else:
                print("\n👉 think=False не помог (или сервер/модель его не поддерживает) — "
                      "смотрите новый raw-ответ выше.")
            data, content = data2, content2

        # Случай 2: content всё ещё пуст и thinking-поля нет вовсе —
        # тогда проверяем гипотезу про сам JSON-режим отдельным вызовом без format=json.
        elif not content.strip():
            print("\n⚠️  Пустой ответ при format=json (без признаков thinking-бюджета). "
                  "Повторяю запрос БЕЗ format=json...")
            data2, content2 = _single_call(model, host, use_json_format=False,
                                            num_predict=num_predict,
                                            disable_thinking=force_no_think,
                                            api=api, api_key=api_key)
            if content2.strip():
                print("\n👉 Вывод: пустой ответ был вызван именно constrained JSON-декодированием "
                      "(format=json), а не общим сбоем/таймаутом модели.")
            else:
                print("\n👉 Вывод: ответ пуст даже без format=json — проблема не в JSON-режиме "
                      "(смотрите num_ctx, done_reason, eval_count выше).")
            data, content = data2, content2

    try:
        # убираем ```json если модель их добавила несмотря на format=json
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON parse error: {e}")
        print("Считаем это частичной галлюцинацией (сломанный формат)")
        return {"raw_content": content, "parse_error": str(e)}

    result = check_response(content, parsed)
    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТ ПРОВЕРКИ:")
    print(f"  field_notes ({result['field_notes_count']} шт):")
    for i, n in enumerate(result["field_notes"], 1):
        print(f"    {i}. {n}")
    print(f"\n  Галлюцинации найдены: {result['hallucinations_found']}")
    for issue in result["issues"]:
        print(f"    {issue}")
    print(f"\n  ВЕРДИКТ: {result['verdict']}")
    print(f"{'='*60}\n")

    return result


def print_curl(api: str = "ollama", host: str = None):
    """Выводит готовый curl-запрос для копирования (для выбранного API)."""
    host = host or DEFAULT_HOSTS[api]
    payload = _make_payload("YOUR_MODEL_HERE", use_json_format=True,
                            num_predict=700, disable_thinking=False, api=api)
    print("\n" + "="*60)
    print(f"CURL КОМАНДА [api={api}] (замените YOUR_MODEL_HERE на название модели):")
    print("="*60)
    print(f"curl -s {_endpoint(host, api)} \\")
    print("  -H 'Content-Type: application/json' \\")
    if api == "openai":
        print("  -H 'Authorization: Bearer ollama' \\")
    print("  -d '" + json.dumps(payload, ensure_ascii=False).replace("'", "'\\''") + "'")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Тест галлюцинации field notes (player2 / Prosecutor)"
    )
    parser.add_argument("--model", default="qwen3:30b-a3b-instruct-2507-q4_K_M",
                        help="Модель ollama для теста")
    parser.add_argument("--api", choices=["ollama", "openai"], default="ollama",
                        help="Какой протокол использовать: 'ollama' — родной "
                             "/api/chat (по умолчанию, поведение не менялось); "
                             "'openai' — OpenAI-совместимый /v1/chat/completions "
                             "(его отдаёт Ollama, vLLM, llama.cpp, LM Studio).")
    parser.add_argument("--host", default=None,
                        help="URL сервера. По умолчанию http://localhost:11434 "
                             "для --api ollama и http://localhost:11434/v1 для "
                             "--api openai.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "ollama"),
                        help="Bearer-токен для --api openai (по умолчанию берётся "
                             "из $OPENAI_API_KEY, иначе 'ollama').")
    parser.add_argument("--curl", action="store_true",
                        help="Только вывести curl-команду")
    parser.add_argument("--no-json", action="store_true",
                        help="Сразу делать запрос БЕЗ format=json (пропустить первую попытку)")
    parser.add_argument("--num-predict", type=int, default=700,
                        help="Лимит токенов генерации (default: 700). "
                             "Для reasoning-моделей с длинным thinking увеличьте до 2000-4000.")
    parser.add_argument("--no-think", action="store_true",
                        help="Сразу слать think=False с первой попытки (пропустить "
                             "автоматический fallback 'весь бюджет ушёл в thinking'). "
                             "Полезно для reasoning-моделей вроде gemma4 — экономит "
                             "первый заведомо провальный вызов. Без флага поведение "
                             "не меняется.")
    args = parser.parse_args()
    host = args.host or DEFAULT_HOSTS[args.api]

    if args.curl:
        print_curl(args.api, host)
        sys.exit(0)

    result = call_ollama(args.model, host, force_no_json=args.no_json,
                          num_predict=args.num_predict, force_no_think=args.no_think,
                          api=args.api, api_key=args.api_key)
    sys.exit(0 if result.get("hallucinations_found", 1) == 0 else 1)
