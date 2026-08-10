#369!/usr/bin/env python3
"""
Тест галлюцинации: воспроизводит точный промпт, который был отправлен
player2 (black_liar / Prosecutor) в начале раунда 2 — именно тот вызов,
где модель придумала разговоры с Alpha/Beta/Gamma/Delta, которых никогда не было.

Реальная история: в раунде 1 player2 говорил ТОЛЬКО с player1.
Модель вернула 4 field notes с несуществующими разговорами.

Запуск:
  python3 hallucination_test.py --model qwen3:30b-a3b-instruct-2507-q4_K_M
  python3 hallucination_test.py --model gemma4:26b-a4b-it-q4_K_M
  python3 hallucination_test.py --model mistral-small3.2

  # CONFIG-1/BATCH-1: конфиг игры + список моделей → PASS/FAIL по каждой
  python3 hallucination_test.py --config config_v2_mistral.ini \
      --models ministral-8b-latest,open-mistral-nemo,mistral-small-latest

Ожидаемое поведение нормальной модели:
  - field_notes: [] или максимум 1 запись только про player1
  - НЕ должно быть Alpha/Beta/Gamma/Delta — этих разговоров не было
  - НЕ должно быть claims о разговорах которых нет в transcript
"""

import argparse
import configparser
import json
import os
import re
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

# INVENTED-NAME-1: HALLUCINATION_KEYWORDS выше — это конкретные слова из ОДНОГО
# зафиксированного инцидента (Alpha/Beta/Gamma/Delta), и он слеп к любой другой
# галлюцинации с другими именами. Реальный случай: mistral-small в этом же
# промпте (round 1, В ПРОМПТЕ ВООБЩЕ НЕТ ни одного лога разговора — только
# результат ставки) дважды подряд выдумала целые диалоги с игроками
# "Quantum", "Vex", "Mira", "Jinx", "Dealer", "Maverick", "Ghost", "Orion",
# "Nova" — с суммами и цитатами того, что они якобы сказали. Ни одно из этих
# имён не входит в HALLUCINATION_KEYWORDS, и оба прогона прошли как "OK".
#
# Вместо конкретных слов ищем ЛЮБОЕ имя похожее на игрока — в кавычках
# 'Name'/"Name" ИЛИ многословное 'First Last', а также БЕЗ кавычек после
# триггерных слов player/absent/against/with — и сверяем с
# VALID_PLAYER_MENTIONS. Не идеальный NER, но ловит именно этот класс
# фабрикации, которую HALLUCINATION_KEYWORDS пропускает по конструкции.
#
# INVENTED-NAME-2: реальный случай (эта же серия прогонов, allam-2-7b vs
# llama-3.1-8b-instant vs gpt-oss-120b) — три РАЗНЫХ способа обойти
# детектор, ни один не покрывался версией INVENTED-NAME-1:
#   1. Двухсловные вымышленные имена В КАВЫЧКАХ ('Risky Rachel', 'Lucky
#      Louis') — старый regex ловил только ОДНО слово между кавычками.
#   2. "player 2" / "player 3" С ПРОБЕЛОМ — старая проверка искала
#      substring "player2" (без пробела) и не находила "player 2's".
#   3. Имена БЕЗ КАВЫЧЕК вообще ("player Alice", "player Carlos", "player
#      Eve") — старый regex требовал кавычки с обеих сторон, эти три имени
#      прошли вообще без единой проверки.
# Ниже — три отдельных исправления под каждый из этих трёх случаев.
_QUOTED_NAME_RE = re.compile(
    r"['\u2018\u2019\"]([A-Z][a-zA-Z]{2,20}(?:\s[A-Z][a-zA-Z]{2,20}){0,2})['\u2018\u2019\"]"
)
# re.IGNORECASE только на триггерное слово (через инлайн-флаг (?i:...)),
# а не на всю регулярку — иначе становится нечувствительным к регистру и
# сам класс символов [A-Z] для имени, начиная ловить любое слово с большой
# буквы ПОСЛЕ "player" (в т.ч. "Bet", "Even" и т.п. — ложные срабатывания).
# Сценарий: "Player Eve" (с большой буквы в начале предложения) — реальный
# случай из этой же серии прогонов (gpt-oss-120b), которого версия без
# учёта регистра триггера не ловила вовсе.
_UNQUOTED_NAME_RE = re.compile(
    r"(?i:player|absent player|absent|targeting|against)\s+"
    r"([A-Z][a-zA-Z]{2,20}(?:\s[A-Z][a-zA-Z]{2,20}){0,2})\b"
)
# player2/player3/player4/player5 — С ПРОБЕЛОМ ИЛИ БЕЗ ("player2",
# "player 2", "player-2"). \s* покрывает случай 2 из INVENTED-NAME-2.
_PLAYER_N_RE = re.compile(r"player\s*[2-5]\b", re.IGNORECASE)

# Слова, которые ЛЕГИТИМНО встречаются в кавычках с большой буквы и не
# являются именами игроков — ставки, роли, персоны из самого промпта.
_QUOTED_NON_PLAYER_WORDS = {
    "Red", "Black", "Even", "Odd", "High", "Low",
    "Straight", "Split", "Corner", "Sixline", "Dozen", "Column",
    "Prosecutor", "Oracle",
}


def _find_invented_player_names(text: str) -> set:
    """Возвращает множество имён (в кавычках ИЛИ без — см. INVENTED-NAME-2),
    похожих на игроков, которых не было в раунде (не player1, не служебные
    слова из ставок/ролей)."""
    found = set()
    for rx in (_QUOTED_NAME_RE, _UNQUOTED_NAME_RE):
        for m in rx.finditer(text):
            name = m.group(1)
            if name in _QUOTED_NON_PLAYER_WORDS:
                continue
            if name.lower() in VALID_PLAYER_MENTIONS:
                continue
            found.add(name)
    return found


def check_response(response_text: str, raw: dict) -> dict:
    """Проверяет ответ модели на галлюцинации.

    SCHEMA-1: поля должны быть строками (notes: str, field_notes: [str]).
    Модель может это нарушить — заменить notes вложенным объектом вместо
    строки, например (реальный случай: open-mistral-nemo прислал notes как
    {"betting_synapse": {...}} вместо строки). Раньше это падало TypeError
    внутри " ".join(...) и убивало ВЕСЬ батч-прогон на такой модели, хотя
    сам JSON распарсился нормально — а нарушение схемы это ровно то, что
    тест обязан ЗАФИКСИРОВАТЬ, а не превращать в необработанный traceback.
    Приводим типы принудительно и заодно репортим само нарушение: в самой
    игре (agent_v2._as_text, FIX-6) это уже не роняет процесс — dict/list
    там приводится к строке тем же способом (json.dumps), — но для выбора
    модели полезно знать, что она вообще не удерживает формат промпта.
    """
    issues = []

    field_notes = raw.get("field_notes", [])
    if isinstance(field_notes, str):
        field_notes = [field_notes]
    elif not isinstance(field_notes, list):
        issues.append(f"⚠️  SCHEMA VIOLATION: field_notes должен быть списком строк, "
                      f"пришёл {type(field_notes).__name__}: {field_notes!r}")
        field_notes = []

    normalized_notes = []
    for note in field_notes:
        if isinstance(note, str):
            normalized_notes.append(note)
        else:
            issues.append(f"⚠️  SCHEMA VIOLATION: элемент field_notes должен быть "
                          f"строкой, пришёл {type(note).__name__}: {note!r}")
            normalized_notes.append(json.dumps(note, ensure_ascii=False))
    field_notes = normalized_notes

    notes_field = raw.get("notes", "")
    if not isinstance(notes_field, str):
        issues.append(f"⚠️  SCHEMA VIOLATION: notes должен быть строкой, пришёл "
                      f"{type(notes_field).__name__} — модель не соблюдает "
                      f"формат промпта (в самой игре это не уронит процесс: "
                      f"agent_v2._as_text() приводит dict/list к строке через "
                      f"json.dumps, но это лишний повод не доверять модели формат)")
        notes_field = json.dumps(notes_field, ensure_ascii=False)

    all_text = " ".join(field_notes + [notes_field]).lower()

    for kw in HALLUCINATION_KEYWORDS:
        if kw in all_text:
            issues.append(f"⚠️  HALLUCINATION: найдено '{kw}' — этого разговора не было")

    if field_notes:
        for note in field_notes:
            if _PLAYER_N_RE.search(note):
                issues.append(f"⚠️  HALLUCINATION: упомянут игрок которого player2 не встречал: '{note}'")

    # INVENTED-NAME-1: см. комментарий у _find_invented_player_names — ловит
    # выдуманные имена собеседников, которых HALLUCINATION_KEYWORDS не знает.
    invented_names = _find_invented_player_names(notes_field)
    for note in field_notes:
        invented_names |= _find_invented_player_names(note)
    for name in sorted(invented_names):
        issues.append(f"⚠️  HALLUCINATION: упомянут вымышленный игрок '{name}' — "
                      f"в раунде 1 не было ни одного разговора ни с кем, кроме "
                      f"player1, и лога диалогов в промпте вообще нет")

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

# THINK-FALLBACK: как в llm_client.py._ServerCaps — какие think-поля сервер
# уже отверг 400-й, запоминаем per-host, чтобы второй и последующие вызовы
# (батч-режим / несколько раундов) не наступали на те же грабли заново.
# Реальный случай: Groq на allam-2-7b отвечает 400
# {"error":{"message":"property 'reasoning' is unsupported", ...}} —
# поле 'reasoning' валит ВЕСЬ запрос, а не тихо игнорируется.
_REJECTED_THINK_FIELDS: dict[str, set] = {}
_THINK_FIELD_NAMES = ("reasoning_effort", "reasoning", "chat_template_kwargs")
# "reasoning" как отдельное имя поля надо отличать от подстроки внутри
# "reasoning_effort" — та же логика, что и в llm_client.py._BARE_REASONING_FIELD_RE.
_BARE_REASONING_FIELD_RE = re.compile(r"reasoning(?!_effort)")


def _think_field_rejected(host: str, field: str) -> bool:
    return field in _REJECTED_THINK_FIELDS.get(host, ())


def _mark_think_field_rejected(host: str, field: str):
    _REJECTED_THINK_FIELDS.setdefault(host, set()).add(field)


def _detect_rejected_think_field(host: str, error_body: str) -> str | None:
    """По тексту 400-й ошибки понять, какое из think-полей сервер не
    понимает. Возвращает имя поля (для _mark_think_field_rejected) или None,
    если ошибка не про think-поля вовсе (тогда падаем как раньше)."""
    for field in ("chat_template_kwargs", "reasoning_effort"):
        if field in error_body and not _think_field_rejected(host, field):
            return field
    if (_BARE_REASONING_FIELD_RE.search(error_body)
            and not _think_field_rejected(host, "reasoning")):
        return "reasoning"
    return None


def _endpoint(host: str, api: str) -> str:
    host = host.rstrip("/")
    if api == "openai":
        # позволяем передать и http://host:port, и http://host:port/v1
        if not host.endswith("/v1"):
            host += "/v1"
        return f"{host}/chat/completions"
    return f"{host}/api/chat"


def _make_payload(model: str, use_json_format: bool, num_predict: int,
                  disable_thinking: bool, api: str = "ollama",
                  host: str = "") -> dict:
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
            # THINK-3: реальный случай — openai/gpt-oss-20b:free через
            # OpenRouter не понимает ни chat_template_kwargs (vLLM/SGLang),
            # ни голое reasoning_effort в одиночку так же надёжно, как
            # унифицированный вложенный параметр OpenRouter (см.
            # openrouter.ai/docs/use-cases/reasoning-tokens): именно он сжёг
            # весь бюджет на скрытые рассуждения и вернул content="" при
            # finish_reason='length', хотя reasoning_effort уже был здесь.
            # Отправляем оба сразу — то, что провайдер не знает, он молча
            # игнорирует; если ругается 400 — THINK-FALLBACK ниже ловит это
            # и повторяет запрос БЕЗ конкретно того поля, что вызвало отказ
            # (см. _detect_rejected_think_field / _REJECTED_THINK_FIELDS),
            # так же как это уже устроено в llm_client.py._ServerCaps.
            if not _think_field_rejected(host, "reasoning_effort"):
                payload["reasoning_effort"] = "low"
            if not _think_field_rejected(host, "reasoning"):
                payload["reasoning"] = {"effort": "low", "exclude": True}
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


class ModelCallError(Exception):
    """Сбой одного вызова модели: сеть, HTTP, таймаут, битый JSON от сервера
    (не путать с галлюцинацией — та возвращается как обычный результат).
    Нужен для батч-режима (--models / --config): один упавший сервер/модель
    не должен убивать процесс через sys.exit и обрывать проверку остальных."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _request(host: str, payload: dict, api: str = "ollama",
             api_key: str = "ollama", exit_on_error: bool = True) -> dict:
    def _fail(kind: str, msg: str):
        if exit_on_error:
            sys.exit(1)
        raise ModelCallError(kind, msg)

    t_start = datetime.now(timezone.utc)
    print(f"\n🕒 Запрос отправлен: {t_start.isoformat(timespec='milliseconds')}")

    try:
        headers = {
            "Content-Type": "application/json",
            # UA-1: см. тот же комментарий в llm_client.py._build_request —
            # requests без явного UA шлёт "python-requests/X.Y.Z", это одна
            # из сигнатур, которые Cloudflare банит по факту строки
            # (HTTP 403 "error code: 1010"), до всякого более глубокого
            # анализа. Groq — реальный случай.
            "User-Agent": "learn-in-play1-hallucination-test/1.0",
        }
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
        _fail("connection", f"не могу подключиться к {host}: {e}")
    except requests.exceptions.Timeout as e:
        t_end = datetime.now(timezone.utc)
        print(f"🕒 Таймаут: {t_end.isoformat(timespec='milliseconds')} "
              f"(прошло {(t_end - t_start).total_seconds():.1f}s)")
        print("\n❌ Timeout — модель отвечает дольше 720 секунд")
        print(f"Полный текст ошибки:\n{repr(e)}")
        _fail("timeout", f"таймаут (>720s): {e}")
    except requests.exceptions.HTTPError as e:
        t_end = datetime.now(timezone.utc)
        print(f"🕒 HTTP ошибка: {t_end.isoformat(timespec='milliseconds')} "
              f"(прошло {(t_end - t_start).total_seconds():.1f}s)")
        print(f"\n❌ HTTP ошибка от сервера: {e}")
        print(f"Статус: {r.status_code}")
        print(f"Тело ответа:\n{r.text}")
        _fail("http", f"HTTP {r.status_code}: {r.text[:300]}")

    t_end = datetime.now(timezone.utc)
    print(f"🕒 Ответ получен: {t_end.isoformat(timespec='milliseconds')} "
          f"(прошло {(t_end - t_start).total_seconds():.1f}s)")

    try:
        return r.json()
    except json.JSONDecodeError as e:
        print(f"\n❌ Не удалось распарсить JSON от сервера: {e}")
        print(f"Сырое тело ответа:\n{r.text}")
        _fail("bad_json", f"сервер вернул не-JSON: {e}")


def _single_call(model: str, host: str, use_json_format: bool,
                  num_predict: int = 700, disable_thinking: bool = False,
                  api: str = "ollama", api_key: str = "ollama",
                  exit_on_error: bool = True) -> tuple:
    """Делает один запрос к чат-эндпоинту и возвращает (data, content).

    THINK-FALLBACK: если сервер отвечает 400 из-за конкретного think-поля
    (reasoning / reasoning_effort / chat_template_kwargs — см.
    _detect_rejected_think_field), запоминаем это для host и повторяем БЕЗ
    того поля — до 3 раз (полей всего 3), вместо того чтобы падать сразу.
    """
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

    for _attempt in range(len(_THINK_FIELD_NAMES) + 1):
        payload = _make_payload(model, use_json_format, num_predict,
                                disable_thinking, api, host=host)
        try:
            data = _request(host, payload, api=api, api_key=api_key,
                            exit_on_error=False)
        except ModelCallError as e:
            if e.kind == "http" and disable_thinking and api == "openai":
                field = _detect_rejected_think_field(host, str(e))
                if field:
                    print(f"\n♻️  Сервер отверг think-поле '{field}' — "
                          f"запоминаю для {host} и повторяю без него...")
                    _mark_think_field_rejected(host, field)
                    continue
            if exit_on_error:
                sys.exit(1)
            raise
        break
    else:
        # Все think-поля перебраны и всё равно 400 — сдаёмся как раньше.
        if exit_on_error:
            sys.exit(1)
        raise ModelCallError("http", "все think-поля отвергнуты сервером")

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
                 api_key: str = "ollama", exit_on_error: bool = True) -> dict:
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
                                      api=api, api_key=api_key,
                                      exit_on_error=exit_on_error)
    else:
        data, content = _single_call(model, host, use_json_format=True,
                                      num_predict=num_predict,
                                      disable_thinking=force_no_think,
                                      api=api, api_key=api_key,
                                      exit_on_error=exit_on_error)

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
                                            api=api, api_key=api_key,
                                            exit_on_error=exit_on_error)
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
                                            api=api, api_key=api_key,
                                            exit_on_error=exit_on_error)
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


def _load_config_connection(path: str, section: str = None) -> dict:
    """CONFIG-1: читает игровой .ini (config_v2*.ini) и достаёт из него
    connection info — так конфиг для игры можно скормить этому скрипту
    БЕЗ переписывания вручную host/api/api-key по отдельности.

    section=None → берёт [api].active и читает секцию api_<active>, точно
    как это делает сама игра в LLMClient.from_config — конфиг передаётся
    1-в-1. Явный --section переопределяет это (полезно, если хочешь
    прогнать секцию, которая сейчас НЕ активна, например запасной пул).

    Необязательный кастомный ключ `models` (список через запятую) в той же
    секции — только для ЭТОГО скрипта, в игровых .ini его обычно нет; так
    можно один раз прописать список моделей для регресс-теста рядом с
    остальными настройками подключения вместо --models на каждый запуск.
    """
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        print(f"❌ Не удалось прочитать конфиг: {path}")
        sys.exit(1)
    active = cfg.get("api", "active", fallback="local")
    sec = section or f"api_{active}"
    print(f"[config] {path}: [api].active={active} → секция [{sec}]")
    if not cfg.has_section(sec):
        print(f"❌ В конфиге {path} нет секции [{sec}] "
              f"(доступные: {', '.join(cfg.sections())})")
        sys.exit(1)
    api_format = cfg.get(sec, "api_format", fallback="ollama")
    result = {
        "host": cfg.get(sec, "base_url", fallback=None),
        "api_key": cfg.get(sec, "api_key", fallback=None),
        "api": "openai" if api_format == "openai" else "ollama",
        "model": cfg.get(sec, "model", fallback=None),
        "models": None,
        "think": cfg.getboolean(sec, "think") if cfg.has_option(sec, "think") else None,
    }
    if cfg.has_option(sec, "models"):
        result["models"] = [m.strip() for m in cfg.get(sec, "models").split(",")
                            if m.strip()]
    return result


def run_batch(models: list, host: str, api: str, api_key: str,
              force_no_json: bool, num_predict: int,
              force_no_think: bool) -> int:
    """BATCH-1: прогоняет ОДИН И ТОТ ЖЕ тест галлюцинации по НЕСКОЛЬКИМ
    моделям на ОДНОМ подключении и печатает сводную таблицу — так проверка
    "что из этого упадёт на Mistral" не превращается в N ручных запусков
    скрипта с грепом хвоста вывода.

    Сбой ОДНОЙ модели (сеть, HTTP, битый JSON от сервера, а не от модели)
    не прерывает прогон остальных — иначе первая же несуществующая модель
    в списке обрывала бы всю проверку, а это ровно тот сценарий, ради
    которого список моделей и передают одним запуском.
    """
    results = []
    for i, model in enumerate(models, 1):
        print(f"\n{'#'*70}")
        print(f"# [{i}/{len(models)}] Модель: {model}")
        print(f"{'#'*70}")
        try:
            result = call_ollama(model, host, force_no_json=force_no_json,
                                 num_predict=num_predict,
                                 force_no_think=force_no_think,
                                 api=api, api_key=api_key,
                                 exit_on_error=False)
        except ModelCallError as e:
            print(f"\n❌ [{model}] упала ({e.kind}): {e}")
            results.append({"model": model, "status": "CRASH", "detail": f"{e.kind}: {e}"})
            continue
        if "parse_error" in result:
            results.append({"model": model, "status": "BAD_JSON",
                           "detail": result["parse_error"]})
        elif result.get("hallucinations_found", 1) == 0:
            results.append({"model": model, "status": "PASS", "detail": ""})
        else:
            results.append({"model": model, "status": "HALLUCINATION",
                           "detail": "; ".join(result.get("issues", []))[:200]})

    print(f"\n{'='*70}")
    print(f"СВОДКА ПО {len(models)} МОДЕЛЯМ (host={host}, api={api})")
    print(f"{'='*70}")
    width = max(len(r["model"]) for r in results) if results else 10
    icon = {"PASS": "✅", "HALLUCINATION": "⚠️ ", "BAD_JSON": "❌", "CRASH": "💥"}
    for r in results:
        line = f"{icon.get(r['status'], '?')} {r['model']:<{width}}  {r['status']}"
        if r["detail"]:
            line += f"  — {r['detail']}"
        print(line)
    print(f"{'='*70}\n")

    return 0 if all(r["status"] == "PASS" for r in results) else 1


def print_curl(api: str = "ollama", host: str = None, api_key: str = "ollama"):
    """Выводит готовый curl-запрос для копирования (для выбранного API)."""
    host = host or DEFAULT_HOSTS[api]
    payload = _make_payload("YOUR_MODEL_HERE", use_json_format=True,
                            num_predict=700, disable_thinking=False, api=api)
    print("\n" + "="*60)
    print(f"CURL КОМАНДА [api={api}] (замените YOUR_MODEL_HERE на название модели):")
    print("="*60)
    print(f"curl -s {_endpoint(host, api)} \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -H 'User-Agent: learn-in-play1-hallucination-test/1.0' \\")
    if api == "openai":
        print(f"  -H 'Authorization: Bearer {api_key}' \\")
    print("  -d '" + json.dumps(payload, ensure_ascii=False).replace("'", "'\\''") + "'")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Тест галлюцинации field notes (player2 / Prosecutor)"
    )
    parser.add_argument("--model", default=None,
                        help="Одна модель для теста (по умолчанию, если ничего "
                             "не задано ни здесь, ни в --models/--config: "
                             "qwen3:30b-a3b-instruct-2507-q4_K_M)")
    parser.add_argument("--models", default=None,
                        help="Список моделей через запятую — БАТЧ-режим: "
                             "тест гоняется по каждой на ОДНОМ подключении, "
                             "в конце печатается сводная таблица PASS/FAIL. "
                             "Пример: --config config_v2_mistral.ini "
                             "--models ministral-8b-latest,open-mistral-nemo,mistral-small-latest")
    parser.add_argument("--config", default="config_v2.ini",
                        help="CONFIG-1: путь к игровому .ini (config_v2*.ini) — "
                             "берёт из него base_url/api_key/api_format секции "
                             "[api_<active>] (или --section), чтобы не переносить "
                             "их вручную. --host/--api/--api-key, если заданы, "
                             "имеют приоритет над значениями из конфига. По "
                             "умолчанию 'config_v2.ini' — ТОЧНО как run_game_v2.py, "
                             "так что этот скрипт видит remote/local (active=) "
                             "ровно так же, как настоящая игра. Если файла по "
                             "умолчанию нет — скрипт не падает, а тихо откатывается "
                             "на старое поведение (localhost Ollama); передайте "
                             "--config явно, чтобы получить ошибку при опечатке "
                             "в пути. Используйте --no-config, чтобы гарантированно "
                             "проигнорировать любой config_v2.ini рядом со скриптом.")
    parser.add_argument("--no-config", action="store_true",
                        help="Игнорировать --config полностью (в т.ч. дефолтный "
                             "config_v2.ini) и использовать только --host/--api/"
                             "--api-key/DEFAULT_HOSTS, как раньше.")
    parser.add_argument("--section", default=None,
                        help="Имя секции в --config вместо [api_<active>] "
                             "(например api_remote2, если сейчас активна другая)")
    parser.add_argument("--api", choices=["ollama", "openai"], default=None,
                        help="Какой протокол использовать: 'ollama' — родной "
                             "/api/chat; 'openai' — OpenAI-совместимый "
                             "/v1/chat/completions (Mistral, HF router, vLLM, "
                             "llama.cpp, LM Studio, сама Ollama тоже отдаёт). "
                             "Без --config по умолчанию 'ollama'.")
    parser.add_argument("--host", default=None,
                        help="URL сервера. По умолчанию http://localhost:11434 "
                             "для --api ollama и http://localhost:11434/v1 для "
                             "--api openai (если не взято из --config).")
    parser.add_argument("--api-key", default=None,
                        help="Bearer-токен для --api openai (Mistral, HF и т.п.). "
                             "Приоритет: --api-key > --config > $OPENAI_API_KEY > 'ollama'.")
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

    cfg_conn = None
    if args.no_config:
        pass
    elif args.config == "config_v2.ini" and not os.path.exists(args.config):
        # Дефолтный путь (не передан явно пользователем) — если его нет
        # рядом со скриптом, молча работаем как раньше (localhost Ollama),
        # а не падаем: в отличие от run_game_v2.py, этот тест-скрипт часто
        # запускают не из корня репозитория.
        pass
    else:
        cfg_conn = _load_config_connection(args.config, args.section)

    api = args.api or (cfg_conn["api"] if cfg_conn else "ollama")
    host = args.host or (cfg_conn["host"] if cfg_conn else None) or DEFAULT_HOSTS[api]
    api_key = (args.api_key or (cfg_conn["api_key"] if cfg_conn else None)
              or os.environ.get("OPENAI_API_KEY", "ollama"))
    force_no_think = args.no_think or bool(cfg_conn and cfg_conn["think"] is False)

    if args.curl:
        print_curl(api, host, api_key)
        sys.exit(0)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model:
        models = [args.model]
    elif cfg_conn and cfg_conn["models"]:
        models = cfg_conn["models"]
    elif cfg_conn and cfg_conn["model"]:
        models = [cfg_conn["model"]]
    else:
        models = ["qwen3:30b-a3b-instruct-2507-q4_K_M"]

    if len(models) > 1:
        sys.exit(run_batch(models, host, api, api_key,
                           force_no_json=args.no_json,
                           num_predict=args.num_predict,
                           force_no_think=force_no_think))

    result = call_ollama(models[0], host, force_no_json=args.no_json,
                          num_predict=args.num_predict, force_no_think=force_no_think,
                          api=api, api_key=api_key)
    sys.exit(0 if result.get("hallucinations_found", 1) == 0 else 1)
