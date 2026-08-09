#!/usr/bin/env bash
# Воспроизводит РОВНО тот запрос, который agent_v2.decide_bet() шлёт через
# llm_client.LLMClient._build_request() при max_bets_per_round > 1.
# Правьте только переменные в блоке ниже — тело запроса ниже не трогайте,
# оно списано 1:1 с кода.

set -euo pipefail

BASE_URL="https://openrouter.ai/api/v1"      # ваш [api_remote] base_url
API_KEY="ВАШ_КЛЮЧ_СЮДА"                       # ваш [api_remote] api_key
MODEL="ВАША_МОДЕЛЬ_СЮДА"                      # ваш [api_remote] model, как в конфиге player1
MAX_TOKENS=700                                # ваш [tokens] bet из конфига
TEMPERATURE=0.5

# Системный промпт — минимальная версия для дебага (без abstract_prompt
# персоны, она не влияет на сам факт filtered/empty choices).
SYSTEM='You are a casino player. TASK: Place casino bet. JSON only.'

# Пользовательский промпт — ХВОСТ реального decide_bet() при
# max_bets_per_round=3, тот самый блок, который меняется в мульти-режиме.
USER='Player: player1  Balance: 76  Recommended max bet: 30

Betting synapse:
(some prior strategy text here)

You are choosing your bet for round 4.

Place your bet(s). You may place UP TO 3 separate bets this round, covering different numbers/selections. Return ONLY JSON:
{"bets": [ {"type": "...", "numbers": [...] OR "selection": "...", "amount": N}, ... up to 3 entries ], "reasoning": "short reason"}
Types: straight(1#,35:1) split(2#,17:1) street(3#,11:1) corner(4#,8:1) sixline(6#,5:1) dozen(sel=1st12/2nd12/3rd12,2:1) column(sel=col1/col2/col3,2:1) even_money(sel=red/black/even/odd/low/high,1:1). The SUM of all amount fields must be ≤ 76.'

PAYLOAD=$(jq -n \
  --arg model "$MODEL" \
  --argjson temperature "$TEMPERATURE" \
  --argjson max_tokens "$MAX_TOKENS" \
  --arg system "$SYSTEM" \
  --arg user "$USER" \
  '{
    model: $model,
    temperature: $temperature,
    max_tokens: $max_tokens,
    messages: [
      {role: "system", content: $system},
      {role: "user", content: $user}
    ],
    stream: false,
    response_format: {type: "json_object"},
    reasoning_effort: "low",
    reasoning: {effort: "low", exclude: true}
  }')

echo "=== PAYLOAD ==="
echo "$PAYLOAD" | jq .
echo "=== ОТВЕТ ==="

curl -sS -i \
  -X POST "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "User-Agent: learn-in-play1-llm-client/1.0 (+https://github.com/homdx/learn-in-play1)" \
  -d "$PAYLOAD"
