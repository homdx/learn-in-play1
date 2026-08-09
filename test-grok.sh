#!/usr/bin/env bash
# Проверяет, какое значение reasoning_effort реально принимает Groq для
# qwen/qwen3.6-27b (и любой другой их thinking-модели).
#
# Причина: chat_template_kwargs и reasoning ("effort":"low") Groq отвергает
# HTTP 400 целиком ("property '...' is unsupported"). А для reasoning_effort
# ошибка другая — не "поле не поддерживается", а
#   "`reasoning_effort` must be one of `none` or `default`"
# то есть поле реально существует, просто наше значение "low" неверное.
# Этот скрипт пробует РАЗНЫЕ значения по очереди и показывает: (а) прошёл
# ли запрос вообще, (б) реально ли content содержит ответ, а не только
# <think>-блок без ничего после.
#
# Правьте только переменные в блоке ниже.

set -euo pipefail

BASE_URL="https://api.groq.com/openai/v1"
API_KEY="your-token-here"                       # ваш [api_remote] api_key для Groq
MODEL="qwen/qwen3.6-27b"                      # как в конфиге, где ловится баг
MAX_TOKENS=700                                # тот же бюджет, что реально не хватает

SYSTEM='You are a casino player. TASK: Place casino bet. JSON only.'
USER='Round 1. Balance 100. No history. Place your bet.
Return ONLY JSON: {"type": "even_money", "selection": "red", "amount": N, "reasoning": "short"}'

try_value() {
  local reasoning_effort_value="$1"
  echo
  echo "=========================================================="
  echo "=== reasoning_effort=\"${reasoning_effort_value}\" ==="
  echo "=========================================================="

  PAYLOAD=$(jq -n \
    --arg model "$MODEL" \
    --argjson max_tokens "$MAX_TOKENS" \
    --arg system "$SYSTEM" \
    --arg user "$USER" \
    --arg re "$reasoning_effort_value" \
    '{
      model: $model,
      temperature: 0.4,
      max_tokens: $max_tokens,
      messages: [
        {role: "system", content: $system},
        {role: "user", content: $user}
      ],
      stream: false,
      reasoning_effort: $re
    }')

  echo "--- PAYLOAD ---"
  echo "$PAYLOAD" | jq .
  echo "--- RESPONSE ---"

  RESPONSE=$(curl -sS -X POST "${BASE_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "User-Agent: learn-in-play1-llm-client/1.0 (+https://github.com/homdx/learn-in-play1)" \
    -d "$PAYLOAD")

  echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"

  # Быстрый вердикт: есть ли ошибка, и если нет — похож ли content на
  # "только <think>, ничего после" (та самая проблема из логов).
  ERR=$(echo "$RESPONSE" | jq -r '.error.message // empty' 2>/dev/null || true)
  if [ -n "$ERR" ]; then
    echo ">>> ОТКАЗ СЕРВЕРА: $ERR"
    return
  fi
  CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
  FINISH=$(echo "$RESPONSE" | jq -r '.choices[0].finish_reason // empty' 2>/dev/null || true)
  COMPLETION_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.completion_tokens // empty' 2>/dev/null || true)
  echo ">>> finish_reason=${FINISH}, completion_tokens=${COMPLETION_TOKENS}"
  if echo "$CONTENT" | grep -q '^\s*<think>' && ! echo "$CONTENT" | grep -qi '</think>'; then
    echo ">>> ПЛОХО: content — незакрытый <think>, весь бюджет ушёл на рассуждение, ответа нет"
  elif [ -z "$CONTENT" ]; then
    echo ">>> ПЛОХО: content пуст"
  else
    echo ">>> ХОРОШО: content содержит что-то помимо думания — проверьте, валидный ли это JSON:"
    echo "$CONTENT"
  fi
}

# Groq прямо сказал в ошибке, что валидны только "none" и "default" —
# пробуем оба. "low" оставлен для контроля (уже знаем, что упадёт, но
# показывает точный текст ошибки на будущее).
try_value "none"
try_value "default"
try_value "low"
