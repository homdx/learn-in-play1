# VERIFY-1 — проверка утверждений о третьих игроках

## Что готово

* `dialogue_archive.py` — читалка `dlg_r*.json`, 16 тестов, зависимостей нет.
* `roles.py` — две строки в персоны (что записи проверяемы) + `verify_block(role, stage)`
  со стадиями `prepare` / `challenged`. Существующие 62 теста проходят.

## Поток, который вы описали

```
ход 1   A → B:  claim_about = {player: "C", round: 5, gist: "..."}
ход 2   B → A:  challenge = true          ← отказ поверить
ход 3   A видит в промпте TABLE RECORD + verify_block(role, "challenged")
                → настаивает или признаёт ошибку
```

Ключевое: **никакого разбора прозы**. Кто именно оболган — приходит полем
`claim_about`, которое заполняет сам говорящий; требование записи — полем
`challenge`, которое ставит слушатель. Оба уже возвращаются моделью в том же
JSON, что и `message`/`transfer`.

## Три точки стыковки

**1. Схема хода в `agent_v2.dialogue_turn`** (~строка 1801, блок `Return ONLY JSON`):

```
 "claim_about": null or {"player": "playerN", "round": 5, "gist": "коротко"},
 "challenge": false,
```

плюс пояснение рядом: `claim_about` заполняется, когда ссылаешься на прошлый
разговор с отсутствующим игроком; `challenge: true` — когда требуешь у
собеседника поднять запись вместо того, чтобы верить на слово.

**2. Сохранение полей в `run_game_v2.py`** (строки 594 и 653, `conversation.append`)
— добавить `"claim_about": turn_x.get("claim_about")` и `"challenge": bool(turn_x.get("challenge"))`.
Больше в оркестраторе ничего не нужно: `conversation` и так передаётся в
`dialogue_turn` целиком.

**3. Сборка промпта в `dialogue_turn`** — перед `user_msg`:

```python
last = conversation[-1] if conversation else None
challenged = bool(last and last.get("from") == partner_id and last.get("challenge"))
claim = None
if challenged:
    # утверждение, которое оспаривают, — последнее МОЁ с claim_about
    claim = next((t.get("claim_about") for t in reversed(conversation)
                  if t.get("from") == self.player_id and t.get("claim_about")), None)

if challenged and claim:
    verify_txt = dialogue_archive.format_evidence(
        self.base_dir, partner_id, claim.get("player"),
        claim.get("round"), claim.get("gist", "")
    ) + roles.verify_block(self.role, "challenged")
else:
    verify_txt = dialogue_archive.format_preparation(
        self.base_dir, self.player_id, self.all_players, round_no
    ) + roles.verify_block(self.role, "prepare")
```

и `+ verify_txt` в `user_msg` — логично сразу после `score_txt`, рядом с
остальными проверяемыми фактами.

Важно: `format_evidence` вызывается **обеими** сторонами с одинаковыми
аргументами и возвращает одинаковый текст. Врун и обвинитель читают один и тот
же протокол — иначе проверка выродилась бы в спор о том, что показывает архив.

## Чего я сознательно не делал

* **Ареитра-круповье нет.** Оказался не нужен для 80% случаев: `no_record` —
  факт файловой системы, его доказывает `os.listdir`, а не модель. Совпадение
  `gist` с реальной репликой пусть оценивает сам собеседник — он и так читает
  дословный текст. Добавить арбитра можно позже, поверх того же `lookup()`.
* **Порога «врал N раз → санкция» нет.** Сначала стоит посмотреть в логах,
  начнут ли агенты вообще ставить `challenge`. Если нет — проблема в цене
  реплики, а не в отсутствии наказания.

## Что смотреть в первом прогоне

1. Ставит ли кто-нибудь `challenge` хоть раз. Если ноль за 20 раундов — роли
   не видят смысла проверять, надо удешевлять сам вопрос.
2. Что делает роль после `no_record`: признаётся или упирается. Оба исхода
   валидны, интересно распределение.
3. Не начал ли `black_liar` заполнять `claim_about` в каждой реплике — это
   удорожает промпт и делает проверку рутиной вместо события.

Прототипировать на `qwen3:8b`: на 30b один лишний обмен репликами — плюс
полчаса на раунд.

---

# POOL-1 — несколько серверов моделей

## Конфиг

```ini
[api]
active      = remote
pool        = api_remote, api_remote2   ; без этого ключа — всё как раньше
max_failover = 1                        ; повторов на другом сервере

[api_remote]
base_url = https://api.one/
model    = qwen3:30b-a3b-cpu

[api_remote2]
base_url = http://localhost:11434
model    = qwen3:8b
```

Модели в пуле могут быть разными: запасной сервер послабее лучше упавшего
раунда.

## Что распараллелено (и что нет)

| этап | параллельно | почему |
|---|---|---|
| Фаза 0, `reflect_betting` | да, N игроков | агент трогает только свои файлы |
| `update_dsyn` после диалога | да, пара | обе стороны пишут по готовому транскрипту |
| `update_checklist` после диалога | да, пара | то же самое |
| диалоги | нет | реплика зависит от предыдущей |
| `decide_next_move`, `plan_round` | нет | зависят от того, кто ещё свободен |
| **ставки** | **нет, намеренно** | BET-1 ставит их сразу после диалогов игрока, чтобы следующие видели; батч убил бы эту видимость |

Замер на заглушках: 6 вызовов по 0.3с → 1.80 / 0.91 / 0.60с на 1/2/3 серверах.

## Отказоустойчивость

Ошибка вызова → повтор на другом свободном сервере (`max_failover`).
`ENDPOINT_FAIL_THRESHOLD` ошибок подряд выводят сервер из ротации на
`ENDPOINT_COOLDOWN_SEC`. `LLMUnavailable` через failover НЕ проходит:
`_Breaker` глобален, это утверждение про пул целиком.
