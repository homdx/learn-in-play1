"""
llm_pool.py — несколько серверов моделей вместо одного (POOL-1).

ЧТО РЕАЛЬНО ПАРАЛЛЕЛИТСЯ

Раунд разобран по зависимостям, и независимых мест оказалось три, а не
«все этапы»:

  * Фаза 0, reflect_betting по каждому игроку. Агент читает и пишет
    только свои файлы (персона, синапсы, история), чужого состояния не
    касается. N игроков — N независимых вызовов.
  * update_dsyn после диалога: две стороны пишут каждая свою синапсу по
    одному и тому же готовому транскрипту. Пара независимых вызовов.
  * update_checklist после диалога — то же самое, ещё пара.

Всё остальное последовательно ПО СУЩЕСТВУ, и это не ограничение
реализации:

  * Диалоги — обмен репликами, каждая следующая зависит от предыдущей.
  * decide_next_move — выбор собеседника зависит от того, кто ещё
    свободен и с кем уже поговорили.
  * plan_round — читает список доступных партнёров, который меняется по
    ходу фазы.
  * СТАВКИ — отдельный случай, и его легко испортить. Может показаться,
    что N ставок в конце раунда просятся в батч, но BET-1 специально
    ставит их сразу после диалогов игрока, чтобы следующие игроки видели
    уже размещённые ставки. Параллельный батч уничтожил бы ровно ту
    видимость, ради которой это было сделано.

КАК УСТРОЕН ДОСТУП

PooledClient повторяет интерфейс LLMClient (chat / chat_json), поэтому
подставляется в agent.client без единой правки в agent_v2. На каждый
вызов он арендует свободный сервер, а после — возвращает. Аренда на
вызов, а не на агента: при пяти агентах и двух серверах жёсткое
закрепление дало бы 3/2 и половину простоя.

ОТКАЗОУСТОЙЧИВОСТЬ

Если сервер отвечает ошибкой, вызов повторяется на ДРУГОМ свободном
сервере. Это и есть смысл пула помимо скорости: одна залипшая модель
больше не роняет раунд. Сервер, отдавший подряд несколько ошибок,
временно выводится из ротации и возвращается после паузы — иначе
каждый вызов продолжал бы тратить на него попытку.

Один сервер в конфиге → поведение ровно прежнее: аренда всегда
успешна и мгновенна, потока управления нет.
"""

from __future__ import annotations

import threading
import time

import llm_client as _llm            # МОДУЛЬ, не имя: тесты и stub-режим
                                     # подменяют llm_client.LLMClient уже
                                     # после импорта, и связанное на импорте
                                     # имя такую подмену не увидело бы.
from llm_client import LLMUnavailable

# Сколько подряд ошибок выводят сервер из ротации.
ENDPOINT_FAIL_THRESHOLD = 3
# На сколько секунд. Не навсегда: удалённый шлюз, отдавший 500, обычно
# оживает сам, и вычёркивать его до конца партии — терять половину пула.
ENDPOINT_COOLDOWN_SEC = 120.0


class _Endpoint:
    """Один сервер: клиент, занятость, счётчик подряд идущих ошибок."""

    def __init__(self, client, name: str):
        self.client = client
        self.name = name
        self.busy = False
        self.fails = 0
        self.blocked_until = 0.0

    def available(self, now: float) -> bool:
        return not self.busy and now >= self.blocked_until

    def note_ok(self):
        self.fails = 0

    def note_fail(self) -> bool:
        """True, если сервер после этой ошибки выведен из ротации."""
        self.fails += 1
        if self.fails >= ENDPOINT_FAIL_THRESHOLD:
            self.blocked_until = time.monotonic() + ENDPOINT_COOLDOWN_SEC
            self.fails = 0
            return True
        return False


class LLMPool:
    """
    Набор серверов с арендой. Потокобезопасен: фаза 0 и пары
    dsyn/checklist ходят в него из нескольких потоков одновременно.
    """

    def __init__(self, clients: list, names: list[str] | None = None):
        if not clients:
            raise ValueError("LLMPool: нужен хотя бы один клиент")
        names = names or [f"ep{i}" for i in range(len(clients))]
        self._eps = [_Endpoint(c, n) for c, n in zip(clients, names)]
        self._cv = threading.Condition()
        self.on_event = None          # необязательный лог: on_event(str)

    def __len__(self):
        return len(self._eps)

    @property
    def size(self) -> int:
        return len(self._eps)

    def _log(self, msg: str):
        if self.on_event:
            try:
                self.on_event(msg)
            except Exception:
                pass

    def acquire(self, exclude: set | None = None, timeout: float = 300.0):
        """
        Занять свободный сервер. exclude — имена уже опробованных: на
        повторе после ошибки нужен именно ДРУГОЙ сервер.

        Возвращает _Endpoint или None, если подходящего не нашлось за
        timeout. None — не авария: вызывающий откатывается на любой
        доступный, включая уже опробованный.
        """
        exclude = exclude or set()
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                now = time.monotonic()
                for ep in self._eps:
                    if ep.name not in exclude and ep.available(now):
                        ep.busy = True
                        return ep
                # Все либо заняты, либо на карантине, либо исключены.
                if not any(ep.name not in exclude for ep in self._eps):
                    return None
                left = deadline - now
                if left <= 0:
                    return None
                # Ждём и освобождения, и истечения карантина, поэтому
                # просыпаемся сами, а не только по notify.
                self._cv.wait(min(left, 1.0))

    def release(self, ep):
        with self._cv:
            ep.busy = False
            self._cv.notify_all()


class PooledClient:
    """
    Замена LLMClient для агента. Тот же интерфейс, внутри — пул.

    Число попыток НЕ равно числу серверов: два сервера не должны
    превращаться в два повтора каждого вызова. Повтор на другом сервере
    делается только при ошибке, максимум max_failover раз.
    """

    def __init__(self, pool: LLMPool, max_failover: int = 1, on_retry=None):
        self.pool = pool
        self.max_failover = max(0, int(max_failover))
        self.on_retry = on_retry

    # Агенты читают эти атрибуты у клиента (логи, отладка). Отдаём
    # первый сервер как представительный, иначе AttributeError на ровном
    # месте в чужом коде.
    @property
    def model(self):
        return self.pool._eps[0].client.model

    @property
    def base_url(self):
        return self.pool._eps[0].client.base_url

    def _call(self, method: str, *a, **kw):
        tried: set = set()
        last = None
        for attempt in range(self.max_failover + 1):
            ep = self.pool.acquire(exclude=tried)
            if ep is None:
                # Другого свободного нет — берём любой, лишь бы не встать.
                ep = self.pool.acquire()
                if ep is None:
                    break
            try:
                result = getattr(ep.client, method)(*a, **kw)
                ep.note_ok()
                return result
            except LLMUnavailable:
                # Предохранитель считает вызовы по ВСЕМ серверам сразу
                # (_Breaker глобален), так что это утверждение про пул
                # целиком. Другой сервер тут не поможет.
                raise
            except Exception as e:
                last = e
                tried.add(ep.name)
                if ep.note_fail():
                    self.pool._log(f"endpoint {ep.name} parked for "
                                   f"{ENDPOINT_COOLDOWN_SEC:.0f}s after "
                                   f"repeated failures")
                if attempt < self.max_failover and self.on_retry:
                    self.on_retry(f"endpoint {ep.name} failed ({type(e).__name__}"
                                  f": {e}) — retrying on another server")
            finally:
                self.pool.release(ep)
        raise last if last else RuntimeError("LLMPool: не удалось занять сервер")

    def chat(self, *a, **kw):
        return self._call("chat", *a, **kw)

    def chat_json(self, *a, **kw):
        return self._call("chat_json", *a, **kw)


# ── конфиг ───────────────────────────────────────────────────────────────

def clients_from_config(cfg) -> tuple[list, list[str]]:
    """
    Читает пул из конфига. Формат — обратно совместимый:

        [api]
        active = remote
        pool = api_remote, api_remote2      ; необязательно

        [api_remote]
        base_url = https://api.one/
        model    = qwen3:30b

        [api_remote2]
        base_url = https://api.two/
        model    = qwen3:8b

    Без ключа pool берётся одна секция api_<active>, как раньше.
    Разные модели в пуле допустимы намеренно: запасной сервер с моделью
    послабее лучше, чем упавший раунд.
    """
    active = cfg.get("api", "active", fallback="local")
    raw = cfg.get("api", "pool", fallback="").strip()
    sections = ([s.strip() for s in raw.split(",") if s.strip()]
                if raw else [f"api_{active}"])

    clients, names = [], []
    for sec in sections:
        if not cfg.has_section(sec):
            raise ValueError(f"[api] pool ссылается на секцию {sec!r}, "
                             f"которой нет в конфиге")
        clients.append(_client_for_section(cfg, sec))
        names.append(sec)
    return clients, names


def _client_for_section(cfg, sec: str):
    """Как LLMClient.from_config, но для явно названной секции."""
    verify_ssl = cfg.getboolean("api", "verify_ssl", fallback=True)
    think_raw = cfg.get(sec, "think", fallback=None)
    return _llm.LLMClient(
        base_url=cfg.get(sec, "base_url"),
        api_key=cfg.get(sec, "api_key", fallback="not-needed"),
        model=cfg.get(sec, "model"),
        api_format=cfg.get(sec, "api_format", fallback="ollama"),
        verify_ssl=verify_ssl,
        num_ctx=cfg.getint(sec, "num_ctx", fallback=0),
        think=None if think_raw is None else cfg.getboolean(sec, "think"),
        timeout=cfg.getint("api", "timeout_seconds", fallback=120),
        retries=cfg.getint("api", "retries", fallback=1),
        json_format=cfg.getboolean("api", "json_format", fallback=True),
    )


def build_client(cfg, on_retry=None):
    """
    Главная точка входа: вернуть то, что класть в agent.client.

    Один сервер → обычный LLMClient (никаких потоков и блокировок на
    пустом месте). Два и больше → PooledClient.
    """
    clients, names = clients_from_config(cfg)
    if len(clients) == 1:
        c = clients[0]
        c.on_retry = on_retry
        return c
    pool = LLMPool(clients, names)
    return PooledClient(pool, max_failover=cfg.getint("api", "max_failover",
                                                      fallback=1),
                        on_retry=on_retry)


_SHARED_POOL: LLMPool | None = None
_SHARED_LOCK = threading.Lock()


def shared_client(cfg, on_retry=None, factory=None):
    """
    То же, что build_client, но ПУЛ ОДИН НА ПРОЦЕСС.

    Это не оптимизация, а условие корректности. Каждый агент строит себе
    клиента (agent_v2.py:729); если каждый построит и собственный пул,
    аренда перестанет что-либо значить — пятеро независимо решат, что
    сервер свободен, и пошлют на него пять параллельных запросов. Ровно
    та же ошибка, что была с флагом json_format на экземпляре.

    Клиент-обёртка у каждого агента свой (у него свой on_retry для лога),
    а вот LLMPool за ними общий.
    """
    global _SHARED_POOL
    # Один сервер — прежний путь буква в букву, включая ФАБРИКУ вызывающего.
    # Строить клиента здесь своими руками нельзя: agent_v2 подменяется в
    # тестах по имени agent_v2.LLMClient, и обход этой подмены превратил бы
    # заглушку в реальный сетевой вызов.
    if not cfg.get("api", "pool", fallback="").strip():
        c = factory() if factory else _llm.LLMClient.from_config(cfg)
        c.on_retry = on_retry
        return c
    clients, names = clients_from_config(cfg)
    with _SHARED_LOCK:
        if _SHARED_POOL is None or _SHARED_POOL.size != len(clients):
            _SHARED_POOL = LLMPool(clients, names)
    return PooledClient(_SHARED_POOL,
                        max_failover=cfg.getint("api", "max_failover", fallback=1),
                        on_retry=on_retry)


def reset_shared_pool():
    """Для тестов: следующий shared_client() построит пул заново."""
    global _SHARED_POOL
    _SHARED_POOL = None


# ── параллельный прогон независимых вызовов ──────────────────────────────

def run_parallel(tasks: list, workers: int, on_error=None) -> list:
    """
    Выполнить независимые задачи (callable без аргументов).

    workers<=1 → выполняет последовательно, В ТОМ ЖЕ ПОРЯДКЕ и без
    участия threading. Это важнее, чем кажется: при одном сервере
    поведение обязано остаться байт-в-байт прежним, иначе сравнивать
    прогоны с пулом и без него будет не с чем.

    Исключение одной задачи не отменяет остальные: у каждого агента своя
    рефлексия, и сбой у одного не повод терять её у четверых. Результат —
    список той же длины, на месте упавшей задачи None.
    """
    if workers <= 1 or len(tasks) <= 1:
        out = []
        for t in tasks:
            try:
                out.append(t())
            except Exception as e:
                if on_error:
                    on_error(e)
                out.append(None)
        return out

    from concurrent.futures import ThreadPoolExecutor
    results: list = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(t): i for i, t in enumerate(tasks)}
        for fut, i in futs.items():
            try:
                results[i] = fut.result()
            except Exception as e:
                if on_error:
                    on_error(e)
    return results
