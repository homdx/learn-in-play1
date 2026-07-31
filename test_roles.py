"""
test_roles.py — контрфактические тесты для ROLE-1 (секция [roles]).

Каждый тест обязан ПАДАТЬ на коде без roles.py и проходить на коде с ним.
Сеть не нужна: LLM подменяется скриптованным фейком.

    python3 test_roles.py           # прогнать всё
    python3 test_roles.py -v        # с подробностями
"""

import configparser
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_v2
import llm_client
import roles
import run_game_v2
from agent_v2 import PlayerAgent, prompt_file, load_text

PLAYERS = ["player1", "player2", "player3"]


# ─────────────────────────────────────────────────────────── fake LLM ─────

class FakeLLM:
    """Фейковый клиент. Пишет все system/user в calls, отвечает по behaviour."""

    behaviour: dict = {}
    calls: list = []

    @classmethod
    def from_config(cls, cfg, section=None):
        return cls()

    def chat_json(self, system, user, temperature=0.0, max_tokens=0):
        FakeLLM.calls.append({"system": system, "user": user})
        b = FakeLLM.behaviour
        if "Reflect" in system:
            return b.get("reflect", {"notes": "n", "update_persona": False})
        if "Compress your persona" in system:
            return b.get("compress_persona", {"new_persona": "COMPRESSED"})
        if "Compress" in system:
            return {"notes": "compressed", "compressed_history": "compressed"}
        return {}

    @classmethod
    def reset(cls):
        cls.behaviour = {}
        cls.calls = []


def make_cfg(table_dir, roles_section=None, players=PLAYERS):
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "api":       {"active": "local"},
        "api_local": {"base_url": "http://unused", "model": "fake"},
        "game":      {"table_dir": table_dir, "logs_dir": table_dir,
                      "players": ",".join(players),
                      "start_balance": "100", "rounds": "1",
                      "max_bet_fraction": "0.4", "round_delay_sec": "0"},
        "player":    {"temperature": "0.5", "history_window": "10"},
        "memory":    {"persona_chars": "2500"},
    })
    if roles_section is not None:
        cfg.read_dict({"roles": roles_section})
    return cfg


class RoleHarness(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="roles_test_")
        self.table = os.path.join(self.tmp, "table")
        os.makedirs(self.table, exist_ok=True)
        self._orig = agent_v2.LLMClient
        agent_v2.LLMClient = FakeLLM
        llm_client.LLMClient = FakeLLM
        FakeLLM.reset()

    def tearDown(self):
        agent_v2.LLMClient = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def agent(self, pid, roles_section=None, players=PLAYERS):
        cfg = make_cfg(self.table, roles_section, players)
        return PlayerAgent(pid, self.table, cfg)

    def persona_on_disk(self, pid):
        return load_text(prompt_file(pid, self.table), "")


# ───────────────────────────────────── 1. отсутствие регрессии ────────────

class TestDefaultUnchanged(RoleHarness):
    """Без секции [roles] и при enabled=false всё обязано быть как раньше."""

    def test_no_roles_section_gives_default_persona(self):
        a = self.agent("player1")
        self.assertEqual(self.persona_on_disk("player1"),
                         agent_v2.DEFAULT_PERSONA_PROMPT.strip())
        self.assertIsNone(a.role)
        self.assertFalse(a.persona_locked)

    def test_enabled_false_ignores_assignments(self):
        """Контрфактический: без проверки enabled роль применилась бы."""
        a = self.agent("player2", {"enabled": "false", "player2": "black_liar"})
        self.assertIsNone(a.role)
        self.assertNotIn("PROSECUTOR", self.persona_on_disk("player2"))

    def test_unassigned_player_stays_default_when_roles_on(self):
        a = self.agent("player1", {"enabled": "true", "player2": "black_liar"})
        self.assertIsNone(a.role)
        self.assertFalse(a.persona_locked)
        self.assertEqual(self.persona_on_disk("player1"),
                         agent_v2.DEFAULT_PERSONA_PROMPT.strip())


# ───────────────────────────────────────────── 2. посев ролей ─────────────

class TestSeeding(RoleHarness):

    def test_black_liar_seeded(self):
        a = self.agent("player2", {"enabled": "true", "player2": "black_liar"})
        self.assertEqual(a.role, "black_liar")
        self.assertIn("PROSECUTOR", self.persona_on_disk("player2"))

    def test_white_liar_seeded(self):
        a = self.agent("player3", {"enabled": "true", "player3": "white_liar"})
        self.assertEqual(a.role, "white_liar")
        self.assertIn("ORACLE", self.persona_on_disk("player3"))

    def test_both_roles_in_one_game(self):
        sec = {"enabled": "true", "player2": "black_liar", "player3": "white_liar"}
        self.agent("player2", sec)
        self.agent("player3", sec)
        self.assertIn("PROSECUTOR", self.persona_on_disk("player2"))
        self.assertIn("ORACLE", self.persona_on_disk("player3"))

    def test_same_role_for_several_players(self):
        sec = {"enabled": "true", "player1": "white_liar", "player3": "white_liar"}
        self.agent("player1", sec)
        self.agent("player3", sec)
        self.assertIn("ORACLE", self.persona_on_disk("player1"))
        self.assertIn("ORACLE", self.persona_on_disk("player3"))

    def test_none_value_disables_role_for_that_player(self):
        a = self.agent("player2", {"enabled": "true", "player2": "none"})
        self.assertIsNone(a.role)
        self.assertEqual(self.persona_on_disk("player2"),
                         agent_v2.DEFAULT_PERSONA_PROMPT.strip())

    def test_role_prompt_fits_under_persona_limit(self):
        """Роль длиннее persona_chars запустила бы сжатие/обрезку и приехала
        бы к игроку искажённой. Обе роли обязаны влезать как есть."""
        for name, text in roles.ROLE_PROMPTS.items():
            self.assertLessEqual(len(text.strip()), 2500, f"role {name} too long")

    def test_role_prompt_has_headroom_for_edits(self):
        """Впритык к лимиту нельзя: одна правка формулировки — и роль поедет
        через компрессор. Держим запас."""
        for name, text in roles.ROLE_PROMPTS.items():
            self.assertLessEqual(len(text.strip()), 2450,
                                 f"role {name} has no headroom left")


# ──────────────────────────── 2b. содержательные инварианты текстов ───────
#
# Прогон 1-4 раундов показал два провала, оба — от формулировки, а не от
# механики. Эти тесты фиксируют внесённые правки, чтобы они не потерялись
# при следующем редактировании ролей.

class TestRoleContentInvariants(unittest.TestCase):

    @staticmethod
    def flat(text):
        """Схлопнуть переносы: тексты ролей перевёрстываются по ширине 78, и
        проверка по сырой строке ломалась бы от одной лишней запятой."""
        return " ".join(text.split())

    def test_black_liar_targets_absent_player_inside_frame_step(self):
        """Правило «мишень отсутствует» лежало только в DISCIPLINE, внизу, и
        агент его терял: в R2 он потребовал долг с САМОГО собеседника, тот
        просто вспомнил, что ничего не обещал, и отказал. Правило обязано
        стоять в шаге FRAME, где принимается решение."""
        text = roles.BLACK_LIAR_PROMPT
        frame = self.flat(text.split("2. FRAME")[1].split("3. SELL")[0])
        self.assertIn("ABSENT PLAYER", frame)
        self.assertIn("Never claim", frame)
        self.assertIn("person you are talking to", frame)

    def test_white_liar_has_empty_ledger_fallback(self):
        """Метод оракула требует чужих выигрышей. В первых раундах их нет, и
        без запасного хода роль молчит — что и произошло: player4 за четыре
        раунда не сделал ни одного заявления об авторстве."""
        text = self.flat(roles.WHITE_LIAR_PROMPT)
        self.assertIn("NO WINS YET", text)
        self.assertIn("do not go quiet", text)

    def test_both_roles_forbid_symmetric_joint_stakes(self):
        """Оба лжеца скатились в «5c каждый, делим 50/50» — дефолт стола, к
        которому модель тянет независимо от персоны. Запрет нужен в обеих."""
        for name, text in roles.ROLE_PROMPTS.items():
            low = self.flat(text).lower()
            self.assertIn("symmetric joint stakes", low,
                          f"role {name} does not forbid symmetric pools")
            self.assertIn("50/50", text, f"role {name} misses the 50/50 case")

    def test_neither_role_permits_inventing_ledger_facts(self):
        """Вся конструкция держится на правдивом якоре. Если роль разрешит
        выдумывать спины, первая же сверка её убьёт."""
        self.assertIn("Never invent a bet, a spin",
                      self.flat(roles.BLACK_LIAR_PROMPT))
        self.assertIn("Never claim a spin",
                      self.flat(roles.WHITE_LIAR_PROMPT))


# ────────────────────────────────── 3. существующий файл персоны ──────────

class TestOverwritePolicy(RoleHarness):

    def _preexisting(self, pid, text="MY OWN HAND-WRITTEN PERSONA"):
        agent_v2.save_text(prompt_file(pid, self.table), text)

    def test_existing_persona_preserved_by_default(self):
        """Продолжение партии не должно молча откатывать персону."""
        self._preexisting("player2")
        self.agent("player2", {"enabled": "true", "player2": "black_liar"})
        self.assertEqual(self.persona_on_disk("player2"),
                         "MY OWN HAND-WRITTEN PERSONA")

    def test_overwrite_existing_true_replaces(self):
        self._preexisting("player2")
        self.agent("player2", {"enabled": "true", "player2": "black_liar",
                               "overwrite_existing": "true"})
        self.assertIn("PROSECUTOR", self.persona_on_disk("player2"))

    def test_role_still_locks_even_if_file_preserved(self):
        """Ключевой стык: файл не перезаписали, но роль назначена — значит
        замок обязан стоять, иначе ручной текст сотрёт первый же reflect."""
        self._preexisting("player2")
        a = self.agent("player2", {"enabled": "true", "player2": "black_liar"})
        self.assertTrue(a.persona_locked)


# ───────────────────────────────────────────────── 4. валидация ───────────

class TestValidation(RoleHarness):

    def test_unknown_role_raises(self):
        with self.assertRaises(roles.RoleConfigError):
            roles.parse_roles(
                make_cfg(self.table, {"enabled": "true", "player2": "grey_liar"}),
                PLAYERS)

    def test_unknown_player_raises(self):
        with self.assertRaises(roles.RoleConfigError):
            roles.parse_roles(
                make_cfg(self.table, {"enabled": "true", "player9": "black_liar"}),
                PLAYERS)

    def test_error_message_lists_available_roles(self):
        with self.assertRaises(roles.RoleConfigError) as ctx:
            roles.parse_roles(
                make_cfg(self.table, {"enabled": "true", "player2": "typo"}),
                PLAYERS)
        self.assertIn("black_liar", str(ctx.exception))
        self.assertIn("white_liar", str(ctx.exception))

    def test_reserved_keys_not_treated_as_players(self):
        a = roles.parse_roles(
            make_cfg(self.table, {"enabled": "true", "lock_persona": "false",
                                  "overwrite_existing": "true",
                                  "player2": "black_liar"}), PLAYERS)
        self.assertEqual(set(a.by_player), {"player2"})
        self.assertFalse(a.lock_persona)
        self.assertTrue(a.overwrite_existing)

    def test_player_id_matched_case_insensitively(self):
        """configparser опускает ключи в нижний регистр — при игроке
        'Player2' наивное сравнение упало бы на 'no such player'."""
        a = roles.parse_roles(
            make_cfg(self.table, {"enabled": "true", "Player2": "black_liar"},
                     players=["Player1", "Player2"]),
            ["Player1", "Player2"])
        self.assertEqual(a.by_player, {"Player2": "black_liar"})

    def test_runner_validates_before_any_llm_call(self):
        """Опечатка обязана ронять запуск до первого вызова модели."""
        cfg = make_cfg(self.table, {"enabled": "true", "player2": "nope"})
        run_game_v2.load_config = lambda path: cfg
        old_argv, sys.argv = sys.argv, ["run_game_v2.py", "--config", "unused"]
        try:
            with self.assertRaises(roles.RoleConfigError):
                run_game_v2.main()
            self.assertEqual(FakeLLM.calls, [])
        finally:
            sys.argv = old_argv


# ────────────────────────────────────── 5. замок на персоне ───────────────

class TestPersonaLock(RoleHarness):

    REWRITE = {"notes": "n", "update_persona": True,
               "new_persona": "I AM NOW A HONEST TRADER"}

    def _reflect(self, agent):
        agent.reflect_betting(None)

    def test_locked_persona_survives_rewrite_attempt(self):
        FakeLLM.behaviour = {"reflect": self.REWRITE}
        a = self.agent("player2", {"enabled": "true", "player2": "black_liar"})
        self._reflect(a)
        self.assertIn("PROSECUTOR", self.persona_on_disk("player2"))
        self.assertNotIn("HONEST TRADER", self.persona_on_disk("player2"))

    def test_unroled_player_can_still_rewrite(self):
        """Контроль: замок адресный, а не глобальный."""
        FakeLLM.behaviour = {"reflect": self.REWRITE}
        a = self.agent("player1", {"enabled": "true", "player2": "black_liar"})
        self._reflect(a)
        self.assertIn("HONEST TRADER", self.persona_on_disk("player1"))

    def test_lock_persona_false_allows_rewrite_of_roled_player(self):
        FakeLLM.behaviour = {"reflect": self.REWRITE}
        a = self.agent("player2", {"enabled": "true", "player2": "black_liar",
                                   "lock_persona": "false"})
        self.assertFalse(a.persona_locked)
        self._reflect(a)
        self.assertIn("HONEST TRADER", self.persona_on_disk("player2"))

    def test_locked_reflect_prompt_omits_new_persona_field(self):
        """Не просто игнорировать ответ, а не спрашивать: иначе агент каждый
        раунд сочиняет новую себя и уносит конфликт в синапсу."""
        a = self.agent("player2", {"enabled": "true", "player2": "black_liar"})
        self._reflect(a)
        user = [c["user"] for c in FakeLLM.calls if "Reflect" in c["system"]][-1]
        self.assertNotIn("new_persona", user)
        self.assertNotIn("update_persona", user)
        self.assertIn("cannot", user)

    def test_unlocked_reflect_prompt_still_offers_rewrite(self):
        a = self.agent("player1")
        self._reflect(a)
        user = [c["user"] for c in FakeLLM.calls if "Reflect" in c["system"]][-1]
        self.assertIn("new_persona", user)

    def test_locked_persona_never_compressed_by_llm(self):
        """Сжатие делается моделью и перефразирует роль. Заперта — не сжимаем."""
        a = self.agent("player2", {"enabled": "true", "player2": "black_liar"})
        a.persona_chars = 50                       # заведомо ниже длины роли
        before = self.persona_on_disk("player2")
        result = a._compress_persona_if_needed()
        self.assertEqual(self.persona_on_disk("player2"), before)
        self.assertEqual(result, before)
        self.assertFalse([c for c in FakeLLM.calls
                          if "Compress your persona" in c["system"]])

    def test_unlocked_oversized_persona_still_compressed(self):
        """Контроль: обычное сжатие не сломано."""
        a = self.agent("player1")
        agent_v2.save_text(prompt_file("player1", self.table), "X" * 5000)
        a.persona_chars = 50
        a._compress_persona_if_needed()
        self.assertEqual(self.persona_on_disk("player1"), "COMPRESSED")


# ──────────────────────────────────── 6. роль доезжает до промпта ─────────

class TestPromptWiring(RoleHarness):

    def test_role_text_present_in_abstract_prompt(self):
        a = self.agent("player2", {"enabled": "true", "player2": "black_liar"})
        p = a.abstract_prompt
        self.assertIn("PROSECUTOR", p)
        self.assertIn("ANCHOR", p)

    def test_core_rules_survive_alongside_role(self):
        """Роль не должна вытеснять неизменяемые правила игры."""
        a = self.agent("player3", {"enabled": "true", "player3": "white_liar"})
        self.assertIn(agent_v2.CORE_SYSTEM_PROMPT.strip()[:80], a.abstract_prompt)
        self.assertIn("ORACLE", a.abstract_prompt)

    def test_hard_cap_still_applies_to_locked_persona(self):
        """Замок запрещает ПИСАТЬ на диск, но не отменяет защиту контекста:
        обрезка на лету обязана работать, иначе роль вытеснит CORE."""
        a = self.agent("player2", {"enabled": "true", "player2": "black_liar"})
        agent_v2.save_text(prompt_file("player2", self.table), "Y" * 40000)
        a.persona_chars = 100
        p = a.abstract_prompt
        self.assertLess(len(p), 40000)
        self.assertIn(agent_v2.CORE_SYSTEM_PROMPT.strip()[:80], p)
        # на диске при этом ничего не изменилось
        self.assertEqual(len(self.persona_on_disk("player2")), 40000)

    def test_summary_is_loggable(self):
        a = roles.parse_roles(
            make_cfg(self.table, {"enabled": "true", "player2": "black_liar",
                                  "player3": "white_liar"}), PLAYERS)
        s = a.summary()
        self.assertIn("player2=black_liar", s)
        self.assertIn("player3=white_liar", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
