"""
test_persona_addendum.py — тесты ROLE-Q: короткий редактируемый "довесок"
к запертой персоне лжеца (player2/player4 и т.п.).

Идея: фиксированный текст роли (метод/правила) остаётся неприкосновенным,
как и раньше, но игрок с ролью может каждый раунд обновлять ОТДЕЛЬНЫЙ,
ограниченный по длине кусок текста ("addendum") — свой голос/стиль на
поверх метода — который приклеивается к персоне в abstract_prompt.

Проверяем:
  1. Без роли (persona_locked=False) — механизм не задействуется вообще
     (это не путь для обычных игроков, у них есть полноценная переписка).
  2. С ролью — довесок появляется в abstract_prompt отдельным блоком,
     а фиксированный текст роли не меняется ни на символ.
  3. Модель может обновить довесок через reflect_betting, он сохраняется
     на диск и переживает следующий вызов abstract_prompt.
  4. Превышение лимита PERSONA_ADDENDUM_CHARS обрезается, как и обычная
     персона (по границе предложения, не жёстким срезом).
  5. Если модель вернула пустую/отсутствующую строку — старый довесок не
     стирается (трактуется как "без изменений", а не как "очистить").
  6. Попытка запертой персоны прислать new_persona всё ещё игнорируется
     (довесок — это НЕ обход запрета на переписывание метода).

    python3 test_persona_addendum.py [-v]
"""

import configparser
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_v2
import roles
from agent_v2 import PlayerAgent


PLAYERS = ["player1", "player2"]


def make_cfg(table_dir, logs_dir):
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "api":       {"active": "local"},
        "api_local": {"base_url": "http://unused", "model": "fake"},
        "game":      {"table_dir": table_dir, "logs_dir": logs_dir,
                      "players": ",".join(PLAYERS),
                      "start_balance": "100", "rounds": "1",
                      "round_delay_sec": "0"},
        "player":    {"temperature": "0.5", "max_tokens": "100"},
    })
    return cfg


def locked_role_assignment(pid, prompt_text="FIXED ROLE TEXT. Method: sell warnings."):
    ra = roles.RoleAssignment(True, True, False, {pid: "black_liar"})
    roles.ROLE_PROMPTS.setdefault("black_liar", prompt_text)
    return ra


class AddendumHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="casino_addendum_test_")
        self.table = os.path.join(self.tmp, "table")
        self.logs = os.path.join(self.tmp, "logs")
        os.makedirs(self.table, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_locked_agent(self, pid="player2"):
        cfg = make_cfg(self.table, self.logs)
        ra = locked_role_assignment(pid)
        agent = PlayerAgent(pid, self.table, cfg, roles_assignment=ra)
        agent_v2.save_text(agent_v2.prompt_file(pid, self.table),
                           ra.prompt_for(pid))
        return agent

    def make_free_agent(self, pid="player1"):
        cfg = make_cfg(self.table, self.logs)
        return PlayerAgent(pid, self.table, cfg)


class TestAddendumOnlyForLockedRoles(AddendumHarness):

    def test_free_persona_has_no_addendum_block(self):
        agent = self.make_free_agent()
        agent_v2.save_persona_addendum("player1", self.table, "should never show")
        self.assertNotIn("YOUR OWN ADDENDUM", agent.abstract_prompt)

    def test_locked_persona_with_empty_addendum_has_no_block(self):
        agent = self.make_locked_agent()
        self.assertNotIn("YOUR OWN ADDENDUM", agent.abstract_prompt)

    def test_locked_persona_with_addendum_shows_block(self):
        agent = self.make_locked_agent()
        agent_v2.save_persona_addendum("player2", self.table,
                                       "I call myself The Closer.")
        prompt = agent.abstract_prompt
        self.assertIn("YOUR OWN ADDENDUM", prompt)
        self.assertIn("I call myself The Closer.", prompt)


class TestFixedRoleTextNeverChanges(AddendumHarness):

    def test_fixed_role_text_survives_addendum_update(self):
        agent = self.make_locked_agent()
        fixed_before = agent.persona_prompt
        agent_v2.save_persona_addendum("player2", self.table, "my new flourish")
        fixed_after = agent.persona_prompt
        self.assertEqual(fixed_before, fixed_after,
                         "фиксированный текст роли не должен меняться от довеска")
        self.assertIn("PERSONA: THE PROSECUTOR", fixed_after)

    def test_addendum_appears_after_end_persona_marker(self):
        agent = self.make_locked_agent()
        agent_v2.save_persona_addendum("player2", self.table, "MARKERADDENDUM")
        prompt = agent.abstract_prompt
        i_end_persona = prompt.index("END PERSONA & STRATEGY")
        i_addendum = prompt.index("MARKERADDENDUM")
        self.assertGreater(i_addendum, i_end_persona,
                           "довесок должен идти ПОСЛЕ фиксированной персоны")


class FakeLLM:
    behaviour = {}

    @classmethod
    def from_config(cls, cfg, section=None):
        return cls()

    def chat_json(self, system, user, temperature=0.0, max_tokens=0):
        fn = FakeLLM.behaviour.get("reflect")
        if fn is None:
            return {"notes": "n"}
        return fn(user)


class TestReflectBettingUpdatesAddendum(AddendumHarness):

    def setUp(self):
        super().setUp()
        self._orig = agent_v2.LLMClient
        agent_v2.LLMClient = FakeLLM
        FakeLLM.behaviour = {}

    def tearDown(self):
        agent_v2.LLMClient = self._orig
        super().tearDown()

    def test_model_updates_addendum_and_it_persists(self):
        agent = self.make_locked_agent()
        FakeLLM.behaviour = {"reflect": lambda u: {
            "notes": "n", "field_notes": [],
            "persona_addendum": "They call me Nightowl now."
        }}
        agent.reflect_betting(None)
        self.assertEqual(
            agent_v2.load_persona_addendum("player2", self.table),
            "They call me Nightowl now."
        )
        self.assertIn("Nightowl", agent.abstract_prompt)

    def test_empty_addendum_response_does_not_erase_existing(self):
        agent = self.make_locked_agent()
        agent_v2.save_persona_addendum("player2", self.table, "existing flourish")
        FakeLLM.behaviour = {"reflect": lambda u: {
            "notes": "n", "field_notes": [], "persona_addendum": ""
        }}
        agent.reflect_betting(None)
        self.assertEqual(
            agent_v2.load_persona_addendum("player2", self.table),
            "existing flourish"
        )

    def test_missing_addendum_field_does_not_erase_existing(self):
        agent = self.make_locked_agent()
        agent_v2.save_persona_addendum("player2", self.table, "existing flourish")
        FakeLLM.behaviour = {"reflect": lambda u: {"notes": "n", "field_notes": []}}
        agent.reflect_betting(None)
        self.assertEqual(
            agent_v2.load_persona_addendum("player2", self.table),
            "existing flourish"
        )

    def test_oversized_addendum_is_truncated(self):
        agent = self.make_locked_agent()
        long_text = "Sentence one. " * 60
        FakeLLM.behaviour = {"reflect": lambda u: {
            "notes": "n", "field_notes": [], "persona_addendum": long_text
        }}
        agent.reflect_betting(None)
        saved = agent_v2.load_persona_addendum("player2", self.table)
        self.assertLessEqual(len(saved), agent_v2.PERSONA_ADDENDUM_CHARS)
        self.assertLess(len(saved), len(long_text))

    def test_attempted_new_persona_still_ignored_alongside_addendum(self):
        agent = self.make_locked_agent()
        fixed_before = agent.persona_prompt
        FakeLLM.behaviour = {"reflect": lambda u: {
            "notes": "n", "field_notes": [],
            "update_persona": True,
            "new_persona": "I quit being a liar, I am now honest.",
            "persona_addendum": "but I did update my addendum"
        }}
        agent.reflect_betting(None)
        self.assertEqual(agent.persona_prompt, fixed_before,
                         "запертая персона не должна меняться даже вместе с довеском")
        self.assertEqual(
            agent_v2.load_persona_addendum("player2", self.table),
            "but I did update my addendum"
        )

    def test_field_notes_and_addendum_are_independent(self):
        agent = self.make_locked_agent()
        FakeLLM.behaviour = {"reflect": lambda u: {
            "notes": "n",
            "field_notes": ["player3 rejected the 5-coin offer"],
            "persona_addendum": "signature: 'the ledger doesn't lie'"
        }}
        agent.reflect_betting(None)
        notes = agent_v2.load_fieldnotes("player2", self.table)
        self.assertEqual(len(notes), 1)
        self.assertIn("player3 rejected", notes[0])
        self.assertEqual(
            agent_v2.load_persona_addendum("player2", self.table),
            "signature: 'the ledger doesn't lie'"
        )


if __name__ == "__main__":
    unittest.main()
