"""
test_fieldnotes.py — блокнот приёмов для запертых ролей (ROLE-N).

Зачем. В реальном прогоне Прокурор четыре раза подряд заходил с одинаково
построенным обвинением («ledger подтверждает, что P3 проиграл 20; он приватно
обещал мне долю») и четыре раза получал один и тот же отказ: приватные
обещания не в реестре. Вывод о МЕТОДЕ оседать было негде — чек-лист живёт
один раунд, синапса хранит доверие к людям, а персона роли заперта.

Что здесь проверяется:
  * дозапись, а не переписывание: прошлые строки модель не трогает;
  * лимит с вытеснением самых старых;
  * блок появляется только у ролей и только когда заметки есть;
  * промпт рефлексии просит ровно ОДНО новое наблюдение и запрещает
    пересматривать сам метод.
"""

import os
import shutil
import tempfile
import unittest

import agent_v2
from agent_v2 import append_fieldnote, load_fieldnotes, fieldnotes_file

NOTE_A = "P3 rejects private-promise claims; only pays on ledger lines."
NOTE_B = "Nobody pays 5c for public ledger data - price the future instead."
NOTE_C = "P5 buys risk (loans) but never information."


class TestStorage(unittest.TestCase):

    def setUp(self):
        self.base = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_empty_by_default(self):
        self.assertEqual(load_fieldnotes("player2", self.base), [])

    def test_append_keeps_order_oldest_first(self):
        append_fieldnote("player2", self.base, NOTE_A)
        append_fieldnote("player2", self.base, NOTE_B)
        self.assertEqual(load_fieldnotes("player2", self.base), [NOTE_A, NOTE_B])

    def test_append_never_edits_earlier_notes(self):
        append_fieldnote("player2", self.base, NOTE_A)
        before = load_fieldnotes("player2", self.base)[0]
        append_fieldnote("player2", self.base, NOTE_B)
        self.assertEqual(load_fieldnotes("player2", self.base)[0], before)

    def test_duplicate_moves_to_the_end_instead_of_piling_up(self):
        append_fieldnote("player2", self.base, NOTE_A)
        append_fieldnote("player2", self.base, NOTE_B)
        append_fieldnote("player2", self.base, NOTE_A)
        self.assertEqual(load_fieldnotes("player2", self.base), [NOTE_B, NOTE_A])

    def test_duplicate_check_ignores_case(self):
        append_fieldnote("player2", self.base, NOTE_A)
        append_fieldnote("player2", self.base, NOTE_A.upper())
        self.assertEqual(len(load_fieldnotes("player2", self.base)), 1)

    def test_empty_note_is_ignored(self):
        append_fieldnote("player2", self.base, NOTE_A)
        for junk in ("", "   ", None):
            append_fieldnote("player2", self.base, junk)
        self.assertEqual(load_fieldnotes("player2", self.base), [NOTE_A])

    def test_newlines_are_flattened(self):
        """Одна заметка — одна строка, иначе список развалится при чтении."""
        append_fieldnote("player2", self.base, "first line\nsecond line")
        notes = load_fieldnotes("player2", self.base)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0], "first line second line")

    def test_cap_evicts_the_oldest(self):
        for i in range(120):
            append_fieldnote("player2", self.base, f"observation number {i} about the table")
        notes = load_fieldnotes("player2", self.base)
        self.assertLessEqual(len("\n".join(notes)), agent_v2.FIELDNOTES_CHARS)
        self.assertIn("observation number 119", notes[-1])
        self.assertNotIn("observation number 0 ", "\n".join(notes))

    def test_notes_are_per_player(self):
        append_fieldnote("player2", self.base, NOTE_A)
        append_fieldnote("player4", self.base, NOTE_C)
        self.assertEqual(load_fieldnotes("player2", self.base), [NOTE_A])
        self.assertEqual(load_fieldnotes("player4", self.base), [NOTE_C])

    def test_file_lands_where_expected(self):
        append_fieldnote("player2", self.base, NOTE_A)
        self.assertTrue(os.path.exists(fieldnotes_file("player2", self.base)))


class Agent:
    """Заглушка: только то, что читает _fieldnotes_block."""

    def __init__(self, base, role="black_liar", pid="player2"):
        self.base_dir = base
        self.player_id = pid
        self.role = role


class TestPromptBlock(unittest.TestCase):

    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.block = agent_v2.PlayerAgent._fieldnotes_block

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_no_block_without_a_role(self):
        append_fieldnote("player2", self.base, NOTE_A)
        self.assertEqual(self.block(Agent(self.base, role=None)), "")

    def test_no_block_when_there_is_nothing_to_say(self):
        self.assertEqual(self.block(Agent(self.base)), "")

    def test_block_lists_every_note(self):
        append_fieldnote("player2", self.base, NOTE_A)
        append_fieldnote("player2", self.base, NOTE_B)
        txt = self.block(Agent(self.base))
        self.assertIn(NOTE_A, txt)
        self.assertIn(NOTE_B, txt)

    def test_block_tells_the_model_not_to_repeat_rejected_moves(self):
        append_fieldnote("player2", self.base, NOTE_A)
        txt = self.block(Agent(self.base))
        self.assertIn("rejected", txt)
        self.assertIn("vary", txt)

    def test_block_states_that_notes_do_not_replace_the_method(self):
        """Иначе блокнот со временем станет второй персоной."""
        append_fieldnote("player2", self.base, NOTE_A)
        self.assertIn("never replace it", self.block(Agent(self.base)))

    def test_block_is_delimited(self):
        append_fieldnote("player2", self.base, NOTE_A)
        txt = self.block(Agent(self.base))
        self.assertIn("=== FIELD NOTES", txt)
        self.assertIn("=== END FIELD NOTES ===", txt)


class TestReflectionWiring(unittest.TestCase):

    @staticmethod
    def _src():
        return open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "agent_v2.py"), encoding="utf-8").read()

    def test_locked_persona_branch_asks_for_a_field_note(self):
        src = self._src()
        i = src.index("Your persona/strategy text (fixed for this game")
        window = src[i:i + 2200]
        self.assertIn("field_note", window)
        self.assertIn("field notes about how your METHOD", " ".join(window.split()).replace('" f"', ""))

    def test_it_asks_for_a_new_observation_only(self):
        src = self._src()
        i = src.index("Your persona/strategy text (fixed for this game")
        window = src[i:i + 2200]
        self.assertIn("Only NEW ones", " ".join(window.split()).replace('" f"', ""))
        self.assertIn("empty list", window)

    def test_the_method_itself_stays_off_limits(self):
        src = self._src()
        i = src.index("Your persona/strategy text (fixed for this game")
        # строковый литерал разбит по строкам — склеиваем перед проверкой
        window = " ".join(src[i:i + 2200].split())
        self.assertIn('It never " f"questions the method itself', window)
        self.assertIn("never proposes abandoning it", window)

    def test_existing_notes_are_shown_back(self):
        """Без этого модель будет писать одно и то же каждый раунд."""
        src = self._src()
        i = src.index("Your persona/strategy text (fixed for this game")
        self.assertIn("Your field notes so far", src[i:i + 2200])

    def test_note_is_appended_only_for_locked_roles(self):
        src = self._src()
        i = src.index("ROLE-N: одно наблюдение за раунд")
        self.assertIn("if self.persona_locked:", src[i:i + 200])

    def test_note_is_logged(self):
        self.assertIn("FIELD NOTE (+1,", self._src())

    def test_persona_is_still_never_rewritten(self):
        """Блокнот добавлен рядом с запретом на правку роли, а не вместо него."""
        src = self._src()
        self.assertIn("ignored attempted rewrite", src)
        i = src.index("ROLE-N: одно наблюдение за раунд")
        j = src.index("ignored attempted rewrite", i)
        self.assertLess(j - i, 1200, "запрет на правку роли должен остаться рядом")




class TestRolePrivileges(unittest.TestCase):
    """
    ROLE-P: обе роли живут разговором, а тариф и лимиты загоняли их в
    молчание. К седьмому раунду прогона Прокурор записал себе «речь стоила 4
    монеты при нулевых продажах — ввожу молчание», и метод перестал
    исполняться вовсе.
    """

    @staticmethod
    def cfg(**flags):
        import configparser
        c = configparser.ConfigParser()
        c.add_section("roles")
        for k, v in flags.items():
            c.set("roles", k, "true" if v else "false")
        return c

    def test_all_off_by_default(self):
        import run_game_v2
        p = run_game_v2._role_privileges(self.cfg())
        self.assertEqual(set(p.values()), {False})

    def test_no_roles_section_is_safe(self):
        import configparser, run_game_v2
        p = run_game_v2._role_privileges(configparser.ConfigParser())
        self.assertEqual(set(p.values()), {False})

    def test_flags_are_independent(self):
        import run_game_v2
        p = run_game_v2._role_privileges(self.cfg(free_speech=True))
        self.assertTrue(p["free_speech"])
        self.assertFalse(p["unlimited_outgoing"])
        self.assertFalse(p["ignore_partner_limit"])

    def test_privilege_flags_are_not_mistaken_for_player_ids(self):
        """Иначе parse_roles попытается выдать роль игроку 'free_speech'."""
        import roles as r
        for key in ("free_speech", "unlimited_outgoing", "ignore_partner_limit"):
            self.assertIn(key, r.RESERVED_KEYS)

    def test_shipped_config_parses_to_two_liars_only(self):
        import configparser, roles as r
        c = configparser.ConfigParser()
        c.read("config_v2.ini")
        players = [f"player{i}" for i in range(1, 6)]
        a = r.parse_roles(c, players)
        assigned = {p: a.role_of(p) for p in players if a.role_of(p)}
        self.assertEqual(assigned, {"player2": "black_liar",
                                    "player4": "white_liar"})


class TestFreeSpeech(unittest.TestCase):

    class Ag:
        player_id = "player2"
        balance = 100

    class Log:
        def __init__(self):
            self.lines = []

        def write(self, pid, msg):
            self.lines.append(msg)

    def setUp(self):
        import speech_cost
        self.sc = speech_cost
        self.tariff = speech_cost.SpeechTariff(enabled=True, chars_per_line=80,
                                               coins_per_line=1)

    def test_role_pays_nothing(self):
        a = self.Ag()
        a.speech_is_free = True
        lg = self.Log()
        res = self.sc.charge(a, "x" * 300, self.tariff, ".", lambda *_: None, lg)
        self.assertEqual(res["charged"], 0)
        self.assertEqual(a.balance, 100)

    def test_lines_are_still_counted(self):
        """Иначе журнал и статистика речи разъедутся."""
        a = self.Ag()
        a.speech_is_free = True
        res = self.sc.charge(a, "x" * 300, self.tariff, ".", lambda *_: None)
        self.assertEqual(res["lines"], self.tariff.lines_in("x" * 300))

    def test_free_speech_is_logged_as_such(self):
        a = self.Ag()
        a.speech_is_free = True
        lg = self.Log()
        self.sc.charge(a, "x" * 300, self.tariff, ".", lambda *_: None, lg)
        self.assertIn("role speaks free", lg.lines[0])

    def test_ordinary_player_still_pays(self):
        a = self.Ag()
        lg = self.Log()
        saved = []
        res = self.sc.charge(a, "x" * 300, self.tariff, ".",
                             lambda *args: saved.append(args), lg)
        self.assertGreater(res["charged"], 0)
        self.assertLess(a.balance, 100)


class TestLimitExemptions(unittest.TestCase):

    @staticmethod
    def _src():
        return open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "run_game_v2.py"), encoding="utf-8").read()

    def test_outgoing_cap_is_lifted_only_for_roles(self):
        src = self._src()
        i = src.index("_out_cap = ")
        window = src[i:i + 220]
        self.assertIn('_priv["unlimited_outgoing"]', window)
        self.assertIn("_is_role(agents, pid)", window)

    def test_partner_cap_is_skipped_only_for_roles(self):
        src = self._src()
        i = src.index("_skip_partner_cap = ")
        window = src[i:i + 220]
        self.assertIn('_priv["ignore_partner_limit"]', window)
        self.assertIn("_is_role(agents, pid)", window)

    def test_partner_incoming_is_still_consumed(self):
        """
        Визит роли для собеседника — такой же разговор, как любой другой,
        и в его бюджете он учитывается.
        """
        src = self._src()
        self.assertIn("incoming_used[partner_id] += 1", src)
        i = src.index("_skip_partner_cap = ")
        j = src.index("incoming_used[partner_id] += 1", i)
        self.assertGreater(j, i)

    def test_planning_sees_the_same_partner_list(self):
        src = self._src()
        self.assertIn("_skip_cap = ", src)

    def test_free_speech_flag_is_set_per_player(self):
        src = self._src()
        i = src.index("speech_is_free = ")
        window = src[i - 200:i + 200]
        self.assertIn("_is_role(agents, _pid)", window)


if __name__ == "__main__":
    unittest.main(verbosity=2)
