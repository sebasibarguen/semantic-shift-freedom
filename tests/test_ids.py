# ABOUTME: Tests for sentence-id minting in src/ids.py.
# ABOUTME: Pins the property the old scheme lacked: two different sentences never share an id.

import unittest

from src.ids import sentence_id, split_id


class TestSentenceId(unittest.TestCase):
    """Ids are the join key for council gold, arena splits, and annotator
    labels. The old scheme hashed (date, speaker, index), so a speaker's
    second speech on the same day reused every id from the first — 2.3% of
    the corpus. The sentence text must participate in the id."""

    def test_same_speaker_same_day_same_index_different_sentence_differ(self):
        a = sentence_id("2020-01-13", "Boris Johnson", 3, "In fact, when I was in Canada.")
        b = sentence_id("2020-01-13", "Boris Johnson", 3, "One of the issues I discussed.")
        self.assertNotEqual(a, b)

    def test_deterministic(self):
        args = ("1970-01-23", "Mr. Heald", 3, "Heald is already an existing freedom.")
        self.assertEqual(sentence_id(*args), sentence_id(*args))

    def test_keeps_prefix_and_index_layout(self):
        sid = sentence_id("2020-01-13", "Boris Johnson", 3, "In fact.")
        prefix, index = split_id(sid)
        self.assertEqual(prefix, "2020-01-13")
        self.assertEqual(index, 3)
        self.assertTrue(sid.endswith("-003"))

    def test_year_only_prefix_for_archive_records(self):
        prefix, index = split_id(sentence_id("1812", "Mr. Canning", 159, "Popery is repugnant."))
        self.assertEqual((prefix, index), ("1812", 159))

    def test_split_id_reads_legacy_ids(self):
        # Old ids had a 6-hex digest; the layout is otherwise identical.
        self.assertEqual(split_id("2020-01-13-1f287f-003"), ("2020-01-13", 3))
        self.assertEqual(split_id("1812-3330ea-002"), ("1812", 2))


if __name__ == "__main__":
    unittest.main()


class TestIdMinter(unittest.TestCase):
    """Volume indexes repeat verbatim across consecutive volumes, so the same
    (date, speaker, index, text) can legitimately occur more than once. Each
    occurrence still needs its own id or downstream joins collapse them."""

    def test_identical_tuple_gets_distinct_ids_per_occurrence(self):
        from src.ids import IdMinter
        m = IdMinter()
        a = m.mint("1831", "Unknown", 0, "Beer Bill, and Trade, freedom of, iv.")
        b = m.mint("1831", "Unknown", 0, "Beer Bill, and Trade, freedom of, iv.")
        c = m.mint("1831", "Unknown", 0, "Beer Bill, and Trade, freedom of, iv.")
        self.assertEqual(len({a, b, c}), 3)
        for sid in (a, b, c):
            self.assertEqual(split_id(sid), ("1831", 0))

    def test_first_occurrence_matches_plain_sentence_id(self):
        from src.ids import IdMinter
        args = ("2020-01-13", "Boris Johnson", 3, "In fact.")
        self.assertEqual(IdMinter().mint(*args), sentence_id(*args))

    def test_processing_order_is_the_only_input(self):
        from src.ids import IdMinter
        seq = [("1831", "Unknown", 0, "x"), ("1831", "Unknown", 0, "x"), ("1832", "A", 1, "y")]
        self.assertEqual([IdMinter().mint(*s) for s in seq][0], [IdMinter().mint(*s) for s in seq][0])
        m1, m2 = IdMinter(), IdMinter()
        self.assertEqual([m1.mint(*s) for s in seq], [m2.mint(*s) for s in seq])
