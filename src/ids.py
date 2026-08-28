# ABOUTME: Mints the stable per-sentence id shared by every extractor and every id-keyed artifact.
# ABOUTME: The sentence text is part of the digest so two sentences can never share an id.

import hashlib

_DIGEST_HEX = 8


def sentence_id(prefix: str, speaker: str, index: int, sentence: str) -> str:
    """Return ``{prefix}-{digest}-{index:03d}`` for one extracted sentence.

    ``prefix`` is the sitting date (``YYYY-MM-DD``) or, for the pre-1909
    archive, the year. ``index`` is the sentence's position in its speech.

    Hashing the sentence text is what keeps the id unique: a speaker's second
    speech on the same day restarts ``index`` at zero, and without the text
    every sentence of that speech would reuse the ids of the first.
    """
    source = f"{prefix}-{speaker}-{index}-{sentence}"
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:_DIGEST_HEX]
    return f"{prefix}-{digest}-{index:03d}"


def split_id(sid: str) -> tuple[str, int]:
    """Recover ``(prefix, index)`` from any id in the current or legacy layout."""
    prefix, _digest, index = sid.rsplit("-", 2)
    return prefix, int(index)


class IdMinter:
    """Issues ids for one extraction run, guaranteeing no id repeats.

    The same (prefix, speaker, index, sentence) can occur more than once —
    Hansard volume indexes repeat verbatim across volumes — and each
    occurrence needs its own id. Repeats fold an occurrence counter into the
    digest, so the outcome depends only on processing order, which the
    extractors keep stable (sorted files, document order within a file).
    """

    def __init__(self):
        self._issued: set[str] = set()

    def mint(self, prefix: str, speaker: str, index: int, sentence: str) -> str:
        sid = sentence_id(prefix, speaker, index, sentence)
        occurrence = 1
        while sid in self._issued:
            sid = sentence_id(prefix, speaker, index, f"{sentence}\x00{occurrence}")
            occurrence += 1
        self._issued.add(sid)
        return sid
